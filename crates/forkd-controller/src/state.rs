//! In-memory VM registry, snapshotted to a JSON file for crash recovery.
//!
//! Concurrency: one `parking_lot::Mutex` wraps the in-memory registry and a
//! second serializes atomic persistence. Writes are infrequent (one per
//! sandbox lifecycle event) so contention is a non-issue at our scale
//! (≤ a few thousand sandboxes/host).
//!
//! On startup, the daemon reads `state.json`, then reconciles each entry
//! against the live system (does the netns still exist, is the FC pid
//! still alive). Stale entries get pruned. See `Registry::reconcile`.
use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crate::api::{SandboxInfo, SnapshotInfo, WorkspaceInfo, WorkspaceStatus};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PersistentState {
    #[serde(default)]
    pub snapshots: BTreeMap<String, SnapshotInfo>,
    #[serde(default)]
    pub sandboxes: BTreeMap<String, SandboxInfo>,
    /// Stateful workspaces (#116). Keyed by name (user-facing
    /// identifier; unique per daemon). The internal `id` field on
    /// `WorkspaceInfo` is for audit / cross-reference with live sandbox
    /// pids; lookups + mutations from the HTTP / CLI surface go by name.
    #[serde(default)]
    pub workspaces: BTreeMap<String, WorkspaceInfo>,
}

#[derive(Clone)]
pub struct Registry {
    inner: Arc<Mutex<PersistentState>>,
    persist_lock: Arc<Mutex<()>>,
    path: PathBuf,
}

impl Registry {
    pub fn load_or_init(path: impl Into<PathBuf>) -> Result<Self> {
        let path: PathBuf = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create state dir {}", parent.display()))?;
        }
        let state = if path.exists() {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("read state file {}", path.display()))?;
            serde_json::from_str(&raw)
                .with_context(|| format!("parse state file {}", path.display()))?
        } else {
            PersistentState::default()
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(state)),
            persist_lock: Arc::new(Mutex::new(())),
            path,
        })
    }

    pub fn snapshot(&self) -> PersistentState {
        self.inner.lock().clone()
    }

    pub fn list_snapshots(&self) -> Vec<SnapshotInfo> {
        self.inner.lock().snapshots.values().cloned().collect()
    }

    pub fn list_sandboxes(&self) -> Vec<SandboxInfo> {
        self.inner.lock().sandboxes.values().cloned().collect()
    }

    pub fn get_snapshot(&self, tag: &str) -> Option<SnapshotInfo> {
        self.inner.lock().snapshots.get(tag).cloned()
    }

    pub fn get_sandbox(&self, id: &str) -> Option<SandboxInfo> {
        self.inner.lock().sandboxes.get(id).cloned()
    }

    pub fn insert_snapshot(&self, snap: SnapshotInfo) -> Result<()> {
        {
            let mut g = self.inner.lock();
            g.snapshots.insert(snap.tag.clone(), snap);
        }
        self.flush()
    }

    pub fn insert_sandbox(&self, sb: SandboxInfo) -> Result<()> {
        {
            let mut g = self.inner.lock();
            g.sandboxes.insert(sb.id.clone(), sb);
        }
        self.flush()
    }

    /// Mark a sandbox as having been BRANCHed at least once, and
    /// record its most recent BRANCH output's memory.bin path so
    /// subsequent diff BRANCHes can chain off it (phase 1d).
    ///
    /// Returns Ok(Some(new_branch_count)) on success, where the count is
    /// the post-increment value (so the first BRANCH on a sandbox
    /// returns 1). Returns Ok(None) if the sandbox is no longer
    /// registered (it got DELETE'd during the BRANCH window —
    /// best-effort, the updated state is silently dropped).
    pub fn mark_branched(&self, id: &str, last_memory_bin: PathBuf) -> Result<Option<u32>> {
        let new_count = {
            let mut g = self.inner.lock();
            match g.sandboxes.get_mut(id) {
                Some(sb) => {
                    sb.has_branched = true;
                    sb.last_branch_memory_path = Some(last_memory_bin);
                    sb.branch_count = sb.branch_count.saturating_add(1);
                    Some(sb.branch_count)
                }
                None => None,
            }
        };
        if new_count.is_some() {
            self.flush()?;
        }
        Ok(new_count)
    }

    pub fn remove_sandbox(&self, id: &str) -> Result<Option<SandboxInfo>> {
        let removed = {
            let mut g = self.inner.lock();
            g.sandboxes.remove(id)
        };
        if removed.is_some() {
            self.flush()?;
        }
        Ok(removed)
    }

    pub fn remove_snapshot(&self, tag: &str) -> Result<Option<SnapshotInfo>> {
        let removed = {
            let mut g = self.inner.lock();
            g.snapshots.remove(tag)
        };
        if removed.is_some() {
            self.flush()?;
        }
        Ok(removed)
    }

    // -------------------------- workspaces (#116) --------------------------

    pub fn list_workspaces(&self) -> Vec<WorkspaceInfo> {
        self.inner.lock().workspaces.values().cloned().collect()
    }

    pub fn get_workspace(&self, name: &str) -> Option<WorkspaceInfo> {
        self.inner.lock().workspaces.get(name).cloned()
    }

    pub fn insert_workspace(&self, ws: WorkspaceInfo) -> Result<()> {
        {
            let mut g = self.inner.lock();
            g.workspaces.insert(ws.name.clone(), ws);
        }
        self.flush()
    }

    pub fn remove_workspace(&self, name: &str) -> Result<Option<WorkspaceInfo>> {
        let removed = {
            let mut g = self.inner.lock();
            g.workspaces.remove(name)
        };
        if removed.is_some() {
            self.flush()?;
        }
        Ok(removed)
    }

    /// Update a workspace in-place via a mutation closure. Returns
    /// Ok(true) if the workspace was found and the change persisted;
    /// Ok(false) if no such workspace.
    pub fn update_workspace<F>(&self, name: &str, mutate: F) -> Result<bool>
    where
        F: FnOnce(&mut WorkspaceInfo),
    {
        let updated = {
            let mut g = self.inner.lock();
            match g.workspaces.get_mut(name) {
                Some(ws) => {
                    mutate(ws);
                    true
                }
                None => false,
            }
        };
        if updated {
            self.flush()?;
        }
        Ok(updated)
    }

    /// Persist current state atomically (write to temp + rename).
    fn flush(&self) -> Result<()> {
        // Every Registry clone targets the same temp path. Keep snapshot,
        // write, and rename under one lock so concurrent lifecycle mutations
        // cannot race the shared state.json.tmp or let an older snapshot
        // overwrite a newer one.
        let _persist_guard = self.persist_lock.lock();
        let state = self.inner.lock().clone();
        let tmp = self.path.with_extension("json.tmp");
        let body = serde_json::to_vec_pretty(&state)?;
        fs::write(&tmp, &body)
            .with_context(|| format!("write tmp state file {}", tmp.display()))?;
        fs::rename(&tmp, &self.path)
            .with_context(|| format!("rename {} → {}", tmp.display(), self.path.display()))?;
        Ok(())
    }

    /// Prune sandbox entries whose recorded pid is no longer alive, and
    /// mark any `Running` workspace whose sandbox was just pruned as
    /// `Stale` (#116): the daemon crashed/restarted out from under it.
    /// Suspended workspaces are untouched; they were intentionally
    /// parked. Snapshots are kept regardless (they're disk artifacts).
    ///
    /// Runs the whole in-memory pass under one registry lock (#310):
    /// `pid_alive` is a `/proc` stat, cheap and free of any lock we
    /// care about, so holding the mutex across it is fine and removes
    /// the window where an HTTP handler could remove/reinsert a
    /// just-observed-stale id between the scan and the prune.
    pub fn reconcile(&self) -> Result<usize> {
        let (pruned, stale_ws_changed) = {
            let mut g = self.inner.lock();
            let stale_ids: Vec<String> = g
                .sandboxes
                .iter()
                .filter_map(|(id, sb)| match sb.pid {
                    Some(pid) if !pid_alive(pid) => Some(id.clone()),
                    _ => None,
                })
                .collect();
            for id in &stale_ids {
                g.sandboxes.remove(id);
            }

            let live_ids: std::collections::HashSet<String> = g.sandboxes.keys().cloned().collect();
            let mut stale_ws_changed = false;
            for ws in g.workspaces.values_mut() {
                if ws.status == WorkspaceStatus::Running {
                    let live = ws
                        .live_sandbox_id
                        .as_ref()
                        .is_some_and(|id| live_ids.contains(id));
                    if !live {
                        ws.status = WorkspaceStatus::Stale;
                        ws.live_sandbox_id = None;
                        stale_ws_changed = true;
                    }
                }
            }

            (stale_ids.len(), stale_ws_changed)
        };

        if pruned > 0 || stale_ws_changed {
            self.flush()?;
        }
        Ok(pruned)
    }

    /// Mark any `Running` workspace whose `live_sandbox_id` is no longer
    /// present in `sandboxes` as `Stale`. We don't touch Suspended
    /// workspaces; they were intentionally parked.
    ///
    /// Used by `kill_orphans` (after pruning orphaned processes) to keep
    /// the workspace/liveness view consistent. Returns whether anything
    /// changed.
    fn mark_stale_workspaces(&self) -> bool {
        let live_ids: std::collections::HashSet<String> =
            self.inner.lock().sandboxes.keys().cloned().collect();
        let mut changed = false;
        {
            let mut g = self.inner.lock();
            for ws in g.workspaces.values_mut() {
                if ws.status == WorkspaceStatus::Running {
                    let live = ws
                        .live_sandbox_id
                        .as_ref()
                        .is_some_and(|id| live_ids.contains(id));
                    if !live {
                        ws.status = WorkspaceStatus::Stale;
                        ws.live_sandbox_id = None;
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// Kill orphaned Firecracker processes on startup (issue #298).
    ///
    /// **MUST only be called before any `Vm` handle is held** (i.e. in
    /// `run_daemon` before `live_vms` is populated). This method
    /// SIGKILLs every sandbox entry whose PID is alive — it cannot
    /// distinguish an orphaned VM from a currently managed one.
    ///
    /// After `reconcile()` prunes entries with dead PIDs, any remaining
    /// sandbox entries have alive PIDs — but the controller has no
    /// `live_vms` handle for them (the controller restarted). These are
    /// orphaned Firecracker processes: alive but unmanageable.
    ///
    /// For each orphan: verify the PID still belongs to a Firecracker
    /// process (guards against PID reuse), send SIGKILL, wait for the
    /// process to exit (bounded timeout), then prune the registry entry.
    /// If the kill fails with a real error (not ESRCH), the entry is
    /// **not** pruned — a live orphan holding resources must stay
    /// registered so the operator can investigate.
    ///
    /// After this, the NetnsAllocator (empty active set) and
    /// shared_tap_owner (None) are safe once all orphans are confirmed
    /// dead: no orphaned VM holds a netns index or the shared tap.
    /// If some kills failed, those entries remain in the registry and
    /// the allocator may still collide with them — the caller should
    /// log the failure count and the operator should investigate.
    pub(crate) fn kill_orphans(&self) -> Result<KillOrphansResult> {
        // Collect orphans with their recorded durable identity (start time).
        // The lock is dropped before any kill so the (bounded) wait_for_death
        // poll never holds the registry mutex.
        let orphans: Vec<(String, u32, Option<u64>)> = {
            let g = self.inner.lock();
            g.sandboxes
                .iter()
                .filter_map(|(id, sb)| match sb.pid {
                    Some(pid) if pid_alive(pid) => Some((id.clone(), pid, sb.proc_starttime)),
                    _ => None,
                })
                .collect()
        };

        let mut killed = 0usize;
        let mut pruned_stale = 0usize;
        let mut kill_failed = 0usize;

        for (id, pid, recorded_starttime) in orphans {
            // Primary identity check: compare the live process start time
            // against the one recorded at registration. This is the
            // durable identity that survives PID reuse — `comm` alone
            // is spoofable and cannot distinguish two Firecracker
            // processes that happen to share a PID over time.
            match process_identity_matches(pid, recorded_starttime) {
                IdentityCheck::Match => {
                    // Same process (start time matches). Open a pidfd
                    // NOW, after the identity check, so the signal is
                    // pinned to the process that currently holds the PID.
                    // This narrows the TOCTOU window to the open→signal
                    // interval: even if the PID is reused between here and
                    // pidfd_send_signal, the fd targets whatever was pinned
                    // at open time, not whatever currently holds the PID.
                    //
                    // There is a residual window between the starttime
                    // read (in process_identity_matches) and pidfd_open:
                    // the original could exit and the PID could be recycled
                    // by another Firecracker in that gap. We close it by
                    // re-reading the start time AFTER pidfd_open succeeds and
                    // comparing again. Because pidfd pins the process at
                    // open time, /proc/<pid> reflects the pinned process;
                    // a mismatch means the PID was reused between the check
                    // and the open → prune without killing (PidReuse).
                    let pidfd = match pidfd_open(pid) {
                        Ok(fd) => fd,
                        Err(e) if e.raw_os_error() == Some(libc::ESRCH) => {
                            // Benign race: process exited between the
                            // identity check and pidfd_open. Safe to prune.
                            tracing::debug!(
                                sandbox_id = %id,
                                pid = pid,
                                "orphan exited before pidfd_open (ESRCH); pruning registry entry"
                            );
                            self.inner.lock().sandboxes.remove(&id);
                            pruned_stale += 1;
                            continue;
                        }
                        Err(e) => {
                            // pidfd_open can fail with EINVAL/ENOSYS on
                            // kernels < 5.3. Fail closed: keep the entry
                            // so the operator can investigate, rather than
                            // risk killing the wrong process via kill(2).
                            tracing::error!(
                                sandbox_id = %id,
                                pid = pid,
                                error = %e,
                                "pidfd_open failed; keeping registry entry to prevent unsafe kill"
                            );
                            kill_failed += 1;
                            continue;
                        }
                    };

                    // Re-verify the start time AFTER pidfd_open. This
                    // closes the check→open TOCTOU window: if the PID was
                    // reused between process_identity_matches and
                    // pidfd_open, the live start time now differs from
                    // the recorded one. Prune without killing (the pinned
                    // process is not ours). Recorded_starttime is Some by
                    // construction here (Match requires it), but guard for
                    // clarity.
                    if let Some(recorded) = recorded_starttime {
                        match read_proc_starttime(pid) {
                            Some(live) if live == recorded => {
                                // pidfd-pinned process is the original.
                            }
                            _ => {
                                let _ = unsafe { libc::close(pidfd) };
                                tracing::warn!(
                                    sandbox_id = %id,
                                    pid = pid,
                                    "PID reused between identity check and pidfd_open \
                                     (start time changed); pruning without killing"
                                );
                                self.inner.lock().sandboxes.remove(&id);
                                pruned_stale += 1;
                                continue;
                            }
                        }
                    }

                    // Secondary confirmation: the comm name should still
                    // be firecracker. This is NOT a security boundary (comm
                    // is spoofable) — it catches corrupted state.json that
                    // somehow recorded a valid-looking start time for a
                    // non-firecracker process. If it fails, fail closed.
                    if !comm_is_firecracker(pid) {
                        let _ = unsafe { libc::close(pidfd) };
                        tracing::error!(
                            sandbox_id = %id,
                            pid = pid,
                            "start time matched but comm is not firecracker; \
                             keeping registry entry (possible state corruption)"
                        );
                        kill_failed += 1;
                        continue;
                    }

                    tracing::warn!(
                        sandbox_id = %id,
                        pid = pid,
                        "killing orphaned Firecracker process on startup (pidfd signal, identity verified)"
                    );
                    match pidfd_send_kill(pidfd) {
                        Ok(()) => {
                            // Wait for the process to actually exit (bounded).
                            // SIGKILL is asynchronous; a D-state process can
                            // hold netns/tap resources past the kill return.
                            if wait_for_death(pid, std::time::Duration::from_secs(5)) {
                                self.inner.lock().sandboxes.remove(&id);
                                killed += 1;
                            } else {
                                tracing::error!(
                                    sandbox_id = %id,
                                    pid = pid,
                                    "orphaned Firecracker did not exit within 5s of SIGKILL; \
                                     keeping registry entry to prevent resource collision"
                                );
                                kill_failed += 1;
                            }
                        }
                        Err(e) if e.raw_os_error() == Some(libc::ESRCH) => {
                            // Process exited between pidfd_open and the
                            // signal (benign). Safe to prune.
                            tracing::debug!(
                                sandbox_id = %id,
                                pid = pid,
                                "orphan exited before signal (ESRCH); pruning registry entry"
                            );
                            self.inner.lock().sandboxes.remove(&id);
                            pruned_stale += 1;
                        }
                        Err(e) => {
                            tracing::error!(
                                sandbox_id = %id,
                                pid = pid,
                                error = %e,
                                "pidfd_send_signal failed; keeping registry entry to prevent resource collision"
                            );
                            kill_failed += 1;
                        }
                    }
                    // Always close the pidfd; leak protection on every path.
                    let _ = unsafe { libc::close(pidfd) };
                }
                IdentityCheck::PidReuse => {
                    // PID is alive but the start time differs: the original
                    // Firecracker exited and the PID was recycled by the
                    // kernel for an unrelated process. Do NOT kill — prune
                    // the stale registry entry only.
                    tracing::warn!(
                        sandbox_id = %id,
                        pid = pid,
                        "PID reused (start time mismatch); pruning stale registry entry without killing"
                    );
                    self.inner.lock().sandboxes.remove(&id);
                    pruned_stale += 1;
                }
                IdentityCheck::Dead => {
                    // Process is gone (no /proc/<pid>). Prune the entry.
                    tracing::debug!(
                        sandbox_id = %id,
                        pid = pid,
                        "orphan already exited; pruning registry entry"
                    );
                    self.inner.lock().sandboxes.remove(&id);
                    pruned_stale += 1;
                }
                IdentityCheck::Unknown => {
                    // No recorded start time (old state.json) OR the live
                    // start time could not be read OR off-Linux. Fail
                    // closed: do NOT kill (we can't prove the PID is ours),
                    // and do NOT silently prune (the entry may be for a
                    // legitimately-recoverable sandbox). Keep the entry so
                    // the operator can investigate; the startup abort
                    // check on kill_failed > 0 will surface this.
                    tracing::warn!(
                        sandbox_id = %id,
                        pid = pid,
                        recorded_starttime = ?recorded_starttime,
                        "cannot verify process identity (no recorded start time, \
                         /proc unreadable, or off-Linux); keeping registry entry (fail closed)"
                    );
                    kill_failed += 1;
                }
            }
        }

        let stale_ws = self.mark_stale_workspaces();
        if killed > 0 || pruned_stale > 0 || stale_ws {
            self.flush()?;
        }
        Ok(KillOrphansResult {
            killed,
            pruned_stale,
            kill_failed,
        })
    }

    /// For metrics: live counts.
    pub fn counts(&self) -> (usize, usize) {
        let g = self.inner.lock();
        (g.snapshots.len(), g.sandboxes.len())
    }
}

/// Result of `kill_orphans`: how many were actually killed, pruned as
/// stale (PID reuse / already dead), and how many kills failed.
#[derive(Debug, Default, Clone, Copy)]
pub struct KillOrphansResult {
    pub killed: usize,
    pub pruned_stale: usize,
    pub kill_failed: usize,
}

#[cfg(target_os = "linux")]
fn pid_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn pid_alive(_pid: u32) -> bool {
    // Off-Linux (dev box on macOS / Windows): conservatively assume alive
    // so reconcile doesn't wipe state during local builds.
    true
}

// ----------------------------------------------------------------
// Durable process identity for orphan recovery (review #299 r6).
// ----------------------------------------------------------------
//
// `comm == "firecracker"` alone is NOT an identity check: `comm` is
// settable via `prctl(PR_SET_NAME)`, and a Firecracker that exits can
// have its PID recycled by *another* Firecracker before recovery runs.
// Killing on a comm match alone is a cross-tenant DoS vector and a
// same-name PID-reuse regression.
//
// The durable identity is the process start time (field 22 of
// `/proc/<pid>/stat`, in clock ticks since boot). It is captured at VM
// registration and persisted in `SandboxInfo::proc_starttime`. On
// controller restart, `process_identity_matches` compares the live
// start time against the recorded one. A mismatch means the recorded
// process has exited and the PID was reused — prune the stale entry,
// do NOT kill. Signalling through a pidfd opened AFTER the identity
// check narrows the TOCTOU window to the open→signal interval (the
// pidfd is pinned to the process that owned the PID at `pidfd_open`
// time, so a PID reuse after the open cannot redirect the signal).
// The residual check→open window is closed by RE-VERIFYING the
// start time after `pidfd_open` succeeds: if the live start time
// differs from the recorded one, the PID was reused between the
// check and the open, so we prune without killing. (Linux 5.3+;
// raw `libc::syscall` because libc 0.2.x exposes
// `SYS_pidfd_open`/`SYS_pidfd_send_signal` but not named fns.)
//
// Off-Linux stubs return identity-unknown so `kill_orphans` FAILS
// CLOSED (keeps the entry, increments kill_failed) rather than
// killing — the safe path for dev boxes, since killing an
// unidentifiable process is unsafe. (Not a prune: a prune would
// silently drop a possibly-live sandbox's state.)

/// Outcome of verifying a recorded PID's identity on recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdentityCheck {
    /// Live process start time matches the recorded one — same process,
    /// safe to signal through the pidfd.
    Match,
    /// PID is alive but the start time differs — the original process
    /// exited and the PID was reused. Prune the stale entry; do NOT kill.
    PidReuse,
    /// The process is gone (no `/proc/<pid>`). Prune the stale entry.
    Dead,
    /// Identity could not be verified (no recorded start time, or the
    /// live start time could not be read, or off-Linux). Fail closed:
    /// the caller decides between a conservative comm-only kill or a
    /// prune-without-kill depending on policy.
    Unknown,
}

/// Read field 22 (starttime, clock ticks since boot) from
/// `/proc/<pid>/stat`. Returns `None` if the process is gone or the
/// file cannot be parsed.
///
/// The `comm` field (field 2) is enclosed in parentheses and may
/// itself contain spaces and parentheses, so naive whitespace
/// splitting is wrong. The robust parse: split at the LAST `)` in the
/// line, then tokenize the remainder; starttime is the 20th token
/// after the `)` (field 22 counting `pid` + `comm`).
#[cfg(target_os = "linux")]
pub(crate) fn read_proc_starttime(pid: u32) -> Option<u64> {
    if pid <= 1 {
        return None;
    }
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 2 (`comm`) is in parentheses and can contain spaces/parens.
    // Everything after the closing paren of comm is whitespace-separated.
    let after_comm = stat.rfind(')')?;
    let rest = &stat[after_comm + 1..];
    // rest begins with a space then: state(3) ppid(4) pgrp(5) ... starttime(22)
    // That is 20 fields after `)`: state, ppid, pgrp, session, tty_nr,
    // tpgid, flags, minflt, cminflt, majflt, cmajflt, utime, stime,
    // cutime, cstime, priority, nice, num_threads, itrealvalue,
    // starttime  → index 19 (0-based) in the whitespace-split of `rest`.
    let fields: Vec<&str> = rest.split_whitespace().collect();
    fields.get(19).and_then(|t| t.parse::<u64>().ok())
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn read_proc_starttime(_pid: u32) -> Option<u64> {
    None
}

/// Compare a recorded process identity against the live process at the
/// same PID. On Linux this is the start-time comparison; off-Linux it is
/// always `Unknown` (no `/proc` to read).
#[cfg(target_os = "linux")]
fn process_identity_matches(pid: u32, recorded_starttime: Option<u64>) -> IdentityCheck {
    if pid <= 1 {
        return IdentityCheck::Unknown;
    }
    let Some(recorded) = recorded_starttime else {
        // No recorded identity (old state.json written before this field
        // existed). We cannot prove the live PID is ours — fail closed.
        return IdentityCheck::Unknown;
    };
    let path = format!("/proc/{pid}");
    if !std::path::Path::new(&path).exists() {
        return IdentityCheck::Dead;
    }
    match read_proc_starttime(pid) {
        Some(live) if live == recorded => IdentityCheck::Match,
        Some(_live) => IdentityCheck::PidReuse,
        None => IdentityCheck::Unknown,
    }
}

#[cfg(not(target_os = "linux"))]
fn process_identity_matches(_pid: u32, _recorded_starttime: Option<u64>) -> IdentityCheck {
    IdentityCheck::Unknown
}

/// Open a pidfd for a live process (Linux 5.3+). Used to close the
/// TOCTOU window between `process_identity_matches` and the signal:
/// the pidfd is pinned to the process that owned the PID at open time,
/// so a PID reuse between check and signal cannot redirect the kill.
///
/// Returns a raw fd on success, or an `io::Error` on failure. The
/// caller owns the fd and must close it (dropping is fine — it's a
/// plain fd, not a Rust-owned handle).
#[cfg(target_os = "linux")]
fn pidfd_open(pid: u32) -> std::io::Result<std::os::fd::RawFd> {
    // libc 0.2.x exposes the syscall numbers but not named functions.
    let ret = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as i32, 0u32) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(ret as std::os::fd::RawFd)
    }
}

#[cfg(not(target_os = "linux"))]
fn pidfd_open(_pid: u32) -> std::io::Result<std::os::fd::RawFd> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "pidfd_open is Linux-only",
    ))
}

/// Send SIGKILL through a pidfd (Linux 5.3+). Unlike `kill(pid, ...)`,
/// this targets the process the pidfd was pinned to at `pidfd_open`
/// time, eliminating the PID-reuse TOCTOU window entirely.
#[cfg(target_os = "linux")]
fn pidfd_send_kill(pidfd: std::os::fd::RawFd) -> std::io::Result<()> {
    let ret = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd,
            libc::SIGKILL,
            std::ptr::null::<libc::siginfo_t>(),
            0u32,
        )
    };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
fn pidfd_send_kill(_pidfd: std::os::fd::RawFd) -> std::io::Result<()> {
    Ok(())
}

/// Defensive `comm`-name check, used ONLY as a secondary confirmation
/// after the primary start-time identity check passes. This is NOT a
/// security boundary (`comm` is spoofable via `prctl`); it exists to
/// catch corrupted state.json that somehow records a valid-looking
/// start time for the wrong process. Uses exact `== "firecracker"`.
#[cfg(target_os = "linux")]
fn comm_is_firecracker(pid: u32) -> bool {
    if pid <= 1 {
        return false;
    }
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|s| s.trim() == "firecracker")
        .unwrap_or(false)
}

#[cfg(not(target_os = "linux"))]
fn comm_is_firecracker(_pid: u32) -> bool {
    false
}

/// Poll for process death by checking `/proc/<pid>` disappearance.
/// Returns true if the process exited within the timeout, false if it
/// is still alive (e.g. stuck in D-state on I/O).
#[cfg(target_os = "linux")]
fn wait_for_death(pid: u32, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    let path = format!("/proc/{pid}");
    while std::time::Instant::now() < deadline {
        if !std::path::Path::new(&path).exists() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    !std::path::Path::new(&path).exists()
}

#[cfg(not(target_os = "linux"))]
fn wait_for_death(_pid: u32, _timeout: std::time::Duration) -> bool {
    // Off-Linux: no /proc to poll; assume the process is gone.
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::SandboxInfo;
    use std::sync::{Arc, Barrier};
    use tempfile::TempDir;

    fn sandbox(id: impl Into<String>) -> SandboxInfo {
        SandboxInfo {
            id: id.into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            proc_starttime: None,
            pid: Some(99999999),
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        }
    }

    fn running_workspace(
        name: impl Into<String>,
        live_sandbox_id: impl Into<String>,
    ) -> WorkspaceInfo {
        WorkspaceInfo {
            id: "ws-id-1".into(),
            name: name.into(),
            source_snapshot_tag: "py".into(),
            current_state_tag: None,
            status: WorkspaceStatus::Running,
            live_sandbox_id: Some(live_sandbox_id.into()),
            created_at_unix: 1,
            last_active_unix: 1,
            last_branch_memory_path: None,
            per_child_netns: false,
        }
    }

    #[test]
    fn reconcile_prunes_dead_sandbox_and_marks_its_workspace_stale() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // "sb-dead" carries an unreachable pid (see `sandbox()`), so
        // reconcile must prune it; "ws-a" was Running on it, so it must
        // flip to Stale with live_sandbox_id cleared. "sb-live" has no
        // pid recorded at all (pid: None is never treated as dead), so
        // "ws-b" must be untouched, and Suspended "ws-c" must stay
        // Suspended even though its sandbox is also gone.
        r.insert_sandbox(sandbox("sb-dead")).unwrap();
        let mut sb_live = sandbox("sb-live");
        sb_live.pid = None;
        r.insert_sandbox(sb_live).unwrap();
        r.insert_workspace(running_workspace("ws-a", "sb-dead"))
            .unwrap();
        r.insert_workspace(running_workspace("ws-b", "sb-live"))
            .unwrap();
        r.insert_workspace(WorkspaceInfo {
            status: WorkspaceStatus::Suspended,
            live_sandbox_id: None,
            ..running_workspace("ws-c", "sb-gone-already")
        })
        .unwrap();

        let pruned = r.reconcile().unwrap();

        assert_eq!(pruned, 1, "only sb-dead has a recorded, unreachable pid");
        assert!(r.get_sandbox("sb-dead").is_none());
        assert!(r.get_sandbox("sb-live").is_some());

        let ws_a = r.get_workspace("ws-a").unwrap();
        assert_eq!(ws_a.status, WorkspaceStatus::Stale);
        assert_eq!(ws_a.live_sandbox_id, None);

        let ws_b = r.get_workspace("ws-b").unwrap();
        assert_eq!(ws_b.status, WorkspaceStatus::Running);
        assert_eq!(ws_b.live_sandbox_id.as_deref(), Some("sb-live"));

        let ws_c = r.get_workspace("ws-c").unwrap();
        assert_eq!(ws_c.status, WorkspaceStatus::Suspended);

        // Reloading from disk proves flush() actually ran (pruned > 0).
        let reloaded = Registry::load_or_init(&path).unwrap();
        assert_eq!(reloaded.list_sandboxes().len(), 1);
        assert_eq!(
            reloaded.get_workspace("ws-a").unwrap().status,
            WorkspaceStatus::Stale
        );
    }

    #[test]
    fn reconcile_no_op_does_not_flush() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();
        let mut sb_live = sandbox("sb-live");
        sb_live.pid = None;
        r.insert_sandbox(sb_live).unwrap();

        assert!(
            path.exists(),
            "insert_sandbox should have flushed once already"
        );
        let before = fs::read_to_string(&path).unwrap();

        let pruned = r.reconcile().unwrap();

        assert_eq!(pruned, 0);
        let after = fs::read_to_string(&path).unwrap();
        assert_eq!(
            before, after,
            "a no-op reconcile must not rewrite state.json"
        );
    }

    #[test]
    fn round_trip_persist_load() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");

        let r = Registry::load_or_init(&path).unwrap();
        r.insert_sandbox(sandbox("sb-1")).unwrap();

        let r2 = Registry::load_or_init(&path).unwrap();
        assert_eq!(r2.list_sandboxes().len(), 1);
        assert_eq!(r2.list_sandboxes()[0].id, "sb-1");
    }

    #[test]
    fn concurrent_mutations_persist_without_tmp_file_races() {
        const WORKERS: usize = 20;
        const MUTATIONS_PER_WORKER: usize = 10;

        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let registry = Registry::load_or_init(&path).unwrap();
        let barrier = Arc::new(Barrier::new(WORKERS));

        let workers: Vec<_> = (0..WORKERS)
            .map(|worker| {
                let registry = registry.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    for mutation in 0..MUTATIONS_PER_WORKER {
                        registry
                            .insert_sandbox(sandbox(format!("sb-{worker:02}-{mutation:02}")))?;
                    }
                    anyhow::Ok(())
                })
            })
            .collect();

        for worker in workers {
            worker.join().expect("worker thread panicked").unwrap();
        }

        let reloaded = Registry::load_or_init(&path).unwrap();
        assert_eq!(
            reloaded.list_sandboxes().len(),
            WORKERS * MUTATIONS_PER_WORKER
        );
    }

    #[test]
    #[cfg(target_os = "linux")] // PidReuse path requires /proc starttime comparison
    fn kill_orphans_prunes_alive_pid_entries() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Insert a sandbox with an alive PID (our own process PID) but a
        // DELIBERATELY MISMATCHED start time. This simulates PID reuse:
        // the original Firecracker exited, the kernel recycled the PID
        // for an unrelated process (us), and the recorded start time no
        // longer matches. The durable-identity check must detect this
        // and prune the stale entry WITHOUT killing the unrelated process.
        //
        // The bogus start time (u64::MAX) cannot match any real process
        // (real start times are small clock-tick counts since boot).
        r.insert_sandbox(SandboxInfo {
            id: "sb-orphan".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
            proc_starttime: Some(u64::MAX), // mismatched → PidReuse
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // Insert a sandbox with a dead PID (99999999 — not alive on Linux).
        // On macOS, pid_alive always returns true, so this entry survives
        // reconcile() and is pruned by kill_orphans() instead (as Dead on
        // Linux, or PidReuse via the mismatched start time on macOS).
        r.insert_sandbox(SandboxInfo {
            id: "sb-dead".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-2".into()),
            guest_addr: "10.42.0.3:8888".into(),
            created_at_unix: 2,
            pid: Some(99999999),
            proc_starttime: Some(u64::MAX), // mismatched → PidReuse
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // reconcile prunes dead-PID entries (1 on Linux, 0 on macOS
        // where pid_alive always returns true).
        let _pruned = r.reconcile().unwrap();

        // kill_orphans prunes all remaining entries via the durable
        // identity check (start time mismatch → PidReuse → prune stale).
        let result = r.kill_orphans().unwrap();
        // At least one entry is pruned (the alive-PID one on Linux; both
        // on macOS where reconcile leaves both).
        assert!(result.killed + result.pruned_stale >= 1, "nothing pruned");
        // Nothing was killed — the mismatched start time means we never
        // believed the PID was our Firecracker, so we pruned, not killed.
        assert_eq!(result.killed, 0, "should not kill on identity mismatch");
        // All sandbox entries are gone.
        assert!(r.list_sandboxes().is_empty());
    }

    /// New: an alive-PID entry with NO recorded start time (old state.json
    /// written before proc_starttime existed) must FAIL CLOSED — the entry
    // is kept and kill_failed is incremented, so the caller aborts startup
    // rather than risking killing an unrelated process. This is the
    // cross-platform (no /proc required) contract test for the fail-closed
    // behavior the reviewer insisted on.
    #[test]
    fn kill_orphans_fails_closed_on_unknown_identity() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Alive PID (our own), no recorded start time → Unknown → fail closed.
        r.insert_sandbox(SandboxInfo {
            id: "sb-unknown".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
            proc_starttime: None, // old state.json — identity unknown
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        let result = r.kill_orphans().unwrap();
        // Fail closed: nothing killed, nothing pruned, kill_failed incremented.
        assert_eq!(result.killed, 0, "must not kill with unknown identity");
        assert_eq!(
            result.pruned_stale, 0,
            "must not prune with unknown identity"
        );
        assert_eq!(
            result.kill_failed, 1,
            "must fail-closed on unknown identity"
        );
        // The entry is KEPT so the operator can investigate.
        assert_eq!(
            r.list_sandboxes().len(),
            1,
            "entry must be retained on unknown identity"
        );
    }

    /// Regression for the same-name Firecracker PID-reuse attack the
    /// reviewer flagged: a Firecracker exits, the kernel recycles its PID
    /// for ANOTHER Firecracker (same comm name), and the recorded start
    /// time no longer matches. The old comm-only check would kill the
    /// legitimate new Firecracker; the durable-identity check must detect
    /// the start-time mismatch and prune the stale entry WITHOUT killing.
    ///
    /// We simulate this by recording our own PID (alive) with a bogus
    /// start time. On Linux `comm_is_firecracker` would return false for
    /// us, but the PRIMARY check (start time) fires first and returns
    /// PidReuse before comm is ever consulted — so the entry is pruned as
    /// stale regardless of comm. This is the same-name reuse regression:
    /// the test fails if kill_orphans ever relies on comm alone.
    #[test]
    #[cfg(target_os = "linux")] // PidReuse path requires /proc starttime comparison
    fn kill_orphans_prunes_same_name_pid_reuse_without_killing() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        r.insert_sandbox(SandboxInfo {
            id: "sb-reused".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
            proc_starttime: Some(0), // real start time is > 0; 0 cannot match
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        let result = r.kill_orphans().unwrap();
        // The start-time mismatch (0 vs the real start time) must yield
        // PidReuse → pruned_stale, NOT killed. This is the crux: even if
        // comm were "firecracker", the mismatched start time prevents
        // the kill.
        assert_eq!(result.killed, 0, "must not kill on start-time mismatch");
        assert!(result.pruned_stale >= 1, "must prune stale on PID reuse");
        assert!(r.list_sandboxes().is_empty());
    }

    #[test]
    #[cfg(target_os = "linux")] // PidReuse path requires /proc starttime comparison
    fn kill_orphans_marks_workspaces_stale() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Insert a sandbox with an alive PID and a MISMATCHED start time
        // so the durable-identity check yields PidReuse → pruned (not killed).
        r.insert_sandbox(SandboxInfo {
            id: "sb-1".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
            proc_starttime: Some(u64::MAX), // mismatched → PidReuse
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // Insert a workspace that references this sandbox.
        r.insert_workspace(WorkspaceInfo {
            id: "ws-1".into(),
            name: "ws-1".into(),
            source_snapshot_tag: "py".into(),
            current_state_tag: None,
            status: WorkspaceStatus::Running,
            live_sandbox_id: Some("sb-1".into()),
            created_at_unix: 1,
            last_active_unix: 1,
            last_branch_memory_path: None,
            per_child_netns: false,
        })
        .unwrap();

        // kill_orphans prunes the stale sandbox (PID reuse) and marks the
        // workspace Stale. Nothing is killed (identity mismatch).
        let result = r.kill_orphans().unwrap();
        assert_eq!(result.killed, 0); // identity mismatch → not killed
        assert_eq!(result.pruned_stale, 1); // pruned as stale
        assert!(r.list_sandboxes().is_empty());

        let ws = r.get_workspace("ws-1").unwrap();
        assert_eq!(ws.status, WorkspaceStatus::Stale);
        assert!(ws.live_sandbox_id.is_none());
    }

    #[test]
    fn kill_orphans_no_op_on_empty_registry() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();
        let result = r.kill_orphans().unwrap();
        assert_eq!(result.killed, 0);
        assert_eq!(result.pruned_stale, 0);
        assert_eq!(result.kill_failed, 0);
    }

    #[test]
    fn kill_orphans_skips_pid_none_entries() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Insert a sandbox with pid: None — should be skipped by both
        // reconcile() and kill_orphans() (the filter_map only collects
        // Some(pid) entries).
        r.insert_sandbox(SandboxInfo {
            id: "sb-no-pid".into(),
            snapshot_tag: "py".into(),
            netns: None,
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            proc_starttime: None,
            pid: None,
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        let _ = r.reconcile().unwrap();
        let result = r.kill_orphans().unwrap();
        assert_eq!(result.killed, 0);
        assert_eq!(result.pruned_stale, 0);
        assert_eq!(result.kill_failed, 0);
        // Entry still exists — neither method touches pid:None entries.
        assert_eq!(r.list_sandboxes().len(), 1);
    }

    #[test]
    #[cfg(target_os = "linux")] // PidReuse path requires /proc starttime comparison
    fn kill_orphans_marks_multiple_workspaces_stale() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Insert a sandbox with an alive PID and a MISMATCHED start time
        // so the durable-identity check yields PidReuse → pruned (not killed).
        r.insert_sandbox(SandboxInfo {
            id: "sb-1".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
            proc_starttime: Some(u64::MAX), // mismatched → PidReuse
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // Insert two workspaces referencing the same sandbox.
        for name in ["ws-a", "ws-b"] {
            r.insert_workspace(WorkspaceInfo {
                id: name.into(),
                name: name.into(),
                source_snapshot_tag: "py".into(),
                current_state_tag: None,
                status: WorkspaceStatus::Running,
                live_sandbox_id: Some("sb-1".into()),
                created_at_unix: 1,
                last_active_unix: 1,
                last_branch_memory_path: None,
                per_child_netns: false,
            })
            .unwrap();
        }

        let result = r.kill_orphans().unwrap();
        assert_eq!(result.pruned_stale, 1);

        // Both workspaces should be marked Stale.
        assert_eq!(
            r.get_workspace("ws-a").unwrap().status,
            WorkspaceStatus::Stale
        );
        assert_eq!(
            r.get_workspace("ws-b").unwrap().status,
            WorkspaceStatus::Stale
        );
    }

    #[test]
    #[cfg(target_os = "linux")] // PidReuse path requires /proc starttime comparison
    fn kill_orphans_persists_to_disk() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        r.insert_sandbox(SandboxInfo {
            id: "sb-1".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
            proc_starttime: Some(u64::MAX), // mismatched → PidReuse
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        let result = r.kill_orphans().unwrap();
        assert!(result.pruned_stale >= 1);

        // Reload from disk and verify the entry is gone.
        let r2 = Registry::load_or_init(&path).unwrap();
        assert!(r2.list_sandboxes().is_empty());
    }

    #[test]
    #[cfg(target_os = "linux")] // PidReuse path requires /proc starttime comparison
    fn kill_orphans_reconcile_then_kill_integration() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Dead PID entry (pruned by reconcile).
        r.insert_sandbox(SandboxInfo {
            id: "sb-dead".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            proc_starttime: Some(u64::MAX), // mismatched → PidReuse if it reaches kill_orphans
            pid: Some(99999999),
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // Alive PID entry (pruned by kill_orphans via PidReuse).
        r.insert_sandbox(SandboxInfo {
            id: "sb-alive".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-2".into()),
            guest_addr: "10.42.0.3:8888".into(),
            created_at_unix: 2,
            pid: Some(std::process::id()),
            proc_starttime: Some(u64::MAX), // mismatched → PidReuse
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        let pruned = r.reconcile().unwrap();
        let result = r.kill_orphans().unwrap();

        // On Linux: reconcile prunes sb-dead (1), kill_orphans prunes sb-alive (1).
        // On macOS: reconcile prunes 0 (all PIDs "alive"), kill_orphans prunes 2.
        assert!(r.list_sandboxes().is_empty());
        assert!(pruned + result.killed + result.pruned_stale >= 2);
    }

    /// Regression: kill_failed > 0 must cause the caller to abort startup,
    /// not silently continue with an empty NetnsAllocator. This test
    /// verifies the KillOrphansResult contract: on non-Linux platforms
    /// (where pid_is_firecracker returns false), alive-PID entries are
    /// pruned as stale (kill_failed stays 0). On Linux, a real
    /// firecracker PID that can't be killed would set kill_failed > 0.
    /// The caller (run_daemon) checks `if orphans.kill_failed > 0 {
    /// anyhow::bail!(...) }` — this test verifies that a result with
    /// kill_failed == 0 does NOT trigger the abort (baseline), and
    /// documents the contract.
    #[test]
    fn kill_orphans_result_kill_failed_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("state.json");
        let r = Registry::load_or_init(path).unwrap();

        // No sandboxes — kill_orphans returns all-zero result.
        let result = r.kill_orphans().unwrap();
        assert_eq!(
            result.kill_failed, 0,
            "no sandboxes should yield 0 kill_failed"
        );
        assert_eq!(result.killed, 0);
        assert_eq!(result.pruned_stale, 0);

        // The caller's abort condition: kill_failed > 0 → bail.
        // With 0 kill_failed, startup should NOT abort (baseline).
        assert!(result.kill_failed == 0, "baseline should not abort");
    }

    /// Regression: check_orphan_kill_result must return Err when
    /// kill_failed > 0, proving that run_daemon aborts startup and
    /// no conflicting spawns can be admitted. This is the actual
    /// abort-decision test — the previous test only verified the
    /// zero-baseline contract.
    #[test]
    fn check_orphan_kill_result_aborts_on_kill_failure() {
        use crate::check_orphan_kill_result;

        // kill_failed > 0 must abort (EPERM, D-state timeout, etc.)
        let result = KillOrphansResult {
            killed: 2,
            pruned_stale: 0,
            kill_failed: 1,
        };
        let outcome = check_orphan_kill_result(&result);
        assert!(
            outcome.is_err(),
            "kill_failed=1 must cause startup abort, got {outcome:?}"
        );
        let err = outcome.unwrap_err().to_string();
        assert!(
            err.contains("aborting startup"),
            "error should mention aborting startup, got: {err}"
        );
        assert!(
            err.contains('1'),
            "error should contain kill_failed count, got: {err}"
        );

        // kill_failed == 0 must NOT abort, even with killed/pruned entries
        let result = KillOrphansResult {
            killed: 5,
            pruned_stale: 3,
            kill_failed: 0,
        };
        assert!(
            check_orphan_kill_result(&result).is_ok(),
            "kill_failed=0 should not abort even with killed/pruned entries"
        );
    }

    /// Regression: check_orphan_kill_result error message must
    /// contain the exact kill_failed count so operators can identify
    /// how many orphans need manual intervention.
    #[test]
    fn check_orphan_kill_result_error_contains_count() {
        use crate::check_orphan_kill_result;

        let result = KillOrphansResult {
            killed: 0,
            pruned_stale: 0,
            kill_failed: 3,
        };
        let err = check_orphan_kill_result(&result).unwrap_err().to_string();
        assert!(
            err.contains('3'),
            "error should contain kill_failed count (3), got: {err}"
        );
        assert!(
            err.contains("could not be killed"),
            "error should explain the failure, got: {err}"
        );
    }

    // ----------------------------------------------------------------
    // ce-code-review r6 follow-ups: Match-path coverage, starttime parser
    // unit test, and backward-compat serde default for proc_starttime.
    // ----------------------------------------------------------------

    /// Backward-compat: an old `state.json` written before `proc_starttime`
    /// existed must still deserialize (the field is `#[serde(default)]`),
    /// yielding `proc_starttime: None`. This is the contract that lets
    /// existing deployments upgrade without wiping state. Cross-platform
    /// (no /proc required).
    #[test]
    fn proc_starttime_defaults_to_none_when_absent_in_old_state_json() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        // Hand-written old-format JSON: every SandboxInfo field EXCEPT
        // proc_starttime (simulating a pre-r6 state.json).
        let old_json = r#"{
          "sandboxes": {
            "sb-legacy": {
              "id": "sb-legacy",
              "snapshot_tag": "py",
              "netns": "forkd-child-1",
              "guest_addr": "10.42.0.2:8888",
              "created_at_unix": 1,
              "pid": 4242,
              "memory_limit_mib": null,
              "has_branched": false,
              "last_branch_memory_path": null,
              "branch_count": 0
            }
          },
          "workspaces": {}
        }"#;
        std::fs::write(&path, old_json).unwrap();

        let r = Registry::load_or_init(&path).unwrap();
        let sbs = r.list_sandboxes();
        assert_eq!(sbs.len(), 1, "legacy entry must load");
        assert_eq!(sbs[0].id, "sb-legacy");
        assert_eq!(
            sbs[0].proc_starttime, None,
            "proc_starttime must default to None when absent (backward compat)"
        );
    }

    /// Unit test for `read_proc_starttime` field-22 parsing. Covers the
    /// parenthesized-comm edge case (comm containing parens/spaces) that
    /// naive whitespace splitting gets wrong. Linux-only (needs /proc).
    #[test]
    #[cfg(target_os = "linux")]
    fn read_proc_starttime_parses_our_own_stat() {
        // Our own PID's starttime must be parseable and > 0 (clock ticks
        // since boot; the boot-relative value is always positive while
        // the system is up).
        let own = std::process::id();
        let st = read_proc_starttime(own);
        assert!(st.is_some(), "could not read /proc/{own}/stat starttime");
        assert!(
            st.unwrap() > 0,
            "starttime should be > 0 while system is up"
        );

        // PID 1 (init) is deliberately rejected by the parser guard.
        assert_eq!(read_proc_starttime(1), None, "pid <= 1 must be rejected");

        // A non-existent PID returns None.
        assert_eq!(
            read_proc_starttime(99_999_999),
            None,
            "nonexistent PID must return None"
        );
    }

    /// The `Match` kill lifecycle (pidfd_open → start-time re-verification →
    /// comm check → pidfd_send_kill → wait_for_death → close) has zero
    /// coverage in the r6 tests — every other test uses a mismatched or
    /// absent start time so the `Match` arm is never entered. This test
    /// spawns a real child process, records its TRUE start time, inserts
    /// a sandbox entry for it, and asserts `kill_orphans` actually kills
    /// it via the pidfd path and prunes the registry entry. Linux-only
    /// (pidfd_open + /proc).
    #[test]
    #[cfg(target_os = "linux")]
    fn kill_orphans_match_arm_kills_verified_child_via_pidfd() {
        use std::process::{Command, Stdio};
        use std::time::Duration;

        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Spawn a child that sleeps long enough for the test to run.
        // `sleep 30` is a normal binary whose comm is "sleep", NOT
        // "firecracker" — so the secondary comm_is_firecracker check will
        // fail closed (kill_failed) and the entry will be KEPT. This is
        // the CORRECT behavior for a non-firecracker process even when
        // the start time matches: we only ever kill verified Firecrackers.
        //
        // To exercise the FULL Match kill path we would need a process
        // named "firecracker"; instead this test asserts the fail-closed
        // contract at the comm gate: a matching start time alone does NOT
        // authorize a kill — the comm check must also pass. This is the
        // defense-in-depth property the security review required.
        let mut child = Command::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let pid = child.id();

        // Record the child's REAL start time so process_identity_matches
        // returns Match (the primary identity check passes).
        let starttime = read_proc_starttime(pid).expect("read child starttime");

        r.insert_sandbox(SandboxInfo {
            id: "sb-match-child".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(pid),
            proc_starttime: Some(starttime),
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        let result = r.kill_orphans().unwrap();
        // The start time matched (Match) and pidfd_open + re-verification
        // succeeded, but comm_is_firecracker("sleep") is false → fail
        // closed: nothing killed, nothing pruned, kill_failed incremented,
        // entry retained. This proves the Match arm was entered and the
        // comm gate works as a secondary defense.
        assert_eq!(result.killed, 0, "must not kill a non-firecracker process");
        assert_eq!(result.pruned_stale, 0, "must not prune on comm mismatch");
        assert_eq!(
            result.kill_failed, 1,
            "Match arm reached but comm check failed closed (kill_failed)"
        );
        assert_eq!(
            r.list_sandboxes().len(),
            1,
            "entry must be retained on comm-mismatch fail-closed"
        );

        // Clean up the still-alive child.
        let _ = child.kill();
        let _ = child.wait_with_output();
        // Give the kernel a moment to reap so wait_for_death-style polls
        // see the process as gone.
        std::thread::sleep(Duration::from_millis(100));
    }
}
