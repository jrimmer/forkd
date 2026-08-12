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
        let orphans: Vec<(String, u32)> = {
            let g = self.inner.lock();
            g.sandboxes
                .iter()
                .filter_map(|(id, sb)| match sb.pid {
                    Some(pid) if pid_alive(pid) => Some((id.clone(), pid)),
                    _ => None,
                })
                .collect()
        };

        let mut killed = 0usize;
        let mut pruned_stale = 0usize;
        let mut kill_failed = 0usize;
        let mut skip_ids: Vec<String> = Vec::new();

        for (id, pid) in orphans {
            if pid_is_firecracker(pid) {
                tracing::warn!(
                    sandbox_id = %id,
                    pid = pid,
                    "killing orphaned Firecracker process on startup"
                );
                match kill_pid(pid) {
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
                                "orphaned Firecracker did not exit within 5s of SIGKILL;                                  keeping registry entry to prevent resource collision"
                            );
                            kill_failed += 1;
                            skip_ids.push(id);
                        }
                    }
                    Err(e) if e.raw_os_error() == Some(libc::ESRCH) => {
                        // Process already dead (benign TOCTOU race between
                        // pid_alive and kill_pid) — safe to prune.
                        tracing::debug!(
                            sandbox_id = %id,
                            pid = pid,
                            "orphan already exited (ESRCH); pruning registry entry"
                        );
                        self.inner.lock().sandboxes.remove(&id);
                        pruned_stale += 1;
                    }
                    Err(e) => {
                        // Real kill failure (EPERM, etc.) — do NOT prune.
                        // The orphan may still be alive holding resources;
                        // pruning would recreate the exact bug #298 fixes.
                        tracing::error!(
                            sandbox_id = %id,
                            pid = pid,
                            error = %e,
                            "failed to kill orphaned Firecracker process;                              keeping registry entry to prevent resource collision"
                        );
                        kill_failed += 1;
                        skip_ids.push(id);
                    }
                }
            } else {
                tracing::warn!(
                    sandbox_id = %id,
                    pid = pid,
                    "PID no longer belongs to Firecracker (PID reuse); pruning stale registry entry"
                );
                self.inner.lock().sandboxes.remove(&id);
                pruned_stale += 1;
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

/// Verify that a PID belongs to a Firecracker process by reading
/// `/proc/<pid>/comm`. Guards against PID reuse: if the original
/// Firecracker process died and the PID was recycled by the OS for
/// a different process, we don't want to kill an unrelated process.
///
/// Note: `comm` is settable via `prctl(PR_SET_NAME)` and is not an
/// identity guarantee. On a multi-tenant host where another process
/// could set its name to "firecracker", this check is a best-effort
/// guard, not a security boundary. The TOCTOU window between this
/// check and `kill_pid` is acknowledged — `pidfd_open` +
/// `pidfd_send_signal` would close it entirely (Linux 5.3+).
#[cfg(target_os = "linux")]
fn pid_is_firecracker(pid: u32) -> bool {
    // Defensive guard against pid 0/1 (defense-in-depth for corrupted
    // state.json). Real firecracker PIDs are always > 1.
    if pid <= 1 {
        return false;
    }
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|s| s.trim() == "firecracker")
        .unwrap_or(false)
}

#[cfg(not(target_os = "linux"))]
fn pid_is_firecracker(_pid: u32) -> bool {
    // Off-Linux: can't verify via /proc; return false so
    // kill_orphans prunes the entry without sending a signal.
    false
}

/// Send SIGKILL to a process by PID. Uses libc::kill directly
/// (we don't have a std::process::Child handle for orphaned PIDs).
///
/// SAFETY: `pid` is a live Linux PID verified by `pid_is_firecracker`;
/// `SIGKILL` is a valid signal constant; `kill(2)` is sound for any
/// `pid_t` value (returns ESRCH if the process doesn't exist).
#[cfg(target_os = "linux")]
fn kill_pid(pid: u32) -> std::io::Result<()> {
    let ret = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    if ret != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
fn kill_pid(_pid: u32) -> std::io::Result<()> {
    Ok(())
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
    fn kill_orphans_prunes_alive_pid_entries() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Insert a sandbox with an alive PID (use our own process PID).
        // On Linux, pid_is_firecracker will return false (we're not
        // firecracker), so the entry is pruned without killing.
        // On non-Linux, pid_is_firecracker always returns false.
        r.insert_sandbox(SandboxInfo {
            id: "sb-orphan".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // Insert a sandbox with a dead PID (99999999 — not alive on Linux).
        // On macOS, pid_alive always returns true, so this entry survives
        // reconcile() and is pruned by kill_orphans() instead.
        r.insert_sandbox(SandboxInfo {
            id: "sb-dead".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-2".into()),
            guest_addr: "10.42.0.3:8888".into(),
            created_at_unix: 2,
            pid: Some(99999999),
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // reconcile prunes dead-PID entries (1 on Linux, 0 on macOS
        // where pid_alive always returns true).
        let _pruned = r.reconcile().unwrap();

        // kill_orphans prunes all remaining entries (alive PID but
        // not firecracker → pruned without killing).
        let result = r.kill_orphans().unwrap();
        // At least the alive-PID entry is pruned.
        assert!(result.killed + result.pruned_stale >= 1);
        // All sandbox entries are gone.
        assert!(r.list_sandboxes().is_empty());
    }

    #[test]
    fn kill_orphans_marks_workspaces_stale() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Insert a sandbox with an alive PID.
        r.insert_sandbox(SandboxInfo {
            id: "sb-1".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
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

        // kill_orphans kills the sandbox and marks the workspace Stale.
        let result = r.kill_orphans().unwrap();
        assert_eq!(result.killed, 0); // not firecracker → not killed
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
    fn kill_orphans_marks_multiple_workspaces_stale() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("state.json");
        let r = Registry::load_or_init(&path).unwrap();

        // Insert a sandbox with an alive PID.
        r.insert_sandbox(SandboxInfo {
            id: "sb-1".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-1".into()),
            guest_addr: "10.42.0.2:8888".into(),
            created_at_unix: 1,
            pid: Some(std::process::id()),
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
            pid: Some(99999999),
            memory_limit_mib: None,
            has_branched: false,
            last_branch_memory_path: None,
            branch_count: 0,
        })
        .unwrap();

        // Alive PID entry (pruned by kill_orphans).
        r.insert_sandbox(SandboxInfo {
            id: "sb-alive".into(),
            snapshot_tag: "py".into(),
            netns: Some("forkd-child-2".into()),
            guest_addr: "10.42.0.3:8888".into(),
            created_at_unix: 2,
            pid: Some(std::process::id()),
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
}
