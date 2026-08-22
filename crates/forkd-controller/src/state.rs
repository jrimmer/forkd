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

    /// For metrics: live counts.
    pub fn counts(&self) -> (usize, usize) {
        let g = self.inner.lock();
        (g.snapshots.len(), g.sandboxes.len())
    }
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
    }
}
