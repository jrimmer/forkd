//! Atomic netns offset allocation for per-child namespaces.
//!
//! `forkd-child-N` namespaces are provisioned on disk (via
//! `scripts/netns-setup.sh N`); a sandbox with `per_child_netns=true`
//! must pick an offset range `[offset+1 .. offset+n+1]` disjoint from
//! every other live sandbox AND from every other in-flight spawn.
//!
//! The old `pick_netns_offset` did a check-then-act scan of `live_vms`:
//! it locked the map only while scanning, then released the lock before
//! Firecracker started and before the VM was inserted, so two concurrent
//! spawns could both receive the same range. This allocator fixes that
//! by making the reservation itself the atomic operation:
//!
//! 1. `reserve(n)` scans the provisioned pool under one lock and marks
//!    the chosen indices as reserved (RAII lease).
//! 2. On spawn success the caller registers the VMs in `live_vms` and
//!    then calls `commit()`, which MOVES the indices from the reserved
//!    set to the ACTIVE set — they stay owned by the live VMs until the
//!    controller explicitly releases them via `release_index()` when the
//!    VM leaves `live_vms` (delete / suspend).
//! 3. On spawn failure the lease is dropped and the reservation is
//!    released automatically (rollback).
//!
//! Committed ranges are therefore never reused while their VMs are
//! still live: `reserve()` checks both the reserved AND the active set.
//! (An earlier version released committed indices immediately, so a
//! second spawn could take the same range as a still-live VM — fixed in
//! review #282 round 2.)

use parking_lot::Mutex;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

/// RAII lease on a range of `forkd-child-N` indices.
///
/// Indices are 1-based on the wire (`forkd-child-1` …); `offset` is the
/// additive offset applied before the within-batch `1..=n` loop, so the
/// lease covers `offset+1 ..= offset+n`.
#[derive(Debug)]
pub struct NetnsReservation {
    offset: usize,
    n: usize,
    /// Set on `commit`; while false the indices stay in the allocator's
    /// reserved set and are released on drop.
    committed: bool,
    /// Shared handle back to the allocator, so dropping the lease can
    /// release the reservation even after the caller's handle moved.
    alloc: Arc<NetnsAllocator>,
}

impl NetnsReservation {
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Transfer the reservation from "in-flight" to "owned by live VMs":
    /// the caller has registered the spawned VMs in `live_vms`, so the
    /// indices become ACTIVE and stay owned until the controller calls
    /// [`NetnsAllocator::release_index`] when the VM is removed.
    /// Idempotent.
    pub fn commit(&mut self) {
        if !self.committed {
            self.committed = true;
            self.alloc.commit_range(self.offset, self.n);
        }
    }
}

impl Drop for NetnsReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.alloc.release_reserved(self.offset, self.n);
        }
    }
}

/// Probe for whether a provisioned namespace index exists.
///
/// Injectable so tests can use an in-memory view instead of real
/// `/var/run/netns/` entries.
pub trait NetnsProbe: Send + Sync {
    fn exists(&self, index: usize) -> bool;
}

/// A probe that checks the real netns directory.
///
/// Caches the provisioned index set at construction time so `reserve()`
/// does not do a filesystem stat per candidate index while holding the
/// allocator locks. The provisioned pool is static until `netns-setup.sh`
/// runs and the daemon restarts (review #282).
pub struct DiskNetnsProbe {
    provisioned: HashSet<usize>,
}

impl DiskNetnsProbe {
    pub fn new(netns_dir: impl Into<std::path::PathBuf>) -> Self {
        let netns_dir = netns_dir.into();
        let mut provisioned = HashSet::new();
        if let Ok(rd) = std::fs::read_dir(&netns_dir) {
            for entry in rd.flatten() {
                if let Some(s) = entry.file_name().to_str() {
                    if let Some(idx) = s.strip_prefix("forkd-child-") {
                        if let Ok(i) = idx.parse::<usize>() {
                            provisioned.insert(i);
                        }
                    }
                }
            }
        }
        Self { provisioned }
    }
}

impl NetnsProbe for DiskNetnsProbe {
    fn exists(&self, index: usize) -> bool {
        self.provisioned.contains(&index)
    }
}

/// A probe that reports every index up to `max` as provisioned (tests).
#[derive(Debug, Clone)]
pub struct RangeProbe {
    pub max: usize,
}

impl NetnsProbe for RangeProbe {
    fn exists(&self, index: usize) -> bool {
        index <= self.max
    }
}

/// Atomic allocator for netns offset ranges.
pub struct NetnsAllocator {
    /// Indices currently reserved by in-flight spawns (not yet
    /// committed to `live_vms`).
    reserved: Mutex<HashSet<usize>>,
    /// Indices owned by LIVE VMs (committed reservations that have not
    /// been released by the controller). Released only when the VM
    /// leaves `live_vms`.
    active: Mutex<HashSet<usize>>,
    /// Upper bound of the provisioned pool: index `provisioned` is the
    /// highest namespace that may be used.
    provisioned: usize,
    probe: Box<dyn NetnsProbe>,
}

impl fmt::Debug for NetnsAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NetnsAllocator")
            .field("provisioned", &self.provisioned)
            .finish()
    }
}

impl NetnsAllocator {
    /// Create an allocator over a provisioned pool of `provisioned`
    /// namespaces (indices `1..=provisioned`), using `probe` to verify
    /// on-disk existence.
    pub fn new(provisioned: usize, probe: Box<dyn NetnsProbe>) -> Arc<Self> {
        Arc::new(Self {
            reserved: Mutex::new(HashSet::new()),
            active: Mutex::new(HashSet::new()),
            provisioned,
            probe,
        })
    }

    /// Discover the provisioned pool by scanning the netns directory
    /// (max existing `forkd-child-N` index). The probe caches the full
    /// provisioned index set at construction time so `reserve()` avoids
    /// per-index filesystem stats (review #282).
    pub fn discover(netns_dir: impl Into<std::path::PathBuf>) -> Arc<Self> {
        let netns_dir = netns_dir.into();
        let probe = DiskNetnsProbe::new(&netns_dir);
        let max = probe.provisioned.iter().copied().max().unwrap_or(0);
        Self::new(max, Box::new(probe))
    }

    /// Highest provisioned index.
    pub fn provisioned(&self) -> usize {
        self.provisioned
    }

    /// Reserve `n` contiguous child indices. Returns `None` when the
    /// provisioned pool cannot satisfy the request.
    ///
    /// The returned lease is held until `commit` or drop; concurrent
    /// reservations are disjoint because the check-then-act happens
    /// under the allocator's lock. Committed (live) indices are also
    /// skipped, so a still-running VM's range is never handed out again.
    pub fn reserve(self: &Arc<Self>, n: usize) -> Option<NetnsReservation> {
        if n == 0 {
            return None;
        }
        let mut reserved = self.reserved.lock();
        let active = self.active.lock();
        // Search offsets 0..=provisioned-n so that offset+n <= provisioned.
        let max_offset = self.provisioned.saturating_sub(n);
        for offset in 0..=max_offset {
            let range_start = offset + 1;
            let range_end = offset + n + 1; // exclusive
            let mut ok = true;
            for i in range_start..range_end {
                if reserved.contains(&i) || active.contains(&i) || !self.probe.exists(i) {
                    ok = false;
                    break;
                }
            }
            if ok {
                for i in range_start..range_end {
                    reserved.insert(i);
                }
                return Some(NetnsReservation {
                    offset,
                    n,
                    committed: false,
                    alloc: Arc::clone(self),
                });
            }
        }
        None
    }

    /// Move a reserved range into the active (live) set. Called by
    /// `NetnsReservation::commit` once the VMs are registered.
    fn commit_range(&self, offset: usize, n: usize) {
        let mut reserved = self.reserved.lock();
        let mut active = self.active.lock();
        for i in (offset + 1)..(offset + n + 1) {
            reserved.remove(&i);
            active.insert(i);
        }
    }

    /// Remove indices from the reserved set (rollback on spawn failure).
    fn release_reserved(&self, offset: usize, n: usize) {
        let mut reserved = self.reserved.lock();
        for i in (offset + 1)..(offset + n + 1) {
            reserved.remove(&i);
        }
    }

    /// Release a single ACTIVE index back to the pool. Called by the
    /// controller when a VM leaves `live_vms` (delete / suspend). The
    /// VM's `netns` string (`forkd-child-N`) provides the index.
    /// Idempotent; releasing an index that is not active is a no-op.
    pub fn release_index(&self, index: usize) {
        self.active.lock().remove(&index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alloc(max: usize) -> Arc<NetnsAllocator> {
        NetnsAllocator::new(max, Box::new(RangeProbe { max }))
    }

    #[test]
    fn reserves_disjoint_ranges() {
        let a = alloc(16);
        let mut r1 = a.reserve(2).expect("r1");
        assert_eq!(r1.offset(), 0); // covers 1,2
        let r2 = a.reserve(1).expect("r2");
        assert_eq!(r2.offset(), 2); // covers 3
                                    // commit r1 — indices 1,2 become ACTIVE (owned by live VMs), not
                                    // free: a committed range must not be reused while its VMs live.
        r1.commit();
        let r3 = a.reserve(1).expect("r3");
        assert_eq!(r3.offset(), 3); // covers 4 (1,2 active; 3 reserved)
        let r4 = a.reserve(5).expect("r4");
        assert_eq!(r4.offset(), 4); // covers 5..9
    }

    #[test]
    fn committed_range_released_by_release_index() {
        let a = alloc(8);
        let mut r1 = a.reserve(2).expect("r1"); // covers 1,2
        r1.commit();
        // While active, the range is NOT reusable.
        assert!(a.reserve(1).is_some()); // offset 2 covers 3
        assert_eq!(a.reserve(2).expect("r2").offset(), 2); // covers 3,4
                                                           // Release index 1 only (VM 1 deleted, VM 2 still live).
        a.release_index(1);
        let r = a.reserve(1).expect("r3");
        assert_eq!(r.offset(), 0); // covers 1 — index 1 free, index 2 active
                                   // Releasing an already-free index is a no-op.
        a.release_index(1);
        a.release_index(99);
    }

    #[test]
    fn drop_releases_reservation() {
        let a = alloc(8);
        let r1 = a.reserve(3).expect("r1"); // covers 1..3
        drop(r1);
        let r2 = a.reserve(3).expect("r2");
        assert_eq!(r2.offset(), 0); // same range reusable
    }

    #[test]
    fn concurrent_reservations_are_disjoint() {
        let a = alloc(256);
        // A Barrier guarantees all 32 reservations are alive at the same
        // time: each thread reserves, then waits. Without it, short
        // sleeps let early threads' leases expire while later threads
        // are still spawning, so later threads legitimately reuse freed
        // ranges (a test bug, not an allocator bug).
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(33));
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let a = Arc::clone(&a);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let r = a.reserve(4).expect("reserve in pool");
                    // All threads reserve before any reports back, so
                    // the reservations overlap in time.
                    barrier.wait();
                    let o = r.offset();
                    (o, o + 4)
                })
            })
            .collect();
        barrier.wait(); // main joins the reservation window
        let mut ranges: Vec<(usize, usize)> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        ranges.sort();
        for pair in ranges.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "ranges overlap: {:?} vs {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn pool_larger_than_256_is_usable() {
        let a = alloc(512);
        // Reserve every single slot (n=1) up to 512.
        let mut leases = Vec::new();
        for _ in 0..512 {
            match a.reserve(1) {
                Some(l) => leases.push(l),
                None => panic!("pool of 512 should satisfy 512 single leases"),
            }
        }
        assert!(a.reserve(1).is_none(), "pool exhausted at 512");
    }

    #[test]
    fn exhaustion_returns_none() {
        let a = alloc(4);
        let _r1 = a.reserve(4).expect("fills 1..4");
        assert!(a.reserve(1).is_none(), "no room left");
        // A request larger than the pool is also rejected.
        assert!(a.reserve(5).is_none());
    }

    #[test]
    fn disk_probe_honors_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("forkd-child-1"), b"").unwrap();
        std::fs::write(dir.path().join("forkd-child-3"), b"").unwrap();
        let a = NetnsAllocator::discover(dir.path());
        assert_eq!(a.provisioned(), 3);
        // Only provisioned indices are usable: offset 0 (covers 1) works,
        // offset 1 (covers 2, which is not provisioned) does not.
        assert!(a.reserve(1).is_some());
        drop(a.reserve(1));
        assert!(a.reserve(2).is_none(), "index 2 is not provisioned");
    }

    #[test]
    fn concurrent_reservations_skip_committed_active() {
        // Thread A reserves n=4 and commits (simulating live VMs owning
        // indices 1..4), then 32 threads reserve(1) under a barrier and
        // assert none overlap the committed range.
        let a = alloc(256);
        let mut committed = a.reserve(4).expect("commit r");
        committed.commit(); // indices 1..4 are now ACTIVE

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(33));
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let a = Arc::clone(&a);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let r = a.reserve(1).expect("reserve in pool");
                    barrier.wait();
                    r.offset() + 1 // the actual index
                })
            })
            .collect();
        barrier.wait();
        let indices: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // No thread should have received an index in the committed
        // range (1..=4).
        for &idx in &indices {
            assert!(idx > 4, "thread got committed index {idx}");
        }
        // All 32 indices must be disjoint.
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 32, "indices not disjoint: {indices:?}");

        // After releasing the committed range, index 1 is reusable.
        for idx in 1..=4 {
            a.release_index(idx);
        }
        assert_eq!(a.reserve(1).unwrap().offset(), 0);
    }
}
