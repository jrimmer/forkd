//! `forkd-controller` library — daemon plumbing (HTTP server, registry).
//!
//! Binary in `src/main.rs` parses CLI args and calls [`run_daemon`].
//! Library shape lets us write integration tests in `tests/`.
pub mod api;
pub mod audit;
pub mod auth;
pub mod http;
pub mod netns;
pub mod state;

use anyhow::{Context, Result};
use axum::middleware;
use axum_server::tls_rustls::RustlsConfig;
use axum_server::Handle;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::audit::AuditSink;
use crate::auth::AuthConfig;
use crate::http::AppState;
use crate::state::Registry;

#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub bind: SocketAddr,
    pub state_file: PathBuf,
    /// Root directory under which `<tag>/vmstate` and `<tag>/memory.bin`
    /// live for each tagged snapshot. Falls back to the canonical
    /// XDG location (`~/.local/share/forkd/snapshots/`) if unset.
    pub snapshot_root: PathBuf,
    /// Path to the audit log file (one JSON line per request, appended).
    pub audit_log: PathBuf,
    /// Optional path to a file whose contents are the daemon's bearer
    /// token. When `None`, the daemon runs unauthenticated — safe only
    /// for loopback-bound, single-tenant developer setups.
    pub token_file: Option<PathBuf>,
    /// PEM-encoded TLS server certificate chain. Required together
    /// with `tls_key` to enable HTTPS. When either is unset the daemon
    /// serves plain HTTP (intended for loopback-only deployments).
    pub tls_cert: Option<PathBuf>,
    /// PEM-encoded TLS private key matching `tls_cert`.
    pub tls_key: Option<PathBuf>,
    /// Scratch directory used when a `POST /v1/sandboxes` request sets
    /// `prewarm: true`. The daemon writes a throwaway snapshot here per
    /// child immediately after restore to amortize the cold-cache penalty
    /// on first BRANCH. tmpfs (`/dev/shm/forkd-prewarm`) is the right
    /// default — the file is deleted immediately and writes never hit
    /// real disk. Must have enough free space to hold one
    /// guest-RAM-sized file per concurrent prewarmed child.
    pub prewarm_scratch_dir: PathBuf,
    /// Maximum concurrent BRANCH operations the daemon will admit. Each
    /// BRANCH writes a full `memory.bin` (typically 256 MiB - 8 GiB), so
    /// the cap bounds peak transient disk usage during fan-outs. `None`
    /// falls back to [`http::DEFAULT_BRANCH_CONCURRENCY`].
    pub branch_concurrency: Option<usize>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8889".parse().unwrap(),
            state_file: PathBuf::from("/var/lib/forkd/state.json"),
            snapshot_root: forkd_vmm::paths::data_dir().join("snapshots"),
            audit_log: PathBuf::from("/var/log/forkd/audit.log"),
            token_file: None,
            tls_cert: None,
            tls_key: None,
            prewarm_scratch_dir: PathBuf::from("/dev/shm/forkd-prewarm"),
            branch_concurrency: None,
        }
    }
}

fn unauthenticated_non_loopback(bind: SocketAddr, token_file: Option<&Path>) -> bool {
    token_file.is_none() && !bind.ip().is_loopback()
}

/// Post-`kill_orphans` startup decision: abort if any orphan could
/// not be killed (EPERM, D-state timeout, etc.). The NetnsAllocator
/// active set and `shared_tap_owner` start empty, so a new spawn could
/// reuse the still-alive orphan's netns index or tap lease — the exact
/// collision #298 is meant to prevent. Returning `Err` here causes
/// `run_daemon` to exit before binding the HTTP listener, so no spawns
/// can be admitted.
pub(crate) fn check_orphan_kill_result(orphans: &crate::state::KillOrphansResult) -> Result<()> {
    if orphans.kill_failed > 0 {
        anyhow::bail!(
            "aborting startup: {} orphaned Firecracker process(es) could not be killed \
             (EPERM or did not exit within timeout); refusing to accept spawns that may \
             collide with still-alive orphans. Kill them manually and restart.",
            orphans.kill_failed
        );
    }
    if orphans.unresolved > 0 {
        anyhow::bail!(
            "aborting startup: {} retained sandbox entr(y/ies) have no recorded PID and \
             cannot be attributed to a live or dead process; refusing to start with an \
             empty allocator/shared-tap ownership that may collide with a live VM still \
             holding those resources (#298). Inspect the registry and remove/repair them, \
             then restart.",
            orphans.unresolved
        );
    }
    Ok(())
}

/// Bring up the controller daemon. Blocks until the listener exits.
/// SIGTERM and SIGINT trigger a graceful shutdown; SIGHUP reopens the
/// configured audit log after external rotation.
pub async fn run_daemon(cfg: DaemonConfig) -> Result<()> {
    let registry = Registry::load_or_init(&cfg.state_file)
        .with_context(|| format!("load state from {}", cfg.state_file.display()))?;
    let pruned = registry.reconcile()?;
    if pruned > 0 {
        tracing::info!(pruned, "reconciled stale sandbox entries on startup");
    }

    // Kill orphaned Firecracker processes left over from a previous
    // controller instance (issue #298). After reconcile() prunes
    // dead-PID entries, any remaining entries have alive PIDs but no
    // live_vms handle — they are unmanageable orphans. Kill them and
    // prune the registry entries so the NetnsAllocator (empty active
    // set) and shared_tap_owner (None) start clean.
    let orphans = registry.kill_orphans()?;
    if orphans.killed > 0 {
        tracing::info!(
            killed = orphans.killed,
            pruned_stale = orphans.pruned_stale,
            kill_failed = orphans.kill_failed,
            unresolved = orphans.unresolved,
            "killed orphaned Firecracker processes on startup"
        );
    } else if orphans.pruned_stale > 0 || orphans.kill_failed > 0 || orphans.unresolved > 0 {
        tracing::warn!(
            pruned_stale = orphans.pruned_stale,
            kill_failed = orphans.kill_failed,
            unresolved = orphans.unresolved,
            "orphan recovery: some entries pruned as stale, unresolved, or kill failed"
        );
    }

    // Fail closed: if any orphan could not be killed (EPERM, D-state
    // timeout, etc.), the controller cannot guarantee a clean resource
    // state. The NetnsAllocator active set and shared_tap_owner start
    // empty, so a new spawn could reuse the still-alive orphan's netns
    // index or tap lease — the exact collision #298 is meant to prevent.
    // Abort startup so the operator intervenes rather than silently
    // admitting conflicting spawns. (review #299)
    check_orphan_kill_result(&orphans)?;

    let audit = AuditSink::open(&cfg.audit_log)
        .with_context(|| format!("open audit log {}", cfg.audit_log.display()))?;
    tracing::info!(audit_log = %audit.path().display(), "audit log open");

    let auth_cfg = match &cfg.token_file {
        Some(p) => {
            let raw = std::fs::read_to_string(p)
                .with_context(|| format!("read token file {}", p.display()))?;
            let tok = raw.trim().to_string();
            validate_token(&tok).with_context(|| format!("validate token from {}", p.display()))?;
            tracing::info!(token_file = %p.display(), "bearer-token auth enabled");
            AuthConfig::with_token(tok)
        }
        None => {
            if unauthenticated_non_loopback(cfg.bind, cfg.token_file.as_deref()) {
                anyhow::bail!(
                    "refusing unauthenticated non-loopback bind {}; pass --token-file \
                     or bind to 127.0.0.1/::1 for local-only use",
                    cfg.bind
                );
            }
            AuthConfig::open()
        }
    };

    let branch_concurrency = cfg
        .branch_concurrency
        .unwrap_or(http::DEFAULT_BRANCH_CONCURRENCY);
    if branch_concurrency == 0 {
        anyhow::bail!(
            "branch_concurrency must be > 0; got 0 (use the default {} if unsure)",
            http::DEFAULT_BRANCH_CONCURRENCY
        );
    }
    if branch_concurrency != http::DEFAULT_BRANCH_CONCURRENCY {
        tracing::info!(
            branch_concurrency,
            default = http::DEFAULT_BRANCH_CONCURRENCY,
            "branch concurrency cap overridden"
        );
    }
    let app_state = Arc::new(AppState {
        registry,
        live_vms: Mutex::new(HashMap::new()),
        snapshot_root: cfg.snapshot_root.clone(),
        branch_in_flight: Mutex::new(std::collections::HashSet::new()),
        branch_sem: std::sync::Arc::new(tokio::sync::Semaphore::new(branch_concurrency)),
        branch_concurrency_cap: branch_concurrency,
        // Atomic netns offset allocator (security review #282): the
        // bound is derived from the provisioned pool on disk instead of
        // a magic constant.
        netns_alloc: crate::netns::NetnsAllocator::discover("/var/run/netns"),
        // #281: shared-tap ownership cell (claim/commit/RAII lifecycle).
        shared_tap_owner: std::sync::Arc::new(parking_lot::Mutex::new(None)),
        // #302: diagnostic counter for orphaned firecracker processes
        // detected during the pre-restore scan. Exposed as a metric.
        orphan_firecrackers_detected: std::sync::atomic::AtomicU64::new(0),
        prewarm_scratch_dir: cfg.prewarm_scratch_dir.clone(),
        #[cfg(target_os = "linux")]
        live_in_flight: Mutex::new(HashMap::new()),
        #[cfg(test)]
        _tempdir: None,
    });

    let auth_layer_cfg = auth_cfg.clone();
    let audit_clone = audit.clone();
    let app = http::router(app_state)
        .layer(middleware::from_fn(move |req, next| {
            let cfg = auth_layer_cfg.clone();
            async move { auth::require_token(cfg, req, next).await }
        }))
        .layer(middleware::from_fn(move |req, next| {
            let sink = audit_clone.clone();
            async move { audit::audit_layer(sink, req, next).await }
        }));

    // axum-server gives us a unified bind path for TLS and plain HTTP,
    // plus a Handle for cooperative shutdown that drains in-flight
    // requests up to a deadline.
    let handle = Handle::new();
    let _signal_task = spawn_signal_handler(handle.clone(), audit.clone());

    let tls = match (&cfg.tls_cert, &cfg.tls_key) {
        (Some(c), Some(k)) => Some(load_tls(c, k).await?),
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("--tls-cert and --tls-key must be supplied together");
        }
        (None, None) => None,
    };

    match tls {
        Some(tls_cfg) => {
            tracing::info!(addr = %cfg.bind, "forkd-controller listening (HTTPS)");
            axum_server::bind_rustls(cfg.bind, tls_cfg)
                .handle(handle)
                .serve(app.into_make_service())
                .await
                .context("axum_server bind_rustls")?;
        }
        None => {
            tracing::info!(addr = %cfg.bind, "forkd-controller listening (plain HTTP)");
            axum_server::bind(cfg.bind)
                .handle(handle)
                .serve(app.into_make_service())
                .await
                .context("axum_server bind")?;
        }
    }
    Ok(())
}

async fn load_tls(cert: &Path, key: &Path) -> Result<RustlsConfig> {
    // axum-server's RustlsConfig wants both PEM files. rustls 0.23
    // requires a crypto provider be installed before any TLS handshake;
    // install aws-lc-rs as the default if nothing's been set yet.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    RustlsConfig::from_pem_file(cert, key)
        .await
        .with_context(|| format!("load TLS cert {} / key {}", cert.display(), key.display()))
}

struct SignalTask(tokio::task::JoinHandle<()>);

impl Drop for SignalTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(unix)]
fn spawn_signal_handler(handle: Handle<SocketAddr>, audit: AuditSink) -> SignalTask {
    let mut interrupt =
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
            Ok(signal) => Some(signal),
            Err(error) => {
                tracing::error!(%error, "failed to install SIGINT handler");
                None
            }
        };
    let mut terminate =
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(signal) => Some(signal),
            Err(error) => {
                tracing::error!(%error, "failed to install SIGTERM handler");
                None
            }
        };
    let mut hangup = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()) {
        Ok(signal) => Some(signal),
        Err(error) => {
            tracing::error!(%error, "failed to install SIGHUP handler");
            None
        }
    };

    SignalTask(tokio::spawn(async move {
        loop {
            tokio::select! {
                signal = recv_unix_signal(&mut interrupt) => {
                    if signal.is_none() {
                        tracing::error!("SIGINT signal stream closed");
                        interrupt = None;
                        continue;
                    }
                    tracing::info!("received SIGINT, shutting down");
                    break;
                }
                signal = recv_unix_signal(&mut terminate) => {
                    if signal.is_none() {
                        tracing::error!("SIGTERM signal stream closed");
                        terminate = None;
                        continue;
                    }
                    tracing::info!("received SIGTERM, shutting down");
                    break;
                }
                signal = recv_unix_signal(&mut hangup) => {
                    if signal.is_none() {
                        tracing::error!("SIGHUP signal stream closed");
                        hangup = None;
                        continue;
                    }
                    match audit.reopen() {
                        Ok(()) => tracing::info!(
                            audit_log = %audit.path().display(),
                            "reopened audit log after SIGHUP"
                        ),
                        Err(error) => tracing::error!(
                            %error,
                            audit_log = %audit.path().display(),
                            "failed to reopen audit log after SIGHUP"
                        ),
                    }
                }
            }
        }
        handle.graceful_shutdown(Some(Duration::from_secs(30)));
    }))
}

#[cfg(unix)]
async fn recv_unix_signal(signal: &mut Option<tokio::signal::unix::Signal>) -> Option<()> {
    match signal {
        Some(signal) => signal.recv().await,
        None => std::future::pending().await,
    }
}

#[cfg(not(unix))]
fn spawn_signal_handler(handle: Handle<SocketAddr>, _audit: AuditSink) -> SignalTask {
    SignalTask(tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => tracing::info!("received interrupt, shutting down"),
            Err(error) => tracing::error!(%error, "interrupt handler failed"),
        }
        handle.graceful_shutdown(Some(Duration::from_secs(30)));
    }))
}

/// Reject tokens that are empty, obvious placeholders, or below a minimum
/// entropy budget. Pure function so it's exercised by unit tests without
/// having to spin up the daemon.
fn validate_token(tok: &str) -> Result<()> {
    if tok.is_empty() {
        anyhow::bail!("token is empty");
    }
    // Reject the literal placeholder shipped in packaging/k8s/. A user who
    // runs `kubectl apply -f` without first running the documented
    // `sed`/Secret-replacement step would otherwise get a daemon protected
    // only by a publicly-known bearer token.
    if tok.starts_with("REPLACE_ME") || tok.starts_with("CHANGE_ME") {
        anyhow::bail!(
            "token still contains the manifest placeholder ({tok}); \
             replace it with a real 32-byte secret before starting the daemon"
        );
    }
    // Reject suspiciously short tokens — sufficient entropy is the user's
    // responsibility, but anything under 16 bytes is almost certainly a
    // copy-paste mistake rather than a deliberate choice.
    if tok.len() < 16 {
        anyhow::bail!(
            "token is only {} bytes; use at least 16 bytes of high-entropy randomness",
            tok.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{unauthenticated_non_loopback, validate_token};
    use std::path::Path;

    #[test]
    fn unauthenticated_non_loopback_is_rejected() {
        assert!(unauthenticated_non_loopback(
            "0.0.0.0:8889".parse().unwrap(),
            None,
        ));
        assert!(unauthenticated_non_loopback(
            "[::]:8889".parse().unwrap(),
            None
        ));
        assert!(!unauthenticated_non_loopback(
            "127.0.0.1:8889".parse().unwrap(),
            None,
        ));
        assert!(!unauthenticated_non_loopback(
            "0.0.0.0:8889".parse().unwrap(),
            Some(Path::new("/etc/forkd/token")),
        ));
    }

    #[test]
    fn rejects_empty_token() {
        assert!(validate_token("").is_err());
    }

    #[test]
    fn rejects_replace_me_placeholder() {
        // Regression: this exact string is shipped in packaging/k8s/
        // forkd-controller.yaml.
        let err =
            validate_token("REPLACE_ME_WITH_32_BYTES_BASE64").expect_err("placeholder accepted");
        let msg = format!("{err:#}");
        assert!(msg.contains("placeholder"), "msg was: {msg}");
    }

    #[test]
    fn rejects_change_me_variant() {
        assert!(validate_token("CHANGE_ME_PLEASE").is_err());
    }

    #[test]
    fn rejects_too_short_token() {
        assert!(validate_token("short").is_err());
        // 15 bytes is one under the cap.
        assert!(validate_token("123456789012345").is_err());
    }

    #[test]
    fn accepts_realistic_token() {
        // 32 hex chars = 16 bytes of entropy if random.
        assert!(validate_token("a1b2c3d4e5f60718293a4b5c6d7e8f90").is_ok());
    }
}
