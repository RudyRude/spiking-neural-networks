use tracing::info;
use tracing_subscriber::{prelude::*, EnvFilter};

/// Initialize tracing/logging with sensible defaults.
/// - Respects RUST_LOG if set (e.g., RUST_LOG=ndfh_api=debug,info)
/// - Defaults to `info,ndfh_api=debug` when RUST_LOG is unset
pub fn init_tracer() {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ndfh_api=debug"));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .compact();

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);

    // Initialize registry (no OTLP/stdout exporter by default to avoid SDK API drift)
    registry.init();
}

/// Lightweight metrics hook for exporter runs.
/// Currently logs via tracing; can be upgraded to real OTLP metrics without changing callsites.
pub fn record_export_metrics(
    hyperedges_total: u64,
    hyperedges_exported: u64,
    filtered_count: u64,
    latency_ms: u64,
) {
    info!(target: "ndfh.export",
        hyperedges_total,
        hyperedges_exported,
        filtered_count,
        latency_ms,
        "export metrics");
}

/// Record a policy decision for auditing/telemetry.
pub fn record_policy_decision(decision: &str, reason: Option<&str>) {
    match reason {
        Some(r) => info!(target: "ndfh.policy", decision, reason = r, "policy decision"),
        None => info!(target: "ndfh.policy", decision, "policy decision"),
    }
}

/// Record snapshot construction metrics.
pub fn record_snapshot_metrics(as_of: i64, hyperedges_count: usize, mode: &str) {
    info!(target: "ndfh.snapshot",
        as_of,
        hyperedges_count,
        mode,
        "snapshot built");
}

/// Gracefully shutdown tracer; a no-op when OTEL is not enabled.
pub fn shutdown_tracer() {
    #[cfg(feature = "otel-stdout")]
    {
        use opentelemetry::global;
        global::shutdown_tracer_provider();
    }
}
