def on_app_start() -> None:
    """Placeholder hook invoked by NeMo once telemetry is configured."""

    # The real implementation sends a lifecycle event to NVIDIA's telemetry
    # backend. For local, no-internet workflows we deliberately no-op.
    return None

