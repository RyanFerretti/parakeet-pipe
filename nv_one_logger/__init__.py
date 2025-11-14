"""
Lightweight stub of NVIDIA's nv-one-logger package.

The upstream library is not published on PyPI, but NeMo references a few of
its core classes when wiring up telemetry callbacks.  This shim provides the
minimal surface area required for NeMo to import and proceed, effectively
no-oping the telemetry integration.
"""

__all__ = ["training_telemetry", "api"]

