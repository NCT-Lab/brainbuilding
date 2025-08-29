"""
Lightweight global context for toggling prediction phase behaviors.

This allows certain transformers to alter behavior (e.g., apply only during
prediction) without tightly coupling code paths. Default is training mode.
"""

from contextlib import contextmanager

_IN_PREDICTION_PHASE: bool = False


@contextmanager
def prediction_phase():
    """Context manager to enable prediction phase temporarily."""
    global _IN_PREDICTION_PHASE
    previous = _IN_PREDICTION_PHASE
    _IN_PREDICTION_PHASE = True
    try:
        yield
    finally:
        _IN_PREDICTION_PHASE = previous


