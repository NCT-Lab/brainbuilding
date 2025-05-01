from contextlib import contextmanager

_IN_PREDICTION_PHASE = False

@contextmanager
def prediction_phase():
    global _IN_PREDICTION_PHASE
    old_value = _IN_PREDICTION_PHASE  # Save current state
    _IN_PREDICTION_PHASE = True       # Set up context (runs at with-statement entry)
    try:
        yield                         # Transfer control back to the with-statement body
    finally:
        _IN_PREDICTION_PHASE = old_value  # Clean up (guaranteed to run when with-block exits)