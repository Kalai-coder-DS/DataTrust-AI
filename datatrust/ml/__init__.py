"""
Phase 7: baseline model.

Trains one simple, fixed model on the user's target column twice - on the data as uploaded
and on the data after the fixes Phases 2-4 recommend - and compares both with a model that
just guesses. Answers two questions the scores alone cannot:

  1. Does cleaning actually change what a model learns from this data?
  2. Is there any signal at all? (Clean data can still be useless for a given target.)
"""
from datatrust.ml.baseline import run_baseline
from datatrust.ml.features import PreparedData, prepare

__all__ = ["PreparedData", "prepare", "run_baseline"]
