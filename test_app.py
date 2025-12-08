"""
Tests for the HealthE Streamlit app.

Directions:
- The main app file is named `app.py`.
- These tests focus on:
  - Verifying that `load_models()` calls joblib.load with the correct paths
    (once, at import time).
  - Verifying that `load_models()` returns 4 objects and that module-level
    globals are set.
"""

import sys
import importlib
from unittest.mock import MagicMock

import joblib


def test_load_models_uses_expected_paths(monkeypatch):
    # Collect the file paths joblib.load gets called with
    called_paths = []

    def fake_load(path):
        called_paths.append(path)
        # return a distinct mock for each call
        return MagicMock(name=f"mock_for_{path}")

    # Patch joblib.load BEFORE importing the app
    monkeypatch.setattr(joblib, "load", fake_load)

    # Make sure we import a fresh copy of the module
    if "app" in sys.modules:
        del sys.modules["app"]

    app = importlib.import_module("app")

    # By now, app.load_models() has been called once at import time:
    #   recovery_model, diet_model, recovery_scaler, diet_scaler = load_models()
    # using our fake joblib.load

    # Check that joblib.load was called exactly four times
    assert len(called_paths) == 4

    # Check the exact paths used
    assert "models/LightGBM_recovery_time.joblib" in called_paths
    assert "models/new_lr_model_final.joblib" in called_paths
    assert "models/recovery_scaler_realistic.joblib" in called_paths
    assert "models/new_diet_scaler_final.joblib" in called_paths

    # Check that the globals on the app module were set
    assert hasattr(app, "recovery_model")
    assert hasattr(app, "diet_model")
    assert hasattr(app, "recovery_scaler")
    assert hasattr(app, "diet_scaler")

    # And that they are non-None (mocks in this case)
    assert app.recovery_model is not None
    assert app.diet_model is not None
    assert app.recovery_scaler is not None
    assert app.diet_scaler is not None


def test_load_models_can_be_called_directly_without_error():
    """
    Because `load_models` is wrapped in `st.cache_resource`, subsequent calls
    may *not* call joblib.load again (they use the cached result).

    So here we only verify:
    - It returns 4 values
    - Those values match the module-level globals
    """

    # Import (or reuse) the app module
    if "app" in sys.modules:
        app = sys.modules["app"]
    else:
        app = importlib.import_module("app")

    models = app.load_models()

    # Should return 4 objects: recovery_model, diet_model, recovery_scaler, diet_scaler
    assert isinstance(models, tuple) or isinstance(models, list)
    assert len(models) == 4

    rec_model, diet_model, rec_scaler, diet_scaler = models

    # They should match the globals set at import time
    assert rec_model is app.recovery_model
    assert diet_model is app.diet_model
    assert rec_scaler is app.recovery_scaler
    assert diet_scaler is app.diet_scaler
