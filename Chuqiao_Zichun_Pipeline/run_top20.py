#!/usr/bin/env python3
"""Run Yuchen Zhou's training pipeline with top-20 feature selection."""

import inspect
import sys
from pathlib import Path

import numpy as np
import sklearn.preprocessing as sk_preprocessing
import sklearn.preprocessing._encoders as sk_encoders

BASE_DIR = Path(__file__).resolve().parent
TRAINING_SRC = BASE_DIR.parent / "YuchenZhou_Pipeline" / "training" / "src"
CONFIG_PATH = BASE_DIR / "config_top20.yaml"

if not TRAINING_SRC.exists():
    raise SystemExit(f"Training src directory not found: {TRAINING_SRC}")

if not CONFIG_PATH.exists():
    raise SystemExit(f"Config file missing: {CONFIG_PATH}")

print("================= TOP-20 FEATURE RUN =================")
print(f"Training src   : {TRAINING_SRC}")
print(f"Using config   : {CONFIG_PATH}")
print("=======================================================\n")

sys.path.insert(0, str(TRAINING_SRC))

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder  # noqa: E402

if 'sparse_output' not in inspect.signature(SklearnOneHotEncoder.__init__).parameters:
    class OneHotEncoderCompat(SklearnOneHotEncoder):  # type: ignore[misc]
        def __init__(
            self,
            *,
            categories='auto',
            drop=None,
            sparse=True,
            dtype=np.float64,
            handle_unknown='error',
            min_frequency=None,
            max_categories=None,
            sparse_output=None,
        ):
            sig = inspect.signature(SklearnOneHotEncoder.__init__)
            supported = set(sig.parameters.keys()) - {'self'}

            if sparse_output is not None and 'sparse' in supported:
                sparse = sparse_output

            kwargs = {}
            if 'categories' in supported:
                kwargs['categories'] = categories
            if 'drop' in supported:
                kwargs['drop'] = drop
            if 'sparse' in supported:
                kwargs['sparse'] = sparse
            if 'dtype' in supported:
                kwargs['dtype'] = dtype
            if 'handle_unknown' in supported:
                kwargs['handle_unknown'] = handle_unknown
            if 'min_frequency' in supported and min_frequency is not None:
                kwargs['min_frequency'] = min_frequency
            if 'max_categories' in supported and max_categories is not None:
                kwargs['max_categories'] = max_categories

            super().__init__(**kwargs)

            # Expose attributes expected by newer sklearn versions during cloning.
            self.categories = categories
            self.drop = drop
            self.sparse = sparse
            self.dtype = dtype
            self.handle_unknown = handle_unknown
            self.min_frequency = min_frequency
            self.max_categories = max_categories
            self.sparse_output = sparse_output

        def get_params(self, deep=True):  # type: ignore[override]
            params = super().get_params(deep=deep)
            params.setdefault('min_frequency', self.min_frequency)
            params.setdefault('max_categories', self.max_categories)
            params.setdefault('sparse_output', self.sparse_output)
            return params

    sk_preprocessing.OneHotEncoder = OneHotEncoderCompat
    sk_encoders.OneHotEncoder = OneHotEncoderCompat

import train  # noqa: E402

argv_backup = sys.argv[:]
sys.argv = [
      "train.py",
      "--config",
      str(CONFIG_PATH),
  ]

try:
    train.main()
finally:
    sys.argv = argv_backup

print("\nTraining finished. Artifacts and reports are stored under:")
print(f"  Artifacts : {BASE_DIR / 'artifacts'}")
print(f"  Reports   : {BASE_DIR / 'reports'}")
