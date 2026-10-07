"""Audited macro-financial network ML pipeline (v2).

Purpose
-------
This module is a publication-oriented replacement for the preliminary notebooks.
It enforces calendar-time nested validation, train-only preprocessing, target-specific
model/transform selection, out-of-fold SHAP for BOTH Ridge and XGBoost, grouped
(drop-column) edge-support diagnostics, persistence benchmarks, and recursive
model-implied response simulation.

Important interpretation rules
------------------------------
* Candidate directed edges come from the literature dictionary REL.
* Self-lags are mandatory persistence controls, not cross-variable network edges.
* W is an out-of-fold predictive attribution network (SHAP), not a causal matrix.
* S is out-of-fold incremental edge-support stability (group drop-column), not a p-value.
* D is SHAP contribution balance, not the sign of a structural coefficient.
* R is a finite-perturbation, model-implied dynamic response, not a structural IRF.

The 2023-2024 period is treated as a supplementary temporal check. Primary model
selection and generalization evidence comes from the nested rolling-origin folds ending in 2022.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

try:
    import shap
except Exception:  # SHAP is optional for data/model audit mode.
    shap = None

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = os.getenv("MACRO_DATA_FILE", str(REPO_ROOT / "data" / "Used Datas ML (24 Variables).xlsx"))
COUNTRY_SHEETS = {
    "TR": "Turkiye Annually",
    "US": "USA Annually",
    "UK": "UK Annually",
    "DE": "Germany Annualy",
    "FR": "France Annualy",
}
COUNTRY_REFERENCE = "FR"
COUNTRIES = list(COUNTRY_SHEETS)

VARS10 = ["CDS", "EXC", "INF", "TPD", "INT", "STC", "CAB", "EID", "GRI", "INV"]
REL: Dict[str, List[str]] = {
    "CDS": ["INV", "EXC", "INF", "TPD", "INT"],
    "EXC": ["CDS", "INV", "INF", "TPD", "INT", "CAB", "GRI"],
    "INF": ["CDS", "INV", "EXC", "TPD", "INT", "STC", "CAB", "GRI"],
    "TPD": ["CDS", "INV", "EXC", "INF", "INT", "CAB", "GRI"],
    "INT": ["INV", "EXC", "INF", "TPD", "STC", "CAB", "GRI"],
    "STC": ["CDS", "INV", "INF", "TPD", "INT", "CAB", "EID"],
    "CAB": ["INV", "EXC", "INF", "TPD", "INT", "STC", "EID", "GRI"],
    "EID": ["CDS", "INV", "INF", "TPD", "INT", "STC", "GRI"],
    "GRI": ["CDS", "INV", "EXC", "INF", "TPD", "INT", "EID"],
    "INV": ["CDS", "EXC", "INF", "TPD", "INT", "STC", "CAB", "GRI"],
}

# Calendar-time outer folds. Folds with insufficient observed target rows are skipped.
OUTER_WINDOWS = [
    (1994, 2007, 2008, 2012),
    (1994, 2012, 2013, 2017),
    (1994, 2017, 2018, 2022),
]
TEMPORAL_CHECK = (2023, 2024)  # supplementary temporal check
MIN_OUTER_TRAIN_ROWS = 30
MIN_INNER_TRAIN_ROWS = 20

# Small pre-specified grids reduce researcher degrees of freedom in n=155 data.
RIDGE_GRID = [0.1, 1.0, 10.0, 100.0]
XGB_PROFILES = [
    dict(n_estimators=120, max_depth=1, learning_rate=0.03, min_child_weight=5,
         reg_alpha=0.5, reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=200, max_depth=1, learning_rate=0.03, min_child_weight=3,
         reg_alpha=0.2, reg_lambda=8.0, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=180, max_depth=2, learning_rate=0.03, min_child_weight=5,
         reg_alpha=0.5, reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=250, max_depth=2, learning_rate=0.03, min_child_weight=3,
         reg_alpha=0.2, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=180, max_depth=2, learning_rate=0.05, min_child_weight=5,
         reg_alpha=0.5, reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8),
]


def forward_target(y: np.ndarray, transform: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    if transform == "asinh":
        return np.arcsinh(y)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_target(z: np.ndarray, transform: str) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if transform == "identity":
        return z
    if transform == "asinh":
        return np.sinh(z)
    raise ValueError(f"Unknown target transform: {transform}")


def target_transform_candidates(target: str) -> List[str]:
    # Pre-specified after pre-2023 distributional diagnostics.
    return ["asinh"] if target == "INF" else ["identity"]


def load_panel(path: str = DATA_FILE, max_year: int = 2024) -> pd.DataFrame:
    frames = []
    for country, sheet in COUNTRY_SHEETS.items():
        d = pd.read_excel(path, sheet_name=sheet)
        d.columns = [str(c).strip() for c in d.columns]
        d = d.rename(columns={"TBD": "TPD"})
        if d.columns.duplicated().any():
            dup = d.columns[d.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns in {sheet}: {dup}")
        if "Date" not in d.columns:
            raise ValueError(f"Date missing in {sheet}")
        d["Date"] = pd.to_numeric(d["Date"], errors="coerce")
        for col in d.columns:
            if col != "Date":
                d[col] = pd.to_numeric(d[col], errors="coerce")
        d["Country"] = country
        d = d[d["Date"].between(1900, max_year)].copy()
        d = d.dropna(subset=["Date"])
        d["Date"] = d["Date"].astype(int)
        if d["Date"].duplicated().any():
            raise ValueError(f"Duplicate years in {sheet}")
        missing_vars = [v for v in VARS10 if v not in d.columns]
        if missing_vars:
            raise ValueError(f"Missing required variables in {sheet}: {missing_vars}")
        frames.append(d)
    full = pd.concat(frames, ignore_index=True).sort_values(["Country", "Date"]).reset_index(drop=True)
    return full


def audit_panel(full: pd.DataFrame) -> Dict[str, Any]:
    panel = full[full["Date"].between(1994, 2024)].copy()
    counts = panel.groupby("Country").size().to_dict()
    duplicate_country_year = int(panel.duplicated(["Country", "Date"]).sum())
    missing = panel.groupby("Country")[VARS10].apply(lambda x: x.isna().sum())
    observed_span = {}
    for v in VARS10:
        observed_span[v] = {}
        for c in COUNTRIES:
            q = panel[(panel.Country == c) & panel[v].notna()]
            observed_span[v][c] = None if q.empty else (int(q.Date.min()), int(q.Date.max()), int(len(q)))
    return {
        "n_rows": int(len(panel)),
        "country_rows": counts,
        "duplicate_country_year": duplicate_country_year,
        "missing_by_country": missing,
        "observed_span": observed_span,
    }


def build_target_frame(full: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    predictors = REL[target]
    d = full[["Country", "Date", target] + predictors].copy().sort_values(["Country", "Date"])
    groups: Dict[str, List[str]] = {}
    features: List[str] = []
    for src in predictors:
        cols = []
        for lag in (1, 2):
            name = f"{src}_lag{lag}"
