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
            d[name] = d.groupby("Country")[src].shift(lag)
            cols.append(name); features.append(name)
        groups[src] = cols
    self_cols = []
    for lag in (1, 2):
        name = f"{target}_self_lag{lag}"
        d[name] = d.groupby("Country")[target].shift(lag)
        self_cols.append(name); features.append(name)
    groups["__SELF__"] = self_cols
    # Four dummies + intercept; avoids exact dummy trap in Ridge.
    for c in COUNTRIES:
        if c == COUNTRY_REFERENCE:
            continue
        name = f"C_{c}"
        d[name] = (d["Country"] == c).astype(float)
        features.append(name)
    groups["__COUNTRY__"] = [f"C_{c}" for c in COUNTRIES if c != COUNTRY_REFERENCE]
    d = d[d["Date"].between(1994, 2024)].copy()
    return d, features, groups


@dataclass
class PanelPreprocessor:
    feature_names: List[str]
    scale: bool
    active_features: List[str] = field(default_factory=list)
    global_median: Dict[str, float] = field(default_factory=dict)
    country_median: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None

    def fit(self, train: pd.DataFrame) -> "PanelPreprocessor":
        self.active_features = []
        self.global_median = {}
        self.country_median = {}
        for f in self.feature_names:
            vals = pd.to_numeric(train[f], errors="coerce")
            if not vals.notna().any():
                # All-missing in this training fold: unavailable, do not create fake zero information.
                continue
            self.active_features.append(f)
            if f.startswith("C_"):
                self.global_median[f] = 0.0
                self.country_median[f] = {c: 0.0 for c in COUNTRIES}
                continue
            gmed = float(vals.median())
            self.global_median[f] = gmed
            cm = {}
            for c in COUNTRIES:
                x = pd.to_numeric(train.loc[train.Country == c, f], errors="coerce")
                cm[c] = float(x.median()) if x.notna().any() else gmed
            self.country_median[f] = cm
        X = self._impute(train)
        if self.scale:
            self.scaler = StandardScaler().fit(X)
        return self

    def _impute(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        countries = df["Country"].astype(str).to_numpy()
        for f in self.active_features:
            x = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
            if not f.startswith("C_"):
                miss = ~np.isfinite(x)
                if miss.any():
                    repl = np.array([self.country_median[f].get(c, self.global_median[f]) for c in countries])
                    x[miss] = repl[miss]
            else:
                x[~np.isfinite(x)] = 0.0
            cols.append(x)
        return np.column_stack(cols) if cols else np.empty((len(df), 0))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._impute(df)
        if self.scale and self.scaler is not None:
            X = self.scaler.transform(X)
        return X


def apply_self_transform(df: pd.DataFrame, target: str, transform: str) -> pd.DataFrame:
    """Transform only target's own lag controls; cross-variable source lags remain raw."""
    out = df.copy()
    if transform != "identity":
        for lag in (1, 2):
            col = f"{target}_self_lag{lag}"
            out[col] = forward_target(pd.to_numeric(out[col], errors="coerce").to_numpy(), transform)
    return out


def outer_splits(d: pd.DataFrame, target: str) -> List[Dict[str, Any]]:
    out = []
    for fold, (a, b, c, e) in enumerate(OUTER_WINDOWS, start=1):
        tr = d[(d.Date.between(a, b)) & d[target].notna()].copy()
        va = d[(d.Date.between(c, e)) & d[target].notna()].copy()
        out.append({
            "fold": fold, "train": tr, "valid": va,
            "train_start": a, "train_end": b, "valid_start": c, "valid_end": e,
            "reliable": len(tr) >= MIN_OUTER_TRAIN_ROWS and len(va) >= 5,
        })
    return out


def inner_splits(train: pd.DataFrame, target: str, n_splits: int = 2) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    years = sorted(train.loc[train[target].notna(), "Date"].unique())
    if len(years) < 8:
        return []
    cut = max(5, int(math.ceil(0.55 * len(years))))
    remaining = years[cut:]
    chunks = [list(x) for x in np.array_split(remaining, n_splits) if len(x)]
    train_years = list(years[:cut])
    out = []
    for chunk in chunks:
        tr = train[train.Date.isin(train_years) & train[target].notna()].copy()
        va = train[train.Date.isin(chunk) & train[target].notna()].copy()
        if len(tr) >= MIN_INNER_TRAIN_ROWS and len(va) >= 5:
            out.append((tr, va))
        train_years += chunk
    return out


def country_scales(train: pd.DataFrame, target: str) -> Dict[str, float]:
    y = pd.to_numeric(train[target], errors="coerce")
    global_sd = float(y.std(ddof=1)) if y.notna().sum() > 1 else 1.0
    if not np.isfinite(global_sd) or global_sd <= 1e-12:
        global_sd = 1.0
    out = {}
    for c in COUNTRIES:
        z = pd.to_numeric(train.loc[train.Country == c, target], errors="coerce")
        sd = float(z.std(ddof=1)) if z.notna().sum() > 1 else global_sd
        out[c] = sd if np.isfinite(sd) and sd > 1e-12 else global_sd
    return out


def balanced_nmae(y_true: np.ndarray, y_pred: np.ndarray, countries: Iterable[str], scales: Dict[str, float]) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float); countries = np.asarray(list(countries))
    scores = []
    for c in np.unique(countries):
        m = countries == c
        if m.sum() == 0:
            continue
        scores.append(float(np.mean(np.abs(y_true[m] - y_pred[m])) / scales[str(c)]))
    return float(np.mean(scores)) if scores else np.nan


def persistence_metrics(train: pd.DataFrame, valid: pd.DataFrame, target: str) -> Dict[str, float]:
    y = pd.to_numeric(valid[target], errors="coerce").to_numpy(float)
    p = pd.to_numeric(valid[f"{target}_self_lag1"], errors="coerce").to_numpy(float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "balanced_nmae": np.nan}
    scales = country_scales(train, target)
    return {
        "r2": float(r2_score(y[m], p[m])),
        "rmse": float(np.sqrt(mean_squared_error(y[m], p[m]))),
        "mae": float(mean_absolute_error(y[m], p[m])),
        "balanced_nmae": balanced_nmae(y[m], p[m], valid.loc[m, "Country"], scales),
    }


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    params: Dict[str, Any]
    transform: str


@dataclass
class FitBundle:
    target: str
    spec: ModelSpec
    features_requested: List[str]
    preprocessor: PanelPreprocessor
    model: Any
    X_train: np.ndarray
    train_frame: pd.DataFrame

    @property
    def active_features(self) -> List[str]:
        return self.preprocessor.active_features

def model_specs(target: str) -> List[ModelSpec]:
    specs = []
    for transform in target_transform_candidates(target):
        for a in RIDGE_GRID:
            specs.append(ModelSpec("Ridge", {"alpha": a}, transform))
        for p in XGB_PROFILES:
            specs.append(ModelSpec("XGB", dict(p), transform))
    return specs


def _make_model(spec: ModelSpec):
    if spec.kind == "Ridge":
        return Ridge(alpha=float(spec.params["alpha"]), fit_intercept=True)
    if spec.kind == "XGB":
        return XGBRegressor(
            **spec.params,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(spec.kind)


def fit_bundle(train_raw: pd.DataFrame, target: str, features: List[str], spec: ModelSpec) -> FitBundle:
    train = apply_self_transform(train_raw, target, spec.transform)
    prep = PanelPreprocessor(features, scale=(spec.kind == "Ridge")).fit(train)
    if not prep.active_features:
        raise ValueError(f"No active features for {target}")
    X = prep.transform(train)
    y_raw = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    y = forward_target(y_raw, spec.transform)
    model = _make_model(spec)
    model.fit(X, y)
    return FitBundle(target, spec, list(features), prep, model, X, train.copy())


def predict_raw(bundle: FitBundle, frame_raw: pd.DataFrame) -> np.ndarray:
    frame = apply_self_transform(frame_raw, bundle.target, bundle.spec.transform)
    X = bundle.preprocessor.transform(frame)
    z = np.asarray(bundle.model.predict(X), float)
    return inverse_target(z, bundle.spec.transform)


def inner_score_spec(train: pd.DataFrame, target: str, features: List[str], spec: ModelSpec) -> float:
    splits = inner_splits(train, target)
    if not splits:
        return np.inf
    scores = []
    for tr, va in splits:
        try:
            b = fit_bundle(tr, target, features, spec)
            pred = predict_raw(b, va)
            y = pd.to_numeric(va[target], errors="coerce").to_numpy(float)
            score = balanced_nmae(y, pred, va.Country, country_scales(tr, target))
        except Exception:
            score = np.inf
        scores.append(score)
    return float(np.mean(scores)) if scores else np.inf


def specs_for_kind(target: str, kind: str) -> List[ModelSpec]:
    return [s for s in model_specs(target) if s.kind == kind]


def tune_kind_on_features(train: pd.DataFrame, target: str, features: List[str], kind: str) -> Tuple[ModelSpec, float, pd.DataFrame]:
    rows = []
    best_spec = None
    best_score = np.inf
    for spec_id, spec in enumerate(specs_for_kind(target, kind)):
        score = inner_score_spec(train, target, features, spec)
        rows.append({"stage": "tune", "kind": kind, "spec_id": spec_id,
                     "transform": spec.transform, "params": json.dumps(spec.params, sort_keys=True),
                     "n_features": len(features), "balanced_nmae": score})
        if score < best_score:
            best_score = score
            best_spec = spec
    if best_spec is None or not np.isfinite(best_score):
        raise ValueError(f"No valid {kind} specification for {target}")
    return best_spec, best_score, pd.DataFrame(rows)


def features_from_sources(groups: Dict[str, List[str]], selected_sources: List[str]) -> List[str]:
    f = []
    for src in selected_sources:
        f.extend(groups[src])
    f.extend(groups["__SELF__"])
    f.extend(groups["__COUNTRY__"])
    return list(dict.fromkeys(f))

