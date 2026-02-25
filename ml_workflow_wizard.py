# ml_workflow_wizard.py
# -----------------------------------------------------------------------------
# One-file Wizard that runs:
#   (A) dataprep_interactive_pro-like cleaning
#   (B) automl_interactive_pro training
#
# This is for convenience (less confusion). It still saves artifacts in:
#   prep_out/cleaned.csv, cleaning_report.json, cleaning_config.json, cleaning_steps.py
#   best_pipeline.joblib, report.json, run_config.json
#
# Run:
#   python ml_workflow_wizard.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import math
import sys
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, GroupKFold, GroupShuffleSplit, TimeSeriesSplit, RepeatedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Task = Literal["classification", "regression"]


def get_env_versions() -> Dict[str, str]:
    """Collect key environment versions for reproducibility/loading joblib artifacts."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": getattr(sklearn, "__version__", "unknown"),
        "joblib": getattr(joblib, "__version__", "unknown"),
    }


def write_requirements_txt(path: Path = Path("requirements.txt")) -> None:
    """Write requirements.txt via `pip freeze` (best effort)."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        path.write_text(out, encoding="utf-8")
    except Exception:
        # Fallback: minimal pins (still helpful if pip freeze isn't available)
        minimal = "\n".join([
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"scikit-learn=={getattr(sklearn, '__version__', 'unknown')}",
            f"joblib=={getattr(joblib, '__version__', 'unknown')}",
        ]) + "\n"
        path.write_text(minimal, encoding="utf-8")


def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1].strip()
    return s


def pct(x: float) -> float:
    return float(round(x * 100.0, 2))


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} [{d}]: ").strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("  Please answer y/n.")


def ask_choice(prompt: str, options: List[str], default_index: int = 0) -> int:
    print(prompt)
    for i, opt in enumerate(options, start=1):
        mark = " (default)" if (i - 1) == default_index else ""
        print(f"  {i}. {opt}{mark}")
    while True:
        ans = input("Select number: ").strip()
        if not ans:
            return default_index
        if ans.isdigit():
            idx = int(ans) - 1
            if 0 <= idx < len(options):
                return idx
        print("  Invalid selection.")


def is_id_like(col: str, s: pd.Series, n_rows: int) -> bool:
    """Heuristic: detect columns that are likely identifiers (good to drop from features).
    We keep this conservative: it only *suggests* dropping; user still confirms.
    """
    # Name-based hints
    name_flag = bool(re.search(r"\b(id|uuid|guid|user|participant|account|customer|record)\b", str(col), re.I))

    # Uniqueness-based hints
    nunique = int(s.nunique(dropna=True))
    uniq_ratio = nunique / max(n_rows, 1)
    uniq_flag = (uniq_ratio >= 0.95) and (nunique >= max(50, int(0.5 * n_rows)))

    # Sequence-like numeric IDs (monotonic, mostly integers)
    seq_flag = False
    if pd.api.types.is_numeric_dtype(s):
        ss = s.dropna()
        if len(ss) >= 50:
            # integer-ish
            as_int = ss.astype("float64").round().astype("int64")
            intish = float((np.abs(ss.astype("float64") - as_int.astype("float64")) < 1e-9).mean()) >= 0.98
            if intish and (as_int.is_monotonic_increasing or as_int.is_monotonic_decreasing) and (as_int.nunique() / max(len(as_int), 1) >= 0.95):
                seq_flag = True

    return bool(name_flag or uniq_flag or seq_flag)




def is_constant_like(s: pd.Series) -> bool:
    nunique = int(s.nunique(dropna=False))
    top_freq = float(s.value_counts(dropna=False, normalize=True).iloc[0]) if len(s) else 1.0
    return bool((nunique <= 1) or (top_freq >= 0.995))


def numeric_like_object(s: pd.Series) -> Tuple[bool, float]:
    """Simple numeric-like detection for object columns (no cleaning)."""
    if s.dtype != "object":
        return False, 0.0
    ss = s.dropna().astype(str).str.strip()
    if ss.empty:
        return False, 0.0
    parsed = pd.to_numeric(ss, errors="coerce")
    rate = float(parsed.notna().mean())
    return bool(rate >= 0.85), rate


_CURRENCY_TOKENS = [
    "₩", "원", "krw", "won",
    "$", "usd", "dollar", "달러",
    "€", "eur",
    "£", "gbp",
    "¥", "jpy", "엔",
]


def _clean_messy_number_str(x: str, percent_as_fraction: bool) -> str:
    """Best-effort cleaning for real-world numeric strings.
    Handles commas, spaces, currency/unit tokens, and parentheses negatives.
    """
    s = x.strip()
    if not s:
        return s

    # Parentheses for negative numbers: (1,234) -> -1234
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # Remove common currency/unit tokens (case-insensitive)
    low = s.lower()
    for tok in _CURRENCY_TOKENS:
        low = low.replace(tok.lower(), "")
    s = low

    # Remove thousands separators and spaces
    s = s.replace(",", "").replace(" ", "")

    # Handle percent
    is_pct = "%" in s
    s = s.replace("%", "")

    # Keep only digits, sign, dot (.)
    s = re.sub(r"[^0-9+\-\.]", "", s)

    if neg and s and not s.startswith("-"):
        s = "-" + s

    # Convert percent to fraction if requested (done later via /100, but we mark here)
    if is_pct and percent_as_fraction and s:
        return f"{s}__PCT__"
    return s


def parse_messy_numeric_series(s: pd.Series, percent_as_fraction: bool = True) -> Tuple[pd.Series, float, Dict[str, Any]]:
    """Parse messy numeric strings (e.g., '10,000', '1000원', '$1,234', '12%').
    Returns (numeric_series, parse_rate, details).
    """
    if s.dtype != "object":
        out = pd.to_numeric(s, errors="coerce")
        return out, float(out.notna().mean()), {"method": "to_numeric_non_object"}

    ss = s.astype("object")
    cleaned = ss.apply(lambda v: _clean_messy_number_str(str(v), percent_as_fraction) if pd.notna(v) else v)

    def _to_num(v):
        if pd.isna(v):
            return np.nan
        st = str(v)
        if st.endswith("__PCT__"):
            base = st[:-7]
            num = pd.to_numeric(base, errors="coerce")
            return (num / 100.0) if pd.notna(num) else np.nan
        return pd.to_numeric(st, errors="coerce")

    out = cleaned.apply(_to_num)
    rate = float(out.notna().mean())

    details = {
        "method": "messy_numeric_clean",
        "percent_as_fraction": percent_as_fraction,
        "sample_raw": s.dropna().astype(str).head(8).tolist(),
        "sample_cleaned": cleaned.dropna().astype(str).head(8).tolist(),
    }
    return out, rate, details


def messy_numeric_like_object(s: pd.Series, percent_as_fraction: bool = True) -> Tuple[bool, float, Dict[str, Any]]:
    """Detect + estimate parse rate for messy numeric text columns."""
    if s.dtype != "object":
        return False, 0.0, {"reason": "not_object"}
    ss = s.dropna().astype(str)
    if ss.empty:
        return False, 0.0, {"reason": "empty"}
    _, rate, details = parse_messy_numeric_series(s, percent_as_fraction=percent_as_fraction)
    return bool(rate >= 0.80), rate, details


def date_like_object(s: pd.Series, dayfirst: bool = False) -> Tuple[bool, float, Dict[str, Any]]:
    """Detect date-like object columns by trying to parse a sample."""
    if s.dtype != "object":
        return False, 0.0, {"reason": "not_object"}
    ss = s.dropna().astype(str).str.strip()
    if ss.empty:
        return False, 0.0, {"reason": "empty"}

    sample = ss.sample(min(len(ss), 200), random_state=42) if len(ss) > 200 else ss
    dt = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True, dayfirst=dayfirst)
    rate = float(dt.notna().mean())

    slash_rate = float(sample.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$").mean())
    amb = bool(slash_rate >= 0.25)

    details = {
        "method": "to_datetime_sample",
        "dayfirst": dayfirst,
        "sample_raw": sample.head(8).tolist(),
        "parse_rate": rate,
        "ambiguous_slash_rate": slash_rate,
    }
    return bool(rate >= 0.80), rate, {"ambiguous": amb, **details}


def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: x.strip() if isinstance(x, str) else x)

def detect_outcome_like_columns(df: pd.DataFrame, target: str) -> List[str]:
    KEYWORDS = [
        "score", "quality", "fatigue", "after", "result", "outcome",
        "completion", "grade", "rating", "label", "target",
    ]
    remove: List[str] = []
    for col in df.columns:
        if col == target:
            continue
        cl = col.lower()
        if any(kw in cl for kw in KEYWORDS):
            remove.append(col)
    return remove


def infer_task(y: pd.Series) -> Task:
    if pd.api.types.is_numeric_dtype(y):
        return "classification" if int(y.nunique(dropna=True)) <= 2 else "regression"
    return "classification"


def needs_proba(metric: str) -> bool:
    return metric in {"roc_auc", "average_precision", "log_loss"}


def print_target_summary(y: pd.Series, task: Task) -> None:
    print("\n[AutoML] Target summary")
    n = int(len(y))
    miss = int(y.isna().sum())
    if task == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        print(f"  n={n}  missing={miss} ({pct(miss/n if n else 0)}%)")
        desc = y_num.describe()
        # describe returns count, mean, std, min, 25%, 50%, 75%, max
        for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            if k in desc:
                v = float(desc[k])
                print(f"  {k:>4}: {round(v, 6)}")
    else:
        vc = y.value_counts(dropna=False)
        print(f"  n={n}  missing={miss} ({pct(miss/n if n else 0)}%)  classes={len(vc)}")
        for cls, cnt in vc.head(10).items():
            print(f"  - {cls}: {cnt} ({pct(cnt/n if n else 0)}%)")
        if len(vc) >= 2:
            ratio = float(vc.max() / max(1, vc.min()))
            print(f"  class balance ratio (max/min): {round(ratio, 3)}")

def sklearn_scoring(task: Task, metric: str) -> str:
    if task == "classification":
        if metric == "accuracy":
            return "accuracy"
        if metric == "f1":
            return "f1_macro"
        if metric == "balanced_accuracy":
            return "balanced_accuracy"
        if metric == "roc_auc":
            # Works for binary and multiclass (OVR); requires predict_proba/decision_function.
            return "roc_auc_ovr_weighted"
        if metric == "average_precision":
            return "average_precision"  # binary-only
        if metric == "log_loss":
            return "neg_log_loss"
        raise ValueError(metric)

    # regression
    if metric == "r2":
        return "r2"
    if metric == "rmse":
        return "neg_root_mean_squared_error"
    if metric == "mae":
        return "neg_mean_absolute_error"
    raise ValueError(metric)


def score_regression(y_true, y_pred, metric: str) -> float:
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(metric)


def score_classification(y_true, y_pred, metric: str, y_proba=None, labels=None) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "roc_auc":
        if y_proba is None:
            raise ValueError("roc_auc requires predicted probabilities (predict_proba).")
        # binary: use positive class probability; multiclass: pass full matrix
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            pos = y_proba[:, 1]
            return float(roc_auc_score(y_true, pos))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    if metric == "average_precision":
        if y_proba is None:
            raise ValueError("average_precision requires predicted probabilities (predict_proba).")
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            pos = y_proba[:, 1]
            return float(average_precision_score(y_true, pos))
        raise ValueError("average_precision is supported for binary classification only.")
    if metric == "log_loss":
        if y_proba is None:
            raise ValueError("log_loss requires predicted probabilities (predict_proba).")
        return float(log_loss(y_true, y_proba, labels=labels))
    raise ValueError(metric)


def full_evaluation_summary(task: Task, estimator, X_test, y_test) -> Dict[str, Any]:
    """Compute a comprehensive evaluation summary on the test set and print an interpreted report.
    This runs independent of the tuning metric (i.e., always prints a full set of key metrics).
    """
    if task == "classification":
        return _full_classification_summary(estimator, X_test, y_test)
    return _full_regression_summary(estimator, X_test, y_test)


def _full_classification_summary(estimator, X_test, y_test) -> Dict[str, Any]:
    y_pred = estimator.predict(X_test)

    out: Dict[str, Any] = {}
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    bal = balanced_accuracy_score(y_test, y_pred)

    out["accuracy"] = float(acc)
    out["f1_weighted"] = float(f1w)
    out["balanced_accuracy"] = float(bal)

    # Confusion matrix (works for binary and multiclass)
    cm = confusion_matrix(y_test, y_pred)
    out["confusion_matrix"] = cm.tolist()

    print("\n=== Final Model Evaluation (Classification) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"  → About {acc*100:.1f}% of predictions matched the true label.")

    print(f"F1 (weighted): {f1w:.4f}")
    print("  → Combines precision & recall; weighted accounts for class imbalance.")

    print(f"Balanced Accuracy: {bal:.4f}")
    print("  → Average recall across classes (useful when classes are imbalanced).")

    # Probability-based metrics if available
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X_test)
            out["has_predict_proba"] = True

            # log loss works for binary + multiclass
            ll = log_loss(y_test, proba)
            out["log_loss"] = float(ll)
            print(f"Log Loss: {ll:.4f}")
            print("  → Lower is better; measures how well-calibrated/confident the probabilities are.")

            # ROC-AUC / PR-AUC
            if proba.ndim == 2 and proba.shape[1] == 2:
                # binary
                pos = proba[:, 1]
                roc = roc_auc_score(y_test, pos)
                pr = average_precision_score(y_test, pos)
                out["roc_auc"] = float(roc)
                out["pr_auc"] = float(pr)
                print(f"ROC-AUC: {roc:.4f}")
                print("  → Higher is better; measures ranking/separation between classes.")
                print(f"PR-AUC: {pr:.4f}")
                print("  → Higher is better; focuses on positive-class detection (useful with imbalance).")
            else:
                # multiclass ROC-AUC (One-vs-Rest, weighted)
                try:
                    roc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                    out["roc_auc_ovr_weighted"] = float(roc)
                    print(f"ROC-AUC (OvR, weighted): {roc:.4f}")
                    print("  → Higher is better; multiclass separation quality (weighted by class frequency).")
                except Exception:
                    pass
        except Exception:
            out["has_predict_proba"] = False

    else:
        out["has_predict_proba"] = False

    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    print(cm)
    # Light interpretation hint
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  → FP (false positives)={int(fp)}, FN (false negatives)={int(fn)}.")
        if fn > fp:
            print("  → More false negatives than false positives: model may be missing positives.")
        elif fp > fn:
            print("  → More false positives than false negatives: model may over-flag positives.")
        else:
            print("  → False positives and false negatives are roughly balanced.")

    return out


def _full_regression_summary(estimator, X_test, y_test) -> Dict[str, Any]:
    y_pred = estimator.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    out: Dict[str, Any] = {
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
    }

    # Target scale context
    y_true = np.asarray(y_test)
    y_std = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    out["y_test_mean"] = float(np.mean(y_true)) if len(y_true) else None
    out["y_test_std"] = y_std

    print("\n=== Final Model Evaluation (Regression) ===")
    print(f"R²: {r2:.4f}")
    if r2 < 0:
        print("  → Worse than predicting the mean (negative R²). Consider better features/model or leakage checks.")
    else:
        print(f"  → Explains about {r2*100:.1f}% of the variance in the target.")

    print(f"RMSE: {rmse:.4f}")
    if y_std > 0:
        print(f"  → Typical error size; compare to target std (~{y_std:.4f}).")
    else:
        print("  → Typical error size.")

    print(f"MAE: {mae:.4f}")
    print("  → Average absolute error (easier to interpret than RMSE).")

    return out




def build_preprocess(numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def candidates_for(task: Task, class_weight_balanced: bool) -> Dict[str, Tuple[Any, Dict[str, List[Any]], bool]]:
    if task == "classification":
        cw = "balanced" if class_weight_balanced else None
        return {
            "logreg": (LogisticRegression(max_iter=3000, class_weight=cw), {"model__C": [0.1, 1.0, 10.0], "model__solver": ["lbfgs"]}, True),
            "rf": (RandomForestClassifier(random_state=42, class_weight=cw), {"model__n_estimators": [300, 600], "model__max_depth": [None, 6, 16]}, False),
            "extratrees": (ExtraTreesClassifier(random_state=42, class_weight=cw), {"model__n_estimators": [400, 800], "model__max_depth": [None, 8, 20]}, False),
            "hgb": (HistGradientBoostingClassifier(random_state=42), {"model__max_depth": [None, 3, 6], "model__learning_rate": [0.05, 0.1], "model__max_iter": [200, 400]}, False),
        }
    return {
        "ridge": (Ridge(random_state=42), {"model__alpha": [0.1, 1.0, 10.0]}, True),
        "rf": (RandomForestRegressor(random_state=42), {"model__n_estimators": [300, 600], "model__max_depth": [None, 6, 16]}, False),
        "extratrees": (ExtraTreesRegressor(random_state=42), {"model__n_estimators": [400, 800], "model__max_depth": [None, 8, 20]}, False),
        "hgb": (HistGradientBoostingRegressor(random_state=42), {"model__max_depth": [None, 3, 6], "model__learning_rate": [0.05, 0.1], "model__max_iter": [200, 400]}, False),
    }


def build_pipeline(model, numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool) -> Pipeline:
    return Pipeline([("preprocess", build_preprocess(numeric_cols, categorical_cols, scale_numeric)), ("model", model)])


@dataclass
class Report:
    task: Task
    metric: str
    cv_folds: int
    best_model_name: str
    best_cv_score: float
    test_score: float
    best_params: Dict[str, Any]
    removed_outcome_like_cols: List[str]
    baseline_score: float

    full_evaluation: Optional[Dict[str, Any]] = None  # full metrics on test set (independent of tuning metric)

    # Reproducibility / diagnostics
    split_mode: str = "random"
    split_details: Dict[str, Any] = None  # e.g., {"group_col": "..."} or {"time_col": "..."}
    stability_cv_mean: Optional[float] = None
    stability_cv_std: Optional[float] = None
    stability_cv_n: Optional[int] = None


def run_dataprep_optimized(df: pd.DataFrame, csv_path: str, outdir: Path) -> Tuple[pd.DataFrame, Path, Dict[str, Any]]:
    """
    Optimized DataPrep:
      - Modes: quick / standard / strict
      - Batch decisions (drop candidates, numeric parsing, date parsing)
      - Confidence-based prompting
      - Save + reuse cleaning_config.json to avoid repeated prompts
    Returns: (cleaned_df, cleaned_csv_path, cleaning_report_dict)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = outdir / "cleaning_config.json"
    report_path = outdir / "cleaning_report.json"
    steps_path = outdir / "cleaning_steps.py"

    # ---------- Reuse config ----------
    if config_path.exists():
        if ask_yes_no(f"\n[DataPrep] Found previous config at {config_path}. Reuse it to clean this CSV automatically?", default=True):
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            protected_cols = set(cfg.get("protected_cols", []))
            high_missing_threshold = float(cfg.get("high_missing_threshold", 0.60))
            drop_duplicates = bool(cfg.get("drop_duplicates", False))
            drop_cols = list(cfg.get("drop_cols", []))
            normalize_text = bool(cfg.get("normalize_text", True))

            numeric_plan = cfg.get("numeric_parse", {}) or {}
            date_plan = cfg.get("date_parse", {}) or {}

            cleaned = df.copy()
            input_shape = list(cleaned.shape)

            if drop_duplicates:
                cleaned = cleaned.drop_duplicates()

            if drop_cols:
                cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns], errors="ignore")

            # Normalize text
            if normalize_text:
                for c in [c for c in cleaned.columns if cleaned[c].dtype == "object"]:
                    cleaned[c] = normalize_text_series(cleaned[c])

            # Numeric parsing plan
            percent_as_fraction = bool(numeric_plan.get("percent_as_fraction", True))
            numeric_cols = list(numeric_plan.get("cols", []))
            for c in numeric_cols:
                if c in cleaned.columns and cleaned[c].dtype == "object":
                    parsed, _, _ = parse_messy_numeric_series(cleaned[c], percent_as_fraction=percent_as_fraction)
                    cleaned[c] = parsed

            # Date parsing plan
            store_dates_as_iso = bool(date_plan.get("store_as_iso", True))
            dayfirst_cols = set(date_plan.get("dayfirst_cols", []))
            date_cols = list(date_plan.get("cols", []))
            for c in date_cols:
                if c in cleaned.columns and cleaned[c].dtype == "object":
                    dt = pd.to_datetime(cleaned[c].astype(str).str.strip(), errors="coerce", infer_datetime_format=True, dayfirst=(c in dayfirst_cols))
                    if store_dates_as_iso:
                        has_time = dt.dropna().astype(str).str.contains(r":\d{2}", regex=True).any()
                        fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
                        cleaned[c] = dt.dt.strftime(fmt)
                    else:
                        cleaned[c] = dt

            cleaned_csv = outdir / "cleaned.csv"
            cleaned.to_csv(cleaned_csv, index=False)

            report = {
                "input_csv": csv_path,
                "output_csv": str(cleaned_csv),
                "input_shape": input_shape,
                "output_shape": list(cleaned.shape),
                "reused_config": True,
                "protected_cols": sorted(list(protected_cols)),
                "high_missing_threshold": high_missing_threshold,
                "drop_duplicates": drop_duplicates,
                "dropped_columns": drop_cols,
                "normalize_text": normalize_text,
                "numeric_parse": numeric_plan,
                "date_parse": date_plan,
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

            # Small script for learning/debugging
            steps = [
                "# cleaning_steps.py (auto-generated from cleaning_config.json)",
                "import pandas as pd",
                "import numpy as np",
                "import re",
                "",
                f"df = pd.read_csv({csv_path!r})",
            ]
            if drop_duplicates:
                steps.append("df = df.drop_duplicates()")
            if drop_cols:
                steps.append(f"df = df.drop(columns={drop_cols!r}, errors='ignore')")
            if normalize_text:
                steps.append("# normalize text columns (strip spaces)")
                steps.append("for c in df.select_dtypes(include=['object']).columns:")
                steps.append("    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)")
            if numeric_cols:
                steps.append("# numeric parsing for messy numeric strings")
                steps.append(f"_PCT_AS_FRACTION = {percent_as_fraction!r}")
                steps.append("_CURRENCY_TOKENS = ['₩','원','krw','won','$','usd','dollar','달러','€','eur','£','gbp','¥','jpy','엔']")
                steps.append("def _clean_num(x: str) -> str:")
                steps.append("    s = x.strip()")
                steps.append("    if not s: return s")
                steps.append("    neg = False")
                steps.append("    if s.startswith('(') and s.endswith(')'):")
                steps.append("        neg = True; s = s[1:-1].strip()")
                steps.append("    low = s.lower()")
                steps.append("    for tok in _CURRENCY_TOKENS:")
                steps.append("        low = low.replace(tok.lower(), '')")
                steps.append("    s = low.replace(',', '').replace(' ', '')")
                steps.append("    is_pct = '%' in s")
                steps.append("    s = s.replace('%','')")
                steps.append("    s = re.sub(r'[^0-9+\\-\\.]', '', s)")
                steps.append("    if neg and s and not s.startswith('-'): s='-'+s")
                steps.append("    if is_pct and _PCT_AS_FRACTION and s: return s+'__PCT__'")
                steps.append("    return s")
                steps.append("def _to_num(v):")
                steps.append("    if pd.isna(v): return np.nan")
                steps.append("    st = _clean_num(str(v))")
                steps.append("    if st.endswith('__PCT__'):")
                steps.append("        base = st[:-7]")
                steps.append("        num = pd.to_numeric(base, errors='coerce')")
                steps.append("        return (num/100.0) if pd.notna(num) else np.nan")
                steps.append("    return pd.to_numeric(st, errors='coerce')")
                for c in numeric_cols:
                    steps.append(f"df[{c!r}] = df[{c!r}].apply(_to_num)")
            if date_cols:
                steps.append("# date parsing")
                steps.append(f"_STORE_AS_ISO = {store_dates_as_iso!r}")
                steps.append(f"_DAYFIRST_COLS = {sorted(list(dayfirst_cols))!r}")
                steps.append("for c in " + repr(date_cols) + ":")
                steps.append("    dt = pd.to_datetime(df[c].astype(str).str.strip(), errors='coerce', infer_datetime_format=True, dayfirst=(c in _DAYFIRST_COLS))")
                steps.append("    if _STORE_AS_ISO:")
                steps.append("        has_time = dt.dropna().astype(str).str.contains(r':\\d{2}', regex=True).any()")
                steps.append("        fmt = '%Y-%m-%d %H:%M:%S' if has_time else '%Y-%m-%d'")
                steps.append("        df[c] = dt.dt.strftime(fmt)")
                steps.append("    else:")
                steps.append("        df[c] = dt")
            steps.append(f"df.to_csv({str(cleaned_csv)!r}, index=False)")
            steps_path.write_text("\n".join(steps) + "\n", encoding="utf-8")

            print("\n[DataPrep] Reused config. Saved cleaned CSV:", cleaned_csv)
            return cleaned, cleaned_csv, report

    # ---------- Mode selection ----------
    mode_idx = ask_choice(
        "\n[DataPrep] Choose cleaning mode (controls how many prompts you get):",
        [
            "Quick (auto-apply safe fixes, minimal prompts)",
            "Standard (recommended: batch prompts + safe defaults)",
            "Strict (ask more before destructive changes)",
        ],
        default_index=1,
    )
    mode = ["quick", "standard", "strict"][mode_idx]

    print("\n[DataPrep] Dataset overview")
    print(f"  shape: {df.shape[0]} rows × {df.shape[1]} cols")
    # show only top 12 cols to avoid spam on wide data; user can inspect CSV separately
    for i, c in enumerate(df.columns[:12]):
        print(f"  - {c} | dtype={df[c].dtype} | missing={pct(df[c].isna().mean())}%")
    if df.shape[1] > 12:
        print(f"  ... ({df.shape[1]-12} more columns)")

    protect_raw = input("\n[DataPrep] Protect columns from being dropped? (comma-separated, optional): ").strip()
    protected_cols = {p.strip() for p in protect_raw.split(",") if p.strip()} if protect_raw else set()

    threshold_str = input("\n[DataPrep] Drop columns with missing% >= ? (default 60): ").strip()
    high_missing_threshold = 0.60 if not threshold_str else max(0.0, min(1.0, float(threshold_str) / 100.0))

    drop_dupes_default = (mode == "quick")
    drop_dupes = ask_yes_no("\n[DataPrep] Drop duplicate rows?", default=drop_dupes_default)

    cleaned = df.copy()
    input_shape = list(cleaned.shape)
    if drop_dupes:
        cleaned = cleaned.drop_duplicates()

    # ---------- Data quality scan (summary) ----------
    # Keep it short; only print details if there are issues.
    quality = {"column_flags": {}, "dataset_flags": {}}
    if cleaned.duplicated().any():
        quality["dataset_flags"]["has_duplicate_rows"] = True
        quality["dataset_flags"]["duplicate_rows_count"] = int(cleaned.duplicated().sum())

    for c in cleaned.columns:
        s = cleaned[c]
        flags = []
        if s.dtype == "object":
            ss = s.dropna().astype(str)
            if not ss.empty:
                if (ss.str.contains(r"\s+$|^\s+", regex=True).mean() >= 0.05):
                    flags.append("whitespace_in_values")
                if ss.str.contains(r"[,$₩€£¥]|원|달러|krw|usd|eur|gbp|jpy", case=False, regex=True).any():
                    flags.append("currency_or_unit_tokens")
                if ss.str.contains(",", regex=False).any():
                    flags.append("commas_in_values")
                if ss.str.contains("%", regex=False).any():
                    flags.append("percent_values")

                # date-ish patterns
                date_pat_rate = float(ss.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean())
                slash_pat_rate = float(ss.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$").mean())
                if date_pat_rate >= 0.30 or slash_pat_rate >= 0.30:
                    flags.append("date_like_strings")

                nunique = int(s.nunique(dropna=True))
                if len(s) > 0 and (nunique / max(len(s), 1) >= 0.50) and nunique >= 50:
                    flags.append("high_cardinality_text_or_id_like")

        if flags:
            quality["column_flags"][c] = flags

    if quality["column_flags"]:
        print("\n[DataPrep] Data quality scan: issues detected in", len(quality["column_flags"]), "column(s).")
        if mode != "quick" and ask_yes_no("[DataPrep] Show the list of flagged columns now?", default=True):
            for c, fl in quality["column_flags"].items():
                print(f"  - {c}: {', '.join(fl)}")
    else:
        print("\n[DataPrep] Data quality scan: no obvious formatting issues detected.")

    # ---------- Drop candidates (batch) ----------
    n_rows = int(cleaned.shape[0])

    id_candidates = []
    constant_candidates = []
    high_missing_candidates = []

    # confidence-ish scoring: high if name looks like id + near-unique
    def _id_confidence(col: str, s: pd.Series) -> float:
        ss = s.dropna()
        nunique = int(ss.nunique(dropna=True))
        uniq_ratio = nunique / max(n_rows, 1)
        name_bonus = 0.35 if re.search(r"\b(id|uuid|guid|user|account|customer|record)\b", str(col), re.I) else 0.0
        seq_bonus = 0.25 if (pd.api.types.is_numeric_dtype(ss) and len(ss) >= 50 and ss.is_monotonic_increasing) else 0.0
        return min(1.0, (uniq_ratio * 0.7) + name_bonus + seq_bonus)

    for c in cleaned.columns:
        if c in protected_cols:
            continue
        s = cleaned[c]
        mr = float(s.isna().mean())
        if mr >= high_missing_threshold:
            high_missing_candidates.append(c)
        if is_constant_like(s):
            constant_candidates.append(c)
        if is_id_like(c, s, n_rows):
            id_candidates.append(c)

    # De-dup in case overlap
    id_candidates = [c for c in id_candidates if c not in protected_cols]
    constant_candidates = [c for c in constant_candidates if c not in protected_cols]
    high_missing_candidates = [c for c in high_missing_candidates if c not in protected_cols]

    drop_cols = set()

    def _batch_drop(prompt_title: str, cols: List[str], default: bool) -> None:
        nonlocal drop_cols
        if not cols:
            return
        print(f"\n[DataPrep] {prompt_title}: {len(cols)} column(s)")
        if mode != "quick":
            print("  " + ", ".join(cols[:12]) + ("" if len(cols) <= 12 else f", ... (+{len(cols)-12} more)"))
        do_drop = ask_yes_no(f"[DataPrep] Drop ALL {prompt_title.lower()} columns?", default=default)
        if do_drop:
            keep_raw = input("[DataPrep] Any columns to KEEP from this drop list? (comma-separated, optional): ").strip()
            keep = {k.strip() for k in keep_raw.split(",") if k.strip()} if keep_raw else set()
            drop_cols |= set([c for c in cols if c not in keep])

    # Defaults by mode
    _batch_drop("ID-like", id_candidates, default=(mode in {"quick", "standard"}))
    _batch_drop("Constant-like", constant_candidates, default=(mode in {"quick", "standard"}))
    _batch_drop(f"High-missing (>= {int(high_missing_threshold*100)}%)", high_missing_candidates, default=False)

    if drop_cols:
        cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns], errors="ignore")

    # ---------- Normalize text ----------
    obj_cols = [c for c in cleaned.columns if cleaned[c].dtype == "object"]
    normalize_default = True if mode in {"quick", "standard"} else False
    normalize_text = bool(obj_cols) and ask_yes_no(f"\n[DataPrep] Normalize {len(obj_cols)} text columns (strip spaces)?", default=normalize_default)
    if normalize_text:
        for c in obj_cols:
            cleaned[c] = normalize_text_series(cleaned[c])

    # ---------- Numeric parsing (batch) ----------
    coerced_numeric_cols: List[str] = []
    percent_as_fraction = True
    numeric_candidates: List[Tuple[str, float]] = []
    for c in obj_cols:
        flag, rate, _ = messy_numeric_like_object(cleaned[c], percent_as_fraction=True)
        if flag:
            numeric_candidates.append((c, rate))
    numeric_candidates.sort(key=lambda t: t[1], reverse=True)

    numeric_parse_applied = False
    if numeric_candidates:
        print(f"\n[DataPrep] Numeric-like text columns detected: {len(numeric_candidates)}")
        if mode != "quick":
            shown = ", ".join([f"{c}({pct(r)}%)" for c, r in numeric_candidates[:10]])
            print("  " + shown + ("" if len(numeric_candidates) <= 10 else f", ... (+{len(numeric_candidates)-10} more)"))
        default_numeric = (mode in {"quick", "standard"})
        if ask_yes_no("[DataPrep] Convert these numeric-like text columns to numeric (handles commas/currency/units/%)?", default=default_numeric):
            any_pct = any(cleaned[c].astype(str).str.contains("%", na=False).any() for c, _ in numeric_candidates)
            if any_pct:
                percent_as_fraction = ask_yes_no("[DataPrep] Convert '12%' -> 0.12 (recommended)?", default=True)

            exclude_raw = input("[DataPrep] Any columns to EXCLUDE from numeric parsing? (comma-separated, optional): ").strip()
            exclude = {e.strip() for e in exclude_raw.split(",") if e.strip()} if exclude_raw else set()

            # quick mode: auto-apply only high parse-rate cols unless user insists
            for c, rate in numeric_candidates:
                if c in exclude:
                    continue
                if mode == "quick" and rate < 0.95:
                    continue
                parsed, _, _ = parse_messy_numeric_series(cleaned[c], percent_as_fraction=percent_as_fraction)
                cleaned[c] = parsed
                coerced_numeric_cols.append(c)
            numeric_parse_applied = True

    # ---------- Date parsing (batch) ----------
    parsed_date_cols: List[str] = []
    dayfirst_cols: List[str] = []
    store_dates_as_iso = True
    date_candidates: List[Tuple[str, float, bool]] = []
    for c in obj_cols:
        if c in coerced_numeric_cols:
            continue
        flag, rate, info = date_like_object(cleaned[c], dayfirst=False)
        if flag:
            date_candidates.append((c, rate, bool(info.get("ambiguous", False))))
    date_candidates.sort(key=lambda t: t[1], reverse=True)

    date_parse_applied = False
    if date_candidates:
        print(f"\n[DataPrep] Date-like text columns detected: {len(date_candidates)}")
        if mode != "quick":
            shown = ", ".join([f"{c}({pct(r)}%)" for c, r, _ in date_candidates[:10]])
            print("  " + shown + ("" if len(date_candidates) <= 10 else f", ... (+{len(date_candidates)-10} more)"))
        default_date = (mode in {"quick", "standard"})
        if ask_yes_no("[DataPrep] Parse these date-like columns as dates?", default=default_date):
            store_dates_as_iso = ask_yes_no("[DataPrep] Store parsed dates as ISO strings in cleaned.csv? (recommended)", default=True)

            # If ambiguous slash dates exist, ask once (standard/quick). Strict asks per column.
            global_dayfirst = False
            any_amb = any(amb for _, _, amb in date_candidates)
            if any_amb and mode != "strict":
                global_dayfirst = ask_yes_no(
                    "[DataPrep] Found many ambiguous slash dates (like 01/02/2026). Treat as day-first? (dd/mm/yyyy)",
                    default=True,
                )

            exclude_raw = input("[DataPrep] Any columns to EXCLUDE from date parsing? (comma-separated, optional): ").strip()
            exclude = {e.strip() for e in exclude_raw.split(",") if e.strip()} if exclude_raw else set()

            for c, _, amb in date_candidates:
                if c in exclude:
                    continue

                dayfirst = False
                if amb:
                    if mode == "strict":
                        dayfirst = ask_yes_no(
                            f"[DataPrep] Column '{c}' seems ambiguous. Treat as day-first? (dd/mm/yyyy)",
                            default=True,
                        )
                    else:
                        dayfirst = global_dayfirst

                dt = pd.to_datetime(cleaned[c].astype(str).str.strip(), errors="coerce", infer_datetime_format=True, dayfirst=dayfirst)
                if store_dates_as_iso:
                    has_time = dt.dropna().astype(str).str.contains(r":\d{2}", regex=True).any()
                    fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
                    cleaned[c] = dt.dt.strftime(fmt)
                else:
                    cleaned[c] = dt

                parsed_date_cols.append(c)
                if dayfirst:
                    dayfirst_cols.append(c)
            date_parse_applied = True

    # ---------- Save outputs ----------
    cleaned_csv = outdir / "cleaned.csv"
    cleaned.to_csv(cleaned_csv, index=False)

    cleaning_report = {
        "input_csv": csv_path,
        "output_csv": str(cleaned_csv),
        "input_shape": input_shape,
        "output_shape": list(cleaned.shape),
        "mode": mode,
        "data_quality_flags": quality,
        "protected_cols": sorted(list(protected_cols)),
        "high_missing_threshold": high_missing_threshold,
        "drop_duplicates": bool(drop_dupes),
        "dropped_columns": sorted(list(drop_cols)),
        "normalize_text": bool(normalize_text),
        "numeric_parse": {
            "enabled": bool(numeric_parse_applied),
            "cols": coerced_numeric_cols,
            "percent_as_fraction": bool(percent_as_fraction),
        },
        "date_parse": {
            "enabled": bool(date_parse_applied),
            "cols": parsed_date_cols,
            "dayfirst_cols": dayfirst_cols,
            "store_as_iso": bool(store_dates_as_iso),
        },
    }
    report_path.write_text(json.dumps(cleaning_report, ensure_ascii=False, indent=2), encoding="utf-8")

    cleaning_config = {
        "high_missing_threshold": high_missing_threshold,
        "protected_cols": sorted(list(protected_cols)),
        "drop_duplicates": bool(drop_dupes),
        "drop_cols": sorted(list(drop_cols)),
        "normalize_text": bool(normalize_text),
        "numeric_parse": {
            "cols": coerced_numeric_cols,
            "percent_as_fraction": bool(percent_as_fraction),
        },
        "date_parse": {
            "cols": parsed_date_cols,
            "dayfirst_cols": dayfirst_cols,
            "store_as_iso": bool(store_dates_as_iso),
        },
    }
    config_path.write_text(json.dumps(cleaning_config, ensure_ascii=False, indent=2), encoding="utf-8")

    # steps script
    steps = [
        "# cleaning_steps.py (auto-generated)",
        "import pandas as pd",
        "import numpy as np",
        "import re",
        "",
        f"df = pd.read_csv({csv_path!r})",
    ]
    if drop_dupes:
        steps.append("df = df.drop_duplicates()")
    if drop_cols:
        steps.append(f"df = df.drop(columns={sorted(list(drop_cols))!r}, errors='ignore')")
    if normalize_text:
        steps.append("# normalize text columns (strip spaces)")
        steps.append("for c in df.select_dtypes(include=['object']).columns:")
        steps.append("    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)")
    if coerced_numeric_cols:
        steps.append("# numeric parsing for messy numeric strings")
        steps.append(f"_PCT_AS_FRACTION = {percent_as_fraction!r}")
        steps.append("_CURRENCY_TOKENS = ['₩','원','krw','won','$','usd','dollar','달러','€','eur','£','gbp','¥','jpy','엔']")
        steps.append("def _clean_num(x: str) -> str:")
        steps.append("    s = x.strip()")
        steps.append("    if not s: return s")
        steps.append("    neg = False")
        steps.append("    if s.startswith('(') and s.endswith(')'):")
        steps.append("        neg = True; s = s[1:-1].strip()")
        steps.append("    low = s.lower()")
        steps.append("    for tok in _CURRENCY_TOKENS:")
        steps.append("        low = low.replace(tok.lower(), '')")
        steps.append("    s = low.replace(',', '').replace(' ', '')")
        steps.append("    is_pct = '%' in s")
        steps.append("    s = s.replace('%','')")
        steps.append("    s = re.sub(r'[^0-9+\\-\\.]', '', s)")
        steps.append("    if neg and s and not s.startswith('-'): s='-'+s")
        steps.append("    if is_pct and _PCT_AS_FRACTION and s: return s+'__PCT__'")
        steps.append("    return s")
        steps.append("def _to_num(v):")
        steps.append("    if pd.isna(v): return np.nan")
        steps.append("    st = _clean_num(str(v))")
        steps.append("    if st.endswith('__PCT__'):")
        steps.append("        base = st[:-7]")
        steps.append("        num = pd.to_numeric(base, errors='coerce')")
        steps.append("        return (num/100.0) if pd.notna(num) else np.nan")
        steps.append("    return pd.to_numeric(st, errors='coerce')")
        for c in coerced_numeric_cols:
            steps.append(f"df[{c!r}] = df[{c!r}].apply(_to_num)")
    if parsed_date_cols:
        steps.append("# date parsing")
        steps.append(f"_STORE_AS_ISO = {store_dates_as_iso!r}")
        steps.append(f"_DAYFIRST_COLS = {dayfirst_cols!r}")
        steps.append(f"for c in {parsed_date_cols!r}:")
        steps.append("    dt = pd.to_datetime(df[c].astype(str).str.strip(), errors='coerce', infer_datetime_format=True, dayfirst=(c in _DAYFIRST_COLS))")
        steps.append("    if _STORE_AS_ISO:")
        steps.append("        has_time = dt.dropna().astype(str).str.contains(r':\\d{2}', regex=True).any()")
        steps.append("        fmt = '%Y-%m-%d %H:%M:%S' if has_time else '%Y-%m-%d'")
        steps.append("        df[c] = dt.dt.strftime(fmt)")
        steps.append("    else:")
        steps.append("        df[c] = dt")
    steps.append(f"df.to_csv({str(cleaned_csv)!r}, index=False)")
    steps_path.write_text("\n".join(steps) + "\n", encoding="utf-8")

    print("\n[DataPrep] Saved cleaned CSV:", cleaned_csv)
    return cleaned, cleaned_csv, cleaning_report


def main() -> None:
    print("=== ML Workflow Wizard (DataPrep -> AutoML) ===")

    csv_path = strip_quotes(input("Enter path to RAW CSV file: "))
    if not csv_path or not Path(csv_path).exists():
        raise SystemExit(f"ERROR: CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ------------------ DataPrep ------------------
    outdir = Path("prep_out")
    cleaned, cleaned_csv, _cleaning_report = run_dataprep_optimized(df, csv_path, outdir)

    # ------------------ AutoML ------------------
    df2 = cleaned
    cols = list(df2.columns)
    print("\n[AutoML] Available columns:")
    for i, c in enumerate(cols, start=1):
        print(f"  {i:>2}. {c}")
    t_idx = ask_choice("\n[AutoML] Select target column:", cols, default_index=max(0, len(cols)-1))
    target = cols[t_idx]
    y = df2[target]

    auto_remove = detect_outcome_like_columns(df2, target)
    if auto_remove:
        print("\n[AutoML] Outcome-like columns detected (by name keywords):")
        for c in auto_remove:
            print(" -", c)
        if not ask_yes_no("[AutoML] Remove these columns from features?", default=True):
            auto_remove = []
            print("[AutoML] Keeping all columns (no outcome-like removal).")
        else:
            print("[AutoML] Removing the above columns from features.")

    X = df2.drop(columns=[target] + auto_remove)
    task = infer_task(y)
    print_target_summary(y, task)

    # ---- Metric selection (expanded) ----
    if task == "classification":
        n_classes = int(y.nunique(dropna=False))
        metric_opts = ["accuracy", "f1", "balanced_accuracy"]
        if n_classes <= 2:
            metric_opts += ["roc_auc", "average_precision", "log_loss"]
        else:
            metric_opts += ["roc_auc", "log_loss"]
    else:
        metric_opts = ["r2", "rmse", "mae"]
    metric = metric_opts[ask_choice("\n[AutoML] Select metric:", metric_opts, default_index=0)]

    # ---- Imbalance hint ----
    class_weight_balanced = False
    if task == "classification":
        vc = y.value_counts(dropna=False)
        if len(vc) >= 2:
            ratio = float(vc.max() / max(1, vc.min()))
            if ratio >= 1.5:
                class_weight_balanced = True
                print("[AutoML] Enabling class_weight='balanced' for applicable models.")

    # ---- Split strategy ----
    split_labels = [
        "random (standard shuffle split)",
        "group (keep same-entity rows together)",
        "time (train first → test last)",
    ]
    split_sel = ask_choice("\n[AutoML] Select split strategy:", split_labels, default_index=0)
    split_mode = ["random", "group", "time"][split_sel]
    split_details: Dict[str, Any] = {}

    groups_full = None
    groups_train = None

    # For stability CV later (may differ for time-based sorting)
    X_stab = X
    y_stab = y
    groups_stab = None

    if split_mode == "random":
        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        cv_for_search = 5

    elif split_mode == "group":
        candidates = [c for c in df2.columns if c != target]
        # default: any column that looks like an id by name
        default_g = 0
        for i, c in enumerate(candidates):
            if re.search(r"\b(id|uuid|guid|user|account|customer|record)\b", str(c), re.I):
                default_g = i
                break
        g_idx = ask_choice("\n[AutoML] Select group column (to prevent group leakage):", candidates, default_index=default_g)
        group_col = candidates[g_idx]
        groups_full = df2[group_col]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups_full))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups_full.iloc[train_idx]

        cv_for_search = GroupKFold(n_splits=5)
        split_details = {"group_col": group_col}
        X_stab, y_stab, groups_stab = X, y, groups_full

    else:  # time
        candidates = [c for c in df2.columns if c != target]
        default_t = 0
        for i, c in enumerate(candidates):
            if re.search(r"(date|time|timestamp|day|month|year)", str(c), re.I):
                default_t = i
                break
        t_idx = ask_choice("\n[AutoML] Select time column (to avoid future leakage):", candidates, default_index=default_t)
        time_col = candidates[t_idx]

        dt = pd.to_datetime(df2[time_col], errors="coerce")
        if dt.notna().any():
            order = np.argsort(dt.fillna(pd.Timestamp.min).to_numpy())
            split_details = {"time_col": time_col, "sorted_by": "datetime"}
        else:
            order = np.argsort(pd.to_numeric(df2[time_col], errors="coerce").fillna(-np.inf).to_numpy())
            split_details = {"time_col": time_col, "sorted_by": "numeric/raw"}

        X_sorted = X.iloc[order].reset_index(drop=True)
        y_sorted = y.iloc[order].reset_index(drop=True)

        n_total = len(X_sorted)
        cut = int(n_total * 0.8)
        X_train, X_test = X_sorted.iloc[:cut], X_sorted.iloc[cut:]
        y_train, y_test = y_sorted.iloc[:cut], y_sorted.iloc[cut:]

        cv_for_search = TimeSeriesSplit(n_splits=5)

        # Stability should respect time order
        X_stab, y_stab = X_sorted, y_sorted

    # baseline
    if task == "classification":
        dum = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        y_pred0 = dum.predict(X_test)
        y_proba0 = dum.predict_proba(X_test) if needs_proba(metric) else None
        base = score_classification(y_test, y_pred0, metric, y_proba=y_proba0, labels=getattr(dum, "classes_", None))
    else:
        dum = DummyRegressor(strategy="mean").fit(X_train, y_train)
        base = score_regression(y_test, dum.predict(X_test), metric)
    print(f"\n[AutoML] Baseline {metric}: {round(float(base), 6)}")

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    scoring = sklearn_scoring(task, metric)
    minimize = (task == "regression" and metric in {"rmse", "mae"})

    best_name = ""
    best_cv = None
    best_est = None
    best_params = {}

    for name, (estimator, grid, needs_scaling) in candidates_for(task, class_weight_balanced).items():
        pipe = build_pipeline(estimator, numeric_cols, categorical_cols, needs_scaling)
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=cv_for_search, n_jobs=-1, refit=True)
        print(f"\n[AutoML] Tuning {name} ...")
        gs.fit(X_train, y_train, groups=groups_train) if split_mode == "group" else gs.fit(X_train, y_train)
        cv_score = float(gs.best_score_)
        if minimize:
            cv_score = -cv_score
        print("  best CV:", round(cv_score, 6))
        if best_cv is None:
            best_name, best_cv, best_est, best_params = name, cv_score, gs.best_estimator_, gs.best_params_
        else:
            better = (cv_score < best_cv) if minimize else (cv_score > best_cv)
            if better:
                best_name, best_cv, best_est, best_params = name, cv_score, gs.best_estimator_, gs.best_params_

    pred = best_est.predict(X_test)
    if task == "classification":
        y_proba = best_est.predict_proba(X_test) if needs_proba(metric) else None
        test_score = score_classification(y_test, pred, metric, y_proba=y_proba, labels=getattr(best_est, "classes_", None))
    else:
        test_score = score_regression(y_test, pred, metric)

    full_eval = full_evaluation_summary(task, best_est, X_test, y_test)

    rep = Report(
        task=task,
        metric=metric,
        cv_folds=5,
        best_model_name=best_name,
        best_cv_score=float(best_cv),
        test_score=float(test_score),
        best_params=best_params,
        removed_outcome_like_cols=auto_remove,
        baseline_score=float(base),
        full_evaluation=full_eval,
        split_mode=split_mode,
        split_details=split_details,
    )

    # ---- Stability evaluation (optional) ----
    if ask_yes_no("\n[AutoML] Run stability evaluation (cross-validated best pipeline)?", default=False):
        if split_mode == "random":
            cv_stab = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) if task == "classification" else RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
            scores = cross_val_score(best_est, X_stab, y_stab, scoring=scoring, cv=cv_stab, n_jobs=-1)
        elif split_mode == "group":
            cv_stab = GroupKFold(n_splits=5)
            scores = cross_val_score(best_est, X_stab, y_stab, scoring=scoring, cv=cv_stab, groups=groups_stab, n_jobs=-1)
        else:
            cv_stab = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(best_est, X_stab, y_stab, scoring=scoring, cv=cv_stab, n_jobs=-1)

        scores_nat = (-scores) if minimize else scores
        rep.stability_cv_mean = float(np.mean(scores_nat))
        rep.stability_cv_std = float(np.std(scores_nat, ddof=1)) if len(scores_nat) > 1 else 0.0
        rep.stability_cv_n = int(len(scores_nat))
        print(f"[AutoML] Stability CV: mean={round(rep.stability_cv_mean, 6)}  std={round(rep.stability_cv_std, 6)}  n={rep.stability_cv_n}")

    write_requirements_txt()

    joblib.dump(best_est, "best_pipeline.joblib")
    Path("report.json").write_text(json.dumps(asdict(rep), ensure_ascii=False, indent=2), encoding="utf-8")
    Path("run_config.json").write_text(json.dumps({
        "input_csv": csv_path,
        "cleaned_csv": str(cleaned_csv),
        "requirements_txt": "requirements.txt",
        "env_versions": get_env_versions(),
        "target": target,
        "task": task,
        "metric": metric,
        "removed_outcome_like_cols": auto_remove,
        "feature_columns_used": list(X.columns),
        "best_model_name": best_name,
        "best_params": best_params,
        "baseline_score": float(base),
        "split_mode": split_mode,
        "split_details": split_details,
        "stability_cv": {
            "mean": rep.stability_cv_mean,
            "std": rep.stability_cv_std,
            "n": rep.stability_cv_n,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Wizard Final Summary ===")
    print(json.dumps(asdict(rep), ensure_ascii=False, indent=2))
    print("\nSaved: best_pipeline.joblib, report.json, run_config.json")


if __name__ == "__main__":
    main()