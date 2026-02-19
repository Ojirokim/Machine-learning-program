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
    name_flag = bool(re.search(r"\b(id|uuid|guid|user|participant|account|customer|record)\b", col, re.I))
    nunique = int(s.nunique(dropna=True))
    uniq_ratio = nunique / max(n_rows, 1)
    uniq_flag = (uniq_ratio >= 0.95) and (nunique >= max(50, int(0.5 * n_rows)))
    return bool(name_flag or uniq_flag)


def is_constant_like(s: pd.Series) -> bool:
    nunique = int(s.nunique(dropna=False))
    top_freq = float(s.value_counts(dropna=False, normalize=True).iloc[0]) if len(s) else 1.0
    return bool((nunique <= 1) or (top_freq >= 0.995))


def numeric_like_object(s: pd.Series) -> Tuple[bool, float]:
    if s.dtype != "object":
        return False, 0.0
    ss = s.dropna().astype(str).str.strip()
    if ss.empty:
        return False, 0.0
    parsed = pd.to_numeric(ss, errors="coerce")
    rate = float(parsed.notna().mean())
    return bool(rate >= 0.85), rate


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


def main() -> None:
    print("=== ML Workflow Wizard (DataPrep -> AutoML) ===")

    csv_path = strip_quotes(input("Enter path to RAW CSV file: "))
    if not csv_path or not Path(csv_path).exists():
        raise SystemExit(f"ERROR: CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ------------------ DataPrep ------------------
    outdir = Path("prep_out")
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n[DataPrep] Dataset overview")
    print(f"  shape: {df.shape[0]} rows × {df.shape[1]} cols")
    for c in df.columns:
        print(f"  - {c} | dtype={df[c].dtype} | missing={pct(df[c].isna().mean())}%")

    threshold_str = input("\n[DataPrep] Drop columns with missing% >= ? (default 60): ").strip()
    high_missing_threshold = 0.60 if not threshold_str else max(0.0, min(1.0, float(threshold_str)/100.0))

    drop_cols: List[str] = []
    for c in df.columns:
        if is_id_like(c, df[c], df.shape[0]) and ask_yes_no(f"[DataPrep] Drop ID-like '{c}'?", default=True):
            drop_cols.append(c)

    for c in df.columns:
        if c in drop_cols:
            continue
        if is_constant_like(df[c]) and ask_yes_no(f"[DataPrep] Drop constant-like '{c}'?", default=True):
            drop_cols.append(c)

    for c in df.columns:
        if c in drop_cols:
            continue
        mr = float(df[c].isna().mean())
        if mr >= high_missing_threshold and ask_yes_no(f"[DataPrep] Drop high-missing '{c}' ({pct(mr)}%)?", default=False):
            drop_cols.append(c)

    cleaned = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    obj_cols = [c for c in cleaned.columns if cleaned[c].dtype == "object"]
    if obj_cols and ask_yes_no(f"[DataPrep] Normalize {len(obj_cols)} text columns (strip spaces)?", default=True):
        for c in obj_cols:
            cleaned[c] = normalize_text_series(cleaned[c])

    # numeric coercion
    for c in obj_cols:
        flag, rate = numeric_like_object(cleaned[c])
        if flag and ask_yes_no(f"[DataPrep] Coerce '{c}' to numeric? (parse_rate={pct(rate)}%)", default=True):
            cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")

    cleaned_csv = outdir / "cleaned.csv"
    cleaned.to_csv(cleaned_csv, index=False)

    (outdir / "cleaning_report.json").write_text(json.dumps({
        "input_csv": csv_path,
        "output_csv": str(cleaned_csv),
        "dropped_columns": drop_cols,
        "high_missing_threshold": high_missing_threshold,
        "input_shape": list(df.shape),
        "output_shape": list(cleaned.shape),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DataPrep] Saved cleaned CSV:", cleaned_csv)

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
        # pick a reasonable default (id-like), otherwise first column
        default_g = 0
        for i, c in enumerate(candidates):
            if is_id_like(c):
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
    # ---- Full evaluation summary (independent of tuning metric) ----
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