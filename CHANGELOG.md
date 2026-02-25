# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning (SemVer).

------------------------------------------------------------------------

## [2.0.0] - 2026-02-23

### Added

-   Advanced real-world numeric parsing in DataPrep:
    -   Handles thousands separators (e.g., `10,000`)
    -   Handles currency/unit suffixes (e.g., `1000원`, `$1000`)
    -   Handles parentheses negatives (e.g., `(1,000)` → `-1000`)
    -   Optional percent-to-fraction conversion (e.g., `12%` → `0.12`)
-   Date parsing for object columns:
    -   Automatic detection of date-like columns
    -   `dayfirst` option for ambiguous formats
    -   Optional ISO-format export for cleaned CSV
-   Data Quality Scan:
    -   Flags currency tokens, comma usage, percent signs
    -   Flags date-like strings
    -   Flags whitespace issues
    -   Flags high-cardinality / ID-like columns
-   Cleaning Modes:
    -   Quick
    -   Standard
    -   Strict
-   Batch decision system:
    -   Drop-all with optional keep-list
    -   Batch numeric parsing decisions
    -   Batch date parsing decisions
-   Protected columns feature
-   Cleaning config persistence:
    -   `cleaning_config.json`
    -   Reuse previous config on subsequent runs
-   Cleaning reproducibility artifacts:
    -   `cleaning_steps.py`
    -   Enhanced `cleaning_report.json`

### Changed

-   Improved ID-like detection:
    -   Includes monotonic sequence detection
    -   Improved near-unique detection
-   Reduced interactive prompt fatigue via batching and mode-based
    defaults
-   Improved user experience with summarized pre-clean overview

### Fixed

-   Group split default selection bug in AutoML phase

------------------------------------------------------------------------

## [1.0.0] - 2026-02-19

### Added

-   Interactive DataPrep step (missing threshold drop, ID-like drop
    prompt, text normalization).
-   Automatic task detection (regression or classification).
-   Baseline model comparison (Dummy model).
-   Model tuning via GridSearchCV (5-fold CV).
-   Supported models:
    -   Ridge
    -   Random Forest
    -   Extra Trees
    -   HistGradientBoosting
-   Full evaluation summary independent of tuning metric:
    -   Classification: Accuracy, F1, Balanced Accuracy, ROC-AUC,
        PR-AUC, Log Loss, Confusion Matrix (with interpretation).
    -   Regression: R², RMSE, MAE (with interpretation).
-   Reproducibility artifacts:
    -   best_pipeline.joblib
    -   report.json
    -   run_config.json
    -   requirements.txt
-   Environment version tracking (Python, NumPy, Pandas, scikit-learn,
    Joblib).
-   Optional stability evaluation (cross-validated summary).

### Changed

-   Outcome-like column removal is now confirmable via user prompt
    before dropping.

### Notes

-   Model files saved with joblib should be loaded using the same
    environment versions.
-   requirements.txt and env_versions in run_config.json help ensure
    reproducibility.
