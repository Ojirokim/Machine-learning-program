
# Changelog
All notable changes to this project will be documented in this file.

This project follows Semantic Versioning (SemVer).

---

## [1.0.0] - 2026-02-19

### Added
- Interactive DataPrep step (missing threshold drop, ID-like drop prompt, text normalization).
- Automatic task detection (regression or classification).
- Baseline model comparison (Dummy model).
- Model tuning via GridSearchCV (5-fold CV).
- Supported models:
  - Ridge
  - Random Forest
  - Extra Trees
  - HistGradientBoosting
- Full evaluation summary independent of tuning metric:
  - Classification: Accuracy, F1, Balanced Accuracy, ROC-AUC, PR-AUC, Log Loss, Confusion Matrix (with interpretation).
  - Regression: R², RMSE, MAE (with interpretation).
- Reproducibility artifacts:
  - best_pipeline.joblib
  - report.json
  - run_config.json
  - requirements.txt
- Environment version tracking (Python, NumPy, Pandas, scikit-learn, Joblib).
- Optional stability evaluation (cross-validated summary).

### Changed
- Outcome-like column removal is now confirmable via user prompt before dropping.

### Notes
- Model files saved with joblib should be loaded using the same environment versions.
- requirements.txt and env_versions in run_config.json help ensure reproducibility.
