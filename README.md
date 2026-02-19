
# ML Workflow Wizard

### Reproducible Supervised Machine Learning CLI (DataPrep → AutoML → Evaluation)

ML Workflow Wizard is an interactive command-line tool for building supervised machine learning models on tabular CSV data.

It guides users through:

- Data cleaning
- Target selection
- Model training and tuning
- Full evaluation reporting
- Reproducible artifact generation

This tool is designed for structured datasets and supports both regression and classification tasks.

---

## Features

### Data Preparation
- Dataset overview (shape, dtype, missing %)
- Drop high-missing columns (configurable threshold)
- Optional ID-like column removal
- Optional text normalization (strip whitespace)
- Cleaned dataset export

### Supervised Learning (AutoML)
- Automatic task detection (regression or classification)
- Baseline model comparison
- Model tuning via GridSearchCV (5-fold CV)
- Models included:
  - Ridge
  - Random Forest
  - Extra Trees
  - HistGradientBoosting

### Full Evaluation Summary (Independent of Tuning Metric)

For classification:
- Accuracy
- F1 Score (weighted)
- Balanced Accuracy
- ROC-AUC (if supported)
- PR-AUC (binary)
- Log Loss
- Confusion Matrix
- Human-readable interpretation

For regression:
- R²
- RMSE
- MAE
- Human-readable interpretation

### Reproducibility
- best_pipeline.joblib
- report.json
- run_config.json
- requirements.txt
- Environment version tracking (Python, NumPy, Pandas, scikit-learn, Joblib)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If requirements.txt is not present yet:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## Usage

Run the wizard:

```bash
python ml_workflow_wizard.py
```

You will be prompted to:

1. Enter a CSV file path
2. Choose preprocessing options
3. Select a target column
4. Select a tuning metric
5. Choose split strategy (random/group/time if enabled)
6. (Optional) Run stability evaluation

---

## Example Output Artifacts

After a run, the following files are generated:

```
prep_out/cleaned.csv
best_pipeline.joblib
report.json
run_config.json
requirements.txt
```

### best_pipeline.joblib
Contains:
- Preprocessing pipeline
- Trained model
- Tuned hyperparameters

### report.json
Contains:
- Best model name
- CV score
- Test score
- Full evaluation metrics
- Baseline score

### run_config.json
Contains:
- Selected target
- Removed columns
- Feature list
- Split configuration
- Environment versions

---

## Design Philosophy

This project emphasizes:

- Clear supervised learning workflow
- Reproducibility
- Interpretability
- Practical evaluation beyond a single metric
- Clean CLI-based experimentation

It is not intended for:
- Deep learning
- Large-scale distributed training
- Production API serving

---

## Versioning

This project follows Semantic Versioning.

Current stable release:

v1.0.0

All updates are documented in:

CHANGELOG.md

---

## License

MIT License\
See LICENSE file for details.