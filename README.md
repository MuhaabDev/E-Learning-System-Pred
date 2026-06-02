# Student Performance Prediction — OULAD

A end-to-end machine learning project on the Open University Learning Analytics Dataset (OULAD). The goal was to predict student assessment outcomes through two tasks: regression (predicting the exact score) and classification (predicting performance band: Fail, Good, Very Good, or Excellent).

Built as part of a university ML course project (team of 5).

---

## Dataset

The Open University Learning Analytics Dataset contains anonymised data from real students across multiple course modules.

| File | Description | Size |
|---|---|---|
| studentAssessment | Submission records with scores | 163,912 rows |
| studentInfo | Student demographics | 32,593 rows |
| studentVle | Clickstream activity logs | 10.6M rows |
| assessments | Assessment metadata | 206 assessments |
| studentRegistration | Enrolment records | 32,593 rows |
| vle | VLE resource metadata | 6,364 rows |
| courses | Module info | 22 modules |

After merging and cleaning: 163,752 rows for regression, 163,912 for classification, with 65+ features before engineering.

---

## Project Structure

```
├── data/                  # Raw CSV files (not included, see below)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_regression.ipynb
│   └── 05_classification.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── models.py
├── reports/
│   └── final_report.pdf
└── README.md
```

> The raw dataset is not included due to size. Download it from the [OULAD official page](https://analyse.kmi.open.ac.uk/open_dataset).

---

## Pipeline

### 1. Preprocessing

- Merged 7 CSV files using student ID, module code, and presentation code as keys
- Aggregated 10.6M VLE click logs into per-student summary statistics (total clicks, interaction days, unique activity types)
- Dropped `final_result` immediately to prevent data leakage
- Handled missing values per column based on what the missingness actually means:
  - `score` — rows deleted (no target, no point)
  - `date`, `date_registration` — filled with median (right-skewed)
  - `date_unregistration` — converted to binary flag `has_unregistered`
  - `imd_band` — mode for regression, 'Unknown' category for classification
  - VLE columns — filled with 0 (no record = no activity)
- Applied ordinal encoding for regression, one-hot encoding for classification

### 2. Feature Engineering

15+ new features created across three groups:

**Submission timing**
- `submission_offset` — days between submission and due date (negative = early)
- `submitted_late` — binary flag
- `submission_timing_ratio` — normalised across modules of different lengths

**Engagement**
- `click_per_day` — click intensity on active days
- `activity_diversity_ratio` — variety of resources used
- `engagement_ratio` — interaction days relative to module length
- `is_heavy_engager` — binary, above 75th percentile in total clicks

**Behaviour**
- `has_prior_attempts` — binary
- `engaged_but_late` — high engagement but still submitted late
- `low_engagement_late` — low engagement and late submission (double risk signal)

**Key finding:** `submitted_late` (|r| = 0.124) was the strongest engineered feature — submission timing mattered more than raw click volume.

### 3. Models

**Regression** — predicting exact assessment score (70/15/15 train/val/test split)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Gradient Boosting | **0.2800** | **15.9** | **12.1** |
| Random Forest | 0.2479 | 16.3 | 12.4 |
| Decision Tree | 0.2023 | 16.8 | 12.8 |
| Ridge | 0.1489 | 17.3 | 13.3 |
| Lasso | 0.1092 | 17.7 | 13.6 |

**Classification** — predicting performance band (80/20 stratified split)

| Model | Test Accuracy | Macro F1 | Train Time |
|---|---|---|---|
| Random Forest (depth=20) | **51.2%** | **0.508** | 42s |
| Gradient Boosting | 48.3% | 0.463 | 564s |
| Decision Tree | 45.6% | 0.434 | 3s |
| Logistic Regression | 44.2% | 0.409 | 7s |

> Chance baseline for 4-class classification = 25%. The Good class was consistently the hardest to predict — its behavioural profile overlaps heavily with neighbouring bands.

---

## Key Findings

- Assessment type and weight were the top regression features — exam vs. quiz differences drove score more than engagement volume
- Submission timing dominated classification — students who submit late are disproportionately likely to fall in Fail or Good
- Tree-based models consistently outperformed linear models, confirming a non-linear relationship between features and scores
- R² = 0.28 for regression is a real result on this data — much of what determines a score (understanding, motivation, question difficulty) simply does not appear in clickstream logs

---

## Tech Stack

- Python, NumPy, Pandas, Matplotlib
- scikit-learn (Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting)
- Jupyter Notebook

---

## How to Run

```bash
git clone https://github.com/MuhaabDev/elearning-performance-prediction
cd elearning-performance-prediction
pip install -r requirements.txt
```

Download the OULAD dataset and place the CSV files in `/data/`, then run the notebooks in order.

---

## References

- Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017). [Open University Learning Analytics Dataset](https://www.nature.com/articles/sdata2017171). Scientific Data.
