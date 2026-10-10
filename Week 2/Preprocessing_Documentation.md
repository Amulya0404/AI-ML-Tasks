# Week 2 — Data Preprocessing Documentation

**Dataset:** Titanic (891 x 15), source: https://github.com/mwaskom/seaborn-data

| Step | Issue found | Action | Result |
|---|---|---|---|
| Missing values | `deck` 77% missing; `age` 20% missing; `embarked` 2 missing | Dropped `deck`; `age` filled with median by (`pclass`, `sex`) group; `embarked` filled with mode | 0 missing values |
| Duplicates | 118 duplicate rows (found after imputation, which makes more rows identical) | Removed | 773 rows |
| Outliers (IQR) | age 13, fare 97, sibsp 39, parch 15 flagged | Values are real, not errors, so kept; only `fare` capped at 99th percentile (262.38), 7 values changed | Reduced skew impact |
| Feature engineering | `sibsp`/`parch` split family info | Added `family_size = sibsp + parch + 1` and `is_alone` | 2 new features |
| Redundant columns | `class`, `alive`, `who`, `adult_male`, `embark_town`, `alone` repeat other columns or the target | Dropped | 12 columns remain |
| Encoding | `sex`, `embarked` are text | `sex` -> 0/1 (male = 1); `embarked` -> one-hot (C, Q, S) | All numeric |
| Scaling | `age`, `fare`, `family_size` on different scales | StandardScaler (mean 0, std 1); binary/one-hot columns untouched | Saved as separate file |

## Output files
- `titanic_week2_cleaned.csv` — encoded, unscaled (use for tree models)
- `titanic_week2_scaled.csv` — same with `age`, `fare`, `family_size` standardized (use for logistic regression / KNN / SVM)

## Notes
- `sibsp`, `parch` and `family_size` are correlated by construction. Keep all for tree models, or drop `family_size` / the originals for linear models.
- The scaler was fit on the whole dataset for this EDA task. In Week 3, fit it on the training split only to avoid data leakage.
- The duplicate count differs from Week 1 (107) because Week 2 imputes `age` by group first, which makes more rows identical. Rows with identical values can also be genuinely different passengers, since the dataset has no ID column.
