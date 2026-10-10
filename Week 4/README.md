# Titanic Survival Prediction — End-to-End ML Project

Predicts whether a Titanic passenger survived from ticket class, sex, age, family aboard, fare and port of embarkation.
Final project (Task 4) of the 4-week AI/ML internship.

**Result:** Gradient Boosting, selected by 5-fold cross-validation, reaches **82.1% test accuracy** and **ROC-AUC 0.881**
(baseline "always died" = 58.7%). Precision 0.86, recall 0.67, F1 0.75 on 156 held-out passengers.

## Project structure
```
data/        titanic_raw.csv (source: https://github.com/mwaskom/seaborn-data), titanic_cleaned.csv
notebooks/   Titanic_Survival_Project.ipynb   - full workflow: cleaning, EDA, preprocessing, models, evaluation
src/         pipeline.py (cleaning + preprocessing + feature engineering), train.py, predict.py
models/      best_model.joblib (deployed), all_tuned_models.joblib, model_info.json
reports/     Project_Report.docx, model_comparison.csv, figures/
screenshots/ Streamlit app screenshots
app.py       Streamlit web app
```

## How to run (from the project root)
```bash
pip install -r requirements.txt
python src/train.py                  # re-train, tune, compare, save models/best_model.joblib
python src/predict.py --pclass 1 --sex female --age 29 --fare 80 --embarked C   # command-line prediction
streamlit run app.py                 # web app at http://localhost:8501
```
`--age` can be left out when unknown (it is imputed).

## Method in brief
1. Remove 111 duplicate rows (891 -> 780); drop `deck` (77% missing) and redundant columns.
2. One scikit-learn pipeline: engineered features (`family_size`, `is_alone`, `log(fare)`), median/most-frequent imputation, scaling, one-hot encoding. Fit on training data only, so no leakage.
3. Stratified 80/20 split; Logistic Regression, Random Forest and Gradient Boosting tuned with 5-fold GridSearchCV (ROC-AUC); model chosen on CV, not on the test set.
4. Final model refit on all 780 rows for deployment; reported test metrics come from the train-only version.

| Model | CV ROC-AUC | Test accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.831 | 0.795 | 0.742 | 0.766 | 0.754 | 0.878 |
| Random Forest | 0.845 | 0.808 | 0.870 | 0.625 | 0.727 | 0.862 |
| **Gradient Boosting** | **0.854** | **0.821** | 0.860 | 0.672 | 0.754 | 0.881 |

## Limitations
Small dataset (test set of 156, so differences between models are within noise); no passenger ID, so some "duplicates" may be different people; the model misses about a third of survivors (recall 0.67); it is an educational demo trained on 1912 data.
