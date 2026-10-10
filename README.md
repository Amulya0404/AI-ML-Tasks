AI/ML Internship — 4-Week Practical Task Plan

Practical work for the Next Gen Software Hub AI/ML internship. All four tasks use one dataset, the Titanic passenger data, and build on each other: from Python basics, to cleaning and EDA, to model building, to a complete project with a web app.

Dataset source: seaborn-data repository, https://github.com/mwaskom/seaborn-data (titanic.csv, 891 passengers, 15 columns).

Repository structure
README.md                         this file
Week1/                            Python & data analysis fundamentals
Week2/                            data preprocessing & EDA
Week3/                            ML model development
Titanic_Survival_Prediction/      Week 4 final end-to-end project
Summary of results
Week	Task	Main outcome
1	Python for Data Science	Dataset loaded, cleaned (891 → 784 rows) and analysed; 4 charts
2	Data Cleaning & EDA	Preprocessed dataset (773 × 12), EDA charts, feature importance
3	Build & Evaluate ML Models	Logistic Regression vs Random Forest, about 80% test accuracy
4	End-to-End Project	Tuned model (80.1% test accuracy), prediction system and Streamlit app
Week 1 — Python & Data Analysis Fundamentals

Folder: Week1/

File	Description
Week1_Python_Data_Analysis.ipynb	Executed notebook
titanic.csv, titanic_cleaned.csv	Original and cleaned datasets
chart1..4_*.png	Matplotlib visualizations
README.md	Short explanation of the work

What was done

Python fundamentals demo (variables, conditionals, loops, functions, lists/tuples/dictionaries, exception handling) and NumPy basics.
Loaded the data with Pandas: 891 rows × 15 columns.
Missing values: age 177 (19.9%), deck 688 (77.2%), embarked and embark_town 2 each.
Removed 107 duplicate rows, leaving 784. Dropped deck, alive and class; filled age with the median and the embarkation fields with the mode. Result: 784 × 12, no missing values.
Statistics: overall survival 41.2%; women 74.1% vs men 21.6%; 1st/2nd/3rd class 63.1% / 50.9% / 25.7%.
Four charts: survival by sex (bar), age distribution (histogram), class share (pie), fare vs age (scatter).
Week 2 — Data Preprocessing & Exploratory Data Analysis

Folder: Week2/

File	Description
Week2_Data_Cleaning_EDA.ipynb	Executed notebook
titanic_week2_cleaned.csv, titanic_week2_scaled.csv	Cleaned datasets (773 × 12), unscaled and scaled
eda1..5_*.png	Outlier boxplots, relationships, correlation heatmap, extra charts, feature importance
Preprocessing_Documentation.md	Every cleaning decision and its reason
Key_Findings.md	Short summary of insights

What was done

Missing values: dropped deck; filled age with the median per class and sex; filled embarked with the mode.
Removed 118 duplicate rows (found after imputation) → 773 rows.
Outliers identified with the IQR method and boxplots; real values kept, only fare capped at its 99th percentile (262.38).
Added family_size and is_alone; encoded sex (0/1) and embarked (one-hot); standardized age, fare, family_size.
Relationship analysis, correlation matrix and charts.
Feature importance with mutual information and Random Forest: sex, class and fare matter most.

Key findings: sex is the strongest signal (correlation −0.51 with survival); women in 1st/2nd class survived at 96.7% / 91.7%; children aged 0–12 survived at 57.4%; families of 2–4 did better than people travelling alone, while families of 5 or more did worst.

Week 3 — Machine Learning Model Development

Folder: Week3/

File	Description
Week3_ML_Model_Development.ipynb	Executed notebook
model_logistic_regression.joblib, model_random_forest.joblib	Trained models
evaluation_results.csv, cross_validation_results.csv	Metrics
cm_confusion_matrices.png, comparison_metrics.png, comparison_roc.png, feature_drivers.png	Charts
Model_Comparison_and_Conclusion.md	Comparison and conclusion

Problem: binary classification, predicting Titanic survival. Stratified 80/20 split (618 train / 155 test); the scaler is fit on the training data only.

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.800	0.754	0.766	0.760	0.879
Random Forest	0.819	0.909	0.625	0.741	0.873

Conclusion: the two models are roughly equal overall (5-fold cross-validated accuracy 0.784 vs 0.796, within one standard deviation) but make different mistakes. Random Forest is cautious, with high precision and low recall; Logistic Regression is more balanced. Sex, class and age drive the predictions.

Week 4 — Final End-to-End Project

Folder: Titanic_Survival_Prediction/

Path	Description
app.py	Streamlit web app
src/pipeline.py	Feature engineering and preprocessing pipeline
src/train.py	Tuning, evaluation, saves model and results
src/predict.py	Prediction function and command-line tool
notebooks/Titanic_Survival_Prediction.ipynb	End-to-end executed notebook
data/titanic.csv	Dataset
models/survival_model.joblib	Trained model (a full pipeline that takes raw passenger fields)
results/	Metrics, comparison table, charts
screenshots/	Screenshots of the running app
Project_Report.pdf / .md	Full 15-section project report
README.md, requirements.txt	Project README and dependencies

What was done

Removed 111 duplicates → 780 passengers; missing values filled inside the pipeline so the test data never influences them.
Features: family_size, is_alone, log_fare; one-hot encoding and scaling.
Four models (Logistic Regression, Random Forest, Gradient Boosting, SVM) tuned with 5-fold cross-validation on the training set only. Gradient Boosting was selected by cross-validated F1.
Selected model on 156 held-out passengers: accuracy 80.1% (baseline 59.0%), precision 0.824, recall 0.656, F1 0.730, ROC-AUC 0.855.
Most important features (permutation importance): sex, then ticket class, then fare and age.

Notes: the model rankings differ between cross-validation and the test set, and the 156-passenger test set is small, so small differences between models should not be over-interpreted. No public demo link is included because the app is not deployed online; it runs locally.

How to run
bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter

# Weeks 1-3: open the notebook in each folder
jupyter notebook Week1/Week1_Python_Data_Analysis.ipynb

# Week 4: run from inside Titanic_Survival_Prediction/
cd Titanic_Survival_Prediction
pip install -r requirements.txt
python -m src.train                  # re-train and regenerate results (optional)
python -m src.predict --pclass 1 --sex female --age 29 --fare 80 --embarked C
streamlit run app.py                 # web app at http://localhost:8501
Technologies

Python 3.12, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit, Jupyter.

Submission notes
All work is original and uses a public dataset, cited above.
Each week's task was completed before moving to the next, and each notebook has been executed with its outputs saved.
