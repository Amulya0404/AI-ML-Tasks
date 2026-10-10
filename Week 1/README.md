# Week 1 — Python for Data Science

**Dataset:** Titanic passengers (`titanic.csv`, 891 rows x 15 columns). Source: seaborn-data, https://github.com/mwaskom/seaborn-data

## Files
- `Week1_Python_Data_Analysis.ipynb` — full executed notebook
- `titanic.csv` — original dataset; `titanic_cleaned.csv` — cleaned output
- `chart1..4_*.png` — Matplotlib visualizations

## What was done
1. Python fundamentals demo (types, conditionals, loops, functions, collections, exception handling) and NumPy basics.
2. Loaded data with Pandas; 891 rows x 15 columns.
3. Inspected structure and dtypes; found missing values in `age` (19.9%), `deck` (77.2%), `embarked`/`embark_town` (2 each).
4. Removed 107 duplicate rows -> 784 rows.
5. Cleaning: dropped `deck`, `alive`, `class`; filled `age` with median and embarkation fields with mode. Result: 784 x 12, no missing values.
6. Statistics: overall survival 41.2%; female 74.1% vs male 21.6%; 1st/2nd/3rd class 63.1% / 50.9% / 25.7%.
7. Four charts: survival by sex (bar), age distribution (histogram), class share (pie), fare vs age (scatter).

## Run
`pip install pandas matplotlib jupyter` then open the notebook.
