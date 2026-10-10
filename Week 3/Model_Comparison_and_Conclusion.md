# Week 3 — Model Comparison & Conclusion

**Problem:** predict Titanic survival (binary classification). **Data:** 773 passengers, 11 features; stratified 80/20 split (618 train / 155 test).

## Test-set results (155 passengers)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.800 | 0.754 | 0.766 | 0.760 | 0.879 |
| Random Forest | 0.819 | 0.909 | 0.625 | 0.741 | 0.873 |

## Confusion matrices (rows: actual)
| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 75 | 16 | 15 | 49 |
| Random Forest | 87 | 4 | 24 | 40 |

## 5-fold cross-validation (all 773 passengers)
| Model | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---|
| Logistic Regression | 0.784 ± 0.022 | 0.733 ± 0.027 |
| Random Forest | 0.796 ± 0.017 | 0.724 ± 0.037 |

## Explanation of results
- **Neither model is clearly better.** Random Forest got 127 of 155 test passengers right versus 124 for Logistic Regression, a difference of 3 people. In cross-validation the accuracy gap (about 1.2 points) is smaller than one standard deviation, and Logistic Regression has the slightly higher F1 and ROC-AUC. With a test set this small, treat the models as roughly equal in overall quality.
- **They make different kinds of errors.** Random Forest is cautious: it rarely predicts survival wrongly (4 false positives, precision 0.91) but misses many real survivors (24 false negatives, recall 0.62). Logistic Regression is balanced (precision 0.75, recall 0.77), which gives it the better F1.
- **Choosing between them** depends on the goal. If missing a survivor is the costly mistake, prefer Logistic Regression. If a positive prediction must be trustworthy, prefer Random Forest.
- **What drives the predictions.** Logistic Regression's largest coefficients are sex (-2.17, being male lowers survival odds), passenger class (-1.08) and age (-0.60, older lowers odds). Random Forest also ranks sex first (0.376), then fare, age and class. Fare gets a near-zero coefficient in the linear model (0.05) because it overlaps with class, but the forest uses it, so some of its value may be non-linear.

## Conclusion
Both models reach about 78–82% accuracy, well above the 58.5% you would get by always predicting "died". Sex, class and age are the key predictors, matching the Week 2 EDA. Because the data is small, gains from further tuning would be hard to measure reliably.

## Limitations & next steps
- Single small test set (155 rows); repeated cross-validation is more trustworthy.
- No hyperparameter tuning beyond sensible defaults; Week 4 could add `GridSearchCV` and more models (Gradient Boosting, SVM).
- Features like title extracted from passenger names (not in this dataset) are known to help on Titanic.
