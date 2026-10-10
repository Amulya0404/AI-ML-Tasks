# Week 2 — Summary of Key Findings

1. **Sex is the strongest signal.** Correlation with survival is -0.51 (male = 1). Women in 1st and 2nd class survived at 96.7% and 91.7%; men in 2nd and 3rd class at 18.5% and 16.1%.
2. **Class matters.** Correlation with `pclass` is -0.34, and `fare` is +0.26 (higher fare, better survival). Third-class women were the exception among women, at 47.2%.
3. **Age:** children aged 0-12 survived at 57.4% versus 40.6% for ages 19-35 and 23.8% for 60+ (only 21 passengers). Linear correlation with age is weak (-0.07), so its value is non-linear.
4. **Travelling with family helped.** Passengers not alone survived at 51.0% versus 34.2% for those alone, but the effect is non-linear: families of 2-4 survived at 55-71%, while families of 5 or more survived at 0-33% (13 passengers at size 5, 22 at size 6). That is why `family_size` has near-zero linear correlation (0.01).
5. **Port:** embarking at Cherbourg correlates positively with survival (+0.17), consistent with Cherbourg carrying the highest share of 1st-class passengers (53.6%, versus 22.2% for Southampton and 3.4% for Queenstown).
6. **Feature importance.** Mutual information ranks sex (0.136), fare (0.120) and pclass (0.108) at the top. Random Forest ranks age (0.275), fare (0.274) and sex (0.227) highest, but random forests tend to favour continuous features, so the mutual-information ranking is the more reliable guide. Embarkation port and `is_alone` are weak predictors.
7. **Recommended features for Week 3:** `sex`, `pclass`, `fare`, `age`, `family_size`/`is_alone`; `embarked_*` optional.
8. **Class balance:** 773 passengers after cleaning, 41.5% survived, so the target is moderately imbalanced but usable without resampling.
