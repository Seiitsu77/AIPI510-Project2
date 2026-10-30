# Abstract, extended background, and recommendations

Supplement to [`statistical_report.md`](statistical_report.md). That file is
regenerated from saved results every time the pipeline runs, so this text is
kept in a separate, hand-written file. All numbers below come from
`results/red/` and `results/white/` and were independently recomputed by
[`scripts/validate_independent.py`](../scripts/validate_independent.py)
(output: `results/validation/independent_validation.json`).

## Abstract

K-nearest neighbors (KNN) predicts a wine's quality from the closest wines in
the training set, so predictors measured in large units can dominate the
distance. We tested whether standardizing the 11 physicochemical predictors
(StandardScaler, fit on training data only) reduces the absolute prediction
error of a k = 5 KNN regressor on the UCI Wine Quality data. Using one fixed,
prespecified 80/20 split of de-duplicated records, we paired each held-out
wine's absolute error under the two models and ran a one-sided paired t-test
(alpha = .05; an a priori power analysis assuming dz = 0.20 required 156 pairs).
In red wine (272 test wines), scaling lowered mean absolute error (MAE) from
0.632 to 0.540 quality points (14.6%): mean paired reduction 0.092 points,
95% CI [0.038, 0.146], t(271) = 3.34, p = .0005, dz = 0.20, so we reject H0. A
secondary white-wine replication (793 test wines) agreed: MAE 0.691 to 0.603,
mean reduction 0.089 [0.055, 0.122], p < .001, dz = 0.18. The effect is
statistically clear but small: in red wine scaling improved 134 of 272
predictions, worsened 94 and left 44 unchanged. Results are conditional on one
training set, one k and one split, so we recommend standardizing predictors
before distance-based modeling while treating the size of the gain as modest.

## Extended background and motivation

**The data.** The Wine Quality data (Cortez et al., 2009) describe red and white
variants of Portuguese Vinho Verde wine: 1,599 red and 4,898 white records, each
with 11 laboratory measurements and an expert quality score. The score is the
median of at least three evaluations by wine experts on a 0 to 10 scale. Quality
is therefore ordinal and subjective; this project treats it as a number, as a
modeling convention, not as a physical quantity.

**What the measurements are.** Fixed acidity is the non-volatile acids (mainly
tartaric); volatile acidity is mostly acetic acid, which tastes of vinegar at
high levels; citric acid adds freshness. Residual sugar is the sugar left after
fermentation. Chlorides measure salt. Free sulfur dioxide is the unbound,
active preservative that protects against microbes and oxidation, and total
sulfur dioxide is free plus bound. Sulphates are an additive that contributes to
sulfur dioxide levels. Density depends mainly on alcohol and sugar, pH measures
acidity on a log scale, and alcohol is percent by volume. These variables are
measured in very different units, which is exactly what makes the scaling
question interesting.

**Why scale matters for KNN.** KNN finds neighbors with Euclidean distance, which
adds up squared differences in raw units. A feature with a large spread therefore
decides who the neighbors are, regardless of whether it is informative. In these
data the standard deviation of total sulfur dioxide is about 33 mg/dm^3 in red
wine, while the standard deviation of density is about 0.0019 g/cm^3, a ratio
near 17,000. Because the expected squared distance between two random wines is
proportional to each feature's variance, we can compute each feature's share of
the unscaled distance. In red wine, total sulfur dioxide accounts for 90.8% and
free sulfur dioxide for 8.7%, so those two features supply 99.5% of the
distance; alcohol supplies about 0.1%. In white wine the shares are 85.7% and
13.0% (98.7% together). Unscaled KNN is therefore close to a two-feature model
of a wine's sulfur dioxide, with acidity, sugar and alcohol nearly invisible.
Standardization gives every predictor unit variance so each can influence the
neighbors.

**Why this is still an empirical question.** Equal weighting is not automatically
better: it also raises the influence of noisy or uninformative predictors. Whether
scaling helps this model on these wines, and by how much, has to be tested and
not assumed. That is the question this project answers, and it is one every
practitioner meets, since preprocessing choices are often made by habit.

## Conclusions and recommendations

**Conclusions.**
- Red wine (primary): reject H0. Scaling reduced KNN mean absolute error by
  0.092 quality points (14.6% of the unscaled MAE), with a 95% CI of
  [0.038, 0.146] and a small standardized effect (dz = 0.20). White wine, a
  secondary replication, shows the same direction and similar size (0.089
  points, 12.8%).
- The gain is real but modest. Quality scores are integers, so the average
  improvement is under a tenth of one score step, and scaling made about a
  third of red-wine predictions worse (94 of 272) while 44 were unchanged.
- Exploratory sensitivity checks that were not in the analysis plan agree with
  the primary test in red wine: Wilcoxon signed-rank p = .003, sign test
  p = .005, paired t-test on squared errors p = .002. They do not replace the
  prespecified result.
- An improved MAE is not a claim that chemistry determines taste or that scaling
  changes wine quality; it says only that the scaled model's predictions land
  closer to expert scores on these held-out wines.

**Recommendations.**
1. Standardize predictors before KNN, and other distance-based or kernel methods,
   when features have different units. It is cheap and helped in both cohorts.
2. Fit the scaler inside a pipeline on training data only, so test wines never
   influence the scaling. This project verified that no leakage occurred.
3. Treat scaling as sound default practice, not a large performance lever. A
   dz near 0.2 is conventionally small (Cohen, 1988).
4. Before claiming benefit beyond these data, repeat the comparison over many
   splits or cross-validation folds, tune k separately for each arm, try other
   scalers such as min-max or robust scaling, and compare with a model that does
   not depend on feature scale. If batch or producer identifiers become
   available, split by group.
5. Remove exact duplicates before splitting, since copies on both sides of the
   split would flatter any nearest-neighbor method.

The main report's Limitations section applies in full: one fitted training set,
unobserved dependence between related wines, a single k, and noisy ordinal
scores.

## Independent validation summary

`scripts/validate_independent.py` does not import the project package. For each
cohort it recomputes the paired t-test, confidence interval, dz, MAE/RMSE and
power analysis from the saved predictions, refits both KNN models from the
cleaned data and saved split, and checks leakage and duplicate handling. All 25
primary checks passed for red and for white, and the recomputed values match
the saved results to numerical precision. The Wilcoxon, sign-test,
squared-error and second-bootstrap checks are labeled exploratory in the
output.

## Additional references

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
  (2nd ed.). Lawrence Erlbaum Associates.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning* (2nd ed.). Springer.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research, 12*, 2825-2830.

(Cortez et al., 2009, and the software documentation are already cited in
`statistical_report.md`.)
