# Analysis plan, fixed before test-set evaluation

Decision date: 2026-10-30. This is a prospective plan within this project,
not an externally registered preregistration. Data inspection has included
schema, ranges, missingness and duplication, but no model predictions or error
comparisons. No final test effect informs the planning effect size.

## Question and hypotheses

Does StandardScaler reduce KNN regression mean absolute error for red Vinho
Verde wine? For each held-out wine, d = |y - prediction_unscaled| -
|y - prediction_scaled|. Positive d favors scaling. H0: population mean d <= 0;
H1: population mean d > 0. One primary paired t-test, one-sided alpha = .05.
The unit is one unique test wine record, never one model prediction or one fold.

## Eligibility and duplicate policy

Use the supplied semicolon-delimited red CSV and all 11 physicochemical
predictors. Validate numeric, finite entries and integer quality from 0 to 10.
Missing, invalid or unexpected columns cause a visible failure rather than an
unplanned imputation. Inspection found no missing/invalid cells.

Remove exact duplicates across all 12 original columns BEFORE the split,
retaining the first source record and a mapping for every original record.
Red: 1,599 -> 1,359 records (240 removed); white: 4,898 -> 3,961 (937 removed).
There are no identical predictor vectors with conflicting quality scores in
either file. If future input contains such conflicts, stop and revise the plan
to use predictor groups, rather than choosing a label or allowing leakage.
This estimates performance over unique observed profiles, not the original
frequency-weighted population. Identical records may be genuine repeated
measurements; their provenance cannot be resolved from these files. Deduplication
does not prove independence or eliminate related batches/near duplicates.

## Split and model specification

Use sklearn train_test_split, shuffle=True, test_size=.20, random_state=42,
without stratification. Red: 1,087 train / 272 test; white: 3,168 / 793.
Freeze the same memberships and training order for both models. Do not search
seeds. KNeighborsRegressor: n_neighbors=5, weights='uniform',
metric='euclidean', algorithm='brute', n_jobs=1 in both arms. Brute force fixes
the neighbor-search method; ties at the kth distance can depend on training
order. Preserve that order. Only treatment adds StandardScaler in a pipeline,
fit on training predictors only; never scale the target. No tuning, feature
selection, outlier removal, rounding of predictions, or quality stratification.

## A priori power

Before evaluation call statsmodels.stats.power.TTestPower.solve_power with
effect_size=.20, alpha=.05, power=.80, alternative='larger'. Here dz is the
population mean difference divided by its SD, a small ASSUMED paired effect,
not an estimate from these test wines. Round required pairs UP to an integer.
Compare planned test sizes and calculate design power using .20 only. Assumes
independent pairs, common difference distribution, and the normal-difference
model used for the t-test power calculation. Adequacy is conditional on those
assumptions and this effect; it does not guarantee significance or power for a
smaller true effect. No observed/post-hoc power.

## Analysis and reporting

Primary endpoint: MAE. RMSE is descriptive, without a second confirmatory test.
Report both metrics, mean d, sample SD (ddof=1), SE, t = mean(d)/(SD/sqrt(n)),
df=n-1, right-tail p, and observed dz=mean(d)/SD. Retain the prespecified
one-sided test even if the observed direction is negative.

Report a TWO-SIDED 95% Student t interval mean(d) +/- t(.975,df)*SE. Also report
the ONE-SIDED 95% lower bound mean(d)-t(.95,df)*SE, whose positive bound matches
the primary rejection rule. A two-sided 95% interval need not exclude zero when
a one-sided .05 test rejects. Intervals condition on the fitted training set.

Assess the DIFFERENCES using histogram, Q-Q plot, skewness, excess kurtosis,
Shapiro-Wilk (descriptive only), quartiles, extremes, ties and leave-one-out
mean range. No assumption-test-driven change to the primary test. Bounded,
discrete differences can be nonnormal; a moderate/large independent test sample
can support a CLT approximation but cannot establish independence. As a
prespecified descriptive sensitivity check, compute a paired percentile
bootstrap 95% interval for the mean d with 10,000 row resamples and seed 4242.
This uses the same test pairs and is not independent confirmation.

Do not automatically substitute Wilcoxon: a location interpretation requires
approximately symmetric differences, and ties/zeros require care. The default
analysis omits it because symmetry and a continuous location-shift model are
not established; the paired bootstrap targets the requested mean directly.

## Secondary replication and conclusions

Run the unchanged protocol on white wine as a separate secondary replication
after the red analysis, using independently trained models. Never pool its
pairs or p-values with red. It is an additional cohort from the same dataset
source, not proof of independence of collection/batches. Red remains the sole
confirmatory test; secondary and diagnostic p-values cannot rescue it.

Use reject/fail to reject, not accept/prove H0. Distinguish statistical evidence
from the observed effect in quality-score units and generalization. Limit claims
to these models, k, split, unique-record population and dataset context. Treat
quality as numeric for this regression comparison while acknowledging its
ordinal, imbalanced expert-score origin. Do not infer effects across new
training sets from this single holdout or claim scaling universally improves
KNN. Save all predictions, errors, memberships, scaler training statistics,
raw-file hashes, software versions and plan hashes.
