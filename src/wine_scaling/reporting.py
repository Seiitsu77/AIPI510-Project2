"""Generate the statistical report directly from saved results."""
from pathlib import Path
import pandas as pd
from .io import read_json


def build_report(root: Path):
    sections = ["""# Feature scaling and KNN wine-quality regression

## Background and research question

Does standardizing the 11 physicochemical predictors reduce KNN regression
absolute prediction error? Wine quality is an ordinal expert score treated as
numeric for this course experiment. KNN uses distances, so predictors measured
on very different scales can dominate neighbor selection. Standardization is
a plausible intervention; its benefit is an empirical question, not a guarantee.

The primary cohort is red Portuguese Vinho Verde wine. White wine, when present,
is a separate secondary replication from the same UCI source. Neither cohort is
pooled with the other. The analysis follows the [prospective project plan](analysis_plan.md),
committed before model evaluation; it is not an external preregistration.

## Hypotheses and experimental design

For a test wine, d = |y − prediction_unscaled| − |y − prediction_scaled|.
H0: μd ≤ 0; H1: μd > 0. Positive values favor scaling. The primary endpoint is
MAE and the primary test is a one-sided paired t-test at α = .05. RMSE is
descriptive. Each retained test observation contributes exactly one pair.

Both arms use the same unstratified 80/20 shuffled split (seed 42), training
order, 11 predictors, and KNeighborsRegressor with k=5, uniform weights,
Euclidean distance, brute-force neighbor search and one computational worker.
The treatment adds only StandardScaler, fitted inside a pipeline on training
predictors. The quality target is never scaled, and predictions stay continuous.
There is no tuning, seed selection, feature selection or outlier deletion.

Remove exact duplicates across predictors and quality before splitting, retaining
the first source record. Both supplied files have no missing/nonfinite cells or
conflicting labels for identical predictor vectors. Input checks fail visibly
if that changes. Deduplication prevents exact copies from straddling the split;
it also changes the estimand to unique recorded profiles. Raw copies, cleaned
records, duplicate mappings and train/test source IDs are saved.

## Statistical methods and assumptions

Let n be test pairs and sd the sample SD of d, with ddof=1. Compute SE=sd/√n,
t=mean(d)/SE, df=n−1, and the right-tail p-value. Observed Cohen's dz=mean(d)/sd
describes the standardized paired effect; it is separate from the assumed
planning dz=.20. A two-sided 95% t interval uses t(.975,df). The additional
one-sided 95% lower bound uses t(.95,df) and corresponds to the .05 directional
test. A two-sided 95% interval can contain zero even if the directional test rejects.

The paired t-test does not require the two arms to have equal error variances
or either error distribution to be normal. Its exact small-sample validity
requires independent, normally distributed differences. Here differences are
bounded and discrete, with ties. Histogram, Q–Q plot, skewness, kurtosis and
Shapiro–Wilk are diagnostics, not gates for choosing a more favorable test.
The CLT supports an approximation with sufficiently many independent pairs,
but diagnostics cannot establish independence. The reported inferential
uncertainty is conditional on this fitted training set and assumes test wines
represent independent draws from a relevant population, not merely a fixed
finite set. We use no finite-population correction for this superpopulation aim.

A prespecified paired percentile bootstrap (10,000 row resamples, seed 4242)
provides a descriptive 95% interval for the same mean. It shares the independence
assumption and is not independent confirmation. Wilcoxon is omitted: a symmetric
location-shift interpretation is unestablished, and tied/zero differences need
care. Its null is not the requested mean-difference null.

## A priori power analysis

Before model evaluation, statsmodels TTestPower.solve_power was called with
effect_size=.20, alpha=.05, power=.80 and alternative='larger'. The small paired
effect dz=.20 is supplied as a planning assumption, not estimated from final
test errors. Required n is rounded upward. This normal-difference planning
model assumes independent pairs; it does not promise power for smaller effects
or establish practical importance. No observed power is calculated.
"""]
    summaries = []
    for color, label in [("red", "Primary red-wine experiment"), ("white", "Secondary white-wine replication")]:
        directory = root / "results" / color
        if not (directory / "inference.json").exists():
            continue
        s = read_json(directory / "inference.json")
        p = read_json(directory / "power_plan.json")
        a = read_json(directory / "data_audit.json")
        manifest = read_json(directory / "evaluation_manifest.json")
        m = pd.read_csv(directory / "metrics.csv")
        unscaled, scaled = m.iloc[0], m.iloc[1]
        lo, hi = s["ci_95_two_sided"]
        blo, bhi = s["bootstrap_ci_95_percentile"]
        decision = "Reject H0" if s["reject_null"] else "Fail to reject H0"
        conclusion = ("The prespecified directional test provides evidence that scaling reduces mean absolute error for this fixed comparison."
                      if s["reject_null"] else
                      "The prespecified directional test does not provide sufficient evidence that scaling reduces mean absolute error for this fixed comparison. This does not establish equivalence or prove no benefit.")
        rel = 100 * s["mean_difference"] / unscaled.mae
        n = s["n_pairs"]
        summaries.append(f"- **{color.capitalize()}**: MAE {unscaled.mae:.4f} → {scaled.mae:.4f}; mean paired reduction {s['mean_difference']:.4f} points (two-sided 95% CI [{lo:.4f}, {hi:.4f}]); one-sided p={s['p_value_one_sided']:.6g}. {decision} at .05.")
        q = s["quantiles"]
        loo_lo, loo_hi = s["leave_one_out_mean_range"]
        sections.append(f"""
## {label}

### Data and planning

{a['raw_rows']:,} raw records − {a['removed_exact_duplicates']:,} exact duplicates =
{a['clean_rows']:,} retained records; {manifest['n_train']:,} train and {n:,} test.
Missing cells: {a['missing_cells']}; conflicting predictor profiles: {a['conflicting_predictor_profiles']}.
The power solver gives {p['required_pairs_continuous']:.4f} pairs, rounded up to
**{p['required_pairs']} pairs**. The test set of {n} is
{'sufficient' if p['sufficient_at_assumed_effect'] else 'insufficient'} for .80
power at the assumed dz=.20. Model-based design power at that assumption is
{p['design_power_at_assumed_dz']:.4f}. This was saved before evaluation.

### Results

| Model | MAE (quality-score points) | RMSE (quality-score points) |
| --- | ---: | ---: |
| Unscaled KNN | {unscaled.mae:.6f} | {unscaled.rmse:.6f} |
| StandardScaler + KNN | {scaled.mae:.6f} | {scaled.rmse:.6f} |

| Paired inference | Value |
| --- | ---: |
| Test pairs | {n} |
| Mean difference, unscaled − scaled | {s['mean_difference']:.6f} |
| SD of differences | {s['sd_difference']:.6f} |
| SE of mean difference | {s['se_difference']:.6f} |
| t-statistic | {s['t_statistic']:.6f} |
| Degrees of freedom | {s['degrees_of_freedom']} |
| One-sided p-value | {s['p_value_one_sided']:.8g} |
| Two-sided 95% t interval | [{lo:.6f}, {hi:.6f}] |
| One-sided 95% lower confidence bound | {s['lower_bound_95_one_sided']:.6f} |
| Observed Cohen's dz | {s['cohens_dz']:.6f} |
| Paired bootstrap 95% percentile interval | [{blo:.6f}, {bhi:.6f}] |

**{decision} at α=.05.** {conclusion}
The observed MAE change is {s['mean_difference']:.4f} score points, or {rel:.2f}%
of control MAE. No smallest practically meaningful improvement was specified;
statistical significance alone cannot determine educational or operational value.

![MAE comparison](../figures/{color}/mae_comparison.png)

Bars start at zero and describe test MAE; uncertainty about the paired contrast
appears in the dedicated mean-improvement interval below.

### Difference distribution and assumptions

There are {s['positive_pairs']} improvements, {s['negative_pairs']} deteriorations,
and {s['zero_pairs']} ties (ties counted with numerical rounding to 10 decimal
places only for diagnostics). Differences range from {q['min']:.3f} to {q['max']:.3f};
Q1={q['q1']:.3f}, median={q['median']:.3f}, Q3={q['q3']:.3f}.
Skewness={s['skewness']:.4f}; excess kurtosis={s['excess_kurtosis']:.4f}.
Shapiro–Wilk W={s['shapiro_w']:.4f}, p={s['shapiro_p_descriptive']:.6g}
(descriptive, not a second primary hypothesis or a selection rule).
{'Normality is not supported at the .05 diagnostic level; discreteness and visible tail departures should be considered.' if s['shapiro_p_descriptive'] < .05 else 'The normality diagnostic does not reject at .05; that does not prove normality.'}

The paired differences are bounded because quality and uniform-neighbor
predictions remain in the score range. With {n} pairs, t inference is an
approximation justified by the mean's sampling behavior if records are
independent, rather than by exact normality of individual differences.
Deleting any single pair changes the mean to a value between {loo_lo:.6f}
and {loo_hi:.6f}; this is an influence diagnostic, not a reason to delete rows.
The bootstrap interval [{blo:.4f}, {bhi:.4f}] shows the descriptive sensitivity
of the mean interval to the normal-based calculation, while retaining the
same fixed training set and independence limitation.

![Paired difference distribution](../figures/{color}/paired_differences.png)

Positive values favor scaling; the dashed line is no difference. The .2-point
lattice follows from integer targets and uniform k=5 predictions.

![Normal Q–Q diagnostic](../figures/{color}/difference_qq.png)

Step patterns are expected for discrete differences; tail departures qualify
the normal-difference assumption. This plot does not test record independence.

![Absolute prediction errors](../figures/{color}/absolute_error_boxplots.png)

Boxes show medians and interquartile ranges, whiskers extend to 1.5 IQR and
points show observations beyond the whiskers (overlapping points may coincide).
They show observation spread, not confidence intervals. The arms are paired.

![Mean improvement and interval](../figures/{color}/mean_improvement_ci.png)

The dot and bar show mean d and a **two-sided 95% Student t interval** in score
points, conditional on the fitted training set. Positive d favors scaling.
""")
    sections.insert(1, "\n## Results at a glance\n\n" + "\n".join(summaries) + "\n")
    sections.append("""
## Limitations

- **Unobserved dependence:** no wine/batch/producer identifiers are available.
  Deduplication removes exact overlap but cannot rule out related or near-identical
  wines. Positive within-batch dependence could make SEs too small and nominal
  power too optimistic. We cannot repair unknown clustering with these files.
- **One fitted training set:** the interval and test reflect paired test-wine
  variation, not variation across new training sets. Pooling repeated holdout or
  cross-validation predictions as independent wines would be invalid.
- **Scope and selection:** only one fixed k, distance rule, split and pair of
  preprocessing choices are evaluated. Neither cohort is a random sample of all
  wines; unique profiles and red Vinho Verde define a narrower target. The white
  cohort comes from the same source and differs in chemistry and score mix.
- **Outcome measurement:** expert scores are ordinal, imbalanced and noisy.
  MAE assumes adjacent score steps are comparable. Treating quality as numeric
  is an explicit modeling convention; an improved MAE is not a causal change in
  wine quality or a sensory-treatment effect.
- **Discreteness and tails:** exact normality of differences is unlikely. Large
  sample t inference and the row bootstrap are approximate and both assume
  independent test records. No robustness diagnostic resolves unobserved batches.
- **Practical meaning and multiplicity:** no practical-equivalence margin was
  specified. Red is the only confirmatory .05 test; white is secondary, RMSE is
  descriptive, and diagnostics are not additional evidence for the primary claim.

## Conclusions and reproducibility

The results above answer the prespecified directional question for these fixed
model comparisons. Interpret the measured mean reduction and interval together
with the rejection decision. Do not infer universal benefit, absence of benefit
from nonrejection, or guaranteed performance for new training samples.
Future work could use genuinely identified independent wines/batches and a
separately planned training-sample replication to address the main limitations.

See [README](../README.md) for commands and [analysis plan](analysis_plan.md)
for choices recorded before evaluation. Each `results/<color>/` folder includes
predictions with source IDs and both absolute errors, split memberships,
duplicate mapping, metrics, inference, data audit, scaler means/scales, power
plan and evaluation manifest. The manifest records file hashes, Python/package
versions and event times; byte-level reproducibility of image metadata across
operating systems is not assumed. Numeric results are reproducible with the
locked software environment; repeated runs do not add new independent evidence.

## Sources and technical references

- Cortez et al. (2009), [Wine Quality dataset, UCI](https://archive.ics.uci.edu/dataset/186/wine+quality),
  [dataset DOI](https://doi.org/10.24432/C56S3T). The local `winequality.names`
  documents predictor definitions, expert scoring and data provenance.
- Cortez et al. (2009), *Modeling wine preferences by data mining from
  physicochemical properties*, Decision Support Systems 47(4), 547–553.
  [Article DOI](https://doi.org/10.1016/j.dss.2009.05.016).
- [statsmodels TTestPower.solve_power](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.solve_power.html)
  supports one-sample/paired standardized-effect planning with alternative='larger'.
- [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
  documents the directional paired test, degrees of freedom and confidence intervals.
- [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
  explains the symmetry null and numerical ties; the default analysis omits it.
""")
    path = root / "reports" / "statistical_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8", newline="\n")
    return path
