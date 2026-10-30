"""Independent validation of the saved KNN feature-scaling results.

Deliberately does NOT import wine_scaling. Every number is recomputed from the
saved predictions / cleaned data using plain pandas, numpy, scipy, statsmodels
and scikit-learn, then compared with results/<color>/inference.json and
power_plan.json. Nothing in results/red or results/white is modified.

Writes results/validation/independent_validation.json and exits with status 1
if any primary check fails. Sensitivity checks (Wilcoxon, sign test, squared
error, a second bootstrap) are EXPLORATORY: they were not in the analysis plan
and do not replace the prespecified primary result.

Usage: python scripts/validate_independent.py [--root PATH]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.power import TTestPower

PROJECT = Path(__file__).resolve().parents[1]
RTOL = 1e-9


def close(a, b, rtol=RTOL, atol=1e-12):
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def validate(root: Path, color: str) -> dict:
    res = root / "results" / color
    saved = json.loads((res / "inference.json").read_text())
    power_saved = json.loads((res / "power_plan.json").read_text())
    scaler_saved = json.loads((res / "scaler_training_statistics.json").read_text())
    cfg = json.loads((res / "prepared_design.json").read_text())["identity"]["config"]
    pred = pd.read_csv(res / "predictions.csv")
    clean = pd.read_csv(root / "data" / "cleaned" / f"winequality-{color}.csv").set_index("source_record_id")
    split = pd.read_csv(res / "split_membership.csv")
    raw = pd.read_csv(root / "data" / "raw" / f"winequality-{color}.csv", sep=";")
    features = [c for c in clean.columns if c != "quality"]

    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    # ---- 1. Primary inference, recomputed from raw predictions -------------
    y = pred["quality"].to_numpy(dtype=float)
    ea = np.abs(y - pred["prediction_unscaled"].to_numpy())
    eb = np.abs(y - pred["prediction_scaled"].to_numpy())
    d = ea - eb
    n, mean, sd = len(d), d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = mean / se
    p_one = stats.t.sf(t, n - 1)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = [mean - tcrit * se, mean + tcrit * se]
    mae_a, mae_b = ea.mean(), eb.mean()
    rmse_a = np.sqrt(np.mean((y - pred["prediction_unscaled"]) ** 2))
    rmse_b = np.sqrt(np.mean((y - pred["prediction_scaled"]) ** 2))
    metrics = pd.read_csv(res / "metrics.csv").set_index("model")

    check("n pairs", n == saved["n_pairs"], f"{n} vs {saved['n_pairs']}")
    check("saved difference column = |err unscaled| - |err scaled|",
          np.allclose(d, pred["difference_unscaled_minus_scaled"], atol=1e-12))
    check("mean difference", close(mean, saved["mean_difference"]))
    check("SD of differences", close(sd, saved["sd_difference"]))
    check("t statistic", close(t, saved["t_statistic"]))
    check("one-sided p-value", close(p_one, saved["p_value_one_sided"], rtol=1e-6))
    check("two-sided 95% CI", close(ci[0], saved["ci_95_two_sided"][0]) and close(ci[1], saved["ci_95_two_sided"][1]))
    check("Cohen's dz", close(mean / sd, saved["cohens_dz"]))
    check("MAE (both arms)", close(mae_a, metrics.loc["unscaled", "mae"]) and close(mae_b, metrics.loc["scaled", "mae"]))
    check("RMSE (both arms)", close(rmse_a, metrics.loc["unscaled", "rmse"]) and close(rmse_b, metrics.loc["scaled", "rmse"]))
    check("reject decision (one-sided alpha)", (p_one < cfg["alpha"]) == saved["reject_null"])

    # ---- 2. Power analysis ---------------------------------------------------
    pw = TTestPower()
    req = pw.solve_power(effect_size=cfg["assumed_dz"], alpha=cfg["alpha"],
                         power=cfg["target_power"], alternative="larger")
    design_power = pw.power(effect_size=cfg["assumed_dz"], nobs=n, alpha=cfg["alpha"], alternative="larger")
    check("required pairs (continuous)", close(req, power_saved["required_pairs_continuous"], rtol=1e-6))
    check("required pairs (rounded up)", int(np.ceil(req)) == power_saved["required_pairs"])
    check("design power at assumed dz", close(design_power, power_saved["design_power_at_assumed_dz"], rtol=1e-6))
    check("test set >= required pairs", n >= int(np.ceil(req)))

    # ---- 3. Independent refit of both models --------------------------------
    train_ids = split[split.split == "train"].sort_values("order_within_split").source_record_id
    test_ids = split[split.split == "test"].sort_values("order_within_split").source_record_id
    Xtr, ytr = clean.loc[train_ids, features], clean.loc[train_ids, "quality"]
    Xte = clean.loc[test_ids, features]
    knn = dict(n_neighbors=cfg["k"], weights=cfg["weights"], metric=cfg["metric"],
               algorithm=cfg["algorithm"], n_jobs=1)
    p_a = KNeighborsRegressor(**knn).fit(Xtr, ytr).predict(Xte)
    scaler = StandardScaler().fit(Xtr)  # training rows only
    p_b = KNeighborsRegressor(**knn).fit(scaler.transform(Xtr), ytr).predict(scaler.transform(Xte))
    check("predictions rows are the test split, in order",
          list(pred["source_record_id"]) == list(test_ids))
    check("refit unscaled predictions match saved",
          np.allclose(p_a, pred["prediction_unscaled"], atol=1e-12))
    check("refit scaled predictions match saved",
          np.allclose(p_b, pred["prediction_scaled"], atol=1e-12))

    # ---- 4. Leakage and data-integrity checks ---------------------------------
    check("no source-record overlap between train and test", not (set(train_ids) & set(test_ids)))
    check("no identical predictor rows across train/test",
          not (set(map(tuple, Xtr.to_numpy())) & set(map(tuple, Xte.to_numpy()))))
    check("scaler statistics come from training rows only",
          scaler_saved["n_training_records_seen"] == len(train_ids)
          and np.allclose(scaler.mean_, scaler_saved["mean"])
          and np.allclose(scaler.var_, scaler_saved["variance_ddof0"]))
    check("split covers every cleaned record exactly once",
          len(train_ids) + len(test_ids) == len(clean) and set(train_ids) | set(test_ids) == set(clean.index))
    check("duplicates removed = exact 12-column duplicates in raw file",
          len(raw) - len(clean) == int(raw.duplicated().sum()),
          f"raw {len(raw)}, cleaned {len(clean)}, exact duplicates {int(raw.duplicated().sum())}")
    check("no conflicting labels for identical predictors", not clean.duplicated(features).any())

    # ---- 5. EXPLORATORY sensitivity checks (not prespecified) -----------------
    rounded = np.round(d, 10)
    pos, neg, zero = int((rounded > 0).sum()), int((rounded < 0).sum()), int((rounded == 0).sum())
    wil = stats.wilcoxon(d, alternative="greater")
    sign = stats.binomtest(pos, pos + neg, 0.5, alternative="greater")
    sq = (y - pred["prediction_unscaled"].to_numpy()) ** 2 - (y - pred["prediction_scaled"].to_numpy()) ** 2
    sq_test = stats.ttest_1samp(sq, 0, alternative="greater")
    rng = np.random.default_rng(20261030)
    boot = rng.choice(d, size=(10000, n), replace=True).mean(axis=1)
    boot_ci = np.quantile(boot, [0.025, 0.975])
    check("second bootstrap CI agrees with saved bootstrap CI (+/-0.005)",
          np.allclose(boot_ci, saved["bootstrap_ci_95_percentile"], atol=0.005),
          f"mine [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}] vs saved {saved['bootstrap_ci_95_percentile']}")

    var = Xtr.var(ddof=0)
    share = (var / var.sum()).sort_values(ascending=False)
    desc = clean[features].describe().loc[["min", "max", "std"]].T

    return {
        "dataset": color,
        "all_primary_checks_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "recomputed": {
            "n_pairs": n, "mae_unscaled": mae_a, "mae_scaled": mae_b,
            "mae_reduction_pct_of_control": 100 * mean / mae_a,
            "rmse_unscaled": rmse_a, "rmse_scaled": rmse_b,
            "mean_difference": mean, "sd_difference": sd, "t": t, "df": n - 1,
            "p_one_sided": p_one, "ci_95_two_sided": ci, "cohens_dz": mean / sd,
            "required_pairs": req, "design_power_at_assumed_dz": design_power,
            "improved": pos, "worsened": neg, "tied": zero,
        },
        "exploratory_sensitivity_not_prespecified": {
            "wilcoxon_one_sided_p": float(wil.pvalue),
            "sign_test_one_sided_p": float(sign.pvalue),
            "paired_t_on_squared_error_one_sided_p": float(sq_test.pvalue),
            "second_bootstrap_ci_95": boot_ci.tolist(),
        },
        "unscaled_distance_variance_share_top5": {k: float(v) for k, v in share.head(5).items()},
        "predictor_ranges": {k: {"min": float(r["min"]), "max": float(r["max"]), "std": float(r["std"])}
                             for k, r in desc.iterrows()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=PROJECT, help="Repository root (default: this repo)")
    args = ap.parse_args()
    root = args.root.resolve()
    out = {c: validate(root, c) for c in ["red", "white"]}
    folder = root / "results" / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "independent_validation.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")

    ok = True
    for color, r in out.items():
        rc = r["recomputed"]
        print(f"\n=== {color}: n={rc['n_pairs']} pairs | MAE {rc['mae_unscaled']:.4f} -> {rc['mae_scaled']:.4f} "
              f"({rc['mae_reduction_pct_of_control']:.2f}% lower) | t({rc['df']})={rc['t']:.4f} "
              f"p(1-sided)={rc['p_one_sided']:.3g} dz={rc['cohens_dz']:.3f}")
        for c in r["checks"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}" + (f"  ({c['detail']})" if c["detail"] and not c["passed"] else ""))
        s = r["exploratory_sensitivity_not_prespecified"]
        print(f"  exploratory: Wilcoxon p={s['wilcoxon_one_sided_p']:.3g}, sign test p={s['sign_test_one_sided_p']:.3g}, "
              f"squared-error t-test p={s['paired_t_on_squared_error_one_sided_p']:.3g}")
        ok &= r["all_primary_checks_passed"]
    print("\nALL PRIMARY CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
