"""Paired inference; positive differences favor scaling."""
import numpy as np
from scipy import stats
from .config import CONFIG


def analyze_errors(unscaled, scaled) -> dict:
    a, b = np.asarray(unscaled, dtype=float), np.asarray(scaled, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 3:
        raise ValueError("Need matching one-dimensional arrays with at least 3 pairs.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Paired errors must be finite.")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("Absolute errors cannot be negative.")
    d = a - b
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        raise ValueError("Zero difference variance: t inference and dz are undefined.")
    se = sd / np.sqrt(n)
    test = stats.ttest_rel(a, b, alternative="greater")
    critical = stats.t.ppf(.975, n - 1)
    # k=5 and integer targets yield errors/differences on a .2 lattice.
    # Round only for diagnostic ties; primary inference uses full precision.
    rounded = np.round(d, 10)
    shapiro = stats.shapiro(d)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    # Batching bounds memory use for larger replication datasets.
    boot = np.concatenate([
        rng.choice(d, size=(min(500, CONFIG["bootstrap_resamples"] - start), n), replace=True).mean(axis=1)
        for start in range(0, CONFIG["bootstrap_resamples"], 500)
    ])
    loo = (d.sum() - d) / (n - 1)
    return {
        "n_pairs": n, "mean_difference": mean, "sd_difference": sd, "se_difference": float(se),
        "t_statistic": float(test.statistic), "degrees_of_freedom": n - 1,
        "p_value_one_sided": float(test.pvalue), "alpha": CONFIG["alpha"],
        "reject_null": bool(test.pvalue < CONFIG["alpha"]), "cohens_dz": mean / sd,
        "ci_95_two_sided": [float(mean - critical * se), float(mean + critical * se)],
        "lower_bound_95_one_sided": float(mean - stats.t.ppf(.95, n - 1) * se),
        "bootstrap_ci_95_percentile": np.quantile(boot, [.025, .975]).tolist(),
        "bootstrap_resamples": CONFIG["bootstrap_resamples"], "bootstrap_seed": CONFIG["bootstrap_seed"],
        "skewness": float(stats.skew(d, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(d, bias=False)),
        "shapiro_w": float(shapiro.statistic), "shapiro_p_descriptive": float(shapiro.pvalue),
        "quantiles": dict(zip(["min", "q1", "median", "q3", "max"], np.quantile(d, [0,.25,.5,.75,1]).tolist(), strict=True)),
        "positive_pairs": int((rounded > 0).sum()), "zero_pairs": int((rounded == 0).sum()),
        "negative_pairs": int((rounded < 0).sum()), "unique_difference_values": len(np.unique(rounded)),
        "leave_one_out_mean_range": [float(loo.min()), float(loo.max())],
        "wilcoxon": {"performed": False, "reason": "Not prespecified as an inferential check: symmetry/location-shift assumptions unestablished; discrete differences have ties/zeros. Paired bootstrap instead targets the mean."},
    }
