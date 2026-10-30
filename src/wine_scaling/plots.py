"""Publication figures from saved observation-level results (PNG + SVG)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BLUE, GOLD, INK = "#225A81", "#B07A19", "#252B33"


def make_figures(predictions, metrics, inference, color: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    d = predictions["difference_unscaled_minus_scaled"].to_numpy()
    n = len(d)
    title = f"{color.capitalize()} wine · {n} unique test records"

    def mae(ax):
        values = metrics["mae"].to_numpy()
        bars = ax.bar(["Unscaled", "StandardScaler"], values, width=.58,
                      color=[BLUE, GOLD], edgecolor=INK, linewidth=.7)
        bars[1].set_hatch("//")
        ax.bar_label(bars, fmt="%.3f", padding=5)
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_ylabel("Mean absolute error (quality-score points)")
        ax.set_title("Mean absolute prediction error", loc="left")

    def histogram(ax):
        # d is on a .2 lattice, since k=5 and all quality scores are integers.
        bins = np.arange(np.floor(d.min() / .2) * .2 - .1,
                         np.ceil(d.max() / .2) * .2 + .3, .2)
        ax.hist(d, bins=bins, color=BLUE, edgecolor="white", linewidth=.7)
        ax.axvline(0, color=INK, linewidth=1.2, linestyle="--", label="No difference")
        ax.axvline(inference["mean_difference"], color=GOLD, linewidth=2, label="Mean")
        ax.set_xlabel("Absolute error: unscaled − scaled (points)")
        ax.set_ylabel("Test records")
        ax.set_title("Paired error differences", loc="left")
        ax.legend(frameon=False, fontsize=9)

    def boxes(ax):
        bp = ax.boxplot([predictions.absolute_error_unscaled, predictions.absolute_error_scaled],
                        tick_labels=["Unscaled", "StandardScaler"], patch_artist=True,
                        widths=.5, medianprops={"color": INK, "linewidth": 1.8},
                        flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": INK, "markersize": 4})
        for patch, fill in zip(bp["boxes"], [BLUE, GOLD], strict=True):
            patch.set_facecolor(fill)
            patch.set_alpha(.65)
        bp["boxes"][1].set_hatch("//")
        ax.set_ylabel("Absolute prediction error (points)")
        ax.set_ylim(bottom=-.05)
        ax.set_title("Absolute-error distributions", loc="left")

    def interval(ax):
        mean = inference["mean_difference"]
        lo, hi = inference["ci_95_two_sided"]
        ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
        ax.errorbar(mean, 0, xerr=[[mean-lo], [hi-mean]], fmt="o", color=BLUE,
                    markersize=8, linewidth=2, capsize=6)
        spread = max(hi, 0) - min(lo, 0)
        ax.set_xlim(min(lo, 0) - spread * .2, max(hi, 0) + spread * .2)
        ax.set_ylim(-.8, .8)
        ax.set_yticks([])
        ax.set_xlabel("Mean absolute-error reduction (quality-score points)")
        ax.text(.5, .16, f"Mean {mean:.3f}; 95% CI [{lo:.3f}, {hi:.3f}]",
                transform=ax.transAxes, ha="center", fontsize=10)
        ax.set_title("Mean improvement with two-sided 95% t interval", loc="left")

    def save(fig, name):
        for extension in ["png", "svg"]:
            fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    for name, draw in [("mae_comparison", mae), ("paired_differences", histogram),
                       ("absolute_error_boxplots", boxes), ("mean_improvement_ci", interval)]:
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        draw(ax)
        fig.suptitle(title, fontsize=10, color="#50565E")
        save(fig, name)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for label, draw, ax in zip("ABCD", [mae, histogram, boxes, interval], axes.flat, strict=True):
        draw(ax)
        ax.text(-.10, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=14)
    fig.suptitle(f"Feature scaling and KNN regression | {title}", fontsize=15)
    save(fig, "experiment_summary")

    fig, ax = plt.subplots(figsize=(6.2, 5), layout="constrained")
    (theoretical, observed), (slope, intercept, _) = stats.probplot(d, dist="norm")
    ax.scatter(theoretical, observed, color=BLUE, s=16, alpha=.65)
    ax.plot(theoretical, intercept + slope * theoretical, color=INK, linestyle="--", linewidth=1.3)
    ax.set(xlabel="Theoretical normal quantile", ylabel="Observed paired difference (points)",
           title="Normal Q–Q plot of paired differences")
    fig.suptitle(title, fontsize=10, color="#50565E")
    save(fig, "difference_qq")
