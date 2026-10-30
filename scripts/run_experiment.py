"""Run from any working directory: python scripts/run_experiment.py --dataset both."""
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".tmp" / "matplotlib"))
sys.path.insert(0, str(PROJECT / "src"))

from wine_scaling.experiment import evaluate, prepare
from wine_scaling.reporting import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["red", "white", "both"], default="red")
    parser.add_argument("--stage", choices=["prepare", "evaluate", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=PROJECT,
                        help="Root for generated data/results/figures/reports; raw inputs stay in the project.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    colors = ["red", "white"] if args.dataset == "both" else [args.dataset]
    # Persist planning for every requested cohort before evaluating either one.
    if args.stage in ["prepare", "all"]:
        for color in colors:
            p = prepare(PROJECT, output, color)
            print(f"Prepared {color}: {p['n_train']} train / {p['n_test']} test; power plan saved.")
    if args.stage in ["evaluate", "all"]:
        # Copy the frozen plan for portable reports in separate artifact roots.
        if output != PROJECT:
            import shutil
            (output / "reports").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / "reports" / "analysis_plan.md", output / "reports" / "analysis_plan.md")
            if (PROJECT / "README.md").exists():
                shutil.copyfile(PROJECT / "README.md", output / "README.md")
        for color in colors:
            s = evaluate(PROJECT, output, color)
            print(f"Evaluated {color}: mean d={s['mean_difference']:.6f}; t({s['degrees_of_freedom']})={s['t_statistic']:.6f}; p(one-sided)={s['p_value_one_sided']:.8g}")
        print(f"Report: {build_report(output)}")


if __name__ == "__main__":
    main()
