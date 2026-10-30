"""Two stages ensure power planning is persisted before predictions are made."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from .config import CONFIG, FEATURES, TARGET
from .data import clean_data
from .io import read_json, sha256, write_json
from .models import make_models
from .plots import make_figures
from .power import plan_power
from .statistics import analyze_errors


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def project_identity(project: Path, color: str):
    return {
        "config": CONFIG, "features": FEATURES,
        "raw_sha256": sha256(project / "data" / "raw" / f"winequality-{color}.csv"),
        "analysis_plan_sha256": sha256(project / "reports" / "analysis_plan.md"),
    }


def prepare(project: Path, output: Path, color: str):
    directory = output / "results" / color
    directory.mkdir(parents=True, exist_ok=True)
    identity = project_identity(project, color)
    plan_file = directory / "prepared_design.json"
    if plan_file.exists():
        old = read_json(plan_file)
        if old["identity"] != identity:
            raise ValueError("Existing plan differs from inputs/design. Use a new output directory and document amendments.")
        return old
    raw = pd.read_csv(project / "data" / "raw" / f"winequality-{color}.csv", sep=";")
    clean, mapping, audit = clean_data(raw)
    clean_dir = output / "data" / "cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_file = clean_dir / f"winequality-{color}.csv"
    clean.to_csv(clean_file, index=False, lineterminator="\n")
    mapping.to_csv(directory / "duplicate_map.csv", index=False, lineterminator="\n")
    train, test = train_test_split(clean.source_record_id.to_numpy(), test_size=CONFIG["test_size"],
                                   random_state=CONFIG["seed"], shuffle=True)
    if len(train) < CONFIG["k"] or len(test) < 3:
        raise ValueError("Too few records for this prespecified design.")
    splits = pd.DataFrame({"source_record_id": np.concatenate([train, test]),
                           "split": ["train"] * len(train) + ["test"] * len(test),
                           "order_within_split": list(range(len(train))) + list(range(len(test)))})
    splits.to_csv(directory / "split_membership.csv", index=False, lineterminator="\n")
    write_json(directory / "data_audit.json", audit)
    write_json(directory / "power_plan.json", plan_power(len(test)))
    plan = {"identity": identity, "dataset": color, "prepared_at_utc": utc_now(),
            "n_train": len(train), "n_test": len(test),
            "clean_sha256": sha256(clean_file),
            "split_sha256": sha256(directory / "split_membership.csv"),
            "power_sha256": sha256(directory / "power_plan.json")}
    write_json(plan_file, plan)
    return plan


def evaluate(project: Path, output: Path, color: str):
    directory = output / "results" / color
    plan_file = directory / "prepared_design.json"
    if not plan_file.exists():
        raise ValueError("Run --stage prepare before evaluation.")
    plan = read_json(plan_file)
    if plan["identity"] != project_identity(project, color):
        raise ValueError("Input/design changed after preparation.")
    clean_file = output / "data" / "cleaned" / f"winequality-{color}.csv"
    for path, expected in [(clean_file, plan["clean_sha256"]),
                           (directory / "split_membership.csv", plan["split_sha256"]),
                           (directory / "power_plan.json", plan["power_sha256"])]:
        if sha256(path) != expected:
            raise ValueError(f"Prepared artifact changed: {path}")
    started = utc_now()
    data = pd.read_csv(clean_file).set_index("source_record_id")
    splits = pd.read_csv(directory / "split_membership.csv")
    train_ids = splits.loc[splits.split == "train"].sort_values("order_within_split").source_record_id
    test_ids = splits.loc[splits.split == "test"].sort_values("order_within_split").source_record_id
    train, test = data.loc[train_ids], data.loc[test_ids]
    if set(train.index) & set(test.index):
        raise ValueError("Source records overlap across train/test.")
    if set(map(tuple, train[FEATURES].to_numpy())) & set(map(tuple, test[FEATURES].to_numpy())):
        raise ValueError("Predictor profiles overlap across train/test.")
    control, treatment = make_models()
    with threadpool_limits(limits=1):
        control.fit(train[FEATURES], train[TARGET])
        treatment.fit(train[FEATURES], train[TARGET])
        pred_a = control.predict(test[FEATURES])
        pred_b = treatment.predict(test[FEATURES])
    y = test[TARGET].to_numpy()
    error_a, error_b = np.abs(y - pred_a), np.abs(y - pred_b)
    predictions = test.reset_index().copy()
    predictions["prediction_unscaled"] = pred_a
    predictions["prediction_scaled"] = pred_b
    predictions["absolute_error_unscaled"] = error_a
    predictions["absolute_error_scaled"] = error_b
    predictions["difference_unscaled_minus_scaled"] = error_a - error_b
    predictions.to_csv(directory / "predictions.csv", index=False, lineterminator="\n")
    metrics = pd.DataFrame([
        {"model": "unscaled", "n_test": len(y), "mae": error_a.mean(), "rmse": np.sqrt(np.mean((y-pred_a)**2))},
        {"model": "scaled", "n_test": len(y), "mae": error_b.mean(), "rmse": np.sqrt(np.mean((y-pred_b)**2))},
    ])
    metrics.to_csv(directory / "metrics.csv", index=False, lineterminator="\n")
    inference = analyze_errors(error_a, error_b)
    write_json(directory / "inference.json", inference)
    scaler = treatment.named_steps["scaler"]
    write_json(directory / "scaler_training_statistics.json", {
        "features": FEATURES, "mean": scaler.mean_.tolist(), "variance_ddof0": scaler.var_.tolist(),
        "scale": scaler.scale_.tolist(), "n_training_records_seen": int(scaler.n_samples_seen_),
    })
    make_figures(predictions, metrics, inference, color, output / "figures" / color)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True)
    write_json(directory / "evaluation_manifest.json", {
        "dataset": color, "evaluation_started_at_utc": started, "evaluation_completed_at_utc": utc_now(),
        "prepared_at_utc": plan["prepared_at_utc"], "n_train": len(train), "n_test": len(test),
        "design": plan["identity"], "prepared_design_sha256": sha256(plan_file),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {m: version(m) for m in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "threadpoolctl"]},
        "git_head_at_run": git.stdout.strip() if git.returncode == 0 else None,
        "requirements_sha256": sha256(project / "requirements.txt"),
        "analysis_source_sha256": {str(p.relative_to(project)).replace("\\", "/"): sha256(p) for p in sorted(list((project / "src").rglob("*.py")) + list((project / "scripts").rglob("*.py")))},
        "artifact_sha256": {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "evaluation_manifest.json"},
        "interpretation_scope": "Conditional on this fitted training set; independence of distinct wine records assumed, not proven.",
    })
    return inference
