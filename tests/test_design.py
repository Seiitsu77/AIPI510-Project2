"""Tests focus on leakage, inferential direction and auditable output integrity."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from wine_scaling.config import CONFIG, FEATURES
from wine_scaling.data import clean_data
from wine_scaling.experiment import evaluate, prepare
from wine_scaling.io import read_json, sha256
from wine_scaling.models import make_models
from wine_scaling.power import plan_power
from wine_scaling.statistics import analyze_errors

PROJECT = Path(__file__).resolve().parents[1]


def test_cleaning_preserves_lineage_and_rejects_conflicting_labels():
    raw = pd.read_csv(PROJECT / "data/raw/winequality-red.csv", sep=";").iloc[:2].copy()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    clean, mapping, audit = clean_data(raw)
    assert len(clean) == 2 and audit["removed_exact_duplicates"] == 1
    assert mapping.retained_source_record_id.to_list() == [1, 2, 1]
    conflicting = raw.copy()
    conflicting.loc[2, "quality"] += 1
    with pytest.raises(ValueError, match="Conflicting"):
        clean_data(conflicting)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_missing_and_nonfinite_fail_explicitly(invalid):
    raw = pd.read_csv(PROJECT / "data/raw/winequality-red.csv", sep=";").iloc[:10].copy()
    raw.loc[0, "density"] = invalid
    with pytest.raises(ValueError, match="Missing or nonfinite"):
        clean_data(raw)


def test_power_rounding_and_direction():
    p = plan_power(272)
    assert p["required_pairs"] == 156
    assert p["required_pairs"] - 1 < p["required_pairs_continuous"] <= p["required_pairs"]
    assert .94 < p["design_power_at_assumed_dz"] < .96


def test_direction_confidence_interval_and_negative_alternative():
    a = np.array([1., .8, .3, .4, 1.2, .6])
    b = np.array([.2, .5, .4, .2, .8, .6])
    s, reverse = analyze_errors(a, b), analyze_errors(b, a)
    d = a - b
    expected_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert s["t_statistic"] == pytest.approx(expected_t)
    assert s["p_value_one_sided"] == pytest.approx(stats.t.sf(expected_t, len(d)-1))
    assert s["p_value_one_sided"] + reverse["p_value_one_sided"] == pytest.approx(1)
    expected_ci = stats.t.interval(.95, len(d)-1, loc=d.mean(), scale=stats.sem(d))
    np.testing.assert_allclose(s["ci_95_two_sided"], expected_ci)
    assert s["cohens_dz"] == pytest.approx(expected_t / np.sqrt(len(d)))


def test_training_only_scaler_and_identical_knn_parameters():
    x = pd.DataFrame({"a": [0., 1., 2., 3., 4., 5.], "b": [3., 5., 7., 9., 11., 13.]})
    y = np.arange(6)
    a, b = make_models()
    assert a.get_params() == b.named_steps["knn"].get_params()
    b.fit(x, y)
    scaler = b.named_steps["scaler"]
    expected = x.mean().to_numpy()
    b.predict(pd.DataFrame({"a": [100000.], "b": [-10000.]}))
    np.testing.assert_allclose(scaler.mean_, expected)
    assert scaler.n_samples_seen_ == len(x)


def test_modified_prepared_split_is_rejected(tmp_path):
    prepare(PROJECT, tmp_path, "red")
    path = tmp_path / "results/red/split_membership.csv"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared artifact changed"):
        evaluate(PROJECT, tmp_path, "red")


@pytest.mark.parametrize("color,raw_n,clean_n", [("red",1599,1359), ("white",4898,3961)])
def test_saved_results_reconstruct_from_raw_and_frozen_split(color, raw_n, clean_n):
    directory = PROJECT / "results" / color
    raw = pd.read_csv(PROJECT / f"data/raw/winequality-{color}.csv", sep=";")
    clean, _, _ = clean_data(raw)
    assert len(raw) == raw_n and len(clean) == clean_n
    clean = clean.set_index("source_record_id")
    split = pd.read_csv(directory / "split_membership.csv")
    train_ids = split.loc[split.split == "train"].source_record_id.to_numpy()
    test_ids = split.loc[split.split == "test"].source_record_id.to_numpy()
    expected_train, expected_test = train_test_split(clean.index.to_numpy(), test_size=.2, random_state=42)
    np.testing.assert_array_equal(train_ids, expected_train)
    np.testing.assert_array_equal(test_ids, expected_test)
    assert set(train_ids).isdisjoint(test_ids)
    train, test = clean.loc[train_ids], clean.loc[test_ids]
    assert set(map(tuple, train[FEATURES].to_numpy())).isdisjoint(map(tuple, test[FEATURES].to_numpy()))
    saved = pd.read_csv(directory / "predictions.csv")
    np.testing.assert_array_equal(saved.source_record_id, test_ids)
    np.testing.assert_array_equal(saved.quality, test.quality)
    a, b = make_models()
    with threadpool_limits(limits=1):
        a.fit(train[FEATURES], train.quality)
        b.fit(train[FEATURES], train.quality)
        np.testing.assert_allclose(a.predict(test[FEATURES]), saved.prediction_unscaled, atol=1e-12)
        np.testing.assert_allclose(b.predict(test[FEATURES]), saved.prediction_scaled, atol=1e-12)
    scaler = read_json(directory / "scaler_training_statistics.json")
    np.testing.assert_allclose(scaler["mean"], train[FEATURES].mean())
    np.testing.assert_allclose(scaler["variance_ddof0"], train[FEATURES].var(ddof=0))
    error_a = np.abs(saved.quality - saved.prediction_unscaled)
    error_b = np.abs(saved.quality - saved.prediction_scaled)
    np.testing.assert_allclose(saved.absolute_error_unscaled, error_a, atol=1e-12)
    np.testing.assert_allclose(saved.absolute_error_scaled, error_b, atol=1e-12)
    d = error_a - error_b
    np.testing.assert_allclose(saved.difference_unscaled_minus_scaled, d, atol=1e-12)
    s = read_json(directory / "inference.json")
    m = pd.read_csv(directory / "metrics.csv")
    np.testing.assert_allclose(m.mae, [error_a.mean(), error_b.mean()])
    np.testing.assert_allclose(m.rmse, [np.sqrt(np.mean(error_a**2)), np.sqrt(np.mean(error_b**2))])
    assert m.mae.iloc[0] - m.mae.iloc[1] == pytest.approx(d.mean())
    assert s["mean_difference"] == pytest.approx(d.mean())
    assert s["t_statistic"] == pytest.approx(stats.ttest_1samp(d, 0, alternative="greater").statistic)
    assert s["p_value_one_sided"] == pytest.approx(stats.t.sf(s["t_statistic"], len(d)-1))
    assert s["reject_null"] == (s["lower_bound_95_one_sided"] > 0)
    manifest = read_json(directory / "evaluation_manifest.json")
    assert manifest["prepared_at_utc"] < manifest["evaluation_started_at_utc"]
    assert manifest["prepared_design_sha256"] == sha256(directory / "prepared_design.json")
    assert manifest["requirements_sha256"] == sha256(PROJECT / "requirements.txt")
    for name, expected in manifest["artifact_sha256"].items():
        assert sha256(directory / name) == expected, name
    for name, expected in manifest["analysis_source_sha256"].items():
        assert sha256(PROJECT / name) == expected, name
