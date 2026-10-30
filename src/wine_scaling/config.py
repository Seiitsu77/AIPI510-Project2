"""Fixed design choices. Changes after evaluation are exploratory."""

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
TARGET = "quality"
CONFIG = {
    "seed": 42, "test_size": 0.20, "k": 5, "weights": "uniform",
    "metric": "euclidean", "algorithm": "brute", "n_jobs": 1,
    "alpha": 0.05, "target_power": 0.80, "assumed_dz": 0.20,
    "bootstrap_seed": 4242, "bootstrap_resamples": 10000,
}
