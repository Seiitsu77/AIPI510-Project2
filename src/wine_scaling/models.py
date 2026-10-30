"""The treatment adds only a scaler fitted inside the training pipeline."""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import CONFIG


def make_models():
    params = {k: CONFIG[k] for k in ["weights", "metric", "algorithm", "n_jobs"]}
    params["n_neighbors"] = CONFIG["k"]
    control = KNeighborsRegressor(**params)
    treatment = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(**params))])
    return control, treatment
