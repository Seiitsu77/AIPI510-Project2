"""Validate inputs and deduplicate before any split."""
import numpy as np
import pandas as pd
from .config import FEATURES, TARGET


def clean_data(raw: pd.DataFrame):
    if list(raw.columns) != FEATURES + [TARGET]:
        raise ValueError("Unexpected schema/order: expected the original 11 predictors and quality.")
    if not all(pd.api.types.is_numeric_dtype(t) for t in raw.dtypes):
        raise ValueError("All source columns must be numeric.")
    if raw.isna().any().any() or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Missing or nonfinite cells: revise the cleaning plan explicitly.")
    y = raw[TARGET]
    if not (y.between(0, 10).all() and (y == np.floor(y)).all()):
        raise ValueError("Quality must be an integer score between 0 and 10.")
    data = raw.reset_index(drop=True).copy()
    data.insert(0, "source_record_id", np.arange(1, len(data) + 1))
    subset = FEATURES + [TARGET]
    clean = data.drop_duplicates(subset=subset, keep="first").copy()
    if clean.duplicated(FEATURES).any():
        raise ValueError("Conflicting targets for identical predictors: use a revised group split plan.")
    keys = list(map(tuple, clean[subset].to_numpy()))
    representatives = dict(zip(keys, clean.source_record_id, strict=True))
    mapping = data[["source_record_id"]].copy()
    mapping["retained_source_record_id"] = [representatives[tuple(x)] for x in data[subset].to_numpy()]
    mapping["removed_duplicate"] = mapping.source_record_id != mapping.retained_source_record_id
    audit = {
        "raw_rows": len(raw), "clean_rows": len(clean),
        "removed_exact_duplicates": len(raw) - len(clean),
        "missing_cells": int(raw.isna().sum().sum()), "conflicting_predictor_profiles": 0,
        "raw_quality_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "clean_quality_counts": {str(k): int(v) for k, v in clean[TARGET].value_counts().sort_index().items()},
        "raw_feature_summary": raw[FEATURES].describe().to_dict(),
        "duplicate_policy": "Remove exact 12-column duplicates before split; retain first raw record.",
    }
    return clean, mapping, audit
