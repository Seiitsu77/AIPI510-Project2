"""A priori paired t-test planning, with no predictions as inputs."""
import math
from statsmodels.stats.power import TTestPower
from .config import CONFIG


def plan_power(n_pairs: int) -> dict:
    c = CONFIG
    analysis = TTestPower()
    required = float(analysis.solve_power(
        effect_size=c["assumed_dz"], alpha=c["alpha"], power=c["target_power"],
        alternative="larger",
    ))
    return {
        "assumed_dz": c["assumed_dz"], "alpha": c["alpha"],
        "target_power": c["target_power"], "alternative": "larger",
        "required_pairs_continuous": required, "required_pairs": math.ceil(required),
        "planned_test_pairs": n_pairs, "sufficient_at_assumed_effect": n_pairs >= math.ceil(required),
        "design_power_at_assumed_dz": float(analysis.power(
            effect_size=c["assumed_dz"], nobs=n_pairs, alpha=c["alpha"], alternative="larger")),
        "assumptions": "Independent paired differences; normal-difference planning model; dz=.20 is assumed, never estimated from final outcomes.",
    }
