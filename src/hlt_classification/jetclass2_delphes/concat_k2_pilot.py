"""Explicit small-pilot recipe; never mutate the registered full campaign."""
from copy import deepcopy
import math

PROFILE = "pilot_100k_50k_60"
AUTHORIZE = "AUTHORIZE JETCLASS2 DZFIX CONCAT K2 PILOT 100K EXACT SPEC"
COUNTS = dict(train=100_000, validation=50_000, final_test=1_000_000)
DOMAIN = "JC2/CONCAT_K2/PILOT100K/v1"


def is_pilot(spec):
    return spec.get("registration", spec).get("experiment_profile") == PROFILE


def training_recipe():
    from .salience_learned_graph import TRAINING
    return {**deepcopy(TRAINING), "batch_size": 128, "maximum_passes": 60,
            "minimum_passes": 30, "patience_clock_start_pass": 30,
            "hold_through_pass": 20, "decay_through_pass": 30}


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= 60:
        raise ValueError("Pilot learning-rate position differs")
    if position <= 3:
        return 3e-4 * position / 3
    if position <= 20:
        return 3e-4
    if position <= 30:
        return 1.5e-5 + (3e-4 - 1.5e-5) * .5 * (1 + math.cos(math.pi * (position-20)/10))
    return 1.5e-5


def kernel_options(spec):
    return {"training_recipe": training_recipe()} if is_pilot(spec) else {}
