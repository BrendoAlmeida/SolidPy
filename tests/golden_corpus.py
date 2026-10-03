# -*- coding: utf-8 -*-
"""Frozen golden corpus used by the backend parity tests.

``tests/golden/corpus_v1.json`` holds the designs (plain data) and ``tests/golden/reference_v1.json`` the
results of the scalar reference (``BurnSimulation``) for each design. Both are written by
``tools/make_golden_corpus.py`` and committed. The reference file records the SHA-256 of the three
reference source files, so a test fails loudly when the scalar physics changes and the data is stale.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from solidpy import BurnSimulation, Environment, Grain, Motor, Propellant

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
CORPUS_PATH = GOLDEN_DIR / "corpus_v1.json"
REFERENCE_PATH = GOLDEN_DIR / "reference_v1.json"

#: The files whose bytes enter ``physics_provider_hash``. The batched backends must not modify them.
REFERENCE_SOURCES = ("Burn.py", "Grain.py", "Propellant.py")

#: Points of the resampled reference curves stored next to the metrics.
CURVE_POINTS = 41
CURVE_CHANNELS = {
    "chamber_pressure_pa": "chamber_pressure_pa",
    "thrust_n": "thrust_n",
    "mdot_nozzle_kg_s": "mdot_nozzle_kg_s",
    "mdot_generated_kg_s": "mdot_generated_kg_s",
}


def igniter_decay(time):
    """Named igniter flow callable (kg/s); callables cannot live in JSON, so cases refer to them by name."""
    return 0.004 * math.exp(-time / 0.4)


def activation_front(time, regressed_length):
    """Named activation callable using the (time, mean regression) call form."""
    return min(1.0, 0.25 + 0.75 * time / 0.2)


CALLABLES = {"igniter_decay": igniter_decay, "activation_front": activation_front}


def reference_sources_sha256():
    digest = hashlib.sha256()
    for name in REFERENCE_SOURCES:
        digest.update((REPO_ROOT / "solidpy" / name).read_bytes())
    return digest.hexdigest()


def load_corpus():
    return json.loads(CORPUS_PATH.read_text())


def load_reference():
    """Return ``{"manifest": {...}, "records": {case_id: record}}``."""
    data = json.loads(REFERENCE_PATH.read_text())
    records = {record["id"]: record for record in data.pop("records")}
    return {"manifest": data, "records": records}


def _source(value):
    """Turn a JSON source description into what ``BurnSimulation`` accepts."""
    if isinstance(value, dict) and "callable" in value:
        return CALLABLES[value["callable"]]
    return value


def build_objects(case):
    """Return ``(grain, motor, propellant, environment, simulation_kwargs)`` for a corpus case."""
    grains = [Grain(**grain) for grain in case["grains"]]
    motor = Motor(grains, **case["motor"])

    spec = dict(case["propellant"])
    table = spec.pop("thermo_table", None)
    scalars = [spec.pop(key) for key in
               ("specific_heat_ratio", "products_molecular_mass", "combustion_temperature", "density")]
    if "interpolation_list" in spec:
        spec["interpolation_list"] = str(REPO_ROOT / spec["interpolation_list"])
    propellant = Propellant(*scalars, **spec)
    if table is not None:
        propellant.load_thermo_table(table)

    environment = Environment(**case.get("environment", {}))
    kwargs = dict(case["simulation"])
    for key in ("igniter_mass_flow", "burn_area_activation"):
        if key in kwargs:
            kwargs[key] = _source(kwargs[key])
    return grains[0], motor, propellant, environment, kwargs


def simulate(case):
    grain, motor, propellant, environment, kwargs = build_objects(case)
    return BurnSimulation(grain, motor, propellant, environment, **kwargs)


def _kind(value, table_tag, scalar_tag, callable_tag):
    if isinstance(value, dict) and "callable" in value:
        return callable_tag
    if value is None:
        return None
    if np.isscalar(value):
        return scalar_tag
    return table_tag


def derive_tags(case):
    """Feature tags computed from the design itself, so they cannot drift from the data."""
    tags = set()
    grains, motor = case["grains"], case["motor"]
    tags.add(f"grains:{motor.get('grain_number') or len(grains)}")
    geometries = {grain["geometry"] for grain in grains}
    tags.add("mixed_geometry" if len(geometries) > 1 else next(iter(geometries)))
    if any(grain["ends_burn"] for grain in grains):
        tags.add("ends_burn")
    if not all(grain["ends_burn"] for grain in grains):
        tags.add("ends_open")
    if "grain_number" in motor:
        tags.add("grain_number_replicated")

    propellant = case["propellant"]
    tags.add("burn_rate_table" if "interpolation_list" in propellant else "power_law")
    tags.add("thermo_table" if "thermo_table" in propellant else "scalar_thermo")
    if propellant.get("erosive_burning_coefficient", 0.0) > 0.0:
        tags.add("erosive")

    simulation = case["simulation"]
    for key, tag in (("eta_c", "eta_c"), ("eta_Cf", "eta_cf"), ("discharge_coefficient", "discharge")):
        if simulation.get(key, 1.0) < 1.0:
            tags.add(tag)

    igniter = _kind(simulation.get("igniter_mass_flow"), "igniter_table", "igniter_scalar", "igniter_callable")
    tags.add(igniter or "igniter_none")
    activation = _kind(
        simulation.get("burn_area_activation"), "activation_table", "activation_scalar", "activation_callable"
    )
    if activation is None:
        activation = "ramp" if simulation.get("ignition_ramp_time", 0.0) > 0.0 else "activation_none"
    tags.add(activation)
    if igniter == "igniter_callable" or activation == "activation_callable":
        tags.add("callable")

    tags.add(f"tail_off_{simulation.get('tail_off_method', 'numerical')}")
    if simulation.get("tail_off_evaluation", True) is False:
        tags.add("tail_off_omitted")
    return sorted(tags)


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def reference_record(simulation, wall_seconds):
    """The stored description of one reference run."""
    result = simulation.result
    history, metrics, status = result["history"], result["metrics"], result["status"]
    time = np.asarray(history["time_s"], dtype=float)
    end = float(time[-1])
    grid = np.linspace(0.0, end, CURVE_POINTS)
    curves = {
        name: np.interp(grid, time, np.asarray(history[key], dtype=float)).tolist()
        for name, key in CURVE_CHANNELS.items()
    }
    return {
        "status": _plain(status),
        "metrics": _plain(metrics),
        "physics_provider_hash": result["provenance"]["physics_provider_hash"],
        "history_points": int(len(time)),
        "end_time_s": end,
        "curves": curves,
        "wall_seconds": round(float(wall_seconds), 4),
    }
