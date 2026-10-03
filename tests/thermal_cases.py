"""Synthetic curves and designs for the thermal ablation tests, and the frozen results of the scalar model.

    python tests/thermal_cases.py        # rewrite tests/golden/thermal_ablation_v1.json from the scalar code

The golden file pins ``simulate_thermal_ablation`` bit for bit so that code around it can be restructured.
Regenerate it only when the thermal physics changes on purpose.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from solidpy import CasingMaterial, NozzleMaterial, geometry_from_components  # noqa: E402
from test_robustness import make_motor_stack  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "thermal_ablation_v1.json"

ALUMINIUM = dict(density_kg_m3=2700.0, thermal_conductivity_w_mk=167.0, heat_capacity_j_kgk=900.0)


def make_geometry(wall_m=0.004):
    grain, motor, propellant, _ = make_motor_stack()
    return geometry_from_components(grain, motor, propellant, casing_wall_thickness_m=wall_m, dry_mass_kg=3.0)


def make_curve(points=120, duration=1.5, pressure_pa=4.5e6, mass_flow=0.6, shape=1.0, pressure_series=True,
               ablation_series=None, throat_growth_mm=0.0, throat_diameter_m=0.035, seed=0):
    """A bell-shaped burn: thrust, nozzle mass flow and chamber pressure over ``points`` samples."""
    time = np.linspace(0.0, duration, points)
    rng = np.random.default_rng(seed)
    bell = np.sin(np.pi * np.clip(time / duration, 0.0, 1.0)) ** shape * (1.0 + 0.03 * rng.standard_normal(points))
    bell = np.maximum(bell, 0.02)
    curve = {
        "time_s": time,
        "thrust_n": 1500.0 * bell,
        "mass_flow_kg_s": mass_flow * bell,
        "mass_nozzle_kg_s": mass_flow * bell,
        "throat_diameter_m": throat_diameter_m + 1e-3 * throat_growth_mm * time / duration,
    }
    if pressure_series:
        curve["chamber_pressure_pa"] = pressure_pa * bell
    if ablation_series is not None:
        curve["throat_ablation_m"] = ablation_series
    return curve


def _ablation_with_gaps(points):
    series = np.linspace(0.0, 4e-4, points)
    series[::7] = np.nan
    return series


def _tiny_steps():
    curve = make_curve(points=80, duration=0.8)
    curve["time_s"] = curve["time_s"].copy()
    curve["time_s"][10:14] = curve["time_s"][9] + 2e-6 * np.arange(1, 5)  # steps below the 1e-5 floor
    curve["time_s"][14:] = np.maximum(curve["time_s"][14:], curve["time_s"][13] + 1e-3 * np.arange(1, 67))
    return curve


#: name -> (curve, keyword arguments of ``simulate_thermal_ablation`` besides the geometry and the curve, wall thickness)
CASES = {
    "steel": (make_curve(), {}, 0.004),
    "steel_coarse_steps": (make_curve(points=50, duration=1.5), {}, 0.004),
    "steel_liner": (make_curve(), {"casing_material": CasingMaterial(liner_thickness_m=0.002)}, 0.004),
    "aluminium": (make_curve(points=90), {"casing_material": CasingMaterial(**ALUMINIUM)}, 0.004),
    "aluminium_liner_thick_wall": (
        make_curve(points=200, duration=3.0, shape=0.6),
        {"casing_material": CasingMaterial(liner_thickness_m=0.003, **ALUMINIUM)}, 0.008),
    "scenario_cold_thick_liner": (
        make_curve(), {"casing_material": CasingMaterial(liner_thickness_m=0.002), "liner_thickness_factor": 1.5,
                       "initial_temperature_k": 250.0}, 0.004),
    "no_pressure_series": (make_curve(pressure_series=False), {}, 0.004),
    "ablation_series_with_gaps": (make_curve(points=105, ablation_series=_ablation_with_gaps(105)), {}, 0.004),
    "hot_gas": (make_curve(), {"flame_temp_k": 3200.0, "gamma": 1.2, "r_specific": 320.0}, 0.004),
    "nozzle_material": (
        make_curve(throat_growth_mm=0.4),
        {"nozzle_material": NozzleMaterial(ablation_rate_scale=2.5, ablation_pressure_exponent=0.5,
                                           ablation_mass_flux_exponent=0.25)}, 0.004),
    "steps_below_the_floor": (_tiny_steps(), {}, 0.004),
    "single_interval": (make_curve(points=2, duration=0.05), {}, 0.004),
    "single_point": (make_curve(points=1, duration=0.05), {}, 0.004),
    "very_thin_wall": (make_curve(points=60), {}, 0.0001),
}


def run_case(name, function=None):
    from solidpy import simulate_thermal_ablation

    curve, kwargs, wall = CASES[name]
    kwargs = dict(kwargs)
    kwargs.setdefault("flame_temp_k", 1600.0)
    return (function or simulate_thermal_ablation)(make_geometry(wall), curve, **kwargs)


def load_golden():
    return json.loads(GOLDEN.read_text())


if __name__ == "__main__":
    golden = {name: run_case(name) for name in CASES}
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN} ({len(golden)} cases)")
