#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU reference benchmark for the reported, non-gating W5 flight dispersion.

The benchmark creates one real SolidPy burn curve, derives a complete motor
geometry, and sends each sampled flight through ``simulate_flight_3dof`` using
``DispersionAnalysis.run``.  Its process pool and report assembly are included
in campaign timings; preparation of the shared burn curve is measured apart.

    python benchmarks/bench_flight_dispersion.py --samples 16 --repeat 2 \
        --out benchmarks/results/w5_cpu_reference_flight_dispersion.json
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solidpy import (  # noqa: E402
    BurnSimulation,
    DispersionAnalysis,
    Environment,
    Grain,
    Motor,
    MotorGeometry,
    Propellant,
    geometry_from_components,
    simulate_flight_3dof,
)

SEED = 20261004
SAMPLE_DISTRIBUTIONS = {
    "launch_angle_deg": {"mean": 82.0, "sigma": 0.4},
    "launch_azimuth_deg": {"mean": 45.0, "sigma": 1.0},
    "wind_speed_m_s": {"mean": 5.0, "sigma": 1.0},
    "wind_direction_deg": {"mean": 270.0, "sigma": 5.0},
    "drag_coefficient_factor": {"mean": 1.0, "sigma": 0.025},
}
FLIGHT_METRIC_KEYS = {
    "landing_downrange_m": "simulation.3dof.landing_downrange_m",
    "landing_crossrange_m": "simulation.3dof.landing_crossrange_m",
    "max_altitude_m": "simulation.3dof.max_altitude_m",
    "flight_time_s": "simulation.3dof.flight_time_s",
}


@dataclass
class FlightCase:
    """Picklable callable that returns actual 3-DOF simulation results."""

    geometry: MotorGeometry
    curve: dict

    def __call__(self, sample):
        return simulate_flight_3dof(
            self.geometry,
            self.curve,
            launch_angle_deg=sample["launch_angle_deg"],
            launch_azimuth_deg=sample["launch_azimuth_deg"],
            wind_speed_m_s=sample["wind_speed_m_s"],
            wind_direction_deg=sample["wind_direction_deg"],
            drag_coefficient_factor=sample["drag_coefficient_factor"],
        )


def _make_case():
    """Build a complete KNSB motor, its reference burn, and flight inputs."""
    grain = Grain(
        outer_radius=71.92 / 2000,
        initial_inner_radius=31.92 / 2000,
        mass=0.7,
    )
    motor = Motor(
        grain,
        grain_number=4,
        chamber_inner_radius=77.92 / 2000,
        nozzle_throat_radius=17.5 / 2000,
        nozzle_exit_radius=44.44 / 2000,
        nozzle_angle=15 * np.pi / 180,
        chamber_length=600 / 1000,
        dry_mass_kg=3.0,
        dry_center_of_mass_position_m=0.0,
    )
    propellant = Propellant(
        specific_heat_ratio=1.1361,
        density=1700,
        products_molecular_mass=39.9e-3,
        combustion_temperature=1600,
        interpolation_list=str(ROOT / "data/burnrate/KNSB3.csv"),
    )
    environment = Environment()

    started = time.perf_counter()
    burn = BurnSimulation(
        grain,
        motor,
        propellant,
        environment,
        max_step_size=0.03,
        tail_off_evaluation=True,
    ).result
    burn_preparation_s = time.perf_counter() - started
    if not burn["status"]["completed"] or not burn["status"]["burnout_completed"]:
        raise RuntimeError(
            "W5 reference burn did not complete: "
            f"{burn['status']['termination_reason']}"
        )

    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=motor.dry_mass_kg,
    )
    history = burn["history"]
    time_s = np.asarray(history["time_s"], dtype=float)
    thrust_n = np.asarray(history["thrust_n"], dtype=float)
    regressions_m = np.asarray(history["regression_m"], dtype=float)
    if (
        time_s.ndim != 1
        or thrust_n.shape != time_s.shape
        or regressions_m.shape != (time_s.size, motor.grain_number)
        or time_s.size < 2
        or not np.all(np.isfinite(time_s))
        or not np.all(np.isfinite(thrust_n))
        or not np.all(np.isfinite(regressions_m))
        or np.any(np.diff(time_s) <= 0.0)
    ):
        raise RuntimeError("reference burn history has inconsistent time-series shapes")

    propellant_mass_kg = np.asarray(
        [
            propellant.density
            * sum(
                grain_i.calculate_remaining_volume(regression)
                for grain_i, regression in zip(motor.grains, row)
            )
            for row in regressions_m
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(propellant_mass_kg)):
        raise RuntimeError("reference burn produced non-finite propellant mass")

    curve = {
        "time_s": time_s,
        "thrust_n": thrust_n,
        "propellant_mass_kg": propellant_mass_kg,
    }
    case = FlightCase(geometry=geometry, curve=curve)
    burn_info = {
        "preparation_s": burn_preparation_s,
        "termination_reason": burn["status"]["termination_reason"],
        "curve_points": int(time_s.size),
        "curve_duration_s": float(time_s[-1] - time_s[0]),
        "peak_thrust_n": float(burn["metrics"]["peak_thrust_n"]),
        "total_impulse_ns": float(burn["metrics"]["total_impulse_ns"]),
        "propellant_mass_initial_kg": float(propellant_mass_kg[0]),
        "propellant_mass_final_kg": float(propellant_mass_kg[-1]),
        "solver_settings": burn["provenance"]["solver_settings"],
    }
    geometry_info = {
        "motor_grains": motor.grain_number,
        "grain_outer_radius_m": grain.outer_radius,
        "grain_initial_inner_radius_m": grain.initial_inner_radius,
        "grain_height_m": grain.initial_height,
        "motor_dry_mass_kg": geometry.dry_mass_kg,
        "motor_initial_mass_kg": geometry.motor_initial_mass_kg,
        "motor_final_mass_kg": geometry.motor_final_mass_kg,
        "propellant_mass_kg": geometry.propellant_mass_kg,
        "mass_scope": geometry.mass_scope,
    }
    return case, burn_info, geometry_info


def _machine_info():
    def package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    cpu_name = platform.processor() or None
    try:
        cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
        model_lines = [line for line in cpu_lines if line.startswith("model name")]
        if model_lines:
            cpu_name = model_lines[0].split(":", 1)[1].strip()
    except OSError:
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_name,
        "logical_cpu_count": os.cpu_count(),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "solidpy": package_version("solidpy"),
        "accelerator_used": False,
    }


def _ellipse_json(ellipse):
    result = dict(ellipse)
    for key in ("center_m", "covariance_matrix", "eigenvalues", "eigenvectors"):
        result[key] = np.asarray(result[key]).tolist()
    return result


def _campaign_result(analysis, samples):
    result = analysis.run(samples)
    impact_points = np.asarray(result["impact_points"], dtype=float)
    if (
        impact_points.shape != (samples, 2)
        or len(result["results"]) != samples
        or len(result["inputs"]) != samples
        or not np.all(np.isfinite(impact_points))
    ):
        raise RuntimeError("W5 flight campaign returned invalid impact coordinates")

    flight_metrics = []
    landing_points = []
    for flight in result["results"]:
        if not isinstance(flight, dict):
            raise RuntimeError("W5 flight campaign returned a non-mapping flight result")
        metrics = {
            name: float(flight[key]) for name, key in FLIGHT_METRIC_KEYS.items()
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError("W5 flight campaign returned non-finite flight metrics")
        trajectory = flight.get("trajectory")
        if not isinstance(trajectory, dict):
            raise RuntimeError("W5 flight result does not contain a trajectory")
        trajectory_axes = [
            np.asarray(trajectory.get(axis), dtype=float)
            for axis in ("x_m", "y_m", "z_m")
        ]
        # simulate_flight_3dof exposes powered-phase samples here; coast landing
        # coordinates are returned separately in the simulation.3dof.landing_* metrics.
        if (
            any(axis.ndim != 1 or axis.size == 0 for axis in trajectory_axes)
            or len({axis.size for axis in trajectory_axes}) != 1
            or not all(np.all(np.isfinite(axis)) for axis in trajectory_axes)
        ):
            raise RuntimeError("W5 flight returned invalid powered-trajectory arrays")
        landing_points.append(
            (metrics["landing_downrange_m"], metrics["landing_crossrange_m"])
        )
        flight_metrics.append(metrics)

    landing_points = np.asarray(landing_points, dtype=float)
    if (
        landing_points.shape != (samples, 2)
        or not np.all(np.isfinite(landing_points))
        or not np.array_equal(impact_points, landing_points)
    ):
        raise RuntimeError(
            "W5 impact points do not match finite landing coordinates from the flight reports"
        )

    return {
        "summary": {key: float(value) for key, value in result["summary"].items()},
        "dispersion_ellipse": _ellipse_json(result["dispersion_ellipse"]),
        "impact_points_m": impact_points.tolist(),
        "flight_metrics": flight_metrics,
        "sampled_inputs": result["inputs"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=int, default=16, help="flight samples (minimum 3)")
    parser.add_argument("--repeat", type=int, default=2, help="timed campaigns after the first call")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3 for a non-degenerate dispersion summary")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    case, burn_info, geometry_info = _make_case()
    analysis = DispersionAnalysis(
        simulation=case,
        parameter_sigmas=SAMPLE_DISTRIBUTIONS,
        random_seed=args.seed,
    )

    started = time.perf_counter()
    first_result = _campaign_result(analysis, args.samples)
    first_call_s = time.perf_counter() - started

    repeat_times_s = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        repeat_result = _campaign_result(analysis, args.samples)
        repeat_times_s.append(time.perf_counter() - started)
    # The fixed seed makes every campaign use the same samples; keep one result
    # payload and report timing separately to avoid duplicating impact arrays.
    if first_result["impact_points_m"] != repeat_result["impact_points_m"]:
        raise RuntimeError("repeated W5 campaigns did not reproduce identical impacts")

    sorted_times = sorted(repeat_times_s)
    median_repeat_s = float(np.median(sorted_times))
    payload = {
        "schema_version": 1,
        "workload": "W5",
        "gating": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "backend": "cpu-reference",
            "service": "flight_dispersion",
            "accelerator_used": False,
            "process_workers": min(args.samples, os.cpu_count() or 1),
        },
        "configuration": {
            "samples": args.samples,
            "repeat": args.repeat,
            "seed": args.seed,
            "sample_distributions": SAMPLE_DISTRIBUTIONS,
            "flight_api": "solidpy.simulate_flight_3dof",
            "dispersion_api": "solidpy.DispersionAnalysis.run",
            "burn_api": "solidpy.BurnSimulation",
            "geometry_api": "solidpy.geometry_from_components",
        },
        "burn": burn_info,
        "geometry": geometry_info,
        "timing": {
            "scope": "DispersionAnalysis.run including process-pool startup and report assembly; shared burn preparation is reported separately",
            "first_call_s": first_call_s,
            "repeat_times_s": repeat_times_s,
            "median_repeat_s": median_repeat_s,
            "samples_per_s": args.samples / median_repeat_s,
        },
        "dispersion": repeat_result["summary"],
        "dispersion_ellipse": repeat_result["dispersion_ellipse"],
        "impact_points_m": repeat_result["impact_points_m"],
        "flight_metrics": repeat_result["flight_metrics"],
        "sampled_inputs": repeat_result["sampled_inputs"],
        "machine": _machine_info(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
