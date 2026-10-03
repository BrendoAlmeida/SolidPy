import numpy as np
import pytest

import golden_corpus as gc
from solidpy import BurnSimulation
from solidpy.batch import ProblemBatch
from solidpy.batch.simulation_view import SimulationView
from solidpy.DetailedBallistics import build_detailed_ballistics

ACTIVATION_FAMILIES = ("tubular", "star", "ramp", "activation-scalar", "activation-table", "igniter-table", "combo-table")
MAX_POINTS = 260


@pytest.fixture(scope="module")
def lanes():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    picked = []
    for family in ACTIVATION_FAMILIES:
        members = [c for c in corpus if c["family"] == family and reference[c["id"]]["history_points"] <= MAX_POINTS
                   and reference[c["id"]]["status"]["termination_reason"] == "completed"]
        picked.append(min(members, key=lambda c: reference[c["id"]]["history_points"]))
    out = []
    for case in picked:
        grain, motor, propellant, environment, kwargs = gc.build_objects(case)
        motor.dry_mass_kg, motor.dry_center_of_mass_position_m = 3.0, 0.0
        simulation = BurnSimulation(grain, motor, propellant, environment, **kwargs)
        batch = ProblemBatch.from_objects(motor, propellant, environment, kwargs)
        out.append((case, simulation, SimulationView.from_lane(batch, 0, simulation.result)))
    return out


def same(a, b, path=""):
    assert type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float))), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            if key in ("canonical_result", "simulation"):
                continue
            same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b or (a != a and b != b), path


def test_the_view_carries_what_the_simulation_it_stands_for_carries(lanes):
    for case, simulation, view in lanes:
        assert view.result is simulation.result
        assert view.motor is simulation.motor or view.motor.nozzle_throat_area == simulation.motor.nozzle_throat_area
        assert view.environment_pressure == simulation.environment_pressure, case["id"]
        assert view.propellant.density == simulation.propellant.density


def test_the_activation_rule_is_the_scalar_one_for_every_kind_of_profile(lanes):
    times = np.linspace(-0.05, 1.5, 41)
    regressions = np.linspace(0.0, 0.03, 41)
    kinds = set()
    for case, simulation, view in lanes:
        kinds.add(type(simulation.burn_area_activation).__name__)
        for t, r in zip(times, regressions):
            assert view.evaluate_burn_area_activation(t, r) == simulation.evaluate_burn_area_activation(t, r), case["id"]
    assert kinds == {"NoneType", "float", "list"}  # no profile (with and without the ramp), a scalar and a table
    assert any(simulation.ignition_ramp_time > 0.0 for _, simulation, _ in lanes)


def test_detailed_ballistics_from_the_view_equals_the_one_from_the_real_simulation(lanes):
    for case, simulation, view in lanes:
        real = build_detailed_ballistics(simulation, resample_step=0.02, max_time_points=300)
        stood_in = build_detailed_ballistics(view, resample_step=0.02, max_time_points=300)

        same(real, stood_in, case["id"])
        assert stood_in["canonical_result"] is simulation.result


def test_detailed_ballistics_without_resampling_also_agrees(lanes):
    case, simulation, view = lanes[0]

    same(build_detailed_ballistics(simulation), build_detailed_ballistics(view), case["id"])
