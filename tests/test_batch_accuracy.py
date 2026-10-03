import copy

import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.batch import ProblemBatch

#: Lanes where the stored reference is visibly inexact: a loose tolerance (rtol 1e-7) and a coarse step grid.
LANES = ("tolerance-013", "ends-tubular-009")
QUANTITIES = (
    ("final time", lambda m: m["nozzle_flow_end_s"]),
    ("burnout time", lambda m: m["grain_burnout_times_s"][0]),
    ("max generated flow", lambda m: m["max_generated_mass_flow_kg_s"]),
    ("peak pressure", lambda m: m["peak_chamber_pressure_pa"]),
)


@pytest.fixture(scope="module")
def errors():
    """Relative error of the stored reference and of the batched solver against a refined reference run."""
    by_id = {c["id"]: c for c in gc.load_corpus()["cases"]}
    stored = gc.load_reference()["records"]
    out = {}
    for lane in LANES:
        case = by_id[lane]
        refined = copy.deepcopy(case)
        refined["simulation"].update(rtol=1e-11, atol=1e-13,
                                     max_step_size=min(case["simulation"].get("max_step_size", 0.01), 0.002))
        truth = gc.simulate(refined).result["metrics"]
        built = gc.build_objects(case)
        ours = backends.get_backend("cpu-vectorized").solve_burn(
            ProblemBatch.from_objects(built[1], built[2], built[3], built[4])).to_results()[0]["metrics"]
        out[lane] = {
            name: (abs(get(stored[lane]["metrics"]) - get(truth)) / abs(get(truth)),
                   abs(get(ours) - get(truth)) / abs(get(truth)))
            for name, get in QUANTITIES
        }
    return out


@pytest.mark.parametrize("lane", LANES)
def test_the_batched_solver_is_no_less_accurate_than_the_reference_against_a_refined_run(errors, lane):
    """The differences between the two are the reference's own integration and grid error, not a bias.

    The tolerances of ``solidpy/backends/_tolerances.py`` were widened to cover them (version 2): on these lanes
    the stored reference misses the refined final time by 1.4e-6 and the maximum generated flow by 2.4e-3, while
    the batched solver is within 1e-8 and 4e-4. The check keeps it that way: the batched error must not exceed
    the reference's by more than a small slack.
    """
    for name, (reference_error, batched_error) in errors[lane].items():
        assert batched_error <= 1.5 * reference_error + 1e-9, (lane, name, reference_error, batched_error)


def test_the_largest_differences_come_from_the_reference_not_from_the_batched_solver(errors):
    reference_time, batched_time = errors["tolerance-013"]["final time"]
    reference_flow, batched_flow = errors["ends-tubular-009"]["max generated flow"]

    assert reference_time > 1e-6 and batched_time < 0.1 * reference_time
    assert reference_flow > 2e-3 and batched_flow < 0.5 * reference_flow
