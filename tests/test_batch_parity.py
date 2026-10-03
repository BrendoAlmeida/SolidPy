"""Whole-simulation parity of the batched backends on the entire golden corpus (architecture document, 8.1.3).

Every design the backend supports is solved in one ``simulate_burn`` call and compared with the stored reference
result. Slow (minutes), so it is marked ``slow``: run it with ``pytest --runslow tests/test_batch_parity.py``.
"""

import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import _tolerances as tol
from solidpy.ensemble import ProblemBatch, simulate_burn

#: Designs whose reference outcome is decided by rounding, see tests/test_batch_solver.py
ROUNDING_DECIDED = ("solver-failure",)

pytestmark = pytest.mark.slow

INTEGRALS = ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg", "igniter_mass_injected_kg")
GRID_SAMPLED = ("peak_chamber_pressure_pa", "peak_thrust_n", "max_nozzle_mass_flow_kg_s", "gas_mass_cutoff_kg")


def corpus_batch(backend_name):
    corpus = gc.load_corpus()["cases"]
    capabilities = backends.get_backend(backend_name).capabilities()
    cases, built = [], []
    for case in corpus:
        grain, motor, propellant, environment, kwargs = gc.build_objects(case)
        lane = ProblemBatch.from_objects(motor, propellant, environment, kwargs)
        if not lane.unsupported(capabilities)[0]:
            cases.append(case)
            built.append((motor, propellant, environment, kwargs))
    batch = ProblemBatch.from_objects([b[0] for b in built], [b[1] for b in built], [b[2] for b in built],
                                      [b[3] for b in built])
    return cases, batch


def relative(a, b, floor=1e-300):
    return abs(a - b) / max(abs(b), floor)


def compare(cases, results):
    reference = gc.load_reference()["records"]
    deltas = {name: [] for name in (*INTEGRALS, *GRID_SAMPLED, "max_generated_mass_flow_kg_s", "burnout", "nozzle_flow_end_s")}
    wrong, failures = [], []
    for case, got in zip(cases, results):
        stored = reference[case["id"]]
        if case["family"] in ROUNDING_DECIDED:
            continue
        if got["status"]["termination_reason"] != stored["status"]["termination_reason"]:
            wrong.append((case["id"], got["status"]["termination_reason"], stored["status"]["termination_reason"]))
            continue
        m, s = got["metrics"], stored["metrics"]
        for name in INTEGRALS:
            if s[name] > 1e-12:
                deltas[name].append((relative(m[name], s[name]), case["id"]))
        for name in GRID_SAMPLED:
            deltas[name].append((relative(m[name], s[name]), case["id"]))
        deltas["max_generated_mass_flow_kg_s"].append((relative(m["max_generated_mass_flow_kg_s"], s["max_generated_mass_flow_kg_s"]), case["id"]))
        if stored["status"]["completed"] or stored["status"]["termination_reason"] == "unsupported_thermochemistry":
            deltas["nozzle_flow_end_s"].append((relative(m["nozzle_flow_end_s"], s["nozzle_flow_end_s"]), case["id"]))
        for t, expected in zip(m["grain_burnout_times_s"], s["grain_burnout_times_s"]):
            if (t is None) != (expected is None):
                failures.append((case["id"], "burnout set"))
            elif expected is not None:
                deltas["burnout"].append((relative(t, expected), case["id"]))
        for key in ("completed", "numerical_blowdown_completed", "burnout_completed"):
            if got["status"][key] != stored["status"][key]:
                failures.append((case["id"], key))
    return deltas, wrong, failures


def report(name, deltas):
    print(f"\n{name}: whole-corpus relative differences (max, p99.9, median, lanes)")
    for key, values in deltas.items():
        if values:
            v = np.array([d for d, _ in values])
            worst = max(values)[1]
            print(f"  {key:30s} max {v.max():.2e} (at {worst})  p99.9 {np.percentile(v, 99.9):.2e}  median {np.median(v):.2e}  n={len(v)}")


LIMITS = {**{name: tol.INTEGRAL_RTOL for name in INTEGRALS}, **{name: tol.GRID_SAMPLED_RTOL for name in GRID_SAMPLED},
          "max_generated_mass_flow_kg_s": tol.GENERATED_FLOW_PEAK_RTOL, "burnout": tol.TIME_RTOL,
          "nozzle_flow_end_s": tol.TIME_RTOL}


def check(name, cases, results):
    deltas, wrong, failures = compare(cases, results)
    report(name, deltas)
    over = {key: [(d, c) for d, c in values if d > LIMITS[key]] for key, values in deltas.items()}
    assert wrong == [] and failures == []
    assert {key: v for key, v in over.items() if v} == {}


def test_the_numpy_backend_matches_the_stored_reference_on_the_whole_supported_corpus():
    cases, batch = corpus_batch("cpu-vectorized")

    result = simulate_burn(batch, backend="cpu-vectorized", strict=True)

    assert len(cases) > 300
    print("\ntiers:", result.execution["tiers"])
    check("cpu-vectorized", cases, result.to_results())


def test_jax_matches_the_stored_reference_on_the_whole_supported_corpus():
    pytest.importorskip("jax")
    cases, batch = corpus_batch("jax")

    result = simulate_burn(batch, backend="jax", strict=True)

    assert len(cases) > 300
    print("\ntiers:", result.execution["tiers"])
    check("jax", cases, result.to_results())
