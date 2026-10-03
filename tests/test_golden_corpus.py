from collections import Counter

import numpy as np
import pytest

import golden_corpus as gc
from solidpy.backends import _tolerances as tol

REGENERATE = (
    "the scalar reference physics changed since the golden corpus was generated; review the change, then "
    "regenerate with `python tools/make_golden_corpus.py` and commit the new files"
)

#: Minimum number of corpus cases that must carry each tag (coverage of the feature matrix).
REQUIRED_TAGS = {
    "tubular": 20, "star": 20, "mixed_geometry": 5,
    "grains:1": 5, "grains:2": 5, "grains:8": 5, "grains:24": 5,
    "ends_burn": 10, "ends_open": 10,
    "power_law": 50, "burn_rate_table": 10, "scalar_thermo": 50, "thermo_table": 5,
    "erosive": 10, "eta_c": 5, "eta_cf": 5, "discharge": 5,
    "igniter_none": 50, "igniter_scalar": 3, "igniter_table": 3, "igniter_callable": 2,
    "activation_none": 50, "ramp": 3, "activation_scalar": 3, "activation_table": 3, "activation_callable": 2,
    "short_burn": 3, "long_burn": 3, "low_kn": 3, "axial_burnout_first": 3, "thin_web": 2,
    "tail_off_analytical": 3, "tail_off_omitted": 2, "grain_number_replicated": 3,
    "igniter_after_burnout": 2, "blowdown_truncated_by_breakpoint": 2,
}

REQUIRED_OUTCOMES = {
    "completed", "burn_timeout", "blowdown_timeout", "solver_failure", "unknown_igniter_duration",
    "unsupported_thermochemistry", "analytical_approximation", "tail_off_omitted",
}

REPRODUCED_FAMILIES = ("tubular", "star", "mixed", "erosive", "efficiency", "ratetable", "igniter-table",
                       "activation-table", "ramp", "quirk", "timeout-burn", "solver-failure")


@pytest.fixture(scope="module")
def corpus():
    return gc.load_corpus()["cases"]


@pytest.fixture(scope="module")
def reference():
    return gc.load_reference()


def test_reference_sources_match_the_frozen_manifest(reference):
    assert gc.reference_sources_sha256() == reference["manifest"]["reference_sources_sha256"], REGENERATE


def test_every_case_has_exactly_one_error_free_reference_record(corpus, reference):
    ids = [case["id"] for case in corpus]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(reference["records"])
    assert reference["manifest"]["cases"] == len(ids)
    assert [i for i, record in reference["records"].items() if "error" in record] == []


def test_stored_tags_contain_the_tags_derived_from_the_design(corpus):
    assert [c["id"] for c in corpus if not set(gc.derive_tags(c)) <= set(c["tags"])] == []


def test_every_design_builds_physical_objects(corpus):
    for case in corpus:
        _, motor, propellant, _, kwargs = gc.build_objects(case)
        assert motor.free_volume > 0.0, case["id"]
        assert propellant.density > 0.0, case["id"]
        assert kwargs.get("max_step_size", 0.01) > 0.0, case["id"]


def test_corpus_covers_the_feature_matrix(corpus, reference):
    tags = Counter(tag for case in corpus for tag in case["tags"])
    short = {tag: (tags[tag], minimum) for tag, minimum in REQUIRED_TAGS.items() if tags[tag] < minimum}
    assert short == {}

    outcomes = Counter(r["status"]["termination_reason"] for r in reference["records"].values())
    assert REQUIRED_OUTCOMES - set(outcomes) == set()


def test_derived_cases_end_the_way_the_generator_intended(corpus, reference):
    wrong = {
        case["id"]: reference["records"][case["id"]]["status"]["termination_reason"]
        for case in corpus
        if "expect_termination_reason" in case
        and reference["records"][case["id"]]["status"]["termination_reason"] != case["expect_termination_reason"]
    }
    assert wrong == {}


def test_the_blowdown_truncation_of_the_reference_is_part_of_the_corpus(corpus, reference):
    """A breakpoint between burnout and the cutoff ends the blowdown early (Burn._integrate_stage)."""
    quirk = [case for case in corpus if "blowdown_truncated_by_breakpoint" in case["tags"]]
    for case in quirk:
        record = reference["records"][case["id"]]
        assert record["status"]["termination_reason"] == "blowdown_timeout"
        assert record["status"]["numerical_blowdown_completed"] is False


def _cheapest(corpus, reference, family):
    members = [case for case in corpus if case["family"] == family]
    return min(members, key=lambda case: reference["records"][case["id"]]["wall_seconds"])


@pytest.mark.parametrize("family", REPRODUCED_FAMILIES)
def test_stored_reference_results_are_reproduced_by_the_current_reference(family, corpus, reference):
    case = _cheapest(corpus, reference, family)
    stored = reference["records"][case["id"]]

    result = gc.simulate(case).result

    assert result["status"]["termination_reason"] == stored["status"]["termination_reason"]
    assert result["status"]["completed"] == stored["status"]["completed"]
    assert result["provenance"]["physics_provider_hash"] == stored["physics_provider_hash"]
    metrics = result["metrics"]
    for name in ("total_impulse_ns", "nozzle_mass_integral_kg", "generated_mass_integral_kg"):
        assert metrics[name] == pytest.approx(stored["metrics"][name], rel=tol.INTEGRAL_RTOL), name
    for name in ("peak_chamber_pressure_pa", "peak_thrust_n"):
        assert metrics[name] == pytest.approx(stored["metrics"][name], rel=tol.GRID_SAMPLED_RTOL), name
    assert np.asarray(result["history"]["time_s"])[-1] == pytest.approx(stored["end_time_s"], rel=tol.TIME_RTOL)
