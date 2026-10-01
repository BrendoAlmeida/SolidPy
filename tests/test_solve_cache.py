"""The solve cache only skips repeated evaluations: results are bit-for-bit the same."""

import numpy as np
import pytest

from solidpy import BurnSimulation, Environment, Grain, Motor, Propellant


def make_stack(geometry="tubular", grain_count=1):
    grains = [
        Grain(0.035, 0.015, initial_height=0.06, geometry=geometry) for _ in range(grain_count)
    ]
    motor = Motor(
        grains, chamber_inner_radius=0.037,
        chamber_length=sum(g.initial_height for g in grains) + 0.02,
        nozzle_throat_radius=0.008, nozzle_exit_radius=0.018,
    )
    propellant = Propellant(
        1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36, burn_rate_n=0.32,
    )
    return grains[0], motor, propellant, Environment()


def simulate(solve_cache, geometry="tubular", grain_count=1, cls=BurnSimulation):
    return cls(
        *make_stack(geometry, grain_count), max_step_size=0.03, solve_cache=solve_cache,
        eta_c=0.93, eta_Cf=0.96, discharge_coefficient=0.98,
    )


def assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert_same(a[key], b[key])
        elif isinstance(a[key], np.ndarray):
            assert np.array_equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("geometry,grain_count", [("tubular", 1), ("star", 1), ("tubular", 3)])
def test_results_are_identical_with_and_without_the_cache(geometry, grain_count):
    cached = simulate(True, geometry, grain_count).result
    plain = simulate(False, geometry, grain_count).result
    for section in ("history", "metrics", "status"):
        assert_same(cached[section], plain[section])


def test_the_cache_is_released_when_the_solve_ends_and_never_serves_stale_values():
    simulation = simulate(True)
    assert simulation._solve_cache is None
    pressure = 4.0e6
    before = simulation._parameters_at_pressure(pressure)
    simulation.eta_c = 0.80  # an input changed after the solve: the next call must see it
    after = simulation._parameters_at_pressure(pressure)
    assert after[0] == pytest.approx(before[0] * (0.80 / 0.93) ** 2)


def test_the_cache_is_released_when_the_solve_fails():
    class Failing(BurnSimulation):
        def evaluate_tail_off_solution(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        Failing(*make_stack(), max_step_size=0.03)
    # a second simulation is unaffected by the failed one
    assert simulate(True).result["status"]["completed"]


def test_only_the_post_processing_evaluations_are_memoised():
    class Counting(BurnSimulation):
        def __init__(self, *args, **kwargs):
            self.calls = {"integrator": 0, "post": 0}
            super().__init__(*args, **kwargs)

        def _state_quantities_uncached(self, time, state, active=None):
            self.calls["post" if active is None else "integrator"] += 1
            return super()._state_quantities_uncached(time, state, active)

    cached = simulate(True, cls=Counting).calls
    plain = simulate(False, cls=Counting).calls
    assert cached["integrator"] == plain["integrator"], "the integrator's calls are never cached"
    assert cached["post"] < plain["post"], "repeated post-processing points are evaluated once"
