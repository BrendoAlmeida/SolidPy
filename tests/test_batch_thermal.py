"""The batched thermal ablation: packing, the NumPy and reference backends and the flux kernel."""

import math

import numpy as np
import pytest

from solidpy import CasingMaterial, backends
from solidpy.backends import SolveOptions, UnsupportedLane
from solidpy.backends import _tolerances as tol
from solidpy.batch import thermal as batch_thermal
from solidpy.batch.integrators import radau
from solidpy.batch.kernels import thermal as thermal_kernels
from solidpy.batch.thermal import NON_FINITE_INPUT, SERIES_MISMATCH, ThermalBatch
from thermal_cases import CASES, case_lane, make_curve, make_geometry, pack_lanes, random_lanes, scalar_thermal

FIXED = [case_lane(name) for name in CASES]
RANDOM = random_lanes(24, seed=5)


@pytest.fixture(scope="module")
def fixed_batch():
    return pack_lanes(FIXED)


@pytest.fixture(scope="module")
def scalar_fixed():
    return [scalar_thermal(lane) for lane in FIXED]


def close(got, want, rtol=tol.THERMAL_RTOL):
    assert set(got) == set(want)
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=rtol, abs=1e-12), key


# -- packing ----------------------------------------------------------------------------------------------------------

def test_a_batch_pads_cells_and_steps_with_inert_entries(fixed_batch):
    a = fixed_batch.arrays
    names = list(CASES)

    assert len(fixed_batch) == len(CASES)
    assert fixed_batch.n_max == 11 and fixed_batch.t_max == 199
    assert a["lower"].shape == a["upper"].shape == (len(CASES), 10) and a["diag"].shape == (len(CASES), 11)
    assert a["n_nodes"][names.index("steel")] == 4 and a["n_nodes"][names.index("steel_liner")] == 6
    for lane in range(len(CASES)):
        n, m = int(a["n_nodes"][lane]), int(a["n_intervals"][lane])
        assert not a["diag"][lane, n:].any() and not a["y0"][lane, n:].any() and not a["source"][lane, n:].any()
        assert not a["lower"][lane, max(n - 1, 0):].any() and not a["upper"][lane, max(n - 1, 0):].any()
        assert (a["dt"][lane, :m] >= 1e-5).all() and (a["dt"][lane, m:] == 1.0).all() and not a["bartz"][lane, m:].any()
    assert a["n_intervals"][names.index("single_point")] == 0 and a["n_intervals"][names.index("single_interval")] == 1


def test_the_start_temperature_is_floored_but_the_ambient_of_the_outer_wall_is_not():
    cold = pack_lanes([dict(FIXED[0], initial_temperature_k=100.0)])

    assert (cold.arrays["y0"][0, :4] == 150.0).all() and cold.arrays["initial_temp_k"][0] == 100.0


def test_select_keeps_the_order_trims_the_padding_and_can_repeat_a_lane(fixed_batch):
    names = list(CASES)
    picked = fixed_batch.select([names.index("steel_liner"), names.index("steel")])

    assert picked.n_max == 6 and picked.t_max == 119
    assert picked.lanes[0] is fixed_batch.lanes[names.index("steel_liner")]
    again = fixed_batch.select([2, 2, 2], trim=False)
    assert len(again) == 3 and again.n_max == fixed_batch.n_max and again.t_max == fixed_batch.t_max
    assert (again.arrays["diag"] == again.arrays["diag"][0]).all()
    assert len(fixed_batch.select([])) == 0


def test_padding_widens_both_axes_and_never_shrinks_them(fixed_batch):
    wide = fixed_batch.with_padding(16, 256)

    assert wide.n_max == 16 and wide.t_max == 256
    np.testing.assert_array_equal(wide.arrays["diag"][:, :11], fixed_batch.arrays["diag"])
    assert not wide.arrays["diag"][:, 11:].any() and (wide.arrays["dt"][:, 199:] == 1.0).all()
    with pytest.raises(ValueError, match="cannot shrink"):
        fixed_batch.with_padding(8, 256)


def test_arguments_are_one_value_for_all_lanes_or_one_per_lane():
    geometry, curve = make_geometry(), make_curve(points=40)

    shared = ThermalBatch.from_objects(geometry, [curve, curve, curve], flame_temp_k=[1500.0, 1600.0, 1700.0],
                                       initial_temperature_k=np.array([250.0, 280.0, 300.0]))
    assert len(shared) == 3 and shared.arrays["flame_temp_k"].tolist() == [1500.0, 1600.0, 1700.0]
    assert shared.arrays["initial_temp_k"].tolist() == [250.0, 280.0, 300.0]
    with pytest.raises(ValueError, match=r"disagree on the number of lanes: \[2, 3\]"):
        ThermalBatch.from_objects(geometry, [curve] * 3, flame_temp_k=[1500.0, 1600.0])
    with pytest.raises(ValueError, match="disagree on the number of lanes"):
        ThermalBatch.from_objects([geometry] * 2, [curve] * 3)
    assert len(ThermalBatch.from_objects(geometry, curve)) == 1


def test_a_curve_without_the_series_the_scalar_model_reads_is_refused_when_packed():
    curve = make_curve(points=20)
    del curve["thrust_n"]

    with pytest.raises(KeyError):
        ThermalBatch.from_objects(make_geometry(), curve)


def test_the_gamma_comes_from_the_argument_then_the_curve_then_the_default():
    plain, tagged = make_curve(points=20), dict(make_curve(points=20), gamma=1.25)
    batch = ThermalBatch.from_objects(make_geometry(), [plain, tagged, tagged], gamma=[None, None, 1.1])

    assert batch.arrays["stagnation"].tolist() == [(1.3 + 1.0) / 2.0, (1.25 + 1.0) / 2.0, (1.1 + 1.0) / 2.0]


def test_lanes_the_batched_solve_cannot_reproduce_carry_a_feature():
    ablation = np.linspace(0.0, 1e-4, 50)
    ablation[7] = np.inf
    nan_thrust = dict(make_curve(points=50))
    nan_thrust["thrust_n"] = nan_thrust["thrust_n"].copy()
    nan_thrust["thrust_n"][3] = np.nan
    short = dict(make_curve(points=50))
    short["thrust_n"] = short["thrust_n"][:40]
    lanes = [FIXED[0], dict(FIXED[0], curve=make_curve(points=50, ablation_series=ablation)), dict(FIXED[0], curve=nan_thrust),
             dict(FIXED[0], curve=short), dict(FIXED[0], flame_temp_k=float("nan"))]

    batch = pack_lanes(lanes)

    assert [sorted(f) for f in batch.features] == [[], [NON_FINITE_INPUT], [NON_FINITE_INPUT], [SERIES_MISMATCH], [NON_FINITE_INPUT]]
    numpy_caps = backends.get_backend("cpu-vectorized").capabilities()
    reference_caps = backends.get_backend("cpu-reference").capabilities()
    assert batch.unsupported(numpy_caps) == [[], [NON_FINITE_INPUT], [NON_FINITE_INPUT], [SERIES_MISMATCH], [NON_FINITE_INPUT]]
    assert batch.unsupported(reference_caps) == [[]] * 5
    assert np.isfinite(batch.arrays["diag"]).all() and np.isfinite(batch.arrays["dt"]).all()  # placeholders stay finite


def test_the_throat_ablation_sum_reproduces_the_order_of_the_scalar_loop():
    rng = np.random.default_rng(2)
    for case in range(30):
        points = int(rng.integers(2, 60))
        time = np.cumsum(rng.uniform(0.001, 0.05, points))
        series = np.full(points, np.nan)
        if case % 3:
            given = rng.random(points) < 0.3
            series[given] = rng.uniform(-1e-5, 4e-4, int(given.sum()))
        args = dict(thrust_n=rng.uniform(0, 2000, points), mass_flow_kg_s=rng.uniform(0, 1, points),
                    pressure_pa=rng.uniform(0, 8e6, points), throat_diameter_m=np.full(points, 0.035))
        dt, bartz, _, ablation = batch_thermal.interval_coefficients(
            time, args["thrust_n"], args["mass_flow_kg_s"], args["pressure_pa"], args["throat_diameter_m"], series,
            1600.0, 287.0, 1.2, 1.5, 0.42, 0.32)

        total = 0.0  # the loop of simulate_thermal_ablation
        for i in range(1, points):
            area = math.pi * max(0.035 * 0.5, 1e-6) ** 2
            pressure = max(args["pressure_pa"][i], 0.0)
            if pressure <= 0.0:
                pressure = max(args["thrust_n"][i], 0.0) / max(area, 1e-9)
            mass_flow = max(args["mass_flow_kg_s"][i], 0.0)
            if not np.isnan(series[i]):
                total = max(float(series[i]), 0.0)
            else:
                total += 1.8e-8 * 1.5 * max(pressure, 1.0) ** 0.42 * max(mass_flow, 1e-9) ** 0.32 * dt[i - 1]
        assert ablation == total, case
        assert len(bartz) == points - 1


# -- the reference and NumPy backends ---------------------------------------------------------------------------------

def test_the_reference_backend_returns_what_the_scalar_model_returns(fixed_batch, scalar_fixed):
    result = backends.get_backend("cpu-reference").thermal_ablation(fixed_batch)

    assert result.results == scalar_fixed and result.backend == "cpu-reference"
    assert result.execution["service"] == "thermal_ablation"


def test_the_reference_backend_gives_the_same_results_through_a_process_pool(fixed_batch, scalar_fixed):
    pooled = backends.get_backend("cpu-reference").thermal_ablation(fixed_batch, SolveOptions(workers=2))

    assert pooled.results == scalar_fixed


def test_the_numpy_backend_matches_the_scalar_model_on_the_fixed_cases(fixed_batch, scalar_fixed):
    result = backends.get_backend("cpu-vectorized").thermal_ablation(fixed_batch)

    assert result.execution["failed_lanes"] == []
    for got, want in zip(result.results, scalar_fixed):
        close(got, want)


def test_the_numpy_backend_matches_the_scalar_model_on_random_lanes():
    result = backends.get_backend("cpu-vectorized").thermal_ablation(pack_lanes(RANDOM))

    assert result.execution["failed_lanes"] == [] and result.execution["radau_steps"] > 24 * 50
    for got, lane in zip(result.results, RANDOM):
        close(got, scalar_thermal(lane))


def test_the_numpy_backend_takes_the_steps_the_scalar_solver_takes(monkeypatch):
    """The total of Radau steps agrees with scipy's own count over the same lanes."""
    from solidpy import Multiphysics

    original = Multiphysics.solve_ivp
    taken = []

    def counting(*args, **kwargs):
        solution = original(*args, **kwargs)
        taken.append(len(solution.t) - 1)
        return solution

    monkeypatch.setattr(Multiphysics, "solve_ivp", counting)
    for lane in RANDOM[:8]:
        scalar_thermal(lane)
    monkeypatch.undo()
    result = backends.get_backend("cpu-vectorized").thermal_ablation(pack_lanes(RANDOM[:8]))

    assert result.execution["radau_steps"] == sum(taken)


def test_a_lane_gives_the_same_result_alone_in_a_batch_and_in_another_order(fixed_batch):
    numpy_backend = backends.get_backend("cpu-vectorized")
    together = numpy_backend.thermal_ablation(fixed_batch).results
    reversed_ = numpy_backend.thermal_ablation(fixed_batch.select(range(len(CASES) - 1, -1, -1))).results[::-1]

    for lane in (0, 4, 5, 9):
        alone = numpy_backend.thermal_ablation(fixed_batch.select([lane])).results[0]
        close(alone, together[lane], rtol=1e-12)
        close(reversed_[lane], together[lane], rtol=1e-12)


def test_an_empty_batch_gives_no_results():
    empty = pack_lanes(RANDOM[:1]).select([])

    assert backends.get_backend("cpu-vectorized").thermal_ablation(empty).results == []
    assert backends.get_backend("cpu-reference").thermal_ablation(empty).results == []


def test_a_lane_whose_integration_does_not_finish_is_returned_as_none_and_listed(fixed_batch, monkeypatch):
    monkeypatch.setattr(radau, "MAX_ATTEMPTS", 1)  # a time step needing a second attempt cannot finish
    names = list(CASES)

    result = backends.get_backend("cpu-vectorized").thermal_ablation(fixed_batch)

    failed = result.execution["failed_lanes"]
    assert failed and [lane for lane, r in enumerate(result.results) if r is None] == failed
    assert names.index("single_point") not in failed  # no time step, nothing to fail
    assert len(failed) < len(CASES)


def test_the_batched_backends_refuse_the_lanes_they_cannot_reproduce():
    batch = pack_lanes([FIXED[0], dict(FIXED[0], flame_temp_k=float("inf"))])

    with pytest.raises(UnsupportedLane, match=r"cpu-vectorized.*1: non_finite_thermal_input"):
        backends.get_backend("cpu-vectorized").thermal_ablation(batch)
    assert len(backends.get_backend("cpu-reference").thermal_ablation(batch.select([0])).results) == 1


def test_options_are_checked_and_the_execution_describes_the_integration(fixed_batch):
    numpy_backend = backends.get_backend("cpu-vectorized")

    with pytest.raises(TypeError, match="SolveOptions"):
        numpy_backend.thermal_ablation(fixed_batch, {"workers": 2})
    execution = numpy_backend.thermal_ablation(fixed_batch.select([0, 1])).execution
    assert execution["integrator"] == "radau-iia5" and execution["rtol"] == 1e-5 and execution["atol"] == 1e-6
    assert execution["radau_attempts"] >= execution["radau_steps"] > 0 and set(numpy_backend.last_timings) == {"solve_s", "assemble_s"}


def test_the_backends_advertise_the_thermal_service():
    assert backends.get_backend("cpu-reference").capabilities().provides("thermal_ablation")
    assert backends.get_backend("cpu-vectorized").capabilities().provides("thermal_ablation")
    assert not backends.Capabilities().provides("thermal_ablation")


def test_a_casing_without_a_liner_reports_the_inner_wall_for_the_hot_face():
    plain = CasingMaterial()
    lanes = [dict(FIXED[0], casing_material=plain), dict(FIXED[0], casing_material=CasingMaterial(liner_thickness_m=0.002))]

    got = backends.get_backend("cpu-vectorized").thermal_ablation(pack_lanes(lanes)).results

    key = "simulation.advanced.thermal."
    assert got[0][key + "liner_hot_face_temp_c"] == got[0][key + "max_inner_wall_temp_c"]
    assert got[1][key + "liner_hot_face_temp_c"] > got[1][key + "max_inner_wall_temp_c"]


def scalar_flux(hot_face_k, bartz, recovery_k, flame_k, stagnation):
    """``heat_flux_and_derivative_from_hot_face`` of ``simulate_thermal_ablation``, transcribed."""
    tw_t0_ratio = hot_face_k / max(flame_k, 1.0)
    sigma_base = 0.5 * tw_t0_ratio / max(stagnation, 1e-9) + 0.5
    sigma = sigma_base ** (-0.68) * stagnation ** (-0.12)
    delta = recovery_k - hot_face_k
    if delta <= 0.0:
        return 0.0, 0.0
    if sigma <= 0.1:
        return bartz * 0.1 * delta, -bartz * 0.1
    sigma_derivative = -0.34 * sigma / max(flame_k * stagnation * sigma_base, 1e-9)
    return bartz * sigma * delta, bartz * (sigma_derivative * delta - sigma)


def test_the_flux_kernel_matches_the_scalar_closure_including_both_clamps():
    """Above the recovery temperature the flux is zero; for a sigma of at most 0.1 (a hot face of ~1e5 K) it is bartz * 0.1 * delta."""
    temperatures = np.concatenate([np.linspace(150.0, 2500.0, 60), [1e4, 1e5, 2e5, 5e5, 5e6]])
    for bartz, recovery, flame, gamma in [(6000.0, 1550.0, 1600.0, 1.136), (2e4, 3000.0, 3400.0, 1.25), (50.0, 1e6, 3000.0, 1.2)]:
        stagnation = (gamma + 1.0) / 2.0
        got = thermal_kernels.flux(np, temperatures, bartz, recovery, flame, stagnation)
        want = np.array([scalar_flux(t, bartz, recovery, flame, stagnation) for t in temperatures])
        np.testing.assert_allclose(got[0], want[:, 0], rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(got[1], want[:, 1], rtol=1e-13, atol=0.0)
    assert (thermal_kernels.flux(np, np.array([1600.0, 1900.0]), 6000.0, 1550.0, 1600.0, 1.07)[0] == 0.0).all()
    assert thermal_kernels.flux(np, np.array([2e5]), 50.0, 1e6, 3000.0, 1.1)[0][0] > 0.0  # the clamped branch, not the cutoff


def test_a_wall_that_starts_hotter_than_the_recovery_temperature_only_cools():
    name = "hot_start_above_the_recovery_temperature"
    lane = case_lane(name)

    got = backends.get_backend("cpu-vectorized").thermal_ablation(pack_lanes([lane])).results[0]

    close(got, scalar_thermal(lane))
    assert got["simulation.advanced.thermal.throat_heat_flux_kw_m2"] == 0.0 and got["simulation.advanced.thermal.heat_load_kj_m2"] == 0.0
