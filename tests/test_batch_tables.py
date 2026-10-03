import numpy as np
import pytest

import golden_corpus as gc
from solidpy import Grain, Motor, Propellant, backends
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch
from solidpy.batch.kernels import propellant as propellant_kernels
from solidpy.batch.kernels import tables

RTOL = tol.KERNEL_RTOL_NUMPY


def make_motor():
    grain = Grain(0.035, 0.015, initial_height=0.12)
    motor = Motor([grain], chamber_inner_radius=0.037, chamber_length=0.14, nozzle_throat_radius=0.008,
                  nozzle_exit_radius=0.018)
    return motor


def table_propellant(path):
    return Propellant(1.1308, 0.04197, 1720.0, density=1879.0, interpolation_list=str(path))


def csv_file(tmp_path, rows, name="rates.csv"):
    path = tmp_path / name
    path.write_text('"Chamber Pressure (MPa)", "Burn Rate (mm/s)"\n' + "\n".join(f"{p},{r}" for p, r in rows) + "\n")
    return path


@pytest.fixture(scope="module")
def propellants(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tables")
    root = gc.REPO_ROOT / "data" / "burnrate"
    items = {
        "knsb3_24_rows": table_propellant(root / "KNSB3.csv"),
        "knsb_77_rows": table_propellant(root / "KNSB.csv"),
        "knsb2_1002_rows": table_propellant(root / "KNSB2.csv"),
        "two_rows": table_propellant(csv_file(folder, [(0.5, 3.0), (9.0, 12.0)], "two.csv")),
        "three_rows": table_propellant(csv_file(folder, [(0.5, 3.0), (3.0, 8.0), (9.0, 9.5)], "three.csv")),
        # rows out of order: interp1d sorts them, but its ends are held at the first and last row of the FILE
        "unsorted_rows": table_propellant(csv_file(folder, [(5.0, 8.0), (1.0, 3.0), (3.0, 6.0), (8.0, 9.0), (2.0, 4.5)],
                                                   "unsorted.csv")),
    }
    items["power_law"] = Propellant(1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36, burn_rate_n=0.32)
    return items


@pytest.fixture(scope="module")
def batch(propellants):
    motor = make_motor()
    return ProblemBatch.from_objects(motor, list(propellants.values()))


def pressure_grid(prop):
    interpolator = prop._burn_rate_interpolator
    if interpolator is None:
        return np.array([0.0, 1e5, 1e6, 3e6, 1e7])
    x = interpolator.x
    knots = x * 1e6
    mids = 0.5 * (knots[:-1] + knots[1:])
    nudges = np.concatenate([knots * (1 - 1e-12), knots * (1 + 1e-12)])
    outside = np.array([-1e5, 0.0, x[0] * 1e6 * 0.5, x[-1] * 1e6 * 1.5, 5e8])
    return np.sort(np.concatenate([knots, mids, nudges, outside]))


def test_the_packer_describes_each_table_with_its_length_and_held_ends(batch, propellants):
    a = batch.arrays
    names = list(propellants)

    assert a["burn_rate_mode"].tolist() == [1, 1, 1, 1, 1, 1, 0]
    assert a["rate_table_n"].tolist() == [24, 77, 1002, 2, 3, 5, 0]
    unsorted = names.index("unsorted_rows")
    assert (a["rate_table_below"][unsorted], a["rate_table_above"][unsorted]) == (8.0, 4.5)  # file order, not sorted
    assert np.isnan(a["burn_rate_a"][:6]).all() and a["burn_rate_a"][6] == 7.36
    assert (np.diff(a["rate_table_x"], axis=1) > 0).all()  # padding is finite and increasing
    n = int(a["rate_table_n"][names.index("knsb2_1002_rows")])
    assert n == 1002 and a["rate_table_x"].shape[1] == 1002


def test_the_table_burn_rate_matches_the_propellant_everywhere_including_the_held_ends(batch, propellants):
    V = {name: array[:, None] for name, array in batch.namespace(np).items()}
    flux = np.zeros((len(propellants), 1))
    for lane, (name, prop) in enumerate(propellants.items()):
        grid = pressure_grid(prop)
        P = {key: np.broadcast_to(value[lane], (len(grid),) + value[lane].shape).copy() for key, value in
             batch.namespace(np).items()}

        got = propellant_kernels.burn_rate(np, grid, np.zeros(len(grid)), P)

        expected = [prop.evaluate_burn_rate(float(p), 0.0) for p in grid]
        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=1e-300, err_msg=name)


def test_the_held_ends_follow_the_file_order_not_the_sorted_order(batch, propellants):
    lane = list(propellants).index("unsorted_rows")
    P = {key: value[lane : lane + 1] for key, value in batch.namespace(np).items()}

    low = propellant_kernels.burn_rate(np, np.array([1e3]), np.zeros(1), P)[0]
    high = propellant_kernels.burn_rate(np, np.array([5e8]), np.zeros(1), P)[0]

    assert low == pytest.approx(8.0 / 1000.0, rel=1e-15) and high == pytest.approx(4.5 / 1000.0, rel=1e-15)


def test_the_bisection_lookup_agrees_with_searchsorted_on_random_tables():
    rng = np.random.default_rng(3)
    lanes, k = 200, 37
    counts = rng.integers(2, k + 1, size=lanes)
    xs = np.zeros((lanes, k))
    for lane in range(lanes):
        n = counts[lane]
        xs[lane, :n] = np.sort(rng.uniform(0.0, 10.0, size=n))
        xs[lane, n:] = xs[lane, n - 1] + np.arange(1, k - n + 1)
    x = np.concatenate([rng.uniform(-1.0, 11.0, size=(lanes, 6)), xs[:, :2]], axis=1)

    got = tables.count_not_above(np, x, xs[:, None, :], counts[:, None])

    for lane in range(lanes):
        expected = np.searchsorted(xs[lane, : counts[lane]], x[lane], side="right")
        np.testing.assert_array_equal(got[lane], expected)
    j = tables.locate(np, x, xs[:, None, :], counts[:, None])
    assert (j >= 0).all() and (j <= counts[:, None] - 2).all()


def test_the_ratetable_family_of_the_corpus_matches_the_reference():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    cases = [c for c in corpus if c["family"] == "ratetable"]
    built = [gc.build_objects(c) for c in cases]
    lanes = ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                      [b[4] for b in built])

    results = backends.get_backend("cpu-vectorized").solve_burn(lanes).to_results()

    assert len(cases) == 18 and {"KNSB", "KNSB2", "KNSB3"} <= {c["propellant"]["interpolation_list"].split("/")[-1][:-4] for c in cases}
    bad = []
    for case, got in zip(cases, results):
        stored = reference[case["id"]]
        if got["status"]["termination_reason"] != stored["status"]["termination_reason"]:
            bad.append((case["id"], "reason"))
            continue
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.INTEGRAL_RTOL, atol=0):
                bad.append((case["id"], key))
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.GRID_SAMPLED_RTOL, atol=0):
                bad.append((case["id"], key))
        for t, s in zip(got["metrics"]["grain_burnout_times_s"], stored["metrics"]["grain_burnout_times_s"]):
            if not np.isclose(t, s, rtol=tol.TIME_RTOL, atol=0):
                bad.append((case["id"], "burnout"))
        table = gc.build_objects(case)[2]._burn_rate_interpolator
        assert got["provenance"]["resolved_inputs"]["burn_rate_table_mm_s"] == table.y.tolist()
    assert bad == []


# ---------------------------------------------------------------------------------------------------------
# Pressure-dependent thermochemistry (``Propellant.load_thermo_table``).

from types import SimpleNamespace  # noqa: E402

from solidpy import Burn  # noqa: E402
from solidpy.batch.kernels import nozzle, rhs  # noqa: E402

MACH_RTOL = 1e-8  # the exit Mach number is a cubic of k on a 33-point grid, not the root finder itself


@pytest.fixture(scope="module")
def thermo_cases():
    corpus = gc.load_corpus()["cases"]
    return [c for c in corpus if c["family"] == "thermotable"]


@pytest.fixture(scope="module")
def thermo_batch(thermo_cases):
    built = [gc.build_objects(c) for c in thermo_cases]
    return ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                     [b[4] for b in built])


def thermo_pressures(propellant):
    x = propellant._temperature_func.x
    mids = 0.5 * (x[:-1] + x[1:])
    return np.sort(np.concatenate([x, mids, x * (1 - 1e-12), x * (1 + 1e-12), [-1.0, 0.0, x[0] * 0.3, x[-1] * 3.0, 5e8]]))


def test_the_thermochemistry_lanes_are_packed_as_tables_with_a_mach_grid(thermo_batch):
    a = thermo_batch.arrays

    assert (a["thermo_mode"] == 1.0).all() and (a["mach_n"] == 33).all()
    assert sorted(set(a["thermo_n"].tolist())) == [2, 3, 4, 5, 6, 8, 10]  # linear (2 and 3 rows) and cubic tables
    assert (np.diff(a["thermo_x"], axis=1) > 0).all() and (np.diff(a["mach_x"], axis=1) > 0).all()
    assert (a["mach_below"] > 1.0).all() and (a["mach_above"] > 1.0).all()


def test_gas_properties_match_the_propellant_along_and_beyond_the_table(thermo_batch):
    P = thermo_batch.namespace(np)
    for lane, propellant in enumerate(thermo_batch.propellants):
        grid = thermo_pressures(propellant)
        view = {name: np.broadcast_to(array[lane], (len(grid),) + array[lane].shape).copy() for name, array in P.items()}

        t0, k = propellant_kernels.gas_properties(np, grid, view)

        eta_squared = thermo_batch.arrays["eta_c"][lane] ** 2
        expected_t = [propellant.Tc_at_pressure(max(float(p), 0.0)) * eta_squared for p in grid]
        expected_k = [propellant.get_gamma(max(float(p), 0.0)) for p in grid]
        np.testing.assert_allclose(t0, expected_t, rtol=RTOL, err_msg=f"Tc lane {lane}")
        np.testing.assert_allclose(k, expected_k, rtol=RTOL, err_msg=f"k lane {lane}")


def exit_mach_for(k, expansion_ratio):
    stand_in = SimpleNamespace(_parameters_at_pressure=lambda _p: (0.0, 0.0, 0.0, float(k), 0.0),
                               motor=SimpleNamespace(expansion_ratio=expansion_ratio), _exit_mach_cache={}, exit_mach=None)
    return Burn.evaluate_exit_mach(stand_in, None)


def test_the_exit_mach_grid_reproduces_the_root_finder_for_any_k_in_range(thermo_batch):
    P = thermo_batch.namespace(np)
    rng = np.random.default_rng(5)
    for lane, propellant in enumerate(thermo_batch.propellants):
        pressure = np.linspace(propellant._gamma_func.x[0], propellant._gamma_func.x[-1], 300)
        k_values = np.asarray(propellant._gamma_func(pressure))
        k = np.concatenate([k_values, rng.uniform(k_values.min(), k_values.max(), 200)])
        view = {name: np.broadcast_to(array[lane], (len(k),) + array[lane].shape).copy() for name, array in P.items()}

        got = propellant_kernels.exit_mach(np, k, view)

        expected = [exit_mach_for(v, thermo_batch.motors[lane].expansion_ratio) for v in k]
        np.testing.assert_allclose(got, expected, rtol=MACH_RTOL, err_msg=f"lane {lane}")


def test_thrust_with_a_k_table_matches_burn_at_pressures_around_the_choking_boundary(thermo_batch, thermo_cases):
    P = thermo_batch.namespace(np)
    for lane, case in enumerate(thermo_cases):
        grain, motor, propellant, environment, kwargs = gc.build_objects(case)
        burn = Burn(grain, motor, propellant, environment, eta_c=kwargs.get("eta_c", 1.0), eta_Cf=kwargs.get("eta_Cf", 1.0),
                    discharge_coefficient=kwargs.get("discharge_coefficient", 1.0))
        ambient = float(thermo_batch.arrays["ambient_pressure"][lane])
        pressure = np.sort(np.concatenate([ambient * np.array([0.5, 1.0, 1.0001, 1.5, 1.9, 2.0, 3.0, 10.0, 40.0]),
                                           np.linspace(propellant._gamma_func.x[0], propellant._gamma_func.x[-1], 25)]))
        view = {name: np.broadcast_to(array[lane], (len(pressure),) + array[lane].shape).copy() for name, array in P.items()}
        temperature = np.full(len(pressure), 0.9 * float(thermo_batch.arrays["source_temperature"][lane]))

        _, k = propellant_kernels.gas_properties(np, pressure, view)
        flow = nozzle.nozzle_mass_flow(np, pressure, temperature, k, view)
        ideal, momentum, pressure_thrust, total = nozzle.thrust_components(
            np, pressure, temperature, flow, k, propellant_kernels.exit_mach(np, k, view), view)

        for i, p in enumerate(pressure):
            scalar = burn.evaluate_thrust_components(float(p), chamber_temperature=float(temperature[i]))
            where = f"{case['id']} p={p!r}"
            np.testing.assert_allclose(flow[i], burn.evaluate_nozzle_mass_flow(float(p), chamber_temperature=float(temperature[i])),
                                       rtol=1e-11, atol=1e-300, err_msg="flow " + where)
            np.testing.assert_allclose(momentum[i], scalar["momentum_n"], rtol=MACH_RTOL, atol=1e-300, err_msg="momentum " + where)
            np.testing.assert_allclose(total[i], scalar["total_n"], rtol=MACH_RTOL, atol=1e-9 * abs(p) * motor.nozzle_exit_area,
                                       err_msg="total " + where)


def test_the_rhs_with_a_k_table_matches_the_scalar_solver_along_trajectories(thermo_cases):
    for case in thermo_cases[:5]:
        simulation = gc.simulate(case)
        raws = [simulation._burn_raw] + ([simulation._tail_raw] if simulation.tail_off_solution is not None else [])
        times, states = simulation._join_segments(raws)
        states = states.T
        batch = ProblemBatch.from_objects(simulation.motor, simulation.propellant, simulation.environment,
                                          {"eta_c": simulation.eta_c, "eta_Cf": simulation.eta_Cf,
                                           "discharge_coefficient": simulation.discharge_coefficient})
        view = {name: array[:, None] for name, array in batch.namespace(np).items()}
        picks = np.unique(np.linspace(0, len(states) - 1, 30).astype(int))
        derivative = rhs.conservative_rhs(np, states[picks][None], None, view, time=times[picks][None])

        for row, index in enumerate(picks):
            expected = np.asarray(simulation._conservative_rhs(times[index], states[index], None), dtype=float)
            scale = np.abs(expected) + np.max(np.abs(expected)) * 1e-3
            bad = np.flatnonzero(~(np.abs(derivative[0, row] - expected) <= 1e-7 * scale))
            assert bad.size == 0, (case["id"], index, bad.tolist(), derivative[0, row][bad], expected[bad])


def test_the_thermotable_family_matches_the_reference_and_keeps_the_legacy_status(thermo_batch, thermo_cases):
    reference = gc.load_reference()["records"]

    results = backends.get_backend("cpu-vectorized").solve_burn(thermo_batch).to_results()

    bad = []
    for case, got in zip(thermo_cases, results):
        stored = reference[case["id"]]
        assert got["status"]["termination_reason"] == "unsupported_thermochemistry" == stored["status"]["termination_reason"]
        assert got["status"]["completed"] is False and got["status"]["numerical_blowdown_completed"] is True
        assert got["provenance"]["thermochemistry_source"] == "pressure_table_legacy" and got["provenance"]["cea_used"] is False
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.INTEGRAL_RTOL, atol=0):
                bad.append((case["id"], key, got["metrics"][key], stored["metrics"][key]))
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.GRID_SAMPLED_RTOL, atol=0):
                bad.append((case["id"], key))
        for t, s in zip(got["metrics"]["grain_burnout_times_s"], stored["metrics"]["grain_burnout_times_s"]):
            if not np.isclose(t, s, rtol=tol.TIME_RTOL, atol=0):
                bad.append((case["id"], "burnout"))
    assert bad == []


def test_the_provider_hash_of_a_k_table_lane_is_the_scalar_hash(thermo_batch, thermo_cases):
    results = backends.get_backend("cpu-vectorized").solve_burn(thermo_batch.select([0, 7])).to_results()

    for lane, got in zip((0, 7), results):
        live = gc.simulate(thermo_cases[lane]).result
        assert got["provenance"]["physics_provider_hash"] == live["provenance"]["physics_provider_hash"]
        assert got["provenance"]["resolved_inputs"]["initial_gas_temperature_k"] == live["provenance"]["resolved_inputs"][
            "initial_gas_temperature_k"]
