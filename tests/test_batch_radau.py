"""The batched Radau IIA(5) against ``solve_ivp(method="Radau")`` on the wall conduction ODE of the thermal model."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.integrate._ivp import radau as scipy_radau

from solidpy.batch.integrators import radau, solver
from solidpy.batch.kernels import thermal as thermal_kernels
from solidpy.Multiphysics import _build_wall_conduction_operator

RTOL, ATOL = 1e-5, 1e-6
FLAME_K, GAMMA = 1600.0, 1.136
STAGNATION = (GAMMA + 1.0) / 2.0
RECOVERY_K = FLAME_K * (1.0 + 0.89 * (GAMMA - 1.0) / 2.0) / (1.0 + (GAMMA - 1.0) / 2.0)

MATERIALS = {  # conductivity [W/mK], density * heat capacity [J/m3K]
    "steel": (16.0, 7850.0 * 520.0),
    "aluminium": (167.0, 2700.0 * 900.0),
    "liner": (0.25, 1100.0 * 1600.0),
}


def make_wall(layers, bartz_base, dt, start_k=298.15):
    """One lane: the operator of a wall of ``layers`` = [(thickness, material)] cells and its heat flux coefficient."""
    dx = np.asarray([thickness for thickness, _ in layers], dtype=float)
    k = np.asarray([MATERIALS[name][0] for _, name in layers])
    rho_cp = np.asarray([MATERIALS[name][1] for _, name in layers])
    jacobian, source, inner = _build_wall_conduction_operator(dx, k, rho_cp, 18.0, 298.15)
    return {"A": jacobian.toarray(), "sparse": jacobian, "source": source, "e0": float(inner[0]), "n": len(dx),
            "bartz": bartz_base, "dt": dt, "y0": np.full(len(dx), start_k)}


def scalar_run(lane):
    """The scalar solve, as ``simulate_thermal_ablation`` does it, and the samples the heat integral uses."""
    def flux(t_hot):
        return thermal_kernels.flux(np, np.float64(t_hot), lane["bartz"], RECOVERY_K, FLAME_K, STAGNATION)

    def fun(_t, y):
        out = lane["A"] @ y + lane["source"]
        out[0] += lane["e0"] * flux(y[0])[0]
        return out

    def jac(_t, y):
        out = lane["A"].copy()
        out[0, 0] += lane["e0"] * flux(y[0])[1]
        return out

    solution = solve_ivp(fun, (0.0, lane["dt"]), lane["y0"], method="Radau", jac=jac, atol=ATOL, rtol=RTOL)
    samples = np.asarray([flux(t_hot)[0] for t_hot in solution.y[0]])
    return solution, float(np.sum(0.5 * (samples[1:] + samples[:-1]) * np.diff(solution.t))), float(samples.max())


def pack(lanes, nodes=None):
    """The padded arrays of ``lanes`` and the callables ``radau.integrate`` takes (NumPy namespace)."""
    nodes = nodes or max(lane["n"] for lane in lanes)
    count = len(lanes)
    A = np.zeros((count, nodes, nodes))
    source = np.zeros((count, nodes))
    y0 = np.zeros((count, nodes))
    for i, lane in enumerate(lanes):
        n = lane["n"]
        A[i, :n, :n], source[i, :n], y0[i, :n] = lane["A"], lane["source"], lane["y0"]
    e0 = np.asarray([lane["e0"] for lane in lanes])
    bartz = np.asarray([lane["bartz"] for lane in lanes])
    unit = np.zeros(nodes)
    unit[0] = 1.0

    def flux(t_hot):
        return thermal_kernels.flux(np, t_hot, bartz, RECOVERY_K, FLAME_K, STAGNATION)

    def fun(_t, y):
        return np.einsum("bij,bj->bi", A, y) + source + (e0 * flux(y[:, 0])[0])[:, None] * unit

    def jac(_t, y, _f):
        return A + (e0 * flux(y[:, 0])[1])[:, None, None] * np.outer(unit, unit)

    return {"fun": fun, "jac": jac, "observe": lambda y: flux(y[:, 0])[0], "y0": y0,
            "t_bound": np.asarray([lane["dt"] for lane in lanes]),
            "size": np.asarray([float(lane["n"]) for lane in lanes])}


def run(problem, active=None, **kwargs):
    active = np.ones(len(problem["y0"]), dtype=bool) if active is None else active
    return radau.integrate(solver.numpy_driver(), problem["fun"], problem["jac"], problem["observe"], problem["y0"],
                           problem["t_bound"], active, problem["size"], RTOL, ATOL, **kwargs)


STEEL4 = [(0.001, "steel")] * 4
LINED = [(0.001, "liner")] * 2 + [(0.001, "steel")] * 4
ALU_LINED = [(0.001, "liner")] * 3 + [(0.001, "aluminium")] * 4
THICK = [(0.001, "steel")] * 12
LANES = [
    make_wall(STEEL4, 6000.0, 0.01),
    make_wall(LINED, 6000.0, 0.03),
    make_wall(ALU_LINED, 9000.0, 0.03, start_k=250.0),
    make_wall(THICK, 4000.0, 0.2),
    make_wall(LINED, 9000.0, 1.5),  # long enough for several steps and rejections
    make_wall(ALU_LINED, 2000.0, 4.0, start_k=400.0),
]


def test_the_constants_are_the_ones_of_the_scipy_solver():
    assert radau.NEWTON_MAXITER == scipy_radau.NEWTON_MAXITER == 6
    assert (radau.MIN_FACTOR, radau.MAX_FACTOR) == (0.2, 10)
    np.testing.assert_array_equal(radau.TI, scipy_radau.TI)
    np.testing.assert_allclose(radau.T @ radau.TI, np.eye(3), atol=1e-12)  # the two transformations are inverses
    assert radau.TI_COMPLEX.dtype == complex and radau.TI_REAL.shape == (3,)
    np.testing.assert_allclose(radau.C, [(4 - 6**0.5) / 10, (4 + 6**0.5) / 10, 1.0])
    assert radau.newton_tolerance(1e-5) == pytest.approx(max(10 * np.finfo(float).eps / 1e-5, min(0.03, 1e-5**0.5)))


def test_every_lane_takes_the_steps_scipy_takes_and_ends_where_it_ends():
    out = run(pack(LANES))

    assert not out["failed"].any()
    for i, lane in enumerate(LANES):
        solution, integral, peak = scalar_run(lane)
        assert int(out["steps"][i]) == len(solution.t) - 1, i
        np.testing.assert_allclose(out["y"][i, : lane["n"]], solution.y[:, -1], rtol=1e-11, atol=0.0, err_msg=str(i))
        assert out["integral"][i] == pytest.approx(integral, rel=1e-11), i
        assert out["peak"][i] == pytest.approx(peak, rel=1e-12), i
        assert not out["y"][i, lane["n"] :].any()  # padded nodes stay at zero


def test_the_cases_exercise_several_steps_and_rejections():
    out = run(pack(LANES))

    assert out["steps"].max() >= 5
    assert (out["attempts"] > out["steps"]).any()  # at least one rejected step or Jacobian refresh


def test_a_lane_gives_the_same_result_alone_with_other_lanes_and_with_padding():
    together = run(pack(LANES))
    padded = run(pack([LANES[1]], nodes=16))

    alone = run(pack([LANES[1]]))
    n = LANES[1]["n"]
    np.testing.assert_allclose(alone["y"][0], together["y"][1, :n], rtol=1e-12)
    np.testing.assert_allclose(padded["y"][0, :n], alone["y"][0], rtol=1e-12)
    assert alone["integral"][0] == pytest.approx(together["integral"][1], rel=1e-12)
    assert alone["steps"][0] == together["steps"][1] == padded["steps"][0]


def test_lanes_that_are_not_active_are_returned_untouched():
    problem = pack(LANES[:3])
    active = np.array([True, False, True])

    out = run(problem, active)

    np.testing.assert_array_equal(out["y"][1], problem["y0"][1])
    assert out["steps"][1] == 0 and not out["failed"][1]
    full = run(problem)
    np.testing.assert_allclose(out["y"][[0, 2]], full["y"][[0, 2]], rtol=1e-12)


def test_a_lane_that_exhausts_its_attempts_is_flagged_failed_and_the_others_are_not():
    out = run(pack([LANES[0], LANES[4]]), max_attempts=2)

    assert out["failed"].tolist() == [False, True]


def test_the_result_is_within_the_tolerance_of_a_tight_reference():
    """The solve at the scalar tolerance is as close to a 1e-12 solve as the tolerance says."""
    problem = pack([LANES[4]])
    base = run(problem)
    solution = solve_ivp(lambda t, y: problem["fun"](np.array([t]), y[None])[0], (0.0, LANES[4]["dt"]), problem["y0"][0],
                         method="Radau", jac=lambda t, y: problem["jac"](np.array([t]), y[None], None)[0],
                         atol=1e-12, rtol=1e-12)

    np.testing.assert_allclose(base["y"][0], solution.y[:, -1], rtol=1e-4)


def random_walls(count, seed=7):
    """Walls of one to two materials, 2 to 10 cells, with the heat flux coefficient and the interval varied widely."""
    rng = np.random.default_rng(seed)
    walls = []
    for _ in range(count):
        layers = []
        for _ in range(rng.integers(1, 3)):
            layers += [(float(rng.choice([0.0005, 0.001, 0.002])), str(rng.choice(list(MATERIALS))))] * int(rng.integers(2, 6))
        walls.append(make_wall(layers, float(10 ** rng.uniform(2.5, 4.5)), float(10 ** rng.uniform(-2, 1.0)),
                               start_k=float(rng.uniform(200, 600))))
    return walls


def test_random_walls_follow_scipy_step_for_step():
    """Lanes of 1 to 43 steps, a third of them with rejected steps: the same number of steps and the same integral."""
    walls = random_walls(160)
    out = run(pack(walls))

    assert not out["failed"].any() and (out["attempts"] > out["steps"]).sum() > 20
    for i, wall in enumerate(walls):
        solution, integral, peak = scalar_run(wall)
        assert int(out["steps"][i]) == len(solution.t) - 1, i
        assert out["integral"][i] == pytest.approx(integral, rel=1e-9), i
        assert out["peak"][i] == pytest.approx(peak, rel=1e-9), i
