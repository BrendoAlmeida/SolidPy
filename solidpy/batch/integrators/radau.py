# -*- coding: utf-8 -*-
"""Batched Radau IIA(5): the controller of ``solve_ivp(method="Radau")`` applied independently to every lane.

The method, the constants and the step controller are those of ``scipy.integrate.Radau``: the initial step
heuristic, the simplified Newton iteration on the transformed system (one real and one complex linear
solve), the error estimate with its second pass after a rejection, the two-step step-size predictor, the
reuse of the Jacobian and of the factorization across steps, and the extrapolation of the previous step's
dense output as the next Newton start. Each lane therefore takes the steps ``solve_ivp`` would take for it.
The Jacobian is tridiagonal (a chain of cells), given as its three diagonals, and the systems are factored by
the Thomas algorithm without pivoting, the matrices being diagonally dominant. The factorization is kept and
reused across steps exactly as scipy reuses its LU (including a factorization made for another step size), so
the iteration and the error estimate see the same matrix. Where scipy pivots, a solve differs by about
``cond * eps`` relative. The state of a padded cell is exactly zero and decoupled.

``integrate`` advances one interval ``[0, t_bound]`` for every lane and also returns the trapezoid of an
observed scalar over the accepted points and its maximum over them, which is how ``simulate_thermal_ablation``
integrates the heat flux. The functions are pure over the array namespace ``xp``; the loop comes from a
``Driver`` (``integrators.solver``), so NumPy and a compiled driver run the same code.
"""

import numpy as np

from . import dop853

# The Butcher tableau of Radau IIA(5) and the transformation matrices of ``scipy.integrate.Radau``, copied so that no private
# scipy module is imported; ``tests/test_batch_radau.py`` compares every one of them with the installed scipy's.
S6 = 6**0.5
C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
MU_COMPLEX = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3)) - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6))
T = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0]])
TI = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
TI_REAL = TI[0]
TI_COMPLEX = TI[1] + 1j * TI[2]
P = np.array([
    [13 / 3 + 7 * S6 / 3, -23 / 3 - 22 * S6 / 3, 10 / 3 + 5 * S6],
    [13 / 3 - 7 * S6 / 3, -23 / 3 + 22 * S6 / 3, 10 / 3 - 5 * S6],
    [1 / 3, -8 / 3, 10 / 3]])
NEWTON_MAXITER = 6
MIN_FACTOR = 0.2
MAX_FACTOR = 10
ERROR_ESTIMATOR_ORDER = 3
EPS = float(np.finfo(float).eps)

#: Attempts one interval may take before its lane is declared failed (scipy has no limit; the lane is rerun there).
MAX_ATTEMPTS = 400


def newton_tolerance(rtol):
    """``Radau.newton_tol``."""
    return max(10.0 * EPS / rtol, min(0.03, rtol**0.5))


def predict_factor(xp, h_abs, h_old, error_norm, error_norm_old):
    """``predict_factor`` per lane; ``NaN`` in ``h_old`` or ``error_norm_old`` stands for scipy's ``None``."""
    one_step = xp.isnan(h_old) | xp.isnan(error_norm_old) | (error_norm == 0.0)
    positive = xp.where(error_norm > 0.0, error_norm, 1.0)
    multiplier = xp.where(one_step, 1.0, h_abs / xp.where(one_step, 1.0, h_old) * (error_norm_old / positive) ** 0.25)
    factor = xp.minimum(1.0, multiplier) * positive**-0.25
    return xp.where(error_norm == 0.0, xp.inf, factor)  # min(1, 1) * 0**-0.25 is inf, as in scipy


def _select(xp, mask, new, old):
    """``where`` with a per-lane mask broadcast over the trailing axes of ``new``."""
    return xp.where(mask.reshape(mask.shape + (1,) * (new.ndim - mask.ndim)), new, old)


def _lane_norm(xp, x, count):
    """scipy's ``norm``: the root mean square of all entries of a lane, ``count`` being their real number."""
    axes = tuple(range(1, x.ndim))
    return xp.sqrt(xp.sum(x * x, axis=axes)) / count**0.5


def tridiagonal_factor(driver, lower, diag, upper):
    """LU without pivoting of tridiagonal matrices, one per row: returns the multipliers ``[B, n-1]`` and the pivots ``[B, n]``.

    ``lower[:, i]`` is the entry below the diagonal in column ``i``, ``upper[:, i]`` the entry above it in column ``i + 1``.
    """
    xp = driver.xp

    def body(previous, xs):
        below, d, above_previous = xs
        multiplier = below / previous
        pivot = d - multiplier * above_previous
        return pivot, (multiplier, pivot)

    first = diag[:, 0]
    _, (multiplier, pivot) = driver.scan(body, first, (lower.T, diag[:, 1:].T, upper.T))
    return multiplier.T, xp.concatenate([first[:, None], pivot.T], axis=1)


def tridiagonal_solve(driver, multiplier, pivot, upper, rhs):
    """Solve with the factors of ``tridiagonal_factor``."""
    xp = driver.xp

    def forward(previous, xs):
        factor, b = xs
        y = b - factor * previous
        return y, y

    first = rhs[:, 0]
    _, rest = driver.scan(forward, first, (multiplier.T, rhs[:, 1:].T))
    y = xp.concatenate([first[:, None], rest.T], axis=1)

    def backward(following, xs):
        y_i, pivot_i, above = xs
        x = (y_i - above * following) / pivot_i
        return x, x

    last = y[:, -1] / pivot[:, -1]
    _, head = driver.scan(backward, last, (y[:, :-1].T, pivot[:, :-1].T, upper.T), reverse=True)
    return xp.concatenate([head.T, last[:, None]], axis=1)


def _factor_pair(driver, jl, jd, ju, h):
    """Factors of ``MU_REAL / h * I - J`` and ``MU_COMPLEX / h * I - J``, stacked on the lane axis (real one first).

    Both are factored in one pass over the cells (the real system as a complex one), because the cost of a pass is
    its number of sequential steps, not its width.
    """
    xp = driver.xp
    lanes = jd.shape[0]
    mu = xp.concatenate([xp.full(lanes, MU_REAL + 0j), xp.full(lanes, MU_COMPLEX)]) / xp.concatenate([h, h])
    diag = mu[:, None] - xp.concatenate([jd, jd]).astype(complex)
    lower = -xp.concatenate([jl, jl]).astype(complex)
    upper = -xp.concatenate([ju, ju]).astype(complex)
    multiplier, pivot = tridiagonal_factor(driver, lower, diag, upper)
    return multiplier, pivot, upper


def _solve_pair(driver, factors, rhs_real, rhs_complex):
    multiplier, pivot, upper = factors
    solved = tridiagonal_solve(driver, multiplier, pivot, upper, driver.xp.concatenate([rhs_real.astype(complex), rhs_complex]))
    lanes = rhs_real.shape[0]
    return solved[:lanes].real, solved[lanes:]


def _solve_real(driver, factors, rhs):
    multiplier, pivot, upper = factors
    lanes = rhs.shape[0]
    return tridiagonal_solve(driver, multiplier[:lanes], pivot[:lanes], upper[:lanes], rhs.astype(complex)).real


def newton(driver, fun, t, y, h, Z0, scale, tol, factors, size, run):
    """``solve_collocation_system`` for every lane in ``run``.

    ``factors`` are those of ``_factor_pair``. Returns ``(converged, n_iter, Z, rate)``: ``Z`` has shape ``[B, 3, n]``
    and ``rate`` is NaN where scipy's is ``None``. ``size`` is the real number of state components of each lane.
    """
    xp = driver.xp
    lanes = y.shape[0]
    m_real = MU_REAL / h
    m_complex = MU_COMPLEX / h
    stage_times = [t + h * C[i] for i in range(3)]
    ti = xp.asarray(TI)
    tm = xp.asarray(T)
    ti_real = xp.asarray(TI_REAL)
    ti_complex = xp.asarray(TI_COMPLEX)
    count = 3.0 * size

    state = {
        "k": xp.asarray(0),
        "running": run,
        "converged": xp.zeros(lanes, dtype=bool),
        "n_iter": xp.zeros(lanes, dtype=xp.asarray(0).dtype),
        "W": xp.einsum("ij,bjn->bin", ti, Z0),
        "Z": Z0,
        "rate": xp.full(lanes, xp.nan),
        "dW_norm_old": xp.full(lanes, xp.nan),
    }

    def body(s):
        k = s["k"]
        running = s["running"]
        Z = s["Z"]
        F = xp.stack([fun(stage_times[i], y + Z[:, i, :]) for i in range(3)], axis=1)
        finite = xp.all(xp.isfinite(F), axis=(1, 2))
        W = s["W"]
        f_real = xp.einsum("bjn,j->bn", F, ti_real) - m_real[:, None] * W[:, 0, :]
        f_complex = xp.einsum("bjn,j->bn", F, ti_complex) - m_complex[:, None] * (W[:, 1, :] + 1j * W[:, 2, :])
        dW_real, dW_complex = _solve_pair(driver, factors, f_real, f_complex)
        dW = xp.stack([dW_real, dW_complex.real, dW_complex.imag], axis=1)
        dW_norm = _lane_norm(xp, dW / scale[:, None, :], count)
        has_rate = k > 0
        rate_now = dW_norm / s["dW_norm_old"]
        slow = has_rate & ((rate_now >= 1.0) | (rate_now ** (NEWTON_MAXITER - k) / (1.0 - rate_now) * dW_norm > tol))
        stop_bad = running & (~finite | slow)
        step = running & ~stop_bad
        W_new = _select(xp, step, W + dW, W)
        Z_new = _select(xp, step, xp.einsum("ij,bjn->bin", tm, W_new), Z)
        done_now = step & ((dW_norm == 0.0) | (has_rate & (rate_now / (1.0 - rate_now) * dW_norm < tol)))
        return {
            "k": k + 1,
            "running": step & ~done_now,
            "converged": s["converged"] | done_now,
            "n_iter": xp.where(running, k + 1, s["n_iter"]),
            "W": W_new,
            "Z": Z_new,
            "rate": xp.where(running & finite & has_rate, rate_now, s["rate"]),
            "dW_norm_old": xp.where(step, dW_norm, s["dW_norm_old"]),
        }

    final = driver.loop(lambda s: xp.any(s["running"]) & (s["k"] < NEWTON_MAXITER), body, state)
    return final["converged"], final["n_iter"], final["Z"], final["rate"]


def _dense_start(xp, c, t, h, y):
    """The Newton start ``sol(t + h * C) - y`` from the previous step's dense output, zero before the first step."""
    sol_h = xp.where(c["has_sol"], c["sol_h"], 1.0)
    x = (t[:, None] + h[:, None] * xp.asarray(C)[None, :] - c["sol_t"][:, None]) / sol_h[:, None]
    x2 = x * x
    powers = xp.stack([x, x2, x2 * x], axis=-1)
    values = xp.einsum("bnk,bsk->bsn", c["Q"], powers) + c["sol_y"][:, None, :]
    return xp.where(c["has_sol"][:, None, None], values - y[:, None, :], 0.0)


def integrate(driver, fun, jac, observe, y0, t_bound, active, size, rtol, atol, max_attempts=MAX_ATTEMPTS):
    """Integrate ``dy/dt = fun(t, y)`` from 0 to ``t_bound`` for every lane in ``active``.

    ``fun(t, y)`` takes ``t`` of shape ``[B]`` and ``y`` of shape ``[B, n]``; ``jac(t, y, f)`` returns the
    tridiagonal Jacobian as ``(lower [B, n-1], diag [B, n], upper [B, n-1])``; ``observe(y)`` is a scalar per lane whose
    trapezoid over the accepted points (including the start) and maximum over them are returned. ``size`` is the real
    state size of each lane. The state of a padded cell must have a zero derivative and zero Jacobian entries.

    Returns a dict with the final ``y``, the trapezoid ``integral`` and ``peak`` of the observed quantity,
    ``steps`` (accepted), ``attempts`` and ``failed`` (the step became too small, the attempt limit was hit
    or the state stopped being finite: the lane must be rerun where scipy decides what happens).
    Lanes outside ``active`` are returned untouched.
    """
    xp = driver.xp
    lanes, n = y0.shape
    float_zero = xp.zeros(lanes)
    int_zero = xp.zeros(lanes, dtype=xp.asarray(0).dtype)
    false = xp.zeros(lanes, dtype=bool)
    t_bound = xp.where(active, t_bound, 1.0)
    rtol_lane = xp.full(lanes, float(rtol))
    atol_lane = xp.full(lanes, float(atol))
    tol = newton_tolerance(rtol)
    e_coefficients = xp.asarray(E)
    p_matrix = xp.asarray(P)

    f0 = fun(float_zero, y0)
    h0 = dop853.initial_step(xp, fun, float_zero, y0, f0, t_bound, xp.inf, rtol_lane, atol_lane, size,
                             ERROR_ESTIMATOR_ORDER)
    g0 = observe(y0)
    nan = xp.full(lanes, xp.nan)
    jl0, jd0, ju0 = jac(float_zero, y0, f0)
    carry = {
        "t": float_zero, "y": y0, "f": f0, "Jl": jl0, "Jd": jd0, "Ju": ju0,
        "h_prop": h0, "h_old": nan, "e_old": nan, "h_cur": h0, "h_old_loc": nan, "e_old_loc": nan,
        "new_step": ~false, "rejected": false, "current_jac": ~false,
        "fac_mult": xp.zeros((2 * lanes, n - 1), dtype=complex), "fac_piv": xp.ones((2 * lanes, n), dtype=complex),
        "fac_up": xp.zeros((2 * lanes, n - 1), dtype=complex), "lu_valid": false,
        "has_sol": false, "Q": xp.zeros((lanes, n, 3)), "sol_t": float_zero, "sol_h": float_zero + 1.0,
        "sol_y": y0,
        "g_prev": g0, "integral": float_zero, "peak": g0,
        "steps": int_zero, "attempts": int_zero, "done": ~active, "failed": false,
    }

    def cond(c):
        return xp.any(~(c["done"] | c["failed"]))

    def body(c):
        t, y, f = c["t"], c["y"], c["f"]
        run = ~(c["done"] | c["failed"])
        new_step = c["new_step"]
        min_step = 10.0 * xp.abs(xp.nextafter(t, xp.inf) - t)
        clamped = new_step & (c["h_prop"] < min_step)
        h_cur = xp.where(new_step, xp.where(clamped, min_step, c["h_prop"]), c["h_cur"])
        h_old = xp.where(new_step, xp.where(clamped, xp.nan, c["h_old"]), c["h_old_loc"])
        e_old = xp.where(new_step, xp.where(clamped, xp.nan, c["e_old"]), c["e_old_loc"])
        rejected = c["rejected"] & ~new_step
        too_small = run & (h_cur < min_step)
        go = run & ~too_small
        t_new = xp.minimum(t + h_cur, t_bound)
        h = t_new - t
        h_abs = xp.abs(h)
        h_use = xp.where(go & (h > 0.0), h, 1.0)

        scale = atol_lane[:, None] + xp.abs(y) * rtol_lane[:, None]
        need = go & ~c["lu_valid"]

        def factorize(operand):
            needed, jl, jd, ju, step, old = operand
            new = _factor_pair(driver, jl, jd, ju, step)
            both = xp.concatenate([needed, needed])
            return tuple(_select(xp, both, a, b) for a, b in zip(new, old))

        factors = driver.branch(
            xp.any(need), factorize, lambda operand: operand[5],
            (need, c["Jl"], c["Jd"], c["Ju"], h_use, (c["fac_mult"], c["fac_piv"], c["fac_up"])),
        )
        converged, n_iter, Z, rate = newton(
            driver, fun, t, y, h_use, _dense_start(xp, c, t, h_use, y), scale, tol, factors, size, go
        )
        converged = go & converged

        y_new = y + Z[:, 2, :]
        ZE = xp.einsum("bjn,j->bn", Z, e_coefficients) / h_use[:, None]
        error = _solve_real(driver, factors, f + ZE)
        error_scale = atol_lane[:, None] + xp.maximum(xp.abs(y), xp.abs(y_new)) * rtol_lane[:, None]
        error_norm = _lane_norm(xp, error / error_scale, size)
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
        second = converged & rejected & (error_norm > 1.0)
        refined = driver.branch(
            xp.any(second),
            lambda operand: _solve_real(driver, factors, fun(t, y + error) + ZE),
            lambda operand: error,
            None,
        )
        error_norm = xp.where(second, _lane_norm(xp, refined / error_scale, size), error_norm)

        over = error_norm > 1.0  # a NaN norm is accepted, as in scipy; the NaN state then fails the lane below
        reject = converged & over
        accept = converged & ~over
        not_converged = go & ~converged
        refresh = not_converged & ~c["current_jac"]
        halve = not_converged & c["current_jac"]

        factor = predict_factor(xp, h_abs, h_old, error_norm, e_old)
        h_next_try = xp.where(
            halve, 0.5 * h_abs, xp.where(reject, h_abs * xp.maximum(MIN_FACTOR, safety * factor), h_cur)
        )

        f_new = fun(t_new, y_new)
        refit = (n_iter > 2) & (rate > 1e-3)
        grown = xp.minimum(MAX_FACTOR, safety * factor)
        keep = ~refit & (grown < 1.2)
        grown = xp.where(keep, 1.0, grown)
        recompute = accept & refit

        def refresh_jacobian(operand):
            after, here = operand
            fresh_after = jac(t_new, y_new, f_new)
            fresh_here = jac(t, y, f)
            return tuple(
                _select(xp, after, a, _select(xp, here, b, old))
                for a, b, old in zip(fresh_after, fresh_here, (c["Jl"], c["Jd"], c["Ju"]))
            )

        jl, jd, ju = driver.branch(
            xp.any(recompute | refresh), refresh_jacobian, lambda operand: (c["Jl"], c["Jd"], c["Ju"]),
            (recompute, refresh),
        )
        g_new = observe(y_new)

        out = dict(c)
        out["t"] = xp.where(accept, t_new, t)
        out["y"] = _select(xp, accept, y_new, y)
        out["f"] = _select(xp, accept, f_new, f)
        out["Jl"], out["Jd"], out["Ju"] = jl, jd, ju
        out["current_jac"] = xp.where(accept, recompute, xp.where(refresh, True, c["current_jac"]))
        out["fac_mult"], out["fac_piv"], out["fac_up"] = factors
        out["lu_valid"] = xp.where(go, accept & keep, c["lu_valid"])
        out["h_prop"] = xp.where(accept, h_abs * grown, c["h_prop"])
        out["h_old"] = xp.where(accept, c["h_prop"], c["h_old"])
        out["e_old"] = xp.where(accept, error_norm, c["e_old"])
        out["h_cur"] = xp.where(go, h_next_try, c["h_cur"])
        out["h_old_loc"] = xp.where(go, h_old, c["h_old_loc"])
        out["e_old_loc"] = xp.where(go, e_old, c["e_old_loc"])
        out["rejected"] = xp.where(go, ~accept & (rejected | reject), c["rejected"])
        out["new_step"] = xp.where(go, accept, c["new_step"])
        out["has_sol"] = c["has_sol"] | accept
        out["Q"] = _select(xp, accept, xp.einsum("bjn,jk->bnk", Z, p_matrix), c["Q"])
        out["sol_t"] = xp.where(accept, t, c["sol_t"])
        out["sol_h"] = xp.where(accept, h_use, c["sol_h"])
        out["sol_y"] = _select(xp, accept, y, c["sol_y"])
        out["g_prev"] = xp.where(accept, g_new, c["g_prev"])
        out["integral"] = c["integral"] + xp.where(accept, 0.5 * (c["g_prev"] + g_new) * h_use, 0.0)
        out["peak"] = xp.where(accept, xp.maximum(c["peak"], g_new), c["peak"])
        out["steps"] = c["steps"] + accept
        attempts = c["attempts"] + go
        out["attempts"] = attempts
        finished = accept & (t_new >= t_bound)
        broken = accept & ~xp.all(xp.isfinite(y_new), axis=1)
        out["done"] = c["done"] | finished | broken
        out["failed"] = c["failed"] | too_small | broken | (go & ~finished & (attempts >= max_attempts))
        return out

    final = driver.loop(cond, body, carry)
    keys = ("y", "integral", "peak", "steps", "attempts", "failed")
    return {key: final[key] for key in keys}
