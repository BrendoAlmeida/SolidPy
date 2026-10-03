# -*- coding: utf-8 -*-
"""Batched Radau IIA(5): the controller of ``solve_ivp(method="Radau")`` applied independently to every lane.

The method, the constants and the step controller are those of ``scipy.integrate.Radau``: the initial step
heuristic, the simplified Newton iteration on the transformed system (one real and one complex linear
solve), the error estimate with its second pass after a rejection, the two-step step-size predictor, the
reuse of the Jacobian and of the factorization across steps, and the extrapolation of the previous step's
dense output as the next Newton start. Each lane therefore takes the steps ``solve_ivp`` would take for it.
Two differences remain, neither of which changes what a step computes:

* the factorization is an explicit inverse, applied by a matrix product (batched and the same on every
  backend), where scipy keeps an LU; a solve differs by about ``cond * eps`` relative;
* the Jacobian is a dense ``[B, n, n]`` array and the state of a padded lane is exactly zero.

``integrate`` advances one interval ``[0, t_bound]`` for every lane and also returns the trapezoid of an
observed scalar over the accepted points and its maximum over them, which is how ``simulate_thermal_ablation``
integrates the heat flux. The functions are pure over the array namespace ``xp``; the loop comes from a
``Driver`` (``integrators.solver``), so NumPy and a compiled driver run the same code.
"""

import numpy as np

# The Butcher tableau and the transformation matrices live in a private scipy module;
# ``tests/test_batch_radau.py`` checks the layout and the values the controller needs.
from scipy.integrate._ivp import radau as _radau

from . import dop853

C = np.asarray(_radau.C)
E = np.asarray(_radau.E)
MU_REAL = float(_radau.MU_REAL)
MU_COMPLEX = complex(_radau.MU_COMPLEX)
T = np.asarray(_radau.T)
TI = np.asarray(_radau.TI)
TI_REAL = np.asarray(_radau.TI_REAL)
TI_COMPLEX = np.asarray(_radau.TI_COMPLEX)
P = np.asarray(_radau.P)
NEWTON_MAXITER = int(_radau.NEWTON_MAXITER)
MIN_FACTOR = float(_radau.MIN_FACTOR)
MAX_FACTOR = float(_radau.MAX_FACTOR)
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


def newton(driver, fun, t, y, h, Z0, scale, tol, inv_real, inv_complex, size, run):
    """``solve_collocation_system`` for every lane in ``run``.

    Returns ``(converged, n_iter, Z, rate)``: ``Z`` has shape ``[B, 3, n]`` and ``rate`` is NaN where scipy's is ``None``.
    ``size`` is the real number of state components of each lane.
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
        dW_real = xp.einsum("bij,bj->bi", inv_real, f_real)
        dW_complex = xp.einsum("bij,bj->bi", inv_complex, f_complex)
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
    ``[B, n, n]`` Jacobian; ``observe(y)`` is a scalar per lane whose trapezoid over the accepted points
    (including the start) and maximum over them are returned. ``size`` is the real state size of each lane.
    The state of a padded component must have a zero derivative and zero rows and columns in the Jacobian.

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
    eye = xp.eye(n)
    c_stage = xp.asarray(C)
    e_coefficients = xp.asarray(E)
    p_matrix = xp.asarray(P)

    f0 = fun(float_zero, y0)
    h0 = dop853.initial_step(xp, fun, float_zero, y0, f0, t_bound, xp.inf, rtol_lane, atol_lane, size,
                             ERROR_ESTIMATOR_ORDER)
    g0 = observe(y0)
    nan = xp.full(lanes, xp.nan)
    carry = {
        "t": float_zero, "y": y0, "f": f0, "J": jac(float_zero, y0, f0),
        "h_prop": h0, "h_old": nan, "e_old": nan, "h_cur": h0, "h_old_loc": nan, "e_old_loc": nan,
        "new_step": ~false, "rejected": false, "current_jac": ~false,
        "inv_real": xp.zeros((lanes, n, n)), "inv_complex": xp.zeros((lanes, n, n), dtype=complex), "lu_valid": false,
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
            needed, jacobian, step, old_real, old_complex = operand
            new_real = xp.linalg.inv((MU_REAL / step)[:, None, None] * eye - jacobian)
            new_complex = xp.linalg.inv((MU_COMPLEX / step)[:, None, None] * eye - jacobian)
            return _select(xp, needed, new_real, old_real), _select(xp, needed, new_complex, old_complex)

        inv_real, inv_complex = driver.branch(
            xp.any(need), factorize, lambda operand: (operand[3], operand[4]),
            (need, c["J"], h_use, c["inv_real"], c["inv_complex"]),
        )
        converged, n_iter, Z, rate = newton(
            driver, fun, t, y, h_use, _dense_start(xp, c, t, h_use, y), scale, tol, inv_real, inv_complex, size, go
        )
        converged = go & converged

        y_new = y + Z[:, 2, :]
        ZE = xp.einsum("bjn,j->bn", Z, e_coefficients) / h_use[:, None]
        error = xp.einsum("bij,bj->bi", inv_real, f + ZE)
        error_scale = atol_lane[:, None] + xp.maximum(xp.abs(y), xp.abs(y_new)) * rtol_lane[:, None]
        error_norm = _lane_norm(xp, error / error_scale, size)
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
        second = converged & rejected & (error_norm > 1.0)
        refined = xp.einsum("bij,bj->bi", inv_real, fun(t, y + error) + ZE)
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
        J = driver.branch(
            xp.any(recompute | refresh),
            lambda operand: _select(xp, operand[0], jac(t_new, y_new, f_new),
                                    _select(xp, operand[1], jac(t, y, f), c["J"])),
            lambda operand: c["J"],
            (recompute, refresh),
        )
        g_new = observe(y_new)

        out = dict(c)
        out["t"] = xp.where(accept, t_new, t)
        out["y"] = _select(xp, accept, y_new, y)
        out["f"] = _select(xp, accept, f_new, f)
        out["J"] = J
        out["current_jac"] = xp.where(accept, recompute, xp.where(refresh, True, c["current_jac"]))
        out["inv_real"], out["inv_complex"] = inv_real, inv_complex
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
