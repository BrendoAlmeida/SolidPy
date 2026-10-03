# -*- coding: utf-8 -*-
"""Batched DOP853: the scipy step controller applied independently to every lane.

The constants and the algorithm are those of ``scipy.integrate.DOP853`` (Hairer, Norsett and Wanner), so
each lane follows the controller ``solve_ivp`` would use for that motor: the same initial-step heuristic,
error norm, safety factor, step-size limits, rejection rule and 7th-order dense output. The sums over
stages are ordered differently from NumPy's, which perturbs the error estimate by about 1e-8 relative, so
the accepted-step sequence agrees with scipy's to rounding only for the first steps and the integrals
agree to ~1e-6 (architecture document, appendix D).

Everything is a function of the array namespace ``xp`` (``numpy`` or ``jax.numpy``) with no in-place
mutation. Time, step size and state are per lane; ``fun(t, y)`` takes ``t`` of shape ``[B]`` and ``y`` of
shape ``[B, n]``.
"""

import numpy as np

# The Butcher tableau lives in a private scipy module; ``tests/test_batch_integrator.py`` checks its layout.
from scipy.integrate._ivp import dop853_coefficients as _coefficients

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
ERROR_EXPONENT = -1.0 / 8.0
ERROR_ESTIMATOR_ORDER = 7
N_STAGES = _coefficients.N_STAGES  # 12 stages, plus one extra derivative evaluation at the end of the step

_A = np.asarray(_coefficients.A[:N_STAGES, :N_STAGES])
_B = np.asarray(_coefficients.B)
_C = np.asarray(_coefficients.C[:N_STAGES])
_E3 = np.asarray(_coefficients.E3)
_E5 = np.asarray(_coefficients.E5)
_D = np.asarray(_coefficients.D)
_A_EXTRA = np.asarray(_coefficients.A[N_STAGES + 1 :])
_C_EXTRA = np.asarray(_coefficients.C[N_STAGES + 1 :])


def rms(xp, x, size=None):
    """Root mean square over the state axis (scipy's ``norm``).

    ``size`` is the number of real state components of each lane. A padded grain adds components that are
    exactly zero, so they do not change the sum, but they must not count in the divisor: scipy divides by
    the size of the state it integrates.
    """
    n = x.shape[-1] if size is None else size
    return xp.sqrt(xp.sum(x * x, axis=-1)) / n**0.5


def initial_step(xp, fun, t0, y0, f0, t_bound, max_step, rtol, atol, size=None):
    """``select_initial_step`` per lane. Returns the initial ``|h|``. ``size``: real state size per lane."""
    interval = xp.abs(t_bound - t0)
    scale = atol[:, None] + xp.abs(y0) * rtol[:, None]
    d0 = rms(xp, y0 / scale, size)
    d1 = rms(xp, f0 / scale, size)
    h0 = xp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / xp.where(d1 > 0, d1, 1.0))
    h0 = xp.minimum(h0, interval)
    y1 = y0 + h0[:, None] * f0
    f1 = fun(t0 + h0, y1)
    d2 = rms(xp, (f1 - f0) / scale, size) / xp.where(h0 > 0.0, h0, 1.0)  # an empty interval gives h0 = 0, as in scipy
    h1 = xp.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        xp.maximum(1e-6, h0 * 1e-3),
        (0.01 / xp.maximum(d1, d2)) ** (1.0 / (ERROR_ESTIMATOR_ORDER + 1)),
    )
    return xp.minimum(xp.minimum(100.0 * h0, h1), xp.minimum(interval, max_step))


def rk_step(xp, fun, t, y, f, h):
    """One DOP853 step per lane. Returns ``(y_new, f_new, K)`` with ``K`` of shape ``[B, 13, n]``."""
    stages = [f]
    for s in range(1, N_STAGES):
        dy = xp.einsum("k,bkn->bn", xp.asarray(_A[s, :s]), xp.stack(stages, axis=1)) * h[:, None]
        stages.append(fun(t + _C[s] * h, y + dy))
    stacked = xp.stack(stages, axis=1)
    y_new = y + h[:, None] * xp.einsum("k,bkn->bn", xp.asarray(_B), stacked)
    f_new = fun(t + h, y_new)
    return y_new, f_new, xp.concatenate([stacked, f_new[:, None, :]], axis=1)


def error_norm(xp, K, h, scale, size=None):
    """``DOP853._estimate_error_norm`` per lane. ``size``: real state size per lane (see ``rms``)."""
    err5 = xp.einsum("k,bkn->bn", xp.asarray(_E5), K) / scale
    err3 = xp.einsum("k,bkn->bn", xp.asarray(_E3), K) / scale
    norm5 = xp.sum(err5 * err5, axis=-1)
    norm3 = xp.sum(err3 * err3, axis=-1)
    denominator = norm5 + 0.01 * norm3
    n = err5.shape[-1] if size is None else size
    value = xp.abs(h) * norm5 / xp.sqrt(xp.where(denominator > 0, denominator, 1.0) * n)
    return xp.where((norm5 == 0) & (norm3 == 0), 0.0, value)


def growth_factor(xp, err):
    """Factor applied to ``|h|`` after an accepted step (before the cap after a rejection)."""
    grown = xp.minimum(MAX_FACTOR, SAFETY * xp.where(err > 0, err, 1.0) ** ERROR_EXPONENT)
    return xp.where(err == 0, MAX_FACTOR, grown)


def shrink_factor(xp, err):
    """Factor applied to ``|h|`` after a rejected step; a NaN error gives the minimum factor.

    ``max(MIN_FACTOR, nan)`` is ``MIN_FACTOR`` in scipy (Python's ``max``), so NaN is mapped to zero here.
    The zero error of an accepted lane is replaced by one to keep the unused value finite.
    """
    raw = SAFETY * xp.where(err > 0, err, 1.0) ** ERROR_EXPONENT
    return xp.maximum(MIN_FACTOR, xp.where(xp.isnan(err), 0.0, raw))


def attempt(xp, fun, t, y, f, h_abs, rejected, t_bound, max_step, rtol, atol, run, size=None):
    """One step attempt for every lane in ``run``, with scipy's ``_step_impl`` semantics.

    A rejected lane retries with a smaller step on the next call (``rejected`` carries that state), which
    is the lockstep equivalent of scipy's inner retry loop. Lanes outside ``run`` are computed but their
    results must be ignored by the caller. ``size`` is the real state size of each lane (padding excluded).
    """
    min_step = 10.0 * xp.abs(xp.nextafter(t, xp.inf) - t)
    clamped = xp.where(h_abs > max_step, max_step, xp.where(h_abs < min_step, min_step, h_abs))
    h_abs = xp.where(rejected, h_abs, clamped)
    too_small = run & rejected & (h_abs < min_step)
    t_new = t + h_abs
    t_new = xp.where(t_new > t_bound, t_bound, t_new)
    h = t_new - t
    h_effective = xp.abs(h)

    y_new, f_new, K = rk_step(xp, fun, t, y, f, h)
    scale = atol[:, None] + xp.maximum(xp.abs(y), xp.abs(y_new)) * rtol[:, None]
    err = error_norm(xp, K, h, scale, size)
    accept = run & ~too_small & (err < 1.0)
    reject = run & ~too_small & ~(err < 1.0)

    grow = growth_factor(xp, err)
    grow = xp.where(rejected, xp.minimum(1.0, grow), grow)
    h_next = xp.where(accept, h_effective * grow, xp.where(reject, h_effective * shrink_factor(xp, err), h_abs))
    rejected_next = xp.where(accept, False, xp.where(reject, True, rejected))
    return {
        "accept": accept,
        "reject": reject,
        "too_small": too_small,
        "t_new": t_new,
        "h": h,
        "y_new": y_new,
        "f_new": f_new,
        "K": K,
        "h_next": h_next,
        "rejected_next": rejected_next,
    }


def dense_coefficients(xp, fun, t_old, y_old, y_new, f_old, f_new, K, h):
    """The seven interpolation rows of ``Dop853DenseOutput`` (three extra stages), shape ``[B, 7, n]``."""
    stages = [K[:, i, :] for i in range(K.shape[1])]
    for j in range(_A_EXTRA.shape[0]):
        s = N_STAGES + 1 + j
        dy = xp.einsum("k,bkn->bn", xp.asarray(_A_EXTRA[j, :s]), xp.stack(stages, axis=1)) * h[:, None]
        stages.append(fun(t_old + _C_EXTRA[j] * h, y_old + dy))
    stacked = xp.stack(stages, axis=1)
    delta = y_new - y_old
    rows = [delta, h[:, None] * f_old - delta, 2.0 * delta - h[:, None] * (f_new + f_old)]
    tail = h[:, None, None] * xp.einsum("jk,bkn->bjn", xp.asarray(_D), stacked)
    rows.extend(tail[:, j, :] for j in range(tail.shape[1]))
    return xp.stack(rows, axis=1)


def dense_eval(xp, F, y_old, x):
    """Evaluate the dense output at ``x`` in [0, 1] (one value per lane, or ``[B, n]`` per component)."""
    y = xp.zeros_like(y_old)
    x = x[:, None] if x.ndim == 1 else x
    for i in range(F.shape[1]):
        y = y + F[:, F.shape[1] - 1 - i, :]
        y = y * x if i % 2 == 0 else y * (1.0 - x)
    return y + y_old
