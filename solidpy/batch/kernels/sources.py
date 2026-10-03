# -*- coding: utf-8 -*-
"""Igniter mass flow and burn-area activation, mirroring ``BurnSimulation.evaluate_igniter_mass_flow`` and
``evaluate_burn_area_activation`` for scalar, table and built-in ramp sources (callables stay on the reference).

Each lane has a mode per source: 0 none, 1 scalar, 2 table. Tables are padded to a common length with finite,
increasing abscissae so that no division by zero or ``inf - inf`` happens in the discarded branches, and a
per-lane count says how many entries are real.
"""

NONE, SCALAR, TABLE = 0.0, 1.0, 2.0


def _interp(xp, x, xs, ys, count):
    """``np.interp(x, xs[:count], ys[:count])`` for one table per lane; ``x`` is clamped to the end values."""
    k = xs.shape[-1]
    valid = xp.arange(k) < count[..., None]
    below = xp.sum((xs <= x[..., None]) & valid, axis=-1) - 1
    j = xp.clip(below, 0, xp.maximum(count - 2, 0))
    last = xp.maximum(count - 1, 0)
    x0 = xp.take_along_axis(xs, j[..., None], axis=-1)[..., 0]
    x1 = xp.take_along_axis(xs, xp.minimum(j + 1, k - 1)[..., None], axis=-1)[..., 0]
    y0 = xp.take_along_axis(ys, j[..., None], axis=-1)[..., 0]
    y1 = xp.take_along_axis(ys, xp.minimum(j + 1, k - 1)[..., None], axis=-1)[..., 0]
    width = x1 - x0
    slope = (y1 - y0) / xp.where(width != 0.0, width, 1.0)  # a one-entry table has no interval to divide by
    inside = slope * (x - x0) + y0
    first_x = xs[..., 0]
    x_last = xp.take_along_axis(xs, last[..., None], axis=-1)[..., 0]
    y_last = xp.take_along_axis(ys, last[..., None], axis=-1)[..., 0]
    return xp.where(x >= x_last, y_last, xp.where(x < first_x, ys[..., 0], inside))


def igniter_flow(xp, time, P):
    """Igniter gas mass flow in kg/s at ``time``."""
    mode = P["igniter_mode"]
    burn_time = P["igniter_burn_time"]
    scalar = xp.where((burn_time > 0.0) & (time < burn_time), xp.maximum(P["igniter_value"], 0.0), 0.0)
    xs, ys, count = P["igniter_table_t"], P["igniter_table_m"], P["igniter_table_n"]
    x_last = xp.take_along_axis(xs, xp.maximum(count - 1, 0)[..., None], axis=-1)[..., 0]
    inside = (time >= xs[..., 0]) & (time < x_last)
    table = xp.where(inside, xp.maximum(_interp(xp, time, xs, ys, count), 0.0), 0.0)
    # a lane whose mode is NaN (a callable source) stays NaN instead of silently getting no igniter
    return xp.where(mode == SCALAR, scalar, xp.where(mode == TABLE, table, 0.0)) + 0.0 * mode


def activation(xp, time, P):
    """Ignited fraction of the burn area in [0, 1] at ``time`` (the built-in ramp when there is no profile)."""
    mode = P["activation_mode"]
    ramp = P["ignition_ramp_time"]
    progress = xp.clip(time / xp.where(ramp > 0.0, ramp, 1.0), 0.0, 1.0)
    smooth = xp.where(ramp > 0.0, progress * progress * (3.0 - 2.0 * progress), 1.0)
    scalar = xp.clip(P["activation_value"], 0.0, 1.0)
    table = xp.clip(_interp(xp, time, P["activation_table_t"], P["activation_table_a"], P["activation_table_n"]), 0.0, 1.0)
    return xp.where(mode == SCALAR, scalar, xp.where(mode == TABLE, table, smooth)) + 0.0 * mode
