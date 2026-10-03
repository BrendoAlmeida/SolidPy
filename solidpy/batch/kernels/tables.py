# -*- coding: utf-8 -*-
"""Piecewise-polynomial tables (burn rate and thermochemistry) evaluated on arrays.

Scalar SolidPy evaluates ``scipy.interpolate.interp1d`` (cubic from four points, linear below) with the ends
held at fixed values. The coefficients are computed once at packing time, with scipy, so the interpolant is
defined in one place; here a table is only looked up (bisection, so a 1,000-row table costs ten steps) and its
cubic evaluated by Horner. Tables are padded per lane with finite, increasing abscissae and a real length.
"""


def count_not_above(xp, x, xs, count):
    """How many of the first ``count`` abscissae of each lane's table are ``<= x`` (bisection, fixed steps)."""
    k = xs.shape[-1]
    lo = xp.zeros(x.shape, dtype=count.dtype)
    hi = xp.broadcast_to(count, x.shape).astype(count.dtype)
    for _ in range(int(k).bit_length() + 1):
        active = lo < hi
        mid = (lo + hi) // 2
        x_mid = xp.take_along_axis(xs, xp.minimum(mid, k - 1)[..., None], axis=-1)[..., 0]
        move = active & (x_mid <= x)
        lo = xp.where(move, mid + 1, lo)
        hi = xp.where(active & ~move, mid, hi)
    return lo


def locate(xp, x, xs, count):
    """The interval ``j`` with ``xs[j] <= x < xs[j + 1]`` of each lane's table, clipped to the real intervals."""
    return xp.clip(count_not_above(xp, x, xs, count) - 1, 0, xp.maximum(count - 2, 0))


def evaluate(xp, x, xs, coefficients, count, below, above):
    """Evaluate the cubic pieces ``c0 dx^3 + c1 dx^2 + c2 dx + c3`` (``dx = x - xs[j]``) with the ends held.

    ``coefficients`` is ``(c0, c1, c2, c3)``, each ``[..., K]`` with one column per interval. Left of the first
    abscissa the value is ``below`` and right of the last one ``above`` (``interp1d``'s ``fill_value``).
    """
    j = locate(xp, x, xs, count)
    take = lambda table: xp.take_along_axis(table, j[..., None], axis=-1)[..., 0]  # noqa: E731
    dx = x - take(xs)
    c0, c1, c2, c3 = (take(c) for c in coefficients)
    inside = ((c0 * dx + c1) * dx + c2) * dx + c3
    x_last = xp.take_along_axis(xs, xp.maximum(count - 1, 0)[..., None], axis=-1)[..., 0]
    return xp.where(x < xs[..., 0], below, xp.where(x > x_last, above, inside))
