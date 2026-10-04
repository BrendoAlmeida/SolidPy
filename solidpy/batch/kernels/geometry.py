# -*- coding: utf-8 -*-
"""Grain geometry kernels, one to one with ``Grain`` (tubular and star, ``ends_burn``, burned-through branch).

Every function takes the array namespace ``xp`` first and works on arrays whose last axis is the padded grain
axis ``G``; ``P`` is the mapping of packed arrays (``ProblemBatch.namespace``). There are no Python loops over
lanes or in-place mutation. Fixed loops over the static grain axis preserve the scalar summation order where
padding would otherwise change floating-point reductions; the same code runs on NumPy and ``jax.numpy`` and can
be jit-compiled.
"""

import math

PI = math.pi

#: True when this Python's ``sum`` compensates float sums (3.12 and later), found by behaviour not by version.
COMPENSATED_SUM = sum([0.1] * 10) == 1.0


def ordered_sum(xp, values, compensated=COMPENSATED_SUM):
    """Sum over the last axis the way this Python's ``sum`` adds floats, as the scalar code does.

    The scalar code sums the remaining volumes of the grains with ``sum()``: plain left to right up to Python
    3.11 and Neumaier compensated summation from 3.12 (``compensated`` follows the running interpreter). A
    pairwise ``sum`` rounds differently, and at the initial state the chamber pressure equals the ambient
    pressure to the last bit, where the nozzle flow depends on the square root of the difference. The loop
    runs over the small static grain axis; trailing zeros of padded grains do not change the result.
    """
    total = values[..., 0]
    compensation = xp.zeros_like(total)
    for i in range(1, values.shape[-1]):
        x = values[..., i]
        t = total + x
        if compensated:
            compensation = compensation + xp.where(xp.abs(total) >= xp.abs(x), (total - t) + x, (x - t) + total)
        total = t
    if compensated:
        total = total + xp.where(xp.isfinite(compensation), compensation, 0.0)
    return total


def valid_prefix_sum(xp, values, counts):
    """Sum each lane's valid grain prefix independently of padded suffix width.

    Through 128 float64 elements, this follows NumPy's contiguous ``sum(axis=-1)`` order: sequential addition
    for fewer than eight elements; eight interleaved accumulators, their fixed pairwise tree, then a sequential
    tail for longer prefixes. Above 128 elements, use the same fixed eight-accumulator order as a deterministic
    fallback. ``counts`` broadcasts to ``values.shape[:-1]`` and gives the valid prefix length per lane.

    The loops are over static axis lengths, never over lanes. This preserves the scalar NumPy reduction when
    zero padding changes ``values.shape[-1]``, and keeps the implementation compatible with JAX/JIT.
    """
    width = values.shape[-1]
    if width == 0:
        return xp.zeros(values.shape[:-1], dtype=values.dtype)

    counts = xp.asarray(counts, dtype=xp.int32)
    counts = xp.broadcast_to(counts, values.shape[:-1])
    short = xp.full(values.shape[:-1], -0.0, dtype=values.dtype)
    for i in range(min(width, 8)):
        short = xp.where(counts > i, short + values[..., i], short)
    short = xp.where(counts == 0, xp.zeros_like(short), short)
    if width < 8:
        return short

    def eight_way_sum(prefix_counts, max_count):
        """NumPy's eight-accumulator reduction for a prefix no longer than ``max_count``."""
        accumulators = values[..., :8]
        full_limit = min(width, max_count) // 8 * 8
        for block_start in range(8, full_limit, 8):
            block = values[..., block_start : block_start + 8]
            accumulators = xp.where(
                (prefix_counts >= block_start + 8)[..., None], accumulators + block, accumulators
            )
        pair_sum = (
            (accumulators[..., 0] + accumulators[..., 1])
            + (accumulators[..., 2] + accumulators[..., 3])
        ) + (
            (accumulators[..., 4] + accumulators[..., 5])
            + (accumulators[..., 6] + accumulators[..., 7])
        )
        tail_start = (prefix_counts // 8) * 8
        offsets = xp.arange(8)
        indices = xp.clip(tail_start[..., None] + offsets, 0, width - 1)
        tail = xp.take_along_axis(values, indices, axis=-1)
        for i in range(8):
            pair_sum = xp.where(prefix_counts % 8 > i, pair_sum + tail[..., i], pair_sum)
        return xp.where(prefix_counts < 8, short, pair_sum)

    exact_counts = xp.minimum(counts, 128)
    exact = eight_way_sum(exact_counts, 128)
    if width <= 128:
        return exact

    # NumPy switches to recursive splitting above 128. A fixed eight-way reduction here remains accurate and
    # padding-invariant for unusually large motors without adding dynamic recursive control flow to JAX.
    fallback = eight_way_sum(counts, width)
    return xp.where(counts <= 128, exact, fallback)


def port_area(xp, regression, P):
    """Gas-port cross-section at ``regression`` (``Grain.evaluate_port_area``)."""
    w = xp.maximum(regression, 0.0)
    outer = P["outer_radius"]
    r_bore = xp.minimum(P["inner_radius0"] + w, outer)
    r_floor = xp.where(w < P["slot_floor_depth"], xp.minimum(P["slot_floor_radius"] + w, outer), outer)
    slot = P["n_points"] * P["epsilon"] * xp.maximum(r_floor**2 - r_bore**2, 0.0)
    return PI * r_bore**2 + xp.where(P["is_star"], slot, 0.0)


def remaining_volume(xp, regression, P):
    """Remaining solid volume per grain (``Grain.calculate_remaining_volume``); zero for padded grains."""
    w = xp.maximum(regression, 0.0)
    height = xp.where(P["ends_burn"], P["height0"], P["height0"] - 2.0 * w)
    solid_area = PI * P["outer_radius"] ** 2 - port_area(xp, w, P)
    volume = xp.maximum(solid_area * height, 0.0)
    volume = xp.where(w >= P["burnout_depth"], 0.0, volume)
    return xp.where(P["grain_valid"], volume, 0.0)


def burn_area(xp, regression, P):
    """Burn area per grain (``Grain.evaluate_burn_area``), zero once the grain is burned through.

    The solver evaluates this at a regression clamped just below the burnout depth; that clamp belongs to
    the caller (``rhs.state_quantities``), as in the scalar code.
    """
    w = xp.maximum(regression, 0.0)
    outer, inner0, height0, ends = P["outer_radius"], P["inner_radius0"], P["height0"], P["ends_burn"]
    web = outer - inner0

    # tubular (``Grain.calculate_tubular_geometry``)
    burned_tubular = xp.where(ends, w >= web, (w >= web) | (w >= height0 / 2))
    inner = xp.minimum(inner0 + w, outer)
    height_tubular = xp.where(ends, height0, xp.maximum(height0 - 2.0 * w, 0.0))
    longitudinal = 2.0 * PI * inner * height_tubular
    transversal = 2.0 * PI * (outer**2 - inner**2)
    area_tubular = xp.where(burned_tubular, 0.0, xp.where(ends, longitudinal, transversal + longitudinal))

    # star, fixed-angle radial front (``Grain.calculate_star_geometry``)
    points, epsilon = P["n_points"], P["epsilon"]
    r_bore = inner0 + w
    height_star = xp.where(ends, height0, height0 - 2.0 * w)
    phase_one = w < P["slot_floor_depth"]
    r_floor = P["slot_floor_radius"] + w
    lateral = xp.where(
        phase_one,
        (2.0 * PI - 2.0 * points * epsilon) * r_bore + 2.0 * points * epsilon * r_floor,
        (2.0 * PI - 2.0 * points * epsilon) * r_bore,
    )
    end_area = xp.where(
        phase_one,
        PI * (outer**2 - r_bore**2) - points * epsilon * (r_floor**2 - r_bore**2),
        (PI - points * epsilon) * (outer**2 - r_bore**2),
    )
    end_faces = xp.where(ends, 0.0, 2.0 * end_area)
    area_star = xp.maximum(lateral * height_star + end_faces, 0.0)
    burned_star = (w >= web) | (~ends & (w >= height0 / 2))
    area_star = xp.where(burned_star, 0.0, area_star)

    return xp.where(P["is_star"], area_star, area_tubular)
