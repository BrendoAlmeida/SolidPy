# -*- coding: utf-8 -*-
"""Burn stage and blowdown stage of ``BurnSimulation`` for a batch of lanes.

Each stage is one loop whose body attempts one DOP853 step for every unfinished lane (own time, step size and
state per lane); finished lanes are masked. The structure follows ``BurnSimulation._integrate_stage``:

1. Burn stage: a terminal event per active grain (regression crossing its burnout depth, direction +1). The
   event time is located on the dense output, the step is cut there, grains within ``64 eps`` of their burnout
   depth are snapped to it and deactivated, and the lane restarts with a fresh initial step, as the scalar
   code calls ``solve_ivp`` again. The stage is cut into segments at the source breakpoints (igniter and
   activation table knots, the end of the igniter, the end of the ignition ramp), each a new call with a fresh
   initial step, and ends when every grain is burned out or at ``burn_timeout_s``.
2. Source-only stage: when the igniter outlasts the burn, the segments up to its end (or the blowdown
   timeout), with no events. Its points count towards the pressure peak the cutoff is built from.
3. Blowdown stage: a terminal event when the chamber pressure falls through
   ``ambient + 0.01 * max(peak - ambient, 0)`` (direction -1), or ``burn end + tail_off_timeout_s``. Only the
   first segment is integrated: ``BurnSimulation._integrate_stage`` stops its breakpoint loop there when no
   grain is active, so a source breakpoint between burnout and cutoff ends the blowdown early. This is
   reproduced because the scalar code is the oracle.

The functions are pure over the array namespace ``xp``. What differs between NumPy and a compiled driver (the
loop, the conditional and scatter into the history buffers) comes from a ``Driver``.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..kernels import rhs
from . import dop853

EPS = float(np.finfo(float).eps)
#: ``BurnSimulation`` snaps a grain to its burnout depth when it is within this fraction of it.
SNAP_TOLERANCE = 64.0 * EPS
#: Bisection steps on the dense output; scipy uses brentq to 4 eps, 64 halvings of [0, 1] go beyond that.
BISECTION_ITERATIONS = 64


@dataclass(frozen=True)
class Driver:
    """The operations that depend on how the loop is executed."""

    xp: Any
    loop: Callable  # loop(cond, body, carry) -> carry
    branch: Callable  # branch(any_lane, if_true, if_false, operand) -> result
    repeat: Callable  # repeat(count, body, state) -> state
    store: Callable  # store(buffer, rows, cols, values) -> buffer, rows/cols/values per lane


def numpy_driver():
    """A Python loop with masks: finished lanes are held, not recomputed."""
    xp = np

    def loop(cond, body, carry):
        with np.errstate(all="ignore"):
            while cond(carry):
                carry = body(carry)
        return carry

    def branch(any_lane, if_true, if_false, operand):
        return if_true(operand) if bool(any_lane) else if_false(operand)

    def repeat(count, body, state):
        for i in range(count):
            state = body(i, state)
        return state

    def store(buffer, rows, cols, values):
        buffer[rows, cols] = values
        return buffer

    return Driver(xp, loop, branch, repeat, store)


@dataclass(frozen=True)
class SolveConfig:
    """Static options of a solve."""

    keep_history: bool = False
    max_steps: int = 100000  # accepted points per lane across both stages; a lane past it is failed


def _next_boundary(xp, P, t, stop):
    """The end of the segment that starts at ``t``: the next source breakpoint, or the stage stop."""
    candidate = xp.min(xp.where(P["breakpoints"] > t[:, None], P["breakpoints"], xp.inf), axis=1)
    return xp.minimum(candidate, stop)


def _fun(xp, P, active, t_seg):
    """The right-hand side of a segment: sources are evaluated just below the segment end at the end."""

    def fun(t, y):
        source_time = xp.where(t >= t_seg, xp.nextafter(t_seg, -xp.inf), t)
        return rhs.conservative_rhs(xp, y, active, P, time=source_time)

    return fun


def _track_flow(xp, c, name, flow, t_point, write):
    """Running version of ``BurnSimulation._flow_interval`` over the stored points, in order.

    The interval starts at the time of the point before the first positive flow and ends at the time of the
    point after the last positive flow (or at the last point itself).
    """
    positive = write & (flow > 0.0)
    first = positive & ~c[name + "_seen"]
    take_end = write & (positive | c[name + "_want"])
    c[name + "_start"] = xp.where(first, c["prev_time"], c[name + "_start"])
    c[name + "_seen"] = c[name + "_seen"] | positive
    c[name + "_end"] = xp.where(take_end, t_point, c[name + "_end"])
    c[name + "_want"] = xp.where(write, positive, c[name + "_want"])


def _record(driver, c, P, cfg, write, t_point, y_point):
    """Store accepted points: running reductions always, the history buffers only if requested.

    A lane past ``max_steps`` stored points is flagged ``overflow`` and stops storing.
    """
    xp = driver.xp
    full = write & (c["hn"] >= cfg.max_steps)
    write = write & ~full
    q = rhs.state_quantities(xp, y_point, None, P, time=t_point)
    out = dict(c)
    out["pmax"] = xp.where(write, xp.maximum(c["pmax"], q["pressure"]), c["pmax"])
    out["tmax"] = xp.where(write, xp.maximum(c["tmax"], q["thrust"]), c["tmax"])
    out["gmax"] = xp.where(write, xp.maximum(c["gmax"], q["generated"]), c["gmax"])
    out["nmax"] = xp.where(write, xp.maximum(c["nmax"], q["nozzle"]), c["nmax"])
    _track_flow(xp, out, "gen", q["generated"], t_point, write)
    _track_flow(xp, out, "noz", q["nozzle"], t_point, write)
    out["prev_time"] = xp.where(write, t_point, c["prev_time"])
    if cfg.keep_history:
        rows = xp.arange(len(write))
        cols = xp.where(write, c["hn"], cfg.max_steps)  # the buffers have a spare column for masked lanes
        out["ht"] = driver.store(c["ht"], rows, cols, t_point)
        out["hy"] = driver.store(c["hy"], rows, cols, y_point)
    out["hn"] = c["hn"] + write.astype(c["hn"].dtype)
    out["overflow"] = c["overflow"] | full
    return out


def initial_carry(driver, P, y0, cfg):
    xp = driver.xp
    b, n = y0.shape
    g = P["grain_valid"].shape[-1]
    zeros = xp.zeros(b)
    flag = xp.zeros(b, dtype=bool)
    q0 = rhs.state_quantities(xp, y0, None, P, time=xp.zeros(b))
    hist = cfg.max_steps + 1 if cfg.keep_history else 1
    ht = xp.zeros((b, hist))
    hy = xp.zeros((b, hist, n))
    if cfg.keep_history:
        hy = driver.store(hy, xp.arange(b), xp.zeros(b, dtype=int), y0)
    carry = dict(
        t=zeros, y=y0, f=xp.zeros_like(y0), h_abs=zeros, rejected=flag, need_init=~flag, t_start=zeros,
        t_seg=_next_boundary(xp, P, zeros, P["burn_timeout_s"]), t_stop=P["burn_timeout_s"],
        active=P["grain_valid"], burn_t=xp.full((b, g), xp.nan),
        g_prev=xp.zeros((b, g)), done=flag, ok=~flag, overflow=flag, hn=xp.ones(b, dtype=int),
        prev_time=zeros, pmax=q0["pressure"], tmax=q0["thrust"], gmax=q0["generated"], nmax=q0["nozzle"],
        ht=ht, hy=hy, iterations=xp.zeros((), dtype=int),
    )
    for name in ("gen", "noz"):
        carry.update({name + "_start": zeros, name + "_end": zeros, name + "_seen": flag, name + "_want": flag})
    all_points = ~flag
    _track_flow(xp, carry, "gen", q0["generated"], zeros, all_points)
    _track_flow(xp, carry, "noz", q0["nozzle"], zeros, all_points)
    return carry


def _init_block(driver, c, P, active, mode, cutoff=None):
    """Initial step and event baseline for lanes that start, or restart, an integration call."""
    xp = driver.xp
    size = P["n_valid_grains"] + 7
    fun = _fun(xp, P, active, c["t_seg"])
    m = c["need_init"] & ~c["done"]
    f0 = fun(c["t"], c["y"])
    h0 = dop853.initial_step(xp, fun, c["t"], c["y"], f0, c["t_seg"], P["max_step_size"], P["rtol"], P["atol"], size)
    if mode == "burn":
        g = c["y"][:, 2 : 2 + P["grain_valid"].shape[-1]] - P["burnout_depth"]
        mask_g = m[:, None]
    else:
        g = rhs.pressure_of(xp, c["y"], P) - cutoff
        mask_g = m
    out = dict(c)
    out["f"] = xp.where(m[:, None], f0, c["f"])
    out["h_abs"] = xp.where(m, h0, c["h_abs"])
    out["need_init"] = xp.where(m, False, c["need_init"])
    out["rejected"] = xp.where(m, False, c["rejected"])
    out["t_start"] = xp.where(m, c["t"], c["t_start"])
    out["g_prev"] = xp.where(mask_g, g, c["g_prev"])
    return out


def _bisect(driver, value_at, shape, below_is_low):
    """Root of ``value_at(x)`` on [0, 1] by bisection; ``below_is_low(value)`` is true left of the root."""
    xp = driver.xp

    def step(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        left = below_is_low(value_at(mid))
        return xp.where(left, mid, lo), xp.where(left, hi, mid)

    lo, hi = driver.repeat(BISECTION_ITERATIONS, step, (xp.zeros(shape), xp.ones(shape)))
    return 0.5 * (lo + hi)


def burn_body(driver, c, P, cfg):
    xp = driver.xp
    g = P["grain_valid"].shape[-1]
    size = P["n_valid_grains"] + 7
    active = c["active"]
    c = driver.branch(
        xp.any(c["need_init"] & ~c["done"]), lambda cc: _init_block(driver, cc, P, active, "burn"), lambda cc: cc, c
    )
    fun = _fun(xp, P, active, c["t_seg"])
    run = ~c["done"] & ~c["need_init"]
    a = dop853.attempt(
        xp, fun, c["t"], c["y"], c["f"], c["h_abs"], c["rejected"], c["t_seg"], P["max_step_size"], P["rtol"],
        P["atol"], run, size,
    )
    depth = P["burnout_depth"]
    accept = a["accept"]
    g_new = a["y_new"][:, 2 : 2 + g] - depth
    trigger = active & (c["g_prev"] <= 0.0) & (g_new >= 0.0) & accept[:, None]

    def with_events(_):
        F = dop853.dense_coefficients(xp, fun, c["t"], c["y"], a["y_new"], c["f"], a["f_new"], a["K"], a["h"])
        Fg, y_old_g = F[:, :, 2 : 2 + g], c["y"][:, 2 : 2 + g]
        root = _bisect(
            driver, lambda x: dop853.dense_eval(xp, Fg, y_old_g, x) - depth, (len(c["t"]), g), lambda v: v < 0.0
        )
        x_event = xp.min(xp.where(trigger, root, xp.inf), axis=1)
        lane = xp.isfinite(x_event)
        return lane, x_event, dop853.dense_eval(xp, F, c["y"], xp.where(lane, x_event, 1.0))

    def no_events(_):
        return xp.zeros_like(accept), xp.zeros_like(c["t"]), a["y_new"]

    event_lane, x_event, y_event = driver.branch(xp.any(trigger), with_events, no_events, None)
    t_event = c["t"] + x_event * a["h"]
    snap = event_lane[:, None] & active & (y_event[:, 2 : 2 + g] >= depth * (1.0 - SNAP_TOLERANCE))
    y_event = xp.concatenate(
        [y_event[:, :2], xp.where(snap, depth, y_event[:, 2 : 2 + g]), y_event[:, 2 + g :]], axis=1
    )
    active_next = active & ~snap
    any_left = xp.any(active_next, axis=1)
    no_progress = event_lane & (t_event <= c["t_start"]) & any_left

    plain = accept & ~event_lane
    t_point = xp.where(event_lane, t_event, a["t_new"])
    y_point = xp.where(event_lane[:, None], y_event, a["y_new"])
    out = _record(driver, c, P, cfg, accept, t_point, y_point)

    at_segment_end = plain & (a["t_new"] >= c["t_seg"])
    reached_bound = at_segment_end & (c["t_seg"] >= c["t_stop"])
    next_segment = at_segment_end & ~reached_bound
    finished = event_lane & ~any_left
    fail = a["too_small"] | no_progress | (accept & out["overflow"])
    out["t"] = xp.where(accept, t_point, c["t"])
    out["y"] = xp.where(accept[:, None], y_point, c["y"])
    out["f"] = xp.where(plain[:, None], a["f_new"], c["f"])
    out["h_abs"] = xp.where(accept | run, a["h_next"], c["h_abs"])
    out["rejected"] = xp.where(event_lane, False, a["rejected_next"])
    out["need_init"] = c["need_init"] | (event_lane & any_left & ~fail) | (next_segment & ~fail)
    out["t_seg"] = xp.where(next_segment, _next_boundary(xp, P, t_point, c["t_stop"]), c["t_seg"])
    out["active"] = active_next
    out["burn_t"] = xp.where(snap, t_event[:, None], c["burn_t"])
    out["g_prev"] = xp.where(plain[:, None], g_new, c["g_prev"])
    out["done"] = c["done"] | reached_bound | finished | fail
    out["ok"] = c["ok"] & ~fail
    out["iterations"] = c["iterations"] + 1
    return out


def stage_body(driver, c, P, cfg, cutoff, first_segment_only):
    """The source-only and blowdown stages: no active grain, an optional pressure event, segments to ``t_stop``.

    ``cutoff`` is the pressure the blowdown event waits for (``-inf``: no event). With ``first_segment_only``
    the stage ends at the end of its first segment, as the scalar blowdown does.
    """
    xp = driver.xp
    size = P["n_valid_grains"] + 7
    active = xp.zeros_like(P["grain_valid"])
    c = driver.branch(
        xp.any(c["need_init"] & ~c["done"]),
        lambda cc: _init_block(driver, cc, P, active, "tail", cutoff),
        lambda cc: cc,
        c,
    )
    fun = _fun(xp, P, active, c["t_seg"])
    run = ~c["done"] & ~c["need_init"]
    a = dop853.attempt(
        xp, fun, c["t"], c["y"], c["f"], c["h_abs"], c["rejected"], c["t_seg"], P["max_step_size"], P["rtol"],
        P["atol"], run, size,
    )
    accept = a["accept"]
    g_new = rhs.pressure_of(xp, a["y_new"], P) - cutoff
    trigger = accept & (c["g_prev"] >= 0.0) & (g_new <= 0.0)

    def with_events(_):
        F = dop853.dense_coefficients(xp, fun, c["t"], c["y"], a["y_new"], c["f"], a["f_new"], a["K"], a["h"])
        x = _bisect(
            driver, lambda s: rhs.pressure_of(xp, dop853.dense_eval(xp, F, c["y"], s), P) - cutoff,
            (len(c["t"]),), lambda v: v > 0.0,
        )
        return x, dop853.dense_eval(xp, F, c["y"], x)

    def no_events(_):
        return xp.zeros_like(c["t"]), a["y_new"]

    x_event, y_event = driver.branch(xp.any(trigger), with_events, no_events, None)
    t_point = xp.where(trigger, c["t"] + x_event * a["h"], a["t_new"])
    y_point = xp.where(trigger[:, None], y_event, a["y_new"])
    out = _record(driver, c, P, cfg, accept, t_point, y_point)

    plain = accept & ~trigger
    at_segment_end = plain & (a["t_new"] >= c["t_seg"])
    at_stop = at_segment_end & (c["t_seg"] >= c["t_stop"])
    next_segment = at_segment_end & ~at_stop & (not first_segment_only)
    stage_done = at_segment_end if first_segment_only else at_stop
    fail = a["too_small"] | (accept & out["overflow"])
    out["t"] = xp.where(accept, t_point, c["t"])
    out["y"] = xp.where(accept[:, None], y_point, c["y"])
    out["f"] = xp.where(plain[:, None], a["f_new"], c["f"])
    out["h_abs"] = xp.where(accept | run, a["h_next"], c["h_abs"])
    out["rejected"] = a["rejected_next"]
    out["need_init"] = c["need_init"] | (next_segment & ~fail)
    out["t_seg"] = xp.where(next_segment, _next_boundary(xp, P, t_point, c["t_stop"]), c["t_seg"])
    out["g_prev"] = xp.where(plain, g_new, c["g_prev"])
    out["reached_cutoff"] = c["reached_cutoff"] | trigger
    out["done"] = c["done"] | trigger | stage_done | fail
    out["ok"] = c["ok"] & ~fail
    out["iterations"] = c["iterations"] + 1
    return out


def solve_burn_and_blowdown(driver, P, y0, cfg=SolveConfig(), max_iterations=None):
    """Solve every lane through the burn stage and, when requested, the blowdown stage.

    ``max_iterations`` caps the iterations of each stage loop (``None``: no cap). A lane that is not finished
    when the cap is hit is reported in ``unfinished`` and its other outputs are partial; callers rerun such lanes
    from the start in a smaller batch (see ``batch.tiers``). The cap is a plain value, or a traced scalar under jit,
    so changing it does not recompile.

    Returns a mapping of per-lane arrays: the final state and time, running maxima, flow-interval trackers,
    the burnout time of every grain, stage flags and, if ``cfg.keep_history``, the stored points.
    """
    xp = driver.xp
    b = y0.shape[0]
    cap = xp.asarray(np.iinfo(np.int64).max if max_iterations is None else max_iterations)
    carry = initial_carry(driver, P, y0, cfg)
    burn = driver.loop(
        lambda c: ~xp.all(c["done"]) & (c["iterations"] < cap), lambda c: burn_body(driver, c, P, cfg), carry
    )
    burned_out = ~xp.any(burn["active"], axis=1) & burn["ok"]
    tail_off = P["tail_off_evaluation"] > 0.5  # per lane: BurnSimulation(tail_off_evaluation=...)
    no_active = xp.zeros_like(P["grain_valid"])
    never = xp.full(b, -xp.inf)  # a cutoff of -inf: p - cutoff is +inf, so the pressure event cannot fire

    # source-only stage: the igniter outlasts the burn, so the gas keeps being fed until it stops
    burn_end = burn["t"]
    stop_blowdown = burn_end + P["tail_off_timeout_s"]
    source_stop = xp.minimum(P["source_end_time"], stop_blowdown)
    run_sources = burned_out & tail_off & (P["source_end_time"] > burn_end)
    stage = dict(burn)
    stage.update(
        t_stop=source_stop, t_seg=_next_boundary(xp, P, burn_end, source_stop), need_init=run_sources,
        done=~run_sources, rejected=xp.zeros(b, dtype=bool), reached_cutoff=xp.zeros(b, dtype=bool), active=no_active,
        g_prev=xp.zeros(b), iterations=xp.zeros((), dtype=int),
    )
    sources = driver.loop(
        lambda c: ~xp.all(c["done"]) & (c["iterations"] < cap),
        lambda c: stage_body(driver, c, P, cfg, never, False), stage,
    )

    # blowdown cutoff from the pressure peak of the burn stage and the source-only stage (not the blowdown itself)
    pa = P["ambient_pressure"]
    peak_burn = sources["pmax"]
    cutoff = pa + 0.01 * xp.maximum(peak_burn - pa, 0.0)
    p_start = rhs.pressure_of(xp, sources["y"], P)
    alive = burned_out & tail_off & sources["ok"]
    past_stop = sources["t"] >= stop_blowdown
    immediate = alive & ~past_stop & (p_start <= cutoff)
    run_tail = alive & ~past_stop & ~immediate
    tail = dict(sources)
    tail.update(
        t_stop=stop_blowdown, t_seg=_next_boundary(xp, P, sources["t"], stop_blowdown), need_init=run_tail,
        done=~run_tail, rejected=xp.zeros(b, dtype=bool), reached_cutoff=immediate, g_prev=xp.zeros(b),
        iterations=xp.zeros((), dtype=int),
    )
    tail = driver.loop(
        lambda c: ~xp.all(c["done"]) & (c["iterations"] < cap),
        lambda c: stage_body(driver, c, P, cfg, cutoff, True), tail,
    )
    return {
        "t": tail["t"], "y": tail["y"], "y0": y0, "n_points": tail["hn"], "ht": tail["ht"], "hy": tail["hy"],
        "pmax": tail["pmax"], "tmax": tail["tmax"], "gmax": tail["gmax"], "nmax": tail["nmax"],
        "gen_start": tail["gen_start"], "gen_end": tail["gen_end"], "noz_start": tail["noz_start"],
        "noz_end": tail["noz_end"], "burn_t": burn["burn_t"], "burn_ok": burn["ok"], "burned_out": burned_out,
        "tail_ok": tail["ok"], "reached_cutoff": tail["reached_cutoff"], "cutoff": cutoff, "peak_burn": peak_burn,
        "overflow": tail["overflow"], "burn_end": burn["t"], "burn_iterations": burn["iterations"],
        "tail_iterations": tail["iterations"], "source_iterations": sources["iterations"],
        "unfinished": ~burn["done"] | ~sources["done"] | ~tail["done"],
    }
