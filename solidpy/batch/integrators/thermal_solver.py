# -*- coding: utf-8 -*-
"""The wall conduction loop of ``simulate_thermal_ablation`` for a batch of lanes.

Each time step of the scalar code is one ``solve_ivp(method="Radau")`` call over the step length with the
wall temperatures of the previous step as the start. Here every step is one ``radau.integrate`` call for all
lanes at once (a lane whose curve is shorter is masked), at the scalar code's tolerances, and the extremes the
scalar code keeps are updated after each step:

* the heat flux into the hot face, its maximum and its trapezoid over the points the integrator accepted,
* the temperatures of the inner wall (first casing cell), the outer wall (last cell), the hot face and the
  liner/casing interface at the end of the step, and the wall gradient.

The functions are pure over ``xp``; the loops come from a ``Driver`` (``integrators.solver``).
"""

import numpy as np

from ..kernels import thermal as kernels
from . import radau

#: Tolerances of the ``solve_ivp`` call in ``simulate_thermal_ablation``.
RTOL = 1e-5
ATOL = 1e-6


def solve_thermal(driver, P, max_attempts=None):
    """Integrate the wall of every lane through all its time steps.

    ``P`` holds the arrays of ``ThermalBatch`` as ``xp`` arrays. Returns per-lane arrays: ``max_heat_flux``,
    ``heat`` (trapezoid of the flux over the accepted points, summed over steps), ``max_inner``, ``max_outer``,
    ``max_hot``, ``max_interface``, ``max_gradient`` (all temperatures in K), ``failed`` (the integration of some
    step did not finish: the lane has to be rerun on the scalar code), ``steps`` and ``attempts`` (Radau steps). ``max_attempts`` is the attempts one time step may take
    before its lane is declared failed (``None``: ``radau.MAX_ATTEMPTS``).
    """
    xp = driver.xp
    max_attempts = radau.MAX_ATTEMPTS if max_attempts is None else max_attempts
    lanes, nodes = P["y0"].shape
    lower, diag, upper, source, e0 = P["lower"], P["diag"], P["upper"], P["source"], P["e0"]
    size = P["n_nodes"].astype(float)
    unit = np.zeros(nodes)
    unit[0] = 1.0
    unit_vector = xp.asarray(unit)
    column = xp.zeros((lanes, 1))
    last = P["n_nodes"] - 1
    inner, interface = P["inner_node"], P["interface_node"]
    start = P["initial_temp_k"]
    zero = xp.zeros(lanes)

    def pick(y, index):
        return xp.take_along_axis(y, index[:, None], axis=1)[:, 0]

    state = {
        "y": P["y0"], "max_heat_flux": zero, "heat": zero, "max_inner": start, "max_outer": start, "max_hot": start,
        "max_interface": start, "max_gradient": zero, "failed": zero > 1.0,
        "steps": xp.zeros(lanes, dtype=P["n_nodes"].dtype), "attempts": xp.zeros(lanes, dtype=P["n_nodes"].dtype),
    }

    def body(i, s):
        active = (i < P["n_intervals"]) & ~s["failed"]
        dt = P["dt"][:, i]
        bartz = P["bartz"][:, i]

        def flux(hot_face):
            return kernels.flux(xp, hot_face, bartz, P["recovery_temp_k"], P["flame_temp_k"], P["stagnation"])

        def fun(_t, y):
            conduction = diag * y + xp.concatenate([column, lower * y[:, :-1]], axis=1) \
                + xp.concatenate([upper * y[:, 1:], column], axis=1)
            return conduction + source + (e0 * flux(y[:, 0])[0])[:, None] * unit_vector

        def jac(_t, y, _f):
            return lower, diag + (e0 * flux(y[:, 0])[1])[:, None] * unit_vector, upper

        out = radau.integrate(driver, fun, jac, lambda y: flux(y[:, 0])[0], s["y"], dt, active, size, RTOL, ATOL,
                              max_attempts)
        ok = active & ~out["failed"]
        y = xp.where(ok[:, None], out["y"], s["y"])
        outer = pick(y, last)
        inner_k = pick(y, inner)
        interface_k = 0.5 * (pick(y, interface) + inner_k)
        gradient = (y[:, 0] - outer) / xp.maximum(P["thickness_m"], 1e-9)
        return {
            "y": y,
            "max_heat_flux": xp.where(ok, xp.maximum(s["max_heat_flux"], out["peak"]), s["max_heat_flux"]),
            "heat": s["heat"] + xp.where(ok, out["integral"], 0.0),
            "max_inner": xp.where(ok, xp.maximum(s["max_inner"], inner_k), s["max_inner"]),
            "max_outer": xp.where(ok, xp.maximum(s["max_outer"], outer), s["max_outer"]),
            "max_hot": xp.where(ok, xp.maximum(s["max_hot"], y[:, 0]), s["max_hot"]),
            "max_interface": xp.where(ok, xp.maximum(s["max_interface"], interface_k), s["max_interface"]),
            "max_gradient": xp.where(ok, xp.maximum(s["max_gradient"], gradient), s["max_gradient"]),
            "failed": s["failed"] | (active & out["failed"]),
            "steps": s["steps"] + xp.where(active, out["steps"], 0),
            "attempts": s["attempts"] + xp.where(active, out["attempts"], 0),
        }

    return driver.repeat(P["dt"].shape[1], body, state)
