# -*- coding: utf-8 -*-
"""Derived quantities and conservative right-hand side of the burn model for a batch of lanes.

``state_quantities`` mirrors ``BurnSimulation._state_quantities_uncached`` and ``conservative_rhs`` mirrors
``BurnSimulation._conservative_rhs``. The state of a lane is
``[gas mass, thermal inventory, regression_0 .. regression_{G-1}, generated mass, igniter mass, nozzle mass,
unscaled thrust impulse, pressure * throat area integral]``.

Scope: lanes without igniter, activation profile or ignition ramp. The scalar code scales the regression rates
and the burn area by an activation factor that depends on time; these kernels take no time and apply none,
so a lane that needs it (``ProblemBatch.lane_features``) must be routed to a backend that declares it. A
batched backend only advertises the features its kernels reproduce.

The kernels are shape agnostic: with lane arrays of shape ``S`` and grain arrays of shape ``S + (G,)`` they
return quantities of shape ``S``. ``S = (B,)`` is the solver case; ``S = (B, T)`` evaluates a stored history
(pass every packed array with a new axis inserted after the lane axis).
"""

from .geometry import burn_area, ordered_sum, port_area, remaining_volume
from .nozzle import TINY, nozzle_mass_flow, thrust_components
from .propellant import burn_rate, gas_properties


def pressure_of(xp, y, P):
    """Chamber pressure only (event function and cutoff checks)."""
    g = P["grain_valid"].shape[-1]
    remaining = ordered_sum(xp, remaining_volume(xp, y[..., 2 : 2 + g], P))
    return P["gas_constant"] * xp.maximum(y[..., 1], TINY) / (P["chamber_volume"] - remaining)


def state_quantities(xp, y, active, P):
    """Derived quantities at a state.

    ``active`` is the per-grain boolean mask the integrator carries, or ``None`` to derive it from the
    regressions as the scalar post-processing does. Padded grains are never active.
    """
    g = P["grain_valid"].shape[-1]
    valid = P["grain_valid"]
    regression = y[..., 2 : 2 + g]

    remaining = ordered_sum(xp, remaining_volume(xp, regression, P))
    volume = P["chamber_volume"] - remaining
    gas_mass = xp.maximum(y[..., 0], TINY)
    thermal = xp.maximum(y[..., 1], TINY)
    temperature = thermal / gas_mass
    pressure = P["gas_constant"] * thermal / volume

    if active is None:
        active = regression < P["burnout_depth"]
    active = active & valid
    # the scalar solver evaluates the burn area at a regression clamped just below the burnout depth
    clamped = xp.minimum(xp.maximum(regression, 0.0), xp.nextafter(P["burnout_depth"], 0.0))
    areas = xp.where(active, burn_area(xp, clamped, P), 0.0)

    # the mean port area runs over every real grain, burned out or not (as np.mean does in the scalar code)
    port_mean = xp.sum(xp.where(valid, port_area(xp, regression, P), 0.0), axis=-1) / P["n_valid_grains"]
    source_temperature, k = gas_properties(xp, pressure, P)
    source_temperature = xp.broadcast_to(source_temperature, pressure.shape)
    nozzle = nozzle_mass_flow(xp, pressure, temperature, k, P)
    port_mass_flux = 0.5 * nozzle / xp.maximum(port_mean, 1e-9)
    rate = burn_rate(xp, xp.maximum(pressure, P["ambient_pressure"]), port_mass_flux, P)
    rate = xp.where(xp.any(active, axis=-1), rate, 0.0)
    rates = xp.where(active, rate[..., None], 0.0)

    generated_grains = P["density"][..., None] * areas * rates
    generated = xp.sum(generated_grains, axis=-1)
    ideal, momentum, pressure_thrust, thrust = thrust_components(
        xp, pressure, temperature, nozzle, k, P["exit_mach"], P
    )
    return {
        "pressure": pressure,
        "volume": volume,
        "temperature": temperature,
        "source_temperature": source_temperature,
        "regression_rates": rates,
        "areas": areas,
        "generated_grains": generated_grains,
        "generated": generated,
        "nozzle": nozzle,
        "momentum_ideal": ideal,
        "momentum": momentum,
        "pressure_thrust": pressure_thrust,
        "thrust": thrust,
    }


def conservative_rhs(xp, y, active, P, igniter_flow=None):
    """Time derivative of the conservative state, same shape as ``y``."""
    q = state_quantities(xp, y, active, P)
    igniter = xp.zeros_like(q["generated"]) if igniter_flow is None else igniter_flow
    mass_rate = q["generated"] + igniter - q["nozzle"]
    thermal_rate = (
        q["generated"] * q["source_temperature"] + igniter * P["igniter_temperature"] - q["nozzle"] * q["temperature"]
    )
    tail = xp.stack(
        [
            q["generated"],
            igniter,
            q["nozzle"],
            q["momentum"] + q["pressure_thrust"],  # unscaled: eta_Cf is applied to the reported impulse
            q["pressure"] * P["throat_area"],
        ],
        axis=-1,
    )
    return xp.concatenate([mass_rate[..., None], thermal_rate[..., None], q["regression_rates"], tail], axis=-1)
