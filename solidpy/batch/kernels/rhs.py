# -*- coding: utf-8 -*-
"""Derived quantities and conservative right-hand side of the burn model for a batch of lanes.

``state_quantities`` mirrors ``BurnSimulation._state_quantities_uncached`` and ``conservative_rhs`` mirrors
``BurnSimulation._conservative_rhs``. The state of a lane is
``[gas mass, thermal inventory, regression_0 .. regression_{G-1}, generated mass, igniter mass, nozzle mass,
unscaled thrust impulse, pressure * throat area integral]``.

Sources: with ``time`` the igniter mass flow and the activation factor (scalar, table or built-in ramp) are
applied as in the scalar code; without it the lane has none (activation 1, no igniter). Callable sources are
not reproduced (their mode is NaN, so the results are NaN rather than plausible); a backend only advertises the
features its kernels reproduce.

The kernels are shape agnostic: with lane arrays of shape ``S`` and grain arrays of shape ``S + (G,)`` they
return quantities of shape ``S``. ``S = (B,)`` is the solver case; ``S = (B, T)`` evaluates a stored history
(pass every packed array with a new axis inserted after the lane axis).
"""

from .geometry import burn_area, ordered_sum, port_area, remaining_volume, valid_prefix_sum
from .nozzle import TINY, nozzle_mass_flow, thrust_components
from .propellant import burn_rate, exit_mach, gas_properties
from .sources import activation as activation_factor
from .sources import igniter_flow as igniter_flow_at


def pressure_of(xp, y, P):
    """Chamber pressure only (event function and cutoff checks)."""
    g = P["grain_valid"].shape[-1]
    remaining = ordered_sum(xp, remaining_volume(xp, y[..., 2 : 2 + g], P))
    return P["gas_constant"] * xp.maximum(y[..., 1], TINY) / (P["chamber_volume"] - remaining)


def state_quantities(xp, y, active, P, detail=False, time=None):
    """Derived quantities at a state.

    ``active`` is the per-grain boolean mask the integrator carries, or ``None`` to derive it from the
    regressions as the scalar post-processing does. Padded grains are never active. ``detail`` adds the exit
    velocity and exit pressure used by stored histories. ``time`` (same shape as the pressure) switches the
    igniter and the activation on.
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
    port_mean = valid_prefix_sum(
        xp, xp.where(valid, port_area(xp, regression, P), 0.0), P["n_valid_grains"]
    ) / P["n_valid_grains"]
    source_temperature, k = gas_properties(xp, pressure, P)
    source_temperature = xp.broadcast_to(source_temperature, pressure.shape)
    nozzle = nozzle_mass_flow(xp, pressure, temperature, k, P)
    port_mass_flux = 0.5 * nozzle / xp.maximum(port_mean, 1e-9)
    rate = burn_rate(xp, xp.maximum(pressure, P["ambient_pressure"]), port_mass_flux, P)
    rate = xp.where(xp.any(active, axis=-1), rate, 0.0)
    if time is None:
        activation, igniter = xp.ones_like(rate), xp.zeros_like(rate)
    else:
        activation, igniter = activation_factor(xp, time, P), igniter_flow_at(xp, time, P)
    rates = xp.where(active, (rate * activation)[..., None], 0.0)

    generated_grains = P["density"][..., None] * areas * rates
    generated = valid_prefix_sum(xp, generated_grains, P["n_valid_grains"])
    parts = thrust_components(xp, pressure, temperature, nozzle, k, exit_mach(xp, k, P), P, detail)
    ideal, momentum, pressure_thrust, thrust = parts[:4]
    quantities = {
        "pressure": pressure,
        "volume": volume,
        "temperature": temperature,
        "source_temperature": source_temperature,
        "regression_rates": rates,
        "areas": areas * activation[..., None],
        "generated_grains": generated_grains,
        "generated": generated,
        "igniter": igniter,
        "activation": activation,
        "nozzle": nozzle,
        "momentum_ideal": ideal,
        "momentum": momentum,
        "pressure_thrust": pressure_thrust,
        "thrust": thrust,
    }
    if detail:
        quantities["exit_velocity"], quantities["exit_pressure"] = parts[4], parts[5]
    return quantities


def conservative_rhs(xp, y, active, P, time=None, return_record_values=False):
    """Time derivative of the conservative state, same shape as ``y``; ``time`` switches the sources on.

    With ``return_record_values=True``, also return ``(pressure, thrust, generated, nozzle)`` from the same
    state-quantity evaluation. The integrator uses these values at ordinary accepted endpoints, where their
    state, active-grain mask and source time exactly match the values needed by its running reductions.
    """
    q = state_quantities(xp, y, active, P, time=time)
    igniter = q["igniter"]
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
    derivative = xp.concatenate([mass_rate[..., None], thermal_rate[..., None], q["regression_rates"], tail], axis=-1)
    if return_record_values:
        return derivative, (q["pressure"], q["thrust"], q["generated"], q["nozzle"])
    return derivative
