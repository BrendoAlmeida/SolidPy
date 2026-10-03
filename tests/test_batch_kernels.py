import numpy as np
import pytest

import golden_corpus as gc
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch
from solidpy.batch import problem as pb
from solidpy import Burn
from solidpy.batch.kernels import geometry, nozzle, propellant

RTOL = tol.KERNEL_RTOL_NUMPY


def close(actual, desired, what, scale=0.0):
    """Relative agreement, with an absolute floor of ``RTOL * scale`` for results of a cancellation.

    The scalar code squares with ``pow``, which is not always the correctly rounded ``x * x`` the kernels
    use, so a difference of two nearly equal areas (a grain one step from burnout) can differ by an ulp of
    the operands. That is a relative error of order one on a result that is ~1e-16 of the grain volume.
    """
    np.testing.assert_allclose(actual, desired, rtol=RTOL, atol=max(RTOL * scale, 1e-300), err_msg=what)


@pytest.fixture(scope="module")
def lanes():
    """Every corpus design, packed once, with the scalar objects each lane came from."""
    cases = gc.load_corpus()["cases"]
    built = [gc.build_objects(case) for case in cases]
    batch = ProblemBatch.from_objects(
        [b[1] for b in built], [b[2] for b in built], [b[3] for b in built], [b[4] for b in built]
    )
    return cases, batch


def regression_samples(a):
    """Regression depths at every edge of the scalar geometry code, shape ``[B, G, S]``."""
    burnout, floor = a["burnout_depth"], a["slot_floor_depth"]
    web = a["outer_radius"] - a["inner_radius0"]
    columns = [
        np.full_like(burnout, -1e-3), np.zeros_like(burnout), np.full_like(burnout, 1e-12),
        0.1 * burnout, 0.37 * burnout, 0.5 * burnout, 0.9 * burnout,
        np.nextafter(burnout, 0.0), burnout * (1 - 1e-12), burnout, burnout * (1 + 1e-12), 1.5 * burnout,
        floor * 0.999999, floor, floor * 1.000001, web * 0.999, web, a["height0"] / 2, 10 * a["outer_radius"],
    ]
    return np.stack(columns, axis=-1)


def test_geometry_kernels_match_the_grain_methods_at_every_edge(lanes):
    cases, batch = lanes
    a = batch.arrays
    P = batch.namespace(np)
    samples = regression_samples(a)
    cross_section = np.pi * a["outer_radius"] ** 2
    scales = {
        "burn area": 2 * np.pi * a["outer_radius"] * (a["outer_radius"] + a["height0"]),
        "port area": cross_section,
        "remaining volume": cross_section * a["height0"],
    }

    for s in range(samples.shape[-1]):
        regression = samples[..., s]
        area = geometry.burn_area(np, regression, P)
        port = geometry.port_area(np, regression, P)
        volume = geometry.remaining_volume(np, regression, P)
        for lane, motor in enumerate(batch.motors):
            for slot, grain in enumerate(motor.grains):
                r = regression[lane, slot]
                where = f"{cases[lane]['id']} grain {slot} regression {r!r}"
                scalar_area = grain.evaluate_burn_area(r)
                close(area[lane, slot], scalar_area, "burn area " + where, scales["burn area"][lane, slot])
                assert (area[lane, slot] == 0.0) == (scalar_area == 0.0), "burned-through branch " + where
                close(port[lane, slot], grain.evaluate_port_area(r), "port area " + where,
                      scales["port area"][lane, slot])
                close(volume[lane, slot], grain.calculate_remaining_volume(r), "remaining volume " + where,
                      scales["remaining volume"][lane, slot])


def test_padded_grains_hold_no_volume_and_the_geometry_stays_finite(lanes):
    _, batch = lanes
    P = batch.namespace(np)
    padded = ~batch.arrays["grain_valid"]
    assert padded.any()

    for s in range(regression_samples(batch.arrays).shape[-1]):
        regression = regression_samples(batch.arrays)[..., s]
        volume = geometry.remaining_volume(np, regression, P)
        assert (volume[padded] == 0.0).all()
        for kernel in (geometry.burn_area, geometry.port_area):
            assert np.isfinite(kernel(np, regression, P)[padded]).all()


def test_regression_beyond_the_burnout_depth_leaves_no_burn_area_or_volume(lanes):
    _, batch = lanes
    P = batch.namespace(np)
    beyond = 1.5 * batch.arrays["burnout_depth"]
    valid = batch.arrays["grain_valid"]

    assert (geometry.remaining_volume(np, beyond, P)[valid] == 0.0).all()
    # tubular and star grains are burned through at (or before) the burnout depth
    assert (geometry.burn_area(np, beyond, P)[valid] == 0.0).all()


def test_shape_agnostic_kernels_accept_a_time_axis(lanes):
    """With lane arrays ``[B, 1]`` and grain arrays ``[B, 1, G]`` the kernels evaluate a whole history."""
    _, batch = lanes
    a = batch.arrays
    P = batch.namespace(np)
    view = {name: array[:, None] for name, array in P.items()}
    samples = np.moveaxis(regression_samples(a), -1, 1)  # [B, S, G]

    stacked = geometry.burn_area(np, samples, view)

    for s in range(samples.shape[1]):
        np.testing.assert_array_equal(stacked[:, s], geometry.burn_area(np, samples[:, s], P))


def test_the_packed_inputs_of_the_geometry_match_the_grain_attributes(lanes):
    _, batch = lanes
    a = batch.arrays
    for lane, motor in enumerate(batch.motors):
        for slot, grain in enumerate(motor.grains):
            assert a["outer_radius"][lane, slot] == grain.outer_radius
            assert a["inner_radius0"][lane, slot] == grain.initial_inner_radius
            assert a["is_star"][lane, slot] == (grain.geometry == "star")
            assert a["burnout_depth"][lane, slot] == grain.burnout_regression_m
    assert pb.UNKNOWN_GEOMETRY not in set().union(*batch.lane_features)


def lane_view(P):
    """Lane arrays as ``[B, 1]`` so they broadcast against ``[B, S]`` samples."""
    return {name: array[:, None] if array.ndim == 1 else array for name, array in P.items()}


def scalar_burn(batch, lane):
    settings = batch.settings[lane]
    return Burn(
        batch.motors[lane].grains[0], batch.motors[lane], batch.propellants[lane], batch.environments[lane],
        eta_c=settings["eta_c"], eta_Cf=settings["eta_Cf"], discharge_coefficient=settings["discharge_coefficient"],
    )


def pressure_samples(a):
    """Chamber pressures around ambient, the choking boundary and far above it, shape ``[B, S]``."""
    ambient, k = a["ambient_pressure"][:, None], a["gamma"][:, None]
    boundary = ambient / nozzle.critical_pressure_ratio(np, k)
    return np.concatenate(
        [
            0.3 * ambient, ambient * (1 - 1e-12), ambient, ambient * (1 + 1e-9), 1.1 * ambient,
            0.5 * (ambient + boundary), boundary * (1 - 1e-9), boundary, boundary * (1 + 1e-9),
            3 * boundary, 10 * boundary, 40 * boundary, 100 * boundary, np.full_like(ambient, 1e8),
        ],
        axis=1,
    )


def test_nozzle_flow_and_thrust_match_burn_at_the_choking_boundary_and_below_ambient(lanes):
    cases, batch = lanes
    keep = [i for i, f in enumerate(batch.lane_features) if pb.THERMO_SCALAR in f]
    sub = batch.select(keep)
    V = lane_view(sub.namespace(np))
    pressure = pressure_samples(sub.arrays)
    temperature = sub.arrays["source_temperature"][:, None] * np.random.default_rng(7).uniform(0.3, 1.05, pressure.shape)

    k = V["gamma"]
    flow = nozzle.nozzle_mass_flow(np, pressure, temperature, k, V)
    ideal, momentum, pressure_thrust, total = nozzle.thrust_components(np, pressure, temperature, flow, k, V["exit_mach"], V)

    for row, lane in enumerate(keep):
        burn = scalar_burn(batch, lane)
        for s in range(pressure.shape[1]):
            p, t = pressure[row, s], temperature[row, s]
            where = f"{cases[lane]['id']} pressure {p!r}"
            scalar = burn.evaluate_thrust_components(p, chamber_temperature=t)
            close(flow[row, s], burn.evaluate_nozzle_mass_flow(p, chamber_temperature=t), "mass flow " + where)
            close(ideal[row, s], scalar["momentum_ideal_n"], "ideal momentum " + where)
            close(momentum[row, s], scalar["momentum_n"], "momentum " + where)
            # pressure thrust is (p_exit - p_ambient) * A_e: a cancellation when the flow is barely choked
            close(pressure_thrust[row, s], scalar["pressure_n"], "pressure thrust " + where,
                  scale=p * sub.arrays["exit_area"][row])
            close(total[row, s], scalar["total_n"], "total thrust " + where,
                  scale=p * sub.arrays["exit_area"][row])


def test_no_flow_and_no_thrust_at_or_below_ambient_pressure(lanes):
    _, batch = lanes
    keep = [i for i, f in enumerate(batch.lane_features) if pb.THERMO_SCALAR in f]
    sub = batch.select(keep)
    V = lane_view(sub.namespace(np))
    ambient = sub.arrays["ambient_pressure"][:, None]
    pressure = ambient * np.array([[0.0, 0.5, 1.0 - 1e-12, 1.0]])
    temperature = np.full_like(pressure, 2000.0)

    flow = nozzle.nozzle_mass_flow(np, pressure, temperature, V["gamma"], V)
    parts = nozzle.thrust_components(np, pressure, temperature, flow, V["gamma"], V["exit_mach"], V)

    assert not flow.any()
    assert all(not part.any() for part in parts)
    assert all(np.isfinite(part).all() for part in parts)


def test_choked_and_unchoked_flow_are_continuous_at_the_boundary(lanes):
    _, batch = lanes
    keep = [i for i, f in enumerate(batch.lane_features) if pb.THERMO_SCALAR in f]
    sub = batch.select(keep)
    V = lane_view(sub.namespace(np))
    k = V["gamma"]
    boundary = sub.arrays["ambient_pressure"][:, None] / nozzle.critical_pressure_ratio(np, k)
    temperature = np.full_like(boundary, 2500.0)

    below = nozzle.nozzle_mass_flow(np, boundary * (1 - 1e-9), temperature, k, V)
    above = nozzle.nozzle_mass_flow(np, boundary * (1 + 1e-9), temperature, k, V)

    np.testing.assert_allclose(below, above, rtol=1e-8)


def test_burn_rate_matches_the_propellant_with_and_without_the_erosive_term(lanes):
    cases, batch = lanes
    keep = [i for i, f in enumerate(batch.lane_features) if pb.BURN_RATE_POWER_LAW in f]
    sub = batch.select(keep)
    V = lane_view(sub.namespace(np))
    pressures = np.array([-5e5, 0.0, 1e5, 1e6, 3.3e6, 8e6, 2e7])
    fluxes = np.array([0.0, 5e-4, 1e-3, 1.0000001e-3, 0.5, 20.0, 300.0, 5e3])
    p = np.repeat(pressures, len(fluxes))[None, :].repeat(len(keep), axis=0)
    g = np.tile(fluxes, len(pressures))[None, :].repeat(len(keep), axis=0)

    rate = propellant.burn_rate(np, p, g, V)

    erosive_lanes = 0
    for row, lane in enumerate(keep):
        scalar = batch.propellants[lane]
        erosive_lanes += pb.EROSIVE_BURNING in batch.lane_features[lane]
        for s in range(p.shape[1]):
            close(rate[row, s], scalar.evaluate_burn_rate(p[row, s], g[row, s]),
                  f"burn rate {cases[lane]['id']} pressure {p[row, s]!r} flux {g[row, s]!r}")
    assert erosive_lanes >= 10


def test_the_erosive_term_switches_on_only_above_the_flux_threshold(lanes):
    _, batch = lanes
    keep = [i for i, f in enumerate(batch.lane_features) if pb.EROSIVE_BURNING in f]
    V = lane_view(batch.select(keep).namespace(np))
    pressure = np.full((len(keep), 1), 3e6)

    plain = propellant.burn_rate(np, pressure, np.full_like(pressure, 1e-3), V)
    eroded = propellant.burn_rate(np, pressure, np.full_like(pressure, 300.0), V)
    base = propellant.burn_rate(np, pressure, np.zeros_like(pressure), V)

    np.testing.assert_array_equal(plain, base)  # G = 1e-3 is not above the threshold
    assert (eroded > base).all()


def test_scalar_thermochemistry_gives_the_lane_constants(lanes):
    _, batch = lanes
    keep = [i for i, f in enumerate(batch.lane_features) if pb.THERMO_SCALAR in f]
    sub = batch.select(keep)
    V = lane_view(sub.namespace(np))

    source_temperature, k = propellant.gas_properties(np, np.full((len(keep), 3), 2e6), V)

    np.testing.assert_array_equal(source_temperature[:, 0], sub.arrays["source_temperature"])
    np.testing.assert_array_equal(k[:, 0], sub.arrays["gamma"])
