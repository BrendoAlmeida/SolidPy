import numpy as np
import pytest

import golden_corpus as gc
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch
from solidpy.batch import problem as pb
from solidpy.batch.kernels import geometry

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
