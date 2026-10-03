import inspect
import math

import numpy as np
import pytest

import golden_corpus as gc
from solidpy import Burn, BurnSimulation, Environment, Grain, Motor, Propellant
from solidpy.backends import Capabilities
from solidpy.batch import FEATURES, ProblemBatch
from solidpy.batch import problem as pb


def make_stack(geometries=("tubular",), **propellant_kwargs):
    grains = [Grain(0.035, 0.015, initial_height=0.12, geometry=g) for g in geometries]
    motor = Motor(
        grains, chamber_inner_radius=0.037, chamber_length=0.12 * len(grains) + 0.02,
        nozzle_throat_radius=0.008, nozzle_exit_radius=0.018, nozzle_angle=0.2618,
    )
    kwargs = dict(burn_rate_a=7.36, burn_rate_n=0.32)
    kwargs.update(propellant_kwargs)
    return motor, Propellant(1.1308, 0.04197, 1720.0, density=1879.0, **kwargs)


def pack(motor, propellant, **settings):
    return ProblemBatch.from_objects(motor, propellant, settings=settings or None)


def test_a_plain_lane_needs_only_the_basic_features():
    batch = pack(*make_stack())

    assert len(batch) == 1
    assert batch.lane_features[0] == {
        pb.TUBULAR_GRAIN, pb.BURN_RATE_POWER_LAW, pb.THERMO_SCALAR, pb.TAIL_OFF_NUMERICAL,
    }
    assert batch.lane_features[0] <= set(FEATURES)


def test_arrays_hold_the_values_of_the_objects_and_pad_the_grain_axis():
    motor, propellant = make_stack(("tubular", "star"), erosive_burning_coefficient=1e-5, erosive_alpha=40.0)
    environment = Environment(altitude=1000.0)

    batch = ProblemBatch.from_objects(
        [motor, make_stack()[0]], [propellant, propellant], environment, {"eta_c": 0.9, "max_step_size": 0.02}
    )

    a = batch.arrays
    assert batch.g_max == 2 and a["grain_valid"].tolist() == [[True, True], [True, False]]
    assert a["is_star"].tolist() == [[False, True], [False, False]]
    assert a["outer_radius"][0, 0] == 0.035 and a["height0"][0, 1] == 0.12
    assert a["burnout_depth"][0, 0] == motor.grains[0].burnout_regression_m
    assert a["n_valid_grains"].tolist() == [2.0, 1.0] and batch.n_grains.tolist() == [2, 1]
    assert a["chamber_volume"][0] == motor.chamber_volume
    assert a["throat_area"][1] == motor.nozzle_throat_area
    assert a["ambient_pressure"][0] == environment.atmospheric_pressure
    assert a["divergence_factor"][0] == pytest.approx(0.5 * (1.0 + math.cos(0.2618)), rel=1e-15)
    assert a["eta_c"].tolist() == [0.9, 0.9] and a["max_step_size"].tolist() == [0.02, 0.02]
    assert a["erosive_coefficient"].tolist() == [1e-5, 1e-5] and a["erosive_alpha"].tolist() == [40.0, 40.0]
    assert a["source_temperature"][0] == pytest.approx(1720.0 * 0.9**2, rel=1e-15)
    assert a["igniter_temperature"][0] == a["source_temperature"][0]


def test_one_object_is_broadcast_to_every_lane_and_settings_can_differ_per_lane():
    motor, propellant = make_stack()

    batch = ProblemBatch.from_objects(motor, propellant, settings=[{"rtol": 1e-6}, {"rtol": 1e-9}, {}])

    assert len(batch) == 3
    default_rtol = inspect.signature(BurnSimulation.__init__).parameters["rtol"].default
    assert batch.arrays["rtol"].tolist() == [1e-6, 1e-9, default_rtol]
    assert batch.arrays["chamber_volume"].tolist() == [motor.chamber_volume] * 3


def test_the_exit_mach_satisfies_the_area_mach_relation():
    batch = ProblemBatch.from_objects(*make_stack())
    k, mach, ratio = batch.arrays["gamma"][0], batch.arrays["exit_mach"][0], batch.arrays["expansion_ratio"][0]

    area_ratio = ((k + 1) / 2) ** (-(k + 1) / (2 * (k - 1))) * (1 + (k - 1) / 2 * mach**2) ** ((k + 1) / (2 * (k - 1))) / mach

    assert mach > 1.0
    assert area_ratio == pytest.approx(ratio, rel=1e-9)


def test_initial_state_is_the_ambient_gas_fill():
    motor, propellant = make_stack(("tubular", "tubular", "star"))
    batch = pack(motor, propellant, eta_c=0.95)
    simulation = BurnSimulation(motor.grains[0], motor, propellant, **{"max_step_size": 0.03, "eta_c": 0.95})

    state = batch.initial_state()

    assert state.shape == (1, 3 + 7)
    assert state[0, 0] == simulation.result["metrics"]["gas_mass_initial_kg"]
    assert state[0, 1] == state[0, 0] * simulation.initial_gas_temperature_k
    assert not state[0, 2:].any()


def test_select_keeps_the_requested_lanes_in_order():
    motor, propellant = make_stack()
    batch = ProblemBatch.from_objects(motor, propellant, settings=[{"rtol": 1e-6}, {"rtol": 1e-7}, {"rtol": 1e-9}])

    sub = batch.select([2, 0])

    assert sub.arrays["rtol"].tolist() == [1e-9, 1e-6]
    assert [s["rtol"] for s in sub.settings] == [1e-9, 1e-6]
    assert sub.motors == [motor, motor]


def _thermo_table(motor, propellant):
    propellant.load_thermo_table([[1e5, 900.0, 1.13, 1700.0], [1e6, 900.0, 1.13, 1720.0], [5e6, 900.0, 1.12, 1740.0]])
    return motor, propellant, {}


def _instance_override(motor, propellant):
    original = propellant.evaluate_burn_rate
    propellant.evaluate_burn_rate = lambda p, g=0.0, _o=original: 1.05 * _o(p, g)  # what Robustness does
    return motor, propellant, {}


def _subclass_propellant(motor, propellant):
    class Custom(Propellant):
        pass

    custom = Custom(1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36, burn_rate_n=0.32)
    return motor, custom, {}


def _unknown_geometry(motor, propellant):
    motor.grains[0].geometry = "finocyl"
    return motor, propellant, {}


def _grain_override(motor, propellant):
    motor.grains[0].evaluate_burn_area = lambda r, update_state=False: 0.0
    return motor, propellant, {}


CASES = {
    "tabulated burn rate": (lambda m, p: (m, Propellant(1.13, 0.04, 1700.0, density=1800.0,
                                                        interpolation_list="data/burnrate/KNSB3.csv"), {}),
                            {pb.BURN_RATE_TABLE}, {pb.BURN_RATE_POWER_LAW}),
    "thermochemistry table": (_thermo_table, {pb.THERMO_TABLE}, {pb.THERMO_SCALAR}),
    "scalar igniter": (lambda m, p: (m, p, {"igniter_mass_flow": 0.01, "igniter_burn_time": 0.5}),
                       {pb.IGNITER_SCALAR}, set()),
    "table igniter": (lambda m, p: (m, p, {"igniter_mass_flow": [[0.0, 0.0], [0.2, 0.01], [0.4, 0.0]]}),
                      {pb.IGNITER_TABLE}, set()),
    "callable igniter": (lambda m, p: (m, p, {"igniter_mass_flow": lambda t: 0.01, "igniter_burn_time": 0.5}),
                         {pb.IGNITER_CALLABLE}, set()),
    "scalar activation": (lambda m, p: (m, p, {"burn_area_activation": 0.5}), {pb.ACTIVATION_SCALAR}, set()),
    "table activation": (lambda m, p: (m, p, {"burn_area_activation": [[0.0, 0.2], [0.3, 1.0]]}),
                         {pb.ACTIVATION_TABLE}, set()),
    "callable activation": (lambda m, p: (m, p, {"burn_area_activation": lambda t: 0.5}),
                            {pb.ACTIVATION_CALLABLE}, set()),
    "ignition ramp": (lambda m, p: (m, p, {"ignition_ramp_time": 0.1}), {pb.IGNITION_RAMP}, set()),
    "analytical tail-off": (lambda m, p: (m, p, {"tail_off_method": "Analytical"}),
                            {pb.TAIL_OFF_ANALYTICAL}, {pb.TAIL_OFF_NUMERICAL}),
    "omitted tail-off": (lambda m, p: (m, p, {"tail_off_evaluation": False}), {pb.TAIL_OFF_OMITTED}, set()),
    "erosive burning": (lambda m, p: (m, Propellant(1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36,
                                                    burn_rate_n=0.32, erosive_burning_coefficient=1e-5), {}),
                        {pb.EROSIVE_BURNING}, set()),
    "instance override of the burn rate": (_instance_override, {pb.INSTANCE_OVERRIDE}, set()),
    "propellant subclass": (_subclass_propellant, {pb.CUSTOM_CLASS}, set()),
    "unknown grain geometry": (_unknown_geometry, {pb.UNKNOWN_GEOMETRY}, {pb.TUBULAR_GRAIN, pb.STAR_GRAIN}),
    "instance override on a grain": (_grain_override, {pb.INSTANCE_OVERRIDE}, set()),
}


@pytest.mark.parametrize("name", list(CASES))
def test_every_unsupported_or_special_feature_is_detected(name):
    build, expected, absent = CASES[name]
    motor, propellant, settings = build(*make_stack())

    features = pack(motor, propellant, **settings).lane_features[0]

    assert expected <= features, features
    assert not (absent & features), features


def test_ends_burn_and_star_are_features_of_the_grains():
    grains = [Grain(0.035, 0.015, initial_height=0.12, geometry="star", ends_burn=True),
              Grain(0.035, 0.015, initial_height=0.12)]
    motor = Motor(grains, chamber_inner_radius=0.037, chamber_length=0.3, nozzle_throat_radius=0.008,
                  nozzle_exit_radius=0.018)

    features = pack(motor, make_stack()[1]).lane_features[0]

    assert {pb.STAR_GRAIN, pb.TUBULAR_GRAIN, pb.ENDS_BURN} <= features


def test_values_the_kernels_cannot_reproduce_are_nan_not_plausible_numbers():
    motor, propellant, _ = _thermo_table(*make_stack())
    tabulated_rate = Propellant(1.13, 0.04, 1700.0, density=1800.0, interpolation_list="data/burnrate/KNSB3.csv")

    thermo = pack(motor, propellant).arrays
    rate = pack(make_stack()[0], tabulated_rate).arrays

    for name in ("gamma", "source_temperature", "exit_mach"):
        assert np.isnan(thermo[name]).all(), name
    for name in ("burn_rate_a", "burn_rate_n"):
        assert np.isnan(rate[name]).all(), name


def test_capabilities_decide_which_lanes_are_unsupported():
    motor, propellant = make_stack()
    batch = ProblemBatch.from_objects(
        motor, propellant, settings=[{}, {"igniter_mass_flow": 0.01, "igniter_burn_time": 0.5}]
    )
    capabilities = Capabilities({
        pb.TUBULAR_GRAIN: "supported", pb.BURN_RATE_POWER_LAW: "supported", pb.THERMO_SCALAR: "supported",
        pb.TAIL_OFF_NUMERICAL: "supported",
    })

    assert batch.unsupported(capabilities) == [[], [pb.IGNITER_SCALAR]]


@pytest.mark.parametrize(
    "settings, error, message",
    [
        ({"eta_c": 1.2}, ValueError, "eta_c must be a finite real number in"),
        ({"discharge_coefficient": 0.0}, ValueError, "discharge_coefficient"),
        ({"max_step_size": -0.01}, ValueError, "max_step_size must be a finite positive"),
        ({"rtol": float("nan")}, ValueError, "rtol must be a finite positive"),
        ({"igniter_burn_time": -1.0}, ValueError, "igniter_burn_time must be a finite non-negative"),
        ({"tail_off_method": "magic"}, ValueError, "tail_off_method must be numerical or analytical"),
        ({"burn_area_activation": 1.5}, ValueError, "must not exceed 1"),
        ({"igniter_mass_flow": [[0.0, 0.0], [0.0, 1.0]]}, ValueError, "increasing non-negative time"),
        ({"igniter_temperature": -5.0}, ValueError, "igniter_temperature must be a finite positive"),
        ({"not_a_setting": 1}, TypeError, r"unexpected simulation setting\(s\) \['not_a_setting'\]"),
    ],
)
def test_invalid_inputs_raise_the_scalar_errors_prefixed_with_the_lane(settings, error, message):
    motor, propellant = make_stack()

    with pytest.raises(error, match=rf"lane 1: .*{message}"):
        ProblemBatch.from_objects(motor, propellant, settings=[{}, settings])


def test_a_propellant_without_burn_rate_data_is_rejected_like_the_scalar_code():
    motor, _ = make_stack()
    bare = Propellant(1.1308, 0.04197, 1720.0, density=1879.0)

    with pytest.raises(TypeError, match="lane 0: Missing arguments"):
        ProblemBatch.from_objects(motor, bare)


def test_mismatched_lane_counts_and_a_small_g_max_are_errors():
    motor, propellant = make_stack(("tubular", "tubular"))

    with pytest.raises(ValueError, match="motors has 2 entries for 3 lanes"):
        ProblemBatch.from_objects([motor, motor], propellant, settings=[{}, {}, {}])
    with pytest.raises(ValueError, match="g_max=1 is smaller"):
        ProblemBatch.from_objects(motor, propellant, g_max=1)


def _snapshot(obj):
    return {k: v for k, v in vars(obj).items() if isinstance(v, (bool, int, float, str, tuple, type(None)))}


def test_packing_does_not_modify_the_objects():
    motor, propellant = make_stack(("tubular", "star"))
    before = [_snapshot(obj) for obj in (motor, propellant, *motor.grains)]

    pack(motor, propellant, eta_c=0.9)

    assert [_snapshot(obj) for obj in (motor, propellant, *motor.grains)] == before


@pytest.fixture(scope="module")
def corpus_lanes():
    cases = gc.load_corpus()["cases"]
    packed = []
    for case in cases:
        grain, motor, propellant, environment, kwargs = gc.build_objects(case)
        packed.append((case, ProblemBatch.from_objects(motor, propellant, environment, kwargs)))
    return packed


TAG_TO_FEATURE = {
    "tubular": pb.TUBULAR_GRAIN, "star": pb.STAR_GRAIN, "ends_burn": pb.ENDS_BURN,
    "power_law": pb.BURN_RATE_POWER_LAW, "burn_rate_table": pb.BURN_RATE_TABLE, "erosive": pb.EROSIVE_BURNING,
    "scalar_thermo": pb.THERMO_SCALAR, "thermo_table": pb.THERMO_TABLE,
    "igniter_scalar": pb.IGNITER_SCALAR, "igniter_table": pb.IGNITER_TABLE, "igniter_callable": pb.IGNITER_CALLABLE,
    "activation_scalar": pb.ACTIVATION_SCALAR, "activation_table": pb.ACTIVATION_TABLE,
    "activation_callable": pb.ACTIVATION_CALLABLE, "ramp": pb.IGNITION_RAMP,
    "tail_off_numerical": pb.TAIL_OFF_NUMERICAL, "tail_off_analytical": pb.TAIL_OFF_ANALYTICAL,
    "tail_off_omitted": pb.TAIL_OFF_OMITTED,
}


def test_features_of_every_corpus_design_agree_with_its_tags(corpus_lanes):
    wrong = []
    for case, batch in corpus_lanes:
        features = batch.lane_features[0]
        expected = {feature for tag, feature in TAG_TO_FEATURE.items() if tag in case["tags"]}
        if "mixed_geometry" in case["tags"]:
            expected |= {pb.TUBULAR_GRAIN, pb.STAR_GRAIN}
        if not expected <= features or (pb.STAR_GRAIN in features) != (
            "star" in case["tags"] or "mixed_geometry" in case["tags"]
        ):
            wrong.append(case["id"])
        assert not (features & pb.REFERENCE_ONLY), case["id"]
    assert wrong == []


def test_every_corpus_design_packs_with_finite_values_where_supported(corpus_lanes):
    for case, batch in corpus_lanes:
        a = batch.arrays
        assert (a["grain_valid"].sum(axis=1) == batch.n_grains).all(), case["id"]
        assert np.isfinite(a["chamber_volume"]).all() and (a["free_volume"] > 0).all(), case["id"]
        scalar = pb.THERMO_SCALAR in batch.lane_features[0]
        assert np.isfinite(a["exit_mach"]).all() == scalar, case["id"]
        assert np.isfinite(batch.initial_state()).all() == scalar, case["id"]


def test_packed_inputs_match_the_stored_reference_initial_state(corpus_lanes):
    reference = gc.load_reference()["records"]
    checked = 0
    for case, batch in corpus_lanes:
        if pb.THERMO_SCALAR not in batch.lane_features[0]:
            continue
        stored = reference[case["id"]]["metrics"]["gas_mass_initial_kg"]
        assert batch.initial_state()[0, 0] == pytest.approx(stored, rel=1e-13), case["id"]
        checked += 1
    assert checked > 300


def test_a_custom_grain_without_a_geometry_is_routed_not_a_crash():
    motor, propellant = make_stack()

    class Bare:
        outer_radius, initial_inner_radius, initial_height, ends_burn = 0.035, 0.015, 0.12, False

    motor.grains = [Bare()]
    features = pb.required_features(motor, propellant, pb.SETTING_DEFAULTS)

    assert {pb.CUSTOM_CLASS, pb.UNKNOWN_GEOMETRY} <= features
    assert pb.REFERENCE_ONLY & features


def test_solve_cache_is_an_accepted_setting_and_does_not_change_the_packed_values():
    motor, propellant = make_stack()

    on = ProblemBatch.from_objects(motor, propellant, settings={"solve_cache": True})
    off = ProblemBatch.from_objects(motor, propellant, settings={"solve_cache": False})

    assert off.settings[0]["solve_cache"] is False
    for name in on.arrays:
        np.testing.assert_array_equal(on.arrays[name], off.arrays[name])


def test_an_empty_batch_is_a_clear_error():
    with pytest.raises(ValueError, match="at least one lane is required"):
        ProblemBatch.from_objects([], [])


def test_the_grain_axis_can_be_padded_without_changing_the_lanes():
    motor, propellant = make_stack(("tubular", "star"))
    batch = ProblemBatch.from_objects(motor, propellant, settings=[{}, {"eta_c": 0.9}])

    wide = batch.with_g_max(8)

    assert wide.g_max == 8 and batch.with_g_max(2) is batch
    assert wide.arrays["grain_valid"].sum(axis=1).tolist() == [2, 2]
    for name in pb.GRAIN_FIELDS:
        np.testing.assert_array_equal(wide.arrays[name][:, :2], batch.arrays[name])
    assert wide.arrays["grain_valid"][:, 2:].sum() == 0 and np.isfinite(wide.arrays["outer_radius"]).all()
    np.testing.assert_array_equal(wide.arrays["eta_c"], batch.arrays["eta_c"])
    assert wide.initial_state().shape == (2, 8 + 7)
    with pytest.raises(ValueError, match="smaller than the current grain axis"):
        batch.with_g_max(1)


def test_select_may_repeat_a_lane_to_fill_a_compiled_bucket():
    motor, propellant = make_stack()
    batch = ProblemBatch.from_objects(motor, propellant, settings=[{"rtol": 1e-6}, {"rtol": 1e-9}])

    padded = batch.select([0, 1, 0, 0])

    assert padded.arrays["rtol"].tolist() == [1e-6, 1e-9, 1e-6, 1e-6]
