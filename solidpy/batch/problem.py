# -*- coding: utf-8 -*-
"""Pack scalar SolidPy objects into a structure-of-arrays batch (one lane per motor).

Packing is strict: a lane carries the set of features it needs (``lane_features``) and a value that the
batched kernels cannot reproduce is stored as NaN instead of a plausible number. Anything the packer does
not recognise is reported as a feature the backends do not support, so a lane is routed to the reference
solver rather than silently simulated with the wrong physics.

Array schema (leading axis ``B`` = lane, ``G`` = padded grain axis):

* grain arrays ``[B, G]``: ``outer_radius``, ``inner_radius0``, ``height0``, ``ends_burn``, ``is_star``,
  ``n_points``, ``epsilon``, ``slot_fraction``, ``slot_floor_radius``, ``slot_floor_depth``,
  ``burnout_depth``, ``grain_valid``. Padded grains have ``grain_valid`` false and harmless geometry.
* lane arrays ``[B]``: chamber, propellant, nozzle, efficiency and solver settings (see ``LANE_FIELDS``).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence

import numpy as np

from ..Burn import Burn, BurnSimulation
from ..Environment import Environment
from ..Grain import Grain
from ..Motor import Motor
from ..Propellant import Propellant

# Lane features. The vocabulary matches the capability matrix of the architecture document (4.6).
TUBULAR_GRAIN = "tubular_grain"
STAR_GRAIN = "star_grain"
ENDS_BURN = "ends_burn"
BURN_RATE_POWER_LAW = "burn_rate_power_law"
BURN_RATE_TABLE = "burn_rate_table"
EROSIVE_BURNING = "erosive_burning"
THERMO_SCALAR = "thermo_scalar"
THERMO_TABLE = "thermo_table"
IGNITER_SCALAR = "igniter_scalar"
IGNITER_TABLE = "igniter_table"
IGNITER_CALLABLE = "igniter_callable"
ACTIVATION_SCALAR = "activation_scalar"
ACTIVATION_TABLE = "activation_table"
ACTIVATION_CALLABLE = "activation_callable"
IGNITION_RAMP = "ignition_ramp"
TAIL_OFF_NUMERICAL = "tail_off_numerical"
TAIL_OFF_ANALYTICAL = "tail_off_analytical"
TAIL_OFF_OMITTED = "tail_off_omitted"
INSTANCE_OVERRIDE = "instance_override"
CUSTOM_CLASS = "custom_class"
UNKNOWN_GEOMETRY = "unknown_geometry"

FEATURES = (
    TUBULAR_GRAIN, STAR_GRAIN, ENDS_BURN, BURN_RATE_POWER_LAW, BURN_RATE_TABLE, EROSIVE_BURNING,
    THERMO_SCALAR, THERMO_TABLE, IGNITER_SCALAR, IGNITER_TABLE, IGNITER_CALLABLE, ACTIVATION_SCALAR,
    ACTIVATION_TABLE, ACTIVATION_CALLABLE, IGNITION_RAMP, TAIL_OFF_NUMERICAL, TAIL_OFF_ANALYTICAL,
    TAIL_OFF_OMITTED, INSTANCE_OVERRIDE, CUSTOM_CLASS, UNKNOWN_GEOMETRY,
)

#: Features no batched backend will ever run: the reference solver handles those lanes.
REFERENCE_ONLY = frozenset({INSTANCE_OVERRIDE, CUSTOM_CLASS, UNKNOWN_GEOMETRY})

GRAIN_FIELDS = (
    "outer_radius", "inner_radius0", "height0", "ends_burn", "is_star", "n_points", "epsilon",
    "slot_fraction", "slot_floor_radius", "slot_floor_depth", "burnout_depth", "grain_valid",
)
LANE_FIELDS = (
    "chamber_volume", "free_volume", "propellant_volume", "throat_area", "exit_area", "expansion_ratio",
    "divergence_factor", "exit_mach", "density", "gas_constant", "gamma", "source_temperature",
    "eta_c", "eta_cf", "discharge_coefficient", "ambient_pressure", "burn_rate_a", "burn_rate_n",
    "erosive_coefficient", "erosive_alpha", "n_valid_grains", "igniter_temperature", "igniter_burn_time",
    "ignition_ramp_time", "max_step_size", "rtol", "atol", "burn_timeout_s", "tail_off_timeout_s",
)

# Geometry of a padded (non-existent) grain. It never burns and holds no volume; the values only need to
# keep every kernel finite.
_PADDING = dict(
    outer_radius=1.0, inner_radius0=0.5, height0=1.0, ends_burn=False, is_star=False, n_points=1.0,
    epsilon=0.1, slot_fraction=0.5, slot_floor_radius=0.75, slot_floor_depth=0.25, burnout_depth=0.5,
    grain_valid=False,
)
_BOOL_GRAIN_FIELDS = ("ends_burn", "is_star", "grain_valid")


def _setting_defaults() -> Dict[str, Any]:
    """Defaults of the ``BurnSimulation`` keyword settings, read from its signature so they cannot drift."""
    skip = {"self", "grain", "motor", "propellant", "environment", "solve_cache"}
    parameters = inspect.signature(BurnSimulation.__init__).parameters
    return {name: parameter.default for name, parameter in parameters.items() if name not in skip}


SETTING_DEFAULTS = _setting_defaults()


def _instance_overrides(obj) -> List[str]:
    """Names of methods that were replaced on the instance (e.g. ``Robustness`` patches ``evaluate_burn_rate``)."""
    return sorted(
        name for name, value in vars(obj).items() if callable(value) and callable(getattr(type(obj), name, None))
    )


def _source_kind(source, callable_feature, scalar_feature, table_feature) -> Optional[str]:
    if source is None:
        return None
    if callable(source):
        return callable_feature
    if np.isscalar(source):
        return scalar_feature
    return table_feature


def required_features(motor, propellant, settings: Mapping[str, Any]) -> FrozenSet[str]:
    """Return the set of features one motor needs, so a router can compare it with a backend's capabilities."""
    features = set()
    grains = list(motor.grains)
    for grain in grains:
        if type(grain) is not Grain:
            features.add(CUSTOM_CLASS)
        features.add(STAR_GRAIN if grain.geometry == "star" else TUBULAR_GRAIN)
        if grain.geometry not in ("tubular", "star"):
            features.add(UNKNOWN_GEOMETRY)
        if grain.ends_burn:
            features.add(ENDS_BURN)
    if type(motor) is not Motor or type(propellant) is not Propellant:
        features.add(CUSTOM_CLASS)
    if _instance_overrides(propellant) or _instance_overrides(motor) or any(_instance_overrides(g) for g in grains):
        features.add(INSTANCE_OVERRIDE)

    if "interpolation_list" in propellant.__dict__:
        features.add(BURN_RATE_TABLE)
    elif "burn_rate_a" in propellant.__dict__ and "burn_rate_n" in propellant.__dict__:
        features.add(BURN_RATE_POWER_LAW)
    else:
        raise TypeError(
            "Missing arguments. You must pass either an `interpolation_list` path or scalar ballistic "
            "coefficients `burn_rate_a` and `burn_rate_n` arguments to Propellant class "
        )
    if getattr(propellant, "erosive_burning_coefficient", 0.0) > 0.0:
        features.add(EROSIVE_BURNING)

    tabulated = any(
        getattr(propellant, name, None) is not None
        for name in ("_thermo_table", "_cstar_func", "_gamma_func", "_temperature_func", "_cea_obj", "cea_formulation")
    )
    features.add(THERMO_TABLE if tabulated else THERMO_SCALAR)

    igniter = _source_kind(settings["igniter_mass_flow"], IGNITER_CALLABLE, IGNITER_SCALAR, IGNITER_TABLE)
    if igniter:
        features.add(igniter)
    activation = _source_kind(
        settings["burn_area_activation"], ACTIVATION_CALLABLE, ACTIVATION_SCALAR, ACTIVATION_TABLE
    )
    if activation:
        features.add(activation)
    elif settings["ignition_ramp_time"] > 0.0:
        features.add(IGNITION_RAMP)

    features.add(TAIL_OFF_ANALYTICAL if settings["tail_off_method"] == "analytical" else TAIL_OFF_NUMERICAL)
    if not settings["tail_off_evaluation"]:
        features.add(TAIL_OFF_OMITTED)
    return frozenset(features)


def _resolve_settings(lane: int, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(overrides) - set(SETTING_DEFAULTS))
    if unknown:
        raise TypeError(f"lane {lane}: unexpected simulation setting(s) {unknown}")
    settings = {**SETTING_DEFAULTS, **overrides}
    for name in ("max_step_size", "rtol", "atol", "burn_timeout_s", "tail_off_timeout_s"):
        settings[name] = BurnSimulation._positive_setting(name, settings[name])
    for name in ("igniter_burn_time", "ignition_ramp_time"):
        settings[name] = BurnSimulation._nonnegative_setting(name, settings[name])
    settings["tail_off_method"] = str(settings["tail_off_method"]).lower()
    if settings["tail_off_method"] not in {"numerical", "analytical"}:
        raise ValueError("tail_off_method must be numerical or analytical")
    BurnSimulation._validate_source_profiles(
        SimpleNamespace(
            igniter_burn_time=settings["igniter_burn_time"],
            ignition_ramp_time=settings["ignition_ramp_time"],
            igniter_mass_flow=settings["igniter_mass_flow"],
            burn_area_activation=settings["burn_area_activation"],
            _nonnegative_setting=BurnSimulation._nonnegative_setting,
        )
    )
    return settings


def _broadcast(values, count: Optional[int], name: str) -> List[Any]:
    items = list(values) if isinstance(values, (list, tuple)) else [values]
    if count is None or len(items) == count:
        return items
    if len(items) == 1:
        return items * count
    raise ValueError(f"{name} has {len(items)} entries for {count} lanes")


def _grain_row(grain) -> Dict[str, Any]:
    outer, inner = float(grain.outer_radius), float(grain.initial_inner_radius)
    floor_radius = inner + float(grain.slot_fraction) * (outer - inner)
    return dict(
        outer_radius=outer, inner_radius0=inner, height0=float(grain.initial_height),
        ends_burn=bool(grain.ends_burn), is_star=grain.geometry == "star", n_points=float(grain.n_points),
        epsilon=float(grain.epsilon), slot_fraction=float(grain.slot_fraction), slot_floor_radius=floor_radius,
        slot_floor_depth=max(outer - floor_radius, 0.0), burnout_depth=float(grain.burnout_regression_m),
        grain_valid=True,
    )


def _lane_values(motor, propellant, environment, settings: Mapping[str, Any], features: FrozenSet[str]):
    """Scalar inputs of one lane. Values the batched kernels cannot reproduce are NaN."""
    burn = Burn(
        motor.grains[0], motor, propellant, environment, eta_c=settings["eta_c"], eta_Cf=settings["eta_Cf"],
        discharge_coefficient=settings["discharge_coefficient"],
    )
    ambient = float(burn.environment_pressure)
    scalar_thermo = THERMO_SCALAR in features
    for name, value in (("propellant_density", propellant.density), ("gas_constant", propellant.products_constant),
                        ("combustion_temperature", propellant.combustion_temperature)):
        BurnSimulation._positive_setting(name, value)
    gamma = float(propellant.specific_heat_ratio)
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("specific_heat_ratio must be finite and greater than one")
    source_temperature = float(propellant.Tc_at_pressure(ambient)) * burn.eta_c ** 2
    BurnSimulation._positive_setting("effective_combustion_temperature", source_temperature)

    igniter_temperature = settings["igniter_temperature"]
    igniter_temperature = BurnSimulation._positive_setting(
        "igniter_temperature", source_temperature if igniter_temperature is None else igniter_temperature
    )
    angle = motor.nozzle_angle
    divergence = 1.0 if angle is None or angle <= 0.0 else 0.5 * (1.0 + math.cos(float(angle)))
    nan = float("nan")
    power_law = BURN_RATE_POWER_LAW in features
    return dict(
        chamber_volume=float(motor.chamber_volume), free_volume=float(motor.free_volume),
        propellant_volume=float(motor.propellant_volume), throat_area=float(motor.nozzle_throat_area),
        exit_area=float(motor.nozzle_exit_area), expansion_ratio=float(motor.expansion_ratio),
        divergence_factor=divergence,
        exit_mach=float(burn.evaluate_exit_mach()) if scalar_thermo else nan,
        density=float(propellant.density), gas_constant=float(propellant.products_constant),
        gamma=gamma if scalar_thermo else nan,
        source_temperature=source_temperature if scalar_thermo else nan,
        eta_c=burn.eta_c, eta_cf=burn.eta_Cf, discharge_coefficient=burn.discharge_coefficient,
        ambient_pressure=ambient,
        burn_rate_a=float(propellant.burn_rate_a) if power_law else nan,
        burn_rate_n=float(propellant.burn_rate_n) if power_law else nan,
        erosive_coefficient=float(getattr(propellant, "erosive_burning_coefficient", 0.0)),
        erosive_alpha=float(getattr(propellant, "erosive_alpha", 35.0)),
        n_valid_grains=float(len(motor.grains)), igniter_temperature=igniter_temperature,
        igniter_burn_time=settings["igniter_burn_time"], ignition_ramp_time=settings["ignition_ramp_time"],
        max_step_size=settings["max_step_size"], rtol=settings["rtol"], atol=settings["atol"],
        burn_timeout_s=settings["burn_timeout_s"], tail_off_timeout_s=settings["tail_off_timeout_s"],
    )


@dataclass
class ProblemBatch:
    """Many independent motors as arrays. Build one with :meth:`from_objects`."""

    arrays: Dict[str, np.ndarray]
    lane_features: List[FrozenSet[str]]
    n_grains: np.ndarray
    motors: List[Any] = field(repr=False)
    propellants: List[Any] = field(repr=False)
    environments: List[Any] = field(repr=False)
    settings: List[Dict[str, Any]] = field(repr=False)

    def __len__(self) -> int:
        return len(self.lane_features)

    @property
    def g_max(self) -> int:
        return int(self.arrays["grain_valid"].shape[1])

    @classmethod
    def from_objects(
        cls,
        motors,
        propellants,
        environments=None,
        settings: Optional[Any] = None,
        g_max: Optional[int] = None,
    ) -> "ProblemBatch":
        """Pack motors, propellants and environments (one per lane, or one broadcast to every lane).

        ``settings`` holds ``BurnSimulation`` keyword settings: one mapping for every lane or one per lane.
        Invalid inputs raise the same errors as ``BurnSimulation`` would, prefixed with the lane index.
        """
        given = {"motors": motors, "propellants": propellants, "environments": environments, "settings": settings}
        count = max((len(value) for value in given.values() if isinstance(value, (list, tuple))), default=1)
        motor_list = _broadcast(motors, count, "motors")
        propellant_list = _broadcast(propellants, count, "propellants")
        environment_list = _broadcast(Environment() if environments is None else environments, count, "environments")
        setting_list = _broadcast({} if settings is None else settings, count, "settings")

        resolved, features, rows = [], [], []
        for lane in range(count):
            try:
                lane_settings = _resolve_settings(lane, setting_list[lane])
                lane_feature_set = required_features(motor_list[lane], propellant_list[lane], lane_settings)
                rows.append(_lane_values(motor_list[lane], propellant_list[lane], environment_list[lane],
                                         lane_settings, lane_feature_set))
            except (ValueError, TypeError) as exc:
                if str(exc).startswith(f"lane {lane}:"):
                    raise
                raise type(exc)(f"lane {lane}: {exc}") from exc
            resolved.append(lane_settings)
            features.append(lane_feature_set)

        n_grains = np.asarray([len(motor.grains) for motor in motor_list], dtype=int)
        needed = int(n_grains.max())
        if g_max is None:
            g_max = needed
        elif g_max < needed:
            raise ValueError(f"g_max={g_max} is smaller than the largest grain count {needed}")

        arrays: Dict[str, np.ndarray] = {
            name: np.full((count, g_max), _PADDING[name], dtype=bool if name in _BOOL_GRAIN_FIELDS else float)
            for name in GRAIN_FIELDS
        }
        for lane, motor in enumerate(motor_list):
            for slot, grain in enumerate(motor.grains):
                for name, value in _grain_row(grain).items():
                    arrays[name][lane, slot] = value
        for name in LANE_FIELDS:
            arrays[name] = np.asarray([row[name] for row in rows], dtype=float)
        return cls(arrays, features, n_grains, motor_list, propellant_list, environment_list, resolved)

    def namespace(self, xp) -> Dict[str, Any]:
        """The arrays as ``xp`` arrays, keyed by name, ready to pass to the kernels."""
        return {name: xp.asarray(array) for name, array in self.arrays.items()}

    def initial_state(self) -> np.ndarray:
        """Initial state ``[B, G + 7]``: gas mass, thermal inventory, regressions, five zero integrals."""
        a = self.arrays
        mass = a["ambient_pressure"] * a["free_volume"] / (a["gas_constant"] * a["source_temperature"])
        state = np.zeros((len(self), self.g_max + 7))
        state[:, 0] = mass
        state[:, 1] = mass * a["source_temperature"]
        return state

    def unsupported(self, capabilities) -> List[List[str]]:
        """For every lane, the needed features that ``capabilities`` does not fully support."""
        return [capabilities.missing(sorted(features)) for features in self.lane_features]

    def select(self, indices: Sequence[int]) -> "ProblemBatch":
        """A sub-batch with the given lanes, in the given order."""
        index = np.asarray(indices, dtype=int)
        return ProblemBatch(
            arrays={name: array[index] for name, array in self.arrays.items()},
            lane_features=[self.lane_features[i] for i in index],
            n_grains=self.n_grains[index],
            motors=[self.motors[i] for i in index],
            propellants=[self.propellants[i] for i in index],
            environments=[self.environments[i] for i in index],
            settings=[self.settings[i] for i in index],
        )
