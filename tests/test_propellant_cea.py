# -*- coding: utf-8 -*-

import builtins
import sys
import types

import pytest

from solidpy import Burn, Environment, Grain, Motor, Propellant
from solidpy.Propellant import FT_TO_M, PA_PER_PSIA, RANKINE_TO_KELVIN


def make_static_propellant(**kwargs):
    defaults = {
        "specific_heat_ratio": 1.2,
        "products_molecular_mass": 30e-3,
        "combustion_temperature": 2800.0,
        "density": 1700.0,
        "burn_rate_a": 5.0,
        "burn_rate_n": 0.3,
    }
    defaults.update(kwargs)
    return Propellant(**defaults)


def test_static_getters_keep_legacy_values():
    propellant = make_static_propellant()

    assert propellant.get_cstar(5.0e6) == pytest.approx(propellant.cstar)
    assert propellant.cstar_at_pressure(5.0e6) == pytest.approx(propellant.cstar)
    assert propellant.get_gamma(5.0e6) == pytest.approx(propellant.specific_heat_ratio)
    assert propellant.k_at_pressure(5.0e6) == pytest.approx(propellant.specific_heat_ratio)


def test_static_cstar_can_be_supplied_directly():
    propellant = make_static_propellant(cstar=1450.0)

    assert propellant.cstar == pytest.approx(1450.0)
    assert propellant.get_cstar(1.0e6) == pytest.approx(1450.0)


def test_cea_formulation_requires_optional_rocketcea(monkeypatch):
    for module_name in list(sys.modules):
        if module_name.startswith("rocketcea"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("rocketcea"):
            raise ImportError("blocked rocketcea")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="pip install rocketcea"):
        make_static_propellant(cea_formulation="name TEST H 2 O 2 wt%=100.0")


def test_cea_formulation_builds_interpolators_with_si_conversions(monkeypatch):
    calls = {"cards": [], "prop_names": [], "cstar_pc": [], "gamma_pc": []}

    rocketcea_module = types.ModuleType("rocketcea")
    cea_obj_module = types.ModuleType("rocketcea.cea_obj")

    def add_new_propellant(name, card):
        calls["cards"].append((name, card))

    class FakeCEAObj:
        def __init__(self, propName):
            calls["prop_names"].append(propName)

        def get_IvacCstrTc(self, Pc, eps, MR):
            calls["cstar_pc"].append(Pc)
            return 250.0, 1000.0 + 100.0 * Pc, 5000.0 + 10.0 * Pc

        def get_Chamber_MolWt_gamma(self, Pc, eps, MR):
            calls["gamma_pc"].append(Pc)
            return 25.0, 1.1 + 0.01 * Pc

    cea_obj_module.CEA_Obj = FakeCEAObj
    cea_obj_module.add_new_propellant = add_new_propellant
    monkeypatch.setitem(sys.modules, "rocketcea", rocketcea_module)
    monkeypatch.setitem(sys.modules, "rocketcea.cea_obj", cea_obj_module)

    card = "name TEST H 2 O 2 wt%=100.0"
    propellant = make_static_propellant(
        cea_formulation=card,
        cea_name="TEST_PROP",
        cea_pressure_range_pa=(PA_PER_PSIA, 2.0 * PA_PER_PSIA),
        cea_n_points=2,
    )

    mid_pressure_pa = 1.5 * PA_PER_PSIA

    assert calls["cards"] == [("TEST_PROP", card)]
    assert calls["prop_names"] == ["TEST_PROP"]
    assert calls["cstar_pc"] == pytest.approx([1.0, 2.0])
    assert calls["gamma_pc"] == pytest.approx([1.0, 2.0])
    assert propellant.get_cstar(mid_pressure_pa) == pytest.approx(1150.0 * FT_TO_M)
    assert propellant.get_gamma(mid_pressure_pa) == pytest.approx(1.115)
    assert propellant.Tc_at_pressure(mid_pressure_pa) == pytest.approx(
        5015.0 * RANKINE_TO_KELVIN
    )


def test_burn_uses_pressure_dependent_propellant_getters():
    class TrackingPropellant(Propellant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gamma_calls = []
            self.temperature_calls = []

        def get_gamma(self, pressure_pa):
            self.gamma_calls.append(float(pressure_pa))
            return super().get_gamma(pressure_pa)

        def Tc_at_pressure(self, pressure_pa):
            self.temperature_calls.append(float(pressure_pa))
            return super().Tc_at_pressure(pressure_pa)

    grain = Grain(
        outer_radius=0.04,
        initial_inner_radius=0.015,
        mass=0.7,
    )
    motor = Motor(
        grain,
        grain_number=1,
        chamber_inner_radius=0.045,
        nozzle_throat_radius=0.006,
        nozzle_exit_radius=0.015,
        chamber_length=0.25,
    )
    propellant = TrackingPropellant(
        specific_heat_ratio=1.2,
        products_molecular_mass=30e-3,
        combustion_temperature=2800.0,
        density=1700.0,
        burn_rate_a=5.0,
        burn_rate_n=0.3,
    )
    burn = Burn(grain, motor, propellant, Environment())

    chamber_pressure = 2.0e6
    assert burn.evaluate_nozzle_mass_flow(chamber_pressure) > 0.0
    assert chamber_pressure in propellant.gamma_calls
    assert chamber_pressure in propellant.temperature_calls
