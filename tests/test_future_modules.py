# -*- coding: utf-8 -*-

import numpy as np
import pytest

from solidpy import (
    CavityResonance,
    DispersionAnalysis,
    estimate_kinetic_lag_isp_loss,
    estimate_thermal_lag_isp_loss,
    estimate_two_phase_isp_loss,
)


def test_acoustics_skeleton_public_api():
    resonance = CavityResonance(
        chamber_length_m=0.6,
        chamber_radius_m=0.04,
        speed_of_sound_m_s=950.0,
    )

    longitudinal = resonance.longitudinal_frequencies()
    assert longitudinal == pytest.approx([791.6666666667, 1583.3333333333, 2375.0])

    tangential = resonance.tangential_frequencies()
    assert len(tangential) == 3
    assert tangential == sorted(tangential)
    assert all(frequency > 0.0 for frequency in tangential)

    time_s = np.linspace(0.0, 0.02, 500)
    pressure = np.sin(2.0 * np.pi * 500.0 * time_s)
    heat_release = 2.0 * pressure
    risk = resonance.evaluate_rayleigh_risk(pressure, heat_release, time_s)
    assert risk["risk_level"] == "high"
    assert risk["normalized_correlation"] == pytest.approx(1.0)
    assert risk["is_destabilizing"] is True


def test_monte_carlo_skeleton_public_api():
    analysis = DispersionAnalysis(
        simulation=object(),
        parameter_sigmas={"dry_mass_kg": 0.05, "wind_speed_m_s": 1.5},
        random_seed=42,
    )

    with pytest.raises(NotImplementedError):
        analysis.sample_inputs(8)
    with pytest.raises(NotImplementedError):
        analysis.run(8)
    with pytest.raises(NotImplementedError):
        analysis.dispersion_ellipse([(0.0, 0.0), (1.0, 1.0)])


def test_two_phase_flow_skeleton_public_api():
    particle_distribution = {"species": "Al2O3", "diameter_m": 5e-6}
    nozzle_state = {
        "pressure_pa": [3.0e6, 101325.0],
        "temperature_k": [3300.0, 1800.0],
        "nozzle_length_m": 0.16,
    }

    thermal = estimate_thermal_lag_isp_loss(
        220.0,
        0.12,
        particle_distribution,
        nozzle_state,
    )
    kinetic = estimate_kinetic_lag_isp_loss(
        220.0,
        0.12,
        particle_distribution,
        nozzle_state,
    )
    combined = estimate_two_phase_isp_loss(
        220.0,
        0.12,
        particle_distribution,
        nozzle_state,
    )

    assert thermal["thermal_loss_s"] > 0.0
    assert kinetic["kinetic_loss_s"] > 0.0
    assert combined["total_loss_s"] > max(
        thermal["thermal_loss_s"],
        kinetic["kinetic_loss_s"],
    )
    assert combined["corrected_isp_s"] < 220.0

    no_condensed = estimate_two_phase_isp_loss(
        220.0,
        0.0,
        particle_distribution,
        nozzle_state,
    )
    assert no_condensed["total_loss_s"] == pytest.approx(0.0)
    assert no_condensed["corrected_isp_s"] == pytest.approx(220.0)
