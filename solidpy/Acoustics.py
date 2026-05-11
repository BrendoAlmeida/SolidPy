# -*- coding: utf-8 -*-

"""Acoustic resonance scaffolding for solid rocket motor chambers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import jnp_zeros


def _validate_positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _validate_modes(modes):
    modes = int(modes)
    if modes < 1:
        raise ValueError("modes must be at least 1.")
    return modes


def _as_time_major(history, time_s, name):
    data = np.asarray(history, dtype=float)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must contain only finite values.")

    n_time = time_s.size
    if data.shape[0] == n_time:
        return data
    if data.ndim > 1 and data.shape[-1] == n_time:
        return np.moveaxis(data, -1, 0)
    if data.ndim == 1 and data.size == n_time:
        return data
    raise ValueError(f"{name} must have a time axis matching time_s.")

@dataclass(frozen=True)
class CavityResonance:
    """Model entry point for chamber acoustic resonance analysis.

    This class is intended to host the next generation of SolidPy acoustic
    stability calculations. The target model will estimate longitudinal and
    tangential chamber modes from the chamber geometry, local speed of sound,
    and operating state. Those modal estimates can then be compared against
    burn-rate response, nozzle admittance, and pressure-coupled heat-release
    histories.

    Planned responsibilities:
        * Calculate longitudinal acoustic natural frequencies along the motor
          axis for open/closed or impedance-corrected end conditions.
        * Calculate tangential and radial mode families for cylindrical
          chambers using appropriate Bessel roots.
        * Evaluate acoustic-instability risk with a Rayleigh-style index that
          compares pressure oscillation phase against unsteady heat release.
        * Provide mode metadata suitable for plotting Campbell diagrams or
          overlaying against measured static-fire pressure spectra.

    The heavy modal math is intentionally deferred. Methods currently raise
    ``NotImplementedError`` so callers can integrate the public API without
    silently receiving placeholder physics.
    """

    chamber_length_m: float
    chamber_radius_m: float
    speed_of_sound_m_s: float
    mean_chamber_pressure_pa: Optional[float] = None
    damping_ratio: float = 0.0

    def longitudinal_frequencies(self, modes=3):
        """Return the first longitudinal acoustic frequencies in hertz.

        Args:
            modes (int): Number of longitudinal modes to evaluate, starting at
                the first non-zero axial mode.

        Returns:
            list[float]: Natural frequencies in hertz.

        Raises:
            NotImplementedError: Until the impedance and boundary-condition
                model is implemented.
        """
        modes = _validate_modes(modes)
        chamber_length_m = _validate_positive(
            self.chamber_length_m,
            "chamber_length_m",
        )
        speed_of_sound_m_s = _validate_positive(
            self.speed_of_sound_m_s,
            "speed_of_sound_m_s",
        )

        mode_numbers = np.arange(1, modes + 1, dtype=float)
        frequencies_hz = mode_numbers * speed_of_sound_m_s / (2.0 * chamber_length_m)
        return [float(frequency) for frequency in frequencies_hz]

    def tangential_frequencies(self, modes=3):
        """Return the first tangential acoustic frequencies in hertz.

        Args:
            modes (int): Number of tangential mode families to evaluate.

        Returns:
            list[float]: Natural frequencies in hertz, ordered from lowest to
            highest.

        Raises:
            NotImplementedError: Until the cylindrical modal-root model is
                implemented.
        """
        modes = _validate_modes(modes)
        chamber_radius_m = _validate_positive(
            self.chamber_radius_m,
            "chamber_radius_m",
        )
        speed_of_sound_m_s = _validate_positive(
            self.speed_of_sound_m_s,
            "speed_of_sound_m_s",
        )

        candidates = []
        max_azimuthal_order = modes + 3
        radial_roots_per_order = modes
        for azimuthal_order in range(1, max_azimuthal_order + 1):
            roots = jnp_zeros(azimuthal_order, radial_roots_per_order)
            for root in roots:
                frequency_hz = (
                    speed_of_sound_m_s * float(root) / (2.0 * np.pi * chamber_radius_m)
                )
                candidates.append(frequency_hz)

        candidates.sort()
        return [float(frequency) for frequency in candidates[:modes]]

    def evaluate_rayleigh_risk(
        self,
        pressure_history,
        heat_release_history,
        time_s,
    ):
        """Evaluate acoustic instability risk with the Rayleigh criterion.

        Args:
            pressure_history: Time-aligned pressure perturbation history.
            heat_release_history: Time-aligned unsteady heat-release or burn
                response history.
            time_s: Monotonic time vector for the histories.

        Returns:
            dict: Planned output containing a Rayleigh index, dominant mode,
            phase estimate, and qualitative risk flag.

        Raises:
            NotImplementedError: Until pressure/heat-release phase integration
                is implemented.
        """
        time_s = np.asarray(time_s, dtype=float)
        if time_s.ndim != 1 or time_s.size < 2:
            raise ValueError("time_s must be a one-dimensional vector with at least 2 points.")
        if not np.all(np.isfinite(time_s)) or not np.all(np.diff(time_s) > 0.0):
            raise ValueError("time_s must be finite and strictly increasing.")

        pressure = _as_time_major(pressure_history, time_s, "pressure_history")
        heat_release = _as_time_major(
            heat_release_history,
            time_s,
            "heat_release_history",
        )
        if pressure.shape != heat_release.shape:
            raise ValueError("pressure_history and heat_release_history must have the same shape.")

        pressure_perturbation = pressure - np.mean(pressure, axis=0, keepdims=True)
        heat_release_perturbation = heat_release - np.mean(
            heat_release,
            axis=0,
            keepdims=True,
        )

        numerator = float(
            np.trapezoid(
                np.reshape(pressure_perturbation * heat_release_perturbation, (time_s.size, -1)),
                time_s,
                axis=0,
            ).sum()
        )
        pressure_energy = float(
            np.trapezoid(
                np.reshape(pressure_perturbation * pressure_perturbation, (time_s.size, -1)),
                time_s,
                axis=0,
            ).sum()
        )
        heat_energy = float(
            np.trapezoid(
                np.reshape(heat_release_perturbation * heat_release_perturbation, (time_s.size, -1)),
                time_s,
                axis=0,
            ).sum()
        )
        denominator = float(np.sqrt(max(pressure_energy * heat_energy, 0.0)))
        normalized_correlation = numerator / denominator if denominator > 0.0 else 0.0
        normalized_correlation = float(np.clip(normalized_correlation, -1.0, 1.0))

        if normalized_correlation >= 0.65:
            risk_level = "high"
        elif normalized_correlation >= 0.25:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "rayleigh_index": float(numerator),
            "normalized_correlation": normalized_correlation,
            "risk_level": risk_level,
            "is_destabilizing": bool(normalized_correlation > 0.0),
            "time_span_s": float(time_s[-1] - time_s[0]),
        }
