# -*- coding: utf-8 -*-

"""Monte Carlo dispersion-analysis scaffolding for flight simulations."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping as MappingABC
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
from scipy.stats import chi2

from ._parallel import safe_process_context


def _validate_iterations(n_iterations):
    n_iterations = int(n_iterations)
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1.")
    return n_iterations


def _parameter_distribution(specification):
    if isinstance(specification, MappingABC):
        sigma = float(
            specification.get(
                "sigma",
                specification.get("std", specification.get("stddev", 0.0)),
            )
        )
        mean = float(specification.get("mean", specification.get("nominal", 0.0)))
    else:
        sigma = float(specification)
        mean = 0.0

    if not np.isfinite(mean):
        raise ValueError("parameter mean must be finite.")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("parameter sigma must be a non-negative finite value.")
    return mean, sigma


def _apply_sample_to_object(simulation, sample):
    cloned_simulation = copy.deepcopy(simulation)
    for name, perturbation in sample.items():
        if hasattr(cloned_simulation, name):
            current_value = getattr(cloned_simulation, name)
            if isinstance(current_value, (int, float, np.number)) and np.isfinite(
                current_value
            ):
                setattr(cloned_simulation, name, float(current_value) + float(perturbation))
            else:
                setattr(cloned_simulation, name, perturbation)
        else:
            setattr(cloned_simulation, name, perturbation)
    return cloned_simulation


def _extract_impact_point(result):
    if isinstance(result, MappingABC):
        key_pairs = (
            (
                "simulation.3dof.landing_downrange_m",
                "simulation.3dof.landing_crossrange_m",
            ),
            ("simulation.3dof.landing_x_m", "simulation.3dof.landing_y_m"),
            ("landing_downrange_m", "landing_crossrange_m"),
            ("landing_x_m", "landing_y_m"),
            ("downrange_m", "crossrange_m"),
        )
        for downrange_key, crossrange_key in key_pairs:
            if downrange_key in result and crossrange_key in result:
                return (
                    float(result[downrange_key]),
                    float(result[crossrange_key]),
                )

        impact = result.get("impact_point", result.get("impact"))
        if isinstance(impact, MappingABC):
            return (
                float(impact.get("downrange_m", impact.get("x_m"))),
                float(impact.get("crossrange_m", impact.get("y_m"))),
            )
        if impact is not None:
            impact = np.asarray(impact, dtype=float).reshape(-1)
            if impact.size >= 2:
                return float(impact[0]), float(impact[1])

        trajectory = result.get("trajectory")
        if isinstance(trajectory, MappingABC) and "x_m" in trajectory and "y_m" in trajectory:
            x_m = np.asarray(trajectory["x_m"], dtype=float).reshape(-1)
            y_m = np.asarray(trajectory["y_m"], dtype=float).reshape(-1)
            if x_m.size and y_m.size:
                return float(x_m[-1]), float(y_m[-1])

    if isinstance(result, (tuple, list, np.ndarray)):
        impact = np.asarray(result, dtype=float).reshape(-1)
        if impact.size >= 2:
            return float(impact[0]), float(impact[1])

    raise ValueError("simulation result does not contain a recognizable impact point.")


def _run_single_simulation(simulation, sample):
    if callable(simulation):
        result = simulation(sample)
    elif hasattr(simulation, "run") and callable(simulation.run):
        result = _apply_sample_to_object(simulation, sample).run()
    elif hasattr(simulation, "simulate") and callable(simulation.simulate):
        result = _apply_sample_to_object(simulation, sample).simulate()
    else:
        raise TypeError("simulation must be callable or provide run()/simulate().")

    return {
        "input": sample,
        "result": result,
        "impact_point": _extract_impact_point(result),
    }


@dataclass
class DispersionAnalysis:
    """Run Monte Carlo dispersion analysis for simulation impact points.

    ``parameter_sigmas`` maps parameter names to either a standard deviation
    or a mapping. For mappings, the standard deviation may use ``sigma``,
    ``std``, or ``stddev`` (in that precedence order), and the mean may use
    ``mean`` or ``nominal`` (in that precedence order); both default to zero.
    ``random_seed`` controls reproducible input sampling. The simulation may be
    a callable that accepts a sample, or an object with a ``run()`` or
    ``simulate()`` method. Object simulations are deep-copied for each sample
    before parameter perturbations are applied.

    :meth:`run` evaluates the samples in a process pool and returns their raw
    results, impact points, summary statistics, and a confidence ellipse.
    """

    simulation: object
    parameter_sigmas: Mapping[str, float] = field(default_factory=dict)
    random_seed: Optional[int] = None

    def sample_inputs(self, n_iterations):
        """Generate normally distributed input samples.

        Args:
            n_iterations (int): Number of normally distributed samples to
                generate.

        Returns:
            list[dict]: One sampled parameter dictionary per iteration. A
            scalar ``parameter_sigmas`` value is treated as a zero-mean
            standard deviation. Mapping values may specify standard deviation
            using ``sigma``, ``std``, or ``stddev``, and mean using ``mean`` or
            ``nominal``; aliases are read in the order listed and default to
            zero.

        Raises:
            ValueError: If the iteration count or a distribution parameter
                is invalid.
        """
        n_iterations = _validate_iterations(n_iterations)
        rng = np.random.default_rng(self.random_seed)
        parameter_names = list(self.parameter_sigmas.keys())
        distributions = [
            _parameter_distribution(self.parameter_sigmas[name])
            for name in parameter_names
        ]

        samples = []
        for _ in range(n_iterations):
            sample = {}
            for name, (mean, sigma) in zip(parameter_names, distributions):
                sample[name] = float(rng.normal(mean, sigma))
            samples.append(sample)
        return samples

    def run(self, n_iterations):
        """Execute ``n_iterations`` perturbed simulations.

        Args:
            n_iterations (int): Number of simulation samples to execute.

        Returns:
            dict: Results with ``inputs`` (sampled dictionaries),
            ``impact_points`` (an ``(N, 2)`` NumPy array), ``results`` (raw
            simulation outputs), ``summary`` (aggregate metrics), and
            ``dispersion_ellipse`` (ellipse geometry and covariance).

        Raises:
            ValueError: If ``n_iterations`` is less than one or a simulation
                result does not contain a recognizable impact point.
            TypeError: If ``simulation`` is not callable and has no callable
                ``run()`` or ``simulate()`` method.

        Workers use the ``spawn`` process start method, so the simulation and
        its runner must be pickleable; define callable runners at module scope.
        Scripts that call this method must protect process-starting code with
        an ``if __name__ == "__main__":`` guard.
        """
        n_iterations = _validate_iterations(n_iterations)
        samples = self.sample_inputs(n_iterations)
        max_workers = min(n_iterations, max(os.cpu_count() or 1, 1))
        ordered_results = [None] * n_iterations
        impact_points = np.zeros((n_iterations, 2), dtype=float)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=safe_process_context(),
        ) as executor:
            futures = {
                executor.submit(_run_single_simulation, self.simulation, sample): index
                for index, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                index = futures[future]
                outcome = future.result()
                ordered_results[index] = outcome["result"]
                impact_points[index] = outcome["impact_point"]

        ellipse = self.dispersion_ellipse(impact_points)
        summary = {
            "simulation.monte_carlo.iterations": float(n_iterations),
            "simulation.monte_carlo.downrange_mean_m": float(np.mean(impact_points[:, 0])),
            "simulation.monte_carlo.crossrange_mean_m": float(np.mean(impact_points[:, 1])),
            "simulation.monte_carlo.downrange_std_m": float(np.std(impact_points[:, 0], ddof=0)),
            "simulation.monte_carlo.crossrange_std_m": float(np.std(impact_points[:, 1], ddof=0)),
            "simulation.monte_carlo.confidence": ellipse["confidence"],
            "simulation.monte_carlo.semi_major_axis_m": ellipse["semi_major_axis_m"],
            "simulation.monte_carlo.semi_minor_axis_m": ellipse["semi_minor_axis_m"],
            "simulation.monte_carlo.orientation_angle_rad": ellipse["orientation_angle_rad"],
        }

        return {
            "inputs": samples,
            "impact_points": impact_points,
            "results": ordered_results,
            "summary": summary,
            "dispersion_ellipse": ellipse,
        }

    def dispersion_ellipse(self, impact_points, confidence=0.95):
        """Compute a planar confidence ellipse from impact points.

        Args:
            impact_points: Iterable of ``(downrange_m, crossrange_m)`` impact
                coordinates.
            confidence (float): Confidence level for the ellipse, typically
                0.95 or 0.997.

        Returns:
            dict: Ellipse center, semi-major and semi-minor axes, orientation,
            sample covariance, eigenvalues and eigenvectors, chi-square scale,
            confidence, and ellipse area. A single point produces zero
            covariance and zero-length axes.

        Raises:
            ValueError: If ``confidence`` is outside ``(0, 1)`` or the input
                is not a non-empty, finite array with shape ``(N, 2)``.
        """
        confidence = float(confidence)
        if not np.isfinite(confidence) or confidence <= 0.0 or confidence >= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        impact_points = np.asarray(impact_points, dtype=float)
        if impact_points.ndim != 2 or impact_points.shape[1] != 2:
            raise ValueError("impact_points must be an array with shape (N, 2).")
        if impact_points.shape[0] < 1:
            raise ValueError("impact_points must contain at least one point.")
        if not np.all(np.isfinite(impact_points)):
            raise ValueError("impact_points must contain only finite values.")

        center = np.mean(impact_points, axis=0)
        if impact_points.shape[0] == 1:
            covariance = np.zeros((2, 2), dtype=float)
        else:
            covariance = np.cov(impact_points, rowvar=False)
            covariance = np.asarray(covariance, dtype=float).reshape(2, 2)

        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]

        scale = math.sqrt(float(chi2.ppf(confidence, df=2)))
        axes = scale * np.sqrt(eigenvalues)
        principal_vector = eigenvectors[:, 0]
        orientation_angle_rad = float(
            math.atan2(float(principal_vector[1]), float(principal_vector[0]))
        )

        return {
            "center_m": (float(center[0]), float(center[1])),
            "semi_major_axis_m": float(max(axes[0], axes[1])),
            "semi_minor_axis_m": float(min(axes[0], axes[1])),
            "orientation_angle_rad": orientation_angle_rad,
            "orientation_angle_deg": float(math.degrees(orientation_angle_rad)),
            "covariance_matrix": covariance,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "chi_square_scale": scale,
            "confidence": confidence,
            "area_m2": float(math.pi * axes[0] * axes[1]),
        }
