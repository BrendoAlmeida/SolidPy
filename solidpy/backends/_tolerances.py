# -*- coding: utf-8 -*-
"""Versioned parity limits between accelerated backends and the scalar reference.

These are the limits CI enforces. The outer contract is ``solidpy.Acceptance`` (peaks 2 %, integrals 1 %,
mass balance 1 %). Against it, the integral limit is 1000x tighter and the limit on quantities read off the
accepted-step grid is 10x tighter. Grid-sampled maxima cannot go lower: two integrators (and two NumPy
environments of the reference itself) take different step sequences, so a tighter limit would test the grid
instead of the backend.

Never edit a value to make a failing test pass. A change needs a written cause in the commit message and a
new ``TOLERANCES_VERSION``.
"""

TOLERANCES_VERSION = "5"

# Version 2 (cause): the first measurement of the batched solver on the golden corpus, 160 completed lanes
# that include tolerances from rtol 1e-6 to 1e-9, gave event times up to 1.5e-6 (median 3e-10) and the maximum
# generated mass flow, read off the step grid, up to 2.04e-3 (median 3e-9). The version 1 values (1e-6 and 2e-3)
# came from a smaller, tighter corpus (architecture document, appendix D).
# Investigated against a refined run of the reference (rtol 1e-11, max_step 2 ms): on the worst lanes it is the
# stored reference that is off, not the batched solver. tolerance-013 (rtol 1e-7): final time of the reference
# 1.4e-6 from the refined value, batched 4e-9; ends-tubular-009: maximum generated flow of the reference 2.4e-3
# from the refined value (its coarse grid misses the peak), batched 3e-4. The limits therefore cover the error
# of the reference itself, not a bias of the batched solver; tests/test_batch_accuracy.py keeps the batched
# error from exceeding the reference's. The peak limit is still 4x inside the 2 % acceptance policy.

# Version 4 (cause): adds limits for what the robustness analysis reads from a run, which the earlier versions never
# compared; no existing value changed. Each new limit is at least 1.5x the worst difference measured between the
# scalar path and the NumPy backend, JAX on the CPU device and JAX on the GPU on eight designs (four tabulated
# four-grain, four power-law two-grain, throat and density varied), each with its nominal run, 5 to 10 default
# scenarios and 8 Latin-hypercube samples, max step 0.02 s. Worst differences (tabulated design; the power-law one is
# smaller on every line):
# * maximum generated mass flow 2.25e-2: a spike narrower than the step grid, under-sampled by every solver, the
#   scalar one included (errors against a refined scalar run, rtol 1e-11 and max step 2 ms, up to 2.4e-2 for the
#   scalar, NumPy and both JAX runs). The corpus limit stays at 1.5e-2;
# * maximum pressure rise rate 3.8e-2: a finite difference of consecutive accepted points; the scalar value is itself
#   up to 4.6e-2 from the refined run;
# * throat ablation 4.2e-5 and final throat diameter 1.6e-7: accumulated over the accepted points;
# * the series the detailed ballistics interpolates from the accepted points (thrust, pressure, mass, exit state)
#   5.5e-3 of their maximum;
# * the mass balance residual is at rounding level (<= 8e-8 %) in both paths.
# The burn area and the generated flow drop to zero when a grain burns out, and the detailed ballistics reads them by
# linear interpolation between accepted points, so a sample that falls inside the last accepted step before a burnout
# reads a fraction of the jump that depends on where the solver put that step: one GPU run differed by 3.1e-1 at that
# one sample (the lane agreed to 5e-12 when rerun on its own, and the GPU's burnout time varies by about 1e-8 between
# runs). Comparisons skip those samples (tests/test_batch_robustness.py).

# Version 5 (cause): adds the limit for the batched thermal ablation, which the earlier versions never compared; no
# existing value changed. The batched wall integration is the Radau IIA(5) controller of the scalar code's
# ``solve_ivp(method="Radau")`` ported step for step (initial step, Newton iteration, error estimate, step prediction,
# reuse of the Jacobian and of the factorization), so each lane takes the same Radau steps as the scalar run (the
# totals agree: 140,287 steps on one set of 600 lanes in all three backends) and the differences are rounding: matrix
# products and a pivot-free tridiagonal factorization where scipy keeps an LU. Worst relative difference between the
# scalar model and the NumPy backend, JAX on the CPU device and JAX on the GPU over every metric of 14 fixed cases
# (tests/thermal_cases.py) and 1,800 random lanes (walls of 4 to 18 cells, 30 to 400 time steps of 0.002 to 0.5 s,
# metal or composite casings with and without a liner): 1.4e-12 (JAX on the CPU device; NumPy 1.2e-13, GPU 2.3e-13).
# The heat load is a trapezoid over the scalar solver's own steps, so it differs by 1e-4 to 1e-3 from the exact
# integral and a different step sequence would show at that size: the limit sits well under that and well over the
# rounding, and the comparison fails if the steps differ.

#: Relative error of one kernel against the scalar method it mirrors.
KERNEL_RTOL_NUMPY = 1e-12
KERNEL_RTOL_GPU = 1e-10

#: Integrals over the whole simulation (impulse, generated and nozzle mass).
INTEGRAL_RTOL = 1e-5

#: Quantities sampled on the accepted-step grid (peak pressure, peak thrust, peak flows).
GRID_SAMPLED_RTOL = 5e-3

#: The maximum generated mass flow. It is a spike of the ignition transient, narrower than the step grid, so it
#: is the least well conditioned metric: on ratetable-006 (a 24-row burn-rate table) the stored reference is
#: 1.4e-2 away from a refined run and the batched solver 7.9e-3, while the pressure and thrust peaks of the same
#: lane agree to 5e-6. The acceptance policy gives this metric 2 % for the same reason (version 3).
GENERATED_FLOW_PEAK_RTOL = 1.5e-2

#: Event times (grain burnout, blowdown cutoff) and the final time.
TIME_RTOL = 5e-6

#: Series the detailed ballistics builds by interpolating the accepted points (thrust, pressure, mass, exit state),
#: relative to the largest value of the series. The solvers take different steps, so the points differ (worst
#: measured 5.5e-3, version 4).
DETAILED_SERIES_RTOL = 1e-2

#: Throat ablation and final throat diameter, accumulated over the accepted points (worst measured 4.2e-5, version 4).
DETAILED_ACCUMULATED_RTOL = 1e-4

#: The maximum generated mass flow of workloads on a coarse step grid (detailed ballistics, max step 0.01 to 0.02 s)
#: with a sharp ignition spike (worst measured 2.25e-2 on tabulated four-grain designs, 1.4e-5 on power-law ones);
#: see the version 4 note. ``GENERATED_FLOW_PEAK_RTOL`` still applies to the corpus.
GENERATED_FLOW_PEAK_COARSE_RTOL = 3.5e-2

#: Maximum pressure rise rate: the largest finite difference between consecutive accepted points, so it depends on
#: the step sequence itself, and the scalar value is itself up to 4.6e-2 from a refined run (worst measured 3.8e-2,
#: version 4).
FINITE_DIFFERENCE_RTOL = 6e-2

#: Mass balance residual in percent. Both paths leave a rounding-level residual (<= 8e-8 %), so the comparison is
#: absolute (version 4).
MASS_BALANCE_ATOL_PCT = 1e-6

#: Every metric of the batched thermal ablation against the scalar model (worst measured 1.4e-12, version 5).
THERMAL_RTOL = 1e-9
