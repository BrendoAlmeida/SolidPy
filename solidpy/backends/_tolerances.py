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

TOLERANCES_VERSION = "4"

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
# compared; no existing value changed. Measured on a four-grain tabulated design (max step 0.02 s), nine robustness
# scenarios, scalar path against the NumPy backend, JAX on the CPU device and JAX on the GPU, and against a refined
# scalar run (rtol 1e-11, max step 2 ms) of every lane:
# * the maximum generated mass flow, a spike narrower than the step grid, is under-sampled by every solver, the
#   scalar one included: errors against the refined run up to 2.4e-2 (scalar), 2.4e-2 (NumPy), 2.2e-2 (JAX CPU) and
#   2.4e-2 (JAX GPU), all low, so solvers differ from one another by up to 2.2e-2 (the corpus limit stays at 1.5e-2);
# * the maximum pressure rise rate, a finite difference of consecutive accepted points, has errors up to 4.6e-2
#   (scalar) and 4.0e-2 (NumPy) against the refined run, and the backends differ from the scalar path by up to 3.5e-2;
# * the series the detailed ballistics interpolates from the accepted points differ by up to 3.3e-3 of their
#   maximum, and the throat ablation, accumulated over those points, by 2.8e-5 in its final value;
# * the mass balance residual is at rounding level (<= 8e-8 %) in both paths.

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

#: Series the detailed ballistics builds by interpolating the accepted points (thrust, pressure, mass, ablation),
#: relative to the largest value of the series. The two solvers take different steps, so the points differ
#: (worst measured 3.3e-3, version 4).
DETAILED_SERIES_RTOL = 1e-2

#: The maximum generated mass flow of workloads on a coarse step grid (detailed ballistics, max step 0.01 to 0.02 s)
#: with a sharp ignition spike; see the version 4 note. ``GENERATED_FLOW_PEAK_RTOL`` still applies to the corpus.
GENERATED_FLOW_PEAK_COARSE_RTOL = 3e-2

#: Maximum pressure rise rate: the largest finite difference between consecutive accepted points, so it depends on
#: the step sequence itself, and the scalar value is itself up to 4.6e-2 from a refined run (version 4).
FINITE_DIFFERENCE_RTOL = 6e-2

#: Mass balance residual in percent. Both paths leave a rounding-level residual (<= 8e-8 %), so the comparison is
#: absolute (version 4).
MASS_BALANCE_ATOL_PCT = 1e-6
