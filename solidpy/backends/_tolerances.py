# -*- coding: utf-8 -*-
"""Versioned parity limits between accelerated backends and the scalar reference.

These are the limits CI enforces. They are 100x tighter than the outer contract of
``solidpy.Acceptance`` (peaks 2 %, integrals 1 %, mass balance 1 %). Never edit a value to make a failing
test pass: a change needs a written cause in the commit message and a new ``TOLERANCES_VERSION``.
"""

TOLERANCES_VERSION = "1"

#: Relative error of one kernel against the scalar method it mirrors.
KERNEL_RTOL_NUMPY = 1e-12
KERNEL_RTOL_GPU = 1e-10

#: Integrals over the whole simulation (impulse, generated and nozzle mass).
INTEGRAL_RTOL = 1e-5

#: Quantities sampled on the accepted-step grid (peak pressure, peak thrust, peak flows). The reference
#: itself varies by this order between two environments, so this is a property of grid sampling.
GRID_SAMPLED_RTOL = 2e-3

#: Event times (grain burnout, blowdown cutoff) and the final time.
TIME_RTOL = 1e-6
