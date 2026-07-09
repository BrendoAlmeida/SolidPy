# -*- coding: utf-8 -*-
"""NumPy version-compatibility shims for SolidPy.

``np.trapezoid`` only exists from NumPy 2.0 onward (it replaced ``np.trapz``,
which emits a DeprecationWarning under 2.x and is slated for removal).  Some
deployment targets — notably VMs pinned to a baseline x86-64-v1 CPU mask —
must run NumPy 1.26.x (whose wheels build for the older baseline), where
``np.trapezoid`` does not exist.  Importing the bare ``np.trapezoid`` attribute
at call time crashes there, killing every simulation.

Use ``from solidpy._numpy_compat import trapezoid`` and call ``trapezoid(...)``
in place of ``np.trapezoid(...)`` so the same code works across NumPy versions.
"""
from __future__ import annotations

import numpy as np

# np.trapezoid was added in NumPy 2.0; np.trapz is the legacy name (still present
# under 2.x but deprecated). Prefer the new name when available.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = ["trapezoid"]
