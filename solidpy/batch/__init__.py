# -*- coding: utf-8 -*-
"""Batched (structure-of-arrays) description and kernels for many independent motors.

This package needs NumPy and SciPy (the packers reuse the scalar models' helpers). Array kernels are written against an
array namespace ``xp`` (``numpy`` or ``jax.numpy``) so the same code runs on the CPU and on an accelerator. The scalar
classes remain the reference; nothing here changes them.
"""

from .problem import FEATURES, ProblemBatch, required_features

__all__ = ["FEATURES", "ProblemBatch", "required_features"]
