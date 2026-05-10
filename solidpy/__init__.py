# -*- coding: utf-8 -*-

"""
SolidPy is Projeto Jupiter propulsion's team attempt to create a sophisticated
internal ballistics simulator for solid propellant rocket engines, with versatility
for different configurations and able to generate all the required data for designed motors.
"""

__author__ = (
    "João Lemes Gribel Soares",
    "Pedro Henrique Marinho Bressan",
    "Thais Silva Melo",
)
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__credits__ = (
    "João Lemes Gribel Soares",
    "Pedro Henrique Marinho Bressan",
    "Thais Silva Melo",
)
__license__ = "MIT"
__version__ = ""
__maintainer__ = "João Lemes Gribel Soares"
__email__ = "jgribel@usp.br"
__status__ = "Production"

import csv
import math
import warnings
import time

import numpy as np
from scipy import integrate
from scipy import linalg
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import scipy.constants as const
import matplotlib.pyplot as plt

from .Grain import Grain
from .Motor import Motor
from .Propellant import Propellant, build_cea_properties
from .Burn import Burn, BurnSimulation
from .BurnEmpirical import BurnEmpirical
from .Environment import Environment
from .Rail import Rail
from .Export import Export
from .DetailedBallistics import (
    build_detailed_ballistics,
    evaluate_nozzle_ablation_rate,
    evaluate_combustion_stability,
    run_detailed_ballistics,
)
from .Robustness import (
    RobustnessScenario,
    build_latin_hypercube_scenarios,
    default_robustness_scenarios,
    run_robustness_analysis,
    summarize_robustness,
)
from .Multiphysics import (
    CasingMaterial,
    MotorGeometry,
    NozzleMaterial,
    evaluate_barrowman_stability,
    evaluate_cd_by_components,
    geometry_from_components,
    simulate_advanced_components,
    simulate_advanced_physics,
    simulate_cfd_proxies,
    simulate_flight_1d,
    simulate_ignition_proxy,
    simulate_structural_response,
    simulate_thermal_ablation,
)
