# SolidPy v12 numerical interface

This contract specifies reproducible scalar internal-ballistics and structural
evaluations. History quantities use SI units. Existing positional solution arrays
remain available; new evaluations should use the named canonical result.

## Independent efficiencies

`Burn` and `BurnSimulation` accept keyword-only `eta_c=1.0`, `eta_Cf=1.0`, and
`discharge_coefficient=1.0`. All must be finite with `0 < value <= 1`;
invalid values raise `ValueError` before solving.

`eta_c` sets `T_combustion_effective = eta_c**2 * T_combustion`, affecting
thermodynamics, pressure, and flow. It is not a second multiplier on thrust.
`discharge_coefficient` scales nozzle flow and momentum thrust. `eta_Cf` scales
reported thrust and impulse without changing pressure, characteristic velocity,
flow, generated mass, or gas inventory.

`Burn.evaluate_thrust_components(chamber_pressure)` returns `momentum_ideal_n`,
`momentum_n`, `pressure_n`, and `total_n`. Conical divergence affects momentum once:

```text
lambda = (1 + cos(nozzle_angle)) / 2  # 1 for an unspecified angle
F_momentum_ideal = lambda * mdot_nozzle_ideal * exhaust_velocity
F_momentum = discharge_coefficient * F_momentum_ideal
F_pressure = (P_exit - P_ambient) * nozzle_exit_area
F_reported = eta_Cf * (F_momentum + F_pressure)
```

At or below ambient pressure, nozzle flow and reported thrust are zero.
Resolved efficiencies use `efficiency_semantics="native_split"`.

## Connected volume and physical geometry

`Motor(..., connected_chamber_volume_m3=None)` accepts a positive finite gas
control volume; omission preserves `chamber_area * chamber_length`.
`chamber_length` retains its physical meaning. A connected frustum contributes
`pi * length / 3 * (R_inlet**2 + R_inlet*R_throat + R_throat**2)` to volume.
Initial free volume subtracts total propellant volume once and must be positive.
Grains must fit within the chamber radius and physical axial length, including
separations. Casing, liner, envelope, and buckling calculations use physical
lengths; the equivalent gas-volume length is not a structural dimension.

## Canonical result

`BurnSimulation.result` contains `history`, `metrics`, `status`, `efficiencies`,
and `provenance`. Its adaptive history includes burnout and numerical blowdown.
`evaluate_complete_solution()` and `total_burn_solution` retain the seven-array
layout: time, pressure, free volume, mean regression, thrust, exit pressure,
and exit velocity.

`result["history"]` contains aligned arrays on strictly increasing `time_s`:

| Field | Unit | Meaning |
| --- | --- | --- |
| `time_s` | s | Adaptive solver time |
| `chamber_pressure_pa` | Pa | Absolute chamber pressure |
| `free_volume_m3` | m³ | Gas control volume |
| `regression_m` | m | Individual grain regressions |
| `burn_area_m2` | m² | Total active burn area |
| `mdot_generated_kg_s` | kg/s | Propellant gas generation |
| `mdot_igniter_kg_s` | kg/s | Igniter mass injection |
| `mdot_nozzle_kg_s` | kg/s | Effective nozzle discharge |
| `thrust_n` | N | Reported thrust after `eta_Cf` |
| `gas_mass_kg` | kg | Control-volume gas inventory |

Momentum and pressure thrust arrays retain the public component decomposition.
Per-grain regression and axial profiles share the history time coordinate.
Interpolation for display or export does not replace canonical integrals.
Gas inventory follows mass continuity, including igniter injection at a different
temperature: its energy contribution does not change its injected mass.

`result["metrics"]` records `propellant_mass_consumed_kg`,
`propellant_burn_duration_s`, `nozzle_flow_duration_s`,
`mass_flow_avg_generated_kg_s`, `max_generated_mass_flow_kg_s`,
`mass_flow_avg_nozzle_kg_s`, `max_nozzle_mass_flow_kg_s`,
`nozzle_mass_integral_kg`, `igniter_mass_injected_kg`, `gas_mass_initial_kg`,
`gas_mass_cutoff_kg`, `mass_balance_residual_kg`, and `mass_flow_balance_error_pct`.
Source/discharge interval endpoints accompany their durations. Averages divide
the corresponding time integral by interval duration; zero duration gives zero.
Integration uses the nonuniform time coordinate, not sample arithmetic means.

```text
mass_balance_residual_kg = propellant_mass_consumed_kg + igniter_mass_injected_kg
                         - nozzle_mass_integral_kg
                         - (gas_mass_cutoff_kg - gas_mass_initial_kg)
                         - other_declared_outflows_kg
```

Relative error divides the absolute residual by
`max(propellant_mass_consumed_kg + igniter_mass_injected_kg, epsilon)`. Initial gas uses
the inventory difference. The model has no other outflow; residual gas is retained.

## Completion and axial diagnostic

Each grain stops generating gas at its burnout event. Numerical blowdown ends
when `P_chamber - P_ambient <= 0.01 * (P_peak - P_ambient)`. The final event state
retains gas inventory. `result["status"]` records completion and termination
reason. Timeout, solver failure, omitted tail-off, or analytical approximation
is incomplete for numerical acceptance. Igniter flow remains included while active.

The axial diagnostic is separate from the erosion model's mean port-flux
approximation. It orders physical stations/grains and records IDs, positions,
nozzle direction, and backend. Local flow accumulates upstream mass sources in
the declared direction; local flux divides by instantaneous grain-specific port
area, including slots. The profile records time and source assumptions.
Summaries include `max_axial_mass_flux_kg_m2_s`, `max_axial_mass_flux_time_s`,
and `max_axial_mass_flux_grain_index`. Temporal/spatial convergence is recorded.
Until thresholds are calibrated, axial flux is diagnostic and does not reject
physical validity or modify the erosive-burning law.

## Mass, structure, and robustness

Dry mass sums modeled casing, liner, and nozzle components; initial motor mass
adds initial propellant mass. Results identify modeled and omitted components.
Physical closures, lengths, and material volumes are counted without overlap.
`simulate_structural_response()` is the canonical kernel for triaxial von Mises,
Tresca/burst, buckling, strain, and closure-fastener responses. Public burst
calculations share strength and safety-factor conventions; the
`casing_burst_pressure_mpa` alias refers to canonical `burst_pressure_mpa`.

`closure_bolt_status` is `configured`, `not_configured`, or `model_not_available`.
Applicability explains absent factors/stresses (`None`, serialized as `null`).
Missing required structural properties cause an error or explicit model failure.
`thermal_service_margin` is the dimensionless maximum service-temperature
margin; legacy `thermoelastic_margin` is its alias, not a thermal-stress result.
Nominal outputs remain separate from scenario/ensemble outputs. Robustness
records `robustness_policy_id`, scenario IDs/factors, provider/hash, and status;
ensemble worst-case margins do not overwrite nominal results.

## Provenance and acceptance

Provenance records `eta_c_applied`, `eta_cf_applied`,
`discharge_coefficient_applied`, `efficiency_semantics`, `thermochemistry_source`,
`cea_used`, `physics_provider_hash`, and `solidpy_git_sha`, together with resolved
inputs and solver settings. The v12 path uses scalar properties (`cea_used=false`).
Optional runtime CEA is deferred: explicit formulation/provider identity and
hash, pressure coverage, coherent molecular mass or gas constant, and an
out-of-range policy are required before it is covered by this contract.

Policy `v12_numerical_acceptance_v1` requires completed numerical blowdown and
mass-balance error <=1%. Coarse/refined runs must agree within 2% for peak
pressure, thrust, generated flow, and nozzle flow; impulse and integrated
generated/nozzle mass must agree within 1%. Deltas are
`abs(refined-coarse)/max(abs(refined), scale_floor)`; floors are recorded per quantity.
Regression coverage includes isolated efficiencies and domain boundaries,
connected frustum volume with physical lengths, nonuniform integration, igniter
mass/energy, burnout/blowdown and residual inventory, incomplete termination,
nullable structural outputs, shared burst formulas, and axial IDs/direction.
Structural limits remain independent from numerical acceptance thresholds.
