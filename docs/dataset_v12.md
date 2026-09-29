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
`grain_axial_positions_m` contains each grain's physical start position in input
order, inferred from its initial height and `grain_separation`. Its origin is
the nozzle-side stack reference; positive coordinates point away from the nozzle,
matching the detailed ballistics center-of-mass convention.

## Grain regression geometry

`Grain.calculate_remaining_volume(regression_m)` returns solid volume without
mutating the grain. `burnout_regression_m` is the smaller of radial web and half
initial height when end faces burn; inhibited end faces use the radial web.
The tubular model is `tubular_radial_axial_v1`. The star/slotted-cylinder model
is `fixed_angle_radial_front_v1`: slot angular boundaries are fixed, radial
sidewalls are inhibited, and the bore and slot-floor arcs regress. This model
is an approximation and does not represent isotropic star regression. Burn area
is the negative derivative of remaining volume, including burning end faces,
so the geometry cannot generate more propellant mass than its initial volume.

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
temperature: its temperature contribution does not change its injected mass.
The scalar solver integrates gas mass `m` and the thermal mixing inventory `m*T`,
with `d(m*T)/dt = mdot_generated*T_combustion_effective + mdot_igniter*T_igniter
- mdot_nozzle*T_gas` and `P*V = m*R*T_gas`. This is the prescribed source-temperature
mixing approximation `prescribed_source_temperature_mixing_v1`; it does not
include expansion work, wall heat loss, or a complete chamber energy equation.
Gas leaving the nozzle uses the actual mixed temperature. Initially, gas is at
the effective combustion temperature, recorded as `initial_gas_temperature_k`.
For identical source temperatures this reduces to the legacy isothermal model.
Activation uses `uniform_front_rate_scaling_v1`: the flame activation fraction
scales regression speed uniformly, and gas generation follows that same front.
Burned grains retain their terminal regression and generate no further gas.

`result["metrics"]` records `propellant_mass_consumed_kg`,
`propellant_burn_duration_s`, `nozzle_flow_duration_s`,
`mass_flow_avg_generated_kg_s`, `max_generated_mass_flow_kg_s`,
`mass_flow_avg_nozzle_kg_s`, `max_nozzle_mass_flow_kg_s`,
`nozzle_mass_integral_kg`, `igniter_mass_injected_kg`, `gas_mass_initial_kg`,
`gas_mass_cutoff_kg`, `mass_balance_residual_kg`, and `mass_flow_balance_error_pct`.
Source/discharge interval endpoints accompany their durations. Averages divide
the corresponding time integral by interval duration; zero duration gives zero.
Canonical generated, igniter, nozzle and impulse integrals use adaptive ODE
quadratures alongside the conserved states (`integration_method="adaptive_ode_quadrature"`).
Cumulative arrays `generated_mass_integral_kg`, `igniter_mass_integral_kg`,
`nozzle_mass_integral_kg`, `impulse_integral_ns`, and `pressure_throat_integral_ns`
make the integrals reproducible on the canonical time coordinate. Instantaneous
rates use the right-hand value at a source cutoff or burnout event; trapezoids
of those rate samples can differ slightly at discontinuities. Display interpolation
never replaces quadratures. `generated_mass_integral_kg` is also a scalar metric;
`integrated_generated_mass_kg` is its alias. `propellant_mass_consumed_kg` comes
independently from initial minus remaining geometric solid volume.
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
when `P_chamber - P_ambient <= 0.01 * (P_peak - P_ambient)`. A known igniter
source remains integrated through its declared end even after propellant burnout;
the reference peak includes that source phase. A callable igniter requires a
positive declared `igniter_burn_time` for a completed result; an unknown future
source duration is explicitly incomplete. `rtol`, `atol`, `burn_timeout_s` and
`tail_off_timeout_s` are configurable keyword settings; tail timeout is measured
from propellant burnout. The final event state
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

`evaluate_axial_mass_flux(motor, result["history"], stations_per_grain=5,
nozzle_direction="negative", flow_arrangement="single_outlet")` evaluates
the independent `source_accumulation_single_outlet_v1` backend. The station
count must be an integer >=2; `positive` reverses the outlet direction.
Split-flow arrangements are rejected. Stations run from upstream toward the
outlet and retain input grain indices and reproducible station IDs. Positions
move with regressing end faces; inhibited faces keep their original coordinates.
After a grain exhausts axially or radially, its station cross-section uses the
chamber area rather than a remaining bore restriction.

Lateral gas generation is distributed uniformly over each instantaneous grain
length. Burning end faces contribute two equal point sources in proportion to
their share of the canonical burn area. Endpoint stations include the
outlet-side trace of each face source. Igniter gas enters at the closed upstream
end. The diagnostic declares that spatial gas accumulation and drainage are
unmodeled: its source-throughput profile can differ from nozzle discharge
during pressurization and is zero during source-free blowdown, even while the
nozzle still ejects stored gas. It is not a transient spatial flow solution.

The diagnostic returns `time_s` with shape `(time,)` and `positions_m`,
`port_area_m2`, `mdot_axial_kg_s`, and `mass_flux_kg_m2_s` with shape `(time, stations)`, plus
station metadata, source decomposition, assumptions, and `metrics`. Its maximum
records time/station array indices, station ID, position, and input grain index.
Status remains `uncalibrated_diagnostic`; convergence starts `not_evaluated`.

`compare_axial_diagnostics(coarse, refined)` reports measured temporal peak
deltas and profile errors at matching stations on their common time interval.
Peak deltas use each entire sampled history. Temporal interpolation retains
sampling error near source discontinuities. Spatial peak errors compare sampled
maxima with outlet endpoint maxima: for this backend the interior source profile
is affine, its port area is constant within each grain, and sources are
nonnegative, so endpoint sampling captures the exact spatial maximum. Both
station counts and the numerical scale floor accompany the report. No calibrated
acceptance threshold or physical validity gate is applied.

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
