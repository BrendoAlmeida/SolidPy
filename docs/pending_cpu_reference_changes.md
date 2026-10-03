# Pending changes to the scalar reference (not made on purpose)

The accelerated-backend work (`docs/gpu_backend_architecture.md`, section 14.4) kept `solidpy/Burn.py`,
`solidpy/Grain.py` and `solidpy/Propellant.py` byte-for-byte unchanged, because the `physics_provider_hash` of every
result is a SHA-256 of the resolved inputs plus the bytes of those three files. While building the batched kernels
the points below were found. They are not fixed here; each one is a decision about the scalar physics.

## How to apply any of them

A byte changed in the three files above changes the hash of every result and makes the batched layer and the golden
corpus stale. In one commit, or a short series:

1. Change the scalar code.
2. Mirror it in the batched kernels (the file named in each item), so the batch stays equal to the oracle.
3. `python tools/make_golden_corpus.py`, review the diff of `tests/golden/reference_v1.json` like a physics change
   (`tests/golden/README.md`); `tests/test_golden_corpus.py` fails until the manifest SHA matches.
4. Re-run the parity suites, including `pytest tests/test_batch_parity.py --runslow`, and check whether the limits in
   `solidpy/backends/_tolerances.py` still hold. Do not widen them without writing the cause.

## 1. A source breakpoint between burnout and cutoff truncates the blowdown

`BurnSimulation._integrate_stage` (`Burn.py`, the breakpoint loop starting at `for boundary in
self._source_breakpoints(...)`, break at the end of that loop) stops after the first breakpoint segment when no grain
is active and `stop_after_burnout=True`. The blowdown is exactly that case (`active` is all false from the start).
A node of the activation table, the end of the igniter or `ignition_ramp_time` that falls between the last burnout
and the cutoff therefore ends the blowdown at the breakpoint; the cutoff event has not fired and the run is reported
as `blowdown_timeout` (`Burn.py`, where `_termination_reason` is set after the blowdown call).

* Reproduces with the `quirk` family of the golden corpus.
* The batched solver copies it on purpose (`solidpy/batch/integrators/solver.py`, `stage_body(...,
  first_segment_only=True)` and the module docstring). Fixing the scalar code means passing `False` there and
  regenerating the corpus.
* Likely fix: do not apply the "no active grain" break to segments of a stage that was entered with no active grain,
  or only break on `cutoff is None`.

## 2. Simultaneous grain burnout can be reported as `solver_failure` by rounding

Identical grains (an absurd erosive coefficient makes it easy) burn out together. scipy locates the first event and
the root of the second lands a few ulps away; the scalar code snaps only the grains whose event fired, the other
event fires again at the old time and the run ends as `solver_failure`. Three `solver-failure` designs of the corpus
are decided by this, not by physics (`tests/test_batch_solver.py`, `ROUNDING_DECIDED`). The batched solver snaps every
grain within `64 * eps` of its depth at the first event and completes them, so the parity tests skip that family.

* Likely fix: snap every grain within the tolerance at the first event in `_integrate_stage`, as the batch does.
  The corpus would then keep these designs but their reference outcome would change from failure to `completed`, and
  `ROUNDING_DECIDED` in `tests/test_batch_result.py`, `tests/test_batch_parity.py` and `tests/test_batch_solver.py`
  could go.

## 3. The provider hash does not see a burn-rate override on the instance

`Robustness._apply_scenario` (`Robustness.py`, `propellant.evaluate_burn_rate = lambda ...`) applies
`burn_rate_factor` by replacing `evaluate_burn_rate` on the propellant instance. `resolved_inputs` in `Burn.py`
lists `burn_rate_a`, `burn_rate_n` and the burn-rate table, but not that override, so two scenarios that differ only
in `burn_rate_factor` get the same `physics_provider_hash`. The Robustness report collects those hashes
(`Robustness.py`, `physics_provider_hashes`), so it cannot tell them apart.

* Not touched here. The batched packer treats an instance override as an unsupported feature and sends those lanes
  to `cpu-reference`, so no wrong result is produced.
* Likely fix: record the factor (or the identity of the override) in `resolved_inputs`, or apply the factor to
  `burn_rate_a` on the copy. The second option would also let Robustness scenarios run as batched lanes
  (`backend=` for `Robustness`, a later phase).

## 4. Two `git` subprocesses per result

Every `BurnSimulation.to_results` call runs `git rev-parse --show-toplevel` and `git rev-parse HEAD` (`Burn.py`, next
to the hash). That is a fixed cost per result, visible in thousand-run ensembles. The batched path reads the SHA once
per batch and caches it by `.git/HEAD` stamp (`solidpy/batch/assemble.py`). Caching it in the scalar code the same way
is a small change, but it changes the bytes of `Burn.py` and so follows the procedure above.

## Not a change request

The blowdown cutoff uses the pressure peak of the burn stage plus the source-only segment, without the blowdown
itself (`Burn.py`, `cutoff = ... 0.01 * max(reference_peak - ...)`). `docs/gpu_backend_architecture.md` section 5.4 said
"maximum of the history"; the code is the reference and the kernels copy it (decision D6 in section 14.3).
