# Golden corpus

Frozen designs and the results of the scalar reference (`BurnSimulation`) for each of them. They are the
fixed input and the fixed expectation of the backend parity tests.

| File | Content |
|---|---|
| `corpus_v1.json` | One design per line: grains, motor, propellant, environment, simulation settings, feature tags |
| `reference_v1.json` | A manifest, then one reference record per design: status, metrics, 41-point resampled curves, `physics_provider_hash`, number of accepted steps |

The designs cover tubular and star grains, 1 to 24 grains, `ends_burn` both ways, power-law and tabulated
burn rate, scalar and tabulated thermochemistry, erosive burning, efficiencies below one, igniter and
activation profiles (none, scalar, table, ramp, callable), numerical and analytical tail-off, timeouts,
solver failure and designs at the edge of the guards in the scalar code. Callables are referred to by name
(`tests/golden_corpus.py`) because JSON cannot hold them.

The manifest stores the SHA-256 of `Burn.py`, `Grain.py` and `Propellant.py`. `tests/test_golden_corpus.py`
fails when it no longer matches, which means the scalar physics changed and these files are stale.

## Regenerating

Only when the scalar physics changes on purpose:

```
python tools/make_golden_corpus.py
```

It takes a minute or two on a many-core machine and produces the same designs from the fixed seed. Review
the diff of `reference_v1.json` like a physics change, and commit both files together with the change that
made them stale.
