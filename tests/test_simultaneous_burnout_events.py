import numpy as np
import pytest

import golden_corpus as gc
from solidpy import BurnSimulation


CASE_IDS = ("solver-failure-000", "solver-failure-001", "solver-failure-002")


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_reported_simultaneous_burnout_events_are_snapped(case_id):
    case = next(case for case in gc.load_corpus()["cases"] if case["id"] == case_id)
    grain, motor, propellant, environment, kwargs = gc.build_objects(case)

    result = BurnSimulation(grain, motor, propellant, environment, **kwargs).result
    burnout_times = result["metrics"]["grain_burnout_times_s"]

    assert result["status"]["termination_reason"] == "completed"
    assert result["status"]["burnout_completed"] is True
    assert all(time is not None for time in burnout_times)
    np.testing.assert_array_equal(burnout_times, np.full(len(burnout_times), burnout_times[0]))
