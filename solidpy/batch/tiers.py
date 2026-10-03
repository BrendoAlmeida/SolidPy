# -*- coding: utf-8 -*-
"""Run a batch in iteration-capped tiers so the slowest lanes do not make every lane pay.

A lockstep solve costs ``iterations of the slowest lane x lanes`` even though finished lanes are masked: they
still occupy the device. Designs differ by orders of magnitude in the steps they need (a few hundred for most,
tens of thousands for a few), so one launch pays the heavy tail for the whole batch. Here every lane first runs
with a small cap on the loop iterations; the lanes that did not finish are rerun from the start in a smaller batch
with a larger cap, and the last tier has no cap. A finished lane's result is exactly what an uncapped run gives
(the lanes are independent), and only the partial work of the unfinished ones is repeated.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Iteration caps of the capped tiers; an uncapped tier always follows. About 90 % of the corpus designs finish in
#: the first tier and about 98 % in the second.
DEFAULT_TIERS = (2048, 16384)


def solve_in_tiers(batch, run: Callable, tiers: Sequence[int] = DEFAULT_TIERS) -> Tuple[Dict[str, np.ndarray], List[tuple]]:
    """Solve ``batch`` with ``run(sub_batch, cap) -> outputs`` tier by tier.

    ``outputs`` maps names to arrays whose first axis is the lane, plus a boolean ``unfinished`` per lane (the
    other, scalar entries are dropped). Returns the merged outputs for all lanes in order and, per tier,
    ``(cap, lanes run, lanes left unfinished)``.
    """
    remaining = np.arange(len(batch))
    merged: Optional[Dict[str, np.ndarray]] = None
    info: List[tuple] = []
    for cap in (*tiers, None):
        sub = batch.select(remaining)
        out = run(sub, cap)
        unfinished = np.asarray(out["unfinished"], dtype=bool)
        if merged is None:
            merged = {name: np.zeros((len(batch),) + np.shape(v)[1:], dtype=np.asarray(v).dtype)
                      for name, v in out.items() if np.ndim(v) >= 1 and np.shape(v)[0] == len(sub)}
        done = ~unfinished
        for name in merged:
            merged[name][remaining[done]] = np.asarray(out[name])[done]
        info.append((cap, len(sub), int(unfinished.sum())))
        remaining = remaining[unfinished]
        if not len(remaining):
            break
    return merged, info
