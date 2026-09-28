# Statistical process control on inspection output: c-chart of defects per board.

from __future__ import annotations
import numpy as np


def c_chart(counts, baseline: int | None = None):
    # Return (cbar, ucl, lcl, out_of_control_indices).

    # Control limits come from the first `baseline` boards (Phase I) if given,
    # otherwise from all boards. c-chart: UCL = cbar + 3*sqrt(cbar), LCL >= 0.

    c = np.asarray(counts, dtype=float)
    ref = c[:baseline] if baseline else c
    cbar = float(ref.mean())
    ucl = cbar + 3 * np.sqrt(cbar)
    lcl = max(cbar - 3 * np.sqrt(cbar), 0.0)
    ooc = [int(i) for i in np.where((c > ucl) | (c < lcl))[0]]
    return cbar, ucl, lcl, ooc
