"""Scoring utilities.

The Phase 2 validator requires each task's score to be strictly between 0 and 1.
Many graders naturally produce exact 0.0 or 1.0 (perfect/zero performance),
so we post-process scores to guarantee they fall within the open interval.
"""

from __future__ import annotations

import math


def strict_unit_interval(value: float, *, decimals: int = 4) -> float:
    """Return a score strictly within (0, 1), rounded to ``decimals``.

    The output is clamped to:
      [10^-decimals, 1 - 10^-decimals]

    This guarantees the rounded value can never be 0.0 or 1.0.
    """

    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0

    if math.isnan(x) or math.isinf(x):
        x = 0.0

    x = round(x, decimals)
    min_score = 10 ** (-decimals)
    max_score = 1.0 - min_score

    if x <= min_score:
        return min_score
    if x >= max_score:
        return max_score
    return x
