# Small statistics helpers: confidence intervals, percentiles, agreement. Pure functions, no I/O.

import math
import statistics
from collections import Counter
from collections.abc import Hashable, Sequence

Z_95 = 1.96  # how many standard deviations cover the middle 95% of a normal distribution


# Wilson score interval for k successes out of n tries: the range of true rates that could
# plausibly have produced k/n. Unlike the textbook p ± z·sqrt(p(1-p)/n), it stays inside
# [0, 1] and doesn't collapse to zero width at 0/n or n/n.
def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # In exact maths the bound is 0 at k=0 and 1 at k=n; floats give -1e-17 or 0.9999999999999999.
    low = 0.0 if k == 0 else centre - half_width
    high = 1.0 if k == n else centre + half_width
    return low, high


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


# The value that p% of `values` are at or below (p between 1 and 99).
# Interpolates between neighbours, like numpy's default.
def percentile(values: list[float], p: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


# Cohen's kappa: how much two raters agree beyond what luck alone would give.
# kappa = (observed - chance) / (1 - chance), where chance = the agreement two raters with the
# same habits would reach by picking at random. 1 = perfect, 0 = no better than luck.
# None when chance is 1 (both always used the same single category): kappa is undefined then.
# Sequence, not list: a list[tuple[int, int]] must be accepted too, and we only read it.
def cohens_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> float | None:
    if not pairs:
        raise ValueError("need at least one pair")
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    chance = sum((counts_a[c] / n) * (counts_b[c] / n) for c in counts_a | counts_b)
    if chance == 1:
        return None
    return (observed - chance) / (1 - chance)


# The usual reading of kappa (Landis & Koch, 1977). A convention, not a law.
def kappa_label(kappa: float | None) -> str:
    if kappa is None:
        return "undefined"
    if kappa <= 0:
        return "poor"
    for limit, label in [
        (0.20, "slight"),
        (0.40, "fair"),
        (0.60, "moderate"),
        (0.80, "substantial"),
    ]:
        if kappa <= limit:
            return label
    return "almost perfect"
