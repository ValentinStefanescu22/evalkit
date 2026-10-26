# Small statistics helpers: confidence intervals and percentiles. Pure functions, no I/O.

import math
import statistics

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
