import pytest

from evalkit.stats import intervals_overlap, percentile, wilson_interval


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [(8, 10, 0.490, 0.943), (0, 5, 0.0, 0.434), (5, 5, 0.566, 1.0), (16, 22, 0.518, 0.868)],
)
def test_wilson_known_values(k: int, n: int, low: float, high: float) -> None:
    assert wilson_interval(k, n) == pytest.approx((low, high), abs=0.001)


def test_wilson_edges_are_exact() -> None:
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0


# Swapping successes and failures mirrors the interval: k/n's low = 1 - (n-k)/n's high.
def test_wilson_is_symmetric() -> None:
    low, high = wilson_interval(3, 10)
    mirror_low, mirror_high = wilson_interval(7, 10)
    assert low == pytest.approx(1 - mirror_high)
    assert high == pytest.approx(1 - mirror_low)


def test_wilson_stays_in_bounds_and_contains_observed_rate() -> None:
    for n in range(1, 31):
        for k in range(n + 1):
            low, high = wilson_interval(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0


@pytest.mark.parametrize(("k", "n"), [(0, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_impossible_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError, match="0 <= k <= n"):
        wilson_interval(k, n)


def test_overlap_decides_significance() -> None:
    # 0/5 vs 5/5: [0, 43%] and [57%, 100%] don't touch -> significant.
    assert not intervals_overlap(wilson_interval(0, 5), wilson_interval(5, 5))
    # One try each can never be significant: [0, 79%] and [21%, 100%] overlap.
    assert intervals_overlap(wilson_interval(0, 1), wilson_interval(1, 1))
    # 1/5 vs 4/5 still overlap -> noise.
    assert intervals_overlap(wilson_interval(1, 5), wilson_interval(4, 5))


def test_percentile() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile([7.0], 95) == 7.0
    # Order of the input doesn't matter.
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
