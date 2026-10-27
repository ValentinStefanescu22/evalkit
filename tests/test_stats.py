from collections.abc import Hashable

import pytest

from evalkit.stats import (
    cohens_kappa,
    intervals_overlap,
    kappa_label,
    percentile,
    wilson_interval,
)


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


def pairs(*groups: tuple[Hashable, Hashable, int]) -> list[tuple[Hashable, Hashable]]:
    return [(a, b) for a, b, count in groups for _ in range(count)]


# The textbook example: 50 items, the raters agree on 35 (70%); by chance they'd agree on 50%.
def test_kappa_textbook_example() -> None:
    data = pairs(("yes", "yes", 20), ("yes", "no", 5), ("no", "yes", 10), ("no", "no", 15))
    assert cohens_kappa(data) == pytest.approx(0.40)


def test_kappa_perfect_and_chance() -> None:
    assert cohens_kappa(pairs((1, 1, 3), (2, 2, 3), (5, 5, 4))) == pytest.approx(1.0)
    # Agreement exactly at chance level: kappa 0.
    assert (
        cohens_kappa(pairs((True, True, 1), (True, False, 1), (False, True, 1), (False, False, 1)))
        == 0
    )


# A judge that passes everything agrees with you 80% of the time, yet kappa says "no better
# than luck": that's why percent agreement alone can fool you.
def test_kappa_always_pass_judge_scores_zero() -> None:
    data = pairs((True, True, 8), (True, False, 2))
    assert sum(a == b for a, b in data) / len(data) == 0.8
    assert cohens_kappa(data) == pytest.approx(0.0)


def test_kappa_undefined_when_both_raters_never_vary() -> None:
    assert cohens_kappa(pairs((4, 4, 5))) is None


def test_kappa_five_categories() -> None:
    # 1-5 scores: agree on 3 of 5. Chance: judge used 4 three times, you 4 twice, etc.
    data = [(4, 4), (4, 3), (4, 4), (5, 5), (2, 1)]
    # observed 0.6; chance = (3/5)(2/5) + (1/5)(1/5) = 0.28; kappa = 0.32/0.72
    assert cohens_kappa(data) == pytest.approx(0.32 / 0.72)


def test_kappa_needs_data() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([])


@pytest.mark.parametrize(
    ("kappa", "label"),
    [
        (None, "undefined"),
        (-0.1, "poor"),
        (0.0, "poor"),
        (0.2, "slight"),
        (0.21, "fair"),
        (0.42, "moderate"),
        (0.8, "substantial"),
        (0.81, "almost perfect"),
        (1.0, "almost perfect"),
    ],
)
def test_kappa_label(kappa: float | None, label: str) -> None:
    assert kappa_label(kappa) == label
