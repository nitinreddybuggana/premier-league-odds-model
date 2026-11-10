import numpy as np
import pytest

from plodds.odds import implied_probabilities, overround


def test_fair_book_has_no_overround():
    assert overround(2.0, 4.0, 4.0) == pytest.approx(0.0)


def test_overround_of_a_typical_book():
    assert overround(1.9, 3.6, 4.2) == pytest.approx(0.042189, abs=1e-6)


def test_implied_probabilities_of_a_fair_book():
    assert implied_probabilities(2.0, 4.0, 4.0) == pytest.approx([0.5, 0.25, 0.25])


def test_implied_probabilities_sum_to_one_and_keep_the_ranking():
    probs = implied_probabilities(1.9, 3.6, 4.2)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]
    # Each one is the raw implied probability scaled down by the same factor.
    assert probs[0] == pytest.approx((1 / 1.9) / 1.042189, abs=1e-6)


def test_works_on_columns_of_matches():
    home = np.array([2.0, 1.5])
    draw = np.array([4.0, 4.5])
    away = np.array([4.0, 7.0])
    probs = implied_probabilities(home, draw, away)
    assert probs.shape == (2, 3)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert overround(home, draw, away).shape == (2,)
