"""Bookmaker odds as probabilities.                         YOURS TO WRITE (day 1)

Decimal odds of 2.50 pay 2.50 per 1 staked, so they imply a probability of
1 / 2.50 = 40%. Add up the implied probabilities for home, draw and away and
you get more than 100%. The excess is the bookmaker's margin, the *overround*.
To compare the market with a model you have to take that margin back out.

Both functions should work on single numbers and on NumPy arrays / pandas
columns (use NumPy operations and you get that for free).

Run `pytest tests/test_odds.py` to check your work.
"""

import numpy as np


def overround(home, draw, away):
    """The bookmaker's margin: the sum of the three raw implied probabilities, minus 1.

    >>> round(overround(2.0, 4.0, 4.0), 6)      # a "fair" book
    0.0
    >>> round(overround(1.9, 3.6, 4.2), 4)
    0.0422
    """
    raise NotImplementedError


def implied_probabilities(home, draw, away):
    """Margin-free probabilities for (home, draw, away).

    Take the raw implied probabilities (1 / odds) and divide each by their sum,
    so the three add up to exactly 1. This is the simplest way to remove the
    margin ("proportional" or "multiplicative" normalisation).

    Returns an array whose last axis has length 3: shape (3,) for single
    numbers, shape (n, 3) for columns of n matches.

    >>> implied_probabilities(2.0, 4.0, 4.0)
    array([0.5 , 0.25, 0.25])

    Stretch: proportional normalisation is known to be slightly unfair to
    favourites (look up "favourite-longshot bias" and "Shin's method").
    """
    raise NotImplementedError
