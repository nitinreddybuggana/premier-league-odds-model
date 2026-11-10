"""How good are a set of probability forecasts?           YOURS TO WRITE (day 4)

Accuracy ("how often was the favourite right?") is a poor measure for
probabilities. These scores reward being *calibrated*: saying 70% for things
that happen 70% of the time.

In every function:

``probs``     array of shape (n, 3): P(home), P(draw), P(away) for n matches
``outcomes``  length-n sequence of 'H', 'D' or 'A': what actually happened

The same functions score the model and the bookmaker (feed in
``implied_probabilities`` of the odds), which is the comparison this whole
project is about.

Run `pytest tests/test_evaluate.py` to check your work.
"""

import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]


def log_loss(probs, outcomes) -> float:
    """Mean of -log(probability given to what actually happened). Lower is better.

    A forecaster who always says 1/3, 1/3, 1/3 scores log(3) = 1.0986. That is
    the number to beat. Being confidently wrong is punished very hard.
    """
    raise NotImplementedError


def brier_score(probs, outcomes) -> float:
    """Mean squared error against the one-hot outcome. Lower is better.

    For each match, turn the outcome into a vector like [1, 0, 0], take the
    squared difference from the forecast, and sum over the three outcomes.
    Then average over matches. The always-1/3 forecaster scores 2/3.
    """
    raise NotImplementedError


def calibration_table(probs, outcomes, bins: int = 10) -> pd.DataFrame:
    """Do forecast probabilities match observed frequencies?

    Pool all 3n (probability, did-it-happen) pairs, cut the probabilities into
    ``bins`` equal-width bins from 0 to 1, and for each non-empty bin report

    ``mean_predicted``  average forecast probability in the bin
    ``observed``        fraction of those forecasts that came true
    ``count``           how many forecasts are in the bin

    For a well-calibrated forecaster the first two columns are close. Plot one
    against the other and you have a reliability diagram for the README.
    """
    raise NotImplementedError
