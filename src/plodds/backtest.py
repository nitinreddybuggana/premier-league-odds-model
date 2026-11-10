"""Walk-forward backtest and betting simulation.          YOURS TO WRITE (days 4-5)

The rule that matters: a prediction for a match may only use information that
existed before that match kicked off. Your trading backtester filled orders at
the same close that generated the signal; this is where you show you know
better.

Run `pytest tests/test_backtest.py` to check your work.
"""

from typing import Callable

import numpy as np
import pandas as pd


def walk_forward(
    matches: pd.DataFrame,
    make_model: Callable[[], object],
    test_seasons: list[str],
    refit_days: int = 7,
) -> pd.DataFrame:
    """Out-of-sample predictions for every match in ``test_seasons``.

    ``matches`` is the output of ``db.load_matches``. ``make_model`` returns a
    fresh, unfitted model (pass the class itself: ``make_model=PoissonModel``).

    Walk through the test matches in date order. Whenever the next match is
    ``refit_days`` or more after the last refit (and always for the first one),
    fit a new model on ALL matches dated strictly before that match's date.
    Predict each match with the most recently fitted model.

    Skip a match if the model raises KeyError for it (a promoted team with no
    history yet). It will become predictable after the next refit that
    includes that team.

    Returns the predicted matches with their original columns plus

    ``p_home``, ``p_draw``, ``p_away``  the model's probabilities
    ``trained_through``                 latest match date in the training data
                                        used for that prediction

    ``trained_through`` is an audit trail: it must be earlier than ``date`` on
    every row, and the tests check exactly that.
    """
    raise NotImplementedError


def kelly_fraction(p: float, odds: float) -> float:
    """Fraction of bankroll the Kelly criterion says to stake.

    For a bet at decimal ``odds`` that wins with probability ``p``:

        f = (p * odds - 1) / (odds - 1)

    Return 0 when the bet has no edge (the formula goes negative).

    >>> round(kelly_fraction(0.6, 2.0), 6)
    0.2
    """
    raise NotImplementedError


def simulate_bets(
    predictions: pd.DataFrame,
    min_edge: float = 0.05,
    staking: str = "flat",
    stake: float = 10.0,
    bankroll: float = 1000.0,
    kelly_scale: float = 0.25,
) -> pd.DataFrame:
    """Turn predictions into a log of bets. One row per bet placed.

    ``predictions`` is the output of ``walk_forward`` and needs odds_home,
    odds_draw and odds_away. Ignore matches with missing odds.

    For each match, in date order, look at the three outcomes. The *edge* of an
    outcome is ``p * odds - 1``: the expected profit per unit staked if the
    model is right. Bet on the outcome with the largest edge, provided it is at
    least ``min_edge``. At most one bet per match.

    Staking:
      'flat'   stake ``stake`` on every bet
      'kelly'  stake ``kelly_scale * kelly_fraction(p, odds) * current bankroll``
               (full Kelly is far too aggressive when p is only an estimate)

    Returned columns: date, home_team, away_team, selection ('H', 'D' or 'A'),
    p, odds, edge, stake, profit, bankroll.
    ``profit`` is ``stake * (odds - 1)`` if the selection matches ``result``,
    otherwise ``-stake``. ``bankroll`` is the running total after the bet,
    starting from ``bankroll``.
    """
    raise NotImplementedError


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fall of an equity curve, as a negative fraction.

    >>> max_drawdown([100, 120, 90, 130])
    -0.25
    """
    raise NotImplementedError


def summarise_bets(bets: pd.DataFrame) -> dict:
    """Headline numbers for a bet log (the output of ``simulate_bets``).

    Keys: n_bets, win_rate, total_staked, profit, roi (profit / total_staked),
    max_drawdown (of the bankroll column).
    """
    raise NotImplementedError
