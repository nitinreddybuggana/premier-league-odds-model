"""A Poisson model of football scores.                    YOURS TO WRITE (days 2-3)

The idea (Maher, 1982): the goals each side scores in a match are Poisson
distributed, with a rate that depends on how good the scoring team's attack
is, how good the other team's defence is, and whether the scoring team is at
home.

One way to write it, for home team i against away team j:

    log(lambda_home) = c + home_advantage + attack[i] - defence[j]
    log(lambda_away) = c                  + attack[j] - defence[i]

    home_goals ~ Poisson(lambda_home)
    away_goals ~ Poisson(lambda_away)        (treated as independent)

The parameters (c, home_advantage, one attack and one defence number per team)
are not known. You estimate them by *maximum likelihood*: find the values that
make the scores that actually happened as probable as possible.

Suggested route
---------------
1. Write a function that takes a flat parameter vector and the data and returns
   the NEGATIVE log-likelihood: minus the sum over matches of
   log P(home_goals | lambda_home) + log P(away_goals | lambda_away).
   `scipy.stats.poisson.logpmf` does the per-match part.
2. Minimise it with `scipy.optimize.minimize`.
3. Unpack the result into the attributes listed in the class docstring.

Things you will run into
------------------------
* Identifiability. Add 0.1 to every attack and every defence and no prediction
  changes, so the optimiser has no unique answer. You need a constraint, e.g.
  attack values sum to zero and defence values sum to zero. (Easy way: optimise
  n-1 values per group and set the last to minus their sum.)
* Speed. Map team names to integer indices once, then use NumPy indexing.
  A Python loop over matches inside the likelihood will be far too slow.
* Promoted teams. A team with no matches in the training data has no
  parameters. `expected_goals` should raise a KeyError for it; the backtest
  decides what to do about that.

Stretch goals (only after the backtest works)
---------------------------------------------
* Time decay: weight recent matches more (Dixon & Coles, 1997, section 4.5).
* The Dixon-Coles low-score correction for 0-0, 1-0, 0-1 and 1-1.

Run `pytest tests/test_model.py` to check your work.
"""

import numpy as np
import pandas as pd


class PoissonModel:
    """Independent Poisson model with attack, defence and home advantage.

    After ``fit`` the model has:

    ``teams_``           list of team names seen in training
    ``home_advantage_``  float, on the log scale (positive means home sides score more)
    ``attack_``          dict of team name -> attack strength
    ``defence_``         dict of team name -> defence strength
    """

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        """Estimate the parameters from past matches and return ``self``.

        ``matches`` has columns home_team, away_team, home_goals, away_goals
        (other columns are ignored).
        """
        raise NotImplementedError

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """(lambda_home, lambda_away) for a fixture. KeyError for an unknown team."""
        raise NotImplementedError

    def score_matrix(self, home_team: str, away_team: str, max_goals: int = 10) -> np.ndarray:
        """Probability of every scoreline up to ``max_goals`` each.

        Returns an array of shape (max_goals + 1, max_goals + 1) where entry
        [h, a] is P(home scores h and away scores a). Because the two counts are
        independent, this is the outer product of two Poisson pmf vectors.
        """
        raise NotImplementedError

    def outcome_probabilities(self, home_team: str, away_team: str) -> np.ndarray:
        """[P(home win), P(draw), P(away win)], summing to 1.

        Sum the right parts of the score matrix (np.tril, np.trace and np.triu
        are handy). The matrix stops at ``max_goals`` so the three sums fall a
        tiny bit short of 1; divide by their total.
        """
        raise NotImplementedError
