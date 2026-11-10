import itertools

import numpy as np
import pandas as pd
import pytest

from plodds.model import PoissonModel

TEAMS = ["Strong", "Good", "Average", "Weak"]
TRUE_ATTACK = {"Strong": 0.4, "Good": 0.1, "Average": -0.1, "Weak": -0.4}
TRUE_DEFENCE = {"Strong": 0.3, "Good": 0.1, "Average": -0.1, "Weak": -0.3}
TRUE_HOME_ADVANTAGE = 0.3
TRUE_INTERCEPT = 0.1


def true_expected_goals(home, away):
    lam_home = np.exp(TRUE_INTERCEPT + TRUE_HOME_ADVANTAGE + TRUE_ATTACK[home] - TRUE_DEFENCE[away])
    lam_away = np.exp(TRUE_INTERCEPT + TRUE_ATTACK[away] - TRUE_DEFENCE[home])
    return lam_home, lam_away


@pytest.fixture(scope="module")
def simulated_matches():
    """2,400 matches drawn from a known Poisson model, so we know the right answer."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        for home, away in itertools.permutations(TEAMS, 2):
            lam_home, lam_away = true_expected_goals(home, away)
            rows.append((home, away, rng.poisson(lam_home), rng.poisson(lam_away)))
    return pd.DataFrame(rows, columns=["home_team", "away_team", "home_goals", "away_goals"])


@pytest.fixture(scope="module")
def model(simulated_matches):
    return PoissonModel().fit(simulated_matches)


def test_fit_returns_the_model(simulated_matches):
    model = PoissonModel()
    assert model.fit(simulated_matches) is model


def test_recovers_the_true_expected_goals(model):
    for home, away in itertools.permutations(TEAMS, 2):
        assert model.expected_goals(home, away) == pytest.approx(
            true_expected_goals(home, away), rel=0.12
        )


def test_recovers_home_advantage(model):
    assert model.home_advantage_ == pytest.approx(TRUE_HOME_ADVANTAGE, abs=0.06)


def test_ranks_attacks_correctly(model):
    assert sorted(TEAMS, key=model.attack_.get, reverse=True) == TEAMS
    assert sorted(model.teams_) == sorted(TEAMS)


def test_score_matrix_is_a_probability_distribution(model):
    matrix = model.score_matrix("Strong", "Weak", max_goals=10)
    assert matrix.shape == (11, 11)
    assert (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(1.0, abs=1e-3)
    lam_home, lam_away = model.expected_goals("Strong", "Weak")
    assert matrix[0, 0] == pytest.approx(np.exp(-lam_home - lam_away))


def test_outcome_probabilities_sum_to_one_and_favour_the_better_side(model):
    probs = model.outcome_probabilities("Strong", "Weak")
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > 0.6

    reverse = model.outcome_probabilities("Weak", "Strong")
    assert reverse[2] > reverse[0]


def test_home_side_is_favoured_between_equal_fixtures(model):
    assert (
        model.outcome_probabilities("Good", "Average")[0]
        > model.outcome_probabilities("Average", "Good")[2]
    )


def test_unknown_team_raises_key_error(model):
    with pytest.raises(KeyError):
        model.expected_goals("Strong", "Newly Promoted")
