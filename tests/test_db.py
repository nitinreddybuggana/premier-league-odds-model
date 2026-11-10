import sqlite3

import pandas as pd
import pytest

from plodds import db


@pytest.fixture
def conn(cleaned_sample):
    conn = db.connect(":memory:")
    db.load_season(conn, *cleaned_sample)
    return conn


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_load_creates_teams_matches_and_odds(conn):
    assert count(conn, "teams") == 4
    assert count(conn, "matches") == 6
    assert count(conn, "odds") == 47


def test_reloading_a_season_replaces_it(conn, cleaned_sample):
    db.load_season(conn, *cleaned_sample)
    assert count(conn, "teams") == 4
    assert count(conn, "matches") == 6
    assert count(conn, "odds") == 47


def test_load_matches_joins_names_and_chosen_odds(conn):
    matches = db.load_matches(conn, bookmaker="B365", timing="pre")
    assert len(matches) == 6
    assert matches["date"].is_monotonic_increasing
    first = matches.iloc[0]
    assert (first["home_team"], first["away_team"]) == ("Northfield", "Eastbrook")
    assert (first["home_goals"], first["away_goals"], first["result"]) == (3, 0, "H")
    assert (first["odds_home"], first["odds_draw"], first["odds_away"]) == (1.40, 5.00, 8.00)


def test_load_matches_keeps_matches_without_odds(conn):
    matches = db.load_matches(conn, bookmaker="PS", timing="closing")
    assert len(matches) == 6
    assert matches["odds_home"].isna().sum() == 1


def test_league_table(conn):
    table = db.league_table(conn, "2023-24")
    # Worked out by hand from the six fixture matches.
    assert table["team"].tolist() == ["Southgate", "Northfield", "Westham Vale", "Eastbrook"]
    assert table["points"].tolist() == [5, 4, 4, 3]
    assert table["goal_difference"].tolist() == [2, 2, 0, -4]
    assert (table["played"] == 3).all()
    assert table["goals_for"].sum() == table["goals_against"].sum()


def test_database_rejects_impossible_rows(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE matches SET home_goals = -1 WHERE match_id = 1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE odds SET home = 0.9 WHERE match_id = 1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO odds (match_id, bookmaker, timing, home, draw, away)"
            " VALUES (999, 'Avg', 'pre', 2.0, 3.0, 4.0)"
        )


def test_summary_counts_coverage(conn):
    summary = db.summary(conn)
    assert summary["season"].tolist() == ["2023-24"]
    assert summary.loc[0, "matches"] == 6
    assert summary.loc[0, "avg_closing"] == 6
    assert summary.loc[0, "pinnacle_closing"] == 5
