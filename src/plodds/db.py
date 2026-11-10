"""SQLite storage: schema, loading and the queries the rest of the project uses.

Three tables:

* ``teams``    one row per club
* ``matches``  one row per match, pointing at two teams
* ``odds``     one row per (match, bookmaker, timing), pointing at a match

The CHECK and UNIQUE constraints make the database reject bad data, so a
cleaning bug shows up when loading and not as a strange backtest result later.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from plodds.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS matches (
    match_id     INTEGER PRIMARY KEY,
    season       TEXT NOT NULL,
    match_date   TEXT NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams (team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams (team_id),
    home_goals   INTEGER NOT NULL CHECK (home_goals >= 0),
    away_goals   INTEGER NOT NULL CHECK (away_goals >= 0),
    result       TEXT NOT NULL CHECK (result IN ('H', 'D', 'A')),
    CHECK (home_team_id != away_team_id),
    UNIQUE (season, home_team_id, away_team_id)
);

CREATE TABLE IF NOT EXISTS odds (
    match_id  INTEGER NOT NULL REFERENCES matches (match_id) ON DELETE CASCADE,
    bookmaker TEXT NOT NULL,
    timing    TEXT NOT NULL CHECK (timing IN ('pre', 'closing')),
    home      REAL NOT NULL CHECK (home > 1),
    draw      REAL NOT NULL CHECK (draw > 1),
    away      REAL NOT NULL CHECK (away > 1),
    PRIMARY KEY (match_id, bookmaker, timing)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open the database (creating the file and schema if needed)."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite leaves these off by default
    conn.executescript(SCHEMA)
    return conn


def load_season(conn: sqlite3.Connection, matches: pd.DataFrame, odds: pd.DataFrame) -> int:
    """Load one cleaned season (the output of ``clean_season``).

    Loading the same season again replaces it, so the pipeline can be re-run
    safely. Everything happens in one transaction: either the whole season
    loads or none of it does. Returns the number of matches loaded.
    """
    seasons = matches["season"].unique()
    if len(seasons) != 1:
        raise ValueError(f"Expected one season per load, got {list(seasons)}")
    season = seasons[0]

    with conn:
        conn.execute("DELETE FROM matches WHERE season = ?", (season,))  # odds rows cascade

        names = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        conn.executemany("INSERT OR IGNORE INTO teams (name) VALUES (?)", [(n,) for n in names])
        team_ids = dict(conn.execute("SELECT name, team_id FROM teams"))

        match_ids = {}
        for row, match in matches.iterrows():
            cursor = conn.execute(
                "INSERT INTO matches (season, match_date, home_team_id, away_team_id,"
                " home_goals, away_goals, result) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    season,
                    match["date"].strftime("%Y-%m-%d"),
                    team_ids[match["home_team"]],
                    team_ids[match["away_team"]],
                    int(match["home_goals"]),
                    int(match["away_goals"]),
                    match["result"],
                ),
            )
            match_ids[row] = cursor.lastrowid

        conn.executemany(
            "INSERT INTO odds (match_id, bookmaker, timing, home, draw, away)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (match_ids[o.row], o.bookmaker, o.timing, float(o.home), float(o.draw), float(o.away))
                for o in odds.itertuples(index=False)
            ],
        )
    return len(matches)


def load_matches(
    conn: sqlite3.Connection, bookmaker: str = "Avg", timing: str = "pre"
) -> pd.DataFrame:
    """Every match, oldest first, with one bookmaker's odds alongside.

    This is the table the model and backtest work from. Matches that have no
    odds from the chosen bookmaker are kept, with missing odds, because the
    model can still learn from their scores.

    ``timing='pre'`` gives odds you could actually have bet at. ``'closing'``
    gives the final odds before kick-off, which are the usual benchmark for how
    much the market knew.
    """
    query = """
        SELECT m.match_id,
               m.season,
               m.match_date AS date,
               home.name    AS home_team,
               away.name    AS away_team,
               m.home_goals,
               m.away_goals,
               m.result,
               o.home       AS odds_home,
               o.draw       AS odds_draw,
               o.away       AS odds_away
        FROM matches AS m
        JOIN teams AS home ON home.team_id = m.home_team_id
        JOIN teams AS away ON away.team_id = m.away_team_id
        LEFT JOIN odds AS o
               ON o.match_id = m.match_id AND o.bookmaker = ? AND o.timing = ?
        ORDER BY m.match_date, m.match_id
    """
    return pd.read_sql_query(query, conn, params=(bookmaker, timing), parse_dates=["date"])


def league_table(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """Final league table for a season, e.g. ``league_table(conn, '2023-24')``.

    Not used by the model. It exists as a check on the data: if this matches
    the real table, the scores were loaded correctly. (Points deductions are
    not in the data, so a club that was docked points will show too many.)
    """
    query = """
        WITH per_team AS (
            SELECT home_team_id AS team_id, home_goals AS scored, away_goals AS conceded
            FROM matches WHERE season = :season
            UNION ALL
            SELECT away_team_id, away_goals, home_goals
            FROM matches WHERE season = :season
        )
        SELECT t.name                                         AS team,
               COUNT(*)                                       AS played,
               SUM(scored > conceded)                         AS won,
               SUM(scored = conceded)                         AS drawn,
               SUM(scored < conceded)                         AS lost,
               SUM(scored)                                    AS goals_for,
               SUM(conceded)                                  AS goals_against,
               SUM(scored) - SUM(conceded)                    AS goal_difference,
               3 * SUM(scored > conceded) + SUM(scored = conceded) AS points
        FROM per_team
        JOIN teams AS t USING (team_id)
        GROUP BY t.team_id
        ORDER BY points DESC, goal_difference DESC, goals_for DESC, team
    """
    table = pd.read_sql_query(query, conn, params={"season": season})
    table.index = range(1, len(table) + 1)
    table.index.name = "position"
    return table


def summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Matches per season and how many of them have each kind of odds."""
    query = """
        SELECT m.season,
               COUNT(DISTINCT m.match_id) AS matches,
               MIN(m.match_date)          AS first_match,
               MAX(m.match_date)          AS last_match,
               COUNT(DISTINCT CASE WHEN o.bookmaker = 'Avg'  AND o.timing = 'pre'     THEN m.match_id END) AS avg_pre,
               COUNT(DISTINCT CASE WHEN o.bookmaker = 'Avg'  AND o.timing = 'closing' THEN m.match_id END) AS avg_closing,
               COUNT(DISTINCT CASE WHEN o.bookmaker = 'B365' AND o.timing = 'pre'     THEN m.match_id END) AS b365_pre,
               COUNT(DISTINCT CASE WHEN o.bookmaker = 'PS'   AND o.timing = 'closing' THEN m.match_id END) AS pinnacle_closing
        FROM matches AS m
        LEFT JOIN odds AS o ON o.match_id = m.match_id
        GROUP BY m.season
        ORDER BY m.season
    """
    return pd.read_sql_query(query, conn)
