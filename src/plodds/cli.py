"""Command line for the data pipeline. Run from the repository root:

    plodds download          fetch the raw season files
    plodds build-db          clean them and load the SQLite database
    plodds summary           matches and odds coverage per season
    plodds table 2023-24     final league table, as a check on the data
"""

import argparse

from plodds import clean, db, download
from plodds.config import DB_PATH, SEASONS


def cmd_download(args):
    seasons = args.seasons or SEASONS
    for path in download.download_all(seasons, force=args.force):
        print(f"  {path}")
    print(f"{len(seasons)} season files in place.")


def cmd_build_db(args):
    conn = db.connect(DB_PATH)
    for season in args.seasons or SEASONS:
        path = download.raw_path(season)
        if not path.exists():
            raise SystemExit(f"{path} not found. Run `plodds download` first.")
        matches, odds = clean.clean_season(clean.read_raw(path), season)
        db.load_season(conn, matches, odds)
        print(f"  {clean.season_label(season)}: {len(matches)} matches, {len(odds)} odds rows")
    print(f"Database written to {DB_PATH}")


def cmd_summary(args):
    print(db.summary(db.connect(DB_PATH)).to_string(index=False))


def cmd_table(args):
    table = db.league_table(db.connect(DB_PATH), args.season)
    if table.empty:
        raise SystemExit(f"No matches for season {args.season!r}. Seasons look like '2023-24'.")
    print(table.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plodds", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("download", help="fetch raw season files")
    p.add_argument("--seasons", nargs="+", help="season codes like 2324 (default: all)")
    p.add_argument("--force", action="store_true", help="re-download files already on disk")
    p.set_defaults(run=cmd_download)

    p = commands.add_parser("build-db", help="clean raw files and load the database")
    p.add_argument("--seasons", nargs="+", help="season codes like 2324 (default: all)")
    p.set_defaults(run=cmd_build_db)

    p = commands.add_parser("summary", help="matches and odds coverage per season")
    p.set_defaults(run=cmd_summary)

    p = commands.add_parser("table", help="final league table for a season")
    p.add_argument("season", help="season label like 2023-24")
    p.set_defaults(run=cmd_table)

    args = parser.parse_args(argv)
    args.run(args)
