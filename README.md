# Can a statistical model beat Premier League odds?

Bookmaker odds are prices. Like any price they imply a probability, they
include a margin, and you can ask whether they are efficient. This project
fits a Poisson model of football scores to ten seasons of Premier League
results, compares its probabilities with the bookmakers', and backtests a
value-betting strategy strictly out of sample.

> **Status: in progress.** The data pipeline and database are done and tested.
> The model, evaluation and backtest are being written. Results will go here.

## How it works

```
football-data.co.uk CSVs ──> clean ──> SQLite ──> Poisson model ──> walk-forward ──> evaluation
     (download.py)        (clean.py)   (db.py)     (model.py)       (backtest.py)   (evaluate.py)
```

1. **Data.** One CSV per season is downloaded, cleaned and loaded into a
   normalised SQLite database (`teams`, `matches`, `odds`). Constraints in the
   schema reject impossible rows.
2. **Model.** Goals are modelled as Poisson counts driven by each team's attack
   and defence strength and home advantage, fitted by maximum likelihood.
3. **Backtest.** For every match in the test seasons, the model is fitted only
   on matches played before it. No prediction ever sees its own result.
4. **Evaluation.** Model and market are scored on log loss, Brier score and
   calibration, and a betting simulation reports ROI and drawdown.

## Setup

```bash
git clone https://github.com/nitinreddybuggana/premier-league-odds-model
cd premier-league-odds-model
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

Run from the repository root.

```bash
plodds download          # fetch the season files into data/raw/
plodds build-db          # clean them and build data/pl.db
plodds summary           # matches and odds coverage per season
plodds table 2023-24     # final league table, as a check on the data
pytest                   # run the tests
```

## Project structure

```
src/plodds/
├── config.py      seasons, paths, bookmaker column names
├── download.py    fetch raw CSVs
├── clean.py       raw CSV -> tidy matches and odds tables
├── db.py          SQLite schema, loading, queries
├── cli.py         the `plodds` command
├── odds.py        odds -> margin-free probabilities
├── model.py       Poisson score model
├── evaluate.py    log loss, Brier score, calibration
└── backtest.py    walk-forward predictions and betting simulation
tests/             one test file per module, plus a small made-up season file
docs/PLAN.md       build plan
```

## Data

Match results and odds come from [football-data.co.uk](https://www.football-data.co.uk/),
which publishes them free of charge. The data is not stored in this repository;
`plodds download` fetches it.
