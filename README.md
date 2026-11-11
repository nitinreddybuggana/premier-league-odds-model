# Can a statistical model beat Premier League odds?

Bookmaker odds are prices. Like any price they imply a probability, they
include a margin, and you can ask whether they are efficient. This project
fits a Poisson model of football scores to ten seasons of Premier League
results, compares its probabilities with the bookmakers', and backtests a
value-betting strategy strictly out of sample.

> **Status: in progress.** The data pipeline, database and odds conversion are
> done and tested. The model, evaluation and backtest are being written.

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

## Findings so far

### The bookmaker's margin

Adding up 1/odds for home, draw and away gives more than 100%. The excess is
the *overround*, the bookmaker's margin. Averaged over 3,800 matches
(2016-17 to 2025-26):

| Odds | Average overround | Why |
|---|---|---|
| Bet365 | 4.68% | One firm's actual margin |
| Market average | 4.41% | Averaging prices leaves roughly a typical firm's margin in place |
| Best price in the market | 0.21% | Taking the top price on each outcome separately removes most of the margin |

The best-price overround was **negative in 1,585 matches (about 42%)**. When it
is negative, backing all three outcomes at the best prices, with stakes in
proportion to 1/odds, returns the same amount whatever the result: at −5%,
£100 staked returns £100 / 0.95 = £105.26. That is arbitrage.

No single bookmaker offers this. The best-price row is stitched together from
different firms, and each firm keeps its own margin: Bet365's overround was
never below 1.7% and the market average never below 3.4%. The opportunity comes
from bookmakers disagreeing with each other, not from any one of them offering
a losing book.

In practice these figures overstate what could be collected. The best three
prices may not have been available at the same moment, bookmakers limit
accounts that arbitrage, and the most extreme case (−12.9%) is more likely a
stale or mistyped quote than a real opportunity.

**Consequence for the backtest:** a betting simulation priced at the best
available odds would look better than anyone could achieve, so it uses Bet365
or market-average odds instead.

### Home advantage and crowds

The 2020-21 season was played behind closed doors, which makes it a natural
experiment on where home advantage comes from.

| Season | Home win % | Away win % | Average home goal margin |
|---|---|---|---|
| 2016-17 | 49.2 | 28.7 | +0.39 |
| 2017-18 | 45.5 | 28.4 | +0.38 |
| 2018-19 | 47.6 | 33.7 | +0.32 |
| 2019-20 | 45.3 | 30.5 | +0.31 |
| **2020-21** | **37.9** | **40.3** | **+0.01** |

| Period | Matches | Home win % | Away win % | Home goal margin |
|---|---|---|---|---|
| With crowds (Aug 2016 – Mar 2020) | 1,428 | 46.9 | 30.3 | +0.35 |
| Behind closed doors (Jun 2020 – May 2021) | 452 | 38.9 | 38.5 | +0.07 |

Without crowds, away teams won as often as home teams and the home goal margin
fell by about 80%. Crowds look like a large part of home advantage. (A few
matches in December 2020 had up to 2,000 fans; too few to change the picture.)

This does not show that home advantage differs *between teams*, which is a
separate question (see below).

*Source for this table: Premier League results from
[footballcsv/england](https://github.com/footballcsv/england), which covers
seasons up to 2020-21.*

## Model assumptions

**One home advantage for the whole league.** Every team gets the same home
boost. Some grounds may be genuinely harder to visit, but each team plays only
19 home games a season, so a separate estimate per team would mostly pick up
luck (a few narrow home wins would look like a "fortress"). One shared value
uses every match and is estimated precisely: a bias–variance trade-off.

Two known issues to test once the backtest works:

1. **Team-specific home advantage**, ideally with shrinkage towards the league
   average. Keep it only if it improves out-of-sample log loss.
2. **The closed-doors period.** Training data that includes 2020-21 pulls the
   shared home advantage down. Options: leave that period out of training, or
   let home advantage vary by season.

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
