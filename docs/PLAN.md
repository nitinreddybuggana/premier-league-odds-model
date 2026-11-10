# Build plan

Seven days, Tue 10 Nov to Mon 16 Nov. Each day ends with tests passing and a
commit. A function that is still a stub shows up as "skipped" in `pytest`, so
the skip count is the to-do list: 31 at the start, 0 at the end.

| Day | Goal | File | Check |
|---|---|---|---|
| 1 Tue | Set up, get the data, odds maths | `odds.py` | `pytest tests/test_odds.py` |
| 2 Wed | Likelihood function and fitting | `model.py` | first 4 tests in `test_model.py` |
| 3 Thu | Score matrix and outcome probabilities | `model.py` | all of `test_model.py` |
| 4 Fri | Scoring rules, walk-forward loop | `evaluate.py`, `backtest.py` | `test_evaluate.py`, walk-forward tests |
| 5 Sat | Betting simulation and metrics | `backtest.py` | all of `test_backtest.py` |
| 6 Sun | Run it for real, make the charts | `notebooks/` or a script | numbers you can explain |
| 7 Mon | README with results, tidy up, CV line | `README.md` | a stranger can follow it |

## Day 1

1. Clone the repo and install it (see the README).
2. `plodds download`, then `plodds build-db`, then `plodds summary`.
   Expect 380 matches in every finished season.
3. `plodds table 2023-24` and compare with the real final table. The points
   should match (Everton and Nottingham Forest were deducted points that
   season, so those two will show more than the official table).
4. Open the database and look around: `sqlite3 data/pl.db`, then `.schema`.
   Write two or three queries of your own, e.g. the average number of home
   and away goals per season. You will want that number on day 2.
5. Write `overround` and `implied_probabilities` in `src/plodds/odds.py`.
6. Use them: what is the average overround for Bet365, for the market
   average, and for the best price in the market? Why are they so different?

## Questions to be able to answer by the end

These are what an interviewer will ask.

* Why Poisson? What does the model assume, and where is that wrong?
* What is maximum likelihood, in one sentence?
* Why can't you estimate every team's attack and defence freely?
* What is look-ahead bias, and how does your backtest avoid it?
* Why score with log loss and not accuracy?
* Your model loses to the market. Why is that the expected result, and why is
  the project still worth something?
* Why fractional Kelly and not full Kelly?
* How many bets did the strategy place, and is that enough to tell skill
  from luck?

## Stretch goals, in order

1. Time-decay weighting in the likelihood (recent matches count more).
2. Dixon-Coles low-score correction.
3. Compare against closing odds as well as pre-match odds.
4. Shin's method for removing the margin.
