# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nestaa404/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nestaa404/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
tracks who will have earned it at **2026-11-26T00:00:00Z**, a week after
launch — as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, scored by
pinball loss on the logit scale.

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
`gta6-trophy-rarity-forecast-01` ($250, 20 paid places).

## How it works

Exophase shows, as of its last refresh, the players tracked *N* and, per
trophy, the earn rate *r = 100·E/N*. Both counts only grow, so a trophy's
rate moves under two forces: **dilution** (new players join holding few
trophies, so *r* falls) and **progression** (tracked players earn more
trophies over time, so *r* rises). A persistence forecast ignores both; a
log-time curve fit confounds them.

`main.py` separates them. Per trophy it fits a progression curve
*F(a) = p·S((a + a0)/τ)* — the share of players holding the trophy *a*
hours after first appearing in Exophase's tracked set — in two families
(Weibull and lognormal, which fit a two-day window equally well but forecast
different weeks), with a global pre-age *a0* (players usually play a while
before they sync). Players arrive in cohorts at each refresh; the tracked
count *N(t)* is extrapolated by blending a power law and a saturating
exponential, weighted by fit. The forecast steps the cohorts forward to the
target: old cohorts age, new cohorts join.

The median is the fit-weighted median over a deterministic 54-scenario
ensemble (3 N shapes × 3 pre-ages × 3 progression speeds × 2 curve
families), blended 75/25 with the log-time curve baseline's median; the
spread is the ensemble's spread of logit rates, with a floor. See
[`dev/MODEL.md`](dev/MODEL.md) for the full design and the local validation.

## Run it

Python 3.13, standard library only, no network, deterministic.

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.
Runs in ~1 s on a final-like case (limit: 8 s per case).

## Local validation

There is no local grade for this competition — the server grades every
entry. `dev/` holds a synthetic-world backtest harness (a simulator whose
delay model deliberately differs in form from the forecaster's, plus
reimplementations of both house baselines and the exact pinball scorer):

```sh
python3 dev/backtest.py --practice   # 6 tuning worlds, final + practice cases
python3 dev/holdout.py               # 6 unseen worlds, final cases only
python3 dev/edge_test.py             # edge cases + determinism (grader flags)
```

Across 12 synthetic worlds (final-like cases: 2 days of history, 4.5-day
horizon), mean pinball loss: **this entry 0.163**, house persistence
baseline 0.808, house curve baseline 0.316 — the entry beats both baselines
in 11 of 12 worlds and ties the 12th. On the kit's practice cases it scores
0.666 vs the baselines' 0.673 (the practice histories are flat, so almost
nothing is identifiable there; the practice pays nothing).

## License

MIT — see [LICENSE](LICENSE).
