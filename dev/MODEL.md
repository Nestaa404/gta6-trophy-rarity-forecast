# Model notes — gta6-trophy-rarity-forecast-01

## The problem

One case (`f00`) at the deadline, 2026-11-21T11:30Z: every valid read of
Exophase's GTA VI PS5 trophy page before then (early recording every
30 min, in-cluster recorder every 10 min), forecasting every trophy's earn
rate at 2026-11-26T00:00Z (horizon 6,510 min = 4.52 days), scored by mean
pinball loss on the logit scale over 5 quantiles × 55 trophies. A prize
needs a loss strictly below the best house entry (persistence baseline,
curve baseline, house model forecast).

The practice cases (p00/p01, 90-min horizons on launch night) are almost
uninformative: their histories are a single flat page (638 players, rates
unchanged for 4.5 h), and their answers are the first refresh. They check
validity and refresh-awareness, not the final's week. The final's history
will contain real refreshes over ~2.5 days.

## Model

Work in counts: at each distinct read, `E_i = r_i * N / 100` (earned) and
`N` (tracked). Both only grow.

**Progression.** Per trophy, fit `F(a) = p * S((a + a0) / tau)`: the share
of players holding the trophy `a` hours after first appearing in the
tracked set. `a0` is a global pre-age (players play a while before they
sync, so they arrive holding trophies). Two families for `S`:

- Weibull `1 - exp(-x^gamma)`, gamma in {1, 1.5, 2} (1 = exponential;
  >1 = S-shaped with sharp saturation);
- lognormal `Phi((ln x - ln tau) / sigma)`, sigma in {0.8, 1.2, 1.8}
  (keeps rising on a log-time scale).

Both families can fit a two-day window almost equally well (e.g. W6
trophy 4: Weibull SSE 17, lognormal SSE 17.8) while forecasting very
different weeks (24% vs 40%), so the ensemble keeps the best fit of EACH
family per trophy instead of picking one. `p` is closed-form weighted
least squares per (tau, shape); weights are inverse variances of the
counts: Poisson noise, the page's 2-decimal rounding `((0.00005*N)^2)`,
and a 2% relative model-error floor. The rounding term matters: without
it the earliest reads (tiny counts, rounding-dominated) outweigh the
precise recent reads and the fit chases the flat early part.

**Tracked players.** `N(t)` is extrapolated by blending a power law in
`(1 + hours since release)` and a saturating exponential, weighted by
in-sample log fit, so a shape the history contradicts is downweighted
(this was the biggest single fix: an unweighted blend overestimated N(T)
in saturating worlds and under-predicted every rate).

**Forecast.** Cohorts (players first seen at each refresh) age to the
target; future cohorts join along the extrapolated N path. `E_i(T)` is
the sum over cohorts of size times `F(age at T)`; `r_i(T) = 100*E_i(T)/N(T)`.

**Ensemble.** 54 scenarios (3 N shapes × 3 pre-ages × 3 speeds × 2
families), weighted by N-shape fit and by each scenario's progression fit
for that trophy. Median = fit-weighted median; spread = weighted std of
logit rates, floored at 0.45, capped at 1.6. The median is blended 75/25
with the log-time curve baseline's median (per-trophy least squares of
logit rate on ln(1 + hours) over the last 24 h of reads, read at the
target — the house curve baseline, reconstructed; it uses each read's own
slot time, stale values included, exactly as the house describes it).

**Fallbacks.** One distinct read (no refresh seen): dilute by the prior
growth sqrt(1 + days) with a wide spread. No history at all: persist 1%
with a wide spread. Never-observed trophy: last rate or 0.05% with a wide
spread. All fallbacks are deterministic.

## What the backtests showed (and what changed the model)

Simulator (`dev/simulate.py`): players sync over the week (power-law or
saturating N), earn each trophy after a shifted lognormal delay (median by
tier/story order, jittered per world), Exophase refreshes every R hours,
reads every 10 min. Its delay model deliberately differs in form from the
forecaster's. 12 worlds, final-like cases (deadline + 4.52-day horizon):

| model | mean loss (12 worlds) |
|---|---|
| this entry | **0.163** |
| house persistence baseline (kit starter) | 0.808 |
| house curve baseline (reconstructed) | 0.316 |

The entry beats both baselines in 11 of 12 worlds and ties the 12th
(W6: 0.236 vs curve 0.234). Tuning worlds W1–W6: 0.167; unseen holdout
worlds W7–W12 (different seeds, delay sigma 0.9/1.5, different regimes):
0.159. Practice cases: entry 0.666 vs baselines 0.673.

Fixes that mattered, in order: N-shape fit weighting (mean 0.440 →
0.230); two progression families (0.241 → 0.192); corrected fit weights
with the rounding term (central fits improved sharply, e.g. W3 trophy 3
48.1% → 62.5% vs true 62.7%); fit-weighted scenario weights; 75/25
blend with the curve baseline (0.194 → 0.167).

## Runtime and determinism

Pure Python stdlib, no network, no randomness, no wall-clock use. ~0.5–1 s
on a final-like case (15–30 distinct snapshots); worst case tested
(refresh every 30 min, 48 thinned snapshots): 3.8 s, under the 8 s cap.
Two runs are byte-identical (checked on a final-like case and on 8 edge
cases, run with the grader's `python3 -P -s` and PYTHONHASHSEED=0).

## Files

- `main.py` — the entry (single file).
- `dev/score.py` — pinball scorer replicating the rules, plus output validation.
- `dev/simulate.py` — synthetic-world generator.
- `dev/baseline_curve.py` — house curve baseline, reconstructed from RULES.txt.
- `dev/backtest.py`, `dev/holdout.py` — run the three forecasters on synthetic cases.
- `dev/blend_test.py` — tuning of the 75/25 median blend (w = 0.25).
- `dev/edge_test.py` — edge cases + determinism.
