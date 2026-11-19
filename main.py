"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, the percent of the
players Exophase tracks who will have earned it at the case's target, as five
quantiles (5%, 25%, 50%, 75%, 95%).

Model (written from scratch for this entry; standard library only, no network,
fully deterministic):

Exophase shows, as of its last refresh, the players tracked N and, per trophy,
the earn rate r = 100 * E / N, where E is how many tracked players hold the
trophy.  Both counts only grow.  A trophy's rate therefore moves under two
forces: dilution (new players join holding few trophies, so r falls) and
progression (tracked players earn more trophies over time, so r rises).  A
persistence forecast ignores both; a log-time curve fit confounds them.

This program separates them.  Per trophy it fits a progression curve

    F(a) = p * (1 - exp(-(a + a0) / tau))

the share of players who hold the trophy `a` hours after they first appear in
Exophase's tracked set (a0 is a global pre-age: players usually play a while
before they sync, so they arrive holding trophies already).  Players arrive
in cohorts at each refresh; the earns at any time are the sum over cohorts of
cohort size times F(age).  With counts E = r * N / 100 at each distinct read,
(p, tau) are fitted per trophy by weighted least squares (closed form in p
for each tau) and a0 is fitted globally.  The tracked count N(t) is
extrapolated to the target by blending a power law and a saturating
exponential, both fitted on the reads.  The forecast then steps the cohorts
forward to the target: old cohorts age (progression), new cohorts join
(dilution).

The median is the median over a deterministic 54-scenario ensemble (3 N
growth shapes x 3 pre-ages a0 x 3 progression speeds x 2 progression-curve
families, Weibull and lognormal), weighted by how well each scenario fits
the history; the spread is the ensemble's spread of logit rates, with a
floor, so the quantiles widen when the history leaves the future genuinely
uncertain.  The median is then
blended 75/25 with the house curve baseline's median (a per-trophy
least-squares line of logit rate on ln(1 + hours since release) over the
last 24 hours of reads, read at the target): the two models fail in
different regimes, so their average is more robust than either alone.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.  -P
keeps this file's directory OFF sys.path, so a module shipped beside main.py
imports only after the sys.path line below; -s drops user site-packages.
This entry is a single file, so no import is needed.

What it must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON {"schema":
    "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes", "target",
    "levels", "game", "trophies", "history", "notes"}, where `trophies` lists
    the case's trophies {id, name, description, tier, points, secret} and
    `history` every read before `at`, oldest first, each {slot, at,
    players_tracked, trophies_earned, rates}, `rates` in the order of
    `trophies` (null where that read lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q":
    [q05, q25, q50, q75, q95]}, ...]}, one row for each of the case's
    trophies, each id once, no other key, each q five finite numbers from 0
    to 100 (percent), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  This entry is one file, but the
# line keeps the layout the kit documents.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- small numeric helpers -------------------------------------------------

def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    c = clamp_rate(pct)
    return math.log(c / (100.0 - c))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000)
    / 10000 rounds it.  A whole number comes back as an int, so json.dumps
    writes 100 where JavaScript writes 100 (not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    # datetime.fromisoformat handles a trailing Z from Python 3.11 on.
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))


def normsinv(p: float) -> float:
    """Acklam's rational approximation of the standard normal quantile."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("normsinv needs 0 < p < 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --- history preparation ---------------------------------------------------

def distinct_snapshots(case: dict):
    """Collapse consecutive identical reads to one snapshot each.

    Exophase's page only changes when it refreshes, so a run of identical
    reads is one observation.  Returns a list of (time_hours, N, rates)
    oldest first, where time_hours is hours since the game's release
    (release_date at 00:00 UTC), and rates is the raw list (may hold None).
    """
    game = case.get("game") or {}
    rel = game.get("release_date") or ""
    try:
        release = datetime.fromisoformat(rel).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        release = None
    snaps = []
    prev_key = None
    for read in case.get("history") or []:
        try:
            t = parse_time(read["slot"])
        except (KeyError, ValueError, TypeError):
            continue
        n = read.get("players_tracked")
        rates = read.get("rates") or []
        if n is None or n <= 0:
            continue
        key = (n, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        if release is None:
            release = t
        hours = (t - release).total_seconds() / 3600.0
        snaps.append((hours, float(n), list(rates)))
    return snaps, release


def last_rates(case: dict, snaps):
    """Each trophy's rate in the last read that lists it (None if never).

    Read from the raw history, not the snapshots: a read with zero players
    tracked still lists rates (all 0.0), and that is information."""
    n_tr = len(case["trophies"])
    out = [None] * n_tr
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or []
        for i in range(n_tr):
            if out[i] is None and i < len(rates) and rates[i] is not None:
                out[i] = float(rates[i])
        if all(v is not None for v in out):
            break
    return out


# --- tracked-players growth ------------------------------------------------# --- progression curve -----------------------------------------------------

# tau grid for the progression curve (hours), log-spaced, then refined.
TAU_GRID = [1.5, 2.5, 4.0, 7.0, 12.0, 20.0, 35.0, 60.0, 100.0, 170.0, 300.0,
            500.0, 850.0, 1500.0, 2500.0, 4500.0]
# Global pre-age grid (hours): how long a player has typically played before
# first appearing in Exophase's tracked set.
A0_GRID = [0.0, 6.0, 12.0, 24.0, 48.0]
# Two families for the progression curve F(a) = p * S((a + a0) / tau):
#   weib: S(x) = 1 - exp(-x^gamma), gamma in WEIBULL_GAMMAS (1 is the
#         exponential; > 1 is S-shaped with a sharp saturation);
#   logn: S(x) = Phi((ln x - ln tau) / sigma), sigma in LOGN_SIGMAS, a
#         lognormal delay, which keeps rising on a log-time scale and
#         saturates only far beyond the horizon.
# Both families can fit a two-day window almost equally well while
# forecasting very different weeks: a trophy whose earn rate is still
# accelerating fits a sharply saturating Weibull and a still-rising
# lognormal about equally.  The ensemble therefore keeps the best fit of
# EACH family per trophy instead of picking one, and lets their disagreement
# widen the quantiles.
WEIBULL_GAMMAS = (1.0, 1.5, 2.0)
LOGN_SIGMAS = (0.8, 1.2, 1.8)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def shape_f(age: float, tau: float, a0: float, kind: str, shape: float) -> float:
    """F(age) without the p factor: the share of players holding the trophy
    `age` hours after first appearing in the tracked set, if every player
    eventually held it."""
    x = age + a0
    if x <= 0:
        return 0.0
    if kind == "weib":
        return 1.0 - math.exp(-((x / tau) ** shape))
    return _phi((math.log(x) - math.log(tau)) / shape)


def thin_snaps(snaps, max_n=48):
    """Keep at most max_n snapshots for the fit (first and last always).
    Refreshes are hours apart, so this rarely bites; it only bounds the
    O(J^2) cohort arithmetic if Exophase ever refreshed every few minutes."""
    if len(snaps) <= max_n:
        return snaps
    keep = [0]
    step = (len(snaps) - 1) / (max_n - 1)
    for k in range(1, max_n - 1):
        keep.append(round(k * step))
    keep.append(len(snaps) - 1)
    return [snaps[i] for i in sorted(set(keep))]


def cohort_factors(snaps, tau, a0, kind="weib", shape=1.0, cohorts=None):
    """G[j] = expected earns per tracked player at snapshot j if every
    player eventually held the trophy (p = 1), given the cohorts.

    Cohort k is the players who first appear at snapshot k: for k = 0 they
    are taken as joining at snapshot 0's slot; for k >= 1 they join uniformly
    over (snapshot k-1's slot, snapshot k's slot].  G[j] sums, over cohorts
    up to j, cohort size times the average of F(age) over the cohort's join
    window.  For the exponential shape the window average is exact in closed
    form; otherwise it is evaluated at the window's midpoint, which is exact
    enough when the window (hours) is short against tau.
    `cohorts`, when given, is the precomputed [(dn, u, v, mid), ...] list.
    """
    times = [s[0] for s in snaps]
    if cohorts is None:
        cohorts = []
        ns = [s[1] for s in snaps]
        for k in range(len(snaps)):
            dn = ns[k] - (ns[k - 1] if k > 0 else 0.0)
            if k == 0:
                u = v = times[0]
            else:
                u, v = times[k - 1], times[k]
            cohorts.append((dn, u, v, 0.5 * (u + v)))
    out = []
    for j in range(len(snaps)):
        tj = times[j]
        total = 0.0
        for k in range(j + 1):
            dn, u, v, mid = cohorts[k]
            if dn <= 0:
                continue
            if kind == "weib" and shape == 1.0 and v - u > 1e-9:
                try:
                    avg_exp = math.exp(-(tj + a0) / tau) * (tau / (v - u)) * \
                        (math.exp(v / tau) - math.exp(u / tau))
                except OverflowError:
                    avg_exp = 0.0
                f = 1.0 - avg_exp
            else:
                f = shape_f(tj - mid, tau, a0, kind, shape)
            total += dn * min(max(f, 0.0), 1.0)
        out.append(total)
    return out


class Fitter:
    """Caches the cohort factors G(tau, a0, kind, shape) (trophy-independent)
    and fits (tau, shape, p) per trophy per family by weighted least squares
    on the earn counts."""

    def __init__(self, snaps, obs_by_trophy):
        self.snaps = snaps
        self.obs = obs_by_trophy
        self.cache = {}
        # Per-trophy precomputed weights: (snapshot index, E, w, w*E).
        # The weights do not depend on (tau, a0, kind, shape), so computing
        # them once keeps the tens of thousands of fit_p calls cheap.
        self.wobs = []
        for obs in obs_by_trophy:
            lst = []
            for (j, e, n) in obs:
                w = 1.0 / (e + 0.5 + (0.00005 * n) ** 2 + (0.02 * e) ** 2)
                lst.append((j, e, w, w * e))
            self.wobs.append(lst)
        # Precomputed cohorts: (size, window start, window end, midpoint).
        times = [s[0] for s in snaps]
        ns = [s[1] for s in snaps]
        self.cohorts = []
        for k in range(len(snaps)):
            dn = ns[k] - (ns[k - 1] if k > 0 else 0.0)
            if k == 0:
                u = v = times[0]
            else:
                u, v = times[k - 1], times[k]
            self.cohorts.append((dn, u, v, 0.5 * (u + v)))

    def g(self, tau, a0, kind="weib", shape=1.0):
        key = (round(tau, 6), round(a0, 6), kind, round(shape, 6))
        got = self.cache.get(key)
        if got is None:
            got = cohort_factors(self.snaps, tau, a0, kind, shape, self.cohorts)
            self.cache[key] = got
        return got

    def fit_p(self, obs, tau, a0, kind="weib", shape=1.0):
        """Best p (closed form) and SSE for one trophy at (tau, a0, kind,
        shape).  Weighted least squares on the earn counts, each snapshot
        weighted by the inverse variance of its count: Poisson noise (E),
        the page's 2-decimal rounding ((0.00005 * N)^2), and a 2% relative
        model-error floor.  Without the rounding term the earliest reads,
        whose counts are tiny and rounding-dominated, would outweigh the
        precise recent reads and the fit would chase the flat early part."""
        g = self.g(tau, a0, kind, shape)
        num = 0.0
        den = 0.0
        for (j, e, w, we) in obs:
            num += we * g[j]
            den += w * g[j] * g[j]
        p = num / den if den > 1e-12 else 0.0
        p = min(max(p, 0.001), 1.0)
        sse = 0.0
        for (j, e, w, we) in obs:
            d = p * g[j] - e
            sse += w * d * d
        return p, sse

    def _refine_tau(self, obs, tau, a0, kind, shape):
        p, sse = self.fit_p(obs, tau, a0, kind, shape)
        for _ in range(2):
            improved = False
            for cand in (tau * 0.75, tau * 1.3333333333):
                if cand < 1.0 or cand > 6000.0:
                    continue
                pc, ssec = self.fit_p(obs, cand, a0, kind, shape)
                if ssec < sse - 1e-12:
                    tau, p, sse = cand, pc, ssec
                    improved = True
            if not improved:
                break
        return tau, p, sse

    def _fit_family(self, wobs, a0, kind):
        """Best (tau, shape, p, sse) of one family for one trophy."""
        shapes = WEIBULL_GAMMAS if kind == "weib" else LOGN_SIGMAS
        best = None
        for shape in shapes:
            bg = None
            for tau in TAU_GRID:
                p, sse = self.fit_p(wobs, tau, a0, kind, shape)
                if bg is None or sse < bg[2]:
                    bg = (tau, p, sse)
            tau, p, sse = self._refine_tau(wobs, bg[0], a0, kind, shape)
            if best is None or sse < best[3]:
                best = (tau, shape, p, sse)
        return (kind,) + best

    def fit_all(self, a0):
        """Best fit of each family per trophy at this a0; returns
        (candidates, total_sse) where candidates[i] is a list of
        (kind, tau, shape, p, sse) and total_sse sums each trophy's best
        family SSE (used to pick the global pre-age)."""
        cands = []
        total = 0.0
        for wobs in self.wobs:
            if len(wobs) < 2:
                cands.append(None)
                continue
            cw = self._fit_family(wobs, a0, "weib")
            cl = self._fit_family(wobs, a0, "logn")
            cands.append([cw, cl])
            total += min(cw[4], cl[4])
        return cands, total


def cohort_age_factor(t_target, u, v, tau, a0, kind="weib", shape=1.0):
    """Average of F(age) at the target over players who joined uniformly in
    (u, v]; u == v means they joined at u.  Exact window average for the
    exponential shape, midpoint for other shapes."""
    if kind == "weib" and shape == 1.0 and v - u > 1e-9:
        try:
            avg_exp = math.exp(-(t_target + a0) / tau) * (tau / (v - u)) * \
                (math.exp(v / tau) - math.exp(u / tau))
        except OverflowError:
            avg_exp = 0.0
        return min(max(1.0 - avg_exp, 0.0), 1.0)
    mid = 0.5 * (u + v)
    return min(max(shape_f(t_target - mid, tau, a0, kind, shape), 0.0), 1.0)


def forecast_counts(snaps, params, a0, n_path, t_target):
    """E at the target per trophy: old cohorts age, future cohorts join.

    n_path is N(t) at steps from the last snapshot to the target; future
    joiners between steps are spread uniformly over the step.  The path is
    first compressed to at most 12 bins (the integral is smooth, so a dozen
    bins are plenty) to keep the 54-scenario ensemble fast.  params[i] is
    (tau, kind, shape, p) for trophy i, or (None, None, None, None).
    """
    times = [s[0] for s in snaps]
    ns = [s[1] for s in snaps]
    # Compress the future path into at most 12 bins.
    bins = []
    if len(n_path) > 13:
        nb = 12
        per = (len(n_path) - 1) / nb
        for b in range(nb):
            i0 = round(b * per)
            i1 = round((b + 1) * per)
            bins.append((n_path[i0][0], n_path[i1][0], n_path[i1][1] - n_path[i0][1]))
    else:
        for s in range(len(n_path) - 1):
            bins.append((n_path[s][0], n_path[s + 1][0], n_path[s + 1][1] - n_path[s][1]))
    out = []
    for (tau, kind, shape, p) in params:
        if tau is None:
            out.append(None)
            continue
        total = 0.0
        for k in range(len(snaps)):
            dn = ns[k] - (ns[k - 1] if k > 0 else 0.0)
            if dn <= 0:
                continue
            if k == 0:
                f = cohort_age_factor(t_target, times[0], times[0], tau, a0, kind, shape)
            else:
                f = cohort_age_factor(t_target, times[k - 1], times[k], tau, a0, kind, shape)
            total += dn * f
        for (t_a, t_b, dn) in bins:
            if dn <= 0:
                continue
            total += dn * cohort_age_factor(t_target, t_a, t_b, tau, a0, kind, shape)
        out.append(p * total)
    return out


# --- ensemble forecast -----------------------------------------------------

# Floor and cap for the logit spread of the quantiles.  The floor keeps the
# forecast honest when the ensemble agrees too well: the future still holds
# refresh timing and model error the ensemble cannot see.
SPREAD_FLOOR = 0.45
SPREAD_CAP = 1.6


def n_models(snaps, t_target):
    """(N at target, hourly path, weight) for each growth shape: power law,
    saturating exponential, and their geometric-mean blend.  The weights
    come from each shape's in-sample log fit, so a shape the history
    contradicts is downweighted instead of dragging the blend.  Returns
    None when the history shows no N growth at all (flat history)."""
    if not snaps:
        return None
    n0 = snaps[-1][1]
    t0 = snaps[-1][0]
    xs = [s[0] for s in snaps]
    ns = [s[1] for s in snaps]
    if len(set(ns)) <= 1 or len(snaps) < 3:
        return None
    lx = [math.log(1.0 + max(x, -0.9)) for x in xs]
    ly = [math.log(n) for n in ns]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    sxx = sum((v - mx) ** 2 for v in lx)
    b = (sum((lx[i] - mx) * (ly[i] - my) for i in range(len(lx))) / sxx) if sxx > 1e-12 else 0.0
    a = my - b * mx
    b = min(max(b, 0.0), 3.0)

    def n_power(t):
        return math.exp(a + b * math.log(1.0 + max(t, -0.9)))

    sse_pow = sum((math.log(max(n_power(x), 1e-9)) - ly[i]) ** 2 for i, x in enumerate(xs))
    best = None
    for tau in (3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0):
        ys = [1.0 - math.exp(-(x + 1.0) / tau) for x in xs]
        den = sum(y * y for y in ys)
        if den <= 1e-12:
            continue
        ninf = sum(ns[i] * ys[i] for i in range(len(ns))) / den
        if ninf <= 0:
            continue
        sse = sum((math.log(max(ninf * ys[i], 1e-9)) - ly[i]) ** 2 for i in range(len(ns)))
        if best is None or sse < best[0]:
            best = (sse, tau, ninf)
    if best is None:
        def n_sat(t):
            return n_power(t)
        sse_sat = sse_pow
    else:
        sse_sat, tau_sat, ninf_sat = best

        def n_sat(t, tau_sat=tau_sat, ninf_sat=ninf_sat):
            return ninf_sat * (1.0 - math.exp(-(t + 1.0) / tau_sat))

    # Likelihood-style weights from the log fits, with a noise floor so a
    # tiny in-sample edge cannot annihilate a shape, and a cap on the ratio.
    npts = len(xs)
    sig2 = max(min(sse_pow, sse_sat) / max(npts - 2, 1), 0.02 ** 2)
    smin = min(sse_pow, sse_sat)
    w_pow = math.exp(-(sse_pow - smin) / (2.0 * sig2))
    w_sat = math.exp(-(sse_sat - smin) / (2.0 * sig2))
    w_pow = min(max(w_pow, 0.1), 1.0)
    w_sat = min(max(w_sat, 0.1), 1.0)
    w_blend = math.sqrt(w_pow * w_sat)

    def path_of(fn):
        span = t_target - t0
        steps = max(1, int(math.ceil(span)))
        path = []
        for k in range(steps + 1):
            t = t0 + span * k / steps
            path.append((t, max(fn(t), n0)))
        nt = max(fn(t_target), n0)
        nt = min(nt, n0 * 100.0)
        path[-1] = (t_target, nt)
        return nt, path

    out = {}
    for name, fn, w in (("power", n_power, w_pow), ("sat", n_sat, w_sat)):
        nt, path = path_of(fn)
        out[name] = (nt, path, w)
    nt = math.sqrt(out["power"][0] * out["sat"][0])
    nt = min(max(nt, n0), n0 * 100.0)
    path = []
    pp = out["power"][1]
    ps = out["sat"][1]
    for k in range(len(pp)):
        path.append((pp[k][0], math.sqrt(max(pp[k][1], 1.0) * max(ps[k][1], 1.0))))
    path[-1] = (t_target, nt)
    out["blend"] = (nt, path, w_blend)
    return out


def weighted_median(pairs):
    """pairs: [(value, weight), ...] -> weighted median."""
    tot = sum(w for _v, w in pairs)
    if tot <= 0:
        vals = sorted(v for v, _w in pairs)
        return vals[len(vals) // 2]
    acc = 0.0
    for v, w in sorted(pairs):
        acc += w
        if acc >= 0.5 * tot:
            return v
    return sorted(pairs)[-1][0]


# The house curve baseline's weight in the final median (see forecast_case):
# the structural model's median is blended 75/25 with a per-trophy
# least-squares line of logit rate on ln(1 + hours since release) over the
# last 24 hours of reads, read at the target.  The two models fail in
# different regimes -- the curve extrapolates a steady log-time rise well
# when growth saturates, the structural model handles dilution and
# progression -- so their average is more robust than either alone.
CURVE_BLEND_W = 0.25


def curve_medians(case, release, t_at, t_target, n_tr):
    """Per-trophy median of the log-time curve fit (the house curve
    baseline, reconstructed): least squares of logit rate on ln(1 + hours
    since release) over the case's last 24 hours of reads, read at the
    target; fewer than 3 points fall back to the last rate.

    Note the convention, the house's: a read counts at its own slot time,
    carrying the page's values as of Exophase's last refresh before it --
    so a stale read at the window's start still counts as a point there.
    Reads are deduplicated per trophy by (players_tracked, rate)."""
    x_target = math.log(1.0 + max(t_target, -0.9))
    at = parse_time(case["at"])
    out = []
    for i in range(n_tr):
        pts = []
        seen = set()
        for read in case.get("history") or []:
            try:
                t = parse_time(read["slot"])
            except (KeyError, ValueError, TypeError):
                continue
            if (at - t).total_seconds() > 24 * 3600:
                continue
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            key = (read.get("players_tracked"), rates[i])
            if key in seen:
                continue
            seen.add(key)
            h = (t - release).total_seconds() / 3600.0 if release else 0.0
            pts.append((math.log(1.0 + max(h, -0.9)), logit(rates[i])))
        last = None
        for read in reversed(case.get("history") or []):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last = float(rates[i])
                break
        if len(pts) < 3:
            out.append(last if last is not None else 1.0)
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx > 1e-12 else 0.0
        a = my - b * mx
        out.append(clamp_rate(inv_logit(a + b * x_target)))
    return out


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zs = []
    for lv in levels:
        if abs(lv - 0.05) < 1e-9:
            zs.append(-1.645)
        elif abs(lv - 0.25) < 1e-9:
            zs.append(-0.674)
        elif abs(lv - 0.5) < 1e-9:
            zs.append(0.0)
        elif abs(lv - 0.75) < 1e-9:
            zs.append(0.674)
        elif abs(lv - 0.95) < 1e-9:
            zs.append(1.645)
        else:
            zs.append(normsinv(lv))
    trophies = case["trophies"]
    n_tr = len(trophies)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    snaps, release = distinct_snapshots(case)
    t_target = (target - release).total_seconds() / 3600.0 if release else horizon_days * 24.0
    rates_last = last_rates(case, snaps)

    # Fallback when there is nothing to learn from: a single distinct read
    # means no refresh has been seen yet.  Persisting the rate would ignore
    # the refresh that is coming; instead dilute by the prior growth
    # sqrt(1 + days) (newcomers arrive holding no trophies yet), with a
    # wide spread.  With no history at all there is no rate to dilute.
    if len(snaps) < 2:
        grow = math.sqrt(1.0 + horizon_days)
        rows = []
        for i, tr in enumerate(trophies):
            r = rates_last[i] if rates_last[i] is not None else 1.0
            lg = logit(r / grow)
            spread = 0.5 * math.sqrt(horizon_days) + 0.5
            rows.append({"id": tr["id"], "q": [round4(clamp_rate(inv_logit(lg + z * spread))) for z in zs]})
        return {"trophies": rows}

    fit_snaps = thin_snaps(snaps)
    # Per-trophy observations: (snapshot index in fit_snaps, E, N).
    obs_by_trophy = []
    for i in range(n_tr):
        obs = []
        for j, (_h, n, rates) in enumerate(fit_snaps):
            if i < len(rates) and rates[i] is not None:
                obs.append((j, float(rates[i]) * n / 100.0, n))
        obs_by_trophy.append(obs)

    fitter = Fitter(fit_snaps, obs_by_trophy)
    # Global pre-age by total fit (each trophy's best family).
    best_a0, best_cands, best_sse = None, None, None
    for a0 in A0_GRID:
        cands, sse = fitter.fit_all(a0)
        if best_sse is None or sse < best_sse:
            best_a0, best_cands, best_sse = a0, cands, sse

    models = n_models(snaps, t_target)
    if models is None:
        # Flat history: no refresh has been seen, so no growth can be
        # fitted.  Assume a modest one instead of none at all: the page is
        # refreshed on Exophase's own schedule and the launch-night
        # recording shows it does move.  N grows like sqrt(1 + days);
        # newcomers arrive holding trophies at the fitted progression
        # curve's pace, so the rates fall by dilution.
        grow = math.sqrt(1.0 + horizon_days)
        t0 = snaps[-1][0]
        n0 = snaps[-1][1]
        n_target = n0 * grow
        steps = max(1, int(math.ceil((t_target - t0))))
        path = []
        for k in range(steps + 1):
            f = 1.0 + (grow - 1.0) * k / steps
            path.append((t0 + (t_target - t0) * k / steps, n0 * f))
        models = {"blend": (n_target, path, 1.0)}

    # Ensemble: 3 N shapes x 3 pre-ages x 3 progression speeds x 2 curve
    # families.  A scenario's weight is its N shape's weight times how well
    # its (pre-age, speed, family) fits that trophy's history, so a
    # parameter combination the data already rules out earns little weight
    # while two families that fit equally well both count.
    a0s = sorted(set([max(best_a0 / 2.0, 0.0), best_a0, best_a0 * 2.0]))
    tau_scales = (0.7, 1.0, 1.4285714286)
    scen_rates = []   # per scenario: [rates per trophy]
    scen_sse = []     # per scenario: [SSE per trophy]
    scen_w = []       # per scenario: N-shape weight
    for mname in ("power", "blend", "sat"):
        if mname not in models:
            continue
        n_target_m, path_m, w_m = models[mname]
        for a0 in a0s:
            for tscale in tau_scales:
                for fam in range(2):
                    params = []
                    sses = []
                    for i, cand_list in enumerate(best_cands):
                        if cand_list is None:
                            params.append((None, None, None, None))
                            sses.append(0.0)
                            continue
                        kind, tau_i, shape_i, _p, _sse = cand_list[fam]
                        tau_s = min(max(tau_i * tscale, 1.0), 6000.0)
                        p_s, sse_s = fitter.fit_p(fitter.wobs[i], tau_s, a0, kind, shape_i)
                        params.append((tau_s, kind, shape_i, p_s))
                        sses.append(sse_s)
                    counts = forecast_counts(fit_snaps, params, a0, path_m, t_target)
                    scen_rates.append([100.0 * c / n_target_m if c is not None else None
                                       for c in counts])
                    scen_sse.append(sses)
                    scen_w.append(w_m)

    # Log-time curve baseline medians, for the 75/25 median blend.
    t_at = (at - release).total_seconds() / 3600.0 if release else 0.0
    curve_med = curve_medians(case, release, t_at, t_target, n_tr)

    rows = []
    for i, tr in enumerate(trophies):
        sse_min = None
        for sses in scen_sse:
            s = sses[i]
            if s > 0 and (sse_min is None or s < sse_min):
                sse_min = s
        vals = []
        for k in range(len(scen_rates)):
            r = scen_rates[k][i]
            if r is None:
                continue
            s = scen_sse[k][i]
            if sse_min is not None and s > 0:
                w_fit = min(max(sse_min / s, 0.05), 1.0)
            else:
                w_fit = 1.0
            vals.append((r, scen_w[k] * w_fit))
        last = rates_last[i]
        if not vals:
            # Never observed: fall back to the last rate if any, else a
            # small rate, with a wide spread.
            med = last if last is not None else 0.05
            spread = 0.5 * math.sqrt(horizon_days) + 0.7
            lg = logit(med)
            rows.append({"id": tr["id"], "q": [round4(clamp_rate(inv_logit(lg + z * spread))) for z in zs]})
            continue
        med = min(max(weighted_median(vals), 0.01), 99.99)
        # Blend with the log-time curve baseline's median (CURVE_BLEND_W).
        lg = (1.0 - CURVE_BLEND_W) * logit(med) + CURVE_BLEND_W * logit(curve_med[i])
        med = min(max(inv_logit(lg), 0.01), 99.99)
        lgs = [(logit(v), w) for (v, w) in vals]
        tot_w = sum(w for _v, w in lgs)
        mean_lg = sum(v * w for (v, w) in lgs) / tot_w
        var = sum(w * (v - mean_lg) ** 2 for (v, w) in lgs) / tot_w
        spread = math.sqrt(var + 0.20 ** 2)
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        lg = logit(med)
        qs = []
        prev = -1.0
        for z in zs:
            q = clamp_rate(inv_logit(lg + z * spread))
            q = max(q, prev)
            qs.append(round4(q))
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
