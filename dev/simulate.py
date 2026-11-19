"""Synthetic-world simulator for backtesting the forecaster.

A world is a generative model of the real process, deliberately different in
form from the forecaster's model (lognormal earn delays and player speeds,
not the forecaster's exponential progression curve), so a win here is
evidence about the real final, not about fitting the forecaster to itself:

  - players start playing over the week; cumulative play-starts follow a
    power law or a saturating exponential in hours since release;
  - each player has a speed v (lognormal); trophy i needs d_i / v hours of
    play to earn (d_i by tier/story order, jittered per world);
  - a player first appears in Exophase's tracked set `pre` hours after
    starting to play, and then counts with the trophies earned by then;
  - Exophase refreshes the page every R hours; reads are every 10 minutes
    and show the last refresh's values, rates rounded to 2 decimals.

Generates final-like cases (deadline 2026-11-21T11:30Z, target
2026-11-26T00:00Z) and practice-like cases (03:00/03:30 checkpoints,
90-minute horizons) cut from the same world.
"""

import bisect
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import normsinv  # noqa: E402  (Acklam inverse normal CDF)

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
FIRST_READ = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)

KIT = "/home/hatch/workspace/makefun-token-radar/campaign/gta6-trophy-rarity-forecast-01/kit/gta6-trophy-rarity-forecast-01-kit"
TROPHIES = json.load(open(f"{KIT}/input/p00.in"))["trophies"]

# Median sync-age (hours) to earn each trophy: phenomenological delays that
# include play intensity, not raw play hours.  Calibrated so chapter 1 is
# ~30% at the deadline (59.5h) and ~60% at the target (168h), the platinum
# ~0.1% -> ~1%, and grind trophies stay in the low single digits.
BASE_D = {
    1: 1500.0, 2: 1.0,
    3: 130.0, 4: 170.0, 5: 220.0, 6: 280.0, 7: 400.0,
    14: 1200.0,
    15: 600.0, 17: 500.0, 25: 450.0, 30: 700.0, 31: 550.0, 40: 400.0,
}


def base_difficulties(seed=7):
    rng = random.Random(seed)
    out = {}
    for t in TROPHIES:
        tid = t["id"]
        if tid in BASE_D:
            out[tid] = BASE_D[tid]
        else:
            # bronze: log-uniform 120..900h of sync age
            out[tid] = math.exp(rng.uniform(math.log(120.0), math.log(900.0)))
    return out


def speed_bins(nbins=9, sigma=0.8):
    bins = []
    for k in range(nbins):
        z = normsinv((k + 0.5) / nbins)
        bins.append((math.exp(sigma * z), 1.0 / nbins))
    return bins


class World:
    """Sync-cohort model: players appear in the tracked set over the week
    (power-law or saturating cumulative N); a player who synced `a` hours
    ago holds trophy i with probability Phi((ln(a + pre) - ln d_i)/sigma),
    a shifted lognormal delay (pre = typical play before first sync)."""

    def __init__(self, seed, shape="power", beta=1.0, tau_n=72.0, total=500_000,
                 pre=12.0, refresh_h=4.0, d_scale=1.0, d_jitter=0.3, sigma=1.2):
        self.rng = random.Random(seed)
        self.shape, self.beta, self.tau_n = shape, beta, tau_n
        self.total, self.pre = total, pre
        self.refresh_h, self.d_scale = refresh_h, d_scale
        self.sigma = sigma
        base = base_difficulties()
        self.d = {tid: base[tid] * d_scale * math.exp(self.rng.gauss(0, d_jitter))
                  for tid in base}
        # hourly sync cohorts: (start_hour, count)
        self.cohorts = []
        prev = 0.0
        for h in range(0, 168):
            n_h = self.n_play(h + 1)
            if n_h - prev > 0:
                self.cohorts.append((float(h), n_h - prev))
            prev = n_h

    def n_play(self, t_h):
        if self.shape == "power":
            return self.total * (max(t_h, 0.0) / 168.0) ** self.beta
        return self.total * (1 - math.exp(-max(t_h, 0.0) / self.tau_n)) / \
            (1 - math.exp(-168.0 / self.tau_n))

    def _phi(self, z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def page_at(self, t_h):
        """True (N, {tid: E}) as of time t_h (hours since release)."""
        n = sum(c for (h, c) in self.cohorts if h <= t_h)
        es = {}
        for tr in TROPHIES:
            tid = tr["id"]
            d = self.d[tid]
            tot = 0.0
            for (h, c) in self.cohorts:
                if h > t_h:
                    break
                age = t_h - h + self.pre
                if age <= 0:
                    continue
                tot += c * self._phi((math.log(age) - math.log(d)) / self.sigma)
            es[tid] = tot
        return n, es

    def reads(self):
        """Every 10-min read from FIRST_READ to TARGET+70min, showing the
        last refresh's values."""
        out = []
        t = FIRST_READ
        end = TARGET + timedelta(minutes=70)
        off = self.refresh_h * 0.37
        # refresh page values, computed once per refresh time
        cache = {}

        def page_for(th):
            if th < off:
                return 0.0, {tid: 0.0 for tid in self.d}
            r_h = off + math.floor((th - off) / self.refresh_h) * self.refresh_h
            if r_h not in cache:
                cache[r_h] = self.page_at(r_h)
            return cache[r_h]

        while t <= end:
            th = (t - RELEASE).total_seconds() / 3600.0
            n, es = page_for(th)
            rates = []
            for tr in TROPHIES:
                rates.append(round(100.0 * es[tr["id"]] / n, 2) if n > 0 else 0.0)
            out.append({
                "slot": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "at": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "players_tracked": int(round(n)),
                "trophies_earned": int(round(sum(es.values()))),
                "rates": rates,
            })
            t += timedelta(minutes=10)
        return out

    def make_case(self, case_id, at, horizon_minutes):
        target = at + timedelta(minutes=horizon_minutes)
        all_reads = self.reads()
        hist = [r for r in all_reads if r["slot"] < at.strftime("%Y-%m-%dT%H:%M:%SZ")]
        lo = target.strftime("%Y-%m-%dT%H:%M:%SZ")
        hi = (target + timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ans = None
        for r in all_reads:
            if lo <= r["slot"] < hi:
                ans = {tr["id"]: r["rates"][k] for k, tr in enumerate(TROPHIES)}
                break
        case = {
            "schema": "makefun/gta6-trophies-case/v1",
            "id": case_id,
            "at": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon_minutes": horizon_minutes,
            "target": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19",
                     "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
            "trophies": TROPHIES,
            "history": hist,
            "notes": "synthetic world (dev only)",
        }
        return case, ans


WORLDS = [
    ("W1 power b=1.0 pre=12 R=4 d=1.0", dict(seed=11, shape="power", beta=1.0, total=500_000, pre=12.0, refresh_h=4.0, d_scale=1.0)),
    ("W2 power b=0.6 pre=24 R=6 d=1.0", dict(seed=12, shape="power", beta=0.6, total=400_000, pre=24.0, refresh_h=6.0, d_scale=1.0)),
    ("W3 sat tau=72 pre=6 R=3 d=0.7", dict(seed=13, shape="sat", tau_n=72.0, total=600_000, pre=6.0, refresh_h=3.0, d_scale=0.7)),
    ("W4 power b=1.4 pre=48 R=8 d=1.5", dict(seed=14, shape="power", beta=1.4, total=800_000, pre=48.0, refresh_h=8.0, d_scale=1.5)),
    ("W5 power b=1.0 pre=0 R=12 d=1.2", dict(seed=15, shape="power", beta=1.0, total=300_000, pre=0.0, refresh_h=12.0, d_scale=1.2)),
    ("W6 sat tau=36 pre=12 R=4 d=1.0", dict(seed=16, shape="sat", tau_n=36.0, total=500_000, pre=12.0, refresh_h=4.0, d_scale=1.0)),
]


if __name__ == "__main__":
    w = World(**WORLDS[0][1])
    case, ans = w.make_case("f00", DEADLINE, 6510)
    print("reads:", len(case["history"]), "last N:", case["history"][-1]["players_tracked"])
    print("answer N at target read:", sum(1 for _ in [0]))
    print("sample actual rates:", {k: ans[k] for k in (1, 2, 3, 4, 7, 14, 15, 32)})
    print("last rates:", {k: case["history"][-1]["rates"][k - 1] for k in (1, 2, 3, 4, 7, 14, 15, 32)})
