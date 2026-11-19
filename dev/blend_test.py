"""Tune the blend of my median with the house-curve baseline's median.

Reads the saved backtest outputs (dev/runs/<W>/f00.{mine,curve}.out and
f00.actual.json) and scores blends: median = inv_logit((1-w) logit(mine)
+ w logit(curve)), spread either kept or widened by the disagreement.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import pinball, LEVELS  # noqa: E402


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(p):
    c = clamp(p)
    return math.log(c / (100.0 - c))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def spread_of(q):
    return (logit(q[4]) - logit(q[0])) / (2 * 1.645)


def score_blend(mine, curve, actual, w, widen):
    losses = []
    for tid, y in actual.items():
        qm = mine[tid]
        qc = curve[tid]
        lg = (1 - w) * logit(qm[2]) + w * logit(qc[2])
        sm = spread_of(qm)
        if widen:
            sm = math.sqrt(sm * sm + (w * abs(logit(qm[2]) - logit(qc[2]))) ** 2)
        sm = min(max(sm, 0.45), 1.6)
        qs = [clamp(inv_logit(lg + z * sm)) for z in (-1.645, -0.674, 0, 0.674, 1.645)]
        for tau, qv in zip(LEVELS, qs):
            losses.append(pinball(y, qv, tau))
    return sum(losses) / len(losses)


def main():
    worlds = ["W1", "W2", "W3", "W4", "W5", "W6"]
    data = {}
    for wn in worlds:
        mine = {r["id"]: r["q"] for r in json.load(open(f"{HERE}/runs/{wn}/f00.mine.out"))["trophies"]}
        curve = {r["id"]: r["q"] for r in json.load(open(f"{HERE}/runs/{wn}/f00.curve.out"))["trophies"]}
        actual = {int(k): v for k, v in json.load(open(f"{HERE}/runs/{wn}/f00.actual.json")).items()}
        data[wn] = (mine, curve, actual)
    print(f"{'w':>5s} {'widen':>6s} " + " ".join(f"{wn:>7s}" for wn in worlds) + "    mean")
    for w in (0.0, 0.15, 0.25, 0.35, 0.5, 0.65):
        for widen in (False, True):
            vals = []
            for wn in worlds:
                mine, curve, actual = data[wn]
                vals.append(score_blend(mine, curve, actual, w, widen))
            print(f"{w:5.2f} {str(widen):>6s} " + " ".join(f"{v:7.4f}" for v in vals) +
                  f"    {sum(vals) / len(vals):.4f}")


if __name__ == "__main__":
    main()
