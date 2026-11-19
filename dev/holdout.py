"""Holdout backtest: worlds never used for tuning, different seeds and
delay-shape sigma, to check the forecaster generalises.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
KIT = "/home/hatch/workspace/makefun-token-radar/campaign/gta6-trophy-rarity-forecast-01/kit/gta6-trophy-rarity-forecast-01-kit"

sys.path.insert(0, HERE)
from score import score_case  # noqa: E402
from simulate import World, DEADLINE  # noqa: E402
from backtest import run_forecaster  # noqa: E402

HOLDOUT = [
    ("W7 power b=0.8 pre=36 R=5 d=0.85", dict(seed=21, shape="power", beta=0.8, total=450_000, pre=36.0, refresh_h=5.0, d_scale=0.85)),
    ("W8 sat tau=120 pre=3 R=2 d=1.3", dict(seed=22, shape="sat", tau_n=120.0, total=700_000, pre=3.0, refresh_h=2.0, d_scale=1.3)),
    ("W9 power b=1.2 pre=18 R=10 d=0.9 s=1.5", dict(seed=23, shape="power", beta=1.2, total=250_000, pre=18.0, refresh_h=10.0, d_scale=0.9, sigma=1.5)),
    ("W10 sat tau=48 pre=24 R=4 d=1.1 s=0.9", dict(seed=24, shape="sat", tau_n=48.0, total=600_000, pre=24.0, refresh_h=4.0, d_scale=1.1, sigma=0.9)),
    ("W11 power b=0.7 pre=8 R=7 d=1.4", dict(seed=25, shape="power", beta=0.7, total=350_000, pre=8.0, refresh_h=7.0, d_scale=1.4)),
    ("W12 power b=1.0 pre=12 R=4 d=1.0 s2", dict(seed=26, shape="power", beta=1.0, total=500_000, pre=12.0, refresh_h=4.0, d_scale=1.0)),
]


def main():
    rows = []
    for wname, wkw in HOLDOUT:
        w = World(**wkw)
        case, ans = w.make_case("f00", DEADLINE, 6510)
        for fname, script in (("mine", os.path.join(WORK, "main.py")),
                              ("persist", os.path.join(KIT, "starter", "main.py")),
                              ("curve", os.path.join(HERE, "baseline_curve.py"))):
            outtxt, dt, err = run_forecaster(script, case, None)
            if outtxt is None:
                print(f"{wname} {fname}: CRASH {err}")
                rows.append((wname, fname, float("nan")))
                continue
            outp = os.path.join(HERE, "runs", wname.split()[0], "f00.%s.out" % fname)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            with open(outp, "w") as f:
                f.write(outtxt)
            s = score_case(outp, ans)
            rows.append((wname, fname, s["loss"], s["typical_miss_pp"], s["coverage_90"], dt))
    print(f"\n{'world':32s} {'model':8s} {'loss':>8s} {'miss_pp':>8s} {'cov90':>6s} {'secs':>6s}")
    for r in rows:
        print(f"{r[0]:32s} {r[1]:8s} {r[2]:8.4f} {r[3]:8.3f} {r[4]:6.2f} {r[5]:6.2f}")
    print("\nMean holdout loss by model:")
    for fname in ("mine", "persist", "curve"):
        vals = [r[2] for r in rows if r[1] == fname]
        print(f"  {fname:8s} {sum(vals) / len(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
