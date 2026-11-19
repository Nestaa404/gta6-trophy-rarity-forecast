"""Backtest: for each synthetic world, cut a final-like case (and two
practice-like cases), run the three forecasters, and score them.

Forecasters:
  mine   work/main.py
  persist  kit starter/main.py (house persistence baseline)
  curve  dev/baseline_curve.py (house curve baseline, reconstructed)

Usage: python3 dev/backtest.py [--worlds W1,W2] [--practice]
Writes cases under dev/runs/<world>/ and prints a loss table.
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
KIT = "/home/hatch/workspace/makefun-token-radar/campaign/gta6-trophy-rarity-forecast-01/kit/gta6-trophy-rarity-forecast-01-kit"

sys.path.insert(0, HERE)
from score import score_case, validate  # noqa: E402
from simulate import WORLDS, World, DEADLINE, TARGET  # noqa: E402

PRACTICE_AT = [datetime(2026, 11, 19, 3, 0, tzinfo=timezone.utc),
               datetime(2026, 11, 19, 3, 30, tzinfo=timezone.utc)]


def run_forecaster(script, case, outdir):
    with tempfile.TemporaryDirectory() as td:
        ind, outd = os.path.join(td, "in"), os.path.join(td, "out")
        os.makedirs(ind)
        os.makedirs(outd)
        with open(os.path.join(ind, f"{case['id']}.in"), "w") as f:
            json.dump(case, f)
        env = dict(os.environ, MAKEFUN_INPUT_DIR=ind, MAKEFUN_OUTPUT_DIR=outd)
        t0 = __import__("time").time()
        r = subprocess.run([sys.executable, script], env=env, capture_output=True,
                           text=True, timeout=60)
        dt = __import__("time").time() - t0
        if r.returncode != 0:
            return None, dt, r.stderr[-500:]
        outp = os.path.join(outd, f"{case['id']}.out")
        if not os.path.exists(outp):
            return None, dt, "no output file: " + r.stderr[-500:]
        with open(outp) as f:
            return f.read(), dt, ""


def main():
    only = None
    if "--worlds" in sys.argv:
        only = set(sys.argv[sys.argv.index("--worlds") + 1].split(","))
    do_practice = "--practice" in sys.argv
    rows = []
    for wname, wkw in WORLDS:
        if only and wname.split()[0] not in only:
            continue
        w = World(**wkw)
        cases = [("f00", *w.make_case("f00", DEADLINE, 6510))]
        if do_practice:
            for k, at in enumerate(PRACTICE_AT):
                cases.append((f"p0{k}", *w.make_case(f"p0{k}", at, 90)))
        for cid, case, ans in cases:
            ans_path = os.path.join(HERE, "runs", wname.split()[0], f"{cid}.actual.json")
            os.makedirs(os.path.dirname(ans_path), exist_ok=True)
            json.dump({str(k): v for k, v in ans.items()}, open(ans_path, "w"))
            for fname, script in (("mine", os.path.join(WORK, "main.py")),
                                  ("persist", os.path.join(KIT, "starter", "main.py")),
                                  ("curve", os.path.join(HERE, "baseline_curve.py"))):
                outtxt, dt, err = run_forecaster(script, case, None)
                if outtxt is None:
                    print(f"{wname} {cid} {fname}: CRASH {err}")
                    rows.append((wname, cid, fname, float("nan"), float("nan"), float("nan"), dt))
                    continue
                outp = os.path.join(HERE, "runs", wname.split()[0], f"{cid}.{fname}.out")
                with open(outp, "w") as f:
                    f.write(outtxt)
                s = score_case(outp, ans)
                probs = validate(outp, os.path.join(HERE, "runs", wname.split()[0], f"{cid}.in.json")) \
                    if False else []
                rows.append((wname, cid, fname, s["loss"], s["typical_miss_pp"], s["coverage_90"], dt))
                # keep the case file for inspection
                json.dump(case, open(os.path.join(HERE, "runs", wname.split()[0], f"{cid}.in.json"), "w"))
    print(f"\n{'world':28s} {'case':4s} {'model':8s} {'loss':>8s} {'miss_pp':>8s} {'cov90':>6s} {'secs':>6s}")
    for wname, cid, fname, loss, miss, cov, dt in rows:
        print(f"{wname:28s} {cid:4s} {fname:8s} {loss:8.4f} {miss:8.3f} {cov:6.2f} {dt:6.2f}")
    print("\nMean final-case loss by model:")
    for fname in ("mine", "persist", "curve"):
        vals = [r[3] for r in rows if r[1] == "f00" and r[2] == fname]
        if vals:
            print(f"  {fname:8s} {sum(vals) / len(vals):.4f}  (n={len(vals)})")
    if do_practice:
        print("Mean practice-case loss by model:")
        for fname in ("mine", "persist", "curve"):
            vals = [r[3] for r in rows if r[1].startswith("p0") and r[2] == fname]
            if vals:
                print(f"  {fname:8s} {sum(vals) / len(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
