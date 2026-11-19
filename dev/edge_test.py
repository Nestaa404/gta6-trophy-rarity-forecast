"""Edge-case and determinism checks for main.py.

Builds synthetic cases: empty history, one read, null rates, all-zero
rates, a 100% trophy, a trophy never observed, a case with a single trophy,
and a case whose history has only one distinct N.  Runs main.py twice on
each (with the grader's -P -s flags and PYTHONHASHSEED=0) and checks the
outputs are valid and byte-identical.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
KIT = "/home/hatch/workspace/makefun-token-radar/campaign/gta6-trophy-rarity-forecast-01/kit/gta6-trophy-rarity-forecast-01-kit"

sys.path.insert(0, HERE)
from score import validate  # noqa: E402

BASE = json.load(open(f"{KIT}/input/p00.in"))


def run(case, tag):
    d = os.path.join(HERE, "runs", "edge", tag)
    ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
    os.makedirs(ind, exist_ok=True)
    os.makedirs(outd, exist_ok=True)
    with open(os.path.join(ind, f"{tag}.in"), "w") as f:
        json.dump(case, f)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=ind, MAKEFUN_OUTPUT_DIR=outd,
               PYTHONHASHSEED="0")
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-P", "-s", os.path.join(WORK, "main.py")],
                           env=env, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return f"CRASH rc={r.returncode}: {r.stderr[-400:]}"
        outs.append(open(os.path.join(outd, f"{tag}.out"), "rb").read())
    if outs[0] != outs[1]:
        return "NON-DETERMINISTIC: two runs differ"
    probs = validate(os.path.join(outd, f"{tag}.out"), os.path.join(ind, f"{tag}.in"))
    if probs:
        return f"INVALID: {probs}"
    size = len(outs[0])
    if size > 65536:
        return f"TOO LARGE: {size} bytes"
    return f"ok ({size} bytes)"


def clone(**kw):
    import copy
    c = copy.deepcopy(BASE)
    c.update(kw)
    return c


def main():
    cases = {}
    # empty history
    c = clone(id="e00")
    c["history"] = []
    cases["e00_empty_history"] = c
    # one read
    c = clone(id="e01")
    c["history"] = c["history"][:1]
    cases["e01_one_read"] = c
    # null rates sprinkled
    c = clone(id="e02")
    for r in c["history"]:
        r["rates"] = [None if (k % 7 == 0) else v for k, v in enumerate(r["rates"])]
    cases["e02_null_rates"] = c
    # all-zero rates
    c = clone(id="e03")
    for r in c["history"]:
        r["rates"] = [0.0] * 55
        r["players_tracked"] = 100
        r["trophies_earned"] = 0
    cases["e03_all_zero"] = c
    # a trophy never observed (always null)
    c = clone(id="e04")
    for r in c["history"]:
        r["rates"][10] = None
    cases["e04_trophy_never_seen"] = c
    # single trophy
    c = clone(id="e05")
    c["trophies"] = c["trophies"][:1]
    for r in c["history"]:
        r["rates"] = r["rates"][:1]
    cases["e05_single_trophy"] = c
    # long horizon, flat history (final-like shape, no movement)
    c = clone(id="e06", horizon_minutes=6510, target="2026-11-26T00:00:00Z",
              at="2026-11-21T11:30:00Z")
    cases["e06_flat_final_horizon"] = c
    # history with a big jump (simulated refresh)
    c = clone(id="e07")
    last = dict(c["history"][-1])
    last["slot"] = "2026-11-19T02:45:00Z"
    last["at"] = "2026-11-19T02:45:00.000Z"
    last["players_tracked"] = 3868
    last["trophies_earned"] = 4276
    last["rates"] = [round(v * 0.2, 2) if v < 100 else 100 for v in last["rates"]]
    c["history"] = c["history"] + [last]
    cases["e07_refresh_jump"] = c
    for tag, case in cases.items():
        print(f"{tag:28s} {run(case, tag)}")


if __name__ == "__main__":
    main()
