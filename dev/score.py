"""Local scorer replicating the competition's pinball loss on the logit scale.

loss item = 2 * max(tau * d, (tau - 1) * d), d = logit(y) - logit(q),
rates in percent clamped to [0.01, 99.99] before the logit.
Loss = mean over every (case, trophy, level).  Also reports typical miss
(mean |y - q50| in pp) and 90% coverage.
"""

import json
import math
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def pinball(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_case(out_path, actual_rates):
    """actual_rates: {trophy_id: rate}. Returns dict with loss, miss, cov."""
    out = json.load(open(out_path))
    rows = {r["id"]: r["q"] for r in out["trophies"]}
    losses = []
    misses = []
    covered = 0
    n = 0
    for tid, y in actual_rates.items():
        q = rows[tid]
        for tau, qv in zip(LEVELS, q):
            losses.append(pinball(y, qv, tau))
        misses.append(abs(y - q[2]))
        if q[0] <= y <= q[4]:
            covered += 1
        n += 1
    return {
        "loss": sum(losses) / len(losses),
        "typical_miss_pp": sum(misses) / len(misses),
        "coverage_90": covered / n if n else 0.0,
        "n_trophies": n,
    }


def validate(out_path, case_path):
    """Check the output shape the grader enforces."""
    case = json.load(open(case_path))
    out = json.load(open(out_path))
    problems = []
    if set(out.keys()) != {"trophies"}:
        problems.append(f"top keys {sorted(out.keys())}")
    ids = [r["id"] for r in out["trophies"]]
    want = [t["id"] for t in case["trophies"]]
    if sorted(ids) != sorted(want):
        problems.append("trophy ids mismatch")
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    for r in out["trophies"]:
        if set(r.keys()) != {"id", "q"}:
            problems.append(f"row keys {sorted(r.keys())}")
            break
        q = r["q"]
        if len(q) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                              or v < 0 or v > 100 for v in q):
            problems.append(f"bad q for id {r['id']}: {q}")
            break
        if any(q[i] > q[i + 1] for i in range(4)):
            problems.append(f"non-monotone q for id {r['id']}: {q}")
            break
    return problems


if __name__ == "__main__":
    # usage: score.py <out.json> <actual.json> [<case.json>]
    out_path, actual_path = sys.argv[1], sys.argv[2]
    actual = json.load(open(actual_path))
    actual = {int(k): v for k, v in actual.items()}
    print(score_case(out_path, actual))
    if len(sys.argv) > 3:
        probs = validate(out_path, sys.argv[3])
        print("validity problems:", probs if probs else "none")
