"""House curve baseline, reconstructed from RULES.txt:

per trophy a least-squares line of logit rate on ln(1 + hours since
2026-11-19T00:00:00Z) over the last 24 hours of reads, read at the target;
spread the random walk's and the fit's residual combined; fewer than 3 reads
fall back to persistence.  Quantiles rounded to 4 decimals half-up, as the
starter does.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.5
T0 = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s):
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))


def forecast_case(case):
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    d = case["horizon_minutes"] / 1440.0
    rw = SIGMA * math.sqrt(d)
    x_target = math.log(1.0 + (target - T0).total_seconds() / 3600.0)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        # distinct reads in the last 24h before `at`
        pts = []
        seen = set()
        for read in case["history"]:
            t = parse_time(read["slot"])
            if (at - t).total_seconds() > 24 * 3600:
                continue
            rates = read["rates"]
            if i >= len(rates) or rates[i] is None:
                continue
            key = (read["players_tracked"], rates[i])
            if key in seen:
                continue
            seen.add(key)
            pts.append((math.log(1.0 + (t - T0).total_seconds() / 3600.0), logit(rates[i])))
        if len(pts) < 3:
            last = None
            for read in reversed(case["history"]):
                if i < len(read["rates"]) and read["rates"][i] is not None:
                    last = read["rates"][i]
                    break
            lg = logit(last if last is not None else 0)
            spread = rw
        else:
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            sxx = sum((p[0] - mx) ** 2 for p in pts)
            b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx > 1e-12 else 0.0
            a = my - b * mx
            lg = a + b * x_target
            resid = [p[1] - (a + b * p[0]) for p in pts]
            rstd = math.sqrt(sum(r * r for r in resid) / len(resid))
            spread = math.sqrt(rw * rw + rstd * rstd)
        q = []
        prev = -1.0
        for z in Z:
            v = clamp(100.0 / (1.0 + math.exp(-(lg + z * spread)))) if lg + z * spread < 700 else 99.99
            v = max(v, prev)
            q.append(round4(v))
            prev = v
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast_case(case), separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
