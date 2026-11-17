"""Recompute the gate-G2 statistics and the informativeness results from the raw run records in outputs/runs
and check them against outputs/tables.  Exits non-zero on any mismatch."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT / "outputs" / "runs"; TAB = ROOT / "outputs" / "tables"

def load(stem):
    rows = []
    for f in sorted(RUN.glob(f"{stem}_*.jsonl")):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows).drop_duplicates("id")

def main():
    c0 = load("stageC0").set_index("draw").cum_backlog
    c = pd.concat([load("stageC1"), load("stageC2")], ignore_index=True)
    assert (c.status == "ok").all() and len(c0) == 30
    c["dB"] = c.cum_backlog - c.draw.map(c0)
    best = c.groupby(["draw", "tag"]).dB.min().unstack()
    g = (best.B3 - best.DTV) / best.B3
    ref = pd.read_csv(TAB / "e2_g2_paired_gains.csv", index_col=0).loc["DTV vs B3 (primary)"]
    ok = abs(g.median() - ref["median"]) < 1e-9 and abs(g.quantile(0.05) - ref["p5"]) < 1e-9
    passed = g.median() >= 0.10 and g.quantile(0.05) > 0
    e5 = pd.read_csv(TAB / "e5b_informativeness_draws.csv", index_col=[0, 1])
    g0 = ((best.B4 - best["DTV-i0.0"]) / best.B4).median()
    ok &= abs(g0 - e5.loc[("informativeness 0 (noise signal)", "B4"), "median"]) < 1e-9
    print(f"G2: median gain over B3 {g.median():.3f}, 5th pct {g.quantile(0.05):.3f}, positive {int((g > 0).sum())}/30 -> "
          f"{'PASS' if passed else 'FAIL'}; noise-signal gain over B4 {g0:.3f}; tables consistent: {ok}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
