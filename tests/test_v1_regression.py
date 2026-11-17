"""Regression against V1 Table 6: regime pattern (waiver selection, backlog eliminated or not)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgw.experiments_v1 import regression_table

def test_v1_regression():
    T = regression_table()
    assert T.regime_match.all(), T[["scenario", "waiver_selected", "v1_selected", "cum_backlog_A", "cum_backlog_B"]]
    return T

if __name__ == "__main__":
    T = test_v1_regression()
    out = Path(__file__).resolve().parents[1] / "outputs" / "tables" / "v1_regression.csv"; T.to_csv(out, index=False)
    print(f"V1 regime pattern reproduced for {len(T)} scenarios -> {out.name}")
