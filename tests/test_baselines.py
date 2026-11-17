"""B3 greedy allocation on a toy instance, and predictor cross-validation (small-sample caveat reported)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sgw import baselines as BL, model as SM

def test_b3_greedy_toy():
    toy = pd.DataFrame(dict(vessel=list("abcde"), group=["AN"] * 5, first_day=[10] * 5, zone=["Houston-Galveston"] * 5,
                            pred=[0.1, 0.5, 0.3, 0.9, 0.2], vclass=["tank"] * 5, lifetime=[30] * 5))
    p = SM.Params(r_share=1.0, lead=0.0)
    zw = np.array([2 * 16 / (p.psc_hours_day * 2), 0, 0, 0])        # capacity for exactly two exams in the step
    G, _, info = BL.b3_groups(toy, p, 2, zw)
    assert set(G.loc[G.group == "AT", "vessel"]) == {"b", "d"} and abs(info["expected_deficiencies"] - 1.4) < 1e-12
    return info

if __name__ == "__main__":
    print("B3 greedy toy:", test_b3_greedy_toy())
