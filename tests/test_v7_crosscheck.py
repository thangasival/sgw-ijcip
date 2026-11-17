"""v7 cross-check: voyage-list review delays reproduce the reference delay table (delay_v4.csv)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pandas as pd
from sgw.policies import voyage_list_vessel_days

NAMES = {"R2": "R2 2026 blanket, initial", "R3": "R3 2026 extension with VAR", "R4": "R4 Proposed HCW-RVP",
         "R4-D": "R4-D HCW-RVP, documentary screens", "R4-T": "R4-T HCW-RVP, risk-triggered exam"}

def test_v7_crosscheck():
    P = pd.read_csv(ROOT / "data" / "v7" / "waiver_voyage_panel_v4.csv", parse_dates=["load_date_parsed"])
    D = pd.read_csv(ROOT / "data" / "v7" / "delay_v4.csv"); D = D[D.delay_group == "ALL"].set_index(["design", "register"]).vessel_days
    n = 0
    for k, full in NAMES.items():
        for reg in ((True, False) if k.startswith("R4") else (True,)):
            got = voyage_list_vessel_days(P, k, reg); exp = D[(full, reg)]
            assert abs(got - exp) < 1e-9, (k, reg, got, exp); n += 1
    assert abs(voyage_list_vessel_days(P, "R3") - 250.0) < 1e-9 and abs(voyage_list_vessel_days(P, "R4-T") - 222.75) < 1e-9
    return n

if __name__ == "__main__":
    print(f"voyage-list delays match v7 delay_v4.csv for {test_v7_crosscheck()} design x register cases")
