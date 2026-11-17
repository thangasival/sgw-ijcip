"""Dual sign convention and finite-difference verification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sgw.lp import LP
from sgw import theory as TH
from sgw.experiments_v1 import run

def test_toy_hand_dual():
    # min 1*x1 + 3*x2  s.t. x1 + x2 = 10, x1 <= 4  ->  rent of the cap = 3 - 1 = 2, demand dual = 3
    lp = LP(); x = lp.var("x", (2,), cost=[1.0, 3.0]); r = lp.le([x[0]], [1.0], 4.0); lp.eq(x, [1.0, 1.0], 10.0)
    s = lp.solve()
    assert abs(s.obj - 22.0) < 1e-9 and abs(s.rent_ub(r) - 2.0) < 1e-9 and abs(s.y_eq[0] - 3.0) < 1e-9
    rc = s.reduced_costs(); assert abs(rc[0]) < 1e-9 and abs(rc[1]) < 1e-9   # both columns basic once the cap row is priced

def test_fd_and_p1_identity():
    _, out = run("Severe")
    o, sol, lp, info, sc = out[False]
    occ = info["rows"]["occ"]; checked = 0
    for key, row in occ.items():
        rent = sol.rent_ub(row)
        if rent <= 1e-6:
            continue
        r, fw, bw = TH.fd_check(lp, sol, row, h=1e-3)
        if abs(fw - bw) < 1e-6:                      # non-degenerate: one-sided differences agree
            assert abs(r - fw) < 1e-6, (key, r, fw, bw); checked += 1
        else:                                         # degenerate: dual lies between one-sided differences
            assert min(fw, bw) - 1e-6 <= r <= max(fw, bw) + 1e-6, (key, r, fw, bw)
    d = TH.p1_decomposition(sol, lp, info, sc)
    assert len(d) > 0 and max(abs(x["gap"]) for x in d) < 1e-9
    return checked, len(d)

if __name__ == "__main__":
    test_toy_hand_dual(); n, m = test_fd_and_p1_identity()
    print(f"toy dual ok; {n} non-degenerate occupancy duals match finite differences; P1 identity holds on {m} columns")
