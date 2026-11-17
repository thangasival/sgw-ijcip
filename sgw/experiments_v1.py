"""V1 synthetic network: regime table (cf. V1 Table 6), P1 decomposition and severity sweep (all-hazards appendix, E8)."""
import numpy as np, pandas as pd
from . import network as NW, v1case as V1, theory as TH

def run(name, **kw):
    net = V1.network(); sc = V1.scenario(name, **kw); out = {}
    for on in (False, True):
        lp, info = NW.build(net, sc, on); info["cls"] = V1.CLS; sol = lp.solve()
        o = NW.outcomes(sol, lp, info, sc, on); o["gen_obj"] = sol.obj + (sc.activation if on else 0.0)
        out[on] = (o, sol, lp, info, sc)
    sel = out[True][0]["gen_obj"] < out[False][0]["gen_obj"] - 1e-9
    A = out[False][0]; B = out[True][0] if sel else A
    thr = 0.1 * A["peak_backlog"]; we = max(sc.window[1], 1)
    clA = NW.clearance(A["backlog_path"], we, thr) if A["peak_backlog"] > 1e-9 else 0
    clB = NW.clearance(B["backlog_path"], we, thr) if A["peak_backlog"] > 1e-9 else 0
    row = dict(scenario=name, cost_per_t_A=A["cost_per_t"], cost_per_t_B=B["cost_per_t"], cum_backlog_A=A["cum_backlog"],
               cum_backlog_B=B["cum_backlog"], peak_A=A["peak_backlog"], peak_B=B["peak_backlog"], clearance_A=clA,
               clearance_B=clB, waiver_selected=int(sel), waiver_kt=B["waiver_kt"] if sel else 0.0,
               delta_direct_cost_musd=B["direct_musd"] - A["direct_musd"])
    return row, out

V1_TABLE6 = {"Normal": (0, 0.0, 0.0), "Moderate": (0, 0.0, 0.0), "Severe": (1, 153.3, 0.0), "Congestion": (0, 150.2, 150.2),
             "Energy": (1, 457.7, 0.0), "Defense": (0, 27.4, 27.4)}     # (y, backlog A, backlog B) as published in V1

def regression_table():
    rows = []
    for s in V1.SCENARIOS:
        r, _ = run(s)
        y, bA, bB = V1_TABLE6[s]
        r.update(v1_selected=y, v1_backlog_A=bA, v1_backlog_B=bB,
                 regime_match=int((r["waiver_selected"] == y) and ((r["cum_backlog_B"] < 1e-6) == (bB < 1e-6)) and ((r["cum_backlog_A"] > 1e-6) == (bA > 1e-6))))
        rows.append(r)
    return pd.DataFrame(rows)

def severity_sweep(losses=np.round(np.arange(0.40, 0.801, 0.02), 2)):
    rows = []
    for loss in losses:
        r, out = run("Severe", loss=float(loss))
        o, sol, lp, info, sc = out[False]
        d = pd.DataFrame(TH.p1_decomposition(sol, lp, info, sc))
        neg = d[d.rc_waiver < -1e-9] if len(d) else d
        rows.append(dict(loss=float(loss), waiver_selected=r["waiver_selected"], cum_backlog_A=r["cum_backlog_A"],
                         cum_backlog_B=r["cum_backlog_B"], delta_direct_cost_musd=r["delta_direct_cost_musd"],
                         priced_out_columns=len(neg), max_identity_gap=float(d.gap.abs().max()) if len(d) else 0.0,
                         max_scarcity_rent=float(d.scarcity_rent.max()) if len(d) else 0.0))
    return pd.DataFrame(rows)
