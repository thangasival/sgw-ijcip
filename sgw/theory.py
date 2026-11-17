"""Theory checks: P1 reduced-cost decomposition and finite-difference verification of LP duals."""
from __future__ import annotations
import numpy as np


def p1_decomposition(sol, lp, info, sc, tol=1e-7):
    """For every waiver column (arc a, day t) in a waiver-off solution, compare its reduced cost with
        cost substitution  r*delta*(m_W - m_v*)  minus  domestic-lift scarcity rent  sum_h pi^{v*}_h,
    where v* is a domestic tier whose column (a, v*, t) is basic (positive flow, reduced cost ~ 0).
    Returns a list of dicts; `gap` is the identity residual (should be ~0)."""
    rc = sol.reduced_costs()
    f = lp.blocks["f"]; x = sol.x
    V = info["V"]; w = V.index("waiver")
    occ = info["rows"]["occ"]; arcs = info["arcs"]; tau = info["tau"]; T = info["T"]
    out = []
    from .v1case import CLS  # class map is the network's; passed through info if available
    cls_of = info.get("cls", CLS)
    for a, (s, j, k) in enumerate(arcs):
        c = cls_of[k]
        for t in range(T):
            for vi, v in enumerate(V):
                if vi == w:
                    continue
                col = f[a, vi, t]
                if x[col] > tol and abs(rc[col]) < 1e-7:
                    window = range(t, min(T, t + 2 * tau[a]))
                    rent_v = sum(sol.rent_ub(occ[(c, v, h)]) for h in window)
                    rent_w = sum(sol.rent_ub(occ[(c, "waiver", h)]) for h in window)
                    subst = info["base_c"][a] * (sc.mult["waiver"] - sc.mult[v])
                    rcw = rc[f[a, w, t]]
                    out.append(dict(arc=f"{s}-{j}-{k}", day=t + 1, vstar=v, rc_waiver=rcw, cost_subst=subst,
                                    scarcity_rent=rent_v, waiver_rent=rent_w, gap=rcw - (subst - rent_v + rent_w)))
                    break
    return out


def fd_check(lp, sol, row, h=1.0):
    """Finite-difference check of the dual of <= row `row`: returns (rent, forward, backward)."""
    b = lp.b_ub.copy()
    b[row] += h; up = lp.solve(b_ub=b).obj
    b[row] -= 2 * h; dn = lp.solve(b_ub=b).obj
    return sol.rent_ub(row), (sol.obj - up) / h, (dn - sol.obj) / h


def p2_rule(env, pol, prm, sol, info, groups=("AT", "AN"), modes=None):
    """P2 (pricing form of Eq. 9): at a solved instance, score each candidate screening mode m for group g by
        sum_ch lambda_ch * eps_{g,m,ch} * Phi_g  +  E_t[ y_Z(g, t + d_m) ]  +  E_t[ theta_{z,t} ] * h_m
    (exposure price + value of the cleared vessel arriving d_m later + inspector-hour price), using the duals of
    the solved LP; weights are the screenings x_{g,c,z,t}.  Returns {g: (predicted mode, scores)}."""
    import numpy as np
    from . import data as SD
    from .model import CH
    modes = modes or {"AT": ("doc", "docp", "ex"), "AN": ("reg", "doc", "docp", "ex")}
    G, C, S, dt = info["G"], info["C"], info["S"], info["dt"]
    rows = info["rows"]
    lam = {ch: (float(-sol.y_ub[r]) if ch in rows.get("exp", {}) else 0.0) for ch, r in
           {ch: rows.get("exp", {}).get(ch) for ch in CH}.items()}
    yZ = np.asarray(sol.y_eq)[rows["Z"]].reshape(len(G), len(C), S)
    thr = rows["thr"]
    th = (-np.asarray(sol.y_ub)[thr]).reshape(len(env.Z), S) if len(thr) else np.zeros((len(env.Z), S))
    x = sol.val("x") + sol.val("xr"); f = sol.val("f")
    kap = info["kap"]
    out = {}
    for g in groups:
        gi = G.index(g)
        w = x[gi]                                     # (c, z, s)
        if w.sum() <= 1e-9:
            continue
        phi = float(sum((kap[ji, ki] * f[ji, ki, 1 + gi, :] / env.parcel[ki]).sum()
                        for ji in range(len(info["J"])) for ki in range(len(info["K"])))) / float(w.sum())
        sc = {}
        for m in modes[g]:
            eps = np.array(env.group_base[g]) * np.array(prm.mode_res[m]) * np.array(prm.res_scale)
            e_term = sum(lam[ch] * eps[i] * phi for i, ch in enumerate(CH))
            L = prm.mode_d[m] / dt
            t = np.arange(S) + L
            d_term = 0.0
            for ci in range(len(C)):
                yi = np.interp(t, np.arange(S), yZ[gi, ci], right=0.0)
                d_term += float((w[ci].sum(axis=0) * yi).sum())
            d_term /= float(w.sum())
            h_term = float((w.sum(axis=0) * th).sum()) / float(w.sum()) * prm.mode_h[m]
            sc[m] = e_term + d_term + h_term
        out[g] = (min(sc, key=sc.get), sc, phi)
    return out
