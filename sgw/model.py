"""SG-WCM: security-gated waiver capacity model with dynamic tiered vetting (DTV).

Every instance is an LP.  A policy fixes the screening mode of each risk group (mu: G -> M) and a re-tiering rule
rho = (u, delta_G, delta_A, informativeness); policies and rules are enumerated outside the solver.
Time is discretised in steps of `dt` days (default 2); delays that are not multiples of dt are represented by
splitting the flow between the two neighbouring steps (linear interpolation), so a 1-day documentary check is half
a step.

Blocks (blueprint numbering):
  supply network   V1 Eqs. (1)-(3), (6): arrivals (transit-lagged), due backlog b, pending p, unloading
  lift             (5)  sum_{arcs of class c} O[j,k,g,s] <= kbar_c Z[g,c,s]; O = lift in circulation (round trip)
  queue            (2)  U[g,c,z,s] = (1-lam dt) U[s-1] + A - x
  cleared stock    (3)  Z[g,c,s] = (1-(lam+r)dt) Z[s-1] + sum_z x[s-d_mu] + sum_z xr[s-d_rung] +/- transitions (10)-(11)
  re-screen queue  (12) Ur[g,c,z,s] = (1-lam dt) Ur[s-1] + share_z (r dt Z + delta n) - xr
  throughput       (4') sum_{g,c} h_mu x + h_rung xr <= R_z dt
  exposure         (6)  E^ch = sum kappa eps^ch_g f / parcel <= Ebar^ch,  ch in {crew, own, cond}
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from .lp import LP
from . import data as SD

KT, MUSD = 1000.0, 1e6
MODES = ["none", "reg", "doc", "docp", "ex", "den"]
# grade-D defaults: delay (days), inspector-hours, residual unscreened share by channel (crew, own, cond)
MODE_D = {"none": 0.0, "reg": 0.25, "doc": 1.0, "docp": 1.5, "ex": 3.0, "den": 1.0}
MODE_H = {"none": 0.0, "reg": 0.5, "doc": 4.0, "docp": 8.0, "ex": 16.0, "den": 4.0}
MODE_RES = {"none": (1.0, 1.0, 1.0), "reg": (0.7, 0.6, 0.5), "doc": (0.3, 0.3, 0.6), "docp": (0.15, 0.1, 0.5),
            "ex": (0.1, 0.3, 0.1), "den": (0.0, 0.0, 0.0)}
NEXT_RUNG = {"none": "doc", "reg": "doc", "doc": "docp", "docp": "ex", "ex": "ex", "den": "den"}
# group baseline risk by channel (crew, own, cond); cond anchored on PSIX deficiencies per exam (0.13 QS21 vs 0.44 targeted)
GROUP_BASE = {"G": (0.8, 0.2, 0.3), "AT": (1.0, 0.3, 1.0), "AN": (1.0, 0.3, 1.0), "UG": (0.8, 0.2, 0.3), "R": (1.0, 1.0, 1.0)}
CH = ["crew", "own", "cond"]
ZONES4 = ["Texas Coast", "Sabine-Lake Charles", "Lower Mississippi", "Other"]
Z6TO4 = {"Houston-Galveston": "Texas Coast", "Corpus Christi": "Texas Coast", "Sabine-Lake Charles": "Sabine-Lake Charles",
         "Lower Mississippi": "Lower Mississippi", "Atlantic-Caribbean": "Other", "Pacific": "Other"}


@dataclass
class Params:
    s_dom: float = 0.05              # share of spot Jones Act tanker DWT available for the emergency increment
    m_dom: float = 1.9               # domestic spot (surge) cost multiplier
    m_w: float = 1.3                 # waiver multiplier on high-ESB routes
    m_w_low: float = 2.0             # waiver multiplier on low-ESB routes (> m_dom: right of first refusal)
    rate: float = 0.02               # USD per tonne-nm
    pen_d: float = 60.0              # USD per tonne-day x priority
    pen_T: float = 6000.0            # USD per tonne x priority, on the last step
    lead: float = 2.0                # days between a vessel's request and its first load date
    r_share: float = 0.05            # share of national PSC inspector-hours redeployable to waiver vetting
    psc_hours_day: float = 24.7 * 16 # national PSC capacity: 8,999 exams/yr ~ 24.7/day x 16 inspector-hours
    valid_amber: float = 30.0        # days an Amber clearance stays valid before a review (30-day reviews)
    d_var: float = 1.0               # VAR market-survey window (days)
    d_dow: float = 0.0               # DoW decision time (days, unobserved; swept)
    unload_scale: float = 1.0
    res_scale: tuple = (1.0, 1.0, 1.0)
    mode_d: dict = field(default_factory=lambda: dict(MODE_D))
    mode_h: dict = field(default_factory=lambda: dict(MODE_H))
    mode_res: dict = field(default_factory=lambda: dict(MODE_RES))
    group_base: dict = field(default_factory=lambda: dict(GROUP_BASE))


@dataclass
class Policy:
    mu: dict                         # group -> mode
    var_routes: tuple = ()           # destinations where the per-voyage VAR applies
    u: float = 0.0                   # P(upgrade) per Amber voyage completed on time and clean
    delta_g: float = 0.0             # P(signal trips) per Green / upgraded voyage
    delta_a: float = 0.0             # P(signal trips) per Amber voyage
    inform: float = 1.0              # informativeness: residual risk of upgraded vessels = G + (1-inform)(AN - G)
    name: str = ""


def lagw(d_days, dt):
    """Split a delay of d_days into (lag_steps, weight) pairs for steps of dt days."""
    x = d_days / dt; L = int(math.floor(x + 1e-9)); a = x - L
    out = [(L, 1.0 - a)] if a < 1 - 1e-9 else []
    if a > 1e-9:
        out.append((L + 1, a))
    return out or [(L, 1.0)]


class Env:
    """Env-R inputs built once and reused across instances."""
    def _group_base(self, V, VG):
        Tr = vessel_truth(V, VG); self.truth = Tr
        out = {}
        for g in SD.GROUPS:
            m = VG[VG.group == (g if g != "UG" else "AN")]
            if len(m) == 0:
                out[g] = GROUP_BASE[g]; continue
            w = m.set_index("vessel").lifetime
            t = Tr.loc[w.index]
            out[g] = tuple(float(np.average(t[c], weights=w)) for c in ("crew", "own", "cond"))
        return out

    def __init__(self, grouping="flag", west_mbbl=28.3, dt=2, extend_arrivals=True, red_share=None, seed=1, VG=None):
        V = SD.voyages(); VG = SD.vessel_groups(V, grouping) if VG is None else VG
        if red_share is not None:                 # E6: re-label a share of non-Green vessels as unresolved ownership
            rng = np.random.default_rng(seed)
            VG = VG.copy(); VG.loc[VG.group == "R", "group"] = "AN"
            n_red = int(round(red_share * len(VG)))
            cand = VG.index[VG.group != "G"].tolist() + VG.index[VG.group == "G"].tolist()
            pick = list(rng.permutation(VG.index[VG.group != "G"]))[:n_red]
            if len(pick) < n_red:
                pick += list(rng.permutation(VG.index[VG.group == "G"]))[:n_red - len(pick)]
            VG.loc[pick, "group"] = "R"
        self.V, self.VG, self.dt = V, VG, dt
        self.days = SD.T; self.S = int(math.ceil(SD.T / dt))
        self.J, self.K, self.Z = SD.DESTS, SD.CLASSES, ZONES4
        qd, self.W = SD.demand(V, west_total_mbbl=west_mbbl)
        self.q = np.zeros((len(self.J), len(self.K), self.S))
        for d in range(SD.T):
            self.q[:, :, d // dt] += qd[:, :, d]
        self.qdaily = qd
        self.H = SD.unloading().reindex(self.J).values
        self.fleet = SD.domestic_fleet()
        zw6 = SD.zone_weights()
        self.zw = np.array([sum(zw6[z6] for z6, z4 in Z6TO4.items() if z4 == z) for z in self.Z])
        self.parcel = V.groupby("cls").kt.mean().reindex(self.K).values
        self.kbar = {"tank": V[V.cls != "gas"].kt.mean(), "gas": V[V.cls == "gas"].kt.mean()}
        G, C = SD.GROUPS, ["tank", "gas"]
        self.Ad = np.zeros((len(G), 2, len(self.Z), SD.T))        # daily arrivals (first-voyage load day)
        for r in VG.itertuples():
            self.Ad[G.index(r.group), C.index(r.vclass), self.Z.index(Z6TO4[r.zone]), r.first_day - 1] += 1
        init_days = (SD.VAR_START - SD.DAY0).days
        load_days = (SD.LOAD_END - SD.DAY0).days + 1
        if extend_arrivals:                       # 'had the initial design continued': new vessels at the initial rate
            rate = self.Ad[..., :init_days].sum(axis=-1) / init_days
            self.Ad[..., init_days:load_days] += rate[..., None]
        self.group_base = self._group_base(V, VG)
        self.lam = {g: 1.0 / max(1.0, VG.loc[VG.group == g, "lifetime"].mean()) if (VG.group == g).any() else 1 / 30 for g in G}
        self.lam["UG"] = self.lam["AN"]
        self.zshare = np.zeros((len(G), len(self.Z)))
        for gi, g in enumerate(G):
            s = self.Ad[gi].sum(axis=(0, 2))
            self.zshare[gi] = s / s.sum() if s.sum() > 0 else self.zw


def vessel_truth(V, VG):
    """Vessel-level 'true' baseline risk used for exposure accounting, independent of the grouping a policy uses:
    crew and ownership channels from the flag-tier/ownership classification (grade D); condition channel from the
    vessel's shrunk PSIX deficiency rate, rescaled so its mean equals the flag-based mean (grade C/D)."""
    from .baselines import predictor
    pred, prior, _ = predictor()
    flag = SD.vessel_groups(V, "flag").set_index("vessel").group
    qs = V.drop_duplicates("vessel").set_index("vessel").flag_tier.eq("QUALSHIP 21")
    rows = []
    for r in VG.itertuples():
        fg = flag.get(r.vessel, r.group)
        crew, own, cond_flag = GROUP_BASE["G" if fg == "G" else ("R" if fg == "R" else "AN")]
        rows.append(dict(vessel=r.vessel, crew=crew, own=own, cond_flag=cond_flag,
                         pred=pred.get(r.vessel, prior(float(qs.get(r.vessel, False)), float(r.vclass == "gas")))))
    Tr = pd.DataFrame(rows).set_index("vessel")
    Tr["cond"] = Tr.pred / Tr.pred.mean() * Tr.cond_flag.mean()
    Tr.loc[flag.reindex(Tr.index).eq("R").values, "cond"] = Tr.cond.clip(lower=1.0)
    return Tr


class _Rows:
    """Accumulates triplets for a block of rows."""
    def __init__(self, n):
        self.n = n; self.r, self.c, self.v = [], [], []
    def add(self, rows, cols, vals):
        rows = np.asarray(rows).ravel(); cols = np.asarray(cols).ravel()
        vals = np.broadcast_to(np.asarray(vals, float), rows.shape).ravel() if np.ndim(vals) else np.full(rows.size, float(vals))
        self.r.append(rows); self.c.append(cols); self.v.append(vals)
    def commit(self, lp, kind, rhs):
        return lp.bulk(kind, self.n, np.concatenate(self.r), np.concatenate(self.c), np.concatenate(self.v), rhs)


def build(env: Env, pol: Policy, prm: Params, ebar=None, fixed_x=None):
    J, K, Zs, S, dt = env.J, env.K, env.Z, env.S, env.dt
    G, C = SD.GROUPS, ["tank", "gas"]
    nJ, nK, nV, nG, nC, nZ = len(J), len(K), 1 + len(G), len(G), 2, len(Zs)
    lp = LP()
    tau = np.array([SD.TRANSIT[j] for j in J], float)
    var_lag = np.array([(prm.d_var + prm.d_dow) if j in pol.var_routes else 0.0 for j in J])
    tauV = np.array([[tau[ji] if vi == 0 else tau[ji] + var_lag[ji] for vi in range(nV)] for ji in range(nJ)])
    win = np.array([[int(math.ceil((2 * tau[ji] + 2 + (0 if vi == 0 else var_lag[ji])) / dt)) for vi in range(nV)] for ji in range(nJ)])
    due = np.array([int(math.ceil((tau[ji] + 1) / dt)) for ji in range(nJ)])
    load_steps = int(math.ceil(((SD.LOAD_END - SD.DAY0).days + 1) / dt))
    # ---------------- variables
    dist = np.array([SD.DIST_NM[j] for j in J], float)
    mW = np.array([prm.m_w_low if j in SD.LOW_ESB else prm.m_w for j in J])
    cost = np.zeros((nJ, nK, nV, S))
    cost[:, :, 0, :] = (prm.rate * dist * prm.m_dom * KT / MUSD)[:, None, None]
    cost[:, :, 1:, :] = (prm.rate * dist * mW * KT / MUSD)[:, None, None, None]
    ub = np.full(cost.shape, np.inf)
    ub[:, K.index("gas"), 0, :] = 0.0
    for gi, g in enumerate(G):
        if pol.mu.get(g) == "den":
            ub[:, :, 1 + gi, :] = 0.0
    ub[:, :, :, load_steps:] = 0.0
    f = lp.var("f", cost.shape, cost=cost, ub=ub)
    pri = np.array([SD.PRIORITY[j] for j in J])
    bc = np.broadcast_to((prm.pen_d * dt * pri * KT / MUSD)[:, None, None], (nJ, nK, S)).copy()
    bc[:, :, -1] += (prm.pen_T * pri * KT / MUSD)[:, None]
    b = lp.var("b", (nJ, nK, S), cost=bc)
    p = lp.var("p", (nJ, nK, S))
    O = lp.var("O", (nJ, nK, nV, S))
    Zv = lp.var("Z", (nG, nC, S)); U = lp.var("U", (nG, nC, nZ, S)); Ur = lp.var("Ur", (nG, nC, nZ, S))
    x = lp.var("x", (nG, nC, nZ, S)) if fixed_x is None else lp.var("x", (nG, nC, nZ, S), lb=fixed_x, ub=fixed_x)
    xr = lp.var("xr", (nG, nC, nZ, S))
    s_ = np.arange(S)
    rows = {}
    # ---------------- balance (due backlog) and pending: rows (j,k,s)
    R1 = _Rows(nJ * nK * S); R2 = _Rows(nJ * nK * S)
    rhs1 = np.zeros(nJ * nK * S); rhs2 = np.zeros(nJ * nK * S)
    for ji in range(nJ):
        for ki in range(nK):
            base = (ji * nK + ki) * S
            rr = base + s_
            R1.add(rr, b[ji, ki, :], 1.0); R1.add(rr[1:], b[ji, ki, :-1], -1.0)
            R2.add(rr, p[ji, ki, :], 1.0); R2.add(rr[1:], p[ji, ki, :-1], -1.0)
            for vi in range(nV):
                for L, w in lagw(tauV[ji, vi], dt):
                    if L < S:
                        R1.add(rr[L:], f[ji, ki, vi, :S - L], w)
                R2.add(rr, f[ji, ki, vi, :], 1.0)
            dq = np.zeros(S); dq[due[ji]:] = env.q[ji, ki, :S - due[ji]]
            rhs1[rr] = dq; rhs2[rr] = env.q[ji, ki, :]
    rows["bal"] = R1.commit(lp, "eq", rhs1); rows["pend"] = R2.commit(lp, "eq", rhs2)
    # ---------------- lift in circulation O
    R3 = _Rows(nJ * nK * nV * S)
    for ji in range(nJ):
        for ki in range(nK):
            for vi in range(nV):
                rr = ((ji * nK + ki) * nV + vi) * S + s_
                R3.add(rr, O[ji, ki, vi, :], 1.0); R3.add(rr[1:], O[ji, ki, vi, :-1], -1.0); R3.add(rr, f[ji, ki, vi, :], -1.0)
                W = win[ji, vi]
                if W < S:
                    R3.add(rr[W:], f[ji, ki, vi, :S - W], 1.0)
    R3.commit(lp, "eq", 0.0)
    # ---------------- unloading (j,s)
    R4 = _Rows(nJ * S)
    for ji in range(nJ):
        rr = ji * S + s_
        for ki in range(nK):
            for vi in range(nV):
                for L, w in lagw(tauV[ji, vi], dt):
                    if L < S:
                        R4.add(rr[L:], f[ji, ki, vi, :S - L], w)
    rows["unl"] = R4.commit(lp, "ub", np.repeat(env.H * prm.unload_scale * dt, S))
    # ---------------- domestic lift (s)
    tank_k = [ki for ki, k in enumerate(K) if SD.VCLASS[k] == "tank"]
    R5 = _Rows(S)
    for ji in range(nJ):
        for ki in tank_k:
            R5.add(s_, O[ji, ki, 0, :], 1.0)
    rows["dom"] = R5.commit(lp, "ub", prm.s_dom * env.fleet["dwt_spot_kt"] * 0.95)
    # ---------------- lift gate (g,c,s)
    R6 = _Rows(nG * nC * S)
    for gi in range(nG):
        for ci, c in enumerate(C):
            rr = (gi * nC + ci) * S + s_
            for ji in range(nJ):
                for ki, k in enumerate(K):
                    if SD.VCLASS[k] == c:
                        R6.add(rr, O[ji, ki, 1 + gi, :], 1.0)
            R6.add(rr, Zv[gi, ci, :], -env.kbar[c])
    rows["lift"] = R6.commit(lp, "ub", 0.0)
    # ---------------- vetting block
    lam = np.array([env.lam[g] for g in G]) * dt
    mode = [pol.mu.get(g, "none") for g in G]
    rung = [NEXT_RUNG[m] for m in mode]
    valid = np.array([dt / prm.valid_amber if g in ("AT", "AN") and mode[gi] not in ("none", "den") else 0.0 for gi, g in enumerate(G)])
    lead_steps = prm.lead / dt
    # arrivals A in steps, shifted earlier by the request lead
    A = np.zeros((nG, nC, nZ, S))
    for d in range(SD.T):
        sd = d - prm.lead
        st = int(max(0, math.floor(sd / dt)))
        A[..., min(S - 1, st)] += env.Ad[..., d]
    def n_terms(R, rr, gi, ci, coef, shift=1):
        """add coef * n[g,c,s-shift] (voyages started) to rows rr"""
        for ji in range(nJ):
            for ki, k in enumerate(K):
                if (SD.VCLASS[k] == "tank") == (C[ci] == "tank"):
                    R.add(rr[shift:], f[ji, ki, 1 + gi, :S - shift], coef / env.parcel[ki])
    gi_of = {g: i for i, g in enumerate(G)}
    R7 = _Rows(nG * nC * S)
    for gi, g in enumerate(G):
        for ci in range(nC):
            rr = (gi * nC + ci) * S + s_
            R7.add(rr, Zv[gi, ci, :], 1.0); R7.add(rr[1:], Zv[gi, ci, :-1], -(1 - lam[gi] - valid[gi]))
            if mode[gi] != "den":
                for zi in range(nZ):
                    for L, w in lagw(prm.mode_d[mode[gi]], dt):
                        if L < S: R7.add(rr[L:], x[gi, ci, zi, :S - L], -w)
                    for L, w in lagw(prm.mode_d[rung[gi]], dt):
                        if L < S: R7.add(rr[L:], xr[gi, ci, zi, :S - L], -w)
            if g in ("AT", "AN") and (pol.u + pol.delta_a) > 0:
                n_terms(R7, rr, gi, ci, pol.u + pol.delta_a)
            if g in ("G", "UG") and pol.delta_g > 0:
                n_terms(R7, rr, gi, ci, pol.delta_g)
            if g == "UG" and pol.u > 0:
                for ga in ("AT", "AN"):
                    n_terms(R7, rr, gi_of[ga], ci, -pol.u)
    rows["Z"] = R7.commit(lp, "eq", 0.0)
    R8 = _Rows(nG * nC * nZ * S); R9 = _Rows(nG * nC * nZ * S)
    rhs8 = np.zeros(nG * nC * nZ * S)
    for gi, g in enumerate(G):
        for ci in range(nC):
            for zi in range(nZ):
                rr = ((gi * nC + ci) * nZ + zi) * S + s_
                R8.add(rr, U[gi, ci, zi, :], 1.0); R8.add(rr, x[gi, ci, zi, :], 1.0); R8.add(rr[1:], U[gi, ci, zi, :-1], -(1 - lam[gi]))
                rhs8[rr] = A[gi, ci, zi, :]
                R9.add(rr, Ur[gi, ci, zi, :], 1.0); R9.add(rr, xr[gi, ci, zi, :], 1.0); R9.add(rr[1:], Ur[gi, ci, zi, :-1], -(1 - lam[gi]))
                sh = env.zshare[gi, zi] if g != "UG" else env.zshare[gi_of["AN"], zi]
                if valid[gi] > 0:
                    R9.add(rr[1:], Zv[gi, ci, :-1], -valid[gi] * sh)
                dlt = pol.delta_a if g in ("AT", "AN") else (pol.delta_g if g in ("G", "UG") else 0.0)
                if dlt > 0:
                    n_terms(R9, rr, gi, ci, -dlt * sh)
    rows["U"] = R8.commit(lp, "eq", rhs8); rows["Ur"] = R9.commit(lp, "eq", 0.0)
    # ---------------- throughput (z,s)
    Rz = prm.r_share * prm.psc_hours_day * env.zw
    hmu = np.array([prm.mode_h[m] for m in mode]); hr = np.array([prm.mode_h[m] for m in rung])
    R10 = _Rows(nZ * S); has = False
    for zi in range(nZ):
        rr = zi * S + s_
        for gi in range(nG):
            for ci in range(nC):
                if hmu[gi] > 0: R10.add(rr, x[gi, ci, zi, :], hmu[gi]); has = True
                if hr[gi] > 0: R10.add(rr, xr[gi, ci, zi, :], hr[gi]); has = True
    rows["thr"] = R10.commit(lp, "ub", np.repeat(Rz * dt, S)) if has else np.array([], int)
    # ---------------- exposure
    eps = exposure_coefs(pol, prm, mode, getattr(env, "group_base", None))
    kap = np.array([[min(1.0, SD.KAPPA[j] + (0.2 if k == "gas" else 0.0)) for k in K] for j in J])
    exp_terms = {}
    for chi, ch in enumerate(CH):
        idx, cf = [], []
        for ji in range(nJ):
            for ki in range(nK):
                for gi in range(nG):
                    w = kap[ji, ki] * eps[gi, chi] / env.parcel[ki]
                    if w > 0:
                        idx.append(f[ji, ki, 1 + gi, :]); cf.append(np.full(S, w))
        idx = np.concatenate(idx) if idx else np.array([], int); cf = np.concatenate(cf) if cf else np.array([])
        exp_terms[ch] = (idx, cf)
        cap = None if ebar is None else ebar.get(ch)
        if cap is not None and np.isfinite(cap):
            rows.setdefault("exp", {})[ch] = lp.le(idx, cf, cap)
    lp.build()
    info = dict(J=J, K=K, G=G, C=C, S=S, dt=dt, tau=tau, rows=rows, mode=mode, rung=rung, Rz=Rz, hmu=hmu, hr=hr,
                kap=kap, eps=eps, exp_terms=exp_terms, load_steps=load_steps)
    return lp, info


def exposure_coefs(pol: Policy, prm: Params, mode, group_base=None):
    G = SD.GROUPS; gb = group_base or prm.group_base
    eps = np.zeros((len(G), len(CH)))
    for gi, g in enumerate(G):
        eps[gi] = np.array(gb[g]) * np.array(prm.mode_res[mode[gi]]) * np.array(prm.res_scale)
    gU = G.index("UG")
    eG = np.array(gb["G"]) * np.array(prm.mode_res[mode[gU]]) * np.array(prm.res_scale)
    eA = np.array(gb["AN"]) * np.array(prm.mode_res[mode[gU]]) * np.array(prm.res_scale)
    eps[gU] = eG + (1 - pol.inform) * (eA - eG)
    return eps


def metrics(sol, lp, info, env: Env, prm: Params):
    f = sol.val("f"); b = sol.val("b"); x = sol.val("x"); xr = sol.val("xr")
    J, dt = info["J"], info["dt"]
    pri = np.array([SD.PRIORITY[j] for j in J])
    B = b.sum(axis=1) * dt                                    # kt-days
    hi = [J.index(j) for j in J if j not in SD.LOW_ESB]
    jul = int(((np.datetime64("2026-08-01") - np.datetime64("2026-03-17")).astype(int)) // dt)
    out = dict(obj=sol.obj, cum_backlog=float(B.sum()), wcum_backlog=float((B * pri[:, None]).sum()),
               high_esb_backlog=float(B[hi].sum()), peak_backlog=float(b.sum(axis=(0, 1)).max()),
               waiver_kt=float(f[:, :, 1:, :].sum()), domestic_kt=float(f[:, :, 0, :].sum()),
               west_waiver_kt_marjul=float(f[J.index("West"), :, 1:, :jul].sum()))
    for ch, (idx, cf) in info["exp_terms"].items():
        out[f"E_{ch}"] = float((sol.x[idx] * cf).sum()) if len(idx) else 0.0
    out["insp_hours"] = float((x * info["hmu"][:, None, None, None]).sum() + (xr * info["hr"][:, None, None, None]).sum())
    out["queue_vessel_days"] = float((sol.val("U").sum() + sol.val("Ur").sum()) * dt)
    thr = info["rows"]["thr"]
    out["theta_mean"] = float(np.mean(-sol.y_ub[thr])) if len(thr) else 0.0
    return out


def solve(env, pol, prm, ebar=None, fixed_x=None):
    lp, info = build(env, pol, prm, ebar=ebar, fixed_x=fixed_x)
    sol = lp.solve()
    return sol, lp, info, metrics(sol, lp, info, env, prm)
