"""Multi-period, multi-commodity maritime network LP (V1 formulation, Eqs. 1-7).

The same builder is used for the all-hazards synthetic network (V1 case) and is extended in model.py
with the security-gated waiver tier.  Units: cargo in kt, time in days, money in USD million.

Constraint map (V1 numbering):
 (1) arrivals a_jkt = sum_{s,v} f_{sjkv,t-tau_sj}         (substituted, not a variable)
 (2) due backlog   b_jkt = b_jk,t-1 + q_jk,t-taubar - a_jkt, b >= 0
 (3) pending       p_jkt = p_jk,t-1 + q_jkt - sum_{s,v} f_sjkvt, p >= 0
 (4) source supply sum_{j,v} f_sjkvt <= S_sk
 (5) occupancy     sum_{arcs of class c} sum_{h=t-2tau+1}^{t} f_{.,v,h} <= Cap_cvt
 (6) unloading     sum_k a_jkt <= H_jt
 (7) objective     sailing cost + priority-weighted backlog + terminal backlog (+ activation, added outside)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .lp import LP

KT = 1000.0          # tonnes per kt
MUSD = 1e6


@dataclass
class Network:
    dests: list                      # destination ports
    commodities: list                # commodity names
    cls: dict                        # commodity -> class ("liquid"/"dry")
    sources: dict                    # commodity -> list of source ports
    dist: dict                       # (s, j) -> nautical miles
    supply: dict                     # (s, k) -> kt/day
    priority: dict                   # k -> omega
    tiers: list = field(default_factory=lambda: ["base", "surge", "waiver"])
    speed_kn: float = 14.0
    tau_override: dict = None        # (s, j) -> days

    def tau(self, s, j):
        if self.tau_override and (s, j) in self.tau_override:
            return int(self.tau_override[(s, j)])
        return int(math.ceil(self.dist[(s, j)] / (self.speed_kn * 24.0)))


@dataclass
class Scenario:
    T: int                           # horizon (days)
    q: np.ndarray                    # demand released, shape (J, K, T)
    cap: dict                        # (class, tier) -> array length T (kt occupied)
    H: np.ndarray                    # unloading kt/day, shape (J, T)
    mult: dict                       # tier -> cost multiplier
    rate: float = 0.02               # USD per tonne-nm
    pen_d: float = 60.0              # USD per tonne-day x priority
    pen_T: float = 6000.0            # USD per tonne x priority (terminal)
    activation: float = 2.0          # USD million
    lag: int = 2                     # waiver unavailable on days 1..lag
    allowance: int = 1               # due offset = min transit + allowance
    window: tuple = (0, 0)           # disruption window (first, last day), 1-indexed


def build(net: Network, sc: Scenario, waiver_on: bool, extra=None):
    """Return (lp, info).  `extra(lp, info)` may add rows/variables before build (used by model.py)."""
    J, K, T = net.dests, net.commodities, sc.T
    jx = {j: i for i, j in enumerate(J)}; kx = {k: i for i, k in enumerate(K)}
    arcs = [(s, j, k) for k in K for s in net.sources[k] for j in J if s != j]
    tau = np.array([net.tau(s, j) for s, j, _ in arcs])
    V = net.tiers
    lp = LP()
    # sailing cost per kt by arc and tier
    base_c = np.array([sc.rate * net.dist[(s, j)] * KT / MUSD for s, j, _ in arcs])
    cost = base_c[:, None, None] * np.array([sc.mult[v] for v in V])[None, :, None] * np.ones((1, 1, T))
    ub = np.full((len(arcs), len(V), T), np.inf)
    if "waiver" in V:
        w = V.index("waiver")
        ub[:, w, :] = 0.0 if not waiver_on else ub[:, w, :]
        if waiver_on:
            ub[:, w, :sc.lag] = 0.0
    f = lp.var("f", (len(arcs), len(V), T), cost=cost, ub=ub)
    om = np.array([net.priority[k] for k in K])
    bcost = np.broadcast_to(sc.pen_d * om[None, :, None] * KT / MUSD, (len(J), len(K), T)).copy()
    bcost[:, :, -1] += sc.pen_T * om[None, :] * KT / MUSD
    b = lp.var("b", (len(J), len(K), T), cost=bcost)
    p = lp.var("p", (len(J), len(K), T))
    # due offset
    taubar = {}
    for k in K:
        for j in J:
            ts = [net.tau(s, j) for s in net.sources[k] if s != j]
            taubar[(j, k)] = (min(ts) + sc.allowance) if ts else None
    into = {}
    for a, (s, j, k) in enumerate(arcs):
        into.setdefault((j, k), []).append(a)
    rows = {"bal": {}, "pend": {}, "sup": {}, "occ": {}, "unl": {}}
    for (j, k), al in into.items():
        ji, ki = jx[j], kx[k]
        for t in range(T):
            idx, cf = [b[ji, ki, t]], [1.0]
            if t > 0:
                idx.append(b[ji, ki, t - 1]); cf.append(-1.0)
            for a in al:
                h = t - tau[a]
                if h >= 0:
                    idx += list(f[a, :, h]); cf += [1.0] * len(V)
            tq = t - taubar[(j, k)]
            rhs = sc.q[ji, ki, tq] if tq >= 0 else 0.0
            rows["bal"][(j, k, t)] = lp.eq(idx, cf, rhs)
            idx, cf = [p[ji, ki, t]], [1.0]
            if t > 0:
                idx.append(p[ji, ki, t - 1]); cf.append(-1.0)
            for a in al:
                idx += list(f[a, :, t]); cf += [1.0] * len(V)
            rows["pend"][(j, k, t)] = lp.eq(idx, cf, sc.q[ji, ki, t])
    # (j,k) pairs without any feasible arc: nothing to deliver; keep b,p at zero
    for j in J:
        for k in K:
            if (j, k) not in into:
                for t in range(T):
                    lp.eq([b[jx[j], kx[k], t]], [1.0], 0.0); lp.eq([p[jx[j], kx[k], t]], [1.0], 0.0)
    # supply
    by_sk = {}
    for a, (s, j, k) in enumerate(arcs):
        by_sk.setdefault((s, k), []).append(a)
    for (s, k), al in by_sk.items():
        for t in range(T):
            rows["sup"][(s, k, t)] = lp.le(np.concatenate([f[a, :, t] for a in al]), 1.0, net.supply[(s, k)])
    # occupancy
    for c in sorted(set(net.cls.values())):
        al = [a for a, (_, _, k) in enumerate(arcs) if net.cls[k] == c]
        for vi, v in enumerate(V):
            for t in range(T):
                idx = [f[a, vi, h] for a in al for h in range(max(0, t - 2 * tau[a] + 1), t + 1)]
                rows["occ"][(c, v, t)] = lp.le(idx, 1.0, sc.cap[(c, v)][t])
    # unloading
    for j in J:
        al = [a for a, (_, jj, _) in enumerate(arcs) if jj == j]
        for t in range(T):
            idx = [f[a, vi, t - tau[a]] for a in al for vi in range(len(V)) if t - tau[a] >= 0]
            if idx:
                rows["unl"][(j, t)] = lp.le(idx, 1.0, sc.H[jx[j], t])
    info = dict(arcs=arcs, tau=tau, taubar=taubar, rows=rows, J=J, K=K, V=V, T=T, base_c=base_c)
    if extra is not None:
        extra(lp, info)
    lp.build()
    return lp, info


def outcomes(sol, lp, info, sc: Scenario, waiver_on: bool):
    f = sol.val("f"); b = sol.val("b")
    arcs, tau, T, V = info["arcs"], info["tau"], info["T"], info["V"]
    sail = float(sum(info["base_c"][a] * sc.mult[v] * f[a, vi, :].sum() for a in range(len(arcs)) for vi, v in enumerate(V)))
    used_w = float(f[:, V.index("waiver"), :].sum()) if "waiver" in V else 0.0
    direct = sail + (sc.activation if (waiver_on and used_w > 1e-6) else 0.0)
    delivered = float(sum(f[a, :, t].sum() for a in range(len(arcs)) for t in range(T) if t + tau[a] < T))
    B = b.sum(axis=(0, 1))
    return dict(direct_musd=direct, cost_per_t=direct * MUSD / (delivered * KT) if delivered > 0 else np.nan,
                cum_backlog=float(B.sum()), peak_backlog=float(B.max()), backlog_path=B,
                waiver_kt=used_w, delivered_kt=delivered, obj=sol.obj)


def clearance(path, window_end, threshold):
    """Days after the disruption window until backlog stays at or below the threshold."""
    T = len(path)
    for d in range(0, T - window_end + 1):
        if all(path[window_end + d - 1 + i] <= threshold + 1e-9 for i in range(0, T - (window_end + d - 1))):
            return d
    return T - window_end
