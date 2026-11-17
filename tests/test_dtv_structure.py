"""Dynamic tiered vetting: static special case and conservation of vessel stocks."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from sgw import model as SM, policies as PO, data as SD
from sgw.model import lagw

def _conservation_residual(sol, info, env, prm, pol):
    G = SD.GROUPS; dt = env.dt; S = env.S
    Z, U, Ur, x, xr = (sol.val(n) for n in ("Z", "U", "Ur", "x", "xr"))
    lam = np.array([env.lam[g] for g in G]) * dt
    mode = info["mode"]; rung = info["rung"]
    # arrivals in steps (same construction as build)
    A = np.zeros_like(U)
    import math
    for d in range(SD.T):
        st = int(max(0, math.floor((d - prm.lead) / dt))); A[..., min(S - 1, st)] += env.Ad[..., d]
    # screenings still in the pipeline at the end of step s
    def pipe(xx, modes):
        P = np.zeros((len(G), 2, S))
        for gi in range(len(G)):
            for L, w in lagw(prm.mode_d[modes[gi]], dt):
                for s in range(S):
                    # x started at s' clears at s'+L: in pipeline for s' <= s < s'+L
                    lo = max(0, s - L + 1)
                    P[gi, :, s] += w * xx[gi, :, :, lo:s + 1].sum(axis=1).sum(axis=-1) if L > 0 else 0.0
        return P
    N = Z + U.sum(axis=2) + Ur.sum(axis=2) + pipe(x, mode) + pipe(xr, rung)
    tot = N.sum(axis=(0, 1))
    dep = np.zeros(S)
    dep[1:] = (lam[:, None, None] * (Z + U.sum(axis=2) + Ur.sum(axis=2))[:, :, :-1]).sum(axis=(0, 1))
    inflow = A.sum(axis=(0, 1, 2))
    resid = tot[1:] - tot[:-1] - (inflow[1:] - dep[1:])
    return np.abs(resid).max()

def test_static_special_case_and_conservation():
    env = SM.Env(dt=2); prm = SM.Params()
    base = PO.named()["R4-T"]
    _, _, _, m_static = SM.solve(env, base, prm)
    _, _, _, m_zero = SM.solve(env, PO.with_rule(base, 0.0, 0.0, 0.0), prm)
    assert abs(m_static["obj"] - m_zero["obj"]) < 1e-6
    pol = PO.with_rule(SM.Policy(mu=dict(G="reg", AT="ex", AN="doc", UG="reg", R="doc"), var_routes=PO.LOW, name="nodeny"), 0.5, 0.02, 0.1)
    sol, lp, info, _ = SM.solve(env, pol, prm)
    r = _conservation_residual(sol, info, env, prm, pol)
    assert r < 1e-5, r
    return m_static["obj"], r

if __name__ == "__main__":
    o, r = test_static_special_case_and_conservation()
    print(f"static special case reproduces static objective {o:.4f}; vessel-stock conservation residual {r:.2e}")
