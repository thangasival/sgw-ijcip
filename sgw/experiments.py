"""Experiment runner for Env-R (E1-E7).  A task is a plain dict so that it can be sent to worker processes:

    dict(id=..., grouping="flag"|"psix"|"pred"|"b3", west_mbbl=28.3, red_share=None, prm={overrides},
         policy=dict(kind="named"|"static"|"uniform"|"b1"|"b3", name=..., rule=(u, dG, dA, inform) or None,
                     var_routes=None), ebar={channel: cap} or None, tag=...)

Workers cache Env objects by (grouping, west_mbbl, red_share, b3-relevant params)."""
from __future__ import annotations
import json, time, traceback
import numpy as np, pandas as pd
from . import data as SD, model as SM, policies as SP, baselines as BL

CH = ("crew", "own", "cond")
_ENV = {}


def make_prm(over):
    over = dict(over or {})
    p = SM.Params()
    for k, v in over.items():
        if k in ("mode_d_scale", "mode_h_scale"):
            continue
        setattr(p, k, tuple(v) if k == "res_scale" else v)
    if "mode_d_scale" in over:
        p.mode_d = {m: d * over["mode_d_scale"] for m, d in p.mode_d.items()}
    if "mode_h_scale" in over:
        p.mode_h = {m: h * over["mode_h_scale"] for m, h in p.mode_h.items()}
    return p


def get_env(t, prm):
    g = t.get("grouping", "flag"); wm = t.get("west_mbbl", 28.3); rs = t.get("red_share")
    ls = t.get("life_scale", 1.0)
    key = (g, wm, rs, ls) + ((prm.r_share, prm.lead, prm.mode_h["ex"]) if g == "b3" else ())
    if key in _ENV:
        return _ENV[key]
    if ls != 1.0:                       # Env-S (synthetic): vessel lifetimes scaled -> repeat-voyage share
        from .synth import env_s
        _ENV[key] = (env_s(dict(t, life_scale=1.0), prm, ls), None)
        return _ENV[key]
    if g in ("flag", "psix"):
        env = SM.Env(grouping=g, west_mbbl=wm, red_share=rs); extra = None
    else:
        base = get_env(dict(grouping="flag", west_mbbl=wm, red_share=rs), prm)[0]
        VGp = BL.predicted_groups(base.VG, base.V)
        if g == "pred":
            env = SM.Env(west_mbbl=wm, VG=VGp); extra = None
        else:
            VG3, _, extra = BL.b3_groups(VGp, prm, base.dt, base.zw)
            env = SM.Env(west_mbbl=wm, VG=VG3)
    _ENV[key] = (env, extra)
    return _ENV[key]


def make_policy(spec):
    k = spec["kind"]; nm = spec.get("name")
    if k == "named":
        pol = SP.named()[nm]
    elif k == "static":
        pol = {p.name: p for p in SP.enumerate_static()}[nm]
    elif k == "uniform":
        pol = BL.uniform(nm)
    elif k == "b1":
        pol = BL.b1()
    elif k == "b3":
        pol = SM.Policy(mu=dict(G="reg", AT="ex", AN="reg", UG="reg", R="den"), var_routes=SP.LOW, name="B3")
    else:
        raise ValueError(k)
    if spec.get("var_routes") is not None:
        pol = SM.Policy(mu=dict(pol.mu), var_routes=tuple(spec["var_routes"]), u=pol.u, delta_g=pol.delta_g,
                        delta_a=pol.delta_a, inform=pol.inform, name=pol.name)
    if spec.get("mu_override"):
        mu = dict(pol.mu); mu.update(spec["mu_override"])
        pol = SM.Policy(mu=mu, var_routes=pol.var_routes, u=pol.u, delta_g=pol.delta_g, delta_a=pol.delta_a,
                        inform=pol.inform, name=pol.name + "|" + ",".join(f"{k}:{v}" for k, v in spec["mu_override"].items()))
    if spec.get("rule"):
        u, dg, da, inf = spec["rule"]
        pol = SP.with_rule(pol, u, dg, da, inf)
    return pol


def run_task(t):
    t0 = time.time()
    try:
        prm = make_prm(t.get("prm"))
        env, extra = get_env(t, prm)
        pol = make_policy(t["policy"])
        sol, lp, info, m = SM.solve(env, pol, prm, ebar=t.get("ebar"))
        out = dict(t, **m, status=str(getattr(sol, "status", getattr(sol, "msg", "ok"))), secs=time.time() - t0)
        thr = info["rows"]["thr"]
        if len(thr):
            th = -np.asarray(sol.y_ub)[thr].reshape(len(env.Z), -1)
            out["theta_zone"] = th.mean(axis=1).tolist()
            out["theta_busy_share"] = float((th > 1e-6).mean())
        for ch, r in info["rows"].get("exp", {}).items():
            out[f"lam_{ch}"] = float(-np.asarray(sol.y_ub)[r]) if np.ndim(r) == 0 else float(-np.asarray(sol.y_ub)[r].sum())
        if extra:
            out.update({f"b3_{k}": v for k, v in extra.items()})
        out["mode"] = list(info["mode"])
        return out
    except Exception as e:  # keep the batch alive; failures are reported in the results table
        return dict(t, status="error", error=repr(e), tb=traceback.format_exc()[-800:], secs=time.time() - t0)


def run_batch(tasks, out_jsonl, workers=4):
    import multiprocessing as mp
    done = set()
    try:
        with open(out_jsonl, encoding="utf-8") as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    except FileNotFoundError:
        pass
    todo = [t for t in tasks if t["id"] not in done]
    if not todo:
        return
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool, open(out_jsonl, "a", encoding="utf-8") as fh:
        for i, r in enumerate(pool.imap_unordered(run_task, todo)):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            print(f"{i + 1}/{len(todo)} {r['id']} {r.get('status')} {r.get('secs', 0):.0f}s", flush=True)


def load(out_jsonl):
    with open(out_jsonl, encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(l) for l in fh if l.strip()])


def caps_from(r2_row, frac):
    return {ch: frac * r2_row[f"E_{ch}"] for ch in CH}


def draw_params(n=30, seed=20261113):
    """Pre-registered parameter ranges (PREREG.md section 4)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(dict(r_share=float(rng.uniform(0.02, 0.10)), mode_d_scale=float(rng.uniform(0.7, 1.3)),
                        mode_h_scale=float(rng.uniform(0.7, 1.3)), res_scale=tuple(float(x) for x in rng.uniform(0.8, 1.2, 3)),
                        s_dom=float(rng.uniform(0.02, 0.10)), west_mbbl=float(rng.uniform(18.8, 34.8)),
                        lead=float(rng.uniform(1.0, 3.0))))
    return out
