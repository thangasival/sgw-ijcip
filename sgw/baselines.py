"""Baselines B0-B4 (blueprint section 6.1): uniform screening, fixed flag rule, static risk prediction (PSIX deficiency
predictor), predict-then-optimise inspection (B3) and the B3-hybrid; B4 is the best static policy (see experiments)."""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import statsmodels.api as smapi
from . import data as SD
from .model import Policy, Env, Params, Z6TO4, ZONES4
from .policies import LOW


def uniform(mode):
    return Policy(mu=dict(G=mode, AT=mode, AN=mode, UG=mode, R="den"), var_routes=LOW, name=f"B0-{mode}")


def b1():
    return Policy(mu=dict(G="reg", AT="ex", AN="doc", UG="reg", R="den"), var_routes=LOW, name="B1 (R4-T flag rule)")


def predictor(shrink=2.0):
    """Poisson GLM of PSC deficiencies (offset: exams) on flag tier and gas carrier, fitted on PSIX vessels with >= 1
    exam; each vessel's prediction is shrunk toward the GLM rate by its own record (empirical Bayes, prior weight
    `shrink` exams).  Returns {normalised vessel name: predicted deficiencies per exam}, plus 5-fold CV pseudo-R2."""
    ps = pd.read_csv(SD.D7 / "psix_vessel_tiers.csv")
    ps["v"] = ps.vessel.map(SD._norm); ps["gas"] = (ps.vessel_type == "Gas carrier").astype(float); ps["green"] = ps.flag_green.astype(float)
    d = ps[ps.psc_exams_3y.fillna(0) > 0]
    X = smapi.add_constant(d[["green", "gas"]])
    m = smapi.GLM(d.deficiencies_3y, X, family=smapi.families.Poisson(), offset=np.log(d.psc_exams_3y)).fit()
    def prior(green, gas):
        return float(np.exp(m.params["const"] + m.params["green"] * green + m.params["gas"] * gas))
    pred = {}
    for r in ps.itertuples():
        pr = prior(r.green, r.gas); ex = 0.0 if pd.isna(r.psc_exams_3y) else r.psc_exams_3y
        dfc = 0.0 if pd.isna(r.deficiencies_3y) else r.deficiencies_3y
        pred[r.v] = (dfc + shrink * pr) / (ex + shrink)
    return pred, prior, m


def predicted_groups(VG: pd.DataFrame, V: pd.DataFrame):
    """B2 grouping: Amber = top predicted-deficiency vessels, same count as flag-tier Amber; Red unchanged."""
    pred, prior, _ = predictor()
    G = VG.copy()
    qs = V.drop_duplicates("vessel").set_index("vessel").flag_tier.eq("QUALSHIP 21")
    G["pred"] = [pred.get(v, prior(float(qs.get(v, False)), float(c == "gas"))) for v, c in zip(G.vessel, G.vclass)]
    n_amber = int((VG.group.isin(["AT", "AN"])).sum())
    cand = G[G.group != "R"].sort_values("pred", ascending=False)
    amber = set(cand.vessel.iloc[:n_amber])
    trig = {v for v, g in V.groupby("vessel") if g.gas_vessel.any() or g.dest.map(SD.KAPPA).eq(1.0).any()}
    G["group"] = [("R" if g == "R" else ("AT" if v in trig else "AN") if v in amber else "G") for v, g in zip(G.vessel, G.group)]
    return G


def b3_groups(VG_pred: pd.DataFrame, prm: Params, env_dt: int, zone_weights: np.ndarray):
    """B3 selection: in each zone and step, inspector-hours examine arriving vessels in descending order of predicted
    deficiencies (expected deficiencies detected per exam); vessels not selected pass with a registry check.
    Group mapping for the supply LP: examined -> 'AT' (mode ex), not examined -> 'AN' (mode reg) or 'G' (reg), Red -> den."""
    G = VG_pred.copy()
    cap = prm.r_share * prm.psc_hours_day * zone_weights * env_dt            # inspector-hours per step by zone
    G["step"] = ((G.first_day - 1 - prm.lead).clip(lower=0) // env_dt).astype(int)
    G["z4"] = G.zone.map(Z6TO4)
    sel = set(); expected = 0.0
    h_ex = prm.mode_h["ex"]
    for (z, s), g in G[G.group != "R"].groupby(["z4", "step"]):
        hours = cap[ZONES4.index(z)]
        for r in g.sort_values("pred", ascending=False).itertuples():
            if hours >= h_ex:
                sel.add(r.vessel); hours -= h_ex; expected += r.pred
    G["group"] = [("R" if g == "R" else ("AT" if v in sel else ("AN" if g in ("AT", "AN") else "G"))) for v, g in zip(G.vessel, G.group)]
    pol = Policy(mu=dict(G="reg", AT="ex", AN="reg", UG="reg", R="den"), var_routes=LOW, name="B3 predict-then-optimise")
    return G, pol, dict(examined=len(sel), expected_deficiencies=expected)
