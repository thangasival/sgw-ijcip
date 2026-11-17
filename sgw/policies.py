"""Named designs, policy enumeration and the voyage-list delay accounting used to cross-check against v7."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from .model import Policy
from . import data as SD

ALL_J = tuple(SD.DESTS)
LOW = tuple(sorted(SD.LOW_ESB))


def named():
    """Observed 2026 designs (R2, R3) and register-based alternatives (R4, R4-D, R4-T) as special cases."""
    none = {g: "none" for g in SD.GROUPS}
    return {
        "R2": Policy(mu=dict(none), name="R2"),
        "R3": Policy(mu=dict(none), var_routes=ALL_J, name="R3"),
        "R4": Policy(mu=dict(G="reg", AT="ex", AN="ex", UG="reg", R="den"), var_routes=LOW, name="R4"),
        "R4-D": Policy(mu=dict(G="reg", AT="doc", AN="doc", UG="reg", R="den"), var_routes=LOW, name="R4-D"),
        "R4-T": Policy(mu=dict(G="reg", AT="ex", AN="doc", UG="reg", R="den"), var_routes=LOW, name="R4-T"),
    }


def enumerate_static(var_routes=LOW):
    """All static mode assignments: G in {reg, doc}, AT in {doc, docp, ex}, AN in {reg, doc, docp, ex}, R denied."""
    out = []
    for mg, mt, mn in itertools.product(["reg", "doc"], ["doc", "docp", "ex"], ["reg", "doc", "docp", "ex"]):
        out.append(Policy(mu=dict(G=mg, AT=mt, AN=mn, UG=mg, R="den"), var_routes=var_routes, name=f"G:{mg}|AT:{mt}|AN:{mn}"))
    return out


def with_rule(pol: Policy, u, dg, da, inform=1.0):
    return Policy(mu=dict(pol.mu), var_routes=pol.var_routes, u=u, delta_g=dg, delta_a=da, inform=inform,
                  name=f"{pol.name}|u={u}|dG={dg}|dA={da}|i={inform}")


# ---------------- voyage-list delay accounting (v7 cross-check) ----------------
V7_HOURS = {"reg": 6.0, "doc_register": 6.0, "doc_day_one": 24.0, "ex": 72.0, "var": 24.0}


def voyage_list_vessel_days(P: pd.DataFrame, design: str, register=True, hours=V7_HOURS):
    """Added pre-loading vessel-days per design on a voyage list with v7 columns (vessel, load_date_parsed, flag_tier,
    vessel_type, kappa, delay_group).  Security review and the low-ESB VAR run in parallel (max of the two)."""
    d = P.sort_values("load_date_parsed")
    first = ~d.vessel.duplicated(); amber = d.flag_tier != "QUALSHIP 21"
    trig = amber & ((d.vessel_type == "Gas carrier") | (d.kappa >= 1.0))
    low = d.delay_group.isin(["east", "intra"])
    if design == "R2":
        h = np.zeros(len(d))
    elif design == "R3":
        h = np.full(len(d), hours["var"])
    else:
        doc = hours["doc_register"] if register else hours["doc_day_one"]
        if design == "R4":
            sec = np.where(amber & first, hours["ex"], hours["reg"]) if register else np.where(first, hours["ex"], hours["reg"])
        elif design == "R4-T":
            sec = np.where(first & trig, hours["ex"], np.where(first & ~register, doc, hours["reg"]))
        elif design == "R4-D":
            sec = np.where(first & ~register, doc, hours["reg"]) if not register else np.full(len(d), hours["reg"])
        var = np.where(low, hours["var"], 0.0)
        h = np.maximum(sec, var)
    return float(np.sum(h) / 24.0)
