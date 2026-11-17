"""Calibrated 2026 network inputs (Env-R).  All inputs are read from data/v7 (read-only, checksummed public-data extracts:
MARAD register, USCG PSIX, EIA imports, MARAD fleet list, and the interrupted time-series fits of PADD 5 imports).

Units: kt (thousand tonnes), days, USD million.  Day 1 = 17 March 2026 (waiver effective).
"""
from __future__ import annotations
import math, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D7 = ROOT / "data" / "v7"
DER = ROOT / "data" / "derived"
DAY0 = pd.Timestamp("2026-03-17")
VAR_START = pd.Timestamp("2026-08-17")
LOAD_END = pd.Timestamp("2026-11-15")
TAIL = 30                                               # run-out days after the loading deadline
T = (LOAD_END - DAY0).days + 1 + TAIL                   # 244 + 30 = 274
BBL_PER_T = {"refined": 7.33, "crude": 7.33, "gas": 11.6}

DESTS = ["West", "GulfEast", "PR", "HI", "AK", "Intra"]
CLASSES = ["refined", "crude", "gas"]
VCLASS = {"refined": "tank", "crude": "tank", "gas": "gas"}
# transit days: v7 TRANSIT for west 20, east 5, intra 2; island split by distance at 12 kn (+3 d canal for Pacific islands)
TRANSIT = {"West": 20, "GulfEast": 5, "PR": 7, "HI": 25, "AK": 26, "Intra": 2}
DIST_NM = {"West": 4500, "GulfEast": 1900, "PR": 1900, "HI": 6200, "AK": 6600, "Intra": 600}
KAPPA = {"West": 0.8, "GulfEast": 0.8, "PR": 1.0, "HI": 1.0, "AK": 1.0, "Intra": 0.6}     # v7 rule (+0.2 gas, cap 1)
PRIORITY = {"West": 0.8, "GulfEast": 0.3, "PR": 1.0, "HI": 1.0, "AK": 1.0, "Intra": 0.2}   # grade D; follows ESB ordering
LOW_ESB = {"GulfEast", "Intra"}                         # established domestic service (VAR right of first refusal)
ZONES = ["Houston-Galveston", "Sabine-Lake Charles", "Corpus Christi", "Lower Mississippi", "Atlantic-Caribbean", "Pacific"]
GROUPS = ["G", "AT", "AN", "UG", "R"]                   # Green, Amber-triggered, Amber-not-triggered, Upgraded (earned Green), Red

_ZMAP = [("Corpus", "Corpus Christi"),
         ("Beaumont|Port Arthur|Nederland|Lake Charles|Sabine|Orange|Westlake", "Sabine-Lake Charles"),
         ("Houston|Galveston|Texas City|Pasadena|Freeport|Seabrook|Baytown|Deer Park|La Porte|Channelview", "Houston-Galveston"),
         ("Louisiana|Mississippi|Alabama|Mobile|Pascagoula|Garyville|Norco|Convent|St\\.", "Lower Mississippi"),
         ("California|Washington|Oregon|Alaska|Hawaii", "Pacific")]


def zone_of(origin: str) -> str:
    o = str(origin)
    for pat, z in _ZMAP:
        if re.search(pat, o, re.I):
            return z
    return "Atlantic-Caribbean"


def _cls(cargo_class: str) -> str | None:
    c = str(cargo_class)
    if c.startswith("Dry bulk"):
        return None
    if c.startswith("Crude"):
        return "crude"
    if c.startswith("Ammonia") or c.startswith("LPG"):
        return "gas"
    return "refined"


def _dest(delay_group, d_region) -> str:
    if delay_group == "west_from_gulf":
        return "West"
    if delay_group == "island":
        return {"Hawaii": "HI", "Alaska": "AK"}.get(d_region, "PR")
    if delay_group == "east":
        return "GulfEast"
    return "Intra"


def _norm(v):
    return re.sub(r"[^a-z]", "", str(v).lower().replace("m/v", ""))


def voyages() -> pd.DataFrame:
    """Initial-period voyages in the 9 Oct 2026 register: the 250-voyage panel plus the 36 later filings.
    Columns: vessel, day (load day index, 1-based), dest, cls, kt, flag_tier, zone, source."""
    P = pd.read_csv(D7 / "waiver_voyage_panel_v4.csv", parse_dates=["load_date_parsed"])
    a = pd.DataFrame(dict(vessel=P.vessel.map(_norm), load=P.load_date_parsed, dest=[_dest(g, r) for g, r in zip(P.delay_group, P.d_region)],
                          cls=P.cargo_class.map(_cls), bbl=P.barrels, mt=P.metric_tons, flag_tier=P.flag_tier, origin=P.origin,
                          gas_vessel=P.vessel_type.eq("Gas carrier"), source="panel"))
    L = pd.read_csv(D7 / "register_later_filings.csv", parse_dates=["ld1"])
    desc = L["Description of Cargo"].astype(str).str.lower()
    lcls = np.where(desc.str.contains("crude"), "crude", np.where(desc.str.contains("propane|butane|lpg|ammonia|ngl"), "gas",
                    np.where(desc.str.contains("coke|coal"), None, "refined")))
    reg = L.d_reg.fillna("")
    b = pd.DataFrame(dict(vessel=L["Vessel Name"].map(_norm), load=L.ld1, dest=[_dest(g, r) for g, r in zip(L.delay_group, reg)],
                          cls=lcls, bbl=L.bbl, mt=np.nan, flag_tier=L.flag_tier, origin=L["Port of Loading"],
                          gas_vessel=(lcls == "gas"), source="later_filing"))
    # later filings carry the route group only; non-contiguous ones are PR unless the discharge port says otherwise
    hon = L["Port of Discharge"].astype(str).str.contains("Hawaii|Honolulu|Barbers", case=False)
    ak = L["Port of Discharge"].astype(str).str.contains("Alaska", case=False)
    b.loc[(b.dest == "PR") & hon.values, "dest"] = "HI"; b.loc[(b.dest == "PR") & ak.values, "dest"] = "AK"
    V = pd.concat([a, b], ignore_index=True)
    V = V[V.cls.notna() & (V.load >= DAY0) & (V.load < VAR_START)].copy()
    V["kt"] = np.where(V.mt.notna(), V.mt / 1000.0, V.bbl / V.cls.map(BBL_PER_T) / 1000.0)
    med = V.groupby("cls").kt.median()
    V["kt"] = V.kt.where(V.kt.notna() & (V.kt > 1.0), V.cls.map(med))       # missing or implausible (<1 kt) quantities
    V["day"] = (V.load - DAY0).dt.days + 1
    V["zone"] = V.origin.map(zone_of)
    return V.reset_index(drop=True)


def vessel_groups(V: pd.DataFrame, grouping: str = "flag") -> pd.DataFrame:
    """One row per vessel: group (flag-tier primary or PSIX-record variant), vessel class, zone of first load, first day,
    availability lifetime (days in the waiver trade)."""
    own = pd.read_csv(D7 / "ownership_register_records.csv")
    red = set(own.loc[own.name_evident_prc.astype(bool), "vessel"].map(_norm))
    ps = pd.read_csv(D7 / "psix_vessel_tiers.csv"); ps["v"] = ps.vessel.map(_norm)
    clean = dict(zip(ps.v, ps.vessel_clean.astype(bool)))
    rows = []
    for v, g in V.sort_values("day").groupby("vessel", sort=False):
        first, last = g.day.min(), g.day.max()
        trig = bool(g.gas_vessel.any() or g.dest.map(KAPPA).eq(1.0).any())
        qs = (g.flag_tier.iloc[0] == "QUALSHIP 21")
        green = qs if grouping == "flag" else clean.get(v, False)
        grp = "R" if v in red else ("G" if green else ("AT" if trig else "AN"))
        life = (last - first) + 2 * TRANSIT[g.dest.iloc[-1]] + 2
        rows.append(dict(vessel=v, group=grp, vclass="gas" if g.gas_vessel.any() else "tank", zone=g.zone.iloc[0],
                         first_day=int(first), last_day=int(last), voyages=len(g), lifetime=int(life)))
    return pd.DataFrame(rows)


def its_monthly():
    """Monthly PADD 5 import shortfall (counterfactual - observed, b/d) from the v7 ITS model, linear 2021+ spec."""
    import sys, os
    v7code = ROOT.parent / "repo" / "jones_act_waiver_2026_replication" / "code"
    cache = DER / "its_monthly_p5.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    cwd = os.getcwd(); sys.path.insert(0, str(v7code))
    import common as C7
    G7 = C7.load_model(); os.chdir(cwd)
    e = G7["load_eia"](str(D7 / "eia_p5_imports.xls")); out, _ = G7["its_fit"](e)
    out["shortfall_bpd"] = out.cf_bpd - out.observed_bpd
    out.to_csv(cache, index=False)
    return out


def demand(V: pd.DataFrame, west_total_mbbl: float = 28.3, cato_absent=None):
    """Emergency (incremental) marine lift requirement q[j,k,t] in kt released per day.
    * West, 17 Mar - 31 Jul: PADD 5 import shortfall (v7 ITS; mean of four specifications = 28.3 Mbbl; range 18.8-34.8),
      monthly shape from the linear specification, split refined/crude by the observed Gulf->West waiver mix.
    * Other routes, 17 Mar - 16 Aug: observed initial-period waiver volumes (panel + later filings + voyages identified as
      unreported) by route-month, spread evenly within each month: a lower bound on the requirement.
    * 17 Aug - 15 Nov (and West from 1 Aug): the initial-period mean daily volume by route and class ('had the initial
      design continued').  No demand is released after the loading deadline."""
    q = np.zeros((len(DESTS), len(CLASSES), T))
    loadT = (LOAD_END - DAY0).days + 1
    V = V.copy(); V["month"] = V.load.dt.to_period("M")
    days = pd.date_range(DAY0, LOAD_END)
    # identified unreported (Cato, 9 Oct; still absent from the register): 3 HI, 2 PR, 1 Marcus Hook, 1 unnamed (-> GulfEast)
    extra = cato_absent if cato_absent is not None else [("HI", "refined", "2026-07-25"), ("HI", "refined", "2026-06-15"),
             ("HI", "gas", "2026-05-22"), ("PR", "refined", "2026-07-22"), ("PR", "refined", "2026-07-27"),
             ("GulfEast", "crude", "2026-08-15"), ("GulfEast", "refined", "2026-07-15")]
    med = V.groupby("cls").kt.median()
    ex = pd.DataFrame([dict(dest=d, cls=c, load=pd.Timestamp(t), kt=med[c]) for d, c, t in extra]); ex["month"] = ex.load.dt.to_period("M")
    W = pd.concat([V[["dest", "cls", "load", "kt", "month"]], ex], ignore_index=True)
    init_days = (VAR_START - DAY0).days
    rate = W.groupby(["dest", "cls"]).kt.sum() / init_days                    # mean kt/day over the initial period
    for (d, c, m), vol in W.groupby(["dest", "cls", "month"]).kt.sum().items():
        if d == "West":
            continue
        md = [i for i, x in enumerate(days) if x.to_period("M") == m and x < VAR_START]
        for i in md:
            q[DESTS.index(d), CLASSES.index(c), i] += vol / len(md)
    # West from the ITS shortfall
    its = its_monthly(); its["month"] = its.date.dt.to_period("M")
    sh = its.set_index("month").shortfall_bpd.clip(lower=0)
    tot_bbl = sum(sh[m] * days.to_period("M").tolist().count(m) for m in sh.index)    # waiver-period days only
    scale = west_total_mbbl * 1e6 / tot_bbl
    wmix = W[W.dest == "West"].groupby("cls").kt.sum(); wmix = wmix / wmix.sum()
    for i, x in enumerate(days):
        m = x.to_period("M")
        if m in sh.index:
            kt_day = sh[m] * scale / BBL_PER_T["refined"] / 1000.0
            for c, share in wmix.items():
                q[DESTS.index("West"), CLASSES.index(c), i] += kt_day * share
    # continuation
    west_end = (pd.Timestamp("2026-08-01") - DAY0).days
    for (d, c), r in rate.items():
        start = west_end if d == "West" else init_days
        q[DESTS.index(d), CLASSES.index(c), start:loadT] += r if d != "West" else 0.0
    # West continuation at the July shortfall level (scaled)
    jul = sh.get(pd.Period("2026-07", "M"), 0.0) * scale / BBL_PER_T["refined"] / 1000.0
    for c, share in wmix.items():
        q[DESTS.index("West"), CLASSES.index(c), west_end:loadT] += jul * share
    return q, W


def unloading():
    """Destination unloading proxy (kt/day): maximum 2025 monthly foreign petroleum receipts at the destination's
    customs ports (EIA company-level imports), an upper bound for marine receipts of the emergency increment."""
    cache = DER / "unloading_proxy.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0).iloc[:, 0]; return s
    EI = pd.read_excel(D7 / "eia_company_imports_2025.xlsx", sheet_name="IMPORTS")
    EI = EI[EI.CNTRY_NAME != "CANADA"].copy(); EI["mon"] = pd.to_datetime(EI.RPT_PERIOD).dt.month
    def dest(r):
        st = str(r.PORT_STATE).upper()
        if "PUERTO" in st: return "PR"
        if "HAWAII" in st: return "HI"
        if "ALASKA" in st: return "AK"
        return {5: "West", 1: "GulfEast", 3: "Intra"}.get(r.PORT_PADD)
    EI["dest"] = EI.apply(dest, axis=1)
    s = EI.groupby(["dest", "mon"]).QUANTITY.sum().groupby("dest").max() / 30 / BBL_PER_T["refined"]
    s.to_csv(cache); return s


def domestic_fleet():
    """Jones Act-eligible self-propelled tankers (MARAD, 4 Apr 2026), excluding Alaska North Slope crude tankers."""
    F = pd.read_excel(D7 / "MARAD_USflag_fleet_2026.xlsx", sheet_name="Ships", header=9)
    F = F[F["Vessel Name"].notna() & (F["Jones Act Eligible"] == "Y") & F["Ship Type"].astype(str).str.contains("Tank", case=False)]
    ans = F.Operator.astype(str).str.contains("Alaska|BP|Polar|Conoco", case=False)
    return dict(n_all=len(F), dwt_all_kt=F["Deadweight Tons"].sum() / 1000, n_spot=int((~ans).sum()),
                dwt_spot_kt=F.loc[~ans, "Deadweight Tons"].sum() / 1000)


def zone_weights():
    """Share of 2025 foreign petroleum receipts by screening zone (proxy for where PSC inspectors are posted)."""
    cache = DER / "zone_weights.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0).iloc[:, 0]
    EI = pd.read_excel(D7 / "eia_company_imports_2025.xlsx", sheet_name="IMPORTS")
    EI = EI[EI.CNTRY_NAME != "CANADA"].copy()
    z = (EI.PORT_CITY.astype(str) + ", " + EI.PORT_STATE.astype(str)).map(zone_of)
    w = EI.groupby(z).QUANTITY.sum(); w = (w / w.sum()).reindex(ZONES).fillna(0)
    w.to_csv(cache); return w
