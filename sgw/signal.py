"""H3a: out-of-time persistence of section 501(c) filing behaviour (report date vs due date), from register timestamps.

This tests a compliance signal only.  No security-incident labels exist; nothing here predicts security risk.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from . import data as SD

SPLIT = pd.Timestamp("2026-06-01")      # fit: loads 17 Mar-31 May; test: loads 1 Jun-16 Aug


def records():
    R = pd.read_csv(SD.D7 / "register_reconciliation_20261009.csv", parse_dates=["ld1", "rep", "due"])
    R = R[~R.status.str.startswith("pre-waiver") & R.rep.notna() & R.due.notna() & (R.ld1 >= SD.DAY0) & (R.ld1 < SD.VAR_START)].copy()
    R["late"] = (R.rep > R.due).astype(int)
    R["vessel"] = R["Vessel Name"].map(SD._norm)
    qs = pd.read_csv(SD.D7 / "waiver_voyage_panel_v4.csv")[["vessel", "flag_tier"]]
    qs["vessel"] = qs.vessel.map(SD._norm); tier = qs.drop_duplicates("vessel").set_index("vessel").flag_tier
    lf = pd.read_csv(SD.D7 / "register_later_filings.csv"); lf["vessel"] = lf["Vessel Name"].map(SD._norm)
    tier = pd.concat([tier, lf.drop_duplicates("vessel").set_index("vessel").flag_tier]); tier = tier[~tier.index.duplicated()]
    R["green"] = R.vessel.map(tier).eq("QUALSHIP 21").astype(int)
    return R.sort_values(["vessel", "ld1"]).reset_index(drop=True)


def _brier(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def persistence(R=None, n_boot=2000, seed=7):
    R = records() if R is None else R
    tr, te = R[R.ld1 < SPLIT], R[R.ld1 >= SPLIT]
    # baseline: flag-tier late rate in the fit window
    base = tr.groupby("green").late.mean()
    # history feature: vessel had a late filing among its fit-window voyages
    hist = tr.groupby("vessel").late.max()
    T = te[te.vessel.isin(hist.index)].copy()
    T["prior_late"] = T.vessel.map(hist)
    # model: late rate by (flag tier, prior_late) estimated on consecutive-voyage pairs inside the fit window
    pr = tr.copy(); pr["prev"] = pr.groupby("vessel").late.shift(); pairs = pr.dropna(subset=["prev"])
    glob = tr.late.mean()
    def cell(g, h):
        s = pairs[(pairs.green == g) & (pairs.prev == h)].late
        k = 2.0                                           # shrink small cells toward the flag-tier rate
        return (s.sum() + k * base.get(g, glob)) / (len(s) + k)
    T["p_base"] = T.green.map(base).fillna(glob)
    T["p_model"] = [cell(g, h) for g, h in zip(T.green, T.prior_late)]
    out = dict(n_fit_records=len(tr), n_test_records=len(te), n_test_with_history=len(T), n_test_vessels=T.vessel.nunique(),
               late_rate_fit=float(tr.late.mean()), late_rate_test=float(te.late.mean()), n_pairs_fit=len(pairs))
    out["test_late_rate_prior_late"] = float(T[T.prior_late == 1].late.mean()) if (T.prior_late == 1).any() else np.nan
    out["test_late_rate_prior_ontime"] = float(T[T.prior_late == 0].late.mean()) if (T.prior_late == 0).any() else np.nan
    tab = pd.crosstab(T.prior_late, T.late).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    out["odds_ratio"], out["fisher_p"] = fisher_exact(tab.values)
    bb, bm = _brier(T.late, T.p_base), _brier(T.late, T.p_model)
    out.update(brier_base=bb, brier_model=bm, bss=1 - bm / bb if bb > 0 else np.nan)
    rng = np.random.default_rng(seed); vs = T.vessel.unique(); bss = []
    for _ in range(n_boot):
        pick = rng.choice(vs, len(vs), replace=True)
        S = pd.concat([T[T.vessel == v] for v in pick])
        b0 = _brier(S.late, S.p_base)
        bss.append(1 - _brier(S.late, S.p_model) / b0 if b0 > 0 else np.nan)
    lo, hi = np.nanpercentile(bss, [2.5, 97.5]); out.update(bss_lo=float(lo), bss_hi=float(hi))
    out["g1_pass"] = bool(lo > 0)
    return out, T


def transition_rates(R=None):
    """Calibration of the re-tiering rule from consecutive voyages by the same vessel (all initial-period records):
    u = P(next filing on time | previous on time) for non-QS21 vessels; delta_G, delta_A = per-voyage late rate."""
    R = records() if R is None else R
    pr = R.copy(); pr["prev"] = pr.groupby("vessel").late.shift(); pairs = pr.dropna(subset=["prev"])
    am = pairs[pairs.green == 0]
    u = float(1 - am[am.prev == 0].late.mean()) if (am.prev == 0).any() else np.nan
    return dict(u=u, n_amber_pairs=int((am.prev == 0).sum()), delta_G=float(R[R.green == 1].late.mean()),
                delta_A=float(R[R.green == 0].late.mean()), n_G=int((R.green == 1).sum()), n_A=int((R.green == 0).sum()))
