"""Figures F1-F6 for the IJCIP manuscript.  Run after the experiment tables exist in outputs/tables.
Writes PDF (vector), 600-dpi TIFF and PNG to outputs/figures."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
MM = 1 / 25.4
W2, W1 = 190 * MM, 90 * MM                     # Elsevier double / single column
COL = {"DTV": "#1f4e79", "B3": "#d95f02", "B3-hybrid": "#f2a15f", "B4": "#6a8a9e", "B2": "#9e9e9e",
       "B1": "#bdbdbd", "B0": "#d9d9d9", "R4-T": "#7b6aa6", "obs": "#333333"}
LABEL = {"B0-doc": "B0 uniform documentary", "B1": "B1 fixed flag rule (R4-T)", "B2": "B2 static risk prediction",
         "B3": "B3 predict-then-optimise", "B3h": "B3-hybrid", "B4": "B4 best static", "DTV": "DTV"}


def _style():
    try:
        from figure_style_kernel import apply_figure_style  # noqa
    except Exception:
        pass
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "legend.frameon": False, "axes.titlelocation": "left", "savefig.dpi": 600})


def _letter(ax, s, dx=-0.14, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=600 if ext == "png" else None)
    fig.savefig(FIG / f"{name}.tif", bbox_inches="tight", pad_inches=0.02, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ---------------------------------------------------------------- F1 schematic (shipping-and-security palette)
NAVY, NAVY_F, NAVY_L = "#14325C", "#E4EBF5", "#F4F7FB"     # security gate: maritime navy
AMB, AMB_F, AMB_L = "#C8701A", "#FCEAD6", "#FDF6EE"        # gating constraints: signal amber
TEAL, TEAL_F, TEAL_L = "#1B6E80", "#E0F0F3", "#F2F8F9"     # logistics: ocean teal
SLATE, INK, GREY = "#4A5568", "#1F2933", "#5F6B7A"


def f1_schematic():
    mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "svg.fonttype": "path"})
    fig,ax=plt.subplots(figsize=(W2,128*MM)); ax.set_xlim(0,100); ax.set_ylim(-4.5,64); ax.axis("off")
    LS=2.25
    def lane(y0,h,fc,ec,lab,col):
        ax.add_patch(FancyBboxPatch((0.5,y0),99,h,boxstyle="round,pad=0,rounding_size=0.8",fc=fc,ec=ec,lw=0.6))
        ax.text(1.5,y0+h-1.0,lab,fontsize=6.8,fontweight="bold",color=col,va="top")
    def box(x,y,w,h,lines,ec,fc,dashed=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.9",fc=fc,ec=ec,lw=1.0,ls=(0,(4,2.5)) if dashed else "-"))
        n=len(lines)
        for i,(t,kw) in enumerate(lines):
            ax.text(x+w/2,y+h/2+(n-1)/2*LS-i*LS,t,ha="center",va="center",fontsize=kw.get("fs",6.6),color=kw.get("c",INK),fontweight=kw.get("fw","normal"))
    def poly(pts,col,ls="-",lw=1.0):
        for (x0,y0),(x1,y1) in zip(pts[:-2],pts[1:-1]): ax.plot([x0,x1],[y0,y1],color=col,lw=lw,ls=ls,solid_capstyle="butt",dash_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle="-|>",mutation_scale=7,lw=lw,color=col,linestyle=ls,shrinkA=0,shrinkB=0))
    S=6.0
    lane(40.5,23,NAVY_L,"#C9D4E5","A. SECURITY GATE: VETTING PIPELINE",NAVY)
    lane(28.5,11.2,AMB_L,"#EED7BC","B. COUPLING: VETTING GATES WAIVER CAPACITY AND EXPOSURE",AMB)
    lane(2.5,25.2,TEAL_L,"#C5DFE4","C. LOGISTICS: LIFT, DELIVERY, BACKLOG",TEAL)
    # A
    box(1.5,47.5,18,10,[("Waiver-vessel requests",{}),(r"$A_{g,p,t}$",{}),(r"risk group $g$, port zone $p$",{"fs":S,"c":GREY})],SLATE,"white",True)
    box(1.5,41.5,18,4.8,[(r"Inspector-hours $R_{p,t}$",{}),("port-zone capacity",{"fs":S,"c":GREY})],AMB,"white",True)
    box(23,47.5,16,10,[("Vetting queue",{}),(r"$U_{g,p,t}$",{})],NAVY,NAVY_F)
    box(42.5,47.5,17.5,10,[(r"Screening $x_{g,p,t}$",{}),(r"mode $\mu(g)$",{}),(r"delay $d_m$, hours $h_m$",{"fs":S,"c":GREY}),("Red: denial",{"fs":S,"c":GREY})],NAVY,NAVY_F)
    box(63.5,47.5,18.5,10,[(r"Cleared stock $Z_{g,c,t}$",{}),("tiers G, AT, AN, UG",{}),("UG = upgraded Amber",{"fs":S,"c":GREY}),(r"class $c$: tank or gas",{"fs":S,"c":GREY})],NAVY,NAVY_F)
    for x0,x1 in [(19.5,23),(39,42.5),(60,63.5)]: poly([(x0,52.5),(x1,52.5)],INK)
    poly([(19.5,43.9),(51.25,43.9),(51.25,47.5)],AMB)
    ax.text(35.5,44.4,r"$\Sigma_{g}\, h_{\mu(g)}x_{g,p,t}\leq R_{p,t}\Delta$;  dual $\theta_{p,t}$",fontsize=S,color=AMB,ha="center",va="bottom")
    poly([(72.75,57.5),(72.75,60.3),(31,60.3),(31,57.5)],NAVY,ls=(0,(4,2)))
    ax.text(67,60.7,r"tripped signal or 30-day review: re-screen at next rung $\mu^{+}(g)$",fontsize=S,color=NAVY,ha="center",va="bottom")
    # B
    box(63.5,30,18.5,7.8,[("Waiver capacity gate",{"fw":"bold"}),(r"$O^{W}_{g,c,t}\leq\bar{\kappa}_c\,Z_{g,c,t}$",{}),(r"$\bar{\kappa}_c$: lift per vessel",{"fs":S,"c":GREY})],AMB,AMB_F)
    box(84.3,30,15,7.8,[("Exposure caps",{"fw":"bold"}),(r"$E^{h}\leq\bar{E}^{h}$",{}),("crew, owner, condition",{"fs":5.6,"c":GREY})],AMB,AMB_F)
    poly([(72.75,47.5),(72.75,37.8)],AMB,lw=1.3); poly([(72.75,30),(72.75,24)],AMB,lw=1.3)
    ax.add_patch(FancyArrowPatch((63.5,19),(63.5,50),arrowstyle="-|>",mutation_scale=7,lw=1.0,color=NAVY,linestyle=(0,(4,2)),connectionstyle="arc3,rad=-0.42"))
    ax.text(55,34.6,"re-tiering on in-waiver evidence:",fontsize=S,color=NAVY,ha="right",va="center")
    ax.text(55,32.4,r"voyages $n_{g,t}$, upgrade $u$, signals $\delta_G$, $\delta_A$",fontsize=S,color=NAVY,ha="right",va="center")
    # C
    box(1.5,14,18,9.5,[("Domestic Jones Act",{}),("tanker fleet",{}),(r"spot share $s^{\mathrm{dom}}$ of DWT",{"fs":S,"c":GREY})],SLATE,"white",True)
    box(23,14,16,9.5,[("Domestic lift",{}),(r"$f^{D}_{j,k,t}$",{}),("first on low-ESB routes",{"fs":S,"c":GREY})],TEAL,TEAL_F)
    box(63.5,14,18.5,10,[("Waiver lift",{"fw":"bold"}),(r"$f^{W}_{j,k,g,t}$",{}),("cleared vessels only",{"fs":S,"c":GREY}),("+ VAR lag (low-ESB routes)",{"fs":5.6,"c":GREY})],TEAL,TEAL_F)
    box(85,14,14,10,[(r"Backlog $b_{j,k,t}$",{}),("unmet demand",{"fs":S,"c":GREY}),(r"unloading $\leq H_j\Delta$",{"fs":S,"c":GREY})],TEAL,TEAL_F)
    box(85,4,14,6.5,[("Emergency",{}),(r"demand $q_{j,k,t}$",{})],SLATE,"white",True)
    poly([(19.5,18.75),(23,18.75)],INK)
    poly([(82,19.8),(85,19.8)],INK); ax.text(83.5,20.3,r"$\tau_j$",fontsize=S,ha="center",va="bottom",color=INK)
    poly([(31,14),(31,8.2),(83.6,8.2),(83.6,16.4),(85,16.4)],INK)
    ax.text(57,8.7,r"domestic delivery after transit $\tau_j$",fontsize=S,ha="center",va="bottom",color=INK)
    poly([(92,10.5),(92,14)],SLATE)
    poly([(78.5,24),(78.5,26.6),(91.8,26.6),(91.8,30)],AMB)
    ax.text(85.3,27.0,r"$\kappa_{jk}\,\varepsilon^{h}_{g,\mu(g)}$ per call",fontsize=S,color=AMB,ha="center",va="bottom")
    # legend, two rows
    def sw(x,y,ec,fc,dash,lab):
        ax.add_patch(FancyBboxPatch((x,y-0.75),3.2,1.5,boxstyle="round,pad=0,rounding_size=0.3",fc=fc,ec=ec,lw=0.9,ls=(0,(3,2)) if dash else "-")); ax.text(x+4,y,lab,fontsize=S,va="center",color=INK)
    def la(x,y,col,ls,lab):
        ax.add_patch(FancyArrowPatch((x,y),(x+4,y),arrowstyle="-|>",mutation_scale=6,lw=1.0,color=col,linestyle=ls)); ax.text(x+5,y,lab,fontsize=S,va="center",color=INK)
    y1=0.6
    sw(1,y1,SLATE,"white",True,"exogenous input"); sw(17,y1,NAVY,NAVY_F,False,"security state"); sw(32,y1,AMB,AMB_F,False,"gating constraint"); sw(49,y1,TEAL,TEAL_F,False,"logistics state")
    y2=-2.6
    la(1,y2,INK,"-","material flow"); la(17,y2,AMB,"-","capacity or exposure constraint"); la(44,y2,NAVY,(0,(4,2)),"information feedback")
    fig.savefig(FIG / "F1_schematic.svg", bbox_inches="tight", pad_inches=0.03)
    _save(fig, "F1_schematic")


# ---------------------------------------------------------------- F2 calibration
def f2_calibration():
    rc = pd.read_csv(TAB / "e1_route_calibration.csv", index_col=0)
    mt = pd.read_csv(TAB / "e1_monthly_timing.csv", index_col=0)
    names = {"West": "West Coast", "GulfEast": "Gulf to East", "PR": "Puerto Rico", "HI": "Hawaii", "AK": "Alaska", "Intra": "Intra-Gulf"}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.42))
    ax = axs[0]; y = np.arange(len(rc))[::-1]
    for yi, (r, row) in zip(y, rc.iterrows()):
        ax.plot([row.observed_kt / 1e3, row.model_R2_kt / 1e3], [yi, yi], color="#bbbbbb", lw=1.2, zorder=1)
    ax.scatter(rc.observed_kt / 1e3, y, s=22, color=COL["obs"], label="Observed (register)", zorder=3)
    ax.scatter(rc.model_R2_kt / 1e3, y, s=22, facecolor="white", edgecolor=COL["DTV"], lw=1.2, label="Model (initial design)", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in rc.index]); ax.set_xlabel("Waiver lift, 17 Mar-31 Jul 2026 (Mt)")
    ax.set_title("Route totals match except Gulf-East, served domestically first"); ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36)); ax.margins(x=0.06, y=0.08)
    _letter(ax, "a", dx=-0.32)
    ax = axs[1]; x = np.arange(len(mt))
    ax.plot(x, mt.obs / 1e3, "-o", color=COL["obs"], ms=3.5, lw=1.2, label="Observed")
    ax.plot(x, mt.model / 1e3, "--o", color=COL["DTV"], ms=3.5, lw=1.2, mfc="white", label="Model")
    ax.set_xticks(x); ax.set_xticklabels([pd.Period(p).strftime("%b") for p in mt.index]); ax.set_ylabel("Waiver lift (Mt per month)")
    r = np.corrcoef(mt.model, mt.obs)[0, 1]
    ax.set_title(f"Monthly timing follows the register (r = {r:.2f})"); ax.legend(loc="lower right"); ax.margins(0.06)
    _letter(ax, "b")
    _save(fig, "F2_calibration")


# ---------------------------------------------------------------- F3 main comparison
def f3_comparison():
    best = pd.read_csv(TAB / "e2_draws_best_dB.csv", index_col=0)
    A = pd.read_csv(TAB / "e2_stageA_central.csv"); Bb = pd.read_csv(TAB / "e2_stageB_central.csv")
    order = ["B0-doc", "B1", "B2", "B3", "B3h", "B4", "DTV"]
    cmap = {"B0-doc": COL["B0"], "B1": COL["B1"], "B2": COL["B2"], "B3": COL["B3"], "B3h": COL["B3-hybrid"], "B4": COL["B4"], "DTV": COL["DTV"]}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 70 * MM), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.36))
    ax = axs[0]; rng = np.random.default_rng(3)
    for i, d in enumerate(order):
        v = best[d].values / 1e3
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=9, color=cmap[d], alpha=0.85, lw=0.3,
                   edgecolor="#555555" if d in ("B0-doc", "B1") else "none", zorder=2)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color="black" if d != "DTV" else COL["DTV"], lw=1.8, zorder=3)
    for k in best.index:                      # paired lines B3 -> DTV
        ax.plot([order.index("B3"), order.index("DTV")], [best.loc[k, "B3"] / 1e3, best.loc[k, "DTV"] / 1e3], color="#d0d0d0", lw=0.4, zorder=1)
    ax.set_yscale("log"); ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["B0\nuniform", "B1\nflag rule", "B2\nstatic\nprediction", "B3\npredict-\nthen-\noptimise", "B3-\nhybrid", "B4\nbest\nstatic", "DTV"])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}k"))
    ax.set_ylabel("Vetting-attributable backlog (kt-days)")
    g = (best.B3 - best.DTV) / best.B3
    ax.set_title(f"DTV beats B3 in 30 of 30 draws (median gain {g.median():.0%})")
    ax.text(0.01, 0.02, "lower = better; 30 paired draws, caps 0.4 x no-vetting exposure", transform=ax.transAxes, fontsize=6.5, color="#555555")
    _letter(ax, "a", dx=-0.13)
    ax = axs[1]
    R2B = float(A[(A.tag == "named") & (A.policy.str.contains("'R2'"))].cum_backlog.iloc[0])
    A["dB"] = A.cum_backlog - R2B; Bb["dB"] = Bb.cum_backlog - R2B
    caps = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    series = {d: A[A.tag == d].groupby("cap").dB.min() for d in ["B0-doc", "B1", "B2", "B3", "B3h", "B4"]}
    series["DTV"] = Bb[Bb.tag == "DTV"].groupby("cap").dB.min()
    for d in ["B0-doc", "B1", "B2", "B3", "B3h", "B4", "DTV"]:
        s = series[d].reindex(caps).dropna()
        ax.plot(s.index * 100, s.values / 1e3, "-o", ms=3 if d != "DTV" else 4, lw=0.9 if d != "DTV" else 1.8,
                color=cmap[d], mec="#555555" if d in ("B0-doc", "B1") else cmap[d], zorder=3 if d == "DTV" else 2)
        ax.plot([], [], "-o", ms=3, color=cmap[d], mec="#555555" if d in ("B0-doc", "B1") else cmap[d], lw=1.8 if d == "DTV" else 0.9,
                label={"B0-doc": "B0", "B3h": "B3-hybrid"}.get(d, d))
    ax.legend(loc="upper left", ncol=2, handlelength=1.6, columnspacing=0.8)
    ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("Exposure cap (% of no-vetting exposure)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}k"))
    ax.set_ylabel("Vetting-attributable backlog (kt-days)")
    ax.set_title("Tighter caps raise backlog for every design (central case)")
    _letter(ax, "b", dx=-0.17)
    _save(fig, "F3_comparison")


# ---------------------------------------------------------------- F4 vetting bottleneck (H1)
def f4_bottleneck():
    tz = pd.read_csv(TAB / "e3_theta_by_zone.csv"); sw = pd.read_csv(TAB / "e3b_rshare_sweep.csv").set_index("r_share")
    p3 = pd.read_csv(TAB / "e3b_p3_threshold.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 64 * MM), gridspec_kw=dict(width_ratios=[0.9, 1.2], wspace=0.38))
    ax = axs[0]; y = np.arange(len(tz))[::-1]
    ax.barh(y + 0.18, tz["DTV cap0.4"], height=0.34, color=COL["DTV"], label="DTV, cap 0.4")
    ax.barh(y - 0.18, tz["R4-T uncapped"], height=0.34, color=COL["R4-T"], label="R4-T (no cap)")
    ax.set_yticks(y); ax.set_yticklabels([z if z != "Other" else "Other U.S. zones" for z in tz.zone]); ax.set_xlabel("Mean shadow price of an inspector-hour\n(USD million per inspector-hour)")
    ax.set_title("Inspector-hours are scarce on the Texas coast"); ax.legend(loc="lower right"); ax.margins(x=0.05)
    _letter(ax, "a", dx=-0.45)
    ax = axs[1]
    for d, lab in [("B3", "B3"), ("B3h", "B3-hybrid"), ("B4", "B4"), ("DTV", "DTV")]:
        c = {"B3h": COL["B3-hybrid"]}.get(d, COL.get(d))
        ax.plot(sw.index * 100, sw[d] / 1e3, "-o", color=c, ms=3 if d != "DTV" else 4, lw=1.8 if d == "DTV" else 0.9)
        ax.text(sw.index[-1] * 100 * 1.08, sw[d].iloc[-1] / 1e3 * {"B3": 1.13, "B3h": 0.9}.get(d, 1.0), lab, color=c, fontsize=6.5, va="center")
    rstar = float(p3[(p3.design == "DTV-base")].r_star_peak.iloc[0])
    ax.axvline(rstar * 100, color="#777777", lw=0.7, ls=":")
    ax.text(rstar * 100 * 1.05, ax.get_ylim()[1] * 0.6 if False else 700, f"peak load =\ncapacity ({rstar:.0%})", fontsize=6.5, color="#555555")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.4, 40)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.set_yticks([50e-0, 100, 200, 500])
    ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}k"))
    ax.set_xlabel("Redeployable share of national PSC inspector-hours (%)"); ax.set_ylabel("Vetting-attributable backlog (kt-days)")
    ax.set_title("Backlog rises steeply once screening load exceeds capacity")
    _letter(ax, "b", dx=-0.16)
    _save(fig, "F4_bottleneck")


# ---------------------------------------------------------------- F5 Env-S region map (synthetic)
def f5_envS():
    S = pd.read_csv(TAB / "e2b_envS_all.csv")
    b4 = S[S.tag == "S-B4"].groupby(["life_scale", "r_share"]).dB.min()
    dt = S[S.tag == "S-DTV"].groupby(["life_scale", "r_share", "inform"]).dB.min()
    fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM), sharey=True, gridspec_kw=dict(wspace=0.08))
    lifes, rs = [0.5, 1.0, 2.0], [0.01, 0.05, 0.2]
    vmax = 160
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for ax, inf in zip(axs, [0.0, 0.5, 1.0]):
        M = np.array([[(b4[(l, r)] - dt[(l, r, inf)]) / 1e3 for r in rs] for l in lifes])
        im = ax.imshow(M, cmap="RdBu", norm=norm, origin="lower", aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(["1%", "5%", "20%"]); ax.set_xlabel("Inspector-hour share")
        ax.set_title({0.0: "Noise signal (informativeness 0)", 0.5: "Informativeness 0.5", 1.0: "Informativeness 1"}[inf])
    axs[0].set_yticks(range(3)); axs[0].set_yticklabels(["x0.5", "x1", "x2"]); axs[0].set_ylabel("Vessel lifetime\n(repeat-voyage share)")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Backlog saved by DTV vs B4\n(thousand kt-days; 0 = no gain)")
    fig.text(0.0, 1.0, "Env-S (synthetic stress test)", fontsize=7, color="#555555", va="bottom")
    for ax, l in zip(axs, "abc"):
        _letter(ax, l, dx=-0.08 if l != "a" else -0.30)
    _save(fig, "F5_envS_regions")


# ---------------------------------------------------------------- F6 mode map (P2)
def f6_modes():
    A = pd.read_csv(TAB / "e2_stageA_central.csv"); p2 = pd.read_csv(TAB / "e4_p2_rule_central.csv")
    A["pname"] = A.policy.map(lambda p: ast.literal_eval(p).get("name"))
    caps = [0.7, 0.5, 0.4, 0.3, 0.2]
    rungs = {"reg": 0, "doc": 1, "docp": 2, "ex": 3}
    lab = {"reg": "registry", "doc": "documentary", "docp": "doc+ attest.", "ex": "examination"}
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 58 * MM))
    groups = ["G", "AT", "AN"]
    cm = mpl.colors.ListedColormap(["#f0f0f0", "#c6dbef", "#6baed6", "#1f4e79"])
    M = np.zeros((3, len(caps)))
    for j, c in enumerate(caps):
        b = A[(A.tag == "B4") & (A.cap == c)].sort_values("obj").pname.iloc[0]
        mu = dict(x.split(":") for x in b.split("|"))
        for i, g in enumerate(groups):
            M[i, j] = rungs[mu[g]]
            ax.text(j, i, lab[mu[g]], ha="center", va="center", fontsize=6, color="white" if M[i, j] >= 3 else "black")
            if g in ("AT", "AN"):
                ok = p2[(p2.cap == c) & (p2.group == g) & (p2.reference == "optimum")].match
                if len(ok) and not bool(ok.iloc[0]):
                    ax.text(j + 0.42, i + 0.30, "*", ha="right", va="center", fontsize=9, color="#d95f02")
    ax.imshow(M, cmap=cm, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(caps))); ax.set_xticklabels([f"{int(c * 100)}%" for c in caps]); ax.set_xlabel("Exposure cap (% of no-vetting exposure)")
    ax.set_yticks(range(3)); ax.set_yticklabels(["Green", "Amber, triggered", "Amber, other"])
    ax.set_title("Examination moves down the risk groups as protection tightens")
    ax.text(0, -0.75 + 3.95, "* P2 pricing rule disagrees with enumeration", fontsize=6, color="#d95f02", va="top", transform=ax.transData)
    _save(fig, "F6_mode_map")


def all_figures():
    _style()
    for f in (f1_schematic, f2_calibration, f3_comparison, f4_bottleneck, f5_envS, f6_modes):
        f()


if __name__ == "__main__":
    all_figures()
