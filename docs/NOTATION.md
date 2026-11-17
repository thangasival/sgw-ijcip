# Notation map: manuscript symbols and code names

The IJCIP revision renamed several symbols to remove collisions. Code names are unchanged; this table maps them.

| Manuscript symbol | Meaning | Code (`sgw/`) |
| :--- | :--- | :--- |
| $\eta_g$ (was $\lambda_g$) | Departure rate from the waiver trade, $1/\eta_g$ = availability (days) | `Env.lam[g]`; `lifetime` in `parameter_table.csv` |
| $\lambda^{h}$ | Dual price of exposure channel $h$ | `lam_crew`, `lam_own`, `lam_cond` in run records |
| $\psi_p$ (was $\omega_p$) | Zone share of national inspector-hours | `Env.zw` (from `data/derived/zone_weights.csv`) |
| $\zeta_{gp}$ | Zone share of group $g$'s requests | `Env.zshare` |
| $\omega_j$ | Continuity priority of destination $j$ | `omega` in `parameter_table.csv` |
| $L_j$ (was $\delta_j$) | Sea distance of route $j$ (nm) | `SD.DIST_NM` |
| $\gamma_0$ (was $r_0$) | Freight rate (USD per tonne-nm) | `Params.rate` |
| $r$ | Redeployable share of PSC inspector-hours | `Params.r_share` |
| $R_{pt}$ (was $R_p$) | Zone inspector-hour capacity per day, $r\,H^{\mathrm{PSC}}\psi_p$ | `Rz` in `model.py` |
| $H^{\mathrm{PSC}}$ | National PSC inspector-hours per day | `Params.psc_hours_day` |
| $\alpha^{h}_{m}$ (was $\rho^{h}_{m}$) | Residual factor of mode $m$ in channel $h$ | `eps_res` in `parameter_table.csv`; `MODE_RES` in `model.py` |
| $\beta^{h}_{g}$ | Group baseline risk (lifetime-weighted mean of vessel-level baselines) | `Env._group_base`, `vessel_truth` |
| $\rho = (u,\delta_G,\delta_A,\iota)$ | Re-tiering rule | `Policy.u`, `Policy.delta_g`, `Policy.delta_a`, `Policy.inform` |
| $Q_{jkt}$ (was $p_{jkt}$) | Pending requirement | pending-demand rows in `model.py` |
| $K^{\mathrm{dom}}$ (was $W^{\mathrm{dom}}$) | Spot Jones Act tanker deadweight (kt) | `env.fleet["dwt_spot_kt"]` |
| $W^{v}_{j}$ | Round-trip occupancy window (periods) | occupancy rows in `model.py` |
| $\chi^{D}_{s}$ (was $\pi^{D}_{s}$) | Scarcity rent of domestic occupancy | dual of `rows["dom"]` |
| $\pi_d$, $\pi_T$ | Daily and terminal backlog penalties | `Params.pen_d`, `Params.pen_T` |
| $\theta_{pt}$ | Shadow price of an inspector-hour (USD million) | `theta_zone` in run records |
| $\bar{\kappa}_c$, $\kappa_{jk}$ | Lift per vessel; terminal-consequence weight | `kbar`; `kappa` in `parameter_table.csv` |
| $d_m$, $h_m$ | Screening delay and inspector-hours of mode $m$ | `MODE_D`, `MODE_H` |

Objective values and duals are in USD million (`MUSD` in `model.py`). Dividing $\theta_{pt}$ by $\pi_d$ (USD 60 per t-day) gives the penalty-free equivalent in priority-weighted kt-days per inspector-hour, which is the top axis of Fig. 3a.
