# Gate record

## G1 - signal

Test: out-of-time persistence of section 501(c) filing lateness. Fit window: loads 17 Mar-31 May 2026 (114 dated records); test window: loads 1 Jun-16 Aug (166 records, of which 26 by 22 vessels with fit-window history).
Model: late rate by (flag tier, previous filing late), estimated on 17 consecutive-voyage pairs in the fit window and shrunk to the flag-tier rate. Baseline: flag-tier late rate.

Result: Brier skill score 0.003 (95% vessel-bootstrap CI -0.242 to 0.228). Late rate after a late fit-window filing 0.33 vs 0.22 after on-time filings (odds ratio 1.80, Fisher p = 1.00).

**Decision: G1 not passed (CI includes 0). H3 is dropped as a claim.** DTV is run with grade-D ranges for the re-tiering rule: u in [0.3, 0.8], delta in [0, 0.2], informativeness in [0, 1]. The paper reports H3a as a real-data finding that the 2026 register does not yet support signal-based re-tiering, and P4/Env-S map where an informative signal would matter.
Calibration facts kept as anchors: P(on-time | previous on-time) for non-QS21 vessels = 0.76 (n = 17); per-voyage late rates QS21 0.179 (n = 195), non-QS21 0.141 (n = 85). Flag tier does not separate filing behaviour.

## Deviation log (written before any E2 result was inspected)
- 13 Nov 2026, after PREREG.md and before the first E2 batch: exposure accounting now uses a vessel-level baseline risk that does not depend on the grouping a design uses (crew and ownership channels from the flag/ownership classification; condition channel from each vessel's shrunk PSIX deficiency rate, rescaled to the flag-based mean). Without this, B2/B3 groupings would change the risk of the same vessels by relabelling them. Group coefficients are lifetime-weighted means of member vessels. Effect on R2 (no vetting): E_cond 188.8 -> 196.0; E_crew and E_own unchanged.
- Request lead drawn as U[1, 3] days (continuous), as written in PREREG.md.
- VAR survey/decision time and vetting are serial in the LP (VAR adds to the delivery lag after clearance). A voyage-list accounting would treat them as parallel; the serial form is conservative for designs with VAR.
- "Best" static policy (B4), best B3-hybrid and best DTV (policy, rule) are selected by the primary metric (cumulative backlog) at the same cap, the same criterion for every design with a selection step. The LP objective (cost plus priority-weighted penalties) is used only inside each solve and for the P2 rule check.
- The pre-registered DTV rule grid has u >= 0.3, so it does not contain the static special case (u = delta = 0); DTV can therefore score below B4 where re-tiering does not pay. DTV results are reported as found; no u = 0 point is added after the fact.

## Gate G2 outcome
- 30 paired draws (seed 20261113), every channel capped at 40% of the draw's own R2 exposure.
- Relative gain over B3, g = (dB_B3 - dB_DTV)/dB_B3: median 0.576 (bootstrap 95% CI 0.541-0.718), 5th percentile 0.361; positive in 30/30 draws.
- **G2 PASSES** (median >= 0.10 and 5th percentile > 0).
- Secondary: DTV vs B4 median 0.269 (p5 0.157); DTV vs B3-hybrid median 0.262 (p5 -0.028; positive in 28/30). At central parameters B3-hybrid is 4% below DTV, so the hybrid result is reported alongside the primary test.
