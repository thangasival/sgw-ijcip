# Pre-registration: E2 superiority test (gate G2)

before any E2 run. Its SHA-256 is recorded in outputs/prereg_hash.txt and cited in the paper.
Already observed before writing this file: the five named designs (R2, R3, R4, R4-D, R4-T) at central parameters with no exposure cap, and gate G1 (failed). No DTV-vs-B3 comparison has been run.

## Environment and protection level
- Env-R (calibrated 2026 network), flag-tier grouping, 2-day steps, central parameters as in outputs/tables/parameter_table.csv.
- Protection ceiling for the primary test: every exposure channel capped at 40% of its R2 (no-vetting) value: E^ch <= 0.4 x E^ch(R2), ch in {crew, own, cond}. The same caps apply to every baseline.
- Frontier (secondary): caps at 100%, 70%, 50%, 40%, 30% and 20% of R2.

## Comparators
- B0: uniform documentary screening (and a uniform-examination variant); Red denied.
- B1: fixed R4-T flag rule (companion design).
- B2: static risk prediction. Groups from a PSIX-based predicted deficiency rate (Poisson GLM on flag tier and gas carrier, offset by exams, with empirical-Bayes shrinkage of each vessel's own record); the top share equal to the flag-Amber share is Amber; R4-T-style modes; no re-tiering.
- B3 (critical comparator): predict-then-optimise inspection. With the B2 predictions, inspector-hours in each zone and step examine queued vessels in descending order of predicted deficiencies (maximising expected deficiencies detected); vessels not selected pass with a registry check. Continuity is not in the inspection objective. The supply LP is then solved with the resulting groups and modes under the same caps.
- B3-hybrid: B2/B3 grouping with modes chosen by the integrated LP (backlog in the objective).
- B4: best static policy (24 mode assignments, u = delta = 0).
- DTV: best (policy, re-tiering rule) with rule grid u in {0.3, 0.5, 0.8}, delta_G in {0, 0.05}, delta_A in {0, 0.1}, informativeness 1 (a separate run uses 0.5 and 0).

## Primary metric and margin
- Metric: vetting-attributable cumulative backlog, Delta B = B(design, capped) - B(R2, uncapped), in kt-days (unweighted). R2 backlog reflects tonnage limits that no vetting policy can change, so the comparison is on the part vetting can affect. The raw cumulative backlog is reported as well.
- Relative gain over B3: g = (Delta B(B3) - Delta B(DTV)) / Delta B(B3).
- **G2 passes if** the median of g over 30 parameter draws is at least 0.10 **and** its 5th percentile is above 0. The same draws are used for every design (paired).
- Parameter draws: uniform over r_share in [0.02, 0.10], mode delays and inspector-hours x U[0.7, 1.3], residual-risk factors x U[0.8, 1.2], s_dom in [0.02, 0.10], West requirement in [18.8, 34.8] Mbbl, request lead in [1, 3] days; seed 20261113.
- Secondary metrics: priority-weighted backlog, high-ESB backlog, inspection hours by zone, queue vessel-days, expected deficiencies detected, exposure by channel.