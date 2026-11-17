# Security-gated waiver capacity model (SG-WCM) with dynamic tiered vetting

Replication package for "Dynamic Security-Gated Maritime Capacity: Adaptive Vessel Vetting for Critical Energy Infrastructure Resilience".

## Quick start

    python -m pip install -r requirements.txt     # Python 3.11
    python run_all.py                             # tests, G2 recomputation, figures (a few minutes)
    python run_all.py --rerun                     # also re-solve all ~1,370 linear programs (several hours)

`run_all.py` exits 0 when every step succeeds.

## Layout

| Path | Content |
| :--- | :--- |
| `sgw/` | Model code: `data.py` (Env-R inputs), `model.py` (SG-WCM with DTV; LP build and solve), `lp.py` (sparse LP builder on SciPy/HiGHS), `policies.py`, `baselines.py` (B0-B4), `theory.py` (P1 decomposition, P2 rule, finite differences), `signal.py` (H3a test), `synth.py` (Env-S), `experiments.py` (task runner), `figures.py`, `network.py`/`v1case.py`/`experiments_v1.py` (original six-scenario network) |
| `data/v7/` | Read-only public-data inputs (MARAD register extracts, USCG PSIX and PSC report, EIA imports, MARAD fleet list), with `MANIFEST.sha256` |
| `data/derived/` | Derived Env-R inputs (voyages, groups, demand, unloading, zone weights) |
| `PREREG.md` | Pre-registration of the G2 test (SHA-256 in `outputs/prereg_hash.txt`, written before any comparison run) |
| `gates.md` | Gate outcomes (G1 failed, G2 passed) and the deviation log |
| `outputs/runs/` | Task lists (`*_tasks.json`) and raw results (`*.jsonl`) of every LP solved |
| `outputs/tables/` | Result tables used in the manuscript |
| `outputs/figures/` | Figures (PDF, 600-dpi TIFF, PNG), numbered as in the manuscript: F1 schematic, F2 calibration, F3 bottleneck, F4 equal-protection comparison, F5 mode map, F6 Env-S |
| `notes/` | Literature-search protocol for Section 2.4: databases, search strings, search dates, screening criteria and the 289 screened records (`literature_log.csv`) |
| `docs/NOTATION.md` | Map from manuscript symbols to code names (symbols renamed in the IJCIP revision) |
| `tests/` | Dual checks, regime regression on the stylised 12-port network, voyage-list delay cross-check, DTV structure, B3 greedy toy test |

## Notes

- The SG-WCM is a deterministic fluid (expected-value) approximation: vessels, queues and voyages are continuous, and upgrades and trips enter at their expected rates.
- Policies and re-tiering rules are chosen by enumeration (24 static assignments x rule grid); each instance is a linear program.
- Objective values and duals are in USD million. Inspector-hour duals divided by the USD 60 per t-day penalty give priority-weighted kt-days per inspector-hour (Fig. 3a, top axis).
- All-hazards checks (E8): `sgw/v1case.py` defines the stylised 12-port, five-commodity network and its six scenarios. Route distances, destination shares and unloading capacities are reconstructed and documented in the module docstring; `tests/test_v1_regression.py` checks the regime pattern, not exact magnitudes.
- `MANIFEST.sha256` is rebuilt with `python scripts/make_manifest.py`, which refuses to run unless `PREREG.md` matches the pre-registered hash.
- Solver: HiGHS interior point with crossover through `scipy.optimize.linprog`. A central instance has 57,814 variables and solves in 9-25 s.
- Env-S results are synthetic and are kept separate from Env-R results in tables and figures.
- The register snapshot is the interim one of 9 October 2026; the post-VAR back-test (E1) is to be re-run on the final register after the 15 November 2026 loading deadline.
- Parallel runs use independent processes per shard (`scripts/run_tasks.py tasks.json out.jsonl k n`).
