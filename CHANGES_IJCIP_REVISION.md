# Changes for the IJCIP revision (17 Nov 2026)

## Made in this package
- `sgw/figures.py`: Fig. 1 lane backgrounds removed; Fig. 2a title ("Most route totals match; Gulf-East goes domestic first"); Fig. 3a top axis in priority-weighted kt-days per inspector-hour; output files renumbered to the manuscript order (F3 bottleneck, F4 comparison, F5 mode map, F6 Env-S).
- `outputs/figures/`: F1 rebuilt without lane backgrounds; F3-F6 files renamed (images unchanged until `run_all.py` is re-run).
- `outputs/tables/parameter_table.csv`: `eps_base` row now gives the vessel-level baselines used in every reported run (AT cond 1.44, AN 0.92, R 1.03; UG inherits AN), matching manuscript Table 2.
- `run_all.py`: manuscript build step runs only if `manuscript/` is present.
- `README.md`: layout and notes updated (figure numbering, `notes/`, `docs/`, fluid approximation, enumeration, dual units, E8 network).
- `docs/NOTATION.md`: manuscript-to-code symbol map.
- `scripts/make_manifest.py`: rebuilds `MANIFEST.sha256`; refuses to run unless `PREREG.md` matches its pre-registered hash.

## Author actions before release
1. Restore `PREREG.md` byte-for-byte from the 13 Nov 2026 original (SHA-256 ef5253d2...6614).
2. Restore `notes/` (at least `literature_log.csv` and the search protocol), which the manifest lists and manuscript Section 2.4 now cites.
3. Re-run `python run_all.py` on a machine with Arial to regenerate F2-F6.
4. Run `python scripts/make_manifest.py`, commit, tag a release and archive it with a DOI.
