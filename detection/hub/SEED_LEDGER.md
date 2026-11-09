# Seed ledger (detection hub)

Every held-out block is written here BEFORE it runs (`hub.eval.harness` refuses a block >= 2000000
that is not in this table). Machine-readable copy: `reports/seed_ledger.csv`.

Rules (`.claude/agents/collapse-detection.md`): tune only on dev seeds (scenarios 0.., shifts
500000..999999). 1000000-1999999 is burned. 2000000+ held-out, one run per frozen config hash,
never tune after seeing it. `thundersim export` defaults to 2000000, so blocks start at 2100000.

| block | date | config hash | git SHA read | purpose | result file |
|---|---|---|---|---|---|
| 0 | 2026-11-07 | (dev, many) | f3337e3 (origin/main), b123c40 (checkout) | dev tuning, scenarios | scratch only |
| 500000 | 2026-11-07 | (dev, many) | f3337e3 / b123c40 | dev shifts | scratch only |
| 2100000 | 2026-11-07 | cf8c7456b67e (frozen collapse_v1) | origin/main 979afef (firmware/backend/docs), checkout b123c40 (detection/models, detection/simulation) | ONE held-out run, written before running: seeds 2100000-2100019; (1) all 35 catalogue scenarios, arms A,B,C,D,E; (2) SYNC, SYNC-KNEEL variant apnea, arms B,D; (3) SYNC, SYNC-KNEEL variant slowstart, arms B,D; (4) X-DISMISS variant dismiss, arms B,D. fw_main, calibrated, fast transport | reports/collapse_v1.md section 6 |
| 2200000 | 2026-11-08 | cf8c7456b67e (frozen collapse_v1, unchanged) | HEAD 86faf2e; simulator pin f05639c99ccacc4635258cbc48aa5fecaf817fc790beb995fb3b071a3b8fd68a (working tree) | ONE held-out run on the NEW simulator, written before running: seeds 2200000-2200019; (1) all 35 catalogue scenarios, arms A,B,C,D,E; (2) SYNC, SYNC-KNEEL variant apnea, arms B,D; (3) SYNC, SYNC-KNEEL variant slowstart, arms B,D; (4) X-DISMISS variant dismiss, arms B,D. fw_main, calibrated, fast transport. Dev re-run on the same pin passed every bar first | reports/collapse_v1.md section 6b |

2026-11-07: config 60323b2f32be failed the dev shift guard (hub-path man-down alerts on shifts) and was
never frozen or run held-out. After the team approved the two man-down fixes, config cf8c7456b67e passed every
dev bar and was frozen; block 2100000 is used once for it (row above). A later change needs a new block
(next free: 2200000) and must be reported as such.

2026-11-08: the simulator's collapse and contact models changed after block 2100000 ran (falls.py,
site.py, uncommitted). The 2100000 results describe the old models; a re-run on the new simulator
uses block 2200000 and must be recorded here first.

2026-11-08 (GO from team lead): dev re-run of frozen cf8c7456b67e on the new simulator. Simulator pin
f05639c99ccacc4635258cbc48aa5fecaf817fc790beb995fb3b071a3b8fd68a (sha256 of the LC_ALL=C-sorted
`shasum -a 256` manifest of detection/simulation/thundersim/**/*.py + calibration.json, 38 files, working
tree, uncommitted), checkout HEAD 86faf2e. Dev seeds: scenarios 0-19, shifts 500000-500008.
