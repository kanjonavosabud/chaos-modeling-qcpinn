# Lorenz63 Classical + DV Training Report

## Scope
- Focused only on `Classical` and `DV` solvers as requested.
- Ran repeated 300-epoch sweeps for three curriculum settings after each major edit.
- Logged every run under `models/lorenz63/...` and raw benchmark logs/CSVs under `testing_checkpoints/lorenz63_benchmark/`.

## Experiment Matrix (applied at each phase)
- Solvers: `Classical`, `DV`
- Curriculum variants:
  - `no_curriculum` -> single-stage on `[0, 2.0]`
  - `curr_short` -> staged `0.5,1.0,2.0`
  - `curr_full` -> staged `0.2,0.5,1.0,1.5,2.0`
- Epoch budget: `300` **per stage** (so curriculum runs consume more total optimizer steps).

## P0 Baseline (Normalization + Curriculum only)

| Solver | Curriculum | Rel L2 % | x% | y% | z% | Max |residual| | Last logged loss | Run path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Classical | no_curriculum | 45.4500 | 78.7500 | 80.8800 | 31.3800 | 68.00 | 2.002e+01 | `models/lorenz63/classical/2026-05-06_13-44-10-024919` |
| Classical | curr_short | 6.8920 | 9.0220 | 11.3800 | 5.6690 | 16.14 | 7.283e-01 | `models/lorenz63/classical/2026-05-06_13-44-13-600421` |
| Classical | curr_full | 6.8800 | 9.3740 | 11.2400 | 5.6320 | 13.89 | 6.250e-01 | `models/lorenz63/classical/2026-05-06_13-44-17-932465` |
| Dv | no_curriculum | 83.8500 | 98.0500 | 97.6700 | 79.9700 | 21.31 | 2.484e+01 | `models/lorenz63/dv/2026-05-06_13-44-23-160692` |
| Dv | curr_short | 9.6970 | 12.6500 | 18.6100 | 7.1350 | 34.02 | 4.033e+00 | `models/lorenz63/dv/2026-05-06_13-44-34-605990` |
| Dv | curr_full | 8.4180 | 10.9800 | 15.9300 | 6.2780 | 28.47 | 2.239e+00 | `models/lorenz63/dv/2026-05-06_13-45-03-177081` |

## P1 After Edit #3 (Fourier Features)

| Solver | Curriculum | Rel L2 % | x% | y% | z% | Max |residual| | Last logged loss | Run path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Classical | no_curriculum | 49.5900 | 81.7000 | 82.9200 | 37.1200 | 42.55 | 1.318e+01 | `models/lorenz63/classical/2026-05-06_13-46-38-574855` |
| Classical | curr_short | 5.3630 | 8.3220 | 9.4270 | 3.9810 | 9.60 | 5.425e-01 | `models/lorenz63/classical/2026-05-06_13-46-42-110349` |
| Classical | curr_full | 4.0170 | 5.3080 | 6.8470 | 3.2330 | 4.61 | 2.014e-01 | `models/lorenz63/classical/2026-05-06_13-46-46-733743` |
| Dv | no_curriculum | 83.9900 | 99.1500 | 96.4600 | 80.2300 | 28.74 | 2.752e+01 | `models/lorenz63/dv/2026-05-06_13-46-52-521918` |
| Dv | curr_short | 5.8780 | 7.7230 | 9.6300 | 4.8530 | 10.76 | 5.957e-01 | `models/lorenz63/dv/2026-05-06_13-47-04-362995` |
| Dv | curr_full | 4.0070 | 5.7680 | 6.5670 | 3.2170 | 22.11 | 4.593e-01 | `models/lorenz63/dv/2026-05-06_13-47-34-055161` |

## P2 After Edit #5 (Adaptive Loss: GradNorm)

| Solver | Curriculum | Rel L2 % | x% | y% | z% | Max |residual| | Last logged loss | Run path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Classical | no_curriculum | 1.0270 | 1.2580 | 2.0050 | 0.7570 | 82.84 | 4.182e-02 | `models/lorenz63/classical/2026-05-06_13-48-53-171313` |
| Classical | curr_short | 0.6976 | 1.0480 | 1.2510 | 0.5168 | 99.78 | 2.202e-02 | `models/lorenz63/classical/2026-05-06_13-48-56-894633` |
| Classical | curr_full | 0.6473 | 0.9813 | 1.3120 | 0.4179 | 59.65 | 1.861e-02 | `models/lorenz63/classical/2026-05-06_13-49-02-016185` |
| Dv | no_curriculum | 2.6600 | 5.3860 | 4.7620 | 1.5820 | 71.37 | 1.658e-01 | `models/lorenz63/dv/2026-05-06_13-49-08-587141` |
| Dv | curr_short | 1.6010 | 2.4750 | 2.6940 | 1.2290 | 40.81 | 6.118e-02 | `models/lorenz63/dv/2026-05-06_13-49-24-553157` |
| Dv | curr_full | 0.9629 | 2.1980 | 1.3990 | 0.6024 | 27.35 | 2.958e-02 | `models/lorenz63/dv/2026-05-06_13-50-05-103708` |

## P3 After Edit #6 (DV Re-uploading)

| Solver | Curriculum | Rel L2 % | x% | y% | z% | Max |residual| | Last logged loss | Run path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Classical | no_curriculum | 1.0270 | 1.2580 | 2.0050 | 0.7570 | 82.84 | 4.182e-02 | `models/lorenz63/classical/2026-05-06_13-51-55-088513` |
| Classical | curr_short | 0.6976 | 1.0480 | 1.2510 | 0.5168 | 99.78 | 2.202e-02 | `models/lorenz63/classical/2026-05-06_13-52-07-946704` |
| Classical | curr_full | 0.6473 | 0.9813 | 1.3120 | 0.4179 | 59.65 | 1.861e-02 | `models/lorenz63/classical/2026-05-06_13-52-25-486255` |
| Dv | no_curriculum | 2.6600 | 5.3860 | 4.7620 | 1.5820 | 71.37 | 1.658e-01 | `models/lorenz63/dv/2026-05-06_13-52-50-565449` |
| Dv | curr_short | 1.6010 | 2.4750 | 2.6940 | 1.2290 | 40.81 | 6.118e-02 | `models/lorenz63/dv/2026-05-06_13-53-39-990244` |
| Dv | curr_full | 0.9629 | 2.1980 | 1.3990 | 0.6024 | 27.35 | 2.958e-02 | `models/lorenz63/dv/2026-05-06_13-56-05-978097` |

## Iterative Findings by Edit

### Edit #1 (Normalization) and Edit #2 (Curriculum)
- Already integrated before this campaign; benchmark confirms curriculum is the dominant early lever.
- `Classical`: `no_curriculum` 45.450% -> `curr_full` 6.880% (improvement `84.86%`).
- `DV`: `no_curriculum` 83.850% -> `curr_full` 8.418% (improvement `89.96%`).

### Edit #3 (Fourier Features for time)
- `Classical` on `curr_full`: 6.880% -> 4.017% (additional `41.61%` gain).
- `DV` on `curr_full`: 8.418% -> 4.007% (additional `52.40%` gain).
- Practical interpretation: Fourier encoding materially improves optimization conditioning for `t -> u(t)` mapping.

### Edit #5 (Adaptive loss weighting, GradNorm)
- `Classical` on `curr_full`: 4.017% -> 0.647% (additional `83.89%` gain).
- `DV` on `curr_full`: 4.007% -> 0.963% (additional `75.97%` gain).
- Biggest jump in this campaign came from GradNorm balancing + curriculum + Fourier combined.

### Edit #6 (DV re-uploading)
- In the main matrix (`num_quantum_layers=1`), metrics are identical with/without re-uploading.
- Reason: re-uploading only has effect when there are at least 2 variational layers (no inter-layer gap when layers=1).

### Edit #4 (Multi-IC flow-map training) status
- Not implemented in this iteration; current architecture remains per-trajectory (`input_dim=1`, single IC per run).
- Recommended as next major structural milestone once current single-trajectory stack is stabilized.

## Creative Ablations (extra)

| Experiment | Rel L2 % | x% | y% | z% | Max |residual| | Run path |
|---|---:|---:|---:|---:|---:|---|
| classical_full_no_norm | 0.4383 | 0.7702 | 0.8207 | 0.2836 | 46.19 | `models/lorenz63/classical/2026-05-06_14-00-42-537710` |
| dv_full_no_norm | 7.4680 | 16.2700 | 16.3800 | 1.7560 | 161.30 | `models/lorenz63/dv/2026-05-06_14-01-03-340155` |
| dv_layers3_no_reupload | 1.8720 | 3.7900 | 3.3980 | 1.0930 | 38.45 | `models/lorenz63/dv/2026-05-06_14-02-38-454122` |
| dv_layers3_with_reupload | 1.7360 | 2.6690 | 3.4420 | 1.1460 | 46.85 | `models/lorenz63/dv/2026-05-06_14-04-48-837239` |

Observations:
- `classical_full_no_norm` reached lower Rel L2 than normalized run, but with much larger residual than best normalized classical run; this suggests better trajectory fitting but weaker physics consistency.
- `dv_full_no_norm` degraded strongly (Rel L2 up to ~7.47%), indicating DV is more sensitive to scale imbalance without normalization.
- Re-uploading test with `num_quantum_layers=3` improved DV Rel L2 modestly (`1.872% -> 1.736%`), validating the mechanism when depth > 1.

## Seed Stability (creative)

| Experiment | Seed | Rel L2 % | x% | y% | z% | Max |residual| | Run path |
|---|---:|---:|---:|---:|---:|---:|---|
| classical_best_norm | 1 | 0.6473 | 0.9813 | 1.3120 | 0.4179 | 59.65 | `models/lorenz63/classical/2026-05-06_14-08-32-365169` |
| dv_best_norm | 1 | 0.9629 | 2.1980 | 1.3990 | 0.6024 | 27.35 | `models/lorenz63/dv/2026-05-06_14-08-38-974764` |
| classical_best_norm | 2 | 0.5022 | 0.7792 | 1.1680 | 0.2353 | 111.30 | `models/lorenz63/classical/2026-05-06_14-09-45-809766` |
| dv_best_norm | 2 | 0.8197 | 1.7070 | 1.2680 | 0.5464 | 15.98 | `models/lorenz63/dv/2026-05-06_14-09-52-929300` |
| classical_best_norm | 3 | 0.8480 | 1.4210 | 1.8990 | 0.4092 | 93.31 | `models/lorenz63/classical/2026-05-06_14-11-01-734526` |
| dv_best_norm | 3 | 1.4140 | 2.7200 | 2.8870 | 0.7197 | 40.95 | `models/lorenz63/dv/2026-05-06_14-11-08-578549` |

- `classical_best_norm` Rel L2 mean=0.6658%, std=0.1418% across seeds 1..3.
- `dv_best_norm` Rel L2 mean=1.0655%, std=0.2532% across seeds 1..3.

## Final Ranking of What Helped Most (empirical)
1. Curriculum (`curr_full`)
2. Adaptive loss weighting (`gradnorm`)
3. Fourier features (`k=4` bands)
4. DV re-uploading (only beneficial when `num_quantum_layers > 1`)
5. Multi-IC training (not yet implemented, expected highest long-term generalization gain)

## Code Changes Applied in This Iteration
- Added `src/nn/FourierFeatures.py`.
- Updated `src/nn/ClassicalSolver.py` to support optional Fourier feature preprocessing.
- Updated `src/nn/DVPDESolver.py` to support optional Fourier feature preprocessing.
- Updated `src/trainer/lorenz63_train.py` with optional GradNorm adaptive weighting.
- Updated `src/nn/DVQuantumLayer.py` with optional inter-layer data re-uploading.
- Updated `src/trainer/lorenz63_hybrid_trainer.py` with new CLI flags:
  - `--fourier-features`, `--fourier-num-bands`
  - `--adaptive-loss {none,gradnorm}`
  - `--dv-reupload`
  - plus existing curriculum + safety flags

## Repro Commands (best-known settings from this report)
- Classical best-known (single-trajectory):
  - `venv/bin/python -m src.trainer.lorenz63_hybrid_trainer --solver Classical --epochs 300 --batch-size 128 --lr 5e-3 --fourier-features --fourier-num-bands 4 --adaptive-loss gradnorm --curriculum --curriculum-t1 0.2,0.5,1.0,1.5,2.0`
- DV best-known (single-trajectory, 1 quantum layer):
  - `venv/bin/python -m src.trainer.lorenz63_hybrid_trainer --solver DV --encoding angle --q-ansatz cascade --num-qubits 5 --num-quantum-layers 1 --epochs 300 --batch-size 64 --lr 5e-3 --fourier-features --fourier-num-bands 4 --adaptive-loss gradnorm --curriculum --curriculum-t1 0.2,0.5,1.0,1.5,2.0`
- DV re-uploading test (requires >1 quantum layer to matter):
  - `venv/bin/python -m src.trainer.lorenz63_hybrid_trainer --solver DV --encoding angle --q-ansatz cascade --num-qubits 5 --num-quantum-layers 3 --epochs 300 --batch-size 64 --lr 5e-3 --fourier-features --fourier-num-bands 4 --adaptive-loss gradnorm --curriculum --curriculum-t1 0.2,0.5,1.0,1.5,2.0 --dv-reupload`
