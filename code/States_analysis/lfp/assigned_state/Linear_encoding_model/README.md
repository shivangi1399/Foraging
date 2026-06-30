# Linear encoding model (GLM) pipeline

LFP encoding GLM for one session/channel: predict the continuous LFP from task/behaviour
regressors (correct/wrong, difficulty, movement, cognitive state, trial/stim/reward/block
onsets, reaction time, saccades, and RF target/distractor/sky/mountain/grass).

Everything is per **session** + **channel**. Outputs live in:
`Results/states_analysis/states_lfp/all_trials/full_length/GLM/<session>/channel<ch>_regressors/`
with fit artifacts under that folder's `results/`.

## Execution order

GLM (run in this folder tree). Steps 3–5 loop over every `<session>/channel<ch>_regressors`
folder under the results dir automatically (auto-discovered; pin a subset via the `SESSIONS` /
`CHANNELS` constants at the top of each script):

1.  `GLM_input/Regressors.py`   -> all regressor `.npz` + `full_mask`/`full_mask_reg`
                                   (ACME: sessions looped, channels parallelised)
2.  `GLM_input/NeuralData.py`   -> `neural_data.npz`  (needs `full_mask_reg.npz` from step 1)
3.  `GLM_fitting/DesignMatrix.py` -> `..._dMatRaw_*` and `..._dMatProcessed_*`
                                   (ACME: channels parallelised, one worker each)
3a. `GLM_fitting/RedundancySubsample.py`       (ACME: channels parallelised, one worker each)
                                   -> per-channel `..._redundancy_<method>.jsonl`
3b. `GLM_fitting/AnalyzeRedundancySubsample.py` -> per-channel `..._redundancy_<method>.png` + Jaccard table
4.  `GLM_fitting/FittingGLM.py`   -> `..._<n>samples.pkl`  (submit with `FittingGLM.sh`)
5.  `GLM_fitting/ExamineFit.py`   -> observed-vs-predicted plot

Steps 3a–3b are the **required redundancy check**: they do NOT feed the fit directly, but you
inspect 3b before fitting and, if a regressor group is flagged redundant, drop/merge it (or raise
the `col_sum` threshold) in `DesignMatrix.py` and re-run step 3. DesignMatrix.py also prints a
quick exact-rank verdict (full-matrix Gram QR) every run; 3a/3b add the subsampling-stability
sweep across downscale factors.

Dependencies in one line: 0 -> 1 -> 2, then 1+2 -> 3 -> 3a -> 3b -> (act on it) -> 4 -> 5.
(Step 2 reads `full_mask_reg.npz` from step 1; step 3 reads step 1's regressors + step 2's
neural data; 3a reads step 3's raw matrix; step 4 reads step 3's processed matrix + neural data.)

## Support modules (not run directly)

- `reg.py`   – `makeDesignMatrix_noTrials` (event -> time-lagged kernel) and `ridge_MML` (ridge penalty).
- `utils.py` – `makeLogical` (event times -> boolean vector).

## How the steps reach the cluster

Two different mechanisms:

- **ACME (steps 1, 2, 3, 3a):** run the script directly in the warping env (`python <script>.py`)
  and acme fans the channels out to SLURM workers itself (`SLURM_PARTITION` / `MAX_WORKERS` /
  `MEM_PER_WORKER` at the top of each script). One worker per channel; each writes its own output,
  so there's no cross-channel contention. `RedundancySubsample.py` runs its whole downscale-factor
  sweep inside each channel's worker and truncates its jsonl per run (no external cleanup needed).
- **`sbatch FittingGLM.sh` (step 4):** the fit is a single heavy job, not channel-parallel — the
  design matrix is ~1.6M samples × thousands of columns and `ridge_MML` densifies it (`.toarray()`),
  so it needs a **big-memory, multi-core** node (`--mem` in the hundreds of GB, `--cpus-per-task=24`
  so `sklearn n_jobs=-1` uses all cores, `--time` in hours). The `.sh` pins the conda env and exports
  `OMP/MKL/OPENBLAS_NUM_THREADS` to match the allocation.

Steps 3b and 5 are light and run directly on the login node.

