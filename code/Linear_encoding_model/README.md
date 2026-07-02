# Linear encoding model (GLM) pipeline

LFP encoding GLM per session/channel: predict the continuous LFP from task/behaviour
regressors (correct/wrong, difficulty, movement, cognitive state, trial/stim/reward/block
onsets, reaction time, saccades, and RF target/distractor/sky/mountain/grass).

Everything is per **session** + **channel**. Outputs live in:
`Results/states_analysis/states_lfp/all_trials/full_length/GLM/<session>/channel<ch>_regressors/`
with fit artifacts under that folder's `results/`. The light plotting steps (3b, 5, 6b) instead
write PDFs into the mirror **plots** tree:
`plots/states_lfp/all_trials/full_length/GLM/<session>/channel<ch>_regressors/results/`.

## Execution order

GLM (run in this folder tree). Steps 3–6 loop over every `<session>/channel<ch>_regressors`
folder under the results dir automatically (auto-discovered; pin a subset via the `SESSIONS` /
`CHANNELS` constants at the top of each script):

1.  `GLM_input/Regressors.py`   -> all regressor `.npz` + `full_mask`/`full_mask_reg`
                                   (ACME: sessions looped, channels parallelised)
2.  `GLM_input/NeuralData.py`   -> `neural_data.npz`  (needs `full_mask_reg.npz` from step 1)
3.  `GLM_fitting/DesignMatrix.py` -> `..._dMatRaw_*` and `..._dMatProcessed_*`
                                   (ACME: channels parallelised, one worker each)
3a. `GLM_fitting/RedundancySubsample.py`       (ACME: channels parallelised, one worker each)
                                   -> per-channel `..._redundancy_<method>.jsonl`
3b. `GLM_fitting/AnalyzeRedundancySubsample.py` -> per-channel `..._redundancy_<method>.pdf` + Jaccard table
4.  `GLM_fitting/FittingGLM.py`   -> `..._<n>samples.pkl`
                                   (ACME: channels parallelised, one worker each)
5.  `GLM_fitting/ExamineFit.py`   -> observed-vs-predicted PDF (in the plots tree)
6a. `GLM_fitting/RegressorContributions.py`        (ACME: channels parallelised, one worker each)
                                   -> per-channel `..._contributions.npz` (betas, per-family dR2 + trace
                                      variance, + per-lag significance: beta_sig / beta_thr / family_p)
6b. `GLM_fitting/AnalyzeRegressorContributions.py` -> per-channel `_kernels.pdf` / `_traces.pdf` / `_summary.pdf`
                                   (significant lags shaded on the kernels; significant families starred)

Steps 3a–3b are the **required redundancy check**: they do NOT feed the fit directly, but you
inspect 3b before fitting and, if a regressor group is flagged redundant, drop/merge it (or raise
the `col_sum` threshold) in `DesignMatrix.py` and re-run step 3. DesignMatrix.py also prints a
quick exact-rank verdict (full-matrix Gram QR) every run; 3a/3b add the subsampling-stability
sweep across downscale factors.

Step 6 is the **optional contribution analysis** (run after a fit exists). 6a reuses each fit's
`alphas` + sample count, refits to recover the weights — the step-4 pickle's `mdl` is unfitted
because `cross_val_predict` clones the estimator — and per regressor family computes its per-lag
kernel, its contribution-trace variance, and its unique `dR2` (cv-R^2 drop when that family's
columns are dropped). It also runs a **circular-shift permutation test** (`N_PERM`, default 1000):
each permutation circularly shifts the target by a large random offset and refits the full ridge,
and per family a max-statistic across its lags gives an FWER-controlled threshold (`beta_thr`), the
significant design columns (`beta_sig`), and a family-level p-value (`family_p`) — same permutation
logic as the saccade STA / RF-entry ETA. Because the null refits the *full* model, significance is
"unique given the other regressors", so a lag can be flat here yet clear in the marginal STA/ETA
(the `stim_onset` vs `target_in_RF` redundancy). Set `N_PERM=0` to skip. 6b plots all three and
shades the significant lags on the kernels (stars significant families on the summary).

Dependencies in one line: 0 -> 1 -> 2, then 1+2 -> 3 -> 3a -> 3b -> (act on it) -> 4 -> 5, and
optionally 4 -> 6a -> 6b. (Step 2 reads `full_mask_reg.npz` from step 1; step 3 reads step 1's
regressors + step 2's neural data; 3a reads step 3's raw matrix; step 4 reads step 3's processed
matrix + neural data; 6a reads step 4's pickle + step 3's processed matrix + neural data.)

## Support modules (not run directly)

- `reg.py`   – `makeDesignMatrix_noTrials` (event -> time-lagged kernel) and `ridge_MML` (ridge penalty).
- `utils.py` – `makeLogical` (event times -> boolean vector).

## How the steps reach the cluster

One mechanism for the heavy steps:

- **ACME (steps 1, 2, 3, 3a, 4, 6a):** run the script directly in the warping env
  (`python <script>.py`) and acme fans the channels out to SLURM workers itself
  (`SLURM_PARTITION` / `MAX_WORKERS` / `MEM_PER_WORKER` at the top of each script). One worker per
  channel; each writes its own output, so there's no cross-channel contention.
  `RedundancySubsample.py` runs its whole downscale-factor sweep inside each channel's worker and
  truncates its jsonl per run (no external cleanup needed).

  Step 4 (`FittingGLM.py`) is the heavy one: the design is ~1.6M samples × thousands of columns and
  `ridge_MML` densifies it (`.toarray()`), so each worker needs a **big-memory, multi-core** node —
  `MEM_PER_WORKER` in the hundreds of GB and a per-core-mem partition so `sklearn n_jobs=-1` gets
  several cores. This replaces the old single-job `FittingGLM.sh` (kept only as a reference for the
  resource request; the ACME path is now the way to run step 4).

Steps 3b, 5 and 6b are light and run directly on the login node (no acme).

