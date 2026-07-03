# Linear encoding model (GLM) pipeline

## What this does 

For each recording electrode, this fits a model that **predicts the brain signal (the LFP) from
what was happening in the task and the animal's behaviour**. The "what was happening" includes:
whether the trial was correct, the difficulty, movement, cognitive state, the timing of trial /
stimulus / reward / block onsets, reaction time, eye movements (saccades), and what was inside that
electrode's receptive field (target, distractor, sky, mountain, grass).

The model is a **ridge regression** (a linear model with a penalty that keeps it stable). It's fit
**one electrode at a time**. Once fit, we ask, for each thing in the model: *how much does it
actually help predict the brain signal, and is that more than we'd expect by chance?*

Everything is organised by **session** (a recording day) and **channel** (an electrode). Results
live under the folder set in `glm_config.py`:
`Results/Linear_encoding_model/all_regressors/100Hz/<session>/channel<ch>_regressors/` (fit files in
its `results/` subfolder). Plots go to the mirror `plots/...` tree.

You can run this **two ways**: each session on its own, or **all sessions pooled** into one combined
fit per electrode. See "How to run it" below.

---

## What you get out of it

For every electrode and every regressor "family" (e.g. `grass_in_RF`, `stim_onset`, `state`):

- **full R²** — how much of the brain signal the *whole* model explains. Higher = better fit.
- **dR2 (unique contribution)** — how much *extra* a family explains that nothing else can. We drop
  that family, refit, and see how much the (cross-validated) prediction drops. If a family repeats
  information already in the model, its dR2 is ~0 even if it looks related — that's "redundant."
- **trace variance** — how much a family swings the predicted signal on its own (a rough, in-sample
  size measure, not corrected for redundancy).
- **significance (permutation test)** — is the effect real or could it happen by chance? We shuffle
  the brain signal 1000× (circular time-shift) to build a "by chance" baseline, and a family counts
  as significant if its real effect beats that baseline. A family can be **significant but tiny**
  (real, reliable, but explains almost nothing unique) — always read dR2 and significance together.
- **dummy** — a deliberately meaningless control regressor. It's the baseline: anything real should
  clearly beat the `dummy` column.

In the array-level overview plot, each cell is the **mean dR2 across the array's channels**, and a
**dot** marks families that pass a proper **array-level permutation test** (significant across the
whole array, not just one channel).

---

## How to run it

Everything runs in the **warping** conda env

`ArrayRun.py` is the one command that runs the whole pipeline. You set a few options at the top of
the file, then run it. It walks the electrodes one array at a time and submits each step to the
cluster.

### Option A — each session on its own (individual runs)

Fits every session separately (one model per session per electrode).

1. In `ArrayRun.py` set:
   ```python
   SESSIONS = ['20230203', '20230208', '20230209', '20230213', '20230214']   # your session names
   POOL_SESSIONS = False
   ```
2. Run it:
   ```
   python ArrayRun.py
   ```
3. Plot one session:
   ```
   python PlotArraySummary.py 20230214
   ```

### Option B — all sessions pooled together (one combined fit per electrode)

Concatenates all the sessions into a single fit per electrode — more data, one answer per electrode.
Under the hood it z-scores each session, adds a "session" nuisance regressor to absorb day-to-day
offsets, and uses session-aware cross-validation + permutations so nothing leaks across the seam
between sessions.

1. In `ArrayRun.py` set:
   ```python
   SESSIONS = ['20230203', '20230208', '20230209', '20230213', '20230214']   # your session names
   POOL_SESSIONS = True
   ```
2. Run it (this builds each session, then automatically concatenates and fits the combined model):
   ```
   python ArrayRun.py
   ```
   The combined results are written under a folder named `pooled_<first>_<last>`, i.e.
   `pooled_20230203_20230214` or whatever else you choose.
3. Plot the pooled result — just pass that folder name:
   ```
   python PlotArraySummary.py pooled_20230203_20230214
   ```

> Tip: on a fresh run, keep the full pipeline in `STEPS`. If some sessions are already built, you can
> trim `STEPS` (e.g. only build the missing sessions' inputs) to avoid recomputing them.

### Which extra scripts work on pooled data

The pooled fit reuses the same file layout, so most analysis scripts "just work" on it — point their
`SESSIONS` at the `pooled_...` folder:

| Script | Individual | Pooled |
|---|---|---|
| `FittingGLM.py`, `RegressorContributions.py` (fit + stats) | ✅ | ✅ (run automatically by `ArrayRun`) |
| `ExamineFit.py` (observed vs predicted) | ✅ | ✅ |
| `AnalyzeRegressorContributions.py` (kernel / trace plots) | ✅ | ✅ |
| `PlotArraySummary.py` (array overview) | ✅ | ✅ |
| `RedundancySubsample.py` + `AnalyzeRedundancySubsample.py` (redundancy check) | ✅ | ❌ **individual sessions only** |

The redundancy scripts need the *raw* design matrix, which the pooled build doesn't save — so run
the redundancy check on individual sessions, not the pooled fit. We already removed the redundant 
regressors we could afford to loose, the left over once are essential even if they are redundant.

---

## Under the hood: the full step list

`ArrayRun.py` runs these in order (each reads the previous step's output). Steps 1–3 run per real
session; for a pooled run the fit steps then run on the concatenated design.

1.  `GLM_input/Regressors.py`   → all regressor `.npz` + `full_mask_reg.npz`
2.  `GLM_input/NeuralData.py`   → `neural_data.npz` (the brain signal; needs step 1)
3.  `GLM_fitting/DesignMatrix.py` → `..._dMatRaw_*` and `..._dMatProcessed_*` (the model's inputs)
    - *(pooled only)* `GLM_fitting/AssemblePooled.py` → concatenates the sessions into the pooled design
4.  `GLM_fitting/FittingGLM.py`   → `..._<n>samples.pkl` (the fitted model + R²)
5.  `GLM_fitting/RegressorContributions.py` → `..._contributions.npz` (per-family dR2, trace
    variance, and the permutation-test significance + saved null distributions)

Plotting / inspection (light — no cluster needed):

- `GLM_fitting/ExamineFit.py` → observed-vs-predicted trace PDF
- `GLM_fitting/AnalyzeRegressorContributions.py` → per-channel kernel / trace / summary PDFs
- `PlotArraySummary.py` → per-array heatmaps + the array-level overview

Redundancy check (individual sessions only, optional but recommended before trusting a fit):

- `GLM_fitting/RedundancySubsample.py` → per-channel `..._redundancy_<method>.jsonl`
- `GLM_fitting/AnalyzeRedundancySubsample.py` → redundancy PDFs + Jaccard table

### The permutation test (how "significance" is decided)

`RegressorContributions.py` runs a **circular-shift permutation test** (`N_PERM`, default 1000): each
shuffle slides the brain signal by a large random offset and refits the full model, and per family a
max-statistic across its time-lags gives a chance threshold. Because the shuffle keeps every other
regressor in place, significance means "real *given everything else in the model*" — so a lag can be
flat here yet show up in a simpler marginal average. The shuffles use a **fixed seed so every channel
of an array gets the same shuffles**, which lets `PlotArraySummary.py` build the honest array-level
test by pooling the per-channel null distributions. Set `N_PERM=0` to skip.

---

## How the steps reach the cluster (ACME)

The heavy steps (1, 2, 3, assemble, 4, 5) use **ACME**: you run the script (or `ArrayRun.py`) in the
warping env and acme fans the channels out to SLURM workers itself — one worker per channel. Tune
`SLURM_PARTITION` / `MAX_WORKERS` / `MEM_PER_WORKER` at the top of each script.

`FittingGLM.py` (step 4) is the heavy one: the design is huge and gets densified, so each worker
needs a **big-memory, multi-core** node. The plotting scripts are light and run on the login node.

## Support modules (not run directly)

- `reg.py`   – `makeDesignMatrix_noTrials` (event → time-lagged kernel) and `ridge_MML` (ridge penalty).
- `utils.py` – `makeLogical` (event times → boolean vector).
