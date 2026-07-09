"""
=============================================================================================
Single source of truth for the GLM output tree + analysis sampling rate.
=============================================================================================
Every pipeline script imports RESULTS_DIR / PLOTS_DIR from here, and DesignMatrix.py imports
NATIVE_FS / DOWNSAMPLE_FACTOR from here, so the rate folder (`<frameRate>Hz`, e.g. `100Hz`) that
sits ABOVE the session/date can never drift out of sync with the DOWNSAMPLE_FACTOR that actually
produced the data.

To change the analysis rate, edit DOWNSAMPLE_FACTOR here ONLY -- the folder label follows:
    DOWNSAMPLE_FACTOR = 10 -> 100 Hz -> .../GLM/100Hz/<session>/...
    DOWNSAMPLE_FACTOR = 4  -> 250 Hz -> .../GLM/250Hz/<session>/...
(NATIVE_FS must be an exact multiple of the target rate.)
"""
import os

# --- analysis sampling rate (authoritative) ---
NATIVE_FS = 1000                              # Hz, rate the raw regressors/neural_data are saved at
DOWNSAMPLE_FACTOR = 10                        # design/target row rate = NATIVE_FS // DOWNSAMPLE_FACTOR
FRAME_RATE = NATIVE_FS // DOWNSAMPLE_FACTOR   # Hz, the analysis (design row) rate
RATE_DIR = f'{FRAME_RATE}Hz'                  # folder label inserted above the session/date

# --- output roots ---
# MODEL is the regressor-set variant (swap it to keep alternative models side by side).
MODEL = 'all_regressors'
_RESULTS_BASE = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/Linear_encoding_model/{MODEL}'
_PLOTS_BASE = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/Linear_encoding_model/{MODEL}'
RESULTS_DIR = os.path.join(_RESULTS_BASE, RATE_DIR)
PLOTS_DIR = os.path.join(_PLOTS_BASE, RATE_DIR)


# =============================================================================================
# Convenience runner: runs the pipeline end-to-end.
# =============================================================================================
# This block only executes when glm_config.py is run directly -- importing it still just loads 
# the config above, nothing runs.
#
# Each step is launched as a SEPARATE subprocess (so each keeps its own module-level config and
# __main__; ArrayRun also submits SLURM jobs via acme), in order, with the SAME python that ran
# this file (so it stays in the warping env). If a step fails, the run stops there.
#
# Toggle steps on/off below, then:   python glm_config.py
if __name__ == '__main__':
    import sys
    import subprocess

    # --- what to run (each True step runs after the previous one finishes) ---
    RUN_ARRAY_RUN      = True    # 1. ArrayRun.py           -> fit + contributions (SLURM/acme)
    RUN_PLOT_SUMMARY   = True    # 2. PlotArraySummary.py   -> per-array dR2 / significance heatmaps
    RUN_PLOT_KERNELS   = True    # 3. PlotPooledKernels.py  -> mean regressor kernels

    # Session the two plot scripts read. Match ArrayRun.POOLED_SESSION (the pooled fit folder).
    POOLED_SESSION = 'pooled_all'

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PY = sys.executable   # same interpreter (== warping env) that launched this script

    # (label, command) in dependency order; the plot scripts take the session as argv[1].
    _steps = []
    if RUN_ARRAY_RUN:
        _steps.append(('ArrayRun',         [_PY, 'ArrayRun.py']))
    if RUN_PLOT_SUMMARY:
        _steps.append(('PlotArraySummary', [_PY, 'PlotArraySummary.py', POOLED_SESSION]))
    if RUN_PLOT_KERNELS:
        _steps.append(('PlotPooledKernels', [_PY, 'PlotPooledKernels.py', POOLED_SESSION]))

    if not _steps:
        raise SystemExit('Nothing to run: all RUN_* toggles are False.')

    print(f'GLM pipeline runner: {len(_steps)} step(s) -> {[n for n, _ in _steps]}')
    for _i, (_name, _cmd) in enumerate(_steps, 1):
        print(f'\n================= STEP {_i}/{len(_steps)}: {_name} =================')
        print(f'  $ {" ".join(_cmd)}')
        _rc = subprocess.run(_cmd, cwd=_HERE).returncode
        if _rc != 0:
            raise SystemExit(f'STEP {_i} ({_name}) failed with exit code {_rc}; stopping here.')
        print(f'  [{_name}] done.')
    print('\nAll steps finished.')
