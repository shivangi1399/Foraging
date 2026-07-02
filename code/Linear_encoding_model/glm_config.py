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
