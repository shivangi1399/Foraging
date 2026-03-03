# Foraging Task Analysis

Analysis of neural and behavioral data from a leaf foraging task recorded in a dome VR environment. LFP recordings are from 6 electrode arrays (192 channels) spanning V1 and V4.

## Task

On each trial, two morphed visual stimuli are presented, and the subject must select one in the discrimination task, which models foraging. Trials vary in difficulty (morph ratio: easy 30/70, hard 49/51) and reward level.

## Analysis overview

### Preprocessing (`code/preprocessing/`)
- **cut_trials.py** — Segments continuous neural data into trials aligned to stimulus onset. Extracts trial info (trial number, reward, difficulty) from Unreal Engine log files and saves in Syncopy and MATLAB formats.
- **cosmos_data_matlab.m** — Converts Syncopy LFP data into FieldTrip format for artifact rejection.
- **artifact_rejection.m** — Visual artifact rejection on FieldTrip-formatted LFP.
- **load_cleaned_data.py** — Converts cleaned data back to Syncopy format. Loads artifact-rejected LFP data back into Python via Syncopy.

### Parameter analysis (`code/Parameter analysis/`)
- **trial_parameter_analysis.py** — Timelock analyses split by reward level and difficulty, plotted per array and channel.
- **spectra_sign.py** / **spectra_timelock.py** — Spectral and timelock significance analyses.
- **across_block_behavior.py** / **trial_history_analysis.py** — Behavioral analyses across blocks and trial history effects.
  
### Cognitive state analysis

Cognitive states are inferred from behavioral emissions (e.g., reaction times) using a hidden Markov model (external to this repo). The predicted state assignments are used to group trials for all downstream analyses.

#### Behavioral (`code/States_analysis/beh/`)
- **states_beh.py** — Characterizes predicted states: state durations, transition probabilities, trial outcomes (correct/incorrect/miss rates), RT distributions by state and difficulty, and state dynamics around block changes.
- **states_beh_stats.py** — Chi-squared/Fisher's exact tests for trial outcome proportions, Mann-Whitney U tests for pairwise RT comparisons (Bonferroni corrected).

#### LFP — assigned states (`code/States_analysis/lfp/assigned_state/`)
- **erp_spectra_stats.py** — Core state-dependent LFP analysis. Computes trial-wise ERPs, power spectra (2-100 Hz, multitaper), and FOOOF aperiodic-corrected residual spectra per state. Runs nonparametric permutation tests (1000 permutations, max/min correction) between state pairs at single-channel, array, and merged-array levels.
- **coherence.py** — Trial-wise inter-array LFP coherence per state with permutation-based significance testing across frequency.
- **state_lfp_fig.py** — Summary figures: ERP/spectra with significance masks, heatmaps, coherence comparisons, and frequency-band summaries.

#### LFP — state probability (`code/States_analysis/lfp/state_prob/`)
- **erp_spec_state_prob.ipynb** — ERP and spectral analysis using continuous state probabilities.

### RF-to-stimulus mapping (`code/RF_VR_mapping/RFarea_stim_collapse_time/`)

Determines whether each channel's receptive field overlaps with the visual stimuli on each trial and time point, accounting for gaze-dependent coordinate transformations, the monkey's position in the VR, and stimulus collapse artifacts. The pipeline runs in sequence:

1. **mapping_corners.py** — Parses stimulus positions from log files, computes stimulus corner coordinates in dome space, transforms to gaze-centered retinal then Cartesian coordinates using eye tracking alignment.
2. **prepare_RF.py** — Converts Gaussian RF fits (azimuth, elevation, radius) from spherical to Cartesian coordinates. Matches each session to the closest RF mapping session.
3. **corner_order.py** — Reorders stimulus corner coordinates to a consistent winding order for the projective transform.
4. **warping_stim.py** — Warps each stimulus image into its on-screen polygon at each time point using a projective transform on the thresholded stimulus outline. Saves warped outlines in HDF5.
5. **RFarea_stim_collapse.py** — Tests RF-stimulus overlap using Shapely geometric intersection. Detects ("collapsed") stimulus polygons and handles them accordingly. Saves per-channel, per-trial, per-time-point overlap results.
6. **RFoverlap_perc_collapse.py** — Plots the percentage of time each channel's RF overlaps with stimulus A vs B.
7. **RF_map_flow.py** — Documents the HDF5 data structure and time alignment between mapping and neural data.

## Shared functions (`code/functions/`)
- **eyetracking/** — Dome-to-retinal coordinate conversions, iRec eye tracker parsing, timestamp alignment.
- **preprocessing/** — Ephys alignment, trial segmentation, spike sorting, DLC video tracking.
- **rf_bar_mapping/** — RF mapping from bar stimuli with dome back-projection.
- **decision_points/** — Log file parsing, reaction time extraction, signal detection metrics.
- **unreal_logfile/** — Unreal Engine log file parser (`TextLog`).
- **convert_unreal_coordinates/** — Coordinate transforms between Unreal and spherical systems.
- **AutoGaussianGaborFits/** — MATLAB tools for Gaussian/Gabor surface fitting to RF data.

## Software (`software/`)
- **syncopy-matlab** — MATLAB interface for Syncopy data format.

## Environment

The conda environment is specified in `software/warping_environment.yml`.

## Data

Neural data (LFP in Syncopy format), behavioral log files, and intermediate results are stored separately and not included in this repository.
