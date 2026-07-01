import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _matlab_length(x):
    """
    MATLAB length(x): max(size(x)) for arrays.
    For Python objects/lists, use len().
    """
    if x is None:
        return 0
    if isinstance(x, (list, tuple)):
        return len(x)
    try:
        a = np.asarray(x)
        if a.ndim == 0:
            return 1
        return int(max(a.shape))
    except Exception:
        return 1

def _index_like_matlab_vector(x, trials_bool):
    """
    MATLAB: x(trials_bool)
    Works for numpy arrays (including dtype=object) and python lists.
    """
    if isinstance(x, list):
        return [xi for xi, keep in zip(x, trials_bool) if keep]
    a = np.asarray(x)
    # ensure 1D behavior like a vector/cell array
    a = a.ravel()
    return a[trials_bool]

def selectBehaviorTrials(bhv, trials, nTrials=None):
    """
    Faithful translation of your MATLAB selectBehaviorTrials for the common case:
    trials is a logical vector (useIdx).
    Includes the crucial 'struct one layer deeper' behavior, so RawEvents.Trial is sliced.
    """

    # --- fieldnames ---
    if bhv is None:
        return bhv
    bFields = list(getattr(bhv, "_fieldnames", []))

    # --- nTrials handling (MATLAB: if isfield(bhv,'nTrials') ...) ---
    if hasattr(bhv, "nTrials"):
        bhv.nTrials = int(np.sum(np.asarray(bhv.nTrials)))
        nTrials = bhv.nTrials
    elif nTrials is None:
        raise ValueError("Need nTrials if bhv has no nTrials field")

    trials = np.asarray(trials, dtype=bool).ravel()

    # MATLAB: if length(trials) ~= nTrials → warning + truncate
    if trials.size != nTrials:
        warnings.warn("Trial index has different length as available trials in behavioral dataset")
        trials = trials[:nTrials]

    # MATLAB: if isfield(bhv,'nTrials') bhv.nTrials = sum(trials);
    if hasattr(bhv, "nTrials"):
        bhv.nTrials = int(np.sum(trials))

    # --- main loop over fields ---
    for fname in bFields:
        val = getattr(bhv, fname)

        # MATLAB: if ~any(ismember(size(val), length(trials))) ...
        # We'll treat "contains trial dimension" as: any dimension equals nTrials
        try:
            shape = np.shape(val)
            contains_trial_dim = any(d == nTrials for d in shape)
        except Exception:
            contains_trial_dim = False

        if not contains_trial_dim:
            # MATLAB: if isstruct(val) check one layer deeper
            if hasattr(val, "_fieldnames"):
                tFields = list(getattr(val, "_fieldnames", []))
                if len(tFields) > 0:
                    first_sub = tFields[0]
                    inner = getattr(val, first_sub)

                    # MATLAB: if length(inner) == length(trials) then slice it
                    if _matlab_length(inner) == trials.size:
                        setattr(val, first_sub, _index_like_matlab_vector(inner, trials))
                        setattr(bhv, fname, val)
                    else:
                        # carry over whole field (do nothing)
                        pass
            else:
                # carry over whole field (do nothing)
                pass

        else:
            # MATLAB: if isvector(val) -> val(trials)
            # else: highD matrix -> slice along trial dimension
            arr = np.asarray(val)

            is_vector = (arr.ndim == 1) or (arr.ndim == 2 and 1 in arr.shape)
            if is_vector:
                setattr(bhv, fname, arr.ravel()[trials])
            else:
                # find first axis with size == nTrials (MATLAB: cIdx = find(...))
                axis = [i for i, d in enumerate(arr.shape) if d == nTrials][0]
                indexer = [slice(None)] * arr.ndim
                indexer[axis] = trials
                setattr(bhv, fname, arr[tuple(indexer)])

    return bhv

def makeLogical(cIdx, vecLength):
    # vecOut = false(vecLength,1);
    vecOut = np.zeros(vecLength, dtype=bool)

    # cIdx = cIdx(~isnan(cIdx) & cIdx <= vecLength);
    cIdx = np.asarray(cIdx)
    cIdx = cIdx[~np.isnan(cIdx)]
    cIdx = cIdx[cIdx <= vecLength]

    # MATLAB indexing is 1-based
    # vecOut(cIdx) = true;
    # → subtract 1 for Python indexing
    vecOut[cIdx.astype(int) - 1] = True

    return vecOut


def cat2_cell(cell_list):
    """
    MATLAB: cat(2, cell{:})
    - ignores empty entries
    - concatenates contents horizontally into a 1D vector
    """
    out = []
    for x in cell_list:
        if x is None:
            continue
        # treat NaN as empty (if you used np.nan placeholders)
        if isinstance(x, float) and np.isnan(x):
            continue

        # scalar
        if np.isscalar(x):
            out.append(float(x))
        else:
            arr = np.asarray(x).ravel()
            # drop NaNs if present
            arr = arr[~np.isnan(arr)]
            out.extend(arr.astype(float).tolist())

    return np.asarray(out)