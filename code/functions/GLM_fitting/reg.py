import numpy as np
import warnings

from scipy import optimize
from sklearn.metrics import mean_squared_error, r2_score

def makeDesignMatrix_noTrials(events, eventType, regLabels, opts):
    """
    Python 1-to-1 translation of:

    [fullMat, eventIdx] = makeDesignMatrix_noTrials(events, eventType, regLabels, opts)

    events:      (nrTimes x nrRegs) binary/bool matrix
    eventType:   length nrRegs, values in {1,2,3}
    regLabels:   length nrRegs (strings)
    opts:        object with attributes OR dict with keys:
                preTrig, postTrig, sPostTime, mPreTime, mPostTime
    """

    # allow opts as dict or attribute-object
    def _get_opt(name):
        return opts[name] if isinstance(opts, dict) else getattr(opts, name)

    nrTimes = events.shape[0]
    nrRegs = len(eventType)

    # fullMat = cell(1,length(eventType)); eventIdx = cell(1,length(eventType));
    fullMat_cells = [None] * nrRegs
    eventIdx_cells = [None] * nrRegs

    # loop over regressor variables
    for iRegs in range(nrRegs):  # MATLAB: 1:nrRegs
        et = eventType[iRegs]

        # determine index for current event type
        if et == 1:
            # kernelIdx = -opts.preTrig : opts.postTrig;
            kernelIdx = np.arange(-_get_opt("preTrig"), _get_opt("postTrig") + 1, 1, dtype=int)
        elif et == 2:
            # kernelIdx = 0 : opts.sPostTime;
            kernelIdx = np.arange(0, _get_opt("sPostTime") + 1, 1, dtype=int)
        elif et == 3:
            # kernelIdx = -opts.mPreTime : opts.mPostTime;
            kernelIdx = np.arange(-_get_opt("mPreTime"), _get_opt("mPostTime") + 1, 1, dtype=int)
        else:
            raise ValueError("Unknown event type. Must be a value between 1 and 3.")

        # build design matrix
        nrCols = len(kernelIdx)
        trace = events[:, iRegs].astype(bool)             # trace = logical(events(:,iRegs));
        trace_idx = np.where(trace)[0] + 1                # MATLAB find(trace) (1-based indices)

        # cIdx = bsxfun(@plus, find(trace), kernelIdx);
        # -> (nEvents x nrCols) 1-based candidate indices
        cIdx = trace_idx[:, None] + kernelIdx[None, :]

        # fullMat{iRegs} = false(nrTimes, nrCols);
        mat = np.zeros((nrTimes, nrCols), dtype=bool)

        # for iCols = 1:nrCols
        for iCols in range(nrCols):
            useIdx = cIdx[:, iCols]  # 1-based
            # useIdx = useIdx(useIdx > 0 & useIdx <= nrTimes);
            useIdx = useIdx[(useIdx > 0) & (useIdx <= nrTimes)]
            # fullMat{iRegs}(useIdx, iCols) = true;
            # convert to 0-based for numpy indexing
            mat[useIdx - 1, iCols] = True

        # cIdx = sum(fullMat{iRegs},1) > 0; %don't use empty regressors
        keep_cols = np.sum(mat, axis=0) > 0

        # if sum(~cIdx) > 0 warning(...)
        removed = int(np.sum(~keep_cols))
        if removed > 0:
            warnings.warn(
                f"Removed {removed} empty regressors from design matrix of regressor {regLabels[iRegs]}."
            )

        # fullMat{iRegs} = fullMat{iRegs}(:,cIdx);
        mat = mat[:, keep_cols]

        # eventIdx{iRegs} = repmat(iRegs, sum(cIdx),1);
        # MATLAB iRegs is 1-based; keep that convention in eventIdx
        event_idx = np.full((int(np.sum(keep_cols)), 1), iRegs + 1, dtype=int)

        fullMat_cells[iRegs] = mat
        eventIdx_cells[iRegs] = event_idx

    # fullMat = cat(2,fullMat{:});
    fullMat = np.concatenate(fullMat_cells, axis=1) if nrRegs > 0 else np.zeros((nrTimes, 0), dtype=bool)

    # eventIdx = cat(1,eventIdx{:});
    eventIdx = np.concatenate(eventIdx_cells, axis=0) if nrRegs > 0 else np.zeros((0, 1), dtype=int)

    return fullMat, eventIdx



def ridge_MML(Y, X, adjust_betas = False, recenter = True, L = None, regress = True):
    """
    This is an implementation of Ridge regression with the Ridge parameter
    lambda determined using the fast algorithm of Karabatsos 2017 (see
    below). I also made some improvements, described below.

    Inputs are Y (the outcome variables) and X (the design matrix, aka the
    regressors). Y may be a matrix. X is a matrix with as many rows as Y, and
    should *not* include a column of ones.

    A separate value of lambda will be found for each column of Y.

    Outputs are the lambdas (the Ridge parameters, one per column of Y); the
    betas (the regression coefficients, again with columns corresponding to
    columns of Y); and a vector of logicals telling you whether fminbnd
    failed to converge for each column of y (this happens frequently).

    If lambdas is supplied, the optimization step is skipped and the betas
    are computed immediately. This obviously speeds things up a lot.


    TECHNICAL DETAILS:

    To allow for un-centered X and Y, it turns out you can simply avoid
    penalizing the intercept when performing the regression. However, no
    matter what it is important to put the columns of X on the same scale
    before performing the regression (though Matlab's ridge.m does not do
    this if you choose not to recenter). This rescaling step is undone in the
    betas that are returned, so the user does not have to worry about it. But
    this step substantially improves reconstruction quality.

    Improvements to the Karabatsos algorithm: as lambda gets large, local
    optima occur frequently. To combat this, I use two strategies. First,
    once we've passed lambda = 25, we stop using the fixed step size of 1/4
    and start using an adaptive step size: 1% of the current lambda. (This
    also speeds up execution time greatly for large lambdas.) Second, we add
    boxcar smoothing to our likelihood values, with a width of 7. We still
    end up hitting local minima, but the lambdas we find are much bigger and
    closer to the global optimum.

    Source: "Marginal maximum likelihood estimation methods for the tuning
    parameters of ridge, power ridge, and generalized ridge regression" by G
    Karabatsos, Communications in Statistics -- Simulation and Computation,
    2017. Page 6.
    http://www.tandfonline.com/doi/pdf/10.1080/03610918.2017.1321119

    Written by Matt Kaufman, 2018. mattkaufman@uchicago.edu
    
    Adapted to Python by Michael Sokoletsky, 2021
    """
    
    ## Optional arguments

    if L is None:
        compute_L = True
    else:
        compute_L = False

    ## If design matrix is a DataFrame, convert to a matrix

    X = np.array(X)

    ## Error checking

    if np.size(Y, 0) != np.size(X, 0):
        raise IndexError('Size mismatch')

    ## Ensure Y is zero-mean

    pY = np.size(Y, 1)

    X[np.isnan(X)] = 0

    ## Renorm (Z-score) if adjusting betas (and if input to this function was not renormed already)
    if adjust_betas:
        X_std = np.std(X, axis=0, ddof=1)
        X = np.divide(X, X_std)
    
    # Recenter if not already done and computing L
    if not recenter and compute_L:
        X_mean = np.mean(X, 0)
        X = np.subtract(X, X_mean)


    ## Optimize lambda

    if compute_L:

        ## SVD the predictors

        U, d, VH = np.linalg.svd(X, full_matrices=False)
        S = np.diag(d)
        V = VH.T.conj()

        ## Find the valid singular values of X, compute d and alpha

        n = np.size(X, 0)  # Observations
        p = np.size(V, 1)  # Predictors

        # Find the number of good singular values. Ensure numerical stability.
        q = np.sum(d.T > abs(np.spacing(U[0,0])) * np.arange(1,p+1))

        d2 = d ** 2

        # Equation 1
        # Eliminated the diag(1 ./ d2) term: it gets cancelled later and only adds
        # numerical instability (since later elements of d may be tiny).
        # alph = V' * X' * Y
        alph = S @ U.T @ Y
        alpha2 = alph ** 2

        ## Compute variance of y
        # In Equation 19, this is shown as y'y

        Y_var = np.sum(Y ** 2, 0)

        ## Compute the lambdasnp.

        L = np.full(pY,np.nan)

        convergence_failures = np.empty(pY, dtype=int)
        
        for i in range(pY):
            
            L[i], flag = ridge_MML_one_Y(q, d2, n, Y_var[i], alpha2[:, i])
            convergence_failures[i] = flag
        
    else:
        p = np.size(X, 1)




    # If requested, perform the actual regression

    if regress:
        if not recenter:
            betas = np.full((p + 1, pY),np.nan)

            # Augment X with a column of ones, to allow for a non-zero intercept
            # (offset). This is what we'll use for regression, without a penalty on
            # the intercept column.

            X = np.c_[np.ones(np.size(X,0)), X]

            XTX = X.T @ X

            # Prep penalty matrix    
            ep = np.identity(p + 1)
            ep[0,0] = 0 # No penalty for intercept column

            # For renorming the betas
            # The 1 is so we don't renorm the intercept column.
            # Note that the rescaling doesn't alter the intercept.
            if adjust_betas:
                renorm = np.insert(X_std, 0, 1)

        else:
            betas = np.full((p, pY), np.nan)

            # You would think you could compute X'X more efficiently as VSSV', but
            # this is numerically unstable and can alter results slightly. Oh well.
            # XTX = V * bsxfun(@times, V', d2)

            XTX = X.T @ X

            # Prep penalty matrix
            ep = np.identity(p)

            if adjust_betas:
                # For renorming the betas
                renorm = X_std.T
            
        # Compute X' * Y all at once, again for speed
        XTY = X.T @ Y

        # Compute betas for renormed X
        if hasattr(L, "__len__"):
            for i in range(0, pY):
                betas[:, i] = np.linalg.solve(XTX + L[i] * ep, XTY[:, i])
        else:
            betas = np.linalg.solve(XTX + L * ep, XTY)
                
        if adjust_betas:
            # Adjust betas to account for renorming.
            betas = np.divide(betas.T, renorm).T

        betas[np.isnan(betas)] = 0



    ## Display fminbnd failures
    
    if compute_L and sum(convergence_failures) > 0:
        print(f'fminbnd failed to converge {sum(convergence_failures)}/{pY} times')
    
    if compute_L and regress:
        return L, betas
    if compute_L:
        return L
    return betas
    
def ridge_MML_one_Y(q, d2, n, Y_var, alpha2):
    
    # Compute the lambda for one column of Y

    # Width of smoothing kernel to use when dealing with large lambda
    
    smooth = 7

    # Value of lambda at which to switch from step size 1/4 to step size L/stepDenom.
    # Value of stepSwitch must be >= smooth/4, and stepSwitch/stepDenom should
    # be >= 1/4.
    step_switch = 25
    step_denom = 100
    
    ## Set up smoothing

    # These rolling buffers will hold the last few values, to average for smoothing
    sm_buffer = np.full(smooth, np.nan)
    test_vals_L = np.full(smooth, np.nan)

    # Initialize index of the buffers where we'll write the next value
    sm_buffer_I = 0
                
    
    # Evaluate the log likelihood of the data for increasing values of lambda
    # This is step 1 of the two-step algorithm at the bottom of page 6.
    # Basically, increment until we pass the peak. Here, I've added trying
    # small steps as normal, then switching over to using larger steps and
    # smoothing to combat local minima.
    
    ## Mint the negative log-likelihood function
    NLL_func = mint_NLL_func(q, d2, n, Y_var, alpha2)



    # Loop through first few values of k before you apply smoothing.
    # Step size 1/4, as recommended by Karabatsos

    done = False
    NLL = np.inf
    for k in range(step_switch * 4+1):
        sm_buffer_I = sm_buffer_I % smooth +1
        prev_NLL = NLL

      # Compute negative log likelihood of the data for this value of lambda
        NLL = NLL_func(k / 4)

      # Add to smoothing buffer
        sm_buffer[int(sm_buffer_I-1)] = NLL
        test_vals_L[int(sm_buffer_I-1)] = k / 4

      # Check if we've passed the minimum
        if NLL > prev_NLL:
            # Compute limits for L
            min_L = (k - 2) / 4
            max_L = k / 4
            done = True
            break
                        
    # If we haven't already hit the max likelihood, continue increasing lambda,
    # but now apply smoothing to try to reduce the impact of local minima that
    # occur when lambda is large

    # Also increase step size from 1/4 to L/stepDenom, for speed and robustness
    # to local minima
    
    if not done:
        
        L = k / 4
        NLL = np.mean(sm_buffer)
        iteration = 0
        
        while not done:
            L += L / step_denom
            sm_buffer_I = sm_buffer_I % smooth + 1
            prev_NLL = NLL
            iteration += 1
            # Compute negative log likelihood of the data for this value of lambda,
            # overwrite oldest value in the smoothing buffer
            sm_buffer[int(sm_buffer_I-1)] = NLL_func(L)
            test_vals_L[int(sm_buffer_I-1)] = L
            NLL = np.mean(sm_buffer)
            
            # Check if we've passed the minimum or hit NaN NLL (L passed double-precision maximum)
            
            if NLL>prev_NLL:
                # Adjust for smoothing kernel (walk back by half the kernel)
                sm_buffer_I -= (smooth - 1) / 2
                sm_buffer_I += smooth * (sm_buffer_I < 0) # wrap around
                
        
                max_L = test_vals_L[int(sm_buffer_I-1)]
            
                # Walk back by two more steps to find min bound
                sm_buffer_I -= 2
                sm_buffer_I += smooth * (sm_buffer_I < 0) # wrap around
                min_L = test_vals_L[int(sm_buffer_I-1)]

                passed_min = True
                done = True

            elif np.isnan(NLL):

                passed_min = False
                done = True
                
    else:

        passed_min = True

 
    ## Bounded optimization of lambda
    # This is step 2 of the two-step algorithm at the bottom of page 6. Note
    # that Karabatsos made a mistake when describing the indexing relative to
    # k*, which is fixed here (we need to go from k*-2 to k*, not k*-1 to k*+1)

    if passed_min:
        L, _, flag, _ = optimize.fminbound(NLL_func, max(0, min_L), max_L, xtol=1e-04, full_output=1, disp=0)
    else:
        flag = 1 # if the above loop could not find the minimum, return failed-to-converge flag
    
    return L, flag

def  mint_NLL_func(q, d2, n, Y_var, alpha2):
    '''
    Mint an anonymous function with L as the only input parameter, with all
    the other terms determined by the data.
    We've modified the math here to eliminate the d^2 term from both alpha
    (Equation 1, in main function) and here (Equation 19), because they
    cancel out and add numerical instability.
    '''
    NLL_func = lambda L: - (q * np.log(L) - np.sum(np.log(L + d2[:q])) \
                - n * np.log(Y_var - np.sum( np.divide(alpha2[:q],(L + d2[:q])))))
    return NLL_func

def array_shrink(data_in, mask, mode='merge'):
    '''
    Code to merge the first two dimensions of matrix 'DataIn' into one and remove
    values based on the 2D index 'mask'. The idea here is that DataIn is a stack
    of images with resolution X*Y and pixels in 'mask' should be removed to
    reduce datasize and computional load of subsequent analysis. The first 
    dimension of 'DataOut' will be the product of the X*Y of 'DataIn' minus 
    pixels in 'mask'. 
    Usage: data_out = array_shrink(data_in,mask,'merge')

    To re-assemble the stack after computations have been done, the code
    can be called with the additional argument 'mode' set to 'split'. This
    will reconstruct the original data structure removed pixels will be
    replaced by NaNs.
    Usage: data_out = array_shrink(data_in,mask,'split')

    Originally written in MATLAB by Simon Musal, 2016

    Adapted to Python by Michael Sokoletsky, 2021
    '''

    d_size = np.shape(data_in)  # size of input matrix
    if d_size[0] == 1:
        data_in = np.squeeze(data_in)  # remove singleton dimensions
        d_size = np.shape(data_in)

    if len(d_size) == 2:
        if d_size[0] == 1:
            data_in = data_in.T
            d_size = np.shape(data_in)  # size of input matrix

        d_size = d_size + (1,)

    if mode == 'merge':  # merge x and y dimension into one

        data_in = np.reshape(data_in,
                             (np.size(mask),
                              np.prod(d_size[mask.ndim:])))
        mask = mask.flatten()  # reshape mask to vector
        data_in = data_in[~mask, :]
        orig_size = [np.size(data_in, 0), *d_size[2:]]
        data_in = np.reshape(data_in, tuple(orig_size))
        data_out = data_in

    elif mode == 'split':  # split first dimension into x- and y- dimension based on mask size

        # check if datatype is single. If not will use double as a default.
        if data_in.dtype == 'float32':
            d_type = 'float32'
        else:
            d_type = 'float64'

        m_size = np.shape(mask)
        mask = mask.flatten()  # reshape mask to vector
        curr_size = [np.size(mask), *d_size[1:]]
        # pre-allocate new matrix
        data_out = np.full(tuple(curr_size), np.nan, dtype=d_type)
        data_out = np.reshape(data_out, (np.size(data_out, 0), -1))
        data_out[~mask, :] = np.reshape(data_in, (np.sum(~mask), -1))
        orig_size = [*m_size, *d_size[1:]]
        data_out = np.squeeze(np.reshape(data_out, tuple(orig_size)))

    return data_out

def vis_score(data, m_svt, opts, frame_idx):
    '''
    Short code to compute the correlation between lowD data Vc and modeled
    lowD data Vm. Vc and Vm are temporal components, u is the spatial
    components. corr_mat is a the correlation between Vc and Vm in each pixel.

    Originally written in MATLAB by Simon Musall, 2019

    Adapted to Python by Michael Sokoletsky, 2021
    '''

    if opts['map_met'] == 'r2':
        if opts['sample_trials'] > 0:
            Vc = data.svt[frame_idx, :].T
            Vm = m_svt[frame_idx, :].T
        else:
            Vc = data.svt.T
            Vm = m_svt.T
        cov_Vc = np.cov(Vc)  # S x S
        cov_Vm = np.cov(Vm)  # % S x S
        c_cov_V = (Vm - np.expand_dims(np.mean(Vm, 1), axis=1)
                ) @ Vc.T / (np.size(Vc, 1) - 1)  # S x S
        cov_P = np.expand_dims(np.sum((data.u_flat @ c_cov_V) * data.u_flat, 1), axis=0)  # 1 x P
        var_P1 = np.expand_dims(np.sum((data.u_flat @ cov_Vc) * data.u_flat, 1), axis=0)  # 1 x Pii
        var_P2 = np.expand_dims(np.sum((data.u_flat @ cov_Vm) * data.u_flat, 1), axis=0)  # 1 x P
        std_Px_Py = var_P1 ** 0.5 * var_P2 ** 0.5  # 1 x P
        corr_mat = (cov_P / std_Px_Py).T
        corr_mat = array_shrink(corr_mat, data.mask, 'split') ** 2
    elif opts['map_met'] == 'R2':
        if opts['sample_trials'] > 0:
            real_act = data.svt[frame_idx, :] @ data.u_flat.T
            model_act = m_svt[frame_idx, :] @ data.u_flat.T
        else:
            real_act = data.svt @ data.u_flat.T
            model_act = m_svt @ data.u_flat.T
        scores = r2_score(real_act, model_act, multioutput='raw_values')
        corr_mat = array_shrink(scores, data.mask, 'split')
        
    return corr_mat

def smoothCol_box(V, win=5):
    V = np.asarray(V, dtype=float)
    if win <= 1:
        return V
    kernel = np.ones(win, dtype=float) / win
    out = np.empty_like(V)
    for j in range(V.shape[1]):
        out[:, j] = np.convolve(V[:, j], kernel, mode="same")
    return out

from scipy.sparse import issparse

class SVDStack(object):
    '''
    This class created the image stack for widefield imaging data.
    '''

    def __init__(self, u, svt, dims=None, dtype='float32'):
        self.u = u.astype('float32')
        self.svt = svt.astype('float32')
        self.issparse = False
        self.mask = np.isnan(u[:, :, 0])  # create the mask
        if issparse(u):
            self.issparse = True
            if dims is None:
                raise ValueError(
                    'Supply dims = [H,W] when using sparse arrays')
            self.u_flat = self.u
        else:
            if dims is None:
                dims = u.shape[:2]
            self.u_flat = array_shrink(u, self.mask)
        self.shape = [svt.shape[1], *dims]
        self.dtype = dtype

    def __len__(self):
        return self.svt.shape[0]
    

def mint_calc_score(data):
    '''
    This function calculates the loadings once so that the calc_score does not have to do so repeatedly.
    '''
    loadings = np.linalg.norm(data.svt, axis=0)**2

    def calc_score(y_true, y_pred):
        '''
        This function returns an R2 score based on the loadings of each component.
        '''
        
        numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        denominator = (
            (y_true - np.mean(y_true, axis=0)) ** 2
        ).sum(axis=0, dtype=np.float64)
        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = np.ones([y_true.shape[1]])
        output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
        # arbitrary set to zero to avoid -inf scores, having a constant
        # y_true is not interesting for scoring a regression anyway
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

        return np.average(output_scores, weights=loadings)

    return calc_score