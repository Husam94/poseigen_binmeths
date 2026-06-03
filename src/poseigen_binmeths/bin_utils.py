import numpy as np
import itertools

import poseigen_seaside.basics as se

from scipy.interpolate import interp1d

import concurrent.futures



def BinParser(inp, uni, categorical = False, multi = False): 
    
    if multi is False: inp = [inp]

    pars = []
    for inpx in inp: 
        if categorical: par = [np.where(inpx == u)[0] for u in uni] 
        else: par = [np.where(np.logical_and(inpx>=u[0], inpx <= u[1]))[0] for u in uni]
        pars.append(par)
    
    if multi is False: pars = pars[0]
        
    return pars


def Binner(inp, uni=None, open_ends=False):
    """
    Fast binning with optional open ends.
    If open_ends=True, values below the lowest bin go to bin 0,
    values above the highest bin go to the last bin.
    """
    inp = np.asarray(inp)
    categorical = isinstance(inp.flat[0], str)
    inpy = inp.reshape(-1)
    inpx = np.zeros(len(inpy), dtype=int)

    if categorical:
        par = BinParser(inpy, uni, categorical=True)
        for ip, p in enumerate(par):
            inpx[p] = ip
    else:
        uni = np.asarray(uni)
        starts = uni[:, 0]
        ends = uni[:, 1]
        # Find which bin each value belongs to
        idx = np.searchsorted(starts, inpy, side='right') - 1
        # Mask for values within bin ranges
        valid = (idx >= 0) & (inpy <= ends[idx])
        inpx[valid] = idx[valid]
        if open_ends:
            # Values below lowest bin
            inpx[inpy < starts[0]] = 0
            # Values above highest bin
            inpx[inpy > ends[-1]] = len(uni) - 1
        # Otherwise, out-of-range values remain 0

    inpx = inpx.reshape(inp.shape)
    return inpx

def HistUni(inp, bins = 10): 
    inp = np.array(inp)
    bin_edges = np.linspace(inp.min(), inp.max(), bins + 1)
    uni = se.ListWindower(bin_edges, win_size = 2)
    return np.array(uni)


def UniProp(uni): 
    startoz, endoz = uni.T
    mino, maxo = startoz.min(), endoz.max()
    diffs = endoz - startoz
    return diffs / (maxo - mino)


def HistBinning(inp, bins = 10):
    uni = HistUni(inp, bins = bins)
    return Binner(inp, uni = uni)


def HistWeights(x, bins = 1000, recip = False, interp = False): 

    # Not the same as in original.
    
    x = np.array(x)
    x_bind = HistBinning(x, bins = bins)

    histoz = np.histogram(x.reshape(-1), bins = bins)
    histoz_scaled = histoz[0] / histoz[0].sum()

    if interp: 

        x = np.array(x)
        x_mi, x_ma = se.MinMax(x)

        oldr = np.linspace(x_mi, x_ma, len(histoz_scaled))
        weis = np.interp(x.reshape(-1), oldr, histoz_scaled)

    else: 
        weis = np.zeros(x.shape)
        for iz, z in enumerate(histoz_scaled):
            weis[x_bind == iz] = z

    if recip: 
        weis = 1 / weis 
        weis = weis / weis.sum()

    return weis


def BinFunc(inp, bind, uni = None,  
            mode = [np.mean, {}]):
    
    #DOES NOT SUPPORT MULTI RIGHT NOW> 
        
    if uni is None: uni = np.unique(bind)
    results = [mode[0](inp[bind == un], **mode[1]) for un in uni]
    
    return results



def BinsOfBins(bind, uni = None):
    #BIND FORMAT IS [LENGTH, TYPE] FOR THIS ONE
    # uni is a list the length of TYPE. for each type, it tells the unique ones. 

    if uni is None: h2, h1 = np.unique(bind, axis = 0, return_counts = True)
    else: h2 = [list(x) for x in list(itertools.product(*uni))]

    div = [np.where((bind == u).all(axis = 1))[0] for u in h2]
    lu = len(h2)
    newbind = np.repeat(0, len(bind))
    for i in range(lu): 
        newbind[div[i]] = i
    
    return newbind



def BinCount(bind, uni=None, multi=False, prop=False, bin_sizes=None):
    """
    Efficient bin counting with optional density scaling.
    Supports different uni and bin_sizes for each bind when multi=True.
    """
    if not multi:
        # Single array mode
        binx = np.asarray(bind).reshape(-1)
        unix = np.sort(np.unique(binx)) if uni is None else np.asarray(uni)
        idx = np.searchsorted(unix, binx)
        valid = (idx < len(unix)) & (unix[idx] == binx)
        counts = np.bincount(idx[valid], minlength=len(unix)).astype(float)
        if prop:
            s = counts.sum()
            if s > 0:
                counts = counts / s
        if bin_sizes is not None:
            bsx = np.asarray(bin_sizes, dtype=float)
            bsx = bsx / bsx.sum() if bsx.sum() > 0 else bsx
            counts = counts / bsx
        return counts
    else:
        # Multi mode: bind, uni, bin_sizes should be lists of same length
        cnvs = []
        for i, binx in enumerate(bind):
            unix = np.sort(np.unique(binx)) if (uni is None or uni[i] is None) else np.asarray(uni[i])
            binx = np.asarray(binx).reshape(-1)
            idx = np.searchsorted(unix, binx)
            valid = (idx < len(unix)) & (unix[idx] == binx)
            counts = np.bincount(idx[valid], minlength=len(unix)).astype(float)
            if prop:
                s = counts.sum()
                if s > 0:
                    counts = counts / s
            if bin_sizes is not None and bin_sizes[i] is not None:
                bsx = np.asarray(bin_sizes[i], dtype=float)
                bsx = bsx / bsx.sum() if bsx.sum() > 0 else bsx
                counts = counts / bsx
            cnvs.append(counts)
        return cnvs


def BinDensity(bind, uni, raw = False):
    
    """
    Slides across inp1, counts the number of observations in each window,
    scales by the total number of observations, and fits a linear interpolation.
    Returns the interpolation object: winden.predict(x)
    """

    startoz, endoz = uni.T
    centers = (endoz + startoz) / 2

    counts = BinCount(bind)
    counts = np.array(counts)

    densities = counts if raw else  counts / len(bind)
    winden = interp1d(centers, densities, kind='linear', bounds_error=False, 
                      fill_value=(densities[0], densities[-1]))
    winden.predict = winden.__call__  # For API consistency

    return winden



def BinSelect(bind, weights = None, select = 0.1): 
    """
    Weighted random sampling from bins without replacement.
    
    Parameters:
    -----------
    bind : array-like
        1D array of bin labels (integers)
    weights : array-like, optional
        Corresponding weights for each observation. If None, uniform weights.
    select : float, int, or list
        - Single int: number of observations to select per bin
        - Single float < 1: proportion of observations to select per bin
        - List: per-bin selection (int or float < 1 for each bin)
    
    Returns:
    --------
    selected_indices : ndarray
        Indices of selected observations
    """
    bind = np.asarray(bind).reshape(-1)
    N = len(bind)
    
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.asarray(weights).reshape(-1)
    
    uni = np.unique(bind)
    num_bins = len(uni)
    
    # Parse select parameter
    if not isinstance(select, (list, np.ndarray)):
        select = np.full(num_bins, select)
    else:
        select = np.asarray(select)
        if len(select) != num_bins:
            raise ValueError(f"select list length ({len(select)}) must match number of bins ({num_bins})")
    
    selected_indices = []
    
    for bin_idx, bin_label in enumerate(uni):
        # Get indices for this bin
        bin_mask = (bind == bin_label)
        bin_indices = np.flatnonzero(bin_mask)
        n_available = len(bin_indices)
        
        if n_available == 0:
            continue
        
        # Determine number to select
        n_select = select[bin_idx]
        if n_select < 1:
            # Proportion
            n_select = int(np.round(n_available * n_select))
        else:
            # Absolute count
            n_select = int(n_select)
        
        # Check if we have enough observations
        if n_select > n_available:
            print('Less than wanted')
            selected_indices.append(bin_indices)
        elif n_select > 0:
            # Normalize weights for this bin
            bin_weights = weights[bin_mask]
            bin_weights_norm = bin_weights / bin_weights.sum()
            
            # Weighted random sampling without replacement
            chosen = np.random.choice(bin_indices, size=n_select, replace=False, p=bin_weights_norm)
            selected_indices.append(chosen)
    
    # Concatenate all selected indices
    if len(selected_indices) > 0:
        selected_indices = np.concatenate(selected_indices)
    else:
        selected_indices = np.array([], dtype=int)
    
    return selected_indices


################## OBJECTIVE BINNING ##################################


def FindScaleFactor(inp, 
                    mode = [se.tanh, {}], 
                    mode_rev = [se.revtanh, {}], 
                 
              bins = 10, 
              maxrange = 0.3, minobs = None,
              start = 1, increm = 0.1, 
              mima = None, 
              lowerbound = None, upperbound = None): 
    
    #lets determine the maximum alpha:

    inpx = inp.reshape(-1) 

    if mima is None: mi, ma = None, None
    elif isinstance(mima, int): mi, ma = (0, int)
    else: mi, ma = mima

    if mi is None: mi = np.min(inp)
    if ma is None: ma = np.max(inp)

    lb = mi if lowerbound is None else lowerbound
    ub = ma if upperbound is None else upperbound

    inpx = inp.reshape(-1)
    inpx[inpx < lb] = lb
    inpx[inpx > ub] = ub

    cent1 = None

    totz = ma - mi 

    starto = start - increm

    vars, scalefactors, edges, hix = [], [], [], []

    if maxrange is None: maxrange = 1
    if minobs is None: minobs = len(inp) 

    lenz_max, mino_obs = 0, 0
    
    while lenz_max <= maxrange and mino_obs <= minobs:

        starto = np.round(starto + increm, 5)

        if cent1 is None:

            liny = mode[0](starto * inpx, **mode[1])

            hist, bin_edges = np.histogram(liny, bins = bins, density=False)

            mino_obs = np.min(hist)

            rt = mode_rev[0](bin_edges, ** mode_rev[1]) / starto

            lens_props = np.array([rt[i+1] - rt[i] for i in np.arange(len(rt) -1)]) / totz
            lenz_max = np.max(lens_props)

            rt[0] = mi
            rt[-1] = ma

            hix.append(hist)
            edges.append(rt)
            vars.append(np.std(hist))
            scalefactors.append(starto)
        
    
    besto = np.argmin(vars)
    #print(besto, vars[besto], edges[besto], hix[besto])

    return scalefactors[besto]



def FindNumBins(inp, 
                mode = [se.tanh, {}], 
                mode_rev = [se.revtanh, {}], 
                
                maxrange = 0.3, minobs = 50, 
                minbins = 5, maxbins = 1000,
                start = 1, increm = 0.1, 
                mima = None, lowerbound = None, upperbound = None): 
    
    #Lets just find the number of bins givin the maxrange range and minobs
    # then find the scale factor that best satistifies whatever. 

    # As we increase the scale factor with the modes, we expect more obs in the bin
    # As we decrease the number of bins, we're going to have to try more until the maxrange is reached. 


    inpx = inp.reshape(-1) 

    if mima is None: mi, ma = None, None
    elif isinstance(mima, int): mi, ma = (0, int)
    else: mi, ma = mima

    if mi is None: mi = np.min(inp)
    if ma is None: ma = np.max(inp)

    lb = mi if lowerbound is None else lowerbound
    ub = ma if upperbound is None else upperbound

    totz = ma - mi 

    inpx = inp.reshape(-1)
    inpx[inpx < lb] = lb
    inpx[inpx > ub] = ub

    starto = start

    lenz_max, mino_obs = 0, 0
    

    for bxo in np.arange(minbins, maxbins): #The maximum number of bins.

        gxzo = None

        print(f'TRYING: {bxo}')

        for incrz in np.arange(start, 100, increm): 
            
            starto = incrz

            liny = mode[0](starto * inpx, **mode[1])

            hist, bin_edges = np.histogram(liny, bins = bxo, density=False)

            mino_obs = np.min(hist)

            rt = mode_rev[0](bin_edges, ** mode_rev[1]) / starto

            lens_props = np.array([rt[i+1] - rt[i] for i in np.arange(len(rt) -1)]) / totz
            lenz_max = np.max(lens_props)

            if lenz_max <= maxrange and mino_obs >= minobs: 
                
                gxzo = starto
            
            elif lenz_max > maxrange: break 

        
        if gxzo is not None: 

            maxb = bxo 
        
        #else: break 
    
    return maxb



def ObjUni(inp, scalefactor, 
           mode = [se.tanh, {}], mode_rev = [se.revtanh, {}], 
           bins = 10, 
           open_ends = False, 
           mima = None, lowerbound = None, upperbound = None):

    inpx = inp.reshape(-1) 

    if mima is None: mi, ma = None, None
    elif isinstance(mima, int): mi, ma = (0, int)
    else: mi, ma = mima

    if mi is None: mi = np.min(inp)
    if ma is None: ma = np.max(inp)

    lb = mi if lowerbound is None else lowerbound
    ub = ma if upperbound is None else upperbound

    inpx = inp.reshape(-1)
    inpx[inpx < lb] = lb
    inpx[inpx > ub] = ub

    # If scalefactor is nan, skip scaling and transformation
    if np.isnan(scalefactor):
        liny = inpx
        bin_edges = np.linspace(lb, ub, bins + 1)
    else:
        liny = mode[0](scalefactor * inpx, **mode[1])
        bin_edges = None  # Will be set by np.histogram below

    hist, bin_edges = np.histogram(liny, bins=bins, density=False) if bin_edges is None else np.histogram(liny, bins=bin_edges, density=False)

    if np.isnan(scalefactor):
        rt = bin_edges
    else:
        rt = mode_rev[0](bin_edges, **mode_rev[1]) / scalefactor

    rt[0] = mi
    rt[-1] = ma

    return np.array(se.ListWindower(rt, win_size=2))




def BinOpt(inp,
           mode = [se.tanh, {}],
           obs_min = 10,
           bins_min = 5, bins_max = 50, bins_increm = 5,
           scalefax_min = 1, scalefax_increm = 0.1, scalefax_max = 50, 
           geometric = False):

    inpx = inp.reshape(-1)
    bins_num_range = np.arange(bins_max, bins_min - 1, -bins_increm)

    for bins_num in bins_num_range:
        # Try normal scale first
        liny = inpx
        hist, _ = np.histogram(liny, bins=bins_num, density=False)
        mino_obs = np.min(hist)
        if mino_obs >= obs_min:
            return bins_num, np.nan  # No scaling needed
        


        # Try scaling if normal scale fails
        scalefax_range = np.arange(scalefax_min, scalefax_max, scalefax_increm)
        def eval_scalefax(scalefax):
            if geometric: 
                bin_ranges = se.ListWindower(se.GeomNumSpacing(np.min(inpx), np.max(inpx), bins_num + 1, scalefax), 2)
                bind = Binner(inpx, bin_ranges)
                bc = BinCount(bind)
                print(bc)
                mino_obs = np.min(bc)

            else: 
                liny = mode[0](scalefax * inpx, **mode[1])
                hist, _ = np.histogram(liny, bins=bins_num, density=False)
                mino_obs = np.min(hist)
            return scalefax, mino_obs

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(eval_scalefax, scalefax_range))
        candidates = [(s, m) for s, m in results if m >= obs_min]
        if candidates:
            best_scalefax = min(candidates, key=lambda x: x[0])[0]
            return bins_num, best_scalefax

    print('Not Possible')
    return None






def absrel_weighted(a, b, w, eps = 1e-12): 
    '''(abs(b-a) / a)'''
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    return np.mean(w * (np.abs(b-a) / (a+eps)))



def BinWeights(inp, bin_sizes=None, mode=[se.Reciprocal, {}]):
    """
    Efficiently computes weights for each observation based on bin membership.
    Supports multi-dimensional inp and bin_sizes with np.nan fillers.
    Ensures output weights have the same shape as inp.
    """

    orig_shape = np.shape(inp)
    inp = np.squeeze(inp)

    if mode is None:
        weights = np.ones_like(inp, dtype=float)
        weights /= weights.size
        weights = weights.reshape(inp.shape)
        if weights.shape != orig_shape:
            weights = weights.reshape(orig_shape)
        return weights

    if isinstance(mode, str): 
        if mode == 'recip': mode = [se.Reciprocal, {}]
        if mode == 'comp': mode = [se.Complement, {}]

    if bin_sizes is not None:
        bin_sizes = np.squeeze(bin_sizes)
    inp_shape = inp.shape

    # Always work with 2D: (N, G)
    if inp.ndim == 1:
        inp = inp.reshape(-1, 1)
    if bin_sizes is not None and bin_sizes.ndim == 1:
        bin_sizes = bin_sizes.reshape(-1, 1)

    N, G = inp.shape
    weights_all = np.empty_like(inp, dtype=float)

    for g in range(G):
        inp_sel = inp[:, g]
        if bin_sizes is not None:
            bin_sizes_sel = bin_sizes[:, g] if bin_sizes.shape[1] > 1 else bin_sizes[:, 0]
            valid_bins = ~np.isnan(bin_sizes_sel)
            bin_labels = np.arange(len(bin_sizes_sel))[valid_bins]
        else:
            bin_labels = np.unique(inp_sel)
            bin_sizes_sel = None

        # Map inp_sel to 0-based indices in bin_labels
        sort_idx = np.argsort(bin_labels)
        sorted_bins = bin_labels[sort_idx]
        bin_indices = np.searchsorted(sorted_bins, inp_sel)
        in_range = (bin_indices < len(sorted_bins)) & (sorted_bins[bin_indices] == inp_sel)
        mapped_indices = np.full_like(bin_indices, -1)
        mapped_indices[in_range] = bin_indices[in_range]

        valid_mask = mapped_indices >= 0
        bincounts = np.bincount(mapped_indices[valid_mask], minlength=len(bin_labels))

        if bin_sizes_sel is not None and not np.any(np.isnan(bin_sizes_sel)):
            bincounts = bincounts / bin_sizes_sel[valid_bins]

        bin_weights = mode[0](bincounts / np.sum(bincounts), **mode[1])
        weights = np.ones_like(inp_sel, dtype=float)
        valid_idx = mapped_indices >= 0
        weights[valid_idx] = bin_weights[mapped_indices[valid_idx]]

        weights /= np.sum(weights)
        weights_all[:, g] = weights

    # Reshape output to match original input shape
    if G == 1:
        weights_all = weights_all[:, 0]
    weights_all = weights_all.reshape(inp_shape)
    if weights_all.shape != orig_shape:
        weights_all = weights_all.reshape(orig_shape)
    return weights_all



def OptSample(inp, bin_sizes=None, select=0.1, n_samples=50, random_state=None):
    """
    Optimal sampling that balances bin proportions across multiple bin dimensions.
    
    Parameters:
    -----------
    inp : array-like
        Input array of shape (N,) or (N, B) or (N, B, 1, 1, ...) where N is observations and B is bin dimensions
    bin_sizes : array-like or list of arrays, optional
        Bin sizes for density correction. If inp is (N, B), must be list of B arrays.
    select : float or int
        Proportion (if < 1) or number (if >= 1) of samples to select
    n_samples : int
        Number of random samples to evaluate
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    selected_indices : ndarray
        Indices of optimally selected observations
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    inp = np.asarray(inp)
    
    # Squeeze to 2D if shape has extra dimensions of size 1
    if inp.ndim > 2:
        inp = inp.squeeze()
        # If still more than 2D after squeeze, reshape to 2D keeping first two dims
        if inp.ndim > 2:
            new_shape = (inp.shape[0], np.prod(inp.shape[1:]))
            inp = inp.reshape(new_shape)
    
    # Ensure 2D: (N, B)
    if inp.ndim == 1:
        inp = inp.reshape(-1, 1)
    
    N, B = inp.shape
    
    # Determine sample size
    sample_size = int(np.round(N * select)) if select < 1 else int(select)
    sample_size = min(sample_size, N)
    
    # Prepare bin_sizes
    if bin_sizes is not None:
        if not isinstance(bin_sizes, list):
            bin_sizes = [bin_sizes]
        if len(bin_sizes) != B:
            raise ValueError(f"bin_sizes length ({len(bin_sizes)}) must match B ({B})")
    else:
        bin_sizes = [None] * B
    
    # Step 2 & 3: Compute combined weights across all B dimensions (vectorized)
    combined_weights = np.ones(N, dtype=float)
    for b in range(B):
        weights_b = BinWeights(inp[:, b], bin_sizes=bin_sizes[b], mode='recip')
        combined_weights *= weights_b
    combined_weights /= combined_weights.sum()
    
    # Step 6: Pre-compute optimal proportions for each B (do once, not per sample)
    optimal_props = []
    unis = []
    for b in range(B):
        inp_b = inp[:, b]
        bs_b = bin_sizes[b]
        
        if bs_b is not None:
            bs_b = np.asarray(bs_b)
            valid = ~np.isnan(bs_b)
            uni_b = np.arange(len(bs_b))
            opt_prop = np.zeros_like(bs_b, dtype=float)
            opt_prop[valid] = 1.0 / bs_b[valid]
            opt_prop /= opt_prop.sum()
        else:
            uni_b = np.unique(inp_b)
            opt_prop = np.ones(len(uni_b), dtype=float) / len(uni_b)
        
        optimal_props.append(opt_prop)
        unis.append(uni_b)
    
    # Step 4, 5, 7 & 8: Generate samples and score them
    best_score = np.inf
    best_sample = None
    
    for _ in range(n_samples):
        # Sample
        sampled_idx = np.random.choice(N, size=sample_size, replace=False, p=combined_weights)
        
        # Score this sample across all B dimensions
        scores_b = []
        for b in range(B):
            inp_b = inp[:, b]
            bs_b = bin_sizes[b]
            uni_b = unis[b]
            
            # Get bin counts for sample (use fast BinCount)
            sample_props = BinCount(inp_b[sampled_idx], uni=uni_b, multi=False, 
                                   prop=True, bin_sizes=bs_b)
            
            opt_prop = optimal_props[b]
            
            # Ensure same length
            if len(sample_props) != len(opt_prop):
                sample_props_full = np.zeros_like(opt_prop)
                min_len = min(len(sample_props), len(opt_prop))
                sample_props_full[:min_len] = sample_props[:min_len]
                sample_props = sample_props_full
            
            # Compute weighted absolute relative error (use existing function)
            weights_score = np.ones_like(opt_prop)
            score = absrel_weighted(sample_props, opt_prop, weights_score)
            scores_b.append(score)
        
        # Average score across B dimensions
        avg_score = np.mean(scores_b)
        
        # Keep best sample
        if avg_score < best_score:
            best_score = avg_score
            best_sample = sampled_idx
    
    return best_sample
