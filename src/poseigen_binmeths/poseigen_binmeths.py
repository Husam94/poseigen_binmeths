import numpy as np
import pandas as pd
import math
import os
import itertools

from collections import Counter

import poseigen_seaside.basics as se

import poseigen_seaside.metrics as mex

from scipy.interpolate import interp1d

import concurrent.futures





#---------------------------------------------------------------------------------------


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



def Harpoon(bind, select, 
            multi = False, 
            repeat = 1, custidx = None): 
    
    #BIND IS NOW INPUT! DOES NOT DO BINNING ANY MORE

    #BIND FORMAT IS [LENGTH, TYPE] FOR THIS ONE
    
    #OCT 2 23, ADDED REPEAT FUNCTION FOR HARPOON TRAINER 
    ##cust idx is custom idx 
    
    #Added Multi feature (June 2) 
    #for multi #Inp is Observations as rows and each output as a column, else its a 1dim array 

    if select < 1: select = np.round(len(bind) * select).astype(int)
    
    if multi: 
        h2, h1 = np.unique(bind, axis = 0, return_counts = True)
        div = [np.where((bind == u).all(axis = 1))[0] for u in h2] 
    else: 
        h2, h1 = np.unique(bind, return_counts = True)
        div = [np.where(bind == u)[0] for u in h2]

    if custidx is not None: div = [np.array(custidx)[d] for d in div]
    
    lu = len(h2)
    nums = np.repeat(0, lu)
    
    j = 0 
    for _ in range(select):
        for i in np.random.permutation(np.arange(lu)):
            if j == select: break 
            elif h1[i] > 0:
                nums[i] += 1
                h1[i] -= 1
                j += 1 
        
    idx = np.stack([np.hstack([np.random.choice(d, replace = False, size = w) for d,w in zip(div, nums)]) 
                    for _ in range(repeat)], 0)
    
    return idx[0] if repeat == 1 else idx



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




# def BinCount(bind, uni = None, multi = False, prop = False):
#     #uni is shared
#     #inp is a list of 1D arrays or lists of values

#     if multi is False: bind = [bind]
#     bind = [np.array(m) for m in bind]
    
#     if uni is None: uni = np.sort(np.unique(bind))

#     hs = [Counter(m.reshape(-1)) for m in bind]
#     max_key = max(max(d) for d in hs)
#     empty = dict.fromkeys(uni, 0)
#     rere = [{**empty, **d} for d in hs]

#     cnv = pd.DataFrame(rere).to_numpy()

#     if multi is False: cnv = cnv[0] 

#     if prop: cnv = cnv / np.sum(cnv)
    
#     return cnv



# def BinCount(bind, uni = None, multi = False, prop = False, bin_sizes = None):
#     #uni is shared
#     #inp is a list of 1D arrays or lists of values

#     if multi is False: bind = [bind]
#     bind = [np.array(m) for m in bind]
#     uni = [uni]
#     bin_sizes = [bin_sizes]

#     cnvs = []
#     for binx, unix, bsx, in zip(bind, uni, bin_sizes): 
#         if unix is None: unix = np.sort(np.unique(binx))
#         hs = Counter(binx.reshape(-1))
#         max_key = max(hs)
#         empty = dict.fromkeys(unix, 0)
#         rere = {**empty, **hs}
#         cnv = pd.DataFrame(rere).to_numpy()
#         if prop is not None: cnv = cnv / np.sum(cnv)
#         if bsx is not None: 
#             bsx = np.array(bsx)
#             bsx = bsx / np.sum(bsx)
#             cnv = cnv / bsx
#         cnvs.append(cnv)
    
#     if multi is False: cnvs = cnvs[0]

#     return cnvs

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




# def ObjUni(inp, scalefactor, 
#            mode = [se.tanh, {}], mode_rev = [se.revtanh, {}], 
#                   bins = 10, 
#                   open_ends = False, 
#                   mima = None, lowerbound = None, upperbound = None):
    
#     inpx = inp.reshape(-1) 

#     if mima is None: mi, ma = None, None
#     elif isinstance(mima, int): mi, ma = (0, int)
#     else: mi, ma = mima

#     if mi is None: mi = np.min(inp)
#     if ma is None: ma = np.max(inp)

#     lb = mi if lowerbound is None else lowerbound
#     ub = ma if upperbound is None else upperbound

#     inpx = inp.reshape(-1)
#     inpx[inpx < lb] = lb
#     inpx[inpx > ub] = ub

#     liny = mode[0](scalefactor * inpx, **mode[1])

#     hist, bin_edges = np.histogram(liny, bins = bins, density=False)

#     rt = mode_rev[0](bin_edges, **mode_rev[1]) / scalefactor

#     rt[0] = mi
#     rt[-1] = ma

#     return np.array(se.ListWindower(rt, win_size = 2))

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




# def BinOpt(inp,
#            mode = [se.tanh, {}], mode_rev = [se.revtanh, {}], 
#            obs_min = 10,
#            bins_min = 5, bins_max = 50, bins_increm = 5,
#            scalefax_min = 1, scalefax_increm = 0.1, scalefax_max = 50): 

#     bins_num_range = np.arange(bins_min, bins_max + 1, bins_increm)[::-1]
#     scalefax_range = np.arange(scalefax_min, scalefax_max, scalefax_increm)
#     inpx = inp.reshape(-1)

#     def eval_bin_scalefax(args):
#         bins_num, scalefax = args
#         liny = mode[0](scalefax * inpx, **mode[1])
#         hist, _ = np.histogram(liny, bins=bins_num, density=False)
#         mino_obs = np.min(hist)
#         return (bins_num, scalefax, mino_obs)

#     combos = [(bins_num, scalefax) for bins_num in bins_num_range for scalefax in scalefax_range]
#     results = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         for res in executor.map(eval_bin_scalefax, combos):
#             results.append(res)

#     # Sort by bins_num descending, scalefax ascending
#     results = sorted(results, key=lambda x: (-x[0], x[1]))

#     for bins_num in bins_num_range:
#         candidates = [r for r in results if r[0] == bins_num and r[2] >= obs_min]
#         if candidates:
#             best = min(candidates, key=lambda x: x[1])
#             return best[0], best[1]

#     print('Not Possible')
#     return None



# def BinOpt(inp,
#            mode = [se.tanh, {}],
#            obs_min = 10,
#            bins_min = 5, bins_max = 50, bins_increm = 5,
#            scalefax_min = 1, scalefax_increm = 0.1, scalefax_max = 50): 

#     bins_num_range = np.arange(bins_min, bins_max + 1, bins_increm)[::-1]
#     scalefax_range = np.arange(scalefax_min, scalefax_max, scalefax_increm)
#     inpx = inp.reshape(-1)

#     def eval_scalefax(scalefax, bins_num):
        
#         if mode is not None: liny = mode[0](scalefax * inpx, **mode[1])
#         else: liny = inpx

#         hist, _ = np.histogram(liny, bins=bins_num, density=False)
#         mino_obs = np.min(hist)
#         return scalefax, mino_obs

#     for bins_num in bins_num_range:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             results = list(executor.map(lambda s: eval_scalefax(s, bins_num), scalefax_range))
#         # Find the smallest scalefax that satisfies obs_min
#         candidates = [(s, m) for s, m in results if m >= obs_min]
#         if candidates:
#             best_scalefax = min(candidates, key=lambda x: x[0])[0]
#             return bins_num, best_scalefax

#     print('Not Possible')
#     return None


# def BinOpt(inp,
#            mode = [se.tanh, {}],
#            obs_min = 10,
#            bins_min = 5, bins_max = 50, bins_increm = 5,
#            scalefax_min = 1, scalefax_increm = 0.1, scalefax_max = 50):

#     inpx = inp.reshape(-1)
#     bins_num_range = np.arange(bins_min, bins_max + 1, bins_increm)

#     # Step 1: Try normal scale (scalefax=1) first
#     def eval_bins_num_normal(bins_num):
#         liny = inpx
#         hist, _ = np.histogram(liny, bins=bins_num, density=False)
#         mino_obs = np.min(hist)
#         return bins_num, mino_obs

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(eval_bins_num_normal, bins_num_range))
#     valid_bins = [b for b, m in results if m >= obs_min]

#     if valid_bins:
#         max_bins_num = max(valid_bins)
#         return max_bins_num, np.nan  # No scaling needed

#     # Step 2: If normal scale fails, proceed to scaling
#     scalefax_range = np.arange(scalefax_min, scalefax_max, scalefax_increm)

#     def eval_scalefax(scalefax):
#         if mode is not None:
#             liny = mode[0](scalefax * inpx, **mode[1])
#         else:
#             liny = inpx
#         hist, _ = np.histogram(liny, bins=bins_min, density=False)
#         mino_obs = np.min(hist)
#         return scalefax, mino_obs

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(eval_scalefax, scalefax_range))
#     candidates = [(s, m) for s, m in results if m >= obs_min]
#     if not candidates:
#         print('Not Possible')
#         return None

#     best_scalefax = min(candidates, key=lambda x: x[0])[0]

#     def eval_bins_num_scaled(bins_num):
#         if mode is not None:
#             liny = mode[0](best_scalefax * inpx, **mode[1])
#         else:
#             liny = inpx
#         hist, _ = np.histogram(liny, bins=bins_num, density=False)
#         mino_obs = np.min(hist)
#         return bins_num, mino_obs

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(eval_bins_num_scaled, bins_num_range))
#     valid_bins = [b for b, m in results if m >= obs_min]
#     if not valid_bins:
#         print('Not Possible')
#         return None

#     max_bins_num = max(valid_bins)
#     return max_bins_num, best_scalefax




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




#====================================== SPLIT TOOLS ======================================


def SplitExpander(X1, X2, Split, ma = None):

    # X1 is the full length of the total split (modified to be able to expand subsplits)
    if isinstance(X1, int) == False: 
        X1 = len(X1)
    
    if ma is None: ma = np.sum([len(x) for x in Split])
 
    #X2 is the multiplier
    if isinstance(X2, int) == False: X2 = len(X2)

    multiplier = X2 // X1 if X2 > X1 else X2 #June 1 mod

    new = [np.hstack([*[e + (ma * m) for m in range(multiplier)]]) for e in Split]
        
    return new


def SplitRepeater(split, reps): 
    #reps is repeats
    #each obs reps times. Designed for btc days
    return [np.hstack([np.arange(o*reps, (o+1)*reps) 
                       for o in sp]) 
            for sp in split]



def SplitData(data, split): return [data[s] for s in split]

def Data2Divided(inp, split): 
    return [[n[s] for n in inp] for s in split]
    
def Data2Grouped(inp, split = None):
    #for Divided dictionary, you need to select which split to use e.g., dict[0] as its the same as divided format
    if split is None or split is False: 
        grouped = [[inp[n][m] for n in range(len(inp))] for m in range(len(inp[0]))]
    else: 
        grouped = [[d[s] for s in split] for d in inp]
    
    return grouped







def Divided2NonDivided(divs):
    #divs in the shape of (obs, features)
    
    stacks = np.hstack(divs) if len(divs[0].shape) == 1 else np.vstack(divs)

    cs = np.cumsum([0] + [len(d) for d in divs])
    split = [np.arange(cs[i], cs[i+1]) for i in range(len(divs))]
    
    return stacks, split

def Idx2Split(X, idx, props = [0.7, 0.3]):
    if isinstance(X, int) == False: X = len(X)
    if idx < 1: idx = se.Round2Int(idx*X)
    
    f = np.arange(X)
    ls = [se.Round2Int(X*p) for p in props]
    lx = [se.Round2Int(np.sum(ls[:r])) for r in range(len(ls))]

    fs = [se.Cutter(se.Cutter(f, u), idx) for u in lx]
    
    return [j[:r] for j,r in zip(fs, ls)]

def UngroupSplits(Splits, group):
    #Splits is a list of splits

    LG = np.arange(len(group))
    Splits = [[LG[np.isin(group, sp)] for sp in split] for split in Splits]

    return Splits



def SplitGen_Random(X, num_splits = 1, proportions = [0.7, 0.3], group = None): 
    
    #group is the inp groups from OverlapGroup or whereever
    
    if group is not None: li = len(np.unique(group))
    else: li = X if isinstance(X, int) else len(X)
    
    #proportions = np.array(proportions) / np.sum(proportions)
    
    ls = [se.Round2Int(li*p) for p in proportions]
    if np.sum(ls) < li: ls[0] = ls[0] + (li - np.sum(ls))
    idx = np.arange(li)
      
    Splits = []
    for _ in range(num_splits):
        LXX = np.random.permutation(idx)
        Splits.append([LXX[e-l:e] for l,e in zip(ls, np.cumsum(ls))])
    
    
    if group is not None: Splits = UngroupSplits(Splits, group)

    if num_splits == 1: Splits = Splits[0]
    
    return Splits



def SplitGen_Window(X, num_splits = 3, 
                    proportions = [0.7, 0.3],
                    everyother = None, start_at = None, end_at = None, 
                    random = True, group = None): 
    
    #if num_splits is None, it returns all possible splits
    
    if group is not None: li = np.max(group)
    else: li = X if isinstance(X, int) else len(X)
    
    #proportions = np.array(proportions) / np.sum(proportions)
    
    ls = [se.Round2Int(li*p) for p in proportions]
    lx = [se.Round2Int(np.sum(ls[:r])) for r in range(len(ls))]
    
    idx = np.arange(li)
    idxs = [se.Cutter(idx, u) for u in lx]
    
    contwin = [se.ContinuousWindower(idx, k) for k in ls] # GOOD

    if num_splits is not None or everyother is not None:
        if everyother is not None: 
            rando = idx[::everyother]
            num_splits = None 
        else:
            rando = np.random.choice(idx, num_splits, replace = False) if random is True else np.arange(num_splits)

        idxs = [we[rando] for we in idxs]
        contwin = [coco[ra] for coco, ra in zip(contwin, idxs)]
    
    num_splits = len(contwin[0])
    Splits = [[a[i] for a in contwin] for i in range(num_splits)]
    
    if group is not None: Splits = UngroupSplits(Splits, group)
    
    if num_splits == 1: Splits = Splits[0]
    
    return Splits


def SubSplitGen(inp, split, onlyfirst = False,
                num_subsamples = 1, proportion = 0.3, group = None, 
                subsample_mode = [se.SubSample_Random, {}]): 
    
    if isinstance(proportion, float) is True: 
        proportion = [proportion] * len(split)
    
    subsample_mode[1].update({'num_subsamples': num_subsamples})
    
    if group is None: ss = [subsample_mode[0](inp[s], proportion = p, **subsample_mode[1]) for s,p in zip(split, proportion)]
    else: ss = [subsample_mode[0](inp[s], group = group[s], proportion = p, **subsample_mode[1]) for s,p in zip(split, proportion)]

    if onlyfirst: 
        addon = [np.arange(len(s)) for s in split[1:]]
        if num_subsamples != 1: addon = [[s] * num_subsamples for s in addon]
        ss = [ss[0], *addon]
    
    if num_subsamples == 1: ss = [[j] for j in ss]
    
    ss = [[s[h] for h in j] for s,j in zip(split, ss)]
    
    ss = [[j[i] for j in ss] for i in range(num_subsamples)]
    
    if num_subsamples == 1: ss = ss[0]
    return ss


def SplitGen_Stratified(bind, num_splits = 1, proportions = [0.7, 0.3]):
    #bind is a 1d list of binned
    #does not do multi currently 

    #Stratified random split

    uni = np.sort(np.unique(bind))
    idxs = [np.where(bind == u)[0] for u in uni]

    counts = BinCount(bind, uni=uni, multi = False)    
    nums = [[np.round(p * c).astype(int) for p in proportions] for c in counts]
    
    Splits = []
    for _ in range(num_splits):
        Split = []
        for iu, u in enumerate(uni): 
            XX = np.random.permutation(idxs[iu])
            Split.append([XX[e-l:e] for l,e in zip(nums[iu], np.cumsum(nums[iu]))])
        Splits.append([np.hstack([s[ip] for s in Split]) for ip in range(len(proportions))])
    
    if num_splits == 1: Splits = Splits[0]
    
    return Splits

def SubSample_Stratified(bind, 
                         proportion = 0.3, num_subsamples = 1, 
                         guarantee1 = False): 
    #bind is a 1d list of binned
    #does not do multi currently 

    li = len(bind)
    if proportion > 1: proportion = proportion / li

    uni = np.sort(np.unique(bind))

    idxs = [np.where(bind == u)[0] for u in uni]
    lidxs = np.array([len(x) for x in idxs])
    
    nums = np.round(lidxs * proportion).astype(int)
    nums = [li if n > li else n for n in nums]
    if guarantee1: nums = [1 if n == 0 and lx > 0 else n for n,lx in zip(nums, lidxs)]
    
    ss = [np.hstack([np.random.choice(b, n, replace = False) for b,n in zip(idxs, nums)]) for _ in range(num_subsamples)]
    
    if num_subsamples == 1: ss = ss[0]
    return ss


###### AKIN SPLIT #############


def Akin_Scorer(bind, idxs, 
                
                uni = None, multi = False, 

                closest = False,
                onlyidx = None, p = 2, reciprocal = False, pseudo = True, weight_bymem = False,
                summarize_mode = [se.Epsilon, {}]):
    
    #idxs  = list of lists where each has 1D arrays or a list of indices to which relate to inp
    #idxs is [split, ... ]

    if multi is False: bind = [bind]
    
    all_histo = []
    for bindx in bind:
        
        if uni is None: uni = np.sort(np.unique(bindx))
        vals = [[bindx[d].reshape(-1) for d in s] for s in idxs] #@@@@@
        d2v = [BinCount(v, uni = uni, multi = True) for v in vals]

        j = BinCount(bindx, uni = uni).reshape(-1) if closest else None

        C = [se.NormalizedDistances(v, j,
                                    onlyidx = onlyidx, p = p, 
                                    reciprocal = reciprocal, pseudo = pseudo,
                                    weight_bymem = weight_bymem, 
                                    summarize_mode = summarize_mode)
             for v in d2v]
        
        all_histo.append(C)
    
    all_histo = np.stack(all_histo, 1) #ALL HISTO SHAPE: (# SPLITS, #OF SCORES)

    if all_histo.shape[1] > 1:
        all_histo = np.array([summarize_mode[0](sc, **summarize_mode[1]) for sc in all_histo])
    else: all_histo = all_histo[:, 0]
    
    return all_histo #is a 1dim array


def SplitGen_Akin(bind,
                  proportions = [0.7, 0.3], num_splits = 1, rando = 100, window = False, group = None, 

                  uni = None, multi = False,
                  closest = False,
                  onlyidx = None, p = 2, reciprocal = False, pseudo = True, weight_bymem = False,
                  summarize_mode = [se.Epsilon, {}], 

                  atatime = 100, pathname = None, pickup = False):
    
    #June 1 mod: added atatime for memory. It operates in rounds of atatime so that you can do alot of them with less memory. 
    
    rounds = np.arange(0, rando, atatime).tolist() + [rando]

    if pathname is not None: 
        if pathname.endswith('.p'): 
            temppath = pathname[:-2] + '_temp' + '.p'
        else:
            pathname = pathname + '.p'
            temppath = pathname + '_temp' + '.p'

    if pathname is None: pickup = False
    
    if pickup and os.path.isfile(temppath): 
        finround, past_splits, past_all_histo = se.PickleLoad(temppath)
    else: finround, past_splits, past_all_histo = 0, [], []

    lb = len(bind.T) if multi else len(bind)
    
    
    for ir in np.arange(len(rounds[:-1]))[finround:]:
        print(f'round {ir + 1} of {len(rounds[:-1])}')
        
        randoz = rounds[ir+1] - rounds[ir]
        if randoz > 1: 
    
            SplitGen_args = {'num_splits': randoz, 'proportions': proportions, 'group': group}                    

            if window: Splits = SplitGen_Window(lb, random = True, **SplitGen_args)
            else: Splits = SplitGen_Random(lb, **SplitGen_args)

            all_histo = Akin_Scorer(bind, Splits,
                                     
                                    uni = uni, multi = multi, 
                
                                    closest = closest,
                                    onlyidx = onlyidx, p = p, reciprocal = reciprocal, pseudo = pseudo, weight_bymem = weight_bymem,
                                    summarize_mode = summarize_mode)
            
            mino = np.argsort(all_histo)[:num_splits]

            past_splits.extend([Splits[m] for m in mino])
            past_all_histo.extend([all_histo[m] for m in mino])

            mino2 = np.argsort(past_all_histo)[:num_splits]
            past_splits = [past_splits[m] for m in mino2] #juet keep the bests
            past_all_histo = [past_all_histo[m] for m in mino2]
            Splits2 = past_splits[0] if num_splits == 1 else past_splits

            print(f'newbest:{past_all_histo[0]}')

            if pathname is not None: 
                se.PickleDump(Splits2,pathname)
                se.PickleDump([ir, past_splits, past_all_histo], temppath) 
    
    pathname: os.remove(temppath)

    return Splits2


#####################################


def SubSample_Akin(bind, 
                   proportion = 0.3, num_subsamples = 1, rando = 10, group = None,
                   
                   uni = None, multi = False,
                   closest = False,
                   onlyidx = None, p = 2, reciprocal = False, pseudo = True, weight_bymem = False,
                   summarize_mode = [se.Epsilon, {}]): 
    
    
    ss = se.se.SubSample_Random(bind, proportion = proportion, num_subsamples = rando, group = group)
    ss_l = [[s] for s in ss]
        
    all_histo = Akin_Scorer(bind.T, ss_l,
                                    
                            uni = uni, multi = multi, 
        
                            closest = closest,
                            onlyidx = onlyidx, p = p, reciprocal = reciprocal, pseudo = pseudo, 
                            weight_bymem = weight_bymem,
                            summarize_mode = summarize_mode)
    
    mino = np.argsort(all_histo)[:num_subsamples]
          
    ss = [ss[m] for m in mino]
    if num_subsamples == 1: ss = ss[0]

    return ss



def SimpleStratifiedSplit(obs_bind, props=[0.5, 0.25, 0.25], prints=False, random=False):
    
    obs_bind = np.asarray(obs_bind).reshape(-1)
    uni = np.unique(obs_bind)
    props = np.array(props)
    props = props / np.sum(props)
    lpx = len(props)
    splits = [[] for _ in range(lpx)]

    for unx in uni:
        idxs = np.where(obs_bind == unx)[0]
        n = len(idxs)

        # Base sizes
        sizes = np.floor(props * n).astype(int)

        # Remainder
        remainder = n - np.sum(sizes)

        if remainder > 0:
            if random:
                # Randomly choose which splits get the extra observations
                extra_bins = np.random.choice(lpx, size=remainder, replace=False)
                for b in extra_bins:
                    sizes[b] += 1
            else:
                # Original behavior: allocate to first splits
                sizes[:remainder] += 1

        # Shuffle and split indices
        np.random.shuffle(idxs)
        cs = np.cumsum(sizes)
        split_idxs = np.split(idxs, cs[:-1])

        for i, s in enumerate(split_idxs):
            splits[i].append(s)

    # Concatenate splits for all classes
    splits = [np.concatenate(s) if len(s) > 0 else np.array([], dtype=int) for s in splits]

    if prints:
        print([BinCount(obs_bind[s], uni=uni) for s in splits])

    return splits




# class BinWeights: 

#     def __init__(self, uni = None, multi = False, minus = True, newrange = True):

#         self.uni = uni
#         self.multi = multi
#         self.minus = minus
#         self.newrange = newrange
        
#     def fit(self, bind): 

#         multi = self.multi
#         uni = self.uni
#         minus = self.minus
#         newrange = self.newrange

#         if multi is False: bind = [bind]

#         all_weights, unis = [], []
#         for bindx in bind: 

#             if uni is None: uni = np.sort(np.unique(bindx))
#             unis.append(uni)
#             cnv = BinCount(bindx, uni = uni, multi = False)
#             cnv_prop = cnv / np.sum(cnv)

#             weights = 1 - cnv_prop if minus else 1 / cnv_prop
            
#             if newrange is not None: 
#                 mi, ma = se.MiMa(weights) 
#                 if isinstance(newrange, tuple): 

#                     newmi, newma = newrange
#                     weights = (((weights - mi) * (newma - newmi)) / (ma - mi)) + newmi
                
#                 elif newrange is True: 
#                     weights = weights / ma
            
#             all_weights.append(weights)
        
#         self.all_weights = all_weights
#         self.unis = unis
    
#         return self
    
#     def eval(self, inp, dtype = np.float32):
         
#         if self.multi is False: inp = [inp]

#         new_weights = []

#         for inpx, weights, uni in zip(inp, self.all_weights, self.unis):
#             weix = np.zeros(inpx.shape, dtype = dtype)
#             for u,w in zip(uni, weights): weix[inpx == u] = w
#             new_weights.append(weix)
            
#         new_weights = np.stack(new_weights) if self.multi else new_weights[0]
        
#         return new_weights


# def BinWeighter(bind, onlyidx = None, byaxis = None, 
#                 uni = None, minus = True, newrange = True,
#                 dtype = np.float32):
    
#     bind = np.array(bind) #just in case
    
#     origshape = bind.shape

#     inps = [bind] if byaxis is None else [bind.take(indices=a, axis=byaxis) for a in range(origshape[byaxis])]

#     inpshape = inps[0].shape

#     newei = []

#     for j in inps: 
#         bine = j if onlyidx is None else j[onlyidx]

#         bw = BinWeights(uni = uni, multi = False, minus = minus, newrange = newrange).fit(bind = bine.reshape(-1))
#         wo = bw.eval(j.reshape(-1), dtype=dtype).reshape(inpshape)
#         newei.append(wo)
    
#     newei = np.stack(newei, axis = byaxis) if byaxis else newei[0] 
    
#     return newei



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





def DA_AError(inp1, inp2, weights = None, 
              
              bind = None, bw_mode = 'recip', bin_sizes = None,

              mean = True, rel = False, 
           expo = 2, root = False, weightbefore = False): 
    
    ''' 
    Weights is a placeholder here 
    inp2 is the y_true
    
    '''
    
    # DENSITY AWARE ABSOLUTE ERROR 
    
    #added revbase and revpseudo to reverse log transformations.
    #Applies to both inputs. 

    weights = BinWeights(bind, bin_sizes = bin_sizes, mode = bw_mode)

    if expo != 2: root = False
    w = mex.WeightsAdjuster(inp1, weights)

    rr = inp2 if rel else 1

    e = abs((inp1-inp2) / rr) #RELATIVE ERROR 

    if weightbefore: w = w **expo
    
    AE = w * (e ** expo)
    
    if mean == True:
        if isinstance(w, int) == False: AE = AE.sum() / w.sum()
        else: AE = AE.mean()

        if root: AE = AE**(1/2)

    return AE


def EBS_allocations(y_bin, sample_size=1000, n_bins=250):
    '''EBS = Exhaustive Balanced Sampling '''

    counts = np.bincount(y_bin, minlength=n_bins)
    remaining_quota = sample_size
    remaining_bins = np.arange(n_bins)
    allocations = np.zeros(n_bins, dtype=int)

    while remaining_quota > 0 and len(remaining_bins) > 0:
        EBS = remaining_quota // len(remaining_bins)
        if EBS == 0:
            for b in remaining_bins[:remaining_quota]:
                allocations[b] += 1
            break
        alloc = np.minimum(EBS, counts[remaining_bins])
        allocations[remaining_bins] += alloc
        remaining_quota -= alloc.sum()
        exhausted = alloc == counts[remaining_bins]
        remaining_bins = remaining_bins[~exhausted]
    return allocations

def EBS_sample(y_bin, sample_size=1000, n_bins=250, random_state=None, obs_weight=None):
    '''EBS = Exhaustive Balanced Sampling with optional observation weights'''
    rng = np.random.default_rng(random_state)
    allocations = EBS_allocations(y_bin, sample_size, n_bins)
    bin_indices = [np.where(y_bin == b)[0] for b in range(n_bins)]
    sampled = []
    for b in range(n_bins):
        n = allocations[b]
        idx = bin_indices[b]
        if n > 0 and len(idx) > 0:
            if n > len(idx):
                # This should not happen, but if it does, fix allocation
                n = len(idx)
            if obs_weight is not None:
                w = obs_weight[idx]
                w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
                chosen = rng.choice(idx, size=n, replace=False, p=w)
            else:
                chosen = rng.choice(idx, size=n, replace=False)
            sampled.append(chosen)
    return np.concatenate(sampled)


















#==========================================


def HarpoonOverlapping(window_belong, select, repeat=1):
    """
    Round robin sampling for overlapping bins (window_belong: (N, max_bins), -1 for unused).
    Each sample can only be selected once, even if it belongs to multiple bins.
    Returns indices sampled without replacement.
    """
    window_belong = np.asarray(window_belong)
    N = window_belong.shape[0]
    num_bins = int(window_belong.max()) + 1 if np.any(window_belong != -1) else 0

    if select < 1:
        select = int(np.round(N * select))

    all_selected = []
    for _ in range(repeat):
        selected_mask = np.zeros(N, dtype=int)
        selected_count = 0
        while selected_count < select:
            for bin_idx in np.random.permutation(np.arange(num_bins)):
                # Find indices where not yet selected and belongs to this bin
                belongs = (window_belong == bin_idx).any(axis=1)
                candidates = np.where((selected_mask == 0) & belongs)[0]
                if len(candidates) > 0:
                    pick = np.random.choice(candidates, 1, replace=False)[0]
                    selected_mask[pick] = 1
                    selected_count += 1
                    if selected_count == select:
                        break
        all_selected.append(np.where(selected_mask == 1)[0])
    return all_selected[0] if repeat == 1 else all_selected








