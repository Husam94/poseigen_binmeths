import numpy as np
import pandas as pd
import math
import os
import itertools

from collections import Counter

import poseigen_seaside.basics as se


def BinParser(inp, uni, categorical = False, multi = False): 
    
    if multi is False: inp = [inp]

    pars = []
    for inpx in inp: 
        if categorical: par = [np.where(inpx == u)[0] for u in uni] 
        else: par = [np.where(np.logical_and(inpx>=u[0], inpx <= u[1]))[0] for u in uni]
        pars.append(par)
    
    if multi is False: pars = pars[0]
        
    return pars


def Binner(inp, uni = None): 

    #This is Binner_v2 in origninal binstuff. 
    
    #should be able to handle a multi-dim shape, bin it and return 
    categorical = False
    if isinstance(inp[0], str): categorical = True
    
    inpy = inp.reshape(-1)
    
    par = BinParser(inpy, uni, categorical = categorical)
    
    inpx = np.zeros(len(inpy), dtype = int)
    for ip, p in enumerate(par): 
        inpx[p] = ip
    
    inpx = inpx.reshape(inp.shape)
    
    return inpx 

def HistUni(inp, bins = 10): 
    inp = np.array(inp)
    bin_edges = np.linspace(inp.min(), inp.max(), bins + 1)
    uni = se.ListWindower(bin_edges, win_size = 2)
    return uni


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




def BinCount(bind, uni = None, multi = False, prop = False):
    #uni is shared
    #inp is a list of 1D arrays or lists of values

    if multi is False: bind = [bind]
    bind = [np.array(m) for m in bind]
    
    if uni is None: uni = np.sort(np.unique(bind))

    hs = [Counter(m.reshape(-1)) for m in bind]
    max_key = max(max(d) for d in hs)
    empty = dict.fromkeys(uni, 0)
    rere = [{**empty, **d} for d in hs]

    cnv = pd.DataFrame(rere).to_numpy()

    if multi is False: cnv = cnv[0] 

    if prop: cnv = cnv / np.sum(cnv)
    
    return cnv



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

    liny = mode[0](scalefactor * inpx, **mode[1])

    hist, bin_edges = np.histogram(liny, bins = bins, density=False)

    rt = mode_rev[0](bin_edges, **mode_rev[1]) / scalefactor

    rt[0] = mi
    rt[-1] = ma

    return se.ListWindower(rt, win_size = 2)











###########################################################

############# SPLIT TOOLS ####################





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




def SimpleStratifiedSplit(obs_bind, props = [0.5, 0.25, 0.25], prints = False): 
    uni = np.unique(obs_bind)
    props = np.array(props)
    props = props / np.sum(props)

    lpx = np.arange(len(props))

    splito = {ipo: [] for ipo in lpx}

    for unx in uni: 
        idxs = np.where(obs_bind == unx)[0]
        dd = np.random.choice(lpx, size = len(idxs), p = props, replace = True)
        for ipo in lpx: splito[ipo].append(idxs[dd == ipo])
    
    splito = [np.concatenate(splito[ipo]) for ipo in lpx]

    if prints: print([BinCount(obs_bind[s], uni = uni) for s in splito])

    return splito



class BinWeights: 

    def __init__(self, uni = None, multi = False, minus = True, newrange = True):

        self.uni = uni
        self.multi = multi
        self.minus = minus
        self.newrange = newrange
        
    def fit(self, bind): 

        multi = self.multi
        uni = self.uni
        minus = self.minus
        newrange = self.newrange

        if multi is False: bind = [bind]

        all_weights, unis = [], []
        for bindx in bind: 

            if uni is None: uni = np.sort(np.unique(bindx))
            unis.append(uni)
            cnv = BinCount(bindx, uni = uni, multi = False)
            cnv_prop = cnv / np.sum(cnv)

            weights = 1 - cnv_prop if minus else 1 / cnv_prop
            
            if newrange is not None: 
                mi, ma = se.MiMa(weights) 
                if isinstance(newrange, tuple): 

                    newmi, newma = newrange
                    weights = (((weights - mi) * (newma - newmi)) / (ma - mi)) + newmi
                
                elif newrange is True: 
                    weights = weights / ma
            
            all_weights.append(weights)
        
        self.all_weights = all_weights
        self.unis = unis
    
        return self
    
    def eval(self, inp, dtype = np.float32):
         
        if self.multi is False: inp = [inp]

        new_weights = []

        for inpx, weights, uni in zip(inp, self.all_weights, self.unis):
            weix = np.zeros(inpx.shape, dtype = dtype)
            for u,w in zip(uni, weights): weix[inpx == u] = w
            new_weights.append(weix)
            
        new_weights = np.stack(new_weights) if self.multi else new_weights[0]
        
        return new_weights


def BinWeighter(bind, onlyidx = None, byaxis = None, 
                uni = None, minus = True, newrange = True,
                dtype = np.float32):
    
    bind = np.array(bind) #just in case
    
    origshape = bind.shape

    inps = [bind] if byaxis is None else [bind.take(indices=a, axis=byaxis) for a in range(origshape[byaxis])]

    inpshape = inps[0].shape

    newei = []

    for j in inps: 
        bine = j if onlyidx is None else j[onlyidx]

        bw = BinWeights(uni = uni, multi = False, minus = minus, newrange = newrange).fit(bind = bine.reshape(-1))
        wo = bw.eval(j.reshape(-1), dtype=dtype).reshape(inpshape)
        newei.append(wo)
    
    newei = np.stack(newei, axis = byaxis) if byaxis else newei[0] 
    
    return newei


























