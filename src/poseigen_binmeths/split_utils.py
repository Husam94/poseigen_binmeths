import numpy as np
import os

import poseigen_seaside.basics as se
import poseigen_binmeths.bin_utils as bu



#---- Basics ----

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


def SplitGen_Stratified(bind, num_splits = 1, proportions = [0.7, 0.3]):
    #bind is a 1d list of binned
    #does not do multi currently 

    #Stratified random split

    uni = np.sort(np.unique(bind))
    idxs = [np.where(bind == u)[0] for u in uni]

    counts = bu.BinCount(bind, uni=uni, multi = False)    
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
        print([bu.BinCount(obs_bind[s], uni=uni) for s in splits])

    return splits


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




# ---- AKIN SPLIT [aka AbsRel split] ---- 


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
        d2v = [bu.BinCount(v, uni = uni, multi = True) for v in vals]

        j = bu.BinCount(bindx, uni = uni).reshape(-1) if closest else None

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
    Splits2 = None
    
    
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
    
    if pathname is not None and os.path.isfile(temppath):
        os.remove(temppath)

    if Splits2 is None:
        raise ValueError('No splits were generated. Increase rando or atatime so at least one round runs.')

    return Splits2


def SubSample_Akin(bind, 
                   proportion = 0.3, num_subsamples = 1, rando = 10, group = None,
                   
                   uni = None, multi = False,
                   closest = False,
                   onlyidx = None, p = 2, reciprocal = False, pseudo = True, weight_bymem = False,
                   summarize_mode = [se.Epsilon, {}]): 
    
    
    ss = se.SubSample_Random(bind, proportion = proportion, num_subsamples = rando, group = group)
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