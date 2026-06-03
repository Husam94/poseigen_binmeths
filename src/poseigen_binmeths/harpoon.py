

import numpy as np
import torch




# def Harpoon(bind, select, 
#             multi = False, 
#             repeat = 1, custidx = None): 
    
#     #BIND IS NOW INPUT! DOES NOT DO BINNING ANY MORE

#     #BIND FORMAT IS [LENGTH, TYPE] FOR THIS ONE
    
#     #OCT 2 23, ADDED REPEAT FUNCTION FOR HARPOON TRAINER 
#     ##cust idx is custom idx 
    
#     #Added Multi feature (June 2) 
#     #for multi #Inp is Observations as rows and each output as a column, else its a 1dim array 

#     if select < 1: select = np.round(len(bind) * select).astype(int)
    
#     if multi: 
#         h2, h1 = np.unique(bind, axis = 0, return_counts = True)
#         div = [np.where((bind == u).all(axis = 1))[0] for u in h2] 
#     else: 
#         h2, h1 = np.unique(bind, return_counts = True)
#         div = [np.where(bind == u)[0] for u in h2]

#     if custidx is not None: div = [np.array(custidx)[d] for d in div]
    
#     lu = len(h2)
#     nums = np.repeat(0, lu)
    
#     j = 0 
#     for _ in range(select):
#         for i in np.random.permutation(np.arange(lu)):
#             if j == select: break 
#             elif h1[i] > 0:
#                 nums[i] += 1
#                 h1[i] -= 1
#                 j += 1 
        
#     idx = np.stack([np.hstack([np.random.choice(d, replace = False, size = w) for d,w in zip(div, nums)]) 
#                     for _ in range(repeat)], 0)
    
#     return idx[0] if repeat == 1 else idx


def Harpoon(bind, select, repeat=1, custidx=None, random_state=None):
    """
    Efficient 1D-only Harpoon sampler.

    Parameters
    ----------
    bind : array-like, shape (N,)
        Bin label for each observation.
    select : float or int
        If < 1, interpreted as proportion of N. Otherwise number of points.
    repeat : int
        Number of repeated samples to draw.
    custidx : array-like, optional
        External indices mapped onto bind positions.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    ndarray
        Shape (select,) if repeat==1, else (repeat, select).
    """
    if random_state is not None:
        np.random.seed(random_state)

    bind = np.asarray(bind).reshape(-1)
    n = bind.size

    if n == 0:
        return np.array([], dtype=int) if repeat == 1 else np.empty((repeat, 0), dtype=int)

    # Convert select to integer count.
    select = int(np.round(n * select)) if select < 1 else int(select)
    select = max(0, min(select, n))
    repeat = int(repeat)

    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    if select == 0:
        return np.array([], dtype=int) if repeat == 1 else np.empty((repeat, 0), dtype=int)

    # Unique bins + inverse map in one pass.
    _, inv, counts = np.unique(bind, return_inverse=True, return_counts=True)
    n_bins = counts.size

    # Build per-bin index groups efficiently via sorting by inverse labels.
    order = np.argsort(inv, kind="stable")
    groups = np.split(order, np.cumsum(counts)[:-1])

    if custidx is not None:
        custidx = np.asarray(custidx)
        if custidx.shape[0] != n:
            raise ValueError("custidx must have the same length as bind")
        groups = [custidx[g] for g in groups]

    # Allocate balanced counts per bin (fast, capacity-aware).
    nums = np.minimum(counts, select // n_bins)
    rem = select - nums.sum()

    # Distribute remainder randomly among bins that still have capacity.
    while rem > 0:
        avail = np.flatnonzero(counts > nums)
        take = min(rem, avail.size)
        chosen = np.random.choice(avail, size=take, replace=False)
        nums[chosen] += 1
        rem -= take

    # Draw samples for each repeat.
    out = np.empty((repeat, select), dtype=np.asarray(groups[0]).dtype)
    for r in range(repeat):
        chunks = []
        for g, w in zip(groups, nums):
            if w:
                g = np.asarray(g)
                # Faster than np.random.choice for without-replacement subset.
                chunks.append(np.random.permutation(g)[:w])
        out[r] = np.concatenate(chunks)

    return out[0] if repeat == 1 else out


def Harpoon_torch(bind, select, repeat=1, custidx=None, random_state=None, num_bins=None):
    """
    Efficient 1D-only Harpoon sampler (PyTorch, GPU) with optional histogram binning.

    Parameters
    ----------
    bind : torch.Tensor, shape (N,)
        Bin label for each observation (on GPU).
    select : float or int
        If < 1, interpreted as proportion of N. Otherwise number of points.
    repeat : int
        Number of repeated samples to draw.
    custidx : torch.Tensor, optional
        External indices mapped onto bind positions (on GPU).
    random_state : int, optional
        Seed for reproducibility.
    num_bins : int, optional
        If not None, bin the bind values into num_bins histogram bins.

    Returns
    -------
    torch.Tensor
        Shape (select,) if repeat==1, else (repeat, select).
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    bind = bind.reshape(-1)
    n = bind.size(0)

    if n == 0:
        dtype = torch.int64
        device = bind.device
        return torch.empty((repeat, 0), dtype=dtype, device=device) if repeat > 1 else torch.empty(0, dtype=dtype, device=device)

    select = int(round(n * select)) if select < 1 else int(select)
    select = max(0, min(select, n))
    repeat = int(repeat)

    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if select == 0:
        dtype = torch.int64
        device = bind.device
        return torch.empty((repeat, 0), dtype=dtype, device=device) if repeat > 1 else torch.empty(0, dtype=dtype, device=device)

    # Histogram binning if num_bins is specified
    if num_bins is not None:
        min_val = bind.min().item()
        max_val = bind.max().item()
        # Avoid zero-width bins
        if min_val == max_val:
            bin_indices = torch.zeros_like(bind, dtype=torch.long)
        else:
            # Compute bin edges
            bin_edges = torch.linspace(min_val, max_val, steps=num_bins+1, device=bind.device)
            # Assign each value to a bin
            bin_indices = torch.bucketize(bind, bin_edges[:-1], right=False)
        bind_for_grouping = bin_indices
    else:
        bind_for_grouping = bind

    # Unique bins + inverse map + counts
    unique, inv = torch.unique(bind_for_grouping, return_inverse=True)
    counts = torch.bincount(inv)
    n_bins = counts.size(0)

    # Build per-bin index groups
    order = torch.argsort(inv)
    splits = torch.cumsum(counts, dim=0)[:-1]
    groups = torch.split(order, tuple(counts.tolist()))

    if custidx is not None:
        if custidx.shape[0] != n:
            raise ValueError("custidx must have the same length as bind")
        groups = [custidx[g] for g in groups]

    nums = torch.minimum(counts, torch.full_like(counts, select // n_bins))
    rem = select - nums.sum().item()

    # Distribute remainder randomly among bins with capacity
    while rem > 0:
        avail = (counts > nums).nonzero(as_tuple=True)[0]
        take = min(rem, avail.size(0))
        if take == 0:
            break
        chosen = avail[torch.randperm(avail.size(0))[:take]]
        nums[chosen] += 1
        rem -= take

    dtype = custidx.dtype if custidx is not None else torch.int64
    device = bind.device
    out = torch.empty((repeat, select), dtype=dtype, device=device)

    for r in range(repeat):
        chunks = []
        for g, w in zip(groups, nums):
            if w > 0:
                g = g.clone()
                perm = torch.randperm(g.size(0), device=device)
                chunks.append(g[perm[:w]])
        out[r] = torch.cat(chunks)

    return out[0] if repeat == 1 else out



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
    else:
        select = int(select)
    select = max(0, min(select, N))

    if num_bins == 0 or select == 0:
        return np.array([], dtype=int) if repeat == 1 else [np.array([], dtype=int) for _ in range(repeat)]

    all_selected = []
    for _ in range(repeat):
        selected_mask = np.zeros(N, dtype=int)
        selected_count = 0
        while selected_count < select:
            made_progress = False
            for bin_idx in np.random.permutation(np.arange(num_bins)):
                # Find indices where not yet selected and belongs to this bin
                belongs = (window_belong == bin_idx).any(axis=1)
                candidates = np.where((selected_mask == 0) & belongs)[0]
                if len(candidates) > 0:
                    pick = np.random.choice(candidates, 1, replace=False)[0]
                    selected_mask[pick] = 1
                    selected_count += 1
                    made_progress = True
                    if selected_count == select:
                        break
            if not made_progress:
                break
        all_selected.append(np.where(selected_mask == 1)[0])
    return all_selected[0] if repeat == 1 else all_selected





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
