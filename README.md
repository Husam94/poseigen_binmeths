# Binmeths

This package implements methods explored in the [Binning Methods paper](https://doi.org/).
poseigen_binmeths is the binning and split-generation package in the Poseigen family.
It provides utilities for bin construction, bin statistics, balanced sampling, and train/validation/test split strategies used across Poseigen workflows.

## Features

poseigen_binmeths is organized into three modules:

- bin_utils: bin parsing and assignment, histogram utilities, objective binning helpers, bin counts and densities, and weighted bin selection.
- split_utils: split generation utilities including random, window, stratified, and akin-based scoring/split selection methods.
- harpoon: balanced per-bin samplers including NumPy and PyTorch implementations, plus overlapping-bin sampling helpers.

Find Epoch Sampling [poseigen_trident](https://github.com/Husam94/poseigen_trident). 

Binning methods are used in the following case studies: 
- [Synthetic Right-skewed Data](https://github.com/Husam94/CS-Synth)
- [CTCF-DNA Interactions](https://github.com/Husam94/CS-CTCF)
- [YY1-DNA Interactions](https://github.com/Husam94/CS-YY1)
- [Enhancer Activity](https://github.com/Husam94/CS-EnhAct)



## Installation

Install from PyPI:

```bash
pip install poseigen_binmeths
```

For local development, install from source using your preferred editable-install workflow.

## Usage

Import from the package namespace (bin utilities are re-exported):

```python
import poseigen_binmeths as pbm
```

Or import modules directly:

```python
import poseigen_binmeths.bin_utils as bu
import poseigen_binmeths.split_utils as su
import poseigen_binmeths.harpoon as harp
```

Example: histogram binning and a random split.

```python
import numpy as np
import poseigen_binmeths.bin_utils as bu
import poseigen_binmeths.split_utils as su

x = np.random.randn(1000)
bind = bu.HistBinning(x, bins=20)
train_idx, test_idx = su.SplitGen_Random(len(bind), proportions=[0.8, 0.2])
```

## Project Status

poseigen_binmeths is in active development and is intended to support binning and sampling workflows across the Poseigen ecosystem.

## Related Projects

- poseigen_seaside: shared utilities and metrics foundation.
- poseigen_trident: model training workflows, including epoch sampling integrations.

## License

This project is released under the MIT License.