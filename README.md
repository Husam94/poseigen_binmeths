# Binmeths

[![PyPI version](https://img.shields.io/pypi/v/poseigen_binmeths)](https://pypi.org/project/poseigen_binmeths/)
[![Python versions](https://img.shields.io/pypi/pyversions/poseigen_binmeths)](https://pypi.org/project/poseigen_binmeths/)
[![License](https://img.shields.io/pypi/l/poseigen_binmeths)](https://pypi.org/project/poseigen_binmeths/)
[![bioRxiv preprint](https://img.shields.io/badge/preprint-bioRxiv-B31B1B)](https://www.biorxiv.org/content/10.1101/2025.06.26.66188)

`poseigen_binmeths` is the binning and split-generation package in the [Poseigen ecosystem](https://github.com/Husam94/poseigen). It provides utilities for bin construction, bin statistics, balanced sampling, and train/validation/test split strategies used across Poseigen workflows.

See the [Binning Methods preprint](https://www.biorxiv.org/content/10.1101/2025.06.26.66188) for the methods implemented here.

## Installation

Requires Python 3.12 or newer. Install from PyPI:

```bash
python -m pip install poseigen_binmeths
```

## Package

poseigen_binmeths is organized into three modules:

- bin_utils: bin parsing and assignment, histogram utilities, objective binning helpers, bin counts and densities, and weighted bin selection.
- split_utils: split generation utilities including random, window, stratified, and akin-based scoring/split selection methods.
- harpoon: balanced per-bin samplers including NumPy and PyTorch implementations, plus overlapping-bin sampling helpers.

For epoch-sampling integrations, see [poseigen_trident](https://github.com/Husam94/poseigen_trident).

## Case studies

- **Synthetic data**
  - [Synthetic right-skewed data](https://github.com/Husam94/CS-Synth)
- **Genomic applications**
  - [CTCF-DNA interactions](https://github.com/Husam94/CS-CTCF)
  - [YY1-DNA interactions](https://github.com/Husam94/CS-YY1)
  - [Enhancer activity](https://github.com/Husam94/CS-EnhAct)


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