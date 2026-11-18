"""EML unit vs. perceptron: single-unit capability study.

See docs/designs/eml_single_unit_study.md. Tensors created without an
explicit dtype default to float64 from here on.
"""

import torch

torch.set_default_dtype(torch.float64)
