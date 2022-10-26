r"""Functional interface"""
from typing import Callable, List, Optional, Tuple, Union
import math
import warnings

import torch
from torch import _VF
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes
# A workaround to support both TorchScript and MyPy:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

from torch._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch.nn import _reduction as _Reduction
from torch.nn import grad  # noqa: F401
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default

Tensor = torch.Tensor
conv1d = _add_docstr(
    torch.conv1d,
    r"""
conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
Applies a 1D convolution over an input signal composed of several input
planes.""")