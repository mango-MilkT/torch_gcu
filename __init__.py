#
# Copyright 2021-2022 Enflame. All Rights Reserved.
#
import torch
import atexit
from . import _GCUC
from .core import pytorch_override
from .core.random import *
from .core.model import *
from .core.device import *
from .core.debug import *
from .core.experiment import *
from .core.version import *
from .utils import data
from . import nn
from . import amp


def _prepare_to_exit():
    _GCUC._prepare_to_exit()


_GCUC._initialize_aten_bindings()
atexit.register(_prepare_to_exit)
