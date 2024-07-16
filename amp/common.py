import torch
from importlib.util import find_spec


def amp_definitely_not_available():
    return not (find_spec('torch_gcu'))
