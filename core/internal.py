#
# Copyright 2021-2023 Enflame. All Rights Reserved.
#

# This file is only for internal user
import torch_gcu
from torch_gcu.core.device import _device_t, _get_device_index, _check_device_valid


def get_exe_default_stream(device: _device_t = None):
    r"""Return default executable running stream of given device

    Args:
        device: Specific device, if :attr:`device` is ``None``, this will use the current default GCU device
    Return:
        Int value, reinterpret cast from stream(void*)
    """
    device_id = _get_device_index(device, True)
    _check_device_valid(device_id)
    return torch_gcu._GCUC._get_exe_default_stream(device_id)


def get_dma_default_stream(device: _device_t = None):
    r"""Return default dma stream of given device

    Args:
        device: Specific device, if :attr:`device` is ``None``, this will use the current default GCU device
    Return:
        Int value, reinterpret cast from stream(void*)
    """
    device_id = _get_device_index(device, True)
    _check_device_valid(device_id)
    return torch_gcu._GCUC._get_dma_default_stream(device_id)
