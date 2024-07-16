#
# Copyright 2020-2021 Enflame. All Rights Reserved.
#

import torch
import torch_gcu
from typing import Optional, Any, Union
from torch import device as _device
from torch_gcu.utils import LazyProperty

_device_t = Union[_device, str, int, None]
__all__ = ['is_available', 'device_count', 'gcu_device', 'device', 'device_of',
           'set_device', 'current_device', 'get_device_name']

_DEVICES_COUNT = LazyProperty(lambda: torch_gcu._GCUC._get_device_count())


def _device_valid(id: int) -> bool:
    return id >= 0 and id < _DEVICES_COUNT.value


def _check_device_valid(id: int) -> bool:
    if not _device_valid(id):
        raise IndexError(
            "Invalid device id, must in [0, {}], got {}".format(_DEVICES_COUNT.value - 1, id))


def _get_device_index(device: Any, optional: bool = False) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a GCU device. Note that for a GCU device without a specified index,
    i.e., ``torch.device('xla')``, this will return the current default GCU
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default GCU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None
    if isinstance(device, torch.device):
        if device.type != 'xla':
            raise ValueError(
                'Expected a gcu device, but got: {}'.format(device))
        device_idx = device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            device_idx = torch_gcu._GCUC._get_current_device_id()
        else:
            raise ValueError('Expected a torch.device with a specified index '
                             'or an integer, but got:{}'.format(device))
    return device_idx


def is_available() -> bool:
    r"""Returns a bool indicating if GCU is currently available."""
    return _DEVICES_COUNT.value > 0


def device_count() -> int:
    r"""Returns the number of GCUs available."""
    return _DEVICES_COUNT.value


def gcu_device(device: _device_t = None) -> torch.device:
    r"""Returns torch.device reference to a GCU device

    Args:
        device (torch.device or int): device must has type xla if it is a torch.device.
    """
    device_id = _get_device_index(device, optional=True)
    return torch.device('xla:{}'.format(device_id))


class device(object):
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if not _device_valid(self.idx):
            return
        self.prev_idx = torch_gcu.current_device()
        if self.prev_idx != self.idx:
            set_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.prev_idx != self.idx:
            set_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GCU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        dev = obj.device
        if dev.type == 'xla':
            idx = dev
        else:
            idx = -1
        super(device_of, self).__init__(idx)


def set_device(device: _device_t) -> None:
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``TOPS_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device_id = _get_device_index(device)
    if _device_valid(device_id):
        torch_gcu._GCUC._set_current_device(device_id)


def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    if is_available():
        return torch_gcu._GCUC._get_current_device_id()


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch_gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    device_id = _get_device_index(device, optional=True)
    _check_device_valid(device_id)
    # Do not support different type GCU device in same machine
    return torch_gcu._GCUC._get_devices_target_name()
