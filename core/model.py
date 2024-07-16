#
# Copyright 2021-2022 Enflame. All Rights Reserved.
#
from __future__ import print_function

import copy
from typing import Union
import torch
import torch_gcu
from torch_gcu.core.device import _device_t, _get_device_index, _check_device_valid
from torch_gcu._GCUC import JitRunMode

__all__ = ['JitRunMode', 'synchronize', 'unlazy', 'sync_lived_tensor', 'optimizer_step',
           'collect_buffers', 'collect_amp_training_params', 'set_scalar_cached_enable',
           'fetch_gradients', 'clear_graph_caches']


def _assert_type_valid(obj, valid_type) -> bool:
    """Return if obj is one of given type

    Args:
      obj: Object to check
      valid_type: Check types
    Return:
      Return True if obj is a object in valid_type else False
    """
    if not isinstance(obj, valid_type):
        raise TypeError(
            "Param must be {}, got {}".format(valid_type, type(obj)))


def _is_gcu_tensor(tensor) -> bool:
    """Return if tensor is on device GCU

    Args:
      tensor: Object to check

    Return:
      return True if tensor is a torch.Tensor on GCU device else False
    """
    return isinstance(tensor, torch.Tensor) and tensor.device.type == "xla"


def _is_gcu_tensor_list_or_tuple(tensors) -> bool:
    """Return if tensor is a list/tuple of torch.Tensors on GCU device

    Args:
      tensor: Object to check

    Return:
      Return True if tensor is a list/tuple of torch.Tensors on GCU device
      or empty list/tuple, else False
    """
    return isinstance(tensors, (list, tuple)) and \
        (len(tensors) == 0 or all(_is_gcu_tensor(t) for t in tensors))


def _assert_gcu_tensor_list_or_tuple(tensors) -> None:
    """Raise error if input is not list/tuple of GCU tensor

    Args:
      tensor: Object to check
    """
    if not _is_gcu_tensor_list_or_tuple(tensors):
        raise TypeError("Tensors must be a list of GCU tensor.")


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on a GCU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`torch_gcu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    with torch_gcu.device(device):
        return torch_gcu._GCUC._synchronize()


"""
Note [JitRunMode]
SYNC, run graph synchronously,
ASYNC, run graph asynchronously, deliver graph without waiting, lead higher HBM peak
SAFEASYNC, run graph asynchronously, wait last graph finish before deliver graph
"""


def unlazy(tensors, mode=JitRunMode.SAFEASYNC):
    """Create a graph using given tensors as outputs

    Args:
      tensors (List/tuple[torch.Tensor]): List or tuple of `torch.Tensor`s to materialize.
        For each Tensor `t` in the list, `t.device` must be a `GCU` device.
        Can not empty.
      mode(JitRunMode): Run graph mode. See Note [JitRunMode].
    """
    _assert_gcu_tensor_list_or_tuple(tensors)
    torch_gcu._GCUC._sync_multi(
        tensors, graph_create_info="model.unlazy()", devices=-1, mode=mode)


def sync_lived_tensor(device: _device_t = None, mode=JitRunMode.SAFEASYNC):
    """Create and run a graph using all lived tensors on device as outputs

    Args:
      device: Specific device, if :attr:`device` is ``None``, this will use the current default GCU device
    device
      mode(JitRunMode): Run graph mode. See Note [JitRunMode].
    """
    device_id = _get_device_index(device, True)
    _check_device_valid(device_id)
    torch_gcu._GCUC._sync_live_tensors(device_id, mode)


def optimizer_step(optimizer: torch.optim.Optimizer, extra_output=None, optimizer_args={},
                   model: torch.nn.Module = None, mode=JitRunMode.SAFEASYNC, device: _device_t = None):
    """Run the provided optimizer step and issue the GCU device step computation.

    Args:
      optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance whose
        `step()` function needs to be called. The `step()` function will be called
        with the `optimizer_args` named arguments.
      extra_output: Only used when Param model is not None. A tensor or list/tuple of tensors,
             these tensor will add to outputs, this is useful when loss need to copy to host.
             Usually used for avoids graph sharding
      optimizer_args (dict, optional): Named arguments dictionary for the
                                       `optimizer.step()` call.
      model(:class:`torch.nn.Module` or None): model to collect tensor as output.
             If model is None, all tensors created on GCU will be outputs for graph.
             This is not recommended for performance.
             If model is passed in, all tensors need to be udpate will be outputs for graph.
      mode(JitRunMode): Run graph mode. See Note [JitRunMode].
      device: specify device to update, only used when model=None

    Returns:
      The same value returned by the `optimizer.step()` call.
    """
    _assert_type_valid(optimizer, torch.optim.Optimizer)
    loss = optimizer.step(**optimizer_args)
    params = []
    if model is not None:
        _assert_type_valid(model, torch.nn.Module)
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.data is not None:
                    params.append(p.data)
                    # add all state in optimizer to outputs
                    state = optimizer.state[p]
                    for v in state.values():
                        if _is_gcu_tensor(v):
                            params.append(v)
        if _is_gcu_tensor(extra_output):
            params.append(extra_output)
        elif _is_gcu_tensor_list_or_tuple(extra_output):
            params.extend(extra_output)
        params.extend(collect_buffers(model))

    if model is None:
        device_id = _get_device_index(device, True)
        _check_device_valid(device_id)
    else:
        device_id = -1

    torch_gcu._GCUC._sync_multi(
        params, "model.optimizer_step()", device_id, mode)

    return loss


def collect_buffers(model):
    """Return all buffers of the model.
    Args:
      model (torch.nn.Module): model to get buffers from
    Return:
      buffer tensors
    """
    _assert_type_valid(model, torch.nn.Module)
    buffers = []
    for buf in model.buffers():
        if isinstance(buf, torch.Tensor):
            buffers.append(buf.data)
    return buffers


def collect_amp_training_params(model, optimizer, scaler):
    """Collect amp training graph outputs form model && optimizer && scaler.
    Args:
      model (torch.nn.Module): model to get buffers from
      optimizer (torch.optim.Optimizer): optimizer to get training params from
      scaler (torch_gcu.amp.GradScaler): scalar to get amp training params from
    Return:
      training param tensors
    """
    _assert_type_valid(model, torch.nn.Module)
    _assert_type_valid(optimizer, torch.optim.Optimizer)
    _assert_type_valid(scaler, torch_gcu.amp.GradScaler)
    params = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.data is not None:
                params.append(p.data)
                state = optimizer.state[p]
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        params.append(v)
    params.extend(collect_buffers(model))
    params += [scaler._scale, scaler._growth_tracker]
    return params


def set_scalar_cached_enable(enabled: bool):
    """should we enabled the scalar paramter cache to avoid unnecessary d2h operation.
    Args:
      enabled (bool): enable scalar paramter cache or disable.
    Note:
      thread unsafe, please use it as env, like:
      import torch_gcu
      torch_gcu.set_scalar_cached_enable(enabled)
    """
    torch_gcu._GCUC._set_scalar_cached_enable(bool(enabled))


def fetch_gradients(optimizer):
    """Return all gradients of weights in optimizer update list.
    Args:
      optimizer (torch.optimizer): optimizer to get gradients from
    Return:
      Gradients tensors
    """
    _assert_type_valid(optimizer, torch.optim.Optimizer)
    gradients = []
    for param_group in optimizer.__getstate__()['param_groups']:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        gradients.append(p.grad.data)
    return gradients


def clear_graph_caches():
    """Clear graph caches."""
    torch_gcu._GCUC._clear_graph_caches()