import torch
import functools
import collections
import warnings
try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
from torch._six import string_classes

from .common import amp_definitely_not_available


class autocast(object):
    r"""
    See :class:`torch.autocast`.
    ``torch.cuda.amp.autocast(args...)`` is equivalent to ``torch.autocast("cuda", args...)``
    """

    def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
        super().__init__()
        self._cache_enabled = torch.is_autocast_cache_enabled()
        if amp_definitely_not_available():
            warnings.warn(
                'User provided device_type of \'xla\', but GCU is not available. Disabling')
            enabled = False
        self.fast_dtype = dtype
        self._cache_enabled = cache_enabled
        if self.fast_dtype != torch.float16 and self.fast_dtype != torch.bfloat16:
            raise RuntimeError('Current GCU Device does not support {}. Please switch dtype to float16 && bfloat16.'.format(
                str(self.fast_dtype)))
        self._enabled = enabled

    def __enter__(self):
        self.prev_cache_enabled = torch.is_autocast_cache_enabled()
        self.prev = torch.is_autocast_enabled()
        self.prev_fastdtype = torch.get_autocast_gpu_dtype()
        torch.set_autocast_gpu_dtype(self.fast_dtype)
        torch.set_autocast_enabled(self._enabled)
        torch.autocast_increment_nesting()
        torch.set_autocast_cache_enabled(self._cache_enabled)

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.prev)
        torch.set_autocast_gpu_dtype(self.prev_fastdtype)
        torch.set_autocast_cache_enabled(self.prev_cache_enabled)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (value.is_floating_point() and value.is_cuda and (
            value.dtype is not torch.float64))
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, string_classes):
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = map(lambda v: _cast(v, dtype), value)
        if isinstance(value, list) or isinstance(value, tuple):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


# custom_fwd is a decorator that may or may not be used with arguments, following
# https://github.com/dabeaz/python-cookbook/tree/master/src/9/defining_a_decorator_that_takes_an_optional_argument.
# this works:
#     @custom_fwd
#     def forward(...):
# this also works:
#     @custom_fwd(cast_inputs=torch.float)
#     def forward(...):
# TODO:  when python 2 support is dropped, change the signature to
# def custom_fwd(fwd=None, *, cast_inputs=None) with internal changes following the link above.
def custom_fwd(fwd=None, **kwargs):
    """
    Helper decorator for ``forward`` methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).  See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point CUDA Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if fwd is None:
        if len(kwargs) == 0:
            cast_inputs = None
        else:
            assert len(kwargs) == 1
            cast_inputs = kwargs["cast_inputs"]
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    if len(kwargs) == 0:
        cast_inputs = None
    else:
        assert len(kwargs) == 1
        cast_inputs = kwargs["cast_inputs"]

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled()
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(enabled=False):
                    return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
            else:
                return fwd(*args, **kwargs)
    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd):
    """
    Helper decorator for backward methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.
    """
    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(args[0]._fwd_used_autocast):
            return bwd(*args, **kwargs)
    return decorate_bwd
