#
# Copyright 2021 Enflame. All Rights Reserved.
#
import warnings

from . import comm
from torch.autograd import Function
from torch._utils import _get_device_index


class Broadcast(Function):

    @staticmethod
    def forward(ctx, target_dtus, *inputs):
        assert all(map(lambda i: i.device.type != 'cpu', inputs)), (
            'Broadcast function not implemented for CPU tensors'
        )
        target_dtus = list(map(lambda x: _get_device_index(x, True), target_dtus))
        ctx.target_dtus = target_dtus
        if len(inputs) == 0:
            return tuple()
        ctx.num_inputs = len(inputs)
        ctx.input_device = inputs[0].get_device()
        outputs = comm.broadcast_coalesced(inputs, ctx.target_dtus)
        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)


class ReduceAddCoalesced(Function):

    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_dtus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]

        grads = [grads[i:i + num_inputs]
                 for i in range(0, len(grads), num_inputs)]
        return comm.reduce_add_coalesced(grads, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None,) + Broadcast.apply(ctx.target_dtus, *grad_outputs)


class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(map(lambda i: i.device.type != 'cpu', inputs)), (
            'Gather function not implemented for CPU tensors'
        )
        target_device = _get_device_index(target_device, True)
        ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn('Was asked to gather along dimension 0, but all '
                          'input tensors were scalars; will instead unsqueeze '
                          'and return a vector.')
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(map(lambda i: i.size(ctx.dim), inputs))
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):

    @staticmethod
    def forward(ctx, target_dtus, chunk_sizes, dim, input):
        target_dtus = list(map(lambda x: _get_device_index(x, True), target_dtus))
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        outputs = comm.scatter(input, target_dtus, chunk_sizes, ctx.dim)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)
