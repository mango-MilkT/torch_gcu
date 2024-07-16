#
# Copyright 2021-2022 Enflame. All Rights Reserved.
#
import torch
import torch._tensor_str
from torch._tensor_str import (get_summarized_data, _Formatter,
                               _tensor_str_with_formatter, PRINT_OPTS)

def is_gcu(self):
    return self.device.type == 'xla'

# Add is_gcu function to torch.Tensor
setattr(torch.Tensor, 'is_gcu', is_gcu)

def _tensor_str_gcu(self, indent):
    if self.is_gcu():
        self = self.cpu()

    if self.numel() == 0:
        return '[]'

    if self.has_names():
        # There are two main codepaths (possibly more) that tensor printing goes through:
        # - tensor data can fit comfortably on screen
        # - tensor data needs to be summarized
        # Some of the codepaths don't fully support named tensors, so we send in
        # an unnamed tensor to the formatting code as a workaround.
        self = self.rename(None)

    summarize = self.numel() > PRINT_OPTS.threshold

    # handle the negative bit
    if self.is_neg():
        self = self.resolve_neg()

    if self.dtype is torch.float16 or self.dtype is torch.bfloat16:
        self = self.float()

    if self.dtype.is_complex:
        # handle the conjugate bit
        self = self.resolve_conj()
        real_formatter = _Formatter(get_summarized_data(
            self.real) if summarize else self.real)
        imag_formatter = _Formatter(get_summarized_data(
            self.imag) if summarize else self.imag)
        return _tensor_str_with_formatter(self, indent, summarize, real_formatter, imag_formatter)
    else:
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        return _tensor_str_with_formatter(self, indent, summarize, formatter)

# Override _tensor_str for print GCU tensor
torch._tensor_str._tensor_str = _tensor_str_gcu

def _rebuild_device_tensor_from_numpy_gcu(data, dtype, device, requires_grad):
    if device.startswith('xla'):
        tensor = torch.from_numpy(data).to(dtype=dtype)
        import torch_gcu
        current_device = 'xla:' + str(torch_gcu.current_device())
        if device == current_device:
            tensor = tensor.to(device)
    else:
        tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor

# Override _rebuild_device_tensor_from_numpy for torch.load can work properly
torch._utils._rebuild_device_tensor_from_numpy = _rebuild_device_tensor_from_numpy_gcu