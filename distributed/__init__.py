#
# Copyright 2019-2021 Enflame. All Rights Reserved.
#
import torch
if not torch.cuda.is_available():
    from .distributed_c10d import *
    from torch_gcu._GCUC import _register_comm_hook, _GradBucket, Reducer,\
        _compute_bucket_assignment_by_size, _broadcast_coalesced_with_group
    __all__ = ['_register_comm_hook', '_GradBucket', 'Reducer', '_compute_bucket_assignment_by_size',
            '_broadcast_coalesced_with_group']

    def is_available():

        return True

    if is_available():
        from .distributed_c10d import (
            _all_gather_base,
            _reduce_scatter_base,
        )