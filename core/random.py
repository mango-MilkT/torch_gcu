#
# Copyright 2021-2022 Enflame. All Rights Reserved.
#
import torch_gcu


def manual_seed_all(seed: int):
    r"""Sets the seed for generating random numbers for all GCUs.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)
    if seed < 0 or seed > 4294967295:
        raise ValueError(
            "Seed must be a unint32 number which is between 0~4294967295, got {}".format(seed))

    torch_gcu._GCUC._set_manual_seed_all(seed)
