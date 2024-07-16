#
# Copyright 2023 Enflame. All Rights Reserved.
#
import torch_gcu

def get_version():
    # get TOPS_VERSION as torch_gcu version
    version = torch_gcu._GCUC._get_version()
    return version
