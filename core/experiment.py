import torch_gcu

def set_graph_threshold(threshold: int):
    r"""Lazy mode will accumulate the torch op. This function is used to set the
    threshold of the number of ops which the jit will run the graph. This value is only
    effect in PT_EVALUATE_ALL_LIVE_TENSORS mode. The count of torch op will be clear
    when threshold is been set to -1.

    Args:
        threshold (int): The threshold of how many torch op to accumulate
          before run the graph.
    """
    torch_gcu._GCUC._set_graph_threshold(threshold)
