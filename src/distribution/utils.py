import numpy as np


def get_v_indices(d, rank):
    idx1 = np.concatenate([np.arange(d)] * rank)
    idx2 = np.concatenate([np.full(d, r) for r in range(rank)])
    return (idx1, idx2)