import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    anchor   : (N,D) or (D,)
    positive : (N,D) or (D,)
    negative : (N,D) or (D,)
    margin   : margin parameter

    returns: scalar mean triplet loss
    """

    # convert to numpy
    a = np.array(anchor, dtype=float)
    p = np.array(positive, dtype=float)
    n = np.array(negative, dtype=float)

    # ensure 2D shape (handles single vector case)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if n.ndim == 1:
        n = n.reshape(1, -1)

    # squared Euclidean distances
    d_ap = np.sum((a - p) ** 2, axis=1)
    d_an = np.sum((a - n) ** 2, axis=1)

    # triplet loss per sample
    loss = np.maximum(0.0, d_ap - d_an + margin)

    return float(np.mean(loss))


# ---------------------------------------
# Example tests
# ---------------------------------------
print(triplet_loss([[1,0]], [[2,0]], [[5,0]], margin=1.0))  # 0.0
print(triplet_loss([[0,0]], [[3,0]], [[1,0]], margin=1.0))  # 9.0
print(triplet_loss([1,0], [2,0], [5,0], margin=1.0))        # 0.0