import numpy as np

def label_smoothing_loss(predictions, target, epsilon=0.1):
    """
    predictions : array-like (K,) probability distribution
    target      : int (correct class index)
    epsilon     : smoothing factor

    returns: scalar loss
    """

    # convert to numpy
    p = np.array(predictions, dtype=float)

    K = len(p)

    # build smoothed target distribution q
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)

    # cross-entropy
    loss = -np.sum(q * np.log(p))

    return float(loss)


# ---------------------------------------
# Example tests
# ---------------------------------------
print(label_smoothing_loss([0.9, 0.05, 0.05], 0, 0.1))  # ≈ 0.3984
print(label_smoothing_loss([0.7, 0.3], 0, 0.2))         # ≈ 0.4413