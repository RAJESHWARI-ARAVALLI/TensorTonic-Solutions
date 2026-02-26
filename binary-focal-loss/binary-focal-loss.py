import numpy as np

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    """
    predictions : (N,) predicted probabilities (0 < p < 1)
    targets     : (N,) binary labels {0,1}
    alpha       : balancing factor
    gamma       : focusing parameter

    returns: mean binary focal loss
    """

    # convert to numpy arrays
    p = np.array(predictions, dtype=float)
    y = np.array(targets, dtype=float)

    # compute p_t
    p_t = np.where(y == 1, p, 1 - p)

    # focal loss
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)

    return float(np.mean(loss))


# ---------------------------------------
# Example tests
# ---------------------------------------
print(binary_focal_loss([0.9], [1], alpha=1.0, gamma=2.0))  # ≈ 0.000263
print(binary_focal_loss([0.1], [1], alpha=1.0, gamma=2.0))  # ≈ 1.8631