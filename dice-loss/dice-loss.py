import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    p : array-like predicted probabilities (N,) or (H,W)
    y : array-like binary ground truth mask (same shape)
    eps : smoothing term

    returns: scalar Dice loss
    """

    # convert to float numpy arrays
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # flatten (handles both 1D & 2D cases safely)
    p = p.reshape(-1)
    y = y.reshape(-1)

    # compute intersection and sums
    intersection = np.sum(p * y)
    union = np.sum(p) + np.sum(y)

    # dice coefficient
    dice = (2 * intersection + eps) / (union + eps)

    # dice loss
    return 1 - dice


# ---------------------------------------
# Example tests
# ---------------------------------------
print(dice_loss([0.9, 0.7, 0.1, 0.0], [1, 1, 0, 0]))   # ≈ 0.135
print(dice_loss([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]))   # 0.0
print(dice_loss([1.0, 1.0], [0, 0]))                   # 1.0