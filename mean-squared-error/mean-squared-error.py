import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    y_pred : array-like (N,)
    y_true : array-like (N,)

    returns: float MSE or None if shape mismatch
    """

    # convert to numpy arrays
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    # check shape mismatch
    if y_pred.shape != y_true.shape:
        return None

    # compute mse
    mse = np.mean((y_pred - y_true) ** 2)

    return float(mse)


# ---------------------------------------
# Example tests
# ---------------------------------------
print(mean_squared_error([2, 3], [1, 1]))        # 2.5
print(mean_squared_error([0, 0, 0], [0, 0, 0]))  # 0.0