import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    y_true : array-like true values
    y_pred : array-like predicted values
    delta  : threshold parameter

    returns: scalar mean Huber loss
    """

    # convert to numpy arrays
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # compute error
    e = y_true - y_pred
    abs_e = np.abs(e)

    # piecewise huber formula
    quadratic = 0.5 * e**2
    linear = delta * (abs_e - 0.5 * delta)

    loss = np.where(abs_e <= delta, quadratic, linear)

    return float(np.mean(loss))


# ---------------------------------------
# Example tests
# ---------------------------------------
print(huber_loss([1,2,3], [1.5,1.7,2.5], delta=1.0))  # ≈ 0.0983
print(huber_loss([0,5], [2,8], delta=1.0))            # 2.0
print(huber_loss([1,2], [1,2], delta=1.0))            # 0.0