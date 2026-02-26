import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    p : (N,) predicted probabilities (0 < p < 1)
    y : (N,) binary labels {0,1}
    gamma : focusing parameter

    returns: mean focal loss (scalar)
    """

    # convert to numpy arrays
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # focal loss formula (vectorized)
    loss = - (1 - p)**gamma * y * np.log(p) \
           - (p**gamma) * (1 - y) * np.log(1 - p)

    return np.mean(loss)


# ---------------------------------------
# Example test cases
# ---------------------------------------
p = [0.9, 0.2, 0.7, 0.1]
y = [1, 0, 1, 0]
print("Loss:", focal_loss(p, y, gamma=2.0))

p = [0.5, 0.5, 0.5, 0.5]
y = [1, 0, 1, 0]
print("Loss:", focal_loss(p, y, gamma=2.0))

p = [0.9, 0.2, 0.7, 0.1]
y = [1, 0, 1, 0]
print("Loss:", focal_loss(p, y, gamma=0.0))   # equals cross entropy