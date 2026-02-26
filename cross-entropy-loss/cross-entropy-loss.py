import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    y_true : list or array of labels (N,)
    y_pred : list or array of probabilities (N, C)
    """

    # convert to numpy arrays (fixes your error)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    N = len(y_true)

    # pick correct class probabilities
    correct_probs = y_pred[np.arange(N), y_true]

    # compute loss
    loss = -np.mean(np.log(correct_probs))

    return loss


# ---------------------------------------
# Example
# ---------------------------------------
y_true = [0, 2, 1]

y_pred = [
    [0.7, 0.2, 0.1],
    [0.1, 0.3, 0.6],
    [0.2, 0.5, 0.3]
]

print("Cross-Entropy Loss:", cross_entropy_loss(y_true, y_pred))