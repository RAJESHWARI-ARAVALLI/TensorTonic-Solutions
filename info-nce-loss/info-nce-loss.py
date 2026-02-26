import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Z1 : (N, D) first embedding batch
    Z2 : (N, D) second embedding batch
    temperature : τ > 0

    returns: scalar mean InfoNCE loss
    """

    # convert to numpy
    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)

    # similarity matrix (N x N)
    S = (Z1 @ Z2.T) / temperature

    # numerically stable softmax (row-wise)
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    probs = exp_S / np.sum(exp_S, axis=1, keepdims=True)

    # positive pair probabilities (diagonal)
    pos_probs = np.diag(probs)

    # InfoNCE loss
    loss = -np.mean(np.log(pos_probs))

    return float(loss)


# ---------------------------------------
# Example tests
# ---------------------------------------
Z1 = [[1, 0], [0, 1]]
Z2 = [[1, 0], [0, 1]]
print("Loss (aligned):", info_nce_loss(Z1, Z2, temperature=0.1))

Z2 = [[0, 1], [1, 0]]
print("Loss (misaligned):", info_nce_loss(Z1, Z2, temperature=0.1))

Z2 = [[1, 0], [0, 1]]
print("Loss (temp=1):", info_nce_loss(Z1, Z2, temperature=1.0))