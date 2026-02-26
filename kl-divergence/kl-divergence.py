import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    p : array-like (N,) reference distribution
    q : array-like (N,) approximate distribution
    eps : numerical stability term

    returns: scalar KL divergence D_KL(P || Q)
    """

    # convert to numpy arrays
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # add epsilon to avoid log(0)
    q = q + eps

    # compute elementwise KL (only where p > 0)
    kl = np.where(p > 0, p * np.log(p / q), 0.0)

    return float(np.sum(kl))


# ---------------------------------------
# Example tests
# ---------------------------------------
print(kl_divergence([0.4, 0.6], [0.5, 0.5]))  # ≈ 0.0201
print(kl_divergence([0.3, 0.7], [0.3, 0.7]))  # 0.0
print(kl_divergence([0.9, 0.1], [0.5, 0.5]))  # ≈ 0.368