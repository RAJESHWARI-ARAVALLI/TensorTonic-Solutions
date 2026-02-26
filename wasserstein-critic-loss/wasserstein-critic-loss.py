import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    real_scores : array-like critic outputs for real samples
    fake_scores : array-like critic outputs for fake samples

    returns: scalar critic loss
    """

    # convert to numpy arrays
    real_scores = np.array(real_scores, dtype=float)
    fake_scores = np.array(fake_scores, dtype=float)

    # compute loss
    loss = np.mean(fake_scores) - np.mean(real_scores)

    return float(loss)


# ---------------------------------------
# Example tests
# ---------------------------------------
print(wasserstein_critic_loss([2.0, 1.5, 3.0], [-1.0, 0.0, 0.5]))  # ≈ -2.333
print(wasserstein_critic_loss([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]))    # 0.0
print(wasserstein_critic_loss([0.0, 0.0], [1.0, 2.0, 3.0]))         # 2.0