import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    """
    x1, x2 : array-like vectors
    label  : +1 (similar) or -1 (dissimilar)
    margin : margin for dissimilar pairs

    returns: scalar cosine embedding loss
    """

    # convert to numpy arrays
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # cosine similarity
    cos_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    # loss based on label
    if label == 1:
        loss = 1 - cos_sim
    else:  # label == -1
        loss = max(0.0, cos_sim - margin)

    return float(loss)


# ---------------------------------------
# Example tests
# ---------------------------------------
print(cosine_embedding_loss([1, 0, 0], [1, 0, 0], 1))  # 0.0
print(cosine_embedding_loss([1, 0, 0], [0, 1, 0], 1))  # 1.0