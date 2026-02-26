import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean"):
    """
    y_true  : (N,) labels in {-1, +1}
    y_score : (N,) real-valued scores
    margin  : hinge margin (default = 1)
    reduction : "mean" or "sum"
    """

    # convert to numpy arrays
    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)

    # shape validation
    if y_true.shape != y_score.shape:
        return None

    # label validation
    if not np.all(np.isin(y_true, [-1, 1])):
        return None

    # hinge loss per sample
    loss = np.maximum(0.0, margin - y_true * y_score)

    # reduction
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("Invalid reduction")