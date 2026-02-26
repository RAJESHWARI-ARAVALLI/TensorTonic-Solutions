import numpy as np

# ---------------------------------------
# Numerically stable sigmoid
# ---------------------------------------
def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# ---------------------------------------
# Training function (fixed)
# ---------------------------------------
def train_logistic_regression(X, y, lr=0.01, steps=1000):
    """
    X : (N, D)
    y : (N,)
    lr : learning rate
    steps : number of gradient descent steps
    """

    N, D = X.shape

    # initialize parameters
    w = np.zeros(D)
    b = 0.0

    for step in range(steps):

        # Forward pass
        z = X @ w + b
        p = _sigmoid(z)

        # Loss
        eps = 1e-9
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        # Gradients
        dw = (1 / N) * (X.T @ (p - y))
        db = (1 / N) * np.sum(p - y)

        # Update
        w -= lr * dw
        b -= lr * db

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    return w, b


# ---------------------------------------
# Example
# ---------------------------------------
if __name__ == "__main__":
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5]])

    y = np.array([0, 0, 1, 1])

    w, b = train_logistic_regression(X, y, lr=0.1, steps=1000)

    print("\nLearned Parameters:")
    print("w =", w)
    print("b =", b)