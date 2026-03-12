import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    
    N, D = X.shape
    
    # Initialize weights w as zeros and bias b as 0.0
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # 1. Forward Pass: compute predictions
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 2. Compute the error
        error = p - y
        
        # 3. Calculate Gradients
        # dw = (1/N) * X^T * error
        dw = (1 / N) * np.dot(X.T, error)
        # db = (1/N) * sum of error
        db = (1 / N) * np.sum(error)
        
        # 4. Update Parameters (Gradient Descent)
        w -= lr * dw
        b -= lr * db
        
    return w, float(b)
