import numpy as np
import torch

EPSILON = 1e-14


def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

        Cross Entropy Loss that assumes the input
        X is post-softmax, so this function only
        does negative loglikelihood. EPSILON is applied
        while calculating log.

    Args:
        X: (n_neurons, n_examples). softmax outputs
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels

    Returns:
        a float number of Cross Entropy Loss (averaged)
    """
    s = -1/X.size()[1]
    print(torch.log(X + epsilon) * y_1hot)
    return s * torch.sum(torch.log(X + epsilon) * y_1hot)


def softmax(X):
    """Softmax

        Regular Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    X_exp = torch.exp(X)
    return X_exp/torch.sum(X_exp, 0)


def stable_softmax(X):
    """Softmax

        Numerically stable Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    c = torch.max(X, 0)[0]
    X_exp = torch.exp(torch.add(X, -c))
    return X_exp / torch.sum(X_exp, 0)


def relu(X):
    """Rectified Linear Unit

        Calculate ReLU

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tenor whereThe shape is the same as X but clamped on 0
    """
    return torch.maximum(X, torch.zeros(X.size()))


def sigmoid(X):
    """Sigmoid Function

        Calculate Sigmoid

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tensor where each element is the sigmoid of the X.
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)

