import numpy as np


def NumericalGradient(func, point):
    """ Calculate the gradient of function at point by numerical method.

    Args:
        func: function, the object function.
        point: list/ndarray, the arbitrary point on Euclid space R^n.

    Returns:
        grad: ndarray, the gradient of the object function at point.
    """
    n = len(point)
    grad = np.zeros(n)
    eps = np.finfo(np.float32).eps
    for i in range(n):
        h = np.zeros(n)
        h[i] = eps
        grad[i] = (func(point+h) - func(point-h)) / (2 * h[i])

    return grad


def NumericalHessian(func, point):
    """ Calculate the Hessian matrix of function at point by numerical method.

    Args:
        func: function, the object function.
        point: list/ndarray, the arbitrary point on Eulicd space R^n.

    Returns:
        H: ndarray, the Hessian matrix of the function func at point.
    """
    n = len(point)
    H = np.zeros((n, n))
    grad0 = NumericalGradient(func, point)
    eps = np.linalg.norm(grad0) * np.finfo(np.float32).eps
    for i in range(n):
        h = np.zeros(n)
        h[i] = eps
        grad1 = NumericalGradient(func, point-h)
        grad2 = NumericalGradient(func, point+h)
        H[:, i] = (grad2 - grad1) / (2 * h[i])

    return H
