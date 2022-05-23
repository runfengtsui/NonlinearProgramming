import numpy as np
from NumericalGradient import *


def ConjugateGradientMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear programming by conjugate gradient method.
    
    Args:
        func: function, the object function.
        init: ndarray, the init point.
        tol: double, the interation accuracy, the default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    k, n = 0, len(init)
    grad = NumericalGradient(func, init)
    Pprior = -grad
    while np.linalg.norm(grad) >= tol:
        if k == 0:
            H = NumericalHessian(func, init)
            lam = np.dot(grad, grad) / np.dot(np.dot(-grad, H), -grad)
            init -= lam * grad
            gradPrior = grad
            grad = NumericalGradient(func, init)
        elif k == n:
            k = 0
        else:
            alpha = np.dot(grad, grad) / np.dot(gradPrior, gradPrior)
            P = -grad + alpha * Pprior
            H = NumericalHessian(func, init)
            lam = np.dot(grad, grad) / np.dot(np.dot(P, H), P)
            init += lam * P
            gradPrior, Pprior = grad, P
            grad = NumericalGradient(func, init)
    
    return init, func(init)


# Test functions
def f1(t):
    return t[0]**2 + t[1]**2


def f2(t):
    return t[0]**2 + 50 * t[1]**2


def f3(t):
    return (1 - t[0])**2 + (t[1] - t[0]**2)**2


if __name__ == "__main__":
    print("By conjugate gradient method,")
    point, value = ConjugateGradientMethod(f1, np.ones(2)*100, 1e-3)
    print("the minimum point of f1 is", point, "and the minimum value is ", value)
    point, value = ConjugateGradientMethod(f2, np.ones(2), 1e-3)
    print("the minimum point of f2 is", point, "and the minimum value is ", value)
    point, value = ConjugateGradientMethod(f3, np.zeros(2), 1e-3)
    print("the minimum point of f3 is", point, "and the minimum value is ", value)
