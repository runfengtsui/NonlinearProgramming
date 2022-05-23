import numpy as np
from NumericalGradient import NumericalGradient
from OneDimensionSearch import Method618


def GradientDescendMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear programming by gradient descend method.
    
    Args:
        func: function, the object function.
        init: ndarray, the initial point.
        tol: double, the iteration accuracy, default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of function func.
        double, the minimum value of function func.
    """
    grad = NumericalGradient(func, init)
    while np.linalg.norm(grad) >= tol:
        f = lambda lam: func(init-lam*grad)
        lam, _ = Method618(f, np.array([0, 1]))
        init = init - lam * grad
        grad = NumericalGradient(func, init)
        
    return init, func(init)


# Test functions
def f1(t):
    return t[0]**2 + t[1]**2


def f2(t):
    return t[0]**2 + 50 * t[1]**2


def f3(t):
    return (1 - t[0])**2 + (t[1] - t[0]**2)**2


if __name__ == '__main__':
    print("By Gradient Descend Method, ")
    point, value = GradientDescendMethod(f1, np.array([100, 100]), 1e-3)
    print("the minimum point of f1 is", point, "and the minimum value is ", value)
    point, value = GradientDescendMethod(f2, np.ones(2), 1e-3)
    print("the minimum point of f2 is", point, "and the nimimum value is ", value)
    point, value = GradientDescendMethod(f3, np.zeros(2), 1e-3)
    print("the minimum point of f3 is", point, "and the minimum value is ", value)
