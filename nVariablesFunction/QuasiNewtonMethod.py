import numpy as np
from NumericalGradient import *
from OneDimensionSearch import Method618


def QuasiNewtonMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear programming by Quasi-Newton method.
    
    Args:
        func: function, the object function.
        init: ndarray, the initial point.
        tol: double, interation accuracy, the default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    grad = NumericalGradient(func, init)
    while np.linalg.norm(grad) >= tol:
        invH = np.linalg.inv(NumericalHessian(func, init))
        P = np.dot(invH, -grad)
        f = lambda lam: func(init+lam*P)
        # if lambda = 1, this method changed to Newton Method.
        lam, _ = Method618(f, np.array([0, 1]))
        init = init + lam * P
        grad = NumericalGradient(func, init)
        
    return init, func(init)


# Test functions
def f1(t):
    return 2*t[0]**2 + 2 * t[0] * t[1] + t[1]**2 + t[0] - t[1]


def f2(t):
    return (1 - t[0])**2 + 2 * (t[1] - t[0]**2)**2


if __name__ == "__main__":
    print("By quasi-Newton method,")
    point, value = QuasiNewtonMethod(f1, np.zeros(2), 1e-3)
    print("the minimum point of f1(x, y) is ", point, "and the minimum value is ", value)
    point, value = QuasiNewtonMethod(f2, np.zeros(2), 1e-3)
    print("the minimum point of f2(x, y) is ", point, "and the minimum value is ", value)
