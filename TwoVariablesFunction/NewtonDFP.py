import numpy as np
import sympy as sp
from sympy.abc import x, y
from OneDimensionSearch import Method618
from GradientMethod import Gradient, Hessian


def QuasiNewtonMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear programming by Quasi-Newton method.
    
    Args:
        func: symbol function, the object function.
        init: ndarray, the initial point.
        tol: double, interation accuracy, the default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    grad = Gradient(func, init)
    while np.linalg.norm(grad) >= tol:
        invH = np.linalg.inv(Hessian(func, init))
        P = np.dot(invH, -grad)
        f = lambda lam: func(init[0]+lam*P[0], init[1]+lam*P[1])
        # if lambda = 1, this method changed to Newton Method.
        lam, _ = Method618(f, np.array([0, 1]))
        init = init + lam * P
        grad = Gradient(func, init)
        
    return init, func(init[0], init[1])


def DFPMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear Programming by DFP method.
    
    Args:
        func: symbol function, the objection function.
        init: ndarray, the initial point.
        tol: double, iteration accuracy, the default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    k, n = 0, len(init)
    grad, H = Gradient(func, init), np.eye(n)
    while np.linalg.norm(grad) >= tol:
        if k < n:
            P = np.dot(H, -grad)
            f = lambda lam: func(init[0]+lam*P[0], init[1]+lam*P[1])
            lam, _ = Method618(f, np.array([0, 1]))
            XDelta = lam * P
            GDelta = Gradient(func, init+XDelta) - Gradient(func, init)
            HDelta = np.outer(XDelta, XDelta.T) / np.dot(GDelta.T, XDelta)
            temp1 = np.dot(np.dot(H, np.outer(GDelta, GDelta.T)), H)
            temp2 = np.dot(np.dot(GDelta.T, H), GDelta)
            HDelta -= temp1 / temp2
            init = init + XDelta
            grad = Gradient(func, init)
        else:
            k = 0
            
    return init, func(init[0], init[1])


# Test function 1.
def f(x, y):
    return 2 * sp.Pow(x, 2) + 2 * x * y + sp.Pow(y, 2) + x - y


# Test function 2.
def g(x, y):
    return sp.Pow(1-x, 2) + 2 * sp.Pow(y-sp.Pow(x, 2), 2)


if __name__ == "__main__":
    # Quasi-Newton method.
    print("Quasi-Newton Method:")
    point, value = QuasiNewtonMethod(f, np.zeros(2), 1e-3)
    print("The minimum point of f(x, y) is ", point, "and the minimum value is ", value)
    point, value = QuasiNewtonMethod(g, np.zeros(2), 1e-3)
    print("The minimum point of g(x, y) is ", point, "and the minimum value is ", value)
    # DFP Method.
    print("DFP Method:")
    point, value = DFPMethod(f, np.zeros(2), 1e-3)
    print("The minimum point of f(x, y) is ", point, "and the minimum value is ", value)
    point, value = DFPMethod(g, np.zeros(2), 1e-3)
    print("The minimum point of g(x, y) is ", point, "and the minimum value is ", value)
