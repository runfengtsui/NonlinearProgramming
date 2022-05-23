import numpy as np
from NumericalGradient import NumericalGradient
from OneDimensionSearch import Method618


def DFPMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear Programming by DFP method.
    
    Args:
        func: function, the objection function.
        init: ndarray, the initial point.
        tol: double, iteration accuracy, the default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    k, n = 0, len(init)
    grad, H = NumericalGradient(func, init), np.eye(n)
    while np.linalg.norm(grad) >= tol:
        if k < n:
            P = np.dot(H, -grad)
            f = lambda lam: func(init+lam*P)
            lam, _ = Method618(f, np.array([0, 1]))
            XDelta = lam * P
            GDelta = NumericalGradient(func, init+XDelta) - NumericalGradient(func, init)
            HDelta = np.outer(XDelta, XDelta.T) / np.dot(GDelta.T, XDelta)
            temp1 = np.dot(np.dot(H, np.outer(GDelta, GDelta.T)), H)
            temp2 = np.dot(np.dot(GDelta.T, H), GDelta)
            HDelta -= temp1 / temp2
            init = init + XDelta
            grad = NumericalGradient(func, init)
        else:
            k = 0
            
    return init, func(init)


# Test functions
def f1(t):
    return 2*t[0]**2 + 2 * t[0] * t[1] + t[1]**2 + t[0] - t[1]


def f2(t):
    return (1 - t[0])**2 + 2 * (t[1] - t[0]**2)**2


if __name__ == '__main__':
    print("By DFP method,")
    point, value = DFPMethod(f1, np.zeros(2), 1e-3)
    print("the minimum point of f1(x, y) is ", point, "and the minimum value is ", value)
    point, value = DFPMethod(f2, np.zeros(2), 1e-3)
    print("the minimum point of f2(x, y) is ", point, "and the minimum value is ", value)
