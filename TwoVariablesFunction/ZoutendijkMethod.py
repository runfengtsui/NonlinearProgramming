import numpy as np
from GradientMethod import Gradient
from sympy import solve, Pow
from sympy.abc import x, y
from OneDimensionSearch import Method618
from scipy import optimize


def ZoutendijkMethod(func, init, A, b, tol=1e-9):
    """ Constrained nonlinear programming by Zoutendijk Method.
    
    Args:
        func: symbol function, the object function.
        init: ndarray, the initial point.
        A: ndarray, coffcients of constrained condition.
        b: ndarray, const vector of constrained condition.
        tol: double, iteration accuracy, the default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    while True:
        # We can't use equal symbol because the number's type is float
        if abs(np.dot(A, init) - b) > 0.01:
            grad = Gradient(func, init)
            if np.linalg.norm(grad)**2 > tol:
                D = -grad
                expr = str(A[0])+'*('+str(init[0])+"+"+str(D[0])+"*x)+"+str(A[1])+"*("+str(init[1])+"+"+str(D[1])+"*x)-"+str(b)
                lamMax = float(solve(expr, x)[0])
                f = lambda lam: func(init[0]+lam*D[0], init[1]+lam*D[1])
                lam, _ = Method618(f, np.array([0, lamMax]))
                init = init + lam * D
            else:
                return init, func(init[0], init[1])
        else:
            A1 = np.append(Gradient(func, init), -1)
            A2 = np.append(A, -1)
            Au = np.vstack((A1, A2))
            bounds = [(-1, 1), (-1, 1), (None, 0)]
            result = optimize.linprog(np.array([0, 0, 1]), A_ub=Au, b_ub=np.zeros(2), bounds=bounds).x
            if abs(result[-1]) <= tol:
                return init, func(init[0], init[1])
            else:
                D = np.array(result[0:2])
                f = lambda lam: func(init[0]+D[0]*lam, init[1]+D[1]*lam)
                lam, _ = Method618(f, np.array([0, 1]))
                init = init + lam * D


# Test function
def f(x, y):
    return -4 * x - 4 * y + Pow(x, 2) + Pow(y, 2)


if __name__ == '__main__':
    point, value = ZoutendijkMethod(f, np.zeros(2), np.array([1, 2]), 4, 0.01)
    print("By Zoutendijk method, the minimum point of the function f(t) is ", point,
            "and the minimum value of f(t) is ", value)
