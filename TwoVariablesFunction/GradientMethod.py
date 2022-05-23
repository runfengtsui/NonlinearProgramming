import numpy as np
import sympy as sp
from sympy.abc import x, y
from OneDimensionSearch import Method618


def Gradient(func, point):
    """ Calculate the gradient of the object funciton at point.
    
    Args:
        func: symbol function, the object function.
        point: ndarray, the arbitrary point on xOy plane.
        
    Returns:
        grad: ndarray, the gradient of the object function at point.
    """
    xpartial = sp.diff(func(x, y), x).subs([(x, point[0]), (y, point[1])])
    ypartial = sp.diff(func(x, y), y).subs([(x, point[0]), (y, point[1])])
    grad = np.array(list(map(float, [xpartial, ypartial])))
    
    return grad


def Hessian(func, point):
    """ Get the Hessian matrix of the object function at point.
    
    Args:
        func: symbol function, the object function.
        point: ndarray, the arbitrary point on xOy plane.
        
    Returns:
        H: ndarray, the Hessian matrix of the object function at point.
    """
    xpartial = sp.diff(func(x, y), x)
    ypartial = sp.diff(func(x, y), y)
    secondOrderList = [sp.diff(xpartial, x).subs([(x, point[0]), (y, point[1])]),
                       sp.diff(xpartial, y).subs([(x, point[0]), (y, point[1])]), 
                       sp.diff(ypartial, y).subs([(x, point[0]), (y, point[1])])]
    secondOrderList = list(map(float, secondOrderList))
    H = np.array([[secondOrderList[0], secondOrderList[1]],
                  [secondOrderList[1], secondOrderList[2]]])
    
    return H


def GradientDescendMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear programming by gradient descend method.
    
    Args:
        func: symbol function, the object function.
        init: ndarray, the initial point.
        tol: double, the iteration accuracy, default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of function func.
        double, the minimum value of function func.
    """
    grad = Gradient(func, init)
    while np.linalg.norm(grad) >= tol:
        f = lambda lam: func(init[0]-lam*grad[0], init[1]-lam*grad[1])
        lam, _ = Method618(f, np.array([0, 1]))
        init = init - lam * grad
        grad = Gradient(func, init)
        
    return init, func(init[0], init[1])


def ConjugateGradientMethod(func, init, tol=1e-9):
    """ Unconstrained nonlinear programming by conjugate gradient Method.
    
    Args:
        func: symbol function, the object function.
        init: ndarray, the init point.
        tol: double, the interation accuracy, the default value is 1e-9.
        
    Returns:
        init: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    k, n = 0, len(init)
    grad = Gradient(func, init)
    while np.linalg.norm(grad) >= tol:
        if k == 0:
            H = Hessian(func, init)
            lam = np.dot(grad, grad) / np.dot(np.dot(-grad, H), -grad)
            init -= lam * grad
            gradPrior, Pprior = grad, -grad
            grad = Gradient(func, init)
        elif k == n:
            k = 0
        else:
            alpha = np.dot(grad, grad) / np.dot(gradPrior, gradPrior)
            P = -grad + alpha * Pprior
            H = Hessian(func, init)
            lam = np.dot(grad, grad) / np.dot(np.dot(P, H), P)
            init += lam * P
            gradPrior, Pprior = grad, P
            grad = Gradient(func, init)
    
    return init, func(init[0], init[1])


# Test function 1.
def f1(x, y):
    return sp.Pow(x, 2) + sp.Pow(y, 2)


# Test function 2.
def f2(x, y):
    return sp.Pow(x, 2) + 50 * sp.Pow(y, 2)


# Test function 3.
def f3(x, y):
    return sp.Pow(1-x, 2) + 2 * sp.Pow(y-sp.Pow(x, 2), 2)


if __name__ == "__main__":
    # Gradient Descend Method.
    print("Gradient Descend Method:")
    point, value = GradientDescendMethod(f1, np.array([100, 100]), 1e-3)
    print("The minimum point of f1 is", point, "and the minimum value is ", value)
    point, value = GradientDescendMethod(f2, np.ones(2), 1e-3)
    print("The minimum point of f2 is", point, "and the nimimum value is ", value)
    point, value = GradientDescendMethod(f3, np.zeros(2), 1e-3)
    print("The minimum point of f3 is", point, "and the minimum value is ", value)
    # Conjugate Gradient Method.
    print("Conjugate Gradient Method:")
    point, value = ConjugateGradientMethod(f1, np.ones(2)*100, 1e-3)
    print("The minimum point of f1 is", point, "and the minimum value is ", value)
    point, value = ConjugateGradientMethod(f2, np.ones(2), 1e-3)
    print("The minimum point of f2 is", point, "and the nimimum value is ", value)
    point, value = ConjugateGradientMethod(f3, np.zeros(2), 1e-3)
    print("The minimum point of f3 is", point, "and the minimum value is ", value)
