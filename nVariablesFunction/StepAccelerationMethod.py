import numpy as np


def Perturbation(func, B, delta):
    """ Get the pertubation with step delta.
    
    Args:
        func: function, the object function.
        B: ndarray, the initial point.
        delta: double, step length, extremely small number.
        
    Returns:
        T: ndarray, the point after perturbation.
    """
    n, T = len(B), B.copy()
    for i in range(n):
        Delta = np.zeros(n)
        Delta[i] = delta
        if func(T+Delta) < func(T):
            T += Delta
        elif func(T-Delta) < func(T):
            T -= Delta
        else:
            T = T
    
    return T


def StepAccelerationMethod(func, B, delta=0.1, alpha=0.01, tol=1e-9):
    """ Unconstrained nonlinear programming by step acceleration method.
    
    Args:
        func: function, the object function.
        B: ndarray, the initial point.
        delta: double, step length, the default value is 0.1.
        alpha: double, shorten factor, the default value is 0.01.
        tol: double, iteration accuracy, the default value is 1e-9.
        
    Returns:
        T: ndarray, the minimum point of the object function.
        double, the minimum value of the object function.
    """
    while True:
        Bi = Perturbation(func, B, delta)
        if func(Bi) < func(B):
            T = B + 2 * (Bi - B)
            if func(T) < func(Bi):
                B = T
            else:
                B = Bi
        elif np.linalg.norm(B - Bi) <= tol and delta <= tol:
            break
        else:
            delta *= alpha
    
    return B, func(B)


# Test function 1.
def f(t):
    return t[0]**2 + t[1]**2


# Test function 2.
def g(t):
    return t[0]**2 + 50 * t[1]**2


# Test function 3.
def h(t):
    return (1 - t[0])**2 + 2 * (t[1] - t[0]**2)**2


if __name__ == "__main__":
    print("Step Acceleration Method:")
    # Test function 1
    point, value = StepAccelerationMethod(f, np.ones(2)*100, 10, 0.1, 1e-5)
    print("The minimum point of f(t) is", point, "and the minimum value is ", value)
    # Test function 2
    T, value = StepAccelerationMethod(g, np.ones(2), 10, 0.1, 1e-5)
    print("The minimum point of g(t) is", T, "and the minimum value is ", value)
    # Test function 3
    T, value = StepAccelerationMethod(h, np.zeros(2), 10, 0.1, 1e-5)
    print("The minimum point of h(t) is", T, "and the minimum value is ", value)
