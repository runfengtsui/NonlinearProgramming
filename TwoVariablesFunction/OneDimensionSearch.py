import numpy as np


def Method618(func, intev, tol=1e-9):
    """ One　dimensional search by 0.618 method.
    
    Args:
        func: function, the unimodal function to be searched.
        intev: ndarray, the searching interval.
        tol: double, iteration accuracy, default value is 1e-9.
    
    Returns:
        ndarray, the minimum point and the minimum value of the function func.
    """
    goldenRatio = (5**0.5 - 1) / 2
    a, b = intev
    if abs(b - a) < tol:
        return np.array([sum(intev) / 2, func(sum(intev)/2)])
    t1 = b - goldenRatio * (b - a)
    t2 = a + goldenRatio * (b - a)
    if func(t1) < func(t2):
        b = t2
    else:
        a = t1
    return Method618(func, np.array([a, b]), tol)


def FibonacciMethod(func, intev, delta=0.1, epsilon=0.01):
    """ One dimensional search by Fibonacci method.
    
    Args:
        func: function, the unimodal function to be searched.
        intev: ndarray, the searching interval.
        delta: double, maximum shortening rate, the default value is 0.1.
        epsilon: double, arbitrarily small number, the default value is 0.01.
        
    Returns:
        ndarray, the shortened interval.
        ndarray, the minimum point and minimum value of the function func.
    """
    a, b = intev
    
    # Determine the number of pilots and generate Fibonacci sequence
    n = 2
    fibonacci = [1, 1]
    while fibonacci[n-1] < 1 / delta:
        n += 1
        fibonacci.append(fibonacci[n-2] + fibonacci[n-3])
    
    # iteration
    while n > 3:
        t1 = a + fibonacci[n-3] / fibonacci[n-1] * (b - a)
        t2 = a + fibonacci[n-2] / fibonacci[n-1] * (b - a)
        if func(t1) < func(t2):
            b = t2
        elif func(t2) <= func(t1):
            a = t1
        n -= 1   
    b = a + (0.5 + epsilon) * (b - a)
    
    return np.array([a, b]), np.array([t2, func(t2)])


# Test function 1.
def f(t):
    return np.exp(t) + np.exp(-t)


# Test function 2.
def g(t):
    # the negative symbol is to find the maximum of the original function.
    return -np.power(np.sin(t), 6) * np.tan(1-t) * np.exp(30*t)


if __name__ == "__main__":
    # Test function 1
    intev = np.array([-1, 1])
    x, fv = Method618(f, intev, 1e-5)
    print("By 0.618 method, the minimum point is ", x, "and the minimum value is ", fv)
    _, [x, fv] = FibonacciMethod(f, intev, delta=1e-5/2, epsilon=1e-5)
    print("By Fibonacci method, the minimum point is ", x, "and the minimum value is ", fv)

    # Test function 2
    intev = np.array([0, 1])
    x, fv = Method618(g, intev, 1e-5)
    print("By 0.618 method, the maximum point is ", x, "and the maximum value is ", -fv)
    _, [x, fv] = FibonacciMethod(g, intev, delta=1e-5/2, epsilon=1e-5)
    print("By Fibonacci method, the maximum point is ", x, "and the maximum value is ", -fv)
