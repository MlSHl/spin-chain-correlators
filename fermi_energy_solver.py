import numpy as np
from scipy.optimize import root_scalar

delta = 1
t = 1
t_prime = 1

# Domain of solutions
m = 0
n = np.pi/2

# Binary search parameters
a = -1.5
b = 0
eps = 0.00001

def solve(f, a, b, y):
    g = lambda x: f(x) - y
    sol = root_scalar(g, bracket=(a, b), method="brentq")
    return sol.root

def E_minus(k):
    return 2*t_prime*np.cos(2*k) - np.sqrt((2*t*np.cos(k))**2 + (delta/2)**2)

def E_plus(k):
    return 2*t_prime*np.cos(2*k) + np.sqrt((2*t*np.cos(k))**2 + (delta/2)**2)

def G(y):
    return solve(E_minus, m, n, y) + solve(E_plus, m, n, y) - np.pi/2

# Binary search
while True:
    y = (a+b)/2
    Gy = G(y)
    print("step: ", y, " value: ", Gy)

    if (abs(Gy) < eps):
        print("Solution is: ", y)
        break
    elif (Gy > 0):
        a = y
    else:
        b = y

