import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

J = 1
J_star = 0
alpha = J_star/J
alpha_critical = 2
N = 100


def k_f_minus():
    if (alpha == 0):
        return np.pi/2
    return np.arccos((1-np.sqrt(1+2*alpha**2))/(2*alpha)) 

def k_f_plus():
    if (alpha == 0):
        return np.pi/2
    return np.arccos((1+np.sqrt(1+2*alpha**2))/(2*alpha)) 

def delta(a, b):
    if (a==b):
        return 1
    else:
        return 0


km = k_f_minus()
kp = k_f_plus()

def g(r):
    if (r==0):
        if alpha < alpha_critical:
            return 2 * km / np.pi - 1
        else:
            return 2 * (km - kp) / np.pi - 1
    if (alpha < alpha_critical):
        return (2/(np.pi*r))*np.sin(km * r)
    else:
        return (2/(np.pi*r))*(np.sin(km * r) - np.sin(kp * r))

def det(matrix):
    return np.linalg.det(matrix)

#--------
toeplitz_matrix_indexes = np.arange(-N, N+1) 
g_vec = np.vectorize(g) 
toeplitz_matrix_elements = g_vec(toeplitz_matrix_indexes)
def K_tr(r):
    i, j = np.indices((r, r))
    A = toeplitz_matrix_elements[j-i+1+N]

    return (1/4)*det(A)

x = np.arange(1, N+1) # 2, 3, 4, ... , N-1, N
y = np.array([K_tr(r) for r in x])

#plt.scatter(x, y )
#plt.xlabel("r")
#plt.ylabel("K_tr(r)")
#plt.show()

#--------
#print("K(r): ", y)

#A = 0.1174
#B = 0.0637
#def analytic_K_tr(x, A, B):
#    return (A/(x**(1/2)) + (B*np.cos(2*k_f_minus()*x))/(x**(5/2)))
#
#ktr = analytic_K_tr(x, A, B)
#plt.plot(x, ktr)
#plt.show()

from scipy.optimize import curve_fit

def magic():
    r_min = 10
    x_fit = x[x >= r_min]
    y_fit = y[x >= r_min]

    if alpha < alpha_critical:
        # Eq. (21)
        def fit_func(r, A, B):
            return (
                A / np.sqrt(r)
                + B * np.cos(2 * km * r) / r**(5/2)
            )

        params, covariance = curve_fit(
            fit_func,
            x_fit,
            y_fit,
            p0=[0.1, 0.01]
        )

        A, B = params

        print("alpha < alpha_c")
        print(f"A = {A}")
        print(f"B = {B}")

        print("\nFitted function:")
        print(
            f"K_tr(r) = {A:.8f}/sqrt(r) "
            f"+ {B:.8f}*cos({2*km:.8f}*r)/r^(5/2)"
        )


    elif alpha > alpha_critical:
        # Eq. (22)
        def fit_func(r, B, C, D, E, phi_minus, phi_plus, phi_1, phi_2):
            return (
                B * np.cos(km * r + phi_minus) / r
                + C * np.cos(kp * r + phi_plus) / r
                + D * np.cos((2 * km - kp) * r + phi_1) / r**3
                + E * np.cos((2 * kp - km) * r + phi_2) / r**3
            )

        params, covariance = curve_fit(
            fit_func,
            x_fit,
            y_fit,
            p0=[
                0.05,   # B
                0.1,    # C
                0.05,   # D
                0.05,   # E
                0.0,    # phi_minus
                0.0,    # phi_plus
                0.0,    # phi_1
                0.0     # phi_2
            ],
            maxfev=100000
        )

        B, C, D, E, phi_minus, phi_plus, phi_1, phi_2 = params

        print("alpha > alpha_c")
        print(f"B = {B}")
        print(f"C = {C}")
        print(f"D = {D}")
        print(f"E = {E}")
        print(f"phi_minus = {phi_minus}")
        print(f"phi_plus = {phi_plus}")
        print(f"phi_1 = {phi_1}")
        print(f"phi_2 = {phi_2}")

        print("\nFitted function:")
        print(
            f"K_tr(r) = {B:.8f}*cos({km:.8f}*r + {phi_minus:.8f})/r\n"
            f"        + {C:.8f}*cos({kp:.8f}*r + {phi_plus:.8f})/r\n"
            f"        + {D:.8f}*cos({2*km-kp:.8f}*r + {phi_1:.8f})/r^3\n"
            f"        + {E:.8f}*cos({2*kp-km:.8f}*r + {phi_2:.8f})/r^3"
        )


    else:
        raise ValueError("alpha = alpha_c needs separate treatment.")


    # Plot exact data and fit
    y_model = fit_func(x, *params)

    plt.scatter(x, y, label="Exact determinant values")
    plt.plot(x, y_model, label="curve_fit")
    plt.xlabel("r")
    plt.ylabel("K_tr(r)")
    plt.legend()
    plt.show()

magic()
