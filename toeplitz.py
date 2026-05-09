import numpy as np
import matplotlib.pyplot as plt

J = 1
J_star = 3
alpha = J_star/J
alpha_critical = 2
N = 1000


def k_f_minus():
    return np.arccos((1-np.sqrt(1+2*alpha**2))/(2*alpha)) 

def k_f_plus():
    return np.arccos((1+np.sqrt(1+2*alpha**2))/(2*alpha)) 

def delta(a, b):
    if (a==b):
        return 1
    else:
        return 0

# period 2*K_fermi

km = k_f_minus()
kp = k_f_plus()
def g(r):
    if (r==0):
        if alpha < alpha_critical:
            return 2 * km / np.pi - 1
        else:
            return 2 * (km - kp) / np.pi - 1
    if (alpha < alpha_critical):
        #print(km())
        return (2/(np.pi*r))*np.sin(km * r) - delta(0, r)
    else:
        #print(kp())
        return (2/(np.pi*r))*(np.sin(km * r) 
                              - np.sin(kp * r))

def det(matrix):
    return np.linalg.det(matrix)

toeplitz_matrix_indexes = np.arange(-N, N+1) 
g_vec = np.vectorize(g) 
toeplitz_matrix_elements = g_vec(toeplitz_matrix_indexes)
def K(r):
    
    i, j = np.indices((r, r))
    A = toeplitz_matrix_elements[j-i+1+N]

    return (1/4)*det(A)


x = np.arange(1, N+1) # 2, 3, 4, ... , N-1, N
y = np.array([K(r) for r in x])

plt.plot(x, y)
plt.xlabel("r")
plt.ylabel("K_tr(r)")
plt.show()
