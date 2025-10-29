import numpy as np
from scipy import integrate, special
from astropy.table import Table
from tqdm import tqdm
import pickle


def calc_F(x):
    """
    Returns values of function F (defined in eq. A7 of Soderberg et al. 2005)
    at x
    """

    def fy1(y):
        return special.kv(5.0 / 3.0, y)

    if isinstance(x, float):
        return x * integrate.quad(fy1, x, np.inf)[0]
    else:
        F_x = np.zeros(len(x))
        for i, x_i in enumerate(x):
            F_x[i] = x_i * integrate.quad(fy1, x_i, np.inf)[0]
        return F_x


def calc_F_2(x, calc_F, p):
    """
    Returns values of function F2 (defined in eq. A7 of Soderberg et al. 2005)
    at x
    """

    def fy2(y):
        return calc_F(y) * (y ** ((p - 2.0) / 2.0))

    if isinstance(x, float):
        return np.sqrt(3) * integrate.quad(fy2, 0, x)[0]
    else:
        F_2_x = np.zeros(len(x))
        for i, x_i in enumerate(x):
            if x_i < 20000:
                F_2_x[i] = np.sqrt(3) * integrate.quad(fy2, 0, x_i)[0]
            else:
                F_2_x[i:] = np.sqrt(3) * integrate.quad(fy2, 0, 20000)[0]
        return F_2_x


x1 = np.arange(0, 20, 1e-3)
x2 = np.arange(20, 700)
x = np.append(x1, x2)

print(x)

p = np.arange(2, 3.5, 0.1)

F2 = []
F2_x = []
F2_p = []

for j in tqdm(range(len(p))):
    print(
        "----------------- calculating for p = " + str(p[j]) + " ----------------------"
    )
    for i in tqdm(range(len(x))):
        F2.append(calc_F_2(x[i], calc_F, p[j]))
        F2_x.append(x[i])
        F2_p.append(p[j])

val_dict = {"x": F2_x, "p": F2_p, "F2": F2}


with open("F2_values.pkl", "wb") as f:
    pickle.dump(val_dict, f)
