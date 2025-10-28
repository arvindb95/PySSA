import numpy as np
from scipy import integrate, special
from astropy.table import Table
from tqdm import tqdm


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


def calc_F_3(x, calc_F, p):
    """
    Returns values of function F3 (defined in eq. A7 of Soderberg et al. 2005)
    at x
    """

    def fy3(y):
        return calc_F(y) * (y ** ((p - 3.0) / 2.0))

    if isinstance(x, float):
        return np.sqrt(3) * integrate.quad(fy3, 0, x)[0]
    else:
        F_3_x = np.zeros(len(x))
        for i, x_i in enumerate(x):
            if x_i < 2000:
                F_3_x[i] = np.sqrt(3) * integrate.quad(fy3, 0, x_i)[0]
            else:
                F_3_x[i:] = np.sqrt(3) * integrate.quad(fy3, 0, 2000)[0]
        return F_3_x


x1 = np.arange(0, 20, 1e-6)
x2 = np.arange(20, 700, 0.01)
x = np.append(x1, x2)

p = np.arange(2, 3.5, 1e-6)

F3 = []

for i in tqdm(range(len(x))):
    for j in tqdm(range(len(p))):
        F3.append(calc_F_3(x[i], calc_F, p[j]))

val_dict = {"x": x, "p": p, "F3": F3}


with open("F3_values.pkl", "wb") as f:
    pickle.dump(val_dict, f)

