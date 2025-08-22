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


x1 = np.arange(0, 20, 1e-6)
x2 = np.arange(20, 700, 0.01)
x = np.append(x1, x2)

F = []

for i in tqdm(range(len(x))):
    F.append(calc_F(x[i]))

tab = Table([x, F], names=["x", "F"])

tab.write("F_values.txt", format="ascii", overwrite=True)
