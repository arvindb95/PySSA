import time
from astropy.table import Table
import pickle

ast_time_start = time.time()

tab = Table.read("F3_values.txt", format="ascii")

x = tab["x"].data
p = tab["p"].data
F3 = tab["F3"].data

ast_time_end = time.time()

print("Time taken to load file using astropy is = ", ast_time_end - ast_time_start)

F_dict = {"x": x, "p": p, "F3": F3}

filehandler = open("F3_values.pkl", "wb")

pickle.dump(F_dict, filehandler)

filehandler.close()

"""
pkl_time_start = time.time()

filehandler = open("F2_values.pkl", "rb")

F_dict = pickle.load(filehandler)

filehandler.close()

pkl_time_end = time.time()

print(F_dict)

print("Time taken to load using pickle = ", pkl_time_end - pkl_time_start)
"""
