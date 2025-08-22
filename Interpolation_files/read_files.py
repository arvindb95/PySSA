# from astropy.table import Table
#
# tab = Table.read("F_values.txt", format="ascii")
#
# x = tab["x"].data
# F = tab["F"].data
#
# print(len(x))

from pandas import read_fwf

t = read_fwf("F2_values.txt")

print(t)
