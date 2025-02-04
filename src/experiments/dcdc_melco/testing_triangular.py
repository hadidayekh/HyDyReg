import numpy as np
import matplotlib.pyplot as plt

data_file = "experiments/dcdc_melco/data_w_triangular.csv"
data = np.genfromtxt(data_file, delimiter=',').T


t = data[0][:]
triangular = data[4]

t_hc = 5e-6
abs_slope = 200000
tri_m = abs_slope*t_hc

def tri_func(t):
    shifted_hc = np.floor(t/t_hc)
    if shifted_hc % 2 == 0:
        return abs_slope * (t - shifted_hc * t_hc)
    else:
        return -abs_slope * (t - shifted_hc * t_hc) + tri_m

tri_sim = np.zeros(t.shape)

for i in range(len(t)):
    tri_sim[i] = tri_func(t[i])

f, a = plt.subplots()

a.scatter(t, triangular, s=0.1, c='b')
a.scatter(t, tri_sim, s=0.1, c='r')

plt.show()

print(np.nanmax((triangular - tri_sim)/tri_sim))