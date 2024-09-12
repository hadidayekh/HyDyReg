import numpy as np

data = np.genfromtxt("experiments/dcdc_melco/melco_data.csv", delimiter=',').T

t = data[0]
Iin0 = data[1]
Iout0 = data[2]
Vout0 = data[3]
gate = data[4]

X = data[1:4].T
indices = np.where(np.diff(gate) != 0)[0]+1

t_list = np.split(t, indices)
x_list = np.split(X, indices)

for i in range(len(t_list)):
    print(t_list[i].shape, x_list[i].shape)