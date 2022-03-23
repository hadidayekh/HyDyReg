
from paper_utils import simulation_ode
import numpy as np
import matplotlib.pyplot as plt


def fvdp2_1(t,y):
    """Single mode, order 2."""
    y0, y1 = y
    dydt = [2*y0-y0*y1, 2*y0**2-y1]
    return dydt

if __name__ == "__main__":
    y0 = [[0.1, 0.2], [4, 0]]#[-0.1, 0.9]]#,[1],[2],[3],[0],[-1],[-2]]
    t_tuple = (0,6)
    stepsize = 0.1
    t_points, y_list = simulation_ode(fvdp2_1, y0, t_tuple, stepsize, eps=0)
    

    print(t_points)
    print(len(y_list))
    #print(y_list[0].shape)
    #print(y_list[1].shape)


    np.savetxt("data/t_3.csv", t_points, delimiter=', ', fmt='%8f')
    for i, temp_y, in enumerate(y_list):
        np.savetxt("data/x" + str(i) + "_3.csv", temp_y, delimiter=',', fmt='%.8f')


    for i, temp_y in enumerate(y_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.scatter(y0_list,y1_list,c='b',s=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
