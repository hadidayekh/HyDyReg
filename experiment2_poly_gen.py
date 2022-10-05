import numpy as np
import matplotlib.pyplot as plt
from paper_utils import get_modetr, get_event, get_labeltest, simulation_ode_3


def case2_mod():
    modetr = get_modetr(0)
    event = get_event(0)
    labeltest = get_labeltest(0)
    y0 = [[-1,1],[1,4]]#,[2,-3]]
    
    T = 3
    stepsize = 0.05
    t_list, y_list = simulation_ode_3(modetr, event, labeltest, y0, T, stepsize)

    np.savetxt("data/t_2.csv", t_list[1], delimiter=', ', fmt='%.8f')

    for i, temp_y, in enumerate(y_list):
        np.savetxt("data/x" + str(i) + "_2.csv", temp_y, delimiter=',', fmt='%.8f')
        # np.savetxt("data/x0_2.csv", y_list[0], delimiter=', ', fmt='%8f')
        # np.savetxt("data/x1_2.csv", y_list[1], delimiter=', ', fmt='%8f')
        # np.savetxt("data/x2_2.csv", y_list[2], delimiter=', ', fmt='%8f')


    colors = ['r', 'm', 'b']
    for i, temp_y in enumerate(y_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.scatter(y0_list,y1_list,c=colors[i],s=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()



if __name__ == "__main__":

    case2_mod()