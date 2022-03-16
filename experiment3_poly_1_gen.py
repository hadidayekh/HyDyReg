
import paper_utils
import numpy as np
import matplotlib.pyplot as plt


def fvdp2_1(t,y):
    """Single mode, order 2."""
    y0, y1 = y
    dydt = [2*y0-y0*y1, 2*y0**2-y1]
    return dydt

if __name__ == "__main__":
    y0 = [[0]]#,[1],[2],[3],[0],[-1],[-2]]
    t_tuple = (0,8)
    stepsize = 1
    order = 3
    t_points, y_list = paper_utils.simulation_ode(fvdp2_1, y0, t_tuple, stepsize, eps=0)