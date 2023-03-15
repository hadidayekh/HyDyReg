import sys
# setting path
sys.path.append('./')
sys.path.append('../')

from snds import SNDS_linear
import plotting
import numpy as np
import matplotlib.pyplot as plt

exp_title = "isolette"

a1 = -0.026
a2 = -0.026
a3 = 98.5

def labelfun(x):
    if x[0] <= a3:
        return 0
    else:
        return 1


event1 = lambda t, x : x[0] - a3
event1.terminal = True
event1.direction = 0

eventlist = [event1]

# Can use the linear SUL for this wrapper with parameters:
a00 = a1
b00 = -a1
c00 = 0

a01 = 0
b01 = 0
c01 = 1

a10 = a2
b10 = -a2
c10 = 0

a11 = 0
b11 = 0
c11 = -1

params = [[[a00, b00, c00],
           [a01, b01, c01]],
          [[a10, b10, c10],
           [a11, b11, c11]]]

params = np.array(params)


##### DEFINING SNDS #####
sul = SNDS_linear(params, eventlist, labelfun)
##### SIMULATING TRAJECTORIES #####
x0s = [[105.5, 80], [97.5, 100]]
time_step = 1
time_range = 50


trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
segments = trajectories.segment_trajectories(0.1)

fig, ax = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, plot_xt=True, segments=segments, labelfun=labelfun)

plt.show()