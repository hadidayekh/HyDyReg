import numpy as np
class SNDS:
    '''
    A switched nonlinear dynamical system.
    
    n = number of modes
    d = dimension of the continuous state vector
    eqs = list of n lists of d differential equations, one per mode and dimension of the continuous state vector
    eventlist = list of events to track when simulating trajectory, this is usually the conditions of crossing from one mode to another
    labeltest = a function that returns the mode in which state x is in, from {0,...,n-1}
    '''
    def __init__(self, n, d, eqs, eventlist, labeltest):
        assert (n == len(eqs))
        self.number_of_modes = n
        self.dim = d
        self.equations = eqs
        self.labeltest = labeltest
        self.eventlist = eventlist

    def simulate_trajectory(self, step, T, x0s):
        num = round(T/step + 1)
        t_points = np.linspace(0, T, num)
        for x0 in x0s:
            print()