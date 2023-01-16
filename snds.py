import numpy as np
from scipy.integrate import solve_ivp
class SNDS:
    '''
    A switched nonlinear dynamical system.
    
    n = number of modes
    d = dimension of the continuous state vector
    eqs = list of n lists of d differential equations, one per mode and dimension of the continuous state vector
    eventlist = list of events to track when simulating trajectory, this is usually the conditions of crossing from one mode to another
    labeltest = a function that returns the mode in which state x is in, from {0,...,n-1}
    '''
    def __init__(self, n, d, eqs, eventlist, labelfun):
        assert (n == len(eqs))
        self.number_of_modes = n
        self.dim = d
        self.equations = eqs
        self.labelfun = labelfun
        self.eventlist = eventlist

    def simulate_trajectory(self, step, T, x0s):
        num = round(T/step + 1)
        t_points = np.linspace(0, T, num)

        t_list = []
        x_list = []

        for k in range(len(x0s)):
            # initial condition in list
            x0 = x0s[k]

            # the mode of the initial condition
            label = self.labelfun(x0)

            status = 1
            ite = 0
            while status == 1:
                ite = ite + 1
                
                # solve the ivp according to the label  of x0 while tracking events in eventlist
                x_object = solve_ivp(self.eqs[label], (0, T + 1.1*step), x0,
                                    t_eval=t_points, method='RK45', rtol=1e-7, atol=1e-9,
                                    max_step=step/10, events=self.eventlist, dense_output=True)
                
                # termination status, if 1 then a terminating event ocurred
                status = x_object.status

                # in case of a terminating event
                if status == 1:
                    # x_object.t_events is a list of time lists of each event type 
                    for a in x_object.t_events:
                        # take one event that is non empty as the time where the event happened
                        # change t_start to be this time
                        # TODO : check if multiple events happened? -
                        if len(a) > 0:
                            t_start = a[0]
                            break

                    # check where t_start occurs in the list t_points
                    # this is in order to keep t_eval consistent with what we want
                    l = t_points.shape[0]
                    for i in range(0,l):
                        if t_points[i] > t_start:
                            break
                    
                    # update t_points to be the times after t_start
                    t_points = t_points[i:]
                    # evaluate the solution at t_start to check which new intial condition to give to 
                    # the solver, and check the (possibly new) label
                    sol = x_object.sol
                    x0 = sol.__call__(t_start + step/1000)
                    label = self.labelfun(x0)

                    x_points = x_object.y.T
                    x_points = x_points[0:i]
                else:
                    x_points = x_object.y.T

                if ite == 1:
                    X_points = x_points
                else:
                    X_points = np.r_[X_points, x_points]

            t_list.append(np.linspace(0, T, num))
            x_list.append(X_points)
        return (t_list, x_list)