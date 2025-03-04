from typing import final
import numpy as np
import pysindy as ps

# Five point stencil differentiation method, using two points from the back and two from the front
def diff_method_fps(t_list, x_list, segments=None, remove_seg=False):
    '''
    - calculates derivatives of trajectory points
    - returns a single matrix X with concatenated data along with the corresponding derivatives Xdot
    - if segments is provided and remove_seg = True:
        * calculates derivatives for each segment not trajectory
        * returns STARTING INDEX of segments INCLUDING 0 for X, Xdot
    '''
    nb_traj = len(t_list) # number of arrays of t-points in t_list
    assert(nb_traj == len(x_list)) # should be equal to number of x-points in x_list
    x_dim = x_list[0][0].shape[0] # dimension of each data point x
    seg_res = []
    for i in range(nb_traj):
        t_points = t_list[i]
        x_points = x_list[i]

        if segments is None:
            nb_points = t_points.shape[0]
            assert(nb_points == x_points.shape[0])

            # assuming t_points are equally spaced
            h = t_points[1] - t_points[0]

            D = nb_points - 4
            
            X = np.zeros([D, x_dim])
            X_dot = np.zeros([D, x_dim])
            Time = np.zeros([D,])

            for j in range(0, D):
                X[j] = x_points[j+2]
                X_dot[j] = (-x_points[j+4] + 8*x_points[j+3] - 8*x_points[j+1] + x_points[j]) / (12.0 * h)
                Time[j] = t_points[j+2]
        else:
            first = True
            seg_curr = segments[i]
            # split t_points and x_points at changepoints in order to calculate
            # derivatives in each segment separately
            splitted_t_points = np.split(t_points, seg_curr)
            splitted_x_points = np.split(x_points, seg_curr)
            # for each segment
            for k in range(len(seg_curr) + 1):
                t_points_seg = splitted_t_points[k]
                x_points_seg = splitted_x_points[k]
                nb_points = t_points_seg.shape[0]
                assert(nb_points == x_points_seg.shape[0])
                # assuming t_points are equally spaced
                D = nb_points - 4
                if D <= 0:
                    if first:
                        X = np.zeros([0, x_dim])
                        X_dot = np.zeros([0, x_dim])
                        Time = np.zeros([0,])
                        first = False
                    else:
                        # if multiple trajectories, concatenate them
                        X = np.r_[X, np.zeros([0, x_dim])]
                        X_dot = np.r_[X_dot, np.zeros([0, x_dim])]
                        Time = np.r_[Time, np.zeros([0,])]
                    continue
                h = t_points_seg[1] - t_points_seg[0]
                X_seg = np.zeros([D, x_dim])
                X_dot_seg = np.zeros([D, x_dim])
                Time_seg = np.zeros([D,])
                for j in range(0, D):
                    X_seg[j] = x_points_seg[j+2]
                    X_dot_seg[j] = (-x_points_seg[j+4] + 8*x_points_seg[j+3] - 8*x_points_seg[j+1] + x_points_seg[j]) / (12.0 * h)
                    Time_seg[j] = t_points_seg[j+2]

                if first:
                    X = X_seg
                    X_dot = X_dot_seg
                    Time = Time_seg
                    first = False
                else:
                    # if multiple trajectories, concatenate them
                    X = np.r_[X, X_seg]
                    X_dot = np.r_[X_dot, X_dot_seg]
                    Time = np.r_[Time, Time_seg]
                
                if i==0:
                    seg_res.append(X.shape[0] - X_seg.shape[0])
                else:
                    seg_res.append(X.shape[0] - X_seg.shape[0] + final_X.shape[0])



            
        if (i==0):
            final_X = X
            final_X_dot = X_dot
            final_Time = Time
        else:
            # if multiple trajectories, concatenate them
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
            final_Time = np.r_[final_Time, Time]

    if len(segments) > 0:
        # print("HERE ARE THE SEGMENTS", final_X.shape, segments, seg_res)
        # exit()
        return final_X, final_X_dot, seg_res, final_Time
    else:
        return final_X, final_X_dot, final_Time


# Three point central differentiation method, using 1 point from the front
# ALLOWS FOR NON-UNIFORM GRID
def diff_method_forward(t_list, x_list, segments=None, remove_seg=False):
    '''
    - calculates derivatives of trajectory points
    - returns a single matrix X with concatenated data along with the corresponding derivatives Xdot
    - if segments is provided and remove_seg = True:
        * calculates derivatives for each segment not trajectory
        * returns STARTING INDEX of segments INCLUDING 0 for X, Xdot
    '''
    nb_traj = len(t_list) # number of arrays of t-points in t_list
    assert(nb_traj == len(x_list)) # should be equal to number of x-points in x_list
    x_dim = x_list[0][0].shape[0] # dimension of each data point x
    seg_res = []
    for i in range(nb_traj):
        t_points = t_list[i]
        x_points = x_list[i]

        if segments is None:
            nb_points = t_points.shape[0]
            assert(nb_points == x_points.shape[0])

            D = nb_points - 1
            
            X = np.zeros([D, x_dim])
            X_dot = np.zeros([D, x_dim])
            Time = np.zeros([D,])

            for j in range(0, D):
                X[j] = x_points[j]
                X_dot[j] = (x_points[j+1] - x_points[j]) / (t_points[j+1] - t_points[j])
                Time[j] = t_points[j]
        else:
            first = True
            seg_curr = segments[i]
            # split t_points and x_points at changepoints in order to calculate
            # derivatives in each segment separately
            splitted_t_points = np.split(t_points, seg_curr)
            splitted_x_points = np.split(x_points, seg_curr)
            # for each segment
            for k in range(len(seg_curr) + 1):
                t_points_seg = splitted_t_points[k]
                x_points_seg = splitted_x_points[k]
                nb_points = t_points_seg.shape[0]
                assert(nb_points == x_points_seg.shape[0])
                # assuming t_points are equally spaced
                D = nb_points - 1
                if D <= 0:
                    if first:
                        X = np.zeros([0, x_dim])
                        X_dot = np.zeros([0, x_dim])
                        Time = np.zeros([0,])
                        first = False
                    else:
                        # if multiple trajectories, concatenate them
                        X = np.r_[X, np.zeros([0, x_dim])]
                        X_dot = np.r_[X_dot, np.zeros([0, x_dim])]
                        Time = np.r_[Time, np.zeros([0,])]
                    continue
                h = t_points_seg[1] - t_points_seg[0]
                X_seg = np.zeros([D, x_dim])
                X_dot_seg = np.zeros([D, x_dim])
                Time_seg = np.zeros([D,])
                for j in range(0, D):
                    X_seg[j] = x_points_seg[j]
                    X_dot_seg[j] = (x_points_seg[j+1] - x_points_seg[j]) / (t_points_seg[j+1] - t_points_seg[j])
                    Time_seg[j] = t_points_seg[j]

                if first:
                    X = X_seg
                    X_dot = X_dot_seg
                    Time = Time_seg
                    first = False
                else:
                    # if multiple trajectories, concatenate them
                    X = np.r_[X, X_seg]
                    X_dot = np.r_[X_dot, X_dot_seg]
                    Time = np.r_[Time, Time_seg]
                
                if i==0:
                    seg_res.append(X.shape[0] - X_seg.shape[0])
                else:
                    seg_res.append(X.shape[0] - X_seg.shape[0] + final_X.shape[0])



            
        if (i==0):
            final_X = X
            final_X_dot = X_dot
            final_Time = Time
        else:
            # if multiple trajectories, concatenate them
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
            final_Time = np.r_[final_Time, Time]

    if len(segments) > 0:
        return final_X, final_X_dot, seg_res, final_Time
    else:
        return final_X, final_X_dot, final_Time


def diff_method_ps(t_list, x_list, segments=None, remove_seg=False, dm=ps.FiniteDifference(order=4, drop_endpoints=True)):
    '''
    - calculates derivatives of trajectory points
    - returns a single matrix X with concatenated data along with the corresponding derivatives Xdot
    - if segments is provided and remove_seg = True:
        * calculates derivatives for each segment not trajectory
        * returns STARTING INDEX of segments INCLUDING 0 for X, Xdot
    '''
    nb_traj = len(t_list) # number of arrays of t-points in t_list
    assert(nb_traj == len(x_list)) # should be equal to number of x-points in x_list
    x_dim = x_list[0][0].shape[0] # dimension of each data point x
    seg_res = []
    try:
        order = dm.order
    except:
        order = dm.get_params()['order']
    n_skip = -(order // -2)
    for i in range(nb_traj):
        t_points = t_list[i]
        x_points = x_list[i]

        if segments is None:
            nb_points = t_points.shape[0]
            assert(nb_points == x_points.shape[0])

            # assuming t_points are equally spaced
            h = t_points[1] - t_points[0]

            D = nb_points - order

            X = np.zeros([D, x_dim])
            X_dot = np.zeros([D, x_dim])
            Time = np.zeros([D,])

            X_dot_ps = dm(x_points, t_points)
            for j in range(0, D):
                X[j] = x_points[j+n_skip]
                X_dot[j] = X_dot_ps[j+n_skip]
                Time[j] = t_points[j+n_skip]
        else:
            first = True
            seg_curr = segments[i]
            # split t_points and x_points at changepoints in order to calculate
            # derivatives in each segment separately
            splitted_t_points = np.split(t_points, seg_curr)
            splitted_x_points = np.split(x_points, seg_curr)
            # for each segment
            for k in range(len(seg_curr) + 1):
                t_points_seg = splitted_t_points[k]
                x_points_seg = splitted_x_points[k]
                nb_points = t_points_seg.shape[0]
                assert(nb_points == x_points_seg.shape[0])
                # assuming t_points are equally spaced
                D = nb_points - order
                if D <= 0:
                    if first:
                        X = np.zeros([0, x_dim])
                        X_dot = np.zeros([0, x_dim])
                        Time = np.zeros([0,])
                        first = False
                    else:
                        # if multiple trajectories, concatenate them
                        X = np.r_[X, np.zeros([0, x_dim])]
                        X_dot = np.r_[X_dot, np.zeros([0, x_dim])]
                        Time = np.r_[Time, np.zeros([0,])]
                    continue
                h = t_points_seg[1] - t_points_seg[0]
                X_seg = np.zeros([D, x_dim])
                X_dot_seg = np.zeros([D, x_dim])
                Time_seg = np.zeros([D,])
                # print(x_points_seg.shape)
                X_dot_seg_ps = dm(x_points_seg, t_points_seg)
                for j in range(0, D):
                    X_seg[j] = x_points_seg[j+n_skip]
                    X_dot_seg[j] = X_dot_seg_ps[j+n_skip]
                    Time_seg[j] = t_points_seg[j+n_skip]

                if first:
                    X = X_seg
                    X_dot = X_dot_seg
                    Time = Time_seg
                    first = False
                else:
                    # if multiple trajectories, concatenate them
                    X = np.r_[X, X_seg]
                    X_dot = np.r_[X_dot, X_dot_seg]
                    Time = np.r_[Time, Time_seg]
                
                if i==0:
                    seg_res.append(X.shape[0] - X_seg.shape[0])
                else:
                    seg_res.append(X.shape[0] - X_seg.shape[0] + final_X.shape[0])



            
        if (i==0):
            final_X = X
            final_X_dot = X_dot
            final_Time = Time
        else:
            # if multiple trajectories, concatenate them
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
            final_Time = np.r_[final_Time, Time]

    if len(segments) > 0:
        # print("HERE ARE THE SEGMENTS", final_X.shape, segments, seg_res)
        # exit()
        return final_X, final_X_dot, seg_res, final_Time
    else:
        return final_X, final_X_dot, final_Time
