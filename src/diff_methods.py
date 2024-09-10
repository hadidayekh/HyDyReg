import numpy as np

# Five point stencil differentiation method, using two points from the back and two from the front
def diff_method_fps(t_list, x_list, segments=[], remove_seg=False):
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

        if not remove_seg:
            nb_points = t_points.shape[0]
            assert(nb_points == x_points.shape[0])

            # assuming t_points are equally spaced
            h = t_points[1] - t_points[0]

            D = nb_points - 4
            
            X = np.zeros([D, x_dim])
            X_dot = np.zeros([D, x_dim])

            for j in range(0, D):
                X[j] = x_points[j+2]
                X_dot[j] = (-x_points[j+4] + 8*x_points[j+3] - 8*x_points[j+1] + x_points[j]) / (12.0 * h)
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
                    continue
                h = t_points_seg[1] - t_points_seg[0]
                X_seg = np.zeros([D, x_dim])
                X_dot_seg = np.zeros([D, x_dim])
                for j in range(0, D):
                    X_seg[j] = x_points_seg[j+2]
                    X_dot_seg[j] = (-x_points_seg[j+4] + 8*x_points_seg[j+3] - 8*x_points_seg[j+1] + x_points_seg[j]) / (12.0 * h)

                if first:
                    X = X_seg
                    X_dot = X_dot_seg
                    first = False
                else:
                    # if multiple trajectories, concatenate them
                    X = np.r_[X, X_seg]
                    X_dot = np.r_[X_dot, X_dot_seg]
                
                if i==0:
                    seg_res.append(X.shape[0] - X_seg.shape[0])
                else:
                    seg_res.append(X.shape[0] - X_seg.shape[0] + final_X.shape[0])



            
        if (i==0):
            final_X = X
            final_X_dot = X_dot
        else:
            # if multiple trajectories, concatenate them
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]

        if len(segments) > 0 and not remove_seg:
            seg_curr = segments[i]
            begin_index = 0
            for j in range(0, i):
                begin_index += segments[j][-1] - 3
            seg_res.append(begin_index)
            for j in range(1, len(seg_curr) - 1):
                seg_res.append(begin_index + seg_curr[j] - 2)
    if len(segments) > 0:
        # print("HERE ARE THE SEGMENTS", final_X.shape, segments, seg_res)
        # exit()
        return final_X, final_X_dot, seg_res
    else:
        return final_X, final_X_dot
