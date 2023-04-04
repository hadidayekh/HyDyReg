import numpy as np

# Diffrerential method used in paper, calculating the derivative of the lagrange interpolating polynomial at said point
def diff_method_paper(t_list, x_list):
    nb_traj = t_list.shape[0] # number of arrays of t-points in t_list
    assert(nb_traj == x_list.shape[0]) # should be equal to number of x-points in x_list
    x_dim = x_list[0][0].shape[0] # dimension of each data point x
    for i in range(nb_traj):
        t_points = t_list[i]
        x_points = x_list[i]
        nb_points = t_points.shape[0]
        assert(nb_points == x_points.shape[0])

        # assuming t_points are equally spaced
        h = t_points[1] - t_points[0]

        D = nb_points - 5
        
        # Calculating both backwards and forwards, so D*2
        X = np.zeros([2*D, x_dim])
        X_dot = np.zeros([2*D, x_dim])

        for j in range(2*D):
            if j < D:
                # forward
                X[j] = x_points[j]
                X_dot[j] = (-137*x_points[j] + 300*x_points[j+1] - 300*x_points[j+2] + 200*x_points[j+3] - 75*x_points[j+4] + 12*x_points[j+5])*h/60
            else:
                # backward
                X[j] = x_points[j-D+5]
                X_dot[j] = (137*x_points[j-D+5] - 300*x_points[j-D+4] + 300*x_points[j-D+3] - 200*x_points[j-D+2] + 75*x_points[j-D+1] - 12*x_points[j-D])*h/60
    
        if (i==0):
            final_X = X
            final_X_dot = X_dot
        else:
            # if multiple trajectories, concatenate them
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
    
    return final_X, final_X_dot

# Central differentiation method that uses one point from back and one point from the front
def diff_method_central(t_list, x_list, segments=[]):
    nb_traj = t_list.shape[0] # number of arrays of t-points in t_list
    assert(nb_traj == x_list.shape[0]) # should be equal to number of x-points in x_list
    x_dim = x_list[0][0].shape[0] # dimension of each data point x
    seg_res = []
    for i in range(nb_traj):
        t_points = t_list[i]
        x_points = x_list[i]
        nb_points = t_points.shape[0]
        assert(nb_points == x_points.shape[0])

        # assuming t_points are equally spaced
        h = t_points[1] - t_points[0]

        D = nb_points - 2
        
        X = np.zeros([D, x_dim])
        X_dot = np.zeros([D, x_dim])

        for j in range(D):
            X[j] = x_points[j+1]
            X_dot[j] = (x_points[j+2] - x_points[j]) / (2.0 * h)

        if (i==0):
            final_X = X
            final_X_dot = X_dot
        else:
            # if multiple trajectories, concatenate them
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]

        if len(segments) > 0:
            seg_curr = segments[i]
            begin_index = 0
            for j in range(0, i):
                begin_index += segments[j][-1] - 1
            seg_res.append(begin_index)
            for j in range(1, len(seg_curr) - 1):
                seg_res.append(begin_index + seg_curr[j] - 1)
    
    if len(segments) > 0:
        return final_X, final_X_dot, seg_res
    else:
        return final_X, final_X_dot

# Five point stencil differentiation method, using two points from the back and two from the front
def diff_method_fps(t_list, x_list, segments=[], remove_seg=False):
    nb_traj = t_list.shape[0] # number of arrays of t-points in t_list
    assert(nb_traj == x_list.shape[0]) # should be equal to number of x-points in x_list
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
            seg_curr = segments[i]
            for k in range(len(seg_curr)-1):
                t_points_seg = t_points[seg_curr[k]:seg_curr[k+1]]
                x_points_seg = x_points[seg_curr[k]:seg_curr[k+1]]
                nb_points = t_points_seg.shape[0]
                assert(nb_points == x_points_seg.shape[0])
                # assuming t_points are equally spaced
                h = t_points_seg[1] - t_points_seg[0]
                D = nb_points - 4
                if D <= 0:
                    continue

                X_seg = np.zeros([D, x_dim])
                X_dot_seg = np.zeros([D, x_dim])
                for j in range(0, D):
                    X_seg[j] = x_points_seg[j+2]
                    X_dot_seg[j] = (-x_points_seg[j+4] + 8*x_points_seg[j+3] - 8*x_points_seg[j+1] + x_points_seg[j]) / (12.0 * h)

                if (k==0):
                    X = X_seg
                    X_dot = X_dot_seg
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
        return final_X, final_X_dot, seg_res
    else:
        return final_X, final_X_dot
