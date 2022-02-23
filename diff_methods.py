import numpy as np

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
        
        X = np.zeros([2*D, x_dim])
        X_dot = np.zeros([2*D, x_dim])

        for j in range(2*D):
            if j < D:
                X[j] = x_points[j]
                X_dot[j] = (-137*x_points[j] + 300*x_points[j+1] - 300*x_points[j+2] + 200*x_points[j+3] - 75*x_points[j+4] + 12*x_points[j+5])*h/60
            else:
                X[j] = x_points[j-D+5]
                X_dot[j] = (137*x_points[j-D+5] - 300*x_points[j-D+4] + 300*x_points[j-D+3] - 200*x_points[j-D+2] + 75*x_points[j-D+1] - 12*x_points[j-D])*h/60
    
        if (i==0):
            final_X = X
            final_X_dot = X_dot
        else:
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
    
    return final_X, final_X_dot


def diff_method_central(t_list, x_list):
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
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
    
    return final_X, final_X_dot

def diff_method_fps(t_list, x_list):
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

        D = nb_points - 4
        
        X = np.zeros([D, x_dim])
        X_dot = np.zeros([D, x_dim])

        for j in range(0, D):
            X[j] = x_points[j+2]
            X_dot[j] = (-x_points[j+4] + 8*x_points[j+3] - 8*x_points[j+1] + x_points[j]) / (12.0 * h)
            
        if (i==0):
            final_X = X
            final_X_dot = X_dot
        else:
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
    
    return final_X, final_X_dot
