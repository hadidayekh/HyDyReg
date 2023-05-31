import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def which_segment(i, segments):
    r = 0
    while r < len(segments) and i >= segments[r]:
        r += 1
    return r - 1

def which_segment_all(N, segments):
    segments_per_point = np.ndarray((N,), int)
    for i in range(N):
        segments_per_point[i] = which_segment(i, segments)
    return segments_per_point

def train_test_split_segments(X, Xdot, segments, segments_per_point, test_size=None):
    for r in range(len(segments)):
        if r < len(segments) - 1:
            X_curr      = X   [segments[r]:segments[r+1]]
            Xdot_curr   = Xdot[segments[r]:segments[r+1]]
        else:
            X_curr      = X   [segments[r]:]
            Xdot_curr   = Xdot[segments[r]:]

        X_train_seg, X_val_seg, Xdot_train_seg, Xdot_val_seg = train_test_split(X_curr, Xdot_curr, test_size=test_size)

        seg_curr    = segments_per_point[segments[r]]
        
        if r == 0:
            X_train     = X_train_seg
            Xdot_train  = Xdot_train_seg
            X_val       = X_val_seg
            Xdot_val    = Xdot_val_seg
            seg_train   = np.ones(X_train_seg.shape[0]).astype(int) * seg_curr
            seg_val     = np.ones(X_val_seg.shape[0]).astype(int)   * seg_curr
        
        else:
            X_train     = np.r_[X_train, X_train_seg]
            Xdot_train  = np.r_[Xdot_train, Xdot_train_seg]
            X_val       = np.r_[X_val, X_val_seg]
            Xdot_val    = np.r_[Xdot_val, Xdot_val_seg]
            seg_train   = np.r_[seg_train, np.ones(X_train_seg.shape[0]).astype(int) * seg_curr]
            seg_val     = np.r_[seg_val, np.ones(X_val_seg.shape[0]).astype(int) * seg_curr]

    return X_train, X_val, Xdot_train, Xdot_val, seg_train, seg_val

def get_modes_val(Chis_seg, seg_val):
    # Chis_seg represents Chis per segment, from which we deduce mode per segment
    # seg_val tells us the segment of each validation point

    # this gives us the mode of each segment
    modes_seg = np.argmin(Chis_seg, axis=0)

    modes_val = np.ndarray((len(seg_val)), int)
    for i in range(len(seg_val)):
        modes_val[i] = modes_seg[seg_val[i]]
    
    return modes_val

def compare_trajs(traj1, traj2):
    assert np.equal(np.array(traj1.t_list), np.array(traj2.t_list)).all(), "Cannot compare different timed trajectories."
    return mean_squared_error(np.array(traj1.x_list).flatten(), np.array(traj2.x_list).flatten())

def compare_snds(snds1, snds2, x0s, time_step, time_range):
    traj1 = snds1.simulate_trajectory(time_step, time_range, x0s)
    traj2 = snds2.simulate_trajectory(time_step, time_range, x0s)
    return compare_trajs(traj1, traj2)