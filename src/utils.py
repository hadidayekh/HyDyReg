import numpy as np
from sklearn.model_selection import train_test_split
def which_segment(i, segments):
    r = 0
    while r < len(segments) and i >= segments[r]:
        r += 1
    return r - 1

def which_segment_all(N, segments):
    segments_per_point = np.zeros(N)
    for i in range(N):
        segments_per_point[i] = which_segment(i, segments)
    return segments_per_point

def train_test_split_segments(X, Xdot, segments, segments_per_point):
    for r in range(len(segments)):
        if r < len(segments) - 1:
            X_curr      = X   [segments[r]:segments[r+1]]
            Xdot_curr   = Xdot[segments[r]:segments[r+1]]
        else:
            X_curr      = X   [segments[r]:]
            Xdot_curr   = Xdot[segments[r]:]

        X_train_seg, X_val_seg, Xdot_train_seg, Xdot_val_seg = train_test_split(X_curr, Xdot_curr)

        seg_curr    = segments_per_point[segments[r]]
        
        if r == 0:
            X_train     = X_train_seg
            Xdot_train  = Xdot_train_seg
            X_val       = X_val_seg
            Xdot_val    = Xdot_val_seg
            seg_train   = np.ones(X_train_seg.shape[0]) * seg_curr
            seg_val     = np.ones(X_val_seg.shape[0])   * seg_curr
        
        else:
            X_train     = np.r_[X_train, X_train_seg]
            Xdot_train  = np.r_[Xdot_train, Xdot_train_seg]
            X_val       = np.r_[X_val, X_val_seg]
            Xdot_val    = np.r_[Xdot_val, Xdot_val_seg]
            seg_train   = np.r_[seg_train, np.ones(X_train_seg.shape[0]) * seg_curr]
            seg_val     = np.r_[seg_val, np.ones(X_val_seg.shape[0]) * seg_curr]

    return X_train, X_val, Xdot_train, Xdot_val, seg_train, seg_val

