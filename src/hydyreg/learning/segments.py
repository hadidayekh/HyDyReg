"""Train/validation split by segments, and segment and mode lookups of points."""
import numpy as np
from sklearn.model_selection import train_test_split


def which_segment(i, segments):
    '''
    Given `segments` containing indices of change points, returns the segment identifier of point `i`.
    '''
    r = 0
    while r < len(segments) and i >= segments[r]:
        r += 1
    return r

def which_segment_all(N, segments):
    '''
    Given `segments` containing indices of change points, as well as a total number of points `N`, returns an array containing the segment identifier of each data point.
    '''
    segments_per_point = np.ndarray((N,), int)
    for i in range(N):
        segments_per_point[i] = which_segment(i, segments)
    return segments_per_point

def train_test_split_segments(X, Xdot, segments, segments_per_point, test_size=None, random_state=None):
    '''
    Splits data of each segment into train and test data sets.

    Parameters
    ----------
    `X` : `ndarray (N x D)`
        The data points to be split.

    `Xdot` : `ndarray (N x D)`
        The corresponding derivatives of points to be split.

    `segments` : `list` of `int`
        The indices of beginnings of each segment in `X` (i.e. list of change point indices).

    `segments_per_point` : `ndarray (N)`
        The segment identifiers of each data point. Equivalent to `which_segment_all(N, segments)`.
        *Note*: bad design. 

    `test_size`: `float`, default=`None`
        If float, should be between 0.0 and 1.0. Represents the portion of data to include in the test split. Defaults to 0.25.

    Returns
    -------
    `X_train` : `ndarray (N_train x D)`
        The training data set.

    `X_val` : `ndarray (N_val x D)`
        The validation or test data set.

    `Xdot_train` : `ndarray (N_train x D)`
        The corresponding derivatives of the training data set.

    `Xdot_val` : `ndarray (N_val x D)`
        The corresponding derivatives of the validation or test data set.

    `seg_train` : `ndarray (N_train)`
        The segment identifiers of each data point in the training data set.

    `seg_val` : `ndarray (N_val)`
        The segment identifiers of each data point in the validation or test data set.

    **Note**: `N_train` + `N_val` = `N`
    '''
    for r in range(len(segments)):
        if r < len(segments) - 1:
            X_curr      = X   [segments[r]:segments[r+1]]
            Xdot_curr   = Xdot[segments[r]:segments[r+1]]
        else:
            X_curr      = X   [segments[r]:]
            Xdot_curr   = Xdot[segments[r]:]

        if X_curr.shape[0] == 1:
            X_train_seg = X_curr
            X_val_seg = np.ndarray((0, X_curr.shape[1]))
            Xdot_train_seg = Xdot_curr
            Xdot_val_seg = np.ndarray((0, X_curr.shape[1]))
        else:
            X_train_seg, X_val_seg, Xdot_train_seg, Xdot_val_seg = train_test_split(X_curr, Xdot_curr, test_size=test_size, random_state=random_state)

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
