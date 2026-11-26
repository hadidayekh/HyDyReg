"""Change-point detection: the jump ("kink") estimator on a window of 2m+1 samples and the choice of its window.

The segmentation statistic of the thesis (three-point forward minus backward derivative) is the member m = 2.
`Trajectories.segment` applies it.
"""
import numpy as np

from .differentiation import WINDOW_LADDER


def kink_weights(m, order=3):
    '''Weights (for h = 1) of the least-squares estimate of a derivative jump at the centre of 2m+1 samples,
    for the model  x(t) = polynomial of degree `order` + J * max(t, 0).'''
    t = np.arange(-m, m + 1, dtype=float)
    A = np.c_[np.vander(t, order + 1, increasing=True), np.maximum(t, 0.0)]
    return np.linalg.pinv(A)[-1]


def kink_statistic(x, h, m=2, order=3):
    '''Norm (over the components) of the estimated derivative jump at every sample of a trajectory x (N x D).
    0 where the window does not fit (the first and last m samples).'''
    N = len(x)
    stat = np.zeros(N)
    if N >= 2 * m + 1:
        w = kink_weights(m, order) / h
        J = np.stack([np.correlate(x[:, d], w, mode='valid') for d in range(x.shape[1])], axis=1)
        stat[m:N - m] = np.linalg.norm(J, axis=1)
    return stat


def kink_noise_gain(m, order=3):
    '''Standard deviation of one component of the estimated jump, for unit noise and unit step (4.18 for m = 2).'''
    return float(np.linalg.norm(kink_weights(m, order)))


def choose_segmentation_window(x_list, h, seg_tol, sigma, z=4.0, ladder=WINDOW_LADDER):
    '''
    Window and threshold of the segmentation statistic for trajectories with measurement noise `sigma`.

    If the noise of the 5-sample statistic allows a jump of size `seg_tol` to be detected
    (z * sigma * gain(2) * sqrt(D) / h <= seg_tol), the thesis statistic and `seg_tol` are kept.
    Otherwise, for each larger window, the level of the statistic where there is no switch is measured on the
    data themselves (median + 5 scaled median absolute deviations of the statistic: most samples are not
    switches; this level contains both the noise and the response to the curvature of the trajectory), and the
    window with the lowest level is chosen; the threshold is that level, but at least `seg_tol`.

    Returns (m, threshold).
    '''
    D = x_list[0].shape[1]
    if z * sigma * kink_noise_gain(2) * np.sqrt(D) / h <= seg_tol:
        return 2, seg_tol
    n_min = min(len(x) for x in x_list)
    best = (2, max(seg_tol, z * sigma * kink_noise_gain(2) * np.sqrt(D) / h))
    for m in ladder[1:]:
        if 4 * m + 2 > n_min:
            break
        st = np.concatenate([kink_statistic(x, h, m)[m:len(x) - m] for x in x_list])
        med = np.median(st)
        level = med + 5 * 1.4826 * np.median(np.abs(st - med))
        thr = max(seg_tol, level)
        if thr < best[1]:
            best = (m, thr)
    return best
