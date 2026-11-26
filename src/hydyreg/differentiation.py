import numpy as np

# Five point stencil differentiation method, using two points from the back and two from the front
def diff_method_fps(t_list, x_list, segments=None):
    '''
    - calculates derivatives of trajectory points
    - returns a single matrix X with concatenated data along with the corresponding derivatives Xdot
    - if segments is provided:
        * calculates derivatives for each segment not trajectory
        * returns STARTING INDEX of segments INCLUDING 0 for X, Xdot
        * a segment of 4 points or fewer gives no derivative and no start index (as in `diff_method_sg`)
    '''
    nb_traj = len(t_list) # number of arrays of t-points in t_list
    assert(nb_traj == len(x_list)) # should be equal to number of x-points in x_list
    x_dim = x_list[0][0].shape[0] # dimension of each data point x
    seg_res = []
    offset = 0 # number of rows of the previous trajectories in final_X
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
                
                seg_res.append(X.shape[0] - X_seg.shape[0] + offset)



            
        if (i==0):
            final_X = X
            final_X_dot = X_dot
            final_Time = Time
        else:
            # if multiple trajectories, concatenate them
            final_X = np.r_[final_X, X]
            final_X_dot = np.r_[final_X_dot, X_dot]
            final_Time = np.r_[final_Time, Time]
        offset = final_X.shape[0]

    if segments is not None and len(segments) > 0:
        return final_X, final_X_dot, seg_res, final_Time
    else:
        return final_X, final_X_dot, final_Time


def estimate_derivative_error(t_list, x_list, segments, per_segment=False):
    '''
    Estimates the error of the five-point derivatives of `diff_method_fps`, from the data alone.

    In each segment, at every point with three neighbours on each side, the five-point estimate
    (error O(h^4)) is compared with the seven-point estimate (error O(h^6)); their difference
    approximates the error of the five-point estimate. The time step is the actual sample spacing,
    as in `diff_method_fps`.

    Returns an array with, for each such point, the largest absolute difference over the components.
    With `per_segment=True`, returns instead one array per segment of at least 5 points, in the order of
    the segments returned by `diff_method_fps` (empty for a segment of 5 or 6 points).
    '''
    errs = []
    for t_points, x_points, seg_curr in zip(t_list, x_list, segments):
        for piece in np.split(np.arange(len(t_points)), seg_curr):
            if len(piece) < 7:
                if per_segment and len(piece) >= 5:
                    errs.append(np.zeros(0))
                continue
            x = x_points[piece]
            h = t_points[piece[1]] - t_points[piece[0]]
            d5 = (-x[5:-1] + 8*x[4:-2] - 8*x[2:-4] + x[1:-5]) / (12*h)
            d7 = (x[6:] - 9*x[5:-1] + 45*x[4:-2] - 45*x[2:-4] + 9*x[1:-5] - x[:-6]) / (60*h)
            errs.append(np.abs(d5 - d7).max(axis=1))
    if per_segment:
        return errs
    return np.concatenate(errs) if errs else np.zeros(0)


def estimate_noise_std(x_list, order=6):
    '''
    Estimates the standard deviation of measurement noise on sampled trajectories (independent noise on
    every sample and component), from the data alone.

    The differences of order `order` of a smooth signal sampled with step h are of size h^order, while
    for independent noise of standard deviation sigma they have standard deviation
    sqrt(binom(2*order, order)) * sigma (30.4 sigma for order 6). The estimate is the median absolute
    deviation of these differences (robust to the few samples around switches), rescaled accordingly.
    Returns about 0 for noise-free data.
    '''
    from scipy.special import comb
    d = [np.diff(x, n=order, axis=0).ravel() for x in x_list if len(x) > order]
    if not d:
        return 0.0
    d = np.concatenate(d)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(comb(2 * order, order)))


# Noise amplification of the finite differences used here, for independent noise of standard deviation sigma
# per sample and component, and sample spacing h:
FPS_NOISE_GAIN = np.sqrt(130) / 12     # five-point derivative: 0.95 sigma / h per component
SEG_NOISE_GAIN = np.sqrt(70) / 2       # segmentation statistic (forward - backward derivative): 4.18 sigma / h per component


# ----------------------------------------------------------------------------------------------------------
# Estimators on a window of half-width m (2m+1 samples). The estimators of the thesis are the members m = 2:
# - the segmentation statistic (three-point forward minus backward derivative) is the least-squares estimate of
#   a derivative jump at the centre of the window, the smooth part being a cubic: weights (-1, 4, -6, 4, -1)/(2h);
# - the five-point derivative is the Savitzky-Golay derivative of degree 4 on 5 samples.
# Larger windows average the measurement noise (gains fall roughly as m^-1.5) at the price of a larger bias.

WINDOW_LADDER = (2, 3, 4, 6, 8, 12, 16, 24)
MIN_NOISY_HALF_WIDTH = 4   # with m > 2, segments use at least 2 * min(m, 4) + 1 samples


def tolerance_noise_gain(m, order=4):
    '''Noise gain to use in tolerances when the derivative window has half-width m: that of the narrowest
    window a kept segment may use (min(m, MIN_NOISY_HALF_WIDTH)). Longer segments use the full window and are
    less noisy than this.'''
    return sg_noise_gain(m if m <= 2 else min(m, MIN_NOISY_HALF_WIDTH), order)



def sg_noise_gain(m, order=4):
    '''Standard deviation of the Savitzky-Golay derivative at the centre of the window, for unit noise and unit
    step (0.95 for m = 2, the five-point derivative).'''
    from scipy.signal import savgol_coeffs
    return float(np.linalg.norm(savgol_coeffs(2 * m + 1, order, deriv=1, use='dot')))


def _sg_piece(x, h, m, order=4):
    '''Savitzky-Golay estimates on one segment x (L x D): (kept indices, smoothed states, derivatives).
    Points near the ends are estimated from the fit of the end window; max(2, m // 3) points are dropped at
    each end, where the estimate is much noisier. A segment shorter than the window uses the largest window
    that fits, down to 2 * min(m, MIN_NOISY_HALF_WIDTH) + 1 samples; shorter segments are dropped (their
    derivatives would be noisier than the tolerances assume, see `tolerance_noise_gain`). m = 2 is the
    five-point derivative on the raw states.'''
    from scipy.signal import savgol_filter
    L = len(x)
    if m > 2:
        win = min(2 * m + 1, L if L % 2 == 1 else L - 1)
        if win < 2 * min(m, MIN_NOISY_HALF_WIDTH) + 1:
            return np.zeros(0, dtype=int), np.zeros((0, x.shape[1])), np.zeros((0, x.shape[1]))
    if m <= 2:
        if L < 5:
            return np.zeros(0, dtype=int), np.zeros((0, x.shape[1])), np.zeros((0, x.shape[1]))
        idx = np.arange(2, L - 2)
        return idx, x[2:-2], (-x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]) / (12 * h)
    e = max(2, ((win - 1) // 2) // 3)
    idx = np.arange(e, L - e)
    xs = savgol_filter(x, win, order, deriv=0, axis=0, mode='interp')
    xd = savgol_filter(x, win, order, deriv=1, delta=h, axis=0, mode='interp')
    return idx, xs[idx], xd[idx]


def diff_method_sg(t_list, x_list, segments=None, m=2, order=4):
    '''
    States and derivatives by local polynomial fits (Savitzky-Golay, degree `order`, 2m+1 samples), segment by
    segment. Same outputs as `diff_method_fps` (X, X_dot, starts of the segments in X, times); the states are
    the smoothed ones when m > 2. With m = 2 the derivative is the five-point derivative of `diff_method_fps`.
    '''
    Xs, Xds, Ts, seg_res, n = [], [], [], [], 0
    for i, (t_points, x_points) in enumerate(zip(t_list, x_list)):
        seg_curr = segments[i] if segments is not None else []
        for piece in np.split(np.arange(len(t_points)), seg_curr):
            if len(piece) < 2:
                continue
            h = t_points[piece[1]] - t_points[piece[0]]
            idx, xs, xd = _sg_piece(x_points[piece], h, m, order)
            if len(idx) == 0:
                continue
            seg_res.append(n)
            n += len(idx)
            Xs.append(xs); Xds.append(xd); Ts.append(t_points[piece][idx])
    D = x_list[0].shape[1]
    X = np.concatenate(Xs) if Xs else np.zeros((0, D))
    X_dot = np.concatenate(Xds) if Xds else np.zeros((0, D))
    T = np.concatenate(Ts) if Ts else np.zeros(0)
    if segments is not None and len(segments) > 0:
        return X, X_dot, seg_res, T
    return X, X_dot, T


def choose_derivative_window(t_list, x_list, segments, sigma, order=4, ladder=WINDOW_LADDER):
    '''
    Window of the Savitzky-Golay derivative for trajectories with measurement noise `sigma`.

    For a half-width m, the error of the derivative has two parts:
    - noise: sigma * gain(m) / h, which decreases with m;
    - bias (the trajectory is not a polynomial of degree `order` over the window), which increases with m and
      is concentrated where the trajectory changes fast. It is estimated on the data as the difference between
      the estimates of degree `order` on 2m+1 samples and of degree `order + 2` on 2m+3 samples (for m = 2:
      five-point minus seven-point derivative), from which the noise of that difference is removed.
    The largest m for which the 99th percentile of the estimated bias stays below the noise is chosen: beyond
    it the error of some points grows quickly, while up to it the error is noise everywhere, so that
    tolerances based on the noise level remain valid for every segment. Without noise this gives m = 2.

    Returns (m, noise standard deviation of the derivative, 99th percentile of the estimated bias).
    '''
    from scipy.signal import savgol_coeffs
    pieces = []
    for t_points, x_points, seg_curr in zip(t_list, x_list, segments):
        for piece in np.split(np.arange(len(t_points)), seg_curr):
            if len(piece) >= 7:
                pieces.append((x_points[piece], t_points[piece[1]] - t_points[piece[0]]))
    h0 = t_list[0][1] - t_list[0][0]
    best = (2, sigma * sg_noise_gain(2, order) / h0, 0.0)
    for m in ladder:
        w_lo = np.r_[0.0, savgol_coeffs(2 * m + 1, order, deriv=1, use='dot'), 0.0]
        w_hi = savgol_coeffs(2 * m + 3, order + 2, deriv=1, use='dot')
        diffs, hs = [], []
        for x, h in pieces:
            if len(x) >= 2 * m + 3:
                d = np.stack([np.correlate(x[:, c], w_lo - w_hi, mode='valid') for c in range(x.shape[1])], axis=1) / h
                diffs.append(np.abs(d).max(axis=1)); hs.append(h)
        if not diffs:
            break
        h = float(np.median(hs))
        noise = sigma * sg_noise_gain(m, order) / h
        q = np.percentile(np.concatenate(diffs), 99)
        bias = np.sqrt(max(q ** 2 - (2.8 * sigma * np.linalg.norm(w_lo - w_hi) / h) ** 2, 0.0))
        if m == ladder[0]:
            best = (m, noise, bias)
        elif bias <= noise:
            best = (m, noise, bias)
        else:
            break
    return best


def auto_deltas(t_list, x_list, segments, X_dot, factor=2.0, rel_floor=1e-6, exclude=None, deriv_noise=0.0):
    '''
    Data-driven insensitivity error for the optimization-based method:
    `factor` times the largest estimated error of the five-point derivatives (see
    `estimate_derivative_error`), but at least `rel_floor` times the largest absolute derivative
    component, so that it does not fall below the precision of the solver.

    `exclude`: indices (in the order of `diff_method_fps`) of segments to leave out, e.g. segments that
    their own polynomial fit does not capture: an undetected switch inside a segment shows as a large
    difference at the points next to it, which is not a derivative error.
    '''
    per_seg = estimate_derivative_error(t_list, x_list, segments, per_segment=True)
    exclude = set(exclude or [])
    kept = [e for k, e in enumerate(per_seg) if k not in exclude and len(e)]
    err = np.concatenate(kept) if kept else np.zeros(0)
    scale = float(np.max(np.abs(X_dot))) if X_dot.size else 1.0
    # with measurement noise, the five-point derivatives also carry noise of standard deviation `deriv_noise`
    return max(factor * (float(err.max()) if err.size else 0.0) + 4 * deriv_noise, rel_floor * scale)


# Three point central differentiation method, using 1 point from the front
# ALLOWS FOR NON-UNIFORM GRID