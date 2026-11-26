"""Relative differences between derivatives, trajectories and systems."""
import numpy as np
from sklearn.metrics import mean_squared_error


def compare_trajs(traj1, traj2, measure="MSE"):
    assert measure == "MSE" or measure == "RD"
    assert np.equal(np.array(traj1.t_list), np.array(traj2.t_list)).all(), "Cannot compare different timed trajectories."
    if measure == "MSE":
        return mean_squared_error(np.array(traj1.x_list).flatten(), np.array(traj2.x_list).flatten())
    elif measure == "RD":
        RDs = []
        for i in range(len(traj1.x_list)):
            xpts1 = traj1.x_list[i]
            xpts2 = traj2.x_list[i]
            RDs.append(relative_diff(xpts1, xpts2))
        return np.mean(RDs), np.max(RDs)


def compare_snds(snds1, snds2, x0s, time_step, time_range, measure="MSE"):
    traj1 = snds1.simulate_trajectory(time_step, time_range, x0s)
    traj2 = snds2.simulate_trajectory(time_step, time_range, x0s)
    return compare_trajs(traj1, traj2, measure=measure)

def relative_diff(x, y):
    if x.ndim == 1:
        return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))
    else:
        return np.linalg.norm(x - y, axis=1) / (np.linalg.norm(x, axis=1) + np.linalg.norm(y,axis=1))

def relative_diff_by_component(x, y):
    res = np.zeros((x.size,))
    for i in range(x.size):
        res[i] = np.abs(x[i]-y[i]) / (np.abs(x[i]) + np.abs(y[i]))
    return res
