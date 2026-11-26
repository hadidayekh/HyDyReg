"""Errors of a learned model: its trajectories against reference trajectories, by relative difference."""
from ..metrics import compare_trajs
from ..systems import SlidingMotionException
from .. import _logging as _log


def evaluate(model, train_ref, train_design, test_ref=None, test_design=None, max_step_factor=10):
    """Simulates the learned model from the initial conditions of the reference trajectories and compares.

    train_ref: reference training trajectories (unsegmented); train_design: (time_step, time_range, x0s).
    test_ref, test_design: the same for the test set (optional).
    Returns (errors, learned trajectories): errors = {'train': (mean, max), 'test': (mean, max)} relative
    differences. A model that cannot be simulated gets errors of 1.
    """
    learned = model.to_SNDS()
    try:
        trajs_train = learned.simulate_trajectory(*train_design, max_step_factor=max_step_factor)
        trajs_test = learned.simulate_trajectory(*test_design, max_step_factor=max_step_factor) if test_ref is not None else None
    except SlidingMotionException:
        # the integrator failed on the learned model (e.g. it diverges): report the largest relative difference
        _log.warning("The learned model could not be simulated on the training or test initial conditions; its errors are reported as 1.")
        errors = {'train': (1.0, 1.0)}
        if test_ref is not None:
            errors['test'] = (1.0, 1.0)
        return errors, (None, None)
    errors = {'train': compare_trajs(train_ref, trajs_train, measure="RD")}
    if test_ref is not None:
        errors['test'] = compare_trajs(trajs_test, test_ref, measure="RD")
    return errors, (trajs_train, trajs_test)


def evaluate_on_data(model, trajs, max_step_factor=10):
    """Errors of a learned model against recorded trajectories: each trajectory is simulated from its first
    sample over its own duration, on its own time grid, and compared by relative difference. Returns
    ((mean, max) over trajectories, learned trajectories). A model that cannot be simulated gets (1, 1)."""
    import numpy as np
    from ..metrics import relative_diff
    learned = model.to_SNDS()
    rds, sims = [], []
    for t, x in zip(trajs.t_list, trajs.x_list):
        t = np.asarray(t, dtype=float)
        h, T = t[1] - t[0], t[-1] - t[0]
        try:
            sim = learned.simulate_trajectory(h, T, [x[0]], max_step_factor=max_step_factor)
        except SlidingMotionException:
            return (1.0, 1.0), None
        xs = sim.x_list[0]
        n = min(len(xs), len(x))
        rds.append(relative_diff(np.asarray(x)[:n], xs[:n]))
        sims.append(sim)
    return (float(np.mean(rds)), float(np.max(rds))), sims
