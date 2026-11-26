"""Checks of SNDS.simulate_trajectory against exact solutions (transversal crossing, Filippov sliding,
sliding exit, learned-model mode scores, no events on non-mode changes)."""
import numpy as np
import pytest

from hydyreg.systems import SNDS, SlidingMotionException

STEP = 0.01


def surface_x2():
    ev = lambda t, x: x[1]
    ev.terminal, ev.direction = True, 0
    return ev


def system(f0, f1, with_scores=False):
    lab = lambda x: 0 if x[1] > 0 else 1
    s = SNDS(2, 2, [f0, f1], [surface_x2()], lab)
    if with_scores:
        s.mode_scores = lambda x: np.array([x[1], -x[1]])
    return s


def at(trajs, t, step=STEP):
    return trajs.x_list[0][int(round(t / step))]


def close(got, exp, tol=1e-6):
    return np.max(np.abs(np.asarray(got) - np.asarray(exp))) < tol


@pytest.mark.parametrize("scores", [False, True], ids=["eventlist", "mode_scores"])
def test_transversal(scores):
    # transversal crossing at t=1, then x2' = -2
    s = system(lambda t, x: [1, -1], lambda t, x: [1, -2], scores)
    tr = s.simulate_trajectory(STEP, 2, [[0, 1]])
    assert close(at(tr, 2), [2, -2])
    assert s.sim_stats_['crossings'] == 1 and s.sim_stats_['sliding_segments'] == 0


@pytest.mark.parametrize("scores", [False, True], ids=["eventlist", "mode_scores"])
def test_sliding(scores):
    # sliding from t=1 along x2=0 with Filippov velocity (1, 0)
    s = system(lambda t, x: [1, -1], lambda t, x: [1, 1], scores)
    tr = s.simulate_trajectory(STEP, 3, [[0, 1]])
    assert close(at(tr, 3), [3, 0])
    assert s.sim_stats_['sliding_segments'] == 1
    with pytest.raises(SlidingMotionException):
        s.simulate_trajectory(STEP, 3, [[0, 1]], stop_at_sliding_motion=True)


@pytest.mark.parametrize("scores", [False, True], ids=["eventlist", "mode_scores"])
def test_sliding_exit(scores):
    # sliding until x1=2 (mode 1 stops pushing up), then exit into mode 1: x2 = -(t-2)^2/2
    s = system(lambda t, x: [1, -1], lambda t, x: [1, 2 - x[0]], scores)
    tr = s.simulate_trajectory(STEP, 3, [[0, 1]])
    assert close(at(tr, 3), [3, -0.5])
    assert close(at(tr, 1.5), [1.5, 0])  # on the surface at t=1.5


def test_no_events_when_lower_ranked_modes_swap():
    # four modes with linear scores; the trajectory stays in mode 0 while the ranks of modes 1-3 swap
    W = np.array([[0, 0, 10], [1, 0, 0], [-1, 0, 0], [0, 1, 0]])  # scores = W @ [x1, x2, 1]
    scores = lambda x: W @ np.array([x[0], x[1], 1.0])
    f = lambda t, x: [1, 0.5]
    s = SNDS(4, 2, [f, f, f, f], None, lambda x: int(np.argmax(scores(x))))
    s.mode_scores = scores
    tr = s.simulate_trajectory(STEP, 4, [[-2, -2]])
    st = s.sim_stats_
    assert st['crossings'] == 0 and st['sliding_segments'] == 0 and st['grazing'] == 0
    assert close(at(tr, 4), [2, 0])  # free flight
