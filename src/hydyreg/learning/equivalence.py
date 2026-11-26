"""Approximate equivalence queries."""
import numpy as np
from tqdm import tqdm

from ..metrics import compare_trajs
from ..systems import SNDS, SlidingMotionException
from ..trajectories import Trajectories
from .. import _logging as _log


def generate_random_points(n_points, bounds, random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng(0)
    x0s = random_generator.random((n_points, bounds.shape[0]))
    for d in range(bounds.shape[0]):
        x0s.T[d] = x0s.T[d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return x0s


def equivalence_query_by_traj(sul: SNDS, learned_snds: SNDS, n_trajs, bounds, time_step, time_range, tol_eq=0.1, N_cexs=1, max_step_factor=10, rng=None):
    '''
    Equivalence query approaximation check.
    Simulates multiple trajectories of the SUL and the learned SNDS and compares them up to a certain tolerance to deduce counterexamples.

    Parameters
    ----------

    `sul` : `SNDS`
        The system-under-learning.

    `learned_snds` : `SNDS`
        The learned switched nonlinear dynamical system to evaluate.

    `n_trajs` : `int`
        The number of trajectories to simulate and compare in an equivalence query check.

    `bounds` : `ndarray` `(d, 2)`
        The lower and upper bounds of each dimension of the state vector.
        Initial conditions are generated within thsese bounds.

    `time_step` : `float`
        The simulation time step of trajectories.

    `time_range` : `float`
        The simulation time range of trajectories.

    `tol_eq` : `float`, default=`0.1`
        The tolerance for the average relative difference between points of a pair of trajectories to decide whether this pair is a counterexample.

    `N_cexs` : `int`, default=`1`
        The maximal number of counterexamples.
        The equivalence query would return at the first `N_cexs` counterexamples rather than continue simulating `n_trajs` trajectories.
    
    `rng` : `numpy.random.Generator`, default=None
        Generator of the initial conditions (None: a new generator with seed 0).

    `max_step_factor` : `int`, default=`10`
        For simulation purposes, indicates the maximum time step of the simulator, such that the simulator maximum time step is `time_step / max_step_factor`.
    `

    Returns
    -------
    `eq_query` : bool
        The result of equivalence.
    
    `cexs_sul` : `Trajectories`
        The SUL part of counterexample pairs of trajectories.
    
    `conforming_trajs_sul` : `Trajectories`
        The SUL part of the conforming (thus non-counterexample) pairs of trajectories.

    `cexs_learned` : `Trajectories`
        The learned SNDS part of counterexample pairs of trajectories.

    `conforming_trajs_learned` : `Trajectories`
        The learned SNDS part of the conforming (thus non-counterexample) pairs of trajectories.
    '''
    cexs_sul = Trajectories([], [])
    cexs_sul.d = sul.dim
    conforming_trajs_sul = Trajectories([], [])
    conforming_trajs_sul.d = sul.dim
    cexs_learned = Trajectories([], [])
    cexs_learned.d = sul.dim
    conforming_trajs_learned = Trajectories([], [])
    conforming_trajs_learned.d = sul.dim
    total_RD = 0
    cexs_RD = 0

    x0s = generate_random_points(n_trajs, bounds, random_generator=rng if rng is not None else np.random.default_rng(0))

    for i in tqdm(range(n_trajs), disable=not _log.enabled()):
        if cexs_sul.num_trajectories == N_cexs:
            return False, cexs_sul, conforming_trajs_sul, cexs_learned, conforming_trajs_learned
        x0 = x0s[i]
        sul_traj = sul.simulate_trajectory(time_step, time_range, [x0], max_step_factor=max_step_factor)
        try:
            learned_traj = learned_snds.simulate_trajectory(time_step, time_range, [
                                                            x0], compare_with_traj=None, max_step_factor=max_step_factor)
        except SlidingMotionException:
            # the integrator failed on the learned model
            cexs_sul.append_trajs(sul_traj)
            continue

        if learned_traj.truncated:
            cexs_sul.append_trajs(sul_traj)
            continue
        comp_traj_test_RD = compare_trajs(learned_traj, sul_traj, measure="RD")[0]
        total_RD += comp_traj_test_RD

        if comp_traj_test_RD >= tol_eq:
            cexs_sul.append_trajs(sul_traj)
            cexs_learned.append_trajs(learned_traj)
            cexs_RD += comp_traj_test_RD
        else:
            conforming_trajs_sul.append_trajs(sul_traj)
            conforming_trajs_learned.append_trajs(learned_traj)

    _log.info("Total RD:", total_RD/n_trajs)
    if cexs_sul.num_trajectories > 0:
        _log.info("Number of counterexamples:", cexs_sul.num_trajectories)
        _log.info("Counterexamples RD:", cexs_RD / cexs_sul.num_trajectories)
    return cexs_sul.num_trajectories == 0, cexs_sul, conforming_trajs_sul, cexs_learned, conforming_trajs_learned
