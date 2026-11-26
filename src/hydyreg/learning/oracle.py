"""What the learner can ask of a system.

An oracle is any object with `simulate_trajectory(step, T, x0s, max_step_factor=...) -> Trajectories` (and
`generate_points_between_segments` for the thesis boundary queries). An `SNDS` is an oracle. `noisy_oracle`
returns a view of a system whose simulated trajectories carry measurement noise.
"""
import copy

import numpy as np

NOISE_SEED = 12345


def noisy_oracle(system, noise_std, seed=NOISE_SEED):
    """A copy of `system` whose `simulate_trajectory` adds independent Gaussian noise of standard deviation
    `noise_std` to every sample and component, drawn in call order from np.random.default_rng(seed) (`seed`: an
    integer, a SeedSequence or a Generator). Methods
    that simulate through `self.simulate_trajectory` (boundary queries) are noisy too. noise_std = 0 returns
    the system itself."""
    if not noise_std > 0:
        return system
    noise_rng = np.random.default_rng(seed)
    obs = copy.copy(system)

    def noisy_simulate(*args, **kwargs):
        trajs = system.simulate_trajectory(*args, **kwargs)
        trajs.x_list = [x + noise_std * noise_rng.normal(size=x.shape) for x in trajs.x_list]
        return trajs
    obs.simulate_trajectory = noisy_simulate
    return obs
