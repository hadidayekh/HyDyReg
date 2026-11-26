"""Fixed tolerances of the learner, in one place. They are the values used for every result of this package; change
them only with a rerun of the benchmarks. `GOOD_RD` is also the default of `LearnConfig.rd_tol`.

Relative differences (RD) are those of `hydyreg.metrics.relative_diff`: |a - b| / (|a| + |b|) per point, averaged.
"""

# tol_RD of the thesis (Algorithm 4): a mode captures a set of points if the mean RD between their estimated
# derivatives and the mode's prediction is below it. Used to assign segments, found and merge modes, and to find
# segments with an undetected switch. Measurement noise raises it (HybridRegressor._rd_tol).
GOOD_RD = 1e-3

# A single point (or a fine point of a boundary query) belongs to a mode if its RD to that mode is below this.
POINT_RD = 1e-2

# Switch location inside a gap (HybridRegressor.locate_switch): the reconstruction may miss the observed states
# by at most SWITCH_ACCEPT x ||x_B - x_A||, and at most SWITCH_RESOLVE x the miss of a switch at either end.
SWITCH_ACCEPT = 1e-2
SWITCH_RESOLVE = 0.2

# A fine-step boundary query is used only if the RD that measurement noise alone gives its derivatives is below this.
QUERY_NOISE_RD = 0.05

# Number of random test initial conditions (SamplingDesign.test_x0s = 'random'), drawn with a fixed seed.
N_TEST_POINTS = 10
TEST_POINTS_SEED = 0
