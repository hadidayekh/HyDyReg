"""HyDyReg: identification of switched nonlinear dynamical systems from trajectories, with active learning.

A switched system has several modes, each with polynomial dynamics dx/dt = f_j(x), and regions of the state
space where each mode is active. HyDyReg learns the modes, their dynamics and their regions from sampled
trajectories, with the two methods of the thesis (https://hal.science/tel-05073907/): the optimization-based
method and the incremental method. Given a system that can be simulated, it can also learn actively (boundary
refinement and equivalence queries).

    import hydyreg as hd

    trajs = hd.Trajectories.from_csv("runs/*.csv")
    result = hd.learn_from_data(trajs, hd.LearnConfig(method="incremental", seg_tol=0.01))
    print(result.equations())
    hd.save_model(result.model, "model.json")

Main names (all importable from `hydyreg`; each links to the page that documents it):

- Data: `hydyreg.trajectories.Trajectories`.
- Systems to simulate: `hydyreg.systems.SNDS`, `hydyreg.systems.SNDS_affine`,
  `hydyreg.systems.SlidingMotionException`.
- Learning: `hydyreg.learning.pipeline.learn_from_data`, `hydyreg.learning.pipeline.learn_from_system`,
  `hydyreg.learning.pipeline.Learner`, `hydyreg.learning.result.LearnResult`.
- Configuration: `hydyreg.learning.config.LearnConfig` and its groups `hydyreg.learning.config.ClassifierConfig`,
  `hydyreg.learning.config.OptimizationConfig`, `hydyreg.learning.config.SimulationConfig`,
  `hydyreg.learning.config.BoundaryConfig`, `hydyreg.learning.config.EquivalenceConfig`,
  `hydyreg.learning.config.NoiseConfig`; `hydyreg.learning.config.SamplingDesign`.
- Model: `hydyreg.model.HybridRegressor`; classifiers `hydyreg.classifier.StandardizedLogisticRegression`,
  `hydyreg.classifier.DecisionListLogisticRegression`, `hydyreg.classifier.PairwiseLogisticRegression`.
- Saving: `hydyreg.persistence.save_model`, `hydyreg.persistence.load_model`.
- Submodules: `hydyreg.plotting`, `hydyreg.export`, `hydyreg.benchmarks`.
"""
import importlib

__version__ = "2.0.0"

_NAMES = {
    "Trajectories": "hydyreg.trajectories",
    "SNDS": "hydyreg.systems", "SNDS_affine": "hydyreg.systems", "SlidingMotionException": "hydyreg.systems",
    "HybridRegressor": "hydyreg.model",
    "StandardizedLogisticRegression": "hydyreg.classifier",
    "DecisionListLogisticRegression": "hydyreg.classifier",
    "PairwiseLogisticRegression": "hydyreg.classifier",
    "LearnConfig": "hydyreg.learning.config", "SamplingDesign": "hydyreg.learning.config",
    "ClassifierConfig": "hydyreg.learning.config", "NoiseConfig": "hydyreg.learning.config",
    "BoundaryConfig": "hydyreg.learning.config",
    "OptimizationConfig": "hydyreg.learning.config", "SimulationConfig": "hydyreg.learning.config",
    "EquivalenceConfig": "hydyreg.learning.config",
    "learn_from_data": "hydyreg.learning.pipeline", "learn_from_system": "hydyreg.learning.pipeline",
    "Learner": "hydyreg.learning.pipeline",
    "LearnResult": "hydyreg.learning.result",
    "save_model": "hydyreg.persistence", "load_model": "hydyreg.persistence",
}
_SUBMODULES = {"plotting", "export", "benchmarks", "learning", "model", "differentiation", "segmentation",
               "metrics", "persistence", "systems", "trajectories", "classifier", "regression", "cli"}

__all__ = sorted(_NAMES) + sorted(_SUBMODULES) + ["verbose"]


def verbose(level="INFO"):
    """Shows hydyreg's messages on the console: 'INFO' (progress and results), 'DEBUG' (details), 'WARNING'."""
    from ._logging import verbose as _verbose
    _verbose(level)


def __getattr__(name):
    if name in _NAMES:
        return getattr(importlib.import_module(_NAMES[name]), name)
    if name in _SUBMODULES:
        return importlib.import_module("hydyreg." + name)
    raise AttributeError(f"module 'hydyreg' has no attribute {name!r}")


def __dir__():
    return __all__
