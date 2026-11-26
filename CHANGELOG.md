# Changelog

## 2.0 (2026-11-26)

HyDyReg becomes the installable Python package `hydyreg`, with improved learning methods. Because the methods
changed, this version no longer reproduces the tables of the thesis; the tag `thesis-reproducible` holds the code that
does, and its README explains the setup. Version 1.0 (tag `thesis-reproducible`) is the code of the thesis.

### The package
- Installation with pip, on Python 3.11 or later. The dependencies are given as lower bounds (numpy >= 2.0,
  scipy >= 1.13, scikit-learn >= 1.5, casadi >= 3.6, matplotlib >= 3.8.4), so that hydyreg installs alongside other
  packages. The reported results come from the reference environment, `requirements-lock.txt` (Python 3.12; see
  "Reference environment" in the README). Licence: GNU GPL v3 or later.
- Python API: `learn_from_data` learns from recorded trajectories (for example CSV files), without a simulator;
  `learn_from_system` learns actively from a system that can be simulated. Both are configured by one `LearnConfig`
  (see "Configuration reference" in the README) and return a `LearnResult`.
- Command line: `hydyreg fit` (from data files), `hydyreg learn` (from a simulated system), and `hydyreg bench`,
  which runs the 26 experiments of the thesis, over several seeds if asked, and compares them with the thesis values.
- Learned models can be saved and loaded as JSON, and exported as text, LaTeX, MATLAB or SpaceEx.
- Every run has a seed (`LearnConfig.seed`, default 0) from which all its random draws come, so a run gives the same
  result every time.
- Unit tests, regression tests on short learning runs, and continuous integration on Linux, with the minimum
  versions, the newest releases and the reference environment.

### Changes to the methods
These change the results compared with the thesis.
- **Simulation.** Mode changes are located exactly, and trajectories can slide along a boundary between two modes.
- **Mode classifier.** The learned boundaries no longer depend on where a solver stops; the mode regions are
  still polytopes. A decision list and one-versus-one classifiers are available as options (`classifier.kind`).
- **Boundary refinement.** Most switches are now located with the learned dynamics, which needs fewer queries to the
  system (`boundaries.points`).
- **Measurement noise.** The tolerances adapt to the noise level estimated from the data (`noise.adapt`).
- **Incremental method.** Undetected switches no longer create spurious modes.
- **Optimization-based method.** More reliable solutions and more accurate dynamics (`optimization.deltas`,
  `optimization.refit`, `optimization.C_space`).
- **Equivalence queries.** A stricter default tolerance (`equivalence.tol` = 0.01; 0.1 in the thesis), so that
  learning no longer stops after a first query that passes too easily.

### Results
On the 26 experiments of the thesis, over seeds 0 to 4, every run terminates (the last equivalence query finds no
counterexample) with the correct number of modes. The final error d3 is the mean relative difference between
trajectories of the learned model and of the system, from the test initial conditions. With seed 0, it is below the
thesis value in all 26 experiments, for example 0.0021 instead of 0.0206 for `affine_3modes` (optimization-based
method) and 0.0006 instead of 0.0118 for the parametric example with 10 modes (incremental method). The intermediate
errors (d1 after the passive step, d2 after the boundary refinement) can be larger than in the thesis; the active
steps correct them. `hydyreg bench compare --ref thesis` shows the comparison.

### Known limitations
Measurement noise remains the weak point. With noisy data, learning fails for the polynomial example
`poly3_2modes_nonlinclass` from a relative noise of 1e-3, for `bio_sys` at 1e-2, and for `ham` already at 1e-4,
where a spurious mode appears.
