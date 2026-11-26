# HyDyReg

HyDyReg identifies **switched nonlinear dynamical systems** from sampled trajectories. A switched system has
several modes, each with polynomial dynamics $\dot x = f_j(x)$, and regions of the state space where each mode
is active. HyDyReg learns the number of modes, their dynamics and their regions, with the two methods of my
[PhD thesis](https://hal.science/tel-05073907/): the *optimization-based* method (a nonlinear program that assigns
segments to a given number of modes) and the *incremental* method (segments are treated one by one and modes are
created when needed). It learns *passively* from recorded data, or *actively* from a system it can simulate
(boundary refinement and approximate equivalence queries).

The thesis tables are reproduced by the version tagged `thesis-reproducible`. This version contains later
improvements, summarized in [`CHANGELOG.md`](CHANGELOG.md). Related papers and how to cite: [Publications](#publications).

## Installation

Python 3.11 or later, on Linux and macOS (x86_64 and arm64). The dependencies (numpy 2, scipy, scikit-learn,
casadi, matplotlib) are given as lower bounds: the test suite is checked with those minimum versions, with the newest
releases, and in the reference environment.

```
uv venv .venv                                      # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -e .      # or: .venv/bin/pip install -e .
```

Extras: `-e ".[dev]"` (pytest, pyflakes), `-e ".[notebook]"` (JupyterLab for the examples).

**Reference environment.** The regression references and the reported results were recorded with Python 3.12 and
the exact versions in `requirements-lock.txt`. Other versions give the same models up to the last digits, but a few
fragile benchmark rows (for example `incr/poly3`) can differ more. To reproduce reported numbers exactly:

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

`hydyreg bench run` prints which versions differ from this environment.

## Quick start

### Python

```python
import hydyreg as hd

trajs = hd.Trajectories.from_csv("examples/data/isolette_*.csv")   # one file per trajectory: t, x1, x2, ...
config = hd.LearnConfig(method="incremental", poly_degree=1, boundary_degree=1, seg_tol=0.01)
result = hd.learn_from_data(trajs, config)

print(result.summary())
print(result.equations())            # result.equations("latex") for LaTeX
result.plot("regions")               # also "comparison", "modes"
hd.save_model(result.model, "model.json")
model = hd.load_model("model.json")
sim = model.to_SNDS().simulate_trajectory(0.25, 25, [[100.0, 90.0]])
```

Active learning on a system that can be simulated (any `SNDS`, here a benchmark of the thesis):

```python
system = hd.benchmarks.systems.isolette()
design = hd.SamplingDesign(time_step=0.25, time_range=25, x0s=[[105.5, 80], [97.5, 100]])
config = hd.LearnConfig(method="incremental", seg_tol=0.01,
                        equivalence=hd.EquivalenceConfig(bounds=[[95, 105], [70, 110]], tol=0.05))
result = hd.learn_from_system(system, design, config)
result.terminated                    # True: the last equivalence query found no counterexample
```

`hd.verbose()` shows progress messages (`hd.verbose("DEBUG")` for details); the library is silent otherwise.

### Command line

```
hydyreg fit examples/data/isolette_*.csv --seg-tol 0.01 --out result/ --latex --plots
hydyreg fit --config examples/configs/fit_optimization.toml --set optimization.n_starts=10 --out result/
hydyreg learn --system hydyreg.benchmarks.systems:isolette --config examples/configs/learn_isolette.toml --out result/
hydyreg simulate result/model.json --x0 100,90 --step 0.25 --time 25 --out sim.csv
hydyreg export result/model.json --format latex          # text, latex, matlab, spaceex
```

`fit` and `learn` write `model.json`, `report.txt` (modes, equations, regions, errors), `run.json` (configuration,
run information, versions), and with `--latex` / `--plots` also `equations.tex` and figures. Options:
`hydyreg <command> --help`. Exit codes: 0 success, 1 a run failed or did not terminate, 2 usage or configuration
error.

### Notebooks

`examples/01_quickstart.ipynb` (a two-mode system from scratch), `02_learn_from_csv.ipynb` (recorded data, both
methods), `03_active_learning.ipynb` (equivalence queries; the decision-list classifier on a T-junction).

## Configuration reference

A `LearnConfig` (Python) or a TOML file (command line; the same names, groups as tables; see
`examples/configs/`). The defaults are the values of the benchmark experiments. The last column gives the
corresponding keyword of the former function `learn_system` (removed; it is the interface at the tag `thesis-reproducible`).

The package does not reproduce the thesis behaviour through options (that is the tag `thesis-reproducible`).
These former options are now fixed behaviour: Algorithm 4 with the split step, gate on new modes and merging
(`robust_segments`); the exact Newton solver of the classifier (`solver`); continuation in C (`opti_warm_start`),
seeded starts (`opti_init_seed=None` removed), slacks kept nonnegative exactly (`opti_bound_relax_factor`), the refit
on all points after C is selected (`opti_refit_all`), splitting of training segments with a missed switch
(`opti_split_segments`), no early stop of the search over C (`break_at_low_MSE`), Filippov sliding in equivalence
queries (`stop_at_sliding_motion`).

| Field | Default | Meaning | Former keyword |
|---|---|---|---|
| `method` | `"incremental"` | `"incremental"` or `"optimization"` | `fit_active_only` |
| `n_modes` | `None` | number of modes (required by the optimization-based method) | `nmodes` |
| `poly_degree` | `1` | largest degree of the dynamics | `poly_degree` |
| `boundary_degree` | `1` | largest degree of the boundary features | `classifier_poly_deg` |
| `seg_tol` | `0.01` | derivative jump that marks a switch | `seg_tol` |
| `rd_tol` | `0.001` | tol_RD of the thesis: a mode captures points whose mean relative difference to its prediction is below it (raised automatically under noise) | — (fixed `good_RD`) |
| `seed` | `0` | seed of the run: every random draw (query initial conditions, validation split, experiment noise, IPOPT and clusterwise starts) comes from it; NumPy's global state is not used | — |
| `classifier.kind` | `"logistic"` | `"logistic"` (polytope regions, thesis 4.3), `"decision_list"` (if / else if / else; T-junctions), `"pairwise"` (one-versus-one) | `classifier` |
| `classifier.C` | `1e6` | inverse regularization strength of the classifier (binary model for two modes, multinomial otherwise) | `classifier` |
| `noise.adapt` | `true` | tolerances, estimator windows and (incremental) ridge fits follow the estimated noise; `false`: fixed tolerances, 5-sample windows, least squares | `adapt_to_noise`, `adapt_windows`, `noisy_ridge` |
| `noise.integral_refit` | `false` | under noise, refit each mode by integral least squares on its runs of consecutive samples (no derivatives; option) | — |
| `boundaries.points` | `"model"` | `"model"` (switches located with the learned modes) or `"query"` (thesis) | `boundary_points` |
| `boundaries.fine_step_factor` | `20` | fine step = time step of the refined trajectories / factor | `gen_fine_time_step_factor` |
| `optimization.solver` | `"ipopt"` | `"ipopt"` (thesis program) or `"clusterwise"` | `opti_solver` |
| `optimization.C_space` | `[0]` | values of C (no effect with the refit; with `refit="none"` and `[0]`, a warning: no regularization) | `C_space` |
| `optimization.deltas` | `"auto"` | insensitivity error (number or `"auto"`) | `deltas` |
| `optimization.n_starts` | `30` | IPOPT starts for the smallest C | `opti_n_starts` |
| `optimization.init_seed` | `None` | seed of the starts (start k uses `init_seed + k`); `None`: from `seed` | `opti_init_seed` |
| `optimization.init` | `"random"` | `"random"` or `"segments"` | `opti_init` |
| `optimization.stop_tol` | `"auto"` | early stop of the starts (`"auto"` or a number) | `opti_stop_tol` |
| `optimization.refit` | `"ridge"` | refit after the program: `"ridge"`, `"ls"`, `"none"` (coefficients computed by IPOPT, thesis) | `apply_LR`, `opti_refit` |
| `optimization.clusterwise_restarts` | `200` | starts of the clusterwise solver | `opti_n_restarts` |
| `simulation.max_step_factor` | `10` | largest integration step = step / factor | `max_step_factor` |
| `equivalence.enabled` | when bounds are given | equivalence queries (active learning) | `do_eq_query` |
| `equivalence.bounds`, `.n_trajs`, `.tol` | `None`, `50`, `0.01` | box of initial conditions, trajectories per query, RD tolerance (thesis: 0.1) | `eq_query_bounds`, `eq_query_ntrajs`, `eq_query_tol` |
| `equivalence.time_step`, `.time_range` | those of the test set | query trajectories | `time_step_test`, `time_range_test` |
| `equivalence.max_counterexamples` | `5` | a query stops after this many | `N_cexs` |
| `equivalence.max_rounds` | `100` | largest number of queries | `max_eq_rounds` |
| `equivalence.update_dynamics` | incremental: true; optimization: false | counterexamples update the dynamics | `improve_cont` |

`SamplingDesign` (active learning only): `time_step`, `time_range`, `x0s` (training), `test_time_step`,
`test_time_range`, `test_x0s` (`"random"`: 10 points in the equivalence bounds), `noise_std` (measurement noise
added to what the learner observes; errors are measured against the noise-free system).

TOML files read by `hydyreg fit` may also have a `[data]` table (`files`, `time_column`, `state_columns`,
`delimiter`, `skip_header`, `dt`, `trajectory_column`); `hydyreg learn` reads a `[design]` table. Unknown keys are
errors.

## The thesis experiments

The 26 rows of Tables 7.2–7.5 (both methods, 8 benchmark systems and the parametric example) are in
`hydyreg.benchmarks` (`systems`, `rows`) and run with:

```
hydyreg bench list                                       # the 26 rows
hydyreg bench run all -j 4 --timeout 30 --out runs/a     # one process per row; resumable
hydyreg bench run 'incr/*' ham --out runs/a --set design.noise_std=1e-3
hydyreg bench table --out runs/a                         # thesis-format tables
hydyreg bench compare --out runs/a --ref runs/b          # or --ref thesis; --printed for printed precision
```

Each row writes `<out>/results/<row>.json` (the 11 columns of the thesis tables, whether the run terminated, run
information, versions). `--set KEY=VALUE` overrides a setting of every selected row (stored in the result);
`--plots` saves figures. On the reference machine (an Apple Silicon laptop), all 26 rows take about 6 minutes with `-j 4`.

**Reproducing the thesis numbers:** `git checkout thesis-reproducible` (original implementation, casadi 3.5.5,
x86_64 Python 3.9, under Rosetta on Apple Silicon); its README explains the setup.

## Development

```
pytest tests -n 4                    # unit tests and golden cases (about 2 minutes)
pytest tests/unit                    # unit tests only (under a minute)
python -m pdoc hydyreg -o docs       # API documentation (pdoc in the dev extra; docs/ is not versioned)
```

The golden cases (`tests/golden`) are short learning runs used as regression smoke tests. They give the same results on
every platform within tolerances: NEQ, the number of modes and termination exactly, final arrays to 1e-6 of their
largest entry, errors to 1e-6 relative (plus 1e-9 absolute, below which they are rounding noise). After a deliberate
change of results, re-record them (`python tests/golden/capture.py all --out tests/golden/ref`) and log the change in
`CHANGELOG.md`. `tests/golden/ref/row_settings.json` records the settings of the 26 rows.

Layout of `src/hydyreg`: `trajectories`, `segmentation`, `differentiation`, `systems` (simulation with Filippov
sliding), `classifier`, `regression`, `model/` (the learner `HybridRegressor`), `learning/` (configuration,
pipeline, equivalence queries, evaluation, results, reports), `export/`, `persistence`, `plotting`, `cli`,
`benchmarks/`.

## Publications

The methods of HyDyReg come from my PhD thesis and the papers below. This version improves on them (see
[`CHANGELOG.md`](CHANGELOG.md)); these improvements are not published yet. The code as described in the publications is the tag `thesis-reproducible`.

- H. Dayekh. *Passive and Active Learning of Switched Nonlinear Dynamical Systems.* PhD thesis, Université
  Grenoble Alpes, 2025. [hal.science/tel-05073907](https://hal.science/tel-05073907/). Both methods, the
  active learning loop and the experiments in full.
- H. Dayekh, N. Basset, T. Dang. *Learning of Switched Nonlinear Dynamical Systems: Passive and Active
  Approaches.* In *Design and Verification of Cyber-Physical Systems* (Essays Dedicated to Martin Fränzle), LNCS
  16060, pp. 198–222, Springer, 2026. [doi:10.1007/978-3-032-16855-9_9](https://doi.org/10.1007/978-3-032-16855-9_9).
  An overview of the passive and active approaches.
- H. Dayekh, N. Basset, T. Dang. *Hybrid System Identification through Optimization and Active Learning.* ADHS
  2024, IFAC-PapersOnLine 58(11), pp. 87–92, 2024.
  [doi:10.1016/j.ifacol.2024.07.430](https://doi.org/10.1016/j.ifacol.2024.07.430). The optimization-based method
  and the active refinement of the mode regions.
- H. Dayekh, N. Basset, T. Dang. *Active Learning of Switched Nonlinear Dynamical Systems.* CDC 2024, pp.
  7834–7839, IEEE, 2024. [doi:10.1109/CDC56724.2024.10886099](https://doi.org/10.1109/CDC56724.2024.10886099),
  [hal-04701115](https://hal.science/hal-04701115). Active learning from counterexample trajectories given by
  approximate equivalence queries using the incremental method.

If you use HyDyReg, please cite a paper rather than the thesis: the LNCS chapter for HyDyReg as a whole, or the
ADHS or CDC paper for the part you use.

## Use of Generative AI

The development of HyDyReg since the thesis (the refactoring into a package, the improvements listed in
[`CHANGELOG.md`](CHANGELOG.md), the tests and the documentation) has been assisted by Claude Code, Anthropic's AI coding assistant, under my direction.
The methods build on those of the publications above.

## Licence

HyDyReg is distributed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See [`LICENSE`](LICENSE).
