# Overview #
HyDyReg is a research project that aims at the identification of switched nonlinear dynamical systems through two main methods : the *optimization-based* and *incremental* approaches, the latter being compatible with active learning by its nature.
Details about the methods can be found in my [PhD thesis](https://hal.science/tel-05073907/).

**Note**: This is academic code supporting my PhD thesis and publications. Experiments are deterministic on a given machine (see [Reproducing the thesis tables](#reproducing-the-thesis-tables)), but the code is not structured for deployment.



# Installation #

This repository requires Python 3.9 (the pinned `casadi==3.5.5` does not provide wheels for newer Python versions). Install the dependencies as follows:
```
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.9 is no longer maintained, so your system may not provide it; [uv](https://github.com/astral-sh/uv) can install it (`uv venv --python 3.9 .venv`, then `uv pip install --python .venv/bin/python -r requirements.txt`).

On Apple Silicon Macs, the pinned `casadi==3.5.5.post2` only exists for x86_64, so use an x86_64 Python 3.9 running under Rosetta 2. For example, with [uv](https://github.com/astral-sh/uv):
```
uv venv --python cpython-3.9.25-macos-x86_64-none .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

# Running Experiments #

## With the experiment runner (recommended)

`src/run_experiments.py` runs any subset of the 26 rows of the thesis tables (7.2–7.5, both methods). It can be called from any directory, and it writes everything to an output directory, leaving `src/` untouched:
```
python src/run_experiments.py --list                       # the 26 rows, their indices and rough runtimes
python src/run_experiments.py incr/isolette --out runs/a   # one row
python src/run_experiments.py 9-16 --out runs/a            # rows by index (here: Table 7.3)
python src/run_experiments.py 'incr/*' ham parametric-4    # globs, short names (both methods)
python src/run_experiments.py --out runs/a --table --compare thesis
```
- Each row runs in a fresh Python process and writes `<out>/results/<row>.json` (log in `<out>/logs/`). Rerunning the same command skips finished rows, so interrupted runs resume. Use `--force` to rerun them.
- `-j N` runs N rows in parallel. This changes the timing columns, not the results.
- `--table` prints the tables in thesis format. `--compare thesis` compares the d-values, NEQ and d_dot with the thesis at the printed precision; `--compare <other out dir>` compares two runs exactly.

### Reproducing the thesis tables
```
python src/run_experiments.py all --threads 10 --out runs/thesis --compare thesis
```
Runs are deterministic on a given machine. On the thesis setup, all 26 rows are reproduced exactly. On other platforms some values can differ, and which rows differ, and by how much, depends on the platform. Each run starts by printing how its platform compares with the thesis setup.

**Why the platform matters.** This is a weakness of the original implementation: the mode classifier is an unregularized `LogisticRegression` fitted on linearly separable data. Such a fit has no finite optimum, so the solver stops at a point that depends on the path it took, and last-bit rounding differences of the platform (BLAS kernels and thread count, math library, compiled solvers) can move the learned boundaries slightly. Later versions of HyDyReg (v2.0 and later) use a well-posed classifier, whose results no longer depend on the thread count.

**The thesis setup.** The thesis tables were produced on an Apple Silicon Mac with 10 cores:
- x86_64 Python 3.9 under Rosetta (see [Installation](#installation)), with the pinned `requirements.txt`;
- OpenBLAS with 10 threads (`--threads 10`), which then uses its Nehalem kernels.

On an Apple Silicon Mac set up this way, the command above reproduces all 26 rows of Tables 7.2–7.5 exactly. On this setup, the thread count alone changes only some rows: d3 for incremental parametric n ≥ 4, and also d2, NEQ and d_dot for n = 20.

**Fewer than 10 cores.** OpenBLAS never uses more threads than there are physical cores. On macOS, `tools/fakecores` emulates more cores; it changes only the core count that OpenBLAS sees:
```
clang -arch x86_64 -dynamiclib -o fakecores_x86_64.dylib tools/fakecores/fakecores.c
python src/run_experiments.py all --threads 10 --fakecores fakecores_x86_64.dylib --out runs/thesis --compare thesis
```

## With the original scripts

All runs need must happen in the `src` folder.

To run the experiments and get a table of results, it suffices to run:
```
python3 experiments/experiment_infos.py
python3 experiments/parametric.py
```

For the **incremental approach**, in the above files, you must set the following :
```python
optimization_based = False
improve_continuous = True
```
For the **optimization-based approach**, in the above files, you must set the following:
```python
optimization_based = True
improve_continuous = False
```
#               Example Flow Using the Incremental Learning Approach               #
```python
from diff_methods import diff_method_fps
from opti import HybridRegressor
from snds import Trajectories
from sklearn.linear_model import LogisticRegression

trajs = Trajectories(t_list, x_list)


segments = trajs_train_sul.segment_trajectories(seg_tol)
X, X_dot, seg_res, _ = diff_method_fps(
    trajs_train_sul.t_list, trajs_train_sul.x_list, segments=segments, remove_seg=True)
    
model = HybridRegressor(train_coefs=True, deg=poly_degree, classifier_poly_deg=classifier_poly_deg)


model.fit_first_incremental(X, X_dot, seg_res)

model.classify_modes(classifier)

print(model.gen_full_latex_eq())
```


# Generating Docs
To generate docs:
```
PYTHONPATH=src PDOC_IMPORTS=disable python3.9 -m pdoc src --output-dir docs
```