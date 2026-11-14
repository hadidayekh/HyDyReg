# Overview #
HyDyReg is a research project that aims at the identification of switched nonlinear dynamical systems through two main methods : the *optimization-based* and *incremental* approaches, the latter being compatible with active learning by its nature.
Details about the methods can be found in my [PhD thesis](https://hal.science/tel-05073907/).

**Note**: This is academic code supporting my PhD thesis and publications. While experiments are fully reproducible, the code is not yet structured for deployment.



# Installation #

This repository requires Python 3.9 (the pinned `casadi==3.5.5` does not provide wheels for newer Python versions). Install the dependencies as follows:
```
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

### Reproducing the thesis numbers exactly
```
python src/run_experiments.py all --thesis --out runs/thesis --compare thesis
```
`--thesis` runs the original implementation with 10 OpenBLAS threads and checks the platform. All 26 rows match exactly only on the thesis platform:
- x86_64 Python 3.9 (on Apple Silicon: under Rosetta, see above);
- the pinned `requirements.txt`;
- 10 OpenBLAS threads.

With this setup, all 26 rows of Tables 7.2–7.5 are reproduced exactly (verified on an Apple Silicon Mac under Rosetta).

**Why the thread count matters.** The mode classifier is an unregularized `LogisticRegression` fitted on linearly separable data. Such a fit has no finite optimum, so the solver stops at a point that depends on the path it took. The last-bit rounding differences that OpenBLAS produces with different thread counts can therefore change the learned boundaries slightly. This affects only some rows: d3 for incremental parametric n ≥ 4, and also d2, NEQ and d_dot for n = 20. All other rows are the same for any thread count. Runs are fully deterministic for a given platform and thread count.

**Fewer than 10 cores.** OpenBLAS never uses more threads than there are physical cores. On macOS, `tools/fakecores` emulates more cores; it changes only the core count that OpenBLAS sees:
```
clang -arch x86_64 -dynamiclib -o fakecores_x86_64.dylib tools/fakecores/fakecores.c
python src/run_experiments.py all --thesis --fakecores fakecores_x86_64.dylib --out runs/thesis --compare thesis
```
On other platforms (other CPUs or BLAS kernels, Linux), expect the same values for most rows and small differences in the sensitive ones.

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