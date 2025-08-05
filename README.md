# Overview #
HyDyReg is a research project that aims at the identification of switched nonlinear dynamical systems through two main methods : the *optimization-based* and *incremental* approaches, the latter being compatible with active learning by its nature.
Details about the methods can be found in my [PhD thesis](https://hal.science/tel-05073907/).

**Note**: This is academic code supporting my PhD thesis and publications. While experiments are fully reproducible, the code is not yet structured for deployment.



# Installation #

This repository requires Python 3.9+ with the following dependencies installed as follows:
```
pip install -r requirements.txt
```

# Running Experiments #

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