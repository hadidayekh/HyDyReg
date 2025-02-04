# INSTALLATION #

This repository requires Python 3.9+ with the following dependencies installed as follows:



#               INCREMENTAL LEARNING APPROACH               #
```python
from diff_methods import diff_method_fps
from opti import HybridRegressor
from snds import Trajectories
from sklearn.linear_model import LogisticRegression

trajs = Trajectories(t_list, x_list)


segments = trajs_train_sul.segment_trajectories(seg_tol)
X, X_dot, seg_res, _ = diff_method_fps(
    trajs_train_sul.t_list, trajs_train_sul.x_list, segments=segments, remove_seg=True)
    
model = HybridRegressor(train_coefs=True, deg=poly_degree, classifier_poly_deg=classifier_poly_deg, classifier_poly_trans=classifier_poly_trans)


model.fit_first_hypothesis(X, X_dot, seg_res)

model.classify_modes(classifier)

print(model.gen_full_latex_eq())
```