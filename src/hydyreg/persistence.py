"""Saving and loading learned models as JSON.

The file holds the polynomial degree and coefficients of each mode, the classifier (kind, classes, weights) and
its feature degree, the variable names and, optionally, the configuration. It is readable, diffable and does not
depend on library versions. A loaded model can predict, be simulated (`to_SNDS`) and be exported; it cannot
continue incremental learning (the training points are not saved).

    save_model(result.model, "model.json")
    model = load_model("model.json")
    trajs = model.to_SNDS().simulate_trajectory(0.01, 5, [[1.0, 2.0]])
"""
import json
import os
from typing import Any, Optional, Sequence, Union

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

FORMAT_VERSION = 1


def _version():
    try:
        from importlib import metadata
        return metadata.version("hydyreg")
    except Exception:
        return None


def _classifier_dict(c, n_modes):
    from .classifier import StandardizedLogisticRegression, DecisionListLogisticRegression, PairwiseLogisticRegression
    if n_modes < 2 or c is None or not hasattr(c, "classes_"):
        return None
    d = dict(classes=np.asarray(c.classes_).tolist(), n_features=int(c.n_features_in_))
    if isinstance(c, DecisionListLogisticRegression):
        d.update(kind="decision_list", C=c.C, order=c.order_.tolist(), W=c.W_.tolist(), B=c.B_.tolist())
    elif isinstance(c, PairwiseLogisticRegression):
        d.update(kind="pairwise", C=c.C, W=c.W_.tolist(), B=c.B_.tolist())
    elif isinstance(c, StandardizedLogisticRegression) or hasattr(c, "coef_"):
        d.update(kind="logistic", C=getattr(c, "C", None), multi_class=getattr(c, "multi_class", "auto"),
                 coef=np.asarray(c.coef_).tolist(), intercept=np.asarray(c.intercept_).tolist())
    else:
        raise ValueError(f"cannot save a classifier of type {type(c).__name__}; supported: logistic regression, "
                         "decision list, one-versus-one")
    return d


def _classifier_from_dict(d):
    from .classifier import StandardizedLogisticRegression, DecisionListLogisticRegression, PairwiseLogisticRegression
    kind = d["kind"]
    if kind == "logistic":
        c = StandardizedLogisticRegression(C=d.get("C") or 1e6, multi_class=d.get("multi_class", "auto"))
        c.coef_, c.intercept_ = np.array(d["coef"], dtype=float), np.array(d["intercept"], dtype=float)
    elif kind == "decision_list":
        c = DecisionListLogisticRegression(C=d.get("C", 1e6))
        c.order_, c.W_, c.B_ = np.array(d["order"], dtype=int), np.array(d["W"], dtype=float), np.array(d["B"], dtype=float)
    elif kind == "pairwise":
        c = PairwiseLogisticRegression(C=d.get("C", 1e6))
        c.W_, c.B_ = np.array(d["W"], dtype=float), np.array(d["B"], dtype=float)
        K = len(d["classes"])
        c._diag = np.where(np.eye(K, dtype=bool), np.inf, 0.0)
    else:
        raise ValueError(f"unknown classifier kind {kind!r} in model file")
    c.classes_ = np.array(d["classes"])
    c.n_features_in_ = int(d["n_features"])
    return c


def save_model(model: Any, path: Union[str, os.PathLike], names: Optional[Sequence[str]] = None,
               config: Any = None) -> None:
    """Writes a fitted HybridRegressor to a JSON file (floats at full precision)."""
    model.check_if_fitted()
    D = int(model.Xfit_.shape[1])
    poly = getattr(model, "poly", None)
    powers = poly.powers_.tolist() if (model.deg > 1 and poly is not None) else np.eye(D, dtype=int).tolist()
    d = dict(
        format_version=FORMAT_VERSION, hydyreg_version=_version(), dim=D, n_modes=int(model.n),
        names=list(names) if names else [f"x{i + 1}" for i in range(D)], poly_degree=int(model.deg),
        feature_powers=powers,
        coefficients=[dict(coef=np.asarray(model.Omegas_[j]).tolist(), intercept=np.asarray(model.betas_[j]).tolist())
                      for j in range(model.n)],
        boundary_degree=int(model.classifier_poly_deg),
        classifier=_classifier_dict(getattr(model, "classifier_", None), model.n),
        single_mode=int(model.modes_classify_[0]) if model.n < 2 and hasattr(model, "modes_classify_") else None,
        config=config.to_dict() if hasattr(config, "to_dict") else config,
    )
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


def load_model(path: Union[str, os.PathLike]) -> Any:
    """Reads a model written by `save_model`. Returns a HybridRegressor that can predict, be simulated and be
    exported."""
    from .model import HybridRegressor
    with open(path) as f:
        d = json.load(f)
    if d.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported model file format {d.get('format_version')!r} (expected {FORMAT_VERSION})")
    D, n, deg, bdeg = d["dim"], d["n_modes"], d["poly_degree"], d["boundary_degree"]
    m = HybridRegressor(n=n, deg=deg, classifier_poly_deg=bdeg)
    zeros = np.zeros((1, D))
    m.poly = PolynomialFeatures(degree=deg, include_bias=False).fit(zeros)
    if deg > 1:
        if m.poly.powers_.tolist() != d["feature_powers"]:
            raise ValueError("the polynomial features of the file do not match those of this version")
    Q = m.poly.n_output_features_ if deg > 1 else D
    m.Omegas_ = np.array([c["coef"] for c in d["coefficients"]], dtype=float).reshape(n, D, Q)
    m.betas_ = np.array([c["intercept"] for c in d["coefficients"]], dtype=float).reshape(n, D)
    m.Xfit_, m.Xfit_trans_, m.X_dot_fit_ = np.zeros((0, D)), np.zeros((0, Q)), np.zeros((0, D))
    m.modesfit_ = np.zeros(0, dtype=int)
    m.names_ = d.get("names")
    if n >= 2:
        classifier_poly = PolynomialFeatures(degree=bdeg, include_bias=False).fit(zeros) if bdeg > 1 else None
        if classifier_poly is not None:
            m.classifier_poly_ = classifier_poly
        m.classifier_ = _classifier_from_dict(d["classifier"])
        m.modes_classify_ = np.array(d["classifier"]["classes"])
        m._install_region_functions(classifier_poly)
    else:
        mode = d.get("single_mode") or 0
        m.classifier_ = None
        m.labelfun_ = lambda x: mode
        m.eventlist_ = None
        m.mode_scores_ = None
        m.modes_classify_ = np.array([mode])
    return m
