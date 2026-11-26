"""Plain-text form of a learned model: equations of each mode and the mode regions."""
import numpy as np


def _poly_text(names_feat, coefs, bias, digits):
    terms = [(c, n) for c, n in zip(coefs, names_feat) if abs(c) > 0]
    s = f"{bias:.{digits}g}"
    for c, n in terms:
        s += f" {'-' if c < 0 else '+'} {abs(c):.{digits}g} {n}"
    return s


def feature_names(model, names=None):
    D = model.Xfit_.shape[1]
    names = names or [f"x{i + 1}" for i in range(D)]
    if model.deg > 1:
        return list(model.poly.get_feature_names_out(names))
    return list(names)


def model_text(model, names=None, digits=4):
    """Equations and regions of a fitted HybridRegressor as text."""
    D = model.Xfit_.shape[1]
    names = names or [f"x{i + 1}" for i in range(D)]
    feats = feature_names(model, names)
    lines = []
    for j in range(model.n):
        lines.append(f"Mode {j + 1}:")
        for d in range(D):
            lines.append(f"  d{names[d]}/dt = " + _poly_text(feats, model.Omegas_[j][d], model.betas_[j][d], digits))
    lines.append(regions_text(model, names, digits))
    return "\n".join(lines)


def regions_text(model, names=None, digits=4):
    """Mode regions: the rules of a decision list, the pairwise boundaries otherwise."""
    c = getattr(model, "classifier_", None)
    if c is None or model.n < 2 or not hasattr(c, "classes_"):
        return "Regions: a single mode everywhere."
    D = model.Xfit_.shape[1]
    names = names or [f"x{i + 1}" for i in range(D)]
    fn = list(model.classifier_poly_.get_feature_names_out(names)) if model.classifier_poly_deg > 1 else list(names)
    fmt = "{:.%dg}" % digits
    if hasattr(c, "describe"):
        return "Regions:\n" + "\n".join("  " + l for l in c.describe(fn, fmt=fmt))
    W, B, classes = np.atleast_2d(c.coef_), np.atleast_1d(c.intercept_), c.classes_
    lines = ["Regions (mode with the largest score; boundaries between modes):"]
    if W.shape[0] == 1:
        pairs = [(0, 1, W[0], B[0])]
    else:
        pairs = [(a, b, W[a] - W[b], B[a] - B[b]) for a in range(len(classes)) for b in range(a + 1, len(classes))]
    for a, b, w, b0 in pairs:
        scale = np.max(np.abs(w)) or 1.0
        s = " ".join(f"{'+' if v >= 0 else '-'} {fmt.format(abs(v / scale))} {n}" for v, n in zip(w, fn)).lstrip("+ ")
        s += f" {'+' if b0 >= 0 else '-'} {fmt.format(abs(b0 / scale))}"
        lines.append(f"  modes {classes[a] + 1} | {classes[b] + 1}: {s} = 0")
    return "\n".join(lines)


def _latex_name(n):
    """'x1' -> 'x_{1}', 'T_box' -> 'T_{\\mathrm{box}}', other names kept (as \\mathrm if longer than one letter)."""
    import re
    m = re.fullmatch(r"([A-Za-z])_?(\d+)", n)
    if m:
        return f"{m.group(1)}_{{{m.group(2)}}}"
    m = re.fullmatch(r"([A-Za-z])_([A-Za-z0-9]+)", n)
    if m:
        return f"{m.group(1)}_{{\\mathrm{{{m.group(2)}}}}}"
    return n if len(n) == 1 else f"\\mathrm{{{n}}}"


def _latex_monomial(feature, names, latex_names):
    """A PolynomialFeatures name such as 'x1^2 x2' (with the given variable names) in LaTeX."""
    out = []
    for factor in feature.split(" "):
        base, _, power = factor.partition("^")
        s = latex_names[names.index(base)] if base in names else base
        out.append(s + (f"^{{{power}}}" if power else ""))
    return " ".join(out)


def _latex_num(v, digits):
    s = ("{:.%dg}" % digits).format(v)
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times 10^{{{int(e)}}}"
    return s


def _latex_linear(w, b, feats, names, ln, digits):
    terms = []
    for c, f in zip(w, feats):
        if abs(c) > 0:
            terms.append(("- " if c < 0 else "+ ") + _latex_num(abs(c), digits) + "\\," + _latex_monomial(f, names, ln))
    if abs(b) > 0:
        terms.append(("- " if b < 0 else "+ ") + _latex_num(abs(b), digits))
    s = " ".join(terms)
    return (s[2:] if s.startswith("+ ") else "-" + s[2:] if s.startswith("- ") else s) or "0"


def regions_latex(model, names=None, digits=4):
    """Mode regions in LaTeX: a cases environment for a decision list, the boundary equations otherwise."""
    c = getattr(model, "classifier_", None)
    D = model.Xfit_.shape[1]
    names = list(names or [f"x{i + 1}" for i in range(D)])
    ln = [_latex_name(n) for n in names]
    if c is None or model.n < 2 or not hasattr(c, "classes_"):
        return "% a single mode everywhere"
    fn = list(model.classifier_poly_.get_feature_names_out(names)) if model.classifier_poly_deg > 1 else names
    if hasattr(c, "order_"):
        rows = []
        for k, (w, b) in enumerate(zip(c.W_, c.B_)):
            rows.append(f"  {c.classes_[c.order_[k]] + 1} & \\text{{if }} {_latex_linear(w, b, fn, names, ln, digits)} > 0"
                        + (" \\text{ (and no previous condition holds)}" if k else ""))
        rows.append(f"  {c.classes_[c.order_[-1]] + 1} & \\text{{otherwise}}")
        return "\\lambda(x) = \\begin{cases}\n" + " \\\\\n".join(rows) + "\n\\end{cases}"
    if hasattr(c, "W_"):
        K = len(c.classes_)
        pairs = [(a, bb, c.W_[a, bb], c.B_[a, bb]) for a in range(K) for bb in range(a + 1, K)]
    else:
        W, B = np.atleast_2d(c.coef_), np.atleast_1d(c.intercept_)
        K = len(c.classes_)
        pairs = [(0, 1, W[0], B[0])] if W.shape[0] == 1 else \
            [(a, bb, W[a] - W[bb], B[a] - B[bb]) for a in range(K) for bb in range(a + 1, K)]
    rows = []
    for a, bb, w, b0 in pairs:
        scale = np.max(np.abs(w)) or 1.0
        rows.append(f"  % boundary between modes {c.classes_[a] + 1} and {c.classes_[bb] + 1}\n  "
                    f"{_latex_linear(w / scale, b0 / scale, fn, names, ln, digits)} &= 0")
    return "% mode = arg max of the classifier scores\n\\begin{aligned}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}"


def model_latex(model, names=None, digits=4):
    """Equations of each mode and the regions in standard LaTeX (amsmath), ready to paste into a document."""
    D = model.Xfit_.shape[1]
    names = list(names or [f"x{i + 1}" for i in range(D)])
    ln = [_latex_name(n) for n in names]
    feats = feature_names(model, names)
    out = []
    for j in range(model.n):
        rows = [f"  \\dot{{{ln[d]}}} &= {_latex_linear(model.Omegas_[j][d], model.betas_[j][d], feats, names, ln, digits)}"
                for d in range(D)]
        out.append(f"% mode {j + 1}\n\\begin{{aligned}}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}")
    out.append("% regions\n" + regions_latex(model, names, digits))
    return "\n\n".join(out) + "\n"
