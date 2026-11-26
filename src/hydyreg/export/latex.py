"""LaTeX (and text) form of the learned equations and boundaries."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def get_latex_eq(polyf : PolynomialFeatures, coefs : np.ndarray, bias, round_dig=3):
    coefs = np.around(coefs, round_dig)
    bias = round(bias, round_dig)
    powers = polyf.powers_
    nfeatures_in = powers.shape[1]
    nfeatures_out = powers.shape[0]
    assert nfeatures_out == len(coefs)

    def x_name(d):
        x_name = f'\\xdim{{{d+1}}}'
        return x_name

    x_names = []
    for d in range(nfeatures_in):
        x_names.append(x_name(d))

    eq_string = ""

    started = False
    for i in range(nfeatures_out):
        if coefs[i] == 0:
            continue
        x_part = ''
        for d in range(nfeatures_in):
            pow = powers[i][d]
            if pow > 0:
                x_part += x_names[d]
            if pow > 1:
                x_part += f'^{{{pow}}}'
        if i > 0 and coefs[i] >= 0 and started:
            eq_string += ' + '
        eq_string += f"{coefs[i]:.4f}" + ' ' + x_part
        started = True
    if not started:
        eq_string += f"{bias:.4f}"
    elif bias != 0:
        if bias > 0:
            eq_string += ' + '
        eq_string += f"{bias:.4f}" 

    return eq_string


def print_classifier_boundaries(classifier, polyfeatures=None):
    try:
        W = classifier.coef_
        B = classifier.intercept_
        ncoef = W.shape[0]
        D = W.shape[1]

        if not polyfeatures is None:
            fn = [f'x{i+1}' for i in range(polyfeatures.n_features_in_)]
            fn = polyfeatures.get_feature_names_out(fn)
        else:
            fn = [f'x{i+1}' for i in range(D)]
        
        if ncoef == 1:
            w = W[0]
            b = B[0]
            max_w = w[np.argmax(np.abs(w))]
            w = w / max_w
            b = b / max_w
            s = "Classifier boundary: "
            for d in range(D):
                s += f'{w[d]:.3f} {fn[d]} + '
            s += f'{b:.3f}'
            print(s)
        else:
            for i in range(ncoef):
                for j in range(i+1, ncoef):
                    s = f"Classifier boundary between {i:2d} and {j:2d}: "
                    w = W[i] - W[j]
                    b = B[i] - B[j]
                    max_w = np.max(np.abs(w))
                    w = w / max_w
                    b = b / max_w
                    for d in range(D):
                        s += f'{w[d]:.3f} {fn[d]} + '
                    s += f'{b:.3f}'
                    print(s)
    except:
        print("Could not print boundary.")

def get_latex_classifier_boundaries(classifier, polyfeatures=None):
    if hasattr(classifier, 'describe'):  # PairwiseLogisticRegression, DecisionListLogisticRegression
        n_in = polyfeatures.n_features_in_ if polyfeatures is not None else classifier.n_features_in_
        fn = [f' \\xdim{{{i+1}}}' for i in range(n_in)]
        if polyfeatures is not None:
            fn = polyfeatures.get_feature_names_out(fn)
        return ''.join(line.replace(':', ': $', 1) + '$\\\\\n' if ':' in line else line + '\\\\\n' for line in classifier.describe(fn))
    try:
        W = classifier.coef_
        B = classifier.intercept_
        classes = classifier.classes_
        ncoef = W.shape[0]
        D = W.shape[1]

        if not polyfeatures is None:
            fn = [f' \\xdim{{{i+1}}}' for i in range(polyfeatures.n_features_in_)]
            fn = polyfeatures.get_feature_names_out(fn)
        else:
            fn = [f' \\xdim{{{i+1}}}' for i in range(D)]
        
        if ncoef == 1:
            w = W[0]
            b = B[0]
            max_w = np.max(np.abs(w))
            w = w / max_w
            b = b / max_w
            s = "Classifier boundary: $"
            for d in range(D):
                s += f'{w[d]:.3f} {fn[d]} + '
            s += f'{b:.3f}$'
            return s
        else:
            S = ''
            for i in range(ncoef):
                for j in range(i+1, ncoef):
                    s = f"Classifier boundary between modes ${classes[i]+1}$ and ${classes[j]+1}$: $"
                    w = W[i] - W[j]
                    b = B[i] - B[j]
                    max_w = w[np.argmax(np.abs(w))]
                    w = w / max_w
                    b = b / max_w
                    for d in range(D):
                        s += f'{w[d]:.3f} {fn[d]} + '
                    s += f'{b:.3f}$\\\\\n'
                    S+=s
            return S
    except:
        print("Could not get boundary.")
        return ''
