"""MATLAB HyEQ class generation."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from .. import _logging as _log


def gen_matlab_hyeq_jumpMap(classifier, polyfeatures : PolynomialFeatures =None, ind=""):
    try:
        W = classifier.coef_
        B = classifier.intercept_
        ncoef = W.shape[0]
        D = W.shape[1]

        if not polyfeatures is None:
            dim_cont = polyfeatures.n_features_in_
            fn = [f'x({i+1})' for i in range(dim_cont)]
            fn = polyfeatures.get_feature_names_out(fn)
            fn = [s.replace(" ", " * ") for s in fn]
        else:
            dim_cont = D
            fn = [f'x({i+1})' for i in range(D)]
        
        s = f"{ind}function xplus = jumpMap(this, x)\n"
        s += f"{ind}\tz = x(1:{dim_cont});\n"
        s += f"{ind}\tmode = x({dim_cont + 1});\n"
        if ncoef == 1:
            w = W[0]
            b = B[0]
            max_w = np.max(np.abs(w))
            w = w / max_w
            b = b / max_w
            s += f"{ind}\tif "
            for d in range(D):
                s += f'({w[d]} * {fn[d]}) + '
            s += f'({b}) >= 0\n'
            s += f"{ind}\t\txplus = [z; 2];\n{ind}\telse\n"
            s += f"{ind}\t\txplus = [z; 1];\n{ind}\tend\n{ind}end"
            return s
        else:
            for i in range(ncoef):
                s += f"{ind}\tif " if i == 0 else f"{ind}\telseif "
                amp = False
                for j in range(ncoef):
                    if j == i:
                        continue
                    if not amp:
                        amp = True
                    else:
                        s += " && "
                    w = W[i] - W[j]
                    b = B[i] - B[j]
                    max_w = np.abs(w[np.argmax(np.abs(w))])
                    w = w / max_w
                    b = b / max_w
                    for d in range(D):
                        s += f'({w[d]})*({fn[d]}) + '
                    s += f'({b}) >= 0'
                s += '\n'
                s += f'{ind}\t\txplus = [z; {i+1}];\n'
            s += f"{ind}\tend\n{ind}end"
            return s
    except:
        _log.warning("Could not get boundary.")
        return ''

def gen_matlab_hyeq_jumpSetIndicator(classifier, polyfeatures : PolynomialFeatures =None, ind=""):
    W = classifier.coef_
    B = classifier.intercept_
    ncoef = W.shape[0]
    D = W.shape[1]

    if not polyfeatures is None:
        dim_cont = polyfeatures.n_features_in_
        fn = [f'x({i+1})' for i in range(dim_cont)]
        fn = polyfeatures.get_feature_names_out(fn)
        fn = [s.replace(" ", " * ") for s in fn]
    else:
        dim_cont = D
        fn = [f'x({i+1})' for i in range(D)]
    
    s = f"{ind}function D = jumpSetIndicator(this, x)\n"
    s += f"{ind}\tz = x(1:{dim_cont});\n"
    s += f"{ind}\tmode = x({dim_cont + 1});\n"
    s += f"{ind}\tswitch mode\n"
    
    if ncoef == 1:
        w = W[0]
        b = B[0]
        max_w = np.max(np.abs(w))
        w = w / max_w
        b = b / max_w
        s += f"{ind}\t\tcase 1\n"
        s += f"{ind}\t\t\tD = "
        bnd = ""
        for d in range(D):
            bnd += f'({w[d]} * {fn[d]}) + '
        bnd += f'({b})'
        s += bnd + ' >= 0;\n'

        s += f"{ind}\t\tcase 2\n"
        s += f"{ind}\t\t\tD = "
        s += bnd + ' < 0;\n'
        
        s += f"{ind}\tend\n{ind}end"
        return s
    else:
        for i in range(ncoef):
            s += f"{ind}\t\tcase {i+1}\n"
            s += f"{ind}\t\t\tD = "
            amp = False
            for j in range(ncoef):
                if j == i:
                    continue
                if not amp:
                    amp = True
                else:
                    s += " || "
                w = W[i] - W[j]
                b = B[i] - B[j]
                max_w = np.abs(w[np.argmax(np.abs(w))])
                w = w / max_w
                b = b / max_w
                for d in range(D):
                    s += f'({w[d]})*({fn[d]}) + '
                s += f'({b}) < 0'
            s += ';\n'
        s += f"{ind}\tend\n{ind}end"
        return s


def gen_matlab_hyeq_flowMap(model, ind=""):
    omegas = model.Omegas_
    betas = model.betas_
    
    polyfeatures = model.poly
    dim_cont = polyfeatures.n_features_in_
    fn = [f'x({i+1})' for i in range(dim_cont)]
    fn = polyfeatures.get_feature_names_out(fn)
    fn = [s.replace(" ", " * ") for s in fn]

    n_modes = model.n

    s = f"{ind}function xdot = flowMap(this, x)\n"
    s += f"{ind}\tz = x(1:{dim_cont});\n"
    s += f"{ind}\tmode = x({dim_cont + 1});\n"

    s += f"{ind}\tz_poly = ["
    for i in range(len(fn)-1):
        s += fn[i] + "; "
    s += fn[-1] + "];\n"

    s += f"{ind}\tswitch mode\n"
    for mode in range(n_modes):
        s += f"{ind}\t\tcase {mode+1}\n"
        s += f"{ind}\t\t\txdot = [this.A{mode+1} * z_poly + this.B{mode+1}; 0];\n"
    s += f"{ind}\tend\n"
    s += f"{ind}end"
    return s

def gen_full_matlab_hyeq_class(model, name="HybridRegressorModel"):
    '''
    Generates a MATLAB class of the learned model to be used with HyEq toolbox.

    Parameters
    ----------
    model : HybridRegressor
        The HybridRegressor learned model.
    '''
    s = "classdef " + name + " < HybridSystem\n"
    
    omegas = model.Omegas_
    betas = model.betas_

    n = omegas.shape[0]
    D = omegas.shape[1]
    Q = omegas.shape[2]

    s += "\tproperties\n"
    for mode in range(n):
        s += f"\t\tA{mode+1} = ["
        for d in range(D-1):
            for q in range(Q-1):
                s += f"{omegas[mode][d][q]}, "
            s += f"{omegas[mode][d][Q-1]}; "
        for q in range(Q-1):
            s += f"{omegas[mode][D-1][q]}, "
        s += f"{omegas[mode][D-1][Q-1]}]\n"
    
    s += "\n"

    for mode in range(n):
        s += f"\t\tB{mode+1} = ["
        for d in range(D-1):
            s += f"{betas[mode][d]}; "
        s += f"{betas[mode][D-1]}]\n"
    
    s += "\tend\n\n"

    s += "\tmethods\n"

    s += gen_matlab_hyeq_flowMap(model, ind = "\t\t") + "\n"
    s += gen_matlab_hyeq_jumpMap(model.classifier_, model.classifier_poly_ if model.classifier_poly_deg > 1 else None, ind = "\t\t") + "\n"
    s += gen_matlab_hyeq_jumpSetIndicator(model.classifier_, model.classifier_poly_ if model.classifier_poly_deg > 1 else None, ind = "\t\t") + "\n"

    s += "\t\tfunction C = flowSetIndicator(this, x)\n\t\t\tC = 1;\n\t\tend\n\n"

    s += "\tend\n"
    s += "end"

    return s.expandtabs(4)
