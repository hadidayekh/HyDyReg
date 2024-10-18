import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
import sobol
def which_segment(i, segments):
    r = 0
    while r < len(segments) and i >= segments[r]:
        r += 1
    return r

def which_segment_all(N, segments):
    segments_per_point = np.ndarray((N,), int)
    for i in range(N):
        segments_per_point[i] = which_segment(i, segments)
    return segments_per_point

def train_test_split_segments(X, Xdot, segments, segments_per_point, test_size=None):
    for r in range(len(segments)):
        if r < len(segments) - 1:
            X_curr      = X   [segments[r]:segments[r+1]]
            Xdot_curr   = Xdot[segments[r]:segments[r+1]]
        else:
            X_curr      = X   [segments[r]:]
            Xdot_curr   = Xdot[segments[r]:]

        if X_curr.shape[0] == 1:
            X_train_seg = X_curr
            X_val_seg = np.ndarray((0, X_curr.shape[1]))
            Xdot_train_seg = Xdot_curr
            Xdot_val_seg = np.ndarray((0, X_curr.shape[1]))
        else:
            X_train_seg, X_val_seg, Xdot_train_seg, Xdot_val_seg = train_test_split(X_curr, Xdot_curr, test_size=test_size)

        seg_curr    = segments_per_point[segments[r]]
        
        if r == 0:
            X_train     = X_train_seg
            Xdot_train  = Xdot_train_seg
            X_val       = X_val_seg
            Xdot_val    = Xdot_val_seg
            seg_train   = np.ones(X_train_seg.shape[0]).astype(int) * seg_curr
            seg_val     = np.ones(X_val_seg.shape[0]).astype(int)   * seg_curr
        
        else:
            X_train     = np.r_[X_train, X_train_seg]
            Xdot_train  = np.r_[Xdot_train, Xdot_train_seg]
            X_val       = np.r_[X_val, X_val_seg]
            Xdot_val    = np.r_[Xdot_val, Xdot_val_seg]
            seg_train   = np.r_[seg_train, np.ones(X_train_seg.shape[0]).astype(int) * seg_curr]
            seg_val     = np.r_[seg_val, np.ones(X_val_seg.shape[0]).astype(int) * seg_curr]

    return X_train, X_val, Xdot_train, Xdot_val, seg_train, seg_val

def get_modes_val(Chis_seg, seg_val):
    # Chis_seg represents Chis per segment, from which we deduce mode per segment
    # seg_val tells us the segment of each validation point

    # this gives us the mode of each segment
    modes_seg = np.argmin(Chis_seg, axis=0)

    modes_val = np.ndarray((len(seg_val)), int)
    for i in range(len(seg_val)):
        modes_val[i] = modes_seg[seg_val[i]]
    
    return modes_val

def compare_trajs(traj1, traj2, measure="MSE"):
    assert measure == "MSE" or measure == "RD"
    assert np.equal(np.array(traj1.t_list), np.array(traj2.t_list)).all(), "Cannot compare different timed trajectories."
    if measure == "MSE":
        return mean_squared_error(np.array(traj1.x_list).flatten(), np.array(traj2.x_list).flatten())
    elif measure == "RD":
        return np.mean(relative_diff(np.array(traj1.x_list).flatten(), np.array(traj2.x_list).flatten()))

def compare_snds(snds1, snds2, x0s, time_step, time_range, measure="MSE"):
    traj1 = snds1.simulate_trajectory(time_step, time_range, x0s)
    traj2 = snds2.simulate_trajectory(time_step, time_range, x0s)
    return compare_trajs(traj1, traj2, measure=measure)

def relative_diff(x, y):
    if x.ndim == 1:
        return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))
    else:
        return np.linalg.norm(x - y, axis=1) / (np.linalg.norm(x, axis=1) + np.linalg.norm(y,axis=1))

def relative_diff_by_component(x, y):
    res = np.zeros((x.size,))
    for i in range(x.size):
        res[i] = np.abs(x[i]-y[i]) / (np.abs(x[i]) + np.abs(y[i]))
    return res
    if x.ndim == 1:
        return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))
    else:
        return np.linalg.norm(x - y, axis=1) / (np.linalg.norm(x, axis=1) + np.linalg.norm(y,axis=1))

def generate_random_points(n_points, bounds, random_generator=None):
    if random_generator is None:
        random_generator = np.random
    x0s = random_generator.random((n_points, bounds.shape[0]))
    for d in range(bounds.shape[0]):
        x0s.T[d] = x0s.T[d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return x0s

def generate_sobol_points(n_points, bounds, skip=0):
    x0s = sobol.sample(dimension=bounds.shape[0], n_points=n_points, skip=skip)
    for d in range(bounds.shape[0]):
        x0s.T[d] = x0s.T[d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return x0s

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
        eq_string += f"{coefs[i]:.10f}" + ' ' + x_part
        started = True
    if not started:
        eq_string += f"{bias:.10f}"
    elif bias != 0:
        if bias > 0:
            eq_string += ' + '
        eq_string += f"{bias:.10f}" 

    return eq_string

def get_SVC_boundary(model_SVC : SVC):
    return
    # sv = model_SVC.support_vectors_
    # if model_SVC.kernel == 'poly' and model_SVC.degree == 2 and sv.shape[1] == 2:
    #     dc = model_SVC.dual_coef_[0]
    #     n = sv.shape[0]

    #     x0 = sv.T[0]
    #     x1 = sv.T[1]

    #     coef1_squared = 0
    #     coef2_squared = 0
    #     coef12 = 0
    #     coef1 = 0
    #     coef2 = 0

    #     g = model_SVC.gamma
    #     c = model_SVC.coef0

    #     for i in range(n):
    #         coef1_squared += g**2 * dc[i] * x0[i]**2
    #         coef2_squared += g**2 * dc[i] * x1[i]**2
    #         coef12 += 2*g**2 * dc[i] * x0[i] * x1[i]
    #         coef1 += 2*g*c * dc[i] * x0[i]
    #         coef2 += 2*g*c * dc[i] * x1[i]

    #     if c == 0:
    #         print(f"x0^2 + {coef2_squared/coef1_squared:.3f} x1^2 + {coef12/coef1_squared:.3f} x0x1 + {model_SVC.intercept_[0]/coef1_squared}")
    #     else:
    #         print(f"x0^2 + {coef2_squared/coef1_squared:.3f} x1^2 + {coef12/coef1_squared:.3f} x0x1 + {coef1/coef1_squared:.3f} x0 + {coef2/coef1_squared:.3f} x1 + {model_SVC.intercept_[0]/coef1_squared:.3f}")

    #     return (coef1_squared, coef2_squared, coef12, coef1, coef2, c**2 + model_SVC.intercept_)
    if True:# model_SVC.kernel == 'linear':
        # W = model_SVC.coef_
        # B = model_SVC.intercept_

        nclasses = len(model_SVC.estimators_)
        assert nclasses != 2
        if nclasses > 1:
            for i in range(nclasses):
                for j in range(i+1, nclasses):
                    for k in range(2):
                        w = model_SVC.estimators_[i].coef_[0] - model_SVC.estimators_[j].coef_[0]
                        b = model_SVC.estimators_[i].intercept_[0] - model_SVC.estimators_[j].intercept_[0]
                        s = f'boundary between modes {i} and {j}: '
                        b = b / w[k]
                        w = w / w[k]
                        # print(w)
                        for d in range(w.shape[0]):
                            s += f'{w[d]:.3f} x{d} + '
                        s += f'{b:.3f}'
                        print(s)
            for i in range(nclasses):
                for k in range(2):
                    w = model_SVC.estimators_[i].coef_[0]
                    b = model_SVC.estimators_[i].intercept_[0]
                    s = 'boundary ' + str(i) + ":"
                    b = b / w[k]
                    w = w / w[k]
                    for d in range(w.shape[0]):
                        s += f'{w[d]:.3f} x{d} + '
                    s += f'{b:.3f}'
                    print(s)
        else:
            w = model_SVC.estimators_[0].coef_[0]
            b = model_SVC.estimators_[0].intercept_[0]
            for k in range(2):
                b = b / w[k]
                w = w / w[k]
                s = f'boundary between modes: '
                for d in range(w.shape[0]):
                    s += f'{w[d]:.3f} x{d} + '
                s += f'{b:.3f}'
                print(s)
    else:
        raise NotImplementedError

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
    try:
        W = classifier.coef_
        B = classifier.intercept_
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
                    s = f"Classifier boundary between modes ${i+1}$ and ${j+1}$: $"
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
        print("Could not get boundary.")
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
        s += bnd + ' <= 0;\n'
        
        s += f"{ind}\tend\n{ind}end"
        return s
    else:
        for i in range(ncoef):
            # s += f"{ind}\tif " if i == 0 else f"{ind}\telseif "
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
                s += f'({b}) <= 0'
            s += ';\n'
        s += f"{ind}\tend\n{ind}end"
        return s


def gen_matlab_hyeq_flowMap(model, ind=""):
    assert model.train_coefs
    omegas = model.Omegas_
    betas = model.betas_
    print(omegas)
    print(betas.shape)
    
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
    s += gen_matlab_hyeq_jumpMap(model.classifier_, model.classifier_poly_ if model.classifier_poly_trans else None, ind = "\t\t") + "\n"
    s += gen_matlab_hyeq_jumpSetIndicator(model.classifier_, model.classifier_poly_ if model.classifier_poly_trans else None, ind = "\t\t") + "\n"

    s += "\t\tfunction C = flowSetIndicator(this, x)\n\t\t\tC = 1;\n\t\tend\n\n"

    s += "\tend\n"
    s += "end"

    return s.expandtabs(4)

def get_SVC_boundary_eq_dim2_poly2(model_SVC):
    if model_SVC.kernel != 'poly' or model_SVC.degree != 2:
        raise NotImplementedError

    n = model_SVC.support_vectors_.shape[0]
    dc = model_SVC.dual_coef_[0]
    sv = model_SVC.support_vectors_

    x0 = sv.T[0]
    x1 = sv.T[1]
    coef_1 = 0
    coef_2 = 0
    coef_12 = 0
    for i in range(n):
        coef_1 += dc[i] * x0[i]**2
        coef_2 += dc[i] * x1[i]**2
        coef_12 += 2*dc[i] * x0[i] * x1[i]
    
    print(f"x0^2 + {coef_2/coef_1:.3f} x1^2 + {coef_12/coef_1:.3f} x0x1 + {model_SVC.intercept_[0]/coef_1}")

    return (coef_1, coef_2, coef_12, model_SVC.intercept_[0])

def check_SVC(model_SVC, X, coefs):
    y = model_SVC.predict(X)
    N = X.shape[0]
    yc = np.zeros(N)
    ydc = model_SVC.decision_function(X)
    for i in range(N):
        yc[i] = coefs[0]*X[i][0]**2 + coefs[1]*X[i][1]**2 + coefs[2]*X[i][0]*X[i][1] + coefs[3]
    return y, yc, ydc

def print_experi_data(DATA):
    # Data is made of exp_name, RD1, RD2, RD3, t_best_C, t_active, t_EQ, count, RDdot, RDmerge, tmerge, RDpwa, tpwa
    for i in range(len(DATA)):
        d = DATA[i]
        exp_name, RD1, RD2, RD3, t1, t2, t3, count, RD_dot, RD_merge, t_merge, RD_piecelinear, t_piecelinear = d
        s = d[0] + ' & '
        for i in range(1, 4): #rd1,rd2,rd3
            s += f'{d[i]:.5f} & '
        for i in range(4, 7): #t1,t2,t3
            s += f'{d[i]:.1f} & '
        s += str(d[7]) + ' & ' #count
        s += f'{d[8]:.1e} & ' #rddot
        s += f'{d[9]:.1e} & ' #rdmerge
        s += f'{d[11]:.1e} & ' #rdpl
        s += f'{d[10]:.1f} & ' #tmerge
        s += f'{d[12]:.1f}\\\\'    #tpl
        print(s)