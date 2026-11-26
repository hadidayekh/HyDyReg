"""SpaceEx XML generation."""
import numpy as np


def gen_spaceex(model, name = 'HybridRegressorModel'):
    '''
    Generates XML SpaceEx format of the learned model.

    Parameters
    ----------
    model : HybridRegressor
        The HybridRegressor learned model.
    '''
    s = '<?xml version="1.0" encoding="UTF-8"?>\n'
    s += '<sspaceex xmlns="http://www-verimag.imag.fr/xml-namespaces/sspaceex" version="0.2" math="SpaceEx">\n'
    s += f'\t<component id="{name}">\n'
    assert model.deg == 1, "SpaceEx only supports affine dynamics."
    assert not model.classifier_poly_deg > 1 , "SpaceEx only supports affine dynamics."
    omegas = model.Omegas_
    betas = model.betas_
    dim_cont = model.poly.n_features_in_

    W = model.classifier_.coef_
    B = model.classifier_.intercept_
    ncoef = W.shape[0]
    D = W.shape[1]

    ind2 = 2*'\t'
    ind3 = 3*'\t'
    for i in range(dim_cont):
        s += f'{ind2}<param name="x{i+1}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'
    
    for i in range(ncoef):
        s += f'{ind2}<location id="{i+1}" name="mode_{i+1}">\n'
        
        s += f'{ind3}<invariant>'
        amp = False
        for j in range(ncoef):
            if j == i:
                continue
            if not amp:
                amp = True
            else:
                s += " &amp; "
            w = W[i] - W[j]
            b = B[i] - B[j]
            max_w = np.abs(w[np.argmax(np.abs(w))])
            w = w / max_w
            b = b / max_w
            for d in range(D):
                s += f'({w[d]})*(x{d+1}) + '
            s += f'({b}) &gt;= 0'
        s += '</invariant>\n'

        s += f'{ind3}<flow>'
        amp = False
        for j in range(dim_cont):
            if not amp:
                amp = True
            else:
                s += " &amp; "

            s += f'x{j+1}\' == '
            for k in range(dim_cont):
                s += f'{omegas[i][j][k]} * x{k+1} + '
            s += f'{betas[i][j]}'
        s += '</flow>\n'

        s += f'{ind2}</location>\n'

    for i in range(ncoef):
        for j in range(ncoef):
            if i == j:
                continue
            s += f'{ind2}<transition source="{i+1}" target="{j+1}">\n'
            
            
            s += f'{ind3}<guard>'
            w = W[i] - W[j]
            b = B[i] - B[j]
            max_w = np.abs(w[np.argmax(np.abs(w))])
            w = w / max_w
            b = b / max_w
            for d in range(D):
                s += f'({w[d]})*(x{d+1}) + '
            s += f'({b}) &lt; 0'
            s += f'</guard>\n'
            
            s += f'{ind2}</transition>\n'
    
    s += f'\t</component>\n'
    s += '</sspaceex>\n'

    return s
