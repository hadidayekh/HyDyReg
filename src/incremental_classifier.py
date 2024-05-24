from typing import List
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LogisticRegression

'''
A polytope is represented by a set of hyperplanes h_i and is defined by a conjunction of these hyperplanes h_i(x) >= 0.
A hyperplane is represented by coefficients w_i and an intercept b_i. 

A hyperplane in the linear space : h_i(x) = w_i.x + b_i = 0
A hyperplane in the polynomial space : h_i(x) = w_i.psi(x) + b_i = 0

We need to keep track of polytopes that share some hyperplanes, notably when h_i(x) = -h_j(x) for i <> j.
'''

class Hyperplane:
    def __init__(self, coef, intercept=0) -> None:
        self.w = coef
        self.b = intercept
        self.Q = coef.shape[0]
        pass

    def get_sign(self, X_trans):
        return np.sign(X_trans @ self.w + self.b)

class Polytope:
    def __init__(self) -> None:
        self.hyperplanes = []
        pass

class IncrementalClassifier:
    def __init__(self, poly_deg=1) -> None:
        self.poly_deg = poly_deg
        self.poly = PolynomialFeatures(degree=poly_deg, include_bias=False)
        pass

    def fit(self, X, y, classifier=LogisticRegression(penalty='none', multi_class='multinomial')):
        X_trans = self.poly.fit_transform(X)
        classifier.fit(X_trans, y)
        n = np.unique(y).shape[0]
        self.hyperplanes : List[Hyperplane] = []
        self.region_matrix = np.zeros((n, n*(n-1)/2))
        self.n = n # number of regions
        for i in range(n):
            for j in range(i+1, n):
                w_ij = classifier.coef_[i] - classifier.coef_[j]
                b_ij = classifier.intercept_[i] - classifier.intercept_[j]
                self.hyperplanes.append(Hyperplane(coef=w_ij, intercept=b_ij))
                
                h_index = i * n + j
                self.region_matrix[i][h_index] = +1
                self.region_matrix[j][h_index] = -1
        self.p = len(self.hyperplanes) # number of hyperplanes

    def predict(self, X):
        N = X_trans.shape[0]
        y_pred = np.ndarray((N,))
        X_trans = self.poly.fit_transform(X)
        
        # get inequality signs for every point for every hyperplane
        hyp_sign_matrix = np.ndarray((self.p, N))
        for i in range(self.p):
            h =  self.hyperplanes[i]
            hyp_sign_matrix[i] = h.get_sign(X_trans)
        hyp_sign_matrix = hyp_sign_matrix.T # now the shape is N by p

        # every point now needs to be compared with all regions
        for i in range(N):
            y_i = -1
            for j in range(self.n):
                rj_xi = self.region_matrix[j] * hyp_sign_matrix[i]
                if (rj_xi >= 0).all():
                    y_i = j
                    break
            y_pred[i] = y_i

        return y_pred