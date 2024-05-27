from typing import List
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

    def plot_hyp(self, ax):
        xmin, xmax = ax.get_xlim()
        ymin = (-self.w[0]*xmin - self.b) / self.w[1]
        ymax = (-self.w[0]*xmax - self.b) / self.w[1]

        ax.plot([xmin, xmax], [ymin, ymax], lw=1, c='k', ls='--')


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
        self.region_matrix = np.zeros((n, n*(n-1)//2))
        self.n = n # number of regions
        for i in range(n):
            for j in range(i+1, n):
                w_ij = classifier.coef_[i] - classifier.coef_[j]
                b_ij = classifier.intercept_[i] - classifier.intercept_[j]
                self.hyperplanes.append(Hyperplane(coef=w_ij, intercept=b_ij))
                
                h_index = i * (n - 1) - i*(i+1)//2 + j - 1
                self.region_matrix[i][h_index] = +1
                self.region_matrix[j][h_index] = -1
        self.p = len(self.hyperplanes) # number of hyperplanes

    def predict(self, X):
        N = X.shape[0]
        y_pred = np.ndarray((N,), dtype=int)
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

    def partial_fit(self, X, y):
        y_pred = self.predict(X)
        X_cex = X[y_pred != y]
        y_cex = y[y_pred != y]

        pass

    def plot_classi(self, X, y, h = 0.05):
        n = np.unique(y).size

        fig, ax = plt.subplots()

        for i in range(n):
            ax.scatter(*X[y == i].T, color=plt.cm.tab10(i), s=4)

        ax.scatter(*X[y == -1].T, color='red', s=4)


        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        for hyp in self.hyperplanes:
            hyp.plot_hyp(ax)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        
        xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        colors = [plt.cm.tab10(i) for i in range(n)]
        colors.append('red')
        cmap = ListedColormap(colors)
        
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

        return fig, ax


from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10, n_features=2, n_informative=2, n_clusters_per_class=1, n_redundant=0, n_classes=4, random_state=1)

model = IncrementalClassifier()

model.fit(X, y)

y_pred = model.predict(X)

model.plot_classi(X, y)
plt.show()
print(y)
print(y_pred)