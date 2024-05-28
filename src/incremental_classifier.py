from typing import List
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import polytope as pc
from sklearn.base import clone
from labellines import labelLine, labelLines

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

    def plot_hyp(self, ax, label=None):
        xmin, xmax = ax.get_xlim()
        ymin = (-self.w[0]*xmin - self.b) / self.w[1]
        ymax = (-self.w[0]*xmax - self.b) / self.w[1]

        ax.plot([xmin, xmax], [ymin, ymax], lw=1, c='k', ls='--', label=label)


# class Polytope:
#     def __init__(self) -> None:
#         self.hyperplanes = []
#         pass

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
        self.regions : List[pc.Polytope] = []
        self.region_matrix = np.zeros((n, n*(n-1)//2))
        self.n = n # number of regions
        self.classifier = classifier
        for i in range(n):
            for j in range(i+1, n):
                w_ij = classifier.coef_[i] - classifier.coef_[j]
                b_ij = classifier.intercept_[i] - classifier.intercept_[j]
                self.hyperplanes.append(Hyperplane(coef=w_ij, intercept=b_ij))
                
                h_index = i * (n - 1) - i*(i+1)//2 + j - 1
                self.region_matrix[i][h_index] = +1
                self.region_matrix[j][h_index] = -1
        self.p = len(self.hyperplanes) # number of hyperplanes

        self.regions = [None]*n
        self.define_region_polytopes(range(n))

        self.X_ = X_trans
        self.y_ = y

    def predict(self, X):
        N = X.shape[0]
        y_pred = np.ndarray((N,), dtype=int)
        X_trans = self.poly.fit_transform(X)
        
        # get inequality signs for every point for every hyperplane
        # hyp_sign_matrix = np.ndarray((self.p, N))
        # for i in range(self.p):
        #     h =  self.hyperplanes[i]
        #     hyp_sign_matrix[i] = h.get_sign(X_trans)
        # hyp_sign_matrix = hyp_sign_matrix.T # now the shape is N by p

        # every point now needs to be compared with all regions
        for i in range(N):
            y_i = -1
            for j in range(self.n):
                # rj_xi = self.region_matrix[j] * hyp_sign_matrix[i]
                # if (rj_xi >= 0).all():
                #     y_i = j
                #     break
                if X_trans[i] in self.regions[j]:
                    y_i = j
                    break
            y_pred[i] = y_i

        return y_pred

    def define_region_polytopes(self, region_indices):
        for R in region_indices:
            A = [[1,0], [0,1], [-1,0], [0,-1]]
            b = [1e6, 1e6, 1e6, 1e6]
            for k in range(self.p):
                if self.region_matrix[R][k] != 0:
                    A.append(self.region_matrix[R][k] * self.hyperplanes[k].w)
                    b.append(self.region_matrix[R][k] * self.hyperplanes[k].b)
            A = -np.array(A)
            b = np.array(b)
            self.regions[R] = pc.reduce(pc.Polytope(A, b))

    def partial_fit(self, X, y):
        y_pred = self.predict(X)
        X_cex = X[y_pred != y]
        y_cex = y[y_pred != y]


        X_all = np.r_[self.X_, X]
        y_all = np.r_[self.y_, y]

        for i in range(X_cex.shape[0]):
            x_i = X_cex[i]
            y_i = y_cex[i]
            print(x_i, y_i)
            # x_i does not belong to polytope y_i
            # need to remove unsatisfied constraints in polytope y_i
            
            hyp_indices = self.region_matrix[y_i].nonzero()[0]
            unsat_hyp = []
            for h_ind in hyp_indices:
                if self.hyperplanes[h_ind].get_sign([x_i])[0] != self.region_matrix[y_i][h_ind]:
                    unsat_hyp.append(h_ind)


            # first track regions that would be affected by this remove:
            affected_regions = []
            for h_ind in unsat_hyp:
                affected_regions.extend(list(self.region_matrix.T[h_ind].nonzero()[0]))
            affected_regions = np.unique(np.array(affected_regions))


            # remove unsat hyp
            print(unsat_hyp, 'fhal;kdjfa;s')
            print(affected_regions)
            unsat_hyp.sort(reverse=True)
            for hyp in unsat_hyp:
                del self.hyperplanes[hyp]
                self.p -= 1
            self.region_matrix = np.delete(self.region_matrix, unsat_hyp, axis=-1)
            

            # redefine affected regions
            self.define_region_polytopes(affected_regions)


            # group all points in affected regions together:
            
            X_affected = []
            y_affected = []
            for k in range(len(y_all)):
                for R in affected_regions:
                    if X_all[k] in self.regions[R]:
                        X_affected.append(X_all[k])
                        y_affected.append(y_all[k])
                        break
            X_affected = np.array(X_affected)
            y_affected = np.array(y_affected)

            LR = clone(self.classifier)
            LR.fit(X_affected, y_affected)
            classes = LR.classes_
            if len(classes) == 2:
                print(self.region_matrix)
                h = Hyperplane(coef=LR.coef_[0], intercept=LR.intercept_[0])
                col = np.zeros((self.n, 1))
                col[classes[0]] = -1
                col[classes[1]] = +1
                self.hyperplanes.append(h)
                self.p += 1
                self.region_matrix = np.append(self.region_matrix, col, axis=-1)
                print(self.region_matrix)
                self.define_region_polytopes(classes)
            else:
                pass
                exit()
        
        self.X_ = X_all
        self.y_ = y_all


    def plot_classi(self, X=None, y=None, res = 0.05, name_hyp=False):
        if X is None or y is None:
            X = self.X_
            y = self.y_

        n = np.unique(y).size

        fig, ax = plt.subplots()

        for i in range(n):
            ax.scatter(*X[y == i].T, color=plt.cm.tab10(i), s=4, label='Class '+ str(i))

        if (X[y == -1].size > 0):
            ax.scatter(*X[y == -1].T, color='red', s=4, label='Undefined')


        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        for k in range(len(self.hyperplanes)):
            self.hyperplanes[k].plot_hyp(ax, label = str(k) if name_hyp else None)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        
        xx, yy = np.meshgrid(np.arange(xmin, xmax, res), np.arange(ymin, ymax, res))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        colors = [plt.cm.tab10(i) for i in range(n)]
        colors.append('red')
        cmap = ListedColormap(colors)
        
        # ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
        ax.legend()

        col_ind = 0
        for p in self.regions:
            p.plot(ax, color=plt.cm.tab10(col_ind), alpha = 0.3)
            col_ind += 1

        if name_hyp:
            labelLines(ax.get_lines(), zorder=2.5)

        return fig, ax


from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10, n_features=2, n_informative=2, n_clusters_per_class=1, n_redundant=0, n_classes=4, random_state=1)

model = IncrementalClassifier()

model.fit(X, y)

y_pred = model.predict(X)

Xn = np.array([[0.5,-2], [-4,0]])
yn = np.array([0, 0])

fig, ax = model.plot_classi()


model.partial_fit(Xn, yn)

# for p in model.regions:
#     a = p.plot()
#     a.set_xlim(ax.get_xlim())
#     a.set_ylim(ax.get_ylim())

model.plot_classi(name_hyp=False)

plt.show()
print(y)
print(y_pred)

