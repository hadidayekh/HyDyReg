from cmath import inf
from copy import deepcopy
import enum
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import polytope as pc


global colormap
class IncrementalLogsiticRGD_withLR:
    def __init__(self, eta=0.0001, nb_epochs=50) -> None:
        self.eta = eta
        self.nb_epochs = nb_epochs

    # Calculates the gradient of L with respect to W and B on x,y
    def grad_L(self, x, y, k):
        W = self.W
        B = self.B
        Z = 0
        
        for j in range(self.K):
            Z += np.exp(np.dot(W[j], x) + B[j])
        
        P = np.exp(np.dot(W[k], x) + B[k])/Z

        one_cond = 1 if y == k else 0

        grad_b_k = P - one_cond
        grad_w_k = x * grad_b_k

        return grad_w_k, grad_b_k

    def define_region_polytopes(self):
        '''
        Defines polytopes in self.regions with the polytope library.
        '''
        self.regions = []
        for k in range(self.K):
            A = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            b = [1e20]*4

            A = [
                [1, 0],   # x >= 0
                [-1, 0],  # x <= 1
                [0, 1],   # y >= 0
                [0, -1],  # y <= 1
            ]
            b = [15, 15, 15, 15]

            for j in range(self.K):
                if j != k:
                    A.append(self.W[k] - self.W[j])
                    b.append(self.B[k] - self.B[j])
            A = -np.array(A)
            b = np.array(b)
            self.regions.append(pc.Polytope(A, b))

    def fit(self, X, y):
        '''
        Runs the fitting algorithm with 0-initialized coefficient vectors.
        '''
        self.K = len(np.unique(y))   # number of classes
        self.D = len(X[0])

        self.W = np.zeros((self.K, self.D))
        self.B = np.zeros(self.K)

        self.X_ = np.ndarray((0, X[0].size))
        self.y_ = np.ndarray((0,))

        return self.partial_fit(X, y)



    def partial_fit(self, X, y, init_method='eps', eps=1, with_prev_portion=0, Xplt=None, yplt=None, colors=lambda x: x):
        '''
        Runs the fitting algorithm using gradient descent and without re-initializing the coefficients.

        init_method:    'lr-sum' in case of a new class, runs logistic regression locally, redefines w for all affected regions by perturbing the barycenter of the weight coefficients of the new class with the result of the local logistic regression
                        'lr-centroid' in case of a new class, runs logistic regression locally, then another optimization problem to define w_new in such a way that borders between classes in the affected regions are close to the result of the local logistic regression
                        'eps' in case of a new class, initializes w_new by perturbing the barycenter of the weight vectors of the affected regions by epsilon.
                        'random' in case of a new class, randomly initializes the new weight vector
                        'zero' in case of a new class, initializes the new weight vector to zero
        '''
        assert len(X[0]) == self.D
        N = X.shape[0]          # number of data points
        new_K = int(np.max(y) + 1)   # this assumes that new classes are always added in ascending order

        # First deal with NEW classes, one at a time
        if new_K > self.K:
            n_new_classes = (new_K - self.K)
            
            for new_class in range(self.K+1, new_K):
                Xnew = X[y == new_class]
                ynew = y[y == new_class]

                # regions contains all regions in which new-class points appear
                regions = []
                for i in range(len(ynew)):
                    for j in range(self.K):
                        if X[i] in self.regions[j]:
                            regions.append(j)
                            break

                regions = np.unique(np.array(regions))
                n_affected_regions = len(regions)
                # print("affected regions", regions)
                
                if init_method == 'random':
                    new_W = np.random.random((1, self.D))
                    new_B = np.random.random((1,))
                else:
                    new_W = np.zeros((1, self.D))
                    new_B = np.zeros((1,))

                self.W = np.r_[self.W, new_W]
                self.B = np.r_[self.B, new_B]
                old_K = self.K
                self.K += 1

                if init_method == 'random' or init_method=='zero':
                    continue

                if init_method == 'lr-centroid' or init_method == 'lr-sum' or n_affected_regions == 1:
                    # run a logistic regression algorithm with all *new* points and
                    # all points in affected regions
                    LR = LogisticRegression(penalty='none')
                    X_lr = Xnew
                    y_lr = ynew
                    for r in regions:
                        X_lr = np.r_[X_lr, self.X_[self.y_ == r]]
                        y_lr = np.r_[y_lr, self.y_[self.y_ == r]]
                    LR.fit(X_lr, y_lr)
                    LR_classes = np.array(LR.classes_, dtype=int) ## must be equal to regions ∪ new_class
                
                # eps = 0.01

                if len(LR_classes) == 2: # equivalent to n_affected_regions == 1
                    self.W[LR_classes[1]] = (LR.coef_[0] + self.W[LR_classes[0]])
                    self.B[LR_classes[1]] = (LR.intercept_[0] + self.B[LR_classes[0]])
                else:
                    # more than one new class / multinomial LR
                    # get the W of the potentially merged regions
                    w_region = self.W[regions[0]] / n_affected_regions
                    b_region = self.B[regions[0]] / n_affected_regions

                    # in case of more than one region, take the mean of W,B                
                    for i in range(1, n_affected_regions):
                        w_region += self.W[regions[i]] / n_affected_regions
                        b_region += self.B[regions[i]] / n_affected_regions

                    '''
                    W_temp = deepcopy(self.W)
                    B_temp = deepcopy(self.B)
                    self.W[-1] = w_region
                    self.B[-1] = b_region
                    for i in range(len(LR_classes)):
                        c = LR_classes[i]
                        if c < self.W.shape[0] - 1:
                            self.W[c] = np.zeros(self.W[c].shape)
                            self.B[c] = np.zeros(self.B[c].shape)
                    if not Xplt is None:
                        self.define_region_polytopes()
                        self.plot_classi(Xplt, yplt, title="After adding new points and merging regions.", colors=colors)
                    self.W = W_temp
                    self.B = B_temp'''

                    if init_method == 'lr-sum':
                        for i, c in enumerate(LR_classes):
                            self.W[c] = w_region + eps * LR.coef_[i]
                            self.B[c] = b_region + eps * LR.intercept_[i]
                    elif init_method == 'eps':
                        self.W[-1] = w_region
                        self.B[-1] = b_region + eps
                    elif init_method == 'lr-centroid':
                        W_new = np.array((n_affected_regions, self.D))
                        B_new = np.array((n_affected_regions,))

                        new_class_LR_index = np.where(LR_classes == new_class)[0][0]
                        alpha_new = LR.coef_[new_class_LR_index]
                        beta_new = LR.coef_[new_class_LR_index]

                        for i, region in enumerate(regions):
                            # find index of class in LR_classes
                            region_LR_index = np.where(LR_classes == region)[0][0]
                            # define w_new and b_new according to the desired result
                            W_new[i] = alpha_new - LR.coef_[region_LR_index] + self.W[region]
                            B_new[i] = beta_new - LR.intercept_[region_LR_index] + self.B[region]
                        
                        # now define w for new class based on the centroid
                        self.W[-1] = np.mean(W_new, axis=0)
                        self.B[-1] = np.mean(B_new, axis=0)

                    # # define initial conditions of W,B for the new and old classes
                    # # based on LR results
                    # for i, c in enumerate(LR_classes):
                    #     if c >= old_K:
                    #         self.W[c] = w_region #+ eps * LR.coef_[i]
                    #         self.B[c] = b_region + 1 #* LR.intercept_[i]
                    #     # self.W[c] = w_region + eps * LR.coef_[i]
                    #     # self.B[c] = b_region + eps * LR.intercept_[i]

            
            if not Xplt is None:
                self.define_region_polytopes()
                self.plot_classi(Xplt, yplt, title="After adding new points and initializing weight vectors before applying SGD.", colors=colors)
            



        X_old = X
        y_old = y
        if with_prev_portion > 0:            
            rand_ints = np.random.permutation(self.X_.shape[0])[:int(self.X_.shape[0]*with_prev_portion)]
            # print(rand_ints)
            X_prev_rand = self.X_[rand_ints]
            y_prev_rand = self.y_[rand_ints]
            X = np.r_[X, X_prev_rand]
            y = np.r_[y, y_prev_rand]
            N = X.shape[0]

        # stochastic gradient descent algorithm
        for _ in range(self.nb_epochs):
            p = np.random.permutation(len(y))
            X_shuffled = X[p]
            y_shuffled = y[p]
            for i in range(N):
                for k in range(self.K):
                    grad_L_w, grad_L_b = self.grad_L(X_shuffled[i], y_shuffled[i], k)
                    self.W[k] = self.W[k] - self.eta * grad_L_w
                    self.B[k] = self.B[k] - self.eta * grad_L_b
        self.define_region_polytopes()
        
        self.X_ = np.r_[self.X_, X_old]
        self.y_ = np.r_[self.y_, y_old]
        
        # for r in self.regions:
        #     print(r.A)
        #     print(r.b)
        return self

    def decision_function(self, X):
        return X @ self.W.T + self.B
    
    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


    def plot_classi(self, X, y, h = 0.05, title="", colors = lambda x : x):
        n = np.unique(y).size
        
        fig, ax = plt.subplots()

        for i in range(n):
            ax.scatter(*X[y == i].T, color=colormap(colors(i)), s=4, label="Class " + str(i))

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        
        xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
        # print(np.c_[xx.ravel(), yy.ravel()].shape)
        Z = np.argmax(self.decision_function(np.c_[xx.ravel(), yy.ravel()]), axis=1).reshape(xx.shape)

        cmap = ListedColormap([colormap(i) for i in range(n)])
        
        # ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
        col_ind = 0
        plot_box_poly = pc.box2poly([[xmin, xmax], [ymin, ymax]])
        for p in self.regions:
            # p = p.intersect(plot_box_poly)
            # print(p)
            p.plot(ax, color=colormap(colors(col_ind)), alpha=0.3)
            col_ind += 1
        if title == "":
            ax.set_title(f"Learning Rate={self.eta}")
        else:
            ax.set_title(title + f'\n$\eta = {self.eta}$', fontsize=10)
        # ax.legend()

        return fig, ax


def generate_random(N):
    return np.random.rand(N, 2)

def which_classes(X, polys):
    y = np.ones((X.shape[0],))*-1
    for i, x in enumerate(X):
        for j, p in enumerate(polys):
            if x in p:
                y[i] = j
                break
    return y

def generate_random_in_poly(N, m, poly):
    X = np.zeros((N, 2))
    i = 0
    while i < N:
        x = np.random.rand(1, 2) * 2 * m - m
        if x in poly:
            X[i] = x
            i += 1
    return X

def plot_with_polys(X, y, polys, colors, title=""):
    f, a = plt.subplots()
    n = max(y)
    for j, p in enumerate(polys):
        a.scatter(*X[y==j].T, color=colormap(colors[j]), s=1)
        p.plot(ax=a, color=colormap(colors[j]), alpha=0.2, linewidth=1)
    a.set_xlim([-0.01,1.01])
    a.set_ylim([-0.01,1.01])

    a.set_title(title)

if __name__ == "__main__":

    from scipy.spatial import Voronoi

    # Sample points for Voronoi
    n_vor_regions = 10
    n_pts = 1000
    partial = 9
    with_prev_portion = 0.3

    etas = [0.01, 0.1, 1, 10]
    nb_epochs = 50
    np.random.seed(2)
    points = np.random.rand(n_vor_regions, 2)  # 10 random points in 2D
    vor = Voronoi(points)

    def get_perp_bis(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        return np.array([x2-x1, y2-y1]), (y2**2-y1**2+x2**2-x1**2)/2

    ridge_points = vor.ridge_points
    polytopes = []
    for i in range(n_vor_regions):
        polytopes.append([np.ndarray((0,2)), np.ndarray((0,))])
    
    for rp in ridge_points:
        i1, i2 = rp
        
        p1 = points[i1]
        p2 = points[i2]

        A_p, b_p = get_perp_bis(p1, p2)

        A1, b1 = polytopes[i1]
        A2, b2 = polytopes[i2]
        polytopes[i1] = [np.vstack([A1, A_p]), np.hstack([b1, b_p])]
        polytopes[i2] = [np.vstack([A2, -A_p]), np.hstack([b2, -b_p])]

    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(ridge_points)

    if n_vor_regions <= 10:
        colors = nx.coloring.equitable_color(G, 10)
        colormap = plt.cm.tab10
    else:
        try:
            colors = nx.coloring.equitable_color(G, 20)
            colormap = plt.cm.tab20
        except:
            colors = nx.coloring.greedy_color(G)
            colormap = plt.cm.tab10

    # Define bounding box constraints, e.g., for a box from [0,0] to [1,1]
    A_box = np.array([
        [1, 0],   # x >= 0
        [-1, 0],  # x <= 1
        [0, 1],   # y >= 0
        [0, -1],  # y <= 1
    ])
    b_box = np.array([1.5, 0.5, 1.5, 0.5])

    # Convert each unbounded region to a bounded polytope within the box
    bounded_polytopes = []
    for A, b in polytopes:
        A_full = np.vstack([A, A_box])
        b_full = np.hstack([b, b_box])
        bounded_polytope = pc.Polytope(A_full, b_full)
        bounded_polytopes.append(bounded_polytope)



    X = generate_random(n_pts)
    y = which_classes(X, bounded_polytopes)


    X_fit = X[y<partial]
    y_fit = y[y<partial]
    X_new = X[y>= partial]
    y_new = y[y>=partial]


    for eta in etas:
        LR = IncrementalLogsiticRGD_withLR(eta=eta, nb_epochs=nb_epochs)

        LR.fit(X_fit,y_fit)
        LR.plot_classi(X_fit,y_fit, title="First fit.", colors=lambda x: colors[x])


        LR.partial_fit(X_new, y_new, init_method='lr-centroid', with_prev_portion=with_prev_portion, Xplt=X, yplt=y, colors=lambda x: colors[x])
        
        LR.plot_classi(X, y, title="After applying GD on new data.", colors=lambda x: colors[x])

        plt.show()


    f, a = plt.subplots()
    a.scatter(*points.T, s=2)
    for i, p in enumerate(bounded_polytopes):
        p.plot(ax = a, color=colormap(colors[i]), alpha=0.2, linewidth=1)
    
    a.set_xlim([-0.01,1.01])
    a.set_ylim([-0.01,1.01])
    a.set_title("Voronoi points and regions")

    plot_with_polys(X, y, bounded_polytopes, colors=colors, title="Sampled points")
    plot_with_polys(X_fit, y_fit, bounded_polytopes[:partial], colors=colors, title="Points given to the first fit")

    plt.show()
    exit()


    m = 10
    boxpoly = pc.box2poly([[-m,m],[-m,m]])

    A1 = np.array([[-1.,-0.], [-0.,-1.],[ 1.,-0.], [-0.,1.], [ 0.99755,  0.06997], [ 0.34677, -0.93795]])
    b1 = np.array([ 1.00000e+20, 1.00000e+20,  1.00000e+20,  1.00000e+20,  2.50281e-01, -6.25994e-01])
    A2 = np.array([[-1.,-0.], [-0.,-1.],[ 1.,-0.], [-0.,1.], [-0.99755, -0.06997], [-0.19863, -0.98008]])
    b2 = np.array([ 1.00000e+20, 1.00000e+20,  1.00000e+20,  1.00000e+20,  -2.50281e-01, -7.65697e-01])
    A3 = np.array([[-1.,-0.], [-0.,-1.],[ 1.,-0.], [-0.,1.], [-0.34677, 0.93795], [0.19863, 0.98008], [0.9, 0.1]])
    b3 = np.array([ 1.00000e+20, 1.00000e+20,  1.00000e+20,  1.00000e+20,  6.25994e-01, 7.65697e-01, -0.5])
    A4 = np.array([[-1.,-0.], [-0.,-1.],[ 1.,-0.], [-0.,1.], [-0.34677, 0.93795], [0.19863, 0.98008], [-0.9, -0.1]])
    b4 = np.array([ 1.00000e+20, 1.00000e+20,  1.00000e+20,  1.00000e+20,  6.25994e-01, 7.65697e-01, 0.5])
    

    

    p1 = pc.Polytope(A1, b1).intersect(boxpoly)
    p2 = pc.Polytope(A2, b2).intersect(boxpoly)
    p3 = pc.Polytope(A3, b3).intersect(boxpoly)
    p4 = pc.Polytope(A4, b4).intersect(boxpoly)
    
    

    polys = [p1, p2, p3]

    X = generate_random(1000, m)
    y = which_classes(X, polys)
    X = X[y != -1]
    y = y[y != -1]

    
    N_new = 200
    X_new = generate_random_in_poly(N_new, m, p4)
    y_new = np.array([3]*N_new)
    


    eta = 0.1
    nb_epochs = 50

    LR = IncrementalLogsiticRGD_withLR(eta=eta, nb_epochs=nb_epochs)

    LR.fit(X,y)
    LR.plot_classi(X,y)

    X_all = np.r_[X, X_new]
    y_all = np.r_[y, y_new]

    LR.partial_fit(X_new, y_new, with_prev_portion=0.3, Xplt=X_all, yplt=y_all)
    LR.plot_classi(X_all, y_all)

    plot_with_polys(X_all, y_all, [p1,p2,p3,p4])

    # p4 = pc.Polytope((p1.A+p2.A)/2,(p1.b+p2.b)/2)
    # p4.plot(ax=a, alpha=0.2)
    plt.show()
    exit()
    n_samples = 200
    n_classes = 4
    eta = 0.1
    nb_epochs = 50

    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                            n_clusters_per_class=1, n_redundant=0, n_classes=n_classes, random_state=1)


    # for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1 , 1]:
    LR = IncrementalLogsiticRGD_withLR(eta=eta, nb_epochs=nb_epochs)


    # LR.W = np.array([[-1.00231, -0.98778], [ 0.99056, -0.85561], [-1.08337,  0.80221], [1.0791,   1.02797]])
    # LR.B = np.array([-0.17557, -0.09156,  0.14826,  0.12181])
    # LR.W = np.array([[-1, -1], [ 1, -1], [-1,  1], [1,   1]])
    # LR.B = np.array([0,0,0,0])
    # LR.W = np.array([[0, -1], [-1,  1], [1,   1]])
    # LR.B = np.array([0,0,0])

    # LR.K = LR.B.size
    # LR.define_region_polytopes()
    # LR.plot_classi(X, y)
    # plt.show()
    # exit()


    LR.fit(X,y)
    LR.plot_classi(X,y)

    print(LR.W)
    print(LR.B)

    # LR.fit(X,y)
    # LR.plot_classi(X,y)
    # plt.show()
    # exit()
    X_new = np.array([[-2,-2], [-1.9, -1.3], [0.3, -1], [-1.5, -3], [-2, -3]])
    y_new = np.array([4, 4, 0, 5, 5])
    # X_new = np.array([[-2,-2], [-1.9, -1.3], [0.3, -1]])
    # y_new = np.array([4, 4, 0])
    X_all = np.r_[X, X_new]
    y_all = np.r_[y, y_new]


    # LR.partial_fit(X_new, y_new, with_prev=True, with_prev_portion=0.4)
    # LR.plot_classi(X_all, y_all)

    # LR.fit(X, y)
    LR.partial_fit(X_new, y_new, with_prev_portion=0.75, Xplt=X_all, yplt=y_all)
    LR.plot_classi(X_all, y_all)

    # LR.fit(X, y)
    # LR.partial_fit(X_all, y_all)
    # LR.plot_classi(X_all, y_all)

    # LR.fit(X_all, y_all)
    # LR.plot_classi(X_all, y_all)

    plt.show()

    ## NEED TO ADD:
    # with_prev needs to select privilege selecting points that are close to boundaries