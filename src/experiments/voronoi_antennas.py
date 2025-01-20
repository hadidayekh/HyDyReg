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
from math import floor, sqrt
import pickle
import sys
import time

# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')

from snds import SNDS
from plotting import draw_trajectories_2D

global colormap

n_vor_regions = 30


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


from scipy.spatial import Voronoi

# Sample points for Voronoi
n_pts = 1000
partial = 9
with_prev_portion = 0.3
init_method='zero'
eps = 1

etas = [0.01, 0.1, 1]
nb_epochs = 50
nb_epochs_partial=1
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

events = []
for rp in ridge_points:
    i1, i2 = rp
    
    p1 = points[i1]
    p2 = points[i2]

    A_p, b_p = get_perp_bis(p1, p2)

    A1, b1 = polytopes[i1]
    A2, b2 = polytopes[i2]
    polytopes[i1] = [np.vstack([A1, A_p]), np.hstack([b1, b_p])]
    polytopes[i2] = [np.vstack([A2, -A_p]), np.hstack([b2, -b_p])]

    events.append(lambda t, x, A_p=A_p, b_p=b_p: A_p.T @ x - b_p)
    
for e in events:
    e.terminal = True
    e.direction = 0
print(len(events))
# exit()
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


f, a = plt.subplots()
a.scatter(*points.T, s=2)
for i, p in enumerate(bounded_polytopes):
    p.plot(ax = a, color=colormap(colors[i]), alpha=0.2, linewidth=1)

a.set_xlim([-0.01,1.01])
a.set_ylim([-0.01,1.01])
a.set_title("Voronoi points and regions")

# plot_with_polys(X, y, bounded_polytopes, colors=colors, title="Sampled points")
# plot_with_polys(X_fit, y_fit, bounded_polytopes[:partial], colors=colors, title="Points given to the first fit")




# Label function to decide which mode
def labelfun(x):
    rs = []
    for i, p in enumerate(bounded_polytopes):
        if x in p:
            # print(i)
            # sleep(0.1)
            return i    


eqs = []
star = 5
p_star = points[star]
for i,p in enumerate(bounded_polytopes):
    if i == star:
        def eq(t, x, i = i):
            # print(i, 'star')
            return p_star - x
        
        eqs.append(eq)
    else:
        def eq(t, x, i = i):
            # if i != 1:
            #     print(i)
            return p_star - points[i]
        
        eqs.append(eq)

sul = SNDS(len(bounded_polytopes), 2, eqs, eventlist=events, labelfun=labelfun)

x0s = [[0.3,0.5], [0.5,0.8], [0.1, 0.9], [0.5,0.5], [0.9,0.1], [0.3,0.1]]
trajs = sul.simulate_trajectory(0.1, 5, x0s)
f, a = draw_trajectories_2D(trajs, mark_beginning=True, fig=f, axs=a)
a.scatter(*p_star)
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