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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists




global colormap

n_vor_regions = 8


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


# Define bounding box constraints, e.g., for a box from [0,0] to [1,1]
A_box = np.array([
    [1, 0],   # x >= 0
    [-1, 0],  # x <= 1
    [0, 1],   # y >= 0
    [0, -1],  # y <= 1
])
b_box = np.array([1, 0, 1, 0])

for i in range(len(b_box)):
    events.append(lambda t, x, A_p=A_box[i], b_p=b_box[i]: A_p.T @ x - b_p)
    
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
b_box = np.array([1, 0, 1, 0])

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



image_path = 'antenna.png'

star = 0
p_star = points[star]


f, a = plt.subplots()
# a.scatter(*points.T, s=4, marker='*', color='k')

imscatter(*points.T, image_path, zoom=0.015, ax=a)
imscatter(*p_star, image_path, zoom=0.025, ax=a)

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
    return -1    


eqs = []
n = len(bounded_polytopes)
A_dy = 10*(np.random.rand(n, 2))
A_dy = np.zeros((n, 2, 2))
for i in range(n):
    A_dy[i] = np.array([[-2, -0.5*i], [0.5*i, -3]])
# A_dy = np.array([[-2.5, -2], [2, -2.5]])
B_dy = np.random.rand(n)-0.5
B_dy = np.array([1,0])
print(A_dy.shape)
print(B_dy.shape)
for i,p in enumerate(bounded_polytopes):
    if i == star:
        def eq(t, x, i = i):
            # print(i, 'star')
            return (p_star - x)
        
        eqs.append(eq)
    else:
        def eq(t, x, i = i):
            # if i != 1:
            #     print(i)
            # return 0.1*(p_star - points[i]) / np.linalg.norm(p_star - points[i])
            return -A_dy[i] @ (points[i]-x) #+ B_dy #A_dy[i] @ (x-points[i]) #+ B_dy[i]
        
        eqs.append(eq)

eqs.append(lambda t, x : np.array([0,0]))

sul = SNDS(len(bounded_polytopes)+1, 2, eqs, eventlist=events, labelfun=labelfun)

# x0s = [[0.3,0.5], [0.5,0.8], [0.1, 0.9], [0.5,0.5], [0.9,0.1], [0.3,0.1]]
x0s = np.random.rand(20, 2) 
trajs = sul.simulate_trajectory(0.01, 10, x0s)




from diff_methods import diff_method_fps
from opti import HybridRegressor
from sklearn.linear_model import Ridge, RidgeCV

# f, a = draw_trajectories_2D(trajs, mark_beginning=True, fig=f, axs=a)#, segments=segments)


segments = trajs.segment_trajectories(0.001)

# f, a = draw_trajectories_2D(trajs, mark_beginning=True, fig=f, axs=a, segments=segments)

X, X_dot, seg_res, _ = diff_method_fps(
    trajs.t_list, trajs.x_list, segments=segments, remove_seg=True)
# print(X.shape)
# exit()
# plt.scatter(*X.T,s=2)
# plt.show()
# exit()


model = HybridRegressor(train_coefs=True, deg=1, classifier_poly_deg=1)

alphas=[1000,100,10,1,.1,.01]
model.fit_first_incremental(X, X_dot, seg_res, good_RD = 0.001)#, regressor = RidgeCV(alpha_per_target=True, alphas=alphas))

model.classify_modes(LogisticRegression())

print(model.gen_full_latex_eq())

learned_hyp = model.to_SNDS()
trajs_learned = learned_hyp.simulate_trajectory(0.02, 10, x0s)

f2, a2 = draw_trajectories_2D(trajs, mark_beginning=True, colors = ['k'])
f2, a2 = draw_trajectories_2D(trajs_learned, fig=f2, axs=a2, mark_beginning=True, labelfun=model.labelfun_, colors = ['r']) 

plt.show()
exit()

