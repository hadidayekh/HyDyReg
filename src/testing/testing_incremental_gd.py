from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
import polytope as pc



class IncrementalLogsiticRGD:
    def __init__(self, learning_rate=0.0001, nb_epochs=50) -> None:
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs

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
        self.regions = []
        for k in range(self.K):
            A = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            b = [1e6, 1e6, 1e6, 1e6]
            for j in range(self.K):
                if j != k:
                    A.append(self.W[k] - self.W[j])
                    b.append(self.B[k] - self.B[j])
            A = -np.array(A)
            b = np.array(b)
            self.regions.append(pc.reduce(pc.Polytope(A, b)))

    def fit(self, X, y):
        self.K = len(np.unique(y))   # number of classes
        self.D = len(X[0])

        self.W = np.zeros((self.K, self.D))
        self.B = np.zeros(self.K)

        self.X_ = np.ndarray((0, X[0].size))
        self.y_ = np.ndarray((0,))

        self.partial_fit(X, y, with_prev=False)



    def partial_fit(self, X, y, with_prev=False):
        assert len(X[0]) == self.D
        N = X.shape[0]          # number of data points
        new_K = np.max(y) + 1   # this assumes that new classes are always added in ascending order
        if new_K > self.K:
            n_new_classes = new_K - self.K
            new_W = np.zeros((n_new_classes, self.D))
            new_B = np.zeros(n_new_classes)
            self.W = np.r_[self.W, new_W]
            self.B = np.r_[self.B, new_B]
            self.K = new_K

        X_old = X
        y_old = y
        if with_prev:
            rand_ints = np.random.randint(0, self.X_.shape[0], int(self.X_.shape[0]*0.1))
            X_prev_rand = self.X_[rand_ints]
            y_prev_rand = self.y_[rand_ints]
            X = np.r_[X, X_prev_rand]
            y = np.r_[y, y_prev_rand]
        for _ in range(self.nb_epochs):
            p = np.random.permutation(len(y))
            X_shuffled = X[p]
            y_shuffled = y[p]
            for i in range(N):
                for k in range(self.K):
                    grad_L_w, grad_L_b = self.grad_L(X_shuffled[i], y_shuffled[i], k)
                    self.W[k] = self.W[k] - self.learning_rate * grad_L_w
                    self.B[k] = self.B[k] - self.learning_rate * grad_L_b
        self.define_region_polytopes()
        
        self.X_ = np.r_[self.X_, X_old]
        self.y_ = np.r_[self.y_, y_old]

    def decision_function(self, X):
        return X @ self.W.T + self.B


    def plot_classi(self, X, y, h = 0.05):
        n = np.unique(y).size

        fig, ax = plt.subplots()

        for i in range(n):
            ax.scatter(*X[y == i].T, color=plt.cm.tab10(i), s=4, label="Class " + str(i))

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        
        xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
        # print(np.c_[xx.ravel(), yy.ravel()].shape)
        Z = np.argmax(self.decision_function(np.c_[xx.ravel(), yy.ravel()]), axis=1).reshape(xx.shape)

        cmap = ListedColormap([plt.cm.tab10(i) for i in range(n)])
        
        # ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
        col_ind = 0
        for p in self.regions:
            p.plot(ax, color=plt.cm.tab10(col_ind), alpha=0.3)
            col_ind += 1
        ax.set_title(f"Learning Rate={self.learning_rate}")
        ax.legend()

        return fig, ax

n_samples = 20
n_classes = 4
learning_rate = 0.01
nb_epochs = 50

X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                           n_clusters_per_class=1, n_redundant=0, n_classes=n_classes, random_state=1)


# for learning_rate in [0.00001, 0.0001, 0.001, 0.01, 0.1 , 1]:
LR = IncrementalLogsiticRGD(learning_rate=learning_rate, nb_epochs=nb_epochs)

LR.fit(X,y)
LR.plot_classi(X,y)
# plt.show()
# exit()
X_new = np.array([[-2,-2], [-1.9, -1.3], [0.3, -1]])
y_new = np.array([4, 4, 0])
X_all = np.r_[X, X_new]
y_all = np.r_[y, y_new]


LR.partial_fit(X_new, y_new)
LR.plot_classi(X_all, y_all)

LR.fit(X, y)
LR.partial_fit(X_all, y_all)
LR.plot_classi(X_all, y_all)

LR.fit(X_all, y_all)
LR.plot_classi(X_all, y_all)

plt.show()