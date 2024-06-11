import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification



class IncrementalLogsiticRGD:
    def __init__(self, learning_rate=0.0001, nb_epochs=50) -> None:
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs

    def grad_L(self, x, y, k):
        W = self.W
        B = self.B
        Z = 0
        
        for j in range(len(x)):
            Z += np.exp(np.dot(W[j], x) + B[j])
        
        P = np.exp(np.dot(W[k], x) + B[k])/Z

        one_cond = 1 if y == k else 0

        grad_b_k = P - one_cond
        grad_w_k = x * grad_b_k

        return grad_w_k, grad_b_k
    
    def fit(self, X, y):
        self.K = len(np.unique(y))   # number of classes

        self.W = np.zeros((self.K, len(X[0])))
        self.B = np.zeros(self.K)

        self.partial_fit(X, y)

    def partial_fit(self, X, y):
        N = X.shape[0]          # number of data points
        for _ in range(self.nb_epochs):
            for i in range(N):
                for k in range(self.K):
                    grad_L = self.grad_L(X[i], y[i], k)
                    self.W[k] = self.W[k] - self.learning_rate * grad_L[0]
                    self.B[k] = self.B[k] - self.learning_rate * grad_L[1]

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
        print(np.c_[xx.ravel(), yy.ravel()].shape)
        Z = np.argmax(self.decision_function(np.c_[xx.ravel(), yy.ravel()]), axis=1).reshape(xx.shape)

        cmap = ListedColormap([plt.cm.tab10(i) for i in range(n)])
        
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
        ax.set_title(f"Learning Rate={self.learning_rate}")
        ax.legend()

        return fig, ax

n_samples = 10
n_classes = 4
learning_rate = 0.005
nb_epochs = 50

X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                           n_clusters_per_class=1, n_redundant=0, n_classes=n_classes, random_state=1)

LR = IncrementalLogsiticRGD(learning_rate=learning_rate, nb_epochs=nb_epochs)

LR.fit(X,y)
LR.plot_classi(X,y)

X_new = np.array([[-4,-2], [0.3, -1]])
y_new = np.array([0, 2])

LR.partial_fit(X_new, y_new)
LR.plot_classi(np.r_[X, X_new], np.r_[y, y_new])

plt.show()