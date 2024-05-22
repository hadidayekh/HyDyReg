from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_classi(X, y, est, h = 0.05):
    n = np.unique(y).size

    fig, ax = plt.subplots()

    for i in range(n):
        ax.scatter(*X[y == i].T, color=plt.cm.tab10(i), s=4)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    Z = np.argmax(est.decision_function(np.c_[xx.ravel(), yy.ravel()]), axis=1).reshape(xx.shape)

    cmap = ListedColormap([plt.cm.tab10(i) for i in range(n)])
    
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

    return fig, ax

est = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss'))

X0 = np.array([[-2,3], [-2,4], [-3,3], [-3,4]])
y0 = np.array([0]*X0.shape[0])
X1 = np.array([[-2,-3], [-2,-4], [-3,-3], [-3,-4], [-2, -2], [-3, -2], [-1,-1], [-4,-1]])
y1 = np.array([1]*X1.shape[0])
X2 = np.array([[1,-1], [1,-2], [2,-2], [3,-2], [4,-2], [4,-3], [4,-4], [3,-3]])
y2 = np.array([2]*X2.shape[0])
X3 = np.array([[2,2], [1,3], [3,4], [4,3], [1,2], [2,4]])
y3 = np.array([3]*X3.shape[0])

X = np.r_[X0, X1, X2]
y = np.r_[y0, y1, y2]

est.fit(X,y)

plot_classi(X, y, est)

X1_rect = np.array([[-2, 0.5], [-3, 0.8], [-1, 1], [-4, 1.5]])
y1_rect = np.array([1]*X1_rect.shape[0])

Xall = np.r_[X, X1_rect]
yall = np.r_[y, y1_rect]

X1_rect_std = est[0].fit_transform(X1_rect)

for i in range(10):
    est[1].partial_fit(X1_rect_std, y1_rect, classes=np.unique(yall))
    plot_classi(Xall, yall, est)

print(est.predict(X1_rect))
plt.show()