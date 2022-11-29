# logistic regression for multi-class classification using built-in one-vs-rest
from turtle import color
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.colors
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
# define dataset
nc=3
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_clusters_per_class=1, n_redundant=0, n_classes=nc, random_state=1)


# define model
#model = LogisticRegression(multi_class='ovr')

model = LogisticRegression()
model = OneVsOneClassifier(model)

# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)


estis = model.estimators_


# clf = model

# # Retrieve the model parameters.
# b = clf.intercept_
# w = clf.coef_

# w1, w2 = w.T
# print(w1, w2)

# # Calculate the intercept and gradient of the decision boundary.
# c = -b/w2
# m = -w1/w2


colors = ['tab:blue', 'tab:orange', 'tab:green']
cmap = matplotlib.colors.ListedColormap(colors)

disp = DecisionBoundaryDisplay.from_estimator(model, X, grid_resolution=10000, alpha=0.3, cmap=cmap)


# Plot the data and the classification with the decision boundary.
xmin, xmax = -4, 4
ymin, ymax = -4, 4
xd = np.array([xmin, xmax])
for i in range(nc):
    clf = estis[i]
    b = clf.intercept_
    w = clf.coef_

    w1, w2 = w.T
    c = -b/w2
    m = -w1/w2
    yd = m*xd + c
    # yd = m[i]*xd + c[i]
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    # if i == 0:
    #     plt.fill_between(xd, yd, ymax, color=colors[i], alpha=0.2)
    #     yd = mbis*xd + cbis
    #     plt.plot(xd, yd, 'r', lw=1)
    # else:
    #     plt.fill_between(xd, yd, ymin, color=colors[i], alpha=0.2)
#plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

for i in range(nc):
    plt.scatter(*X[yhat==i].T, s=8, alpha=0.7, color=colors[i])

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')

plt.show()


exit()
for i in range(y.size):
    c = ""
    if yhat[i] == 0:
    #if y[i] == 0:
        c = "b"
    elif yhat[i] == 1:
    #elif y[i] == 1:
        c = "r"
    else:
        c = "g"
    plt.scatter(X[i][0], X[i][1], color=c)

plt.show()

print(yhat)