# logistic regression for multi-class classification using built-in one-vs-rest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.colors
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
# define dataset
nc=4
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_clusters_per_class=1, n_redundant=0, n_classes=nc, random_state=1)


# define model
#model = LogisticRegression(multi_class='ovr')

model = LogisticRegression(penalty='l2')
model = OneVsRestClassifier(model)

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

fig,ax = plt.subplots()
ax.set_aspect(1)

colors = ['tab:blue', 'tab:orange', 'tab:green']
if nc == 4:
    colors.append('red')
coloors=['r','b']
cmap = matplotlib.colors.ListedColormap(colors)

#disp = DecisionBoundaryDisplay.from_estimator(model, X, grid_resolution=1000, alpha=0.3, cmap=cmap)


# Plot the data and the classification with the decision boundary.
xmin, xmax = -4, 4
ymin, ymax = -4, 4
xd = np.array([xmin, xmax])

# plot estimators
for i in range(len(estis)):
    clf = estis[i]
    b = clf.intercept_
    w = clf.coef_

    w1, w2 = w.T
    c = -b/w2
    m = -w1/w2
    yd = m*xd + c
    # yd = m[i]*xd + c[i]
    ax.plot(xd, yd, 'k', lw=1, ls='--')
    # if i == 0:
    #     ax.fill_between(xd, yd, ymax, color=colors[i], alpha=0.2)
    #     yd = mbis*xd + cbis
    #     ax.plot(xd, yd, 'r', lw=1)
    # else:
    #     ax.fill_between(xd, yd, ymin, color=colors[i], alpha=0.2)

xd = np.linspace(xmin, xmax,1000)

for i in range(len(estis)):
    esti1 = estis[i]
    w1 = esti1.coef_
    b1 = esti1.intercept_
    slope1 = -w1.T[0]/w1.T[1]
    wnorm1 = np.linalg.norm(w1)
    for j in range(i+1, len(estis)):
        if i==0 and j==1:
            xd = np.linspace(0.1, 0.3,1000)
        else:
            xd = np.linspace(xmin, xmax,1000)

        esti2 = estis[j]

        w2 = esti2.coef_
        b2 = esti2.intercept_
        slope2 = -w2.T[0]/w2.T[1]
        wnorm2 = np.linalg.norm(w2)

        # decision function
        w = w1 - w2
        b = b1 - b2

        a1, a2 = w.T
        c = -b/a2
        m = -a1/a2
        slope_mid = m
        yd = m*xd + c

        pr_min = model.predict_proba([[xd[0], yd[0]]])[0]
        pr_max = model.predict_proba([[xd[-1], yd[-1]]])[0]
        
        max1 = np.max(pr_min)
        i1 = np.argmax(pr_min)
        max3 = np.max(pr_max)
        i3 = np.argmax(pr_max)
        
        pr_min[i1] = -1
        pr_max[i3] = -1

        max2 = np.max(pr_min)
        i2 = np.argmax(pr_min)
        max4 = np.max(pr_max)
        i4 = np.argmax(pr_max)

        newx = []
        newy = []
        for q in range(len(xd)):
            if model.predict([[xd[q], yd[q]]])[0] == i or model.predict([[xd[q], yd[q]]])[0] == j:
                newx.append(xd[q])
                newy.append(yd[q])
        ax.plot(newx, newy, 'k', lw = 1)


        # BISECTOR
        w_bis = w1/wnorm1 - w2/wnorm2
        b_bis = b1/wnorm1 - b2/wnorm2
        a1, a2 = w_bis.T
        c = -b_bis/a2
        m = -a1/a2
        slope_bis = m
        yd = m*xd + c
        xd = np.linspace(xmin, xmax,1000)


        angle_12  = np.arctan((slope1 - slope2)/(1 + slope1*slope2))
        angle_w_1 = np.arctan((slope1 - slope_mid)/(1 + slope1*slope_mid))     
        angle_w_2 = np.arctan((slope_mid - slope2)/(1 + slope_mid*slope2))

        print(angle_12, angle_w_1, angle_w_2, angle_w_1 + angle_w_2)

h = 0.05
xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

# individual estimator i think??
def pred_func(x):

    N = x.shape[0]
    y = np.zeros(N)

    estis0 = estis[0].predict(x)
    estis1 = estis[1].predict(x)
    estis2 = estis[2].predict(x)
    estis3 = estis[3].predict(x)

    for i in range(N):
        if   estis0[i] == 1 and estis1[i] == 0 and estis2[i] == 0 and estis3[i] == 0:
            y[i] = 0
        elif estis0[i] == 0 and estis1[i] == 1 and estis2[i] == 0 and estis3[i] == 0:
            y[i] = 1
        elif estis0[i] == 0 and estis1[i] == 0 and estis2[i] == 1 and estis3[i] == 0:
            y[i] = 2
        elif estis0[i] == 0 and estis1[i] == 0 and estis2[i] == 0 and estis3[i] == 1:
            y[i] = 3
        else:
            y[i] = 4
    return y

def dec_function_manual(X):
    res = np.ndarray((X.shape[0], len(estis)))
    for i in range(X.shape[0]):
        for j in range(len(estis)):
            b = estis[j].intercept_
            w = estis[j].coef_
            res[i][j] = np.dot(w, X[i]) + b
    return res


Z1 = np.array(np.argmax(model.decision_function(np.c_[xx.ravel(), yy.ravel()]), axis=1))
Z2 = np.array(np.argmax(model.predict_proba(np.c_[xx.ravel(), yy.ravel()]), axis=1))
Z3 = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z4 = np.array(np.argmax(dec_function_manual(np.c_[xx.ravel(), yy.ravel()]), axis=1))

print(np.array_equal(Z1, Z2))
print(np.array_equal(Z3, Z2))

Z = Z4.reshape(xx.shape)
print(Z)
ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

#Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
#print(Z)
#Z = Z.T[1].reshape(xx.shape)


for i in range(nc):
    ax.scatter(*X[yhat==i].T, s=8, alpha=0.9, color=colors[i])

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_ylabel(r'$x_2$')
ax.set_xlabel(r'$x_1$')

fig.savefig("plots/fig.pdf")

