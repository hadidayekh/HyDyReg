import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.random((50,1)) - 0.5
x2 = np.random.random((50,1)) - 0.5
R = 0.3

def dist(x1,x2):
    return np.sqrt(x1**2 + x2**2)

def plot_circle(x,y,r,ax,color='r'):
    th = np.linspace(0, np.pi*2, 100)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y

    ax.plot(xunit, yunit, color=color)

def plot_circle2(x,y,r,ax,color='r'):
    th = np.linspace(0, np.pi*2, 100)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y

    ax.plot(xunit**2, yunit**2, color=color)

f1, a1 = plt.subplots()
f2, a2 = plt.subplots()

for i in range(x1.shape[0]):
    if dist(x1[i], x2[i]) < R:
        a1.scatter(x1[i], x2[i], color='black')
        a2.scatter(x1[i]**2, x2[i]**2, color='black')
    else:
        a1.scatter(x1[i], x2[i], color='red',facecolors='none')
        a2.scatter(x1[i]**2, x2[i]**2, color='red',facecolors='none')

plot_circle(0, 0, R, a1)
plot_circle2(0, 0, R, a2)

a1.set_xlabel('$x^{(1)}$')
a1.set_ylabel('$x^{(2)}$')

a2.set_xlabel('${x^{(1)}}^2$')
a2.set_ylabel('${x^{(2)}}^2$')

a1.set_aspect('equal', adjustable='box')
a2.set_aspect('equal', adjustable='box')


plt.show()
