import numpy as np
import matplotlib.pyplot as plt

# Define the coefficients
a = 1
b = 1
c = 0
d = -0.51

a2 = 1
b2 = 1.004
c2 = -0.002
d2 = -0.5113

# Create a grid of x and y values
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# Calculate the equation for each point on the grid
Z = a * X**2 + b * Y**2 + c * X * Y + d

Z2 = a2 * X**2 + b2 * Y**2 + c2 * X * Y + d2


# Create a contour plot
plt.contour(X, Y, Z, levels=[0], colors='r')
plt.contour(X, Y, Z2, levels=[0], colors='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of ax^2 + by^2 + cxy + d = 0')
plt.grid(True)
plt.show()
