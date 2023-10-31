import matplotlib.pyplot as plt
import numpy as np

# Generate data
num_points = 1000  
x = np.random.randn(num_points)
y = np.random.randn(num_points)
z = np.random.randn(num_points)

# Plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make axes lines invisible
ax.set_axis_off()

# Plot scatterplot
ax.scatter(x, y, z, s=20, edgecolor=None)

# Show plot
plt.show()