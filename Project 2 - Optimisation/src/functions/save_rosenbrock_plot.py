import numpy as np
import matplotlib.pyplot as plt

FOLDER_PATH = '../Plots/'

def rosenbrock(x):
	return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def save_rosenbrock_plot(path, filename='contour_plot.png', cmap='jet'):
	X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
	Z = rosenbrock(np.array([X, Y]))
	
	# Create the contour plot
	contour = plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap=cmap)
	
	# Plot the optimization path
	plt.plot(path[:, 0], path[:, 1], 'k-', linewidth=2)
	
	# Mark the points along the path with red dots
	plt.plot(path[:, 0], path[:, 1], 'co')
	
	# Save the plot to a file
	plt.savefig(FOLDER_PATH + filename)