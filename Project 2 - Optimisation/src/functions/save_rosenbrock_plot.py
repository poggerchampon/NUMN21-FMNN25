import numpy as np
import matplotlib.pyplot as plt

FOLDER_PATH = '../Plots/'

def rosenbrock(x):
	return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def save_rosenbrock_plot(path, filename='contour_plot.png', cmap='viridis'):
	X, Y = np.meshgrid(np.linspace(-0.5, 1.9, 400), np.linspace(-0.5, 4, 400))
	Z = rosenbrock(np.array([X, Y]))
	
	fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
	
	# Create the contour plot
	contour = ax.contour(X, Y, Z, levels=np.logspace(-1, 5, 18), cmap=cmap)
	cbar = plt.colorbar(contour, ax=ax)
	cbar.set_label('Function Value', labelpad=12, fontsize=12)
	
	# Label contour lines
	ax.clabel(contour, inline=1, fontsize=10, fmt='%1.1f')
	
	# Plot the path & mark the points
	ax.plot(path[:, 0], path[:, 1], 'k-', linewidth=3, label='Optimisation Path')
	ax.plot(path[:, 0], path[:, 1], 'co', markersize=7, label='Points')
	
	ax.set_xlabel('X-axis')
	ax.set_ylabel('Y-axis')
	ax.set_title('Rosenbrock Function Contour Plot')
	ax.legend()
	ax.grid(True)
	
	# ensure everything fits
	plt.tight_layout()
	
	# Save the plot to a file
	plt.savefig(FOLDER_PATH + filename, dpi=300)
