import matplotlib.pyplot as plt
import numpy as np

def plot_temperature(u, room_number, min_temperature=0, max_temperature=40):
    """
    Plots the temperature distribution for a given room based on a 2D numpy array of temperatures.

    Parameters:
    - u : A 2D numpy array representing the temperature at each grid point. The edges of this array are treated as boundary conditions and are not plotted.
    - room_number: The identifier for the room, used in the plot title.
    - min_temperature (optional): The minimum temperature value for color scaling. Defaults to 0.
    - max_temperature (optional): The maximum temperature value for color scaling. Defaults to 40.
    """
    
    u_plot = u[1:-1, 1:-1] # Ignore the edges, as they are boundary conditions
    
    plt.figure(figsize=(8, 8))
    plt.imshow(u_plot, cmap='hot', interpolation='nearest', vmin=min_temperature, vmax=max_temperature)
    cbar = plt.colorbar(label='Temperature')
    
    plt.title(f"Temperature distribution in Room {room_number}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
