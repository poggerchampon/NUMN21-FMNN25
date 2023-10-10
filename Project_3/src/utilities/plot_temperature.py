import matplotlib.pyplot as plt
import numpy as np

def plot_temperature(u, room_number, min_temperature=0, max_temperature=40):
    # Ignore the edges, as they are boundary conditions
    u_plot = u[1:-1, 1:-1]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(u_plot, cmap='hot', interpolation='nearest', vmin=min_temperature, vmax=max_temperature)
    cbar = plt.colorbar(label='Temperature')
    
    plt.title(f"Temperature distribution in Room {room_number}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
