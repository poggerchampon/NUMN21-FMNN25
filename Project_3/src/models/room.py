import numpy as np

from src.utilities.constants import u_normal, u_heater, u_window

class Room:
    def __init__(self, dimensions, boundary_conditions, adjacent_rooms=None):
        self.dimensions = dimensions
        self.boundary_conditions = boundary_conditions # dictionary of values for boundary conditions (walls)
        self.adjacent_rooms = adjacent_rooms if adjacent_rooms else {} # dictionary of adjacent room numbers
        self.temperature = np.full(self.dimensions, u_normal)
        
        self.constant_mapping = {'u_heater': u_heater, 'u_window': u_window, 'u_normal': u_normal} # Mapping of string 
        
        # set the boundary conditions
        self.set_boundary_conditions()

    def set_boundary_conditions(self):
        # Apply the boundary conditions
        for wall, temp_value in self.boundary_conditions.items():
            boundary_val = self.constant_mapping.get(temp_value, temp_value)  # Look up the value from the mapping
            match wall:
                case 'top':
                    self.temperature[0, :] = boundary_val
                case 'bottom':
                    self.temperature[-1, :] = boundary_val
                case 'left':
                    self.temperature[:, 0] = boundary_val
                case 'right':
                    self.temperature[:, -1] = boundary_val
