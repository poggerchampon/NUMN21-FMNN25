import numpy as np

from src.utilities.constants import u_normal

class Room:
    def __init__(self, dimensions, boundary_conditions, adjacent_rooms=None):
        self.dimensions = dimensions
        self.boundary_conditions = boundary_conditions # dictionary of values for boundary conditions (walls)
        self.adjacent_rooms = adjacent_rooms if adjacent_rooms else {} # dictionary of adjacent room numbers
        self.temperature = np.full(self.dimensions, u_normal)

    def set_boundary_conditions(self):
        # Apply the boundary conditions
        for wall, temp_value in self.boundary_conditions.items():
            match wall:
                case 'top':
                    self.temperature[0, :] = temp_value
                case 'bottom':
                    self.temperature[-1, :] = temp_value
                case 'left':
                    self.temperature[:, 0] = temp_value
                case 'right':
                    self.temperature[:, -1] = temp_value
