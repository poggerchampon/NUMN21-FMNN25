import numpy as np

from src.solver.laplace_solver import solve_laplace
from src.utilities.help_functions import interpolate_boundary

class DirichletNeumannSolver:
	def __init__(self, rooms, comm, rank, num_iter=10, omega=0.8, dx=1/20):
		self.rooms = rooms
		self.comm = comm
		self.rank = rank
		
		#parameters
		self.num_iter = num_iter
		self.omega = omega
		self.dx = dx
		
	def solve(self):
		"""
		Perform Dirichlet-Neumann iteration for a dictionary of rooms.
		
		Parameters:
		- rooms: Dictionary of Room objects
		- comm: MPI communicator
		- rank: MPI rank of the current process
		
		Returns:
		- u_dict: Dictionary of updated temperatures for each room
		"""
		u_dict = {i: room.temperature for i, room in enumerate(self.rooms)} # initialise u_dict
		
		for k in range(self.num_iter):
			for i, room in enumerate(self.rooms):
				if self.rank == i:
					u = u_dict[i]
					adjacent_rooms = room.adjacent_rooms
					
					# Send the conditions to the adjacent rooms and collect their conditions
					self.send_conditions(u, adjacent_rooms)
					u = self.receive_and_update_conditions(u, adjacent_rooms)
					
					# Solve laplace equation with the updated conditions
					u_new = solve_laplace(u, self.dx)
					
					# Relaxation
					u_dict[i] = self.omega * u_new + (1 - self.omega) * u
					
		return u_dict
	
	def send_conditions(self, u, adjacent_rooms):
		# Possible to add top and bottom as well. Current .json layout only has left and right
		for direction, adj_info in adjacent_rooms.items():
			adj_rank = adj_info['rank']
			
			# Send rightmost boundary if the adjacent room is to the right
			if direction == "right":
				self.comm.send(u[:, -1], dest=adj_rank, tag=100 + self.rank)
				
			# Send the leftmost boundary if the adjacent room is to the left
			elif direction == "left":
				self.comm.send(u[:, 0], dest=adj_rank, tag=100 + self.rank)
				
	def receive_and_update_conditions(self, u, adjacent_rooms):
		for direction, adj_info in adjacent_rooms.items():
			adj_rank = adj_info['rank']
			adj_type = adj_info['type']
			
			# Receiving the boundary condition from the adjacent room
			u_received = self.comm.recv(source=adj_rank, tag=100 + adj_rank)
			
			# Updating boundary condition based on its type (Dirichlet or Neumann)
			if adj_type == "Dirichlet":
				u = self.update_dirichlet(u, u_received, direction)
			elif adj_type == "Neumann":
				u = self.update_neumann(u, u_received, direction)
				
		return u
				
	def update_dirichlet(self, u, u_received, direction):
		# Interpolate to avoid mismatched matrices. Not the best practice but the current layout
		# provides no information about where on the walls adjacent rooms are located
		u_received_interpolated = interpolate_boundary(u_received, u.shape[0])
		if direction == "right":
			u[:, -1] = u_received_interpolated
		elif direction == "left":
			u[:, 0] = u_received_interpolated
		return u
			
	def update_neumann(self, u, u_received, direction):
		# Interpolate to avoid mismatched matrices. Not the best practice but the current layout
		# provides no information about where on the walls adjacent rooms are located
		u_received_interpolated = interpolate_boundary(u_received, u.shape[0])
		
		if direction == "right":
			g = (u[:, -2] - u[:, -1]) / self.dx  # Compute Neumann condition using first-order differences
			u[:, -1] = u[:, -2] + self.dx * (u_received_interpolated - g)
		elif direction == "left":
			g = (u[:, 1] - u[:, 0]) / self.dx  # Compute Neumann condition using first-order differences
			u[:, 0] = u[:, 1] - self.dx * (g - u_received_interpolated)
		return u
