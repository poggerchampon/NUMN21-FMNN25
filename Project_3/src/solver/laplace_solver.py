import numpy as np
from mpi4py import MPI

from src.solver.laplace_solver import solve_laplace
from src.utilities.constants import u_normal, u_heater, u_window

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
		u_dict = {i: room.temperature for i, room in enumerate(self.rooms)}
		
		for k in range(self.num_iter):
			for i, room in enumerate(self.rooms):
				if self.rank == i:
					u = u_dict[i]
					adjacent_rooms = room.adjacent_rooms
					self.send_conditions(u, adjacent_rooms)
					u = self.receive_and_update_conditions(u, adjacent_rooms)
					
					u_new = solve_laplace(u, self.dx)
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
		# u_received[0] is just because of potentially mismatched arrays
		# boundary is always constant
		if direction == "right":
			u[:, -1] = u_received[0] 
		elif direction == "left":
			u[:, 0] = u_received[0]
			
		return u
			
	def update_neumann(self, u, u_received, direction):
		if direction == "right":
			flux = (u[:, -2] - u[:, -1]) * self.dx
			u[:, -1] = u[:, -2] - self.dx * flux
		elif direction == "left":
			flux = (u[:, 1] - u[:, 0]) * self.dx
			u[:, 0] = u[:, 1] - self.dx * flux
			
		return u
