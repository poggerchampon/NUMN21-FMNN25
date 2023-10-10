from mpi4py import MPI
import numpy as np

from utilities.constants import omega, dx, num_iter, u_normal, u_heater, u_window
from solver.laplace_solver import solve_laplace
from utilities.plot_temperature import plot_temperature

def initialise_room(rank):
	if rank == 1:
		u = np.full((41, 21), u_normal) # init room 2
	else:
		u = np.full((21, 21), u_normal)  # init room 1 and 3
	return u

def set_boundary_conditions(u, rank):
	if rank == 1: # Room 2
		u[0, :] = u_heater # top wall heater
		u[-1, :] = u_window # bottom wall window
	elif rank == 0:
		u[:, 0] = u_heater # Room 1 left wall heater
	elif rank == 2:
		u[:, -1] = u_heater # Room 3 right wall heater
	return u
		
def dirichlet_neumann_iteration(u, comm, rank):
	# Dirichlet-Neumann Iteration
	for k in range(num_iter):
		u = solve_laplace(u)
		
		if rank == 1:
			comm.send(u[:, 0], dest=0, tag=11)
			comm.send(u[:, -1], dest=2, tag=12)
		elif rank == 0:
			u_k1_2 = comm.recv(source=1, tag=11)
			u[:, -1] = u_k1_2[0]  # Update right wall room 1 with Dirichlet conditions from room 2
		else:
			u_k1_2 = comm.recv(source=1, tag=12)
			u[:, 0] = u_k1_2[0]  # Update left wall room 3 with Dirichlet conditions from room 2
			
	return u

def output_results(u, rank):
	room_number = rank + 1
	plot_temperature(u, room_number)

def main():
	# Initialise MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	
	# Initialise the rooms and set the boundary conditions
	u = initialise_room(rank)
	u = set_boundary_conditions(u, rank)
	
	u_new = dirichlet_neumann_iteration(u, comm, rank)
	
	output_results(u_new, rank)
		
	# Terminate MPI
	MPI.Finalize()
	
if __name__ == "__main__":
	main()
