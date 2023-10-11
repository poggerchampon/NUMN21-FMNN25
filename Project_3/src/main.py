from mpi4py import MPI
import numpy as np

from solver.dirichlet_neumann_solver import DirichletNeumannSolver
from utilities.plot_temperature import plot_temperature
from models.load_apartment import load_apartment_layout
		
def output_results(u_dic, rank):
	room_number = rank + 1
	plot_temperature(u_dic[rank], room_number)

def main():
	# Initialise MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	
	# Load apartment layout with conditions set
	rooms = load_apartment_layout(filename="models/apartment.json")
	
	# Solve using DirichletNeumann method
	solver = DirichletNeumannSolver(rooms, comm, rank)
	u_new = solver.solve()
	
	# Plot temperature distribution
	output_results(u_new, rank)
		
	# Terminate MPI
	MPI.Finalize()
	
if __name__ == "__main__":
	main()
	
"""
Running the program:
----------------------
Navigate to dir containing main.py and run:
	
	mpirun -np 3 python3 -B main.py

If module error, run:
	PYTHONPATH="/path/to/Project_3" mpirun -np 3 python3 -B main.py
"""
