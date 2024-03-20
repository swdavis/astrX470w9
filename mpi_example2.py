# mpi_example1.py
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    my_rank = comm.Get_rank()
    print("Hello world! My rank: ", my_rank)

    # ensure that this executes after each rank has completed
    # printing Hello World
    comm.Barrier()
    if my_rank == 0:
        print("Number of processes: ", nprocs)

if __name__ == "__main__":
    main()
