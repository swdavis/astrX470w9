import numpy as np
from mpi4py import MPI

def main():

    # get the MPI communicator
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    my_rank = comm.Get_rank()
 
    N = 10000000

    # determine the part of the work done by each rank
    npart = [ N // nprocs for i in range(nprocs) ]
    # add any extra if not precisely divisible
    for i in range(N % nprocs):
        npart[i] += 1

    my_beg = 0
    for i in range(my_rank):
        my_beg += npart[i]

    # initialize a
    start_time = MPI.Wtime()
    a = np.empty(npart[my_rank])
    for i in range(npart[my_rank]):
        a[i] = 2*(i + my_beg)
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Time to initialize a: ",end_time-start_time)

    # initialize b
    start_time = MPI.Wtime()
    b = np.empty(npart[my_rank])
    for i in range(npart[my_rank]):
        b[i] = (i + my_beg)**2
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Time to initialize b:  ",end_time-start_time)

    # add the two arrays in array c
    c = np.empty(npart[my_rank])
    start_time = MPI.Wtime()
    for i in range(npart[my_rank]):
        c[i] = a[i] + b[i]
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Time to add arrays: ",end_time-start_time)

    # average the result
    start_time = MPI.Wtime()
    sum = 0
    for i in range(npart[my_rank]):
        sum += c[i]
    local_sum = np.array([sum])
    global_sum = np.zeros(1) 
    comm.Reduce(sum, global_sum, op=MPI.SUM, root=0)

    average = global_sum / N
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Average: ",average[0])
        print("Time to compute average: ",end_time-start_time)

if __name__ == "__main__":
    main()
