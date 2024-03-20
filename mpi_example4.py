from mpi4py import MPI
import numpy as np

def main():
    
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    nprocs = comm.Get_size()

    prev = my_rank - 1
    next = my_rank + 1

    if my_rank == 0:
        prev = nprocs - 1
    if my_rank == nprocs - 1:
        next = 0

    rbuf1 = np.empty(1, dtype=int)
    rbuf2 = np.empty(1, dtype=int)
    sbuf = np.array([my_rank], dtype=int)
    
    tag1 = 1
    tag2 = 2

    # blocking sends
    comm.Send([sbuf, MPI.INT], dest=prev, tag=tag2)
    comm.Send([sbuf, MPI.INT], dest=next, tag=tag1)
    # blocking receives
    comm.Recv([rbuf1, MPI.INT], source=prev, tag=tag1)
    comm.Recv([rbuf2, MPI.INT], source=next, tag=tag2)

    print("My rank: {:d} and my neighbors: {:d} {:d}".format(my_rank, rbuf1[0], rbuf2[0]))
    
if __name__ == "__main__":
    main()

