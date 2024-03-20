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
    reqs = [None] * 4

    # non-blocking receives
    reqs[0] = comm.Irecv([rbuf1, MPI.INT], source=prev, tag=tag1)
    reqs[1] = comm.Irecv([rbuf2, MPI.INT], source=next, tag=tag2)
    # non-blocking sends
    reqs[2] = comm.Isend([sbuf, MPI.INT], dest=prev, tag=tag2)
    reqs[3] = comm.Isend([sbuf, MPI.INT], dest=next, tag=tag1)

    # can do some work here if you want, e.g. print current
    # state of the receive buffers
    print(rbuf1[0], rbuf2[0])
    
    MPI.Request.Waitall(reqs)
    print("My rank: {:d} and my neighbors: {:d} {:d}".format(my_rank, rbuf1[0], rbuf2[0]))
    
if __name__ == "__main__":
    main()

