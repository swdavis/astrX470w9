import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def rhs(x):
    """
    Function for the right hand side of Poisson equation
    """
    return np.sin(x)

def error(phi1, phi2):
    """
    Normalized relative change in phi between two iterations
    """
    TINY = 1e-10 # avoids potential NaN
    dphi = np.abs((phi2-phi1)[1:-1]/(0.5*np.abs(phi1+phi2)+TINY)[1:-1])
    return dphi.sum()

def jacobi(phi, x, rho):
    """
    Jacobi iteration update of poisson's equation
    """
    dx = 0.5*(x[2:]-x[:-2])
    phi[1:-1] = 0.5*(phi[:-2] + phi[2:] - dx*dx*rho[1:-1])
    return phi

def main():
    """
    Solve Poisson's equation using relaxation
    """

    ## initialize for MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    nprocs = comm.Get_size()
    # get start time for speed comparison
    start_time = MPI.Wtime()
    
    # set parameters for iteration
    eps = 1e-7
    itermax = 1000000
    # initialize grid and rhs on this process
    x0 = 0
    x1 = 1
    nx = 82
    
    # The following code works with one or multiple processes
    xg = np.linspace(x0,x1,nx)
    phig = np.zeros(xg.size)
    nxl = (nx-2) // nprocs
    x = xg[my_rank*nxl:(my_rank+1)*nxl+2]
    phi = np.zeros(x.size)
    rho = rhs(x)

    err = 10*eps # ensure while executes at least once
    iterations = 0
    while err > eps:
        iterations += 1
        phinew = phi.copy()
        phinew = jacobi(phinew, x, rho)

        # CHANGE: Add code using MPI Send and Recv to pass the
        # the buffers with the neighboring points on the other process
        if nprocs > 1:
            tag1 = 1
            tag2 = 2
            # blocking sends
            if my_rank != 0:
                comm.Send([phinew[1], MPI.FLOAT], dest=my_rank-1, tag=tag2)
            if my_rank != nprocs-1: 
                comm.Send([phinew[-2], MPI.FLOAT], dest=my_rank+1, tag=tag1)
            # blocking receives
            rbuf1 = np.empty(1, dtype=float)
            rbuf2 = np.empty(1, dtype=float)
            if my_rank != 0:
                comm.Recv([rbuf1, MPI.FLOAT], source=my_rank-1, tag=tag1)
                phinew[0] = rbuf1[0]
            if my_rank != nprocs-1:
                comm.Recv([rbuf2, MPI.FLOAT], source=my_rank+1, tag=tag2)
                phinew[-1] = rbuf2[0]

        
        err = error(phi, phinew) # local error
        if nprocs > 1:
            # CHANGE: use MPI Allreduce to compute global error
            # this should use the MPI.SUM operation
            gerr = np.zeros(1)
            comm.Allreduce(np.array([err]), gerr, op=MPI.SUM)
            err = gerr[0]
        
        phi = phinew

        if iterations == itermax:
            break

    if my_rank == 0:
        print("Total iteration: {0} {1:.3e}".format(iterations, err))

    if nprocs > 1:
        # Use gather to set the global array on the root grid for output
        xg[1:-1] = np.array(comm.gather(x[1:-1], root=0)).flatten()
        phig[1:-1] = np.array(comm.gather(phi[1:-1], root=0)).flatten()
    else:
        xg = x
        phig = phi

    # print total time
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Time with {:d} processors: {:e}".format(nprocs, end_time-start_time))
    if my_rank == 0:
        plt.plot(xg, phig, '.',label="numerical")
        plt.plot(xg,-np.sin(xg)+xg*np.sin(1), label="analytic")
        plt.legend(loc="best")
        plt.savefig("jacobi.png")

if __name__ == "__main__":
    main()
