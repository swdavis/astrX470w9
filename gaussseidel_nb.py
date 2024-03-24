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


def gaussseidel_red(phi, x, rho): 
    dx = 0.5*(x[3::2] - x[1:-2:2])
    phi[2:-1:2] = 0.5*(phi[1:-2:2] + phi[3::2] - dx**2*rho[2:-1:2])
    return phi

def gaussseidel_black(phi, x, rho):
    dx = 0.5*(x[2::2] - x[:-2:2])
    phi[1:-1:2] = 0.5*(phi[:-2:2]  + phi[2::2] - dx**2*rho[1:-1:2])
    return phi

def main():
    """
    Solve Poisson's equation using relaxation
    """

    ## initialize for MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    nprocs = comm.Get_size()
    reqr = [None] * 2
    reqb = [None] * 2
    # get start time for speed comparison
    start_time = MPI.Wtime()
    
    # set parameters for iteration
    eps = 1e-7
    itermax = 10000000
    #itermax = 3
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

    err = 10*eps # ensure while executes
    iterations = 0
    while err > eps:
        iterations += 1
        phinew = phi.copy()

        # CHANGE: Add code using MPI ISend and Recv to pass the
        # the buffer with the neighboring points on the other process
        if nprocs > 1:

            dx = 0.5*(x[-1]-x[-3])
            phie = 0.5*(phinew[-3]+phinew[-1]-dx**2*rho[-2])
            sbuf1 = np.array([phie], dtype=float)
            tag1 = 1
            # non-blocking sends
            if my_rank != nprocs-1: 
                reqr[0] = comm.Isend([sbuf1, MPI.FLOAT], dest=my_rank+1, tag=tag1)
            # non-blocking receives
            rbuf1 = np.empty(1, dtype=float)
            if my_rank != 0:
                reqr[1] = comm.Irecv([rbuf1, MPI.FLOAT], source=my_rank-1, tag=tag1)

        phinew = gaussseidel_red(phinew, x, rho)           

        if nprocs > 1:
            if my_rank != 0:
                MPI.Request.Wait(reqr[1])
                phinew[0] = rbuf1[0]
        # CHANGE: Add code using MPI Send and Recv to pass the
        # the buffer with the neighboring points on the other process
        if nprocs > 1:

            dx = 0.5*(x[2]-x[0])
            phis = 0.5*(phinew[0]+phinew[2]-dx**2*rho[1])
            sbuf2 = np.array([phis], dtype=float)
            tag2 = 2
            # non-blocking sends
            if my_rank != 0:
                reqb[0] = comm.Isend([sbuf2, MPI.FLOAT], dest=my_rank-1, tag=tag2)
            # non-blocking receives
            rbuf2 = np.empty(1, dtype=float)
            if my_rank != nprocs-1:
                reqb[1] = comm.Irecv([rbuf2, MPI.FLOAT], source=my_rank+1, tag=tag2)
        
        phinew = gaussseidel_black(phinew, x, rho)
            
        if nprocs > 1:
            if my_rank != nprocs-1:
                MPI.Request.Wait(reqb[1])
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
        plt.savefig("gaussseidel.png")

if __name__ == "__main__":
    main()
