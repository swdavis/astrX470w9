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
    Solve Poisson's equation using jacobi relaxation with
    non-blocking communication
    """

    ## initialize for MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    nprocs = comm.Get_size()
    req = [None] * 4 # initialize req array
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
        
        # Add code using MPI Isend and Irecv to pass the
        # the buffers with the neighboring points on the other process
        # using non-blocking communication

        if nprocs > 1:
            # Initiate the non-blocking send and receive *before*
            # running the jacobi function i.e. start communication
            # before the work is done. We can do this because only
            # the end points need to be passed so lets compute them
            # seperately
            tag1 = 1
            tag2 = 2
            dx = 0.5*(x[2]-x[0])
            phis = 0.5*(phinew[0]+phinew[2]-dx**2*rho[1])
            sbuf1 = np.array([phis], dtype=float)
            dx = 0.5*(x[-1]-x[-3])
            phie = 0.5*(phinew[-3]+phinew[-1]-dx**2*rho[-2])
            sbuf2 = np.array([phie], dtype=float)

            # CHANGE: add non-blocking sends here
            if my_rank != 0:
                req[0] = comm.Isend([sbuf1, MPI.FLOAT], dest=my_rank-1, tag=tag2)
            if my_rank != nprocs-1: 
                req[1] = comm.Isend([sbuf2, MPI.FLOAT], dest=my_rank+1, tag=tag1)
            rbuf1 = np.empty(1, dtype=float)
            rbuf2 = np.empty(1, dtype=float)
            # CHANGE: add non-blocking receives here
            if my_rank != 0:
                req[2] = comm.Irecv([rbuf1, MPI.FLOAT], source=my_rank-1, tag=tag1)
            if my_rank != nprocs-1:
                req[3] = comm.Irecv([rbuf2, MPI.FLOAT], source=my_rank+1, tag=tag2)
        phinew = jacobi(phinew, x, rho)
        
        if nprocs > 1:
            # CHANGE: use MPI.Request.Wait to confirm both receives
            # have terminated, then set phinew[0] and phinew[-1]
            # as needed
            if my_rank != 0:
                MPI.Request.Wait(req[2])
                phinew[0] = rbuf1
            if my_rank != nprocs-1:
                MPI.Request.Wait(req[3])
                phinew[-1] = rbuf2


        err = error(phi, phinew) # local error
        if nprocs > 1:
            # CHANGE: use MPI Allreduce to compute global error
            # accross all processes this should use the MPI.SUM operation
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

    # print total time and plot
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Time with {:d} processors: {:e}".format(nprocs, end_time-start_time))
        plt.plot(xg, phig, '.',label="numerical")
        plt.plot(xg,-np.sin(xg)+xg*np.sin(1), label="analytic")
        plt.legend(loc="best")
        plt.savefig("jacobi_nb.png")

if __name__ == "__main__":
    main()
