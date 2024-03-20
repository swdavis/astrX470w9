#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  
  int numtasks, my_rank, prev, next;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  prev = my_rank - 1;
  next = my_rank + 1;
  
  if (my_rank == 0) {
    prev = numtasks - 1;
  }
  if (my_rank == numtasks - 1) {
    next = 0;
  }

  int sbuf = my_rank;
  int rbuf1, rbuf2;
  
  MPI_Request reqs[4];

  // Non-blocking receives
  MPI_Irecv(&rbuf1, 1, MPI_INT, prev, 1, MPI_COMM_WORLD, &reqs[0]);
  MPI_Irecv(&rbuf2, 1, MPI_INT, next, 2, MPI_COMM_WORLD, &reqs[1]);
  // Non-blocking sends
  MPI_Isend(&sbuf, 1, MPI_INT, prev, 2, MPI_COMM_WORLD, &reqs[2]);
  MPI_Isend(&sbuf, 1, MPI_INT, next, 1, MPI_COMM_WORLD, &reqs[3]);

  // Can do some work here if needed
  // For example, print current state of the receive buffers
  printf("%d %d\n", rbuf1, rbuf2);
  
  // Wait for all non-blocking operations to complete
  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
  
  printf("My rank: %d and my neighbors: %d %d\n", my_rank, rbuf1, rbuf2);
  
  MPI_Finalize();
  return 0;
}
