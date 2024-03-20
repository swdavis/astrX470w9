#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  
  int numtasks, my_rank, prev, next;
  MPI_Status status;
  
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
  
  // Blocking sends
  MPI_Send(&sbuf, 1, MPI_INT, prev, 2, MPI_COMM_WORLD);
  MPI_Send(&sbuf, 1, MPI_INT, next, 1, MPI_COMM_WORLD);
  
  // Blocking receives
  MPI_Recv(&rbuf1, 1, MPI_INT, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&rbuf2, 1, MPI_INT, next, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
  
  printf("My rank: %d and my neighbors: %d %d\n", my_rank, rbuf1, rbuf2);
  
  MPI_Finalize();
  return 0;
}
