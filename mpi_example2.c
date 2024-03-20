#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  
  int numtasks, my_rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  printf("Hello world! My rank: %d\n", my_rank);

  // Ensure that all processes complete printing before proceeding
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (my_rank == 0) {
    printf("Number of processes: %d\n", numtasks);
  }
  
  MPI_Finalize();
  return 0;
}
