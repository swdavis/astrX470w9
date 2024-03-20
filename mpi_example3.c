#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10000000

int main(int argc, char *argv[]) {
  
  int numtasks, my_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
  int *npart = (int *)malloc(numtasks*sizeof(int));
  
  // Determine the part of the work done by each rank
  for (int i = 0; i < numtasks; i++) {
    npart[i] = N / numtasks;
  }
  // Add any extra if not precisely divisible
  for (int i = 0; i < N % numtasks; i++) {
    npart[i] += 1;
  }
  
  int my_beg = 0;
  for (int i = 0; i < my_rank; i++) {
    my_beg += npart[i];
  }
  int my_end = my_beg + npart[my_rank];
  
  // Initialize array a
  double *a = (double *)calloc(npart[my_rank], sizeof(double));
  double start_time = MPI_Wtime();
  for (int i = 0; i < npart[my_rank]; i++) {
    double n = (double) (i+my_beg);
    a[i] = 2. * n;
  }
  double end_time = MPI_Wtime();
  if (my_rank == 0) {
    printf("Time to initialize a: %f\n", end_time-start_time);
  }
  
  // Initialize array b
  double *b = (double *)calloc(npart[my_rank], sizeof(double));
  start_time = MPI_Wtime();
  for (int i = 0; i < npart[my_rank]; i++) {
    double n = (double) (i+my_beg);
    b[i] = n * n;
  }
  end_time = MPI_Wtime();
  if (my_rank == 0) {
    printf("Time to initialize b: %f\n", end_time-start_time);
  }
  
  // Add arrays a and b into array c
  double *c = (double *)calloc(npart[my_rank], sizeof(double));
  start_time = MPI_Wtime();
  for (int i = 0; i < npart[my_rank]; i++) {
    c[i] = a[i] + b[i];
  }
  end_time = MPI_Wtime();
  if (my_rank == 0) {
    printf("Time to add arrays: %f\n", end_time-start_time);
  }
  
  // Average the result
  start_time = MPI_Wtime();
  double sum = 0;
  for (int i = 0; i < npart[my_rank]; i++) {
    sum += c[i];
  }
  double global_sum;
  MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double average = global_sum / N;
  end_time = MPI_Wtime();
  if (my_rank == 0) {
    printf("Average: %f\n", average);
    printf("Time to compute average: %f\n", end_time-start_time);
  }

  // Free up memory
  free(a);
  free(b);
  free(c);
  free(npart);
  MPI_Finalize();
  return 0;
}
