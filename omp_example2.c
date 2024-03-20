#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  
  int numtasks, my_rank;

  // Initialize OpenMP
  #pragma omp parallel
  {
    #pragma omp single
    {
      numtasks = omp_get_num_threads();
      my_rank = omp_get_thread_num();
    }
    printf("Hello world! My rank: %d\n", my_rank);
  }

  // Ensure that all threads complete printing before proceeding
  #pragma omp barrier

  // Only one thread (rank 0) prints the number of threads
  #pragma omp single
  {
    printf("Number of processes: %d\n", numtasks);
  }

  return 0;
}
