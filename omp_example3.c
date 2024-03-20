#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000000

int main() {

  omp_set_num_threads(12);
  double a[N], b[N], c[N];

  int i;
  double n;

  // Initialize array a
  double start_time = omp_get_wtime();
  #pragma omp parallel for private(i,n)
  for (int i = 0; i < N; i++) {
    n = (double)(i);
    a[i] = 2. * n;
  }
  double end_time = omp_get_wtime();
  printf("Time to initialize a: %f\n", end_time - start_time);
  
  // Initialize array b
  start_time = omp_get_wtime();
  #pragma omp parallel for private(i,n)
  for (int i = 0; i < N; i++) {
    n = (double) (i);
    b[i] = n * n;
  }
  end_time = omp_get_wtime();
  printf("Time to initialize b: %f\n", end_time - start_time);
  
  // Add arrays a and b into array c
  start_time = omp_get_wtime();
  #pragma omp parallel for private(i)
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
  end_time = omp_get_wtime();
  printf("Time to add arrays: %f\n", end_time - start_time);
  
  // Average the result
  start_time = omp_get_wtime();
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < N; i++) {
    sum += c[i];
  }
  double average = sum / N;
  end_time = omp_get_wtime();

  printf("Average: %f\n", average);
  printf("Time to compute average: %f\n", end_time - start_time);
  
  return 0;
}
