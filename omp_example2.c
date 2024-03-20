#include <stdio.h>
#include <omp.h>

int main() {
  
  omp_set_num_threads(12); // set max threads to 12

  int nthread, tid;
  // start parallel region
  #pragma omp parallel private(tid)
  {
    nthread = omp_get_num_threads();
    tid = omp_get_thread_num();
    printf("Hello world! My thread id: %d\n", tid);
  }
  // Back to serial region so nly one thread  prints
  // the number of threads
  printf("Number of threads: %d\n", nthread);

  return 0;
}
