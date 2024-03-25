/****************************************************************************************************
openmp_mmult.c performs square matrix multiplication using OpenMP. The program generates random matrices of the dimension specified by the user and performs multiplication using a simple three-loop algorithm. Parallelism is achieved by dividing the first matrix into a group of rows for each thread for multiplication. Each thread then multiplies their group of rows with the entire second matrix, calculating a group of rows of the resultant matrix. Since the threads calculate different portions of the matrix, there was not any need to use locks or other mechanisms to protect shared memory.

The program takes two arguments: 
(1) The first is the number of threads to be used. If no arguments are provided, the program assumes 4 threads.
(2) The second is the dimension of the matrices to be multiplied. If only the first argument is provided, the program uses 100x100 matrices.

The program prints the time required for multiplication and writes thre three matrices to the files 'A.txt', 'B.txt', and 'C.txt'.

Author: Justin Krometis
Modified: February 13, 2012
****************************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

//Functions to be used ( see definitions below main() )
void mat_mult(double *mat1, double *mat2, double *result, int dim);
void mat_mult_thr(double *mat1, double *mat2, double *result, int dim, int nthr);
void fprint_mat(char flnm[], double *mat, int dim);
void rand_mat(double *mat, int dim);

//main() does the work
int main(int argc, char *argv[] ) {
  double *mat1=NULL, *mat2=NULL, *mat3=NULL; //matrices to be multiplied, and the result
  int nthr; //number of threads
  int dim;  //matrix dimension

  time_t mmult_st=0, prog_st=0, mmult_end=0, prog_end=0;  //timer variables
  double mmult_tm=0, prog_tm=0;                           //timer results

  time(&prog_st);

  //the first argument is the number of threads. if none is provided, assume 4.
  if(argc>1)
    nthr=atoi(argv[1]);  
  else {
    //directive in case a user doesn't know the parameters
    printf("\nProgram parameters are: [number of threads] [matrix dimension].");
    printf("\nNo number of threads provided. Assuming 4.");
    nthr=4;
  }

  //the second argument is the number of rows of the first matrix. if none is provided, use 100.
  if(argc>2)
    dim=atoi(argv[2]);
  else {
    printf("\nNo matrix dimension provided. Using 100-by-100 square matrices.");
    dim=100;
  }

  //allocate memory for the three matrices
  mat1 = (double *)malloc(dim*dim*sizeof(double));
  if(mat1==NULL) printf("\nUnable to allocate memory for matrix 1.\n");
  mat2 = (double *)malloc(dim*dim*sizeof(double));
  if(mat2==NULL) printf("\nUnable to allocate memory for matrix 2.\n");
  mat3 = (double *)malloc(dim*dim*sizeof(double));
  if(mat3==NULL) printf("\nUnable to allocate memory for matrix 3.\n");

  //get the two matrices to be multiplied
  printf("\nGenerating two random matrices...");
  srand( (unsigned int)time(NULL) );
  printf("mat1 (%d x %d)...",dim,dim);
  rand_mat(mat1, dim);
  printf("Done. mat2 (%d x %d)...",dim,dim);
  rand_mat(mat2, dim);
  printf("Done.\n");

  //multiply the matrices (the results come back in mat3)
  time(&mmult_st);
  printf("Multiplying the matrices...");
  
  //if the user specifies 1 thread, just do serial matrix multiplication
  //otherwise, do parallel matrix multiplication
  if (nthr == 1)
    mat_mult(mat1,mat2,mat3,dim);
  else
    mat_mult_thr(mat1,mat2,mat3,dim,nthr);
  
  printf("Done.\n");

  //print stuff (to screen and to file)
  time(&mmult_end);
  mmult_tm = difftime(mmult_end,mmult_st);
  printf("Multiplication completed in %1.2f seconds.\n", mmult_tm);

  //write the three matrices to their respective files
  fprint_mat("A.txt",mat1,dim);
  fprint_mat("B.txt",mat2,dim);
  fprint_mat("C.txt",mat3,dim);
  
  time(&prog_end);
  prog_tm = difftime(prog_end,prog_st);
  printf("Program completed (%d thread(s) with dimension %d) in %1.2f seconds.\nThe three matrices were written to A.txt, B.txt, and C.txt.\n\n", nthr,dim,prog_tm);

  //free all of the memory
  free(mat1);
  free(mat2);
  free(mat3);

  return 0;
} //end main()

//mat_mult() does serial square matrix multiplication
//  mat1 and mat2 are the matrices to be multiplied
//  result is the result matrix
//  dim is the number of rows/columns of each matrix
void mat_mult(double *mat1, double *mat2, double *result, int dim)
{
  int i,j,k; //iterators

  for(i=0; i<dim; i++) //iterate through the rows of the result
    for(j=0; j<dim; j++) //iterate through the columns of the result
    {
      *( result+(j+i*dim) ) = 0; //initialize
      
      //iterate through the inner dimension (columns of first matrix/rows of second matrix)
      for(k=0; k<dim; k++)
        *( result+(j+i*dim) ) += *( mat1+(k+i*dim) )*( *( mat2+(j+k*dim) ));
    }
} //end mat_mult()

//mat_mult_thr() does parallel (threaded) matrix multiplication
//  mat1 and mat2 are the matrices to be multiplied
//  result is the result matrix
//  dim is the number of rows/columns of each matrix
//  nthr is the number of threads
void mat_mult_thr(double *mat1, double *mat2, double *result, int dim, int nthr)
{
  int part_rows, th_id;
  part_rows = dim/nthr;
  
  omp_set_num_threads(nthr); //set the number of threads
  #pragma omp parallel shared(mat1,mat2,result,part_rows) private(th_id)
  {
    int i,j,k; //iterators
    th_id = omp_get_thread_num(); //th_id holds the thread number for each thread

    //Split the first for loop among the threads
    #pragma omp for schedule(guided,part_rows)
    for(i=0; i<dim; i++) //iterate through the rows of the result
    {
      //printf("Thread #%d is doing row %d.\n",th_id,i); //Uncomment this line to see which thread is doing each row
      for(j=0; j<dim; j++) //iterate through the columns of the result
      {
        *( result+(j+i*dim) ) = 0; //initialize
      
        //iterate through the inner dimension (columns of first matrix/rows of second matrix)
        for(k=0; k<dim; k++)
          *( result+(j+i*dim) ) += *( mat1+(k+i*dim) )*( *( mat2+(j+k*dim) ));
      }
    }
  }
} //end mat_mult_thr()

//fprint_mat() prints square matrix 'mat' of dimension 'dim' to the file 'flnm'
void fprint_mat(char flnm[], double *mat, int dim)
{
  int i,j;
  FILE *fl;

  fl = fopen(flnm,"w"); //open the file

  for(i=0; i<dim; i++)
  {
    for(j=0; j<dim; j++)
      fprintf(fl,"%1.4f ",*( mat+(j+i*dim) ));

    fprintf(fl,"\n");
  }

  fclose(fl); //close the file
} //end fprint_mat()

//rand_mat() generates a random square matrix of dimension 'dim'
void rand_mat(double *mat, int dim)
{
  int i,j; //iterators
 
  for(i=0; i<dim; i++) //iterate through the rows
    for(j=0; j<dim; j++) //iterate through the columns
      *(mat+(j+i*dim)) = (10.0*rand())/RAND_MAX;
} //end rand_mat()
