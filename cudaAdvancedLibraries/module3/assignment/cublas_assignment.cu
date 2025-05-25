#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#include <iostream>

#define HA 2
#define WA 9
#define WB 2
#define HB WA 
#define WC WB   
#define HC HA  
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P,int uWP,int uHP){
  int i;
  int j;
  for(i=0;i<uHP;i++){
      printf("\n");
      for(j=0;j<uWP;j++) {
        printf("%f ",P[index(i,j,uHP)]);
      }
  }
}

/*
/// REMARK: We could have used this to encapsulate the cudaMalloc within initializeDeviceMemoryFromHostMemory
__host__ float* initializeHostMemory(int height, int width, bool random, float nonRandomValue) {
  // TODO allocate host memory of type float of size height * width called hostMatrix

  // TODO fill hostMatrix with either random data (if random is true) else set each value to nonRandomValue

  return hostMatrix;
}
*/

__host__ float *initializeDeviceMemoryFromHostMemory(int height, int width, float *hostMatrix) {
  // TODO allocate device memory of type float of size height * width called deviceMatrix
  float* d_retPtr;
  cudaError_t err = cudaMalloc((void **)&d_retPtr, height*width*sizeof(float));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector d_r (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  // TODO set deviceMatrix to values from hostMatrix
  cudaError_t errCpy = cudaMemcpy(d_retPtr, hostMatrix, height*width*sizeof(float), cudaMemcpyHostToDevice);
  if (errCpy != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector r from host to device (error code %s)!\n", cudaGetErrorString(errCpy));
    exit(EXIT_FAILURE);
  }
  return d_retPtr;
}

__host__ void retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory) {
  // TODO get matrix values from deviceMatrix and place results in hostMemory
  cudaError_t err = cudaMemcpy(hostMemory, deviceMatrix, height*width*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector r from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return;
}

__host__ void printMatrices(float *A, float *B, float *C){
  printf("\nMatrix A:\n");
  printMat(A,WA,HA);
  printf("\n");
  printf("\nMatrix B:\n");
  printMat(B,WB,HB);
  printf("\n");
  printf("\nMatrix C:\n");
  printMat(C,WC,HC);
  printf("\n");
}

__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
  free( A );  free( B );  free ( C );
  cublasStatus status = cublasFree(AA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(BB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(CC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int  main (int argc, char** argv) {
  cublasStatus status;
  cublasInit();

  // TODO initialize matrices A and B (2d arrays) of floats of size based on the HA/WA and HB/WB to be filled with random data
  /// allocate memory
  float *A = (float*)malloc(HA*WA*sizeof(float));
  float *B = (float*)malloc(HB*WB*sizeof(float));
  // fill with random data
  for (int i = 0; i < HA*WA; i++) {
    A[i] = (float)rand();
  }
  for (int j = 0; j < HB*WB; j++) {
    B[j] = (float)rand();
  }
  if( A == 0 || B == 0){
    return EXIT_FAILURE;
  } else {
    // TODO create arrays of floats C filled with random value
    float *C = (float*)malloc(HC*WC*sizeof(float));
    for (int i = 0; i < HC*WC; i++) {
      C[i] = (float)rand();
    }
    // TODO create arrays of floats alpha filled with 1's
    float *alpha = (float*)malloc(HA*WA*sizeof(float));
    float fill_value = 1.0f; // Der Wert, mit dem wir füllen wollen
    std::fill(alpha, alpha + HA*WA, fill_value);
    // TODO create arrays of floats beta filled with 0's
    float *beta = (float*)malloc(HB*WB*sizeof(float));
    memset(beta, 0, HB*WB * sizeof(float));
    /// allocate memory on device (GPU) and copy data from host (CPU) to device (GPU) to be able to do calculations on it on GPU
    // TODO use initializeDeviceMemoryFromHostMemory to create AA from matrix A
    float* AA = initializeDeviceMemoryFromHostMemory(HA, WA, A);
    // TODO use initializeDeviceMemoryFromHostMemory to create BB from matrix B
    float* BB = initializeDeviceMemoryFromHostMemory(HB, WB, B);
    // TODO use initializeDeviceMemoryFromHostMemory to create CC from matrix C
    float* CC = initializeDeviceMemoryFromHostMemory(HC, WC, C);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // TODO perform Single-Precision Matrix to Matrix Multiplication, GEMM, on AA and BB and place results in CC
    cublasSgemm('n','n',HA,WB,WA,1,AA,HA,BB,HB,0,CC,HC); // Sgemm - Single precision floating point numbers matrix-matrix multiplication level 3

    retrieveDeviceMemory(HC, WC, CC, C);

    printMatrices(A, B, C);

    freeMatrices(A, B, C, AA, BB, CC);
    
    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

}
