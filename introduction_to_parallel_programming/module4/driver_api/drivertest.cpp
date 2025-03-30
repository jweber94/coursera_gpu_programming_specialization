/*
 * drivertest.cpp
 * Vector addition (host code)
 *
 * Andrei de A. Formiga, 2012-06-04
 */

/**
 * This is cuda drive code - this means we have deeper acess to GPU functionallity.
 * This is common for GPU accellerated libraries. Here we have a minimal library that takes care of the copy to and from the
 * GPU as well as how to apply an kernel to a vector based input data.
 * 
 * Commonly, libraries or basic functionallity like this are stored as .fatbin or .ptx to be able to port them
 * between nvidia GPUs.
 * 
 * This files only contains cuda runtime API calls (e.g. cuMemAlloc) but no real cuda accellerated parallel code. Therefore you
 * can compile this as a normal .cpp file and link against libcuda.so. As soon as you use kernel function directives (e.g. __global__ or __device__)
 * or you use a cuda library (e.g. cufft, cudnn, ...), you need to use .cu as the file ending.
 */

 #include <stdio.h>
 #include <stdlib.h>
 
 #include <cuda.h>
 #include <builtin_types.h>
 
 #define N 100
 
 // This will output the proper CUDA error strings
 // in the event that a CUDA host call returns an error
 #define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
 
 inline void __checkCudaErrors( CUresult err, const char *file, const int line )
 {
     if( CUDA_SUCCESS != err) {
         fprintf(stderr,
                 "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                 err, file, line );
         exit(-1);
     }
 }
 
 // --- global variables ----------------------------------------------------
 CUdevice   device;
 CUcontext  context;
 CUmodule   module;
 CUfunction function;
 size_t     totalGlobalMem;
 
 //char       *module_file = (char*) "matSumKernel.fatbin";
 char       *module_file = (char*) "matSumKernel.ptx";
 char       *kernel_name = (char*) "matSum";
 
 
 // --- functions -----------------------------------------------------------
 void initCUDA()
 {
     int deviceCount = 0;
     CUresult err = cuInit(0);
     int major = 0, minor = 0;
 
     if (err == CUDA_SUCCESS)
         checkCudaErrors(cuDeviceGetCount(&deviceCount));
 
     if (deviceCount == 0) {
         fprintf(stderr, "Error: no devices supporting CUDA\n");
         exit(-1);
     }
 
     // get first CUDA device
     checkCudaErrors(cuDeviceGet(&device, 0));
     char name[100];
     cuDeviceGetName(name, 100, device);
     printf("> Using device 0: %s\n", name);
 
     // get compute capabilities and the devicename
     checkCudaErrors( cuDeviceComputeCapability(&major, &minor, device) );
     printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
 
     checkCudaErrors( cuDeviceTotalMem(&totalGlobalMem, device) );
     printf("  Total amount of global memory:   %llu bytes\n",
            (unsigned long long)totalGlobalMem);
     printf("  64-bit Memory Address:           %s\n",
            (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
            "YES" : "NO");
 
     err = cuCtxCreate(&context, 0, device);
     if (err != CUDA_SUCCESS) {
         fprintf(stderr, "* Error initializing the CUDA context.\n");
         cuCtxDetach(context);
         exit(-1);
     }
 
     err = cuModuleLoad(&module, module_file);
     if (err != CUDA_SUCCESS) {
         fprintf(stderr, "* Error loading the module %s\n", module_file);
         cuCtxDetach(context);
         exit(-1);
     }
 
     err = cuModuleGetFunction(&function, module, kernel_name);
 
     if (err != CUDA_SUCCESS) {
         fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
         cuCtxDetach(context);
         exit(-1);
     }
 }
 
 void finalizeCUDA()
 {
     cuCtxDetach(context);
 }
 
 void setupDeviceMemory(CUdeviceptr *d_a, CUdeviceptr *d_b, CUdeviceptr *d_c)
 {
     checkCudaErrors( cuMemAlloc(d_a, sizeof(int) * N) );
     checkCudaErrors( cuMemAlloc(d_b, sizeof(int) * N) );
     checkCudaErrors( cuMemAlloc(d_c, sizeof(int) * N) );
 }
 
 void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
 {
     checkCudaErrors( cuMemFree(d_a) );
     checkCudaErrors( cuMemFree(d_b) );
     checkCudaErrors( cuMemFree(d_c) );
 }
 
 void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
 {
     void *args[3] = { &d_a, &d_b, &d_c };
 
     // grid for kernel: <<<N, 1>>>
     checkCudaErrors( cuLaunchKernel(function, N, 1, 1,  // Nx1x1 blocks
                                     1, 1, 1,            // 1x1x1 threads
                                     0, 0, args, 0) );
 }
 
 int main(int argc, char **argv)
 {
     int a[N], b[N], c[N];
     CUdeviceptr d_a, d_b, d_c; // handles to the cuda device adresses
 
     // initialize host arrays
     for (int i = 0; i < N; ++i) {
         a[i] = N - i;
         b[i] = i * i;
     }
 
     // initialize
     printf("- Initializing...\n");
     initCUDA();
 
     // allocate memory on GPU and save the resulting starting adresses in the d_a, d_b and d_c variables
     setupDeviceMemory(&d_a, &d_b, &d_c); // this is a wrapper around native cuda calls
 
     // copy arrays to device - this is the copying from the CPU memory to the GPU
     checkCudaErrors( cuMemcpyHtoD(d_a, a, sizeof(int) * N) ); // these are native cuda calls
     checkCudaErrors( cuMemcpyHtoD(d_b, b, sizeof(int) * N) );
 
     // run
     printf("# Running the kernel...\n");
     runKernel(d_a, d_b, d_c); // cuda as well as opencl used the concept of calcuation kernels that are applied to the data
     printf("# Kernel complete.\n");
 
     // copy results to host and report - back copy from GPU to CPU
     checkCudaErrors( cuMemcpyDtoH(c, d_c, sizeof(int) * N) ); // this is a native cuda call
     for (int i = 0; i < N; ++i) {
         if (c[i] != a[i] + b[i])
             printf("* Error at array position %d: Expected %d, Got %d\n",
                    i, a[i]+b[i], c[i]);
     }
     printf("*** All checks complete.\n");
 
 
     // finish
     printf("- Finalizing...\n");
     releaseDeviceMemory(d_a, d_b, d_c); // no memory leak
     finalizeCUDA();
     return 0;
 }