#include "merge_sort.h"
#include <limits.h>
#include <iostream>

#define min(a, b) (a < b ? a : b)
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu

__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char** argv) 
{
    int numElements = 32;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'n':
                    i++;
                    numElements = stoi(argv[i]);
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    return {threadsPerBlock, blocksPerGrid, numElements};
}

__host__ long *generateRandomLongArray(int numElements)
{
    //TODO generate random array of long integers of size numElements
    long *randomLongs;
    randomLongs = static_cast<long*>(malloc(numElements*sizeof(long)));
    for (int i = 0; i < numElements; i++) {
        long tmpRand = static_cast<long>(rand()); // rand() returns a random number from 0 to RAND_MAX, then we scale to the range and add the lower bound to have a valid number in the range
        randomLongs[i] = (tmpRand > RAND_MAX/2)*(-1)*tmpRand + (tmpRand <= RAND_MAX/2)*tmpRand; // make positive as well as negative values by using a simple heuristic
    }
    return randomLongs;
}

__host__ void printHostMemory(long *host_mem, int num_elments)
{
    // Output results
    for(int i = 0; i < num_elments; i++)
    {
        printf("%d ",host_mem[i]);
    }
    printf("\n");
}

__host__ int main(int argc, char** argv) 
{

    auto[threadsPerBlock, blocksPerGrid, numElements] = parseCommandLineArguments(argc, argv);

    long *data = generateRandomLongArray(numElements);

    printf("Unsorted data: ");
    printHostMemory(data, numElements);

    data = mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

    printf("Sorted data: ");
    printHostMemory(data, numElements);
}

__host__ void deallocateMemory(long* dataPtr, long* swpPtr, dim3* threadPtr, dim3* blocksPtr) {
    cudaError_t ret = cudaFree(dataPtr);
    if (ret != cudaSuccess) {
        std::cerr << "Could not free memory for the data on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaError_t ret = cudaFree(swpPtr);
    if (ret != cudaSuccess) {
        std::cerr << "Could not free memory for the swp on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaError_t ret = cudaFree(threadPtr);
    if (ret != cudaSuccess) {
        std::cerr << "Could not free memory for the threads on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaError_t ret = cudaFree(blocksPtr);
    if (ret != cudaSuccess) {
        std::cerr << "Could not free memory for the blocks on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple <long* ,long* ,dim3* ,dim3*> allocateMemory(long* data, int numElements, dim3 threadsPerBlock, dim3 blocksPerGrid)
{
    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data, D_swp;
    dim3* D_threads, D_blocks;
    
    // Actually allocate the two arrays
    cudaError_t ret = cudaMalloc(D_data, numElements*sizeof(long));
    if (ret != cudaSuccess) {
        std::cerr << "Could not allocate memory on the GPU for D_data" << std::endl;
        exit(EXIT_FAILURE);
    }

    ret = cudaMalloc(D_swp, numElements*sizeof(long));
    if (ret != cudaSuccess) {
        std::cerr << "Could not allocate memory on the GPU for D_swp" << std::endl;
        exit(EXIT_FAILURE);
    }

    ret = cudaMalloc(D_threads, sizeof(dim3));
    if (ret != cudaSuccess) {
        std::cerr << "Could not allocate memory on the GPU for D_threads" << std::endl;
        exit(EXIT_FAILURE);
    }

    ret = cudaMalloc(D_blocks, sizeof(dim3));
    if (ret != cudaSuccess) {
        std::cerr << "Could not allocate memory on the GPU for D_blocks" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy from our input list into the first array
    ret = cudaMemcpy(D_data, data, numElements*sizeof(long), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        std::cerr << "Could not copy data from host to D_data on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy the thread / block info to the GPU as well
    ret = cudaMemcpy(D_threads, threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        std::cerr << "Could not copy data from host to D_threads on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }

    ret = cudaMemcpy(D_blocks, blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        std::cerr << "Could not copy data from host to D_blocks on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }
    return {D_data, D_swp, D_threads, D_blocks};
}

__host__ long* mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    auto[D_data, D_swp, D_threads, D_blocks] = allocateMemory(data, size, threadsPerBlock, blocksPerGrid); // on GPU and copy the needed data to it

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    // TODO Initialize timing metrics variable(s). The implementation of this is up to you

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        // Actually call the kernel
        //arguments are long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(D_data, D_swp, size, width, slices, D_threads, D_blocks); //TODO You will need to populate arguments for the kernel

        // Switch the input / output arrays instead of copying them around

    }

    // TODO calculate and print to stdout kernel execution time

    // Free the GPU memory
    deallocateMemory(D_data, D_swp, D_threads, D_blocks);
    // CAUTION: We need to copy the result from the GPU to the data of the input or generate a new array for the result and return it here - we need to copy BEFORE deallocation on the GPU happens
    return data;
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    // TODO initialize 3 long variables start, middle, and end
    // middle and end do not have values set,
    // while start is set to the width of the merge sort data span * the thread index * number of slices that this kernel will sort

    for (long slice = 0; slice < slices; slice++) {
        // Break from loop when the start variable is >= size of the input array

        // Set middle to be minimum middle index (start index plus 1/2 width) and the size of the input array

        // Set end to the minimum of the end index (start index plus the width of the current data window) and the size of the input array
       
        // Perform bottom up merege given the two available arrays and the start, middle, and end variables
        
        // Increase the start index by the width of the current data window
    }
}

//
// Finally, sort something gets called by gpu_mergesort() for each slice
// Note that the pseudocode below is not necessarily 100% complete you may want to review the merge sort algorithm.
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;

    // Create a for look that iterates between the start and end indexes
    for () {
        // if i is before the middle index and (j is the final index or the value at i <  the value at j)
        if () {
            // set the value in the destination array at index k to the value at index i in the source array

            // increment i
        } else {
            // set the value in the destination array at index k to the value at index j in the source array

            // increment k
        }
    }
}