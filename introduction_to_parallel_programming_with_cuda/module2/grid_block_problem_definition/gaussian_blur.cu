/**
 * CAUTION: Generated with google gemini for learning purposes
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1920
#define HEIGHT 1080
#define DURATION 1800 // 1 minute of video at 30 frames per second
#define DATA_SIZE (WIDTH * HEIGHT * DURATION * sizeof(float))
#define NUM_THREADS_BLOCK_X_Y 16
#define NUM_BLOCKS_X (WIDTH / NUM_THREADS_BLOCK_X_Y)
#define NUM_BLOCKS_Y (HEIGHT / NUM_THREADS_BLOCK_X_Y)

// Definition der Maske als const __device__ Variable
__device__ const float SIMPLE_3D_MASK[] = {
    1, 2, 1,
    1, 2, 1,
    1, 2, 1,

    1, 2, 1,
    1, 4, 1,
    1, 2, 1,

    1, 2, 1,
    1, 2, 1,
    1, 2, 1
};
#define MASK_WEIGHT_SUM 38

__device__ void threeDSlice(const float *input3dData, dim3 sliceDimensions, float *input3dDataSlice, int x_center, int y_center, int z_center, int width, int height, int depth) {
    int slice_x, slice_y, slice_z;
    int global_x, global_y, global_z;

    for (int dz = 0; dz < sliceDimensions.z; ++dz) {
        for (int dy = 0; dy < sliceDimensions.y; ++dy) {
            for (int dx = 0; dx < sliceDimensions.x; ++dx) {
                slice_x = dx;
                slice_y = dy;
                slice_z = dz;

                global_x = x_center + (slice_x - sliceDimensions.x / 2);
                global_y = y_center + (slice_y - sliceDimensions.y / 2);
                global_z = z_center + (slice_z - sliceDimensions.z / 2);

                // Clamp boundary conditions (optional, but often needed)
                int slice_index = dz * sliceDimensions.y * sliceDimensions.x + dy * sliceDimensions.x + dx;
                if (global_x >= 0 && global_x < width &&
                    global_y >= 0 && global_y < height &&
                    global_z >= 0 && global_z < depth) {
                    int global_index = global_z * width * height + global_y * width + global_x;
                    input3dDataSlice[slice_index] = input3dData[global_index];
                } else {
                    input3dDataSlice[slice_index] = 0.0f; // Or handle boundary differently
                }
            }
        }
    }
}

__device__ float threeDGaussianBlurPixel(const float *input3dDataSlice, dim3 sliceDimensions) {
    float pixelValueSum = 0.0f;
    for (int z = 0; z < sliceDimensions.z; ++z) {
        for (int y = 0; y < sliceDimensions.y; ++y) {
            for (int x = 0; x < sliceDimensions.x; ++x) {
                int index = z * sliceDimensions.y * sliceDimensions.x + y * sliceDimensions.x + x;
                pixelValueSum += input3dDataSlice[index] * SIMPLE_3D_MASK[index];
            }
        }
    }
    return pixelValueSum / MASK_WEIGHT_SUM;
}

__global__ void multidimension_blur_kernel(const float *input3DData, float *output3DData, int width, int height, int duration) {
    dim3 sliceDimensions(3, 3, 3);
    __shared__ float input3dDataSliceShared[3 * 3 * 3]; // Shared memory for the slice

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x < width && y < height && z < duration) {
        // Calculate the global index for the current output pixel
        int output_index = z * width * height + y * width + x;

        // Extract the 3D slice around the current pixel
        threeDSlice(input3DData, sliceDimensions, input3dDataSliceShared, x, y, z, width, height, duration);

        // Apply the 3D Gaussian blur
        float blurredPixelValue = threeDGaussianBlurPixel(input3dDataSliceShared, sliceDimensions);

        // Write the blurred value to the output
        output3DData[output_index] = blurredPixelValue;
    }
}

// Dummy function for loading video data (replace with your actual loading logic)
void loadVideoData(float (*data)[HEIGHT][DURATION]) {
    printf("Loading video data...\n");
    for (int z = 0; z < DURATION; ++z) {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                data[x][y][z] = (float)(x + y + z); // Example data
            }
        }
    }
}

// Dummy function for storing video data (replace with your actual storing logic)
void storeVideoData(float (*data)[HEIGHT][DURATION]) {
    printf("Storing video data...\n");
    // You can add code here to save the output data
}

int main(int argc, char *argv[]) {
    float (*inputHostVideoData)[HEIGHT][DURATION] = (float (*)[HEIGHT][DURATION])malloc(DATA_SIZE);
    float (*outputHostVideoData)[HEIGHT][DURATION] = (float (*)[HEIGHT][DURATION])malloc(DATA_SIZE);

    if (!inputHostVideoData || !outputHostVideoData) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    loadVideoData(inputHostVideoData);

    float *inputDeviceVideoData, *outputDeviceVideoData;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void **)&inputDeviceVideoData, DATA_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for inputDeviceVideoData: %s\n", cudaGetErrorString(cudaStatus));
        free(inputHostVideoData);
        free(outputHostVideoData);
        return 1;
    }

    cudaStatus = cudaMalloc((void **)&outputDeviceVideoData, DATA_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for outputDeviceVideoData: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(inputDeviceVideoData);
        free(inputHostVideoData);
        free(outputHostVideoData);
        return 1;
    }

    cudaStatus = cudaMemcpy(inputDeviceVideoData, inputHostVideoData, DATA_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed to copy input data to device: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(inputDeviceVideoData);
        cudaFree(outputDeviceVideoData);
        free(inputHostVideoData);
        free(outputHostVideoData);
        return 1;
    }

    dim3 blockXY(NUM_THREADS_BLOCK_X_Y, NUM_THREADS_BLOCK_X_Y, 1); // Ensure 3D block dimension
    dim3 gridXYZ(NUM_BLOCKS_X, NUM_BLOCKS_Y, DURATION);

    multidimension_blur_kernel<<<gridXYZ, blockXY>>>(inputDeviceVideoData, outputDeviceVideoData, WIDTH, HEIGHT, DURATION);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(inputDeviceVideoData);
        cudaFree(outputDeviceVideoData);
        free(inputHostVideoData);
        free(outputHostVideoData);
        return 1;
    }

    cudaStatus = cudaMemcpy(outputHostVideoData, outputDeviceVideoData, DATA_SIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed to copy output data to host: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(inputDeviceVideoData);
        cudaFree(outputDeviceVideoData);
        free(inputHostVideoData);
        free(outputHostVideoData);
        return 1;
    }

    storeVideoData(outputHostVideoData);

    cudaFree(inputDeviceVideoData);
    cudaFree(outputDeviceVideoData);
    free(inputHostVideoData);
    free(outputHostVideoData);

    return 0;
}