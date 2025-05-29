#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <tuple>

using namespace std;

/**
 * Minimal example to apply sigmoid activation on a tensor 
 * using cuDNN.
 * Based on the GitHub Gist at https://gist.github.com/rohitdwivedula/127fbea8eab55a0cfe9d72d99fd3969f
 * Which is based on the Medium article at 
 **/

__host__ cudnnHandle_t createCudaHandleAndOutputHWSpecs()
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); // use GPU0
    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;
    return handle_;
}

__host__ std::tuple<cudnnTensorDescriptor_t, float *, int> createTensorAndDescriptor()
{
    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 1; // number of input images
    int c = 1; // number of feature maps (layers of the filter-kernel)
    int h = 1; // height
    int w = 10; // width
    int NUM_ELEMENTS = n * c * h * w;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w); // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensor4ddescriptor

    // create the tensor
    float *x;
    cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(float)); // float since the cudnnDataType_t is FLOAT - managed memory abstracts the cudaMalloc and cudaMemcpy for the user - data are available on CPU as well as on GPU
    // initialize values on CPU and GPU
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        x[i] = i * 1.00f;
    }

    return {x_desc, x, NUM_ELEMENTS}; // tensor descriptor, data associated with the tensor, number of elements in the tensor
}

__host__ float *runCuDnnActivationForward(cudnnHandle_t handle_, cudnnTensorDescriptor_t x_desc, float *x)
{
    // create activation function descriptor
    /// REMARK: an activation filter consists of output=alpha⋅activation(input)+beta⋅output_previous - here we set beta=0 and alpha=1 such that we have output = activation(input)
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    // propagate data from x* (with the data layout described by x_desc) through the activation function
    cudnnActivationForward( // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnactivationforward
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x); // this is where the output is stored - we do an inplace calculation. The result is where the input was before the application of the function

    return x;
}

__host__ void printFloatArray(float *x, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
        std::cout << x[i] << " ";
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    cudnnHandle_t handle_ = createCudaHandleAndOutputHWSpecs(); // create cudnn handle
    auto [x_desc, x, numElements] = createTensorAndDescriptor();
    std::cout << "Original array: ";
    printFloatArray(x, numElements);

    runCuDnnActivationForward(handle_, x_desc, x);

    cudnnDestroy(handle_); // cleanup the cudnn handle
    std::cout << std::endl << "Destroyed cuDNN handle." << std::endl;

    std::cout << "After array: ";
    printFloatArray(x, numElements);

    cudaFree(x); // free GPU memory used by this program
    return 0;
}