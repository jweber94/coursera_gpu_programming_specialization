#include "cufft_example.h"

//Based on example found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example

__device__ Complex complexScaleMult(Complex a, Complex b, int scalar) // functions that are only available on GPU (__device__ prefix) can have return values and can be used as normal free functions in C
{
    //TODO Create a variable of type Complex named c - DONE
    Complex c;

    //TODO Calculate the x value for c by scalar * (a.x * b.x) - DONE
    c.x = scalar*(a.x * b.x);

    //TODO Calculate the y value for c by scalar * (a.y * b.y) - DONE
    c.y = scalar*(a.y * b.y);

    return c;
}

__global__ void complexProcess(Complex *a, Complex *b, Complex *c, int size, int scalar) // global must be void since we can only copy data back via cudaMemcpy
{
    // TODO calculate threadId variable - DONE
    int blockId = blockIdx.x + blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z; 
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // TODO process complexScalarMult on values in a and b at index threadID and the passed scalar, place the result in c[threadId] - DONE
    if (threadId < size) {
        c[threadId] = complexScaleMult(a[threadId], b[threadId], scalar);
    }
}

__host__ std::tuple<int, int> parseCommandLineArguments(int argc, char** argv) 
{
    // parse command line input for argument -n and place in variable N
    int N = 16;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'n':
                    N = (unsigned int) strtol(argv[i], 0, 10);
                    i++;
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    // TODO Set variable SIZE equal to N squared - DONE
    int SIZE = N*N;
    return {N, SIZE};
}

__host__ Complex *generateComplexPointer(int SIZE)
{
    Complex *complex = new Complex[SIZE];
    // TODO populate properties x and y of variable complex at index i to 2 and 3 respectively - DONE
    for (int i = 0; i < SIZE; i++) {
        complex[i].x = 2;
        complex[i].y = 3;
    }

    return complex;
}

__host__ void printComplexPointer(Complex *complex, int N)
{
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << complex[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;
}

__host__ cufftComplex *generateCuFFTComplexPointerFromHostComplex(int mem_size, Complex *hostComplex)
{
    // COMMENT - based on what the main says, we want to return a pointer to the GPU memory (d_a, d_b, d_c where by convention d_* stands for device aka GPU)
    // Complex *complex = new Complex(SIZE); // CAUTION: This is bullshit since it allocates memory on host (CPU)
    // TODO populate properties x and y of variable complex at index i to 2 and 3 respectively - DONE
    // allocate memory
    cufftComplex* d_mem; // cufftComplex is the same as Complex aka float2 on host
    cudaError_t errAlloc = cudaMalloc(&d_mem, mem_size);
    if (errAlloc != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector a (error code %s)!\n", cudaGetErrorString(errAlloc));
        exit(EXIT_FAILURE);
    }
    // copy the data from host to device
    cudaError_t errCpy = cudaMemcpy(d_mem, hostComplex, mem_size, cudaMemcpyHostToDevice);
    if (errCpy != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(errCpy));
        exit(EXIT_FAILURE);
    }

    return d_mem;
}

__host__ cufftHandle transformFromTimeToSignalDomain(int N, cufftComplex *d_a, cufftComplex *d_b, cufftComplex *d_c)
{
    //TODO create a cufftHandle of size N*N and from Complex input to Complex output - DONE
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);

    //TODO execute Complex 2 Complex Forward Transformation based on the cufftHandle for d_a, d_b, d_c - DONE
    printf("Performing Forward Transformation of a, b, and c");
    cufftResult reta = cufftExecC2C(plan, (cufftComplex *)d_a, (cufftComplex *)d_a, CUFFT_FORWARD);
    if (CUFFT_SUCCESS != reta) {
        fprintf(stderr, "Failed to do the complex to complex cufft for data a within the memorychunk of d_a itself");
        exit(EXIT_FAILURE);
    }
    cufftResult retb = cufftExecC2C(plan, (cufftComplex *)d_b, (cufftComplex *)d_b, CUFFT_FORWARD);
    if (CUFFT_SUCCESS != retb) {
        fprintf(stderr, "Failed to do the complex to complex cufft for data a within the memorychunk of d_a itself");
        exit(EXIT_FAILURE);
    }
    cufftResult retc = cufftExecC2C(plan, (cufftComplex *)d_c, (cufftComplex *)d_c, CUFFT_FORWARD);
    if (CUFFT_SUCCESS != retc) {
        fprintf(stderr, "Failed to do the complex to complex cufft for data a within the memorychunk of d_a itself");
        exit(EXIT_FAILURE);
    }

    // TODO return cufftHandle for later use - was already DONE
    return plan;
}

__host__ Complex *transformFromSignalToTimeDomain(cufftHandle plan, int SIZE, cufftComplex *d_c)
{
    // TODO Initialize a Complex pointer with name results of size SIZE - DONE
    Complex* results = new Complex[SIZE]; 

    // TODO Perform Complex to Complex INVERSE transformation of cufftComplex using the passed in plan and d_c
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex*)d_c, (cufftComplex*)d_c, CUFFT_INVERSE); // only transform d_c back to bild bereich

    // TODO Perform memory copy from d_c into Complex variable results - DONE
    cudaError_t errCpy = cudaMemcpy(results, d_c, SIZE, cudaMemcpyDeviceToHost);
    if (errCpy != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(errCpy));
        exit(EXIT_FAILURE);
    }
    return results;
}

int main(int argc, char** argv)
{
    auto[N, SIZE] = parseCommandLineArguments(argc, argv);

    // create three complex arrays on host (CPU)
    Complex *a = generateComplexPointer(SIZE);
    Complex *b = generateComplexPointer(SIZE);
    Complex *c = generateComplexPointer(SIZE);

    cout << "Input random data a:" << endl;
    printComplexPointer(a, N);
    cout << "Input random data b:" << endl;
    printComplexPointer(b, N);

    // determine the size of the host array to allocate a same-sized chunk of memory on device (GPU)
    int mem_size = sizeof(Complex)* SIZE;

    cufftComplex *d_a = generateCuFFTComplexPointerFromHostComplex(mem_size, a);
    cufftComplex *d_b = generateCuFFTComplexPointerFromHostComplex(mem_size, b);
    cufftComplex *d_c = generateCuFFTComplexPointerFromHostComplex(mem_size, c);

    cufftHandle plan = transformFromTimeToSignalDomain(N, d_a, d_b, d_c);

    printf("Launching Complex Division and Subtraction\n");
    int scalar = (rand() % 5) + 1;
    cout << "Scalar value: " << scalar << endl;
    complexProcess <<< N, N >> >(d_a, d_b, d_c, SIZE, scalar);

    Complex *results = transformFromSignalToTimeDomain(plan, SIZE, d_c);
    cout << "Output data c: " << endl;
    printComplexPointer(results, N);

    delete results, a, b, c;
    cufftDestroy(plan);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}