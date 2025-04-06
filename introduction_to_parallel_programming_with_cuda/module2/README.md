# Module 2: Executing Cuda Kernels
+ Remark: Most of the cuda stuff can be found in OpenCL as well, since OpenCL is a generalization of Cuda which is more hardware independened.
    - CUDA is better documented then OpenCL and therefore easier to use as an application developer

## Ressources to understand the Kernel- and Thread Model of GP-GPUs (General Purpose - Graphics Processing Units):
+ CUDA = "Compute Unified Device Architecture"
    - Goes back to 2007 NVidida Tesla Architecture which was the first "freely programmable" GPU which was capable to program scientific computing/arbitary parallel code which are non-graphics code on the GPU
        * Before 2007, GPUs were hardwired for doing shading of different kinds with configurable graphics/shading pipelines.
+ https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
    - General oveview
+ https://www.cs.cmu.edu/afs/cs/academic/class/15418-s18/www/lectures/06_gpuarch.pdf
    - Very good explaination about how threads, thread blocks and (Nvidia specific) wraps play together
    
