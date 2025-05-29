# CuDNN lab
## Important Remarks
+ CuDNN uses a dualism between the data on GPU and a description how the data is organized
    - Creating Tensors, Filter Kernels etc on GPU is step 1
    - Creating a descriptor with cudnn which described how the data should be interpreted by cudnn is step 2
    - Using both data types with the (free) cudnn function allows you to do complex machine learning processing on GPU