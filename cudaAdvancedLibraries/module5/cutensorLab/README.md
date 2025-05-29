# CuTensor Lab
## Remarks
+ CuTensor is one abstraction layer less then cuDnn: You can define your own tensor-contraction operations (tensor-contraction is mathmatically the settlement of multiple tensors and scalars to a new tensor)
+ It is commonly used by researchers that need to have control over complex matrix operations that are not standart for Deep Neural Networks etc
    - You can setup complex index associations by yourself with cuTensor
+ What does CuTensor compared to "programming it all yourself"?
    - CuTensor abstracts the looping and thread modelling on GPU away. You can define your tensor contractions by using the [Einsteinsche Summenkonvention](https://de.wikipedia.org/wiki/Einsteinsche_Summenkonvention)
    - The focus is on scientific computing, research and less on application development