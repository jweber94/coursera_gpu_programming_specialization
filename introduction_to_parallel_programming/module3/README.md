# Parallel programming in Python and C++

## Parallel programming in Python
+ Good ressource for the multiprocessing library:
    - https://www.machinelearningplus.com/python/parallel-processing-python/
+ ```import threading``` library:
    - The ```Lock()``` object is the same as a ```std::mutex``` in C++
    - The ```BoundedSemaphore()``` has the same API as the ```Lock()``` object in order to acquire or release the sychronization primitive. This is a very handy solution,  since python is dynamically typed, so you can reuse functions that have synchronization primitives as parameters.