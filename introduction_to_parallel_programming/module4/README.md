# CAUTION: This is copied from the compiler lab but with my personal annotations

For this assignment, you will need to execute the following commands across all of the assignment parts:

Driver API example code based on code found at https://gist.github.com/tautologico/2879581

#FATBIN assignment part
1. Ensure that you are in the project/driver_api folder
2. Update the drivertest.cpp to use the matSumKernel.fatbin
3. nvcc -o matSumKernel.fatbin -fatbin matSumKernel.cu -lcuda # recompile matSumKernel.cu and link against cuda, see $ nvcc -h | grep library and for the --fatbin argument $ nvcc -h | grep fatbin
4. nvcc -o drivertest.o -c drivertest.cpp -lcuda # compile the cpp file with the nvcc compiler wrapper and ensure that all libcuda.so (cuda runtime api) calls are available for the compiler - NO LINKING IS DONE AT THIS STAGE
5. nvcc -o drivertest drivertest.o -lcuda # link the libcuda.so and create an executable out of the object file from the previous step
6. ./drivertest > ~/project/output-fatbin.txt # execute

#PTX assignment part
1. Update the drivertest.cpp to use the matSumKernel.ptx
2. nvcc -o matSumKernel.ptx -ptx matSumKernel.cu -lcuda # create a ptx file from the calculation kernel definition
3. nvcc -o drivertest.o -c drivertest.cpp -lcuda # compile the test programm (this was done before but with the module as .fatbin, so we need to do it again)
4. nvcc -o drivertest drivertest.o -lcuda # create an executable out of the object file
5. ./drivertest > ~/project/output-ptx.txt # execute the program

Runtime API example code based on code found at https://gist.github.com/al-indigo/4dd93d48a2886db6b1ac

#Runtime assignment part
Be aware: this has nothing to do with the driver_api code. Even if we do the same kind of calculation. The calls differ from the drive api (e.g. cuMemAlloc in driver API is same as cudaMalloc in the runtime API)
1. Ensure that you are in the project/runtime_api folder
2. nvcc -o vector_add vector_add.cu
3. ./vector_add > ~/project/output-runtime.txt
