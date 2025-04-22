IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: multi_gpu.cu
	$(COMPILER) $(COMPILER_FLAGS) multi_gpu.cu -o multi_gpu.exe -Wno-deprecated-gpu-targets

clean:
	rm -f multi_gpu.exe

run:
	./multi_gpu.exe

all: clean build run
