# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: build_cublas build_cusolver build_cusparse

build_cublas: cublas_example.cu
	$(CXX) cublas_example.cu --std c++17 `pkg-config opencv --cflags --libs` -o cublas_example.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcublas

build_cusolver: cusolver_example.cu
	$(CXX) cusolver_example.cu --std c++17 `pkg-config opencv --cflags --libs` -o cusolver_example.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcublas -lcusolver

build_cusparse: cusparse_example.cu
	$(CXX) cusparse_example.cu --std c++17 `pkg-config opencv --cflags --libs` -o cusparse_example.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcublas -lcusparse

run: run_cublas run_cusolver run_cusparse

run_cublas:
	./cublas_example.exe

run_cusolver:
	./cusolver_example.exe

run_cusparse:
	./cusparse_example.exe

clean:
	rm -f *.exe output*.txt 