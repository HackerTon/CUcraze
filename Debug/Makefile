all: cuda

cuda: ../src/cuda.cu
	nvcc -g -G -lineinfo -arch=compute_52 -code=compute_52 --ftz=true --default-stream per-thread -std=c++11 -m=64 -o cuda ../src/cuda.cu