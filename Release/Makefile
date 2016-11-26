all: cuda

cuda: ../src/cuda.cu
	nvcc -arch=compute_52 -code=compute_52 -lineinfo --ftz=true --default-stream per-thread -std=c++11 -m=64 -o cuda ../src/cuda.cu

clean:
	rm -rf ../Release/cuda