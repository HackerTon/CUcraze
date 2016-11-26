#include <iostream>
#include <device_functions.h>
#include "helper.cuh"

__global__ void add_3(float *stack, int number){
	int myid = blockIdx.x * blockDim.x + threadIdx.x;

	int new_id = myid + 1;

	temp[myid] = stack[myid];

	__syncthreads();

	extern __shared__ float temp[];

	if (myid < number){

		stack[myid] = temp[new_id] + temp[new_id - 1];

	}
}

int main(int argc, char** argv){

	//CUDA Event Function Declaration

	cudaEvent_t start, stop;

	cuda(cudaEventCreate(&start));
	cuda(cudaEventCreate(&stop));

	float time = 0;

	int data_size = 512;

	float *dStack = 0;
	float *hStack = 0;

	cuda(cudaMalloc(&dStack, sizeof(float)*data_size));
	cuda(cudaMallocHost(&hStack, sizeof(float)*data_size));

	for (int i = 0; i < data_size; i++){
		hStack[i] = i;
	}

	cuda(cudaMemcpy(dStack, hStack, sizeof(float)*data_size, cudaMemcpyHostToDevice));

	dim3 thread(data_size);
	dim3 block(1);

	cuda(cudaEventRecord(start, 0));

	add_3<<<block, thread, sizeof(float)*data_size>>>(dStack, thread.x);

	cuda(cudaEventRecord(stop, 0));

	cuda(cudaEventSynchronize(stop));

	cuda(cudaPeekAtLastError());

	cuda(cudaEventElapsedTime(&time, start, stop));

	cuda(cudaMemcpy(hStack, dStack, sizeof(float)*data_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < data_size; i++){
		std::cout << hStack[i] << " space " << i << std::endl;
	}

	std::cout << "Time taken for the kernel to finish = " << time << "ms" << std::endl;

	cuda(cudaFree(dStack));
	cuda(cudaFreeHost(hStack));
}
