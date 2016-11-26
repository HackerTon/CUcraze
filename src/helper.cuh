#include <iostream>

#define cuda(error) cuda_error(error, __LINE__)

void cuda_error(cudaError_t status, int line){
	if (status != 0){
		std::cout << "ERROR at " << line << " with code name " << status << std::endl;
	}
}
