
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <gtest/gtest.h>


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

TEST(test, TestName)
{
    //This Test will work
    int a[4] = { 1, 2, 3, 4 };
	int b[4] = { 5, 6, 7, 8 };
	int c[4] = { 0, 0, 0, 0 };
    
	int* d_a, * d_b, * d_c;

    cudaMalloc(&d_a, 4 * sizeof(int));
	cudaMalloc(&d_b, 4 * sizeof(int));
	cudaMalloc(&d_c, 4 * sizeof(int));

    cudaMemcpy(d_a, a, 4* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, 4 * sizeof(int), cudaMemcpyHostToDevice);

	addKernel <<<1, 4 >> > (d_c, d_a, d_b);
	cudaMemcpy(c, d_c, 4 * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(c[i], a[i] + b[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main(int argc, char** argv) {
    // Optional: CUDA setup check
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices available: " << deviceCount << std::endl;

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    return RUN_ALL_TESTS();
}
