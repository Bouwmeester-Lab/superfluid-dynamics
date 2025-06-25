
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <gtest/gtest.h>
#include "MatrixMTests.cuh"
#include "ODESolverTests.cuh"
#include "ComplexFunctionsTests.cuh"

int main(int argc, char** argv) {
    // Optional: CUDA setup check
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices available: " << deviceCount << std::endl;

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int res = RUN_ALL_TESTS();

    plt::show();
	return res;
}
