#define TEST

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <string>
#include <gtest/gtest.h>
#include "MatrixMTests.cuh"
#include "ODESolverTests.cuh"
#include "ComplexFunctionsTests.cuh"
#include "IntegrationPathEquivalenceTests.cuh"
#include "ExportFunctionTests.cuh"
#include "RadialBoundaryNodesTests.cuh"
#include "DeviceBufferTests.cuh"
#include "CudaChecksTests.cuh"

int main(int argc, char** argv) {
    // Optional: CUDA setup check
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices available: " << deviceCount << std::endl;

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int res = RUN_ALL_TESTS();

    // Figures made by tests are shown only on request, so automated runs never block.
    const char* showPlots = std::getenv("CUSH_SHOW_PLOTS");
    if (showPlots != nullptr && std::string(showPlots) == "1") {
        plt::show();
    }
	return res;
}
