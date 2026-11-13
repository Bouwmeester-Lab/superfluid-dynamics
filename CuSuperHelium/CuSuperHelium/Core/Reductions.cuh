#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/std/complex>

#ifdef __INTELLISENSE__ // for Visual Studio IntelliSense https://stackoverflow.com/questions/77769389/intellisense-in-visual-studio-cannot-find-cuda-cooperative-groups-namespace
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__
namespace cg = cooperative_groups;


template<size_t BLOCK_SIZE, typename Op, typename Type>
__device__ Type block_reduce_sum(Type local) {
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);       // 32-thread tile

	// 1) reduce within each warp
	Type warp_sum = cg::reduce(warp, local, Op());

	// 2) write one value per warp to shared mem
	__shared__ Type smem[BLOCK_SIZE / 32];
	if (warp.thread_rank() == 0)
		smem[warp.meta_group_rank()] = warp_sum;
	block.sync();

	// 3) first warp reduces the per-warp sums
	Type block_sum = 0.0;
	if (warp.meta_group_rank() == 0) {
		int num_warps = (BLOCK_SIZE + 31) / 32;
		Type v = (warp.thread_rank() < num_warps) ? smem[warp.thread_rank()] : 0.0;
		block_sum = cg::reduce(warp, v, Op());  // result in lane 0 of this warp
	}
	return (warp.meta_group_rank() == 0 && warp.thread_rank() == 0) ? block_sum : 0.0;
}