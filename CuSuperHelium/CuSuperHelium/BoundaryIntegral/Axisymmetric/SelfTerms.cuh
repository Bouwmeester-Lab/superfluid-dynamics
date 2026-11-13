#pragma once
#ifndef RADIAL_SELF_TERMS_H
#define RADIAL_SELF_TERMS_H

#include "cuda_runtime.h"
#include "Core/constants.cuh"
#include "BoundaryIntegral/Axisymmetric/BoundaryNodes.cuh"

/// <summary>
/// Self-interaction (k == j) entries of the axisymmetric single- and double-layer matrices, from the local
/// logarithmic behaviour of the free-space ring kernel G ~ ln(8r/|s|) / (2 pi r), integrated over the node's own cell
/// with the source measure r ds. Only the direct ring is included (no image or wall correction).
/// </summary>
namespace RadialSelfTerms
{
	/// <summary>
	/// S_kk = ds/(2 pi) (ln(16 r/ds) + 1)
	/// </summary>
	__device__ __forceinline__ double single(const BoundaryNodes& nodes, size_t k)
	{
		const double dsk = nodes.ds[k];
		return dsk / (2 * PI_d) * (log(16.0 * nodes.r[k] / dsk) + 1);
	}

	/// <summary>
	/// D_kk without the 1/2 jump term: ds/(4 pi) (curvature - n_r/r ln(16 r/ds))
	/// </summary>
	__device__ __forceinline__ double doubleLayer(const BoundaryNodes& nodes, size_t k)
	{
		const double dsk = nodes.ds[k];
		const double rk = nodes.r[k];
		return dsk / (4.0 * PI_d) * (nodes.curvature[k] - nodes.nr[k] / rk * log(16.0 * rk / dsk));
	}
}

#endif // RADIAL_SELF_TERMS_H
