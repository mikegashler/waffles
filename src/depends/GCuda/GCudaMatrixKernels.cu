/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include "GCudaMatrix.h"
#include "../../GClasses/GError.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void cuda_activateTanh(double* pA, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		pA[id] = tanh(pA[id]);
	}
}

__global__ void cuda_deactivateTanh(double* pE, const double* pA, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		pE[id] *= (1.0 - (pA[id] * pA[id]));
	}
}
/*
__global__ void cuda_activateBend(double* pA, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		double x = pA[id];
		pA[id] = 0.5 * sqrt(x * x + 1) + x - 0.5
	}
}

__global__ void cuda_deactivateBend(double* pE, const double* pA, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		double x = pE[id];
		pE[id] *= 0.5 * (x / sqrt(x * x + 1)) + 1;
	}
}
*/

namespace GClasses {

void GCudaVector::activateTanh(GCudaEngine& engine)
{
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (m_size + blockSize - 1) / blockSize;
	cuda_activateTanh<<<gridSize, blockSize>>>(this->d_vals, (int)m_size);
}

void GCudaVector::deactivateTanh(GCudaEngine& engine, const GCudaVector& activation)
{
	GAssert(activation.m_size == m_size);
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (m_size + blockSize - 1) / blockSize;
	cuda_deactivateTanh<<<gridSize, blockSize>>>(this->d_vals, activation.d_vals, (int)m_size);
}
/*
void GCudaVector::activateBend(GCudaEngine& engine)
{
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (m_size + blockSize - 1) / blockSize;
	cuda_activateBend<<<gridSize, blockSize>>>(this->d_vals, (int)m_size);
}

void GCudaVector::deactivateBend(GCudaEngine& engine, const GCudaVector& activation)
{
	GAssert(activation.m_size == m_size);
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (m_size + blockSize - 1) / blockSize;
	cuda_deactivateBend<<<gridSize, blockSize>>>(this->d_vals, activation.d_vals, (int)m_size);
}
*/
} // namespace GClasses
