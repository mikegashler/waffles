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

__global__ void cuda_addAndApplyTanh(double* pA, const double* pB, int n)
{
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < n) {
                pA[id] = pA[id] + pB[id];
                pA[id] = tanh(pA[id]);
	}
}

namespace GClasses {

void GCudaVector::addAndApplyTanh(GCudaEngine& engine, const GCudaVector& that)
{
	GAssert(that.m_size == m_size);
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (m_size + blockSize - 1) / blockSize;
	cuda_addAndApplyTanh<<<gridSize, blockSize>>>(this->d_vals, that.d_vals, (int)m_size);
}

} // namespace GClasses
