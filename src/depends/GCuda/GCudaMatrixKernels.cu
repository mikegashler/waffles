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
//#include "GCudaLayers.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void cuda_fill(double* pVec, double val, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n)
		pVec[n] = val;
}

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

__global__ void cuda_conv2D_ff(double* pA, double* pNet, const double* in, const double* pKernels, const double* pBias, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputRows, size_t inputCols, size_t inputChannels, size_t padding, size_t stride)
{
	// Do all values for i, j, and k in parallel
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = id % outputCols;
	id /= outputCols;
	size_t j = id % outputRows;
	id /= outputRows;
	if(id >= kernelCount)
		return;
	size_t k = id;

	// Compute some intermediate values
	size_t outChannelOffset = k * outputRows * outputCols;
	size_t outRowOffset = j * outputCols;
	int inRowOffset = j * stride - padding;

	// This block of code is derived from the serial implementation
	size_t kk = k * inputChannels * kernelRows * kernelCols;
	size_t index = outChannelOffset + outRowOffset + i;
	int inColOffset = i * stride - padding;
	pNet[index] = pBias[k];
	for(size_t z = 0; z < inputChannels; z++)
	{
		size_t kernelChannelOffset = z * kernelRows * kernelCols;
		size_t inChannelOffset = z * inputRows * inputCols;
		for(size_t y = 0; y < kernelRows; y++)
		{
			size_t kernelRowOffset = y * kernelCols;
			int inRow = inRowOffset + y;
			for(size_t x = 0; x < kernelCols; x++)
			{
				int inCol = inColOffset + x;
				if(inRow >= 0 && inRow < (int)inputRows && inCol >= 0 && inCol < (int)inputRows)
				{
					size_t idx = inChannelOffset + inputCols * inRow + inCol;
					pNet[index] += pKernels[kk + kernelChannelOffset + kernelRowOffset + x] * in[idx];
				}
			}
		}
	}

	//a[index] = pThis->m_pActivationFunction->squash(n[index]);
	pA[index] = tanh(pNet[index]);
}

__global__ void cuda_conv2D_deactivate(double* err, const double* net, const double* activation, size_t outputs)
{	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= outputs)
		return;
	err[id] *= (1.0 - activation[id] * activation[id]);
}

__global__ void cuda_conv2D_backPropError(double* upStreamError, const double* err, const double* pKernels, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputChannels, size_t inputRows, size_t inputCols, size_t padding, size_t stride)
{
	// Do all values for i, j, and k in parallel
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = id % inputChannels;
	id /= inputChannels;
	size_t j = id % outputRows;
	id /= outputRows;
	if(id >= kernelCount)
		return;
	size_t k = id;

	// Compute some intermediate values
	size_t outChannelOffset = k * outputRows * outputCols;
	size_t outRowOffset = j * outputCols;
	int inRowOffset = j * stride - padding;

	// This block of code is derived from the serial implementation
	size_t kk = k * inputChannels * kernelRows * kernelCols;
	size_t index = outChannelOffset + outRowOffset + i;
	int inColOffset = i * stride - padding;
	for(size_t z = 0; z < inputChannels; z++)
	{
		size_t kernelChannelOffset = z * kernelRows * kernelCols;
		size_t inChannelOffset = z * inputRows * inputCols;
		for(size_t y = 0; y < kernelRows; y++)
		{
			size_t kernelRowOffset = y * kernelCols;
			int inRow = inRowOffset + y;
			for(size_t x = 0; x < kernelCols; x++)
			{
				int inCol = inColOffset + x;
				if(inRow >= 0 && inRow < (int)inputRows && inCol >= 0 && inCol < (int)inputRows)
				{
					size_t idx = inChannelOffset + inputCols * inRow + inCol;
					upStreamError[idx] += pKernels[kk + kernelChannelOffset + kernelRowOffset + x] * err[index];
				}
			}
		}
	}

}

__global__ void cuda_conv2D_updateDeltas(double* delta, double* biasDelta, const double* upStreamActivation, const double* err, double momentum, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputChannels, size_t inputRows, size_t inputCols, size_t padding, size_t stride)
{
	// Do all values for i, j, and k in parallel
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = id % outputCols;
	id /= outputCols;
	size_t j = id % outputRows;
	id /= outputRows;
	if(id >= kernelCount)
		return;
	size_t k = id;

	// Compute some intermediate values
	size_t outChannelOffset = k * outputRows * outputCols;
	size_t outRowOffset = j * outputCols;
	int inRowOffset = j * stride - padding;

	// This block of code is derived from the serial implementation
	size_t kk = k * inputChannels * kernelRows * kernelCols;
	size_t index = outChannelOffset + outRowOffset + i;
	int inColOffset = i * stride - padding;
	biasDelta[k] += err[index];
	for(size_t z = 0; z < inputChannels; z++)
	{
		size_t kernelChannelOffset = z * kernelRows * kernelCols;
		size_t inChannelOffset = z * inputRows * inputCols;
		for(size_t y = 0; y < kernelRows; y++)
		{
			size_t kernelRowOffset = y * kernelCols;
			int inRow = inRowOffset + y;
			for(size_t x = 0; x < kernelCols; x++)
			{
				int inCol = inColOffset + x;
				if(inRow >= 0 && inRow < (int)inputRows && inCol >= 0 && inCol < (int)inputRows)
				{
					size_t idx = inChannelOffset + inputCols * inRow + inCol;
					delta[kk + kernelChannelOffset + kernelRowOffset + x] += err[index] * upStreamActivation[idx];
				}
			}
		}
	}
}

namespace GClasses {

void GCudaVector::fill(GCudaEngine& engine, double val)
{
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (m_size + blockSize - 1) / blockSize;
	cuda_fill<<<gridSize, blockSize>>>(this->d_vals, val, (int)m_size);
}

void GCudaMatrix::fill(GCudaEngine& engine, double val)
{
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (m_rows * m_cols + blockSize - 1) / blockSize;
	cuda_fill<<<gridSize, blockSize>>>(this->d_vals, val, (int)(m_rows * m_cols));
}

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

void Conv2D_feedForward(GCudaEngine& engine, GCudaVector& activation, GCudaVector& net, GCudaVector& incoming, const GCudaMatrix& kernels, const GCudaVector& bias, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputRows, size_t inputCols, size_t inputChannels, size_t padding, size_t stride)
{
	size_t n = kernelCount * outputRows * outputCols;
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (n + blockSize - 1) / blockSize;
	cuda_conv2D_ff<<<gridSize, blockSize>>>(activation.d_vals, net.d_vals, incoming.d_vals, kernels.d_vals, bias.d_vals, kernelCount, kernelRows, kernelCols, outputRows, outputCols, inputRows, inputCols, inputChannels, padding, stride);
}

void Conv2D_deactivate(GCudaEngine& engine, GCudaVector& error, const GCudaVector& net, const GCudaVector& activation, size_t outputs)
{
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (outputs + blockSize - 1) / blockSize;
	cuda_conv2D_deactivate<<<gridSize, blockSize>>>(error.d_vals, net.d_vals, activation.d_vals, outputs);
}

void Conv2D_backPropError(GCudaEngine& engine, GCudaVector& upStreamError, const GCudaVector& err, const GCudaMatrix& kernels, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputChannels, size_t inputRows, size_t inputCols, size_t padding, size_t stride)
{
	upStreamError.fill(engine, 0.0);
	size_t n = kernelCount * outputRows * inputChannels;
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (n + blockSize - 1) / blockSize;
	cuda_conv2D_backPropError<<<gridSize, blockSize>>>(upStreamError.d_vals, err.d_vals, kernels.d_vals, kernelCount, kernelRows, kernelCols, outputRows, outputCols, inputChannels, inputRows, inputCols, padding, stride);
}

void Conv2D_updateDeltas(GCudaEngine& engine, GCudaMatrix& delta, GCudaVector& biasDelta, const GCudaVector& upStreamActivation, const GCudaVector& err, double momentum, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputChannels, size_t inputRows, size_t inputCols, size_t padding, size_t stride)
{
	if(momentum != 1.0)
	{
		delta.scale(engine, momentum);
		biasDelta.scale(engine, momentum);
	}
	size_t n = kernelCount * outputRows * outputCols;
	size_t blockSize = engine.m_blockSize;
	size_t gridSize = (n + blockSize - 1) / blockSize;
	cuda_conv2D_updateDeltas<<<gridSize, blockSize>>>(delta.d_vals, biasDelta.d_vals, upStreamActivation.d_vals, err.d_vals, momentum, kernelCount, kernelRows, kernelCols, outputRows, outputCols, inputChannels, inputRows, inputCols, padding, stride);
}



} // namespace GClasses
