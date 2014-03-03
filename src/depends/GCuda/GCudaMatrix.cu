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
#include "../../GClasses/GMatrix.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace GClasses {

GCudaEngine::GCudaEngine()
{
	if(cublasCreate((cublasHandle_t*)&m_handle) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasCreate failed");
	m_blockSize = 1024;
}

GCudaEngine::~GCudaEngine()
{
	if(cublasDestroy((cublasHandle_t)m_handle) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDestroy failed");
}










GCudaVector::GCudaVector()
: m_size(0), d_vals(NULL)
{
}

GCudaVector::~GCudaVector()
{
	cudaFree(d_vals);
}

void GCudaVector::resize(size_t size)
{
	cudaFree(d_vals);
	if(cudaMalloc((void**)&d_vals, size * sizeof(double)) != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
	m_size = size;
}

void GCudaVector::upload(const double* pHostVector, size_t size)
{
	if(m_size != size)
		resize(size);
	if(cublasSetVector(m_size, sizeof(double), pHostVector, 1, d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasSetVector failed");
}

void GCudaVector::download(double* pOutHostVector)
{
	if(cublasGetVector(m_size, sizeof(double), d_vals, 1, pOutHostVector, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasGetVector failed");
}

void GCudaVector::copy(GCudaEngine& engine, const GCudaVector& that)
{
	if(m_size != that.m_size)
		resize(that.m_size);
	if(cublasDcopy((cublasHandle_t)engine.m_handle, m_size, that.d_vals, 1, d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDcopy failed");
}

void GCudaVector::add(GCudaEngine& engine, GCudaVector& that, double thatScalar)
{
	GAssert(m_size == that.m_size);
	if(cublasDaxpy((cublasHandle_t)engine.m_handle, m_size, &thatScalar,
		that.d_vals, 1, d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDaxpy failed");
}

void GCudaVector::scale(GCudaEngine& engine, double scalar)
{
	if(cublasDscal((cublasHandle_t)engine.m_handle, m_size, &scalar, d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDscal failed");
}









GCudaMatrix::GCudaMatrix()
: m_rows(0), m_cols(0), d_vals(NULL)
{
}

GCudaMatrix::~GCudaMatrix()
{
	cudaFree(d_vals);
}

void GCudaMatrix::resize(size_t rows, size_t cols)
{
	cudaFree(d_vals);
	if(cudaMalloc((void**)&d_vals, rows * cols * sizeof(double)) != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
	m_rows = rows;
	m_cols = cols;
}

void GCudaMatrix::upload(const GMatrix& m)
{
	if(m_rows != m.rows() || m_cols != m.cols())
		resize(m.rows(), m.cols());
	double* pVals = d_vals;
	for(size_t i = 0; i < m_rows; i++)
	{
		if(cublasSetVector(m_cols, sizeof(double), m[i], 1, pVals, 1) != CUBLAS_STATUS_SUCCESS)
			throw Ex("cublasSetVector failed");
		pVals += m_cols;
	}
}

void GCudaMatrix::download(GMatrix& m)
{
	if(m.rows() != m_rows || m.cols() != m_cols)
		m.resize(m_rows, m_cols);
	double* pVals = d_vals;
	for(size_t i = 0; i < m_rows; i++)
	{
		if(cublasGetVector(m_cols, sizeof(double), pVals, 1, m[i], 1) != CUBLAS_STATUS_SUCCESS)
			throw Ex("cublasGetVector failed");
		pVals += m_cols;
	}
}

void GCudaMatrix::scale(GCudaEngine& engine, double scalar)
{
	if(cublasDscal((cublasHandle_t)engine.m_handle, m_rows * m_cols, &scalar, d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDscal failed");
}

void GCudaMatrix::rowVectorTimesThis(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out)
{
	GAssert(in.m_size == m_rows);
	if(out.size() != m_cols)
		out.resize(m_cols);
	double alpha = 1.0f;
	double beta = 0.0f;
	if(cublasDgemv((cublasHandle_t)engine.m_handle, CUBLAS_OP_N,
		m_cols, m_rows, &alpha, d_vals,
		m_cols, in.d_vals, 1, &beta, out.d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDgemv failed");
}

void GCudaMatrix::thisTimesColumnVector(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out)
{
	GAssert(in.m_size == m_cols);
	if(out.m_size != m_rows)
		out.resize(m_rows);
	double alpha = 1.0f;
	double beta = 0.0f;
	if(cublasDgemv((cublasHandle_t)engine.m_handle, CUBLAS_OP_T,
		m_cols, m_rows, &alpha, d_vals,
		m_cols, in.d_vals, 1, &beta, out.d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDgemv failed");
}

void GCudaMatrix::feedIn(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out, size_t inputStart)
{
	GAssert(inputStart + in.m_size <= m_rows);
	GAssert(out.m_size == m_cols);
	double alpha = 1.0f;
	double beta = 1.0f;
	if(cublasDgemv((cublasHandle_t)engine.m_handle, CUBLAS_OP_N,
		m_cols, in.m_size, &alpha, d_vals + inputStart * m_cols,
		m_cols, in.d_vals, 1, &beta, out.d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDgemv failed");
}

void GCudaMatrix::backPropError(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out, size_t inputStart)
{
	GAssert(in.m_size == m_cols);
	double alpha = 1.0f;
	double beta = 0.0f;
	if(cublasDgemv((cublasHandle_t)engine.m_handle, CUBLAS_OP_T,
		in.size(), m_rows, &alpha, d_vals,
		m_cols, in.d_vals + inputStart, 1, &beta, out.d_vals, 1) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDgemv failed");
}

void GCudaMatrix::updateWeights(GCudaEngine& engine, GCudaVector& upStreamInput, GCudaVector& downStreamError, double learningRate)
{
	if(cublasDger((cublasHandle_t)engine.m_handle, m_cols, m_rows, &learningRate,
		downStreamError.d_vals, 1, upStreamInput.d_vals, 1,
		d_vals, m_cols) != CUBLAS_STATUS_SUCCESS)
		throw Ex("cublasDger failed");
}



} // namespace GClasses
