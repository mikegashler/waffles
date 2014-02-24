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

#ifndef __GCUDAMATRIX_H__
#define __GCUDAMATRIX_H__

#include <cstddef>

namespace GClasses {

class GMatrix;

/// Provides handles and values necessary to perform some GPU operations.
class GCudaEngine
{
public:
	void* m_handle; // This should really be a cublasHandle_t, but I would rather not include <cublas_v2.h> in this file, and C++ does not provide a mechanism to forward-declare typedefs.
	size_t m_blockSize;

	GCudaEngine();
	~GCudaEngine();

};


/// Represents a vector on the GPU. Performs operations in parallel.
class GCudaVector
{
public:
	size_t m_size;
	double* d_vals;

	/// Makes a vector of size 0.
	GCudaVector();
	~GCudaVector();

	/// Resizes this vector.
	void resize(size_t size);

	/// Returns the size of this vector.
	size_t size() { return m_size; }

	/// Copies a vector from the host (CPU memory) to the device (GPU memory).
	/// Resizes this vector if necessary.
	void upload(const double* pHostVector, size_t size);

	/// Copies a vector from the device (GPU memory) to the host (CPU memory).
	void download(double* pOutHostVector);

	/// Copies that into this. (Resizes this vector if necessary.)
	void copy(GCudaEngine& engine, const GCudaVector& that);

	/// Adds that vector to this vector, then applies the tanh function element-wise to this vector.
	void addAndApplyTanh(GCudaEngine& engine, const GCudaVector& that);
};



/// Represents a matrix on the GPU. Performs operations in parallel.
class GCudaMatrix
{
public:
	size_t m_rows;
	size_t m_cols;
	double* d_vals;

	/// Makes a 0x0 matrix.
	GCudaMatrix();
	~GCudaMatrix();

	/// Resizes this matrix
	void resize(size_t rows, size_t cols);

	/// Copies m from the host (CPU memory) to the device (GPU memory).
	/// Resizes this matrix if necessary.
	void upload(const GMatrix& m);

	/// Copies from this device (GPU memory) into m on the host (CPU memory).
	/// Resizes m if necessary.
	void download(GMatrix& m);

	/// Vector-matrix multiply
	void rowVectorTimesThis(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out);

	/// Matrix-vector multiply
	void thisTimesColumnVector(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out);

};


} // namespace GClasses

#endif // __GCUDAMATRIX_H__
