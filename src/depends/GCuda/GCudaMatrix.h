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
protected:
	size_t m_size;
	double* d_vals;

public:
	/// Makes a vector of size 0.
	GCudaVector();
	~GCudaVector();

	/// Resizes this vector.
	void resize(size_t size);

	/// Returns the size of this vector.
	size_t size() const { return m_size; }

	/// Copies a vector from the host (CPU memory) to the device (GPU memory).
	/// Resizes this vector if necessary.
	void upload(const double* pHostVector, size_t size);

	/// Copies a vector from the device (GPU memory) to the host (CPU memory).
	void download(double* pOutHostVector);

	/// Copies that into this. (Resizes this vector if necessary.)
	void copy(GCudaEngine& engine, const GCudaVector& that);

	/// Applies the tanh function element-wise to this vector.
	void activateTanh(GCudaEngine& engine);

	/// Multiplies each element in this vector by (1.0-(a*a)), where a is the corresponding element in activation.
	void deactivateTanh(GCudaEngine& engine, const GCudaVector& activation);

	/// Adds thatScalar * that vector to this vector.
	/// For example, if thatScalar is -1, then this method will subtract that vector from this vector.
	void add(GCudaEngine& engine, GCudaVector& that, double thatScalar = 1.0);

	/// Multiplies this vector by scalar
	void scale(double scalar);
};



/// Represents a matrix on the GPU. Performs operations in parallel.
class GCudaMatrix
{
protected:
	size_t m_rows;
	size_t m_cols;
	double* d_vals;

public:
	/// Makes a 0x0 matrix.
	GCudaMatrix();
	~GCudaMatrix();

	/// Returns the number of rows in this matrix
	size_t rows() const { return m_rows; }

	/// Returns the number of columns in this matrix
	size_t cols() const { return m_cols; }

	/// Resizes this matrix
	void resize(size_t rows, size_t cols);

	/// Copies m from the host (CPU memory) to the device (GPU memory).
	/// Resizes this matrix if necessary.
	void upload(const GMatrix& m);

	/// Copies from this device (GPU memory) into m on the host (CPU memory).
	/// Resizes m if necessary.
	void download(GMatrix& m);

	/// Multiplies this matrix by scalar
	void scale(double scalar);

	/// Vector-matrix multiply
	void rowVectorTimesThis(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out);

	/// Matrix-vector multiply
	void thisTimesColumnVector(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out);

	/// Multiplies a row-vector, in, times the sub-matrix from (inputStart,0) to (inputStart + in.size(), cols()),
	/// and adds the result to whatever is in out.
	void feedIn(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out, size_t inputStart);

	/// Multiplies the sub-matrix from (0,inputStart) to (rows(), inputStart + in.size()) times a column vector, in.
	/// Puts the results in out.
	void backPropError(GCudaEngine& engine, const GCudaVector& in, GCudaVector& out, size_t inputStart);

	/// This method is used by implementations of gradient descent to update the weights
	void updateWeights(GCudaEngine& engine, GCudaVector& upStreamInput, GCudaVector& downStreamError, double learningRate);
};


} // namespace GClasses

#endif // __GCUDAMATRIX_H__
