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
#include "../../GClasses/GVec.h"

namespace GClasses {

class GMatrix;

/// Provides handles and values necessary to perform some GPU operations.
class GCudaEngine
{
public:
	void* m_handle; // This should really be a cublasHandle_t, but I would rather not include <cublas_v2.h> in this file, and C++ does not provide a mechanism to forward-declare typedefs.
	size_t m_blockSize;
	void* m_prng; // This should be a curandGenerator_t.
	bool m_hogWild;

	GCudaEngine();
	~GCudaEngine();

	/// Synchronizes the GPU with the CPU. That is, block until all of the GPU processors complete
	/// the most-recent operation.
	void sync();

	/// Specify whether to use hog wild.
	/// if b is true, then no synchronization will be used.
	/// if b is false, then synchronization will be used again.
	void setHogWild(bool b) { m_hogWild = b; }
};


/// Represents a vector on the GPU. The methods in this class perform
/// operations asynchronously on the GPU. That is, they return immediately
/// and begin processing in parallel on the GPU. Before you can rely on
/// any results, you must call GCudaEngine::sync(), which blocks until
/// all of the GPU processors are ready again.
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
	size_t size() const { return m_size; }

	/// Fills the vector with the specified value
	void fill(GCudaEngine& engine, double val);

	/// Copies a vector from the host (CPU memory) to the device (GPU memory).
	/// Resizes this vector if necessary.
	void upload(const GVec& hostVector);

	/// Copies a vector from the device (GPU memory) to the host (CPU memory).
	void download(GVec& hostVector) const;

	/// Copies that into this. (Resizes this vector if necessary.)
	void copy(GCudaEngine& engine, const GCudaVector& that);

	/// Applies the tanh function element-wise to this vector.
	void activateTanh(GCudaEngine& engine);

	/// Multiplies each element in this vector by (1.0-(a*a)), where a is the corresponding element in activation.
	void deactivateTanh(GCudaEngine& engine, const GCudaVector& activation);
/*
	/// Applies the bend function element-wise to this vector.
	void activateBend(GCudaEngine& engine);

	/// Multiplies each element in this vector by the derivative of the bend function.
	void deactivateBend(GCudaEngine& engine, const GCudaVector& activation);
*/
	/// Adds thatScalar * that vector to this vector.
	/// For example, if thatScalar is -1, then this method will subtract that vector from this vector.
	void add(GCudaEngine& engine, GCudaVector& that, double thatScalar = 1.0);

	/// Multiplies this vector by scalar
	void scale(GCudaEngine& engine, double scalar);

	/// Fills this vector with random values from a standard uniform distribution.
	void randomUniform(GCudaEngine& engine);

	/// Fills this vector with random values from a Normal distribution.
	void randomNormal(GCudaEngine& engine, double mean, double dev);
};



/// Represents a matrix on the GPU. The methods in this class perform
/// operations asynchronously on the GPU. That is, they return immediately
/// and begin processing in parallel on the GPU. Before you can rely on
/// any results, you must call GCudaEngine::sync(), which blocks until
/// all of the GPU processors are ready again.
class GCudaMatrix
{
public:
	size_t m_rows;
	size_t m_cols;
	double* d_vals;

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
	void download(GMatrix& m) const;

	/// Adds that multipled by thatScalar to this.
	void add(GCudaEngine& engine, GCudaMatrix& that, double thatScalar);

	/// Copies the contents of that matrix into this matrix. Resizes if necessary.
	void copy(GCudaEngine& engine, GCudaMatrix& that);

	/// Multiplies this matrix by scalar
	void scale(GCudaEngine& engine, double scalar);

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

	/// Adds the outer product of upStreamInput and downStreamError, multiplied by learningRate, to this matrix.
	void addOuterProduct(GCudaEngine& engine, GCudaVector& upStreamInput, GCudaVector& downStreamError, double learningRate);

	/// Returns the sum of absolute values of elements in the specified row.
	double rowSumAbs(GCudaEngine& engine, size_t row);

	/// Returns the sum of squared values of elements in the specified row.
	double rowSumSquare(GCudaEngine& engine, size_t row);

	/// Returns the sum of absolute values of elements in the specified column.
	double colSumAbs(GCudaEngine& engine, size_t col);

	/// Returns the sum of squared values of elements in the specified column.
	double colSumSquare(GCudaEngine& engine, size_t col);

	/// Scales the values in the specified row.
	void scaleRow(GCudaEngine& engine, size_t row, double scalar);

	/// Scales the values in the specified column.
	void scaleCol(GCudaEngine& engine, size_t col, double scalar);

	/// Fills the matrix with the specified value
	void fill(GCudaEngine& engine, double val);

	/// Fills this matrix with random values from a Normal distribution.
	void fillNormal(GCudaEngine& engine, double mean, double dev);
};


} // namespace GClasses

#endif // __GCUDAMATRIX_H__
