/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
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

#ifndef __GSPARSEMATRIX_H__
#define __GSPARSEMATRIX_H__

#include <map>
#include <vector>
#include <iostream>

namespace GClasses {

class GMatrix;
class GRand;
class GDomNode;
class GDom;
class GVec;

typedef std::map<size_t,double> SparseVec;

/// This class stores a row-compressed sparse matrix. That is,
/// each row consists of a map from a column-index to a value.
class GSparseMatrix
{
protected:
	size_t m_cols;
	std::vector<SparseVec> m_rows;
	double m_defaultValue;

public:
	/// Construct a sparse matrix with the specified number of rows and columns.
	/// defaultValue specifies the common value that is not stored. (Typically,
	/// defaultValue is 0, but for some applications it may make more sense to
	/// set it to UNKNOWN_REAL_VALUE.)
	GSparseMatrix(size_t rows, size_t cols, double defaultValue = 0.0);

	/// Deserializes a sparse matrix
	GSparseMatrix(const GDomNode* pNode);

	~GSparseMatrix();

	static void test();

	typedef SparseVec::const_iterator Iter;

	/// Serializes this object
	GDomNode* serialize(GDom* pDoc) const;

	/// Returns the default value--the common value that is not stored.
	double defaultValue() { return m_defaultValue; }

	/// Returns the number of rows (as if this matrix were dense)
	size_t rows() const { return m_rows.size(); }

	/// Returns the number of columns (as if this matrix were dense)
	size_t cols() const { return m_cols; }

	/// Copies a row into a non-sparse vector
	void fullRow(GVec& outFullRow, size_t row);

	/// Returns a const_iterator to the beginning of a row. The iterator
	/// references a pair, such that first is the column, and second is the value.
	Iter rowBegin(size_t i) const { return m_rows[i].begin(); }

	/// Returns a const_iterator to the end of a row. The iterator
	/// references a pair, such that first is the column, and second is the value.
	Iter rowEnd(size_t i) const { return m_rows[i].end(); }

	/// Returns the specified sparse row.
	SparseVec& row(size_t i) { return m_rows[i]; }

	/// Returns the number of non-default-valued elements in the specified row.
	size_t rowNonDefValues(size_t i) { return m_rows[i].size(); }

	/// Returns the value at the specified position in the matrix. Returns the
	/// default value if no element is stored at that position.
	double get(size_t row, size_t col) const;

	/// Sets a value at the specified position in the matrix. (If val is the default
	/// value, it removes the element from the matrix.)
	void set(size_t row, size_t col, double val);

	/// Copies values from "that" into "this".
	/// Keeps values in "this" that are not overwritten by "that".
	/// Any default-valued elements in that will be left the same.
	/// Any non-default-valued elements will be
	/// copied over the value in this. If the matrices are different
	/// sizes, any non-overlapping elements will be left at the default value,
	/// no-matter what value it has in that.
	void copyFrom(const GSparseMatrix* that);

	/// Copies values from "that" into "this". If the matrices are different
	/// sizes, any non-overlapping elements will be left at the default value.
	void copyFrom(const GMatrix* that);

	/// Adds a new empty row to this matrix
	void newRow();

	/// Adds n new empty rows to this matrix
	void newRows(size_t n);

	/// Adds a new row to this matrix by copying the parameter row
	void copyRow(SparseVec& row);

	/// Empties the contents of this matrix.
	void clear();

	/// Converts to a full matrix
	GMatrix* toFullMatrix();

	/// Multiplies the matrix by a scalar value
	void multiply(double scalar);

	/// Multiplies this sparse matrix by pThat dense matrix, and returns the resulting dense matrix.
	/// If transposeThat is true, then it multiplies by the transpose of pThat.
	GMatrix* multiply(GMatrix* pThat, bool transposeThat);

	/// Swaps the two specified columns. (This method is a lot slower than swapRows.)
	void swapColumns(size_t a, size_t b);

	/// Swaps the two specified rows. (This method is a lot faster than swapColumns.)
	void swapRows(size_t a, size_t b);

	/// Shuffles the rows in this matrix. If pLabels is non-NULL, then it
	/// will also be shuffled in a manner that preserves corresponding rows with this sparse matrix.
	void shuffle(GRand* pRand, GMatrix* pLabels = NULL);

	/// Returns a sub-matrix of this matrix
	GSparseMatrix* subMatrix(size_t row, size_t col, size_t height, size_t width);

	/// Returns the transpose of this matrix
	GSparseMatrix* transpose();

	/// Performs singular value decomposition. (Takes advantage of sparsity to
	/// perform the decomposition efficiently.) Throws an exception if the default
	/// value is not 0.0.
	void singularValueDecomposition(GSparseMatrix** ppU, double** ppDiag, GSparseMatrix** ppV, bool throwIfNoConverge = false, size_t maxIters = 80);

	/// Computes the first principal component about the origin. (This method
	/// expects the default value to be 0.0.) The size of pOutVector will be the
	/// number of columns in this matrix.
	/// (To compute the next principal component, call RemoveComponent,
	/// then call this method again.)
	void principalComponentAboutOrigin(GVec& outVector, GRand* pRand);

	/// Projects this sparse matrix onto the specified components. Returns a dense matrix.
	GMatrix* project(GMatrix& components);

	/// Delete the last row in this sparse matrix. (Note that you can call swapRows to move any row
	/// you want into the last position before you call this method.)
	void deleteLastRow();

	/// Modifies this matrix such that it now has dimensions 'rows' x 'cols'.
    void resize(size_t rows, size_t cols);

protected:
	void singularValueDecompositionHelper(GSparseMatrix** ppU, double** ppDiag, GSparseMatrix** ppV, bool throwIfNoConverge, size_t maxIters);
};

/// Provides static methods for operating on sparse vectors
class GSparseVec
{
public:
	/// Returns the index of the element with the largest magnitude
	static size_t indexOfMaxMagnitude(SparseVec& sparse);

	/// Computes the dot product of a sparse vector with a dense vector
	static double dotProduct(SparseVec& sparse, GVec& dense);

	/// Computes the dot product of two sparse vectors
	static double dotProduct(SparseVec& a, SparseVec& b);

	/// Returns the number of elements that the two vectors both specify in common
	static size_t count_matching_elements(SparseVec& a, SparseVec& b);
};


} // namespace GClasses

#endif // __GSPARSEMATRIX_H__
