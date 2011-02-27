/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GSPARSEMATRIX_H__
#define __GSPARSEMATRIX_H__

#include <map>
#include <iostream>

namespace GClasses {

class GMatrix;
class GRand;
class GTwtNode;
class GTwtDoc;

/// This class stores a row-compressed sparse matrix. That is,
/// each row consists of a map from a column-index to a value.
class GSparseMatrix
{
protected:
	typedef std::map<size_t,double> Map;
	size_t m_rows, m_cols;
	Map* m_pRows;
	double m_defaultValue;

public:
	/// Construct a sparse matrix with the specified number of rows and columns.
	/// defaultValue specifies the common value that is not stored. (Typically,
	/// defaultValue is 0, but for some applications it may make more sense to
	/// set it to UNKNOWN_REAL_VALUE.)
	GSparseMatrix(size_t rows, size_t cols, double defaultValue = 0.0);

	/// Deserializes a sparse matrix
	GSparseMatrix(GTwtNode* pNode);

	~GSparseMatrix();

#ifndef NO_TEST_CODE
	static void test();
#endif

	typedef Map::const_iterator Iter;

	/// Serializes this object
	GTwtNode* toTwt(GTwtDoc* pDoc);

	/// Returns the default value--the common value that is not stored.
	double defaultValue() { return m_defaultValue; }

	/// Returns the number of rows (as if this matrix were dense)
	size_t rows() { return m_rows; }

	/// Returns the number of columns (as if this matrix were dense)
	size_t cols() { return m_cols; }

	/// Copies a row into a non-sparse vector
	void fullRow(double* pOutFullRow, size_t row);

	/// Returns a const_iterator to the beginning of a row. The iterator
	/// references a pair, such that first is the column, and second is the value.
	Iter rowBegin(size_t i) { return m_pRows[i].begin(); }

	/// Returns a const_iterator to the end of a row. The iterator
	/// references a pair, such that first is the column, and second is the value.
	Iter rowEnd(size_t i) { return m_pRows[i].end(); }

	/// Returns the specified sparse row.
	std::map<size_t,double>& row(size_t i) { return m_pRows[i]; }

	/// Returns the number of non-default-valued elements in the specified row.
	size_t rowNonDefValues(size_t i) { return m_pRows[i].size(); }

	/// Returns the value at the specified position in the matrix. Returns the
	/// default value if no element is stored at that position.
	double get(size_t row, size_t col);

	/// Sets a value at the specified position in the matrix. (If val is the default
	/// value, it removes the element from the matrix.)
	void set(size_t row, size_t col, double val);

	/// Copies values from "that" into "this". Any default-valued elements in
	/// that will be left the same. Any non-default-valued elements will be
	/// copied over the value in this. If the matrices are different
	/// sizes, any non-overlapping elements will be left at the default value,
	/// no-matter what value it has in that.
	void copyFrom(GSparseMatrix* that);

	/// Copies values from "that" into "this". If the matrices are different
	/// sizes, any non-overlapping elements will be left at the default value.
	void copyFrom(GMatrix* that);

	/// Converts to a full matrix
	GMatrix* toFullMatrix();

	/// Multiplies the matrix by a scalar value
	void multiply(double scalar);

	/// Swaps the two specified columns. (This method is a lot slower than swapRows.)
	void swapColumns(size_t a, size_t b);

	/// Swaps the two specified rows. (This method is a lot faster than swapColumns.)
	void swapRows(size_t a, size_t b);

	/// Shuffles the rows in this matrix
	void shuffle(GRand* pRand);

	/// Returns a sub-matrix of this matrix
	GSparseMatrix* subMatrix(int row, int col, int height, int width);
};

} // namespace GClasses

#endif // __GSPARSEMATRIX_H__
