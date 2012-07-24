/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GMATRIX_H__
#define __GMATRIX_H__

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

#include "GError.h"
#include "GHolders.h"


namespace GClasses {

#define UNKNOWN_REAL_VALUE -1e308

// Why do we need a different value for unknown discrete values? Because it's
// common to store discrete values in an integer. Integers can't store -1e308,
// and we can't use -1 for unknown reals b/c it's not an insane value.
#define UNKNOWN_DISCRETE_VALUE -1


class GMatrix;
class GPrediction;
class GRand;
class GHeap;
class GDom;
class GDomNode;
class GArffTokenizer;
class GDistanceMetric;
class GSimpleAssignment;

/// \brief Holds the metadata for a dataset.
///
/// The metadata includes which attributes are continuous or nominal,
/// and how many values each nominal attribute supports.
class GRelation
{
public:
	enum RelationType
	{
		UNIFORM,
		MIXED,
		ARFF
	};

	GRelation() {}
	virtual ~GRelation() {}

	/// \brief Returns the type of relation
	virtual RelationType type() const = 0;

	/// \brief Marshal this object into a DOM, which can then be
	/// converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// \brief Returns the number of attributes (columns)
	virtual size_t size() const = 0;

	/// \brief Returns the number of values in the specified
	/// attribute. (Returns 0 for continuous attributes.)
	virtual size_t valueCount(size_t nAttr) const = 0;

	/// \brief Returns true of all of the attributes in the specified
	/// range are continuous
	virtual bool areContinuous(size_t first = 0, size_t count = (size_t)-1) const = 0;

	/// \brief Returns true of all of the attributes in the specified
	/// range are nominal
	virtual bool areNominal(size_t first, size_t count) const = 0;

	/// \brief Makes a deep copy of this relation
	virtual GRelation* clone() const = 0;

	/// \brief Makes a deep copy of the specified subset of this relation
	virtual GRelation* cloneSub(size_t start, size_t count) const = 0;

	/// \brief Deletes the specified attribute
	virtual void deleteAttribute(size_t index) = 0;

	/// \brief Swaps two attributes
	virtual void swapAttributes(size_t nAttr1, size_t nAttr2) = 0;

	/// \brief Prints as an ARFF file to the specified stream. (pData
	/// can be NULL if data is not available)
	void print(std::ostream& stream, const GMatrix* pData, size_t precision) const;

	/// \brief Prints the specified attribute name to a stream
	virtual void printAttrName(std::ostream& stream, size_t column) const;

	/// \brief Prints the specified value to a stream
	virtual void printAttrValue(std::ostream& stream, size_t column, double value) const;

	/// \brief Returns the name of the attribute with index \a nAttr as
	/// a standard string object or "" if the atribute has no name
	///
	/// \param nAttr the index of the attribute whose name is returned
	///
	/// \return the name of the attribute with index \a nAttr as a
	/// standard string object or "" if the atribute has no name
	virtual std::string attrNameStr(std::size_t nAttr) const{ return ""; }

	/// \brief Returns true iff this and that have the same number of
	/// values for each attribute
	virtual bool isCompatible(const GRelation& that) const;

	/// \brief Print a single row of data in ARFF format
	void printRow(std::ostream& stream, const double* pRow, const char* separator) const;

	/// \brief Counts the size of the corresponding real-space vector
	size_t countRealSpaceDims(size_t nFirstAttr, size_t nAttrCount) const;

	/// \brief Converts a row (pIn) to a real-space vector (pOut) (pIn
	/// should point to the nFirstAttr'th element, not the first
	/// element)
	void toRealSpace(const double* pIn, double* pOut, size_t nFirstAttr, size_t nAttrCount) const;

	/// \brief Converts a real-space vector (pIn) to a row (pOut)
	///
	/// nFirstAttr and nAttrCount refer to the row indexes
	void fromRealSpace(const double* pIn, double* pOut, size_t nFirstAttr, size_t nAttrCount, GRand* pRand) const;

	/// \brief Converts a real-space vector (pIn) to an array of
	/// predictions (pOut)
	///
	/// nFirstAttr and nAttrCount refer to the prediction indexes
	void fromRealSpace(const double* pIn, GPrediction* pOut, size_t nFirstAttr, size_t nAttrCount) const;

	/// \brief Load from a DOM.
	static smart_ptr<GRelation> deserialize(GDomNode* pNode);

	/// \brief Saves to a file
	void save(const GMatrix* pData, const char* szFilename, size_t precision) const;

#ifndef MIN_PREDICT
	/// \brief Performs unit tests for this class. Throws an exception
	/// if there is a failure.
	static void test();
#endif // !MIN_PREDICT

 protected:
	/// \brief Returns a copy of aString modified to escape internal
	/// instances of comma, apostrophe, space, percent, back-slash, and
	/// double-quote
	static std::string quote(const std::string aString);
};

typedef smart_ptr<GRelation> sp_relation;


/// \brief A relation with a minimal memory footprint that assumes all
/// attributes are continuous, or all of them are nominal and have the
/// same number of possible values.
class GUniformRelation : public GRelation
{
protected:
	size_t m_attrCount;
	size_t m_valueCount;

public:
	GUniformRelation(size_t attrCount, size_t valueCount = 0)
	: m_attrCount(attrCount), m_valueCount(valueCount)
	{
	}

	GUniformRelation(GDomNode* pNode);

	virtual RelationType type() const { return UNIFORM; }
	
	/// \brief Serializes this object
	virtual GDomNode* serialize(GDom* pDoc) const;
	
	/// \brief Returns the number of attributes (columns)
	virtual size_t size() const { return m_attrCount; }
	
	/// \brief Returns the number of values in each nominal attribute
	/// (or 0 if the attributes are continuous)
	virtual size_t valueCount(size_t) const { return m_valueCount; }
	
	/// \brief See the comment for GRelation::areContinuous
	virtual bool areContinuous(size_t, size_t) const { return m_valueCount == 0; }
	
	/// \brief See the comment for GRelation::areNominal
	virtual bool areNominal(size_t, size_t) const { return m_valueCount != 0; }

	/// \brief Returns a copy of this object
	virtual GRelation* clone() const { return new GUniformRelation(m_attrCount, m_valueCount); }

	/// \brief Returns a deep copy of the specified subset of this
	/// relation
	virtual GRelation* cloneSub(size_t, size_t count) const { return new GUniformRelation(count, m_valueCount); }

	/// \brief Drop the specified attribute
	virtual void deleteAttribute(size_t index);
	
	/// \brief Swap two attributes (since all attributes are identical, does nothing)
	virtual void swapAttributes(size_t, size_t) {}

	/// \brief Returns true iff this and that have the same number of
	/// values for each attribute
	virtual bool isCompatible(const GRelation& that) const;
};



class GMixedRelation : public GRelation
{
protected:
	std::vector<size_t> m_valueCounts;

public:
	/// \brief Makes an empty relation
	GMixedRelation();

	/// \brief Construct a mixed relation. attrValues specifies the
	/// number of nominal values in each attribute (column), or 0 for
	/// continuous attributes.
	GMixedRelation(std::vector<size_t>& attrValues);

	/// \brief Loads from a DOM.
	GMixedRelation(GDomNode* pNode);

	/// \brief Makes a copy of pCopyMe
	GMixedRelation(const GRelation* pCopyMe);

	/// \brief Makes a copy of the specified range of pCopyMe
	GMixedRelation(const GRelation* pCopyMe, size_t firstAttr, size_t attrCount);

	virtual ~GMixedRelation();

	virtual RelationType type() const { return MIXED; }

	/// \brief Marshalls this object to a DOM, which can be saved to a
	/// variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// \brief Makes a deep copy of this relation
	virtual GRelation* clone() const;

	/// \brief Makes a deep copy of the specified subset of this
	/// relation
	virtual GRelation* cloneSub(size_t start, size_t count) const;

	/// \brief Deletes all the attributes
	virtual void flush();

	/// \brief If nValues is zero, adds a real attribute. If nValues is
	/// > 0, adds an attribute with "nValues" nominal values
	void addAttr(size_t nValues);

	/// \brief Adds "attrCount" new attributes, each with "valueCount"
	/// values. (Use valueCount=0 for continuous attributes.)
	void addAttrs(size_t attrCount, size_t valueCount);

	/// \brief Copies the specified attributes and adds them to this
	/// relation.  If attrCount < 0, then it will copy all attributes
	/// from firstAttr to the end.
	void addAttrs(const GRelation* pCopyMe, size_t firstAttr = 0, size_t attrCount = (size_t)-1);

	/// \brief Flushes this relation and then copies all of the
	/// attributes from pCopyMe
	void copy(const GRelation* pCopyMe);

	/// \brief Adds a copy of the specified attribute to this relation
	virtual void copyAttr(const GRelation* pThat, size_t nAttr);

	/// \brief Returns the total number of attributes in this relation
	virtual size_t size() const
	{
		return m_valueCounts.size();
	}

	/// \brief Returns the number of nominal values in the specified
	/// attribute
	virtual size_t valueCount(size_t nAttr) const
	{
		return m_valueCounts[nAttr];
	}

	/// \brief Sets the number of values for this attribute
	virtual void setAttrValueCount(size_t nAttr, size_t nValues);

	/// \brief Returns true iff all attributes in the specified range are continuous
	virtual bool areContinuous(size_t first = 0, size_t count = (size_t)-1) const;

	/// \brief Returns true iff all attributes in the specified range are nominal
	virtual bool areNominal(size_t first = 0, size_t count = (size_t)-1) const;

	/// \brief Swaps two columns
	virtual void swapAttributes(size_t nAttr1, size_t nAttr2);

	/// \brief Deletes an attribute.
	virtual void deleteAttribute(size_t nAttr);
};


class GArffAttribute
{
public:
	std::string m_name;
	std::vector<std::string> m_values;

	GDomNode* serialize(GDom* pDoc, size_t valCount) const;
};


/// \brief ARFF = Attribute-Relation File Format. This stores richer
/// information than GRelation. This includes a name, a name for each
/// attribute, and names for each supported nominal value.
class GArffRelation : public GMixedRelation
{
friend class GMatrix;
protected:
	std::string m_name;
	std::vector<GArffAttribute> m_attrs;

public:
	/// General-purpose constructor
	GArffRelation();

	/// Deserializing constructor
	GArffRelation(GDomNode* pNode);

	virtual ~GArffRelation();

	virtual RelationType type() const { return ARFF; }

	/// \brief Marshalls this object to a DOM, which can be saved to a
	/// variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// \brief Returns a deep copy of this object
	virtual GRelation* clone() const;

	/// \brief Makes a deep copy of the specified subset of this relation
	virtual GRelation* cloneSub(size_t start, size_t count) const;

	/// \brief Deletes all the attributes
	virtual void flush();

	/// \brief Prints the specified attribute name to a stream
	virtual void printAttrName(std::ostream& stream, size_t column) const;

	/// \brief Prints the specified value to a stream
	virtual void printAttrValue(std::ostream& stream, size_t column, double value) const;

	/// \brief Returns true iff the attributes in both relations have
	/// the same names, the same number of values, and the names of
	/// those values all match. (Empty strings are considered to match
	/// everything.)
	virtual bool isCompatible(const GRelation& that) const;

	/// \brief Adds a new attribute (column) to the relation
	void addAttribute(const char* szName, size_t nValues, std::vector<const char*>* pValues);

	/// \brief Adds a copy of the specified attribute to this relation
	virtual void copyAttr(const GRelation* pThat, size_t nAttr);

	/// \brief Returns the name of the relation
	const char* name() const { return m_name.c_str(); }

	/// \brief Sets the name of this relation
	void setName(const char* szName);

	/// \brief Returns the name of the attribute with index \a nAttr
	///
	/// \param nAttr the index of the attribute whose name is returned
	///
	/// \return the name of the attribute with index \a nAttr as a
	/// standard string object or "" if the atribute has no name
	const char* attrName(size_t nAttr) const;

	/// \brief Returns the name of the attribute with index \a nAttr as
	/// a standard string object or "" if the atribute has no name
	///
	/// \param nAttr the index of the attribute whose name is returned
	///
	/// \return the name of the attribute with index \a nAttr as a
	/// standard string object or "" if the atribute has no name
	virtual std::string attrNameStr(std::size_t nAttr) const { 
		return attrName(nAttr); }

#ifndef MIN_PREDICT
    /// \brief Sets the name of the specified attribute.
    void setAttrName(size_t attr, const char* szNewName);
#endif // MIN_PREDICT

	/// \brief Adds a new possible value to a nominal attribute. Returns
	/// the numerical form of the new value.
	int addAttrValue(size_t nAttr, const char* szValue);

	/// \brief Sets the number of values for the specified attribute
	virtual void setAttrValueCount(size_t nAttr, size_t nValues);

	/// \brief Swaps two columns
	virtual void swapAttributes(size_t nAttr1, size_t nAttr2);

	/// \brief Deletes an attribute
	virtual void deleteAttribute(size_t nAttr);

	/// \brief Returns the nominal index for the specified attribute
	/// with the given value
	int findEnumeratedValue(size_t nAttr, const char* szValue) const;

	/// \brief Parses a value
	double parseValue(size_t attr, const char* val);

	/// \brief Parses the meta-data for an attribute
	void parseAttribute(GArffTokenizer& tok);

#ifndef MIN_PREDICT
	/// \brief Drops the specified value from the list of possible values.
	/// (Swaps the last value in to fill its slot.)
	void dropValue(size_t attr, int val);
#endif // MIN_PREDICT
};

/// \brief Represents a matrix or a database table. 
///
/// Elements can be discrete or continuous.
///
/// References a GRelation object, which stores the meta-information about each column.
class GMatrix
{
protected:
	sp_relation m_pRelation;
	GHeap* m_pHeap;
	std::vector<double*> m_rows;

public:
	/// \brief Construct a rows x cols matrix with all elements of the
	/// matrix assumed to be continuous.
	///
	/// It is okay to initially set rows to 0 and later call newRow to
	/// add each row. Adding columns later, however, is not very
	/// computationally efficient.)
	GMatrix(size_t rows, size_t cols, GHeap* pHeap = NULL);

	/// \brief Construct a matrix with a mixed relation. That is, one
	/// with some continuous attributes (columns), and some nominal
	/// attributes (columns).
	///
	/// attrValues specifies the number of nominal values suppored in
	/// each attribute (column), or 0 for a continuous attribute.
	///
	/// Initially, this matrix will have 0 rows, but you can add more
	/// rows by calling newRow or newRows.
	GMatrix(std::vector<size_t>& attrValues, GHeap* pHeap = NULL);

	/// \brief Create an empty matrix whose attributes/column types are
	/// specified by pRelation
	///
	/// pRelation is a smart-pointer to a relation, which specifies the
	/// type of each attribute (column) in the data set.
	///
	/// Initially, this matrix will have 0 rows, but you can add more
	/// rows by calling newRow or newRows.
	GMatrix(sp_relation& pRelation, GHeap* pHeap = NULL);

	///\brief Copy-constructor
	///
	///Copies \a orig, making a new relation object and new storage for
	///the rows (with the same content), but uses the same GHeap object
	///as \a orig
	///
	///\param orig the GMatrix object to copy
	GMatrix(const GMatrix& orig);


	//I put the operator= right after the copy constructor because you
	//should always have both or neither in a class, this makes that
	//easy to verify

	///\brief Make *this into a copy of orig
	///
	///Copies \a orig, making a new relation object and new storage for
	///the rows (with the same content), but uses the same GHeap object
	///as \a orig
	///
	///\param orig the GMatrix object to copy
	///
	///\return a reference to this GMatrix object
	GMatrix& operator=(const GMatrix& orig);


	/// \brief Load from a DOM.
	GMatrix(GDomNode* pNode, GHeap* pHeap = NULL);

	~GMatrix();

	/// \brief Returns true iff all the entries in *this and \a
	/// other are identical and their relations are compatible, and they
	/// are the same size
	///
	/// \return true iff all the entries in *this and \a other are
	/// identical, their relations are compatible, and they are the same
	/// size
	bool operator==(const GMatrix& other) const;

	/// \brief Adds a new row to the dataset. (The values in the row are
	/// not initialized)
	double* newRow();

	/// \brief Adds "nRows" uninitialized rows to the data set
	void newRows(size_t nRows);

	/// \brief Matrix add
	///
	/// Adds the values in pThat to this. (If transpose is true, adds
	/// the transpose of pThat to this.) Both datasets must have the
	/// same dimensions. Behavior is undefined for nominal columns.
	void add(GMatrix* pThat, bool transpose);

	/// \brief Returns a new dataset that contains a subset of the
	/// attributes in this dataset
	GMatrix* attrSubset(size_t firstAttr, size_t attrCount);

	/// \brief This computes the square root of this matrix. (If you
	/// take the matrix that this returns and multiply it by its
	/// transpose, you should get the original dataset again.)
	/// (Returns a lower-triangular matrix.)
	///
	/// Behavior is undefined if there are nominal attributes. If
	/// tolerant is true, it will return even if it cannot compute
	/// accurate results. If tolerant is false (the default) and this
	/// matrix is not positive definate, it will throw an exception.
	GMatrix* cholesky(bool tolerant = false);

	/// \brief Makes a deep copy of this dataset
	GMatrix* clone();

	/// \brief Makes a deep copy of the specified rectangular region of
	/// this matrix
	GMatrix* cloneSub(size_t rowStart, size_t colStart, size_t rowCount, size_t colCount);

	/// \brief Copies the specified column into pOutVector
	void col(size_t index, double* pOutVector);

	/// \brief Returns the number of columns in the dataset
	size_t cols() const { return m_pRelation->size(); }

	/// \brief Copies all the data from pThat. (Just references the same
	/// relation)
	void copy(const GMatrix* pThat);

	/// \brief Copies the specified block of columns from pSource to
	/// this dataset. 
	///
	/// pSource must have the same number of rows as this dataset.
	void copyColumns(size_t nDestStartColumn, const GMatrix* pSource, size_t nSourceStartColumn, size_t nColumnCount);

	/// \brief Adds a copy of the row to the data set
	void copyRow(const double* pRow);

	/// \brief Computes the determinant of this matrix
	double determinant();

#ifndef MIN_PREDICT
	/// \brief Drops any occurrences of the specified value, and removes it as a possible value
	void dropValue(size_t attr, int val);
#endif // MIN_PREDICT

	/// \brief Computes the eigenvalue that corresponds to the specified
	/// eigenvector of this matrix
	double eigenValue(const double* pEigenVector);

	/// \brief Computes the eigenvector that corresponds to the
	/// specified eigenvalue of this matrix. Note that this method
	/// trashes this matrix, so make a copy first if you care.
	void eigenVector(double eigenvalue, double* pOutVector);

	/// \brief Computes y in the equation M*y=x (or y=M^(-1)x), where M
	/// is this dataset, which must be a square matrix, and x is pVector
	/// as passed in, and y is pVector after the call.
	///
	/// If there are multiple solutions, it finds the one for which all
	/// the variables in the null-space have a value of 1. If there are
	/// no solutions, it returns false. Note that this method trashes
	/// this dataset (so make a copy first if you care).
	bool gaussianElimination(double* pVector);

	/// \brief Returns the heap used to allocate rows for this dataset
	GHeap* heap() { return m_pHeap; }

	/// \brief Performs an in-place LU-decomposition, such that the
	/// lower triangle of this matrix (including the diagonal) specifies
	/// L, and the uppoer triangle of this matrix (not including the
	/// diagonal) specifies U, and all values of U along the diagonal
	/// are ones. (The upper triangle of L and the lower triangle of U
	/// are all zeros.)
	void LUDecomposition();

	/// \brief This computes K=kabsch(A,B), such that K is an n-by-n
	/// matrix, where n is pA->cols().  K is the optimal orthonormal
	/// rotation matrix to align A and B, such that A(K^T) minimizes
	/// sum-squared error with B, and BK minimizes sum-squared error
	/// with A.  (This rotates around the origin, so typically you will
	/// want to subtract the centroid from both pA and pB before calling
	/// this.)
	static GMatrix* kabsch(GMatrix* pA, GMatrix* pB);

	/// \brief This uses the Kabsch algorithm to rotate and translate pB
	/// in order to minimize RMS with pA. (pA and pB must have the same
	/// number of rows and columns.)
	static GMatrix* align(GMatrix* pA, GMatrix* pB);


#ifndef MIN_PREDICT
	/// \brief Loads an ARFF file and returns the data. This will throw
	/// an exception if there's an error.
	static GMatrix* loadArff(const char* szFilename);

	/// \brief Loads a file in CSV format.
	static GMatrix* loadCsv(const char* szFilename, char separator, bool columnNamesInFirstRow, bool tolerant);
#endif // MIN_PREDICT

	/// \brief Sets this dataset to an identity matrix. (It doesn't
	/// change the number of columns or rows. It just stomps over
	/// existing values.)
	void makeIdentity();

	/// \brief copies one of the triangular submatrices over the other,
	/// making a symmetric matrix.
	///
	/// \param upperToLower If true, copies the upper triangle of this
	///                     matrix over the lower triangle.  Otherwise,
	///                     copies the lower triangle of this matrix
	///                     over the upper triangle
	void mirrorTriangle(bool upperToLower);

	/// \brief Merges two datasets side-by-side. The resulting dataset
	/// will contain the attributes of both datasets. Both pSetA and
	/// pSetB (and the resulting dataset) must have the same number of
	/// rows
	static GMatrix* mergeHoriz(GMatrix* pSetA, GMatrix* pSetB);

	/// \brief Steals all the rows from pData and adds them to this set.
	/// (You still have to delete pData.) Both datasets must have the
	/// same number of columns.
	void mergeVert(GMatrix* pData);

	/// \brief Computes nCount eigenvectors and the corresponding
	/// eigenvalues using the power method (which is only accurate if a
	/// small number of eigenvalues/vectors are needed.)
	///
	/// If mostSignificant is true, the largest eigenvalues are
	/// found. If mostSignificant is false, the smallest eigenvalues are
	/// found.
	GMatrix* eigs(size_t nCount, double* pEigenVals, GRand* pRand, bool mostSignificant);

	/// \brief Multiplies every element in the dataset by scalar.
	/// Behavior is undefined for nominal columns.
	void multiply(double scalar);

	/// \brief Multiplies this matrix by the column vector pVectorIn to
	/// get pVectorOut.
	///
	/// (If transpose is true, then it multiplies the transpose of this
	/// matrix by pVectorIn to get pVectorOut.) 
	///
	/// pVectorIn should have
	/// the same number of elements as columns (or rows if transpose is
	/// true) 
	///
	/// pVectorOut should have the same number of elements as
	/// rows (or cols, if transpose is true.) 
	///
	/// \note if transpose is true, then pVectorIn is treated as a
	/// row vector and is multiplied by this matrix to get pVectorOut.
	void multiply(const double* pVectorIn, double* pVectorOut, bool transpose = false);

	/// \brief Matrix multiply. 
	///
	/// For convenience, you can also specify that neither, one, or both
	/// of the inputs are virtually transposed prior to the
	/// multiplication. (If you want the results to come out transposed,
	/// you can use the equality (AB)^T=(B^T)(A^T) to figure out how to
	/// specify the parameters.)
	static GMatrix* multiply(GMatrix& a, GMatrix& b, bool transposeA, bool transposeB);

#ifndef MIN_PREDICT
	/// \brief Parses an ARFF file and returns the data. 
	///
	/// This will throw an exception if there's an error.
	static GMatrix* parseArff(const char* szFile, size_t nLen);

	///\brief Imports data from a text file. Determines the meta-data
	///automatically.
	///
	///\note This method does not support Mac line-endings. You should
	///      first replace all '\\r' with '\\n' if your data comes from
	///      a Mac.  As a special case, if separator is '\\0', then it
	///      assumes data elements are separated by any number of
	///      whitespace characters, that element values themselves
	///      contain no whitespace, and that there are no missing
	///      elements. (This is the case when you save a Matlab matrix
	///      to an ascii file.)
	static GMatrix* parseCsv(const char* pFile, size_t len, char separator, bool columnNamesInFirstRow, bool tolerant = false);
#endif // MIN_PREDICT

	/// \brief Computes the Moore-Penrose pseudoinverse of this matrix
	/// (using the SVD method). You are responsible to delete the
	/// matrix this returns.
	GMatrix* pseudoInverse();

	/// \brief Returns a relation object, which holds meta-data about
	/// the attributes (columns)
	sp_relation& relation() { return m_pRelation; }

	/// \brief Returns a relation object, which holds meta-data about
	/// the attributes (columns) (const version)
	smart_ptr<const GRelation> relation() const { 
		return smart_ptr<const GRelation> (m_pRelation, NULL);
	}

	/// \brief Allocates space for the specified number of patterns (to
	/// avoid superfluous resizing)
	void reserve(size_t n) { m_rows.reserve(n); }

	/// \brief Returns the number of rows in the dataset
	size_t rows() const { return m_rows.size(); }

#ifndef MIN_PREDICT
	/// \brief Saves the dataset to a file in ARFF format
	void saveArff(const char* szFilename);
#endif // MIN_PREDICT

	/// \brief Sets the relation for this dataset
	void setRelation(sp_relation& pRelation) { m_pRelation = pRelation; }

	/// \brief Performs SVD on A, where A is this m-by-n matrix.
	///
	/// You are responsible to delete(*ppU), delete(*ppV), and delete[]
	/// *ppDiag.
	///
	/// \param ppU *ppU will be set to an m-by-m matrix where the
	///            columns are the *eigenvectors of A(A^T).
	///
	/// \param ppDiag *ppDiag will be set to an array of n doubles
	///               holding the square roots of the corresponding
	///               eigenvalues.
	///
	/// \param ppV *ppV will be set to an n-by-n matrix where the rows
	///             are the eigenvectors of (A^T)A.
	///
	/// \param throwIfNoConverge if true, throws an exception if the SVD
	///                          solver does not converge.  does nothing
	///                          otherwise
	///
	/// \param maxIters the maximum number of iterations to perform in
	///                 the SVD solver
	void singularValueDecomposition(GMatrix** ppU, double** ppDiag, GMatrix** ppV, bool throwIfNoConverge = false, size_t maxIters = 80);

	/// \brief Matrix subtract. Subtracts the values in *pThat from *this. 
	///
	/// (If transpose is true, subtracts the transpose of *pThat from
	/// this.) Both datasets must have the same dimensions. Behavior is
	/// undefined for nominal columns.
	///
	/// \param pThat pointer to the matrix to subtract from *this
	///
	/// \param transpose If true, the transpose of *pThat is subtracted.
	///                  If false, *pThat is subtracted
	void subtract(GMatrix* pThat, bool transpose);

	/// \brief Returns the sum squared difference between this matrix
	/// and an identity matrix
	double sumSquaredDiffWithIdentity();

	/// \brief Adds an already-allocated row to this dataset. The row must
	/// be allocated in the same heap that this dataset uses. (There is no way
	/// for this method to verify that, so be careful.)
	void takeRow(double* pRow);

	/// \brief Converts the matrix to reduced row echelon form
	size_t toReducedRowEchelonForm();

	/// \brief Copies all the data from this dataset into pVector. 
	///
	/// pVector must be big enough to hold rows() x cols() doubles.
	void toVector(double* pVector);

#ifndef MIN_PREDICT
	/// \brief Marshalls this object to a DOM, which may be saved to a variety of serial formats.
	GDomNode* serialize(GDom* pDoc) const;
#endif // MIN_PREDICT

	/// \brief Returns the sum of the diagonal elements
	double trace();

	/// \brief Returns a pointer to a new dataset that is this dataset
	/// transposed. (All columns in the returned dataset will be
	/// continuous.)
	///
	/// The returned matrix is newly allocated on the system heap with
	/// operator new and must be deleted by the caller.
	///
	/// \return A pointer to a new dataset that is this dataset
	///         transposed. All columns in the returned dataset will be
	///         continuous.  The caller is responsible for deleting the
	///         returned dataset.
	GMatrix* transpose();

	/// \brief Copies the data from pVector over this dataset. 
	///
	/// nRows specifies the number of rows of data in pVector.
	void fromVector(const double* pVector, size_t nRows);

	/// \brief Returns a pointer to the specified row
	inline double* row(size_t index) { return m_rows[index]; }

	/// \brief Returns a pointer to the specified row
	inline double* operator [](size_t index) { return m_rows[index]; }

	/// \brief Returns a const pointer to the specified row
	inline const double* row(size_t index) const { return m_rows[index]; }

	/// \brief Returns a const pointer to the specified row
	inline const double* operator [](size_t index) const { 
	  return m_rows[index]; }

	/// \brief Sets all elements in this dataset to the specified value
	void setAll(double val);

	/// \brief Copies pVector over the specified column
	void setCol(size_t index, const double* pVector);

	/// \brief Swaps the two specified rows
	void swapRows(size_t a, size_t b);

	/// \brief Swaps two columns
	void swapColumns(size_t nAttr1, size_t nAttr2);

	/// \brief Deletes a column
	void deleteColumn(size_t index);

	/// \brief Swaps the specified row with the last row, and then
	/// releases it from the dataset.
	///
	/// If this dataset does not have its own heap, then you must delete
	/// the row this returns
	double* releaseRow(size_t index);

	/// \brief Swaps the specified row with the last row, and then deletes it.
	void deleteRow(size_t index);

	/// \brief Releases the specified row from the dataset and shifts
	/// everything after it up one slot.
	///
	/// If this dataset does not have its own heap, then you must delete
	/// the row this returns
	double* releaseRowPreserveOrder(size_t index);

	/// \brief Deletes the specified row and shifts everything after it
	/// up one slot
	void deleteRowPreserveOrder(size_t index);

	/// \brief Replaces any occurrences of NAN in the matrix with the
	/// corresponding values from an identity matrix.
	void fixNans();

	/// \brief Deletes all the data
	void flush();

	/// \brief Abandons (leaks) all the rows of data
	void releaseAllRows();

	/// \brief Randomizes the order of the rows. 
	///
	/// If pExtension is non-NULL, then it will also be shuffled such
	/// that corresponding rows are preserved.
	void shuffle(GRand& rand, GMatrix* pExtension = NULL);

	/// \brief Shuffles the order of the rows. Also shuffles the rows in
	/// "other" in the same way, such that corresponding rows are
	/// preserved.
	void shuffle2(GRand& rand, GMatrix& other);

	/// \brief This is an inferior way to shuffle the data
	void shuffleLikeCards();

	/// \brief Sorts the data from smallest to largest in the specified
	/// dimension
	void sort(size_t nDimension);

	/// This partially sorts the specified column, such that the specified row
	/// will contain the same row as if it were fully sorted, and previous
	/// rows will contain a value <= to it in that column, and later rows
	/// will contain a value >= to it in that column. Unlike sort, which
	/// has O(m*log(m)) complexity, this method has O(m) complexity. This might
	/// be useful, for example, for efficiently finding the row with a median
	/// value in some attribute, or for separating data by a threshold in
	/// some value.
	void sortPartial(size_t row, size_t col);

	/// \brief Reverses the row order
	void reverseRows();

	/// \brief Sorts rows according to the specified compare
	/// function. (Return true to indicate that the first row comes
	/// before the second row.)
	template<typename CompareFunc>
	void sort(CompareFunc& pComparator)
	{
		std::sort(m_rows.begin(), m_rows.end(), pComparator);
	}

	/// \brief Splits this set of data into two sets. Values
	/// greater-than-or-equal-to dPivot stay in this data set. Values
	/// less than dPivot go into pLessThanPivot
	///
	/// If pExtensionA is non-NULL, then it will also split pExtensionA
	/// such that corresponding rows are preserved.
	void splitByPivot(GMatrix* pGreaterOrEqual, size_t nAttribute, double dPivot, GMatrix* pExtensionA = NULL, GMatrix* pExtensionB = NULL);

	/// \brief Moves all rows with the specified value in the specified
	/// attribute into pSingleClass
	///
	/// If pExtensionA is non-NULL, then it will also split pExtensionA
	/// such that corresponding rows are preserved.
	void splitByNominalValue(GMatrix* pSingleClass, size_t nAttr, int nValue, GMatrix* pExtensionA = NULL, GMatrix* pExtensionB = NULL);

	/// \brief Removes the last nOtherRows rows from this data set and
	/// puts them in pOtherData
	void splitBySize(GMatrix* pOtherData, size_t nOtherRows);

	/// \brief Measures the entropy of the specified attribute
	double entropy(size_t nColumn);

	/// \brief Finds the min and the range of the values of the
	/// specified attribute
	void minAndRange(size_t nAttribute, double* pMin, double* pRange);

	/// \brief Estimates the actual min and range based on a random sample
	void minAndRangeUnbiased(size_t nAttribute, double* pMin, double* pRange);

	/// \brief Shifts the data such that the mean occurs at the origin.
	/// Only continuous values are affected.  Nominal values are left
	/// unchanged.
	void centerMeanAtOrigin();

	/// \brief Computes the arithmetic mean of the values in the
	/// specified column
	double mean(size_t nAttribute);

#ifndef MIN_PREDICT
	/// \brief Computes the median of the values in the specified column
	double median(size_t nAttribute);
#endif // MIN_PREDICT

	/// \brief Computes the arithmetic means of all attributes
	void centroid(double* pOutCentroid);

	/// \brief Computes the average variance of a single attribute
	double variance(size_t nAttr, double mean);

	/// \brief Normalizes the specified attribute values
	void normalize(size_t nAttribute, double dInputMin, double dInputRange, double dOutputMin, double dOutputRange);

	/// \brief Normalize a value from the input min and range to the
	/// output min and range
	static double normalize(double dVal, double dInputMin, double dInputRange, double dOutputMin, double dOutputRange);

	/// \brief Returns the mean if the specified attribute is
	/// continuous, otherwise returns the most common nominal value in
	/// the attribute.
	double baselineValue(size_t nAttribute);

	/// \brief Returns true iff the specified attribute contains
	/// homogenous values. (Unknowns are counted as homogenous with
	/// anything)
	bool isAttrHomogenous(size_t col);

	/// \brief Returns true iff each of the last labelDims columns in
	/// the data are homogenous
	bool isHomogenous();

	/// \brief Replace missing values with the appropriate measure of
	/// central tendency.
	///
	/// If the specified attribute is continuous, replaces all
	/// missing values in that attribute with the mean.  If the
	/// specified attribute is nominal, replaces all missing values in
	/// that attribute with the most common value.
	void replaceMissingValuesWithBaseline(size_t nAttr);

	/// \brief Replaces all missing values by copying a randomly
	/// selected non-missing value in the same attribute.
	void replaceMissingValuesRandomly(size_t nAttr, GRand* pRand);

	/// \brief This is an efficient algorithm for iteratively computing
	/// the principal component vector (the eigenvector of the
	/// covariance matrix) of the data.
	///
	/// See "EM Algorithms for PCA and SPCA"
	/// by Sam Roweis, 1998 NIPS.
	///
	/// The size of pOutVector will be the number of columns in this matrix.
	/// (To compute the next principal component, call RemoveComponent,
	/// then call this method again.)
	void principalComponent(double* pOutVector, const double* pMean, GRand* pRand);

	/// \brief Computes the first principal component assuming the mean
	/// is already subtracted out of the data
	void principalComponentAboutOrigin(double* pOutVector, GRand* pRand);

	/// \brief Computes principal components, while ignoring missing
	/// values
	void principalComponentIgnoreUnknowns(double* pOutVector, const double* pMean, GRand* pRand);

	/// \brief Computes the first principal component of the data with
	/// each row weighted according to the vector pWeights. (pWeights
	/// must have an element for each row.)
	void weightedPrincipalComponent(double* pOutVector, const double* pMean, const double* pWeights, GRand* pRand);

	/// \brief Computes the eigenvalue that corresponds to \a *pEigenvector.
	///
	/// After you compute the principal component, you can call this to
	/// obtain the eigenvalue that corresponds to that principal
	/// component vector (eigenvector).
	double eigenValue(const double* pMean, const double* pEigenVector, GRand* pRand);

	/// \brief Removes the component specified by pComponent from the
	/// data. (pComponent should already be normalized.)
	///
	/// This might be useful, for example, to remove the first principal
	/// component from the data so you can then proceed to compute the
	/// second principal component, and so forth.
	void removeComponent(const double* pMean, const double* pComponent);

	/// \brief Removes the specified component assuming the mean is zero.
	void removeComponentAboutOrigin(const double* pComponent);

	/// \brief Computes the minimum number of principal components
	/// necessary so that less than the specified portion of the
	/// deviation in the data is unaccounted for.
	///
	/// For example, if the data projected onto the first 3 principal
	/// components contains 90 percent of the deviation that the
	/// original data contains, then if you pass the value 0.1 to this
	/// method, it will return 3.
	size_t countPrincipalComponents(double d, GRand* pRand);

	/// \brief Computes the sum-squared distance between pPoint and all
	/// of the points in the dataset.
	///
	/// If pPoint is NULL, it computes the sum-squared distance with the origin.
	///
	/// \note that this is equal to the sum of all the eigenvalues times
	///       the number of dimensions, so you can efficiently compute
	///       eigenvalues as the difference in sumSquaredDistance with
	///       the mean after removing the corresponding component, and
	///       then dividing by the number of dimensions. This is more
	///       efficient than calling eigenValue.
	double sumSquaredDistance(const double* pPoint);

	/// \brief Computes the sum-squared distance between the specified
	/// column of this and that. If the column is a nominal attribute,
	/// then Hamming distance is used.
	double columnSumSquaredDifference(GMatrix& that, size_t col);

	/// \brief Computes the squared distance between this and that.
	///
	/// If transpose is true, computes the difference between this and
	/// the transpose of that.
	double sumSquaredDifference(GMatrix& that, bool transpose = false);

	/// \brief Computes the linear coefficient between the two specified
	/// attributes.
	///
	/// Usually you will want to pass the mean values for attr1Origin
	/// and attr2Origin.
	double linearCorrelationCoefficient(size_t attr1, double attr1Origin, size_t attr2, double attr2Origin);

	/// \brief Computes the covariance between two attributes
	double covariance(size_t nAttr1, double dMean1, size_t nAttr2, double dMean2);

	/// \brief Computes the covariance matrix of the data
	GMatrix* covarianceMatrix();

	/// \brief Performs a paired T-Test with data from the two specified
	/// attributes.
	///
	/// pOutV will hold the degrees of freedom. pOutT will hold the T-value.
	/// You can use GMath::tTestAlphaValue to convert these to a P-value.
	void pairedTTest(size_t* pOutV, double* pOutT, size_t attr1, size_t attr2, bool normalize);

	/// \brief Performs the Wilcoxon signed ranks test from the two
	/// specified attributes.
	///
	/// If two values are closer than tolerance, they are considered to
	/// be equal.
	void wilcoxonSignedRanksTest(size_t attr1, size_t attr2, double tolerance, int* pNum, double* pWMinus, double* pWPlus);

	/// \brief Prints the data to the specified stream
	void print(std::ostream& stream);

	/// \brief Returns the number of ocurrences of the specified value
	/// in the specified attribute
	size_t countValue(size_t attribute, double value);

	/// \brief Returns true iff this matrix is missing any values.
	bool doesHaveAnyMissingValues();

	/// \brief Throws an exception if this data contains any missing
	/// values in a continuous attribute
	void ensureDataHasNoMissingReals();

	/// \brief Throws an exception if this data contains any missing
	/// values in a nominal attribute
	void ensureDataHasNoMissingNominals();

	/// \brief Computes the sum entropy of the data (or the sum variance
	/// for continuous attributes)
	double measureInfo();

	/// \brief Computes the vector in this subspace that has the
	/// greatest distance from its projection into pThat subspace.
	///
	/// Returns true if the results are computed. 
	///
	/// Returns false if the subspaces are so nearly parallel that pOut
	/// cannot be computed with accuracy.
	bool leastCorrelatedVector(double* pOut, GMatrix* pThat, GRand* pRand);

	/// \brief Computes the cosine of the dihedral angle between this
	/// subspace and pThat subspace
	double dihedralCorrelation(GMatrix* pThat, GRand* pRand);

	/// \brief Projects pPoint onto this hyperplane (where each row
	/// defines one of the orthonormal basis vectors of this hyperplane)
	///
	/// This computes (A^T)Ap, where A is this matrix, and p is pPoint.
	void project(double* pDest, const double* pPoint);

	/// \brief Projects pPoint onto this hyperplane (where each row
	/// defines one of the orthonormal basis vectors of this hyperplane)
	void project(double* pDest, const double* pPoint, const double* pOrigin);

	/// \brief Performs a bipartite matching between the rows of \a a
	/// and \a b using the Linear Assignment Problem (LAP) optimizer
	///
	/// Treats the rows of the matrices \a a and \a b as vectors and
	/// calculates the distances between these vectors using \a cost.
	/// Returns an optimal assignment from rows of \a a to rows of \a b
	/// that minimizes sum of the costs of the assignments.
	///
	/// Each row is considered to be a vector in multidimensional space.
	/// The cost is the distance given by \a cost when called on each
	/// row of \a a and row of \a b in turn.  The cost must not be
	/// \f$-\infty\f$ for any pair of rows.  Other than that, there are no
	/// limitations on the cost function.
	///
	/// Because of the limitations of GDistanceMetric, \a a and \a b
	/// must have the same number of columns.
	///
	/// If \f$m\f$ is \f$\max(rows(a), rows(b))\f$ then this routine
	/// requires \f$\Theta(rows(a) \cdot rows(b))\f$ memory and
	/// \f$O(m^3)\f$ time.
	///
	/// \param a the matrix containing the vectors of set a.  Must have
	///          the same number of columns as the matrix containing the
	///          vectors of set b.  Each row is considered to be a
	///          vector in multidimensional space.
	///
	/// \param b the matrix containing the vectors of set b.  Must have
	///          the same number of columns as the matrix containing the
	///          vectors of set a.  Each row is considered to be a
	///          vector in multidimensional space.
	///
	/// \param metric given a row of \a a and a row of \a b, returns the
	///             cost of assigning a to b.
	///
	/// \return the optimal assignment in which each of the rows of \a a
	///         or \a b (whichever has fewer rows) is assigned to a row
	///         of the other matrix
	static GSimpleAssignment bipartiteMatching(GMatrix& a, GMatrix& b, GDistanceMetric& metric);

#ifndef MIN_PREDICT
	/// \brief Performs unit tests for this class. Throws an exception
	/// if there is a failure.
	static void test();
#endif // MIN_PREDICT
protected:
	double determinantHelper(size_t nEndRow, size_t* pColumnList);
	void inPlaceSquareTranspose();
	void singularValueDecompositionHelper(GMatrix** ppU, double** ppDiag, GMatrix** ppV, bool throwIfNoConverge, size_t maxIters);
};



/// \brief This is a special holder that guarantees the data set will
/// release all of its data before it is deleted
class GReleaseDataHolder
{
protected:
	GMatrix* m_pData;

public:
	GReleaseDataHolder(GMatrix* pData)
	{
		m_pData = pData;
	}

	~GReleaseDataHolder()
	{
		m_pData->releaseAllRows();
	}
};

/// \brief This class guarantees that the rows in b are merged
/// vertically back into a when this object goes out of scope.
class GMergeDataHolder
{
protected:
	GMatrix& m_a;
	GMatrix& m_b;

public:
	GMergeDataHolder(GMatrix& a, GMatrix& b) : m_a(a), m_b(b) {}
	~GMergeDataHolder()
	{
		m_a.mergeVert(&m_b);
	}
};


/// \brief Represents an array of matrices or datasets that all have
/// the same number of columns.
class GMatrixArray
{
protected:
	sp_relation m_pRelation;
	std::vector<GMatrix*> m_sets;

public:
	GMatrixArray(sp_relation& pRelation);
	GMatrixArray(size_t cols);
	~GMatrixArray();
	std::vector<GMatrix*>& sets() { return m_sets; }

	/// \brief Adds a new dataset to the array and preallocates the
	/// specified number of rows
	GMatrix* newSet(size_t rows);

	/// \brief Adds count new datasets to the array, and preallocates
	/// the specified number of rows in each one
	void newSets(size_t count, size_t rows);

	/// \brief Deletes all the datasets
	void flush();

	/// \brief Returns the index of the largest data set
	size_t largestSet();

	/// \brief Returns the number of data sets with zero rows
	size_t countEmptySets();
};

} // namespace GClasses

#endif // __GMATRIX_H__
