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

#ifndef __GMATRIX_H__
#define __GMATRIX_H__

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

#include "GError.h"
#include "GVec.h"


namespace GClasses {

#define UNKNOWN_REAL_VALUE -1e308

// Why do we need a different value for unknown discrete values? Because it's
// common to store discrete values in an integer. Integers can't store -1e308,
// and we can't use -1 for unknown reals b/c it's not an insane value.
#define UNKNOWN_DISCRETE_VALUE -1


class GMatrix;
class GPrediction;
class GRand;
class GDom;
class GDomNode;
class GArffTokenizer;
class GDistanceMetric;
class GSimpleAssignment;
class GDistanceMetric;
class GTokenizer;


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

	/// \brief Returns true iff all of the attributes in the specified
	/// range are continuous. The default range is all of them.
	virtual bool areContinuous(size_t first = 0, size_t count = INVALID_INDEX) const = 0;

	/// \brief Returns true iff all of the attributes in the specified
	/// range are nominal. The default range is all of them.
	virtual bool areNominal(size_t first = 0, size_t count = INVALID_INDEX) const = 0;

	/// \brief Makes a deep copy of this relation
	virtual GRelation* clone() const = 0;

	/// \brief Returns a relation containing the same value counts as this object, but
	/// not containing superfluous things, such as human-readable strings. (The returned
	/// relation may be of a different type.)
	virtual GRelation* cloneMinimal() const = 0;

	/// \brief Makes a deep copy of the specified subset of this relation
	virtual GRelation* cloneSub(size_t start, size_t count) const = 0;

	/// \brief Deletes the specified attribute
	virtual void deleteAttributes(size_t index, size_t count) = 0;

	/// \brief Swaps two attributes
	virtual void swapAttributes(size_t nAttr1, size_t nAttr2) = 0;

	/// \brief Prints this relation in ARFF format to the specified stream
	void print(std::ostream& stream) const;

	/// \brief Prints the specified attribute name to a stream
	virtual void printAttrName(std::ostream& stream, size_t column) const;

	/// \brief Prints the specified value to a stream
	virtual void printAttrValue(std::ostream& stream, size_t column, double value, const char* missing = "?") const;

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
	void printRow(std::ostream& stream, const double* pRow, char separator = ',', const char* missing = "?") const;

	/// \brief Load from a DOM.
	static GRelation* deserialize(const GDomNode* pNode);

	/// \brief Saves to a file
	void save(const GMatrix* pData, const char* szFilename) const;

	/// \brief Performs unit tests for this class. Throws an exception
	/// if there is a failure.
	static void test();

};


/// \brief A relation with a minimal memory footprint that assumes all
/// attributes are continuous, or all of them are nominal and have the
/// same number of possible values.
class GUniformRelation : public GRelation
{
protected:
	size_t m_attrCount;
	size_t m_valueCount;

public:
	GUniformRelation(size_t attrCount, size_t values = 0)
	: m_attrCount(attrCount), m_valueCount(values)
	{
	}

	GUniformRelation(const GDomNode* pNode);

	virtual RelationType type() const { return UNIFORM; }

	/// \brief Serializes this object
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// \brief Returns the number of attributes (columns)
	virtual size_t size() const { return m_attrCount; }

	/// \brief Returns the number of values in each nominal attribute
	/// (or 0 if the attributes are continuous)
	virtual size_t valueCount(size_t) const { return m_valueCount; }

	/// \brief See the comment for GRelation::areContinuous
	virtual bool areContinuous(size_t first = 0, size_t count = INVALID_INDEX) const { return m_valueCount == 0; }

	/// \brief See the comment for GRelation::areNominal
	virtual bool areNominal(size_t first = 0, size_t count = INVALID_INDEX) const { return m_valueCount != 0; }

	/// \brief Returns a copy of this object
	virtual GRelation* clone() const { return new GUniformRelation(m_attrCount, m_valueCount); }

	/// \brief Returns a copy of this object
	virtual GRelation* cloneMinimal() const { return clone(); }

	/// \brief Returns a deep copy of the specified subset of this
	/// relation
	virtual GRelation* cloneSub(size_t, size_t count) const { return new GUniformRelation(count, m_valueCount); }

	/// \brief Drop the specified attribute
	virtual void deleteAttributes(size_t index, size_t count);

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
	GMixedRelation(const GDomNode* pNode);

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

	/// \brief Returns a relation containing the same value counts as this object, but
	/// not containing superfluous things, such as human-readable strings. (The returned
	/// relation may be of a different type.)
	virtual GRelation* cloneMinimal() const;

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
	void addAttrs(const GRelation& copyMe, size_t firstAttr = 0, size_t attrCount = INVALID_INDEX);

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
	virtual bool areContinuous(size_t first = 0, size_t count = INVALID_INDEX) const;

	/// \brief Returns true iff all attributes in the specified range are nominal
	virtual bool areNominal(size_t first = 0, size_t count = INVALID_INDEX) const;

	/// \brief Swaps two columns
	virtual void swapAttributes(size_t nAttr1, size_t nAttr2);

	/// \brief Deletes an attribute.
	virtual void deleteAttributes(size_t nAttr, size_t count);
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
	GArffRelation(const GDomNode* pNode);

	virtual ~GArffRelation();

	virtual RelationType type() const { return ARFF; }

	/// \brief Marshalls this object to a DOM, which can be saved to a
	/// variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// \brief Returns a deep copy of this object
	virtual GRelation* clone() const;

	/// \brief Returns a relation containing the same value counts as this object, but
	/// not containing superfluous things, such as human-readable strings. (The returned
	/// relation will be of a different type.)
	virtual GRelation* cloneMinimal() const;

	/// \brief Makes a deep copy of the specified subset of this relation
	virtual GRelation* cloneSub(size_t start, size_t count) const;

	/// \brief Deletes all the attributes
	virtual void flush();

	/// \brief Prints the specified attribute name to a stream
	virtual void printAttrName(std::ostream& stream, size_t column) const;

	/// \brief Prints the specified value to a stream
	virtual void printAttrValue(std::ostream& stream, size_t column, double value, const char* missing = "?") const;

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

    /// \brief Sets the name of the specified attribute.
    void setAttrName(size_t attr, const char* szNewName);

	/// \brief Adds a new possible value to a nominal attribute. Returns
	/// the numerical form of the new value.
	int addAttrValue(size_t nAttr, const char* szValue);

	/// \brief Sets the number of values for the specified attribute
	virtual void setAttrValueCount(size_t nAttr, size_t nValues);

	/// \brief Swaps two columns
	virtual void swapAttributes(size_t nAttr1, size_t nAttr2);

	/// \brief Deletes an attribute
	virtual void deleteAttributes(size_t nAttr, size_t count);

	/// \brief Returns the nominal index for the specified attribute
	/// with the given value
	int findEnumeratedValue(size_t nAttr, const char* szValue) const;

	/// \brief Parses a value
	double parseValue(size_t attr, const char* val);

	/// \brief Parses the meta-data for an attribute
	void parseAttribute(GArffTokenizer& tok);

	/// \brief Drops the specified value from the list of possible values.
	/// (Swaps the last value in to fill its slot.)
	void dropValue(size_t attr, int val);
};

/// \brief Represents a matrix or a database table.
///
/// Elements can be discrete or continuous.
///
/// References a GRelation object, which stores the meta-information about each column.
class GMatrix
{
protected:
	GRelation* m_pRelation;
	std::vector<GVec*> m_rows;

public:
	/// \brief Makes an empty 0x0 matrix.
	GMatrix();

	/// \brief Construct a rows x cols matrix with all elements of the
	/// matrix assumed to be continuous.
	///
	/// It is okay to initially set rows to 0 and later call newRow to
	/// add each row. Adding columns later, however, is not very
	/// computationally efficient.)
	GMatrix(size_t rows, size_t cols);

	/// \brief Construct a matrix with a mixed relation. That is, one
	/// with some continuous attributes (columns), and some nominal
	/// attributes (columns).
	///
	/// attrValues specifies the number of nominal values suppored in
	/// each attribute (column), or 0 for a continuous attribute.
	///
	/// Initially, this matrix will have 0 rows, but you can add more
	/// rows by calling newRow or newRows.
	GMatrix(std::vector<size_t>& attrValues);

	/// \brief Create an empty matrix whose attributes/column types are
	/// specified by pRelation
	///
	/// Takes ownership of pRelation. That is, the destructor will delete pRelation.
	///
	/// Initially, this matrix will have 0 rows, but you can add more
	/// rows by calling newRow or newRows.
	GMatrix(GRelation* pRelation);

	///\brief Copy-constructor
	///
	///Copies \a orig, making a new relation object and new storage for
	///the rows (with the same content).
	///
	///\param orig the GMatrix object to copy
	GMatrix(const GMatrix& orig, size_t rowStart = 0, size_t colStart = 0, size_t rowCount = (size_t)-1, size_t colCount = (size_t)-1);


	//I put the operator= right after the copy constructor because you
	//should always have both or neither in a class, this makes that
	//easy to verify

	///\brief Make *this into a copy of orig
	///
	///Copies \a orig, making a new relation object and new storage for
	///the rows (with the same content).
	///
	///\param orig the GMatrix object to copy
	///
	///\return a reference to this GMatrix object
	GMatrix& operator=(const GMatrix& orig);


	/// \brief Load from a DOM.
	GMatrix(const GDomNode* pNode);

	~GMatrix();

	/// \brief Returns true iff all the entries in *this and \a
	/// other are identical and their relations are compatible, and they
	/// are the same size
	///
	/// \return true iff all the entries in *this and \a other are
	/// identical, their relations are compatible, and they are the same
	/// size
	bool operator==(const GMatrix& other) const;

	/// \brief Sets the relation for this dataset, which specifies the number of columns, and their data types.
	/// If there are one or more rows in this matrix, and the new relation
	/// does not have the same number of columns as the old relation, then
	/// this will throw an exception.
	/// Takes ownership of pRelation. That is, the destructor will delete it.
	void setRelation(GRelation* pRelation);

	/// \brief Resizes this matrix. Assigns all columns to be continuous, and
	/// replaces all element values with garbage.
	void resize(size_t rows, size_t cols);

	/// \brief Adds a new row to the matrix. (The values in the row are
	/// not initialized.) Returns a reference to the new row.
	GVec& newRow();

	/// \brief Adds 'n' new columns to the matrix. (This resizes every row and copies all the
	/// existing data, which is rather inefficient.) The values in the new columns are not initialized.
	void newColumns(size_t n);

	/// \brief Adds "nRows" uninitialized rows to this matrix.
	void newRows(size_t nRows);

	/// \brief Matrix add
	///
	/// Adds scalar * pThat to this. (If transpose is true, adds
	/// scalar * the transpose of pThat to this.) Both datasets must have the
	/// same dimensions. Behavior is undefined for nominal columns.
	void add(const GMatrix* pThat, bool transpose = false, double scalar = 1.0);

	/// \brief Copies the specified range of columns (including meta-data) from that matrix into this matrix,
	/// replacing all data currently in this matrix.
	void copyCols(const GMatrix& that, size_t firstCol, size_t colCount);

	/// \brief Copies (deep) all the data and metadata from that.
	void copy(const GMatrix& that, size_t rowStart = 0, size_t colStart = 0, size_t rowCount = (size_t)-1, size_t colCount = (size_t)-1);

	/// \brief Copies the transpose of that into this matrix.
	void copyTranspose(GMatrix& that);

	/// \brief This computes the square root of this matrix. (If you
	/// take the matrix that this returns and multiply it by its
	/// transpose, you should get the original dataset again.)
	/// (Returns a lower-triangular matrix.)
	///
	/// Behavior is undefined if there are nominal attributes. If
	/// tolerant is true, it will return even if it cannot compute
	/// accurate results. If tolerant is false (the default) and this
	/// matrix is not positive definite, it will throw an exception.
	GMatrix* cholesky(bool tolerant = false);

	/// \brief Copies the specified column into pOutVector
	void col(size_t index, double* pOutVector);

	/// \brief Returns the number of columns in this matrix
	size_t cols() const { return m_pRelation->size(); }

	/// \brief Computes the determinant of this matrix
	double determinant();

	/// \brief Drops any occurrences of the specified value, and removes it as a possible value
	void dropValue(size_t attr, int val);

	/// \brief Computes the eigenvalue that corresponds to the specified
	/// eigenvector of this matrix
	double eigenValue(const GVec& eigenVector);

	/// \brief Computes the eigenvector that corresponds to the
	/// specified eigenvalue of this matrix. Note that this method
	/// trashes this matrix, so make a copy first if you care.
	void eigenVector(double eigenvalue, GVec& outVector);

	/// \brief Computes y in the equation M*y=x (or y=M^(-1)x), where M
	/// is this dataset, which must be a square matrix, and x is pVector
	/// as passed in, and y is pVector after the call.
	///
	/// If there are multiple solutions, it finds the one for which all
	/// the variables in the null-space have a value of 1. If there are
	/// no solutions, it returns false. Note that this method trashes
	/// this dataset (so make a copy first if you care).
	bool gaussianElimination(double* pVector);

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


	/// \brief Loads an ARFF file and replaces the contents of this matrix with it.
	void loadArff(const char* szFilename, size_t maxRows = (size_t)-1);

	/// \brief Loads a raw (binary) file and replaces the contents of this matrix with it.
	void loadRaw(const char* szFilename);

	/// \brief Loads a file and automatically detects ARFF or raw (binary)
	void load(const char* szFilename);

	/// \brief Parses an ARFF file and replaces the contents of this matrix with it.
	void parseArff(const char* szFile, size_t nLen, size_t maxRows = (size_t)-1);

	/// \brief Parses an ARFF file and replaces the contents of this matrix with it.
	void parseArff(GArffTokenizer& tok, size_t maxRows = (size_t)-1);


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
	static GMatrix* mergeHoriz(const GMatrix* pSetA, const GMatrix* pSetB);

	/// \brief Steals all the rows from pData and adds them to this set.
	/// (You still have to delete pData.) Both datasets must have the
	/// same number of columns.
	void mergeVert(GMatrix* pData, bool ignoreMismatchingName = false);

	/// \brief Computes nCount eigenvectors and the corresponding
	/// eigenvalues using the power method (which is only accurate if a
	/// small number of eigenvalues/vectors are needed.)
	///
	/// If mostSignificant is true, the largest eigenvalues are
	/// found. If mostSignificant is false, the smallest eigenvalues are
	/// found.
	GMatrix* eigs(size_t nCount, GVec& eigenVals, GRand* pRand, bool mostSignificant);

	/// \brief Multiplies every element in this matrix by a scalar.
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
	void multiply(const GVec& vectorIn, GVec& vectorOut, bool transpose = false) const;

	/// \brief Matrix multiply.
	///
	/// For convenience, you can also specify that neither, one, or both
	/// of the inputs are virtually transposed prior to the
	/// multiplication. (If you want the results to come out transposed,
	/// you can use the equality (AB)^T=(B^T)(A^T) to figure out how to
	/// specify the parameters.)
	static GMatrix* multiply(const GMatrix& a, const GMatrix& b, bool transposeA, bool transposeB);

	/// \brief Computes the Moore-Penrose pseudoinverse of this matrix
	/// (using the SVD method). You are responsible to delete the
	/// matrix this returns.
	GMatrix* pseudoInverse();

	/// \brief Returns a const reference to the relation object, which holds meta-data about
	/// the attributes (columns)
	const GRelation& relation() const { return *m_pRelation; }

	/// \brief Allocates space for the specified number of patterns (to
	/// avoid superfluous resizing)
	void reserve(size_t n) { m_rows.reserve(n); }

	/// \brief Returns the number of rows in this matrix
	size_t rows() const { return m_rows.size(); }

	/// \brief Saves this matrix to a file in ARFF format
	void saveArff(const char* szFilename);

	/// \brief Saves this matrix to a file in raw (binary) format
	void saveRaw(const char* szFilename);

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
	void subtract(const GMatrix* pThat, bool transpose);

	/// \brief Returns the sum squared difference between this matrix
	/// and an identity matrix
	double sumSquaredDiffWithIdentity();

	/// \brief Adds an already-allocated row to this dataset.
	/// If pos is specified, the new row will be inserted and the speicified position.
	void takeRow(GVec* pRow, size_t pos = (size_t)-1);

	/// \brief Converts the matrix to reduced row echelon form
	size_t toReducedRowEchelonForm();

	/// \brief Copies all the data from this dataset into pVector.
	///
	/// pVector must be big enough to hold rows() * cols() doubles.
	void toVector(double* pVector) const;

	/// \brief Marshalls this object to a DOM, which may be saved to a variety of serial formats.
	GDomNode* serialize(GDom* pDoc) const;

	/// \brief Returns the sum of the diagonal elements
	double trace();

	/// \brief Returns a pointer to a new dataset that is this dataset
	/// transposed. (All columns in the returned dataset will be
	/// continuous.)
	///
	/// The returned matrix must be deleted by the caller.
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
	inline GVec& row(size_t index) { return *m_rows[index]; }

	/// \brief Returns a const reference to the specified row
	inline const GVec& row(size_t index) const { return *m_rows[index]; }

	/// \brief Returns a reference to the first row
	inline GVec& front() { return *m_rows[0]; }
	inline const GVec& front() const { return *m_rows[0]; }

	/// \brief Returns a reference to a row indexed from the back of the matrix.
	/// index 0 (default) is the last row, index 1 is the second-to-last row, etc.
	inline GVec& back(size_t reverse_index = 0) { return *m_rows[m_rows.size() - 1 - reverse_index]; }
	inline const GVec& back(size_t reverse_index = 0) const { return *m_rows[m_rows.size() - 1 - reverse_index]; }

	/// \brief Returns a reference to the specified row
	inline GVec& operator [](size_t index) { return *m_rows[index]; }

	/// \brief Returns a const reference to the specified row
	inline const GVec& operator [](size_t index) const { return *m_rows[index]; }

	/// \brief Fills all elements in the specified range of columns with the specified value.
	/// If no column ranges are specified, the default is to set all of them.
	void fill(double val, size_t colStart = 0, size_t colCount = INVALID_INDEX);

	/// \brief Fills all elements with random values from a uniform distribution.
	void fillUniform(GRand& rand, double min = 0.0, double max = 1.0);

	/// \brief Fills all elements with random values from a Normal distribution.
	void fillNormal(GRand& rand, double deviation = 1.0);

	/// \brief Copies pVector over the specified column
	void setCol(size_t index, const double* pVector);

	/// \brief Swaps the two specified rows
	void swapRows(size_t a, size_t b);

	/// \brief Swap pNewRow in for row i, and return row i. The caller is then responsible to delete the row that is returned.
	GVec* swapRow(size_t i, GVec* pNewRow);

	/// \brief Swaps two columns
	void swapColumns(size_t nAttr1, size_t nAttr2);

	/// \brief Deletes some columns.
	/// This does not reallocate the rows, but it does shift the elements,
	/// which is a slow operation, especially if there are many columns
	/// that follow those being deleted.
	void deleteColumns(size_t index, size_t count);

	/// \brief Swaps the specified row with the last row, and then
	/// releases it from this matrix.
	///
	/// The caller is responsible to delete the row (array of doubles) this method returns.
	GVec* releaseRow(size_t index);

	/// \brief Swaps the specified row with the last row, and then deletes it.
	void deleteRow(size_t index);

	/// \brief Releases the specified row from this matrix and shifts
	/// everything after it up one slot.
	///
	/// The caller is responsible to delete the row this method returns.
	GVec* releaseRowPreserveOrder(size_t index);

	/// \brief Deletes the specified row and shifts everything after it up one slot
	void deleteRowPreserveOrder(size_t index);

	/// \brief Replaces any occurrences of NAN in the matrix with the
	/// corresponding values from an identity matrix.
	void fixNans();

	/// \brief Deletes all the rows in this matrix.
	void flush();

	/// \brief Abandons (leaks) all the rows in this matrix.
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
	void splitCategoricalKeepIfNotEqual(GMatrix* pSingleClass, size_t nAttr, int nValue, GMatrix* pExtensionA = NULL, GMatrix* pExtensionB = NULL);

	/// \brief Moves all rows with the specified value in the specified
	/// attribute into pOtherValues
	///
	/// If pExtensionA is non-NULL, then it will also split pExtensionA
	/// such that corresponding rows are preserved.
	void splitCategoricalKeepIfEqual(GMatrix* pOtherValues, size_t nAttr, int nValue, GMatrix* pExtensionA = NULL, GMatrix* pExtensionB = NULL);

	/// \brief Removes the last nOtherRows rows from this data set and
	/// puts them in "other". (Order is preserved.)
	void splitBySize(GMatrix& other, size_t nOtherRows);

	/// \brief Measures the entropy of the specified attribute
	double entropy(size_t nColumn) const;

	/// \brief Returns the minimum value in the specified column (not counting UNKNOWN_REAL_VALUE).
	/// Returns 1e300 if there are no known values in the column.
	double columnMin(size_t nAttribute) const;

	/// \brief Returns the maximum value in the specified column (not counting UNKNOWN_REAL_VALUE).
	/// Returns -1e300 if there are no known values in the column.
	double columnMax(size_t nAttribute) const;

	/// \brief Returns the sum of the values in the specified column.
	double columnSum(size_t col) const;

	/// \brief Computes the arithmetic mean of the values in the specified column
	/// If pWeights is NULL, then each row is given equal weight.
	/// If pWeights is non-NULL, then it is assumed to be a vector of weights, one for each row in this matrix.
	/// If there are no values in this column with any weight, then it will throw an exception if throwIfEmpty is true,
	/// or else return UNKNOWN_REAL_VALUE.
	double columnMean(size_t nAttribute, const GVec* pWeights = NULL, bool throwIfEmpty = true) const;

	/// Returns the squared magnitude of the vector in the specified column.
	double columnSquaredMagnitude(size_t col) const;

	/// \brief Computes the sample variance of a single attribute
	double columnVariance(size_t nAttr, double mean) const;

	/// \brief Scales the column by the specified scalar.
	void scaleColumn(size_t col, double scalar);

	/// \brief Computes the median of the values in the specified column
	/// If there are no values in this column, then it will throw an exception if throwIfEmpty is true,
	/// or else return UNKNOWN_REAL_VALUE.
	double columnMedian(size_t nAttribute, bool throwIfEmpty = true) const;

	/// \brief Shifts the data such that the mean occurs at the origin.
	/// Only continuous values are affected.  Nominal values are left
	/// unchanged.
	void centerMeanAtOrigin();

	/// \brief Computes the arithmetic means of all attributes
	/// If pWeights is non-NULL, then it is assumed to be a vector of weights, one for each row in this matrix.
	void centroid(GVec& outCentroid, const GVec* pWeights = NULL) const;

	/// \brief Normalizes the specified column
	void normalizeColumn(size_t col, double dInMin, double dInMax, double dOutMin = 0.0, double dOutMax = 1.0);

	/// \brief Clips the values in the specified column to fall beween dMin and dMax (inclusively).
	void clipColumn(size_t col, double dMin, double dMax);

	/// \brief Normalize a value from the input min and max to the output min and max.
	static double normalizeValue(double dVal, double dInMin, double dInMax, double dOutMin = 0.0, double dOutMax = 1.0);

	/// \brief Returns the mean if the specified attribute is
	/// continuous, otherwise returns the most common nominal value in
	/// the attribute.
	double baselineValue(size_t nAttribute) const;

	/// \brief Returns true iff the specified attribute contains
	/// homogenous values. (Unknowns are counted as homogenous with
	/// anything)
	bool isAttrHomogenous(size_t col) const;

	/// \brief Returns true iff each of the last labelDims columns in
	/// the data are homogenous
	bool isHomogenous() const;

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
	void principalComponent(GVec& outVector, const GVec& centroid, GRand* pRand) const;

	/// \brief Computes the first principal component assuming the mean
	/// is already subtracted out of the data
	void principalComponentAboutOrigin(GVec& outVector, GRand* pRand) const;

	/// \brief Computes principal components, while ignoring missing
	/// values
	void principalComponentIgnoreUnknowns(GVec& outVector, const GVec& centroid, GRand* pRand) const;

	/// \brief Computes the first principal component of the data with
	/// each row weighted according to the vector pWeights. (pWeights
	/// must have an element for each row.)
	void weightedPrincipalComponent(GVec& outVector, const GVec& centroid, const double* pWeights, GRand* pRand) const;

	/// \brief Computes the eigenvalue that corresponds to \a *pEigenvector.
	///
	/// After you compute the principal component, you can call this to
	/// obtain the eigenvalue that corresponds to that principal
	/// component vector (eigenvector).
	double eigenValue(const double* pMean, const double* pEigenVector, GRand* pRand) const;

	/// \brief Removes the component specified by pComponent from the
	/// data. (pComponent should already be normalized.)
	///
	/// This might be useful, for example, to remove the first principal
	/// component from the data so you can then proceed to compute the
	/// second principal component, and so forth.
	void removeComponent(const GVec& centroid, const GVec& component);

	/// \brief Removes the specified component assuming the mean is zero.
	void removeComponentAboutOrigin(const GVec& component);

	/// \brief Computes the minimum number of principal components
	/// necessary so that less than the specified portion of the
	/// deviation in the data is unaccounted for.
	///
	/// For example, if the data projected onto the first 3 principal
	/// components contains 90 percent of the deviation that the
	/// original data contains, then if you pass the value 0.1 to this
	/// method, it will return 3.
	size_t countPrincipalComponents(double d, GRand* pRand) const;

	/// \brief Computes the sum-squared distance between pPoint and all
	/// of the rows in this matrix.
	///
	/// If pPoint is NULL, it computes the sum-squared distance with the origin.
	///
	/// \note that this is equal to the sum of all the eigenvalues times
	///       the number of dimensions, so you can efficiently compute
	///       eigenvalues as the difference in sumSquaredDistance with
	///       the mean after removing the corresponding component, and
	///       then dividing by the number of dimensions. This is more
	///       efficient than calling eigenValue.
	double sumSquaredDistance(const GVec& point) const;

	/// \brief Computes the sum-squared distance between the specified
	/// column of this and that. If the column is a nominal attribute,
	/// then Hamming distance is used.
	/// if pOutSAE is not NULL, the sum absolute error will be placed there.
	double columnSumSquaredDifference(const GMatrix& that, size_t col, double* pOutSAE = NULL) const;

	/// \brief Computes the squared distance between this and that.
	///
	/// If transpose is true, computes the difference between this and
	/// the transpose of that.
	double sumSquaredDifference(const GMatrix& that, bool transpose = false) const;

	/// \brief Computes the linear coefficient between the two specified
	/// attributes.
	///
	/// Usually you will want to pass the mean values for attr1Origin
	/// and attr2Origin.
	double linearCorrelationCoefficient(size_t attr1, double attr1Origin, size_t attr2, double attr2Origin) const;

	/// \brief Finds a sphere that tightly bounds all the points in the specified vector of row-indexes.
	///
	/// Returns the squared radius of the sphere, and stores its center in pOutCenter.
	double boundingSphere(GVec& outCenter, size_t* pIndexes, size_t indexCount, GDistanceMetric* pMetric) const;

	/// \brief Computes the covariance between two attributes.
	/// If pWeights is NULL, each row is given a weight of 1.
	/// If pWeights is non-NULL, then it is assumed to be a vector of weights, one for each row in this matrix.
	double covariance(size_t nAttr1, double dMean1, size_t nAttr2, double dMean2, const double* pWeights = NULL) const;

	/// \brief Computes the covariance matrix of the data
	GMatrix* covarianceMatrix() const;

	/// \brief Performs a paired T-Test with data from the two specified
	/// attributes.
	///
	/// pOutV will hold the degrees of freedom. pOutT will hold the T-value.
	/// You can use GMath::tTestAlphaValue to convert these to a P-value.
	void pairedTTest(size_t* pOutV, double* pOutT, size_t attr1, size_t attr2, bool normalize) const;

	/// \brief Performs the Wilcoxon signed ranks test from the two
	/// specified attributes.
	///
	/// If two values are closer than tolerance, they are considered to
	/// be equal.
	void wilcoxonSignedRanksTest(size_t attr1, size_t attr2, double tolerance, int* pNum, double* pWMinus, double* pWPlus) const;

	/// \brief Prints this matrix in ARFF format to the specified stream
	void print(std::ostream& stream = std::cout, char separator = ',') const;

	/// \brief Returns the number of ocurrences of the specified value
	/// in the specified attribute
	size_t countValue(size_t attribute, double value) const;

	/// \brief Returns true iff this matrix is missing any values.
	bool doesHaveAnyMissingValues() const;

	/// \brief Throws an exception if this data contains any missing
	/// values in a continuous attribute
	void ensureDataHasNoMissingReals() const;

	/// \brief Throws an exception if this data contains any missing
	/// values in a nominal attribute
	void ensureDataHasNoMissingNominals() const;

	/// \brief Computes the sum entropy of the data (or the sum variance
	/// for continuous attributes)
	double measureInfo() const;

	/// \brief Computes the vector in this subspace that has the
	/// greatest distance from its projection into pThat subspace.
	///
	/// Returns true if the results are computed.
	///
	/// Returns false if the subspaces are so nearly parallel that pOut
	/// cannot be computed with accuracy.
	bool leastCorrelatedVector(GVec& out, const GMatrix* pThat, GRand* pRand) const;

	/// \brief Computes the cosine of the dihedral angle between this
	/// subspace and pThat subspace
	double dihedralCorrelation(const GMatrix* pThat, GRand* pRand) const;

	/// \brief Projects pPoint onto this hyperplane (where each row
	/// defines one of the orthonormal basis vectors of this hyperplane)
	///
	/// This computes (A^T)Ap, where A is this matrix, and p is pPoint.
//	void project(double* pDest, const double* pPoint) const;

	/// \brief Projects pPoint onto this hyperplane (where each row
	/// defines one of the orthonormal basis vectors of this hyperplane)
//	void project(double* pDest, const double* pPoint, const double* pOrigin) const;

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

	/// \brief Copies values from a rectangular region of the source matrix into this matrix.
	/// The wid and hgt values are clipped if they exceed the size of the source matrix.
	/// An exception is thrown if the destination is not big enough to hold the values at the specified location.
	/// If checkMetaData is true, then this will throw an exception if the data types are incompatible.
	void copyBlock(const GMatrix& source, size_t srcRow = 0, size_t srcCol = 0, size_t hgt = INVALID_INDEX, size_t wid = INVALID_INDEX, size_t destRow = 0, size_t destCol = 0, bool checkMetaData = true);

	/// Counts the number of unique values in the specified column. If maxCount
	/// unique values are found, it immediately returns maxCount.
	size_t countUniqueValues(size_t col, size_t maxCount = (size_t)-1) const;

	/// \brief Performs unit tests for this class. Throws an exception
	/// if there is a failure.
	static void test();

protected:
	double determinantHelper(size_t nEndRow, size_t* pColumnList);
	void inPlaceSquareTranspose();
	void singularValueDecompositionHelper(GMatrix** ppU, double** ppDiag, GMatrix** ppV, bool throwIfNoConverge, size_t maxIters);
};



/// A class for parsing CSV files (or tab-separated files, or whitespace separated files, etc.).
/// (This class does not support Mac line endings, so you should replace all '\r' with '\n' before using this class if your
/// data comes from a Mac.)
class GCSVParser
{
protected:
	bool m_single_quotes; // If false, treats apostrophes as normal text. If true, treats them as quotation marks.
	char m_separator;
	bool m_columnNamesInFirstRow;
	bool m_tolerant;
	size_t m_clearlyNumericalThreshold;
	size_t m_maxVals;
	std::vector<std::string> m_report;
	std::map<size_t, std::string> m_formats;
	std::map<size_t, size_t> m_specifiedReal;
	std::map<size_t, size_t> m_specifiedNominal;
	std::map<size_t, size_t> m_stripQuotes;

public:
	GCSVParser();
	~GCSVParser();

	/// Specify the separating character. '\0' indicates that an arbitrary amount of whitespace is used for separation.
	void setSeparator(char c) { m_separator = c; }

	/// Indicate that the first row specifies column names
	void columnNamesInFirstRow() { m_columnNamesInFirstRow = true; }

	/// Specify to ignore inconsistencies in the number of values in each row. (Using this is very dangerous.)
	void tolerant() { m_tolerant = true; }

	/// Specify the number of unique numerical values before a column is deemed to be clearly numerical.
	void setClearlyNumericalThreshold(size_t n) { m_clearlyNumericalThreshold = n; }

	/// Specify the maximum number of values to allow in a categorical attribute. The parsing of any columns that
	/// contain non-numerical values, and contain more than this number of unique values, will be aborted.
	void setMaxVals(size_t n) { m_maxVals = n; }

	/// Specify that a certain attribute should be expected to be a date or time stamp that follows a given format.
	/// For example, szFormat might be "YYYY-MM-DD hh:mm:ss".
	void setTimeFormat(size_t attr, const char* szFormat);

	/// Indiciate that the specified attribute should be treated as nominal.
	void setNominalAttr(size_t attr);

	/// Indiciate that the specified attribute should be treated as real.
	void setRealAttr(size_t attr);

	/// Indiciate that the specified attribute should have enclosing quotes stripped.
	void setStripQuotes(size_t attr);

	/// Load the specified file, and parse it.
	void parse(GMatrix& outMatrix, const char* szFilename);

	/// Parse the given string.
	void parse(GMatrix& outMatrix, const char* pString, size_t len);

	/// Return a string that reports the status of the specified column. (This should only be called after parsing.)
	std::string& report(size_t column) { return m_report[column]; }
};



/// \brief This is a special holder that guarantees the data set will
/// release all of its data before it is deleted
class GReleaseDataHolder
{
protected:
	GMatrix* m_pData;

public:
	GReleaseDataHolder(GMatrix* pData = NULL)
	{
		m_pData = pData;
	}

	~GReleaseDataHolder()
	{
		if(m_pData)
			m_pData->releaseAllRows();
	}

	void reset(GMatrix* pData = NULL)
	{
		if(m_pData && m_pData != pData)
			m_pData->releaseAllRows();
		m_pData = pData;
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






/// This class divides a matrix into two parts. The left-most columns are the features.
/// The right-most columns are the labels.
class GDataColSplitter
{
protected:
	GMatrix* m_pFeatures;
	GMatrix* m_pLabels;

public:
	/// Splits a dataset into a feature matrix and a label matrix. The right-most "labels" columns
	/// are put in the label matrix
	GDataColSplitter(const GMatrix& data, size_t labels);
	~GDataColSplitter();

	/// Returns a reference to the feature matrix
	GMatrix& features() { return *m_pFeatures; }

	/// Returns a reference to the label matrix
	GMatrix& labels() { return *m_pLabels; }
};





/// This class divides a features and labels matrix into two parts
/// by randomly assigning each row to one of the two parts, keeping the corresponding
/// rows together. The rows are shallow-copied. The destructor of this class releases
/// all of the row references.
class GDataRowSplitter
{
protected:
	GMatrix m_f1;
	GMatrix m_f2;
	GMatrix m_l1;
	GMatrix m_l2;

public:
	/// features and labels are expected to remain valid for the duration of this object.
	/// proportion specifies the proportion of rows that will be referenced
	/// by part 1 of the data. (In case of an exact tie, part 2 gets the extra row.)
	GDataRowSplitter(const GMatrix& features, const GMatrix& labels, GRand& rand, size_t part1Rows);
	~GDataRowSplitter();

	/// Returns a reference to the first part of the features matrix
	const GMatrix& features1() { return m_f1; }

	/// Returns a reference to the second part of the features matrix
	const GMatrix& features2() { return m_f2; }

	/// Returns a reference to the first part of the labels matrix
	const GMatrix& labels1() { return m_l1; }

	/// returns a reference to the second part of the labels matrix
	const GMatrix& labels2() { return m_l2; }
};


/// A matrix in which each row may have a differing number of columns.
class GRaggedMatrix
{
protected:
	std::vector<GVec*> m_rows;
	size_t m_minCols;
	size_t m_maxCols;

public:
	GRaggedMatrix();
	~GRaggedMatrix();

	/// \brief Removes all the rows in this ragged matrix.
	void flush();
	
	/// \brief Returns the number of rows in this matrix.
	size_t rows() const { return m_rows.size(); }

	/// \brief Returns the number of columns in the shortest row
	size_t minCols() const { return m_minCols; }

	/// \brief Returns the number of columns in the longest row
	size_t maxCols() const { return m_maxCols; }

	/// \brief Returns a reference to the specified row
	inline GVec& operator [](size_t index) { return *m_rows[index]; }

	/// \brief Returns a const reference to the specified row
	inline const GVec& operator [](size_t index) const { return *m_rows[index]; }

	/// \brief Adds a new row with the specified size to this ragged matrix.
	GVec& newRow(size_t size);

	/// \brief Parses a CSV file.
	void parseCSV(GTokenizer& tok);

	/// \brief Loads from a CSV file.
	void loadCSV(const char* szFilename);
};


} // namespace GClasses

#endif // __GMATRIX_H__
