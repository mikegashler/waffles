/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    Michael R. Smith,
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

#ifndef __GDISTANCE_H__
#define __GDISTANCE_H__

#include "GMatrix.h"
#include <map>
#include <vector>
#include "GKernelTrick.h"

namespace GClasses {

class GKernel;


/// This class enables you to define a distance (or dissimilarity) metric between two vectors.
/// pScaleFactors is an optional parameter (it can be NULL) that lets the calling class
/// scale the significance of each dimension. Distance metrics that do not mix with
/// this concept may simply ignore any scale factors.
/// Typically, classes that use this should be able to assume that the triangle
/// inequality will hold, but do not necessarily enforce the parallelogram law.
class GDistanceMetric
{
protected:
	const GRelation* m_pRelation;
	bool m_ownRelation;
	GVec m_scaleFactors;

public:
	GDistanceMetric() : m_pRelation(NULL), m_ownRelation(false) {}
	GDistanceMetric(GDomNode* pNode);
	virtual ~GDistanceMetric();

	static void test();

	/// Returns the name of this class
	virtual const char* name() const = 0;

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// This must be called before squaredDistance can be called. Takes ownership of pRelation iff own is true.
	virtual void init(const GRelation* pRelation, bool own) = 0;

	/// Computes the squared distance (or squared dissimilarity) between the two specified vectors
	virtual double squaredDistance(const GVec& a, const GVec& b) const = 0;

	/// Return squaredDistance(pA, pB).  Allows dissimilarity metrics to
	/// be used as function objects.  Do not override.  Override
	/// squaredDistance(pA,pB) instead.  See GDistanceMetric::squaredDistance(const GVec&, const GVec&)
	inline double operator()(const GVec& a, const GVec& b) const
	{
		return squaredDistance(a, b);
	}

	/// Returns the relation that specifies the meaning of the vector elements
	const GRelation* relation() const { return m_pRelation; }

	/// Deserializes a distance metric
	static GDistanceMetric* deserialize(GDomNode* pNode);

	/// Returns a reference to the vector of attribute scalars.
	virtual GVec& scaleFactors() { return m_scaleFactors; }

protected:
	GDomNode* baseDomNode(GDom* pDoc) const;

	/// Sets the relation to use with this metric. Takes ownership
	/// of the relation iff own is true.
	void setRelation(const GRelation* pRelation, bool own);
};



/// This uses a combination of Euclidean distance for continuous
/// attributes, and Hamming distance for nominal attributes.  In
/// particular, for each attribute, it calculates pA[i]-pB[i], squares
/// it and takes the square root of that sum.  For nominal attributes
/// pA[i]-pB[i] is 0 if they are the same and 1 if they are different.
class GRowDistance : public GDistanceMetric
{
protected:
	double m_diffWithUnknown;

public:
	GRowDistance();
	GRowDistance(GDomNode* pNode);
	virtual ~GRowDistance() {}

	/// Returns the name of this class
	virtual const char* name() const { return "GRowDistance"; }

	/// See the comment for GDistanceMetric::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GDistanceMetric::init.
	virtual void init(const GRelation* pRelation, bool own);

	/// Returns the distance between a and b
	virtual double squaredDistance(const GVec& a, const GVec& b) const;

	/// Specify the difference to use when one or more of the values is unknown.
	/// (If your data contains unknown values, you may want to normalize the
	/// known values to fall within some pre-determined range, so that it will
	/// be possible to select a reasonable value for this purpose.)
	void setDiffWithUnknown(double d) { m_diffWithUnknown = d; }
};




/// Interpolates between manhattan distance (norm=1), Euclidean
/// distance (norm=2), and Chebyshev distance (norm=infinity). For
/// nominal attributes, Hamming distance is used.
class GLNormDistance : public GDistanceMetric
{
protected:
	double m_norm;
	double m_diffWithUnknown;

public:
	GLNormDistance(double norm);
	GLNormDistance(GDomNode* pNode);
	virtual ~GLNormDistance() {}

	/// Returns the name of this class
	virtual const char* name() const { return "GLNormDistance"; }

	/// See the comment for GDistanceMetric::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GDistanceMetric::init.
	virtual void init(const GRelation* pRelation, bool own);

	/// Returns the distance (using the norm passed to the constructor) between pA and pB
	virtual double squaredDistance(const GVec& a, const GVec& b) const;

	/// Specify the difference to use when one or more of the values is unknown.
	/// (If your data contains unknown values, you may want to normalize the
	/// known values to fall within some pre-determined range, so that it will
	/// be possible to select a reasonable value for this purpose.)
	void setDiffWithUnknown(double d) { m_diffWithUnknown = d; }
};




/// Returns 1 minus the cosine of the angle between the two vectors with the origin.
class GDenseCosineDistance : public GDistanceMetric
{
public:
	GDenseCosineDistance();
	GDenseCosineDistance(GDomNode* pNode);
	virtual ~GDenseCosineDistance() {}

	/// Returns the name of this class
	virtual const char* name() const { return "GDenseCosineDistance"; }

	/// See the comment for GDistanceMetric::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GDistanceMetric::init.
	virtual void init(const GRelation* pRelation, bool own);

	/// Returns the distance (using the norm passed to the constructor) between pA and pB
	virtual double squaredDistance(const GVec& a, const GVec& b) const;
};




/// Returns 1 minus the cosine of the angle between the two vectors with the origin.
class GKernelDistance : public GDistanceMetric
{
protected:
	GKernel* m_pKernel;
	bool m_ownKernel;

public:
	GKernelDistance(GKernel* pKernel, bool own);
	GKernelDistance(GDomNode* pNode);
	virtual ~GKernelDistance();

	/// Returns the name of this class
	virtual const char* name() const { return "GKernelDistance"; }

	/// See the comment for GDistanceMetric::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GDistanceMetric::init.
	virtual void init(const GRelation* pRelation, bool own);

	/// Returns the distance (using the norm passed to the constructor) between pA and pB
	virtual double squaredDistance(const GVec& a, const GVec& b) const;
};




/// The base class for similarity metrics that operate on sparse vectors.
class GSparseSimilarity
{
protected:
	double m_regularizer;

public:
	GSparseSimilarity() : m_regularizer(0.0) {}
	virtual ~GSparseSimilarity() {}

	/// Set a regularizing term to add to the denominator
	void setRegularizer(double d) { m_regularizer = d; }

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// Computes the similarity between two sparse vectors
	virtual double similarity(const std::map<size_t,double>& a, const std::map<size_t,double>& b) = 0;

	/// Computes the similarity between a sparse and a dense vector
	virtual double similarity(const std::map<size_t,double>& a, const GVec& b) = 0;

	/// Computes the similarity between two dense vectors
	virtual double similarity(const GVec& a, const GVec& b) = 0;

	/// Load from a DOM.
	static GSparseSimilarity* deserialize(GDomNode* pNode);

protected:
	/// A helper method used internally
	GDomNode* baseDomNode(GDom* pDoc, const char* szClassName) const;
};


/// This is a similarity metric that computes the cosine of the angle bewtween two sparse vectors
class GCosineSimilarity : public GSparseSimilarity
{
public:
	GCosineSimilarity() : GSparseSimilarity() {}
	GCosineSimilarity(GDomNode* pNode) : GSparseSimilarity() {}
	virtual ~GCosineSimilarity() {}

	/// See the comment for GSparseSimilarity::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Computes the similarity between two sparse vectors
	virtual double similarity(const std::map<size_t,double>& a, const std::map<size_t,double>& b);

	/// Computes the similarity between a sparse and a dense vector
	virtual double similarity(const std::map<size_t,double>& a, const GVec& b);

	/// Computes the similarity between two dense vectors
	virtual double similarity(const GVec& a, const GVec& b);
};


/// This is a similarity metric that computes the Pearson correlation between two sparse vectors
class GPearsonCorrelation : public GSparseSimilarity
{
public:
	GPearsonCorrelation() : GSparseSimilarity() {}
	GPearsonCorrelation(GDomNode* pNode) : GSparseSimilarity() {}
	virtual ~GPearsonCorrelation() {}

	/// See the comment for GSparseSimilarity::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Computes the similarity between two sparse vectors
	virtual double similarity(const std::map<size_t,double>& a, const std::map<size_t,double>& b);

	/// Computes the similarity between a sparse and a dense vector
	virtual double similarity(const std::map<size_t,double>& a, const GVec& b);

	/// Computes the similarity between two dense vectors
	virtual double similarity(const GVec& a, const GVec& b);
};


/// This computes the reciprocal of Euclidean distance, where all missing values are simply ignored.
class GEuclidSimilarity : public GSparseSimilarity
{
protected:
	double m_squaredMissingPenalty;

public:
	GEuclidSimilarity() : GSparseSimilarity() {}
	GEuclidSimilarity(GDomNode* pNode) : GSparseSimilarity() {}
	virtual ~GEuclidSimilarity() {}

	/// See the comment for GSparseSimilarity::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Computes the similarity between two sparse vectors
	virtual double similarity(const std::map<size_t,double>& a, const std::map<size_t,double>& b);

	/// Computes the similarity between a sparse and a dense vector
	virtual double similarity(const std::map<size_t,double>& a, const GVec& b);

	/// Computes the similarity between two dense vectors
	virtual double similarity(const GVec& a, const GVec& b);
};


} // namespace GClasses

#endif // __GDISTANCE_H__
