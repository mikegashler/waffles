/*
	Copyright (C) 2010, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GDISTANCE_H__
#define __GDISTANCE_H__

#include "GMatrix.h"
#include <map>

namespace GClasses {

/// This class enables you to define dissimilarity (distance) metrics between two vectors.
/// pScaleFactors is an optional parameter (it can be NULL) that lets the calling class
/// scale the significance of each dimension. Distance metrics that do not mix with
/// this concept may simply ignore any scale factors.
/// Typically, classes that use this should be able to assume that the triangle
/// inequality will hold, but do not necessarily enforce the parallelogram law.
class GDissimilarityMetric
{
protected:
	sp_relation m_pRelation;

public:
	GDissimilarityMetric() {}
	GDissimilarityMetric(GTwtNode* pNode);
	virtual ~GDissimilarityMetric() {}

	/// Serialize this metric to a text-based format
	virtual GTwtNode* toTwt(GTwtDoc* pDoc) = 0;

	/// This must be called before dissimilarity can be called
	virtual void init(sp_relation& pRelation) = 0;

	/// Computes the dissimilarity between the two specified vectors
	virtual double dissimilarity(const double* pA, const double* pB) = 0;

	/// Returns the relation that specifies the meaning of the vector elements
	sp_relation& relation() { return m_pRelation; }

	/// Deserializes a dissimilarity metric
	static GDissimilarityMetric* fromTwt(GTwtNode* pNode);

	/// Returns a pointer to the vector of scale factors
	virtual double* scaleFactors() { return NULL; }

protected:
	GTwtNode* baseTwtNode(GTwtDoc* pDoc, const char* szClassName);
};



/// This uses Euclidean distance for continuous attributes, and
/// Hamming distance for nominal attributes.
class GRowDistance : public GDissimilarityMetric
{
public:
	GRowDistance()
	: GDissimilarityMetric()
	{
	}

	GRowDistance(GTwtNode* pNode);

	virtual ~GRowDistance() {}

	/// See the comment for GDissimilarityMetric::toTwt
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// See the comment for GDissimilarityMetric::init
	virtual void init(sp_relation& pRelation);

	/// Returns the distance between pA and pB
	virtual double dissimilarity(const double* pA, const double* pB);
};




/// This uses Euclidean distance for continuous attributes, and
/// Hamming distance for nominal attributes.
class GRowDistanceScaled : public GDissimilarityMetric
{
protected:
	double* m_pScaleFactors;

public:
	GRowDistanceScaled() : m_pScaleFactors(NULL) {}
	GRowDistanceScaled(GTwtNode* pNode);

	virtual ~GRowDistanceScaled()
	{
		delete[] m_pScaleFactors;
	}

	/// See the comment for GDissimilarityMetric::toTwt
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// See the comment for GDissimilarityMetric::init
	virtual void init(sp_relation& pRelation);

	/// Returns the scaled distance between pA and pB
	virtual double dissimilarity(const double* pA, const double* pB);

	/// Returns the vector of scalar values associated with each dimension
	virtual double* scaleFactors() { return m_pScaleFactors; }
};




/// Interpolates between manhattan distance (norm=1), Euclidean distance (norm=2),
/// and Chebyshev distance (norm=infinity). Throws an exception if any of the
/// attributes are nominal.
class GMinkowskiDistance : public GDissimilarityMetric
{
protected:
	double m_norm;

public:
	GMinkowskiDistance(double norm)
	: GDissimilarityMetric(), m_norm(norm)
	{
	}

	GMinkowskiDistance(GTwtNode* pNode);

	/// See the comment for GDissimilarityMetric::toTwt
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// See the comment for GDissimilarityMetric::init
	virtual void init(sp_relation& pRelation);

	/// Returns the distance (using the norm passed to the constructor) between pA and pB
	virtual double dissimilarity(const double* pA, const double* pB);
};




/// The base class for similarity metrics that operate on sparse vectors
class GSparseSimilarity
{
protected:
	double m_regularizer;

public:
	GSparseSimilarity() : m_regularizer(0.0) {}
	virtual ~GSparseSimilarity() {}

	/// Set a regularizing term to add to the denominator
	void setRegularizer(double d) { m_regularizer = d; }

	/// Serialize this metric to a text-based format
	virtual GTwtNode* toTwt(GTwtDoc* pDoc) = 0;

	/// Computes the similarity between two sparse vectors
	virtual double similarity(const std::map<size_t,double>& a, const std::map<size_t,double>& b) = 0;

	/// Computes the similarity between a sparse and a dense vector
	virtual double similarity(const std::map<size_t,double>& a, const double* pB) = 0;

	/// Deserialize from a text-based format
	static GSparseSimilarity* fromTwt(GTwtNode* pNode);

protected:
	/// A helper method used internally
	GTwtNode* baseTwtNode(GTwtDoc* pDoc, const char* szClassName);
};


/// This is a similarity metric that computes the cosine of the angle bewtween two sparse vectors
class GCosineSimilarity : public GSparseSimilarity
{
public:
	GCosineSimilarity() : GSparseSimilarity() {}
	GCosineSimilarity(GTwtNode* pNode) : GSparseSimilarity() {}
	virtual ~GCosineSimilarity() {}

	/// See the comment for GSparseSimilarity::toTwt
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// Computes the similarity between two sparse vectors
	virtual double similarity(const std::map<size_t,double>& a, const std::map<size_t,double>& b);

	/// Computes the similarity between a sparse and a dense vector
	virtual double similarity(const std::map<size_t,double>& a, const double* pB);
};


/// This is a similarity metric that computes the Pearson correlation between two sparse vectors
class GPearsonCorrelation : public GSparseSimilarity
{
public:
	GPearsonCorrelation() : GSparseSimilarity() {}
	GPearsonCorrelation(GTwtNode* pNode) : GSparseSimilarity() {}
	virtual ~GPearsonCorrelation() {}

	/// See the comment for GSparseSimilarity::toTwt
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// Computes the similarity between two sparse vectors
	virtual double similarity(const std::map<size_t,double>& a, const std::map<size_t,double>& b);

	/// Computes the similarity between a sparse and a dense vector
	virtual double similarity(const std::map<size_t,double>& a, const double* pB);
};


} // namespace GClasses

#endif // __GDISTANCE_H__
