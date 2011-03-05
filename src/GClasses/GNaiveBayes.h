/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GNAIVEBAYES_H__
#define __GNAIVEBAYES_H__

#include "GLearner.h"

namespace GClasses {

class GXMLTag;
struct GNaiveBayesOutputAttr;

/// A naive Bayes classifier
class GNaiveBayes : public GIncrementalLearner
{
protected:
	sp_relation m_pFeatureRel;
	sp_relation m_pLabelRel;
	size_t m_nSampleCount;
	GNaiveBayesOutputAttr** m_pOutputs;
	double m_equivalentSampleSize;
	GRand* m_pRand;

public:
	GNaiveBayes(GRand* pRand);

	/// Load from a text-based format
	GNaiveBayes(GTwtNode* pNode, GRand* pRand);

	virtual ~GNaiveBayes();

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif

	/// Save to a text-based format
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// See the comment for GIncrementalLearner::enableIncrementalLearning
	virtual void enableIncrementalLearning(sp_relation& pFeatureRel, sp_relation& pLabelRel);

	/// Adds a single training sample to the collection
	virtual void trainIncremental(const double* pIn, const double* pOut);

	/// See the comment for GIncrementalLearner::trainSparse
	/// This method assumes that the values in pData are all binary values (0 or 1).
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);

	/// To ensure that unsampled values don't dominate the joint
	/// distribution by multiplying by a zero, each value is given
	/// at least as much representation as specified here. (The default
	/// is 0.5, which is as if there were half of a sample for each value.)
	void setEquivalentSampleSize(double d) { m_equivalentSampleSize = d; }

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);

	/// See the comment for GTransducer::canImplicitlyHandleContinuousFeatures
	virtual bool canImplicitlyHandleContinuousFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }
};


/// This modeler is very similar to Naive Bayes, except even simpler. It just counts the
/// frequency of each output give each input. To generalize, it assumes conditional
/// independence and computes the maximum likelihood output based on the training rows
/// with similar input values.
class GNaiveMLE : public GSupervisedLearner
{
protected:
	GCategoricalDistribution* m_pPredictions;
	size_t m_nValues;
	double m_dLimboValue;
	double m_dEquivalentSampleSize;
	sp_relation m_pFeatureRel;
	sp_relation m_pLabelRel;

public:
	/// labelDims specifies the number of output dimensions
	GNaiveMLE(sp_relation& pRelation);

	/// Load from a text-based format
	GNaiveMLE(GTwtNode* pNode, GRand& rand);

	virtual ~GNaiveMLE();

	/// Save to a text-based format
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// Specify a linear interpolating factor between making the prediction
	/// in linear space and logarithmic space.
	void setLimboValue(double d) { m_dLimboValue = d; }

	/// To ensure that unsampled values don't dominate the joint
	/// distribution by multiplying by a zero, each value is given
	/// at least as much representation as specified here.
	void setEquivalentSampleSize(double d) { m_dEquivalentSampleSize = d; }

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);

	/// See the comment for GTransducer::canImplicitlyHandleContinuousFeatures
	virtual bool canImplicitlyHandleContinuousFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }
};

} // namespace GClasses

#endif // __GNAIVEBAYES_H__
