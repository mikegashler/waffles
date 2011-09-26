/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GENSEMBLE_H__
#define __GENSEMBLE_H__

#include "GLearner.h"
#include <vector>
#include <exception>

namespace GClasses {

class GRelation;
class GRand;


typedef void (*EnsembleProgressCallback)(void* pThis, size_t i, size_t n);

/// This is a helper-class used by GBag
class GWeightedModel
{
public:
	double m_weight;
	GSupervisedLearner* m_pModel;

	/// General-purpose constructor
	GWeightedModel(double weight, GSupervisedLearner* pModel)
	: m_weight(weight), m_pModel(pModel)
	{
	}

	/// Load from a DOM.
	GWeightedModel(GDomNode* pNode, GLearnerLoader& ll);
	~GWeightedModel();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	GDomNode* serialize(GDom* pDoc);
};


/// BAG stands for bootstrap aggregator. It represents an ensemble
/// of voting modelers. Each model is trained with a slightly different
/// training set, which is produced by drawing randomly from the original
/// training set with replacement until we have a new training set of
/// the same size. Each model is given equal weight in the vote.
class GBag : public GSupervisedLearner
{
protected:
	sp_relation m_pLabelRel;
	std::vector<GWeightedModel*> m_models;
	size_t m_nAccumulatorDims;
	double* m_pAccumulator;
	EnsembleProgressCallback m_pCB;
	void* m_pThis;

public:
	/// General-purpose constructor.
	GBag(GRand& rand);

	/// Deserializing constructor.
	GBag(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GBag();

#ifndef NO_TEST_CODE
	static void test();
#endif

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc);

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Removes and deletes all the learners
	void flush();

	/// Adds a learner to the bag. This takes ownership of pLearner (so
	/// it will delete it when it's done with it)
	void addLearner(GSupervisedLearner* pLearner);

	/// If you want to be notified when another instance begins training, you can set this callback
	void setProgressCallback(EnsembleProgressCallback pCB, void* pThis)
	{
		m_pCB = pCB;
		m_pThis = pThis;
	}

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);

	/// Assigns uniform weight to all models. (This method is deliberately
	/// virtual so that you can overload it if you want non-uniform weighting.)
	virtual void determineWeights(GMatrix& features, GMatrix& labels);

	/// Scales the weights so they sum to 1.0.
	void normalizeWeights();

	/// Adds the vote from one of the models.
	void castVote(double weight, const double* pOut);

	/// Counts all the votes from the models in the bag, assuming you are
	/// interested in knowing the distribution.
	void tally(GPrediction* pOut);

	/// Counts all the votes from the models in the bag, assuming you only
	/// care to know the winner, and do not care about the distribution.
	void tally(double* pOut);
};



/// This is an ensemble that uses the bagging approach for training, and Bayesian
/// Model Averaging to combine the models. That is, it trains each model with data
/// drawn randomly with replacement from the original training data. It combines
/// the models with weights proporitional to their likelihood as computed using
/// Bayes' law.
class GBayesianModelAveraging : public GBag
{
public:
	/// General-purpose constructor
	GBayesianModelAveraging(GRand& rand) : GBag(rand) {}

	/// Deserializing constructor.
	GBayesianModelAveraging(GDomNode* pNode, GLearnerLoader& ll) : GBag(pNode, ll) {}

	virtual ~GBayesianModelAveraging() {}

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc);

protected:
	/// See the comment for GLearner::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }

	/// Determines the weights in the manner of Bayesian model averaging,
	/// with the assumption of uniform priors.
	virtual void determineWeights(GMatrix& features, GMatrix& labels);
};




class GBayesianModelCombination : public GBag
{
protected:
	size_t m_samples;

public:
	/// General-purpose constructor
	GBayesianModelCombination(GRand& rand) : GBag(rand), m_samples(100) {}

	/// Deserializing constructor.
	GBayesianModelCombination(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GBayesianModelCombination() {}

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Returns the number of samples from which to estimate the combination weights
	size_t samples() { return m_samples; }

	/// Sets the number of samples to use to estimate the combination weights
	void setSamples(size_t n) { m_samples = n; }

protected:
	/// See the comment for GLearner::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }

	/// Determines the weights in the manner of Bayesian model averaging,
	/// with the assumption of uniform priors.
	virtual void determineWeights(GMatrix& features, GMatrix& labels);
};




/// When Train is called, this performs cross-validation on the training
/// set to determine which learner is the best. It then trains that learner
/// with the entire training set.
class GBucket : public GSupervisedLearner
{
protected:
	size_t m_nBestLearner;
	std::vector<GSupervisedLearner*> m_models;

public:
	/// nInitialSize tells it how fast to grow the dynamic array that holds the
	/// models. It's not really important to get it right, just guess how many
	/// models will go in the ensemble.
	GBucket(GRand& rand);

	/// Load from a DOM.
	GBucket(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GBucket();

#ifndef NO_TEST_CODE
	static void test();
#endif

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc);

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Removes and deletes all the learners
	void flush();

	/// Adds a modeler to the list. This takes ownership of pLearner (so
	/// it will delete it when it's done with it)
	void addLearner(GSupervisedLearner* pLearner);

	/// Returns the modeler that did the best with the training set. It is
	/// your responsibility to delete the modeler this returns. Throws if
	/// you haven't trained yet.
	GSupervisedLearner* releaseBestModeler();

	/// If one of the algorithms throws during training,
	/// it will catch it and call this no-op method. Overload
	/// this method if you don't want to ignore exceptions.
	virtual void onError(std::exception& e);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);
};


} // namespace GClasses

#endif // __GENSEMBLE_H__
