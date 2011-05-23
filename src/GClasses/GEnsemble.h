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

/// BAG stands for bootstrap aggregator. It represents an ensemble
/// of voting modelers. Each model is trained with a slightly different
/// training set, which is produced by drawing randomly from the original
/// training set with replacement until we have a new training set of
/// the same size. Each model is given equal weight in the vote.
class GBag : public GSupervisedLearner
{
protected:
	sp_relation m_pLabelRel;
	std::vector<GSupervisedLearner*> m_models;
	size_t m_nAccumulatorDims;
	double* m_pAccumulator;
	GRand* m_pRand;
	EnsembleProgressCallback m_pCB;
	void* m_pThis;

public:
	/// nInitialSize tells it how fast to grow the dynamic array that holds the
	/// models. It's not really important to get it right, just guess how many
	/// models will go in the ensemble.
	GBag(GRand* pRand);

	/// Load from a DOM.
	GBag(GDomNode* pNode, GRand* pRand, GLearnerLoader* pLoader);

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

	void accumulate(const double* pOut);
	void tally(size_t nCount, GPrediction* pOut);
	void tally(size_t nCount, double* pOut);
};



/// When Train is called, this performs cross-validation on the training
/// set to determine which learner is the best. It then trains that learner
/// with the entire training set.
class GBucket : public GSupervisedLearner
{
protected:
	size_t m_nBestLearner;
	std::vector<GSupervisedLearner*> m_models;
	GRand* m_pRand;

public:
	/// nInitialSize tells it how fast to grow the dynamic array that holds the
	/// models. It's not really important to get it right, just guess how many
	/// models will go in the ensemble.
	GBucket(GRand* pRand);

	/// Load from a DOM.
	GBucket(GDomNode* pNode, GRand* pRand, GLearnerLoader* pLoader);

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
