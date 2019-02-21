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

#ifndef __GENSEMBLE_H__
#define __GENSEMBLE_H__

#include "GLearner.h"
#include <vector>
#include <exception>

namespace GClasses {

class GRelation;
class GRand;
class GMasterThread;
class GNeuralNetLearner;


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

	/// Sets the weight of this model
	void setWeight(double w) { m_weight = w; }

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	GDomNode* serialize(GDom* pDoc) const;
};


/// This is a base-class for ensembles that combine the
/// predictions from multiple weighed models.
class GEnsemble : public GSupervisedLearner
{
protected:
	GRelation* m_pLabelRel;
	std::vector<GWeightedModel*> m_models;
	GVec m_accumulator; // a buffer for tallying votes (ballot box?)

	size_t m_workerThreads;
	GMasterThread* m_pPredictMaster;
public:
	volatile const GVec* m_pPredictInput;

	/// General-purpose constructor. See also the comment for GSupervisedLearner::GSupervisedLearner.
	GEnsemble();

	/// Deserializing constructor.
	GEnsemble(const GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GEnsemble();

	/// Returns a reference to the models in the ensemble
	std::vector<GWeightedModel*>& models() { return m_models; }

	/// Adds the vote from one of the models. (This is called internally. Users typically
	/// do not need to call it.)
	void castVote(double weight, const GVec& label);

	/// Specify the number of worker threads to use. If count is 1,
	/// then no additional threads will be spawned, but the work will
	/// all be done by the same thread. If count is 2 or more, that
	/// number of worker threads will be spawned. (Note that with fast models,
	/// the overhead associated with worker threads is often too high to be
	/// worthwhile.) The worker threads are spawned when the first prediction
	/// is made. They are kept alive until clear() is called or this object is
	/// deleted. If you only want to use worker threads during training, but
	/// not when making predictions, you can call this method again to set it back
	/// to 1 after training is complete. Since the inheriting class is
	/// responsible to implement the train method, some child classes may not
	/// implement multi-threaded training. GBag, GBomb, GBayesianModelAveraging,
	/// and GBayesianModelCombination all implement multi-threaded training.
	void setWorkerThreads(size_t count) { m_workerThreads = count; }

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

protected:
	/// Base classes should call this method to serialize the base object
	/// as part of their implementation of the serialize method.
	virtual void serializeBase(GDom* pDoc, GDomNode* pNode) const;

	/// Calls clear on all of the models, and resets the accumulator buffer
	virtual void clearBase();

	/// Sets up the accumulator buffer (ballot box) then calls trainInnerInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// Implement this method to train the ensemble.
	virtual void trainInnerInner(const GMatrix& features, const GMatrix& labels) = 0;

	/// Scales the weights of all the models so they sum to 1.0.
	void normalizeWeights();

	/// Counts all the votes from the models in the bag, assuming you are
	/// interested in knowing the distribution.
	void tally(GPrediction* pOut);

	/// Counts all the votes from the models in the bag, assuming you only
	/// care to know the winner, and do not care about the distribution.
	void tally(GVec& label);
};



/// BAG stands for bootstrap aggregator. It represents an ensemble
/// of voting models. Each model is trained with a slightly different
/// training set, which is produced by drawing randomly from the original
/// training set with replacement until we have a new training set of
/// the same size. Each model is given equal weight in the vote.
class GBag : public GEnsemble
{
protected:
	EnsembleProgressCallback m_pCB;
	void* m_pThis;
	double m_trainSize;

public:
	/// General-purpose constructor. See also the comment for GSupervisedLearner::GSupervisedLearner.
	GBag();

	/// Deserializing constructor.
	GBag(const GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GBag();

	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Calls clears on all of the learners, but does not delete them.
	virtual void clear();

	/// Removes and deletes all the learners.
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
	/// See the comment for GEnsemble::trainInnerInner
	virtual void trainInnerInner(const GMatrix& features, const GMatrix& labels);

	/// Assigns uniform weight to all models. (This method is deliberately
	/// virtual so that you can overload it if you want non-uniform weighting.)
	virtual void determineWeights(const GMatrix& features, const GMatrix& labels);
};


/// This is an ensemble that uses the bagging approach for training, and Bayesian
/// Model Averaging to combine the models. That is, it trains each model with data
/// drawn randomly with replacement from the original training data. It combines
/// the models with weights proporitional to their likelihood as computed using
/// Bayes' law.
class GBayesianModelAveraging : public GBag
{
public:
	/// General-purpose constructor. See also the comment for GSupervisedLearner::GSupervisedLearner.
	GBayesianModelAveraging() : GBag() {}

	/// Deserializing constructor.
	GBayesianModelAveraging(const GDomNode* pNode, GLearnerLoader& ll) : GBag(pNode, ll) {}

	virtual ~GBayesianModelAveraging() {}

	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const override;

protected:
	/// See the comment for GLearner::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() override {
	  return false;
	}

	/// Determines the weights in the manner of Bayesian model averaging,
	/// with the assumption of uniform priors.
	virtual void determineWeights(const GMatrix& features, const GMatrix& labels) override;
};




class GBayesianModelCombination : public GBag
{
protected:
	size_t m_samples;

public:
	/// General-purpose constructor. See also the comment for GSupervisedLearner::GSupervisedLearner.
	GBayesianModelCombination() : GBag(), m_samples(100) {}

	/// Deserializing constructor.
	GBayesianModelCombination(const GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GBayesianModelCombination() {}

	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Returns the number of samples from which to estimate the combination weights
	size_t samples() { return m_samples; }

	/// Sets the number of samples to use to estimate the combination weights
	void setSamples(size_t n) { m_samples = n; }

protected:
	/// See the comment for GLearner::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() override
	{
		return false;
	}

	/// Determines the weights in the manner of Bayesian model averaging,
	/// with the assumption of uniform priors.
	virtual void determineWeights(const GMatrix& features, const GMatrix& labels) override;
};



/// This is an implementation of AdaBoost, except instead of using weighted samples,
/// it resamples the training set by giving each sample a probability proportional to
/// its weight. This difference enables it to work with algorithms that do not
/// support weighted samples.
class GResamplingAdaBoost : public GEnsemble
{
protected:
	GSupervisedLearner* m_pLearner;
	bool m_ownLearner;
	GLearnerLoader* m_pLoader;
	double m_trainSize;
	size_t m_ensembleSize;

public:
	/// General purpose constructor. pLearner is the learning algorithm
	/// that you wish to boost. If ownLearner is true, then this object
	/// will delete pLearner when it is deleted.
	/// pLoader is a GLearnerLoader that can load the model you wish to boost.
	/// (If it is a custom model, then you also need to make a class that inherits
	/// from GLearnerLoader that can load your custom class.) Takes ownership
	/// of pLoader (meaning this object will delete pLoader when it is deleted).
	GResamplingAdaBoost(GSupervisedLearner* pLearner, bool ownLearner, GLearnerLoader* pLoader);

	/// Deserializing constructor
	GResamplingAdaBoost(const GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GResamplingAdaBoost();

	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Deletes all of the models in this ensemble, and calls clear on the base learner.
	virtual void clear();

	/// Specify the size of the drawn set to train with (as a portion of the training
	/// set). The default is 1.0.
	void setTrainSize(double d) { m_trainSize = d; }

	/// Specify the maximum number of learners to ensemble. The default is 30.
	void setSize(size_t n) { m_ensembleSize = n; }

protected:
	/// See the comment for GLearner::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }

	/// See the comment for GEnsemble::trainInnerInner
	virtual void trainInnerInner(const GMatrix& features, const GMatrix& labels);
};



class GGradBoost : public GEnsemble
{
protected:
	GSupervisedLearner* m_pLearner;
	bool m_ownLearner;
	GLearnerLoader* m_pLoader;
	double m_trainSize;
	size_t m_ensembleSize;
	GVec m_labelCentroid;

public:
	/// General purpose constructor. pLearner is the learning algorithm
	/// that you wish to boost. If ownLearner is true, then this object
	/// will delete pLearner when it is deleted.
	/// pLoader is a GLearnerLoader that can load the model you wish to boost.
	/// (If it is a custom model, then you also need to make a class that inherits
	/// from GLearnerLoader that can load your custom class.) Takes ownership
	/// of pLoader (meaning this object will delete pLoader when it is deleted).
	GGradBoost(GSupervisedLearner* pLearner, bool ownLearner, GLearnerLoader* pLoader);

	/// Deserializing constructor
	GGradBoost(const GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GGradBoost();

	// todo: this method does not yet benefit from multiple threads
	virtual void predict(const GVec& in, GVec& out);

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Deletes all of the models in this ensemble, and calls clear on the base learner.
	virtual void clear();

	/// Specify the portion of the training set to be used for training each model. The default is 1.0.
	void setTrainSize(double d) { m_trainSize = d; }

	/// Specify the maximum number of learners to ensemble. The default is 30.
	void setSize(size_t n) { m_ensembleSize = n; }

protected:
	/// See the comment for GLearner::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }

	/// See the comment for GEnsemble::trainInnerInner
	virtual void trainInnerInner(const GMatrix& features, const GMatrix& labels);
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
	/// General-purpose constructor. See also the comment for GSupervisedLearner::GSupervisedLearner.
	GBucket();

	/// Deserializing constructor
	GBucket(const GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GBucket();

	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

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

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);
};


} // namespace GClasses

#endif // __GENSEMBLE_H__
