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

#ifndef __GKNN_H__
#define __GKNN_H__

#include "GLearner.h"

namespace GClasses {

class GNeighborFinderGeneralizing;
class GRand;
class GKnnScaleFactorCritic;
class GOptimizer;
class GRowDistanceScaled;
class GSparseSimilarity;
class GBitTable;


/// The k-Nearest Neighbor learning algorithm
class GKNN : public GIncrementalLearner
{
public:
	enum InterpolationMethod
	{
		Linear,
		Mean,
		Learner,
	};

	enum TrainMethod
	{
		StoreAll,
		ValidationPrune,
		DrawRandom,
	};

protected:
	// Settings
	GMatrix* m_pFeatures;
	GSparseMatrix* m_pSparseFeatures;
	GMatrix* m_pLabels;
	size_t m_nNeighbors;
	InterpolationMethod m_eInterpolationMethod;
	TrainMethod m_eTrainMethod;
	double m_trainParam;
	GSupervisedLearner* m_pLearner;
	bool m_bOwnLearner;

	// Scale Factor Optimization
	bool m_normalizeScaleFactors;
	bool m_optimizeScaleFactors;
	GDistanceMetric* m_pDistanceMetric;
	GSparseSimilarity* m_pSparseMetric;
	bool m_ownMetric;
	GKnnScaleFactorCritic* m_pCritic;
	GOptimizer* m_pScaleFactorOptimizer;

	// Working Buffers
	GVec m_valueCounts;

	// Neighbor Finding
	GNeighborFinderGeneralizing* m_pNeighborFinder;

public:
	/// General-purpose constructor
	GKNN();

	/// Load from a DOM.
	GKNN(const GDomNode* pNode);

	virtual ~GKNN();

	/// Returns the number of neighbors
	size_t neighborCount() { return m_nNeighbors; }

	/// Specify the number of neighbors to use. (The default is 1.)
	void setNeighborCount(size_t k);

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GIncrementalLearner::trainSparse
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);

	/// Discard any training (but not any settings) so it can be trained again
	virtual void clear();

	/// Sets the distance metric to use for finding neighbors. If own is true, then
	/// this object will delete pMetric when it is done with it.
	void setMetric(GDistanceMetric* pMetric, bool own);

	/// Sets the sparse similarity metric to use for finding neighbors. If own is true, then
	/// this object will delete pMetric when it is done with it.
	void setMetric(GSparseSimilarity* pMetric, bool own);

	/// Sets the technique for interpolation. (If you want to use the "Learner" method,
	/// you should call SetInterpolationLearner instead of this method.)
	void setInterpolationMethod(InterpolationMethod eMethod);

	/// Sets the interpolation method to "Learner" and sets the learner to use. If
	/// bTakeOwnership is true, it will delete the learner when this object is deleted.
	void setInterpolationLearner(GSupervisedLearner* pLearner, bool bTakeOwnership);

	/// Adds a copy of pVector to the internal set.
	size_t addVector(const GVec& in, const GVec& out);

	/// Returns the dissimilarity metric
	GDistanceMetric* metric() { return m_pDistanceMetric; }

	/// Specify whether to normalize the scaling of each attribute. (The default is to normalize.)
	void setNormalizeScaleFactors(bool b);

	/// If you set this to true, it will use a hill-climber to optimize the
	/// attribute scaling factors. If you set it to false (the default), it won't.
	void setOptimizeScaleFactors(bool b);

	/// Returns the internal feature set
	GMatrix* features() { return m_pFeatures; }

	/// Returns the internal set of sparse features
	GSparseMatrix* sparseFeatures() { return m_pSparseFeatures; }

	/// Returns the internal label set
	GMatrix* labels() { return m_pLabels; }

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data.
	void autoTune(GMatrix& features, GMatrix& labels);

	/// Specify to train by drawing 'n' random patterns from the training set.
	void drawRandom(size_t n)
	{
		m_eTrainMethod = DrawRandom;
		m_trainParam = (double)n;
	}

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	/// Adds a vector to the internal set. Also, if the (k+1)th nearest
	/// neighbor of that vector is less than "elbow room" from it, then
	/// the closest neighbor is deleted from the internal set. (You might
	/// be wondering why the decision to delete the closest neighbor is
	/// determined by the distance of the (k+1)th neigbor. This enables a
	/// clump of k points to form in the most frequently sampled locations.
	/// Also, If you make this decision based on a closer neighbor, then big
	/// holes may form in the model if points are sampled in a poor order.)
	/// Call SetElbowRoom to specify the elbow room distance.
	virtual void trainIncremental(const GVec& in, const GVec& out);

	/// Finds the nearest neighbors of pVector. Returns the number of neighbors found.
	size_t findNeighbors(const GVec& vector);

	/// Interpolate with each neighbor having equal vote
	void interpolateMean(size_t nc, const GVec& in, GPrediction* pOut, GVec* pOut2);

	/// Interpolate with each neighbor having a linear vote. (Actually it's linear with
	/// respect to the squared distance instead of the distance, because this is faster
	/// to compute.)
	void interpolateLinear(size_t nc, const GVec& in, GPrediction* pOut, GVec* pOut2);

	/// Interpolates with the provided supervised learning algorithm
	void interpolateLearner(size_t nc, const GVec& in, GPrediction* pOut, GVec* pOut2);

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }
};


/// An instance-based transduction algorithm
class GNeighborTransducer : public GTransducer
{
protected:
	size_t m_friendCount;

public:
	/// General-purpose constructor
	GNeighborTransducer();

	/// Returns the number of neighbors.
	size_t neighbors() { return m_friendCount; }

	/// Specify the number of neighbors to use with each point.
	void setNeighbors(size_t k) { m_friendCount = k; }

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data.
	void autoTune(GMatrix& features, GMatrix& labels);

protected:
	/// See the comment for GTransducer::transduce
	virtual std::unique_ptr<GMatrix> transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }
};


/// This represents a grid of values. It might be useful as a Q-table with Q-learning.
class GInstanceTable : public GIncrementalLearner
{
protected:
	size_t m_dims;
	size_t* m_pDims;
	size_t* m_pScales;
	double* m_pTable;
	size_t m_product;

public:
	/// dims specifies the number of feature dimensions.
	/// pDims specifies the number of discrete zero-based values for each feature dim.
	GInstanceTable(size_t dims, size_t* pDims);
	virtual ~GInstanceTable();

	/// Serialize this table
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GIncrementalLearner::trainSparse
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);

	/// Clears the internal model
	virtual void clear();

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const GVec& in, const GVec& out);
};


/// An experimental instance-based learner that prunes attribute-values (or predicates) instead of entire instances.
/// This may be viewed as a form of rule-learning that beings with instance-based learning and then prunes.
class GSparseInstance : public GSupervisedLearner
{
protected:
	size_t m_neighborCount;
	GSparseMatrix* m_pInstanceFeatures;
	GMatrix* m_pInstanceLabels;
	GSparseSimilarity* m_pMetric;
	GBitTable* m_pSkipRows;

public:
	/// General-purpose constructor
	GSparseInstance();

	/// Load from a DOM.
	GSparseInstance(const GDomNode* pNode);

	virtual ~GSparseInstance();

	/// Returns the number of neighbors
	size_t neighborCount() { return m_neighborCount; }

	/// Specify the number of neighbors to use. (The default is 1.)
	void setNeighborCount(size_t k);

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Discard any training (but not any settings) so it can be trained again
	virtual void clear();

	/// Sets the distance metric. Takes ownership of pMetric.
	void setMetric(GSparseSimilarity* pMetric);

protected:
	/// Throw out elements and/or entire instances one at-a-time until no improvements can be found
	void prune(const GMatrix& holdOutFeatures, const GMatrix& holdOutLabels);

	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GSupervisedLearner::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GSupervisedLearner::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }
};

} // namespace GClasses

#endif // __GKNN_H__
