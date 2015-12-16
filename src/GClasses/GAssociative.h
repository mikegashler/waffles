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

#ifndef __GASSOCIATIVE_H__
#define __GASSOCIATIVE_H__

#include "GLearner.h"
#include <vector>
#include <iostream>
#include "GTree.h"

#define MAX_NODES_PER_LAYER ((size_t)1024 * (size_t)1024 * (size_t)1024)

namespace GClasses {

class GActivationFunction;
class GAssociative;


/// A class used by GAssociative
class GAssociativeLayer
{
public:
	GActivationFunction* m_pActivationFunction;
	GMatrix m_forw; // rows = upstream layer, cols = this layer
	GMatrix m_back; // rows = downstream layer, cols = this layer
	GMatrix m_bias; // row 0 is activation, row 1 is bias, row 2 is clamp, row 3 is delta, row 4 is error, row 5 is net

public:
	GAssociativeLayer(size_t bef, size_t cur, size_t aft, GActivationFunction* pActivationFunction = NULL);
	~GAssociativeLayer();

	size_t units() { return m_forw.cols(); }
	GVec& activation() { return m_bias[0]; }
	GVec& bias() { return m_bias[1]; }
	GVec& clamp() { return m_bias[2]; }
	GVec& delta() { return m_bias[3]; }
	GVec& error() { return m_bias[4]; }
	GVec& net() { return m_bias[5]; }

	void clipWeightMagnitudes(double min, double max);
	void init(GRand& rand);
	void print(std::ostream& stream, size_t i);
};



// Helper class used by GAssociative. The user should not need to use this class
// Col 0 = squared delta, larger first
// Col 1 = node {layer,index}
class GAssociativeNodeComparer
{
protected:
	GAssociative* m_pThat;

public:
	GAssociativeNodeComparer(GAssociative* pThat)
	: m_pThat(pThat)
	{
	}

	size_t cols() const { return 2; }

	bool operator ()(size_t a, size_t b, size_t col) const;
	void print(std::ostream& stream, size_t row) const;
};




/// A class that implements a Hopfield-like associative neural network.
/// The difference is that this class supports different weights in each direction.
/// It also supports multiple layers.
/// It uses a training method that I invented to refine the weights.
class GAssociative : public GIncrementalLearner
{
friend class GAssociativeNodeComparer;
protected:
	std::vector<GAssociativeLayer*> m_layers;
	GAssociativeNodeComparer m_comp;
	GRelationalTable<size_t,GAssociativeNodeComparer> m_table;
	double m_epsilon;

public:
	GAssociative();

	/// Load from a text-format
	GAssociative(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GAssociative();

	void addLayer(GAssociativeLayer* pLayer);

	void print(std::ostream& stream);

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Saves the model to a text file.
	virtual GDomNode* serialize(GDom* pDoc) const;
#endif // MIN_PREDICT

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const GVec& in, const GVec& out);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GIncrementalLearner::trainSparse
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedFeatureRange(double* pOutMin, double* pOutMax)
	{
		*pOutMin = -1.0;
		*pOutMax = 1.0;
		return false;
	}

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedLabelRange(double* pOutMin, double* pOutMax)
	{
		*pOutMin = -1.0;
		*pOutMax = 1.0;
		return false;
	}

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	void clampValues(const GVec& in, const GVec& out);
	void updateDelta(GAssociativeLayer* pLayer, size_t layer, size_t unit, double delta);
	void clamp(GAssociativeLayer* pLayer, size_t layer, size_t unit, double value);
	void updateActivation(GAssociativeLayer* pLay, size_t layer, size_t unit, double netDelta);
	void activatePull(GAssociativeLayer* pLay, size_t lay, size_t unit);
	void propagateActivation();
	void updateBlame(GAssociativeLayer* pLay, size_t layer, size_t unit, double delta);
	double seedBlame();
	void propagateBlame();
	void updateWeights(double learningRate);
};


} // namespace GClasses

#endif // __GASSOCIATIVE_H__

