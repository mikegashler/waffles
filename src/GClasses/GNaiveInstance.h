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
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#ifndef __GNAIVEINSTANCE_H__
#define __GNAIVEINSTANCE_H__

#include "GLearner.h"

namespace GClasses {

class GNaiveInstanceAttr;
class GHeap;


/// This is an instance-based learner. Instead of finding the k-nearest
/// neighbors of a feature vector, it finds the k-nearst neighbors in each
/// dimension. That is, it finds n*k neighbors, considering each dimension
/// independently. It then combines the label from all of these neighbors
/// to make a prediction. Finding neighbors in this way makes it more robust to
/// high-dimensional datasets. It tends to perform worse than k-nn in low-dimensional space, and better
/// than k-nn in high-dimensional space. (It may be thought of as a cross
/// between a k-nn instance learner and a Naive Bayes learner. It only
/// supports continuous features and labels (so it is common to wrap it
/// in a Categorize filter which will convert nominal features to a categorical
/// distribution of continuous values).
class GNaiveInstance : public GIncrementalLearner
{
protected:
	size_t m_nNeighbors;
	GNaiveInstanceAttr** m_pAttrs;
	GVec m_pValueSums;
	GVec m_pWeightSums;
	GVec m_pSumBuffer;
	GVec m_pSumOfSquares;
	GHeap* m_pHeap;

public:
	/// nNeighbors is the number of neighbors (in each dimension)
	/// that will contribute to the output value.
	GNaiveInstance();

	/// Deserializing constructor
	GNaiveInstance(const GDomNode* pNode);
	virtual ~GNaiveInstance();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Specify the number of neighbors to use.
	void setNeighbors(size_t k) { m_nNeighbors = k; }

	/// Returns the number of neighbors.
	size_t neighbors() { return m_nNeighbors; }

	/// See the comment for GIncrementalLearner::trainSparse.
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::clear.
	virtual void clear();

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data.
	void autoTune(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// Incrementally train with a single instance
	virtual void trainIncremental(const GVec& in, const GVec& out);

protected:
	void evalInput(size_t nInputDim, double dInput);

	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);
};

} // namespace GClasses

#endif // __GNAIVEINSTANCE_H__

