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
	size_t m_nSampleCount;
	GNaiveBayesOutputAttr** m_pOutputs;
	double m_equivalentSampleSize;

public:
	GNaiveBayes();

	/// Load from a DOM.
	GNaiveBayes(const GDomNode* pNode);

	virtual ~GNaiveBayes();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GIncrementalLearner::trainSparse
	/// This method assumes that the values in pData are all binary values (0 or 1).
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);

	/// To ensure that unsampled values don't dominate the joint
	/// distribution by multiplying by a zero, each value is given
	/// at least as much representation as specified here. (The default
	/// is 0.5, which is as if there were half of a sample for each value.)
	void setEquivalentSampleSize(double d) { m_equivalentSampleSize = d; }

	/// Returns the equivalent sample size. (The number of samples of each
	/// possible value that is added by default to prevent zeros.)
	double equivalentSampleSize() { return m_equivalentSampleSize; }

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data.
	void autoTune(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// Adds a single training sample to the collection
	virtual void trainIncremental(const GVec& in, const GVec& out);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GTransducer::canImplicitlyHandleContinuousFeatures
	virtual bool canImplicitlyHandleContinuousFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);
};

} // namespace GClasses

#endif // __GNAIVEBAYES_H__
