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

#ifndef __GGAUSSIANPROCESS_H__
#define __GGAUSSIANPROCESS_H__

#include "GMatrix.h"
#include "GLearner.h"

namespace GClasses {

class GKernel;

/// Computes a running covariance matrix about the origin.
class GRunningCovariance
{
protected:
	GMatrix m_counts; // upper triangle contains the counts of the number of relevant samples.
	GMatrix m_sums; // upper triangle contains sums of coproducts. Last row contains sums of samples.

public:
	/// Prepares to compute a dims x dims covariance matrix. All attributes are
	/// assumed to be continuous.
	GRunningCovariance(size_t dims);
	~GRunningCovariance();

	static void test();

	/// Adds a vector to the covariance. (Remember that covariance is computed
	/// about the origin, not the centroid. If pVec comes from a distribution with a
	/// non-zero mean, then you may need to subtract the mean before calling this method.)
	/// It is okay for some elements of pVec to be UNKNOWN_REAL_VALUE.
	void add(const GVec& vec);

	/// Multiplies relevant internal counts and sums by gamma. (This is used
	/// for computing "moving covariance" instead of "running covariance".)
	void decay(double gamma);

	/// Returns the current specified element of the covariance matrix.
	/// Returns UNKNOWN_REAL_VALUE if one or fewer relevant samples was added.
	double element(size_t row, size_t col);
};


/// A Gaussian Process model. This class was implemented according to the specification
/// in Algorithm 2.1 on page 19 of chapter 2 of http://www.gaussianprocesses.org/gpml/chapters/
/// by Carl Edward Rasmussen and Christopher K. I. Williams.
class GGaussianProcess : public GSupervisedLearner
{
protected:
	double m_noiseVar;
	double m_weightsPriorVar;
	size_t m_maxSamples;
	GMatrix* m_pLInv;
	GMatrix* m_pAlpha;
	GMatrix* m_pStoredFeatures;
	GMatrix* m_pBuf;
	GKernel* m_pKernel;

public:
	/// General-purpose constructor
	GGaussianProcess();

	/// Deserialization constructor
	GGaussianProcess(const GDomNode* pNode);

	/// Destructor
	virtual ~GGaussianProcess();

	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Sets the kernel to use. Takes ownership of pKernel. (The default is the
	/// identity kernel, which just does linear regression, so you almost certainly
	/// will want to change the kernel before using this model.)
	void setKernel(GKernel* pKernel);

	/// Sets the noise variance term. (The default is 0.0.)
	void setNoiseVariance(double v) { m_noiseVar = v; }

	/// Sets the weight prior variance term. (The default is 1024.0.)
	void setWeightsPriorVariance(double v) { m_weightsPriorVar = v; }

	/// Sets the maximum number of samples to train with. If the training data
	/// contains more than 'm' samples, it will sub-sample the training data
	/// in order to train efficiently. The default is 350.
	void setMaxSamples(size_t m) { m_maxSamples = m; }

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// Called by trainInner
	void trainInnerInner(const GMatrix& features, const GMatrix& labels);
};

} // namespace GClasses

#endif
