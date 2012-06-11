/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
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

#ifndef NO_TEST_CODE
	static void test();
#endif

	/// Adds a vector to the covariance. (Remember that covariance is computed
	/// about the origin, not the centroid. If pVec comes from a distribution with a
	/// non-zero mean, then you may need to subtract the mean before calling this method.)
	/// It is okay for some elements of pVec to be UNKNOWN_REAL_VALUE.
	void add(const double* pVec);

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
	GGaussianProcess(GRand& rand);

	/// Deserialization constructor
	GGaussianProcess(GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GGaussianProcess();

#ifndef NO_TEST_CODE
	static void test();
#endif

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
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// Called by trainInner
	void trainInnerInner(GMatrix& features, GMatrix& labels);
};

} // namespace GClasses

#endif
