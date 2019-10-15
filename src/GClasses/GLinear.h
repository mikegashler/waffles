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

#ifndef __GLINEAR_H__
#define __GLINEAR_H__

#include "GLearner.h"
#include <vector>

namespace GClasses {

class GPCA;

/// A linear regression model. Let f be a feature vector of real values, and let l be a label vector of real values,
/// then this model estimates l=Bf+e, where B is a matrix of real values, and e is a
/// vector of real values. (In the Wikipedia article on linear regression, B is called
/// "beta", and e is called "epsilon". The approach used by this model to compute
/// beta and epsilon, however, is much more efficient than the approach currently
/// described in that article.)
class GLinearRegressor : public GSupervisedLearner
{
protected:
	GMatrix* m_pBeta;
	GVec m_epsilon;

public:
	GLinearRegressor();

	/// Load from a text-format
	GLinearRegressor(const GDomNode* pNode);

	virtual ~GLinearRegressor();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Saves the model to a text file. (This doesn't save the short-term
	/// memory used for incremental learning, so if you're doing "incremental"
	/// learning, it will wake up with amnesia when you load it again.)
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Returns the matrix that represents the linear transformation.
	GMatrix* beta() { return m_pBeta; }

	/// Returns the vector that is added to the results after the linear transformation is applied.
	GVec& epsilon() { return m_epsilon; }

	/// Performs on-line gradient descent to refine the model
	void refine(const GMatrix& features, const GMatrix& labels, double learningRate, size_t epochs, double learningRateDecayFactor);

	/// This model has no parameters to tune, so this method is a noop.
	void autoTune(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& pIn, GVec& pOut);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& pIn, GPrediction* pOut);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }
};



/// A linear regression model that predicts a distribution. (This algorithm also differs from GLinearRegressor
/// in that it computes its model in closed form instead of using a numerical approach to find it. In general,
/// GLinearRegressor seems to be a bit more accurate. Perhaps the method used in this algorithm is not very
/// numerically stable.)
class GLinearDistribution : public GSupervisedLearner
{
protected:
	double m_noiseDev;
	GMatrix* m_pAInv;
	GMatrix* m_pWBar;
	GVec m_buf;

public:
	/// General-purpose constructor
	GLinearDistribution();

	/// Deserialization constructor
	GLinearDistribution(const GDomNode* pNode);

	/// Destructor
	virtual ~GLinearDistribution();

	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Specify the prior expected deviation of the noise (The default is 1.0.)
	void setNoiseDeviation(double d) { m_noiseDev = d; }

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& pIn, GVec& pOut);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& pIn, GPrediction* pOut);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }
};





class GLinearProgramming
{
public:
	/// Compute x that maximizes c*x, subject to Ax<=b, x>=0.
	/// The size of pB is the number of rows in pA.
	/// The size of pC is the number of columns in pA.
	/// leConstraints specifies the number of <= constraints. (These must come first in order.)
	/// geConstraints specifies the number of >= constraints. (These come next.)
	/// The remaining constraints are assumed to be = constraints.
	/// The answer is put in pOutX, which is the same size as pC.
	/// Returns false if there is no solution, and true if it finds a solution.
	static bool simplexMethod(GMatrix* pA, const double* pB, int leConstraints, int geConstraints, const double* pC, double* pOutX);

	/// Perform unit tests for this class. Throws an exception if any tests fail. Returns if they all pass.
	static void test();
};




} // namespace GClasses

#endif // __GLINEAR_H__

