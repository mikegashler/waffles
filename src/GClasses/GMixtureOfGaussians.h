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

#ifndef __GMIXTUREOFGAUSSIANS_H__
#define __GMIXTUREOFGAUSSIANS_H__

#include "GDistribution.h"

namespace GClasses {

class GMatrix;


/// This class uses Expectency Maximization to find the mixture of Gaussians that best approximates
/// the data in a specified real attribute of a data set.
class GMixtureOfGaussians
{
protected:
	int m_nKernelCount;
	int m_nAttribute;
	double* m_pArrMeanVarWeight;
	double* m_pCatLikelihoods;
	double* m_pTemp;
	GMatrix* m_pData;
	GNormalDistribution m_dist;
	double m_dMinVariance;

public:
	GMixtureOfGaussians(int nKernelCount, GMatrix* pData, int nAttribute, double minVariance, GRand* pRand);
	virtual ~GMixtureOfGaussians();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// This tries to fit the data from several random starting points, and returns the best model it finds
	static GMixtureOfGaussians* stochasticHammer(int nMinKernelCount, int nMaxKernelCount, int nItters, int nTrials, GMatrix* pData, int nAttribute, double minVariance, GRand* pRand);

	/// Returns the log likelihood of the current parameters
	double iterate();

	/// Returns the current parameters of the specified kernel
	void params(int nKernel, double* pMean, double* pVariance, double* pWeight);

protected:
	double evalKernel(double x, int nKernel);
	double likelihoodOfEachCategoryGivenThisFeature(double x);
};

} // namespace GClasses

#endif // __GMIXTUREOFGAUSSIANS_H__
