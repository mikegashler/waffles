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

#ifndef __GSTABSEARCH_H__
#define __GSTABSEARCH_H__

#include "GOptimizer.h"
#include "GRand.h"
#include "GVec.h"

namespace GClasses {

/// This performs a brute-force grid search with uniform sampling over the
/// unit hypercube with increasing granularity. (Your target function should scale
/// the candidate vectors as necessary to cover the desired space.) This grid-search
/// increases the granularity after each pass, and carefully avoids sampling anywhere
/// that it has sampled before.
class GGridSearch : public GOptimizer
{
protected:
	GVec m_pCandidate;
	GVec m_pBestVector;
	double m_bestError;
	GCoordVectorIterator* m_pCvi;

public:
	GGridSearch(GTargetFunction* pCritic);
	virtual ~GGridSearch();

	/// Each pass will complete after ((2^n)+1)^d iterations. The distance between
	/// samples at that point will be 1/(2^n). After it completes n=30, it will begin repeating.
	virtual double iterate();

	/// Returns the best vector yet found
	virtual const GVec& currentVector();
};



/// At each iteration, this tries a random vector from the unit
/// hypercube. (Your target function should scale
/// the candidate vectors as necessary to cover the desired space.)
class GRandomSearch : public GOptimizer
{
protected:
	GRand* m_pRand;
	GVec m_pCandidate;
	GVec m_pBestVector;
	double m_bestError;

public:
	GRandomSearch(GTargetFunction* pCritic, GRand* pRand);
	virtual ~GRandomSearch();

	/// Try another random vector
	virtual double iterate();

	/// Returns the best vector yet found
	virtual const GVec& currentVector();
};



/// This is a hill climber for semi-linear error surfaces that minimizes testing with
/// an approach like binary-search.
/// It only searches approximately within the unit cube (although it may stray a little
/// outside of it). It is the target function's responsibility
/// to map this into an appropriate space.
class GMinBinSearch : public GOptimizer
{
protected:
	size_t m_curDim;
	double m_stepSize;
	GVec m_pCurrent;
	double m_curErr;

public:
	GMinBinSearch(GTargetFunction* pCritic);
	virtual ~GMinBinSearch();

	/// Try another random vector
	virtual double iterate();

	/// Returns the best vector yet found
	virtual const GVec& currentVector() { return m_pCurrent; }
};



/// This is somewhat of a multi-dimensional version of binary-search.
/// It greedily probes the best choices first, but then starts trying
/// the opposite choices at the higher divisions so that it can also
/// handle non-monotonic target functions.
/// Each iteration performs a binary (divide-and-conquer) search
/// within the unit hypercube. (Your target function should scale
/// the candidate vectors as necessary to cover the desired space.)
/// Because the high-level divisions are typically less correlated
/// with the quality of the final result than the low-level divisions,
/// it searches through the space of possible "probes" by toggling choices in
/// the order from high level to low level. In low-dimensional space, this
/// algorithm tends to quickly find good solutions, especially if the
/// target function is somewhat smooth. In high-dimensional space, the
/// number of iterations to find a good solution seems to grow exponentially.
class GProbeSearch : public GOptimizer
{
protected:
	GRand m_rand;
	size_t m_nDimensions;
	unsigned int m_nMask[4];
	GVec m_pMins;
	GVec m_pMaxs;
	GVec m_pVector;
	GVec m_pBestYet;
	double m_bestError;
	size_t m_nStabDepth;
	size_t m_nCurrentDim;
	size_t m_nDepth;
	size_t m_nStabs;
	size_t m_samples;

public:
	GProbeSearch(GTargetFunction* pCritic);
	virtual ~GProbeSearch();

	/// Do a little bit more work toward finding a good vector
	virtual double iterate();

	/// Returns the best vector yet found
	virtual const GVec& currentVector() { return m_pBestYet; }

	/// Specify the number of times to divide the space before
	/// satisfactory accuracy is obtained. Larger values will
	/// result in more computation, but will find more precise
	/// values. For most problems, 20 to 30 should be sufficient.
	void setStabDepth(size_t n) { m_nStabDepth = m_nDimensions * n; }

	/// Returns the total number of completed stabs
	size_t stabCount() { return m_nStabs; }

	/// Specify the number of vectors to use to sample each side of a
	/// binary-split division.
	void setSampleCount(size_t n) { m_samples = n; }

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

protected:
	void resetStab();
	void reset();
	double sample(bool greater);
};


} // namespace GClasses

#endif // __GSTABSEARCH_H__
