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

#ifndef __GGREEDYSEARCH_H__
#define __GGREEDYSEARCH_H__

#include "GOptimizer.h"
#include "GVec.h"
#include "GRand.h"

namespace GClasses {


/// At each iteration this algorithm moves in only one
/// dimension. If the situation doesn't improve it tries
/// the opposite direction. If both directions are worse,
/// it decreases the step size for that dimension, otherwise
/// it increases the step size for that dimension.
class GMomentumGreedySearch : public GOptimizer
{
protected:
	size_t m_nDimensions;
	size_t m_nCurrentDim;
	GVec m_pStepSizes;
	GVec m_pVector;
	double m_dError;
	double m_dChangeFactor;

public:
	GMomentumGreedySearch(GTargetFunction* pCritic);
	virtual ~GMomentumGreedySearch();

	/// Performs unit testing. Throws an exception if any test fails.
	static void test();

	/// Returns a pointer to the state vector
	virtual const GVec& currentVector() { return m_pVector; }

	/// Set all the current step sizes to this value
	void setAllStepSizes(double dStepSize);

	/// Returns the vector of step sizes
	GVec& stepSizes();

	virtual double iterate();

	/// d should be a value between 0 and 1
	void setChangeFactor(double d) { m_dChangeFactor = d; }

protected:
	void reset();
	double iterateOneDim();
};


/// In each dimension, tries 5 candidate adjustments:
/// a lot smaller, a little smaller, same spot, a little bigger, and a lot bigger.
/// If it picks a smaller adjustment, the step size in that dimension is made smaller.
/// If it picks a bigger adjustment, the step size in that dimension is made bigger.
class GHillClimber : public GOptimizer
{
protected:
	size_t m_nDims;
	size_t m_dim;
	GVec m_pStepSizes;
	GVec m_pVector;
	GVec m_pAnnealCand;
	double m_dError;
	double m_dChangeFactor;

public:
	GHillClimber(GTargetFunction* pCritic);
	virtual ~GHillClimber();

	/// Performs unit testing. Throws an exception if any test fails.
	static void test();

	/// Returns a pointer to the current vector
	virtual const GVec& currentVector() { return m_pVector; }

	/// Returns the error for the current vector
	double currentError() { return m_dError; }

	/// Set all the current step sizes to this value
	void setStepSizes(double size);

	/// Returns the vector of step sizes
	GVec& stepSizes();

	virtual double iterate();

	/// You can call this method to simulate one annealing jump with the
	/// specified deviation in all dimensions.
	double anneal(double dev, GRand* pRand);

	/// d should be a value between 0 and 1
	void setChangeFactor(double d) { m_dChangeFactor = d; }

protected:
	void reset();
};



/// Perturbs the current vector in a random direction.
/// If it made the vector worse, restores the previous vector.
/// Decays the deviation of perturbation over time.
class GAnnealing : public GOptimizer
{
protected:
	double m_initialDeviation;
	double m_deviation;
	double m_decay;
	size_t m_dims;
	GVec m_pVector;
	GVec m_pCandidate;
	double m_dError;
	GRand* m_pRand;

public:
	GAnnealing(GTargetFunction* pTargetFunc, GRand* pRand);
	virtual ~GAnnealing();

	/// Performs unit testing. Throws an exception if any test fails.
	static void test();

	/// Performs a little more optimization. (Call this in a loop until
	/// acceptable results are found.)
	virtual double iterate();

	/// Returns the best vector yet found.
	virtual const GVec& currentVector() { return m_pVector; }

	/// Specify the current deviation to use for annealing. (A random vector
	/// from a Normal distribution with the specified deviation will be added to each
	/// candidate vector in order to simulate annealing.)
	void setDeviation(double d) { m_deviation = d; }

protected:
	void reset();
};




/// This algorithm picks a random direction, then uses binary search
/// to determine how far to step, and repeats
class GRandomDirectionBinarySearch : public GOptimizer
{
protected:
	GVec m_direction;
	GVec m_current;
	size_t m_dims;
	double m_stepSize;
	double m_err;
	GRand* m_pRand;

public:
	GRandomDirectionBinarySearch(GTargetFunction* pTargetFunc, GRand* pRand);
	virtual ~GRandomDirectionBinarySearch();

	/// Performs unit testing. Throws an exception if any test fails.
	static void test();

	/// Performs a little more optimization. (Call this in a loop until
	/// acceptable results are found.)
	virtual double iterate();

	/// Returns the best vector yet found.
	virtual const GVec& currentVector() { return m_current; }
};




/// This algorithm does a gradient descent by feeling a small distance
/// out in each dimension to measure the gradient. For efficiency reasons,
/// it only measures the gradient in one dimension (which it cycles
/// round-robin style) per iteration and uses the remembered gradient
/// in the other dimensions.
class GEmpiricalGradientDescent : public GOptimizer
{
protected:
	double m_dLearningRate;
	size_t m_nDimensions;
	GVec m_pVector;
	GVec m_pGradient;
	GVec m_pDelta;
	double m_dFeelDistance;
	double m_dMomentum;
	GRand* m_pRand;

public:
	GEmpiricalGradientDescent(GTargetFunction* pCritic, GRand* pRand);
	virtual ~GEmpiricalGradientDescent();

	/// Returns the best vector yet found.
	virtual const GVec& currentVector() { return m_pVector; }

	/// Performs a little more optimization. (Call this in a loop until
	/// acceptable results are found.)
	virtual double iterate();

	/// Sets the learning rate
	void setLearningRate(double d) { m_dLearningRate = d; }

	/// Sets the momentum value
	void setMomentum(double d) { m_dMomentum = d; }

protected:
	void reset();
};



/// This is a variant of empirical gradient descent that tries to estimate
/// the gradient using a minimal number of samples. It is more efficient
/// than empirical gradient descent, but it only works well if the optimization
/// surface is quite locally linear.
class GSampleClimber : public GOptimizer
{
protected:
	GRand* m_pRand;
	double m_dStepSize;
	double m_alpha;
	double m_error;
	size_t m_dims;
	GVec m_pVector;
	GVec m_pDir;
	GVec m_pCand;
	GVec m_pGradient;

public:
	GSampleClimber(GTargetFunction* pCritic, GRand* pRand);
	virtual ~GSampleClimber();

	/// Returns the best vector yet found
	virtual const GVec& currentVector() { return m_pVector; }
	
	/// Performs a little more optimization. (Call this in a loop until
	/// acceptable results are found.)
	virtual double iterate();

	/// Sets the current step size
	void setStepSize(double d) { m_dStepSize = d; }

	/// Sets the alpha value. It should be small (like 0.01)
	/// A very small value updates the gradient estimate
	/// slowly, but precisely. A bigger value updates the
	/// estimate quickly, but never converges very close to
	/// the precise gradient.
	void setAlpha(double d) { m_alpha = d; }

protected:
	void reset();
};


} // namespace GClasses

#endif // __GGREEDYSEARCH_H__
