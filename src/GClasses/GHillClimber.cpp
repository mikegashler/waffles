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

#include "GHillClimber.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "GImage.h"
#include "GBitTable.h"
#include <cmath>

using namespace GClasses;

GMomentumGreedySearch::GMomentumGreedySearch(GTargetFunction* pCritic)
: GOptimizer(pCritic), m_pStepSizes(pCritic->relation()->size()), m_pVector(pCritic->relation()->size())
{
	if(!pCritic->relation()->areContinuous(0, pCritic->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_nDimensions = pCritic->relation()->size();
	m_nCurrentDim = 0;
	m_dChangeFactor = .87;
	reset();
}

/*virtual*/ GMomentumGreedySearch::~GMomentumGreedySearch()
{
}

void GMomentumGreedySearch::reset()
{
	setAllStepSizes(0.1);
	m_pCritic->initVector(m_pVector);
	if(m_pCritic->isStable())
		m_dError = m_pCritic->computeError(m_pVector);
	else
		m_dError = 1e308;
}

void GMomentumGreedySearch::setAllStepSizes(double dStepSize)
{
	m_pStepSizes.fill(dStepSize);
}

GVec& GMomentumGreedySearch::stepSizes()
{
	return m_pStepSizes;
}

double GMomentumGreedySearch::iterateOneDim()
{
	m_pVector[m_nCurrentDim] += m_pStepSizes[m_nCurrentDim];
	double dError = m_pCritic->computeError(m_pVector);
	if(dError >= m_dError)
	{
		m_pVector[m_nCurrentDim] -= m_pStepSizes[m_nCurrentDim];
		m_pVector[m_nCurrentDim] -= m_pStepSizes[m_nCurrentDim];
		dError = m_pCritic->computeError(m_pVector);
		if(dError >= m_dError)
			m_pVector[m_nCurrentDim] += m_pStepSizes[m_nCurrentDim];
	}
	if(dError >= m_dError)
		m_pStepSizes[m_nCurrentDim] *= m_dChangeFactor;
	else
	{
		m_pStepSizes[m_nCurrentDim] /= m_dChangeFactor;
		if(m_pStepSizes[m_nCurrentDim] > 1e12)
			m_pStepSizes[m_nCurrentDim] = 1e12;
		m_dError = dError;
	}
	if(++m_nCurrentDim >= m_nDimensions)
		m_nCurrentDim = 0;
	return m_dError;
}

/*virtual*/ double GMomentumGreedySearch::iterate()
{
	for(size_t i = 1; i < m_nDimensions; i++)
		iterateOneDim();
	return iterateOneDim();
}

// static
void GMomentumGreedySearch::test()
{
	GOptimizerBasicTestTargetFunction target;
	GMomentumGreedySearch opt(&target);
	opt.basicTest(1e-32);
}

// --------------------------------------------------------------------------------


GHillClimber::GHillClimber(GTargetFunction* pCritic)
: GOptimizer(pCritic), m_dim(0), m_pStepSizes(pCritic->relation()->size()), m_pVector(pCritic->relation()->size())
{
	if(!pCritic->relation()->areContinuous(0, pCritic->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_nDims = pCritic->relation()->size();
	m_dChangeFactor = .83;
	reset();
}

/*virtual*/ GHillClimber::~GHillClimber()
{
}

void GHillClimber::reset()
{
	setStepSizes(0.1);
	m_pCritic->initVector(m_pVector);
	if(m_pCritic->isStable())
		m_dError = m_pCritic->computeError(m_pVector);
	else
		m_dError = 1e308;
}

void GHillClimber::setStepSizes(double size)
{
	m_pStepSizes.fill(size);
}

GVec& GHillClimber::stepSizes()
{
	return m_pStepSizes;
}

/*virtual*/ double GHillClimber::iterate()
{
	double decel = m_pStepSizes[m_dim] * m_dChangeFactor;
	if(std::abs(decel) < 1e-16)
		decel = 0.1;
	double accel = m_pStepSizes[m_dim] / m_dChangeFactor;
	if(std::abs(accel) > 1e14)
		accel = 0.1;
	if(!m_pCritic->isStable())
		m_dError = m_pCritic->computeError(m_pVector); // Current spot
	m_pVector[m_dim] += decel;
	double decScore = m_pCritic->computeError(m_pVector); // Forward decelerated
	m_pVector[m_dim] -= decel; // undo
	m_pVector[m_dim] += accel;
	double accScore = m_pCritic->computeError(m_pVector); // Forward accelerated
	if(m_dError < decScore && m_dError < accScore)
	{
		m_pVector[m_dim] -= accel; // undo
		m_pVector[m_dim] -= decel;
		decScore = m_pCritic->computeError(m_pVector); // Reverse decelerated
		m_pVector[m_dim] += decel; // undo
		m_pVector[m_dim] -= accel;
		accScore = m_pCritic->computeError(m_pVector); // Reverse accelerated
		if(m_dError < decScore && m_dError < accScore)
		{
			// Stay put and decelerate
			m_pVector[m_dim] += accel;
			m_pStepSizes[m_dim] = decel;
		}
		else if(decScore < accScore)
		{
			// Reverse and decelerate
			m_pVector[m_dim] += accel;
			m_pVector[m_dim] -= decel;
			m_dError = decScore;
			m_pStepSizes[m_dim] = -decel;
		}
		else
		{
			// Reverse and accelerate
			m_dError = accScore;
			m_pStepSizes[m_dim] = -accel;
		}
	}
	else if(decScore < accScore)
	{
		// Forward and decelerate
		m_pVector[m_dim] -= accel;
		m_pVector[m_dim] += decel;
		m_dError = decScore;
		m_pStepSizes[m_dim] = decel;
	}
	else if(decScore == accScore)
	{
		if(m_dError == decScore)
		{
			// Neither accelerate nor decelerate. If we're on a temporary plateau
			// on the target function, slowing down would be bad because we might
			// never get off the plateau. If we're at the max error, speeding up
			// would be bad, because we'd just run off to infinity. We will, however
			// move at the accelerated rate, just so we're going somewhere.
		}
		else
		{
			// Forward and accelerate
			m_dError = accScore;
			m_pStepSizes[m_dim] = accel;
		}
	}
	else
	{
		// Forward and accelerate
		m_dError = accScore;
		m_pStepSizes[m_dim] = accel;
	}

	if(++m_dim >= m_nDims)
		m_dim = 0;
	return m_dError;
}

double GHillClimber::anneal(double dev, GRand* pRand)
{
	if(!m_pCritic->isStable())
		m_dError = m_pCritic->computeError(m_pVector); // Current spot
	m_pAnnealCand.resize(m_nDims);
	for(size_t i = 0; i < m_nDims; i++)
		m_pAnnealCand[i] = m_pVector[i] + pRand->normal() * dev;
	double err = m_pCritic->computeError(m_pAnnealCand);
	if(err < m_dError)
	{
		m_dError = err;
		m_pAnnealCand.swapContents(m_pVector);
	}
	return m_dError;
}

// static
void GHillClimber::test()
{
	GOptimizerBasicTestTargetFunction target;
	GHillClimber opt(&target);
	opt.basicTest(1.39e-17);
}


// --------------------------------------------------------------------------------


GAnnealing::GAnnealing(GTargetFunction* pTargetFunc, GRand* pRand)
: GOptimizer(pTargetFunc), m_initialDeviation(1.0), m_pVector(pTargetFunc->relation()->size()), m_pCandidate(pTargetFunc->relation()->size()), m_pRand(pRand)
{
	if(!pTargetFunc->relation()->areContinuous(0, pTargetFunc->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_dims = pTargetFunc->relation()->size();
	reset();
}

/*virtual*/ GAnnealing::~GAnnealing()
{
}

void GAnnealing::reset()
{
	m_deviation = m_initialDeviation;
	m_pCritic->initVector(m_pVector);
	if(m_pCritic->isStable())
		m_dError = m_pCritic->computeError(m_pVector);
}

/*virtual*/ double GAnnealing::iterate()
{
	for(size_t j = 0; j < 5; j++)
	{
		if(!m_pCritic->isStable())
			m_dError = m_pCritic->computeError(m_pVector);
		for(size_t i = 0; i < m_dims; i++)
			m_pCandidate[i] = m_pVector[i] + m_pRand->normal() * m_deviation;
		double cand = m_pCritic->computeError(m_pCandidate);
		if(cand < m_dError)
		{
			m_pVector.swapContents(m_pCandidate);
			m_dError = cand;
			m_deviation *= 1.5;
		}
		m_deviation *= 0.95;
		if(m_deviation < 1e-14)
			m_deviation = m_initialDeviation;
	}
	return m_dError;
}

// static
void GAnnealing::test()
{
	GRand rand(0);
	GOptimizerBasicTestTargetFunction target;
	GAnnealing opt(&target, &rand);
	opt.basicTest(0.00017);
}

// --------------------------------------------------------------------------------


GRandomDirectionBinarySearch::GRandomDirectionBinarySearch(GTargetFunction* pTargetFunc, GRand* pRand)
: GOptimizer(pTargetFunc), m_stepSize(1e-6), m_pRand(pRand)
{
	if(!pTargetFunc->relation()->areContinuous(0, pTargetFunc->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_dims = pTargetFunc->relation()->size();
	m_current.resize(m_dims);
	m_direction.resize(m_dims);
	m_current.fill(0.0);
	m_err = m_pCritic->computeError(m_current);
}

// virtual
GRandomDirectionBinarySearch::~GRandomDirectionBinarySearch()
{
}

// virtual
double GRandomDirectionBinarySearch::iterate()
{
	m_direction.fillSphericalShell(*m_pRand);
	double sum = 0.0;
	for(size_t i = 0; i < 20; i++)
	{
		m_current.addScaled(m_stepSize, m_direction);
		double pos = m_pCritic->computeError(m_current);
		if(pos < m_err)
		{
			sum += m_stepSize;
			m_stepSize *= 1.189207115; // pow(2.0, 0.25)
			m_err = pos;
		}
		else
		{
			m_current.addScaled(-2.0 * m_stepSize, m_direction);
			double neg = m_pCritic->computeError(m_current);
			if(neg < m_err)
			{
				sum -= m_stepSize;
				m_stepSize *= -1.189207115; // pow(2.0, 0.25)
				m_err = neg;
			}
			else
			{
				m_current.addScaled(m_stepSize, m_direction);
				m_stepSize *= 0.5;
				if(m_stepSize < 1e-16)
					m_stepSize = 1.0; // No progress. Might as well try something new.
			}
		}
	}
	m_stepSize = sum;
	return m_err;
}

// static
void GRandomDirectionBinarySearch::test()
{
	GRand rand(0);
	GOptimizerBasicTestTargetFunction target;
	GRandomDirectionBinarySearch opt(&target, &rand);
	opt.basicTest(1.92e-05);
}



// --------------------------------------------------------------------------------

GEmpiricalGradientDescent::GEmpiricalGradientDescent(GTargetFunction* pCritic, GRand* pRand)
: GOptimizer(pCritic), m_pVector(pCritic->relation()->size()), m_pGradient(pCritic->relation()->size()), m_pDelta(pCritic->relation()->size())
{
	if(!pCritic->relation()->areContinuous(0, pCritic->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_nDimensions = pCritic->relation()->size();
	m_dFeelDistance = 0.03125;
	m_dMomentum = 0.8;
	m_dLearningRate = 0.1;
	m_pRand = pRand;
	reset();
}

/*virtual*/ GEmpiricalGradientDescent::~GEmpiricalGradientDescent()
{
}

void GEmpiricalGradientDescent::reset()
{
	m_pCritic->initVector(m_pVector);
	m_pDelta.fill(0.0);
}

/*virtual*/ double GEmpiricalGradientDescent::iterate()
{
	// Feel the gradient in each dimension using one random pattern
	double dCurrentError = m_pCritic->computeError(m_pVector);
	double d = m_dFeelDistance * m_dLearningRate;
	for(size_t i = 0; i < m_nDimensions; i++)
	{
		m_pVector[i] += d;
		m_pGradient[i] = (m_pCritic->computeError(m_pVector) - dCurrentError) / d;
		m_pVector[i] -= d;
		m_pDelta[i] = m_dMomentum * m_pDelta[i] - m_dLearningRate * m_pGradient[i];
		m_pVector[i] += m_pDelta[i];
	}
	return dCurrentError;
}

// --------------------------------------------------------------------------------

GSampleClimber::GSampleClimber(GTargetFunction* pCritic, GRand* pRand)
: GOptimizer(pCritic), m_pRand(pRand), m_pVector(pCritic->relation()->size()), m_pDir(pCritic->relation()->size()), m_pCand(pCritic->relation()->size()), m_pGradient(pCritic->relation()->size())
{
	if(!pCritic->relation()->areContinuous(0, pCritic->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_dims = pCritic->relation()->size();
	m_dStepSize = 0.1;
	m_alpha = 0.01;
	reset();
}

// virtual
GSampleClimber::~GSampleClimber()
{
}

void GSampleClimber::reset()
{
	m_dStepSize = .1;
	m_pCritic->initVector(m_pVector);
	m_error = m_pCritic->computeError(m_pVector);
}

// virtual
double GSampleClimber::iterate()
{
	// Improve our moving gradient estimate with a new sample
	m_pDir.fillSphericalShell(*m_pRand);
	m_pCand.copy(m_pVector);
	m_pCand.addScaled(m_dStepSize * 0.015625, m_pDir);
	m_pCand += m_pVector;
	double w;
	double err = m_pCritic->computeError(m_pCand);
	for(size_t i = 0; i < m_dims; i++)
	{
		w = m_alpha * m_pDir[i] * m_pDir[i];
		m_pGradient[i] *= (1.0 - w);
		m_pGradient[i] += w * (err - m_error) / m_pDir[i];
	}
	m_pDir.copy(m_pGradient);
	m_pDir.normalize();

	// Step
	m_pVector.addScaled(m_dStepSize, m_pDir);
	err = m_pCritic->computeError(m_pVector);
	if(m_error < err)
	{
		// back up and slow down
		m_pVector.addScaled(-m_dStepSize, m_pDir);
		m_dStepSize *= 0.87;
	}
	else
	{
		m_error = err;
		m_dStepSize *= 1.15;
	}
	return m_error;
}
