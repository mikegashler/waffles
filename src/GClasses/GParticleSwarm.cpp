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

#include "GParticleSwarm.h"
#include <string.h>
#include "GVec.h"
#include "GRand.h"
#include <cmath>

namespace GClasses {

GParticleSwarm::GParticleSwarm(GTargetFunction* pCritic, size_t nPopulation, double dMin, double dRange, GRand* pRand)
: GOptimizer(pCritic), m_pRand(pRand)
{
	if(!pCritic->relation()->areContinuous(0, pCritic->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_dLearningRate = .2;
	m_nDimensions = pCritic->relation()->size();
	m_nPopulation = nPopulation;
	m_pPositions = new double[m_nPopulation * m_nDimensions];
	m_pVelocities = new double[m_nPopulation * m_nDimensions];
	m_pBests = new double[m_nPopulation * m_nDimensions];
	m_pErrors = new double[m_nPopulation];
	m_dMin = dMin;
	m_dRange = dRange;
	reset();
}

/*virtual*/ GParticleSwarm::~GParticleSwarm()
{
	delete(m_pErrors);
	delete(m_pBests);
	delete(m_pVelocities);
	delete(m_pPositions);
}

void GParticleSwarm::reset()
{
	for(size_t i = 0; i < m_nPopulation; i++)
	{
		for(size_t n = 0; n < m_nDimensions; n++)
		{
			m_pPositions[m_nDimensions * i + n] = m_pRand->uniform() * m_dRange + m_dMin;
			m_pVelocities[m_nDimensions * i + n] = m_pRand->uniform() * m_dRange + m_dMin;
			m_pBests[m_nDimensions * i + n] = m_pPositions[m_nDimensions * i + n];
		}
		m_pErrors[i] = 1e100;
	}
	m_nGlobalBest = 0;
}

/*virtual*/ double GParticleSwarm::iterate()
{
	// Advance
	size_t n = m_nPopulation * m_nDimensions;
	for(size_t i = 0; i < n; i++)
		m_pPositions[i] += m_pVelocities[i];

	// Critique the current spots and find the global best
	double dError;
	double dGlobalBest = 1e100;
	for(size_t i = 0; i < m_nPopulation; i++)
	{
		size_t nPos = m_nDimensions * i;
		dError = m_pCritic->computeError(&m_pPositions[nPos]);
		if(dError < m_pErrors[i])
		{
			m_pErrors[i] = dError;
			memcpy(&m_pBests[nPos], &m_pPositions[nPos], sizeof(double) * m_nDimensions);
		}
		if(m_pErrors[i] < dGlobalBest)
		{
			dGlobalBest = m_pErrors[i];
			m_nGlobalBest = i;
		}
	}

	// Update velocities
	size_t nPos = 0;
	n = m_nDimensions * m_nGlobalBest;
	for(size_t i = 0; i < m_nPopulation; i++)
	{
		for(size_t j = 0; j < m_nDimensions; j++)
		{
			m_pVelocities[nPos + j] += m_dLearningRate * m_pRand->uniform() * (m_pBests[nPos + j] - m_pPositions[nPos + j]) + m_dLearningRate * m_pRand->uniform() * (m_pPositions[n + j] - m_pPositions[nPos + j]);
		}
		nPos += m_nDimensions;
	}

	return dGlobalBest;
}


} // namespace GClasses

