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
: GOptimizer(pCritic), m_pPositions(m_nPopulation, pCritic->relation()->size()), m_pVelocities(m_nPopulation, pCritic->relation()->size()), m_pBests(m_nPopulation, pCritic->relation()->size()), m_pErrors(m_nPopulation), m_pRand(pRand)
{
	if(!pCritic->relation()->areContinuous(0, pCritic->relation()->size()))
		throw Ex("Discrete attributes are not supported");
	m_dLearningRate = .2;
	m_nDimensions = pCritic->relation()->size();
	m_nPopulation = nPopulation;
	m_dMin = dMin;
	m_dRange = dRange;
	reset();
}

/*virtual*/ GParticleSwarm::~GParticleSwarm()
{
}

void GParticleSwarm::reset()
{
	for(size_t i = 0; i < m_nPopulation; i++)
	{
		m_pPositions[i].fillUniform(*m_pRand, m_dMin, m_dMin + m_dRange);
		m_pVelocities[i].fillUniform(*m_pRand, m_dMin, m_dMin + m_dRange);
		m_pBests[i].copy(m_pPositions[i]);
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
		dError = m_pCritic->computeError(m_pPositions[i]);
		if(dError < m_pErrors[i])
		{
			m_pErrors[i] = dError;
			m_pBests[i].copy(m_pPositions[i]);
		}
		if(m_pErrors[i] < dGlobalBest)
		{
			dGlobalBest = m_pErrors[i];
			m_nGlobalBest = i;
		}
	}

	// Update velocities
	size_t nPos = 0;
	for(size_t i = 0; i < m_nPopulation; i++)
	{
		for(size_t j = 0; j < m_nDimensions; j++)
			m_pVelocities[i][j] += m_dLearningRate * m_pRand->uniform() * (m_pBests[i][j] - m_pPositions[i][j]) + m_dLearningRate * m_pRand->uniform() * (m_pPositions[m_nGlobalBest][j] - m_pPositions[i][j]);
		nPos += m_nDimensions;
	}

	return dGlobalBest;
}







GBouncyBalls::GBouncyBalls(GTargetFunction* pCritic, size_t population, GRand& rand, double probTeleport, double propSpurt)
: GOptimizer(pCritic),
m_positions(population, pCritic->relation()->size()),
m_velocities(population, pCritic->relation()->size()),
m_errors(population),
m_bestIndex(0),
m_rand(rand),
m_probTeleport(probTeleport),
m_probSpurt(propSpurt)
{
	m_positions.setAll(0.0);
	for(size_t i = 0; i < population; i++)
		m_velocities[i].fillNormal(m_rand);
	m_errors.fill(1e200);
}

GBouncyBalls::~GBouncyBalls()
{
}

// virtual
double GBouncyBalls::iterate()
{
	for(size_t i = 0; i < m_positions.rows(); i++)
	{
		m_positions[i] += m_velocities[i];
		double newErr = m_pCritic->computeError(m_positions[i]);
		if(newErr > m_errors[i])
		{
			m_positions[i] -= m_velocities[i];
			double mag = std::sqrt(m_velocities[i].squaredMagnitude());
			m_velocities[i].fillSphericalShell(m_rand);
			m_velocities[i] *= (mag * 0.8);
			if(m_rand.uniform() < m_probTeleport)
				m_positions[i].copy(m_positions[m_bestIndex]);
		}
		else
		{
			m_velocities[i] *= 1.2;
			m_errors[i] = newErr;
			if(m_errors[i] < m_errors[m_bestIndex])
				m_bestIndex = i;
			if(m_rand.uniform() < m_probSpurt)
				m_velocities[i] *= 10;
		}
	}
	return m_errors[m_bestIndex];
}




} // namespace GClasses

