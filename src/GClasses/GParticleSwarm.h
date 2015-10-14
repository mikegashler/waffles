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

#ifndef __GPARTICLESWARM_H__
#define __GPARTICLESWARM_H__

#include "GOptimizer.h"
#include "GMatrix.h"

namespace GClasses {

class GRand;


/// An optimization algorithm inspired by flocking birds
class GParticleSwarm : public GOptimizer
{
protected:
	double m_dMin, m_dRange;
	double m_dLearningRate;
	size_t m_nDimensions;
	size_t m_nPopulation;
	double* m_pPositions;
	double* m_pVelocities;
	double* m_pBests;
	double* m_pErrors;
	size_t m_nGlobalBest;
	GRand* m_pRand;

public:
	GParticleSwarm(GTargetFunction* pCritic, size_t nPopulation, double dMin, double dRange, GRand* pRand);
	virtual ~GParticleSwarm();

	/// Perform a little more optimization
	virtual double iterate();

	/// Specify the learning rate
	void setLearningRate(double d) { m_dLearningRate = d; }

protected:
	void reset();
};


} // namespace GClasses

#endif // __GPARTICLESWARM_H__
