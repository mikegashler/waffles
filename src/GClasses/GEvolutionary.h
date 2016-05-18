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

#ifndef __GEVOLUTIONARY_H__
#define __GEVOLUTIONARY_H__

#include "GOptimizer.h"
#include <vector>

namespace GClasses {

class GRand;
class GEvolutionaryOptimizerNode;
class GDiscreteEvolutionaryOptimizerNode;

/// Uses an evolutionary process to optimize a vector.
class GEvolutionaryOptimizer : public GOptimizer
{
protected:
	double m_tournamentProbability;
	GRand* m_pRand;
	std::vector<GEvolutionaryOptimizerNode*> m_population;
	double m_bestErr;
	size_t m_bestIndex;

public:
	/// moreFitSurvivalRate is the probability that the more fit member (in a tournament selection) survives
	GEvolutionaryOptimizer(GTargetFunction* pCritic, size_t population, GRand* pRand, double moreFitSurvivalRate);
	virtual ~GEvolutionaryOptimizer();

	/// Returns the best vector found in recent iterations.
	virtual const GVec& currentVector();

	/// Do a little bit more optimization. (This method is typically called in a loop
	/// until satisfactory results are obtained.)
	virtual double iterate();

protected:
	/// Returns the index of the tournament loser (who should typically die and be replaced).
	size_t doTournament();

	void recomputeError(size_t index, GEvolutionaryOptimizerNode* pNode, const GVec& vec);

	GEvolutionaryOptimizerNode* node(size_t index);
};


} // namespace GClasses

#endif // __GEVOLUTIONARY_H__
