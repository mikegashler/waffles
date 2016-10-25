/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include "GReinforcement.h"
#include "GVec.h"
#include "GNeuralNet.h"
#include "GKNN.h"
#include "GRand.h"
#include "GHolders.h"
#include <string.h>

namespace GClasses {

GQLearner::GQLearner(const GRelation& relation, int actionDims, double* pInitialState, GRand* pRand, GAgentActionIterator* pActionIterator)
: GPolicyLearner(relation, actionDims), m_pRand(pRand), m_pActionIterator(pActionIterator)
{
	m_learningRate = 1;
	m_discountFactor = 0.98;
	m_pSenses = new double[m_senseDims + m_actionDims];
	m_pAction = m_pSenses + m_senseDims;
	memcpy(m_pSenses, pInitialState, m_senseDims * sizeof(double));
	m_actionCap = 50;
}

// virtual
GQLearner::~GQLearner()
{
	delete[] m_pSenses;
}

void GQLearner::setLearningRate(double d)
{
	m_learningRate = d;
}

// Sets the factor for discounting future rewards.
void GQLearner::setDiscountFactor(double d)
{
	m_discountFactor = d;
}

// virtual
void GQLearner::refinePolicyAndChooseNextAction(const double* pSenses, double* pOutActions)
{
	double reward;
	if(m_teleported)
		reward = UNKNOWN_REAL_VALUE;
	else
		reward = rewardFromLastAction();
	if(reward != UNKNOWN_REAL_VALUE)
	{
		// Find the best next action
		double maxQ = 0;
		double q;
		m_pActionIterator->reset(pSenses);
		int i;
		for(i = 0; i < m_actionCap; i++)
		{
			if(!m_pActionIterator->nextAction(pOutActions))
				break;
			q = getQValue(pSenses, pOutActions);
			if(q > maxQ)
				maxQ = q;
		}

		// Update the Q-values
		q = reward + m_discountFactor * maxQ;
		setQValue(m_pSenses, m_pAction, (1.0 - m_learningRate) * getQValue(m_pSenses, m_pAction) + m_learningRate * q);
	}

	// Decide what to do next
	memcpy(m_pSenses, pSenses, m_senseDims * sizeof(double));
	chooseAction(pSenses, pOutActions);
	memcpy(m_pAction, pOutActions, m_actionDims * sizeof(double));
	m_teleported = false;
}

// -----------------------------------------------------------------

GIncrementalLearnerQAgent::GIncrementalLearnerQAgent(const GRelation& obsControlRelation, GIncrementalLearner* pQTable, int actionDims, double* pInitialState, GRand* pRand, GAgentActionIterator* pActionIterator, double softMaxThresh)
: GQLearner(obsControlRelation, actionDims, pInitialState, pRand, pActionIterator), m_buf(m_senseDims + m_actionDims)
{
	// Enable incremental learning
	m_pQTable = pQTable;
	GUniformRelation qRel(1);
	pQTable->beginIncrementalLearning(obsControlRelation, qRel);

	// Init other stuff
	m_softMaxThresh = softMaxThresh;
	m_pActionIterator = pActionIterator;
	pActionIterator->reset(pInitialState);
}

// virtual
GIncrementalLearnerQAgent::~GIncrementalLearnerQAgent()
{
}

// virtual
double GIncrementalLearnerQAgent::getQValue(const double* pState, const double* pAction)
{
	memcpy(m_buf.data(), pState, m_senseDims * sizeof(double));
	memcpy(m_buf.data() + m_senseDims, pAction, m_actionDims * sizeof(double));
	GVec out(1);
	m_pQTable->predict(m_buf, out);
	GAssert(out[0] > -1e200);
	return out[0];
}

// virtual
void GIncrementalLearnerQAgent::setQValue(const double* pState, const double* pAction, double qValue)
{
	memcpy(m_buf.data(), pState, m_senseDims * sizeof(double));
	memcpy(m_buf.data() + m_senseDims, pAction, m_actionDims * sizeof(double));
	GVec tmp(1);
	tmp[0] = qValue;
	m_pQTable->trainIncremental(m_buf, tmp);
}

// virtual
void GIncrementalLearnerQAgent::chooseAction(const double* pSenses, double* pActions)
{
	m_pActionIterator->reset(pSenses);
	if(m_explore && m_pRand->uniform() >= m_softMaxThresh)
	{
		// Explore
		m_pActionIterator->randomAction(pActions, m_pRand);
	}
	else
	{
		// Exploit
		double bestQ = -1e200;
		double q;
		int i;
		GTEMPBUF(double, pCand, m_actionDims);
		for(i = 1; i < m_actionCap; i++)
		{
			if(!m_pActionIterator->nextAction(pCand))
				break;
			q = getQValue(pSenses, pCand);
			if(q > bestQ)
			{
				bestQ = q;
				memcpy(pActions, pCand, m_actionDims * sizeof(double));
			}
		}
	}
}

} // namespace GClasses

