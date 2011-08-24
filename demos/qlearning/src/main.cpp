// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GTime.h>
#include <GClasses/GThread.h>
#include <GClasses/GFile.h>
#include <GClasses/GRand.h>
#include <GClasses/GKNN.h>
#include <GClasses/GReinforcement.h>
#include <math.h>
#include <exception>
#include <iostream>

using namespace GClasses;
using std::cerr;

const char* g_szAppPath = NULL;

#define WORLD_SIZE 8

class TestQAgent;

void PrintPolicy(TestQAgent* pAgent);
void PrintQTable(TestQAgent* pAgent);
bool DoesHitWall(double* pState, double* pAction);
void Move(double* pState, double* pAction);


class TestQAgent : public GIncrementalLearnerQAgent
{
protected:
	double m_lastReward;
	double m_soberness;
	double m_curState[3];
	double m_reward;
	double m_penalty;
	bool m_warpRandom;
	int m_journeyCount;
	GIncrementalLearner* m_pLearner;

public:
	TestQAgent(sp_relation& pRelation, double* pInitialState, GRand* prng, GAgentActionIterator* pActionIterator, double softMaxThresh, double soberness, double reward, double penalty, bool warpRandom)
	: GIncrementalLearnerQAgent(pRelation, MakeLearner(prng), 1/*actionDims*/, pInitialState, prng, pActionIterator, softMaxThresh)
	{
		m_lastReward = UNKNOWN_REAL_VALUE;
		m_soberness = soberness;
		m_curState[0] = pInitialState[0];
		m_curState[1] = pInitialState[1];
		m_reward = reward;
		m_penalty = penalty;
		m_warpRandom = warpRandom;
		m_journeyCount = 0;
	}

	virtual ~TestQAgent()
	{
		delete(m_pLearner);
	}

	GIncrementalLearner* MakeLearner(GRand* pRand)
	{
		size_t dims[3];
		dims[0] = WORLD_SIZE; // x
		dims[1] = WORLD_SIZE; // y
		dims[2] = 4; // action
		m_pLearner = new GInstanceTable(3, dims, *pRand);
		return m_pLearner;
	}

	virtual double rewardFromLastAction()
	{
		return m_lastReward;
	}

	virtual void Iterate()
	{
		refinePolicyAndChooseNextAction(m_curState, &m_curState[2]);
		m_lastReward = Act();
	}

	int GetJourneyCount()
	{
		return m_journeyCount;
	}

	double Act()
	{
		// See if we're at the goal
		if(m_curState[0] >= WORLD_SIZE - 1 && m_curState[1] >= WORLD_SIZE - 1)
		{
			if(m_warpRandom)
			{
				m_curState[0] = (double)m_pRand->next(WORLD_SIZE);
				m_curState[1] = (double)m_pRand->next(WORLD_SIZE);
			}
			else
			{
				m_curState[0] = 0;
				m_curState[1] = 0;
			}
			m_journeyCount++;
			if(m_journeyCount % 1000 == 0)
			{
				PrintPolicy(this);
				printf("Journey: %d\n", m_journeyCount);
			}
			return UNKNOWN_REAL_VALUE; // don't update the q-table when we teleport to a new spot
		}

		// Debug spew
		//printf("<%lg,%lg,%lg>\n", pSenses[0], pSenses[1], pAction[0]);
		//PrintQTable(this);
/*
//################## todo: remove me
if(m_curState[2] == 2)
	return 10;
else
	return 0;
*/
		// Randomly mess up the action
		if(m_pRand->uniform() >= m_soberness)
			m_curState[2] = (double)m_pRand->next(4);

		// See if we hit a wall
		if(DoesHitWall(m_curState, m_curState + 2))
			return -m_penalty;
		else
		{
			Move(m_curState, m_curState + 2);
			if(m_curState[0] >= WORLD_SIZE - 1 && m_curState[1] >= WORLD_SIZE - 1)
				return m_reward;
			else
				return 0;
		}
	}
};

void Move(double* pState, double* pAction)
{
	if((int)pAction[0] == 0)
		++pState[0];
	else if((int)pAction[0] == 1)
		++pState[1];
	else if((int)pAction[0] == 2)
		--pState[0];
	else if((int)pAction[0] == 3)
		--pState[1];
	else
		GAssert(false); // unrecognized move
}

// todo: rewrite this method. The world shouldn't be hard-coded like this.
bool DoesHitWall(double* pState, double* pAction)
{
	bool bHitWall = false;
	if((int)pAction[0] == 0)
	{
		++pState[0];
		if(pState[0] > WORLD_SIZE - 1 || ((int)pState[0] == 4 && (int)pState[1] != 0 && (int)pState[1] != 5))
			bHitWall = true;
		pState[0]--;
	}
	else if((int)pAction[0] == 1)
	{
		if((int)pState[1] >= WORLD_SIZE - 1 || ((int)pState[0] == 4 && ((int)pState[1] == 0 || (int)pState[1] == 5)))
			bHitWall = true;
	}
	else if((int)pAction[0] == 2)
	{
		--pState[0];
		if(pState[0] < 0 || ((int)pState[0] == 4 && (int)pState[1] != 0 && (int)pState[1] != 5))
			bHitWall = true;
		pState[0]++;
	}
	else if((int)pAction[0] == 3)
	{
		if((int)pState[1] <= 0 || ((int)pState[0] == 4 && (int)pState[1] == 5))
			bHitWall = true;
	}
	return bHitWall;
}

double DetermineBestMove(TestQAgent* pAgent, double* pState)
{
	double action;
	int i;
	double q;
	double maxQ = -1e200;
	double bestMove = 0;
	for(i = 0; i < 4; i++)
	{
		action = (double)i;
		q = pAgent->getQValue(pState, &action);
		if(q > maxQ)
		{
			maxQ = q;
			bestMove = action;
		}
	}
	return bestMove;
}

void PrintPolicy(TestQAgent* pAgent)
{
	double state[2];
	printf("\n\n\n\n");
	double action;
	int x, y;
	for(y = WORLD_SIZE - 1; y >= 0 ; y--)
	{
		state[1] = y;
		for(x = 0; x < WORLD_SIZE; x++)
		{
			state[0] = x;
			if((int)state[0] == 4 && (int)state[1] != 0 && (int)state[1] != 5)
				printf(" #");
			else
			{
				action = DetermineBestMove(pAgent, state);
				if(action == 0)
					printf(" >");
				else if(action == 1)
					printf(" ^");
				else if(action == 2)
					printf(" <");
				else
					printf(" v");
			}
		}
		printf("\n");
	}
}

void PrintQTable(TestQAgent* pAgent)
{
	double state[2];
	printf("\n\n\n\n");
	double q0, q1, q2, q3, action;
	int x, y;
	for(y = WORLD_SIZE - 1; y >= 0 ; y--)
	{
		state[1] = y;
		for(x = 0; x < WORLD_SIZE; x++)
		{
			state[0] = x;
			action = 0;
			q0 = pAgent->getQValue(state, &action);
			action = 1;
			q1 = pAgent->getQValue(state, &action);
			action = 2;
			q2 = pAgent->getQValue(state, &action);
			action = 3;
			q3 = pAgent->getQValue(state, &action);
			printf("<%lg,%lg,%lg,%lg>", q0, q1, q2, q3);
		}
		printf("\n");
	}
}

void DoTest(GRand* prng, double rcoeff, double alpha, double gamma, double reward, double penalty, double soft_max_thresh, bool warpRandom)
{
	printf("---------------------------------------\n");
	printf("rcoeff=%lg, alpha=%lg, gamma=%lg, reward=%lg, penalty=%lg, softMaxThresh=%lg, warpRandom=%s\n", rcoeff, alpha, gamma, reward, penalty, soft_max_thresh, warpRandom ? "true" : "false");
	printf("---------------------------------------\n");
	fflush(stdout);
	soft_max_thresh /= 100;
	double soberness = ((rcoeff * 4 / 100) - 1) / 3;
	sp_relation rel;
	GMixedRelation* pRel = new GMixedRelation();
	rel = pRel;
	pRel->addAttr(0); // x
	pRel->addAttr(0); // y
	pRel->addAttr(4); // {E,N,W,S}
	GDiscreteActionIterator it(4);
	double initialstate[2];
	initialstate[0] = 0;
	initialstate[1] = 0;
	TestQAgent agent(rel, initialstate, prng, &it, soft_max_thresh, soberness, reward, penalty, warpRandom);
	agent.setLearningRate(alpha);
	agent.setDiscountFactor(gamma);
	while(agent.GetJourneyCount() < 10000)
		agent.Iterate();
}

void DoIt()
{
	GRand prng(0);

	// rcoeff: 100 = always do what you try to do. 0 = when you try to go one way, you always randomly go another.
	// alpha: 1 = totally confident about all observations. 0 = so insecure that it never learns anything.
	// gamma: 0 = think short-run only. Close to 1 = think long-run.
	// reward: the reward obtained for getting to the top-right corner.
	// penalty: the penalty for hitting a wall or a #.
	// soft_max_thresh: 100 = always exploit. 0 = always explore.
	// warpRandom: true = teleport to a random spot after hitting the goal. false = teleport to (0,0) after goal.
	DoTest(&prng, 85/*rcoeff*/, .03/*alpha*/, .99/*gamma*/, 1/*reward*/, .5/*penalty*/, 95/*soft_max_thresh*/, true/*warpRandom*/);
}

int main(int argc, char *argv[])
{
	int nRet = 0;
	try
	{
		DoIt();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

