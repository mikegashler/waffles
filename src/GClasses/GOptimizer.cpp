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

#include "GOptimizer.h"
#include <string.h>
#include "GVec.h"
#include "GRand.h"
#include <math.h>

namespace GClasses {

using std::vector;

GTargetFunction::GTargetFunction(size_t dims)
{
	m_pRelation = new GUniformRelation(dims, 0);
}

// virtual
GTargetFunction::~GTargetFunction()
{
	delete(m_pRelation);
}

// virtual
void GTargetFunction::initVector(double* pVector)
{
	GVec::setAll(pVector, 0.0, m_pRelation->size());
}

// -------------------------------------------------------

// virtual
double GOptimizerBasicTestTargetFunction::computeError(const double* pVector)
{
	double a = pVector[0] - 0.123456789;
	double b = pVector[1] + 9.876543210;
	double c = pVector[2] - 3.333333333;
	return sqrt(a * a + b * b + c * c);
}

// -------------------------------------------------------


GOptimizer::GOptimizer(GTargetFunction* pCritic)
: m_pCritic(pCritic)
{
}

// virtual
GOptimizer::~GOptimizer()
{
}

double GOptimizer::searchUntil(size_t nBurnInIterations, size_t nIterations, double dImprovement)
{
	for(size_t i = 0; i < nBurnInIterations; i++)
		iterate();
	double dPrevError;
	double dError = iterate();
	while(true)
	{
		dPrevError = dError;
		for(size_t i = 0; i < nIterations; i++)
			iterate();
		dError = iterate();
		if((dPrevError - dError) / dPrevError >= dImprovement && dError > 0.0)
		{
		}
		else
			break;
	}
	return dError;
}

#ifndef MIN_PREDICT
void GOptimizer::basicTest(double minAccuracy, double warnRange)
{
	double d = searchUntil(5, 20, 0.001);
	if(d > minAccuracy)
		throw Ex("Optimizer accuracy has regressed. Expected ", to_str(minAccuracy), ". Got ", to_str(d));
	if(d < minAccuracy - warnRange)
		std::cout << "Accuracy is much better than expected. Expected " << to_str(minAccuracy) << ". Got " << to_str(d) << ". Please tighten the expected accuracy for this test.\n";
}
#endif
// -------------------------------------------------------

GParallelOptimizers::GParallelOptimizers(size_t dims)
{
	if(dims > 0)
		m_pRelation = new GUniformRelation(dims, 0);
}

GParallelOptimizers::~GParallelOptimizers()
{
	for(vector<GOptimizer*>::iterator it = m_optimizers.begin(); it != m_optimizers.end(); it++)
		delete(*it);
	for(vector<GTargetFunction*>::iterator it = m_targetFunctions.begin(); it != m_targetFunctions.end(); it++)
		delete(*it);
}

void GParallelOptimizers::add(GTargetFunction* pTargetFunction, GOptimizer* pOptimizer)
{
	m_targetFunctions.push_back(pTargetFunction);
	m_optimizers.push_back(pOptimizer);
}

double GParallelOptimizers::iterateAll()
{
	double err = 0;
	for(vector<GOptimizer*>::iterator it = m_optimizers.begin(); it != m_optimizers.end(); it++)
		err += (*it)->iterate();
	return err;
}

double GParallelOptimizers::searchUntil(size_t nBurnInIterations, size_t nIterations, double dImprovement)
{
	for(size_t i = 1; i < nBurnInIterations; i++)
		iterateAll();
	double dPrevError;
	double dError = iterateAll();
	while(true)
	{
		dPrevError = dError;
		for(size_t i = 1; i < nIterations; i++)
			iterateAll();
		dError = iterateAll();
		if((dPrevError - dError) / dPrevError < dImprovement)
			break;
	}
	return dError;
}

// -------------------------------------------------------

class GAction
{
protected:
        int m_nAction;
        GAction* m_pPrev;
        unsigned int m_nRefs;

        ~GAction()
        {
                if(m_pPrev)
                        m_pPrev->Release(); // todo: this could overflow the stack with recursion
        }
public:
        GAction(int nAction, GAction* pPrev)
         : m_nAction(nAction), m_nRefs(0)
        {
                m_pPrev = pPrev;
                if(pPrev)
                        pPrev->AddRef();
        }

        void AddRef()
        {
                m_nRefs++;
        }

        void Release()
        {
                if(--m_nRefs == 0)
                        delete(this);
        }

        GAction* GetPrev() { return m_pPrev; }
        int GetAction() { return m_nAction; }
};

// -------------------------------------------------------

GActionPath::GActionPath(GActionPathState* pState) : m_pLastAction(NULL), m_nPathLen(0)
{
        m_pHeadState = pState;
}

GActionPath::~GActionPath()
{
        if(m_pLastAction)
                m_pLastAction->Release();
        delete(m_pHeadState);
}

void GActionPath::doAction(size_t nAction)
{
        GAction* pPrevAction = m_pLastAction;
        m_pLastAction = new GAction((int)nAction, pPrevAction);
        m_pLastAction->AddRef(); // referenced by m_pLastAction
        if(pPrevAction)
                pPrevAction->Release(); // no longer referenced by m_pLastAction
        m_nPathLen++;
        m_pHeadState->performAction(nAction);
}

GActionPath* GActionPath::fork()
{
        GActionPath* pNewPath = new GActionPath(m_pHeadState->copy());
        pNewPath->m_nPathLen = m_nPathLen;
        pNewPath->m_pLastAction = m_pLastAction;
        if(m_pLastAction)
                m_pLastAction->AddRef();
        return pNewPath;
}

void GActionPath::path(size_t nCount, size_t* pOutBuf)
{
        while(nCount > m_nPathLen)
                pOutBuf[--nCount] = -1;
        size_t i = m_nPathLen;
        GAction* pAction = m_pLastAction;
        while(i > nCount)
        {
                i--;
                pAction = pAction->GetPrev();
        }
        while(i > 0)
        {
                pOutBuf[--i] = pAction->GetAction();
                pAction = pAction->GetPrev();
        }
}

double GActionPath::critique()
{
        return m_pHeadState->critiquePath(m_nPathLen, m_pLastAction);
}

} // namespace GClasses

