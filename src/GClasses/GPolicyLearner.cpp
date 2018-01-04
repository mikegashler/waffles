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

#include "GPolicyLearner.h"
#include "GKNN.h"
#include "GDecisionTree.h"
#include "GNeighborFinder.h"
#include "GOptimizer.h"
#include <stdlib.h>
#include "GVec.h"
#include "GRand.h"
//#include "GImage.h"
#include "GHeap.h"
#include "GHillClimber.h"
#include "GDom.h"
#include <deque>
#include <math.h>

namespace GClasses {

using std::deque;

// virtual
void GDiscreteActionIterator::reset(const double* pState)
{
	m_action = 0;
}

// virtual
void GDiscreteActionIterator::randomAction(double* pOutAction, GRand* pRand)
{
	*pOutAction = (double)pRand->next(m_count);
}

// virtual
bool GDiscreteActionIterator::nextAction(double* pOutAction)
{
	if(m_action < m_count)
	{
		*pOutAction = m_action++;
		return true;
	}
	else
		return false;
}

// -----------------------------------------------------------------------------

GPolicyLearner::GPolicyLearner(const GRelation& relation, int actionDims)
: m_pRelation(relation.clone())
{
	m_senseDims = (int)m_pRelation->size() - actionDims;
	if(m_senseDims < 0)
		throw Ex("more action dims than relation dims");
	m_actionDims = actionDims;
	m_teleported = true;
	m_explore = true;
}

GPolicyLearner::GPolicyLearner(GDomNode* pAgent)
{
	m_pRelation = GRelation::deserialize(pAgent->get("relation"));
	m_actionDims = (int)pAgent->getInt("actionDims");
	m_senseDims = (int)m_pRelation->size() - m_actionDims;
	m_teleported = true;
}

// virtual
GPolicyLearner::~GPolicyLearner()
{
	delete(m_pRelation);
}


GDomNode* GPolicyLearner::baseDomNode(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "actionDims", (long long)m_actionDims);
	pNode->add(pDoc, "relation", m_pRelation->serialize(pDoc));
	return pNode;
}


} // namespace GClasses

