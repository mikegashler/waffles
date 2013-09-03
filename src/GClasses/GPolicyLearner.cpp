/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GPolicyLearner.h"
#include "GNeuralNet.h"
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
	m_pRelation = GRelation::deserialize(pAgent->field("relation"));
	m_actionDims = (int)pAgent->field("actionDims")->asInt();
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
	pNode->addField(pDoc, "actionDims", pDoc->newInt(m_actionDims));
	pNode->addField(pDoc, "relation", m_pRelation->serialize(pDoc));
	return pNode;
}


} // namespace GClasses

