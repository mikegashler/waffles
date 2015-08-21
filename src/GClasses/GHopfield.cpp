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

#include "GHopfield.h"
#include "GRand.h"
#include "GTree.h"
#include "GActivation.h"

using std::vector;

namespace GClasses {

bool GHopfieldNodeComparer::operator ()(size_t a, size_t b, size_t col) const
{
	size_t layA = a / MAX_NODES_PER_LAYER;
	size_t layB = b / MAX_NODES_PER_LAYER;
	if(col == 0)
	{
		double deltaA = m_pThat->m_layers[layA]->delta()[a % MAX_NODES_PER_LAYER];
		double deltaB = m_pThat->m_layers[layB]->delta()[b % MAX_NODES_PER_LAYER];
		return deltaA * deltaA > deltaB * deltaB;
	}
	else
	{
		if(layA < layB)
			return true;
		else if(layB < layA)
			return false;
		else if(a % MAX_NODES_PER_LAYER < b % MAX_NODES_PER_LAYER)
			return true;
		else
			return false;
	}
}





GHopfieldLayer::GHopfieldLayer(size_t bef, size_t cur, size_t aft, GActivationFunction* pActivationFunction)
: m_forw(bef, cur),
m_back(aft, cur),
m_bias(4, cur)
{
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationTanH();
}

void GHopfieldLayer::init(GRand& rand)
{
	size_t bef = m_forw.rows();
	size_t cur = m_forw.cols();
	size_t aft = m_back.rows();
	double magForw = std::max(0.03, 1.0 / bef);
	double magBack = std::max(0.03, 1.0 / aft);
	for(size_t i = 0; i < bef; i++)
	{
		double* pRow = m_forw[i];
		for(size_t j = 0; j < cur; j++)
			*(pRow++) = rand.normal() * magForw;
	}
	for(size_t i = 0; i < aft; i++)
	{
		double* pRow = m_back[i];
		for(size_t j = 0; j < cur; j++)
			*(pRow++) = rand.normal() * magBack;
	}
	m_bias.setAll(0.0);
}









GHopfield::GHopfield()
: m_comp(this), m_table(m_comp)
{
}

void GHopfield::addLayer(GHopfieldLayer* pLayer)
{
	m_layers.push_back(pLayer);
}

void GHopfield::setActivation(GHopfieldLayer* pLayer, size_t layer, size_t index, double newActivation)
{
	double* pAct = pLayer->activation();
	double oldActivation = pAct[index];
	pAct[index] = newActivation;
	size_t nodeId = layer * MAX_NODES_PER_LAYER + index;
	GRelationalRow<size_t>* pMatch = m_table.find(nodeId, 1, NULL);
	if(pMatch)
		m_table.remove(pMatch);
	pLayer->delta()[index] += (newActivation - oldActivation);
	m_table.insert(nodeId);
}

double GHopfield::relaxPush()
{
	GRelationalRow<size_t>* pFirst = m_table.get(0, 0);
	size_t nodeId = pFirst->row;
	size_t layer = nodeId / MAX_NODES_PER_LAYER;
	size_t unit = nodeId % MAX_NODES_PER_LAYER;
	GHopfieldLayer* pLayerSrc = m_layers[layer];
	double delta = pLayerSrc->delta()[unit];
	if(layer > 0)
	{
		GHopfieldLayer* pLay = m_layers[layer - 1];
		for(size_t i = 0; i < pLay->units(); i++)
		{
			double w = pLay->m_back[unit][i];
			pLay->net()[i] += w * delta;
			double newActivation = pLay->m_pActivationFunction->squash(pLay->net()[i], i);
			setActivation(pLay, layer - 1, i, newActivation);
		}
	}
	if(layer < m_layers.size() - 1)
	{
		GHopfieldLayer* pLay = m_layers[layer + 1];
		for(size_t i = 0; i < pLay->units(); i++)
		{
			double w = pLay->m_forw[unit][i];
			pLay->net()[i] += w * delta;
			double newActivation = pLay->m_pActivationFunction->squash(pLay->net()[i], i);
			setActivation(pLay, layer + 1, i, newActivation);
		}
	}
	m_table.remove(pFirst);
	return delta;
}

void GHopfield::relaxPull(size_t lay)
{
	GHopfieldLayer* pLayDest = m_layers[lay];
	double* pNet = pLayDest->net();
	GVec::copy(pNet, pLayDest->bias(), pLayDest->units());
	if(lay > 0)
	{
		GHopfieldLayer* pLaySrc = m_layers[lay - 1];
		double* pAct = pLaySrc->activation();
		for(size_t i = 0; i < pLaySrc->units(); i++)
			GVec::addScaled(pNet, pAct[i], pLayDest->m_back[i], pLayDest->units());
	}
	if(lay < m_layers.size() - 1)
	{
		GHopfieldLayer* pLaySrc = m_layers[lay + 1];
		double* pAct = pLaySrc->activation();
		for(size_t i = 0; i < pLaySrc->units(); i++)
			GVec::addScaled(pNet, pAct[i], pLayDest->m_forw[i], pLayDest->units());
	}
	double* pAct = pLayDest->activation();
	for(size_t i = 0; i < pLayDest->units(); i++)
		*(pAct++) = pLayDest->m_pActivationFunction->squash(*(pNet++), i);
}

void GHopfield::relax()
{
	// Push for a while
	double delta = 1000;
	size_t safety = 0;
	while(delta > 1e-3)
	{
		if(m_table.size() == 0)
			break;
		if(++safety >= 10000)
		{
			throw Ex("not converging?");
			break;
		}
		delta = relaxPush();
	}

	// Pull for a while
	for(size_t j = 0; j < 4; j++)
	{
		for(size_t i = 0; i < m_layers.size() - 1; i++)
			relaxPull(i);
		for(size_t i = m_layers.size() - 1; i > 0; i--)
			relaxPull(i);
	}
}

// virtual
void GHopfield::trainIncremental(const double* pIn, const double* pOut)
{
	
}

} // namespace GClasses

