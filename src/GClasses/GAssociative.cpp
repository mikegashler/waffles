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

#include "GAssociative.h"
#include "GRand.h"
#include "GTree.h"
#include "GActivation.h"

using std::vector;

namespace GClasses {

bool GAssociativeNodeComparer::operator ()(size_t a, size_t b, size_t col) const
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

void GAssociativeNodeComparer::print(std::ostream& stream, size_t row) const
{
	size_t layer = row / MAX_NODES_PER_LAYER;
	size_t unit = row % MAX_NODES_PER_LAYER;
	double delta = m_pThat->m_layers[layer]->delta()[unit];
	stream << "Lay:" << to_str(layer) << ", Unit:" << to_str(unit) << ", Delta:" << to_str(delta);
}



GAssociativeLayer::GAssociativeLayer(size_t bef, size_t cur, size_t aft, GActivationFunction* pActivationFunction)
: m_forw(bef, cur),
m_back(aft, cur),
m_bias(6, cur)
{
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationTanH();
}

GAssociativeLayer::~GAssociativeLayer()
{
	delete(m_pActivationFunction);
}

void GAssociativeLayer::clipWeightMagnitudes(double min, double max)
{
	size_t bef = m_forw.rows();
	size_t cur = m_forw.cols();
	size_t aft = m_back.rows();
	for(size_t i = 0; i < cur; i++)
	{
		double sqmag = 0.0;
		for(size_t j = 0; j < bef; j++)
			sqmag += m_forw[j][i] * m_forw[j][i];
		for(size_t j = 0; j < aft; j++)
			sqmag += m_back[j][i] * m_back[j][i];
		if(sqmag < min * min)
		{
			double scal = min / sqrt(sqmag);
			for(size_t j = 0; j < bef; j++)
				m_forw[j][i] *= scal;
			for(size_t j = 0; j < aft; j++)
				m_back[j][i] *= scal;
		}
		else if(sqmag > max * max)
		{
			double scal = max / sqrt(sqmag);
			for(size_t j = 0; j < bef; j++)
				m_forw[j][i] *= scal;
			for(size_t j = 0; j < aft; j++)
				m_back[j][i] *= scal;
		}
	}
}

void GAssociativeLayer::init(GRand& rand)
{
	size_t bef = m_forw.rows();
	size_t aft = m_back.rows();
	for(size_t i = 0; i < bef; i++)
		m_forw[i].fillNormal(rand);
	for(size_t i = 0; i < aft; i++)
		m_back[i].fillNormal(rand);
	clipWeightMagnitudes(0.8, 0.8);
	m_bias.setAll(0.0);
	clamp().fill(UNKNOWN_REAL_VALUE);
}

void GAssociativeLayer::print(std::ostream& stream, size_t i)
{
	stream << "act: "; activation().print(stream); stream << "\n";
	stream << "bias: "; bias().print(stream); stream << "\n";
	stream << "clamp: "; clamp().print(stream); stream << "\n";
	stream << "delta: "; delta().print(stream); stream << "\n";
	stream << "err: "; error().print(stream); stream << "\n";
	stream << "net: "; net().print(stream); stream << "\n";
	stream << "Weights " << to_str(i - 1) << " -> " << to_str(i) << ": "; m_forw.print(stream);
	stream << "Weights " << to_str(i + 1) << " -> " << to_str(i) << ": "; m_back.print(stream);
}







GAssociative::GAssociative()
: m_comp(this), m_table(m_comp), m_epsilon(1e-6)
{
}

// virtual
GAssociative::~GAssociative()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
}

// virtual
GDomNode* GAssociative::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GAssociative");
	throw Ex("Sorry, not implemented yet");
	return pNode;
}

void GAssociative::addLayer(GAssociativeLayer* pLayer)
{
	m_layers.push_back(pLayer);
}

void GAssociative::print(std::ostream& stream)
{
	stream << "============================================\n";
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		stream << "---Layer " << to_str(i) << "\n";
		m_layers[i]->print(stream, i);
	}
	stream << "---Priority queue:\n";
	m_table.print(stream, 0);
	stream << "============================================\n";
}

// virtual
void GAssociative::trainInner(const GMatrix& features, const GMatrix& labels)
{
	beginIncrementalLearningInner(features.relation(), labels.relation());
	GRandomIndexIterator ii(features.rows(), m_rand);
	for(size_t i = 0; i < 10; i++)
	{
		ii.reset();
		size_t index;
		while(ii.next(index))
			trainIncremental(features[index], labels[index]);
	}
}

// virtual
void GAssociative::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	if(featureRel.size() != m_layers[0]->units())
		throw Ex("mismatching number of input units");
	if(labelRel.size() != m_layers[m_layers.size() - 1]->units())
		throw Ex("mismatching number of output units");
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->init(m_rand);
}

void GAssociative::updateDelta(GAssociativeLayer* pLayer, size_t layer, size_t unit, double delta)
{
	GAssert(std::abs(delta) < 100.0);
	size_t nodeId = layer * MAX_NODES_PER_LAYER + unit;
	GRelationalRow<size_t>* pMatch = m_table.find(nodeId, 1);
	if(pMatch)
		m_table.remove(pMatch);
	pLayer->delta()[unit] += delta;
	m_table.insert(nodeId);
}

void GAssociative::updateActivation(GAssociativeLayer* pLay, size_t layer, size_t unit, double netDelta)
{
	if(netDelta == 0.0)
		return;
	pLay->net()[unit] = std::max(-5.0, std::min(5.0, pLay->net()[unit] + netDelta));
	double oldActivation = pLay->activation()[unit];
	double newActivation = pLay->m_pActivationFunction->squash(pLay->net()[unit], unit);
	pLay->activation()[unit] = newActivation;
	if(pLay->clamp()[unit] != UNKNOWN_REAL_VALUE)
		return;
	updateDelta(pLay, layer, unit, newActivation - oldActivation);
}

void GAssociative::propagateActivation()
{
	// Push for a while
	double sqdelta = 1.0;
	size_t safety = 0;
	while(sqdelta > m_epsilon)
	{
		if(m_table.size() == 0)
			break;
		if(++safety >= 1000)
			break;

		// Pop from the priority queue
		GRelationalRow<size_t>* pFirst = m_table.get(0, 0);
		size_t nodeId = pFirst->row;
		size_t layer = nodeId / MAX_NODES_PER_LAYER;
		size_t unit = nodeId % MAX_NODES_PER_LAYER;
		GAssociativeLayer* pLayerSrc = m_layers[layer];
		double delta = pLayerSrc->delta()[unit];

		// Push to neighbors
		if(layer > 0)
		{
			GAssociativeLayer* pLayDest = m_layers[layer - 1];
			for(size_t i = 0; i < pLayDest->units(); i++)
				updateActivation(pLayDest, layer - 1, i, pLayDest->m_back[unit][i] * delta);
		}
		if(layer < m_layers.size() - 1)
		{
			GAssociativeLayer* pLayDest = m_layers[layer + 1];
			for(size_t i = 0; i < pLayDest->units(); i++)
				updateActivation(pLayDest, layer + 1, i, pLayDest->m_forw[unit][i] * delta);
		}
		pLayerSrc->delta()[unit] = 0.0;
		m_table.remove(pFirst);
		sqdelta = delta * delta;
	}
	m_table.clear();
}

void GAssociative::updateBlame(GAssociativeLayer* pLay, size_t layer, size_t unit, double delta)
{
	delta *= pLay->m_pActivationFunction->derivativeOfNet(pLay->net()[unit], pLay->activation()[unit], unit);
	if(delta == 0.0)
		return;
	double oldBlame = pLay->error()[unit];
	double newBlame = oldBlame + delta;
	pLay->error()[unit] = newBlame;
	updateDelta(pLay, layer, unit, newBlame - oldBlame);
}

double GAssociative::seedBlame()
{
	double sse = 0.0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GAssociativeLayer* pLay = m_layers[i];
		GVec& pAct = pLay->activation();
		GVec& pClamp = pLay->clamp();
		GVec& pDelta = pLay->delta();
		GVec& pErr = pLay->error();
		for(size_t j = 0; j < pLay->units(); j++)
		{
			pErr[j] = 0.0;
			pDelta[j] = 0.0;
			if(pClamp[j] != UNKNOWN_REAL_VALUE)
			{
				double e = pClamp[j] - pAct[j];
				updateBlame(pLay, i, j, e);
				sse += (e * e);
			}
		}
	}
	return sse;
}

void GAssociative::propagateBlame()
{
	double sqdelta = 1.0;
	size_t safety = 0;
	while(sqdelta > m_epsilon)
	{
		if(m_table.size() == 0)
			break;
		if(++safety >= 1000)
			break;

		GRelationalRow<size_t>* pFirst = m_table.get(0, 0);
		size_t nodeId = pFirst->row;
		size_t layer = nodeId / MAX_NODES_PER_LAYER;
		size_t unit = nodeId % MAX_NODES_PER_LAYER;
		GAssociativeLayer* pLaySrc = m_layers[layer];
		double delta = pLaySrc->delta()[unit];
		if(layer > 0)
		{
			GAssociativeLayer* pLayDest = m_layers[layer - 1];
			for(size_t i = 0; i < pLayDest->units(); i++)
			{
				if(pLayDest->clamp()[i] == UNKNOWN_REAL_VALUE)
					updateBlame(pLayDest, layer - 1, i, pLaySrc->m_forw[i][unit] * delta);
			}
		}
		if(layer < m_layers.size() - 1)
		{
			GAssociativeLayer* pLayDest = m_layers[layer + 1];
			for(size_t i = 0; i < pLayDest->units(); i++)
			{
				if(pLayDest->clamp()[i] == UNKNOWN_REAL_VALUE)
					updateBlame(pLayDest, layer + 1, i, pLaySrc->m_back[i][unit] * delta);
			}
		}
		pLaySrc->delta()[unit] = 0.0;
		m_table.remove(pFirst);
		sqdelta = delta * delta;
	}
	m_table.clear();
}

void GAssociative::activatePull(GAssociativeLayer* pLay, size_t lay, size_t unit)
{
	double oldnet = pLay->net()[unit];
	double net = pLay->bias()[unit];
	if(lay > 0)
	{
		GAssociativeLayer* pLaySrc = m_layers[lay - 1];
		GVec& pAct = pLaySrc->activation();
		GVec& pClamp = pLaySrc->clamp();
		for(size_t i = 0; i < pLaySrc->units(); i++)
			net += (pClamp[i] == UNKNOWN_REAL_VALUE ? pAct[i] : pClamp[i]) * pLay->m_forw[i][unit];
	}
	if(lay < m_layers.size() - 1)
	{
		GAssociativeLayer* pLaySrc = m_layers[lay + 1];
		GVec& pAct = pLaySrc->activation();
		GVec& pClamp = pLaySrc->clamp();
		for(size_t i = 0; i < pLaySrc->units(); i++)
			net += (pClamp[i] == UNKNOWN_REAL_VALUE ? pAct[i] : pClamp[i]) * pLay->m_back[i][unit];
	}
	updateActivation(pLay, lay, unit, net - oldnet);
}

void GAssociative::updateWeights(double learningRate)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GAssociativeLayer* pLay = m_layers[i];
		GVec& pErr = pLay->error();
		GVec& pDelta = pLay->delta();
		for(size_t j = 0; j < pLay->units(); j++)
		{
			if(pErr[j] != 0.0)
			{
				pLay->bias()[j] += learningRate * pErr[j];
				if(i > 0)
				{
					GAssociativeLayer* pLayIn = m_layers[i - 1];
					for(size_t k = 0; k < pLayIn->units(); k++)
					{
						double act = pLayIn->clamp()[k];
						if(act == UNKNOWN_REAL_VALUE)
							act = pLayIn->activation()[k];
						pLay->m_forw[k][j] += learningRate * pErr[j] * act;
					}
				}
				if(i < m_layers.size() - 1)
				{
					GAssociativeLayer* pLayIn = m_layers[i + 1];
					for(size_t k = 0; k < pLayIn->units(); k++)
					{
						double act = pLayIn->clamp()[k];
						if(act == UNKNOWN_REAL_VALUE)
							act = pLayIn->activation()[k];
						pLay->m_back[k][j] += learningRate * pErr[j] * act;
					}
				}
			}
			pErr[j] = 0.0;
			pDelta[j] = 0.0;
			activatePull(pLay, i, j);
		}
	}
}

void GAssociative::clamp(GAssociativeLayer* pLayer, size_t layer, size_t unit, double value)
{
	double oldValue = pLayer->clamp()[unit];
	if(oldValue == UNKNOWN_REAL_VALUE)
		oldValue = pLayer->activation()[unit];
	pLayer->clamp()[unit] = value;
	if(value == UNKNOWN_REAL_VALUE)
		value = pLayer->activation()[unit];
	updateDelta(pLayer, layer, unit, value - oldValue);
}

void GAssociative::clampValues(const GVec& pIn, const GVec& pOut)
{
	// Features
	GAssociativeLayer* pLayIn = m_layers[0];
	for(size_t i = 0; i < pLayIn->units(); i++)
		clamp(pLayIn, 0, i, pIn[i]);

	// Labels
	size_t lastLayer = m_layers.size() - 1;
	GAssociativeLayer* pLayOut = m_layers[lastLayer];
	for(size_t i = 0; i < pLayOut->units(); i++)
		clamp(pLayOut, lastLayer, i, pOut[i]);
}

// virtual
void GAssociative::trainIncremental(const GVec& pIn, const GVec& pOut)
{
	double learningRate = 0.02;
	clampValues(pIn, pOut);
	for(size_t i = 0; i < 10; i++)
	{
		propagateActivation();
		seedBlame();
		propagateBlame();
		updateWeights(learningRate);
	}
}

// virtual
void GAssociative::predict(const GVec& pIn, GVec& pOut)
{
	// Clamp the input values
	GAssociativeLayer* pLayIn = m_layers[0];
	for(size_t i = 0; i < pLayIn->units(); i++)
		clamp(pLayIn, 0, i, pIn[i]);
	size_t lastLayer = m_layers.size() - 1;
	GAssociativeLayer* pLayOut = m_layers[lastLayer];
	for(size_t i = 0; i < pLayOut->units(); i++)
		clamp(pLayOut, lastLayer, i, UNKNOWN_REAL_VALUE);

	// Activate the neurons
	propagateActivation();

	// Retrieve the activations of the output units
	pOut.copy(pLayOut->activation());
}

// virtual
void GAssociative::predictDistribution(const GVec& pIn, GPrediction* pOut)
{
	throw new Ex("Sorry, not implemented yet");
}

// virtual
void GAssociative::trainSparse(GSparseMatrix& features, GMatrix& labels)
{
	throw new Ex("Sorry, not implemented yet");
}

// virtual
void GAssociative::clear()
{
}


#ifndef MIN_PREDICT
// static
void GAssociative::test()
{
	GAssociative model;
	model.addLayer(new GAssociativeLayer(0, 2, 10));
	model.addLayer(new GAssociativeLayer(2, 10, 4));
	model.addLayer(new GAssociativeLayer(10, 4, 0));
	GUniformRelation relTwo(2);
	GUniformRelation relFour(4);
	model.beginIncrementalLearning(relTwo, relFour);
	GVec in(2);
	GVec out(4);
	for(size_t i = 0; i < 5000; i++)
	{
		in[0] = model.rand().uniform() - 0.5;
		in[1] = model.rand().uniform() - 0.5;
		out[0] = in[0] * in[1];
		out[1] = in[0] * in[0];
		out[2] = 0.5 * (in[0] + in[1]);
		out[3] = 0.5 * (in[0] * in[0] - in[1] * in[1]);
		model.trainIncremental(in, out);
	}
	GVec pred(4);
	double sumDist = 0.0;
	for(size_t i = 0; i < 100; i++)
	{
		in[0] = model.rand().uniform() - 0.5;
		in[1] = model.rand().uniform() - 0.5;
		out[0] = in[0] * in[1];
		out[1] = in[0] * in[0];
		out[2] = 0.5 * (in[0] + in[1]);
		out[3] = 0.5 * (in[0] * in[0] - in[1] * in[1]);
		model.predict(in, pred);
		double dist = sqrt(out.squaredDistance(pred));
		sumDist += dist;
	}
	sumDist *= 0.01;
	if(sumDist > 0.15)
		throw Ex("failed");
}
#endif // MIN_PREDICT

} // namespace GClasses

