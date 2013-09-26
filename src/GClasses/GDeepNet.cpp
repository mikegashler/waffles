/*
	Copyright (C) 2013, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GDeepNet.h"
#include "GRand.h"
#include "GVec.h"
#include "GMath.h"
#include "GNeuralNet.h"

using std::vector;

namespace GClasses {

GDeepNetLayer::GDeepNetLayer(size_t hidden, size_t visible, GRand& rand)
: m_hiddenCount(hidden),
m_visibleCount(visible),
m_rand(rand)
{
}

GDeepNetLayer::~GDeepNetLayer()
{
}

void GDeepNetLayer::trainUnsupervised(const GMatrix& observations, size_t epochs, double initialLearningRate, double decay)
{
	if(observations.cols() != m_visibleCount)
		throw Ex("Expected ", to_str(m_visibleCount), " columns. Got ", to_str(observations.cols()));
	size_t* pIndexes = new size_t[observations.rows()];
	ArrayHolder<size_t> hIndexes(pIndexes);
	GIndexVec::makeIndexVec(pIndexes, observations.rows());
	double learningRate = initialLearningRate;
	for(size_t ep = 0; ep < epochs; ep++)
	{
		GIndexVec::shuffle(pIndexes, observations.rows(), &m_rand);
		size_t* pInd = pIndexes;
		for(size_t i = 0; i < observations.rows(); i++)
			update(observations[*(pInd++)], learningRate);
		learningRate *= decay;
	}
}

GMatrix* GDeepNetLayer::mapToHidden(const GMatrix& observations)
{
	if(observations.cols() != m_visibleCount)
		throw Ex("Expected ", to_str(m_visibleCount), " columns. Got ", to_str(observations.cols()));
	GMatrix* pHidden = new GMatrix(observations.rows(), m_hiddenCount);
	Holder<GMatrix> hHidden(pHidden);
	for(size_t i = 0; i < observations.rows(); i++)
	{
		propToHidden(observations[i]);
		GVec::copy(pHidden->row(i), activationHidden(), m_hiddenCount);
	}
	return hHidden.release();
}









GRestrictedBoltzmannMachine::GRestrictedBoltzmannMachine(size_t hidden, size_t visible, GRand& rand)
: GDeepNetLayer(hidden, visible, rand),
m_w(visible, hidden),
m_delta(visible, hidden)
{
	// Initialize the weights
	for(size_t i = 0; i < visible; i++)
	{
		double* pRow = m_w[i];
		for(size_t j = 0; j < hidden; j++)
		{
			*pRow = 0.01 * m_rand.normal();
			pRow++;
		}
	}

	m_activationVisible = new double[visible + visible + visible + hidden + hidden + hidden];
	m_biasVisible = m_activationVisible + visible;
	m_activationHidden = m_biasVisible + visible;
	m_biasHidden = m_biasVisible + hidden;
	m_blameHidden = m_biasHidden + hidden;
	m_blameVisible = m_blameHidden + hidden;
	GVec::setAll(m_biasVisible, 0.0, visible);
	GVec::setAll(m_biasHidden, 0.0, hidden);
}

GRestrictedBoltzmannMachine::~GRestrictedBoltzmannMachine()
{
	delete[] m_activationVisible;
}

void GRestrictedBoltzmannMachine::propToHidden(const double* pVisible)
{
	m_w.multiply(pVisible, m_activationHidden, true);
	GVec::add(m_activationHidden, m_biasHidden, m_hiddenCount);
	double* pH = m_activationHidden;
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		*pH = GMath::logistic(*pH);
		pH++;
	}
}

void GRestrictedBoltzmannMachine::propToVisible(const double* pHidden)
{
	m_w.multiply(pHidden, m_activationVisible, false);
	GVec::add(m_activationVisible, m_biasVisible, m_visibleCount);
	double* pV = m_activationVisible;
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*pV = GMath::logistic(*pV);
		pV++;
	}
}

void GRestrictedBoltzmannMachine::sampleHidden()
{
	double* pH = m_activationHidden;
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		*pH = m_rand.uniform() < *pH ? 1.0 : 0.0;
		pH++;
	}
}

void GRestrictedBoltzmannMachine::sampleVisible()
{
	double* pV = m_activationVisible;
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*pV = m_rand.uniform() < *pV ? 1.0 : 0.0;
		pV++;
	}
}

void GRestrictedBoltzmannMachine::draw(size_t iters)
{
	double* pH = m_activationHidden;
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		*pH = ((m_rand.next() & 1) == 0 ? 0.0 : 1.0);
		pH++;
	}
	for(size_t i = 0; i < iters; i++)
	{
		propToVisible(m_activationHidden);
		//sampleVisible();
		propToHidden(m_activationVisible);
		sampleHidden();
	}
	propToVisible(m_activationHidden);
}

double GRestrictedBoltzmannMachine::freeEnergy(const double* pVisibleSample)
{
	propToHidden(pVisibleSample);
	double* pBuf = m_delta[0];
	m_w.multiply(m_activationVisible, pBuf, true);
	return -GVec::dotProduct(m_activationHidden, pBuf, m_hiddenCount) -
		GVec::dotProduct(m_biasVisible, m_activationVisible, m_visibleCount) -
		GVec::dotProduct(m_biasHidden, m_activationHidden, m_hiddenCount);
}

void GRestrictedBoltzmannMachine::contrastiveDivergenceUpdate(const double* pVisibleSample, double learningRate, double momentum, size_t gibbsSamples)
{
	// Sample hidden vector
	propToHidden(pVisibleSample);
	//sampleHidden(); // commented out as explained in Section 3.3 at http://axon.cs.byu.edu/~martinez/classes/678/Papers/guideTR.pdf

	// Compute positive gradient
	m_delta.multiply(momentum);
	for(size_t v = 0; v < m_visibleCount; v++)
		GVec::addScaled(m_delta[v], pVisibleSample[v], m_activationHidden, m_hiddenCount);

	// Add positive gradient to the biases
	GVec::addScaled(m_biasVisible, learningRate, pVisibleSample, m_visibleCount);
	GVec::addScaled(m_biasHidden, learningRate, m_activationHidden, m_hiddenCount);

	// Resample
	for(size_t i = 1; i < gibbsSamples; i++)
	{
		propToVisible(m_activationHidden);
		//sampleVisible(); // commented out explained in Section 3.2 at http://axon.cs.byu.edu/~martinez/classes/678/Papers/guideTR.pdf
		propToHidden(m_activationVisible);
		sampleHidden();
	}
	propToVisible(m_activationHidden);
	//sampleVisible(); // commented out explained in Section 3.2 at http://axon.cs.byu.edu/~martinez/classes/678/Papers/guideTR.pdf
	propToHidden(m_activationVisible);
	// Note that we do not sample the hidden for the final iteration, as explained at http://axon.cs.byu.edu/~martinez/classes/678/Papers/guideTR.pdf

	// Compute negative gradient
	for(size_t v = 0; v < m_visibleCount; v++)
		GVec::addScaled(m_delta[v], -m_activationVisible[v], m_activationHidden, m_hiddenCount);

	// Subtract negative gradient from biases
	GVec::addScaled(m_biasVisible, -learningRate, m_activationVisible, m_visibleCount);
	GVec::addScaled(m_biasHidden, -learningRate, m_activationHidden, m_hiddenCount);

	// Update the weights
	for(size_t v = 0; v < m_visibleCount; v++)
		GVec::addScaled(m_w[v], learningRate, m_delta[v], m_hiddenCount);
}

void GRestrictedBoltzmannMachine::maximumLikelihoodUpdate(const double* pVisibleSample, double learningRate, double momentum)
{
	propToHidden(pVisibleSample);
	m_delta.multiply(momentum);
	for(size_t v = 0; v < m_visibleCount; v++)
		GVec::addScaled(m_delta[v], pVisibleSample[v], m_activationHidden, m_hiddenCount);
	for(size_t v = 0; v < m_visibleCount; v++)
		GVec::addScaled(m_w[v], learningRate, m_delta[v], m_hiddenCount);
	GVec::addScaled(m_biasVisible, learningRate, pVisibleSample, m_visibleCount);
	GVec::addScaled(m_biasHidden, learningRate, m_activationHidden, m_hiddenCount);
}

// virtual
void GRestrictedBoltzmannMachine::update(const double* pVisible, double learningRate)
{
	contrastiveDivergenceUpdate(pVisible, learningRate, 0.0, 1);
}

// virtual
void GRestrictedBoltzmannMachine::backpropHelper1(const double* pInputs, double* pInputBlame, double learningRate)
{
	throw Ex("Sorry, not implemented yet");
}

// virtual
void GRestrictedBoltzmannMachine::backpropHelper2(const double* pInputs, double* pInputBlame, double learningRate)
{
	throw Ex("Sorry, not implemented yet");
}







GStackableAutoencoder::GStackableAutoencoder(size_t hidden, size_t visible, GRand& rand)
: GDeepNetLayer(hidden, visible, rand),
m_weightsEncode(hidden, visible),
m_weightsDecode(visible, hidden),
m_biasHidden(3, hidden),
m_biasVisible(3, visible)
{
	// Initialize the weights
	for(size_t i = 0; i < hidden; i++)
	{
		double* pRow = m_weightsEncode[i];
		for(size_t j = 0; j < visible; j++)
		{
			*pRow = 0.01 * m_rand.normal();
			pRow++;
		}
	}
	double* pB = biasVisible();
	for(size_t j = 0; j < visible; j++)
		*(pB++) = 0.01 * m_rand.normal();
	for(size_t i = 0; i < visible; i++)
	{
		double* pRow = m_weightsDecode[i];
		for(size_t j = 0; j < hidden; j++)
		{
			*pRow = 0.01 * m_rand.normal();
			pRow++;
		}
	}
	pB = biasHidden();
	for(size_t j = 0; j < hidden; j++)
		*(pB++) = 0.01 * m_rand.normal();
}

GStackableAutoencoder::~GStackableAutoencoder()
{
}

// virtual
void GStackableAutoencoder::propToHidden(const double* pVisible)
{
	double* pA = activationHidden();
	m_weightsEncode.multiply(pVisible, pA);
	GVec::add(pA, biasHidden(), m_hiddenCount);
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		*pA = tanh(*pA);
		pA++;
	}
}

// virtual
void GStackableAutoencoder::propToVisible(const double* pHidden)
{
	double* pA = activationVisible();
	m_weightsDecode.multiply(pHidden, pA);
	GVec::add(pA, biasVisible(), m_visibleCount);
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*pA = tanh(*pA);
		pA++;
	}
}

// virtual
void GStackableAutoencoder::draw(size_t iters)
{
	double* pH = activationHidden();
	for(size_t i = 0; i < m_hiddenCount; i++)
		*(pH++) = 0.01 * m_rand.normal();
	for(size_t i = 0; i < iters; i++)
	{
		propToVisible(activationHidden());
		propToHidden(activationVisible());
	}
	propToVisible(activationHidden());
}

// virtual
void GStackableAutoencoder::update(const double* pVisible, double learningRate)
{
	// Compute net and activation values
	if(m_noiseDeviation > 0.0)
	{
		double* pAV = activationVisible();
		GVec::copy(pAV, pVisible, m_visibleCount);
		GVec::perturb(pAV, m_noiseDeviation, m_visibleCount, m_rand);
		propToHidden(pAV);
	}
	else
		propToHidden(pVisible);
	propToVisible(activationHidden());

	// Convert visible blame term
	double* pBlame = blameVisible();
	const double* pV = pVisible;
	double* pAV = activationVisible();
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*(pBlame++) = (*pV - *pAV) * (*pAV) * (1.0 - *pAV);
		pAV++;
		pV++;
	}

	// Backpropagate
	backpropHelper1(activationHidden(), blameHidden(), learningRate);
	backpropHelper2(pVisible, blameVisible(), learningRate);
}

// virtual
void GStackableAutoencoder::backpropHelper1(const double* pInputs, double* pInputBlame, double learningRate)
{
	// Backpropagate the blame
	double* pBlame = blameVisible();
	m_weightsDecode.multiply(pBlame, pInputBlame, true);
	const double* pInp = pInputs;
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		*(pInputBlame++) *= (1.0 - (*pInp) * (*pInp));
		pInp++;
	}

	// Update weights
	double* pBias = biasVisible();
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		GVec::addScaled(m_weightsDecode[i], learningRate * (*pBlame), pInputs, m_hiddenCount);
		*(pBias++) += learningRate * (*pBlame);
		pBlame++;
	}
}

// virtual
void GStackableAutoencoder::backpropHelper2(const double* pInputs, double* pInputBlame, double learningRate)
{
	// Backpropagate the blame
	double* pBlame = blameHidden();
	m_weightsEncode.multiply(pBlame, pInputBlame, true);
	const double* pInp = pInputs;
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*(pInputBlame++) *= (1.0 - (*pInp) * (*pInp));
		pInp++;
	}

	// Update weights
	double* pBias = biasHidden();
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		GVec::addScaled(m_weightsEncode[i], learningRate * (*pBlame), pInputs, m_visibleCount);
		*(pBias++) += learningRate * (*pBlame);
		pBlame++;
	}
}







GDeepNet::GDeepNet(GRand& rand)
: m_rand(rand)
{
}

// virtual
GDeepNet::~GDeepNet()
{
	for(vector<GDeepNetLayer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++)
		delete(*it);
}

void GDeepNet::addLayer(GDeepNetLayer* pNewLayer)
{
	if(m_layers.size() > 0)
	{
		GDeepNetLayer* pNextLayer = m_layers[0];
		if(pNewLayer->visibleCount() != pNextLayer->hiddenCount())
			throw Ex("Expected the new layer to have ", to_str(pNextLayer->hiddenCount()), " visible units. Got ", to_str(pNewLayer->visibleCount()));
	}
	m_layers.insert(m_layers.begin(), pNewLayer);
}

double* GDeepNet::draw(size_t iters)
{
	GDeepNetLayer* pLayer = m_layers[0];
	pLayer->draw(iters);
	for(size_t layer = 1; layer < m_layers.size(); layer++)
	{
		GDeepNetLayer* pNextLayer = m_layers[layer];
		pNextLayer->propToVisible(pLayer->activationVisible());
		pLayer = pNextLayer;
	}
	return pLayer->activationVisible();
}

double* GDeepNet::forward(const double* pIntrinsic)
{
	GDeepNetLayer* pLayer = m_layers[0];
	pLayer->propToVisible(pIntrinsic);
	double* pIn = pLayer->activationVisible();
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		pLayer = m_layers[i];
		pLayer->propToVisible(pIn);
		pIn = pLayer->activationVisible();
	}
	return pIn;
}

double* GDeepNet::backward(const double* pObserved)
{
	size_t i = m_layers.size() - 1;
	GDeepNetLayer* pLayer = m_layers[i];
	pLayer->propToHidden(pObserved);
	double* pIn = pLayer->activationHidden();
	for(i--; i < m_layers.size(); i--)
	{
		pLayer = m_layers[i];
		pLayer->propToHidden(pIn);
		pIn = pLayer->activationHidden();
	}
	return pIn;
}

void GDeepNet::trainLayerwise(GMatrix& observations, size_t epochs, double initialLearningRate, double decay)
{
	GMatrix* pObs = &observations;
	Holder<GMatrix> hObs(NULL);
	for(size_t i = m_layers.size() - 1; i < m_layers.size(); i--)
	{
		// Train the layer
		m_layers[i]->trainUnsupervised(*pObs, epochs, initialLearningRate, decay);

		// Map the observations for the next layer
		if(i > 0)
		{
			pObs = m_layers[i]->mapToHidden(*pObs);
			hObs.reset(pObs);
		}
	}
}

void GDeepNet::refineBackprop(const double* pObservation, double learningRate)
{
	// Compute blame term on the visible layer
	forward(backward(pObservation));
	GDeepNetLayer* pVisibleLayer = m_layers[m_layers.size() - 1];
	double* pAct = pVisibleLayer->activationVisible();
	double* pBlame = pVisibleLayer->blameVisible();
	const double* pObs = pObservation;
	for(size_t i = 0; i < pVisibleLayer->visibleCount(); i++)
	{
		*(pBlame++) = (*pObs - *pAct) * (1.0 - (*pAct) * (*pAct));
		pAct++;
		pObs++;
	}

	// Backpropagate blame and update weights
	for(size_t i = m_layers.size() - 1; i > 0; i--)
		m_layers[i]->backpropHelper1(m_layers[i - 1]->activationVisible(), m_layers[i - 1]->blameVisible(), learningRate);
	m_layers[0]->backpropHelper1(m_layers[0]->activationHidden(), m_layers[0]->blameHidden(), learningRate);
	for(size_t i = 0; i + 1 < m_layers.size(); i++)
		m_layers[i]->backpropHelper2(m_layers[i + 1]->activationHidden(), m_layers[i + 1]->blameHidden(), learningRate);
	m_layers[m_layers.size() - 1]->backpropHelper2(pObservation, m_layers[m_layers.size() - 1]->blameVisible(), learningRate);
}

#ifndef MIN_PREDICT
// static
void GDeepNet::test()
{
	// Make a deep net
	GRand rand(0);
	GDeepNet dn(rand);
	dn.addLayer(new GStackableAutoencoder(3, 4, rand));
	dn.addLayer(new GStackableAutoencoder(2, 3, rand));

	// Make a neural net
	GNeuralNet nn(rand);
	nn.setLearningRate(0.8);
	vector<size_t> topo;
	topo.push_back(3);
	topo.push_back(2);
	topo.push_back(3);
	nn.setTopology(topo);
	GUniformRelation ends(4);
	nn.beginIncrementalLearning(ends, ends);

	// Copy the weights from the deep network to the neural net
	Holder<GMatrix> h(NULL);
	h.reset(((GStackableAutoencoder*)dn.m_layers[1])->weightsEncode().transpose());
	nn.getLayer(0)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(0)->bias(), ((GStackableAutoencoder*)dn.m_layers[1])->biasHidden(), 3);

	h.reset(((GStackableAutoencoder*)dn.m_layers[0])->weightsEncode().transpose());
	nn.getLayer(1)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(1)->bias(), ((GStackableAutoencoder*)dn.m_layers[0])->biasHidden(), 2);

	h.reset(((GStackableAutoencoder*)dn.m_layers[0])->weightsDecode().transpose());
	nn.getLayer(2)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(2)->bias(), ((GStackableAutoencoder*)dn.m_layers[0])->biasVisible(), 3);

	h.reset(((GStackableAutoencoder*)dn.m_layers[1])->weightsDecode().transpose());
	nn.getLayer(3)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(3)->bias(), ((GStackableAutoencoder*)dn.m_layers[1])->biasVisible(), 4);

	// Train both of them with a phony observation
	double obs[4];
	rand.cubical(obs, 4);
	nn.trainIncremental(obs, obs);
	dn.refineBackprop(obs, 0.8);

	// Check that the weights were updated identically
	double err = 0.0;
	err += nn.getLayer(0)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.m_layers[1])->weightsEncode(), true);
	err += nn.getLayer(1)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.m_layers[0])->weightsEncode(), true);
	err += nn.getLayer(2)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.m_layers[0])->weightsDecode(), true);
	err += nn.getLayer(3)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.m_layers[1])->weightsDecode(), true);
	err += GVec::squaredDistance(nn.getLayer(0)->bias(), ((GStackableAutoencoder*)dn.m_layers[1])->biasHidden(), 3);
	err += GVec::squaredDistance(nn.getLayer(1)->bias(), ((GStackableAutoencoder*)dn.m_layers[0])->biasHidden(), 2);
	err += GVec::squaredDistance(nn.getLayer(2)->bias(), ((GStackableAutoencoder*)dn.m_layers[0])->biasVisible(), 3);
	err += GVec::squaredDistance(nn.getLayer(3)->bias(), ((GStackableAutoencoder*)dn.m_layers[1])->biasVisible(), 4);
	if(err > 1e-12)
		throw Ex("failed");
}
#endif // MIN_PREDICT


} // namespace GClasses
