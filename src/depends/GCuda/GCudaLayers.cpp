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

#include "GCudaLayers.h"
#include "../../GClasses/GNeuralNet.h"
#include "../../GClasses/GVec.h"

namespace GClasses {

GCudaLayer::GCudaLayer(GCudaEngine& engine)
: GNeuralNetLayer(), m_engine(engine)
{
}

GCudaLayer::GCudaLayer(GDomNode* pNode, GCudaEngine& engine)
: GNeuralNetLayer(), m_engine(engine)
{
	throw Ex("First deserialize to a GLayerClassic, then call GLayerClassicCuda::upload");
}












GLayerClassicCuda::GLayerClassicCuda(GCudaEngine& engine, size_t inputs, size_t outputs)
: GCudaLayer(engine)
{
	resize(inputs, outputs, NULL);
}

GLayerClassicCuda::~GLayerClassicCuda()
{
}

GDomNode* GLayerClassicCuda::serialize(GDom* pDoc)
{
	GLayerClassic tmp(inputs(), outputs());
	download(tmp);
	return tmp.serialize(pDoc);
}

void GLayerClassicCuda::resize(size_t inputCount, size_t outputCount, GRand* pRand, double deviation)
{
	if(inputCount == inputs() && outputCount == outputs())
		return;
	if(pRand)
		throw Ex("Sorry, GLayerClassicCuda does not support preserving resizes");

	m_weights.resize(inputCount, outputCount);
	m_delta.resize(inputCount, outputCount);
	m_bias.resize(outputCount);
	m_biasDelta.resize(outputCount);
	m_activation.resize(outputCount);
	m_error.resize(outputCount);
	m_outgoing.resize(0);
}

// virtual
void GLayerClassicCuda::resetWeights(GRand& rand)
{
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	GMatrix mTmp(inputCount, outputCount);
	double mag = 1.0 / inputCount;
	for(size_t i = 0; i < inputCount; i++)
	{
		GVec& pW = mTmp[i];
		for(size_t j = 0; j < outputCount; j++)
			pW[j] = rand.normal() * mag;
	}
	m_weights.upload(mTmp);
	GVec vTmp(outputCount);
        vTmp.fillNormal(rand, mag);
	m_bias.upload(vTmp);
	m_delta.scale(m_engine, 0.0);
	m_biasDelta.scale(m_engine, 0.0);
}

// virtual
void GLayerClassicCuda::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	// Perturb weights
	GMatrix m;
	m_weights.download(m);
	size_t n = std::min(outputs() - start, count);
	for(size_t j = 0; j < m_weights.rows(); j++)
		GVec::perturb(m[j].data() + start, deviation, n, rand);
	m_weights.upload(m);

	// Perturb biases
	GVec v(outputs());
	m_bias.download(v);
	GVec::perturb(v.data() + start, deviation, n, rand);
	m_bias.upload(v);
}

// virtual
GVec& GLayerClassicCuda::activation()
{
	m_activation.download(m_outgoing);
	return m_outgoing;
}

// virtual
GVec& GLayerClassicCuda::error()
{
	m_error.download(m_outgoing);
	return m_outgoing;
}

// virtual
void GLayerClassicCuda::copyBiasToNet()
{
	m_activation.copy(m_engine, m_bias);
	m_engine.sync();
}

// virtual
void GLayerClassicCuda::feedIn(const GVec& in)
{
	m_incoming.upload(in);
	m_weights.feedIn(m_engine, m_incoming, m_activation, 0);
	m_engine.sync();
}

// virtual
void GLayerClassicCuda::feedIn(GNeuralNetLayer* pUpStreamLayer)
{
	if(pUpStreamLayer->usesGPU())
	{
		m_weights.feedIn(m_engine, ((GCudaLayer*)pUpStreamLayer)->deviceActivation(), m_activation, 0);
		m_engine.sync();
	}
	else
		feedIn(pUpStreamLayer->activation());
}

// virtual
void GLayerClassicCuda::activate()
{
	m_activation.activateTanh(m_engine);
	m_engine.sync();
}

// virtual
void GLayerClassicCuda::feedForward(const GVec& in)
{
	copyBiasToNet();
	m_incoming.upload(in);
	m_weights.feedIn(m_engine, m_incoming, m_activation, 0);
	m_engine.sync();
	activate();
}

// virtual
void GLayerClassicCuda::feedForward(GNeuralNetLayer* pUpStreamLayer)
{
	copyBiasToNet();
	feedIn(pUpStreamLayer);
	activate();
}

// virtual
void GLayerClassicCuda::dropOut(GRand& rand, double probOfDrop)
{
	throw Ex("sorry, not implemented yet");
}

void GLayerClassicCuda::computeError(const GVec& target)
{
	m_error.upload(target);
	m_error.add(m_engine, m_activation, -1.0);
	m_engine.sync();
}

void GLayerClassicCuda::deactivateError()
{
	m_error.deactivateTanh(m_engine, m_activation);
	m_engine.sync();
}

void GLayerClassicCuda::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	if(pUpStreamLayer->usesGPU())
	{
		m_weights.backPropError(m_engine, m_error, ((GCudaLayer*)pUpStreamLayer)->deviceError(), 0);
		m_engine.sync();
	}
	else
	{
		if(m_incoming.size() != inputs())
			m_incoming.resize(inputs());
		m_weights.backPropError(m_engine, m_error, m_incoming, 0);
		m_engine.sync();
		m_incoming.download(pUpStreamLayer->error());
	}
}

// virtual
void GLayerClassicCuda::updateBias(double learningRate, double momentum)
{
	m_bias.add(m_engine, m_error, learningRate);
	m_engine.sync();
}

// virtual
void GLayerClassicCuda::updateDeltas(const GVec& upStreamActivation, double momentum)
{
	// Assume that the input was already uploaded into m_incoming when feedForward was called
	if(momentum != 1.0)
	{
		m_delta.scale(m_engine, momentum);
		m_engine.sync();
	}
	m_delta.addOuterProduct(m_engine, m_incoming, m_error, 1.0);
 	m_biasDelta.add(m_engine, m_error, 1.0);
	m_engine.sync();
}

// virtual
void GLayerClassicCuda::updateDeltas(GNeuralNetLayer* pUpStreamLayer, double momentum)
{
	if(momentum != 1.0)
	{
		m_delta.scale(m_engine, momentum);
		m_engine.sync();
	}
	if(pUpStreamLayer->usesGPU())
	{
		m_delta.addOuterProduct(m_engine, ((GCudaLayer*)pUpStreamLayer)->deviceActivation(), m_error, 1.0);
	}
	else
	{
		// Assume that the input was already uploaded into m_incoming when feedForward was called
		m_delta.addOuterProduct(m_engine, m_incoming, m_error, 1.0);
	}
	m_biasDelta.add(m_engine, m_error, 1.0);
	m_engine.sync();
}

void GLayerClassicCuda::applyDeltas(double learningRate)
{
	m_weights.add(m_engine, m_delta, learningRate);
	m_engine.sync();
}

// virtual
void GLayerClassicCuda::applyAdaptive()
{
	throw Ex("Sorry, GLayerClassicCuda::applyAdaptive is not yet implemented");
}

// virtual
void GLayerClassicCuda::updateWeightsAndRestoreDroppedOnes(const GVec& upStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	throw Ex("Sorry, not implemented yet");
}

// virtual
void GLayerClassicCuda::updateWeightsAndRestoreDroppedOnes(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double momentum)
{
	throw Ex("Sorry, not implemented yet");
}

void GLayerClassicCuda::scaleWeights(double factor, bool scaleBiases)
{
	m_weights.scale(m_engine, factor);
	if(scaleBiases)
		m_bias.scale(m_engine, factor);
	m_engine.sync();
}

void GLayerClassicCuda::diminishWeights(double amount, bool diminishBiases)
{
	throw Ex("Sorry, GLayerClassicCuda::diminishWeights is not yet implemented");
}

// virtual
void GLayerClassicCuda::maxNorm(double min, double max)
{
	throw Ex("Sorry, GLayerClassicCuda::maxNorm is not yet implemented");
}

// virtual
size_t GLayerClassicCuda::countWeights()
{
	throw Ex("Sorry, GLayerClassicCuda::countWeights is not yet implemented");
	//return 0;
}

// virtual
size_t GLayerClassicCuda::weightsToVector(double* outVector)
{
	throw Ex("Sorry, GLayerClassicCuda::weightsToVector is not yet implemented");
	//return 0;
}

// virtual
size_t GLayerClassicCuda::vectorToWeights(const double* vector)
{
	throw Ex("Sorry, GLayerClassicCuda::vectorToWeights is not yet implemented");
	//return 0;
}

// virtual
void GLayerClassicCuda::copyWeights(const GNeuralNetLayer* pSource)
{
	throw Ex("Sorry, GLayerClassicCuda::copyWeights is not yet implemented");
}

// virtual
void GLayerClassicCuda::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	throw Ex("Sorry, GLayerClassicCuda::renormalizeInput is not yet implemented");
}

// virtual
void GLayerClassicCuda::refineActivationFunction(double learningRate)
{
	throw Ex("Sorry, GLayerClassicCuda::refineActivationFunction does not yet support this method");
}

// virtual
void GLayerClassicCuda::regularizeActivationFunction(double lambda)
{
	throw Ex("Sorry, GLayerClassicCuda::regularizeActivationFunction does not yet support this method");
}


void GLayerClassicCuda::upload(const GLayerClassic& source)
{
	m_weights.upload(source.weights());
	m_bias.upload(source.bias());
}

{
	m_weights.download(dest.weights());
	m_bias.download(dest.bias());
}


} // namespace GClasses

