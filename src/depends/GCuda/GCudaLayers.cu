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

GCudaLayer::GCudaLayer(GDomNode* pNode)
: GNeuralNetLayer()
{
	throw Ex("Sorry, GCudaLayer does not support serialization");
}

GDomNode* GCudaLayer::serialize(GDom* pDoc)
{
	throw Ex("Sorry, GNeuralNetLayerCuda does not support serialization");
	return NULL;
}











GNeuralNetLayerCuda::GNeuralNetLayerCuda(GCudaEngine engine, size_t inputs, size_t outputs)
: GCudaLayer(engine), m_pOutgoing(NULL)
{
	resize(inputs, outputs, NULL);
}

GNeuralNetLayerCuda::~GNeuralNetLayerCuda()
{
	delete[] m_pOutgoing;
}

void GNeuralNetLayerCuda::resize(size_t inputCount, size_t outputCount, GRand* pRand)
{
	if(inputCount == inputs() && outputCount == outputs())
		return;
	if(pRand)
		throw Ex("Sorry, GNeuralNetLayerCuda does not support preserving resizes");

	m_weights.resize(inputCount, outputCount);
	m_delta.resize(inputCount, outputCount);
	m_bias.resize(6, outputCount);
	m_net.resize(6, outputCount);
	m_activation.resize(6, outputCount);
	m_error.resize(6, outputCount);
	m_biasDelta.resize(6, outputCount);
	delete[] m_pOutgoing;
	m_pOutgoing = NULL;
}

// virtual
void GNeuralNetLayerCuda::resetWeights(GRand& rand)
{
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	GMatrix mTmp(inputCount, outputCount);
	double mag = 1.0 / inputCount;
	for(size_t i = 0; i < inputCount; i++)
	{
		double* pW = mTmp[i];
		for(size_t j = 0; j < inputCount; j++)
			*(pW++) = rand.normal() * mag;
	}
	m_weights.upload(mTmp);
	mTmp.setAll(0.0);
	m_delta.upload(mTmp);
	GVec vTmp(outputCount);
	double* pB = vTmp.v;
	for(size_t i = 0; i < outputCount; i++)
		*(pB++) = rand.normal() * mag;
	m_bias.upload(vTmp.v, outputCount);
	GVec::setAll(vTmp.v, 0.0, outputCount);
	m_biasDelta.upload(vTmp.v, outputCount);
}

// virtual
void GNeuralNetLayerCuda::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	// Perturb weights
	GMatrix m;
	m_weights.download(m);
	size_t n = std::min(outputs() - start, count);
	for(size_t j = 0; j < m_weights.rows(); j++)
		GVec::perturb(m[j] + start, deviation, n, rand);
	m_weights.upload(m);

	// Perturb biases
	GVec v(outputCount());
	m_bias.download(v.v);
	GVec::perturb(v.v + start, deviation, n, rand);
	m_bias.upload(v.v, outputCount());
}

// virtual
double* GNeuralNetLayerCuda::activation()
{
	if(!m_pOutgoing)
		m_pOutgoing = new double[outputCount()];
	m_activation.download(m_pOutgoing, outputCount());
	return m_pOutgoing;
}

// virtual
double* GNeuralNetLayerCuda::error()
{
	if(!m_pOutgoing)
		m_pOutgoing = new double[outputCount()];
	m_error.download(m_pOutgoing, outputCount());
	return m_pOutgoing;
}

// virtual
void GNeuralNetLayerCuda::copyBiasToNet()
{
	m_activation.copy(m_bias);
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

// virtual
void GNeuralNetLayerCuda::feedIn(const double* pIn, size_t inputStart, size_t inputCount)
{
	m_incoming.upload(pIn, inputCount());
	m_weights.feedIn(m_engine, m_incoming, m_activation, inputStart);
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

// virtual
void GNeuralNetLayerCuda::feedIn(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
{
	if(pUpStreamLayer->usesGPU())
	{
		m_weights.feedIn(m_engine, ((GCudaLayer*)pUpStreamLayer)->deviceActivation(), m_activation, inputStart);
		if(cudaDeviceSynchronize() != cudaSuccess)
			throw Ex(cudaGetErrorString(cudaGetLastError()));
	}
	else
		feedIn(pUpStreamLayer->activation(), inputStart, pUpStreamLayer->outputs());
}

// virtual
void GNeuralNetLayerCuda::activate()
{
	m_activation.activateTanh(m_engine);
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

void GNeuralNetLayerCuda::computeError(const double* pTarget)
{
	m_error.upload(pTarget, outputCount());
	m_error.add(m_engine, m_activation, -1.0);
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

void GNeuralNetLayerCuda::deactivateError()
{
	m_error.deactivateError(m_activation);
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

void GNeuralNetLayerCuda::backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
{
	if(pUpStreamLayer->usesGPU())
	{
		m_weights.backPropError(m_error, ((GCudaLayer*)pUpStreamLayer)->deviceError(), inputStart);
		if(cudaDeviceSynchronize() != cudaSuccess)
			throw Ex(cudaGetErrorString(cudaGetLastError()));
	}
	else
	{
		if(m_incoming.size() != inputCount())
			m_incoming.resize(inputCount());
		m_weights.backPropError(m_error, m_incoming, inputStart);
		if(cudaDeviceSynchronize() != cudaSuccess)
			throw Ex(cudaGetErrorString(cudaGetLastError()));
		m_incoming.download(pUpStreamLayer->error(), inputCount());
	}
}

// virtual
void GNeuralNetLayerCuda::adjustWeights(const double* pUpStreamActivation, double learningRate, double momentum)
{
	// Assume that the input was already uploaded into m_incoming when feedForward was called
	m_weights.updateWeights(m_engine, m_incoming, m_error, learningRate);
	m_bias.add(m_engine, m_error, learningRate);
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

// virtual
void GNeuralNetLayerCuda::adjustWeights(GNeuralNetLayer* pUpStreamLayer, double learningRate, double momentum)
{
	if(pUpStreamLayer->usesGPU())
	{
		m_weights.updateWeights(m_engine, ((GCudaLayer*)pUpStreamLayer)->deviceActivation(), m_error, learningRate);
		m_bias.add(m_engine, m_error, learningRate);
	}
	else
	{
		// Assume that the input was already uploaded into m_incoming when feedForward was called
		m_weights.updateWeights(m_engine, m_incoming, m_error, learningRate);
		m_bias.add(m_engine, m_error, learningRate);
	}
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

void GNeuralNetLayerCuda::scaleWeights(double factor)
{
	m_weights.scale(factor);
	m_bias.scale(factor);
	size_t outputCount = outputs();
	if(cudaDeviceSynchronize() != cudaSuccess)
		throw Ex(cudaGetErrorString(cudaGetLastError()));
}

void GNeuralNetLayerCuda::diminishWeights(double amount)
{
	throw Ex("Sorry, GNeuralNetLayerCuda::diminishWeights is not yet implemented");
}

// virtual
void GNeuralNetLayerCuda::clipWeights(double max)
{
	throw Ex("Sorry, GNeuralNetLayerCuda::clipWeights is not yet implemented");
}

// virtual
size_t GNeuralNetLayerCuda::countWeights()
{
	throw Ex("Sorry, GNeuralNetLayerCuda::countWeights is not yet implemented");
	return 0;
}

// virtual
size_t GNeuralNetLayerCuda::weightsToVector(double* pOutVector)
{
	throw Ex("Sorry, GNeuralNetLayerCuda::weightsToVector is not yet implemented");
	return 0;
}

// virtual
size_t GNeuralNetLayerCuda::vectorToWeights(const double* pVector)
{
	throw Ex("Sorry, GNeuralNetLayerCuda::vectorToWeights is not yet implemented");
	return 0;
}

// virtual
void GNeuralNetLayerCuda::copyWeights(const GNeuralNetLayer* pSource)
{
	throw Ex("Sorry, GNeuralNetLayerCuda::copyWeights is not yet implemented");
	return 0;
}

