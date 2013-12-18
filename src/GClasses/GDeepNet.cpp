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

#include "GDeepNet.h"
#include "GRand.h"
#include "GVec.h"
#include "GMath.h"
#include "GNeuralNet.h"
#include "GManifold.h"
#include "GActivation.h"
#include "GHolders.h"

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

void GDeepNetLayer::trainUnsupervised(const GMatrix& observations, size_t epochs, double startLearningRate, double endLearningRate)
{
	double decay = pow(endLearningRate / startLearningRate, 1.0 / epochs);
	if(observations.cols() != m_visibleCount)
		throw Ex("Expected ", to_str(m_visibleCount), " columns. Got ", to_str(observations.cols()));
	GRandomIndexIterator ii(observations.rows(), m_rand);
	size_t index;
	double learningRate = startLearningRate;
	for(size_t ep = 0; ep < epochs; ep++)
	{
		ii.reset();
		while(ii.next(index))
			update(observations[index], learningRate);
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
		GVec::copy(pHidden->row(i), activationEncode(), m_hiddenCount);
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
m_biasEncode(3, hidden),
m_biasDecode(3, visible),
m_noiseDeviation(0.0)
{
	double perturbation = 0.01;

	// Initialize the weights
	for(size_t i = 0; i < hidden; i++)
	{
		double* pRow = m_weightsEncode[i];
		for(size_t j = 0; j < visible; j++)
		{
			*pRow = perturbation * m_rand.normal();
			pRow++;
		}
	}
	double* pB = biasDecode();
	for(size_t j = 0; j < visible; j++)
		*(pB++) = perturbation * m_rand.normal();
	for(size_t i = 0; i < visible; i++)
	{
		double* pRow = m_weightsDecode[i];
		for(size_t j = 0; j < hidden; j++)
		{
			*pRow = perturbation * m_rand.normal();
			pRow++;
		}
	}
	pB = biasEncode();
	for(size_t j = 0; j < hidden; j++)
		*(pB++) = perturbation * m_rand.normal();
}

GStackableAutoencoder::~GStackableAutoencoder()
{
}

// virtual
void GStackableAutoencoder::propToHidden(const double* pVisible)
{
	double* pA = activationEncode();
	m_weightsEncode.multiply(pVisible, pA);
	GVec::add(pA, biasEncode(), m_hiddenCount);
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		*pA = tanh(*pA);
		pA++;
	}
}

// virtual
void GStackableAutoencoder::propToVisible(const double* pHidden)
{
	double* pA = activationDecode();
	m_weightsDecode.multiply(pHidden, pA);
	GVec::add(pA, biasDecode(), m_visibleCount);
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*pA = tanh(*pA);
		pA++;
	}
}

// virtual
void GStackableAutoencoder::draw(size_t iters)
{
	double* pH = activationEncode();
	for(size_t i = 0; i < m_hiddenCount; i++)
		*(pH++) = 0.01 * m_rand.normal();
	for(size_t i = 0; i < iters; i++)
	{
		propToVisible(activationEncode());
		propToHidden(activationDecode());
	}
	propToVisible(activationEncode());
}

// virtual
void GStackableAutoencoder::update(const double* pVisible, double learningRate)
{
	// Compute net and activation values
	if(m_noiseDeviation > 0.0)
	{
		double* pAV = activationDecode();
		GVec::copy(pAV, pVisible, m_visibleCount);
		GVec::perturb(pAV, m_noiseDeviation, m_visibleCount, m_rand);
		propToHidden(pAV);
	}
	else
		propToHidden(pVisible);
	propToVisible(activationEncode());

	// Compute visible blame term
	double* pBlame = blameDecode();
	const double* pV = pVisible;
	double* pAV = activationDecode();
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*(pBlame++) = (*pV - *pAV) * (1.0 - *pAV * *pAV);
		pAV++;
		pV++;
	}

	// Backpropagate
	backpropHelper1(activationEncode(), blameEncode(), learningRate);
	backpropHelper2(pVisible, blameDecode(), learningRate);
}

// virtual
void GStackableAutoencoder::backpropHelper1(const double* pInputs, double* pInputBlame, double learningRate)
{
	// Backpropagate the blame
	double* pBlame = blameDecode();
	m_weightsDecode.multiply(pBlame, pInputBlame, true);
	const double* pInp = pInputs;
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		*(pInputBlame++) *= (1.0 - (*pInp) * (*pInp));
		pInp++;
	}

	// Update weights
	double* pBias = biasDecode();
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
	double* pBlame = blameEncode();
	m_weightsEncode.multiply(pBlame, pInputBlame, true);
	const double* pInp = pInputs;
	for(size_t i = 0; i < m_visibleCount; i++)
	{
		*(pInputBlame++) *= (1.0 - (*pInp) * (*pInp));
		pInp++;
	}

	// Update weights
	double* pBias = biasEncode();
	for(size_t i = 0; i < m_hiddenCount; i++)
	{
		GVec::addScaled(m_weightsEncode[i], learningRate * (*pBlame), pInputs, m_visibleCount);
		*(pBias++) += learningRate * (*pBlame);
		pBlame++;
	}
}

GMatrix* GStackableAutoencoder::trainDimRed(const GMatrix& observations, GNeighborFinderCacheWrapper& nf)
{
	if(observations.cols() != m_visibleCount)
		throw Ex("Expected ", to_str(m_visibleCount), " cols. Got ", to_str(observations.cols()));

	// Reduce dimensionality while training a neural network to do the encoding
	GUniformRelation rel(m_visibleCount);
	GNeuralNet nnEncoder;
	nnEncoder.setTopology(m_hiddenCount);
	nnEncoder.beginIncrementalLearning(rel, rel);
	nnEncoder.getLayer(1)->setActivationFunction(new GActivationIdentity());
	nnEncoder.getLayer(0)->setToWeaklyApproximateIdentity();
	nnEncoder.getLayer(0)->perturbWeights(m_rand, 0.03);
	nnEncoder.getLayer(1)->setToWeaklyApproximateIdentity();
	nnEncoder.getLayer(1)->perturbWeights(m_rand, 0.03);
	GMatrix intrinsic;
	intrinsic.copy(&observations);
	GScalingUnfolder su;
	su.unfold(intrinsic, nf, 5, &nnEncoder, &observations); // This line trains nnEncoder

	// Copy the weights into the encoder
	GMatrix* pEncTranspose = nnEncoder.getLayer(0)->m_weights.transpose();
	Holder<GMatrix> hEncTranspose(pEncTranspose);
	m_weightsEncode.copy(pEncTranspose);
	hEncTranspose.reset();
	GVec::copy(biasEncode(), nnEncoder.getLayer(0)->bias(), m_hiddenCount);

	// Initialize the decoder
	GMatrix* pEncoding = mapToHidden(observations);
	Holder<GMatrix> hEncoding(pEncoding);

	// Train a decoder network
	GNeuralNet nnDecoder;
	nnDecoder.train(*pEncoding, observations);

	// Copy the weights into the decoder
	GMatrix* pDecTranspose = nnDecoder.getLayer(0)->m_weights.transpose();
	Holder<GMatrix> hDecTranspose(pDecTranspose);
	m_weightsDecode.copy(pDecTranspose);
	GVec::copy(biasDecode(), nnDecoder.getLayer(0)->bias(), m_visibleCount);
	return hEncoding.release();
}




/*
G2DConvolutionalLayer::G2DConvolutionalLayer(size_t visibleWidth, size_t visibleHeight, size_t visibleChannels, size_t kernelSize, size_t kernelCount, GRand& rand)
: GDeepNetLayer((visibleWidth + 2 * kernelSize - 2) * (visibleHeight + 2 * kernelSize - 2) * visibleChannels * kernelCount, visibleWidth * visibleHeight * visibleChannels, rand),
m_visibleWidth(visibleWidth),
m_visibleHeight(visibleHeight),
m_visibleChannels(visibleChannels),
m_kernelCount(kernelCount)
{
	m_pWeightsEncode = new GMatrix[m_visibleChannels * m_kernelCount];
	m_pWeightsDecode = new GMatrix[m_visibleChannels * m_kernelCount];
	m_pBiasEncode = new double[m_visibleChannels * m_kernelCount];
	m_pBiasDecode = new double[m_visibleChannels * m_kernelCount];
	m_pActivationEncode = new double[m_visibleCount];
	m_pActivationDecode = new double[m_hiddenCount];
	m_pBlameEncode = new double[m_visibleCount];
	m_pBlameDecode = new double[m_hiddenCount];
	for(size_t k = 0; k < m_visibleChannels * m_kernelCount; k++)
	{
		m_pWeightsEncode[k].resize(kernelSize, kernelSize);
		m_pWeightsDecode[k].resize(kernelSize, kernelSize);
		for(size_t i = 0; i < kernelSize; i++)
		{
			double* pRowEnc = m_pWeightsEncode[k][i];
			double* pRowDec = m_pWeightsDecode[k][i];
			for(size_t j = 0; j < kernelSize; j++)
			{
				*(pRowEnc++) = 0.01 * m_rand.normal();
				*(pRowDec++) = 0.01 * m_rand.normal();
			}
		}
		m_pBiasEncode[k] = 0.01 * m_rand.normal();
		m_pBiasDecode[k] = 0.01 * m_rand.normal();
	}
}

G2DConvolutionalLayer::~G2DConvolutionalLayer()
{
	delete[] m_pWeightsEncode;
	delete[] m_pWeightsDecode;
	delete[] m_pBiasEncode;
	delete[] m_pBiasDecode;
	delete[] m_pActivationEncode;
	delete[] m_pActivationDecode;
	delete[] m_pBlameEncode;
	delete[] m_pBlameDecode;
}
*/







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
		pNextLayer->propToVisible(pLayer->activationDecode());
		pLayer = pNextLayer;
	}
	return pLayer->activationDecode();
}

double* GDeepNet::decode(const double* pIntrinsic)
{
	GDeepNetLayer* pLayer = m_layers[0];
	pLayer->propToVisible(pIntrinsic);
	double* pIn = pLayer->activationDecode();
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		pLayer = m_layers[i];
		pLayer->propToVisible(pIn);
		pIn = pLayer->activationDecode();
	}
	return pIn;
}

double* GDeepNet::encode(const double* pObserved)
{
	size_t i = m_layers.size() - 1;
	GDeepNetLayer* pLayer = m_layers[i];
	pLayer->propToHidden(pObserved);
	double* pIn = pLayer->activationEncode();
	for(i--; i < m_layers.size(); i--)
	{
		pLayer = m_layers[i];
		pLayer->propToHidden(pIn);
		pIn = pLayer->activationEncode();
	}
	return pIn;
}

void GDeepNet::trainLayerwise(GMatrix& observations, size_t epochs, double startLearningRate, double endLearningRate)
{
	GMatrix* pObs = &observations;
	Holder<GMatrix> hObs(NULL);
	for(size_t i = m_layers.size() - 1; i < m_layers.size(); i--)
	{
		// Train the layer
		m_layers[i]->trainUnsupervised(*pObs, epochs, startLearningRate, endLearningRate);

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
	decode(encode(pObservation));
	GDeepNetLayer* pVisibleLayer = m_layers[m_layers.size() - 1];
	double* pAct = pVisibleLayer->activationDecode();
	double* pBlame = pVisibleLayer->blameDecode();
	const double* pObs = pObservation;
	for(size_t i = 0; i < pVisibleLayer->visibleCount(); i++)
	{
		*(pBlame++) = (*pObs - *pAct) * (1.0 - (*pAct) * (*pAct));
		pAct++;
		pObs++;
	}

	// Backpropagate blame and update weights
	for(size_t i = m_layers.size() - 1; i > 0; i--)
		m_layers[i]->backpropHelper1(m_layers[i - 1]->activationDecode(), m_layers[i - 1]->blameDecode(), learningRate);
	m_layers[0]->backpropHelper1(m_layers[0]->activationEncode(), m_layers[0]->blameEncode(), learningRate);
	for(size_t i = 0; i + 1 < m_layers.size(); i++)
		m_layers[i]->backpropHelper2(m_layers[i + 1]->activationEncode(), m_layers[i + 1]->blameEncode(), learningRate);
	m_layers[m_layers.size() - 1]->backpropHelper2(pObservation, m_layers[m_layers.size() - 1]->blameDecode(), learningRate);
}

#ifndef MIN_PREDICT
void GDeepNet_testUpdate()
{
	// Make a deep net
	GRand rand(0);
	GDeepNet dn(rand);
	dn.addLayer(new GStackableAutoencoder(3, 4, rand));

	// Make a neural net
	GNeuralNet nn;
	nn.setLearningRate(0.8);
	nn.setTopology(3);
	GUniformRelation ends(4);
	nn.beginIncrementalLearning(ends, ends);

	// Copy the weights from the deep network to the neural net
	Holder<GMatrix> h(NULL);
	h.reset(((GStackableAutoencoder*)dn.layers()[0])->weightsEncode().transpose());
	nn.getLayer(0)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(0)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasEncode(), 3);

	h.reset(((GStackableAutoencoder*)dn.layers()[0])->weightsDecode().transpose());
	nn.getLayer(1)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(1)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasDecode(), 4);

	// Train both of them with a phony observation
	double obs[4];
	rand.cubical(obs, 4);
	nn.trainIncremental(obs, obs);
	((GStackableAutoencoder*)dn.layers()[0])->update(obs, 0.8);

	// Check that the weights were updated identically
	double err = 0.0;
	err += nn.getLayer(0)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.layers()[0])->weightsEncode(), true);
	err += nn.getLayer(1)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.layers()[0])->weightsDecode(), true);
	err += GVec::squaredDistance(nn.getLayer(0)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasEncode(), 3);
	err += GVec::squaredDistance(nn.getLayer(1)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasDecode(), 4);
	if(err > 1e-12)
		throw Ex("failed");
}

void GDeepNet_testBackprop()
{
	// Make a deep net
	GRand rand(0);
	GDeepNet dn(rand);
	dn.addLayer(new GStackableAutoencoder(3, 4, rand));
	dn.addLayer(new GStackableAutoencoder(2, 3, rand));

	// Make a neural net
	GNeuralNet nn;
	nn.setLearningRate(0.8);
	nn.setTopology(3, 2, 3);
	GUniformRelation ends(4);
	nn.beginIncrementalLearning(ends, ends);

	// Copy the weights from the deep network to the neural net
	Holder<GMatrix> h(NULL);
	h.reset(((GStackableAutoencoder*)dn.layers()[1])->weightsEncode().transpose());
	nn.getLayer(0)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(0)->bias(), ((GStackableAutoencoder*)dn.layers()[1])->biasEncode(), 3);

	h.reset(((GStackableAutoencoder*)dn.layers()[0])->weightsEncode().transpose());
	nn.getLayer(1)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(1)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasEncode(), 2);

	h.reset(((GStackableAutoencoder*)dn.layers()[0])->weightsDecode().transpose());
	nn.getLayer(2)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(2)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasDecode(), 3);

	h.reset(((GStackableAutoencoder*)dn.layers()[1])->weightsDecode().transpose());
	nn.getLayer(3)->m_weights.copy(h.get());
	GVec::copy(nn.getLayer(3)->bias(), ((GStackableAutoencoder*)dn.layers()[1])->biasDecode(), 4);

	// Train both of them with a phony observation
	double obs[4];
	rand.cubical(obs, 4);
	nn.trainIncremental(obs, obs);
	dn.refineBackprop(obs, 0.8);

	// Check that the weights were updated identically
	double err = 0.0;
	err += nn.getLayer(0)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.layers()[1])->weightsEncode(), true);
	err += nn.getLayer(1)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.layers()[0])->weightsEncode(), true);
	err += nn.getLayer(2)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.layers()[0])->weightsDecode(), true);
	err += nn.getLayer(3)->m_weights.sumSquaredDifference(((GStackableAutoencoder*)dn.layers()[1])->weightsDecode(), true);
	err += GVec::squaredDistance(nn.getLayer(0)->bias(), ((GStackableAutoencoder*)dn.layers()[1])->biasEncode(), 3);
	err += GVec::squaredDistance(nn.getLayer(1)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasEncode(), 2);
	err += GVec::squaredDistance(nn.getLayer(2)->bias(), ((GStackableAutoencoder*)dn.layers()[0])->biasDecode(), 3);
	err += GVec::squaredDistance(nn.getLayer(3)->bias(), ((GStackableAutoencoder*)dn.layers()[1])->biasDecode(), 4);
	if(err > 1e-12)
		throw Ex("failed");
}

// static
void GDeepNet::test()
{
	GDeepNet_testUpdate();
	GDeepNet_testBackprop();
}
#endif // MIN_PREDICT


} // namespace GClasses
