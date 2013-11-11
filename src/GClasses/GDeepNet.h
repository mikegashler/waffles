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

#ifndef __GDEEPNET_H__
#define __GDEEPNET_H__

#include "GMatrix.h"

namespace GClasses {

class GRand;


/// Abstract parent class of network layers that are suitable for stacking to form a deep network
class GDeepNetLayer
{
protected:
	size_t m_hiddenCount;
	size_t m_visibleCount;
	GRand& m_rand;

public:
	/// General-purpose constructor
	GDeepNetLayer(size_t hidden, size_t visible, GRand& rand);
	virtual ~GDeepNetLayer();

	/// Returns the number of visible units
	size_t visibleCount() { return m_visibleCount; }

	/// Returns the number of hidden units
	size_t hiddenCount() { return m_hiddenCount; }

	/// Trains this layer in an unsupervised manner
	void trainUnsupervised(const GMatrix& observations, size_t epochs = 100, double initialLearningRate = 0.1, double decay = 0.97);

	/// Map observations through this layer to generate a matrix suitable for training the layer that feeds into this layer.
	/// The caller is responsible to delete the returned matrix.
	GMatrix* mapToHidden(const GMatrix& observations);

	/// Return a vector of hidden activation values
	virtual double* activationEncode() = 0;

	/// Return a vector of visible activation values
	virtual double* activationDecode() = 0;

	/// Return a vector of hidden blame values
	virtual double* blameEncode() = 0;

	/// Return a vector of visible blame values
	virtual double* blameDecode() = 0;

	/// Feeds pVisible backward through this layer to generate a hidden vector
	/// The result can be retrieved by calling activationEncode().
	virtual void propToHidden(const double* pVisible) = 0;

	/// Feeds pHidden forward through this layer to generate a predicted visible vector
	/// The result can be retrieved by calling activationDecode().
	virtual void propToVisible(const double* pHidden) = 0;

	/// Draws a sample from this layer at the end of an MCMC chain of length iters.
	/// The result can be retrieved by calling activationDecode().
	virtual void draw(size_t iters) = 0;

	/// Present pVisible to this layer for training in an on-line manner.
	virtual void update(const double* pVisible, double learningRate) = 0;

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper1(const double* pInputs, double* pInputBlame, double learningRate) = 0;

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper2(const double* pInputs, double* pInputBlame, double learningRate) = 0;
};




/// Implements a restricted boltzmann machine (RBM).
class GRestrictedBoltzmannMachine : public GDeepNetLayer
{
protected:
	GMatrix m_w;
	GMatrix m_delta;
	double* m_biasHidden;
	double* m_biasVisible;
	double* m_activationHidden;
	double* m_activationVisible;
	double* m_blameHidden;
	double* m_blameVisible;

public:
	GRestrictedBoltzmannMachine(size_t hidden, size_t visible, GRand& rand);
	~GRestrictedBoltzmannMachine();

	/// Returns the vector of blame terms
	virtual double* blameEncode() { return m_blameHidden; }

	/// Returns the vector of blame terms
	virtual double* blameDecode() { return m_blameVisible; }

	/// Returns the vector of hidden activation values.
	virtual double* activationEncode() { return m_activationHidden; }

	/// Returns the vector of hidden activation values.
	virtual double* activationDecode() { return m_activationVisible; }

	/// Propagates to compute the hidden activations.
	virtual void propToHidden(const double* pObserved);

	/// Propagates to compute the visibile activations.
	virtual void propToVisible(const double* pHidden);

	/// Assumes the visible activations have already been set. Returns an inferred sample of the hiddens.
	void sampleHidden();

	/// Assumes the hidden activations have already been set. Returns an inferred sample of the visibles.
	void sampleVisible();

	/// Sets the hidden activations to random values, then iterates the specified number of times
	virtual void draw(size_t iters);

	/// Computes the free energy for the given observation.
	double freeEnergy(const double* pVisibleSample);

	/// Update the weights by contrastive divergence.
	void contrastiveDivergenceUpdate(const double* pVisibleSample, double learningRate, double momentum, size_t gibbsSamples = 1);

	/// Update the weights in the maximum likelihood manner.
	void maximumLikelihoodUpdate(const double* pVisibleSample, double learningRate, double momentum);

	/// Present a single visible vector, and update all the weights by on-line gradient descent.
	virtual void update(const double* pVisible, double learningRate);

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper1(const double* pInputs, double* pInputBlame, double learningRate);

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper2(const double* pInputs, double* pInputBlame, double learningRate);
};



/// A stackable autoencoder. It can be thought of as two single-layer perceptrons, one that goes in each direction
/// between the hidden and visible layers. It differs from a Restricted Boltzmann Machine (RBM) in that it uses a
/// separate weights matrix for each of the two directions, instead of using the same weights matrix for both
/// directions, as the RBM does.
class GStackableAutoencoder : public GDeepNetLayer
{
protected:
	GMatrix m_weightsEncode; // The weights that map from the input to the hidden layer
	GMatrix m_weightsDecode; // The weights that map from the hidden to the output layer
	GMatrix m_biasEncode; // The bias and activation values for the hidden layer
	GMatrix m_biasDecode; // The bias and activation values for the output layer
	double m_noiseDeviation;

public:
	/// General-purpose constructor
	GStackableAutoencoder(size_t hidden, size_t visible, GRand& rand);
	~GStackableAutoencoder();

	/// Returns the number of visible units
	virtual size_t visibleCount() { return m_visibleCount; }

	/// Returns the number of hidden units
	virtual size_t hiddenCount() { return m_hiddenCount; }

	/// Train this autoencoder to denoise by injecting Gaussian noise with
	/// the specified deviation into all training observations.
	void denoise(double deviation) { m_noiseDeviation = deviation; }

	/// Returns the vector of hidden biases.
	double* biasEncode() { return m_biasEncode[0]; }

	/// Returns the vector of visible biases.
	double* biasDecode() { return m_biasDecode[0]; }

	/// Returns the weights for the encoder
	GMatrix& weightsEncode() { return m_weightsEncode; }

	/// Returns the weights for the decoder
	GMatrix& weightsDecode() { return m_weightsDecode; }

	/// Returns the vector of blame terms
	virtual double* blameEncode() { return m_biasEncode[2]; }

	/// Returns the vector of blame terms
	virtual double* blameDecode() { return m_biasDecode[2]; }

	/// Returns the vector of hidden activation values.
	virtual double* activationEncode() { return m_biasEncode[1]; }

	/// Returns the vector of hidden activation values.
	virtual double* activationDecode() { return m_biasDecode[1]; }

	/// Computes a hidden vector from the given visible vector
	virtual void propToHidden(const double* pVisible);

	/// Computes a visible vector from the given hidden vector
	virtual void propToVisible(const double* pHidden);

	/// Draws a random sample from this layer
	virtual void draw(size_t iters);

	/// Present a single visible vector, and update all the weights by on-line gradient descent.
	virtual void update(const double* pVisible, double learningRate);

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper1(const double* pInputs, double* pInputBlame, double learningRate);

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper2(const double* pInputs, double* pInputBlame, double learningRate);

	/// Trains the layer using a dimensionality reduction technique.
	/// Returns the data as mapped through to the input of this layer.
	GMatrix* trainDimRed(const GMatrix& observations);
};


/*
class G2DConvolutionalLayer : public GDeepNetLayer
{
protected:
	size_t m_visibleWidth, m_visibleHeight, m_visibleChannels, m_kernelCount;
	GMatrix* m_pWeightsEncode; // Array of weights that map from the input to the hidden layer
	GMatrix* m_pWeightsDecode; // Array of weights that map from the hidden to the output layer
	double* m_pBiasEncode;
	double* m_pBiasDecode;
	double* m_pActivationEncode;
	double* m_pActivationDecode;
	double* m_pBlameEncode;
	double* m_pBlameDecode;

public:
	/// General-purpose constructor.
	/// visibleWidth, visibleHeight, and visibleChannels all specify the dimensions of the visible image.
	/// kernelSize specifies both the width and height of the convolutional kernel.
	/// kernelCount specifies the number of convolutional kernels to train.
	/// The Each hidden mapping will have a width of (visibleWidth + kernelSize - 1) and a
	/// height of (visibleHeight + kernelSize - 1). There will be (visibleChannels * kernelCount)
	/// hidden channels.
	G2DConvolutionalLayer(size_t visibleWidth, size_t visibleHeight, size_t visibleChannels, size_t kernelSize, size_t kernelCount, GRand& rand);
	~G2DConvolutionalLayer();

	/// Returns the number of visible units
	virtual size_t visibleCount() { return m_visibleWidth * m_visibleHeight * m_visibleChannels; }

	/// Returns the number of hidden units
	virtual size_t hiddenCount() { return m_hiddenCount; }

	/// Returns the vector of blame terms
	virtual double* blameEncode() { return m_pBlameEncode; }

	/// Returns the vector of blame terms
	virtual double* blameDecode() { return m_pBlameDecode; }

	/// Returns the vector of hidden activation values.
	virtual double* activationEncode() { return m_pActivationEncode; }

	/// Returns the vector of hidden activation values.
	virtual double* activationDecode() { return m_pActivationDecode; }
/ *
	/// Computes a hidden vector from the given visible vector
	virtual void propToHidden(const double* pVisible);

	/// Computes a visible vector from the given hidden vector
	virtual void propToVisible(const double* pHidden);

	/// Draws a random sample from this layer
	virtual void draw(size_t iters);

	/// Present a single visible vector, and update all the weights by on-line gradient descent.
	virtual void update(const double* pVisible, double learningRate);

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper1(const double* pInputs, double* pInputBlame, double learningRate);

	/// A helper method called by GDeepNet::refineBackprop. You should probably not call this method directly.
	virtual void backpropHelper2(const double* pInputs, double* pInputBlame, double learningRate);
* /
};
*/


/// A collection of GDeepNetLayer instances.
class GDeepNet
{
protected:
	std::vector<GDeepNetLayer*> m_layers; // the last layer is the visible end
	GRand& m_rand;

public:
	GDeepNet(GRand& rand);
	virtual ~GDeepNet();

#ifndef MIN_PREDICT
	/// Run some unit tests. Throws an exception if any tests fail.
	static void test();
#endif

	/// Adds a new layer to this deep net. Takes ownership of pNewLayer.
	/// The first layer added is the visible layer. The last layer added
	/// is the intrinsic or input layer.
	void addLayer(GDeepNetLayer* pNewLayer);

	/// Draw a sample from the hidden-most layer, then feed it forward through
	/// all the layers to return a sample predicted observation.
	double* draw(size_t iters);

	/// Feed pIntrinsic forward through all the layers to return a predicted observation.
	double* forward(const double* pIntrinsic);

	/// Feed pObserved backward through all the layers to return a predicted intrinsic representation.
	double* backward(const double* pObserved);

	/// This performs greedy layer-wise training. That is, it trains each layer (starting with the
	/// visible end) for many epochs, then maps the data through the layer, and trains the next layer,
	/// until all layers have been trained. Typically, this is done as a pre-processing step to find
	/// a good set of initial weights for the deep network.
	void trainLayerwise(GMatrix& observations, size_t epochs = 100, double initialLearningRate = 0.1, double decay = 0.97);

	/// Present a single observation to refine all of the layers by backpropagation. (Note that this only
	/// refines the forward-direction component of the layers. Its effect on backward-direction effectiveness
	/// may not be good.)
	void refineBackprop(const double* pObservation, double learningRate);
};





} // namespace GClasses

#endif // __GDEEPNET_H__
