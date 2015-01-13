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

#ifndef __GNEURALNET_H__
#define __GNEURALNET_H__

#include "GLearner.h"
#include <vector>

namespace GClasses {

/// This value is used for the number of inputs or outputs of a neural net layer when
/// you do not wish to specify a fixed size. For example, it may be used for the inputs
/// of the first layer or the outputs of the last layer, because the training data will
/// provide these sizes. (In fact, those ends will be resized to fit the data whether or
/// not FLEXIBLE_SIZE is used.) FLEXIBLE_SIZE should probably not be used on an end that
/// will be connected to another end with FLEXIBLE_SIZE because then both ends will stay
/// at a size of zero, which will result in approximately baseline predictions.
#define FLEXIBLE_SIZE 0

class GNeuralNet;
class GRand;
class GBackProp;
class GImage;
class GActivationFunction;
class LagrangeVals;

/// Represents a layer of neurons in a neural network
class GNeuralNetLayer
{
public:
	GNeuralNetLayer() {}
	virtual ~GNeuralNetLayer() {}

	/// Returns the type of this layer
	virtual const char* type() = 0;

	/// Returns true iff this layer does its computations in parallel on a GPU.
	virtual bool usesGPU() { return false; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) = 0;

	/// Unmarshalls the specified DOM node into a layer object.
	static GNeuralNetLayer* deserialize(GDomNode* pNode);

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() = 0;

	/// Returns the number of values that this layer outputs.
	virtual size_t outputs() = 0;

	/// Resizes this layer. If pRand is non-NULL, then it preserves existing weights when possible
	/// and initializes any others to small random values.
	virtual void resize(size_t inputs, size_t outputs, GRand* pRand = NULL, double deviation = 0.03) = 0;

	/// Returns a buffer where the activation from the most-recent call to feedForward is stored.
	virtual double* activation() = 0;

	/// Returns a buffer where the error terms for each unit are stored.
	virtual double* error() = 0;

	/// Copies the bias vector into the net vector. (This should be done before feedIn is called.)
	virtual void copyBiasToNet() = 0;

	/// Feeds the inputs (or a portion of the inputs) through the weights and updates the net.
	virtual void feedIn(const double* pIn, size_t inputStart, size_t inputCount) = 0;

	/// Feeds the previous layer's activation into this layer. (Implementations
	/// for specialized hardware may override this method to avoid shuttling the previous
	/// layer's activation back to host memory.)
	virtual void feedIn(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
	{
		feedIn(pUpStreamLayer->activation(), inputStart, pUpStreamLayer->outputs());
	}

	/// Applies the activation function to the net vector to compute the activation vector.
	virtual void activate() = 0;

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop) = 0;

	/// Randomly sets some of the weights to 0. (The dropped weights are restored when you call updateWeightsAndRestoreDroppedOnes.)
	virtual void dropConnect(GRand& rand, double probOfDrop) = 0;

	/// Feeds in the bias and pIn, then computes the activation of this layer.
	void feedForward(const double* pIn);

	/// Computes the error term of the activation.
	virtual void computeError(const double* pTarget) = 0;

	/// Converts the error term to refer to the net input.
	virtual void deactivateError() = 0;

	/// Computes the activation error of the layer that feeds into this one.
	/// inputStart is used if multiple layers feed into this one. It specifies
	/// the starting index of all the inputs where this layer feeds in.
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart = 0) = 0;

	/// Updates the bias of this layer by gradient descent. (Assumes the error has already been computed and deactivated.)
	virtual void updateBias(double learningRate, double momentum) = 0;

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum) = 0;

	/// Refines the weights by gradient descent.
	virtual void updateWeights(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double momentum)
	{
		updateWeights(pUpStreamLayer->activation(), inputStart, pUpStreamLayer->outputs(), learningRate, momentum);
	}

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum) = 0;

	/// Refines the weights by gradient descent.
	virtual void updateWeightsAndRestoreDroppedOnes(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double momentum)
	{
		updateWeightsAndRestoreDroppedOnes(pUpStreamLayer->activation(), inputStart, pUpStreamLayer->outputs(), learningRate, momentum);
	}

	/// Multiplies all the weights by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) = 0;

	/// Moves all weights in the direction of zero by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) = 0;

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t countWeights() = 0;

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) = 0;

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) = 0;

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(GNeuralNetLayer* pSource) = 0;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) = 0;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise. The
	/// default values apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX) = 0;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double max) = 0;

	/// Compute the L1 norm (sum of absolute values) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL1Norm(size_t unit) = 0;

	/// Compute the L2 norm (sum of squares) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL2Norm(size_t unit) = 0;

	/// Compute the L1 norm (sum of absolute values) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL1Norm(size_t input) = 0;

	/// Compute the L2 norm (sum of squares) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL2Norm(size_t input) = 0;

	/// Scale weights that feed into the specified unit
	virtual void scaleUnitIncomingWeights(size_t unit, double scalar) = 0;

	/// Scale weights that feed into this layer from the specified input
	virtual void scaleUnitOutgoingWeights(size_t input, double scalar) = 0;

	/// Adjusts weights such that values in the new range will result in the
	/// same behavior that previously resulted from values in the old range.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0) = 0;

	/// Feeds a matrix through this layer, one row at-a-time, and returns the resulting transformed matrix.
	GMatrix* feedThrough(const GMatrix& data);

	/// Gets the weights and bias of a single neuron.
	virtual void setWeightsSingleNeuron(size_t outputNode, const double* weights)
	{
		throw Ex("Not yet implemented");
	}

	/// Gets the weights and bias of a single neuron.
	virtual void getWeightsSingleNeuron(size_t outputNode, double*& weights)
	{
		throw Ex("Not yet implemented");
	}

	virtual void copySingleNeuronWeights(size_t source, size_t dest)
	{
		throw Ex("Not yet implemented");
	}

protected:
	GDomNode* baseDomNode(GDom* pDoc);
};


class GLayerClassic : public GNeuralNetLayer
{
friend class GNeuralNet;
protected:
	GMatrix m_weights; // Each row is an upstream neuron. Each column is a downstream neuron.
	GMatrix m_delta; // Used to implement momentum
	GMatrix m_bias; // Row 0 is the bias. Row 1 is the net. Row 2 is the activation. Row 3 is the error. Row 4 is the biasDelta. Row 5 is the slack.
	GActivationFunction* m_pActivationFunction;

public:
using GNeuralNetLayer::feedIn;
using GNeuralNetLayer::updateWeights;
using GNeuralNetLayer::updateWeightsAndRestoreDroppedOnes;

	/// General-purpose constructor. Takes ownership of pActivationFunction.
	/// If pActivationFunction is NULL, then GActivationTanH is used.
	GLayerClassic(size_t inputs, size_t outputs, GActivationFunction* pActivationFunction = NULL);

	/// Deserializing constructor
	GLayerClassic(GDomNode* pNode);
	~GLayerClassic();

	/// Returns the type of this layer
	virtual const char* type() { return "classic"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() { return m_weights.rows(); }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() { return m_weights.cols(); }

	/// Resizes this layer. If pRand is non-NULL, then it preserves existing weights when possible
	/// and initializes any others to small random values.
	virtual void resize(size_t inputs, size_t outputs, GRand* pRand = NULL, double deviation = 0.03);

	/// Returns the activation values from the most recent call to feedForward().
	virtual double* activation() { return m_bias[2]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual double* error() { return m_bias[3]; }

	/// Copies the bias vector into the net vector.
	virtual void copyBiasToNet();

	/// Feeds a portion of the inputs through the weights and updates the net.
	virtual void feedIn(const double* pIn, size_t inputStart, size_t inputCount);

	/// Applies the activation function to the net vector to compute the activation vector.
	virtual void activate();

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Randomly sets some of the weights to 0. (The dropped weights are restored when you call updateWeightsAndRestoreDroppedOnes.)
	virtual void dropConnect(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const double* pTarget);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart = 0);

	/// Updates the bias of this layer by gradient descent. (Assumes the error has already been computed and deactivated.)
	virtual void updateBias(double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases);

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases);

	/// Contracts all the weights. (Assumes contractive error terms have already been set.)
	void contractWeights(double factor, bool contractBiases);

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t countWeights();

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector);

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector);

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double max);

	/// Compute the L1 norm (sum of absolute values) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL1Norm(size_t unit);

	/// Compute the L2 norm (sum of squares) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL2Norm(size_t unit);

	/// Compute the L1 norm (sum of absolute values) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL1Norm(size_t input);

	/// Compute the L2 norm (sum of squares) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL2Norm(size_t input);

	/// Scale weights that feed into the specified unit
	virtual void scaleUnitIncomingWeights(size_t unit, double scalar);

	/// Scale weights that feed into this layer from the specified input
	virtual void scaleUnitOutgoingWeights(size_t input, double scalar);

	/// Returns a reference to the weights matrix of this layer
	GMatrix& weights() { return m_weights; }

	/// Returns the bias vector of this layer.
	double* bias() { return m_bias[0]; }

	/// Returns the bias vector of this layer.
	const double* bias() const { return m_bias[0]; }

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	double* net() { return m_bias[1]; }

	/// Returns a buffer used to store delta values for each bias in this layer.
	double* biasDelta() { return m_bias[4]; }

	/// Returns a vector used to specify slack terms for each unit in this layer.
	double* slack() { return m_bias[5]; }

	/// Returns a pointer to the activation function used in this layer
	GActivationFunction* activationFunction() { return m_pActivationFunction; }

	/// Feeds a vector forward through this layer. Uses the first value in pIn as an input bias.
	void feedForwardWithInputBias(const double* pIn);

	/// Feeds a vector forward through this layer to compute only the one specified output value.
	void feedForwardToOneOutput(const double* pIn, size_t output, bool inputBias);

	/// This is the same as computeError, except that it only computes the error of a single unit.
	void computeErrorSingleOutput(double target, size_t output);

	/// Same as deactivateError, but only applies to a single unit in this layer.
	void deactivateErrorSingleOutput(size_t output);

	/// Backpropagates the error from a single output node to a hidden layer.
	/// (Assumes that the error in the output node has already been deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	void backPropErrorSingleOutput(size_t output, double* pUpStreamError);

	/// Gets the weights and bias of a single neuron.
	void setWeightsSingleNeuron(size_t outputNode, const double* weights);

	/// Gets the weights and bias of a single neuron.
	void getWeightsSingleNeuron(size_t outputNode, double*& weights);

	/// Updates the weights and bias of a single neuron. (Assumes the error has already been computed and deactivated.)
	void updateWeightsSingleNeuron(size_t outputNode, const double* pUpStreamActivation, double learningRate, double momentum);

	/// Sets the weights of this layer to make it weakly approximate the identity function.
	/// start specifies the first unit whose incoming weights will be adjusted.
	/// count specifies the maximum number of units whose incoming weights are adjusted.
	void setWeightsToIdentity(size_t start = 0, size_t count = (size_t)-1);

	/// Adjusts the value of each weight to, w = w - factor * pow(w, power).
	/// If power is 1, this is the same as calling scaleWeights.
	/// If power is 0, this is the same as calling diminishWeights.
	void regularizeWeights(double factor, double power);

	/// Transforms the weights of this layer by the specified transformation matrix and offset vector.
	/// transform should be the pseudoinverse of the transform applied to the inputs. pOffset should
	/// be the negation of the offset added to the inputs after the transform, or the transformed offset
	/// that is added before the transform.
	void transformWeights(GMatrix& transform, const double* pOffset);

	/// Adjusts weights such that values in the new range will result in the
	/// same behavior that previously resulted from values in the old range.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);
	void copySingleNeuronWeights(size_t source, size_t dest);
};



class GLayerSoftMax : public GLayerClassic
{
public:
	GLayerSoftMax(size_t inputs, size_t outputs);
	GLayerSoftMax(GDomNode* pNode);
	virtual ~GLayerSoftMax() {}

	/// Returns the type of this layer
	virtual const char* type() { return "softmax"; }

	/// Applies the logistic activation function to the net vector to compute the activation vector,
	/// and also adjusts the weights so that the activations sum to 1.
	virtual void activate();

	/// This method is a no-op, since cross-entropy training does not multiply by the derivative of the logistic function.
	virtual void deactivateError() {}
};




/// Facilitates mixing multiple types of layers side-by-side into a single layer.
class GLayerMixed : public GNeuralNetLayer
{
protected:
	GMatrix m_inputError;
	GMatrix m_activation;
	std::vector<GNeuralNetLayer*> m_components;

public:
using GNeuralNetLayer::feedIn;
using GNeuralNetLayer::updateWeights;
using GNeuralNetLayer::updateWeightsAndRestoreDroppedOnes;

	/// General-purpose constructor. (You should call addComponent at least twice to mix some layers, after constructing this object.)
	GLayerMixed();

	/// Deserializing constructor
	GLayerMixed(GDomNode* pNode);
	~GLayerMixed();

	/// Returns the type of this layer
	virtual const char* type() { return "mixed"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Adds another component of this layer. In other words, make this layer bigger by adding pComponent to it,
	/// as a peer beside the other components in this layer.
	void addComponent(GNeuralNetLayer* pComponent);

	/// Returns the specified component.
	GNeuralNetLayer& component(size_t i) { return *m_components[i]; }

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs();

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs();

	/// Throws an exception if the specified dimensions would change anything. Also
	/// throws an exception if pRand is not NULL.
	virtual void resize(size_t inputs, size_t outputs, GRand* pRand = NULL, double deviation = 0.03);

	/// Returns the activation values from the most recent call to feedForward().
	virtual double* activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual double* error() { return m_activation[1]; }

	/// Calls copyBiasToNet for each component.
	virtual void copyBiasToNet();

	/// Feeds a portion of the inputs through the weights and updates the net for each component.
	virtual void feedIn(const double* pIn, size_t inputStart, size_t inputCount);

	/// Applies the activation function to the net vector to compute the activation vector
	/// in each component, then aggregates all the activation vectors into a single activation for this layer.
	virtual void activate();

	/// Calls dropOut for each component.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Randomly sets some of the weights to 0. (The dropped weights are restored when you call updateWeightsAndRestoreDroppedOnes.)
	virtual void dropConnect(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const double* pTarget);

	/// Copies the error vector into the corresponding buffer for each component,
	/// then calls deactivateError for each component.
	virtual void deactivateError();

	/// Calls backPropError for each component, and adds them up into the upstreams error buffer.
	/// (Note that the current implementation of this method may not be compatible with GPU-optimized layers.
	/// This method still needs to be audited for compatibility with such layers.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart = 0);

	/// Calls updateBias for each component.
	virtual void updateBias(double learningRate, double momentum);

	/// Calls updateWeights for each component.
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Calls scaleWeights for each component.
	virtual void scaleWeights(double factor, bool scaleBiases);

	/// Calls diminishWeights for each component.
	virtual void diminishWeights(double amount, bool regularizeBiases);

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t countWeights();

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector);

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector);

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(GNeuralNetLayer* pSource);

	/// Calls resetWeights for each component.
	virtual void resetWeights(GRand& rand);

	/// Calls perturbWeights for each component.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Calls maxNorm for each component.
	virtual void maxNorm(double max);

	/// Compute the L1 norm (sum of absolute values) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL1Norm(size_t unit);

	/// Compute the L2 norm (sum of squares) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL2Norm(size_t unit);

	/// Compute the L1 norm (sum of absolute values) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL1Norm(size_t input);

	/// Compute the L2 norm (sum of squares) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL2Norm(size_t input);

	/// Scale weights that feed into the specified unit
	virtual void scaleUnitIncomingWeights(size_t unit, double scalar);

	/// Scale weights that feed into this layer from the specified input
	virtual void scaleUnitOutgoingWeights(size_t input, double scalar);

	/// Adjusts weights such that values in the new range will result in the
	/// same behavior that previously resulted from values in the old range.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);
};



class GLayerRestrictedBoltzmannMachine : public GNeuralNetLayer
{
protected:
	GMatrix m_weights; // Each column is an upstream neuron. Each row is a downstream neuron.
	GMatrix m_delta;
	GMatrix m_bias; // Row 0 is the bias. Row 1 is the net. Row 2 is the activation. Row 3 is the error. Row 4 is the delta.
	GMatrix m_biasReverse; // Row 0 is the bias. Row 1 is the net. Row 2 is the activation. Row 3 is the error. Row 4 is the delta.
	GActivationFunction* m_pActivationFunction;

public:
using GNeuralNetLayer::feedIn;
using GNeuralNetLayer::updateWeights;
using GNeuralNetLayer::updateWeightsAndRestoreDroppedOnes;

	/// General-purpose constructor. Takes ownership of pActivationFunction.
	GLayerRestrictedBoltzmannMachine(size_t inputs, size_t outputs, GActivationFunction* pActivationFunction = NULL);

	/// Deserializing constructor
	GLayerRestrictedBoltzmannMachine(GDomNode* pNode);

	~GLayerRestrictedBoltzmannMachine();

	/// Returns the type of this layer
	virtual const char* type() { return "rbm"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Returns the number of visible units.
	virtual size_t inputs() { return m_weights.cols(); }

	/// Returns the number of hidden units.
	virtual size_t outputs() { return m_weights.rows(); }

	/// Resizes this layer. If pRand is non-NULL, then it preserves existing weights when possible
	/// and initializes any others to small random values.
	virtual void resize(size_t inputs, size_t outputs, GRand* pRand = NULL, double deviation = 0.03);

	/// Returns the activation values on the hidden end.
	virtual double* activation() { return m_bias[2]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual double* error() { return m_bias[3]; }

	/// Copies the bias vector into the net vector.
	virtual void copyBiasToNet();

	/// Feeds a portion of the inputs through the weights and updates the net.
	virtual void feedIn(const double* pIn, size_t inputStart, size_t inputCount);

	/// Applies the activation function to the net vector to compute the activation vector.
	virtual void activate();

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Randomly sets some of the weights to 0. (The dropped weights are restored when you call updateWeightsAndRestoreDroppedOnes.)
	virtual void dropConnect(GRand& rand, double probOfDrop);

	/// Feed a vector from the hidden end to the visible end. The results are placed in activationReverse();
	void feedBackward(const double* pIn);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const double* pTarget);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart = 0);

	/// Updates the bias of this layer by gradient descent. (Assumes the error has already been computed and deactivated.)
	virtual void updateBias(double learningRate, double momentum);

	/// Adjust weights that feed into this layer. (Assumes the error has already been deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases);

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases);

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t countWeights();

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector);

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector);

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// Also perturbs the bias.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double max);

	/// Compute the L1 norm (sum of absolute values) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL1Norm(size_t unit);

	/// Compute the L2 norm (sum of squares) of weights feeding into the specified unit
	virtual double unitIncomingWeightsL2Norm(size_t unit);

	/// Compute the L1 norm (sum of absolute values) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL1Norm(size_t input);

	/// Compute the L2 norm (sum of squares) of weights feeding into this layer from the specified input
	virtual double unitOutgoingWeightsL2Norm(size_t input);

	/// Scale weights that feed into the specified unit
	virtual void scaleUnitIncomingWeights(size_t unit, double scalar);

	/// Scale weights that feed into this layer from the specified input
	virtual void scaleUnitOutgoingWeights(size_t input, double scalar);

	/// Returns a reference to the weights matrix of this layer
	GMatrix& weights() { return m_weights; }

	/// Returns the bias for the hidden end of this layer.
	double* bias() { return m_bias[0]; }

	/// Returns the bias for the hidden end of this layer.
	const double* bias() const { return m_bias[0]; }

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	double* net() { return m_bias[1]; }

	/// Returns the delta vector for the bias.
	double* biasDelta() { return m_bias[4]; }

	/// Returns the bias for the visible end of this layer.
	double* biasReverse() { return m_biasReverse[0]; }

	/// Returns the delta vector for the reverse bias.
	double* biasReverseDelta() { return m_biasReverse[4]; }

	/// Returns the net for the visible end of this layer.
	double* netReverse() { return m_biasReverse[1]; }

	/// Returns the activation for the visible end of this layer.
	double* activationReverse() { return m_biasReverse[2]; }

	/// Returns the error term for the visible end of this layer.
	double* errorReverse() { return m_biasReverse[3]; }

	/// Performs binomial resampling of the activation values on the output end of this layer.
	void resampleHidden(GRand& rand);

	/// Performs binomial resampling of the activation values on the input end of this layer.
	void resampleVisible(GRand& rand);

	/// Draws a sample observation from "iters" iterations of Gibbs sampling.
	/// The resulting sample is placed in activationReverse(), and the corresponding
	/// encoding will be in activation().
	void drawSample(GRand& rand, size_t iters);

	/// Returns the free energy of this layer.
	double freeEnergy(const double* pVisibleSample);

	/// Refines this layer by contrastive divergence.
	/// pVisibleSample should point to a vector of inputs that will be presented to this layer.
	void contrastiveDivergence(GRand& rand, const double* pVisibleSample, double learningRate, size_t gibbsSamples = 1);

	/// Adjusts weights such that values in the new range will result in the
	/// same behavior that previously resulted from values in the old range.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);
};



class GLayerConvolutional1D : public GNeuralNetLayer
{
protected:
	size_t m_inputSamples;
	size_t m_inputChannels;
	size_t m_outputSamples;
	size_t m_kernelsPerChannel;
	GMatrix m_kernels;
	GMatrix m_delta;
	GMatrix m_activation; // Row 0 is the activation. Row 1 is the net. Row 2 is the error.
	GMatrix m_bias; // Row 0 is the bias. Row 1 is the bias delta.
	GActivationFunction* m_pActivationFunction;

public:
using GNeuralNetLayer::feedIn;
using GNeuralNetLayer::updateWeights;
using GNeuralNetLayer::updateWeightsAndRestoreDroppedOnes;

	/// General-purpose constructor.
	/// For example, if you collect 19 samples from 3 sensors, then the total input size will be 57 (19*3=57).
	/// The three values collected at time 0 will come first, followed by the three values collected at
	/// time 1, and so forth. If kernelSize is 5, then the output will consist of 15 (19-5+1=15) samples.
	/// If kernelsPerChannel is 2, then there will be 6 (3*2=6) channels in the output, for a total of 90 (15*6=90)
	/// output values. The first six channel values will appear first in the output vector, followed by the next six,
	/// and so forth. (kernelSize must be <= inputSamples.)
	GLayerConvolutional1D(size_t inputSamples, size_t inputChannels, size_t kernelSize, size_t kernelsPerChannel, GActivationFunction* pActivationFunction = NULL);

	/// Deserializing constructor
	GLayerConvolutional1D(GDomNode* pNode);

	virtual ~GLayerConvolutional1D();

	/// Returns the type of this layer
	virtual const char* type() { return "conv1"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() { return m_inputSamples * m_inputChannels; }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() { return m_outputSamples * m_inputChannels * m_kernelsPerChannel; }

	/// Resizes this layer. If pRand is non-NULL, an exception is thrown.
	virtual void resize(size_t inputs, size_t outputs, GRand* pRand = NULL, double deviation = 0.03);

	/// Returns the activation values from the most recent call to feedForward().
	virtual double* activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual double* error() { return m_activation[2]; }

	/// Copies the bias vector into the net vector.
	virtual void copyBiasToNet();

	/// Feeds a portion of the inputs through the weights and updates the net.
	virtual void feedIn(const double* pIn, size_t inputStart, size_t inputCount);

	/// Applies the activation function to the net vector to compute the activation vector.
	virtual void activate();

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Throws an exception, because convolutional layers do not support dropConnect.
	virtual void dropConnect(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const double* pTarget);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart = 0);

	/// Updates the bias of this layer by gradient descent. (Assumes the error has already been computed and deactivated.)
	virtual void updateBias(double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases);

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases);

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t countWeights();

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector);

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector);

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start, size_t count);

	/// Clips each kernel weight (not including the bias) to fall between -max and max.
	virtual void maxNorm(double max);

	/// Throws an exception.
	virtual double unitIncomingWeightsL1Norm(size_t unit);

	/// Throws an exception.
	virtual double unitIncomingWeightsL2Norm(size_t unit);

	/// Throws an exception.
	virtual double unitOutgoingWeightsL1Norm(size_t input);

	/// Throws an exception.
	virtual double unitOutgoingWeightsL2Norm(size_t input);

	/// Throws an exception.
	virtual void scaleUnitIncomingWeights(size_t unit, double scalar);

	/// Throws an exception.
	virtual void scaleUnitOutgoingWeights(size_t input, double scalar);

	/// Throws an exception.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	double* net() { return m_activation[1]; }

	double* bias() { return m_bias[0]; }
	double* biasDelta() { return m_bias[0]; }
	GMatrix& kernels() { return m_kernels; }
};




class GLayerConvolutional2D : public GNeuralNetLayer
{
protected:
	size_t m_inputCols;
	size_t m_inputRows;
	size_t m_inputChannels;
	size_t m_outputCols;
	size_t m_outputRows;
	size_t m_kernelsPerChannel;
	size_t m_kernelCount;
	GMatrix m_kernels;
	GMatrix m_delta;
	GMatrix m_activation; // Row 0 is the activation. Row 1 is the net. Row 2 is the error.
	GMatrix m_bias; // Row 0 is the bias. Row 1 is the bias delta.
	GActivationFunction* m_pActivationFunction;

public:
using GNeuralNetLayer::feedIn;
using GNeuralNetLayer::updateWeights;
using GNeuralNetLayer::updateWeightsAndRestoreDroppedOnes;

	/// General-purpose constructor.
	/// For example, if your input is a 64x48 color (RGB) image, then inputCols will be 64, inputRows will be 48,
	/// and inputChannels will be 3. The total input size will be 9216 (64*48*3=9216).
	/// If kernelSize is 5, then the output will consist of 60 columns (64-5+1=60) and 44 rows (48-5+1=44).
	/// If kernelsPerChannel is 2, then there will be 6 (3*2=6) channels in the output, for a total of
	/// 15840 (60*44*6=15840) output values. (kernelSize must be <= inputSamples.)
	GLayerConvolutional2D(size_t inputCols, size_t inputRows, size_t inputChannels, size_t kernelSize, size_t kernelsPerChannel, GActivationFunction* pActivationFunction = NULL);

	/// Deserializing constructor
	GLayerConvolutional2D(GDomNode* pNode);

	virtual ~GLayerConvolutional2D();

	/// Returns the type of this layer
	virtual const char* type() { return "conv2"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() { return m_inputRows * m_inputCols * m_inputChannels; }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() { return m_outputRows * m_outputCols * m_inputChannels * m_kernelsPerChannel; }

	/// Resizes this layer. If pRand is non-NULL, an exception is thrown.
	virtual void resize(size_t inputs, size_t outputs, GRand* pRand = NULL, double deviation = 0.03);

	/// Returns the activation values from the most recent call to feedForward().
	virtual double* activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual double* error() { return m_activation[2]; }

	/// Copies the bias vector into the net vector.
	virtual void copyBiasToNet();

	/// Feeds a portion of the inputs through the weights and updates the net.
	virtual void feedIn(const double* pIn, size_t inputStart, size_t inputCount);

	/// Applies the activation function to the net vector to compute the activation vector.
	virtual void activate();

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Throws an exception, because convolutional layers do not support dropConnect.
	virtual void dropConnect(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const double* pTarget);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart = 0);

	/// Updates the bias of this layer by gradient descent. (Assumes the error has already been computed and deactivated.)
	virtual void updateBias(double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases);

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases);

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t countWeights();

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector);

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector);

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start, size_t count);

	/// Clips each kernel weight (not including the bias) to fall between -max and max.
	virtual void maxNorm(double max);

	/// Throws an exception.
	virtual double unitIncomingWeightsL1Norm(size_t unit);

	/// Throws an exception.
	virtual double unitIncomingWeightsL2Norm(size_t unit);

	/// Throws an exception.
	virtual double unitOutgoingWeightsL1Norm(size_t input);

	/// Throws an exception.
	virtual double unitOutgoingWeightsL2Norm(size_t input);

	/// Throws an exception.
	virtual void scaleUnitIncomingWeights(size_t unit, double scalar);

	/// Throws an exception.
	virtual void scaleUnitOutgoingWeights(size_t input, double scalar);

	/// Throws an exception.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	double* net() { return m_activation[1]; }

	double* bias() { return m_bias[0]; }
	double* biasDelta() { return m_bias[0]; }
	GMatrix& kernels() { return m_kernels; }
};




/// A feed-forward artificial neural network, or multi-layer perceptron.
class GNeuralNet : public GIncrementalLearner
{
protected:
	std::vector<GNeuralNetLayer*> m_layers;
	double m_learningRate;
	double m_momentum;
	double m_validationPortion;
	double m_minImprovement;
	size_t m_epochsPerValidationCheck;
	bool m_useInputBias;

public:
	GNeuralNet();

	/// Load from a text-format
	GNeuralNet(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GNeuralNet();

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Saves the model to a text file.
	virtual GDomNode* serialize(GDom* pDoc) const;
#endif // MIN_PREDICT

	/// Returns the number of layers in this neural network. These include the hidden
	/// layers and the output layer. (The input vector does not count as a layer.)
	size_t layerCount() const { return m_layers.size(); }

	/// Returns a reference to the specified layer.
	GNeuralNetLayer& layer(size_t n) { return *m_layers[n]; }

	/// Returns a reference to the last layer.
	GNeuralNetLayer& outputLayer() { return *m_layers[m_layers.size() - 1]; }

	/// Adds pLayer to the network at the specified position.
	/// (The default position is at the end in feed-forward order.)
	/// Takes ownership of pLayer.
	/// If the number of inputs and/or outputs do not align with the
	/// previous and/or next layers, then any layers with FLEXIBLE_SIZE inputs or
	/// FLEXIBLE_SIZE outputs will be resized to accomodate. If both layers have
	/// fixed sizes that do not align, then an exception will be thrown.
	void addLayer(GNeuralNetLayer* pLayer, size_t position = INVALID_INDEX);

	/// Drops the layer at the specified index. Returns a pointer to
	/// the layer. You are then responsible to delete it. (This doesn't
	/// resize the remaining layers to fit with each other, so the caller
	/// is responsible to repair any such issues before using the neural
	/// network again.)
	GNeuralNetLayer* releaseLayer(size_t index);

	/// Set the portion of the data that will be used for validation. If the
	/// value is 0, then all of the data is used for both training and validation.
	void setValidationPortion(double d) { m_validationPortion = d; }

	/// Counts the number of weights in the network. (This value is not cached, so
	/// you should cache it rather than frequently call this method.)
	size_t countWeights() const;
	size_t countWeights(size_t layer) const;

	/// Perturbs all weights in the network by a random normal offset with the
	/// specified deviation.
	void perturbAllWeights(double deviation);

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double max);

	/// Multiplies all weights in the network by the specified factor. This can be used
	/// to implement L2 regularization, which prevents weight saturation.
	void scaleWeights(double factor, bool scaleBiases = true, size_t startLayer = 0, size_t layerCount = INVALID_INDEX);

	/// Diminishes all weights in the network by the specified amount. This can be used
	/// to implemnet L1 regularization, which promotes sparse representations. That is,
	/// it makes many of the weights approach zero.
	void diminishWeights(double amount, bool regularizeBiases = true, size_t startLayer = 0, size_t layerCount = INVALID_INDEX);

	/// Just like scaleWeights, except it only scales the weights in one of the output units.
	void scaleWeightsSingleOutput(size_t output, double lambda);

	/// Contract all the weights in this network by the specified factor.
	void contractWeights(double factor, bool contractBiases);

	/// Adjust the magnitudes of the incoming and outgoing connections by amount beta,
	/// such that sum of the absolute values of the weights remains constant. If beta
	/// is 0, no bleeding will occur. If beta is 1, total bleeding will occur, such that
	/// the sum of the absolute values of the input and output weights are the same. Typically,
	/// only a small amount of bleeding is desirable, so values close to 0 (like 0.00001)
	/// are used.
	void bleedWeightsL1(double beta);

	/// Adjust the magnitudes of the incoming and outgoing connections by amount beta,
	/// such that sum-squared-weights remains constant. If beta
	/// is 0, no bleeding will occur. If beta is 1, total bleeding will occur, such that
	/// the sum of the squares of the input and output weights are the same. Typically,
	/// only a small amount of bleeding is desirable, so values close to 0 (like 0.00001)
	/// are used.
	void bleedWeightsL2(double beta);

	/// Returns the current learning rate
	double learningRate() const { return m_learningRate; }

	/// Set the learning rate
	void setLearningRate(double d) { m_learningRate = d; }

	/// Returns the current momentum value
	double momentum() const { return m_momentum; }

	/// Momentum has the effect of speeding convergence and helping
	/// the gradient descent algorithm move past some local minimums
	void setMomentum(double d) { m_momentum = d; }

	/// Returns the threshold ratio for improvement.
	double improvementThresh() { return m_minImprovement; }

	/// Specifies the threshold ratio for improvement that must be
	/// made since the last validation check for training to continue.
	/// (For example, if the mean squared error at the previous validation check
	/// was 50, and the mean squared error at the current validation check
	/// is 49, then training will stop if d is > 0.02.)
	void setImprovementThresh(double d) { m_minImprovement = d; }

	/// Returns the number of epochs to perform before the validation data
	/// is evaluated to see if training should stop.
	size_t windowSize() { return m_epochsPerValidationCheck; }

	/// Sets the number of epochs that will be performed before
	/// each time the network is tested again with the validation set
	/// to determine if we have a better best-set of weights, and
	/// whether or not it's achieved the termination condition yet.
	/// (An epochs is defined as a single pass through all rows in
	/// the training set.)
	void setWindowSize(size_t n) { m_epochsPerValidationCheck = n; }

#ifndef MIN_PREDICT
	/// See the comment for GIncrementalLearner::trainSparse
	/// Assumes all attributes are continuous.
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);
#endif // MIN_PREDICT

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Train the network until the termination condition is met.
	/// Returns the number of epochs required to train it.
	size_t trainWithValidation(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& validateFeatures, const GMatrix& validateLabels);

	/// Gets the internal training data set
	GMatrix* internalTraininGMatrix();

	/// Gets the internal validation data set
	GMatrix* internalValidationData();

	/// Sets all the weights from an array of doubles. The number of
	/// doubles in the array can be determined by calling countWeights().
	void setWeights(const double* pWeights);
	void setWeights(const double* pWeights, size_t layer);

	/// Copy the weights from pOther. It is assumed (but not checked) that
	/// pOther already has the same network structure as this neural network.
	/// This method is faster than copyStructure.
	void copyWeights(GNeuralNet* pOther);

	/// Makes this neural network into a deep copy of pOther, including layers, nodes, settings and weights.
	void copyStructure(GNeuralNet* pOther);

	/// Serializes the network weights into an array of doubles. The
	/// number of doubles in the array can be determined by calling
	/// countWeights().
	void weights(double* pOutWeights) const;
	void weights(double* pOutWeights, size_t layer) const;

	/// Evaluates a feature vector. (The results will be in the nodes of the output layer.)
	/// The maxLayers parameter can limit how far into the network values are propagated.
	void forwardProp(const double* pInputs, size_t maxLayers = INVALID_INDEX);

	/// This is the same as forwardProp, except it only propagates to a single output node.
	/// It returns the value that this node outputs. If bypassInputWeights is true, then
	/// pInputs is assumed to have the same size as the first layer, and it is fed into the
	/// net of this layer, instead of the inputs.
	double forwardPropSingleOutput(const double* pInputs, size_t output);

	/// This method assumes forwardProp has been called. It copies the predicted vector into pOut.
	void copyPrediction(double* pOut);

	/// This method assumes forwardProp has been called. It computes the sum squared prediction error
	/// with the specified target vector.
	double sumSquaredPredictionError(const double* pTarget);

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data. That is, this method will add a good number of hidden
	/// layers, pick a good momentum value, etc.
	void autoTune(GMatrix& features, GMatrix& labels);

	/// Specify whether to use an input bias. (The default is false.) This feature is
	/// used with generative-backpropagation, which adjusts inputs to create latent features.
	void setUseInputBias(bool b) { m_useInputBias = b; }

	/// Returns whether this neural network utilizes an input bias.
	bool useInputBias() const { return m_useInputBias; }

	/// Inverts the weights of the specified node, and adjusts the weights in
	/// the next layer (if there is one) such that this will have no effect
	/// on the output of the network.
	/// (Assumes this model is already trained.)
	void invertNode(size_t layer, size_t node);

	/// Swaps two nodes in the specified layer. If layer specifies one of the hidden
	/// layers, then this will have no net effect on the output of the network.
	/// (Assumes this model is already trained.)
	void swapNodes(size_t layer, size_t a, size_t b);

	/// Swaps nodes in hidden layers of this neural network to align with those in
	/// that neural network, as determined using bipartite matching. (This might
	/// be done, for example, before averaging weights together.)
	void align(const GNeuralNet& that);

	/// Prints weights in a human-readable format
	void printWeights(std::ostream& stream);

	/// Performs principal component analysis (without reducing dimensionality) on the features to shift the
	/// variance of the data to the first few columns. Adjusts the weights on the input layer accordingly,
	/// such that the network output remains the same. Returns the transformed feature matrix.
	GMatrix* compressFeatures(GMatrix& features);

	/// This method assumes that the error term is already set at every unit in the output layer. It uses back-propagation
	/// to compute the error term at every hidden unit. (It does not update any weights.)
	void backpropagate(const double* pTarget, size_t startLayer = INVALID_INDEX);

	/// Backpropagates error from a single output node over all of the hidden layers. (Assumes the error term is already set on
	/// the specified output node.)
	void backpropagateSingleOutput(size_t outputNode, double target, size_t startLayer = INVALID_INDEX);

	/// This method assumes that the error term is already set for every network unit (by a call to backpropagate). It adjusts weights to descend the
	/// gradient of the error surface with respect to the weights.
	void descendGradient(const double* pFeatures, double learningRate, double momentum);

	/// This method assumes that the error term has been set for a single output network unit, and all units that feed into
	/// it transitively (by a call to backpropagateSingleOutput). It adjusts weights to descend the gradient of the error surface with respect to the weights.
	void descendGradientSingleOutput(size_t outputNeuron, const double* pFeatures, double learningRate, double momentum);

	/// This method assumes that the error term is already set for every network unit. It calculates the gradient
	/// with respect to the inputs. That is, it points in the direction of changing inputs that makes the error bigger.
	/// (Note that this calculation depends on the weights, so be sure to call this method before you call descendGradient.
	/// Also, note that descendGradient depends on the input features, so be sure not to update them until after you call descendGradient.)
	void gradientOfInputs(double* pOutGradient);

	/// This method assumes that the error term is already set for every network unit. It calculates the gradient
	/// with respect to the inputs. That is, it points in the direction of changing inputs that makes the error bigger.
	/// This method assumes that error is computed for only one output neuron, which is specified.
	/// (Note that this calculation depends on the weights, so be sure to call this method before you call descendGradientSingleOutput.)
	/// Also, note that descendGradientSingleOutput depends on the input features, so be sure not to update them until after you call descendGradientSingleOutput.)
	void gradientOfInputsSingleOutput(size_t outputNeuron, double* pOutGradient);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const double* pIn, const double* pOut);

	/// Presents a pattern for training. Applies dropout to the activations of hidden layers.
	/// Note that when training with dropout is complete, you should call
	/// scaleWeights(1.0 - probOfDrop, false, 1) to compensate for the scaling effect
	/// dropout has on the weights.
	void trainIncrementalWithDropout(const double* pIn, const double* pOut, double probOfDrop);

	/// Presents a pattern for training. Applies dropConnect to the weights.
	/// Note that when training with dropConnect is complete, you should call
	/// scaleWeights(1.0 - probOfDrop, false) to compensate for the scaling effect
	/// dropConnect has on the weights.
	void trainIncrementalWithDropConnect(const double* pIn, const double* pOut, double probOfDrop);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const double* pIn, double* pOut);

	/// Pretrains the network using the method of stacked autoencoders.
	/// This method performs the following steps: 1- Start with the first
	/// layer. 2- Create an autoencoder using the current layer as the
	/// encoder and a temporary layer as the decoder. 3- Train the
	/// autoencoder with the features. 4- Discard the decoder. 5- Map the
	/// features through the encoder to obtain a set of features for
	/// training the next layer, and go to step 2 until all (or maxLayers)
	/// layers have been pretrained in this manner.
	void pretrainWithAutoencoders(const GMatrix& features, size_t maxLayers = INVALID_INDEX);

	/// Finds the column in the intrinsic matrix with the largest deviation, then
	/// centers the matrix at the origin and renormalizes so the largest deviation
	/// is 1. Also renormalizes the input layer so these changes will have no effect.
	void containIntrinsics(GMatrix& intrinsics);

#ifndef MIN_PREDICT
	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const double* pIn, GPrediction* pOut);
#endif // MIN_PREDICT

	/// Generate a neural network that is initialized with the Fourier transform
	/// to reconstruct the given time-series data. The number of rows in the given
	/// time-series data is expected to be a power of 2. The resulting neural network will
	/// accept one input, representing time. The outputs will match the number of columns
	/// in the given time-series data. The series is assumed to represent one period
	/// of time in a repeating cycle. The duration of this period is specified as the
	/// parameter, period. The returned network has already had
	/// beginIncrementalLearning called.
	static GNeuralNet* fourier(GMatrix& series, double period = 1.0);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedFeatureRange(double* pOutMin, double* pOutMax);

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedLabelRange(double* pOutMin, double* pOutMax);

protected:
	/// Measures the sum squared error against the specified dataset
	double validationSquaredError(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);
};


/// A helper class used by GNeuralNetPseudoInverse
class GNeuralNetInverseLayer
{
public:
	GActivationFunction* m_pActivationFunction;
	std::vector<double> m_unbias;
	GMatrix* m_pInverseWeights;

	GNeuralNetInverseLayer()
	: m_pActivationFunction(NULL)
	{
	}

	~GNeuralNetInverseLayer();
};

/// Approximates the inverse of a neural network. (This only
/// works well if the neural network is mostly invertible. For
/// example, if the neural network only deviates a little from
/// the identity function, then this will work well. With many
/// interesting problems, this gives very poor results.)
/// Note: This class assumes that the activation functions used
/// within each layer of the neural network are homogeneous.
class GNeuralNetPseudoInverse
{
protected:
	double m_padding;
	std::vector<GNeuralNetInverseLayer*> m_layers;
	double* m_pBuf1;
	double* m_pBuf2;

public:
	/// padding specifies a margin in which label values will be clipped inside
	/// the activation function output range to avoid extreme feature values (-inf, inf, etc.).
	GNeuralNetPseudoInverse(GNeuralNet* pNN, double padding = 0.01);
	~GNeuralNetPseudoInverse();

	/// Computes the input features from the output labels. In cases of
	/// under-constraint, the feature vector with the minimum magnitude is chosen.
	/// In cases of over-constraint, the feature vector is chosen with a corresponding
	/// label vector that minimizes sum-squared error with the specified label
	/// vector.
	void computeFeatures(const double* pLabels, double* pFeatures);

#ifndef MIN_PREDICT
	static void test();
#endif // MIN_PREDICT
};

/// This model uses a randomely-initialized network to map the inputs into
/// a higher-dimensional space, and it uses a layer of perceptrons to learn
/// in this augmented space.
class GReservoirNet : public GIncrementalLearner
{
protected:
	GIncrementalLearner* m_pModel;
	GNeuralNet* m_pNN;
	double m_weightDeviation;
	size_t m_augments;
	size_t m_reservoirLayers;

public:
	/// General-purpose constructor
	GReservoirNet();

	/// Deserializing constructor
	GReservoirNet(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GReservoirNet();

#ifndef MIN_PREDICT
	static void test();
#endif // MIN_PREDICT

	/// Specify the deviation of the random weights in the reservoir
	void setWeightDeviation(double d) { m_weightDeviation = d; }

	/// Specify the number of additional attributes to augment the data with
	void setAugments(size_t n) { m_augments = n; }

	/// Specify the number of hidden layers in the reservoir
	void setReservoirLayers(size_t n) { m_reservoirLayers = n; }

#ifndef MIN_PREDICT
	/// Marshall this object to a DOM
	virtual GDomNode* serialize(GDom* pDoc) const;
#endif // MIN_PREDICT

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const double* pIn, GPrediction* pOut);

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const double* pIn, const double* pOut);

	/// See the comment for GIncrementalLearner::trainSparse
	/// Assumes all attributes are continuous.
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedFeatureRange(double* pOutMin, double* pOutMax);

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedLabelRange(double* pOutMin, double* pOutMax);
};


} // namespace GClasses

#endif // __GNEURALNET_H__

