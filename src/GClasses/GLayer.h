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

#ifndef __GLAYER_H__
#define __GLAYER_H__

#include <vector>
#include "GMatrix.h"

namespace GClasses {

/// This value is used for the number of inputs or outputs of a neural net layer when
/// you do not wish to specify a fixed size. For example, it may be used for the inputs
/// of the first layer or the outputs of the last layer, because the training data will
/// provide these sizes. (In fact, those ends will be resized to fit the data whether or
/// not FLEXIBLE_SIZE is used.) FLEXIBLE_SIZE should probably not be used on an end that
/// will be connected to another end with FLEXIBLE_SIZE because then both ends will stay
/// at a size of zero, which will result in approximately baseline predictions.
#define FLEXIBLE_SIZE 0

class GActivationFunction;

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

	/// Updates the bias of this layer by gradient descent, but does not change any element by more than learningRate * max.
	virtual void updateBiasClipped(double learningRate, double max) = 0;

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum) = 0;

	/// Refines the weights by gradient descent.
	virtual void updateWeights(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double momentum)
	{
		updateWeights(pUpStreamLayer->activation(), inputStart, pUpStreamLayer->outputs(), learningRate, momentum);
	}

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent,
	/// but does not change any element by more than learningRate * max.
	virtual void updateWeightsClipped(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double max) = 0;

	/// Wraps the other updateWeightsClipped.
	virtual void updateWeightsClipped(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double max)
	{
		updateWeightsClipped(pUpStreamLayer->activation(), inputStart, pUpStreamLayer->outputs(), learningRate, max);
	}

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum) = 0;

	/// Refines the weights by gradient descent.
	virtual void updateWeightsAndRestoreDroppedOnes(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double momentum)
	{
		updateWeightsAndRestoreDroppedOnes(pUpStreamLayer->activation(), inputStart, pUpStreamLayer->outputs(), learningRate, momentum);
	}

	/// Zero out the weight and bias deltas.
	virtual void resetDeltas() = 0;

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate) = 0;

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateBias() = 0;

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateWeights(const double* pFeat) = 0;

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

	/// Refines the activation function by stochastic gradient descent
	virtual void refineActivationFunction(double learningRate) = 0;

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda) = 0;

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
using GNeuralNetLayer::updateWeightsClipped;
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

	/// Updates the bias of this layer by gradient descent, but does not change any element by more than learningRate * max.
	virtual void updateBiasClipped(double learningRate, double max);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent,
	/// but does not change any element by more than learningRate * max.
	virtual void updateWeightsClipped(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double max);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Zero out the weight and bias deltas.
	virtual void resetDeltas();

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateBias();

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateWeights(const double* pFeat);

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

	/// Refines the activation function by stochastic gradient descent
	virtual void refineActivationFunction(double learningRate);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

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
using GNeuralNetLayer::updateWeightsClipped;
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

	/// Updates the bias of this layer by gradient descent, but does not change any element by more than learningRate * max.
	virtual void updateBiasClipped(double learningRate, double max);

	/// Calls updateWeights for each component.
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent,
	/// but does not change any element by more than learningRate * max.
	virtual void updateWeightsClipped(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double max);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Zero out the weight and bias deltas.
	virtual void resetDeltas() { throw Ex("Sorry, not implemented yet"); }

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate) { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateBias() { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateWeights(const double* pFeat) { throw Ex("Sorry, not implemented yet"); }

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

	/// Refines the activation function by stochastic gradient descent
	virtual void refineActivationFunction(double learningRate);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

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
using GNeuralNetLayer::updateWeightsClipped;
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

	/// Updates the bias of this layer by gradient descent, but does not change any element by more than learningRate * max.
	virtual void updateBiasClipped(double learningRate, double max);

	/// Adjust weights that feed into this layer. (Assumes the error has already been deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent,
	/// but does not change any element by more than learningRate * max.
	virtual void updateWeightsClipped(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double max);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Zero out the weight and bias deltas.
	virtual void resetDeltas() { throw Ex("Sorry, not implemented yet"); }

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate) { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateBias() { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateWeights(const double* pFeat) { throw Ex("Sorry, not implemented yet"); }

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

	/// Refines the activation function by stochastic gradient descent
	virtual void refineActivationFunction(double learningRate);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

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
using GNeuralNetLayer::updateWeightsClipped;
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

	/// Updates the bias of this layer by gradient descent, but does not change any element by more than learningRate * max.
	virtual void updateBiasClipped(double learningRate, double max);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent,
	/// but does not change any element by more than learningRate * max.
	virtual void updateWeightsClipped(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double max);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Zero out the weight and bias deltas.
	virtual void resetDeltas() { throw Ex("Sorry, not implemented yet"); }

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate) { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateBias() { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateWeights(const double* pFeat) { throw Ex("Sorry, not implemented yet"); }

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

	/// Refines the activation function by stochastic gradient descent
	virtual void refineActivationFunction(double learningRate);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

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
using GNeuralNetLayer::updateWeightsClipped;
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

	/// Updates the bias of this layer by gradient descent, but does not change any element by more than learningRate * max.
	virtual void updateBiasClipped(double learningRate, double max);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent,
	/// but does not change any element by more than learningRate * max.
	virtual void updateWeightsClipped(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double max);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Zero out the weight and bias deltas.
	virtual void resetDeltas() { throw Ex("Sorry, not implemented yet"); }

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate) { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateBias() { throw Ex("Sorry, not implemented yet"); }

	/// Add to the delta buffer for batch updating.
	virtual void batchUpdateWeights(const double* pFeat) { throw Ex("Sorry, not implemented yet"); }

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

	/// Refines the activation function by stochastic gradient descent
	virtual void refineActivationFunction(double learningRate);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

	/// Throws an exception.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	double* net() { return m_activation[1]; }

	double* bias() { return m_bias[0]; }
	double* biasDelta() { return m_bias[0]; }
	GMatrix& kernels() { return m_kernels; }
};



} // namespace GClasses

#endif // __GLAYER_H__

