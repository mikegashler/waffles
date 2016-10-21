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
#include <ostream>

namespace GClasses {

/// This value is used for the number of inputs or outputs of a neural net layer when
/// you do not wish to specify a fixed size. For example, it may be used for the inputs
/// of the first layer or the outputs of the last layer, because the training data will
/// provide these sizes. (In fact, those ends will be resized to fit the data whether or
/// not FLEXIBLE_SIZE is used.) FLEXIBLE_SIZE should probably not be used on an end that
/// will be connected to another end with FLEXIBLE_SIZE because then both ends will stay
/// at a size of zero, which will result in approximately baseline predictions.
#define FLEXIBLE_SIZE (size_t)0

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

	/// Makes a string representation of this layer
	virtual std::string to_str() = 0;

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const = 0;

	/// Returns the number of values that this layer outputs.
	virtual size_t outputs() const = 0;

	/// Resizes this layer. If pRand is non-NULL, then it preserves existing weights when possible
	/// and initializes any others to small random values.
	virtual void resize(size_t inputs, size_t outputs) = 0;

	/// Resizes the inputs of this layer (as in the above function) given the upstream layer to calculate needed inputs.
	virtual void resizeInputs(GNeuralNetLayer* pUpStreamLayer)
	{
		resize(pUpStreamLayer->outputs(), outputs());
	}

	/// Returns a buffer where the activation from the most-recent call to feedForward is stored.
	virtual GVec& activation() = 0;

	/// Returns a buffer where the error terms for each unit are stored.
	virtual GVec& error() = 0;

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop) = 0;

	/// Feeds an input vector through this layer to compute the output of this layer.
	virtual void feedForward(const GVec& in) = 0;

	/// Feeds the activation of the previous layer through this layer to compute the output of this layer.
	virtual void feedForward(GNeuralNetLayer* pUpStreamLayer)
	{
		feedForward(pUpStreamLayer->activation());
	}

	/// Computes the error term of the activation.
	virtual void computeError(const GVec& target) = 0;

	/// Converts the error term to refer to the net input.
	virtual void deactivateError() = 0;

	/// Computes the activation error of the layer that feeds into this one.
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer) = 0;

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum) = 0;

	/// Updates the deltas for updating the weights by gradient descent.
	virtual void updateDeltas(GNeuralNetLayer* pUpStreamLayer, double momentum)
	{
		updateDeltas(pUpStreamLayer->activation(), momentum);
	}

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate) = 0;

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
	virtual void copyWeights(const GNeuralNetLayer* pSource) = 0;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) = 0;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise. The
	/// default values apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX) = 0;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max) = 0;

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda) = 0;

	/// Feeds a matrix through this layer, one row at-a-time, and returns the resulting transformed matrix.
	GMatrix* feedThrough(const GMatrix& data);

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
using GNeuralNetLayer::feedForward;
using GNeuralNetLayer::updateDeltas;

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

	/// Makes a string representation of this layer
	virtual std::string to_str();

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const { return m_weights.rows(); }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() const { return m_weights.cols(); }

	/// Resizes this layer. If pRand is non-NULL, then it preserves existing weights when possible
	/// and initializes any others to small random values.
	virtual void resize(size_t inputs, size_t outputs);

	/// Returns the activation values from the most recent call to feedForward().
	virtual GVec& activation() { return m_bias[2]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error() { return m_bias[3]; }

	/// Feeds a the inputs through this layer.
	virtual void feedForward(const GVec& in);

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

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
	virtual void copyWeights(const GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double min, double max);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

	/// Returns a reference to the weights matrix of this layer
	GMatrix& weights() { return m_weights; }
	const GMatrix& weights() const { return m_weights; }

	GMatrix& deltas() { return m_delta; }

	/// Returns the bias vector of this layer.
	GVec& bias() { return m_bias[0]; }

	/// Returns the bias vector of this layer.
	const GVec& bias() const { return m_bias[0]; }

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	GVec& net() { return m_bias[1]; }

	/// Returns a buffer used to store delta values for each bias in this layer.
	GVec& biasDelta() { return m_bias[4]; }

	/// Returns a vector used to specify slack terms for each unit in this layer.
	GVec& slack() { return m_bias[5]; }

	/// Returns a pointer to the activation function used in this layer
	GActivationFunction* activationFunction() { return m_pActivationFunction; }

	/// Feeds a vector forward through this layer to compute only the one specified output value.
	void feedForwardToOneOutput(const GVec& in, size_t output);

	/// This is the same as computeError, except that it only computes the error of a single unit.
	void computeErrorSingleOutput(double target, size_t output);

	/// Same as deactivateError, but only applies to a single unit in this layer.
	void deactivateErrorSingleOutput(size_t output);

	/// Backpropagates the error from a single output node to a hidden layer.
	/// (Assumes that the error in the output node has already been deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	void backPropErrorSingleOutput(size_t output, GVec& upStreamError);

	/// Updates the weights and bias of a single neuron. (Assumes the error has already been computed and deactivated.)
	void updateWeightsSingleNeuron(size_t outputNode, const GVec& upStreamActivation, double learningRate, double momentum);

	/// Sets the weights of this layer to make it weakly approximate the identity function.
	/// start specifies the first unit whose incoming weights will be adjusted.
	/// count specifies the maximum number of units whose incoming weights are adjusted.
	void setWeightsToIdentity(size_t start = 0, size_t count = (size_t)-1);

	/// Transforms the weights of this layer by the specified transformation matrix and offset vector.
	/// transform should be the pseudoinverse of the transform applied to the inputs. pOffset should
	/// be the negation of the offset added to the inputs after the transform, or the transformed offset
	/// that is added before the transform.
	void transformWeights(GMatrix& transform, const GVec& offset);

	/// Adjusts weights such that values in the new range will result in the
	/// same behavior that previously resulted from values in the old range.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);
	void copySingleNeuronWeights(size_t source, size_t dest);
	void printSummary(std::ostream& stream);
};



/// Multiplies each pair of values together to produce the output
class GLayerProductPooling : public GNeuralNetLayer
{
friend class GNeuralNet;
protected:
	GMatrix m_activation; // Row 0 is the activation. Row 1 is the error.

public:
using GNeuralNetLayer::feedForward;
using GNeuralNetLayer::updateDeltas;

	/// General-purpose constructor.
	GLayerProductPooling(size_t inputs);

	/// Deserializing constructor
	GLayerProductPooling(GDomNode* pNode);
	~GLayerProductPooling();

	/// Returns the type of this layer
	virtual const char* type() { return "productpooling"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Makes a string representation of this layer
	virtual std::string to_str();

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const { return m_activation.cols() * 2; }

	/// Returns the number of outputs that this layer produces.
	virtual size_t outputs() const { return m_activation.cols(); }

	/// Resizes this layer. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs);

	/// Resizes the inputs of this layer (as in the above function) given the upstream layer to calculate needed inputs.
	virtual void resizeInputs(GNeuralNetLayer* pUpStreamLayer)
	{
		resize(pUpStreamLayer->outputs(), pUpStreamLayer->outputs() / 2);
	}

	/// Returns the activation values from the most recent call to feedForward().
	virtual GVec& activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error() { return m_activation[1]; }

	/// Feeds a the inputs through this layer.
	virtual void feedForward(const GVec& in);

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

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
	virtual void copyWeights(const GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double min, double max);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);
};



class GLayerMaxOut : public GNeuralNetLayer
{
friend class GNeuralNet;
protected:
	GMatrix m_weights; // Each row is an upstream neuron. Each column is a downstream neuron.
	GMatrix m_bias; // Row 0 is the bias. Row 1 is the bias delta.
	GMatrix m_delta; // Used to implement momentum
	GMatrix m_activation; // Row 0 is the activation. Row 1 is the error.
	GIndexVec m_winners; // The indexes of the winning inputs.

public:
using GNeuralNetLayer::feedForward;
using GNeuralNetLayer::updateDeltas;

	/// General-purpose constructor. Takes ownership of pActivationFunction.
	/// If pActivationFunction is NULL, then GActivationTanH is used.
	GLayerMaxOut(size_t inputs, size_t outputs);

	/// Deserializing constructor
	GLayerMaxOut(GDomNode* pNode);
	~GLayerMaxOut();

	/// Returns the type of this layer
	virtual const char* type() { return "maxnet"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Makes a string representation of this layer
	virtual std::string to_str();

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const { return m_weights.rows(); }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() const { return m_weights.cols(); }

	/// Resizes this layer. If pRand is non-NULL, then it preserves existing weights when possible
	/// and initializes any others to small random values.
	virtual void resize(size_t inputs, size_t outputs);

	/// Returns the activation values from the most recent call to feedForward().
	virtual GVec& activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error() { return m_activation[1]; }

	/// Feeds a the inputs through this layer.
	virtual void feedForward(const GVec& in);

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

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
	virtual void copyWeights(const GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double min, double max);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

	/// Returns a reference to the weights matrix of this layer
	GMatrix& weights() { return m_weights; }
	const GMatrix& weights() const { return m_weights; }

	GMatrix& deltas() { return m_delta; }

	/// Returns the bias vector of this layer.
	GVec& bias() { return m_bias[0]; }

	/// Returns the bias vector of this layer.
	const GVec& bias() const { return m_bias[0]; }

	/// Returns a buffer used to store delta values for each bias in this layer.
	GVec& biasDelta() { return m_bias[1]; }

	/// Sets the weights of this layer to make it weakly approximate the identity function.
	/// start specifies the first unit whose incoming weights will be adjusted.
	/// count specifies the maximum number of units whose incoming weights are adjusted.
	void setWeightsToIdentity(size_t start = 0, size_t count = (size_t)-1);

	/// Transforms the weights of this layer by the specified transformation matrix and offset vector.
	/// transform should be the pseudoinverse of the transform applied to the inputs. pOffset should
	/// be the negation of the offset added to the inputs after the transform, or the transformed offset
	/// that is added before the transform.
	void transformWeights(GMatrix& transform, const GVec& offset);

	void copySingleNeuronWeights(size_t source, size_t dest);
	void printSummary(std::ostream& stream);
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
using GNeuralNetLayer::feedForward;
using GNeuralNetLayer::updateDeltas;

	/// General-purpose constructor. (You should call addComponent at least twice to mix some layers, after constructing this object.)
	GLayerMixed();

	/// Deserializing constructor
	GLayerMixed(GDomNode* pNode);
	~GLayerMixed();

	/// Returns the type of this layer
	virtual const char* type() { return "mixed"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Makes a string representation of this layer
	virtual std::string to_str();

	/// Adds another component of this layer. In other words, make this layer bigger by adding pComponent to it,
	/// as a peer beside the other components in this layer.
	void addComponent(GNeuralNetLayer* pComponent);

	/// Returns the specified component.
	GNeuralNetLayer& component(size_t i) { return *m_components[i]; }

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const;

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() const;

	/// Throws an exception if the specified dimensions would change anything. Also
	/// throws an exception if pRand is not NULL.
	virtual void resize(size_t inputs, size_t outputs);

	/// Returns the activation values from the most recent call to feedForward().
	virtual GVec& activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error() { return m_activation[1]; }

	/// Feeds the inputs through each component to compute an aggregated activation
	virtual void feedForward(const GVec& in);

	/// Calls dropOut for each component.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Copies the error vector into the corresponding buffer for each component,
	/// then calls deactivateError for each component.
	virtual void deactivateError();

	/// Calls backPropError for each component, and adds them up into the upstreams error buffer.
	/// (Note that the current implementation of this method may not be compatible with GPU-optimized layers.
	/// This method still needs to be audited for compatibility with such layers.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

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
	virtual void copyWeights(const GNeuralNetLayer* pSource);

	/// Calls resetWeights for each component.
	virtual void resetWeights(GRand& rand);

	/// Calls perturbWeights for each component.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Calls maxNorm for each component.
	virtual void maxNorm(double min, double max);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);
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
using GNeuralNetLayer::feedForward;
using GNeuralNetLayer::updateDeltas;

	/// General-purpose constructor. Takes ownership of pActivationFunction.
	GLayerRestrictedBoltzmannMachine(size_t inputs, size_t outputs, GActivationFunction* pActivationFunction = NULL);

	/// Deserializing constructor
	GLayerRestrictedBoltzmannMachine(GDomNode* pNode);

	~GLayerRestrictedBoltzmannMachine();

	/// Returns the type of this layer
	virtual const char* type() { return "rbm"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Makes a string representation of this layer
	virtual std::string to_str();

	/// Returns the number of visible units.
	virtual size_t inputs() const { return m_weights.cols(); }

	/// Returns the number of hidden units.
	virtual size_t outputs() const { return m_weights.rows(); }

	/// Resizes this layer. If pRand is non-NULL, then it preserves existing weights when possible
	/// and initializes any others to small random values.
	virtual void resize(size_t inputs, size_t outputs);

	/// Returns the activation values on the hidden end.
	virtual GVec& activation() { return m_bias[2]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error() { return m_bias[3]; }

	/// Feeds pIn forward through this layer.
	virtual void feedForward(const GVec& in);

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Feed a vector from the hidden end to the visible end. The results are placed in activationReverse();
	void feedBackward(const GVec& in);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

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
	virtual void copyWeights(const GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// Also perturbs the bias.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double min, double max);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

	/// Returns a reference to the weights matrix of this layer
	GMatrix& weights() { return m_weights; }

	/// Returns the bias for the hidden end of this layer.
	GVec& bias() { return m_bias[0]; }

	/// Returns the bias for the hidden end of this layer.
	const GVec& bias() const { return m_bias[0]; }

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	GVec& net() { return m_bias[1]; }

	/// Returns the delta vector for the bias.
	GVec& biasDelta() { return m_bias[4]; }

	/// Returns the bias for the visible end of this layer.
	GVec& biasReverse() { return m_biasReverse[0]; }

	/// Returns the delta vector for the reverse bias.
	GVec& biasReverseDelta() { return m_biasReverse[4]; }

	/// Returns the net for the visible end of this layer.
	GVec& netReverse() { return m_biasReverse[1]; }

	/// Returns the activation for the visible end of this layer.
	GVec& activationReverse() { return m_biasReverse[2]; }

	/// Returns the error term for the visible end of this layer.
	GVec& errorReverse() { return m_biasReverse[3]; }

	/// Performs binomial resampling of the activation values on the output end of this layer.
	void resampleHidden(GRand& rand);

	/// Performs binomial resampling of the activation values on the input end of this layer.
	void resampleVisible(GRand& rand);

	/// Draws a sample observation from "iters" iterations of Gibbs sampling.
	/// The resulting sample is placed in activationReverse(), and the corresponding
	/// encoding will be in activation().
	void drawSample(GRand& rand, size_t iters);

	/// Returns the free energy of this layer.
	double freeEnergy(const GVec& visibleSample);

	/// Refines this layer by contrastive divergence.
	/// pVisibleSample should point to a vector of inputs that will be presented to this layer.
	void contrastiveDivergence(GRand& rand, const GVec& visibleSample, double learningRate, size_t gibbsSamples = 1);
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
using GNeuralNetLayer::feedForward;
using GNeuralNetLayer::updateDeltas;

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
	virtual const char* type() { return "conv1d"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Makes a string representation of this layer
	virtual std::string to_str();

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const { return m_inputSamples * m_inputChannels; }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() const { return m_outputSamples * m_inputChannels * m_kernelsPerChannel; }

	/// Resizes this layer. If pRand is non-NULL, an exception is thrown.
	virtual void resize(size_t inputs, size_t outputs);

	/// Returns the activation values from the most recent call to feedForward().
	virtual GVec& activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error() { return m_activation[2]; }

	/// Feeds a the inputs through this layer.
	virtual void feedForward(const GVec& in);

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Throws an exception, because convolutional layers do not support dropConnect.
	virtual void dropConnect(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

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
	virtual void copyWeights(const GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start, size_t count);

	/// Clips each kernel weight (not including the bias) to fall between -max and max.
	virtual void maxNorm(double min, double max);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);

	/// Returns the net vector (that is, the values computed before the activation function was applied)
	/// from the most recent call to feedForward().
	GVec& net() { return m_activation[1]; }

	const GVec& bias() const { return m_bias[0]; }
	GVec& bias() { return m_bias[0]; }
	GVec& biasDelta() { return m_bias[1]; }
	const GMatrix& kernels() const { return m_kernels; }
	GMatrix& kernels() { return m_kernels; }
};




class GLayerConvolutional2D : public GNeuralNetLayer
{
protected:
	/// Image abstraction to facilitate convolution
	struct Image
	{
		static size_t npos;
		
		Image(GVec *data, size_t width, size_t height, size_t channels);
		size_t index(size_t x, size_t y, size_t z) const;
		double read(size_t x, size_t y, size_t z = 0) const;
		double &at(size_t x, size_t y, size_t z = 0);
		
		/// image data
		GVec *data;
		size_t width, height, channels;
		bool interlaced;
		
		/// viewport data
		mutable size_t dx, dy, dz;	///< offset
		mutable size_t px, py;		///< padding
		mutable size_t sx, sy;		///< stride
		mutable bool invertStride;	///< whether the stride should be inverted (i.e. sx or sy zeroes between each value)
		mutable bool flip;			///< whether to "flip" the image (i.e. 180 degree rotation)
	};
	
	/// Input dimensions
	size_t m_width, m_height, m_channels;
	
	/// Kernel dimensions (kernel channels = input channels)
	size_t m_kWidth, m_kHeight;
	
	/// Output dimensions (derived; output channels = kernel count)
	size_t m_outputWidth, m_outputHeight;
	
	/// Data
	GVec m_bias, m_biasDelta;
	GMatrix m_kernels, m_deltas;
	GMatrix m_activation; // Row 0 is the net. Row 1 is the activation. Row 2 is the error.
	GActivationFunction *m_pActivationFunction;
	
	/// Data as images
	Image m_kernelImage, m_deltaImage;
	Image m_inputImage, m_upStreamErrorImage;
	Image m_netImage, m_actImage, m_errImage;

private:
	/// Helper functions for convolution
	double filterSum(const Image &in, const Image &filter, size_t channels);
	void addScaled(const Image &in, double scalar, Image &out);
	void convolve(const Image &in, const Image &filter, Image &out, size_t channels = none);
	void convolveFull(const Image &in, const Image &filter, Image &out, size_t channels = none);
	void updateOutputSize();

public:
	static size_t none;
	
	/// General-purpose constructor.
	GLayerConvolutional2D(size_t width, size_t height, size_t channels, size_t kWidth, size_t kHeight, size_t kCount = 0, GActivationFunction *pActivationFunction = NULL);

	/// Constructor that will automatically use the upstream convolutional layer when added to a neural network
	GLayerConvolutional2D(size_t kWidth, size_t kHeight, size_t kCount = 0, GActivationFunction *pActivationFunction = NULL);

	GLayerConvolutional2D(GDomNode* pNode);
	virtual ~GLayerConvolutional2D();

	virtual const char *type() { return "conv2d"; }
	virtual GDomNode *serialize(GDom *pDoc);
	virtual std::string to_str();
	virtual size_t inputs() const { return m_width * m_height * m_channels; }
	virtual size_t outputs() const { return m_outputWidth * m_outputHeight * m_bias.size(); }
	virtual void resize(size_t inputs, size_t outputs);
	virtual void resizeInputs(GNeuralNetLayer *pUpStreamLayer);
	virtual GVec &activation() { return m_activation[1]; }
	virtual GVec &error() { return m_activation[2]; }

	virtual void feedForward(const GVec &in);
	virtual void dropOut(GRand &rand, double probOfDrop);
	virtual void dropConnect(GRand &rand, double probOfDrop);
	virtual void computeError(const GVec &target);
	virtual void deactivateError();
	virtual void backPropError(GNeuralNetLayer *pUpStreamLayer);
	virtual void updateDeltas(const GVec &upStreamActivation, double momentum);
	virtual void applyDeltas(double learningRate);
	virtual void scaleWeights(double factor, bool scaleBiases);
	virtual void diminishWeights(double amount, bool regularizeBiases);
	virtual size_t countWeights();
	virtual size_t weightsToVector(double *pOutVector);
	virtual size_t vectorToWeights(const double *pVector);
	virtual void copyWeights(const GNeuralNetLayer *pSource);
	virtual void resetWeights(GRand &rand);
	virtual void perturbWeights(GRand &rand, double deviation, size_t start = 0, size_t count = INVALID_INDEX);
	virtual void maxNorm(double min, double max);
	virtual void regularizeActivationFunction(double lambda);

	void setPadding(size_t px, size_t py = none);
	void setStride(size_t sx, size_t sy = none);
	void setInterlaced(bool interlaced);
	void setInputInterlaced(bool interlaced);
	void setKernelsInterlaced(bool interlaced);
	void setOutputInterlaced(bool interlaced);
	void addKernel();
	void addKernels(size_t n);

	size_t inputWidth() const { return m_width; }
	size_t inputHeight() const { return m_height; }
	size_t inputChannels() const { return m_channels; }

	size_t kernelWidth() const { return m_kWidth; }
	size_t kernelHeight() const { return m_kHeight; }
	size_t kernelChannels() const { return m_channels; }

	size_t outputWidth() const { return m_outputWidth; }
	size_t outputHeight() const { return m_outputHeight; }
	size_t outputChannels() const { return m_bias.size(); }
	
	size_t kernelCount() const { return m_kernels.rows(); }
	GVec &net() { return m_activation[0]; }
	const GMatrix &kernels() const { return m_kernels; }
	GMatrix &kernels() { return m_kernels; }
	const GVec &bias() const { return m_bias; }
	GVec &bias() { return m_bias; }
};



class GMaxPooling2D : public GNeuralNetLayer
{
protected:
	size_t m_inputCols;
	size_t m_inputRows;
	size_t m_inputChannels;
	size_t m_regionSize;
	GMatrix m_activation; // Row 0 is the activation. Row 1 is the error.

public:
using GNeuralNetLayer::feedForward;
using GNeuralNetLayer::updateDeltas;

	/// General-purpose constructor.
	/// For example, if your input is a 64x48 color (RGB) image, then inputCols will be 64, inputRows will be 48,
	/// and inputChannels will be 3. The total input size will be 9216 (64*48*3=9216).
	/// The values should be presented in the following order: c1x1y1, c2x1y1, c1x2y1, c2x2y1, c1x1y2, ...
	/// If kernelSize is 5, then the output will consist of 60 columns (64-5+1=60) and 44 rows (48-5+1=44).
	/// If kernelsPerChannel is 2, then there will be 6 (3*2=6) channels in the output, for a total of
	/// 15840 (60*44*6=15840) output values. (kernelSize must be <= inputSamples.)
	GMaxPooling2D(size_t inputCols, size_t inputRows, size_t inputChannels, size_t regionSize = 2);

	/// Deserializing constructor
	GMaxPooling2D(GDomNode* pNode);

	virtual ~GMaxPooling2D();

	/// Returns the type of this layer
	virtual const char* type() { return "maxpool2"; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc);

	/// Makes a string representation of this layer
	virtual std::string to_str();

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const { return m_inputRows * m_inputCols * m_inputChannels; }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() const { return m_inputRows * m_inputCols * m_inputChannels / (m_regionSize * m_regionSize); }

	/// Resizes this layer. If pRand is non-NULL, an exception is thrown.
	virtual void resize(size_t inputs, size_t outputs);

	/// Returns the activation values from the most recent call to feedForward().
	virtual GVec& activation() { return m_activation[0]; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error() { return m_activation[1]; }

	/// Feeds a the inputs through this layer.
	virtual void feedForward(const GVec& in);

	/// Randomly sets the activation of some units to 0.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Throws an exception, because convolutional layers do not support dropConnect.
	virtual void dropConnect(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	/// (Assumes the error for this layer has already been computed.)
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream layer's error vector.
	/// (Assumes that the error in this layer has already been computed and deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the deltas for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Add the weight and bias deltas to the weights.
	virtual void applyDeltas(double learningRate);

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
	virtual void copyWeights(const GNeuralNetLayer* pSource);

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand);

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	virtual void perturbWeights(GRand& rand, double deviation, size_t start, size_t count);

	/// Clips each kernel weight (not including the bias) to fall between -max and max.
	virtual void maxNorm(double min, double max);

	/// Regularizes the activation function
	virtual void regularizeActivationFunction(double lambda);
};



} // namespace GClasses

#endif // __GLAYER_H__
