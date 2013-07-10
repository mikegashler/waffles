/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GNEURALNET_H__
#define __GNEURALNET_H__

#include "GLearner.h"
#include <vector>

namespace GClasses {

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
	GNeuralNetLayer(size_t inputs, size_t outputs, GActivationFunction* pActivationFunction = NULL);
	GNeuralNetLayer(GDomNode* pNode);
	~GNeuralNetLayer();

	GMatrix m_weights; // Each row is an upstream neuron. Each column is a downstream neuron.
	GMatrix m_bias; // Row 0 is the bias. Row 1 is the net. Row 2 is the activation.
	GActivationFunction* m_pActivationFunction;

	GDomNode* serialize(GDom* pDoc);
	void resize(size_t inputs, size_t outputs);
	void resizePreserve(size_t inputCount, size_t outputCount, GRand& rand);
	void resetWeights(GRand* pRand);
	size_t inputs() const { return m_weights.rows(); }
	size_t outputs() const { return m_weights.cols(); }
	double* bias() { return m_bias[0]; }
	const double* bias() const { return m_bias[0]; }
	double* net() { return m_bias[1]; }
	double* activation() { return m_bias[2]; }
	void feedForward(const double* pIn);
	void feedForwardWithInputBias(const double* pIn);
	void feedForwardToOneOutput(const double* pIn, size_t output, bool inputBias);
};


/// An internal class used by GBackProp
class GBackPropLayer
{
public:
	GMatrix m_delta;
	GMatrix m_blame;

	void resize(size_t inputs, size_t outputs);
	double* blame() { return m_blame[0]; }
	double* biasDelta() { return m_blame[1]; }
	double* slack() { return m_blame[2]; }
};

/// This class performs backpropagation on a neural network. (I made it a separate
/// class because it is only needed during training. There is no reason to waste
/// this space after training is complete, or if you choose to use a different
/// technique to train the neural network.)
class GBackProp
{
friend class GNeuralNet;
public:
	enum TargetFunction
	{
		squared_error, /// (default) best for regression
		cross_entropy, /// best for classification
		sign, /// uses the sign of the error, as in the perceptron training rule
		uniform, /// sets all blame values on the output units to 1.0.
	};

protected:
	GNeuralNet* m_pNN;
	std::vector<GBackPropLayer> m_layers;

public:
	/// This class will adjust the weights in pNN
	GBackProp(GNeuralNet* pNN);

	~GBackProp();

	/// Returns a layer (not a layer of the neural network, but a corresponding layer of values used for back-prop)
	GBackPropLayer& layer(size_t layer)
	{
		return m_layers[layer];
	}

	/// This method computes the error terms for each node in the output layer.
	/// It assumes that forwardProp has already been called.
	/// After calling this method, it is typical to call backpropagate(), to compute the error on
	/// the hidden nodes, and then to call backProp()->descendGradient to update
	/// the weights. pTarget contains the target values for the output nodes.
	void computeBlame(const double* pTarget, size_t layer = (size_t)-1, TargetFunction eTargetFunction = squared_error);

	/// This is the same as computeBlame, except that it only sets
	/// the error on a single output node.
	void computeBlameSingleOutput(double target, size_t output, size_t layer = (size_t)-1, TargetFunction eTargetFunction = squared_error);

	/// Backpropagates the error from the downstream layer to the upstream layer.
	static void backPropLayer(GNeuralNetLayer* pNNDownStreamLayer, GNeuralNetLayer* pNNUpStreamLayer, GBackPropLayer* pBPDownStreamLayer, GBackPropLayer* pBPUpStreamLayer);

	/// Backpropagates the error from a single output node to a hidden layer.
	void backPropFromSingleNode(size_t outputNode, GNeuralNetLayer* pNNDownStreamLayer, GNeuralNetLayer* pNNUpStreamLayer, GBackPropLayer* pBPDownStreamLayer, GBackPropLayer* pBPUpStreamLayer);

	/// This method assumes that the error term is already set at every unit in the output layer. It uses back-propagation
	/// to compute the error term at every hidden unit. (It does not update any weights.)
	void backpropagate(size_t startLayer = (size_t)-1);

	/// Backpropagates error from a single output node over all of the hidden layers. (Assumes the error term is already set on
	/// the specified output node.)
	void backpropagateSingleOutput(size_t outputNode, size_t startLayer = (size_t)-1);

	/// This method assumes that the error term is already set for every network unit (by a call to backpropagate). It adjusts weights to descend the
	/// gradient of the error surface with respect to the weights.
	void descendGradient(const double* pFeatures, double learningRate, double momentum);

	/// This method assumes that the error term has been set for a single output network unit, and all units that feed into
	/// it transitively (by a call to backpropagateSingleOutput). It adjusts weights to descend the gradient of the error surface with respect to the weights.
	void descendGradientSingleOutput(size_t outputNeuron, const double* pFeatures, double learningRate, double momentum);

	/// This method assumes that the error term is already set for every network unit (by a call to backpropagate). It adjusts weights on the specified layer
	/// to descend the gradient of the error surface with respect to the weights.
	/// Returns the sum-squared delta.
	void descendGradientOneLayer(size_t layer, const double* pFeatures, double learningRate, double momentum);

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

protected:
	/// Adjust weights in pNNDownStreamLayer. (The error for pNNDownStreamLayer layer must have already been computed.)
	static void adjustWeights(GNeuralNetLayer* pNNDownStreamLayer, const double* pUpStreamActivation, GBackPropLayer* pBPDownStreamLayer, double learningRate, double momentum);

	/// Adjust the weights of a single neuron that follows a hidden layer. (Assumes the error of this neuron has already been computed).
	static void adjustWeightsSingleNeuron(size_t outputNode, GNeuralNetLayer* pNNDownStreamLayer, const double* pUpStreamActivation, GBackPropLayer* pBPDownStreamLayer, double learningRate, double momentum);

	/// Adjust the weights in a manner that uses Lagrange multipliers to regularize the weights. (Experimental.)
	void adjustWeightsLagrange(GNeuralNetLayer* pNNDownStreamLayer, const double* pUpStreamActivation, GBackPropLayer* pBPDownStreamLayer, LagrangeVals& lv);
};



/// An artificial neural network
class GNeuralNet : public GIncrementalLearner
{
friend class GBackProp;
protected:
	std::vector<size_t> m_topology;
	std::vector<GNeuralNetLayer*> m_layers;
	GBackProp* m_pBackProp;
	double m_learningRate;
	double m_momentum;
	double m_validationPortion;
	double m_minImprovement;
	size_t m_epochsPerValidationCheck;
	GBackProp::TargetFunction m_backPropTargetFunction;
	bool m_useInputBias;

public:
	GNeuralNet(GRand& rand);

	/// Load from a text-format
	GNeuralNet(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GNeuralNet();

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Saves the model to a text file.
	virtual GDomNode* serialize(GDom* pDoc) const;
#endif // MIN_PREDICT

	/// Specify the number of nodes in each hidden layer in feed-forward order. For example,
	/// if topology contains the values [3,7], then the network will have two hidden layers.
	/// The first hidden layer (in feed-forward order) will have 3 nodes. The next hidden
	/// layer will have 7 nodes. (The output layer will be automatically added with the
	/// number of nodes to match the columns in the training labels.)
	void setTopology(const std::vector<size_t>& topology) { m_topology = topology; }

	/// Returns the number of layers in this neural network. These include the hidden
	/// layers and the output layer. (The input vector does not count as a layer.)
	size_t layerCount() const { return m_layers.size(); }

	/// Returns a reference to the specified layer.
	GNeuralNetLayer* getLayer(size_t n) { return m_layers[n]; }

	/// Adds a new node at the end of the specified layer. (The new node is initialized
	/// with small weights, so this operation should initially have little impact on
	/// predictions.)
	void addNode(size_t layer);

	/// Removes the specified node from the specified layer. (An exception will be thrown
	/// the layer only has one node.)
	void dropNode(size_t layer, size_t node);

	/// Returns the backprop object associated with this neural net (if there is one)
	GBackProp* backProp() { return m_pBackProp; }

	/// Set the portion of the data that will be used for validation. If the
	/// value is 0, then all of the data is used for both training and validation.
	void setValidationPortion(double d) { m_validationPortion = d; }

	/// Counts the number of weights in the network. (This value is not cached, so
	/// you should cache it rather than frequently call this method.)
	size_t countWeights() const;

	/// Perturbs all weights in the network by a random normal offset with the
	/// specified deviation.
	void perturbAllWeights(double deviation);

	/// Clips all non-bias weights to fall within the range [-max, max].
	void clipWeights(double max);

	/// Multiplies all weights in the network by the specified factor.
	void scaleWeights(double factor);

	/// Just like scaleWeights, except it only scales the weights in one of the output units.
	void scaleWeightsSingleOutput(size_t output, double lambda);

	/// Multiplies all weights (including biases) in the specified layer by "factor".
	void scaleWeightsOneLayer(double factor, size_t lay);

	/// Adjust the magnitudes of the incoming and outgoing connections by amount alpha,
	/// such that sum-squared magnitude remains constant. A small value for alpha, such as
	/// 0.0001, will bring the magnitudes closer together by a small amount (so the bigger
	/// one will be scaled down, and the smaller one will be scaled up).
	void bleedWeights(double alpha);

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

	/// Specify the target function to use for back-propagation. The default is squared_error.
	/// cross_entropy tends to be faster, and is well-suited for classification tasks.
	void setBackPropTargetFunction(GBackProp::TargetFunction eTF) { m_backPropTargetFunction = eTF; }

	/// Returns the enumeration of the target function used for backpropagation
	GBackProp::TargetFunction backPropTargetFunction() { return m_backPropTargetFunction; }

#ifndef MIN_PREDICT
	/// See the comment for GIncrementalLearner::trainSparse
	/// Assumes all attributes are continuous.
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);
#endif // MIN_PREDICT

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Train the network until the termination condition is met.
	/// Returns the number of epochs required to train it.
	size_t trainWithValidation(GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& validateFeatures, GMatrix& validateLabels);

	/// Some extra junk is allocated when training to make it efficient.
	/// This method is called when training is done to get rid of that
	/// extra junk.
	void releaseTrainingJunk();

	/// Gets the internal training data set
	GMatrix* internalTraininGMatrix();

	/// Gets the internal validation data set
	GMatrix* internalValidationData();

	/// Sets all the weights from an array of doubles. The number of
	/// doubles in the array can be determined by calling countWeights().
	void setWeights(const double* pWeights);

	/// Copy the weights from pOther. It is assumed (but not checked) that
	/// pOther has the same network structure as this neural network.
	void copyWeights(GNeuralNet* pOther);

	/// Copies the layers, nodes, and settings from pOther (but not the
	/// weights). beginIncrementalLearning must have been called on pOther
	/// so that it has a complete structure.
	void copyStructure(GNeuralNet* pOther);

	/// Serializes the network weights into an array of doubles. The
	/// number of doubles in the array can be determined by calling
	/// countWeights().
	void weights(double* pOutWeights) const;

	/// Evaluates a feature vector. (The results will be in the nodes of the output layer.)
	/// The maxLayers parameter can limit how far into the network values are propagated.
	void forwardProp(const double* pInputs, size_t maxLayers = (size_t)-1);

	/// This is the same as forwardProp, except it only propagates to a single output node.
	/// It returns the value that this node outputs.
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

	/// Returns true iff train or beginIncrementalTraining has been called.
	bool hasTrainingBegun() const { return m_layers.size() > 0; }

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

	/// Adjusts weights on the first layer such that new inputs will be expected to fall in
	/// the new range instead of the old range.
	void normalizeInput(size_t index, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);

	/// Inserts a new layer at the specified position.
	/// Its weights will be initialized in a manner that loosely approximates the identity function
	/// with some random perturbation, and without changing any weights in other layers. (Note that
	/// a better approximation for adding a new layer with no net effect on the overall behavior of
	/// the network could be implemented if the weights in other layers were also adjusted, but that
	/// might exacerbate weight saturation.)
	/// The current implementation makes the unnecessary assumptions that the logistic function is used
	/// as the activation function in all layers.
	void insertLayer(size_t position, size_t nodeCount);

protected:
#ifndef MIN_PREDICT
	/// A helper method used by serialize.
	GDomNode* serializeInner(GDom* pDoc, const char* szClassName) const;
#endif // MIN_PREDICT

	/// Measures the sum squared error against the specified dataset
	double validationSquaredError(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

#ifndef MIN_PREDICT
	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);
#endif // MIN_PREDICT

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

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(sp_relation& pFeatureRel, sp_relation& pLabelRel);

	/// See the comment for GIncrementalLearner::trainIncrementalInner
	virtual void trainIncrementalInner(const double* pIn, const double* pOut);
};


/// A helper class used by GNeuralNetPseudoInverse
class GNeuralNetInverseLayer
{
public:
	GActivationFunction* m_pActivationFunction;
	std::vector<double> m_unbias;
	GMatrix* m_pInverseWeights;

	~GNeuralNetInverseLayer()
	{
		delete(m_pInverseWeights);
	}
};

/// Computes the pseudo-inverse of a neural network.
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
class GReservoirNet : public GNeuralNet
{
protected:
	double m_weightDeviation;
	size_t m_augments;
	size_t m_reservoirLayers;

public:
	/// General-purpose constructor
	GReservoirNet(GRand& rand);

	/// Deserializing constructor
	GReservoirNet(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GReservoirNet() {}

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

	/// See the comment for GSupervisedLearner::clearFeatureFilter.
	virtual void clearFeatureFilter();
#endif // MIN_PREDICT
};


} // namespace GClasses

#endif // __GNEURALNET_H__

