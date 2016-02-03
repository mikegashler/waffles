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

#include "GLayer.h"
#include "GLearner.h"
#include "GVec.h"
#include <vector>

namespace GClasses {

class GNeuralNetLayer;
class GActivationFunction;


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

public:
	GNeuralNet();

	/// Load from a text-format
	GNeuralNet(const GDomNode* pNode);

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

	/// Perturbs all weights in the network by a random normal offset with the
	/// specified deviation.
	void perturbAllWeights(double deviation);

	/// Scales weights if necessary such that the magnitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max, bool outputLayer = false);

	/// Multiplies all weights in the network by the specified factor. This can be used
	/// to implement L2 regularization, which prevents weight saturation.
	void scaleWeights(double factor, bool scaleBiases = true, size_t startLayer = 0, size_t layerCount = INVALID_INDEX);

	/// Diminishes all weights in the network by the specified amount. This can be used
	/// to implemnet L1 regularization, which promotes sparse representations. That is,
	/// it makes many of the weights approach zero.
	void diminishWeights(double amount, bool regularizeBiases = true, size_t startLayer = 0, size_t layerCount = INVALID_INDEX);

	/// Just like scaleWeights, except it only scales the weights in one of the output units.
	void scaleWeightsSingleOutput(size_t output, double lambda);

	/// Regularizes all the activation functions
	void regularizeActivationFunctions(double lambda);

	/// Contract all the weights in this network by the specified factor.
	void contractWeights(double factor, bool contractBiases);

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
	void copyWeights(const GNeuralNet* pOther);

	/// Makes this neural network into a deep copy of pOther, including layers, nodes, settings and weights.
	void copyStructure(const GNeuralNet* pOther);

	/// Copy the errors from pOther into this neural network.
	void copyErrors(const GNeuralNet* pOther);

	/// Serializes the network weights into an array of doubles. The
	/// number of doubles in the array can be determined by calling
	/// countWeights().
	void weights(double* pOutWeights) const;
	void weights(double* pOutWeights, size_t layer) const;

	/// Evaluates a feature vector. (The results will be in the nodes of the output layer.)
	/// The maxLayers parameter can limit how far into the network values are propagated.
	void forwardProp(const GVec& inputs, size_t maxLayers = INVALID_INDEX);

	/// This is the same as forwardProp, except it only propagates to a single output node.
	/// It returns the value that this node outputs. If bypassInputWeights is true, then
	/// pInputs is assumed to have the same size as the first layer, and it is fed into the
	/// net of this layer, instead of the inputs.
	double forwardPropSingleOutput(const GVec& inputs, size_t output);

	/// This method assumes forwardProp has been called. It copies the predicted vector into pOut.
	void copyPrediction(GVec& out);

	/// This method assumes forwardProp has been called. It computes the sum squared prediction error
	/// with the specified target vector.
	double sumSquaredPredictionError(const GVec& target);

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data. That is, this method will add a good number of hidden
	/// layers, pick a good momentum value, etc.
	void autoTune(GMatrix& features, GMatrix& labels);

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

	/// Prints a summary of this neural network. Assumes all layers are GLayerClassic.
	void printSummary(std::ostream& stream);

	/// Performs principal component analysis (without reducing dimensionality) on the features to shift the
	/// variance of the data to the first few columns. Adjusts the weights on the input layer accordingly,
	/// such that the network output remains the same. Returns the transformed feature matrix.
	GMatrix* compressFeatures(GMatrix& features);

	/// Backpropagates, assuming the error has already been computed for the output layer
	void backpropagateErrorAlreadySet();

	/// This method assumes that the error term is already set at every unit in the output layer. It uses back-propagation
	/// to compute the error term at every hidden unit. (It does not update any weights.)
	void backpropagate(const GVec& target, size_t startLayer = INVALID_INDEX);

	/// Backpropagates, and adjusts weights to keep errors from diminishing or exploding
	double backpropagateAndNormalizeErrors(const GVec& target, double alpha);

	/// Backpropagate from a downstream layer
	void backpropagateFromLayer(GNeuralNetLayer* pDownstream);

	/// Backpropagates from a layer, and adjusts weights to keep errors from diminishing or exploding
	void backpropagateFromLayerAndNormalizeErrors(GNeuralNetLayer* pDownstream, double errMag, double alpha);

	/// Backpropagates error from a single output node over all of the hidden layers. (Assumes the error term is already set on
	/// the specified output node.)
	void backpropagateSingleOutput(size_t outputNode, double target, size_t startLayer = INVALID_INDEX);

	/// This method assumes that the error term is already set for every network unit (by a call to backpropagate). It adjusts weights to descend the
	/// gradient of the error surface with respect to the weights.
	void descendGradient(const GVec& features, double learningRate, double momentum);

	/// This method assumes that the error term has been set for a single output network unit, and all units that feed into
	/// it transitively (by a call to backpropagateSingleOutput). It adjusts weights to descend the gradient of the error surface with respect to the weights.
	void descendGradientSingleOutput(size_t outputNeuron, const GVec& features, double learningRate, double momentum);

	/// Update the delta buffer in each layer with the gradient for a single pattern presentation.
	void updateDeltas(const GVec& features, double momentum);

	/// Tell each layer to apply its deltas. (That is, take a step in the direction specified in the delta buffer.)
	void applyDeltas(double learningRate);

	/// This method assumes that the error term is already set for every network unit. It calculates the gradient
	/// with respect to the inputs. That is, it points in the direction of changing inputs that makes the error bigger.
	/// (Note that this calculation depends on the weights, so be sure to call this method before you call descendGradient.
	/// Also, note that descendGradient depends on the input features, so be sure not to update them until after you call descendGradient.)
	void gradientOfInputs(GVec& outGradient);

	/// This method assumes that the error term is already set for every network unit. It calculates the gradient
	/// with respect to the inputs. That is, it points in the direction of changing inputs that makes the error bigger.
	/// This method assumes that error is computed for only one output neuron, which is specified.
	/// (Note that this calculation depends on the weights, so be sure to call this method before you call descendGradientSingleOutput.)
	/// Also, note that descendGradientSingleOutput depends on the input features, so be sure not to update them until after you call descendGradientSingleOutput.)
	void gradientOfInputsSingleOutput(size_t outputNeuron, GVec& outGradient);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const GVec& in, const GVec& out);

	/// Performs a single step of batch gradient descent.
	void trainIncrementalBatch(const GMatrix& features, const GMatrix& labels);

	/// Presents a pattern for training. Applies dropout to the activations of hidden layers.
	/// Note that when training with dropout is complete, you should call
	/// scaleWeights(1.0 - probOfDrop, false, 1) to compensate for the scaling effect
	/// dropout has on the weights.
	void trainIncrementalWithDropout(const GVec& in, const GVec& out, double probOfDrop);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

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
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);
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



/// Trains an Elman-style recurrent neural network.
class GBackPropThroughTime
{
protected:
	GNeuralNet& m_transition;
	GNeuralNet& m_observation;
	size_t m_unfoldDepth;
	double m_unfoldReciprocal;
	size_t m_obsParamCount;
	std::vector<GNeuralNet*> m_parts;
	double* m_buf;
	double m_errorNormalizationTerm;

public:
	/// The purpose of this class is to train "transition" and "observation".
	/// transition provides the layers in the recurrent portion of the Elman-style network.
	/// observation provides the layers that follow the recurrent portion of the Elman-style network.
	/// k is the number of times the transition portion will occur in the unfolded network.
	/// It is assumed that transition.beginIncrementalLearning and observation.beginIncrementalLearning
	/// have already been called before they were passed to this class.
	GBackPropThroughTime(GNeuralNet& transition, GNeuralNet& observation, size_t unfoldDepth);
	~GBackPropThroughTime();

	void setErrorNormalizationTerm(double d)
	{
		m_errorNormalizationTerm = d;
	}

	/// initialState is the initial values for the first part of the input into the transition function.
	/// controls should provide a control vector (concatenated after the state) for each unfolding of the transition function.
	/// obsParams provides any additional values fed into the observation function. In most cases, it will be NULL.
	/// targetObs provides the target outputs, or labels.
	void trainIncremental(const GVec& initialState, const GMatrix& controls, const GVec& obsParams, const GVec& targetObs);
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
	GReservoirNet(const GDomNode* pNode, GLearnerLoader& ll);

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
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const GVec& in, const GVec& out);

	/// See the comment for GIncrementalLearner::trainSparse
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

