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
#include "GOptimizer.h"
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
	double m_validationPortion;
	double m_minImprovement;
	size_t m_epochsPerValidationCheck;
	bool m_ready;

public:
	GNeuralNet();
	GNeuralNet(const GDomNode* pNode);
	virtual ~GNeuralNet();

	/// Unused supervised learner methods.
	/// \todo These should probably be removed from supervised learner.
	virtual void trainIncremental(const GVec &in, const GVec &out) { throw Ex("Use GDifferentiableOptimizer instead!"); }
	virtual void trainSparse(GSparseMatrix &features, GMatrix &labels) { throw Ex("Use GDifferentiableOptimizer instead!"); }

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
	/// The factor for L2 regularization should be less than 1.0, but most likely somewhat close to 1.
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

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

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

	/// Serializes the network weights into an array of doubles. The
	/// number of doubles in the array can be determined by calling
	/// countWeights().
	void weights(double* pOutWeights) const;
	void weights(double* pOutWeights, size_t layer) const;

	/// Evaluates a feature vector. (The results will be in the nodes of the output layer.)
	/// The maxLayers parameter can limit how far into the network values are propagated.
	void forwardProp(const GVec& inputs, size_t maxLayers = INVALID_INDEX);

	/// This method assumes forwardProp has been called. It copies the predicted vector into pOut.
	void copyPrediction(GVec& out);

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

	/// Backpropagates given the blame terms for the output layer.
	/// (Blame is the partial derivative of the objective function with respect to the prediction).
	void backpropagate(const GVec &blame);

	/// Backpropagate from a downstream layer
	void backpropagateFromLayer(GNeuralNetLayer* pDownstream);

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

	/// Convenience method for adding a basic layer
	void addLayer(size_t outputSize)
	{
		addLayer(new GLayerClassic(FLEXIBLE_SIZE, outputSize));
	}

	/// Convenience method for adding multiple layers (base case)
	template <typename T>
	void addLayers(T layer)
	{
		addLayer(layer);
	}

	/// Convenience method for adding multiple layers
	template <typename T, typename ... Ts>
	void addLayers(T layer, Ts... layers)
	{
		addLayers(layer);
		addLayers(layers...);
	}

protected:
	/// Unused supervised learner method.
	/// \todo This should probably be removed from supervised learner.
	virtual void trainInner(const GMatrix& features, const GMatrix& labels) { throw Ex("Use GDifferentiableOptimizer instead!"); }

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);
};



/// A class that facilitates training a neural network with an arbitrary optimization algorithm
class GNeuralNetTargetFunction : public GTargetFunction
{
protected:
	GNeuralNet& m_nn;
	const GMatrix& m_features;
	const GMatrix& m_labels;

public:
	/// You should call GNeuralNet::beginIncrementalLearning before passing your neural network to this constructor.
	/// features and labels should be pre-filtered to contain only continuous values for the neural network.
	GNeuralNetTargetFunction(GNeuralNet& nn, const GMatrix& features, const GMatrix& labels)
	: GTargetFunction(nn.countWeights()), m_nn(nn), m_features(features), m_labels(labels)
	{
	}

	virtual ~GNeuralNetTargetFunction() {}

	/// Copies the neural network weights into the vector.
	virtual void initVector(GVec& pVector)
	{
		m_nn.weights(pVector.data());
	}

	/// Copies the vector into the neural network and measures sum-squared error.
	virtual double computeError(const GVec& pVector)
	{
		m_nn.setWeights(pVector.data());
		return m_nn.sumSquaredError(m_features, m_labels);
	}
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
