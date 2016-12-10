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

#include "GBlock.h"
#include "GLearner.h"
#include "GOptimizer.h"
#include "GVec.h"
#include <vector>
#include "GDom.h"

namespace GClasses {

class GBlock;
class GContextLayer;
class GContextNeuralNet;
class GContextRecurrent;


/// GNeuralNet contains GLayers stacked upon each other.
/// GLayer contains GBlocks concatenated beside each other. (GNeuralNet is a type of GBlock.)
/// Each GBlock is an array of differentiable network units (artificial neurons).
/// The user must add at least one GBlock to each GLayer.
class GLayer
{
protected:
	size_t m_inputs, m_outputs, m_weightCount;
	std::vector<GBlock*> m_blocks;

public:
	GLayer();
	GLayer(GDomNode* pNode);
	virtual ~GLayer();

	/// Marshal this object into a dom node.
	GDomNode* serialize(GDom* pDoc) const;

	/// Allocates a new GContextLayer object, which can be used to train or predict with this layer.
	/// (Behavior is undefined if you add any blocks after you call newContext.)
	GContextLayer* newContext() const;

	/// Returns the number of blocks in this layer.
	size_t blockCount() const { return m_blocks.size(); }

	/// Returns a reference to the specified block.
	GBlock& block(size_t i) { return *m_blocks[i]; }
	const GBlock& block(size_t i) const { return *m_blocks[i]; }

	/// Adds a block of network units (artificial neurons) to this layer.
	/// inPos specifies the index of the first output from the previous layer that will feed into this block of units.
	void add(GBlock* pBlock, size_t inPos = 0);

	/// Recounts the number of inputs, outputs, and weights in this layer.
	void recount();

	/// Returns the number of inputs that this layer consumes.
	size_t inputs() const;

	/// Returns the number of outputs that this layer produces.
	size_t outputs() const;

	/// Resets the weights in all of the blocks in this layer
	void resetWeights(GRand& rand);

	/// Returns the total number of weights in this layer
	size_t weightCount() const;

	/// Marshals all the weights in this layer into a vector
	size_t weightsToVector(double* pOutVector) const;

	/// Unmarshals all the weights in this layer from a vector
	size_t vectorToWeights(const double *pVector);

	/// Copies the weights from pOther. (Assumes pOther has the same structure.)
	void copyWeights(const GLayer* pOther);

	/// Perturbs all the weights in this layer by adding Gaussian noise with the specified deviation.
	void perturbWeights(GRand& rand, double deviation);

	/// Clips the magnitude of the weight vector in each network unit to fall within the specified range.
	void maxNorm(double min, double max);

	/// Scales all the weights in this layer
	void scaleWeights(double factor, bool scaleBiases);

	/// Moves all weights by a constant amount toward 0
	void diminishWeights(double amount, bool diminishBiases);

	/// Take a step to descend the gradient by updating the weights.
	void step(double learningRate, const GVec &gradient);
};






/// The base class for the buffers that a thread needs to
/// use (train or predict with) a neural network component.
class GContext
{
public:
	GContext() {};
	virtual ~GContext() {}

	/// Resets the state of all recurrent blocks.
	/// (This is called whenever a recurrent neural network begins with a new sequence,
	/// either for training or testing.)
	virtual void resetState() = 0;
};






/// Contains the buffers that a thread needs to train or use a GLayer.
/// Each thread should use a separate GContextLayer object.
/// Call GLayer::newContext to obtain a new GContextLayer object.
class GContextLayer : public GContext
{
friend class GLayer;
public:
	const GLayer& m_layer;
	GVec m_activation;
	GVec m_blame;
	std::vector<GContextRecurrent*> m_recurrents;
	std::vector<GContextNeuralNet*> m_components;

protected:
	GContextLayer(const GLayer& layer); // deliberately protected. Call GLayer::newContext to construct one.

public:
	~GContextLayer();

	/// See the comment for GContext::resetState.
	virtual void resetState() override;

	/// Feeds input forward through the layer that was used to construct this object.
	void forwardProp(const GVec& input, GVec& output);

	/// Identical to forwardProp, except recurrent blocks additionally propagate through time during training.
	void forwardProp_training(const GVec& input, GVec& output);

	/// Backpropagates the blame through the layer that was used to construct this object.
	void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame);

	/// Updates the gradient for the layer that was used to construct this object.
	void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const;

};





/// GNeuralNet contains GLayers stacked upon each other.
/// GLayer contains GBlocks concatenated beside each other.
/// (GNeuralNet is a type of GBlock, so you can nest.)
/// Each GBlock is an array of differentiable network units (artificial neurons).
/// The user must add at least one GBlock to each GLayer.
class GNeuralNet : public GBlock
{
protected:
	size_t m_weightCount;
	std::vector<GLayer*> m_layers;

public:
	GNeuralNet();
	GNeuralNet(GDomNode* pNode);
	virtual ~GNeuralNet();

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_neuralnet; }

	/// Marshal this object into a dom node.
	GDomNode* serialize(GDom* pDoc) const override;

	/// Allocates a new GContextNeuralNet object, which can be used to train or predict with this neural net.
	/// (Behavior is undefined if you add or modify any layers after you call newContext.)
	GContextNeuralNet* newContext() const;

	/// Adds a new layer, and returns a reference to it.
	GLayer& newLayer();

	/// Returns the number of layers in this neural net.
	/// (Layers within neural networks embedded within this one are not counted.)
	size_t layerCount() const { return m_layers.size(); }

	/// Returns the specified layer.
	GLayer& layer(size_t i) { return *m_layers[i]; }
	const GLayer& layer(size_t i) const { return *m_layers[i]; }

	/// Returns a reference to the last layer.
	GLayer& outputLayer() { return *m_layers[m_layers.size() - 1]; }
	const GLayer& outputLayer() const { return *m_layers[m_layers.size() - 1]; }

	/// Returns a string representation of this object
	virtual std::string to_str() const override;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
	virtual size_t inputs() const override { return m_layers[0]->inputs(); }

	/// Returns the number of outputs this layer produces
	virtual size_t outputs() const override { return outputLayer().outputs(); }

	/// Take a step to descend the gradient by updating the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Recounts the number of weights.
	void recount();

	/// Returns the number of weights.
	virtual size_t weightCount() const override;

	/// Serializes the network weights into an array of doubles. The
	/// number of doubles in the array can be determined by calling
	/// weightCount().
	virtual size_t weightsToVector(double* pOutWeights) const override;

	/// Sets all the weights from an array of doubles. The number of
	/// doubles in the array can be determined by calling weightCount().
	virtual size_t vectorToWeights(const double* pWeights) override;

	/// Copy the weights from pOther. It is assumed (but not checked) that
	/// pOther already is a GNeuralNet with the same structure as this one.
	/// This method is faster than copyStructure.
	virtual void copyWeights(const GBlock* pOther) override;

	/// Makes this object into a deep copy of pOther, including layers, nodes, settings and weights.
	void copyStructure(const GNeuralNet* pOther);

	/// Initialize the weights, usually with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs all weights in the network by a random normal offset with the
	/// specified deviation.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Scales weights if necessary such that the magnitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max) override;

	/// Multiplies all weights in the network by the specified factor. This can be used
	/// to implement L2 regularization, which prevents weight saturation.
	/// The factor for L2 regularization should be less than 1.0, but most likely somewhat close to 1.
	virtual void scaleWeights(double factor, bool scaleBiases = true) override;

	/// Diminishes all weights in the network by the specified amount. This can be used
	/// to implemnet L1 regularization, which promotes sparse representations. That is,
	/// it makes many of the weights approach zero.
	virtual void diminishWeights(double amount, bool regularizeBiases = true) override;

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

	/// Measures the loss with respect to some data.
	/// Returns sum-squared error.
	double measureLoss(const GMatrix& features, const GMatrix& labels, double* pOutSAE = nullptr);

	/// Performs principal component analysis (without reducing dimensionality) on the features to shift the
	/// variance of the data to the first few columns. Adjusts the weights on the input layer accordingly,
	/// such that the network output remains the same. Returns the transformed feature matrix.
//	GMatrix* compressFeatures(GMatrix& features);

	/// Finds the column in the intrinsic matrix with the largest deviation, then
	/// centers the matrix at the origin and renormalizes so the largest deviation
	/// is 1. Also renormalizes the input layer so these changes will have no effect.
//	void containIntrinsics(GMatrix& intrinsics);

	/// Generate a neural network that is initialized with the Fourier transform
	/// to reconstruct the given time-series data. The number of rows in the given
	/// time-series data is expected to be a power of 2. The resulting neural network will
	/// accept one input, representing time. The outputs will match the number of columns
	/// in the given time-series data. The series is assumed to represent one period
	/// of time in a repeating cycle. The duration of this period is specified as the
	/// parameter, period. The returned network has already had
	/// beginIncrementalLearning called.
//	static GNeuralNet* fourier(GMatrix& series, double period = 1.0);

protected:
	/// Deliberately protected
	/// Throws an exception telling you to call GContextNeuralNet::forwardProp instead.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Deliberately protected.
	/// Throws an exception telling you to call GContextNeuralNet::backProp instead.
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Deliberately protected.
	/// Throws an exception telling you to call GContextNeuralNet::updateGradient instead.
	void updateGradient(const GVec &x, const GVec& outBlame, GVec& inBlame) const override;
};





/// Contains the buffers that a thread needs to train or use a GNeuralNet.
/// Each thread should use a separate GContextNeuralNet object.
/// Call GNeuralNet::newContext to obtain a new GContextNeuralNet object.
class GContextNeuralNet : public GContext
{
friend class GNeuralNet;
protected:
	const GNeuralNet& m_nn;
	std::vector<GContextLayer*> m_layers;
	GContextLayer* m_pOutputLayer; // redundant pointer to the last layer for efficiency purposes

	GContextNeuralNet(const GNeuralNet& nn); // deliberately protected. Call GNeuralNet::newContext to construct one.

public:
	~GContextNeuralNet();

	size_t layerCount() const { return m_layers.size(); }
	GContextLayer& layer(size_t i) { return *m_layers[i]; }

	/// See the comment for GContext::resetState.
	virtual void resetState() override;

	/// Feeds input forward through the component that was used to construct this object.
	void forwardProp(const GVec& input, GVec& output) const;

	/// Idential to forwardProp, except recurrent blocks additionally propagate through time during training.
	void forwardProp_training(const GVec& input, GVec& output) const;

	/// Backpropagates the blame across the component that was used to construct this object.
	/// (If pInBlame is non-NULL, it will also compute the blame for the inputs.)
	void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec* pInBlame);

	/// Updates the gradient for the component that was used to construct this object.
	void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const;

	/// Returns the activation buffer for the output layer
	GVec& predBuf() { return m_pOutputLayer->m_activation; }

	/// Returns the blame buffer for the output layer
	GVec& blameBuf() { return m_pOutputLayer->m_blame; }
};





/// A thin wrapper around a GNeuralNet that implements the GIncrementalLearner interface.
class GNeuralNetLearner : public GIncrementalLearner
{
protected:
	GNeuralNet m_nn;
	GNeuralNetOptimizer* m_pOptimizer;
	bool m_ready;

public:
	GNeuralNetLearner();
	GNeuralNetLearner(const GDomNode* pNode);
	virtual ~GNeuralNetLearner();

	/// Returns a reference to the neural net that this class wraps
	GNeuralNet& nn() { return m_nn; }

	/// Lazily creates an optimizer for the neural net that this class wraps, and returns a reference to it.
	GNeuralNetOptimizer& optimizer();

	/// Convenience method. Adds a layer and returns a reference to it.
	GLayer& newLayer() { return m_nn.newLayer(); }

	virtual void trainIncremental(const GVec &in, const GVec &out) override;
	virtual void trainSparse(GSparseMatrix &features, GMatrix &labels) override;

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Saves the model to a text file.
	virtual GDomNode* serialize(GDom* pDoc) const override;
#endif // MIN_PREDICT

	/// See the comment for GSupervisedLearner::clear
	virtual void clear() override;

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out) override;

#ifndef MIN_PREDICT
	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut) override;
#endif // MIN_PREDICT

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() override { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedFeatureRange(double* pOutMin, double* pOutMax) override;

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() override { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() override { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedLabelRange(double* pOutMin, double* pOutMax) override;

protected:
	/// See the comment for GIncrementalLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels) override;

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel) override;
};




/// A class that facilitates training a neural network with an arbitrary optimization algorithm
class GNeuralNetTargetFunction : public GTargetFunction
{
protected:
	GNeuralNet& m_nn;
	const GMatrix& m_features;
	const GMatrix& m_labels;

public:
	/// features and labels should be pre-filtered to contain only continuous values for the neural network.
	GNeuralNetTargetFunction(GNeuralNet& nn, const GMatrix& features, const GMatrix& labels)
	: GTargetFunction(nn.weightCount()), m_nn(nn), m_features(features), m_labels(labels)
	{
	}

	virtual ~GNeuralNetTargetFunction() {}

	/// Copies the neural network weights into the vector.
	virtual void initVector(GVec& pVector)
	{
		m_nn.weightsToVector(pVector.data());
	}

	/// Copies the vector into the neural network and measures sum-squared error.
	virtual double computeError(const GVec& pVector)
	{
		m_nn.vectorToWeights(pVector.data());
		return m_nn.measureLoss(m_features, m_labels);
	}
};





/// This model uses a randomely-initialized network to map the inputs into
/// a higher-dimensional space, and it uses a layer of perceptrons to learn
/// in this augmented space.
class GReservoirNet : public GIncrementalLearner
{
protected:
	GIncrementalLearner* m_pModel;
	GNeuralNetLearner* m_pNN;
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
