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

#ifndef __GCUDALAYERS_H__
#define __GCUDALAYERS_H__

#include "GCudaMatrix.h"
#include "../../GClasses/GNeuralNet.h"

namespace GClasses {

class GCudaLayer : public GNeuralNetLayer
{
protected:
	GCudaEngine& m_engine;

public:
	/// Standard constructor
	GCudaLayer(GCudaEngine& engine);

	/// Throws an exception.
	GCudaLayer(GDomNode* pNode, GCudaEngine& engine);
	virtual ~GCudaLayer();

	/// Returns true
	virtual bool usesGPU() { return true; }

	/// Return the activation vector on device memory
	virtual GCudaVector& deviceActivation() = 0;

	/// Return the error vector on device memory
	virtual GCudaVector& deviceError() = 0;
};



class GLayerClassicCuda : public GCudaLayer
{
protected:
	GCudaMatrix m_weights; // Each row is an upstream neuron. Each column is a downstream neuron.
	GCudaMatrix m_delta;
	GCudaVector m_bias;
	GCudaVector m_biasDelta;
	GCudaVector m_activation;
	GCudaVector m_incoming;
	GCudaVector m_error;
	GVec m_outgoing;

public:
	/// General-purpose constructor. Takes ownership of pActivationFunction.
	GLayerClassicCuda(GCudaEngine& engine, size_t inputs, size_t outputs);
	
	// Unmarshaling constructor
	GLayerClassicCuda(GDomNode* pNode, GCudaEngine& engine);

	virtual ~GLayerClassicCuda();

	/// Returns the type of this layer
	virtual const char* type() { return "classiccuda"; }

	/// Converts to a classic layer and serializes.
	/// (The serialized form will not remember that it was trained with CUDA,
	/// so it can be loaded on machines without a GPU. If you want to resume training
	/// on a GPU, you will need to call "upload" to get back to a GLayerClassicCuda.)
	GDomNode* serialize(GDom* pDoc);

	virtual std::string to_str();

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() const { return m_weights.rows(); }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() const { return m_weights.cols(); }

	/// Resizes this layer. If pRand is non-NULL, then it throws an exception.
	virtual void resize(size_t inputs, size_t outputs);

	/// Returns the activation vector in device memory.
	virtual GCudaVector& deviceActivation() { return m_activation; }

	/// Downloads the activation vector from device memory to host memory, and returns a pointer to the host memory copy.
	virtual GVec& activation();

	/// Returns the error vector in device memory.
	virtual GCudaVector& deviceError() { return m_error; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual GVec& error();

	/// Uploads pIn to the GPU, then feeds it through this layer
	virtual void feedForward(const GVec& in);

	/// Feeds the activation from the upstream layer through this layer.
	/// (If the upstream layer is a GPU-optimized layer, then it will be faster because
	/// the data can stay on the GPU.)
	virtual void feedForward(GNeuralNetLayer* pUpStreamLayer);

	/// Throws an exception because it is not implemented for this layer type yet.
	virtual void dropOut(GRand& rand, double probOfDrop);

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const GVec& target);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream error vector.
	/// (Assumes that the error in this layer has already been deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	/// Note that this method does not sync with the GPU. It assumes that you will yet call
	/// updateBias, which does sync with the GPU.
	virtual void updateDeltas(const GVec& upStreamActivation, double momentum);

	/// Refines the weights by gradient descent.
	virtual void updateDeltas(GNeuralNetLayer* pUpStreamLayer, double momentum);

	/// Adds the deltas to the weights.
	virtual void applyDeltas(double learningRate);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(const GVec& upStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// This is a special weight update method for use with drop-connect. It updates the weights, and restores
	/// the weights that were previously dropped by a call to dropConnect.
	virtual void updateWeightsAndRestoreDroppedOnes(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double momentum);

	/// Updates the bias of this layer by gradient descent. (Assumes the error has already been
	/// computed and deactivated.) This method also syncs with the GPU, so it should be
	/// called after updateWeights.
	virtual void updateBias(double learningRate, double momentum);

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases);

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool diminishBiases);

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t countWeights();

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* outVector);

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* vector);

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

	/// Throws an exception
	virtual void refineActivationFunction(double learningRate);

	/// Throws an exception
	virtual void regularizeActivationFunction(double lambda);

	/// Copies the weights and bias vector from GLayerClassic layer into this layer.
	void upload(const GLayerClassic& source);

	/// Copies the weights and bias vector from this layer into a GLayerClassic layer.
	void download(GLayerClassic& dest) const;

protected:
	void copyBiasToNet();

	void feedIn(const GVec& in);

	void feedIn(GNeuralNetLayer* pUpStreamLayer);

	void activate();

};


class GLayerConvolutional2DCuda : public GCudaLayer
{
protected:
	// primary properties
	size_t m_inputRows;
	size_t m_inputCols;
	size_t m_inputChannels;
	size_t m_kernelRows;
	size_t m_kernelCols;
	size_t m_kernelCount;
	size_t m_stride;
	size_t m_padding;

	// derived properties
	size_t m_outputRows;
	size_t m_outputCols;

	// parameters
	GCudaVector m_incoming;
	GCudaVector m_bias, m_biasDelta;
	GCudaMatrix m_kernels;
	GCudaMatrix m_delta;
	GCudaVector m_net;
	GCudaVector m_activation;
	GCudaVector m_error;
	GVec m_outgoing;

public:
	/// General-purpose constructor.
	/// For example, if your input is a 64x48 color (RGB) image, then inputCols will be 64, inputRows will be 48,
	/// and inputChannels will be 3. The total input size will be 9216 (64*48*3=9216).
	/// The values should be presented as inputChannels 2d images (i.e. a 64x48x1 image for red, a 64x48 image for blue, and a 64x48 image for green) in row major order.
	/// kernelCount determines the number of output channels.
	/// kernelRows, kernelCols, stride, and padding determine the size of the output.
	GLayerConvolutional2DCuda(GCudaEngine& engine, size_t inputCols, size_t inputRows, size_t inputChannels, size_t kernelRows, size_t kernelCols, size_t kernelCount, size_t stride = 1, size_t padding = 0);

	/// Constructor that uses the upstream convolutional layer to determine input dimensions
	GLayerConvolutional2DCuda(GCudaEngine& engine, const GLayerConvolutional2DCuda& upstream, size_t kernelRows, size_t kernelCols, size_t kernelCount, size_t stride = 1, size_t padding = 0);

	GLayerConvolutional2DCuda(GDomNode* pNode, GCudaEngine& engine);
	virtual ~GLayerConvolutional2DCuda();

	virtual const char *type() { return "conv2dcuda"; }
	virtual GDomNode *serialize(GDom *pDoc);
	virtual std::string to_str();
	virtual size_t inputs() const { return m_inputRows * m_inputCols * m_inputChannels; }
	virtual size_t outputs() const { return m_outputRows * m_outputCols * m_kernelCount; }
	virtual void resize(size_t inputs, size_t outputs);
	virtual void resizeInputs(GNeuralNetLayer *pUpStreamLayer);

	/// Uploads pIn to the GPU, then feeds it through this layer
	virtual void feedForward(const GVec &in);

	/// Feeds the activation from the upstream layer through this layer.
	/// (If the upstream layer is a GPU-optimized layer, then it will be faster because
	/// the data can stay on the GPU.)
	virtual void feedForward(GNeuralNetLayer* pUpStreamLayer);

	virtual void dropOut(GRand &rand, double probOfDrop);
	virtual void dropConnect(GRand &rand, double probOfDrop);
	virtual void computeError(const GVec &target);
	virtual void deactivateError();
	virtual void backPropError(GNeuralNetLayer *pUpStreamLayer);
	virtual void updateDeltas(const GVec &upStreamActivation, double momentum);
	virtual void updateDeltas(GNeuralNetLayer* pUpStreamLayer, double momentum);
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
	virtual GVec& activation();
	virtual GVec& error();
	virtual GCudaVector& deviceActivation() { return m_activation; }
	virtual GCudaVector& deviceError() { return m_error; }

	/// Copies the kernels and biases from a GLayerConvolutional2D layer into this layer.
	void upload(const GLayerConvolutional2D& source);

	/// Copies the weights and bias vector from this layer into a GLayerConvolutional2D layer.
	void download(GLayerConvolutional2D& dest) const;

	size_t kernelRows() const { return m_kernelRows; }
	size_t kernelCols() const { return m_kernelCols; }

	size_t outputRows() const { return m_outputRows; }
	size_t outputCols() const { return m_outputCols; }
	size_t outputChannels() const { return m_kernelCount; }
};

} // namespace GClasses

#endif // __GCUDALAYERS_H__
