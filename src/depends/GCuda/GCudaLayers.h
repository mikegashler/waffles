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
	GCudaLayer(GCudaEngine& engine) : GNeuralNetLayer(), m_engine(engine) {}

	/// Throws an exception.
	GCudaLayer(GDomNode* pNode, GCudaEngine& engine);
	virtual ~GCudaLayer() {}

	/// Throws an exception
	virtual GDomNode* serialize(GDom* pDoc);

	/// Returns true
	virtual bool usesGPU() { return true; }

	/// Return the activation vector on device memory
	virtual GCudaVector& deviceActivation() = 0;

	/// Return the error vector on device memory
	virtual GCudaVector& deviceError() = 0;
};


class GLayerCuda : public GCudaLayer
{
protected:
	GCudaMatrix m_weights; // Each row is an upstream neuron. Each column is a downstream neuron.
	GCudaVector m_bias;
	GCudaVector m_activation;
	GCudaVector m_incoming;
	GCudaVector m_error;
	double* m_pOutgoing;

public:
	/// General-purpose constructor. Takes ownership of pActivationFunction.
	GLayerCuda(GCudaEngine& engine, size_t inputs, size_t outputs);
	virtual ~GLayerCuda();

	/// Returns the type of this layer
	virtual const char* type() { return "cuda"; }

	/// Returns the number of values expected to be fed as input into this layer.
	virtual size_t inputs() { return m_weights.rows(); }

	/// Returns the number of nodes or units in this layer.
	virtual size_t outputs() { return m_weights.cols(); }

	/// Resizes this layer. If pRand is non-NULL, then it throws an exception.
	virtual void resize(size_t inputs, size_t outputs, GRand* pRand = NULL);

	/// Returns the activation vector in device memory.
	virtual GCudaVector& deviceActivation() { return m_activation; }

	/// Downloads the activation vector from device memory to host memory, and returns a pointer to the host memory copy.
	virtual double* activation();

	/// Returns the error vector in device memory.
	virtual GCudaVector& deviceError() { return m_error; }

	/// Returns a buffer used to store error terms for each unit in this layer.
	virtual double* error();

	virtual void copyBiasToNet();

	virtual void feedIn(const double* pIn, size_t inputStart, size_t inputCount);

	virtual void feedIn(GNeuralNetLayer* pUpStreamLayer, size_t inputStart);

	virtual void activate();

	/// Computes the error terms associated with the output of this layer, given a target vector.
	/// (Note that this is the error of the output, not the error of the weights. To obtain the
	/// error term for the weights, deactivateError must be called.)
	virtual void computeError(const double* pTarget);

	/// Multiplies each element in the error vector by the derivative of the activation function.
	/// This results in the error having meaning with respect to the weights, instead of the output.
	virtual void deactivateError();

	/// Backpropagates the error from this layer into the upstream error vector.
	/// (Assumes that the error in this layer has already been deactivated.
	/// The error this computes is with respect to the output of the upstream layer.)
	virtual void backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart = 0);

	/// Updates the weights that feed into this layer (not including the bias) by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	/// Note that this method does not sync with the GPU. It assumes that you will yet call
	/// updateBias, which does sync with the GPU.
	virtual void updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum);

	/// Refines the weights by gradient descent.
	/// Note that this method does not sync with the GPU. It assumes that you will yet call
	/// updateBias, which does sync with the GPU.
	virtual void updateWeights(GNeuralNetLayer* pUpStreamLayer, size_t inputStart, double learningRate, double momentum);

	/// Updates the bias of this layer by gradient descent. (Assumes the error has already been
	/// computed and deactivated.) This method also syncs with the GPU, so it should be
	/// called after updateWeights.
	virtual void updateBias(double learningRate, double momentum);

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor);

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount);

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

	/// Copies the weights and bias vector from this layer into a GLayerClassic layer.
	void upload(GLayerClassic& source);

	/// Copies the weights and bias vector from GLayerClassic layer into this layer.
	void download(GLayerClassic& dest);
};

} // namespace GClasses

#endif // __GCUDALAYERS_H__
