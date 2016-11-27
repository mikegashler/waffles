/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Stephen Ashmore,
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

#ifndef __GBLOCK_H__
#define __GBLOCK_H__

#include <vector>
#include "GMatrix.h"
#include <ostream>

namespace GClasses {


/// Represents a block of network units (artificial neurons) in a neural network.
class GBlock
{
protected:
	size_t m_inPos;

public:
	enum BlockType
	{
		block_neuralnet,
		block_tanh,
		block_logistic,
		block_bentidentity,
		block_softroot,
		block_sigexp,
		block_gaussian,
		block_sine,
		block_rectifier,
		block_leakyrectifier,
		block_softplus,
		block_linear,
		block_activation,
		block_productpooling,
		block_additionpooling,
		block_maxout,
		block_softmax,
		block_restrictedboltzmannmachine,
		block_convolutional1d,
		block_convolutional2d,
		block_maxpooling,
	};

	GBlock();
	GBlock(GDomNode* pNode);
	virtual ~GBlock() {}

	/// Returns the type of this layer
	virtual BlockType type() const = 0;

	/// Returns true iff this layer operates only on individual elements
	virtual bool elementWise() const { return false; }

	/// Returns true iff this layer does its computations in parallel on a GPU.
	virtual bool usesGPU() { return false; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// Unmarshalls the specified DOM node into a layer object.
	static GBlock* deserialize(GDomNode* pNode);

	/// Returns the offset in the previous layer's output where values are fed as input to this block.
	size_t inPos() const { return m_inPos; }

	/// Sets the starting offset in the previous layer's output where values will be fed as input to this block.
	void setInPos(size_t n) { m_inPos = n; };

	/// Makes a string representation of this layer
	virtual std::string to_str() const = 0;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) = 0;

	/// Returns the number of inputs this layer consumes
	virtual size_t inputs() const = 0;

	/// Returns the number of outputs this layer produces
	virtual size_t outputs() const = 0;

	/// Evaluates the input, sets the output.
	virtual void forwardProp(const GVec& input, GVec& output) const = 0;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) = 0;

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t weightCount() const = 0;

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const = 0;

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) = 0;

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(const GBlock* pSource) = 0;

	/// Initialize the weights, usually with small random values.
	virtual void resetWeights(GRand& rand) = 0;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise. The
	/// default values apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation) = 0;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max) = 0;

	/// Multiplies all the weights by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) = 0;

	/// Moves all weights in the direction of zero by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) = 0;

	/// Evaluate the input and outBlame, update the gradient for updating the weights by gradient descent.
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const = 0;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) = 0;

protected:
	GDomNode* baseDomNode(GDom* pDoc) const;
};




/// Standard fully-connected layer of weights. Often followed by a GBlockActivation.
class GBlockLinear : public GBlock
{
protected:
	GMatrix m_weights; // An (inputs+1)-by-outputs matrix of weights. The last row contains the bias values.

public:
	GBlockLinear(size_t outputs, size_t inputs = 0);
	GBlockLinear(GDomNode* pNode);

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_linear; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this layer
	virtual std::string to_str() const override;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
	virtual size_t inputs() const override { return m_weights.rows() - 1; }

	/// Returns the number of outputs this layer produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Multiplies all the weights by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Moves all weights in the direction of zero by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Applies contractive regularization to the weights in this layer.
	void contractWeights(double factor, bool contractBiases, const GVec& output);

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(const GBlock* pSource) override;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise. The
	/// default values apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max) override;

	/// Returns the bias vector of this layer.
	GVec& bias() { return m_weights.back(); }

	/// Returns the bias vector of this layer.
	const GVec& bias() const { return m_weights.back(); }

	/// Get the entire weights matrix
	GMatrix &weights() { return m_weights; }

	/// Get the entire weights matrix
	const GMatrix &weights() const { return m_weights; }

	/// Transforms the weights of this layer by the specified transformation matrix and offset vector.
	/// transform should be the pseudoinverse of the transform applied to the inputs. pOffset should
	/// be the negation of the offset added to the inputs after the transform, or the transformed offset
	/// that is added before the transform.
	void transformWeights(GMatrix& transform, const GVec& offset);

	/// Adjusts weights such that values in the new range will result in the
	/// same behavior that previously resulted from values in the old range.
	virtual void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);
};






/*
class GBlockMaxOut : public GBlock
{
protected:
	GMatrix m_weights; // Each row is an upstream neuron. Each column is a downstream neuron.
	GMatrix m_bias; // Row 0 is the bias. Row 1 is the bias delta.
	GIndexVec m_winners; // The indexes of the winning inputs.

public:
	/// General-purpose constructor.
	GBlockMaxOut(size_t outputs, size_t inputs = 0);

	/// Deserializing constructor
	GBlockMaxOut(GDomNode* pNode);
	~GBlockMaxOut();

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_maxout; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this layer
	virtual std::string to_str() const override;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
	virtual size_t inputs() const override { return m_weights.rows(); }

	/// Returns the number of outputs this layer produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(const GBlock* pSource) override;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double min, double max) override;

	/// Returns a reference to the weights matrix of this layer
	GMatrix& weights() { return m_weights; }

	/// Returns a reference to the weights matrix of this layer
	const GMatrix& weights() const { return m_weights; }

	/// Returns the bias vector of this layer.
	GVec& bias() { return m_bias[0]; }

	/// Returns the bias vector of this layer.
	const GVec& bias() const { return m_bias[0]; }

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
};
*/





class GBlockRestrictedBoltzmannMachine : public GBlock
{
protected:
	GMatrix m_weights; // Each column is an upstream neuron. Each row is a downstream neuron.
	GVec m_bias; // Row 0 is the bias.
	GMatrix m_biasReverse; // Row 0 is the bias. Row 1 is the activation. Row 2 is the blame

public:
	/// General-purpose constructor.
	GBlockRestrictedBoltzmannMachine(size_t outputs, size_t inputs = 0);

	/// Deserializing constructor
	GBlockRestrictedBoltzmannMachine(GDomNode* pNode);

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_restrictedboltzmannmachine; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this layer
	virtual std::string to_str() const override;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
	virtual size_t inputs() const override { return m_weights.cols(); }

	/// Returns the number of outputs this layer produces
	virtual size_t outputs() const override { return m_weights.rows(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;

	/// Feed a vector backwards through this layer.
	void feedBackward(const GVec& output, GVec& input) const;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(const GBlock* pSource) override;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// Also perturbs the bias.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	/// The default values for these parameters apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are <= max.
	virtual void maxNorm(double min, double max) override;

	/// Returns a reference to the weights matrix of this layer
	GMatrix& weights() { return m_weights; }

	/// Returns the bias for the hidden end of this layer.
	GVec& bias() { return m_bias; }

	/// Returns the bias for the hidden end of this layer.
	const GVec& bias() const { return m_bias; }

	/// Returns the bias for the visible end of this layer.
	GVec& biasReverse() { return m_biasReverse[0]; }
	const GVec& biasReverse() const { return m_biasReverse[0]; }

	/// Performs binomial resampling of the activation values on the output end of this layer.
	void resampleHidden(GRand& rand, GVec& output);

	/// Performs binomial resampling of the activation values on the input end of this layer.
	void resampleVisible(GRand& rand, GVec& input);

	/// Draws a sample observation from "iters" iterations of Gibbs sampling.
	/// The resulting sample is placed in activationReverse(), and the corresponding
	/// encoding will be in activation().
	void drawSample(GRand& rand, size_t iters, GVec& output, GVec& input);

	
/*  *** Note that these two commented-out methods are pretty-much the whole point of RBMs, so this class is pretty-much useless until they are restored. ***
	/// Returns the free energy of this layer. Assumes that a pattern has already been set in the 
	double freeEnergy(const GVec& visibleSample);

	/// Refines this layer by contrastive divergence.
	/// pVisibleSample should point to a vector of inputs that will be presented to this layer.
	void contrastiveDivergence(GRand& rand, const GVec& visibleSample, double learningRate, size_t gibbsSamples = 1);
*/
};





class GBlockConvolutional1D : public GBlock
{
protected:
	size_t m_inputSamples;
	size_t m_inputChannels;
	size_t m_outputSamples;
	size_t m_kernelsPerChannel;
	GMatrix m_kernels;
	GVec m_bias;

public:
	/// General-purpose constructor.
	/// For example, if you collect 19 samples from 3 sensors, then the total input size will be 57 (19*3=57).
	/// The three values collected at time 0 will come first, followed by the three values collected at
	/// time 1, and so forth. If kernelSize is 5, then the output will consist of 15 (19-5+1=15) samples.
	/// If kernelsPerChannel is 2, then there will be 6 (3*2=6) channels in the output, for a total of 90 (15*6=90)
	/// output values. The first six channel values will appear first in the output vector, followed by the next six,
	/// and so forth. (kernelSize must be <= inputSamples.)
	GBlockConvolutional1D(size_t inputSamples, size_t inputChannels, size_t kernelSize, size_t kernelsPerChannel);

	/// Deserializing constructor
	GBlockConvolutional1D(GDomNode* pNode);

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_convolutional1d; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this layer
	virtual std::string to_str() const override;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
        virtual size_t inputs() const override { return m_inputSamples * m_inputChannels; }

	/// Returns the number of outputs this layer produces
        virtual size_t outputs() const override { return m_outputSamples * m_inputChannels * m_kernelsPerChannel; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Multiplies all the weights in this layer by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this layer into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this layer into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this layer. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this layer. (Assumes pSource is the same type of layer.)
	virtual void copyWeights(const GBlock* pSource) override;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise.
	/// start specifies the first unit whose incoming weights are perturbed.
	/// count specifies the maximum number of units whose incoming weights are perturbed.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Clips each kernel weight (not including the bias) to fall between -max and max.
	virtual void maxNorm(double min, double max) override;

	const GVec& bias() const { return m_bias; }
	GVec& bias() { return m_bias; }
	const GMatrix& kernels() const { return m_kernels; }
	GMatrix& kernels() { return m_kernels; }
};



/*
class GBlockConvolutional2D : public GBlock
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
	GMatrix m_kernels;

	/// Data as images
	Image m_kernelImage, m_deltaImage;
	Image m_inputImage, m_upStreamErrorImage;
	Image m_actImage, m_errImage;

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
	GBlockConvolutional2D(size_t width, size_t height, size_t channels, size_t kWidth, size_t kHeight, size_t kCount = 0);

	/// Constructor that will automatically use the upstream convolutional layer when added to a neural network
	GBlockConvolutional2D(size_t kWidth, size_t kHeight, size_t kCount = 0);

	GBlockConvolutional2D(GDomNode* pNode);

	virtual BlockType type() const override { return block_convolutional2d; }
	virtual GDomNode* serialize(GDom* pDoc) const override;
	virtual std::string to_str() const override;
	virtual void resize(size_t inputs, size_t outputs) override;
        virtual size_t inputs() const override { return m_width * m_height * m_channels; }
        virtual size_t outputs() const override { return m_outputWidth * m_outputHeight * m_bias.size(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;


	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	virtual void scaleWeights(double factor, bool scaleBiases) override;
	virtual void diminishWeights(double amount, bool regularizeBiases) override;
	virtual size_t weightCount() const override;
	virtual size_t weightsToVector(double* pOutVector) const override;
	virtual size_t vectorToWeights(const double *pVector) override;
	virtual void copyWeights(const GBlock* pSource) override;
	virtual void resetWeights(GRand& rand) override;
	virtual void perturbWeights(GRand& rand, double deviation) override;
	virtual void maxNorm(double min, double max) override;

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
	const GMatrix &kernels() const { return m_kernels; }
	GMatrix &kernels() { return m_kernels; }
	const GVec &bias() const { return m_bias; }
	GVec &bias() { return m_bias; }
};
*/


} // namespace GClasses

#endif // __GBLOCK_H__
