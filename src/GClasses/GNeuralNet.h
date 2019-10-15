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
#include "GOptimizer.h"
#include "GVec.h"
#include <vector>
#include "GDom.h"
#include <cmath>

namespace GClasses {

class GContextLayer;
class GContextNeuralNet;
class GContextRecurrent;
class GLayer;










/// Represents a block of network units (artificial neurons) in a neural network.
class GBlock
{
protected:
	size_t inputCount;
	size_t outputCount;
	size_t m_inPos;

public:
	GVec outputBuf;
	GVec outBlameBuf;
	GVec weightsBuf;
	GVec gradientBuf;
	GConstVecWrapper input;
	GVecWrapper output;
	GVecWrapper outBlame;
	GVecWrapper inBlame;
	GVecWrapper weights;
	GVecWrapper gradient;

	enum BlockType
	{
		block_neuralnet,

		// activation functions
		block_identity,
		block_tanh,
		block_scaledtanh,
		block_logistic,
		block_el,
		block_elpos,
		block_bentidentity,
		block_sigexp,
		block_gaussian,
		block_sine,
		block_rectifier,
		block_leakyrectifier,
		block_softplus,
		block_softroot,
		block_softmax,

		// weights transfer
		block_linear,
		block_fanout,
		block_conv,
		block_temperedlinear,
		block_pal,
		block_hinge,
		block_elbow,
		block_softexp,
		block_hypercube,
		block_catin,
		block_catout,
		block_weight_digester,
		block_optional,

		// weightless transfer
		block_scalarsum,
		block_scalarproduct,
		block_biasproduct,
		block_switch,
		block_spectral,
		block_spreader,
		block_repeater,

		// recurrent
		block_lstm,

		// still needed
		// block_maxout,
		// block_batch_normalization,
		// block_drop_out,
		// block_drop_connect,
		// block_softmax,
		// block_gru,
	};

	GBlock(size_t inputs, size_t outputs);
	GBlock(const GBlock& that);
	GBlock(const GDomNode* pNode);
	virtual ~GBlock();

	/// Returns the type of this block
	virtual BlockType type() const = 0;

	/// Returns the name of this block in the form of a string
	virtual std::string name() const = 0;

	/// Returns a string representation of this block
	virtual std::string to_str(bool includeWeights = false, bool includeActivations = false) const;

	/// Returns true iff this block is recurrent
	virtual bool isRecurrent() const { return false; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Unmarshalls the specified DOM node into a block object.
	static GBlock* deserialize(GDomNode* pNode, GRand& rand);

	/// Returns a copy of this block
	virtual GBlock* clone() const = 0;

	/// Attaches this block to the buffers it will operate on.
	/// If any of pOutput, pOutBlame, pWeights, or pGradient are nullptr, it allocates its own buffers for these.
	/// If pInput or pInBlame are nullptr, it just leaves them unbound.
	virtual void bind(const GVec* pInput, GVec* pOutput, GVec* pOutBlame, GVec* pInBlame, GVec* pWeights, GVec* pGradient);

	/// Returns the offset in the previous layer's output where values are fed as input to this block.
	size_t inPos() const { return m_inPos; }

	/// Sets the starting offset in the previous layer's output where values will be fed as input to this block.
	void setInPos(size_t n);

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const { return inputCount; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const { return outputCount; }

	/// Evaluate the input, compute the output. (Assumes bind has already been called.)
	virtual void forwardProp() = 0;

	/// Computes the blame on the output of this block.
	/// Returns the SSE. Some blocks, such as SoftMax, may override it for comparable behavior.
	/// (Assumes forwardProp has already been called.)
	virtual double computeBlame(const GVec& target);

	/// Resets the state in any recurrent connections.
	virtual void resetState() {}

	/// Resets the state in any recurrent connections.
	virtual GBlock* advanceState(size_t unfoldedInstances) { return this; }

	/// Evaluate outBlame, update inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	/// (Assumes computeBlame has already been called.)
	virtual void backProp() = 0;

	/// Evaluate the input and outBlame, update the gradient of the weights.
	/// (Assumes backProp has already been called.)
	virtual void updateGradient() = 0;

	/// By default, just calls updateGradient. Blocks that implement this method
	/// update the gradient using only the sign of the input, ignoring the magnitude of the input.
	virtual void updateGradientNormalized() { updateGradient(); }

	/// Adds the gradient scaled by the learning rate to the weights.
	/// (Assumes updateGradient has already been called.)
	virtual void step(double learningRate, double momentum);

	/// Same as step, but also adds random noise proportional by jitter to the gradient magnitude to the step.
	virtual void step_jitter(double learningRate, double momentum, double jitter, GRand& rand);

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const = 0;

	/// Returns the number of double-precision elements necessary to represent the gradient of this block.
	virtual size_t gradCount() const
	{
		return weightCount();
	}

	/// Initialize the weights, usually with small random values.
	virtual void initWeights(GRand& rand) = 0;

	/// Blocks that support this method will initialize the weights in a manner that causes the
	/// block to implement the identity function. For all other blocks with weights, this will throw an exception.
	virtual void init_identity(GVec& weights);

	/// Return true iff the weights for this block make it implement identity.
	virtual bool is_identity(GVec& weights);

	/// Returns the number of weights this layer would have if its input and output sizes were adjusted by the given values.
	/// Throws an exception for blocks that do not implement this method.
	virtual size_t adjustedWeightCount(int inAdjustment, int outAdjustment);

	/// Add newInputs units to the input and newOutput units to the output of this block.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand);

	/// If input is not INVALID_INDEX, then the specified input will be dropped.
	/// If output is not INVALID_INDEX, then the specified output will be dropped.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft);

	/// Puts a 1 in elements that correspond with bias weights and a 0 in all other elements.
	virtual void biasMask(GVec& mask);

	/// Returns the index of the first input that this layer completely ignores, or INVALID_INDEX if there are none.
	virtual size_t firstIgnoredInput(const GVec& weights);

	/// Uses central differencing to test that backProp and updateGradient are implemented correctly.
	/// Throws an exception if a problem is found.
	void finiteDifferencingTest(double tolerance = 0.0005);

protected:
	GDomNode* baseDomNode(GDom* pDoc) const;

	/// Exercises some basic functionality that all blocks have in common
	void basicTest();
};




/// The base class of blocks that have no weights
class GBlockWeightless : public GBlock
{
public:
	GBlockWeightless(size_t inputs, size_t outputs) : GBlock(inputs, outputs) {}
	GBlockWeightless(const GBlockWeightless& that) : GBlock(that) {}
	GBlockWeightless(const GDomNode* pNode) : GBlock(pNode) {}
	virtual ~GBlockWeightless() {}

	virtual size_t weightCount() const override { return 0; }
	virtual void initWeights(GRand& rand) override {}
	virtual void updateGradient() override {}
	virtual void step(double learningRate, double momentum) override {}
};



/// The base class of blocks that apply an activation function, such as tanh, in an element-wise manner.
class GBlockActivation : public GBlockWeightless
{
public:
	GBlockActivation(size_t size);
	GBlockActivation(const GBlockActivation& that) : GBlockWeightless(that) {}
	GBlockActivation(const GDomNode* pNode);
	virtual ~GBlockActivation() {}

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Computes the input that would produce the specified output.
	/// (May throw an exception if this activation function is not invertible.)
	void inverseProp(const GVec& output, GVec& input);

	/// Evaluates the activation function
	virtual double eval(double x) const = 0;

	/// Evaluates the derivative of the activation function.
	/// x is the net input, and f_x is the output activation--the value obtained by calling eval(x).
	virtual double derivative(double x, double f_x) const = 0;

	virtual double inverse(double y) const
	{
		throw Ex("Sorry, this activation function is not invertible");
	}
};




/// Applies the [Identity function](https://en.wikipedia.org/wiki/Identity_function) element-wise to the input.
/// Serves as a pass-through block of units in a neural network.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = x \f]   | ![](Activation_identity.png)
///
class GBlockIdentity : public GBlockActivation
{
public:
	GBlockIdentity(size_t size) : GBlockActivation(size) {}
	GBlockIdentity(const GBlockIdentity& that) : GBlockActivation(that) {}
	GBlockIdentity(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockIdentity() {}
	virtual BlockType type() const override { return block_identity; }
	virtual std::string name() const override { return "GBlockIdentity"; }
	virtual GBlockIdentity* clone() const override { return new GBlockIdentity(*this); }
	virtual double eval(double x) const override { return x; }
	virtual double derivative(double x, double f_x) const override { return 1.0; }
	virtual double inverse(double y) const override { return y; }
};






/// Applies the [TanH function](https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent) element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = tanh(x) \f]   | ![](Activation_tanh.png)
///
// (Note, the following code is a bit faster:
// 		if(std::abs(x) >= 700.0)
// 			return (x >= 0 ? 1.0 : -1.0);
// 		double a = exp(x);
// 		double b = 1.0 / a;
// 		return (a - b) / (a + b);
// and here is a fast version of the derivative
// 		if(std::abs(x) >= 700.0)
// 			return (x >= 0 ? 1.0 : 0.0);
// 		double a = exp(x);
// 		double b = 1.0 / a;
// 		double d = 2.0 / (a + b); // sech(x)
// 		return d * d;
class GBlockTanh : public GBlockActivation
{
public:
	GBlockTanh(size_t size) : GBlockActivation(size) {}
	GBlockTanh(const GBlockTanh& that) : GBlockActivation(that) {}
	GBlockTanh(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockTanh() {}
	virtual BlockType type() const override { return block_tanh; }
	virtual std::string name() const override { return "GBlockTanh"; }
	virtual GBlockTanh* clone() const override { return new GBlockTanh(*this); }
	virtual double eval(double x) const override { return std::tanh(x); }
	virtual double derivative(double x, double f_x) const override { return 1.0 - (f_x * f_x); }
	virtual double inverse(double y) const override { return 0.5 * std::log(-(y + 1) / (y - 1)); }
};




/// Applies a scaled TanH function element-wise to the input.
/// | Equation  |
/// | --------- |
/// | \f[ f(x) = tanh(x \times 0.66666667) \times  1.7159\f] |
/// LeCun et al. suggest scale_in=2/3 and scale_out=1.7159. By carefully matching
/// scale_in and scale_out, the nonlinearity can also be tuned to preserve the mean and variance of its input:
/// - scale_in=0.5, scale_out=2.4: If the input is a random normal variable, the output will have zero mean and unit variance.
/// - scale_in=1, scale_out=1.6: Same property, but with a smaller linear regime in input space.
/// - scale_in=0.5, scale_out=2.27: If the input is a uniform normal variable, the output will have zero mean and unit variance.
/// - scale_in=1, scale_out=1.48: Same property, but with a smaller linear regime in input space.
///
class GBlockScaledTanh : public GBlockActivation
{
	const double SCALE_IN = 0.66666667;
	const double SCALE_OUT = 1.7159;
public:
	GBlockScaledTanh(size_t size) : GBlockActivation(size) {}
	GBlockScaledTanh(const GBlockScaledTanh& that) : GBlockActivation(that) {}
	GBlockScaledTanh(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockScaledTanh() {}
	virtual BlockType type() const override { return block_scaledtanh; }
	virtual std::string name() const override { return "GBlockScaledTanh"; }
	virtual GBlockScaledTanh* clone() const override { return new GBlockScaledTanh(*this); }
	virtual double eval(double x) const override { return std::tanh(x * SCALE_IN) * SCALE_OUT; }
	virtual double derivative(double x, double f_x) const override { return SCALE_IN/SCALE_OUT*(SCALE_OUT-f_x)*(SCALE_OUT+f_x); }
	virtual double inverse(double y) const override { return 0.5 / SCALE_IN * std::log(-(y / SCALE_OUT + 1) / (y / SCALE_OUT - 1)); }
};



/// Applies the [Logistic function](https://en.wikipedia.org/wiki/Logistic_function) element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = \frac{1}{1 + e^{-x}} \f]   | ![](Activation_logistic.png)
///
class GBlockLogistic : public GBlockActivation
{
public:
	GBlockLogistic(size_t size) : GBlockActivation(size) {}
	GBlockLogistic(const GBlockLogistic& that) : GBlockActivation(that) {}
	GBlockLogistic(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockLogistic() {}
	virtual BlockType type() const override { return block_logistic; }
	virtual std::string name() const override { return "GBlockLogistic"; }
	virtual GBlockLogistic* clone() const override { return new GBlockLogistic(*this); }
	virtual double eval(double x) const override
	{
		if(x >= 700.0) // Don't trigger a floating point exception
			return 1.0;
		else if(x < -700.0) // Don't trigger a floating point exception
			return 0.0;
		else return 1.0 / (std::exp(-x) + 1.0);
	}
	virtual double derivative(double x, double f_x) const override { return f_x * (1.0 - f_x); }
	virtual double inverse(double y) const override { return std::log(y / (1.0 - y)); }
};




/// An exponential-linear unit
class GBlockEl : public GBlockActivation
{
public:
	GBlockEl(size_t size) : GBlockActivation(size) {}
	GBlockEl(const GBlockEl& that) : GBlockActivation(that) {}
	GBlockEl(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockEl() {}
	virtual BlockType type() const override { return block_el; }
	virtual std::string name() const override { return "GBlockEl"; }
	virtual GBlockEl* clone() const override { return new GBlockEl(*this); }
	virtual double eval(double x) const override { return x < 0 ? std::exp(x) - 1.0 : x; }
	virtual double derivative(double x, double f_x) const override { return x < 0 ? std::exp(x) : 1.0; }
	virtual double inverse(double y) const override { return y < 0 ? std::log(y + 1) : y; }
};




/// An exponential-linear unit.
/// As a special case, this block passes UNKNOWN_REAL_VALUE straight through.
class GBlockElPos : public GBlockActivation
{
public:
	GBlockElPos(size_t size) : GBlockActivation(size) {}
	GBlockElPos(const GBlockElPos& that) : GBlockActivation(that) {}
	GBlockElPos(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockElPos() {}
	virtual BlockType type() const override { return block_elpos; }
	virtual std::string name() const override { return "GBlockElPos"; }
	virtual GBlockElPos* clone() const override { return new GBlockElPos(*this); }
	virtual double eval(double x) const override { return x == UNKNOWN_REAL_VALUE ? UNKNOWN_REAL_VALUE : (x < 0 ? std::exp(x) : x + 1); }
	virtual double derivative(double x, double f_x) const override { return x == UNKNOWN_REAL_VALUE ? 0.0 : (x < 0 ? std::exp(x) : 1.0); }
	virtual double inverse(double y) const override { return y = UNKNOWN_REAL_VALUE ? UNKNOWN_REAL_VALUE : (y < 0 ? std::log(y) : y + 1); }
};




#define BEND_AMOUNT 0.5
#define BEND_SIZE 0.1
/// Applies the Bent identity element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = \frac{\sqrt{x^2+0.01}-0.1}{2}+x \f]   | ![](Activation_bent_identity.png)
///
class GBlockBentIdentity : public GBlockActivation
{
public:
	GBlockBentIdentity(size_t size) : GBlockActivation(size) {}
	GBlockBentIdentity(const GBlockBentIdentity& that) : GBlockActivation(that) {}
	GBlockBentIdentity(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockBentIdentity() {}
	virtual BlockType type() const override { return block_bentidentity; }
	virtual std::string name() const override { return "GBlockBentIdentity"; }
	virtual GBlockBentIdentity* clone() const override { return new GBlockBentIdentity(*this); }
	virtual double eval(double x) const override { return BEND_AMOUNT * (std::sqrt(x * x + BEND_SIZE * BEND_SIZE) - BEND_SIZE) + x; }
	virtual double derivative(double x, double f_x) const override { return BEND_AMOUNT * x / std::sqrt(x * x + BEND_SIZE * BEND_SIZE) + 1.0; }
	virtual double inverse(double y) const override { return (std::sqrt(2.0 * BEND_AMOUNT * BEND_AMOUNT * BEND_AMOUNT * BEND_SIZE * y + BEND_AMOUNT * BEND_AMOUNT * BEND_SIZE * BEND_SIZE + BEND_AMOUNT * BEND_AMOUNT * y * y) - BEND_AMOUNT * BEND_SIZE - y) / (BEND_AMOUNT * BEND_AMOUNT - 1.0); }
};



/// An element-wise nonlinearity block
/// This activation function forms a sigmoid shape by splicing exponential and logarithmic functions together.
class GBlockSigExp : public GBlockActivation
{
public:
	GBlockSigExp(size_t size) : GBlockActivation(size) {}
	GBlockSigExp(const GBlockSigExp& that) : GBlockActivation(that) {}
	GBlockSigExp(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockSigExp() {}
	virtual BlockType type() const override { return block_sigexp; }
	virtual std::string name() const override { return "GBlockSigExp"; }
	virtual GBlockSigExp* clone() const override { return new GBlockSigExp(*this); }
	virtual double eval(double x) const override { return (x <= 0.0 ? exp(x) - 1.0 : std::log(x + 1.0)); }
	virtual double derivative(double x, double f_x) const override { return (x <= 0.0 ? std::exp(x) : 1.0 / (x + 1.0)); }
	virtual double inverse(double y) const override { return (y > 0.0 ? exp(y) - 1.0 : std::log(y + 1.0)); }
};



/// Applies the [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function) element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = e^{-x^2} \f]   | ![](Activation_gaussian.png)
///
class GBlockGaussian : public GBlockActivation
{
public:
	GBlockGaussian(size_t size) : GBlockActivation(size) {}
	GBlockGaussian(const GBlockGaussian& that) : GBlockActivation(that) {}
	GBlockGaussian(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockGaussian() {}
	virtual BlockType type() const override { return block_gaussian; }
	virtual std::string name() const override { return "GBlockGaussian"; }
	virtual GBlockGaussian* clone() const override { return new GBlockGaussian(*this); }
	virtual double eval(double x) const override { return std::exp(-(x * x)); }
	virtual double derivative(double x, double f_x) const override { return -2.0 * x * f_x; }
};



/// Applies the [Sinusoid](https://en.wikipedia.org/wiki/Sine_wave) element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = \sin(x) \f]   | ![](Activation_sinusoid.png)
///
class GBlockSine : public GBlockActivation
{
public:
	GBlockSine(size_t size) : GBlockActivation(size) {}
	GBlockSine(const GBlockSine& that) : GBlockActivation(that) {}
	GBlockSine(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockSine() {}
	virtual BlockType type() const override { return block_sine; }
	virtual std::string name() const override { return "GBlockSine"; }
	virtual GBlockSine* clone() const override { return new GBlockSine(*this); }
	virtual double eval(double x) const override { return std::sin(x); }
	virtual double derivative(double x, double f_x) const override { return std::cos(x); }
};




/// Applies the [Rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (ReLU) element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = \left \{ \begin{array}{rcl} 0 & \mbox{for} & x < 0 \\ x & \mbox{for} & x \ge 0\end{array} \right. \f]   | ![](Activation_rectified_linear.png)
///
class GBlockRectifier : public GBlockActivation
{
public:
	GBlockRectifier(size_t size) : GBlockActivation(size) {}
	GBlockRectifier(const GBlockSine& that) : GBlockActivation(that) {}
	GBlockRectifier(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockRectifier() {}
	virtual BlockType type() const override { return block_rectifier; }
	virtual std::string name() const override { return "GBlockRectifier"; }
	virtual GBlockRectifier* clone() const override { return new GBlockRectifier(*this); }
	virtual double eval(double x) const override { return std::max(0.0, x); }
	virtual double derivative(double x, double f_x) const override { return (x >= 0.0 ? 1.0 : 0.0); }
};




/// Applies the Leaky rectified linear unit (Leaky ReLU) element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = \left \{ \begin{array}{rcl} 0.01x & \mbox{for} & x < 0\\ x & \mbox{for} & x \ge 0\end{array} \right. \f]   | ![](Activation_prelu.png)
///
class GBlockLeakyRectifier : public GBlockActivation
{
public:
	GBlockLeakyRectifier(size_t size) : GBlockActivation(size) {}
	GBlockLeakyRectifier(const GBlockLeakyRectifier& that) : GBlockActivation(that) {}
	GBlockLeakyRectifier(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockLeakyRectifier() {}
	virtual BlockType type() const override { return block_leakyrectifier; }
	virtual std::string name() const override { return "GBlockLeakyRectifier"; }
	virtual GBlockLeakyRectifier* clone() const override { return new GBlockLeakyRectifier(*this); }
	virtual double eval(double x) const override { return x >= 0.0 ? x : 0.01 * x; }
	virtual double derivative(double x, double f_x) const override { return x >= 0.0 ? 1.0 : 0.01; }
	virtual double inverse(double y) const override { return (y > 0.0 ? y : 100.0 * y); }
};




/// Applies the SoftPlus function element-wise to the input.
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x)=\ln(1+e^x) \f]   | ![](Activation_softplus.png)
///
// (Note: A similar, but less well-known function is the integral of the logistic function. I think it is slightly faster to compute.)
class GBlockSoftPlus : public GBlockActivation
{
public:
	GBlockSoftPlus(size_t size) : GBlockActivation(size) {}
	GBlockSoftPlus(const GBlockSoftPlus& that) : GBlockActivation(that) {}
	GBlockSoftPlus(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockSoftPlus() {}
	virtual BlockType type() const override { return block_softplus; }
	virtual std::string name() const override { return "GBlockSoftPlus"; }
	virtual GBlockSoftPlus* clone() const override { return new GBlockSoftPlus(*this); }
	virtual double eval(double x) const override { return x > 500 ? x : log(1.0 + exp(x)); }
	virtual double derivative(double x, double f_x) const override { return 1.0 / (1.0 + exp(-x)); }
	virtual double inverse(double y) const override { return log(exp(y) - 1.0); }
};




/// An element-wise nonlinearity block.
/// This function is shaped like a sigmoid, but with both a co-domain and domain
/// that spans the continuous values. At very negative values,
/// it is shaped like y=-sqrt(-2x). Near zero, it is shaped
/// like y=x. At very positive values, it is shaped like y=sqrt(2x).
class GBlockSoftRoot : public GBlockActivation
{
public:
	GBlockSoftRoot(size_t size) : GBlockActivation(size) {}
	GBlockSoftRoot(const GBlockSoftRoot& that) : GBlockActivation(that) {}
	GBlockSoftRoot(const GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockSoftRoot() {}
	virtual BlockType type() const override { return block_softroot; }
	virtual std::string name() const override { return "GBlockSoftRoot"; }
	virtual GBlockSoftRoot* clone() const override { return new GBlockSoftRoot(*this); }
	virtual double eval(double x) const override
	{
		double d = std::sqrt(x * x + 1.0);
		return std::sqrt(d + x) - std::sqrt(d - x);
	}
	virtual double derivative(double x, double f_x) const override
	{
		if(std::abs(x) > 1e7)
			return 0.0;
		double d = std::sqrt(x * x + 1.0);
		double t = x / d;
		return (t + 1.0) / (2.0 * std::sqrt(d + x)) - (t - 1.0) / (2.0 * std::sqrt(d - x));
	}
	virtual double inverse(double y) const override { return 0.5 * y * std::sqrt(y * y + 4.0); }
};




/// A softmax block.
class GBlockSoftMax : public GBlockWeightless
{
public:
	GBlockSoftMax(size_t size) : GBlockWeightless(size, size) {}
	GBlockSoftMax(const GBlockSoftMax& that) : GBlockWeightless(that) {}
	GBlockSoftMax(const GDomNode* pNode) : GBlockWeightless(pNode) {}
	virtual ~GBlockSoftMax() {}
	virtual BlockType type() const override { return block_softmax; }
	virtual std::string name() const override { return "GBlockSoftMax"; }
	virtual GBlockSoftMax* clone() const override { return new GBlockSoftMax(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Computes the blame with cross-entropy
	virtual double computeBlame(const GVec& target) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};





/// A block of sinusoid units arranged with exponentially growing wavelengths
/// Assumes there is exactly 1 input.
class GBlockSpectral : public GBlockWeightless
{
protected:
	double m_freq_start;
	double m_freq_scale;
	double m_freq_shift;

public:
	GBlockSpectral(double min_wavelength, double max_wavelength, size_t units, bool linear_spacing = false);
	GBlockSpectral(const GBlockSpectral& that) : GBlockWeightless(that) {}
	GBlockSpectral(const GDomNode* pNode);
	virtual ~GBlockSpectral() {}
	virtual BlockType type() const override { return block_spectral; }
	virtual std::string name() const override { return "GBlockSpectral"; }
	virtual GBlockSpectral* clone() const override { return new GBlockSpectral(*this); }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};





/// Expands a layer by repeating it as needed to acheive the desired number of units.
class GBlockRepeater : public GBlockWeightless
{
public:
	GBlockRepeater(size_t in, size_t out) : GBlockWeightless(in, out) { if(in > out) throw Ex("Expected more outputs than inputs"); }
	GBlockRepeater(const GBlockRepeater& that) : GBlockWeightless(that) {}
	GBlockRepeater(const GDomNode* pNode) : GBlockWeightless(pNode) {}
	virtual ~GBlockRepeater() {}
	virtual BlockType type() const override { return block_repeater; }
	virtual std::string name() const override { return "GBlockRepeater"; }
	virtual GBlockRepeater* clone() const override { return new GBlockRepeater(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};





/// Fans out and shuffles in the values to break spatial locality.
class GBlockSpreader : public GBlockWeightless
{
protected:
	size_t m_spread;
	GIndexVec m_forw;
	GIndexVec m_back;

public:
	GBlockSpreader(size_t units, size_t spread);
	GBlockSpreader(const GBlockSpreader& that);
	GBlockSpreader(const GDomNode* pNode);
	virtual ~GBlockSpreader() {}
	virtual BlockType type() const override { return block_spreader; }
	virtual std::string name() const override { return "GBlockSpreader"; }
	virtual GBlockSpreader* clone() const override { return new GBlockSpreader(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};





/// Standard fully-connected block of weights. Often followed by a GBlockActivation.
class GBlockLinear : public GBlock
{
public:
	/// General-purpose constructor
	GBlockLinear(size_t inputs, size_t outputs);

	/// Copy constructor
	GBlockLinear(const GBlockLinear& that) : GBlock(that) {}

	/// Unmarshalling constructor
	GBlockLinear(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockLinear() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_linear; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockLinear"; }

	/// Returns a copy of this block
	virtual GBlockLinear* clone() const override { return new GBlockLinear(*this); }

	/// Returns a string representation of this block
	virtual std::string to_str(bool includeWeights = false, bool includeActivations = false) const;

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes backProp has already been called.)
	virtual void updateGradient() override;

	/// Updates the gradient using only the sign of the input, ignoring the magnitude of the input.
	virtual void updateGradientNormalized() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;

	/// Computes weights using Ordinary Least Squares
	void ordinaryLeastSquares(const GMatrix& features, const GMatrix& labels, GVec& outWeights);

	/// Adjusts the bias and weights to compensate for an adjustment in the input range
	/// Warning: the gradient is still scaled by the input values, so this is NOT equivalent for training.
	void adjustInputRange(GVec& weights, size_t inputIndex, double oldMin, double oldMax, double newMin, double newMax);

	/// Initializes weights to implement the identity function
	virtual void init_identity(GVec& weights) override;

	/// Puts a 1 in elements that correspond with bias weights and a 0 in all other elements.
	virtual void biasMask(GVec& mask) override;

	/// Returns the number of weights this layer would have if its input and output sizes were adjusted by the given values.
	virtual size_t adjustedWeightCount(int inAdjustment, int outAdjustment) override;

	/// Returns the index of the first input that this layer completely ignores, or INVALID_INDEX if there are none.
	virtual size_t firstIgnoredInput(const GVec& weights) override;

	/// Add newInputs units to the input and newOutput units to the output of this block.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand) override;

	/// If input is not INVALID_INDEX, then the specified input will be dropped.
	/// If output is not INVALID_INDEX, then the specified output will be dropped.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft) override;

	/// Pass in two linear layers with their corresponding weights.
	/// The weights will be combined into a single layer that performs the function of both layers.
	/// If targetA is true, the fused weights will be placed in a. Otherwise they will be placed in b.
	/// The non-target layer must have the same number of inputs and outputs
	/// (to ensure that it doesn't change the weights size).
	static void fuseLayers(GBlockLinear& aa, GVec& a, GBlockLinear& bb, GVec& b, bool targetA);

	/// An experimental regularizer that promotes anti-symmetric weights and layers that mirror each other.
	void regularize_square(double lambda, const GVec* pWeightsNext);
};





/// For each input, produces n outputs, where each output is a linear function of one input.
/// As a special case, inputs with a value of UNKNOWN_REAL_VALUE are replicated in the output.
class GBlockFanOut : public GBlock
{
public:
	/// General-purpose constructor
	GBlockFanOut(size_t inputs, size_t outputs);

	/// Copy constructor
	GBlockFanOut(const GBlockFanOut& that) : GBlock(that) {}

	/// Unmarshalling constructor
	GBlockFanOut(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockFanOut() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_fanout; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockFanOut"; }

	/// Returns a copy of this block
	virtual GBlockFanOut* clone() const override { return new GBlockFanOut(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes backProp has already been called.)
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};





/// This is a running version of batch normalization. That is, it uses running means and running variances
/// instead of batch means and batch variances to normalize inputs.
class GBlockRunningNormalizer : public GBlock
{
protected:
	double batch_size;
	double inv_bs;
	double decay_scalar;
	double epsilon;

public:
	/// General-purpose constructor
	GBlockRunningNormalizer(size_t units, double effective_batch_size);

	/// Copy constructor
	GBlockRunningNormalizer(const GBlockRunningNormalizer& that) : GBlock(that) {}

	/// Unmarshalling constructor
	GBlockRunningNormalizer(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockRunningNormalizer() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_linear; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockRunningNormalizer"; }

	/// Returns a copy of this block
	virtual GBlockRunningNormalizer* clone() const override { return new GBlockRunningNormalizer(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes backProp has already been called.)
	virtual void updateGradient() override;

	/// Updates the weights
	virtual void step(double learningRate, double momentum) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Returns the number of gradient elements
	virtual size_t gradCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};





class GBlockWeightDigester : public GBlock
{
protected:
	GBlock* m_pDigestMyWeights;

public:
	/// General-purpose constructor
	GBlockWeightDigester(GBlock* pDigestMyWeights, size_t extraInputs, size_t outputs);

	/// Copy constructor
	GBlockWeightDigester(const GBlockWeightDigester& that) : GBlock(that) {}

	/// Unmarshalling constructor
	GBlockWeightDigester(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockWeightDigester() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_weight_digester; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockWeightDigester"; }

	/// Returns a copy of this block
	virtual GBlockWeightDigester* clone() const override { return new GBlockWeightDigester(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes backProp has already been called.)
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};





/// A linear layer with protections against saturation.
class GBlockTemperedLinear : public GBlock
{
protected:
	double deviationCap;
	double forgetRate;
	GVec moment1;
	GVec moment2;

public:
	/// General-purpose constructor
	GBlockTemperedLinear(size_t inputs, size_t outputs, double deviation_cap = 3.0, double forget_rate = 0.001);

	/// Copy constructor
	GBlockTemperedLinear(const GBlockTemperedLinear& that);

	/// Unmarshalling constructor
	GBlockTemperedLinear(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockTemperedLinear() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_temperedlinear; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockTemperedLinear"; }

	/// Returns a copy of this block
	virtual GBlockTemperedLinear* clone() const override { return new GBlockTemperedLinear(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes backProp has already been called.)
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};





/// A convolutional layer.
class GBlockConv : public GBlock
{
protected:
	size_t filterSize;
	GTensor tensorInput;
	GTensor tensorFilter;
	GTensor tensorOutput;
	size_t filterCount;
	size_t outputsPerFilter;

public:
	/// General-purpose constructor. Example:
	///  nn.add(new GBlockConv( {28, 28}, {5, 5, 8}, {28, 28, 8} ));
	GBlockConv(const std::initializer_list<size_t>& inputDims, const std::initializer_list<size_t>& filterDims, const std::initializer_list<size_t>& outputDims);

	/// Copy constructor
	GBlockConv(const GBlockConv& that);

	/// Unmarshalling constructor
	GBlockConv(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockConv() {}

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Returns the type of this block
	virtual BlockType type() const override { return block_conv; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockConv"; }

	/// Returns a copy of this block
	virtual GBlockConv* clone() const override { return new GBlockConv(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes backProp has already been called.)
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;

	static void test();
};





/// Treats the input as two concatenated vectors.
/// Adds each corresponding pair of values together to produce the output.
class GBlockMaxPooling2D : public GBlockWeightless
{
protected:
	size_t width, height, channels;

public:
	/// General-purpose constructor.
	/// width is the width of the input. height is the height of the input. channels is the number of channels in the input.
	GBlockMaxPooling2D(size_t width, size_t height, size_t channels);

	/// Copy constructor
	GBlockMaxPooling2D(const GBlockMaxPooling2D& that) : GBlockWeightless(that) {}

	/// Deserializing constructor
	GBlockMaxPooling2D(GDomNode* pNode);

	/// Destructor
	~GBlockMaxPooling2D();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_scalarsum; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockMaxPooling2D"; }

	/// Returns a copy of this block
	virtual GBlockMaxPooling2D* clone() const override { return new GBlockMaxPooling2D(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};




/// Treats the input as two concatenated vectors.
/// Adds each corresponding pair of values together to produce the output.
class GBlockScalarSum : public GBlockWeightless
{
public:
	/// General-purpose constructor.
	GBlockScalarSum(size_t outputs);

	/// Copy constructor
	GBlockScalarSum(const GBlockScalarSum& that) : GBlockWeightless(that) {}

	/// Deserializing constructor
	GBlockScalarSum(GDomNode* pNode);

	/// Destructor
	~GBlockScalarSum();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_scalarsum; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockScalarSum"; }

	/// Returns a copy of this block
	virtual GBlockScalarSum* clone() const override { return new GBlockScalarSum(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};





/// Treats the input as two concatenated vectors.
/// Multiplies each corresponding pair of values together to produce the output.
class GBlockScalarProduct : public GBlockWeightless
{
public:
	/// General-purpose constructor.
	GBlockScalarProduct(size_t outputs);

	/// Copy constructor
	GBlockScalarProduct(const GBlockScalarProduct& that) : GBlockWeightless(that) {}

	/// Deserializing constructor
	GBlockScalarProduct(GDomNode* pNode);

	/// Destructor
	~GBlockScalarProduct();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_scalarproduct; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockScalarProduct"; }

	/// Returns a copy of this block
	virtual GBlockScalarProduct* clone() const override { return new GBlockScalarProduct(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};






/// Like GBlockScalarProduct, except it adds the first two elements instead of multiplying them.
/// (This is intended to be used as a substitute for matrix factorization.)
class GBlockBiasProduct : public GBlockWeightless
{
public:
	/// General-purpose constructor.
	GBlockBiasProduct(size_t outputs);

	/// Copy constructor
	GBlockBiasProduct(const GBlockBiasProduct& that) : GBlockWeightless(that) {}

	/// Deserializing constructor
	GBlockBiasProduct(GDomNode* pNode);

	/// Destructor
	~GBlockBiasProduct();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_biasproduct; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockBiasProduct"; }

	/// Returns a copy of this block
	virtual GBlockBiasProduct* clone() const override { return new GBlockBiasProduct(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};






/// Treats the input as three concatenated vectors: a, b, and c.
/// (The values in 'a' typically fall in the range [0,1].)
/// The output is computed element-wise as a*b + (1-a)*c.
class GBlockSwitch : public GBlockWeightless
{
public:
	/// General-purpose constructor.
	GBlockSwitch(size_t outputs);

	/// Copy constructor
	GBlockSwitch(const GBlockScalarProduct& that) : GBlockWeightless(that) {}

	/// Deserializing constructor
	GBlockSwitch(GDomNode* pNode);

	/// Destructor
	~GBlockSwitch();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_switch; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockSwitch"; }

	/// Returns a copy of this block
	virtual GBlockSwitch* clone() const override { return new GBlockSwitch(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};








/// A parameterized activation function (a.k.a. adaptive transfer function).
/// When alpha is 0, this activation function implements identity.
/// When alpha is positive, it bends upward. When alpha is negative, it bends downward.
/// Beta specifies approximately how big the bending curve is. When beta is 0, it bends on a point.
class GBlockHinge : public GBlock
{
public:
	/// General-purpose constructor
	/// Size specifies the number of units in this layer.
	GBlockHinge(size_t size);

	/// Copy constructor
	GBlockHinge(const GBlockHinge& that) : GBlock(that) {}

	GBlockHinge(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockHinge() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_hinge; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockHinge"; }

	/// Returns a copy of this block
	virtual GBlockHinge* clone() const override { return new GBlockHinge(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Updates the weights
	virtual void step(double learningRate, double momentum) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;

	/// Initialize the weights to implement the identity function.
	virtual void init_identity(GVec& weights) override;

	/// Return true iff the weights for this block make it implement identity.
	virtual bool is_identity(GVec& weights) override;

	/// Returns the number of weights this layer would have if its input and output sizes were adjusted by the given values.
	virtual size_t adjustedWeightCount(int inAdjustment, int outAdjustment) override;

	/// Add newInputs units to the input and newOutput units to the output of this block.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand) override;

	/// If input is not INVALID_INDEX, then the specified input will be dropped.
	/// If output is not INVALID_INDEX, then the specified output will be dropped.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft) override;
};





/// A parameterized activation function (a.k.a. adaptive transfer function).
/// When alpha is 0, this activation function implements identity.
/// When alpha is 1, it implements eponential-linear.
/// When alpha is -1, it implements the anty-symmetric exponetial-linear.
class GBlockElbow : public GBlock
{
public:
	/// General-purpose constructor
	/// Size specifies the number of units in this layer.
	GBlockElbow(size_t size);

	/// Copy constructor
	GBlockElbow(const GBlockElbow& that) : GBlock(that) {}

	GBlockElbow(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockElbow() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_elbow; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockElbow"; }

	/// Returns a copy of this block
	virtual GBlockElbow* clone() const override { return new GBlockElbow(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Updates the weights
	virtual void step(double learningRate, double momentum) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;

	/// Initialize the weights to implement the identity function.
	virtual void init_identity(GVec& weights) override;

	/// Return true iff the weights for this block make it implement identity.
	virtual bool is_identity(GVec& weights) override;

	/// Returns the number of weights this layer would have if its input and output sizes were adjusted by the given values.
	virtual size_t adjustedWeightCount(int inAdjustment, int outAdjustment) override;

	/// Add newInputs units to the input and newOutput units to the output of this block.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand) override;

	/// If input is not INVALID_INDEX, then the specified input will be dropped.
	/// If output is not INVALID_INDEX, then the specified output will be dropped.
	/// The weights in weightsAft will be set to reflect the adjustments.
	virtual void dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft) override;
};





/// A parameterized activation function (a.k.a. adaptive transfer function).
/// When alpha=0, it is the identity function.
/// As alpha approaches infinity, it approaches tanh.
class GBlockLeakyTanh : public GBlock
{
public:
	/// General-purpose constructor
	/// Size specifies the number of units in this layer.
	GBlockLeakyTanh(size_t size);

	/// Copy constructor
	GBlockLeakyTanh(const GBlockLeakyTanh& that) : GBlock(that) {}

	GBlockLeakyTanh(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockLeakyTanh() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_hinge; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockLeakyTanh"; }

	/// Returns a copy of this block
	virtual GBlockLeakyTanh* clone() const override { return new GBlockLeakyTanh(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Updates the weights
	virtual void step(double learningRate, double momentum) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};





/// A parameterized activation function (a.k.a. adaptive transfer function).
class GBlockSoftExp : public GBlock
{
protected:
	double m_beta;

public:
	/// General-purpose constructor
	/// When beta is 0, this activation function always approximates identity near the origin, but approximates e^x-1 when alpha is 1.
	/// When beta is 1, this activation function approximates e^x when alpha is 1.
	GBlockSoftExp(size_t size, double beta = 0.0);

	/// Copy constructor
	GBlockSoftExp(const GBlockSoftExp& that) : GBlock(that), m_beta(that.m_beta) {}

	GBlockSoftExp(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockSoftExp() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_softexp; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockSoftExp"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Returns a copy of this block
	virtual GBlockSoftExp* clone() const override { return new GBlockSoftExp(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};











/// A Probabilistically Activating Linear block.
/// This is an experimental block type.
class GBlockPAL : public GBlock
{
protected:
	GVec m_probs; // stores the probability for activating of each unit
	GRand m_rand;

public:
	/// General-purpose constructor
	GBlockPAL(size_t inputs, size_t outputs, GRand& rand);

	/// Copy constructor
	GBlockPAL(const GBlockPAL& that) : GBlock(that), m_probs(that.m_probs.size()), m_rand(that.m_rand) {}

	GBlockPAL(GDomNode* pNode, GRand& rand);

	/// Destructor
	virtual ~GBlockPAL() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_pal; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockPAL"; }

	/// Returns a copy of this block
	virtual GBlockPAL* clone() const override { return new GBlockPAL(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};








/// Number of inputs: 2^dims * vertexSizeIn.
/// Number of outputs: 2^dims * vertexSizeOut.
/// Each vertex is fully connected to all the vertices indicated by the edges of a hypercube.
class GBlockHypercube : public GBlock
{
protected:
	size_t m_dims;
	size_t m_vertexSizeIn;
	size_t m_vertexSizeOut;

public:
	/// General-purpose constructor
	GBlockHypercube(size_t dims, size_t vertexSizeIn, size_t vertexSizeOut);

	/// Copy constructor
	GBlockHypercube(const GBlockHypercube& that) : GBlock(that) {}

	/// Unmarshalling constructor
	GBlockHypercube(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockHypercube() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_hypercube; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockHypercube"; }

	/// Returns a copy of this block
	virtual GBlockHypercube* clone() const override { return new GBlockHypercube(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};







class GBlockOptional : public GBlock
{
public:
	/// General-purpose constructor
	GBlockOptional(size_t inputs, size_t outputs);

	/// Copy constructor
	GBlockOptional(const GBlockOptional& that) : GBlock(that) {}

	/// Unmarshalling constructor
	GBlockOptional(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockOptional() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_optional; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockOptional"; }

	/// Returns a copy of this block
	virtual GBlockOptional* clone() const override { return new GBlockOptional(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};






/// A special block that expects one categorical input value (encoded as a zero-indexed int casted as a double).
/// It learns the best encoding for each input value using each of its units.
/// It dynamically adjusts to accomodate input values not previously seen.
/// It does not backpropagate any error, so it is only suitable for inputs.
class GBlockCatIn : public GBlock
{
protected:
	size_t m_valueCount;

public:
	/// General-purpose constructor
	GBlockCatIn(size_t valueCount, size_t units);

	/// Copy constructor
	GBlockCatIn(const GBlockCatIn& that) : GBlock(that) {}

	/// Unmarshalling constructor
	GBlockCatIn(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockCatIn() {}

	/// Returns the type of this block
	virtual BlockType type() const override { return block_catin; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockCatIn"; }

	/// Returns a copy of this block
	virtual GBlockCatIn* clone() const override { return new GBlockCatIn(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Updates the weights
	virtual void step(double learningRate, double momentum) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Returns the number of elements for the gradient vector.
	virtual size_t gradCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;
};




/// A special block that uses a one-hot encoding to predict categorical outputs.
/// It trains to predict the most probably category by at least a specified margin.
class GBlockCatOut : public GBlockWeightless
{
protected:
	double m_margin;

public:
	GBlockCatOut(size_t categories, double margin = 0.3);
	GBlockCatOut(const GBlockCatOut& that) : GBlockWeightless(that), m_margin(that.m_margin) {}
	GBlockCatOut(const GDomNode* pNode);
	virtual ~GBlockCatOut() {}

	virtual BlockType type() const override { return block_catout; }
	virtual std::string name() const override { return "GBlockCatOut"; }
	virtual GBlockCatOut* clone() const override { return new GBlockCatOut(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Computes blame for this layer. (Assumes target is a one-dimensional index value casted as a double)
	virtual double computeBlame(const GVec& target);

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;
};





//  Here is a diagram of this LSTM unit.
//  C_t and h-t feed into this same block in the next time step.
//  x_t comes from the previous layer.
//  y_t feeds into the next layer.
//
//  C_t-1 ----(*)--------(+)----------------------> C_t
//             ^          ^                 |
//             |          |               [tanh]
//             +-->(1-)->(*)                |
//             |          |                 v
//             f          t        o------>(*)
//             ^          ^        ^        |   y_t (output)
//             |          |        |        |    ^
//        [logistic]   [tanh]  [logistic]   |    |
//             |          |        |        |    |
//  h_t-1 --------------------------        -------> h_t
//          ^
//          |
//         x_t (input)
//
/// A Long-short-term-memory block
class GBlockLSTM : public GBlock
{
protected:
	GVec n; // net input consisting of x, h, and c
	GVec f; // controls what stays in memory
	GVec t; // computed value that may (or may not) be written to memory
	GVec o; // controls when to output the memory
	GVec c; // incoming memory values
	GVec h; // incoming previous output
	GVec blame_h; // blame on the output of h
	GVec blame_c; // blame on the output of c
	GBlockLSTM* pPrevInstance; // backward around the ring
	GBlockLSTM* pNextInstance; // forward around the ring
	GBlockLSTM* pSpare; // chain (not ring) of unused instances

public:
	/// General-purpose constructor
	GBlockLSTM(size_t inputs, size_t outputs);

	/// Copy constructor
	GBlockLSTM(const GBlockLSTM& that);

	/// Unmarshalling constructor
	GBlockLSTM(GDomNode* pNode);

	/// Destructor
	virtual ~GBlockLSTM();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_lstm; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockLSTM"; }

	/// Returns a copy of this block
	virtual GBlockLSTM* clone() const override { return new GBlockLSTM(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp() override;

	/// Advances through time
	virtual void resetState() override;

	/// Advances through time
	virtual GBlock* advanceState(size_t unfoldedInstances) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp() override;

	/// Updates the gradient for updating the weights by gradient descent.
	virtual void updateGradient() override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand) override;

protected:
	void stepInTime();
	void forwardProp_instance(const GVec& weights);
	void backProp_instance(const GVec& weights, bool current);
	void updateGradient_instance(GVec& weights, GVec& gradient);
};











//
//
//
//
//
//
//
//              Layer
//
//
//
//
//
//
//

/// GNeuralNet contains GLayers stacked upon each other.
/// GLayer contains GBlocks concatenated beside each other. (GNeuralNet is a type of GBlock.)
/// Each GBlock is an array of differentiable network units (artificial neurons).
/// The user must add at least one GBlock to each GLayer.
class GLayer
{
protected:
	size_t input_count;
	size_t output_count;
	size_t weight_count;
	size_t grad_count;
	std::vector<GBlock*> m_blocks;

public:
	GVec outputBuf;
	GVec outBlameBuf;
	GVecWrapper output;
	GVecWrapper outBlame;

	GLayer();
	GLayer(const GLayer& that, GLayer* pPrevLayer);
	GLayer(GDomNode* pNode, GRand& rand);
	virtual ~GLayer();

	/// Marshal this object into a dom node.
	GDomNode* serialize(GDom* pDoc) const;

	/// Returns the number of blocks in this layer.
	size_t blockCount() const { return m_blocks.size(); }

	/// Returns a reference to the specified block.
	GBlock& block(size_t i) { return *m_blocks[i]; }
	const GBlock& block(size_t i) const { return *m_blocks[i]; }

	/// Adds a block of network units (artificial neurons) to this layer.
	/// inPos specifies the index of the first output from the previous layer that will feed into this block of units.
	void add(GBlock* pBlock, size_t inPos);

	/// Recounts the number of inputs, outputs, and weights in this layer.
	void recount();

	/// Returns the number of inputs that this layer consumes.
	size_t inputs() const;

	/// Returns the number of outputs that this layer produces.
	size_t outputs() const;

	/// Allocates output and outBlame buffers and binds each block to the appropriate parts of all these buffers.
	void bind(const GVec* pInput, GVec* pOutput, GVec* pOutBlame, GVec* pInBlame, GVec& _weights, GVec& _gradient);

	/// Binds to the specified input buffer
	void bindInput(const GVec& _input);

	/// Binds to the speicified inBlame buffer
	void bindInBlame(GVec& _inBlame);

	/// Resets the weights in all of the blocks in this layer
	void initWeights(GRand& rand);

	/// Returns the total number of weights in this layer
	size_t weightCount() const;

	/// Returns the total number of gradient elements necessary for updating this layer
	size_t gradCount() const;

	/// Evaluate the input, set the output.
	void forwardProp();

	/// Computes the out blame for all of the blocks in this layer
	double computeBlame(const GVec& target);

	/// Resets the state in any recurrent connections
	void resetState();

	/// Resets the state in any recurrent connections
	void advanceState(size_t unfoldedInstances);

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	void backProp();

	/// Updates the gradient for updating the weights by gradient descent.
	void updateGradient();

	/// Updates the gradient using only the sign of the inputs.
	void updateGradientNormalized();

	/// Adds the gradient scaled by the learningRate to the weights.
	void step(double learningRate, double momentum);

	/// Adds the gradient scaled by the learningRate to the weights and also jitters it.
	void step_jitter(double learningRate, double momentum, double jitter, GRand& rand);

	/// Puts a 1 in elements that correspond with bias weights and a 0 in all other elements.
	void biasMask(GVec& mask);
};






//
//
//
//
//
//
//
//             NeuralNet
//
//
//
//
//
//
//

/// GNeuralNet contains GLayers stacked upon each other.
/// GLayer contains GBlocks concatenated beside each other.
/// (GNeuralNet is a type of GBlock, so you can nest.)
/// Each GBlock is an array of differentiable network units (artificial neurons).
/// The user must add at least one GBlock to each GLayer.
class GNeuralNet : public GBlock
{
friend class GNeuralNetOptimizer;
protected:
	size_t m_weightCount;
	size_t m_gradCount;
	std::vector<GLayer*> m_layers;

public:
	/// General-purpose constructor
	GNeuralNet();

	/// Copy constructor
	GNeuralNet(const GNeuralNet& that);

	/// Deserializing constructor
	GNeuralNet(GDomNode* pNode, GRand& rand);

	/// Destructor
	virtual ~GNeuralNet();

	/// Deletes all the layers
	void deleteAllLayers();

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_neuralnet; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GNeuralNet"; }

	/// Retuns a copy of this neural network
	virtual GNeuralNet* clone() const override { return new GNeuralNet(*this); }

	/// Marshal this object into a dom node.
	GDomNode* serialize(GDom* pDoc) const override;

	/// Unmarshalls this object in place from a dom node.
	void deserialize(GDomNode* pNode, GRand& rand);

	/// Adds a block as a new layer to this neural network.
	GBlock* add(GBlock* pBlock);
	void add(GBlock* a, GBlock* b) { add(a); add(b); }
	void add(GBlock* a, GBlock* b, GBlock* c) { add(a); add(b, c); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d) { add(a); add(b, c, d); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d, GBlock* e) { add(a); add(b, c, d, e); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d, GBlock* e, GBlock* f) { add(a); add(b, c, d, e, f); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d, GBlock* e, GBlock* f, GBlock* g) { add(a); add(b, c, d, e, f, g); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d, GBlock* e, GBlock* f, GBlock* g, GBlock* h) { add(a); add(b, c, d, e, f, g, h); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d, GBlock* e, GBlock* f, GBlock* g, GBlock* h, GBlock* i) { add(a); add(b, c, d, e, f, g, h, i); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d, GBlock* e, GBlock* f, GBlock* g, GBlock* h, GBlock* i, GBlock* j) { add(a); add(b, c, d, e, f, g, h, i, j); }
	void add(GBlock* a, GBlock* b, GBlock* c, GBlock* d, GBlock* e, GBlock* f, GBlock* g, GBlock* h, GBlock* i, GBlock* j, GBlock* k) { add(a); add(b, c, d, e, f, g, h, i, j, k); }

	/// Attaches this neural network to the relevant buffers for it to operate on.
	virtual void bind(const GVec* pInput, GVec* pOutput, GVec* pOutBlame, GVec* pInBlame, GVec* pWeights, GVec* pGradient) override;

	/// Concatenates a block to the last (output-most) layer in this neural network.
	/// (inPos specifies the starting position of the inputs into this block.)
	GBlock* concat(GBlock* pBlock, size_t inPos);
/*
	/// Inserts a block into a new layer at the specified position.
	/// Requires the current weights and returns a new weight vector (which may be the
	/// same vector if pBlock has no weights). New weights will be initialized to identity for a linear layer.
	/// The caller is still responsible to delete pOldWeights (if it is not equal to the returned value)
	/// and initialize a new gradient vector.
	GVec* insert(size_t position, GBlock* pBlock, GVec* pOldWeights, GRand& rand);

	/// Drops a layer from this neural network. The layer must have the same number of inputs and outputs,
	/// and it cannot be the output layer.
	/// Requires the current weights and returns a new weight vector (which may be the
	/// same vector if pBlock has no weights).
	/// The caller is still responsible to delete pOldWeights (if it is not equal to the returned value)
	/// and initialize a new gradient vector.
	GVec* drop(size_t layer, GVec* pOldWeights);

	/// Adds a unit to each hidden layer.
	/// startLayer and layerCount specify which layers will be adjusted.
	/// Assumes all specified layers have the same size.
	GVec* increaseWidth(size_t newUnitsPerLayer, GVec* pWeights, size_t startLayer, size_t layerCount, GRand& rand);

	/// If each linear layer ignores at least one unit that feeds into it,
	/// then this removes an ignored unit from every layer to make the whole network
	/// thinner by one unit. If this condition is not met, then it returns nullptr.
	GVec* decrementWidth(GVec* pWeights, size_t startLayer, size_t layerCount);
*/
	/// An experimental regularization method that promotes anti-symmetry and mirroring in square linear layers
	void regularize_square(double lambda);

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
	virtual std::string to_str(bool includeWeights = false, bool includeActivations = false) const override;

	/// Same as to_str, but it lets the use specify a string to prepend to each line, and optionally includes the weights
	std::string to_str(const std::string& line_prefix, bool includeWeights = false, bool includeActivations = false) const;

	/// Returns the number of inputs this layer consumes
	virtual size_t inputs() const override { return m_layers[0]->inputs(); }

	/// Returns the number of outputs this layer produces
	virtual size_t outputs() const override { return outputLayer().outputs(); }

	/// Recounts the number of weights.
	void recount();

	/// Returns the number of weights.
	virtual size_t weightCount() const override;

	/// Returns the number of elements in the gradient.
	virtual size_t gradCount() const override;

	/// Copies all the layers from other.
	void copyTopology(const GNeuralNet& other);

	/// Initialize the weights. (This is called by init.)
	virtual void initWeights(GRand& rand) override;

	/// Allocates buffers needed for training, binds all the layers to these buffers,
	/// and to each other, initializes the weights with small random values, and
	/// initializes the gradient vector with zeros.
	void init(GRand& rand, GNeuralNet* pBase = nullptr);

	/// Initializes with pre-trained weights
	void init(GVec& weights);

	/// Measures the loss with respect to some data. Returns sum-squared error.
	/// if pOutSAE is not nullptr, then sum-absolute error will be storead where it points.
	/// As a special case, if labels have exactly one categorical column, then it will be assumed
	/// that the maximum output unit of this neural network represents a categorical prediction,
	/// and sum hamming loss will be returned.
	double measureLoss(const GMatrix& features, const GMatrix& labels, double* pOutSAE = nullptr);

	/// Evaluates the input vector. Returns an output vector.
	GVec& forwardProp(const GVec& input);

	/// Computes blame on the output of this neural network.
	virtual double computeBlame(const GVec& target) override;

	/// Resets the state in any recurrent connections
	virtual void resetState() override;

	/// Resets the state in any recurrent connections
	virtual GBlock* advanceState(size_t unfoldedInstances) override;

	/// Backpropagates the error. If inputBlame is non-null, the blame for the inputs will also be computed.
	void backpropagate(GVec* inputBlame = nullptr);

	/// Updates the gradient vector
	virtual void updateGradient() override;

	/// Update the gradient using only the sign of the input, ignoring the magnitude of the input.
	virtual void updateGradientNormalized() override;

	/// Adds the gradient scaled by the learning rate to the weights
	virtual void step(double learningRate, double momentum) override;

	/// Adds the gradient scaled by the learningRate to the weights and also jitters it.
	virtual void step_jitter(double learningRate, double momentum, double jitter, GRand& rand);

	/// Returns a mathematical expression of this neural network.
	/// (Currently only supports linear, tanh, and scalarProduct blocks in one-block layers.)
	std::string toEquation();

	/// Puts a 1 in elements that correspond with bias weights and a 0 in all other elements.
	virtual void biasMask(GVec& mask);

	/// Returns the index of the first weight associated with the specified layer.
	size_t layerStart(size_t layer);

	/// Run unit tests for this class
	static void test();

protected:

	/// Internal method to forward propagate. Assumes the input and output have already been hooked up.
	virtual void forwardProp() override;

	/// Internal method to backpropate error. Assumes inBlame and outBlame have already been hooked up.
	virtual void backProp() override;

	/// Like backProp, but it doesn't compute blame on the inputs.
	void backPropFast();
};







class GBlockResidual : public GNeuralNet
{
	virtual void forwardProp() override;

	virtual void backProp() override;
};








/// A thin wrapper around a GNeuralNet that implements the GIncrementalLearner interface.
class GNeuralNetLearner : public GIncrementalLearner
{
protected:
	GNeuralNet m_nn;
	GNeuralNetOptimizer* m_pOptimizer;

public:
	GNeuralNetLearner();
	//GNeuralNetLearner(const GDomNode* pNode);
	virtual ~GNeuralNetLearner();

	/// Returns a reference to the neural net that this class wraps
	GNeuralNet& nn() { return m_nn; }

	/// Lazily creates an optimizer for the neural net that this class wraps, and returns a reference to it.
	GNeuralNetOptimizer& optimizer();

	virtual void trainIncremental(const GVec &in, const GVec &out) override;
	virtual void trainSparse(GSparseMatrix &features, GMatrix &labels) override;

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Saves the model to a text file.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear() override;

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out) override;

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut) override;

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



/*
/// A class that facilitates training a neural network with an arbitrary optimization algorithm
class GNeuralNetTargetFunction : public GTargetFunction
{
protected:
	GNeuralNet& m_nn;
	const GMatrix& m_features;
	const GMatrix& m_labels;
	GRand& m_rand;

public:
	/// features and labels should be pre-filtered to contain only continuous values for the neural network.
	GNeuralNetTargetFunction(GNeuralNet& nn, const GMatrix& features, const GMatrix& labels, GRand& rand)
	: GTargetFunction(nn.weightCount()), m_nn(nn), m_features(features), m_labels(labels), m_rand(rand)
	{
	}

	virtual ~GNeuralNetTargetFunction() {}

	/// Copies the neural network weights into the vector.
	virtual void initVector(GVec& vector)
	{
		GRand rand(0);
		vector.copy(m_nn.weights);
	}

	/// Copies the vector into the neural network and measures sum-squared error.
	virtual double computeError(const GVec& vec)
	{
		return m_nn.measureLoss(vec, m_features, m_labels);
	}
};
*/



} // namespace GClasses

#endif // __GNEURALNET_H__
