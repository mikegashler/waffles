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
	GConstVecWrapper input;
	GVecWrapper output;
	GVecWrapper outBlame;
	GVecWrapper inBlame;

	enum BlockType
	{
		block_neuralnet,

		// activation functions
		block_identity,
		block_tanh,
		block_scaledtanh,
		block_logistic,
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
		block_conv,
		block_temperedlinear,
		block_pal,
		block_hinge,
		block_softexp,

		// weightless transfer
		block_scalarsum,
		block_scalarproduct,
		block_switch,
		block_spectral,

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
	GBlock(GDomNode* pNode);
	virtual ~GBlock() {}

	/// Returns the type of this block
	virtual BlockType type() const = 0;

	/// Returns the name of this block in the form of a string
	virtual std::string name() const = 0;

	/// Returns a string representation of this block
	virtual std::string to_str() const;

	/// Returns true iff this block is recurrent
	virtual bool isRecurrent() const { return false; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Unmarshalls the specified DOM node into a block object.
	static GBlock* deserialize(GDomNode* pNode, GRand& rand);

	/// Returns a copy of this block
	virtual GBlock* clone() const = 0;

	/// Returns the offset in the previous layer's output where values are fed as input to this block.
	size_t inPos() const { return m_inPos; }

	/// Sets the starting offset in the previous layer's output where values will be fed as input to this block.
	void setInPos(size_t n, GLayer* pPrevLayer);

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const { return inputCount; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const { return outputCount; }

	/// Evaluate the input, compute the output.
	virtual void forwardProp(const GVec& weights) = 0;

	/// Computes the blame on the output of this block.
	/// The default implementation computes it for SSE, but specialty blocks such as SoftMax may override it as needed.
	virtual void computeBlame(const GVec& target);

	/// Resets the state in any recurrent connections.
	virtual void resetState() {}

	/// Resets the state in any recurrent connections.
	virtual GBlock* advanceState(size_t unfoldedInstances) { return this; }

	/// Evaluate outBlame, update inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) = 0;

	/// Evaluate the input and outBlame, update the gradient of the weights.
	virtual void updateGradient(GVec& weights, GVec& gradient) = 0;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const = 0;

	/// Initialize the weights, usually with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) = 0;

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
	GBlockWeightless(GDomNode* pNode) : GBlock(pNode) {}
	virtual ~GBlockWeightless() {}

	virtual size_t weightCount() const override { return 0; }
	virtual void initWeights(GRand& rand, GVec& weights) override {}
	virtual void updateGradient(GVec& weights, GVec& gradient) override {}
};



/// The base class of blocks that apply an activation function, such as tanh, in an element-wise manner.
class GBlockActivation : public GBlockWeightless
{
public:
	GBlockActivation(size_t size);
	GBlockActivation(const GBlockActivation& that) : GBlockWeightless(that) {}
	GBlockActivation(GDomNode* pNode);
	virtual ~GBlockActivation() {}

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

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
	GBlockIdentity(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockTanh(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockScaledTanh(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockLogistic(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockBentIdentity(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockSigExp(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockGaussian(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual ~GBlockGaussian() {}
	virtual BlockType type() const override { return block_gaussian; }
	virtual std::string name() const override { return "GBlockGaussian"; }
	virtual GBlockGaussian* clone() const override { return new GBlockGaussian(*this); }
	virtual double eval(double x) const override { return std::exp(-(x * x)); }
	virtual double derivative(double x, double f_x) const override { return -2.0 * x * std::exp(-(x * x)); }
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
	GBlockSine(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockLeakyRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockSoftPlus(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockSoftRoot(GDomNode* pNode) : GBlockActivation(pNode) {}
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
	GBlockSoftMax(GDomNode* pNode) : GBlockWeightless(pNode) {}
	virtual ~GBlockSoftMax() {}
	virtual BlockType type() const override { return block_softmax; }
	virtual std::string name() const override { return "GBlockSoftMax"; }
	virtual GBlockSoftMax* clone() const override { return new GBlockSoftMax(*this); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& weights) override;

	/// Computes the blame with cross-entropy
	virtual void computeBlame(const GVec& target) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;
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
	GBlockSpectral(GDomNode* pNode);
	virtual ~GBlockSpectral() {}
	virtual BlockType type() const override { return block_spectral; }
	virtual std::string name() const override { return "GBlockSpectral"; }
	virtual GBlockSpectral* clone() const override { return new GBlockSpectral(*this); }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;
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

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;

	/// Computes weights using Ordinary Least Squares
	void ordinaryLeastSquares(const GMatrix& features, const GMatrix& labels, GVec& outWeights);
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;
};





/// A convolutional layer.
class GBlockConv : public GBlock
{
protected:
	size_t filterSize;
	GTensor tensorInput; // BUG: this member variable makes this block non-thread-safe.
	GTensor tensorFilter; // BUG: this member variable makes this block non-thread-safe.
	GTensor tensorOutput; // BUG: this member variable makes this block non-thread-safe.
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;

#ifndef NO_TEST_CODE
	static void test();
#endif // NO_TEST_CODE
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;
};








/// A parameterized activation function (a.k.a. adaptive transfer function).
class GBlockHinge : public GBlock
{
public:
	/// General-purpose constructor
	/// When alpha is 0, this activation function always approximates identity. When alpha is positive, it bends upward. When alpha is negative, it bends downward.
	/// Beta specifies approximately how big the bending curve is. When beta is 0, it bends on a point.
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;
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
	virtual void forwardProp(const GVec& weights) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;
};











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
	virtual void forwardProp(const GVec& weights) override;

	/// Advances through time
	virtual void resetState() override;

	/// Advances through time
	virtual GBlock* advanceState(size_t unfoldedInstances) override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& weights) override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Initialize the weights with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;

protected:
	void stepInTime();
	void forwardProp_instance(const GVec& weights);
	void backProp_instance(const GVec& weights, bool current);
	void updateGradient_instance(GVec& weights, GVec& gradient);
};











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
	std::vector<GBlock*> m_blocks;

public:
	GVec output;
	GVec outBlame;

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
	void add(GBlock* pBlock, GLayer* pPrevLayer, size_t inPos);

	/// Recounts the number of inputs, outputs, and weights in this layer.
	void recount();

	/// Returns the number of inputs that this layer consumes.
	size_t inputs() const;

	/// Returns the number of outputs that this layer produces.
	size_t outputs() const;

	/// Hooks up an input vector to this layer
	void setInput(const GVec& in);

	/// Hooks up an input vector to this layer
	void setInBlame(GVec& inBlame);

	/// Resets the weights in all of the blocks in this layer
	void initWeights(GRand& rand, GVec& weights);

	/// Returns the total number of weights in this layer
	size_t weightCount() const;

	/// Evaluate the input, set the output.
	void forwardProp(const GVec& weights);

	/// Computes the out blame for all of the blocks in this layer
	void computeBlame(const GVec& target);

	/// Resets the state in any recurrent connections
	void resetState();

	/// Resets the state in any recurrent connections
	void advanceState(size_t unfoldedInstances);

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	void backProp(const GVec& weights);

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	void updateGradient(GVec& weights, GVec& gradient);
};






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

	/// Adds a block as a new layer to this neural network.
	void add(GBlock* pBlock);
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

	/// Concatenates a block to the last (output-most) layer in this neural network.
	/// (inPos specifies the starting position of the inputs into this block.)
	void concat(GBlock* pBlock, size_t inPos = 0);

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

	/// Same as to_str, but it lets the use specify a string to prepend to each line
	std::string to_str(const std::string& line_prefix) const;

	/// Returns the number of inputs this layer consumes
	virtual size_t inputs() const override { return m_layers[0]->inputs(); }

	/// Returns the number of outputs this layer produces
	virtual size_t outputs() const override { return outputLayer().outputs(); }

	/// Recounts the number of weights.
	void recount();

	/// Returns the number of weights.
	virtual size_t weightCount() const override;

	/// Makes this object into a deep copy of pOther, including layers, nodes, settings and weights.
	void copyStructure(const GNeuralNet* pOther, GRand& rand);

	/// Initialize the weights, usually with small random values.
	virtual void initWeights(GRand& rand, GVec& weights) override;

	/// Measures the loss with respect to some data. Returns sum-squared error.
	/// if pOutSAE is not nullptr, then sum-absolute error will be storead where it points.
	/// As a special case, if labels have exactly one categorical column, then it will be assumed
	/// that the maximum output unit of this neural network represents a categorical prediction,
	/// and sum hamming loss will be returned.
	double measureLoss(const GVec& weights, const GMatrix& features, const GMatrix& labels, double* pOutSAE = nullptr);

	/// Evaluates the input vector. Returns an output vector.
	GVec& forwardProp(const GVec& weights, const GVec& input);

	/// Computes blame on the output of this neural network.
	virtual void computeBlame(const GVec& target) override;

	/// Resets the state in any recurrent connections
	virtual void resetState() override;

	/// Resets the state in any recurrent connections
	virtual GBlock* advanceState(size_t unfoldedInstances) override;

	/// Backpropagates the error. If inputBlame is non-null, the blame for the inputs will also be computed.
	void backpropagate(const GVec& weights, GVec* inputBlame = nullptr);

	/// Updates the gradient vector
	virtual void updateGradient(GVec& weights, GVec& gradient) override;

	/// Returns a mathematical expression of this neural network.
	/// (Currently only supports linear, tanh, and scalarProduct blocks in one-block layers.)
	std::string toEquation(const GVec& weights);

#ifndef MIN_PREDICT
	static void test();
#endif

protected:

	/// Internal method to forward propagate. Assumes the input and output have already been hooked up.
	virtual void forwardProp(const GVec& weights) override;

	/// Internal method to backpropate error. Assumes inBlame and outBlame have already been hooked up.
	virtual void backProp(const GVec& weights) override;

	/// Like backProp, but it doesn't compute blame on the inputs.
	void backPropFast(const GVec& weights);
};









/// A thin wrapper around a GNeuralNet that implements the GIncrementalLearner interface.
class GNeuralNetLearner : public GIncrementalLearner
{
protected:
	GNeuralNet m_nn;
	GNeuralNetOptimizer* m_pOptimizer;

public:
	GNeuralNetLearner();
	GNeuralNetLearner(const GDomNode* pNode);
	virtual ~GNeuralNetLearner();

	/// Returns a reference to the neural net that this class wraps
	GNeuralNet& nn() { return m_nn; }

	/// Lazily creates an optimizer for the neural net that this class wraps, and returns a reference to it.
	GNeuralNetOptimizer& optimizer();

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
		m_nn.initWeights(m_rand, vector);
	}

	/// Copies the vector into the neural network and measures sum-squared error.
	virtual double computeError(const GVec& vec)
	{
		return m_nn.measureLoss(vec, m_features, m_labels);
	}
};




} // namespace GClasses

#endif // __GNEURALNET_H__
