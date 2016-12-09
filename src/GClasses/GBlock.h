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

#include "GMatrix.h"
#include <vector>
#include <ostream>
#include <cmath>

namespace GClasses {

class GContextRecurrent;
class GContextRecurrentInstance;


/// Represents a block of network units (artificial neurons) in a neural network.
class GBlock
{
protected:
	size_t m_inPos;

public:
	enum BlockType
	{
		block_neuralnet,
		block_identity,
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
		block_scalarproduct,
		block_scalarsum,
		block_switch,
		block_maxout,
		block_softmax,
		block_restrictedboltzmannmachine,
		block_convolutional1d,
		block_convolutional2d,
		block_maxpooling,
		block_lstm,
		block_gru,
	};

	GBlock();
	GBlock(GDomNode* pNode);
	virtual ~GBlock() {}

	/// Returns the type of this block
	virtual BlockType type() const = 0;

	/// Returns true iff this block operates only on individual elements
	virtual bool elementWise() const { return false; }

	/// Returns true iff this block is recurrent
	virtual bool isRecurrent() const { return false; }

	/// Returns true iff this block does its computations in parallel on a GPU.
	virtual bool usesGPU() { return false; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// Unmarshalls the specified DOM node into a block object.
	static GBlock* deserialize(GDomNode* pNode);

	/// Returns the offset in the previous layer's output where values are fed as input to this block.
	size_t inPos() const { return m_inPos; }

	/// Sets the starting offset in the previous layer's output where values will be fed as input to this block.
	void setInPos(size_t n) { m_inPos = n; };

	/// Makes a string representation of this block
	virtual std::string to_str() const = 0;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) = 0;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const = 0;

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const = 0;

	/// Evaluates the input, sets the output.
	virtual void forwardProp(const GVec& input, GVec& output) const = 0;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const = 0;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const = 0;

	/// Serialize the weights in this block into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const = 0;

	/// Deserialize from a vector to the weights in this block. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) = 0;

	/// Copy the weights from pSource to this block. (Assumes pSource is the same type of block.)
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




/// The base class of blocks that have no weights
class GBlockWeightless : public GBlock
{
public:
	GBlockWeightless() : GBlock() {}
	GBlockWeightless(GDomNode* pNode) : GBlock(pNode) {}
	virtual ~GBlockWeightless() {}

	virtual size_t weightCount() const override { return 0; }
	virtual size_t weightsToVector(double* pOutVector) const override { return 0; }
	virtual size_t vectorToWeights(const double* pVector) override { return 0; }
	virtual void copyWeights(const GBlock* pSource) override {}
	virtual void resetWeights(GRand& rand) override {}
	virtual void perturbWeights(GRand& rand, double deviation) override {}
	virtual void maxNorm(double min, double max) override {}
	virtual void scaleWeights(double factor, bool scaleBiases) override {}
	virtual void diminishWeights(double amount, bool regularizeBiases) override {}
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override {}
	virtual void step(double learningRate, const GVec &gradient) override {}
};





/// Treats the input as two concatenated vectors.
/// Adds each corresponding pair of values together to produce the output.
class GBlockScalarSum : public GBlockWeightless
{
protected:
	size_t m_outputCount;

public:
	/// General-purpose constructor.
	GBlockScalarSum(size_t outputs);

	/// Deserializing constructor
	GBlockScalarSum(GDomNode* pNode);
	~GBlockScalarSum();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_scalarsum; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_outputCount * 2; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_outputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// A special forward prop that accepts two vectors of the same size.
	void forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// A convenience version of backProp that mirrors forwardProp2.
	void backProp2(const GVec& outBlame, GVec& inBlame1, GVec& inBlame2) const;
};





/// Treats the input as two concatenated vectors.
/// Multiplies each corresponding pair of values together to produce the output.
class GBlockScalarProduct : public GBlockWeightless
{
protected:
	size_t m_outputCount;

public:
	/// General-purpose constructor.
	GBlockScalarProduct(size_t outputs);

	/// Deserializing constructor
	GBlockScalarProduct(GDomNode* pNode);
	~GBlockScalarProduct();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_scalarproduct; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_outputCount * 2; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_outputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// A special forward prop that accepts two vectors of the same size.
	void forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// A convenience version of backProp pthat mirrors forwardProp2.
	void backProp2(const GVec& input1, const GVec& input2, const GVec& outBlame, GVec& inBlame1, GVec& inBlame2) const;
};






/// Treats the input as three concatenated vectors: a, b, and c.
/// (The values in 'a' typically fall in the range [0,1].)
/// The output is computed element-wise as a*b + (1-a)*c.
class GBlockSwitch : public GBlockWeightless
{
protected:
	size_t m_outputCount;

public:
	/// General-purpose constructor.
	GBlockSwitch(size_t outputs);

	/// Deserializing constructor
	GBlockSwitch(GDomNode* pNode);
	~GBlockSwitch();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_switch; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_outputCount * 3; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_outputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// A special forward prop that accepts three vectors of the same size.
	void forwardProp3(const GVec& inA, const GVec& inB, const GVec& inC, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// A convenience version of backProp that mirrors forwardProp3
	void backProp3(const GVec& inA, const GVec& inB, const GVec& inC, const GVec& outBlame, GVec& inBlameA, GVec& inBlameB, GVec& inBlameC) const;
};





class GMaxPooling2D : public GBlockWeightless
{
protected:
	size_t m_inputCols;
	size_t m_inputRows;
	size_t m_inputChannels;
	size_t m_regionSize;

public:
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

	/// Returns the type of this block
	virtual BlockType type() const override { return block_maxpooling; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_inputRows * m_inputCols * m_inputChannels; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_inputRows * m_inputCols * m_inputChannels / (m_regionSize * m_regionSize); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;
};










/// The base class of blocks that apply an activation function, such as tanh, in an element-wise manner.
class GBlockActivation : public GBlockWeightless
{
protected:
	size_t m_units;

public:
	GBlockActivation(size_t size = 0);
	GBlockActivation(GDomNode* pNode);

	/// Returns true iff this block operates only on individual elements
	virtual bool elementWise() const { return true; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_units; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_units; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Evaluates the activation function
	virtual double eval(double x) const = 0;

	/// Evaluates the derivative of the activation function.
	/// x is the net input, and f_x is the output activation--the value obtained by calling eval(x).
	virtual double derivative(double x, double f_x) const = 0;
};



/// The identity function. Serves as a pass-through block of units in a neural network.
class GBlockIdentity : public GBlockActivation
{
public:
	GBlockIdentity(size_t size = 0) : GBlockActivation(size) {}
	GBlockIdentity(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_identity; }
	virtual double eval(double x) const override { return x; }
	virtual double derivative(double x, double f_x) const override { return 1.0; }
};




/// An element-wise nonlinearity block
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
	GBlockTanh(size_t size = 0) : GBlockActivation(size) {}
	GBlockTanh(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_tanh; }
	virtual double eval(double x) const override { return std::tanh(x); }
	virtual double derivative(double x, double f_x) const override { return 1.0 - (f_x * f_x); }
};



/// An element-wise nonlinearity block
class GBlockLogistic : public GBlockActivation
{
public:
	GBlockLogistic(size_t size = 0) : GBlockActivation(size) {}
	GBlockLogistic(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_logistic; }
	virtual double eval(double x) const override
	{
		if(x >= 700.0) // Don't trigger a floating point exception
			return 1.0;
		else if(x < -700.0) // Don't trigger a floating point exception
			return 0.0;
		else return 1.0 / (std::exp(-x) + 1.0);
		
	}
	virtual double derivative(double x, double f_x) const override { return f_x * (1.0 - f_x); }
};



#define BEND_AMOUNT 0.5
#define BEND_SIZE 0.5
/// An element-wise nonlinearity block
class GBlockBentIdentity : public GBlockActivation
{
public:
	GBlockBentIdentity(size_t size = 0) : GBlockActivation(size) {}
	GBlockBentIdentity(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_bentidentity; }
	virtual double eval(double x) const override { return BEND_AMOUNT * (std::sqrt(x * x + BEND_SIZE * BEND_SIZE) - BEND_SIZE) + x; }
	virtual double derivative(double x, double f_x) const override { return BEND_AMOUNT * x / std::sqrt(x * x + BEND_SIZE * BEND_SIZE) + 1.0; }
};



/// An element-wise nonlinearity block
/// This activation function forms a sigmoid shape by splicing exponential and logarithmic functions together.
class GBlockSigExp : public GBlockActivation
{
public:
	GBlockSigExp(size_t size = 0) : GBlockActivation(size) {}
	GBlockSigExp(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_sigexp; }
	virtual double eval(double x) const override { return (x <= 0.0 ? exp(x) - 1.0 : std::log(x + 1.0)); }
	virtual double derivative(double x, double f_x) const override { return (x <= 0.0 ? std::exp(x) : 1.0 / (x + 1.0)); }
};



/// An element-wise nonlinearity block
class GBlockGaussian : public GBlockActivation
{
public:
	GBlockGaussian(size_t size = 0) : GBlockActivation(size) {}
	GBlockGaussian(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_gaussian; }
	virtual double eval(double x) const override { return std::exp(-(x * x)); }
	virtual double derivative(double x, double f_x) const override { return -2.0 * x * std::exp(-(x * x)); }
};



/// An element-wise nonlinearity block
class GBlockSine : public GBlockActivation
{
public:
	GBlockSine(size_t size = 0) : GBlockActivation(size) {}
	GBlockSine(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_sine; }
	virtual double eval(double x) const override { return std::sin(x); }
	virtual double derivative(double x, double f_x) const override { return std::cos(x); }
};




/// An element-wise nonlinearity block
class GBlockRectifier : public GBlockActivation
{
public:
	GBlockRectifier(size_t size = 0) : GBlockActivation(size) {}
	GBlockRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_rectifier; }
	virtual double eval(double x) const override { return std::max(0.0, x); }
	virtual double derivative(double x, double f_x) const override { return (x >= 0.0 ? 1.0 : 0.0); }
};




/// An element-wise nonlinearity block
class GBlockLeakyRectifier : public GBlockActivation
{
public:
	GBlockLeakyRectifier(size_t size = 0) : GBlockActivation(size) {}
	GBlockLeakyRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_leakyrectifier; }
	virtual double eval(double x) const override { return x >= 0.0 ? x : 0.01 * x; }
	virtual double derivative(double x, double f_x) const override { return x >= 0.0 ? 1.0 : 0.01; }
};



/// An element-wise nonlinearity block
// (Note: A similar, but less well-known function is the integral of the logistic function. I think it is slightly faster to compute.)
class GBlockSoftPlus : public GBlockActivation
{
public:
	GBlockSoftPlus(size_t size = 0) : GBlockActivation(size) {}
	GBlockSoftPlus(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_softplus; }
	virtual double eval(double x) const override { return x > 500 ? x : log(1.0 + exp(x)); }
	virtual double derivative(double x, double f_x) const override { return 1.0 / (1.0 + exp(-x)); }
};




/// An element-wise nonlinearity block.
/// This function is shaped like a sigmoid, but with both a co-domain and domain
/// that spans the continuous values. At very negative values,
/// it is shaped like y=-sqrt(-2x). Near zero, it is shaped
/// like y=x. At very positive values, it is shaped like y=sqrt(2x).
class GBlockSoftRoot : public GBlockActivation
{
public:
	GBlockSoftRoot(size_t size = 0) : GBlockActivation(size) {}
	GBlockSoftRoot(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_softroot; }
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
};






/// Standard fully-connected block of weights. Often followed by a GBlockActivation.
class GBlockLinear : public GBlock
{
protected:
	GMatrix m_weights; // An (inputs+1)-by-outputs matrix of weights. The last row contains the bias values.

public:
	GBlockLinear(size_t outputs, size_t inputs = 0);
	GBlockLinear(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_linear; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.rows() - 1; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// A convenience method that concatenates two vectors before feeding into this block
	void forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// A convenience method that mirrors forwardProp2.
	void backProp2(const GVec& outBlame, GVec& inBlame1, GVec& inBlame2) const;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// A convenience method that goes with forwardProp2 and backProp2.
	void updateGradient2(const GVec& in1, const GVec& in2, const GVec& outBlame, GVec &gradient) const;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Applies contractive regularization to the weights in this block.
	void contractWeights(double factor, bool contractBiases, const GVec& output);

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this block into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this block. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this block. (Assumes pSource is the same type of block.)
	virtual void copyWeights(const GBlock* pSource) override;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise. The
	/// default values apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max) override;

	/// Multiplies all the weights by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Moves all weights in the direction of zero by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Returns the bias vector of this block.
	GVec& bias() { return m_weights.back(); }

	/// Returns the bias vector of this block.
	const GVec& bias() const { return m_weights.back(); }

	/// Get the entire weights matrix
	GMatrix &weights() { return m_weights; }

	/// Get the entire weights matrix
	const GMatrix &weights() const { return m_weights; }

	/// Transforms the weights of this block by the specified transformation matrix and offset vector.
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

	/// Returns the type of this block
	virtual BlockType type() const override { return block_maxout; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.rows(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Multiplies all the weights in this block by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this block into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this block. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this block. (Assumes pSource is the same type of block.)
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

	/// Returns a reference to the weights matrix of this block
	GMatrix& weights() { return m_weights; }

	/// Returns a reference to the weights matrix of this block
	const GMatrix& weights() const { return m_weights; }

	/// Returns the bias vector of this block.
	GVec& bias() { return m_bias[0]; }

	/// Returns the bias vector of this block.
	const GVec& bias() const { return m_bias[0]; }

	/// Sets the weights of this block to make it weakly approximate the identity function.
	/// start specifies the first unit whose incoming weights will be adjusted.
	/// count specifies the maximum number of units whose incoming weights are adjusted.
	void setWeightsToIdentity(size_t start = 0, size_t count = (size_t)-1);

	/// Transforms the weights of this block by the specified transformation matrix and offset vector.
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

	/// Returns the type of this block
	virtual BlockType type() const override { return block_restrictedboltzmannmachine; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.cols(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.rows(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Feed a vector backwards through this block.
	void feedBackward(const GVec& output, GVec& input) const;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Multiplies all the weights in this block by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this block into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this block. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this block. (Assumes pSource is the same type of block.)
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

	/// Returns a reference to the weights matrix of this block
	GMatrix& weights() { return m_weights; }

	/// Returns the bias for the hidden end of this block.
	GVec& bias() { return m_bias; }

	/// Returns the bias for the hidden end of this block.
	const GVec& bias() const { return m_bias; }

	/// Returns the bias for the visible end of this block.
	GVec& biasReverse() { return m_biasReverse[0]; }
	const GVec& biasReverse() const { return m_biasReverse[0]; }

	/// Performs binomial resampling of the activation values on the output end of this block.
	void resampleHidden(GRand& rand, GVec& output);

	/// Performs binomial resampling of the activation values on the input end of this block.
	void resampleVisible(GRand& rand, GVec& input);

	/// Draws a sample observation from "iters" iterations of Gibbs sampling.
	/// The resulting sample is placed in activationReverse(), and the corresponding
	/// encoding will be in activation().
	void drawSample(GRand& rand, size_t iters, GVec& output, GVec& input);

	
/*  *** Note that these two commented-out methods are pretty-much the whole point of RBMs, so this class is pretty-much useless until they are restored. ***
	/// Returns the free energy of this block. Assumes that a pattern has already been set in the 
	double freeEnergy(const GVec& visibleSample);

	/// Refines this block by contrastive divergence.
	/// pVisibleSample should point to a vector of inputs that will be presented to this block.
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

	/// Returns the type of this block
	virtual BlockType type() const override { return block_convolutional1d; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this block
	virtual std::string to_str() const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
        virtual size_t inputs() const override { return m_inputSamples * m_inputChannels; }

	/// Returns the number of outputs this block produces
        virtual size_t outputs() const override { return m_outputSamples * m_inputChannels * m_kernelsPerChannel; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

	/// Multiplies all the weights in this block by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Diminishes all the weights (that is, moves them in the direction toward 0) by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this block into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this block. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this block. (Assumes pSource is the same type of block.)
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

	/// Constructor that will automatically use the upstream convolutional block when added to a neural network
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
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;


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






/// Base class of recurrent blocks.
//
// A recurrent block unfolded through time:
//
//        output_t              output_t_1              output_t+2
//           ^                     ^                       ^
//         __|__                 __|__                   __|__
//        |     |               |     |                 |     |
// ... >--|     |--> state_t >--|     |--> state_t+1 >--|     |--> ...
//        |_____|               |_____|                 |_____|
//           |                     |                       |
//           ^                     ^                       ^
//         input_t               input_t+1               input_t+2
//
class GBlockRecurrent : public GBlock
{
public:
	GBlockRecurrent() : GBlock() {}
	GBlockRecurrent(GDomNode* pNode) : GBlock(pNode) {}
	virtual ~GBlockRecurrent() {}

	/// Returns true.
	virtual bool isRecurrent() const { return true; }

	/// Returns the number of instances of this block unfolded through time that will be used during training.
	virtual size_t depth() = 0;

	/// Returns a new context object for this recurrent block.
	/// The recurrent state should be initialized to the starting state.
	virtual GContextRecurrentInstance* newContext() = 0;

protected:
	/// Deliberately protected.
	/// Throws an exception telling you to call GContextRecurrent::forwardProp instead.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Deliberately protected.
	/// Throws an exception telling you to call GContextRecurrent::backProp instead.
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Deliberately protected.
	/// Throws an exception telling you to call GContextRecurrent::updateGradient instead.
	virtual void updateGradient(const GVec& input, const GVec& outBlame, GVec& gradient) const override;
};

/// A special context object for recurrent blocks
class GContextRecurrent
{
public:
	GBlockRecurrent& m_block;
	GContextRecurrentInstance* m_pInitialInstance;
	GVec m_emptyBlame;
	GVec m_bogusBlame;
	std::vector<GContextRecurrentInstance*> m_contextHistory; // Stores recent context objects
	std::vector<GContextRecurrentInstance*> m_contextSpares; // Stores unused context objects
	std::vector<GVec*> m_inputHistory; // Stores recent inputs into the recurrent blocks
	std::vector<GVec*> m_inputSpares; // Stores unused vector objects
	size_t m_pos; // Position of the oldest recorded input vector in the history

	GContextRecurrent(GBlockRecurrent& block);
	virtual ~GContextRecurrent();

	void resetState();

	void forwardProp(const GVec& input, GVec& output);

	void forwardPropThroughTime(const GVec& input, GVec& output);

	void backPropThroughTime(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame);

	void updateGradient(const GVec& input, const GVec& outBlame, GVec& gradient) const;

};

/// A single instance of a recurrent context that has been unfolded through time
class GContextRecurrentInstance
{
public:
	virtual ~GContextRecurrentInstance() {}

	virtual void clearBlame() = 0;
	virtual void forwardProp(GContextRecurrentInstance* pPrev, const GVec& input, GVec& output) = 0;
	virtual void backProp(GContextRecurrentInstance* pPrev, const GVec& outBlame, GVec& inBlame) = 0;
	virtual void updateGradient(GContextRecurrentInstance* prev, const GVec& input, GVec& gradient) const = 0;
};



/// A classic Long Short Term Memory block (with coupled forget and input gates)
class GBlockLSTM : public GBlockRecurrent
{
friend class GContextLSTM;
protected:
	GBlockScalarProduct m_product;
	GBlockSwitch m_switch;
	GBlockLogistic m_logistic;
	GBlockTanh m_tanh;
	GBlockLinear m_write;
	GBlockLinear m_val;
	GBlockLinear m_read;

public:
	GBlockLSTM(size_t outputs, size_t inputs = 0);
	GBlockLSTM(GDomNode* pNode);
	virtual ~GBlockLSTM();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_gru; }

	/// Returns the number of instances of this block unfolded through time that will be used during training.
	virtual size_t depth() override { return 4; }

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_write.outputs(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_write.inputs(); }

	/// Makes a new context object for this block
	virtual GContextRecurrentInstance* newContext() override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Returns a string representation of this block
	virtual std::string to_str() const override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this block into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this block. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this block. (Assumes pSource is the same type of block.)
	virtual void copyWeights(const GBlock* pSource) override;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise. The
	/// default values apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max) override;

	/// Multiplies all the weights by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Moves all weights in the direction of zero by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

#ifndef MIN_PREDICT
	static void test();
#endif

};

/// Context class for LSTM blocks
class GContextLSTM : public GContextRecurrentInstance
{
public:
	const GBlockLSTM& m_block;
	GVec m_c;
	GVec m_h;
	GVec m_f;
	GVec m_t;
	GVec m_o;
	GVec m_blamec;
	GVec m_blameh;
	GVec m_blamef;
	GVec m_blamet;
	GVec m_blameo;
	GVec m_buf1;
	GVec m_buf2;

	GContextLSTM(GBlockLSTM& block);

	virtual void forwardProp(GContextRecurrentInstance* pPrev, const GVec& input, GVec& output) override;
	virtual void clearBlame() override;
	virtual void backProp(GContextRecurrentInstance* pPrev, const GVec& outBlame, GVec& inBlame) override;
	virtual void updateGradient(GContextRecurrentInstance* prev, const GVec& input, GVec& gradient) const override;
};





/// A block of Gated Recurrent Units
class GBlockGRU : public GBlockRecurrent
{
friend class GContextGRU;
protected:
	GBlockScalarProduct m_product;
	GBlockSwitch m_switch;
	GBlockLogistic m_logistic;
	GBlockTanh m_tanh;
	GBlockLinear m_update;
	GBlockLinear m_remember;
	GBlockLinear m_val;

public:
	GBlockGRU(size_t outputs, size_t inputs = 0);
	GBlockGRU(GDomNode* pNode);
	virtual ~GBlockGRU();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_gru; }

	/// Returns the number of instances of this block unfolded through time that will be used during training.
	virtual size_t depth() override { return 4; }

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_update.outputs(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_update.inputs(); }

	/// Makes a new context object for this block
	GContextRecurrentInstance* newContext();

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Returns a string representation of this block
	virtual std::string to_str() const override;

	/// Returns the number of double-precision elements necessary to serialize the weights of this block into a vector.
	virtual size_t weightCount() const override;

	/// Serialize the weights in this block into a vector. Return the number of elements written.
	virtual size_t weightsToVector(double* pOutVector) const override;

	/// Deserialize from a vector to the weights in this block. Return the number of elements consumed.
	virtual size_t vectorToWeights(const double* pVector) override;

	/// Copy the weights from pSource to this block. (Assumes pSource is the same type of block.)
	virtual void copyWeights(const GBlock* pSource) override;

	/// Initialize the weights with small random values.
	virtual void resetWeights(GRand& rand) override;

	/// Perturbs the weights that feed into the specifed units with Gaussian noise. The
	/// default values apply the perturbation to all units.
	virtual void perturbWeights(GRand& rand, double deviation) override;

	/// Scales weights if necessary such that the manitude of the weights (not including the bias) feeding into each unit are >= min and <= max.
	virtual void maxNorm(double min, double max) override;

	/// Multiplies all the weights by the specified factor.
	virtual void scaleWeights(double factor, bool scaleBiases) override;

	/// Moves all weights in the direction of zero by the specified amount.
	virtual void diminishWeights(double amount, bool regularizeBiases) override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;
};

/// Context class for GRU blocks
class GContextGRU : public GContextRecurrentInstance
{
public:
	const GBlockGRU& m_block;
	GVec m_h;
	GVec m_z;
	GVec m_r;
	GVec m_t;
	GVec m_blameh;
	GVec m_blamez;
	GVec m_blamer;
	GVec m_blamet;
	GVec m_buf1;
	GVec m_buf2;

	GContextGRU(GBlockGRU& block);

	virtual void forwardProp(GContextRecurrentInstance* pPrev, const GVec& input, GVec& output) override;
	virtual void clearBlame() override;
	virtual void backProp(GContextRecurrentInstance* pPrev, const GVec& outBlame, GVec& inBlame) override;
	virtual void updateGradient(GContextRecurrentInstance* prev, const GVec& input, GVec& gradient) const override;

};



} // namespace GClasses

#endif // __GBLOCK_H__
