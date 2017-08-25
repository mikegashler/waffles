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
#include "GSparseMatrix.h"
#include "GCudaMatrix.h"
#include <vector>
#include <ostream>
#include <cmath>

namespace GClasses {

class GContext;
class GContextRecurrent;
class GContextRecurrentInstance;
class GNeuralNet;


/// The base class for the buffers that a thread needs to
/// use (train or predict with) a neural network component.
class GContext
{
public:
	GRand& m_rand;
#ifdef GCUDA
	GVec m_scratchIn;
	GVec m_scratchOut;
	GVec m_scratchInBlame;
	GVec m_scratchOutBlame;
	GVec m_scratchGradient;
#endif

	GContext(GRand& rand) : m_rand(rand) {};
	virtual ~GContext() {}

	/// Resets the state of all recurrent blocks.
	/// (This is called whenever a recurrent neural network begins with a new sequence,
	/// either for training or testing.)
	virtual void resetState() = 0;

#ifdef GCUDA
	virtual GCudaEngine& cudaEngine() = 0;
#endif
};







/// Represents a block of network units (artificial neurons) in a neural network.
class GBlock
{
protected:
	size_t m_inPos;

public:
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

		// weights transfer
		block_linear,
		block_pal,
		block_sparse,
		block_featureselector,
		block_fuzzy,
		block_restrictedboltzmannmachine,
		block_convolutional1d,
		block_convolutional2d,
		block_hinge,
		block_softexp,

		// weightless transfer
		block_scalarsum,
		block_scalarproduct,
		block_switch,
		block_maxpooling,
		block_allpairings,

		// recurrent
		block_lstm,
		block_gru,

		// still needed
		// block_softmax,
		// block_maxout,
		// block_batch_normalization,
		// block_drop_out,
		// block_drop_connect,
	};

	GBlock();
	GBlock(GDomNode* pNode);
	virtual ~GBlock() {}

	/// Returns the type of this block
	virtual BlockType type() const = 0;

	/// Returns the name of this block in the form of a string
	virtual std::string name() const = 0;

	/// Returns a string representation of this block
	virtual std::string to_str() const;

	/// Returns true iff this block operates only on individual elements
	virtual bool elementWise() const { return false; }

	/// Returns true iff this block is recurrent
	virtual bool isRecurrent() const { return false; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// Unmarshalls the specified DOM node into a block object.
	static GBlock* deserialize(GDomNode* pNode);

	/// Returns the offset in the previous layer's output where values are fed as input to this block.
	size_t inPos() const { return m_inPos; }

	/// Sets the starting offset in the previous layer's output where values will be fed as input to this block.
	void setInPos(size_t n) { m_inPos = n; };

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) = 0;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const = 0;

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const = 0;

	/// Evaluates the input, sets the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const = 0;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const = 0;

	/// Evaluate the input and outBlame, update the gradient for updating the weights by gradient descent.
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec& gradient) const = 0;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec& gradient) = 0;

#ifdef GCUDA
	/// Uploads weights onto the GPU
	virtual void uploadCuda()
	{
		// The default implementation just keeps weights on the CPU.
	}

	/// Downloads weights from the GPU
	virtual void downloadCuda()
	{
		// The default implementation just keeps weights on the CPU.
	}

	virtual void forwardPropCuda(GContext& ctx, const GCudaVector& input, GCudaVector& output) const
	{
		input.download(ctx.m_scratchIn);
		ctx.m_scratchOut.resize(output.size());
		forwardProp(ctx, ctx.m_scratchIn, ctx.m_scratchOut);
		output.upload(ctx.m_scratchOut);
	}

	virtual void backPropCuda(GContext& ctx, const GCudaVector& input, const GCudaVector& output, const GCudaVector& outBlame, GCudaVector& inBlame) const
	{
		input.download(ctx.m_scratchIn);
		output.download(ctx.m_scratchOut);
		outBlame.download(ctx.m_scratchOutBlame);
		ctx.m_scratchInBlame.resize(input.size());
		backProp(ctx, ctx.m_scratchIn, ctx.m_scratchOut, ctx.m_scratchOutBlame, ctx.m_scratchInBlame);
		inBlame.upload(ctx.m_scratchInBlame);
	}

	virtual void updateGradientCuda(GContext& ctx, const GCudaVector& input, const GCudaVector& outBlame, GCudaVector& gradient) const
	{
		input.download(ctx.m_scratchIn);
		outBlame.download(ctx.m_scratchOutBlame);
		gradient.download(ctx.m_scratchGradient);
		updateGradient(ctx, ctx.m_scratchIn, ctx.m_scratchOutBlame, ctx.m_scratchGradient);
		gradient.upload(ctx.m_scratchGradient);
	}

	virtual void stepCuda(GContext& ctx, double learningRate, const GCudaVector& gradient)
	{
		gradient.download(ctx.m_scratchGradient);
		step(learningRate, ctx.m_scratchGradient);
	}
#endif

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

	/// Drops the specified unit from this block
	virtual void dropUnit(size_t index)
	{
		throw Ex("Sorry, not implemented yet");
	}

	/// Clones the specified unit in this block
	virtual void cloneUnit(size_t index)
	{
		throw Ex("Sorry, not implemented yet");
	}

protected:
	GDomNode* baseDomNode(GDom* pDoc) const;

	/// Exercises some basic functionality that all blocks have in common
	void basicTest();
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
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override {}
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

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockScalarSum"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_outputCount * 2; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_outputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// A special forward prop that accepts two vectors of the same size.
	void forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

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

	/// Returns the name of this block
	virtual std::string name() const { return "GBlockScalarProduct"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_outputCount * 2; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_outputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// A special forward prop that accepts two vectors of the same size.
	void forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

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

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockSwitch"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_outputCount * 3; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_outputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// A special forward prop that accepts three vectors of the same size.
	void forwardProp3(const GVec& inA, const GVec& inB, const GVec& inC, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

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

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockMaxPooling"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_inputRows * m_inputCols * m_inputChannels; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_inputRows * m_inputCols * m_inputChannels / (m_regionSize * m_regionSize); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;
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
	virtual bool elementWise() const override { return true; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_units; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_units; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Computes the input that would produce the specified output.
	/// (May throw an exception if this activation function is not invertible.)
	void inverseProp(GContext& ctx, const GVec& output, GVec& input) const;

	/// Evaluates the activation function
	virtual double eval(double x) const = 0;

	/// Evaluates the derivative of the activation function.
	/// x is the net input, and f_x is the output activation--the value obtained by calling eval(x).
	virtual double derivative(double x, double f_x) const = 0;

	virtual void dropUnit(size_t index)
	{
		m_units--;
	}

	virtual void cloneUnit(size_t index)
	{
		m_units++;
	}

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
	GBlockIdentity(size_t size = 0) : GBlockActivation(size) {}
	GBlockIdentity(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_identity; }
	virtual std::string name() const override { return "GBlockIdentity"; }
	virtual double eval(double x) const override { return x; }
	virtual double derivative(double x, double f_x) const override { return 1.0; }
	virtual double inverse(double y) const { return y; }
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
	GBlockTanh(size_t size = 0) : GBlockActivation(size) {}
	GBlockTanh(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_tanh; }
	virtual std::string name() const override { return "GBlockTanh"; }
	virtual double eval(double x) const override { return std::tanh(x); }
	virtual double derivative(double x, double f_x) const override { return 1.0 - (f_x * f_x); }
	virtual double inverse(double y) const { return 0.5 * std::log(-(y + 1) / (y - 1)); }
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
	GBlockScaledTanh(size_t size = 0) : GBlockActivation(size) {}
	GBlockScaledTanh(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_scaledtanh; }
	virtual std::string name() const override { return "GBlockScaledTanh"; }
	virtual double eval(double x) const override { return std::tanh(x * SCALE_IN) * SCALE_OUT; }
	virtual double derivative(double x, double f_x) const override { return SCALE_IN/SCALE_OUT*(SCALE_OUT-f_x)*(SCALE_OUT+f_x); }
	virtual double inverse(double y) const { return 0.5 / SCALE_IN * std::log(-(y / SCALE_OUT + 1) / (y / SCALE_OUT - 1)); }
};



/// Applies the [Logistic function](https://en.wikipedia.org/wiki/Logistic_function) element-wise to the input. 
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = \frac{1}{1 + e^{-x}} \f]   | ![](Activation_logistic.png)
///
class GBlockLogistic : public GBlockActivation
{
public:
	GBlockLogistic(size_t size = 0) : GBlockActivation(size) {}
	GBlockLogistic(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_logistic; }
	virtual std::string name() const override { return "GBlockLogistic"; }
	virtual double eval(double x) const override
	{
		if(x >= 700.0) // Don't trigger a floating point exception
			return 1.0;
		else if(x < -700.0) // Don't trigger a floating point exception
			return 0.0;
		else return 1.0 / (std::exp(-x) + 1.0);

	}
	virtual double derivative(double x, double f_x) const override { return f_x * (1.0 - f_x); }
	virtual double inverse(double y) const { return std::log(y / (1.0 - y)); }
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
	GBlockBentIdentity(size_t size = 0) : GBlockActivation(size) {}
	GBlockBentIdentity(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_bentidentity; }
	virtual std::string name() const override { return "GBlockBentIdentity"; }
	virtual double eval(double x) const override { return BEND_AMOUNT * (std::sqrt(x * x + BEND_SIZE * BEND_SIZE) - BEND_SIZE) + x; }
	virtual double derivative(double x, double f_x) const override { return BEND_AMOUNT * x / std::sqrt(x * x + BEND_SIZE * BEND_SIZE) + 1.0; }
	virtual double inverse(double y) const { return (std::sqrt(2.0 * BEND_AMOUNT * BEND_AMOUNT * BEND_AMOUNT * BEND_SIZE * y + BEND_AMOUNT * BEND_AMOUNT * BEND_SIZE * BEND_SIZE + BEND_AMOUNT * BEND_AMOUNT * y * y) - BEND_AMOUNT * BEND_SIZE - y) / (BEND_AMOUNT * BEND_AMOUNT - 1.0); }
};



/// An element-wise nonlinearity block
/// This activation function forms a sigmoid shape by splicing exponential and logarithmic functions together.
class GBlockSigExp : public GBlockActivation
{
public:
	GBlockSigExp(size_t size = 0) : GBlockActivation(size) {}
	GBlockSigExp(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_sigexp; }
	virtual std::string name() const override { return "GBlockSigExp"; }
	virtual double eval(double x) const override { return (x <= 0.0 ? exp(x) - 1.0 : std::log(x + 1.0)); }
	virtual double derivative(double x, double f_x) const override { return (x <= 0.0 ? std::exp(x) : 1.0 / (x + 1.0)); }
	virtual double inverse(double y) const { return (y > 0.0 ? exp(y) - 1.0 : std::log(y + 1.0)); }
};



/// Applies the [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function) element-wise to the input. 
/// | Equation  | Plot
/// | --------- | -------
/// | \f[ f(x) = e^{-x^2} \f]   | ![](Activation_gaussian.png)
///
class GBlockGaussian : public GBlockActivation
{
public:
	GBlockGaussian(size_t size = 0) : GBlockActivation(size) {}
	GBlockGaussian(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_gaussian; }
	virtual std::string name() const override { return "GBlockGaussian"; }
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
	GBlockSine(size_t size = 0) : GBlockActivation(size) {}
	GBlockSine(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_sine; }
	virtual std::string name() const override { return "GBlockSine"; }
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
	GBlockRectifier(size_t size = 0) : GBlockActivation(size) {}
	GBlockRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_rectifier; }
	virtual std::string name() const override { return "GBlockRectifier"; }
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
	GBlockLeakyRectifier(size_t size = 0) : GBlockActivation(size) {}
	GBlockLeakyRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_leakyrectifier; }
	virtual std::string name() const override { return "GBlockLeakyRectifier"; }
	virtual double eval(double x) const override { return x >= 0.0 ? x : 0.01 * x; }
	virtual double derivative(double x, double f_x) const override { return x >= 0.0 ? 1.0 : 0.01; }
	virtual double inverse(double y) const { return (y > 0.0 ? y : 100.0 * y); }
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
	GBlockSoftPlus(size_t size = 0) : GBlockActivation(size) {}
	GBlockSoftPlus(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_softplus; }
	virtual std::string name() const override { return "GBlockSoftPlus"; }
	virtual double eval(double x) const override { return x > 500 ? x : log(1.0 + exp(x)); }
	virtual double derivative(double x, double f_x) const override { return 1.0 / (1.0 + exp(-x)); }
	virtual double inverse(double y) const { return log(exp(y) - 1.0); }
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
	virtual std::string name() const override { return "GBlockSoftRoot"; }
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
	virtual double inverse(double y) const { return 0.5 * y * std::sqrt(y * y + 4.0); }
};






/// A parameterized activation function (a.k.a. adaptive transfer function).
class GBlockHinge : public GBlock
{
protected:
	GVec m_alpha;
	GVec m_beta;

public:
	// When alpha is 0, this activation function always approximates identity. When alpha is positive, it bends upward. When alpha is negative, it bends downward.
	// Beta specifies approximately how big the bending curve is. When beta is 0, it bends on a point.
	// Size specifies the number of units in this layer.
	GBlockHinge(double alpha = 0.0, double beta = 0.1, size_t size = 0);
	GBlockHinge(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_hinge; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockHinge"; }

	/// Returns true iff this block operates only on individual elements
	virtual bool elementWise() const { return true; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_alpha.size(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_alpha.size(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

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

	virtual void dropUnit(size_t index)
	{
		std::swap(m_alpha[index], m_alpha[m_alpha.size() - 1]);
		m_alpha.erase(m_alpha.size() - 1);
		std::swap(m_beta[index], m_beta[m_beta.size() - 1]);
		m_beta.erase(m_beta.size() - 1);
	}

	virtual void cloneUnit(size_t index)
	{
		m_alpha.resizePreserve(m_alpha.size() + 1);
		m_alpha[m_alpha.size() - 1] = m_alpha[index];
		m_beta.resizePreserve(m_beta.size() + 1);
		m_beta[m_beta.size() - 1] = m_beta[index];
	}

	/// Get the alpha vector
	GVec& alpha() { return m_alpha; }

	/// Get the alpha vector
	const GVec& alpha() const { return m_alpha; }

	/// Get the beta vector
	GVec& beta() { return m_beta; }

	/// Get the beta vector
	const GVec& beta() const { return m_beta; }
};





/// A parameterized activation function (a.k.a. adaptive transfer function).
class GBlockSoftExp : public GBlock
{
protected:
	GVec m_alpha;
	double m_beta;

public:
	// When beta is 0, this activation function always approximates identity near the origin, but approximates e^x-1 when alpha is 1.
	// When beta is 1, this activation function approximates e^x when alpha is 1.
	GBlockSoftExp(double beta = 0.0, size_t size = 0);
	GBlockSoftExp(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_softexp; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockSoftExp"; }

	/// Returns true iff this block operates only on individual elements
	virtual bool elementWise() const { return true; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_alpha.size(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_alpha.size(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

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

	/// Get the alpha vector
	GVec& alpha() { return m_alpha; }

	/// Get the alpha vector
	const GVec& alpha() const { return m_alpha; }

	/// Returns beta
	double beta() const { return m_beta; }
};





/// Standard fully-connected block of weights. Often followed by a GBlockActivation.
class GBlockLinear : public GBlock
{
protected:
	GMatrix m_weights; // An inputs-by-outputs matrix of weights. (Rows = inputs, Cols = outputs.)
	GVec m_bias;
#ifdef GCUDA
	GCudaMatrix m_weightsCuda;
	GCudaVector m_biasCuda;
#endif

public:
	GBlockLinear(size_t outputs, size_t inputs = 0);
	GBlockLinear(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_linear; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockLinear"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.rows(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// A convenience method that concatenates two vectors before feeding into this block
	void forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// A convenience method that mirrors forwardProp2.
	void backProp2(const GVec& outBlame, GVec& inBlame1, GVec& inBlame2) const;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// A convenience method that goes with forwardProp2 and backProp2.
	void updateGradient2(const GVec& in1, const GVec& in2, const GVec& outBlame, GVec &gradient) const;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

#ifdef GCUDA
	/// Uploads weights onto the GPU
	virtual void uploadCuda();

	/// Downloads weights from the GPU
	virtual void downloadCuda();

	// Forwardprops on the GPU
	virtual void forwardPropCuda(GContext& ctx, const GCudaVector& input, GCudaVector& output) const override;
	
	// Backprops on the GPU
	virtual void backPropCuda(GContext& ctx, const GCudaVector& input, const GCudaVector& output, const GCudaVector& outBlame, GCudaVector& inBlame) const override;
	
	// Updates the gradient on the GPU
	virtual void updateGradientCuda(GContext& ctx, const GCudaVector& input, const GCudaVector& outBlame, GCudaVector &gradient) const override;

	virtual void stepCuda(GContext& ctx, double learningRate, const GCudaVector& gradient) override;
#endif

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
	GVec& bias() { return m_bias; }

	/// Returns the bias vector of this block.
	const GVec& bias() const { return m_bias; }

	/// Get the entire weights matrix
	GMatrix& weights() { return m_weights; }

	/// Get the entire weights matrix
	const GMatrix& weights() const { return m_weights; }

	/// Adjusts the weights to change the specified output by amount delta.
	void adjustOutput(const GVec& input, size_t outputIndex, double delta);

	/// Adjusts the weights as needed to keep all output values in the range [min, max].
	/// (Assumes output is the output vector computed by this layer when input is fed in.)
	void clipOutput(const GVec& input, const GVec& output, double min, double max);

	/// Transforms the weights of this block by the specified transformation matrix and offset vector.
	/// transform should be the pseudoinverse of the transform applied to the inputs. pOffset should
	/// be the negation of the offset added to the inputs after the transform, or the transformed offset
	/// that is added before the transform.
	void transformWeights(GMatrix& transform, const GVec& offset);

	/// Adjusts weights such that values in the new range will result in the
	/// same behavior that previously resulted from values in the old range.
	void renormalizeInput(size_t input, double oldMin, double oldMax, double newMin = 0.0, double newMax = 1.0);

	/// Drops an input from this block
	void dropInput(size_t input);

	/// Drops an output from this block
	void dropOutput(size_t output);

	/// Trains the weights of this layer using closed-form Ordinary Least Squares from some training data and an optional vector of sample weights
	void trainOLS(const GMatrix& features, const GMatrix& labels, const GVec* sampleWeights = nullptr);

#ifndef MIN_PREDICT
	static void test();
#endif
};





/// A Probabilistically Activating Linear block.
/// This is an experimental block type.
class GBlockPAL : public GBlock
{
protected:
	GMatrix m_weights; // Weights for the activation value
	GVec m_bias;
	GMatrix m_weightsProb; // Weights for the probability of activating
	GVec m_biasProb;
	GVec m_probs; // BUG: This should be part of the context
	GVec m_acts; // BUG: This should be part of the context
	GVec m_inps; // BUG: This should be part of the context

public:
	GBlockPAL(size_t outputs, size_t inputs = 0);
	GBlockPAL(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_pal; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockPAL"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.rows(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

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
	GVec& bias() { return m_bias; }

	/// Returns the bias vector of this block.
	const GVec& bias() const { return m_bias; }

	/// Get the entire weights matrix
	GMatrix& weights() { return m_weights; }

	/// Get the entire weights matrix
	const GMatrix& weights() const { return m_weights; }
};





/// A layer with random sparse connections.
class GBlockSparse : public GBlock
{
protected:
	GSparseMatrix m_weights; // An inputs-by-outputs matrix of weights.
	GVec m_bias;
	size_t m_connections;

public:
	/// connections specifies how many connections are made.
	/// connections must be >= max(inputs, outputs).
	/// connections must be <= inputs * outputs.
	GBlockSparse(size_t outputs, size_t inputs, GRand& rand, size_t connections);

	// Create a sparse block that is 'fillPercentage' full. This value should
	// be between 0.0 and 1.0.
	GBlockSparse(size_t outputs, size_t inputs, GRand& rand, double fillPercentage);

	GBlockSparse(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_sparse; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockSparse"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.rows(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

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
	GVec& bias() { return m_bias; }

	/// Returns the bias vector of this block.
	const GVec& bias() const { return m_bias; }

	/// Get the entire weights matrix
	GSparseMatrix& weights() { return m_weights; }

	/// Get the entire weights matrix
	const GSparseMatrix& weights() const { return m_weights; }
};





/// A linear block with no bias. All weights are constrained to fall between 0 and 1, and to sum to 1.
/// Regularization is implicitly applied during training to drive the weights such that
/// each output will settle on giving a weight of 1 to exactly one input unit.
class GBlockFeatureSelector : public GBlock
{
protected:
	GMatrix m_weights; // An inputs-by-outputs matrix of weights. The last row contains the bias values.
	double m_lambda;

public:
	GBlockFeatureSelector(size_t outputs, double lambda, size_t inputs = 0);
	GBlockFeatureSelector(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_featureselector; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockFeatureSelector"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.rows(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

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

	/// Get the entire weights matrix
	GMatrix &weights() { return m_weights; }

	/// Get the entire weights matrix
	const GMatrix &weights() const { return m_weights; }

	/// Returns true iff all of the units in this layer have assigned all of their weight to just one unit.
	bool hasConverged();
};





/// Consumes n values and produces a big vector in the form of two vectors (concatenated together)
/// whose corresponding values form all pairs of inputs, plus each input is also paired with each of "lo" and "hi".
/// Example: If the input is {1,2,3}, then the output will be {1,1,1,1,2,2,2,3,3,    2,3,lo,hi,3,lo,hi,lo,hi}.
/// Thus, if there are n inputs, there will be n(n-1)+4n outputs.
class GBlockAllPairings : public GBlockWeightless
{
protected:
	size_t m_inputCount;
	double m_lo, m_hi;

public:
	/// General-purpose constructor.
	GBlockAllPairings(size_t inputs, double lo, double hi);

	/// Deserializing constructor
	GBlockAllPairings(GDomNode* pNode);
	~GBlockAllPairings();

	/// Returns the type of this block
	virtual BlockType type() const override { return block_allpairings; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockAllPairings"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_inputCount; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_inputCount * (m_inputCount - 1) + 4 * m_inputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Returns the input unit that is connected with the specifed output unit.
	/// Returns inputs() for "lo", and inputs()+1 for "hi".
	/// (The current implementation is not very efficient.)
	size_t findSource(size_t outputUnit);
};





/// A block of fuzzy logic units.
/// Treats the inputs as two concatenated vectors, whose corresponding values each form a pair
/// to be combined with fuzzy logic to produce one output value.
/// Example: The input {1,2,3,4,5,6} will apply fuzzy logic to the three pairs {1,4}, {2,5} and {3,6} to produce a vector of 3 output values.
class GBlockFuzzy : public GBlock
{
protected:
	GVec m_alpha;

public:
	GBlockFuzzy(size_t outputs);
	GBlockFuzzy(GDomNode* pNode);

	/// Returns the type of this block
	virtual BlockType type() const override { return block_fuzzy; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockFuzzy"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_alpha.size() * 2; }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_alpha.size(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

	/// Add the weight and bias gradient to the weights.
	virtual void step(double learningRate, const GVec &gradient) override;

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

	/// Returns the vector of fuzzy logic parameters.
	GVec& alpha() { return m_alpha; }

	/// Returns the vector of fuzzy logic parameters.
	const GVec& alpha() const { return m_alpha; }
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

	/// Returns the name of this block
	virtual std::string name() const { return "GBlockMaxOut"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.rows(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.cols(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

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

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockRestrictedBoltzmannMachine"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_weights.cols(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_weights.rows(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Feed a vector backwards through this block.
	void feedBackward(const GVec& output, GVec& input) const;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

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
	void drawSample(GContext& ctx, size_t iters, GVec& output, GVec& input);


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

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockConvolutional1D"; }

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this block consumes
        virtual size_t inputs() const override { return m_inputSamples * m_inputChannels; }

	/// Returns the number of outputs this block produces
        virtual size_t outputs() const override { return m_outputSamples * m_inputChannels * m_kernelsPerChannel; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

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




class GBlockConvolutional2D : public GBlock
{
protected:
	/// Image abstraction to facilitate convolution
	struct Image
	{
		static size_t npos;

		Image(size_t width, size_t height, size_t channels);
		Image(GVec* data, const Image& copyMyParams);
		size_t index(size_t x, size_t y, size_t z) const;
		double read(size_t x, size_t y, size_t z = 0) const;
		double& at(size_t x, size_t y, size_t z = 0);

		/// image data
		GVec* data;
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
	double filterSum(const Image &in, const Image &filter, size_t channels) const;
	void addScaled(const Image &in, double scalar, Image &out) const;
	void convolve(const Image &in, const Image &filter, Image &out, size_t channels = none) const;
	void convolveFull(const Image &in, const Image &filter, Image &out, size_t channels = none) const;
	void updateOutputSize();

public:
	static size_t none;

	/// General-purpose constructor.
	GBlockConvolutional2D(size_t width, size_t height, size_t channels, size_t kWidth, size_t kHeight, size_t kCount = 0);

	/// Constructor that will automatically use the upstream convolutional block when added to a neural network
	GBlockConvolutional2D(size_t kWidth, size_t kHeight, size_t kCount = 0);

	GBlockConvolutional2D(GDomNode* pNode);

	virtual BlockType type() const override { return block_convolutional2d; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockConvolutional2D"; }

	virtual GDomNode* serialize(GDom* pDoc) const override;
	virtual void resize(size_t inputs, size_t outputs) override;
        virtual size_t inputs() const override { return m_width * m_height * m_channels; }
        virtual size_t outputs() const override { return m_outputWidth * m_outputHeight * m_bias.size(); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;


	/// Updates the gradient for updating the weights by gradient descent.
	/// (Assumes the error has already been computed and deactivated.)
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const override;

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
	virtual bool isRecurrent() const override { return true; }

	/// Returns the number of instances of this block unfolded through time that will be used during training.
	virtual size_t depth() = 0;

	/// Returns a new context object for this recurrent block.
	/// The recurrent state should be initialized to the starting state.
	virtual GContextRecurrentInstance* newContext(GRand& rand) = 0;

protected:
	/// Deliberately protected.
	/// Throws an exception telling you to call GContextRecurrent::forwardProp instead.
	virtual void forwardProp(GContext& ctx, const GVec& input, GVec& output) const override;

	/// Deliberately protected.
	/// Throws an exception telling you to call GContextRecurrent::backProp instead.
	virtual void backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const override;

	/// Deliberately protected.
	/// Throws an exception telling you to call GContextRecurrent::updateGradient instead.
	virtual void updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec& gradient) const override;

#ifndef MIN_PREDICT
	static double testEngine(GNeuralNet& nn);
#endif

};






/// A special context object for recurrent blocks
class GContextRecurrent : public GContext
{
public:
	GBlockRecurrent& m_block;
	GVec m_emptyBlame;
	GVec m_bogusBlame;
	std::vector<GContextRecurrentInstance*> m_contextHistory; // Stores recent context objects
	std::vector<GContextRecurrentInstance*> m_contextSpares; // Stores unused context objects
	std::vector<GVec*> m_inputHistory; // Stores recent inputs into the recurrent blocks
	std::vector<GVec*> m_inputSpares; // Stores unused vector objects
	size_t m_pos; // Position of the oldest recorded input vector in the history

	GContextRecurrent(GRand& rand, GBlockRecurrent& block);
	virtual ~GContextRecurrent();

	virtual void resetState() override;

	void forwardProp(const GVec& input, GVec& output);

	void forwardPropThroughTime(const GVec& input, GVec& output);

	void backPropThroughTime(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame);

	void updateGradient(const GVec& input, const GVec& outBlame, GVec& gradient) const;

#ifdef GCUDA
	virtual GCudaEngine& cudaEngine()
	{
		throw Ex("Sorry, not implemented");
	}

	void forwardPropCuda(const GCudaVector& input, GCudaVector& output)
	{
		throw Ex("Sorry, not implemented");
	}

	void forwardPropThroughTimeCuda(const GCudaVector& input, GCudaVector& output)
	{
			throw Ex("Sorry, not implemented");
	}

	void backPropThroughTimeCuda(const GCudaVector& input, const GCudaVector& output, const GCudaVector& outBlame, GCudaVector& inBlame)
	{
                throw Ex("Sorry, not implemented");
	}

	void updateGradientCuda(const GCudaVector& input, const GCudaVector& outBlame, GCudaVector& gradient) const
	{
                throw Ex("Sorry, not implemented");
	}
#endif

};

/// A single instance of a recurrent context that has been unfolded through time
class GContextRecurrentInstance : public GContext
{
public:
	GContextRecurrentInstance(GRand& rand) : GContext(rand) {}
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
	virtual BlockType type() const override { return block_lstm; }

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockLSTM"; }

	/// Returns the number of instances of this block unfolded through time that will be used during training.
	virtual size_t depth() override { return 4; }

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_write.inputs() - m_write.outputs(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_write.outputs(); }

	/// Makes a new context object for this block
	virtual GContextRecurrentInstance* newContext(GRand& rand) override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

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

	GContextLSTM(GRand& rand, GBlockLSTM& block);

	virtual void resetState() override;
	virtual void clearBlame() override;
	virtual void forwardProp(GContextRecurrentInstance* pPrev, const GVec& input, GVec& output) override;
	virtual void backProp(GContextRecurrentInstance* pPrev, const GVec& outBlame, GVec& inBlame) override;
	virtual void updateGradient(GContextRecurrentInstance* prev, const GVec& input, GVec& gradient) const override;

#ifdef GCUDA
	virtual GCudaEngine& cudaEngine()
	{
		throw Ex("Sorry, not implemented");
	}
#endif
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

	/// Returns the name of this block
	virtual std::string name() const override { return "GBlockGRU"; }

	/// Returns the number of instances of this block unfolded through time that will be used during training.
	virtual size_t depth() override { return 4; }

	/// Returns the number of inputs this block consumes
	virtual size_t inputs() const override { return m_update.inputs() - m_update.outputs(); }

	/// Returns the number of outputs this block produces
	virtual size_t outputs() const override { return m_update.outputs(); }

	/// Makes a new context object for this block
	virtual GContextRecurrentInstance* newContext(GRand& rand) override;

	/// Resizes this block.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Marshall this block into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

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

	GContextGRU(GRand& rand, GBlockGRU& block);

	virtual void forwardProp(GContextRecurrentInstance* pPrev, const GVec& input, GVec& output) override;
	virtual void resetState() override;
	virtual void clearBlame() override;
	virtual void backProp(GContextRecurrentInstance* pPrev, const GVec& outBlame, GVec& inBlame) override;
	virtual void updateGradient(GContextRecurrentInstance* prev, const GVec& input, GVec& gradient) const override;

#ifdef GCUDA
	virtual GCudaEngine& cudaEngine()
	{
		throw Ex("Sorry, not implemented");
	}
#endif
};



} // namespace GClasses

#endif // __GBLOCK_H__
