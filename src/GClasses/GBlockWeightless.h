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

#ifndef __GBLOCKWEIGHTLESS_H__
#define __GBLOCKWEIGHTLESS_H__

#include "GBlock.h"

namespace GClasses {


/// The base class of layers that have no weights
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





/// Multiplies each pair of values together to produce the output
class GBlockProductPooling : public GBlockWeightless
{
protected:
	size_t m_outputCount;

public:
	/// General-purpose constructor.
	GBlockProductPooling(size_t outputs, size_t inputs);

	/// Deserializing constructor
	GBlockProductPooling(GDomNode* pNode);
	~GBlockProductPooling();

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_productpooling; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this layer
	virtual std::string to_str() const override;

	/// Resizes this layer. outputs must be 2*inputs.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
        virtual size_t inputs() const override { return m_outputCount / 2; }

	/// Returns the number of outputs this layer produces
        virtual size_t outputs() const override { return m_outputCount; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;
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

	/// Returns the type of this layer
	virtual BlockType type() const override { return block_maxpooling; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this layer
	virtual std::string to_str() const override;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
        virtual size_t inputs() const override { return m_inputRows * m_inputCols * m_inputChannels; }

	/// Returns the number of outputs this layer produces
        virtual size_t outputs() const override { return m_inputRows * m_inputCols * m_inputChannels / (m_regionSize * m_regionSize); }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;
};










/// The base class of layers that apply an activation function, such as tanh, in an element-wise manner.
class GBlockActivation : public GBlockWeightless
{
protected:
	size_t m_units;

public:
	GBlockActivation(size_t size = 0);
	GBlockActivation(GDomNode* pNode);

	/// Returns true iff this layer operates only on individual elements
	virtual bool elementWise() const { return true; }

	/// Marshall this layer into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) const override;

	/// Makes a string representation of this layer
	virtual std::string to_str() const override;

	/// Resizes this layer.
	virtual void resize(size_t inputs, size_t outputs) override;

	/// Returns the number of inputs this layer consumes
        virtual size_t inputs() const override { return m_units; }

	/// Returns the number of outputs this layer produces
        virtual size_t outputs() const override { return m_units; }

	/// Evaluate the input, set the output.
	virtual void forwardProp(const GVec& input, GVec& output) const override;

	/// Evaluates outBlame, and adds to inBlame.
	/// (Note that it "adds to" the inBlame because multiple blocks may fork from a common source.)
	virtual void backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) override;

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




/// An element-wise nonlinearity layer
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



/// An element-wise nonlinearity layer
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
/// An element-wise nonlinearity layer
class GBlockBentIdentity : public GBlockActivation
{
public:
	GBlockBentIdentity(size_t size = 0) : GBlockActivation(size) {}
	GBlockBentIdentity(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_bentidentity; }
	virtual double eval(double x) const override { return BEND_AMOUNT * (std::sqrt(x * x + BEND_SIZE * BEND_SIZE) - BEND_SIZE) + x; }
	virtual double derivative(double x, double f_x) const override { return BEND_AMOUNT * x / std::sqrt(x * x + BEND_SIZE * BEND_SIZE) + 1.0; }
};



/// An element-wise nonlinearity layer
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



/// An element-wise nonlinearity layer
class GBlockGaussian : public GBlockActivation
{
public:
	GBlockGaussian(size_t size = 0) : GBlockActivation(size) {}
	GBlockGaussian(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_gaussian; }
	virtual double eval(double x) const override { return std::exp(-(x * x)); }
	virtual double derivative(double x, double f_x) const override { return -2.0 * x * std::exp(-(x * x)); }
};



/// An element-wise nonlinearity layer
class GBlockSine : public GBlockActivation
{
public:
	GBlockSine(size_t size = 0) : GBlockActivation(size) {}
	GBlockSine(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_sine; }
	virtual double eval(double x) const override { return std::sin(x); }
	virtual double derivative(double x, double f_x) const override { return std::cos(x); }
};




/// An element-wise nonlinearity layer
class GBlockRectifier : public GBlockActivation
{
public:
	GBlockRectifier(size_t size = 0) : GBlockActivation(size) {}
	GBlockRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_rectifier; }
	virtual double eval(double x) const override { return std::max(0.0, x); }
	virtual double derivative(double x, double f_x) const override { return (x >= 0.0 ? 1.0 : 0.0); }
};




/// An element-wise nonlinearity layer
class GBlockLeakyRectifier : public GBlockActivation
{
public:
	GBlockLeakyRectifier(size_t size = 0) : GBlockActivation(size) {}
	GBlockLeakyRectifier(GDomNode* pNode) : GBlockActivation(pNode) {}
	virtual BlockType type() const override { return block_leakyrectifier; }
	virtual double eval(double x) const override { return x >= 0.0 ? x : 0.01 * x; }
	virtual double derivative(double x, double f_x) const override { return x >= 0.0 ? 1.0 : 0.01; }
};



/// An element-wise nonlinearity layer
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




/// An element-wise nonlinearity layer.
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




} // namespace GClasses

#endif // __GBLOCKWEIGHTLESS_H__
