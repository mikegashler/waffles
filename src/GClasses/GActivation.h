/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
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

#ifndef __GACTIVATION_H__
#define __GACTIVATION_H__

#include <cmath>
#include <math.h>
#include "GError.h"
#ifndef MIN_PREDICT
#include "GMath.h"
#include "GVec.h"
#endif // MIN_PREDICT
namespace GClasses {

class GDomNode;
class GDom;


/// The base class for activation functions. Typically, this are
/// sigmoid-shaped functions used to "squash" the output of a network
/// node. These are typically used in conjunction with the GNeuralNet class.
class GActivationFunction
{
public:
	GActivationFunction() {}
	virtual ~GActivationFunction() {}

	/// Returns the name of this activation function
	virtual const char* name() const = 0;

	/// The activation function
	virtual double squash(double x, int index) = 0;

	/// The derivative of the activation function
	virtual double derivative(double x, int index) = 0;

	/// The inverse of the activation function. (This function may throw an exception
	/// if the activation function cannot be inverted.)
	virtual double inverse(double y, int index) = 0;

	/// Returns a clone of this object
	virtual GActivationFunction* clone() = 0;

	/// This computes the derivative of the net value. (Sometimes, such as with
	/// GActivationLogistic, it is more efficient to compute this from the activation
	/// value, so both are provided.)
	virtual double derivativeOfNet(double net, double activation, int index) { return derivative(net, index); }

	/// Serialize this object
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Refines the parameters of this activation function by stochastic gradient descent
	virtual void refine(const double* pNet, const double* pError, double learningRate) {}

	/// Regularizes the parameters of this activation function
	virtual void regularize(double lambda) {}

	/// Deserialize this object
	static GActivationFunction* deserialize(GDomNode* pNode);

//	/// Calculate the scaling factor for this activation function that minimizes the vanishing gradient
//	double measureWeightScale(size_t width, size_t depth, size_t seed);
};

/// The logistic activation function
class GActivationLogistic : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "logistic"; }

	/// The logistic function. Returns 1.0/(e^(-x)+1.0)
	virtual double squash(double x, int index)
	{
		if(x >= 700.0) // Don't trigger a floating point exception
			return 1.0;
		if(x < -700.0) // Don't trigger a floating point exception
			return 0.0;
		return 1.0 / (exp(-x) + 1.0);
	}

	/// Returns d*(1.0-d), where d=squash(x)
	virtual double derivative(double x, int index) { double d = squash(x, index); return d * (1.0 - d); }

	/// The logit function. Returns log(y)-log(1.0-y)
	virtual double inverse(double y, int index)
	{
		if(y >= 1.0)
			return 700.0;
		if(y <= 0.0)
			return -700.0;
		return (log(y) - log(1.0 - y));
	}

	/// Returns y*(1.0-y)
	virtual double derivativeOfNet(double net, double activation, int index) { return activation * (1.0 - activation); }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationLogistic(); }
};

/// The arctan activation function
class GActivationArcTan : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "arctan"; }

	/// Returns atan(x). The result will be in the range -PI/2 <= y <= PI/2
	virtual double squash(double x, int index) { return atan(x); }

	/// Returns 1/(x*x+1.0)
	virtual double derivative(double x, int index) { return 1.0 / (x * x + 1.0); }

	/// Returns tan(y), where -PI/2 <= y <= PI/2
	virtual double inverse(double y, int index) { return tan(y); }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationArcTan(); }
};

/// The hyperbolic tangent activation function
class GActivationTanH : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "tanh"; }

	/// Returns tanh(x). The result is in the range -1 <= y <= 1
	virtual double squash(double x, int index)
	{
		//return tanh(x);
		if(x >= 700.0)
			return 1.0;
		if(x < -700.0)
			return -1.0;
		double a = exp(x);
		double b = 1.0 / a;
		return (a - b) / (a + b);
	}

	/// Returns sech(x)*sech(x)
	virtual double derivative(double x, int index)
	{
		if(x >= 700.0)
			return 1.0;
		if(x < -700.0)
			return 0.0;
		double a = exp(x);
		double b = 1.0 / a;
		double d = 2.0 / (a + b); // sech(x)
		return d * d;
	}

	/// Returns atanh(y), where -1 <= y <= 1
	virtual double inverse(double y, int index)
	{
#ifdef WINDOWS
		return 0.5 * (log(1.0 + y) - log(1.0 - y));
#else
		return atanh(y);
#endif
	}

	/// Returns 1-(y*y)
	virtual double derivativeOfNet(double net, double activation) { return 1.0 - (activation * activation); }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationTanH(); }
};

/// An algebraic activation function. (This is a hyperbola divided by x.)
class GActivationAlgebraic : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "algebraic"; }

	/// Returns x/(sqrt(x*x+1.0). The result is in the range -1 <= y <= 1
	virtual double squash(double x, int index) { return x / (sqrt(x * x + 1.0)); }

	/// Returns 1.0/(sqrt(x*x+1))-(x*x)/pow(x*x+1,1.5)
	virtual double derivative(double x, int index) { return 1.0 / (sqrt(x * x + 1)) - (x * x) / pow(x * x + 1, 1.5); }

	/// Returns y / (sqrt(1.0 - (y * y)))
	virtual double inverse(double y, int index) { return y / (sqrt(1.0 - (y * y))); }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationAlgebraic(); }
};

/// Use this function when you do not want to squash the net. For example,
/// using this activation function with a network that has no hidden layers
/// makes a perceptron model. Also, it is common to use this activation
/// function on the output layer for regression problems.
class GActivationIdentity : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "identity"; }

	/// Returns x
	virtual double squash(double x, int index) { return x; }

	/// Returns 1.0
	virtual double derivative(double x, int index) { return 1.0; }

	/// Returns y
	virtual double inverse(double y, int index) { return y; }

	/// Returns 1.0
	virtual double derivativeOfNet(double net, double activation, int index) { return 1.0; }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationIdentity(); }
};


#define BEND_AMOUNT 0.5

/// This provides an alternative to using GActivationIdentity on the output layer
/// for regression problems. It may add more power because it is non-linear, but
/// like the identity function, its co-domain is the same as its domain. At
/// very positive values, this is shaped like y=1.5*x. At very negative values,
/// this is shaped like y=0.5*x. Around 0, it is shaped like y=x.
class GActivationBend : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "bend"; }

	/// Returns the bend function of x
	virtual double squash(double x, int index)
	{
		return BEND_AMOUNT * (sqrt(x * x + 1) - 1) + x;
	}

	/// Returns the derivative of the bend function
	virtual double derivative(double x, int index)
	{
		return BEND_AMOUNT * x / sqrt(x * x + 1) + 1;
	}

	/// Returns the inverse of the bend function
	virtual double inverse(double y, int index)
	{
		return (BEND_AMOUNT * sqrt(y * y + 2.0 * BEND_AMOUNT * y + 1) - y - BEND_AMOUNT) / (BEND_AMOUNT * BEND_AMOUNT - 1);
	}

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationBend(); }
};



class GActivationHinge : public GActivationFunction
{
protected:
	size_t m_units;
	GVec m_hinges;

public:
	/// General-purpose constructor
	GActivationHinge(size_t units);

	/// Unmarshaling constructor
	GActivationHinge(GDomNode* pNode);

	/// Returns the name of this activation function
	virtual const char* name() const { return "hinge"; }

	/// Marshals this object to a JSON DOM.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Returns the bend function of x
	virtual double squash(double x, int index)
	{
		return m_hinges.v[index] * (sqrt(x * x + 1) - 1) + x;
	}

	/// Returns the derivative of the bend function
	virtual double derivative(double x, int index)
	{
		return m_hinges.v[index] * x / sqrt(x * x + 1) + 1;
	}

	/// Returns the inverse of the bend function
	virtual double inverse(double y, int index)
	{
		double v = m_hinges.v[index];
		return (v * sqrt(y * y + 2.0 * v * y + 1) - y - v) / (v * v - 1);
	}

	/// Refines the hinge values by stochastic gradient descent
	virtual void refine(const double* pNet, const double* pError, double learningRate);

	/// Regularizes the parameters of this activation function
	virtual void regularize(double lambda);

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone()
	{
		GActivationHinge* pClone = new GActivationHinge(m_units);
		GVec::copy(pClone->m_hinges.v, m_hinges.v, m_units);
		return pClone;
	}
};


/// This is an output-layer activation function shaped
/// like a sigmoid, but with both a co-domain and domain
/// that spans the continuous values. At very negative values,
/// this is shaped like y=-sqrt(-2x). Near zero, it is shaped
/// like y=x. At very positive values, it is shaped like y=sqrt(2x).
class GActivationBiDir : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "bidir"; }

	virtual double squash(double x, int index)
	{
		double d = sqrt(x * x + 1.0);
		return sqrt(d + x) - sqrt(d - x);
	}

	virtual double derivative(double x, int index)
	{
		if(std::abs(x) > 1e7)
			return 0.0;
		double d = sqrt(x * x + 1.0);
		double t = x / d;
		return (t + 1.0) / (2.0 * sqrt(d + x)) - (t - 1.0) / (2.0 * sqrt(d - x));
	}

	virtual double inverse(double y, int index)
	{
		double d = y * y;
		if(y >= 0.0)
			return 0.5 * sqrt(d * d + 4.0 * d);
		else
			return -0.5 * sqrt(d * d + 4.0 * d);
	}

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationBiDir(); }
};

/// This is a simple Gaussian function.
class GActivationGaussian : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "gaussian"; }

	virtual double squash(double x, int index) { return exp(-(x * x)); }

	virtual double derivative(double x, int index) { return -2.0 * x * exp(-(x * x)); }

	virtual double inverse(double y, int index)
	{
		throw Ex("Not invertible");
		return 0;
	}

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationGaussian(); }
};



/// The sine wave.
class GActivationSin : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "sin"; }

	virtual double squash(double x, int index) { return sin(x); }

	virtual double derivative(double x, int index) { return cos(x); }

	virtual double inverse(double y, int index) { return asin(y); }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationSin(); }
};



/// This is a canonical wavelet. Has a "Mexican-hat" shape.
class GActivationSinc : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "sinc"; }

	virtual double squash(double x, int index) { return x == 0 ? 1.0 : sin(x) / x; }

	virtual double derivative(double x, int index) { return x == 0 ? 0.0 : cos(x) / x - sin(x) / (x * x); }

	virtual double inverse(double y, int index)
	{
		throw Ex("The Sinc function cannot be inverted");
		return 0;
	}

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationSinc(); }
};


/// Uses the derivative of the logistic function as an activation function. This is shaped somewhat like a bell-curve.
class GActivationLogisticDerivative : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "logisticderiv"; }

	/// The derivative of the logistic function.
	virtual double squash(double x, int index)
	{
		if(x >= 700.0) // Don't trigger a floating point exception
			return 0.0;
		if(x < -700.0) // Don't trigger a floating point exception
			return 0.0;
		double y = 1.0 / (exp(-x) + 1.0);
		return y * (1.0 - y);
	}

	/// Returns d*(1.0-d), where d=squash(x)
	virtual double derivative(double x, int index) { double d = squash(x, index); return d * (1.0 - d) * (1.0 - 2.0 * d); }

	/// The logit function. Returns log(y)-log(1.0-y)
	virtual double inverse(double y, int index)
	{
		throw Ex("This function is not easily invertible");
	}

	/// Returns y*(1.0-y)
	virtual double derivativeOfNet(double net, double activation, int index)
	{
		double t = 1.0 - 2.0 / (exp(-net) + 1.0);
		return activation * t;
	}

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationLogisticDerivative(); }
};


/// The activation function for rectified linear units (ReLU).
class GActivationRectifiedLinear : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "relu"; }

	virtual double squash(double x, int index) { return std::max(0.0, x); }

	virtual double derivative(double x, int index) { return (x >= 0.0 ? 1.0 : 0.0); }

	virtual double inverse(double y, int index) { return y; }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationRectifiedLinear(); }
};


/// The integral of the logsitic function. At very negative values, this converges toward y=0. At very positive values, this converges to y=x.
class GActivationSoftPlus : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "softplus"; }

	virtual double squash(double x, int index) { return x > 500 ? x : log(1.0 + exp(x)); }

	virtual double derivative(double x, int index) { return 1.0 / (1.0 + exp(-x)); }

	virtual double inverse(double y, int index) { return log(exp(y) - 1.0); }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationSoftPlus(); }
};


/// The integral of the logsitic function. At very negative values, this converges toward y=0. At very positive values, this converges to y=x.
class GActivationSoftPlus2 : public GActivationFunction
{
public:
	/// Returns the name of this activation function
	virtual const char* name() const { return "softplus2"; }

	virtual double squash(double x, int index) { return 0.5 * (sqrt(x * x + 1) + x); }

	virtual double derivative(double x, int index) { return 0.5 * (x / sqrt(x * x + 1) + 1.0); }

	virtual double inverse(double y, int index) { return (2 * y - 1) * (2 * y + 1) / (4 * y); }

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone() { return new GActivationSoftPlus2(); }
};


// y=0.25*sqrt(x * x + 1)+x-0.25
// dy/dx = 0.25 * x / sqrt(x * x + 1) + 1

} // namespace GClasses

#endif // __GACTIVATION_H__

