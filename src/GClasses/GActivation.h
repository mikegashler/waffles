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

/*

#define BEND_AMOUNT 0.5
#define BEND_SIZE 0.5

// A parameterized version of the bent identity activation function
class GActivationHinge : public GActivationFunction
{
protected:
	size_t m_units;
	GVec m_error;
	GVec m_hinges;
	GVec m_delta;
	GVec m_rates;

public:
	/// General-purpose constructor
	GActivationHinge();

	/// Unmarshaling constructor
	GActivationHinge(GDomNode* pNode);

#ifndef MIN_PREDICT
	/// Performs unit testing. Throws an exception if any test fails.
	static void test();
#endif

	/// Returns the name of this activation function
	virtual const char* name() const { return "hinge"; }

	/// Returns the internal vector of hinge values
	GVec& alphas() { return m_hinges; }

	/// Marshals this object to a JSON DOM.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Returns the bend function of x
	virtual double squash(double x, size_t index)
	{
		return m_hinges[index] * (sqrt(x * x + BEND_SIZE * BEND_SIZE) - BEND_SIZE) + x;
	}

	/// Returns the derivative of the bend function
	virtual double derivative(double x, size_t index)
	{
		return m_hinges[index] * x / sqrt(x * x + BEND_SIZE * BEND_SIZE) + 1.0;
	}

	/// Returns the inverse of the bend function
	virtual double inverse(double y, size_t index)
	{
		double v = m_hinges[index];
		return BEND_SIZE * (v * sqrt(y * y / (BEND_SIZE * BEND_SIZE) + 2.0 * v / BEND_SIZE * y + 1.0) - y / BEND_SIZE - v) / (v * v - 1.0);
	}

	/// Resizes the layer
	virtual void resize(size_t units);

	/// Sets the error term for this activation function. Used in stochastic gradient descent. (The default behavior is nothing because most activation functions have no parameters to refine.)
	virtual void setError(const GVec& error);

	/// Computes the deltas necessary to refine the parameters of this activation function by gradient descent
	virtual void updateDeltas(const GVec& net, const GVec& activation, double momentum);

	/// Updates the deltas
	virtual void updateDeltas(const GVec &net, const GVec &activation, GVec &deltas);

	/// Applies the deltas to refine the parameters of this activation function by gradient descent
	virtual void applyDeltas(double learningRate);

	/// Applies the deltas
	virtual void applyDeltas(double learningRate, const GVec &deltas);

	/// Adaptively updates per-weight learning rates, and updates the weights based on the signs of the gradient
	virtual void applyAdaptive();

	/// Regularizes the parameters of this activation function
	virtual void regularize(double lambda);

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone();

	/// Returns the number of weights in this activation function.
	virtual size_t countWeights();

	/// Serialize the weights in this activation function.
	virtual size_t weightsToVector(double* pOutVector);

	/// Serialize the weights in this activation function.
	virtual size_t vectorToWeights(const double* pVector);

	/// Copies the weights from another instance.
	virtual void copyWeights(const GActivationFunction* pOther);
};



// The SoftExponential activation function, as published in Luke B. Godfrey and Gashler, Michael S.
// A Continuum among Logarithmic, Linear, and Exponential Functions, and Its Potential to Improve Generalization in Neural Networks.
// In Proceedings of the 7th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management: KDIR, pages 481-486. Lisbon, Portugal, November, 2015.
class GActivationSoftExponential : public GActivationFunction
{
protected:
	size_t m_units;
	GVec m_error;
	GVec m_alphas;
	GVec m_delta;
	GVec m_rates;

public:
	/// General-purpose constructor
	GActivationSoftExponential();

	/// Unmarshaling constructor
	GActivationSoftExponential(GDomNode* pNode);

#ifndef MIN_PREDICT
	/// Performs unit testing. Throws an exception if any test fails.
	static void test();
#endif

	/// Returns the name of this activation function
	virtual const char* name() const { return "softexp"; }

	/// Returns the internal vector of parameter values
	GVec& alphas() { return m_alphas; }

	/// Marshals this object to a JSON DOM.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Returns the softExponential function of x with the parameterized alpha value
	virtual double squash(double x, size_t index)
	{
		return std::max(-500.0, std::min(500.0, GMath::softExponential(m_alphas[index], x)));
	}

	/// Returns the derivative of the softExponential function
	virtual double derivative(double x, size_t index)
	{
		double a = m_alphas[index];
		double d;
		if(a < -1e-12)
			d = 1.0 / std::max(0.0033, 1.0 - a * (a + x)); // maxes out at about 300
		else if(a > 1e-12)
			d = exp(std::min(5.8, a * x)); // maxes out at about 300
		else
			d = 1.0;
		return d;//tanh(d);
	}

	/// Returns the inverse of the bend function
	virtual double inverse(double y, size_t index)
	{
		double a = m_alphas[index];
		return GMath::softExponential(-a, y);
	}

	/// Resizes the layer
	virtual void resize(size_t units);

	/// Sets the error term for this activation function. Used in stochastic gradient descent. (The default behavior is nothing because most activation functions have no parameters to refine.)
	virtual void setError(const GVec& error);

	/// Computes the deltas necessary to refine the parameters of this activation function by gradient descent
	virtual void updateDeltas(const GVec& net, const GVec& activation, double momentum);

	/// Updates the deltas
	virtual void updateDeltas(const GVec &net, const GVec &activation, GVec &deltas);

	/// Applies the deltas to refine the parameters of this activation function by gradient descent
	virtual void applyDeltas(double learningRate);

	/// Applies the deltas
	virtual void applyDeltas(double learningRate, const GVec &deltas);

	/// Adaptively updates per-weight learning rates, and updates the weights based on the signs of the gradient
	virtual void applyAdaptive();

	/// Regularizes the parameters of this activation function
	virtual void regularize(double lambda);

	/// See the comment for GActivationFunction::clone
	virtual GActivationFunction* clone();

	/// Returns the number of weights in this activation function.
	virtual size_t countWeights();

	/// Serialize the weights in this activation function.
	virtual size_t weightsToVector(double* pOutVector);

	/// Serialize the weights in this activation function.
	virtual size_t vectorToWeights(const double* pVector);

	/// Copies the weights from another instance.
	virtual void copyWeights(const GActivationFunction* pOther);
};


*/

} // namespace GClasses

#endif // __GACTIVATION_H__

