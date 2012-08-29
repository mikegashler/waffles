#ifndef __GKERNELTRICK_H__
#define __GKERNELTRICK_H__

#include "GLearner.h"
#include "GVec.h"
#include "GDom.h"
#include <math.h>
#include <cmath>

namespace GClasses {

class GRand;

/// The base class for kernel functions. Classes which implement this
/// must provide an "apply" method that applies the kernel to two
/// vectors. Kernels may be combined together to form a more complex
/// kernel, to which the kernel trick will still apply.
class GKernel
{
public:
	GKernel() {}
	virtual ~GKernel() {}

	/// Returns the name of this kernel.
	virtual const char* name() const = 0;

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc) = 0;

	/// Applies the kernel to the two specified vectors.
	virtual double apply(const double* pA, const double* pB, size_t dims) = 0;

	/// Deserializes a kernel object
	static GKernel* deserialize(GDomNode* pNode);

protected:
	/// Helper method used by the serialize methods in child classes
	GDomNode* makeBaseNode(GDom* pDoc) const;
};

/// The identity kernel
class GKernelIdentity : public GKernel
{
public:
	GKernelIdentity() : GKernel() {}
	GKernelIdentity(GDomNode* pNode) : GKernel() {}
	virtual ~GKernelIdentity() {}

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = GKernel::makeBaseNode(pDoc);
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "identity"; }

	/// Computes A*B
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return GVec::dotProductIgnoringUnknowns(pA, pB, dims);
	}
};

/// Chi Squared kernel
class GKernelChiSquared : public GKernel
{
public:
	GKernelChiSquared() : GKernel() {}
	GKernelChiSquared(GDomNode* pNode) : GKernel() {}
	virtual ~GKernelChiSquared() {}

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "chisquared"; }

	/// Computes the sum over each element of 2 * a * b / (a + b)
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		double d = 0.0;
		for(size_t i = 0; i < dims; i++)
		{
			if(*pA != UNKNOWN_REAL_VALUE && *pB != UNKNOWN_REAL_VALUE)
				d += 2.0 * (*pA) * (*pB) / ((*pA) + (*pB));
			pA++;
			pB++;
		}
		return d;
	}
};

/// A polynomial kernel
class GKernelPolynomial : public GKernel
{
protected:
	double m_offset;
	unsigned int m_order;

public:
	GKernelPolynomial(double offset, unsigned int order) : GKernel(), m_offset(std::abs(offset)), m_order(order) {}
	GKernelPolynomial(GDomNode* pNode) : GKernel(), m_offset(pNode->field("offset")->asDouble()), m_order((unsigned int)pNode->field("order")->asInt()) {}
	virtual ~GKernelPolynomial() {}

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "offset", pDoc->newDouble(m_offset));
		pObj->addField(pDoc, "order", pDoc->newInt(m_order));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "polynomial"; }

	/// Computes (A * B + offset)^order
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return pow(GVec::dotProductIgnoringUnknowns(pA, pB, dims) + m_offset, (int)m_order);
	}
};

/// A Gaussian RBF kernel
class GKernelGaussianRBF : public GKernel
{
protected:
	double m_variance;

public:
	GKernelGaussianRBF(double variance) : GKernel(), m_variance(std::abs(variance)) {}
	GKernelGaussianRBF(GDomNode* pNode) : GKernel(), m_variance(pNode->field("var")->asDouble()) {}
	virtual ~GKernelGaussianRBF() {}

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "var", pDoc->newDouble(m_variance));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "rbf"; }

	/// Computes e^(-0.5 * ||A - B||^2 / variance)
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return exp(-0.5 * GVec::estimateSquaredDistanceWithUnknowns(pA, pB, dims) / m_variance);
	}
};

/// A translation kernel
class GKernelTranslate : public GKernel
{
protected:
	GKernel* m_pK;
	double m_value;

public:
	/// Takes ownership of pK
	GKernelTranslate(GKernel* pK, double value) : GKernel(), m_pK(pK), m_value(std::abs(value)) {}
	GKernelTranslate(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->field("k"))), m_value(pNode->field("v")->asDouble()) {}
	virtual ~GKernelTranslate() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "k", m_pK->serialize(pDoc));
		pObj->addField(pDoc, "v", pDoc->newDouble(m_value));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "translate"; }

	/// Computes K(A, B) + value
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return m_pK->apply(pA, pB, dims) + m_value;
	}
};

/// A scalar kernel
class GKernelScale : public GKernel
{
protected:
	GKernel* m_pK;
	double m_value;

public:
	/// Takes ownership of pK
	GKernelScale(GKernel* pK, double value) : GKernel(), m_pK(pK), m_value(std::abs(value)) {}
	GKernelScale(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->field("k"))), m_value(pNode->field("v")->asDouble()) {}
	virtual ~GKernelScale() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "k", m_pK->serialize(pDoc));
		pObj->addField(pDoc, "v", pDoc->newDouble(m_value));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "scale"; }

	/// Computes K(A, B) * value
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return m_pK->apply(pA, pB, dims) * m_value;
	}
};

/// An addition kernel
class GKernelAdd : public GKernel
{
protected:
	GKernel* m_pK1;
	GKernel* m_pK2;

public:
	/// Takes ownership of pK1 and pK2
	GKernelAdd(GKernel* pK1, GKernel* pK2) : GKernel(), m_pK1(pK1), m_pK2(pK2) {}
	GKernelAdd(GDomNode* pNode) : GKernel(), m_pK1(GKernel::deserialize(pNode->field("k1"))), m_pK2(GKernel::deserialize(pNode->field("k2"))) {}
	virtual ~GKernelAdd() { delete(m_pK1); delete(m_pK2); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "k1", m_pK1->serialize(pDoc));
		pObj->addField(pDoc, "k2", m_pK2->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "add"; }

	/// Computes K1(A, B) + K2(A, B)
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return m_pK1->apply(pA, pB, dims) + m_pK2->apply(pA, pB, dims);
	}
};

/// A multiplication kernel
class GKernelMultiply : public GKernel
{
protected:
	GKernel* m_pK1;
	GKernel* m_pK2;

public:
	/// Takes ownership of pK1 and pK2
	GKernelMultiply(GKernel* pK1, GKernel* pK2) : GKernel(), m_pK1(pK1), m_pK2(pK2) {}
	GKernelMultiply(GDomNode* pNode) : GKernel(), m_pK1(GKernel::deserialize(pNode->field("k1"))), m_pK2(GKernel::deserialize(pNode->field("k2"))) {}
	virtual ~GKernelMultiply() { delete(m_pK1); delete(m_pK2); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "k1", m_pK1->serialize(pDoc));
		pObj->addField(pDoc, "k2", m_pK2->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "multiply"; }

	/// Computes K1(A, B) * K2(A, B)
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return m_pK1->apply(pA, pB, dims) * m_pK2->apply(pA, pB, dims);
	}
};

/// A power kernel
class GKernelPow : public GKernel
{
protected:
	GKernel* m_pK;
	double m_value;

public:
	/// Takes ownership of pK
	GKernelPow(GKernel* pK, unsigned int value) : GKernel(), m_pK(pK), m_value(value) {}
	GKernelPow(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->field("k"))), m_value(pNode->field("v")->asDouble()) {}
	virtual ~GKernelPow() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "k", m_pK->serialize(pDoc));
		pObj->addField(pDoc, "v", pDoc->newDouble(m_value));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "pow"; }

	/// Computes K(A, B)^value
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return pow(m_pK->apply(pA, pB, dims), m_value);
	}
};

/// The Exponential kernel
class GKernelExp : public GKernel
{
protected:
	GKernel* m_pK;

public:
	/// Takes ownership of pK
	GKernelExp(GKernel* pK) : GKernel(), m_pK(pK) {}
	GKernelExp(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->field("k"))) {}
	virtual ~GKernelExp() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "k", m_pK->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "exp"; }

	/// Computes e^K(A, B)
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return exp(m_pK->apply(pA, pB, dims));
	}
};

/// A Normalizing kernel
class GKernelNormalize : public GKernel
{
protected:
	GKernel* m_pK;

public:
	/// Takes ownership of pK
	GKernelNormalize(GKernel* pK) : GKernel(), m_pK(pK) {}
	GKernelNormalize(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->field("k"))) {}
	virtual ~GKernelNormalize() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->addField(pDoc, "k", m_pK->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "normalize"; }

	/// Computes K(A, B) / sqrt(K(A, A) * K(B, B))
	virtual double apply(const double* pA, const double* pB, size_t dims)
	{
		return m_pK->apply(pA, pB, dims) / sqrt(m_pK->apply(pA, pA, dims) * m_pK->apply(pB, pB, dims));
	}
};


} // namespace GClasses

#endif // __GKERNELTRICK_H__
