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
	virtual double apply(const GVec& pA, const GVec& pB) = 0;

	/// Deserializes a kernel object
	static GKernel* deserialize(GDomNode* pNode);

	/// Returns a complex kernel made by combining several other kernels.
	/// This might be used to exercise kernel functionality or to test non-linear metrics.
	/// The caller is responsible to delete the object this returns.
	static GKernel* kernelComplex1();


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
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return pA.dotProductIgnoringUnknowns(pB);
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
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		double d = 0.0;
		for(size_t i = 0; i < pA.size(); i++)
		{
			if(pA[i] != UNKNOWN_REAL_VALUE && pB[i] != UNKNOWN_REAL_VALUE)
				d += 2.0 * pA[i] * pB[i] / (pA[i] + pB[i]);
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
	GKernelPolynomial(GDomNode* pNode) : GKernel(), m_offset(pNode->getDouble("offset")), m_order((unsigned int)pNode->getInt("order")) {}
	virtual ~GKernelPolynomial() {}

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "offset", m_offset);
		pObj->add(pDoc, "order", (long long)m_order);
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "polynomial"; }

	/// Computes (A * B + offset)^order
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return pow(pA.dotProductIgnoringUnknowns(pB) + m_offset, (int)m_order);
	}
};

/// A Gaussian RBF kernel
class GKernelGaussianRBF : public GKernel
{
protected:
	double m_variance;

public:
	GKernelGaussianRBF(double variance) : GKernel(), m_variance(std::abs(variance)) {}
	GKernelGaussianRBF(GDomNode* pNode) : GKernel(), m_variance(pNode->getDouble("var")) {}
	virtual ~GKernelGaussianRBF() {}

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "var", m_variance);
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "rbf"; }

	/// Computes e^(-0.5 * ||A - B||^2 / variance)
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return exp(-0.5 * pA.estimateSquaredDistanceWithUnknowns(pB) / m_variance);
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
	GKernelTranslate(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->get("k"))), m_value(pNode->getDouble("v")) {}
	virtual ~GKernelTranslate() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "k", m_pK->serialize(pDoc));
		pObj->add(pDoc, "v", m_value);
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "translate"; }

	/// Computes K(A, B) + value
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return m_pK->apply(pA, pB) + m_value;
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
	GKernelScale(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->get("k"))), m_value(pNode->getDouble("v")) {}
	virtual ~GKernelScale() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "k", m_pK->serialize(pDoc));
		pObj->add(pDoc, "v", m_value);
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "scale"; }

	/// Computes K(A, B) * value
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return m_pK->apply(pA, pB) * m_value;
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
	GKernelAdd(GDomNode* pNode) : GKernel(), m_pK1(GKernel::deserialize(pNode->get("k1"))), m_pK2(GKernel::deserialize(pNode->get("k2"))) {}
	virtual ~GKernelAdd() { delete(m_pK1); delete(m_pK2); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "k1", m_pK1->serialize(pDoc));
		pObj->add(pDoc, "k2", m_pK2->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "add"; }

	/// Computes K1(A, B) + K2(A, B)
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return m_pK1->apply(pA, pB) + m_pK2->apply(pA, pB);
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
	GKernelMultiply(GDomNode* pNode) : GKernel(), m_pK1(GKernel::deserialize(pNode->get("k1"))), m_pK2(GKernel::deserialize(pNode->get("k2"))) {}
	virtual ~GKernelMultiply() { delete(m_pK1); delete(m_pK2); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "k1", m_pK1->serialize(pDoc));
		pObj->add(pDoc, "k2", m_pK2->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "multiply"; }

	/// Computes K1(A, B) * K2(A, B)
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return m_pK1->apply(pA, pB) * m_pK2->apply(pA, pB);
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
	GKernelPow(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->get("k"))), m_value(pNode->getDouble("v")) {}
	virtual ~GKernelPow() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "k", m_pK->serialize(pDoc));
		pObj->add(pDoc, "v", m_value);
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "pow"; }

	/// Computes K(A, B)^value
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return pow(m_pK->apply(pA, pB), m_value);
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
	GKernelExp(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->get("k"))) {}
	virtual ~GKernelExp() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "k", m_pK->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "exp"; }

	/// Computes e^K(A, B)
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return exp(m_pK->apply(pA, pB));
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
	GKernelNormalize(GDomNode* pNode) : GKernel(), m_pK(GKernel::deserialize(pNode->get("k"))) {}
	virtual ~GKernelNormalize() { delete(m_pK); }

	/// Marshalls this object into a DOM.
	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = makeBaseNode(pDoc);
		pObj->add(pDoc, "k", m_pK->serialize(pDoc));
		return pObj;
	}

	/// Returns the name of this kernel
	virtual const char* name() const { return "normalize"; }

	/// Computes K(A, B) / sqrt(K(A, A) * K(B, B))
	virtual double apply(const GVec& pA, const GVec& pB)
	{
		return m_pK->apply(pA, pB) / sqrt(m_pK->apply(pA, pA) * m_pK->apply(pB, pB));
	}
};


} // namespace GClasses

#endif // __GKERNELTRICK_H__
