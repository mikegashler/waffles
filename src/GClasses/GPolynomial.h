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

#ifndef __GPOLYNOMIAL_H__
#define __GPOLYNOMIAL_H__

#include "GLearner.h"
#include <vector>

namespace GClasses {

class GPolynomialSingleLabel;


/// This regresses a multi-dimensional polynomial to fit the data
class GPolynomial : public GSupervisedLearner
{
protected:
	size_t m_controlPoints;
	std::vector<GPolynomialSingleLabel*> m_polys;

public:
	/// It will have the same number of control points in every feature dimension
	GPolynomial();

	/// Load from a DOM.
	GPolynomial(const GDomNode* pNode);

	virtual ~GPolynomial();

	/// Set the number of control points in the Bezier representation of the
	/// polynomial (which is one more than the polynomial order). The default
	/// is 3.
	void setControlPoints(size_t n);

	/// Returns the number of control points.
	size_t controlPoints();

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data.
	void autoTune(GMatrix& features, GMatrix& labels);

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }
};


} // namespace GClasses

#endif // __GPOLYNOMIAL_H__
