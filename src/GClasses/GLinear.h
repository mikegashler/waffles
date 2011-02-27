/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GLINEAR_H__
#define __GLINEAR_H__

#include "GLearner.h"
#include <vector>

namespace GClasses {

class GPCA;

/// A linear regression algorithm. Only supports 1 label dim.
class GLinearRegressor : public GSupervisedLearner
{
protected:
	GRand* m_pRand;
	GPCA* m_pPCA;

public:
	GLinearRegressor(GRand* pRand);

	/// Load from a text-format
	GLinearRegressor(GTwtNode* pNode, GRand* pRand);

	virtual ~GLinearRegressor();

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif

	/// Saves the model to a text file. (This doesn't save the short-term
	/// memory used for incremental learning, so if you're doing "incremental"
	/// learning, it will wake up with amnesia when you load it again.)
	virtual GTwtNode* toTwt(GTwtDoc* pDoc);

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Returns the random number generator associated with this learner
	GRand* getRand() { return m_pRand; }

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);

	/// See the comment for GSupervisedLearner::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GSupervisedLearner::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }
};



class GLinearProgramming
{
public:
	/// Compute x that maximizes c*x, subject to Ax<=b, x>=0.
	/// The size of pB is the number of rows in pA.
	/// The size of pC is the number of columns in pA.
	/// leConstraints specifies the number of <= constraints. (These must come first in order.)
	/// geConstraints specifies the number of >= constraints. (These come next.)
	/// The remaining constraints are assumed to be = constraints.
	/// The answer is put in pOutX, which is the same size as pC.
	/// Returns false if there is no solution, and true if it finds a solution.
	static bool simplexMethod(GMatrix* pA, const double* pB, int leConstraints, int geConstraints, const double* pC, double* pOutX);

#ifndef NO_TEST_CODE
	/// Perform unit tests for this class. Throws an exception if any tests fail. Returns if they all pass.
	static void test();
#endif
};


} // namespace GClasses

#endif // __GLINEAR_H__

