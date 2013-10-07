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

#include "GKalman.h"
#include "GMatrix.h"
#include "GVec.h"
#include "GHolders.h"

using namespace GClasses;

GExtendedKalmanFilter::GExtendedKalmanFilter(int stateDims, int observationDims, int controlDims)
: m_stateDims(stateDims), m_obsDims(observationDims)
{
	m_x = new double[stateDims + observationDims + stateDims];
	m_z = m_x + stateDims;
	m_zz = m_z + observationDims;
	GVec::setAll(m_x, 0.0, stateDims);
	m_pP = new GMatrix(stateDims, stateDims);
	m_pP->makeIdentity();
	for(size_t i = 0; i < m_pP->rows(); i++)
		m_pP->row(i)[i] *= 1000.0;
}

GExtendedKalmanFilter::~GExtendedKalmanFilter()
{
	delete[] m_x;
	delete(m_pP);
}

void GExtendedKalmanFilter::advance(const double* pControl, GMatrix* pA)
{
	// Check values
	GAssert(pA->rows() == m_stateDims && pA->cols() == m_stateDims); // transition Jacobian wrong size

	// Compute uncorrected next estimated state
	transition(m_x, pControl);

	// Compute uncorrected next estimated covariance of state
	GMatrix* pTemp = GMatrix::multiply(*m_pP, *pA, false, true);
	Holder<GMatrix> hTemp(pTemp);
	delete(m_pP);
	m_pP = GMatrix::multiply(*pA, *pTemp, false, false);
	addTransitionNoise(m_pP);
}

void GExtendedKalmanFilter::correct(const double* pObservation, GMatrix* pH)
{
	// Check values
	GAssert(pH->rows() == m_obsDims && pH->cols() == m_stateDims); // observation Jacobian wrong size

	// Compute the Kalman gain
	GMatrix* pK;
	{
		GMatrix* pTemp = GMatrix::multiply(*m_pP, *pH, false, true);
		Holder<GMatrix> hTemp(pTemp);
		GMatrix* pTemp2 = GMatrix::multiply(*pH, *pTemp, false, false);
		Holder<GMatrix> hTemp2(pTemp2);
		addObservationNoise(pTemp2);
		GMatrix* pTemp3 = pTemp2->pseudoInverse();
		Holder<GMatrix> hTemp3(pTemp3);
		pK = GMatrix::multiply(*pTemp, *pTemp3, false, false);
	}
	Holder<GMatrix> hK(pK);

	// Correct the estimated state
	observation(m_z, m_x);
	GVec::multiply(m_z, -1.0, m_obsDims);
	GVec::add(m_z, pObservation, m_obsDims);
	pK->multiply(m_z, m_zz, false);
	GVec::add(m_x, m_zz, m_stateDims);

	// Correct the estimated covariance of state
	{
		GMatrix* pTemp = GMatrix::multiply(*pK, *pH, false, false);
		Holder<GMatrix> hTemp(pTemp);
		pTemp->multiply(-1.0);
		for(size_t i = 0; i < m_stateDims; i++)
			pTemp->row(i)[i] += 1.0;
		GMatrix* pTemp2 = GMatrix::multiply(*pTemp, *m_pP, false, false);
		delete(m_pP);
		m_pP = pTemp2;
	}
}
