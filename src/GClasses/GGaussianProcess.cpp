/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GGaussianProcess.h"
#include "GRand.h"
#include "GDom.h"
#include "GVec.h"
#include "GDistribution.h"
#include "GKernelTrick.h"
#include <cmath>

namespace GClasses {

GRunningCovariance::GRunningCovariance(size_t dims) : m_counts(dims, dims), m_sums(dims + 1, dims)
{
	m_counts.setAll(0.0);
	m_sums.setAll(0.0);
}

GRunningCovariance::~GRunningCovariance()
{
}

double GRunningCovariance::element(size_t row, size_t col)
{
	if(col < row)
		std::swap(row, col);
	double n = m_counts[row][col];
	if(n <= 1.0)
		return UNKNOWN_REAL_VALUE;
	return n / (n - 1) * (m_sums[row][col] / m_counts[row][col] - (m_sums[m_counts.rows()][row] * m_sums[m_counts.rows()][col]) / (m_counts[row][row] * m_counts[col][col]));
}

void GRunningCovariance::add(const double* pVec)
{
	for(size_t i = 0; i < m_counts.rows(); i++)
	{
		if(pVec[i] != UNKNOWN_REAL_VALUE)
		{
			m_sums[m_counts.rows()][i] += pVec[i];
			for(size_t j = i; j < m_counts.rows(); j++)
			{
				if(pVec[j] != UNKNOWN_REAL_VALUE)
				{
					m_counts[i][j]++;
					m_sums[i][j] += pVec[i] * pVec[j];
				}
			}
		}
	}
}

void GRunningCovariance::decay(double gamma)
{
	m_counts.multiply(gamma);
	m_sums.multiply(gamma);
}

#ifndef NO_TEST_CODE
// static
void GRunningCovariance::test()
{
	GRand rand(0);
	GMatrix m(13, 7);
	for(size_t i = 0; i < m.rows(); i++)
		rand.cubical(m[i], m.cols());
	GMatrix* pCov1 = m.covarianceMatrix();
	Holder<GMatrix> hCov1(pCov1);
	m.centerMeanAtOrigin();
	GRunningCovariance rc(m.cols());
	for(size_t i = 0; i < m.rows(); i++)
		rc.add(m[i]);
	for(size_t i = 0; i < m.cols(); i++)
	{
		for(size_t j = 0; j < m.cols(); j++)
		{
			if(std::abs(rc.element(i, j) - pCov1->row(i)[j]) > 1e-12)
				throw Ex("failed");
		}
	}
}
#endif









GGaussianProcess::GGaussianProcess(GRand& rand)
: GSupervisedLearner(rand), m_noiseVar(1.0), m_weightsPriorVar(1024.0), m_maxSamples(350), m_pLInv(NULL), m_pAlpha(NULL), m_pStoredFeatures(NULL), m_pBuf(NULL)
{
	m_pKernel = new GKernelIdentity();
}

GGaussianProcess::GGaussianProcess(GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode, ll), m_pBuf(NULL)
{
	m_weightsPriorVar = pNode->field("wv")->asDouble();
	m_noiseVar = pNode->field("nv")->asDouble();
	m_maxSamples = (size_t)pNode->field("ms")->asInt();
	m_pLInv = new GMatrix(pNode->field("l"));
	m_pAlpha = new GMatrix(pNode->field("a"));
	m_pStoredFeatures = new GMatrix(pNode->field("feat"));
	m_pKernel = GKernel::deserialize(pNode->field("kernel"));
}

// virtual
GGaussianProcess::~GGaussianProcess()
{
	clear();
	delete(m_pKernel);
}

#ifndef NO_TEST_CODE
// static
void GGaussianProcess::test()
{
	GRand prng(0);
	GGaussianProcess gp(prng);
	gp.basicTest(0.693, 0.776);
	gp.clear();
	GGaussianProcess gp2(prng);
	gp2.setKernel(new GKernelGaussianRBF(0.2));
	gp.basicTest(0.680, 0.774);
}
#endif

// virtual
GDomNode* GGaussianProcess::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GGaussianProcess");
	pNode->addField(pDoc, "wv", pDoc->newDouble(m_weightsPriorVar));
	pNode->addField(pDoc, "nv", pDoc->newDouble(m_noiseVar));
	pNode->addField(pDoc, "ms", pDoc->newInt(m_maxSamples));
	pNode->addField(pDoc, "l", m_pLInv->serialize(pDoc));
	pNode->addField(pDoc, "a", m_pAlpha->serialize(pDoc));
	pNode->addField(pDoc, "feat", m_pStoredFeatures->serialize(pDoc));
	pNode->addField(pDoc, "kernel", m_pKernel->serialize(pDoc));
	return pNode;
}

// virtual
void GGaussianProcess::clear()
{
	delete(m_pLInv);
	m_pLInv = NULL;
	delete(m_pAlpha);
	m_pAlpha = NULL;
	delete(m_pStoredFeatures);
	m_pStoredFeatures = NULL;
	delete(m_pBuf);
	m_pBuf = NULL;
}

// virtual
void GGaussianProcess::trainInner(GMatrix& features, GMatrix& labels)
{
	if(features.rows() <= m_maxSamples)
	{
		trainInnerInner(features, labels);
		return;
	}
	GMatrix f(features.relation());
	GReleaseDataHolder hF(&f);
	GMatrix l(labels.relation());
	GReleaseDataHolder hL(&l);
	for(size_t i = 0; i < features.rows(); i++)
	{
		f.takeRow(features[i]);
		l.takeRow(labels[i]);
	}
	GRand rand(0);
	while(f.rows() > m_maxSamples)
	{
		size_t i = (size_t)rand.next(f.rows());
		f.releaseRow(i);
		l.releaseRow(i);
	}
	trainInnerInner(f, l);
}

void GGaussianProcess::trainInnerInner(GMatrix& features, GMatrix& labels)
{
	clear();
	size_t dims = features.cols();
	GMatrix* pL;
	{
		// Compute the kernel matrix
		GMatrix k(features.rows(), features.rows());
		for(size_t i = 0; i < features.rows(); i++)
		{
			double* pRow = k[i];
			double* pA = features[i];
			for(size_t j = 0; j < features.rows(); j++)
			{
				double* pB = features[j];
				*pRow = m_weightsPriorVar * m_pKernel->apply(pA, pB, dims);
				pRow++;
			}
		}

		// Add the noise variance to the diagonal of the kernel matrix
		for(size_t i = 0; i < features.rows(); i++)
			k[i][i] += m_noiseVar;

		// Compute L
		pL = k.cholesky(true);
	}
	Holder<GMatrix> hL(pL);

	// Compute the model
	m_pLInv = pL->pseudoInverse();
	GMatrix* pTmp = GMatrix::multiply(*m_pLInv, labels, false, false);
	Holder<GMatrix> hTmp(pTmp);
	GMatrix* pLTrans = pL->transpose();
	Holder<GMatrix> hLTrans(pLTrans);
	GMatrix* pLTransInv = pLTrans->pseudoInverse();
	Holder<GMatrix> hLTransInv(pLTransInv);
	m_pAlpha = GMatrix::multiply(*pLTransInv, *pTmp, false, false);
	GAssert(m_pAlpha->rows() == features.rows());
	GAssert(m_pAlpha->cols() == labels.cols());
	m_pStoredFeatures = features.clone();
}

// virtual
void GGaussianProcess::predictInner(const double* pIn, double* pOut)
{
	if(!m_pBuf)
		m_pBuf = new GMatrix(1, m_pStoredFeatures->rows());

	// Compute k*
	double* pK = m_pBuf->row(0);
	size_t dims = m_pStoredFeatures->cols();
	for(size_t i = 0; i < m_pStoredFeatures->rows(); i++)
		*(pK++) = m_weightsPriorVar * m_pKernel->apply(m_pStoredFeatures->row(i), pIn, dims);

	// Compute the prediction
	m_pAlpha->multiply(m_pBuf->row(0), pOut, true);
}

// virtual
void GGaussianProcess::predictDistributionInner(const double* pIn, GPrediction* pOut)
{
	if(!m_pBuf)
		m_pBuf = new GMatrix(2, m_pStoredFeatures->rows());
	else if(m_pBuf->rows() < 2)
		m_pBuf->newRow();

	// Compute k*
	double* pK = m_pBuf->row(0);
	size_t dims = m_pStoredFeatures->cols();
	for(size_t i = 0; i < m_pStoredFeatures->rows(); i++)
		*(pK++) = m_weightsPriorVar * m_pKernel->apply(m_pStoredFeatures->row(i), pIn, dims);

	// Compute the prediction
	GTEMPBUF(double, pred, m_pAlpha->cols());
	m_pAlpha->multiply(m_pBuf->row(0), pred, true);

	// Compute the variance
	double* pV = m_pBuf->row(1);
	m_pLInv->multiply(pK, pV);
	double variance = m_pKernel->apply(pK, pK, m_pStoredFeatures->rows()) - GVec::squaredMagnitude(pV, m_pLInv->rows());

	// Store the results
	for(size_t i = 0; i < m_pAlpha->cols(); i++)
	{
		GNormalDistribution* pNorm = pOut->makeNormal();
		pNorm->setMeanAndVariance(*pred, variance);
		pred++;
	}
}

void GGaussianProcess::setKernel(GKernel* pKernel)
{
	delete(m_pKernel);
	m_pKernel = pKernel;
}


} // namespace GClasses
