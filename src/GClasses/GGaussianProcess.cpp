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

#include "GGaussianProcess.h"
#include "GRand.h"
#include "GDom.h"
#include "GVec.h"
#include "GDistribution.h"
#include "GKernelTrick.h"
#include "GHolders.h"
#include <cmath>
#include <memory>

namespace GClasses {

GRunningCovariance::GRunningCovariance(size_t dims) : m_counts(dims, dims), m_sums(dims + 1, dims)
{
	m_counts.fill(0.0);
	m_sums.fill(0.0);
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

void GRunningCovariance::add(const GVec& vec)
{
	for(size_t i = 0; i < m_counts.rows(); i++)
	{
		if(vec[i] != UNKNOWN_REAL_VALUE)
		{
			m_sums[m_counts.rows()][i] += vec[i];
			for(size_t j = i; j < m_counts.rows(); j++)
			{
				if(vec[j] != UNKNOWN_REAL_VALUE)
				{
					m_counts[i][j]++;
					m_sums[i][j] += vec[i] * vec[j];
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

// static
void GRunningCovariance::test()
{
	GRand rand(0);
	GMatrix m(13, 7);
	for(size_t i = 0; i < m.rows(); i++)
		m[i].fillUniform(rand);
	GMatrix* pCov1 = m.covarianceMatrix();
	std::unique_ptr<GMatrix> hCov1(pCov1);
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









GGaussianProcess::GGaussianProcess()
: GSupervisedLearner(), m_noiseVar(1.0), m_weightsPriorVar(1024.0), m_maxSamples(350), m_pLInv(NULL), m_pAlpha(NULL), m_pStoredFeatures(NULL), m_pBuf(NULL)
{
	m_pKernel = new GKernelIdentity();
}

GGaussianProcess::GGaussianProcess(const GDomNode* pNode)
: GSupervisedLearner(pNode), m_pBuf(NULL)
{
	m_weightsPriorVar = pNode->getDouble("wv");
	m_noiseVar = pNode->getDouble("nv");
	m_maxSamples = (size_t)pNode->getInt("ms");
	m_pLInv = new GMatrix(pNode->get("l"));
	m_pAlpha = new GMatrix(pNode->get("a"));
	m_pStoredFeatures = new GMatrix(pNode->get("feat"));
	m_pKernel = GKernel::deserialize(pNode->get("kernel"));
}

// virtual
GGaussianProcess::~GGaussianProcess()
{
	clear();
	delete(m_pKernel);
}

// static
void GGaussianProcess::test()
{
	GAutoFilter af(new GGaussianProcess());
	af.basicTest(0.693, 0.94);
	af.clear();
	GGaussianProcess* pGP = new GGaussianProcess();
	pGP->setKernel(new GKernelGaussianRBF(0.2));
	GAutoFilter af2(pGP);
	af2.basicTest(0.67, 0.92);
}

// virtual
GDomNode* GGaussianProcess::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GGaussianProcess");
	pNode->add(pDoc, "wv", m_weightsPriorVar);
	pNode->add(pDoc, "nv", m_noiseVar);
	pNode->add(pDoc, "ms", m_maxSamples);
	pNode->add(pDoc, "l", m_pLInv->serialize(pDoc));
	pNode->add(pDoc, "a", m_pAlpha->serialize(pDoc));
	pNode->add(pDoc, "feat", m_pStoredFeatures->serialize(pDoc));
	pNode->add(pDoc, "kernel", m_pKernel->serialize(pDoc));
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
void GGaussianProcess::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(!features.relation().areContinuous())
		throw Ex("GGaussianProcess only supports continuous features. Perhaps you should wrap it in a GAutoFilter.");
	if(!labels.relation().areContinuous())
		throw Ex("GGaussianProcess only supports continuous labels. Perhaps you should wrap it in a GAutoFilter.");
	if(features.rows() <= m_maxSamples)
	{
		trainInnerInner(features, labels);
		return;
	}
	GMatrix f(features.relation().clone());
	GReleaseDataHolder hF(&f);
	GMatrix l(labels.relation().clone());
	GReleaseDataHolder hL(&l);
	for(size_t i = 0; i < features.rows(); i++)
	{
		f.takeRow((GVec*)&features[i]);
		l.takeRow((GVec*)&labels[i]);
	}
	while(f.rows() > m_maxSamples)
	{
		size_t i = (size_t)m_rand.next(f.rows());
		f.releaseRow(i);
		l.releaseRow(i);
	}
	trainInnerInner(f, l);
}

void GGaussianProcess::trainInnerInner(const GMatrix& features, const GMatrix& labels)
{
	clear();
	GMatrix* pL;
	{
		// Compute the kernel matrix
		GMatrix k(features.rows(), features.rows());
		for(size_t i = 0; i < features.rows(); i++)
		{
			GVec& row = k[i];
			const GVec& a = features[i];
			for(size_t j = 0; j < features.rows(); j++)
			{
				const GVec& b = features[j];
				row[j] = m_weightsPriorVar * m_pKernel->apply(a, b);
			}
		}

		// Add the noise variance to the diagonal of the kernel matrix
		for(size_t i = 0; i < features.rows(); i++)
			k[i][i] += m_noiseVar;

		// Compute L
		pL = k.cholesky(true);
	}
	std::unique_ptr<GMatrix> hL(pL);

	// Compute the model
	m_pLInv = pL->pseudoInverse();
	GMatrix* pTmp = GMatrix::multiply(*m_pLInv, labels, false, false);
	std::unique_ptr<GMatrix> hTmp(pTmp);
	GMatrix* pLTrans = pL->transpose();
	std::unique_ptr<GMatrix> hLTrans(pLTrans);
	GMatrix* pLTransInv = pLTrans->pseudoInverse();
	std::unique_ptr<GMatrix> hLTransInv(pLTransInv);
	m_pAlpha = GMatrix::multiply(*pLTransInv, *pTmp, false, false);
	GAssert(m_pAlpha->rows() == features.rows());
	GAssert(m_pAlpha->cols() == labels.cols());
	m_pStoredFeatures = new GMatrix();
	m_pStoredFeatures->copy(features);
}

// virtual
void GGaussianProcess::predict(const GVec& in, GVec& out)
{
	if(!m_pBuf)
		m_pBuf = new GMatrix(1, m_pStoredFeatures->rows());

	// Compute k*
	GVec& k = m_pBuf->row(0);
	for(size_t i = 0; i < m_pStoredFeatures->rows(); i++)
		k[i] = m_weightsPriorVar * m_pKernel->apply(m_pStoredFeatures->row(i), in);

	// Compute the prediction
	m_pAlpha->multiply(m_pBuf->row(0), out, true);
}

// virtual
void GGaussianProcess::predictDistribution(const GVec& in, GPrediction* out)
{
	if(!m_pBuf)
		m_pBuf = new GMatrix(2, m_pStoredFeatures->rows());
	else if(m_pBuf->rows() < 2)
		m_pBuf->newRow();

	// Compute k*
	GVec& k = m_pBuf->row(0);
	for(size_t i = 0; i < m_pStoredFeatures->rows(); i++)
		k[i] = m_weightsPriorVar * m_pKernel->apply(m_pStoredFeatures->row(i), in);

	// Compute the prediction
	GVec pred;
	m_pAlpha->multiply(m_pBuf->row(0), pred, true);

	// Compute the variance
	GVec& v = m_pBuf->row(1);
	m_pLInv->multiply(k, v);
	double variance = m_pKernel->apply(k, k) - v.squaredMagnitude();

	// Store the results
	for(size_t i = 0; i < m_pAlpha->cols(); i++)
	{
		GNormalDistribution* pNorm = out->makeNormal();
		pNorm->setMeanAndVariance(pred[i], variance);
	}
}

void GGaussianProcess::setKernel(GKernel* pKernel)
{
	delete(m_pKernel);
	m_pKernel = pKernel;
}


} // namespace GClasses
