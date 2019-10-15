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

#include "GNaiveBayes.h"
#include "GError.h"
#include <math.h>
#include <stdlib.h>
#include "GVec.h"
#include "GDom.h"
#include "GDistribution.h"
#include "GRand.h"
#include "GTransform.h"
#include "GSparseMatrix.h"
#include "GHolders.h"
#include <cmath>
#include <memory>

namespace GClasses {

struct GNaiveBayesInputAttr
{
	size_t m_nValues;
	size_t* m_pValueCounts;

	GNaiveBayesInputAttr(size_t nValues)
	{
		m_nValues = nValues;
		m_pValueCounts = new size_t[m_nValues];
		memset(m_pValueCounts, '\0', sizeof(size_t) * m_nValues);
	}

	GNaiveBayesInputAttr(GDomNode* pNode)
	{
		GDomListIterator it(pNode);
		m_nValues = it.remaining();
		m_pValueCounts = new size_t[m_nValues];
		for(size_t i = 0; i < m_nValues; i++)
		{
			m_pValueCounts[i] = (size_t)it.currentInt();
			it.advance();
		}
	}

	~GNaiveBayesInputAttr()
	{
		delete[] m_pValueCounts;
	}

	GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pNode = pDoc->newList();
		for(size_t i = 0; i < m_nValues; i++)
			pNode->add(pDoc, m_pValueCounts[i]);
		return pNode;
	}

	void AddTrainingSample(const int inputValue)
	{
		if(inputValue >= 0 && (size_t)inputValue < m_nValues)
			m_pValueCounts[inputValue]++;
		else
			GAssert(inputValue == UNKNOWN_DISCRETE_VALUE);
	}

	size_t eval(const int inputValue)
	{
		if(inputValue >= 0 && (size_t)inputValue < m_nValues)
			return m_pValueCounts[inputValue];
		else
			return 0;
	}
};

// --------------------------------------------------------------------

struct GNaiveBayesOutputValue
{
	size_t m_nCount;
	size_t m_featureDims;
	struct GNaiveBayesInputAttr** m_pInputs;

	GNaiveBayesOutputValue(GRelation* pFeatureRel, size_t nInputCount)
	{
		m_nCount = 0;
		m_featureDims = nInputCount;
		m_pInputs = new struct GNaiveBayesInputAttr*[nInputCount];
		for(size_t n = 0; n < m_featureDims; n++)
			m_pInputs[n] = new struct GNaiveBayesInputAttr(pFeatureRel->valueCount(n));
	}

	GNaiveBayesOutputValue(GDomNode* pNode, size_t nInputCount)
	{
		GDomListIterator it(pNode);
		if(it.remaining() != nInputCount + 1)
			throw Ex("Unexpected number of inputs");
		m_nCount = (size_t)it.currentInt();
		it.advance();
		m_featureDims = nInputCount;
		m_pInputs = new struct GNaiveBayesInputAttr*[m_featureDims];
		for(size_t n = 0; n < m_featureDims; n++)
		{
			m_pInputs[n] = new struct GNaiveBayesInputAttr(it.current());
			it.advance();
		}
	}

	~GNaiveBayesOutputValue()
	{
		for(size_t n = 0; n < m_featureDims; n++)
			delete(m_pInputs[n]);
		delete[] m_pInputs;
	}

	GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pNode = pDoc->newList();
		pNode->add(pDoc, m_nCount);
		for(size_t i = 0; i < m_featureDims; i++)
			pNode->add(pDoc, m_pInputs[i]->serialize(pDoc));
		return pNode;
	}

	void AddTrainingSample(const double* pIn)
	{
		for(size_t n = 0; n < m_featureDims; n++)
			m_pInputs[n]->AddTrainingSample((int)pIn[n]);
		m_nCount++;
	}

	double eval(const double* pInputVector, double equivalentSampleSize)
	{
		// The prior output probability
		double dLogProb = log((double)m_nCount);

		// The probability of inputs given this output
		for(size_t n = 0; n < m_featureDims; n++)
		{
			dLogProb += log(std::max(1e-300,
					(
						(double)m_pInputs[n]->eval((int)pInputVector[n]) +
						(equivalentSampleSize / m_pInputs[n]->m_nValues)
					) /
					(equivalentSampleSize + m_nCount)
				));
		}
		return dLogProb;
	}
};

// --------------------------------------------------------------------

struct GNaiveBayesOutputAttr
{
	size_t m_nValueCount;
	struct GNaiveBayesOutputValue** m_pValues;

	GNaiveBayesOutputAttr(GRelation* pFeatureRel, size_t nInputCount, size_t nValueCount)
	{
		m_nValueCount = nValueCount;
		m_pValues = new struct GNaiveBayesOutputValue*[m_nValueCount];
		for(size_t n = 0; n < m_nValueCount; n++)
			m_pValues[n] = new struct GNaiveBayesOutputValue(pFeatureRel, nInputCount);
	}

	GNaiveBayesOutputAttr(GDomNode* pNode, size_t nInputCount, size_t nValueCount)
	{
		GDomListIterator it(pNode);
		if(it.remaining() != nValueCount)
			throw Ex("Unexpected number of values");
		m_nValueCount = nValueCount;
		m_pValues = new struct GNaiveBayesOutputValue*[m_nValueCount];
		for(size_t n = 0; n < m_nValueCount; n++)
		{
			m_pValues[n] = new struct GNaiveBayesOutputValue(it.current(), nInputCount);
			it.advance();
		}
	}

	~GNaiveBayesOutputAttr()
	{
		for(size_t n = 0; n < m_nValueCount; n++)
			delete(m_pValues[n]);
		delete[] m_pValues;
	}

	GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pNode = pDoc->newList();
		for(size_t i = 0; i < m_nValueCount; i++)
			pNode->add(pDoc, m_pValues[i]->serialize(pDoc));
		return pNode;
	}

	void AddTrainingSample(const double* pIn, int out)
	{
		if(out >= 0 && (size_t)out < m_nValueCount)
			m_pValues[out]->AddTrainingSample(pIn);
	}

	void eval(const double* pIn, GPrediction* pOut, double equivalentSampleSize)
	{
		GCategoricalDistribution* pDist = pOut->makeCategorical();
		GVec& values = pDist->values(m_nValueCount);
		for(size_t n = 0; n < m_nValueCount; n++)
			values[n] = m_pValues[n]->eval(pIn, equivalentSampleSize);
		pDist->normalizeFromLogSpace();
	}

	double predict(const double* pIn, double equivalentSampleSize, GRand* pRand)
	{
		GQUICKVEC(values, m_nValueCount);
		for(size_t n = 0; n < m_nValueCount; n++)
			values[n] = m_pValues[n]->eval(pIn, equivalentSampleSize);
		return (double)values.indexOfMax();
	}
};

// --------------------------------------------------------------------

GNaiveBayes::GNaiveBayes()
: GIncrementalLearner()
{
	m_pOutputs = NULL;
	m_equivalentSampleSize = 0.5;
	m_nSampleCount = 0;
}

GNaiveBayes::GNaiveBayes(const GDomNode* pNode)
: GIncrementalLearner(pNode)
{
	m_nSampleCount = (size_t)pNode->getInt("sampleCount");
	m_equivalentSampleSize = pNode->getDouble("ess");
	GDomNode* pOutputs = pNode->get("outputs");
	GDomListIterator it(pOutputs);
	if(it.remaining() != m_pRelLabels->size())
		throw Ex("Wrong number of outputs");
	m_pOutputs = new struct GNaiveBayesOutputAttr*[m_pRelLabels->size()];
	for(size_t i = 0; i < m_pRelLabels->size(); i++)
	{
		m_pOutputs[i] = new struct GNaiveBayesOutputAttr(it.current(), m_pRelFeatures->size(), m_pRelLabels->valueCount(i));
		it.advance();
	}
}

GNaiveBayes::~GNaiveBayes()
{
	clear();
}

// virtual
GDomNode* GNaiveBayes::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNaiveBayes");
	pNode->add(pDoc, "sampleCount", m_nSampleCount);
	pNode->add(pDoc, "ess", m_equivalentSampleSize);
	GDomNode* pOutputs = pNode->add(pDoc, "outputs", pDoc->newList());
	for(size_t i = 0; i < m_pRelLabels->size(); i++)
		pOutputs->add(pDoc, m_pOutputs[i]->serialize(pDoc));
	return pNode;
}

// virtual
void GNaiveBayes::clear()
{
	m_nSampleCount = 0;
	if(m_pOutputs)
	{
		for(size_t n = 0; n < m_pRelLabels->size(); n++)
			delete(m_pOutputs[n]);
		delete[] m_pOutputs;
	}
	m_pOutputs = NULL;
}

// virtual
void GNaiveBayes::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	clear();
	m_pOutputs = new struct GNaiveBayesOutputAttr*[m_pRelLabels->size()];
	for(size_t n = 0; n < m_pRelLabels->size(); n++)
		m_pOutputs[n] = new struct GNaiveBayesOutputAttr(m_pRelFeatures, m_pRelFeatures->size(), m_pRelLabels->valueCount(n));
}

// virtual
void GNaiveBayes::trainIncremental(const GVec& in, const GVec& out)
{
	for(size_t n = 0; n < m_pRelLabels->size(); n++)
		m_pOutputs[n]->AddTrainingSample(in.data(), (int)out[n]);
	m_nSampleCount++;
}

// virtual
void GNaiveBayes::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(!features.relation().areNominal())
		throw Ex("GNaiveBayes only supports nominal features. Perhaps you should wrap it in a GAutoFilter.");
	if(!labels.relation().areNominal())
		throw Ex("GNaiveBayes only supports nominal labels. Perhaps you should wrap it in a GAutoFilter.");
	beginIncrementalLearningInner(features.relation(), labels.relation());
	for(size_t n = 0; n < features.rows(); n++)
		trainIncremental(features[n], labels[n]);
}

// virtual
void GNaiveBayes::trainSparse(GSparseMatrix& features, GMatrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	size_t featureDims = features.cols();
	GUniformRelation featureRel(featureDims, 2);
	beginIncrementalLearning(featureRel, labels.relation());
	GVec fullRow(featureDims);
	for(size_t n = 0; n < features.rows(); n++)
	{
		features.fullRow(fullRow, n);
		for(size_t i = 0; i < featureDims; i++)
		{
			if(fullRow[i] < 1e-6)
				fullRow[i] = 0.0;
			else
				fullRow[i] = 1.0;
		}
		trainIncremental(fullRow, labels[n]);
	}
}

void GNaiveBayes::predictDistribution(const GVec& in, GPrediction* out)
{
	if(m_nSampleCount <= 0)
		throw Ex("You must call train before you call eval");
	for(size_t n = 0; n < m_pRelLabels->size(); n++)
		m_pOutputs[n]->eval(in.data(), &out[n], m_equivalentSampleSize);
}

void GNaiveBayes::predict(const GVec& in, GVec& out)
{
	if(m_nSampleCount <= 0)
		throw Ex("You must call train before you call eval");
	for(size_t n = 0; n < m_pRelLabels->size(); n++)
		out[n] = m_pOutputs[n]->predict(in.data(), m_equivalentSampleSize, &m_rand);
}

void GNaiveBayes::autoTune(GMatrix& features, GMatrix& labels)
{
	// Find the best ess value
	double bestEss = 0.0;
	double bestErr = 1e308;
	for(double i = 0.0; i < 8; i += 0.25)
	{
		m_equivalentSampleSize = i;
		double d = crossValidate(features, labels, 2);
		if(d < bestErr)
		{
			bestErr = d;
			bestEss = i;
		}
		else if(i >= 2.0)
			break;
	}

	// Set the best values
	m_equivalentSampleSize = bestEss;
}

void GNaiveBayes_CheckResults(double yprior, double ycond, double nprior, double ncond, GPrediction* out)
{
	double py = yprior * ycond;
	double pn = nprior * ncond;
	double sum = py + pn;
	py /= sum;
	pn /= sum;
	GCategoricalDistribution* pCat = out->asCategorical();
	GVec& vals = pCat->values(2);
	if(std::abs(vals[0] - py) > 1e-8)
		throw Ex("wrong");
	if(std::abs(vals[1] - pn) > 1e-8)
		throw Ex("wrong");
}

void GNaiveBayes_testMath()
{
	const char* trainFile =
	"@RELATION test\n"
	"@ATTRIBUTE a {t,f}\n"
	"@ATTRIBUTE b {r,g,b}\n"
	"@ATTRIBUTE c {y,n}\n"
	"@DATA\n"
	"t,r,y\n"
	"f,r,n\n"
	"t,g,y\n"
	"f,g,y\n"
	"f,g,n\n"
	"t,r,n\n"
	"t,r,y\n"
	"t,b,y\n"
	"f,r,y\n"
	"f,g,n\n"
	"f,b,y\n"
	"t,r,n\n";
	GMatrix train;
	train.parseArff(trainFile, strlen(trainFile));
	GMatrix* pFeatures = new GMatrix(train, 0, 0, train.rows(), 2);
	std::unique_ptr<GMatrix> hFeatures(pFeatures);
	GMatrix* pLabels = new GMatrix(train, 0, 2, train.rows(), 1);
	std::unique_ptr<GMatrix> hLabels(pLabels);
	GNaiveBayes nb;
	nb.setEquivalentSampleSize(0.0);
	nb.train(*pFeatures, *pLabels);
	GPrediction out;
	GVec pat(2);
	pat[0] = 0; pat[1] = 0;
	nb.predictDistribution(pat, &out);
	GNaiveBayes_CheckResults(7.0/12.0, 4.0/7.0*3.0/7.0, 5.0/12.0, 2.0/5.0*3.0/5.0, &out);
	pat[0] = 0; pat[1] = 1;
	nb.predictDistribution(pat, &out);
	GNaiveBayes_CheckResults(7.0/12.0, 4.0/7.0*2.0/7.0, 5.0/12.0, 2.0/5.0*2.0/5.0, &out);
	pat[0] = 0; pat[1] = 2;
	nb.predictDistribution(pat, &out);
	GNaiveBayes_CheckResults(7.0/12.0, 4.0/7.0*2.0/7.0, 5.0/12.0, 2.0/5.0*0.0/5.0, &out);
	pat[0] = 1; pat[1] = 0;
	nb.predictDistribution(pat, &out);
	GNaiveBayes_CheckResults(7.0/12.0, 3.0/7.0*3.0/7.0, 5.0/12.0, 3.0/5.0*3.0/5.0, &out);
	pat[0] = 1; pat[1] = 1;
	nb.predictDistribution(pat, &out);
	GNaiveBayes_CheckResults(7.0/12.0, 3.0/7.0*2.0/7.0, 5.0/12.0, 3.0/5.0*2.0/5.0, &out);
	pat[0] = 1; pat[1] = 2;
	nb.predictDistribution(pat, &out);
	GNaiveBayes_CheckResults(7.0/12.0, 3.0/7.0*2.0/7.0, 5.0/12.0, 3.0/5.0*0.0/5.0, &out);
}

// static
void GNaiveBayes::test()
{
	GNaiveBayes_testMath();
	GAutoFilter af(new GNaiveBayes());
	af.basicTest(0.77, 0.94);
}

} // namespace GClasses
