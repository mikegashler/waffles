/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GNaiveBayes.h"
#include "GError.h"
#include <math.h>
#include <stdlib.h>
#include "GVec.h"
#include "GTwt.h"
#include "GDistribution.h"
#include "GRand.h"
#include "GTransform.h"
#include "GSparseMatrix.h"
#include <cmath>

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

	GNaiveBayesInputAttr(GTwtNode* pNode)
	{
		m_nValues = pNode->itemCount();
		m_pValueCounts = new size_t[m_nValues];
		for(size_t i = 0; i < m_nValues; i++)
			m_pValueCounts[i] = (size_t)pNode->item(i)->asInt();
	}

	~GNaiveBayesInputAttr()
	{
		delete[] m_pValueCounts;
	}

	GTwtNode* toTwt(GTwtDoc* pDoc)
	{
		GTwtNode* pNode = pDoc->newList(m_nValues);
		for(size_t i = 0; i < m_nValues; i++)
			pNode->setItem(i, pDoc->newInt(m_pValueCounts[i]));
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

	GNaiveBayesOutputValue(GTwtNode* pNode, size_t nInputCount)
	{
		if(pNode->itemCount() != nInputCount + 1)
			ThrowError("Unexpected number of inputs");
		m_nCount = (size_t)pNode->item(0)->asInt();
		m_featureDims = nInputCount;
		m_pInputs = new struct GNaiveBayesInputAttr*[m_featureDims];
		for(size_t n = 0; n < m_featureDims; n++)
			m_pInputs[n] = new struct GNaiveBayesInputAttr(pNode->item(n + 1));
	}

	~GNaiveBayesOutputValue()
	{
		for(size_t n = 0; n < m_featureDims; n++)
			delete(m_pInputs[n]);
		delete[] m_pInputs;
	}

	GTwtNode* toTwt(GTwtDoc* pDoc)
	{
		GTwtNode* pNode = pDoc->newList(m_featureDims + 1);
		pNode->setItem(0, pDoc->newInt(m_nCount));
		for(size_t i = 0; i < m_featureDims; i++)
			pNode->setItem(i + 1, m_pInputs[i]->toTwt(pDoc));
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

	GNaiveBayesOutputAttr(GTwtNode* pNode, size_t nInputCount, size_t nValueCount)
	{
		if(pNode->itemCount() != nValueCount)
			ThrowError("Unexpected number of values");
		m_nValueCount = nValueCount;
		m_pValues = new struct GNaiveBayesOutputValue*[m_nValueCount];
		for(size_t n = 0; n < m_nValueCount; n++)
			m_pValues[n] = new struct GNaiveBayesOutputValue(pNode->item(n), nInputCount);
	}

	~GNaiveBayesOutputAttr()
	{
		for(size_t n = 0; n < m_nValueCount; n++)
			delete(m_pValues[n]);
		delete[] m_pValues;
	}

	GTwtNode* toTwt(GTwtDoc* pDoc)
	{
		GTwtNode* pNode = pDoc->newList(m_nValueCount);
		for(size_t i = 0; i < m_nValueCount; i++)
			pNode->setItem(i, m_pValues[i]->toTwt(pDoc));
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
		double* pValues = pDist->values(m_nValueCount);
		for(size_t n = 0; n < m_nValueCount; n++)
			pValues[n] = m_pValues[n]->eval(pIn, equivalentSampleSize);
		pDist->normalizeFromLogSpace();
	}

	double predict(const double* pIn, double equivalentSampleSize, GRand* pRand)
	{
		GTEMPBUF(double, pValues, m_nValueCount);
		for(size_t n = 0; n < m_nValueCount; n++)
			pValues[n] = m_pValues[n]->eval(pIn, equivalentSampleSize);
		return (double)GVec::indexOfMax(pValues, m_nValueCount, pRand);
	}
};

// --------------------------------------------------------------------

GNaiveBayes::GNaiveBayes(GRand* pRand)
: GIncrementalLearner(), m_pRand(pRand)
{
	m_pOutputs = NULL;
	m_equivalentSampleSize = 0.5;
	m_nSampleCount = 0;
}

GNaiveBayes::GNaiveBayes(GTwtNode* pNode, GRand* pRand)
: GIncrementalLearner(pNode, *pRand), m_pRand(pRand)
{
	m_pFeatureRel = GRelation::fromTwt(pNode->field("featurerel"));
	m_pLabelRel = GRelation::fromTwt(pNode->field("labelrel"));
	m_nSampleCount = (size_t)pNode->field("sampleCount")->asInt();
	m_equivalentSampleSize = pNode->field("ess")->asDouble();
	GTwtNode* pOutputs = pNode->field("outputs");
	if(pOutputs->itemCount() != m_pLabelRel->size())
		ThrowError("Wrong number of outputs");
	m_pOutputs = new struct GNaiveBayesOutputAttr*[m_pLabelRel->size()];
	for(size_t i = 0; i < m_pLabelRel->size(); i++)
		m_pOutputs[i] = new struct GNaiveBayesOutputAttr(pOutputs->item(i), m_pFeatureRel->size(), m_pLabelRel->valueCount(i));
}

GNaiveBayes::~GNaiveBayes()
{
	clear();
}

// virtual
GTwtNode* GNaiveBayes::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = baseTwtNode(pDoc, "GNaiveBayes");
	pNode->addField(pDoc, "featurerel", m_pFeatureRel->toTwt(pDoc));
	pNode->addField(pDoc, "labelrel", m_pLabelRel->toTwt(pDoc));
	pNode->addField(pDoc, "sampleCount", pDoc->newInt(m_nSampleCount));
	pNode->addField(pDoc, "ess", pDoc->newDouble(m_equivalentSampleSize));
	GTwtNode* pOutputs = pNode->addField(pDoc, "outputs", pDoc->newList(m_pLabelRel->size()));
	for(size_t i = 0; i < m_pLabelRel->size(); i++)
		pOutputs->setItem(i, m_pOutputs[i]->toTwt(pDoc));
	return pNode;
}

// virtual
void GNaiveBayes::clear()
{
	m_nSampleCount = 0;
	if(m_pOutputs)
	{
		for(size_t n = 0; n < m_pLabelRel->size(); n++)
			delete(m_pOutputs[n]);
		delete[] m_pOutputs;
	}
	m_pOutputs = NULL;
}

// virtual
void GNaiveBayes::enableIncrementalLearning(sp_relation& pFeatureRel, sp_relation& pLabelRel)
{
	clear();
	m_pFeatureRel = pFeatureRel;
	m_pLabelRel = pLabelRel;
	m_featureDims = pFeatureRel->size();
	m_labelDims = pLabelRel->size();
	m_pOutputs = new struct GNaiveBayesOutputAttr*[m_pLabelRel->size()];
	for(size_t n = 0; n < m_pLabelRel->size(); n++)
		m_pOutputs[n] = new struct GNaiveBayesOutputAttr(m_pFeatureRel.get(), m_pFeatureRel->size(), m_pLabelRel->valueCount(n));
}

// virtual
void GNaiveBayes::trainIncremental(const double* pIn, const double* pOut)
{
	for(size_t n = 0; n < m_pLabelRel->size(); n++)
		m_pOutputs[n]->AddTrainingSample(pIn, (int)pOut[n]);
	m_nSampleCount++;
}

// virtual
void GNaiveBayes::trainInner(GMatrix& features, GMatrix& labels)
{
	enableIncrementalLearning(features.relation(), labels.relation());
	for(size_t n = 0; n < features.rows(); n++)
		trainIncremental(features[n], labels[n]);
}

// virtual
void GNaiveBayes::trainSparse(GSparseMatrix* pData, size_t labelDims)
{
	sp_relation pFeatureRel = new GUniformRelation(pData->cols() - labelDims, 2);
	sp_relation pLabelRel = new GUniformRelation(labelDims, 2);
	enableIncrementalLearning(pFeatureRel, pLabelRel);
	double* pFullRow = new double[pData->cols()];
	ArrayHolder<double> hFullRow(pFullRow);
	for(size_t n = 0; n < pData->rows(); n++)
	{
		pData->fullRow(pFullRow, n);
		trainIncremental(pFullRow, pFullRow + pFeatureRel->size());
	}
}

void GNaiveBayes::predictDistributionInner(const double* pIn, GPrediction* pOut)
{
	if(m_nSampleCount <= 0)
		ThrowError("You must call train before you call eval");
	for(size_t n = 0; n < m_pLabelRel->size(); n++)
		m_pOutputs[n]->eval(pIn, &pOut[n], m_equivalentSampleSize);
}

void GNaiveBayes::predictInner(const double* pIn, double* pOut)
{
	if(m_nSampleCount <= 0)
		ThrowError("You must call train before you call eval");
	for(size_t n = 0; n < m_pLabelRel->size(); n++)
		pOut[n] = m_pOutputs[n]->predict(pIn, m_equivalentSampleSize, m_pRand);
}

#ifndef NO_TEST_CODE
void GNaiveBayes_CheckResults(double yprior, double ycond, double nprior, double ncond, GPrediction* out)
{
	double py = yprior * ycond;
	double pn = nprior * ncond;
	double sum = py + pn;
	py /= sum;
	pn /= sum;
	GCategoricalDistribution* pCat = out->asCategorical();
	double* pVals = pCat->values(2);
	if(std::abs(pVals[0] - py) > 1e-8)
		ThrowError("wrong");
	if(std::abs(pVals[1] - pn) > 1e-8)
		ThrowError("wrong");
}

void GNaiveBayes_testMath(GRand* pRand)
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
	GMatrix* pTrain = GMatrix::parseArff(trainFile, strlen(trainFile));
	Holder<GMatrix> hTrain(pTrain);
	GMatrix* pFeatures = pTrain->cloneSub(0, 0, pTrain->rows(), 2);
	Holder<GMatrix> hFeatures(pFeatures);
	GMatrix* pLabels = pTrain->cloneSub(0, 2, pTrain->rows(), 1);
	Holder<GMatrix> hLabels(pLabels);
	GNaiveBayes nb(pRand);
	nb.setEquivalentSampleSize(0.0);
	nb.train(*pFeatures, *pLabels);
	GPrediction out;
	double pat[2];
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
	GRand prng(0);
	GNaiveBayes_testMath(&prng);
	GNaiveBayes nb(&prng);
	nb.basicTest(0.77, 0.77, &prng);
}
#endif // !NO_TEST_CODE

// -----------------------------------------------------------------------------

GNaiveMLE::GNaiveMLE(sp_relation& pRelation)
: GSupervisedLearner()
{
	m_pPredictions = NULL;
	m_dLimboValue = 1.0;
	m_dEquivalentSampleSize = .001;
}

GNaiveMLE::GNaiveMLE(GTwtNode* pNode, GRand& rand)
: GSupervisedLearner(pNode, rand)
{
	m_pFeatureRel = GRelation::fromTwt(pNode->field("featurerel"));
	m_pLabelRel = GRelation::fromTwt(pNode->field("labelrel"));
	m_dLimboValue = pNode->field("limbo")->asDouble();
	m_dEquivalentSampleSize = pNode->field("ess")->asDouble();
	GTwtNode* pPredictions = pNode->field("predictions");
	m_nValues = (size_t)pPredictions->itemCount();
	m_pPredictions = new GCategoricalDistribution[m_nValues];
	for(size_t i = 0; i < m_nValues; i++)
		m_pPredictions[i].fromTwt(pPredictions->item(i));
}

// virtual
GNaiveMLE::~GNaiveMLE()
{
	delete[] m_pPredictions;
}

GTwtNode* GNaiveMLE::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = baseTwtNode(pDoc, "GNaiveMLE");
	pNode->addField(pDoc, "featurerel", m_pFeatureRel->toTwt(pDoc));
	pNode->addField(pDoc, "labelrel", m_pLabelRel->toTwt(pDoc));
	pNode->addField(pDoc, "limbo", pDoc->newDouble(m_dLimboValue));
	pNode->addField(pDoc, "ess", pDoc->newDouble(m_dEquivalentSampleSize));
	GTwtNode* pPredictions = pNode->addField(pDoc, "predictions", pDoc->newList(m_nValues));
	for(size_t i = 0; i < m_nValues; i++)
		pPredictions->setItem(i, m_pPredictions[i].toTwt(pDoc));
	return pNode;
}

// virtual
void GNaiveMLE::clear()
{
	delete[] m_pPredictions;
	m_pPredictions = NULL;
}

// virtual
void GNaiveMLE::trainInner(GMatrix& features, GMatrix& labels)
{
	clear();
	m_pFeatureRel = features.relation();
	m_pLabelRel = labels.relation();
	if(m_pLabelRel->size() != 1)
		ThrowError("Sorry, only one-dimensional labels are supported");
	m_nValues = 0;
	for(size_t i = 0; i < m_pFeatureRel->size(); i++)
		m_nValues += m_pFeatureRel->valueCount(i);
	m_pPredictions = new GCategoricalDistribution[m_nValues];

	// Clear the predictions
	size_t nOutputValues = m_pLabelRel->valueCount(0);
	int valIn, valOut;;
	GCategoricalDistribution* pCat;
	double* pValues;
	int nPrediction = 0;
	for(size_t j = 0; j < m_pFeatureRel->size(); j++)
	{
		size_t valCount = m_pFeatureRel->valueCount(j);
		for(size_t k = 0; k < valCount; k++)
		{
			pCat = &m_pPredictions[nPrediction++];
			pValues = pCat->values(nOutputValues);
			for(size_t i = 0; i < nOutputValues; i++)
				pValues[i] = m_dEquivalentSampleSize;
		}
	}

	// Accumulate the predictions
	for(size_t i = 0; i < features.rows(); i++)
	{
		nPrediction = 0;
		for(size_t j = 0; j < m_pFeatureRel->size(); j++)
		{
			int valCount = (int)m_pFeatureRel->valueCount(j);
			valIn = (int)features[i][j];
			valOut = (int)labels[i][0];
			if(valIn >= 0 && valIn < (int)valCount && valOut >= 0 && valOut < (int)nOutputValues)
			{
				pCat = &m_pPredictions[nPrediction + valIn];
				pValues = pCat->values(nOutputValues);
				pValues[valOut]++;
			}
			nPrediction += valCount;
		}
	}

	// Normalize the predictions
	nPrediction = 0;
	for(size_t j = 0; j < m_pFeatureRel->size(); j++)
	{
		size_t valCount = m_pFeatureRel->valueCount(j);
		for(size_t k = 0; k < valCount; k++)
		{
			pCat = &m_pPredictions[nPrediction++];
			pCat->normalize();
		}
	}
}

// virtual
void GNaiveMLE::predictDistributionInner(const double* pIn, GPrediction* pOut)
{
	size_t nOutputValues = m_pLabelRel->valueCount(0);
	GCategoricalDistribution* pCatOut = pOut->makeCategorical();
	double* pValuesOut = pCatOut->values(nOutputValues);
	GVec::setAll(pValuesOut, 0.0, nOutputValues);
	GCategoricalDistribution* pCatIn;
	int nPrediction = 0;
	double x;
	for(size_t i = 0; i < m_pFeatureRel->size(); i++)
	{
		int valCount = (int)m_pFeatureRel->valueCount(i);
		int val = (int)pIn[i];
		if(val >= 0 && val < valCount)
		{
			pCatIn = &m_pPredictions[nPrediction + val];
			for(size_t j = 0; j < nOutputValues; j++)
			{
				x = pCatIn->likelihood((int)j);
				pValuesOut[j] += (1.0 - m_dLimboValue) * x + m_dLimboValue * log(x);
			}
		}
		nPrediction += valCount;
	}
	GAssert((size_t)nPrediction == m_nValues);
	pCatOut->normalizeFromLogSpace();
}

// virtual
void GNaiveMLE::predictInner(const double* pIn, double* pOut)
{
#ifdef WINDOWS
	GPrediction* out = new GPrediction[m_pLabelRel->size()];
	ArrayHolder<GPrediction> hOut(out);
#else
	GPrediction out[m_pLabelRel->size()];
#endif
	predictDistribution(pIn, out);
	GPrediction::predictionArrayToVector(m_pLabelRel->size(), out, pOut);
}

} // namespace GClasses
