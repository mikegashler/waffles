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

#include "GKNN.h"
#include <math.h>
#include "GError.h"
#include <stdlib.h>
#include "GDom.h"
#include "GDistribution.h"
#include "GHeap.h"
#include "GNeighborFinder.h"
#include "GVec.h"
#include "GHillClimber.h"
#include "GCluster.h"
#include "GBitTable.h"
#include "GDistance.h"
#include "GSparseMatrix.h"
#include "GHolders.h"
#include "GBitTable.h"
#include <map>
#include <queue>
#include <memory>

using std::multimap;
using std::map;
using std::pair;
using std::priority_queue;
using std::vector;


namespace GClasses {

class GKnnScaleFactorCritic : public GTargetFunction
{
protected:
	size_t m_labelDims;
	GKNN* m_pLearner;
	double* m_pAccuracy;

public:
	GKnnScaleFactorCritic(GKNN* pLearner, size_t featureDims, size_t labelDims)
	: GTargetFunction(featureDims), m_labelDims(labelDims)
	{
		m_pLearner = pLearner;
		m_pAccuracy = new double[labelDims];
	}

	virtual ~GKnnScaleFactorCritic()
	{
		delete[] m_pAccuracy;
	}

	virtual void initVector(double* pVector)
	{
		GDistanceMetric* pMetric = m_pLearner->metric();
		GVec::copy(pVector, pMetric->scaleFactors().data(), pMetric->scaleFactors().size());
	}

	virtual bool isStable() { return false; }
	virtual bool isConstrained() { return false; }

protected:
	virtual double computeError(const double* pVector)
	{
		// todo: this method is WAAAY too inefficient
		GMatrix* pFeatures = m_pLearner->features();
		GMatrix* pLabels = m_pLearner->labels();
		GKNN temp;
		temp.setNeighborCount(m_pLearner->neighborCount());
		temp.beginIncrementalLearning(pFeatures->relation(), pLabels->relation());
		temp.metric()->scaleFactors().set(pVector, relation()->size());
		return temp.crossValidate(*pFeatures, *pLabels, 2);
	}
};


GKNN::GKNN()
: GIncrementalLearner()
{
	m_eInterpolationMethod = Linear;
	m_eTrainMethod = StoreAll;
	m_trainParam = 0.0;
	m_pLearner = NULL;
	m_bOwnLearner = false;
	m_nNeighbors = 1;
	m_pFeatures = NULL;
	m_pSparseFeatures = NULL;
	m_pLabels = NULL;
	m_pNeighborFinder = NULL;
	m_pEvalNeighbors = new size_t[m_nNeighbors + 1];
	m_pEvalDistances = new double[m_nNeighbors + 1];
	m_normalizeScaleFactors = true;
	m_optimizeScaleFactors = false;
	m_pDistanceMetric = NULL;
	m_pSparseMetric = NULL;
	m_ownMetric = false;
	m_pValueCounts = NULL;
	m_pCritic = NULL;
	m_pScaleFactorOptimizer = NULL;
}

GKNN::GKNN(const GDomNode* pNode)
: GIncrementalLearner(pNode)
{
	m_pNeighborFinder = NULL;
	m_pCritic = NULL;
	m_pScaleFactorOptimizer = NULL;
	m_pLearner = NULL;
	m_pValueCounts = NULL;
	m_bOwnLearner = false;
	m_nNeighbors = (size_t)pNode->field("neighbors")->asInt();
	m_eInterpolationMethod = (InterpolationMethod)pNode->field("interpMethod")->asInt();
	m_eTrainMethod = (TrainMethod)pNode->field("trainMethod")->asInt();
	m_trainParam = pNode->field("trainParam")->asDouble();
	m_normalizeScaleFactors = pNode->field("normalize")->asBool();
	m_optimizeScaleFactors = pNode->field("optimize")->asBool();
	GMatrix* pFeatures = NULL;
	GSparseMatrix* pSparseFeatures = NULL;
	GDomNode* pFeaturesNode = pNode->fieldIfExists("features");
	if(pFeaturesNode)
		pFeatures = new GMatrix(pFeaturesNode);
	else
		pSparseFeatures = new GSparseMatrix(pNode->field("sparseFeatures"));
	GMatrix* pLabels = new GMatrix(pNode->field("labels"));
	GDomNode* pMetricNode = pNode->fieldIfExists("metric");
	m_pDistanceMetric = NULL;
	m_pSparseMetric = NULL;
	if(pMetricNode)
		m_pDistanceMetric = GDistanceMetric::deserialize(pNode->field("metric"));
	else
		m_pSparseMetric = GSparseSimilarity::deserialize(pNode->field("sparseMetric"));
	m_ownMetric = true;
	m_pFeatures = NULL;
	m_pSparseFeatures = NULL;
	m_pLabels = NULL;
	m_pEvalNeighbors = new size_t[m_nNeighbors + 1];
	m_pEvalDistances = new double[m_nNeighbors + 1];
	if(pFeatures)
		beginIncrementalLearningInner(pFeatures->relation(), pLabels->relation());
	else
	{
		GUniformRelation rel(pSparseFeatures->cols(), 0);
		beginIncrementalLearningInner(rel, pLabels->relation());
	}
	delete(m_pFeatures);
	delete(m_pSparseFeatures);
	delete(m_pLabels);
	m_pFeatures = pFeatures;
	m_pSparseFeatures = pSparseFeatures;
	m_pLabels = pLabels;
}

GKNN::~GKNN()
{
	delete(m_pNeighborFinder);
	delete(m_pFeatures);
	delete(m_pSparseFeatures);
	delete(m_pLabels);
	delete[] m_pEvalNeighbors;
	delete[] m_pEvalDistances;
	delete[] m_pValueCounts;
	delete(m_pScaleFactorOptimizer);
	delete(m_pCritic);
	if(m_ownMetric)
	{
		delete(m_pDistanceMetric);
		delete(m_pSparseMetric);
	}
}

// virtual
GDomNode* GKNN::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GKNN");
	pNode->addField(pDoc, "neighbors", pDoc->newInt(m_nNeighbors));
	if(m_eInterpolationMethod == Learner)
		throw Ex("Sorry, serialize is not supported for the \"Learner\" interpolation method");
	pNode->addField(pDoc, "interpMethod", pDoc->newInt(m_eInterpolationMethod));
	pNode->addField(pDoc, "trainMethod", pDoc->newInt(m_eTrainMethod));
	pNode->addField(pDoc, "trainParam", pDoc->newDouble(m_trainParam));
	pNode->addField(pDoc, "normalize", pDoc->newBool(m_normalizeScaleFactors));
	pNode->addField(pDoc, "optimize", pDoc->newBool(m_optimizeScaleFactors));
	if(m_pFeatures)
		pNode->addField(pDoc, "features", m_pFeatures->serialize(pDoc));
	else
		pNode->addField(pDoc, "sparseFeatures", m_pSparseFeatures->serialize(pDoc));
	pNode->addField(pDoc, "labels", m_pLabels->serialize(pDoc));
	if(m_pDistanceMetric)
		pNode->addField(pDoc, "metric", m_pDistanceMetric->serialize(pDoc));
	else
		pNode->addField(pDoc, "sparseMetric", m_pSparseMetric->serialize(pDoc));
	return pNode;
}

void GKNN::autoTune(GMatrix& feats, GMatrix& labs)
{
	// Find the best value for k
	size_t cap = size_t(floor(sqrt(double(feats.rows()))));
	size_t bestK = 1;
	double bestErr = 1e308;
	for(size_t i = 1; i < cap; i *= 3)
	{
		setNeighborCount(i);
		double d = crossValidate(feats, labs, 2);
		if(d < bestErr)
		{
			bestErr = d;
			bestK = i;
		}
		else if(i >= 27)
			break;
	}

	// Set the best values
	m_nNeighbors = bestK;

	// Try without normalization
	m_normalizeScaleFactors = false;
	double d = crossValidate(feats, labs, 2);
	if(d >= bestErr)
		m_normalizeScaleFactors = true;
}

void GKNN::setNeighborCount(size_t k)
{
	delete[] m_pEvalNeighbors;
	delete[] m_pEvalDistances;
	m_nNeighbors = k;
	m_pEvalNeighbors = new size_t[m_nNeighbors + 1];
	m_pEvalDistances = new double[m_nNeighbors + 1];
}

void GKNN::setInterpolationMethod(InterpolationMethod eMethod)
{
	if(eMethod == Learner)
		throw Ex("You should call SetInterpolationLearner instead");
	m_eInterpolationMethod = eMethod;
}

void GKNN::setInterpolationLearner(GSupervisedLearner* pLearner, bool bOwnLearner)
{
	if(m_bOwnLearner)
		delete(m_pLearner);
	m_pLearner = pLearner;
	m_eInterpolationMethod = Learner;
	m_bOwnLearner = bOwnLearner;
}

size_t GKNN::addVector(const GVec& feat, const GVec& lab)
{
	// Store the features
	size_t index;
	if(m_pNeighborFinder)
	{
		delete(m_pNeighborFinder);
		m_pNeighborFinder = NULL;
	}
	index = m_pFeatures->rows();
	m_pFeatures->newRow().copy(feat);

	// Store the labels
	m_pLabels->newRow().copy(lab);
	return index;
}

void GKNN::setNormalizeScaleFactors(bool b)
{
	m_normalizeScaleFactors = b;
}

void GKNN::setOptimizeScaleFactors(bool b)
{
	m_optimizeScaleFactors = b;
}

void GKNN::setMetric(GDistanceMetric* pMetric, bool own)
{
	if(m_ownMetric)
	{
		delete(m_pDistanceMetric);
		delete(m_pSparseMetric);
	}
	m_pDistanceMetric = pMetric;
	m_pSparseMetric = NULL;
	m_ownMetric = own;
}

void GKNN::setMetric(GSparseSimilarity* pMetric, bool own)
{
	if(m_ownMetric)
	{
		delete(m_pDistanceMetric);
		delete(m_pSparseMetric);
	}
	m_pDistanceMetric = NULL;
	m_pSparseMetric = pMetric;
	m_ownMetric = own;
}

// virtual
void GKNN::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	clear();
	if(!m_pDistanceMetric && !m_pSparseMetric)
		setMetric(new GRowDistance(), true);
	if(m_pDistanceMetric)
	{
		m_pFeatures = new GMatrix(featureRel.cloneMinimal());
		m_pDistanceMetric->init(&m_pFeatures->relation(), false);

		// Scale factor optimization
		if(m_optimizeScaleFactors)
		{
			m_pCritic = new GKnnScaleFactorCritic(this, featureRel.size(), labelRel.size());
			m_pScaleFactorOptimizer = new GMomentumGreedySearch(m_pCritic);
		}
	}
	else if(m_pSparseMetric)
	{
		if(!featureRel.areContinuous(0, featureRel.size()))
			throw Ex("Sorry, nominal features cannot be used in conjunction with sparse metrics");
		m_pSparseFeatures = new GSparseMatrix(0, featureRel.size(), UNKNOWN_REAL_VALUE);
	}
	else
		throw Ex("Some sort of distance or similarity metric is required");

	m_pLabels = new GMatrix(labelRel.cloneMinimal());

	// Allocate a buffer for counting values
	size_t maxOutputValueCount = 0;
	for(size_t n = 0; n < labelRel.size(); n++)
		maxOutputValueCount = std::max(maxOutputValueCount, labelRel.valueCount(n));
	m_pValueCounts = new double[maxOutputValueCount];
}

// virtual
void GKNN::trainIncremental(const GVec& in, const GVec& out)
{
	// Make a copy of the vector
	GAssert(m_pDistanceMetric);
	addVector(in, out);

	// Learn how to scale the attributes
	if(m_pScaleFactorOptimizer && m_pFeatures->rows() > 50)
	{
		m_pScaleFactorOptimizer->iterate();
		m_pDistanceMetric->scaleFactors().set(m_pScaleFactorOptimizer->currentVector(), m_pFeatures->cols());
	}
}

void GKNN::trainInner(const GMatrix& feats, const GMatrix& labs)
{
	if(m_pSparseMetric)
		throw Ex("This method is not compatible with sparse similarity metrics. You should either use trainSparse instead, or use a dense dissimilarity metric.");
	beginIncrementalLearningInner(feats.relation(), labs.relation());

	// Give each attribute an equal chance by scaling out the deviation
	GVec& scaleFactors = m_pDistanceMetric->scaleFactors();
	if(m_normalizeScaleFactors)
	{
		for(size_t i = 0; i < feats.cols(); i++)
		{
			if(feats.relation().valueCount(i) == 0)
			{
				double m = feats.columnMean(i);
				double d = sqrt(feats.columnVariance(i, m));
				if(d >= 1e-8)
					scaleFactors[i] = 1.0 / (2.0 * d);
				else
					scaleFactors[i] = 1.0;
			}
			else
				scaleFactors[i] = 1.0;
		}
	}
	else
		scaleFactors.fill(1.0);

	if(m_eTrainMethod == StoreAll)
	{
		m_pFeatures->reserve(feats.rows());
		m_pLabels->reserve(feats.rows());
		for(size_t i = 0; i < feats.rows(); i++)
			addVector(feats[i], labs[i]);
	}
	else if(m_eTrainMethod == ValidationPrune)
	{
		throw Ex("Sorry, this training method is not implemented yet");
	}
	else if(m_eTrainMethod == DrawRandom)
	{
		size_t n = (size_t)m_trainParam;
		m_pFeatures->reserve(n);
		m_pLabels->reserve(n);
		for(size_t i = 0; i < n; i++)
		{
			size_t index = (size_t)m_rand.next(feats.rows());
			addVector(feats[index], labs[index]);
		}
	}

	// Learn to scale the attributes
	if(m_pScaleFactorOptimizer)
	{
		if(!m_pNeighborFinder)
		{
			m_pNeighborFinder = new GKdTree(m_pFeatures, m_nNeighbors, m_pDistanceMetric, false);
		}
		for(size_t j = 0; j < 50; j++)
		{
			m_pScaleFactorOptimizer->iterate();
			m_pNeighborFinder->reoptimize();
		}
		scaleFactors.set(m_pScaleFactorOptimizer->currentVector(), feats.cols());
	}
}

// virtual
void GKNN::trainSparse(GSparseMatrix& feats, GMatrix& labs)
{
	if(feats.rows() != labs.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	if(m_pDistanceMetric)
		throw Ex("This method is not compatible with dense dissimilarity metrics. You should either use the train method instead, or use a sparse similarity metric.");
	if(!m_pSparseMetric)
		setMetric(new GCosineSimilarity(), true);
	GUniformRelation featureRel(feats.cols(), 0);
	beginIncrementalLearning(featureRel, labs.relation());

	// Copy the training data
	m_pSparseFeatures->newRows(feats.rows());
	m_pSparseFeatures->copyFrom(&feats);
	m_pLabels->copy(&labs);
}

void GKNN::findNeighbors(const GVec& vec)
{
	if(m_pDistanceMetric)
	{
		if(!m_pNeighborFinder)
		{
			//m_pNeighborFinder = new GBruteForceNeighborFinder(m_pFeatures, m_nNeighbors, m_pDistanceMetric, false);
			m_pNeighborFinder = new GKdTree(m_pFeatures, m_nNeighbors, m_pDistanceMetric, false);
		}
		GAssert(m_pNeighborFinder->neighborCount() == m_nNeighbors);
		m_pNeighborFinder->neighbors(m_pEvalNeighbors, m_pEvalDistances, vec);
	}
	else
	{
		if(!m_pSparseMetric)
			throw Ex("train, trainSparse, or beginIncrementalLearning must be called before this method");
		multimap<double,size_t> priority_queue;
		for(size_t i = 0; i < m_pSparseFeatures->rows(); i++)
		{
			map<size_t,double>& row = m_pSparseFeatures->row(i);
			double similarity = m_pSparseMetric->similarity(row, vec);
			priority_queue.insert(pair<double,size_t>(similarity, i));
			if(priority_queue.size() > m_nNeighbors)
				priority_queue.erase(priority_queue.begin());
		}
		size_t pos = 0;
		size_t* pNeigh = m_pEvalNeighbors;
		double* pDist = m_pEvalDistances;
		for(multimap<double,size_t>::iterator it = priority_queue.begin(); it != priority_queue.end(); it++)
		{
			*pNeigh = it->second;
			*pDist = 1.0;
			pos++;
			pNeigh++;
			pDist++;
		}
		while(pos < m_nNeighbors)
		{
			*pNeigh = INVALID_INDEX;
			*pDist = UNKNOWN_REAL_VALUE;
			pos++;
			pNeigh++;
			pDist++;
		}
	}
}

void GKNN::interpolateMean(const GVec& in, GPrediction* out, GVec* pOut2)
{
	for(size_t i = 0; i < m_pLabels->cols(); i++)
	{
		if(m_pLabels->relation().valueCount(i) == 0)
		{
			// Continuous label
			double dSum = 0;
			double dSumOfSquares = 0;
			size_t count = 0;
			for(size_t j = 0; j < m_nNeighbors; j++)
			{
				size_t k = m_pEvalNeighbors[j];
				if(k < m_pLabels->rows())
				{
					GVec& neighbor = m_pLabels->row(k);
					dSum += neighbor[i];
					dSumOfSquares += (neighbor[i] * neighbor[i]);
					count++;
				}
			}
			if(out)
			{
				if(count > 0)
				{
					double mean = dSum / count;
					out[i].makeNormal()->setMeanAndVariance(mean, dSumOfSquares / count - (mean * mean));
				}
				else
					out[i].makeNormal()->setMeanAndVariance(0, 1);
			}
			if(pOut2)
			{
				if(count > 0)
					(*pOut2)[i] = dSum / count;
				else
					(*pOut2)[i] = 0;
			}
		}
		else
		{
			// Nominal label
			size_t nValueCount = m_pLabels->relation().valueCount(i);
			GVec::setAll(m_pValueCounts, 0.0, nValueCount);
			for(size_t j = 0; j < m_nNeighbors; j++)
			{
				size_t k = m_pEvalNeighbors[j];
				if(k < m_pLabels->rows())
				{
					GVec& neighbor = m_pLabels->row(k);
					int val = (int)neighbor[i];
					if(val < 0 || val >= (int)nValueCount)
						throw Ex("GKNN doesn't support unknown label values");
					m_pValueCounts[val]++;
				}
			}
			if(out)
				out[i].makeCategorical()->setValues(nValueCount, m_pValueCounts);
			if(pOut2)
				(*pOut2)[i] = (double)GVec::indexOfMax(m_pValueCounts, nValueCount, &m_rand);
		}
	}
}

void GKNN::interpolateLinear(const GVec& in, GPrediction* out, GVec* pOut2)
{
	for(size_t i = 0; i < m_pLabels->cols(); i++)
	{
		if(m_pLabels->relation().valueCount(i) == 0)
		{
			// Continuous label
			double dSum = 0;
			double dSumOfSquares = 0;
			double dTot = 0;
			for(size_t j = 0; j < m_nNeighbors; j++)
			{
				size_t k = m_pEvalNeighbors[j];
				if(k < m_pLabels->rows())
				{
					GVec& neighbor = m_pLabels->row(k);
					if(neighbor[i] == UNKNOWN_REAL_VALUE)
						throw Ex("GKNN doesn't support unknown label values");
					double d = 1.0 / std::max(sqrt(m_pEvalDistances[j]), 1e-9); // the weight
					dTot += d;
					d *= neighbor[i]; // weighted sum
					dSum += d;
					d *= neighbor[i]; // weighted sum of squares
					dSumOfSquares += d;
				}
			}
			if(out)
			{
				if(dTot > 0)
				{
					double d = dSum / dTot;
					out[i].makeNormal()->setMeanAndVariance(d, dSumOfSquares / dTot - (d * d));
				}
				else
					out[i].makeNormal()->setMeanAndVariance(0, 1);
			}
			if(pOut2)
			{
				if(dTot > 0)
					(*pOut2)[i] = dSum / dTot;
				else
					(*pOut2)[i] = 0;
			}
		}
		else
		{
			// Nominal label
			int nValueCount = (int)m_pLabels->relation().valueCount(i);
			GVec::setAll(m_pValueCounts, 0.0, nValueCount);
			double dSumWeight = 0;
			for(size_t j = 0; j < m_nNeighbors; j++)
			{
				size_t k = m_pEvalNeighbors[j];
				if(k < m_pLabels->rows())
				{
					GVec& neighbor = m_pLabels->row(k);
					double d = 1.0 / std::max(m_pEvalDistances[j], 1e-9); // to be truly "linear", we should use sqrt(d) instead of d, but this is faster to compute and arguably better for nominal values anyway
					int val = (int)neighbor[i];
					if(val < 0 || val >= nValueCount)
						throw Ex("GKNN doesn't support unknown label values");
					m_pValueCounts[val] += d;
					dSumWeight += d;
				}
			}
			if(out)
				out[i].makeCategorical()->setValues(nValueCount, m_pValueCounts);
			if(pOut2)
				(*pOut2)[i] = (double)GVec::indexOfMax(m_pValueCounts, nValueCount, &m_rand);
		}
	}
}

void GKNN::interpolateLearner(const GVec& in, GPrediction* out, GVec* pOut2)
{
	GAssert(m_pLearner); // no learner is set
	GMatrix dataFeatures(m_pFeatures->relation().cloneMinimal());
	GReleaseDataHolder hDataFeatures(&dataFeatures);
	dataFeatures.reserve(m_nNeighbors);
	GMatrix dataLabels(m_pLabels->relation().cloneMinimal());
	GReleaseDataHolder hDataLabels(&dataLabels);
	dataLabels.reserve(m_nNeighbors);
	for(size_t i = 0; i < m_nNeighbors; i++)
	{
		size_t nNeighbor = m_pEvalNeighbors[i];
		if(nNeighbor < m_pFeatures->rows())
		{
			dataFeatures.takeRow(&m_pFeatures->row(nNeighbor));
			dataLabels.takeRow(&m_pLabels->row(nNeighbor));
		}
	}
	m_pLearner->train(dataFeatures, dataLabels);
	if(out)
		m_pLearner->predictDistribution(in, out);
	if(pOut2)
		m_pLearner->predict(in, *pOut2);
}

// virtual
void GKNN::predictDistribution(const GVec& in, GPrediction* out)
{
	findNeighbors(in);
	switch(m_eInterpolationMethod)
	{
		case Linear: interpolateLinear(in, out, NULL); break;
		case Mean: interpolateMean(in, out, NULL); break;
		case Learner: interpolateLearner(in, out, NULL); break;
		default:
			GAssert(false); // unexpected enumeration
			break;
	}
}

// virtual
void GKNN::predict(const GVec& in, GVec& out)
{
	findNeighbors(in);
	switch(m_eInterpolationMethod)
	{
		case Linear: interpolateLinear(in, NULL, &out); break;
		case Mean: interpolateMean(in, NULL, &out); break;
		case Learner: interpolateLearner(in, NULL, &out); break;
		default:
			GAssert(false); // unexpected enumeration
			break;
	}
}

// virtual
void GKNN::clear()
{
	delete(m_pNeighborFinder); m_pNeighborFinder = NULL;
	delete(m_pFeatures); m_pFeatures = NULL;
	delete(m_pSparseFeatures); m_pSparseFeatures = NULL;
	delete(m_pLabels); m_pLabels = NULL;
	delete(m_pScaleFactorOptimizer); m_pScaleFactorOptimizer = NULL;
	delete(m_pCritic); m_pCritic = NULL;
	delete[] m_pValueCounts; m_pValueCounts = NULL;
}

#ifndef NO_TEST_CODE
//static
void GKNN::test()
{
	GKNN knn;
	knn.setNeighborCount(3);
	knn.basicTest(0.72, 0.924, 0.1);
}
#endif

// ---------------------------------------------------------------------------------------

GNeighborTransducer::GNeighborTransducer()
: GTransducer(), m_friendCount(12)
{
}

void GNeighborTransducer::autoTune(GMatrix& features, GMatrix& labels)
{
	// Find the best value for k
	size_t cap = size_t(floor(sqrt(double(features.rows()))));
	size_t bestK = 1;
	double bestErr = 1e308;
	for(size_t i = 1; i < cap; i *= 3)
	{
		m_friendCount = i;
		double d = crossValidate(features, labels, 2);
		if(d < bestErr)
		{
			bestErr = d;
			bestK = i;
		}
		else if(i >= 27)
			break;
	}

	// Set the best values
	m_friendCount = bestK;
}

// virtual
std::unique_ptr<GMatrix> GNeighborTransducer::transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2)
{
	// Make a dataset containing all rows
	GMatrix featuresAll(features1.relation().cloneMinimal());
	featuresAll.reserve(features1.rows() + features2.rows());
	GReleaseDataHolder hFeaturesAll(&featuresAll);
	for(size_t i = 0; i < features2.rows(); i++)
		featuresAll.takeRow((GVec*)&features2[i]);
	for(size_t i = 0; i < features1.rows(); i++)
		featuresAll.takeRow((GVec*)&features1[i]);
	auto out = std::unique_ptr<GMatrix>(new GMatrix(labels1.relation().clone()));
	out->newRows(features2.rows());

	// Find friends
	GNeighborFinder* pNF = new GNeighborGraph(new GKdTree(&featuresAll, m_friendCount, NULL, true), true);
	std::unique_ptr<GNeighborFinder> hNF(pNF);
	GTEMPBUF(size_t, neighs, m_friendCount);

	// Transduce
	for(size_t lab = 0; lab < labels1.cols(); lab++)
	{
		size_t labelValues = labels1.relation().valueCount(lab);
		double* tallys = new double[labelValues];
		std::unique_ptr<double[]> hTallys(tallys);

		// Label the unlabeled patterns
		GBitTable labeled(features2.rows());
		GMatrix labelList(features2.rows(), 3); // pattern index, most likely label, confidence
		for(size_t i = 0; i < features2.rows(); i++)
			labelList[i][0] = (double)i;
		while(labelList.rows() > 0)
		{
			// Compute the most likely label and the confidence for each pattern
			for(size_t i = 0; i < labelList.rows(); i++)
			{
				// Find the most common label
				GVec& row = labelList.row(i);
				size_t index = (size_t)row[0];
				pNF->neighbors(neighs, index);
				GVec::setAll(tallys, 0.0, labelValues);
				for(size_t j = 0; j < m_friendCount; j++)
				{
					if(neighs[j] >= featuresAll.rows())
						continue;
					if(neighs[j] >= features2.rows())
					{
						int label = (int)labels1[neighs[j] - features2.rows()][lab];
						if(label >= 0 && label < (int)labelValues)
							tallys[label]++;
					}
					else if(labeled.bit(neighs[j]))
					{
						int label = (int)out->row(neighs[j])[lab];
						if(label >= 0 && label < (int)labelValues)
							tallys[label] += 0.6;
					}
				}
				int label = (int)GVec::indexOfMax(tallys, labelValues, &m_rand);
				double conf = tallys[label];

				// Penalize for dissenting votes
				for(size_t j = 0; j < m_friendCount; j++)
				{
					if(neighs[j] >= featuresAll.rows())
						continue;
					if(neighs[j] >= features2.rows())
					{
						int l2 = (int)labels1[neighs[j] - features2.rows()][lab];
						if(l2 != label)
							conf *= 0.5;
					}
					else if(labeled.bit(neighs[j]))
					{
						int l2 = (int)out->row(neighs[j])[lab];
						if(l2 != label)
							conf *= 0.8;
					}
				}
				row[1] = label;
				row[2] = conf;
			}
			labelList.sort(2);

			// Assign the labels to the patterns we are most confident about
			size_t maxCount = std::max((size_t)4, features1.rows() / 8);
			size_t count = 0;
			for(size_t i = labelList.rows() - 1; i < labelList.rows(); i--)
			{
				GVec& row = labelList.row(i);
				size_t index = (size_t)row[0];
				int label = (int)row[1];
				out->row(index)[lab] = label;
				labeled.set(index);
				labelList.deleteRow(i);
				if(count >= maxCount)
					break;
				count++;
			}
		}
	}
	return out;
}

GInstanceTable::GInstanceTable(size_t dims, size_t* pDims)
: GIncrementalLearner(), m_dims(dims)
{
	m_pDims = new size_t[dims];
	memcpy(m_pDims, pDims, sizeof(size_t) * dims);
	m_product = 1;
	m_pScales = new size_t[dims];
	for(size_t i = 0; i < dims; i++)
	{
		m_pScales[i] = m_product;
		m_product *= pDims[i];
		m_pDims[i] = pDims[i];
	}
	m_pTable = NULL;
	clear();
}

// virtual
GInstanceTable::~GInstanceTable()
{
	delete[] m_pDims;
	delete[] m_pScales;
	clear();
}

// virtual
GDomNode* GInstanceTable::serialize(GDom* pDoc) const
{
	throw Ex("not implemented yet");
	return NULL;
}

// virtual
void GInstanceTable::trainSparse(GSparseMatrix& features, GMatrix& labels)
{
	throw Ex("Sorry, trainSparse is not implemented yet in GInstanceTable");
}

// virtual
void GInstanceTable::trainInner(const GMatrix& features, const GMatrix& labels)
{
	beginIncrementalLearningInner(features.relation(), labels.relation());
	for(size_t i = 0; i < features.rows(); i++)
		trainIncremental(features[i], labels[i]);
}

// virtual
void GInstanceTable::predictDistribution(const GVec& in, GPrediction* out)
{
	throw Ex("Sorry, this model cannot predict a distribution");
}

// virtual
void GInstanceTable::predict(const GVec& in, GVec& out)
{
	size_t pos = 0;
	for(size_t i = 0; i < m_dims; i++)
	{
		size_t n = (size_t)floor(in[i] + 0.5);
		if(n >= m_pDims[i])
			throw Ex("dim=", to_str(i), ", index=", to_str(in[i]), ", out of range. Expected >= 0 and < ", to_str(m_pDims[i]));
		pos += n * m_pScales[i];
	}
	size_t labelDims = m_pRelLabels->size();
	out.set(m_pTable + pos * labelDims, labelDims);
}

// virtual
void GInstanceTable::clear()
{
	delete[] m_pTable;
	m_pTable = NULL;
}

// virtual
void GInstanceTable::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	// Allocate the table
	clear();
	size_t total = m_product * labelRel.size();
	m_pTable = new double[total];

	// Initialize with small random values
	double* p = m_pTable;
	for(size_t i = 0; i < total; i++)
		*(p++) = m_rand.uniform() * 0.1;

	m_dims = featureRel.size();
}

// virtual
void GInstanceTable::trainIncremental(const GVec& in, const GVec& out)
{
	size_t pos = 0;
	for(size_t i = 0; i < m_dims; i++)
	{
		size_t n = (size_t)floor(in[i] + 0.5);
		if(n >= m_pDims[i])
			throw Ex("dim=", to_str(i), ", index=", to_str(in[i]), ", out of range. Expected >= 0 and < ", to_str(m_pDims[i]));
		pos += n * m_pScales[i];
	}
	size_t labelDims = m_pRelLabels->size();
	GVec::copy(m_pTable + pos * labelDims, out.data(), labelDims);
}











GSparseInstance::GSparseInstance()
: GSupervisedLearner(), m_neighborCount(1), m_pInstanceFeatures(NULL), m_pInstanceLabels(NULL), m_pMetric(NULL), m_pSkipRows(NULL)
{
}

GSparseInstance::GSparseInstance(const GDomNode* pNode)
: GSupervisedLearner(pNode)
{
	m_neighborCount = (size_t)pNode->field("neighbors")->asInt();
	m_pInstanceFeatures = new GSparseMatrix(pNode->field("if"));
	m_pInstanceLabels = new GMatrix(pNode->field("il"));
	m_pMetric = GSparseSimilarity::deserialize(pNode->field("metric"));
}

GSparseInstance::~GSparseInstance()
{
	clear();
}

// virtual
GDomNode* GSparseInstance::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GSparseInstance");
	pNode->addField(pDoc, "neighbors", pDoc->newInt(m_neighborCount));
	pNode->addField(pDoc, "if", m_pInstanceFeatures->serialize(pDoc));
	pNode->addField(pDoc, "il", m_pInstanceLabels->serialize(pDoc));
	pNode->addField(pDoc, "metric", m_pMetric->serialize(pDoc));
	return pNode;
}

void GSparseInstance::setNeighborCount(size_t k)
{
	m_neighborCount = k;
}

void GSparseInstance::trainInner(const GMatrix& features, const GMatrix& labels)
{
	// Split the training data into two parts
	if(features.rows() != labels.rows())
		throw Ex("Expected the same number of rows in the features and labels");
	clear();
	GMatrix f1(features.relation().clone());
	GMatrix f2(features.relation().clone());
	GMatrix l1(labels.relation().clone());
	GMatrix l2(labels.relation().clone());
	GReleaseDataHolder hf1(&f1);
	GReleaseDataHolder hf2(&f2);
	GReleaseDataHolder hl1(&l1);
	GReleaseDataHolder hl2(&l2);
	for(size_t i = 0; i < features.rows(); i++)
	{
		if(i & 1)
		{
			f2.takeRow((GVec*)&features[i]);
			l2.takeRow((GVec*)&labels[i]);
		}
		else
		{
			f1.takeRow((GVec*)&features[i]);
			l1.takeRow((GVec*)&labels[i]);
		}
	}

	// Copy the training portion of the data to be the stored instances
	m_pInstanceFeatures = new GSparseMatrix(f1.rows(), f1.cols(), UNKNOWN_REAL_VALUE);
	m_pInstanceFeatures->copyFrom(&f1);
	m_pInstanceLabels = new GMatrix();
	m_pInstanceLabels->copy(&l1);

	// Prune the instances
	prune(f2, l2);
}

void GSparseInstance::prune(const GMatrix& holdOutFeatures, const GMatrix& holdOutLabels)
{
	// Make a list of all known elements
	vector< std::pair<size_t,size_t> > elements;
	for(size_t i = 0; i < m_pInstanceFeatures->rows(); i++)
	{
		SparseVec::const_iterator endit = m_pInstanceFeatures->rowEnd(i);
		for(SparseVec::const_iterator it = m_pInstanceFeatures->rowBegin(i); it != endit; it++)
			elements.push_back( std::pair<size_t,size_t>(i, it->first) );
	}

	// Prune the stored instances
	m_pSkipRows = new GBitTable(m_pInstanceFeatures->rows());
	double bestErr = sumSquaredError(holdOutFeatures, holdOutLabels);
	while(elements.size() > 0)
	{
		bool improved = false;
		std::random_shuffle(elements.begin(), elements.end());
		vector< std::pair<size_t,size_t> >::iterator it;
		for(it = elements.begin(); it != elements.end(); it++)
		{
			if(m_pSkipRows->bit(it->first))
			{
				// drop the element from the list
				std::swap(*it, *(elements.end() - 1));
				it--;
				elements.pop_back();
			}
			else
			{
				// Try dropping an element
				double oldVal = m_pInstanceFeatures->get(it->first, it->second);
				m_pInstanceFeatures->set(it->first, it->second, UNKNOWN_REAL_VALUE);
				double candErr = sumSquaredError(holdOutFeatures, holdOutLabels);
				if(candErr <= bestErr)
				{
					bestErr = candErr;
					improved = true;

					// Drop the element from the list of elements to consider removing
					std::swap(*it, *(elements.end() - 1));
					it--;
					elements.pop_back();
				}
				else
				{
					// Put it back
					m_pInstanceFeatures->set(it->first, it->second, oldVal);

					// Try dropping the whole row
					m_pSkipRows->set(it->first);
					candErr = sumSquaredError(holdOutFeatures, holdOutLabels);
					if(candErr <= bestErr)
					{
						bestErr = candErr;
						improved = true;
					}
					else
						m_pSkipRows->unset(it->first); // Put it back
				}
			}
		}
		if(!improved)
			break;
	}

	// Get rid of the black list
	for(size_t i = m_pInstanceFeatures->rows() - 1; i < m_pInstanceFeatures->rows(); i--)
	{
		if(m_pSkipRows->bit(i))
		{
			m_pInstanceFeatures->swapRows(i, m_pInstanceFeatures->rows() - 1);
			m_pInstanceFeatures->deleteLastRow();
			m_pInstanceLabels->swapRows(i, m_pInstanceFeatures->rows() - 1);
			m_pInstanceLabels->deleteRow(m_pInstanceLabels->rows() - 1);
		}
	}
	delete(m_pSkipRows);
	m_pSkipRows = NULL;
}

void GSparseInstance::setMetric(GSparseSimilarity* pMetric)
{
	delete(m_pMetric);
	m_pMetric = pMetric;
}

// virtual
void GSparseInstance::predictDistribution(const GVec& in, GPrediction* out)
{
	throw Ex("Sorry, this learner cannot predict ditributions");
}

class GSparseInstance_comparator
{
public:
	bool operator()(const std::pair<double,size_t>& a, const std::pair<double,size_t>& b)
	{
		return a.first > b.first;
	}
};

// virtual
void GSparseInstance::predict(const GVec& in, GVec& out)
{
	// Make sure we have a metric to use
	if(!m_pMetric)
		setMetric(new GEuclidSimilarity());

	// Find the k-nearest neighbors
	priority_queue< std::pair<double,size_t>, std::vector<std::pair<double,size_t> >, GSparseInstance_comparator > neighbors;
	for(size_t i = 0; i < m_pInstanceFeatures->rows(); i++)
	{
		if(!m_pSkipRows || !m_pSkipRows->bit(i))
		{
			double similarity = m_pMetric->similarity(m_pInstanceFeatures->row(i), in);
			neighbors.push( std::pair<double,size_t>(similarity, i) );
			if(neighbors.size() > m_neighborCount)
				neighbors.pop();
		}
	}

	// Combine the labels of the neighbors, weighted by similarity
	out.fill(0.0);
	double sumWeight = 0.0;
	while(neighbors.size() > 0)
	{
		size_t index = neighbors.top().second;
		double similarity = neighbors.top().first;
		out += (m_pInstanceLabels->row(index) * similarity);
		sumWeight += similarity;
		neighbors.pop();
	}
	out *= (1.0 / std::max(1e-12, sumWeight));
}

// virtual
void GSparseInstance::clear()
{
	delete(m_pInstanceFeatures);
	m_pInstanceFeatures = NULL;
	delete(m_pInstanceLabels);
	m_pInstanceLabels = NULL;
	delete(m_pMetric);
	m_pMetric = NULL;
	delete(m_pSkipRows);
	m_pSkipRows = NULL;
}

#ifndef NO_TEST_CODE
//static
void GSparseInstance::test()
{
	GSparseInstance learner;
	learner.setNeighborCount(3);
	learner.basicTest(0.0, 0.0);
}
#endif




} // namespace GClasses
