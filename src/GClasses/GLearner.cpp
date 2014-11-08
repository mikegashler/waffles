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

#include "GLearner.h"
#include <stdlib.h>
#include <string.h>
#include "GError.h"
#include "GVec.h"
#include "GHeap.h"
#include "GDom.h"
#ifndef MIN_PREDICT
#include "GGaussianProcess.h"
#include "GImage.h"
#endif // MIN_PREDICT
#include "GNeuralNet.h"
#ifndef MIN_PREDICT
#include "GKNN.h"
#include "GDecisionTree.h"
#include "GNaiveInstance.h"
#include "GLinear.h"
#include "GNaiveBayes.h"
#include "GEnsemble.h"
#include "GPolynomial.h"
#endif // MIN_PREDICT
#include "GTransform.h"
#include "GRand.h"
#include "GHolders.h"
#ifndef MIN_PREDICT
#include "GPlot.h"
#include "GDistribution.h"
#include "GRecommender.h"
#endif // MIN_PREDICT
#include <cmath>
#include <iostream>

using std::vector;

namespace GClasses {

#ifndef MIN_PREDICT
GPrediction::~GPrediction()
{
	delete(m_pDistribution);
}

bool GPrediction::isContinuous()
{
	return m_pDistribution->type() == GUnivariateDistribution::normal;
}

// static
void GPrediction::predictionArrayToVector(size_t nOutputCount, GPrediction* pOutputs, double* pVector)
{
	for(size_t i = 0; i < nOutputCount; i++)
		pVector[i] = pOutputs[i].mode();
}

// static
void GPrediction::vectorToPredictionArray(GRelation* pRelation, size_t nOutputCount, double* pVector, GPrediction* pOutputs)
{
	size_t nInputs = pRelation->size() - nOutputCount;
	for(size_t i = 0; i < nOutputCount; i++)
	{
		size_t nValueCount = pRelation->valueCount(nInputs + i);
		if(nValueCount == 0)
			pOutputs[i].makeNormal()->setMeanAndVariance(pVector[i], 1);
		else
			pOutputs[i].makeCategorical()->setSpike(nValueCount, (size_t)pVector[i], 1);
	}
}

double GPrediction::mode()
{
	return m_pDistribution->mode();
}


GCategoricalDistribution* GPrediction::makeCategorical()
{
	if(!m_pDistribution || m_pDistribution->type() != GUnivariateDistribution::categorical)
	{
		delete(m_pDistribution);
		m_pDistribution = new GCategoricalDistribution();
	}
	return (GCategoricalDistribution*)m_pDistribution;
}

GNormalDistribution* GPrediction::makeNormal()
{
	if(!m_pDistribution || m_pDistribution->type() != GUnivariateDistribution::normal)
	{
		delete(m_pDistribution);
		m_pDistribution = new GNormalDistribution();
	}
	return (GNormalDistribution*)m_pDistribution;
}

GCategoricalDistribution* GPrediction::asCategorical()
{
	if(!m_pDistribution || m_pDistribution->type() != GUnivariateDistribution::categorical)
		throw Ex("The current distribution is not a categorical distribution");
	return (GCategoricalDistribution*)m_pDistribution;
}

GNormalDistribution* GPrediction::asNormal()
{
	if(!m_pDistribution || m_pDistribution->type() != GUnivariateDistribution::normal)
		throw Ex("The current distribution is not a normal distribution");
	return (GNormalDistribution*)m_pDistribution;
}
#endif // MIN_PREDICT
// ---------------------------------------------------------------

GTransducer::GTransducer()
: m_rand(0)
{
}

GTransducer::~GTransducer()
{
}
#ifndef MIN_PREDICT
class GTransducerTrainAndTestCleanUpper
{
protected:
	GMatrix* m_pData;
	size_t m_nTestSize;

public:
	GTransducerTrainAndTestCleanUpper(GMatrix* pData, size_t nTestSize)
	: m_pData(pData), m_nTestSize(nTestSize)
	{
	}

	~GTransducerTrainAndTestCleanUpper()
	{
		while(m_pData->rows() > m_nTestSize)
			m_pData->releaseRow(m_pData->rows() - 1);
	}
};

// virtual
GMatrix* GTransducer::transduce(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2)
{
	if(features1.rows() != labels1.rows())
		throw Ex("Expected features1 and labels1 to have the same number of rows");
	if(features1.cols() != features2.cols())
		throw Ex("Expected both feature matrices to have the same number of cols");

	// Convert the features to a form that this algorithm can handle
	const GMatrix* pF1 = &features1;
	const GMatrix* pF2 = &features2;
	Holder<GMatrix> hF1(NULL);
	Holder<GMatrix> hF2(NULL);
	if(!canImplicitlyHandleNominalFeatures())
	{
		if(!canImplicitlyHandleContinuousFeatures())
			throw Ex("Can't handle nominal or continuous features");

		// Convert nominal features to continuous
		if(!features1.relation().areContinuous())
		{
			GNominalToCat ntc;
			ntc.train(*pF1);
			GMatrix* pTemp = ntc.transformBatch(*pF1);
			hF1.reset(pTemp);
			pF1 = pTemp;
			pTemp = ntc.transformBatch(*pF2);
			hF2.reset(pTemp);
			pF2 = pTemp;
		}
	}
	if(!canImplicitlyHandleContinuousFeatures())
	{
		if(!canImplicitlyHandleNominalFeatures())
			throw Ex("Can't handle nominal or continuous features");

		// Convert continuous features to nominal
		if(!features1.relation().areNominal())
		{
			GDiscretize disc;
			disc.train(*pF1); // todo: should really use both feature sets here
			GMatrix* pTemp = disc.transformBatch(*pF1);
			hF1.reset(pTemp);
			pF1 = pTemp;
			pTemp = disc.transformBatch(*pF2);
			hF2.reset(pTemp);
			pF2 = pTemp;
		}
	}
	if(canImplicitlyHandleContinuousFeatures())
	{
		// Normalize feature values to fall within a supported range
		double fmin, fmax;
		if(!supportedFeatureRange(&fmin, &fmax))
		{
			GNormalize norm(fmin, fmax);
			norm.train(*pF1); // todo: should really use both feature sets here
			GMatrix* pTemp = norm.transformBatch(*pF1);
			hF1.reset(pTemp);
			pF1 = pTemp;
			pTemp = norm.transformBatch(*pF2);
			hF2.reset(pTemp);
			pF2 = pTemp;
		}
	}

	// Take care of the labels
	if(!canImplicitlyHandleContinuousLabels())
	{
		if(!canImplicitlyHandleNominalLabels())
			throw Ex("This algorithm says it cannot handle nominal or continuous labels");
		if(labels1.relation().areNominal())
			return transduceInner(*pF1, labels1, *pF2);
		else
		{
			GDiscretize disc;
			disc.train(labels1);
			GMatrix* pL1 = disc.transformBatch(labels1);
			Holder<GMatrix> hL1(pL1);
			GMatrix* pL2 = transduceInner(*pF1, *pL1, *pF2);
			Holder<GMatrix> hL2(pL2);
			return disc.untransformBatch(*pL2);
		}
	}
	else
	{
		if(canImplicitlyHandleNominalLabels() || labels1.relation().areContinuous())
		{
			double lmin, lmax;
			if(supportedLabelRange(&lmin, &lmax))
				return transduceInner(*pF1, labels1, *pF2);
			else
			{
				GNormalize norm(lmin, lmax);
				norm.train(labels1);
				GMatrix* pL1 = norm.transformBatch(labels1);
				Holder<GMatrix> hL1(pL1);
				GMatrix* pL2 = transduceInner(*pF1, *pL1, *pF2);
				Holder<GMatrix> hL2(pL2);
				return norm.untransformBatch(*pL2);
			}
		}
		else
		{
			double lmin, lmax;
			if(supportedLabelRange(&lmin, &lmax))
			{
				GNominalToCat ntc;
				ntc.train(labels1);
				GMatrix* pL1 = ntc.transformBatch(labels1);
				Holder<GMatrix> hL1(pL1);
				GMatrix* pL2 = transduceInner(*pF1, *pL1, *pF2);
				Holder<GMatrix> hL2(pL2);
				return ntc.untransformBatch(*pL2);
			}
			else
			{
				// todo: both nominalToCat and normalization filters are necessary in this case
				throw Ex("case not yet supported");
				return NULL;
			}
		}
	}
}

// virtual
double GTransducer::trainAndTest(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& testFeatures, const GMatrix& testLabels)
{
	// Check assumptions
	if(testFeatures.rows() != testLabels.rows())
		throw Ex("Expected the test features to have the same number of rows as the test labels");
	if(trainFeatures.cols() != testFeatures.cols())
		throw Ex("Expected the training features and test features to have the same number of columns");

	// Transduce
	GMatrix* pPredictedLabels = transduce(trainFeatures, trainLabels, testFeatures);
	Holder<GMatrix> hPredictedLabels(pPredictedLabels);

	// Evaluate the results
	size_t labelDims = trainLabels.cols();
	double sse = 0.0;
	for(size_t i = 0; i < labelDims; i++)
		sse += testLabels.columnSumSquaredDifference(*pPredictedLabels, i);
	return sse;
}

// virtual
void GTransducer::transductiveConfusionMatrix(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& testFeatures, const GMatrix& testLabels, std::vector<GMatrix*>& stats)
{
	// Check assumptions
	if(testFeatures.rows() != testLabels.rows())
		throw Ex("Expected the test features to have the same number of rows as the test labels");
	if(trainFeatures.cols() != testFeatures.cols())
		throw Ex("Expected the training features and test features to have the same number of columns");

	// Transduce
	GMatrix* pPredictedLabels = transduce(trainFeatures, trainLabels, testFeatures);
	Holder<GMatrix> hPredictedLabels(pPredictedLabels);

	// Evaluate the results
	size_t labelDims = trainLabels.cols();
	stats.resize(labelDims);
	for(size_t j = 0; j < labelDims; j++)
	{
		size_t vals = testLabels.relation().valueCount(j);
		if(vals > 0)
		{
			stats[j] = new GMatrix(vals, vals);
			stats[j]->setAll(0.0);
		}
		else
			stats[j] = NULL;
	}
	for(size_t i = 0; i < pPredictedLabels->rows(); i++)
	{
		const double* pTarget = testLabels[i];
		double* pPred = pPredictedLabels->row(i);
		for(size_t j = 0; j < labelDims; j++)
		{
			if(stats[j])
			{
				if((int)*pTarget >= 0 && (int)*pPred >= 0)
					stats[j]->row((int)*pTarget)[(int)*pPred]++;
			}
			pTarget++;
			pPred++;
		}
	}
}

double GTransducer::crossValidate(const GMatrix& features, const GMatrix& labels, size_t folds, RepValidateCallback pCB, size_t nRep, void* pThis)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");

	// Do cross-validation
	GMatrix trainFeatures(features.relation().cloneMinimal());
	trainFeatures.reserve(features.rows());
	GMatrix testFeatures(features.relation().cloneMinimal());
	testFeatures.reserve(features.rows() / folds + 1);
	GMatrix trainLabels(labels.relation().cloneMinimal());
	trainLabels.reserve(labels.rows());
	GMatrix testLabels(labels.relation().cloneMinimal());
	testLabels.reserve(labels.rows() / folds + 1);
	double sse = 0.0;
	for(size_t i = 0; i < folds; i++)
	{
		// Divide into a training set and a test set
		GReleaseDataHolder hTrainFeatures(&trainFeatures);
		GReleaseDataHolder hTestFeatures(&testFeatures);
		GReleaseDataHolder hTrainLabels(&trainLabels);
		GReleaseDataHolder hTestLabels(&testLabels);
		size_t foldStart = i * features.rows() / folds;
		size_t foldEnd = (i + 1) * features.rows() / folds;
		for(size_t j = 0; j < foldStart; j++)
		{
			trainFeatures.takeRow((double*)features[j]);
			trainLabels.takeRow((double*)labels[j]);
		}
		for(size_t j = foldStart; j < foldEnd; j++)
		{
			testFeatures.takeRow((double*)features[j]);
			testLabels.takeRow((double*)labels[j]);
		}
		for(size_t j = foldEnd; j < features.rows(); j++)
		{
			trainFeatures.takeRow((double*)features[j]);
			trainLabels.takeRow((double*)labels[j]);
		}

		// Evaluate
		double foldsse = trainAndTest(trainFeatures, trainLabels, testFeatures, testLabels);
		sse += foldsse;
		if(pCB)
			pCB(pThis, nRep, i, foldsse, testLabels.rows());
	}
	return sse;
}

double GTransducer::repValidate(const GMatrix& features, const GMatrix& labels, size_t reps, size_t folds, RepValidateCallback pCB, void* pThis)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	GMatrix f(features.relation().cloneMinimal());
	GReleaseDataHolder hF(&f);
	GMatrix l(labels.relation().cloneMinimal());
	GReleaseDataHolder hL(&l);
	for(size_t i = 0; i < features.rows(); i++)
	{
		f.takeRow((double*)features[i]);
		l.takeRow((double*)labels[i]);
	}
	double ssse = 0.0;
	for(size_t i = 0; i < reps; i++)
	{
		f.shuffle(m_rand, &l);
		ssse += crossValidate(f, l, folds, pCB, i, pThis);
	}
	return ssse / reps;
}
#endif // MIN_PREDICT

// ---------------------------------------------------------------

GSupervisedLearner::GSupervisedLearner()
: GTransducer(), m_pRelFeatures(NULL), m_pRelLabels(NULL)
{
}

GSupervisedLearner::GSupervisedLearner(GDomNode* pNode, GLearnerLoader& ll)
: GTransducer()
{
	m_pRelFeatures = GRelation::deserialize(pNode->field("_rf"));
	m_pRelLabels = GRelation::deserialize(pNode->field("_rl"));
}

GSupervisedLearner::~GSupervisedLearner()
{
	delete(m_pRelFeatures);
	delete(m_pRelLabels);
}

const GRelation& GSupervisedLearner::relFeatures()
{
	if(!m_pRelFeatures)
		throw Ex("Training has not begun yet");
	return *m_pRelFeatures;
}

const GRelation& GSupervisedLearner::relLabels()
{
	if(!m_pRelLabels)
		throw Ex("Training has not begun yet");
	return *m_pRelLabels;
}

#ifndef MIN_PREDICT
GDomNode* GSupervisedLearner::baseDomNode(GDom* pDoc, const char* szClassName) const
{
	if(!m_pRelLabels)
		throw Ex("The model must be trained before it is serialized.");
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "class", pDoc->newString(szClassName));
	pNode->addField(pDoc, "_rf", m_pRelFeatures->serialize(pDoc));
	pNode->addField(pDoc, "_rl", m_pRelLabels->serialize(pDoc));
	return pNode;
}

std::string to_str(const GSupervisedLearner& learner)
{
	GDom doc;
	learner.serialize(&doc);
	return to_str(doc);
}

void GSupervisedLearner::train(const GMatrix& features, const GMatrix& labels)
{
	// Check assumptions
	if(features.rows() != labels.rows())
		throw Ex("Expected features and labels to have the same number of rows");
	if(labels.cols() == 0)
		throw Ex("Expected at least one label dimension");
	delete(m_pRelFeatures);
	m_pRelFeatures = features.relation().cloneMinimal();
	delete(m_pRelLabels);
	m_pRelLabels = labels.relation().cloneMinimal();
	trainInner(features, labels);
}

void GSupervisedLearner::confusion(GMatrix& features, GMatrix& labels, std::vector<GMatrix*>& stats)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and rows to have the same number of rows");
	size_t labelDims = labels.cols();
	stats.resize(labelDims);
	for(size_t j = 0; j < labelDims; j++)
	{
		size_t vals = labels.relation().valueCount(j);
		if(vals > 0)
		{
			stats[j] = new GMatrix(vals, vals);
			stats[j]->setAll(0.0);
		}
		else
			stats[j] = NULL;
	}
	GTEMPBUF(double, prediction, labelDims);
	for(size_t i = 0; i < features.rows(); i++)
	{
		predict(features[i], prediction);
		double* target = labels[i];
		for(size_t j = 0; j < labelDims; j++)
		{
			if(labels.relation().valueCount(j) > 0)
			{
				if((int)target[j] >= 0 && (int)prediction[j] >= 0)
					stats[j]->row((int)target[j])[(int)prediction[j]]++;
			}
		}
	}
}

double GSupervisedLearner::sumSquaredError(const GMatrix& features, const GMatrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	if(!m_pRelFeatures->isCompatible(features.relation()))
		throw Ex("Features incompatible with this learner");
	if(!m_pRelLabels->isCompatible(labels.relation()))
		throw Ex("Labels incompatible with this learner");
	size_t labelDims = labels.cols();
	GTEMPBUF(double, prediction, labelDims);
	double sse = 0.0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		predict(features[i], prediction);
		const double* targ = labels[i];
		double* pred = prediction;
		for(size_t j = 0; j < labelDims; j++)
		{
			if(labels.relation().valueCount(j) == 0)
			{
				if(*targ != UNKNOWN_REAL_VALUE)
				{
					double d = *targ - *pred;
					sse += (d * d);
				}
			}
			else
			{
				if(*targ != UNKNOWN_DISCRETE_VALUE && (int)*targ != (int)*pred)
					sse += 1.0;
			}
			targ++;
			pred++;
		}
	}
	return sse;
}

// virtual
GMatrix* GSupervisedLearner::transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2)
{
	// Train
	train(features1, labels1);

	// Predict
	GMatrix* pOut = new GMatrix(labels1.relation().clone());
	pOut->newRows(features2.rows());
	for(size_t i = 0; i < features2.rows(); i++)
		predict(features2.row(i), pOut->row(i));
	return pOut;
}

// virtual
double GSupervisedLearner::trainAndTest(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& testFeatures, const GMatrix& testLabels)
{
	train(trainFeatures, trainLabels);
	return sumSquaredError(testFeatures, testLabels);
}

size_t GSupervisedLearner::precisionRecallContinuous(GPrediction* pOutput, double* pFunc, GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, size_t label)
{
	// Predict the variance for each pattern
	train(trainFeatures, trainLabels);
	GMatrix stats(testFeatures.rows(), 2);
	for(size_t i = 0; i < testFeatures.rows(); i++)
	{
		predictDistribution(testFeatures[i], pOutput);
		double* pResultsVec = stats.row(i);
		pResultsVec[0] = testLabels[i][label];
		if(pResultsVec[0] < 0.0 || pResultsVec[0] > 1.0)
			throw Ex("Expected continuous labels to range from 0 to 1");
		GNormalDistribution* pDist = pOutput[label].asNormal();
		pResultsVec[1] = pDist->mean();
	}

	// Make the precision/recall data
	stats.sort(1); // biggest mean last
	stats.reverseRows(); // biggest mean first
	double sumRelevantRetrieved = 0.0;
	for(size_t i = 0; i < stats.rows(); i++)
	{
		double* pVecIn = stats.row(i);
		sumRelevantRetrieved += pVecIn[0];
		pFunc[i] = sumRelevantRetrieved / (i + 1);
	}
	return stats.rows();
}

size_t GSupervisedLearner::precisionRecallNominal(GPrediction* pOutput, double* pFunc, GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, size_t label, int value)
{
	// Predict the likelihood that each pattern is relevant
	train(trainFeatures, trainLabels);
	GMatrix stats(testFeatures.rows(), 2);
	size_t nActualRelevant = 0;
	for(size_t i = 0; i < testFeatures.rows(); i++)
	{
		predictDistribution(testFeatures[i], pOutput);
		double* pStatsVec = stats.row(i);
		pStatsVec[0] = testLabels[i][label];
		if((int)pStatsVec[0] == value)
			nActualRelevant++;
		GCategoricalDistribution* pDist = pOutput[label].asCategorical();
		pStatsVec[1] = pDist->likelihood((double)value); // predicted confidence that it is relevant
	}

	// Make the precision/recall data
	stats.sort(1); // most confident last
	size_t nFoundRelevant = 0;
	size_t nFoundTotal = 0;
	for(size_t i = stats.rows() - 1; i < stats.rows(); i--)
	{
		double* pVecIn = stats.row(i);
		nFoundTotal++;
		if((int)pVecIn[0] == value) // if actually relevant
		{
			nFoundRelevant++;
			if(nFoundTotal <= 1)
				pFunc[nFoundRelevant - 1] = 1.0;
			else
				pFunc[nFoundRelevant - 1] = (double)(nFoundRelevant - 1) / (nFoundTotal - 1);
		}
	}
	GAssert(nFoundRelevant == nActualRelevant);
	return nActualRelevant;
}

void GSupervisedLearner::precisionRecall(double* pOutPrecision, size_t nPrecisionSize, GMatrix& features, GMatrix& labels, size_t label, size_t nReps)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	size_t nFuncs = std::max((size_t)1, labels.relation().valueCount(label));
	GVec::setAll(pOutPrecision, 0.0, nFuncs * nPrecisionSize);
	double* pFunc = new double[features.rows()];
	ArrayHolder<double> hFunc(pFunc);
	GPrediction* out = new GPrediction[labels.cols()];
	ArrayHolder<GPrediction> hOut(out);
	GMatrix otherFeatures(features.relation().cloneMinimal());
	GMatrix otherLabels(labels.relation().cloneMinimal());
	size_t valueCount = labels.relation().valueCount(label);
	for(size_t nRep = 0; nRep < nReps; nRep++)
	{
		// Split the data
		GMergeDataHolder hFeatures(features, otherFeatures);
		GMergeDataHolder hLabels(labels, otherLabels);
		features.shuffle(m_rand, &labels);
		size_t otherSize = features.rows() / 2;
		features.splitBySize(otherFeatures, otherSize);
		labels.splitBySize(otherLabels, otherSize);

		// Measure precision/recall and merge with the data we've gotten so far
		if(valueCount == 0)
		{
			size_t relevant = precisionRecallContinuous(out, pFunc, features, labels, otherFeatures, otherLabels, label);
			GVec::addInterpolatedFunction(pOutPrecision, nPrecisionSize, pFunc, relevant);
			relevant = precisionRecallContinuous(out, pFunc, otherFeatures, otherLabels, features, labels, label);
			GVec::addInterpolatedFunction(pOutPrecision, nPrecisionSize, pFunc, relevant);
		}
		else
		{
			for(int i = 0; i < (int)valueCount; i++)
			{
				size_t relevant = precisionRecallNominal(out, pFunc, features, labels, otherFeatures, otherLabels, label, i);
				GVec::addInterpolatedFunction(pOutPrecision + nPrecisionSize * i, nPrecisionSize, pFunc, relevant);
				relevant = precisionRecallNominal(out, pFunc, otherFeatures, otherLabels, features, labels, label, i);
				GVec::addInterpolatedFunction(pOutPrecision + nPrecisionSize * i, nPrecisionSize, pFunc, relevant);
			}
		}
	}
	GVec::multiply(pOutPrecision, 1.0 / (2 * nReps), nFuncs * nPrecisionSize);
}


#define TEST_SIZE 5000
// static
void GSupervisedLearner::test()
{
/*	// Make a probabilistic training set
	GRand rand(0);
	vector<size_t> vals1;
	vals1.push_back(3);
	vector<size_t> vals2;
	vals2.push_back(2);
	GMatrix f(vals1);
	GMatrix l(vals2);
	f.newRows(TEST_SIZE);
	l.newRows(TEST_SIZE);
	for(size_t i = 0; i < TEST_SIZE; i++)
	{
		size_t n = size_t(rand.next(3));
		if(n == 0)
		{
			if(rand.uniform() < 0.15)
				l[i][0] = 0;
			else
				l[i][0] = 1;
		}
		else if(n == 1)
		{
			if(rand.uniform() < 0.3)
				l[i][0] = 0;
			else
				l[i][0] = 1;
		}
		else
		{
			if(rand.uniform() < 0.85)
				l[i][0] = 0;
			else
				l[i][0] = 1;
		}
		f[i][0] = double(n);
	}

	// Train the model
	GNeuralNet model;
	model.addLayer(new GNeuralNetLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	model.train(f, l);
	GPrediction out;
	double d, prob;

// Uncomment this block if you want to see how it does without calibration (which should be a little worse than with it).
// 	d = 0;
// 	model.predictDistribution(&d, &out);
// 	prob = out.asCategorical()->values(2)[0];
// 	d = 1;
// 	model.predictDistribution(&d, &out);
// 	prob = out.asCategorical()->values(2)[0];
// 	d = 2;
// 	model.predictDistribution(&d, &out);
// 	prob = out.asCategorical()->values(2)[0];

	// Calibrate the model
	model.calibrate(f, l);

	// Check that the predicted distributions are close to the expected distributions
	d = 0;
	model.predictDistribution(&d, &out);
	prob = out.asCategorical()->values(2)[0];
	if(std::abs(prob - 0.15) > .11)
		throw Ex("failed");
	d = 1;
	model.predictDistribution(&d, &out);
	prob = out.asCategorical()->values(2)[0];
	if(std::abs(prob - 0.30) > .16)
		throw Ex("failed");
	d = 2;
	model.predictDistribution(&d, &out);
	prob = out.asCategorical()->values(2)[0];
	if(std::abs(prob - 0.85) > .11)
		throw Ex("failed");*/
}

void GSupervisedLearner_basicTestEngine(GSupervisedLearner* pLearner, GMatrix& features, GMatrix& labels, GMatrix& testFeatures, GMatrix& testLabels, double minAccuracy, GRand* pRand, double warnRange, double deviation, bool printAccuracy)
{
	// Train the model
	pLearner->train(features, labels);

	// free up some memory, just because we can
	features.flush();
	labels.flush();

	// Test the accuracy
	double resultsBefore = 1.0 - pLearner->sumSquaredError(testFeatures, testLabels) / testFeatures.rows();
	if(printAccuracy){
	  std::cerr << "AccBeforeSerial: " << resultsBefore;
	}
	if(resultsBefore < minAccuracy)
		throw Ex("accuracy has regressed. Expected at least ", to_str(minAccuracy), ". Only got ", to_str(resultsBefore), ". (Sometimes, harmless changes that affect random orderings can trigger small regressions, so don't panic yet.)");
	if(resultsBefore >= minAccuracy + warnRange)
		std::cout << "\nThe measured accuracy (" << resultsBefore << ") is much better than expected (" << minAccuracy << "). Please increase the expected accuracy value so that any future regressions will be caught.\n";

	// Roundtrip the model through serialization
	const GRelation& relLabelsBefore = pLearner->relLabels();
	GDom doc;
	doc.setRoot(pLearner->serialize(&doc));
	pLearner->clear(); // free up some memory, just because we can
	GLearnerLoader ll;
	GSupervisedLearner* pModel = ll.loadLearner(doc.root());
	Holder<GSupervisedLearner> hModel(pModel);
	if(!relLabelsBefore.isCompatible(pModel->relLabels()))
		throw Ex("The label relation failed to round-trip. Did your deserializing constructor call the base class constructor?");

	// Test the accuracy again
	double resultsAfter = 1.0 - pModel->sumSquaredError(testFeatures, testLabels) / testFeatures.rows();
	if(printAccuracy){
	  std::cerr << "  AccAfterSerial: " << resultsAfter << std::endl;
	}
	if(std::abs(resultsAfter - resultsBefore) > deviation)
		throw Ex("serialization shouldn't influence accuracy this much");
}

void GSupervisedLearner_basicTest1(GSupervisedLearner* pLearner, double minAccuracy, GRand* pRand, double warnRange, double deviation, bool printAccuracy)
{
	GMatrix features(0, 2);
	vector<size_t> vals;
	vals.push_back(3);
	GMatrix labels(vals);
	for(size_t i = 0; i < 2000; i++)
	{
		int c = (int)pRand->next(3);
		double* pF = features.newRow();
		pF[0] = pRand->normal() + (c == 1 ? 2.0 : 0.0);
		pF[1] = pRand->normal() + (c == 2 ? 2.0 : 0.0);
		double* pL = labels.newRow();
		pL[0] = (double)c;
	}
	size_t testSize = features.rows() / 2;
	GMatrix testFeatures(features.relation().clone());
	features.splitBySize(testFeatures, testSize);
	GMatrix testLabels(labels.relation().clone());
	labels.splitBySize(testLabels, testSize);
	GSupervisedLearner_basicTestEngine(pLearner, features, labels, testFeatures, testLabels, minAccuracy, pRand, warnRange, deviation, printAccuracy);
}

void GSupervisedLearner_basicTest2(GSupervisedLearner* pLearner, double minAccuracy, GRand* pRand, double warnRange, double deviation, bool printAccuracy)
{
	if(minAccuracy == -1.0)
		return; // skip this test
	vector<size_t> featureVals;
	size_t cols = 10;
	for(size_t i = 0; i < cols; i++)
		featureVals.push_back(3);
	GMatrix features(featureVals);
	vector<size_t> labelVals;
	labelVals.push_back(3);
	GMatrix labels(labelVals);
	for(size_t i = 0; i < 1000; i++)
	{
		int c = (int)pRand->next(3);
		double* pF = features.newRow();
		for(size_t j = 0; j < cols; j++)
		{
			if(pRand->next(2) == 0)
				*pF = (double)c;
			else
				*pF = (double)pRand->next(3);
			pF++;
		}
		double* pL = labels.newRow();
		pL[0] = (double)c;
	}
	size_t testSize = features.rows() / 2;
	GMatrix testFeatures(features.relation().clone());
	features.splitBySize(testFeatures, testSize);
	GMatrix testLabels(labels.relation().clone());
	labels.splitBySize(testLabels, testSize);
	GSupervisedLearner_basicTestEngine(pLearner, features, labels, testFeatures, testLabels, minAccuracy, pRand, warnRange, deviation, printAccuracy);
}

void GSupervisedLearner::basicTest(double minAccuracy1, double minAccuracy2, double deviation, bool printAccuracy, double warnRange)
{
	GSupervisedLearner_basicTest1(this, minAccuracy1, &m_rand, warnRange, deviation, printAccuracy);
	GSupervisedLearner_basicTest2(this, minAccuracy2, &m_rand, warnRange, deviation * 2, printAccuracy);
}
#endif // MIN_PREDICT

// ---------------------------------------------------------------

void GIncrementalLearner::beginIncrementalLearning(const GRelation& featureRel, const GRelation& labelRel)
{
	delete(m_pRelFeatures);
	m_pRelFeatures = featureRel.cloneMinimal();
	delete(m_pRelLabels);
	m_pRelLabels = labelRel.cloneMinimal();
	beginIncrementalLearningInner(featureRel, labelRel);
}

// ---------------------------------------------------------------

// virtual
GIncrementalTransform* GLearnerLoader::loadIncrementalTransform(GDomNode* pNode)
{
	const char* szClass = pNode->field("class")->asString();
	if(szClass[0] == 'G')
	{
		if(szClass[1] < 'N')
		{
			if(szClass[1] < 'I')
			{
#ifndef MIN_PREDICT
				if(strcmp(szClass, "GAttributeSelector") == 0)
					return new GAttributeSelector(pNode, *this);
				else
#endif // MIN_PREDICT
				if(strcmp(szClass, "GDataAugmenter") == 0)
					return new GDataAugmenter(pNode, *this);
				else if(strcmp(szClass, "GDiscretize") == 0)
					return new GDiscretize(pNode, *this);
			}
			else
			{
#ifndef MIN_PREDICT
				if(strcmp(szClass, "GImputeMissingVals") == 0)
					return new GImputeMissingVals(pNode, *this);
				else
#endif // MIN_PREDICT
				if(strcmp(szClass, "GIncrementalTransformChainer") == 0)
					return new GIncrementalTransformChainer(pNode, *this);
			}
		}
		else
		{
			if(szClass[1] < 'P')
			{
				if(strcmp(szClass, "GNoiseGenerator") == 0)
					return new GNoiseGenerator(pNode, *this);
				else if(strcmp(szClass, "GNominalToCat") == 0)
					return new GNominalToCat(pNode, *this);
				else if(strcmp(szClass, "GNormalize") == 0)
					return new GNormalize(pNode, *this);
			}
			else
			{
				if(strcmp(szClass, "GPairProduct") == 0)
					return new GPairProduct(pNode, *this);
				else if(strcmp(szClass, "GPCA") == 0)
					return new GPCA(pNode, *this);
				else if(strcmp(szClass, "GReservoir") == 0)
					return new GReservoir(pNode, *this);
			}
		}
	}
	if(m_throwIfClassNotFound)
		throw Ex("Unrecognized class: ", szClass);
	return NULL;
}

// virtual
GSupervisedLearner* GLearnerLoader::loadLearner(GDomNode* pNode)
{
#ifndef MIN_PREDICT
	const char* szClass = pNode->field("class")->asString();
	if(szClass[0] == 'G')
	{
		if(szClass[1] < 'J')
		{
			if(szClass[1] < 'C')
			{
				if(strcmp(szClass, "GAutoFilter") == 0)
					return new GAutoFilter(pNode, *this);
				else if(strcmp(szClass, "GBag") == 0)
					return new GBag(pNode, *this);
				else if(strcmp(szClass, "GBaselineLearner") == 0)
					return new GBaselineLearner(pNode, *this);
				else if(strcmp(szClass, "GBayesianModelAveraging") == 0)
					return new GBayesianModelAveraging(pNode, *this);
				else if(strcmp(szClass, "GBayesianModelCombination") == 0)
					return new GBayesianModelCombination(pNode, *this);
				else if(strcmp(szClass, "GBucket") == 0)
					return new GBucket(pNode, *this);
			}
			else
			{
				if(strcmp(szClass, "GDecisionTree") == 0)
					return new GDecisionTree(pNode, *this);
				else if(strcmp(szClass, "GFeatureFilter") == 0)
					return new GFeatureFilter(pNode, *this);
				else if(strcmp(szClass, "GGaussianProcess") == 0)
					return new GGaussianProcess(pNode, *this);
				else if(strcmp(szClass, "GIdentityFunction") == 0)
					return new GIdentityFunction(pNode, *this);
			}
		}
		else
		{
			if(szClass[1] < 'N')
			{
				if(strcmp(szClass, "GLinearDistribution") == 0)
					return new GLinearDistribution(pNode, *this);
				else if(strcmp(szClass, "GKNN") == 0)
					return new GKNN(pNode, *this);
				else if(strcmp(szClass, "GLabelFilter") == 0)
					return new GLabelFilter(pNode, *this);
				else if(strcmp(szClass, "GLinearRegressor") == 0)
					return new GLinearRegressor(pNode, *this);
				else if(strcmp(szClass, "GMeanMarginsTree") == 0)
					return new GMeanMarginsTree(pNode, *this);
			}
			else
			{
				if(szClass[1] < 'P')
				{
					if(strcmp(szClass, "GNaiveBayes") == 0)
						return new GNaiveBayes(pNode, *this);
					else if(strcmp(szClass, "GNaiveInstance") == 0)
						return new GNaiveInstance(pNode, *this);
					else if(strcmp(szClass, "GNeuralNet") == 0)
						return new GNeuralNet(pNode, *this);
				}
				else
				{
					if(strcmp(szClass, "GPolynomial") == 0)
						return new GPolynomial(pNode, *this);
					else if(strcmp(szClass, "GRandomForest") == 0)
						return new GRandomForest(pNode, *this);
					else if(strcmp(szClass, "GResamplingAdaBoost") == 0)
						return new GResamplingAdaBoost(pNode, *this);
					else if(strcmp(szClass, "GReservoirNet") == 0)
						return new GReservoirNet(pNode, *this);
					else if(strcmp(szClass, "GWag") == 0)
						return new GWag(pNode, *this);
				}
			}
		}
	}
#endif // MIN_PREDICT
	if(m_throwIfClassNotFound)
		throw Ex("Unrecognized class: ", szClass);
	return NULL;
}

#ifndef MIN_PREDICT
// virtual
GCollaborativeFilter* GLearnerLoader::loadCollaborativeFilter(GDomNode* pNode)
{
	const char* szClass = pNode->field("class")->asString();
	if(szClass[0] == 'G')
	{
		if(strcmp(szClass, "GBagOfRecommenders") == 0)
			return new GBagOfRecommenders(pNode, *this);
		else if(strcmp(szClass, "GBaselineRecommender") == 0)
			return new GBaselineRecommender(pNode, *this);
		else if(strcmp(szClass, "GMatrixFactorization") == 0)
			return new GMatrixFactorization(pNode, *this);
		else if(strcmp(szClass, "GNeuralRecommender") == 0)
			return new GNonlinearPCA(pNode, *this);
	}
	if(m_throwIfClassNotFound)
		throw Ex("Unrecognized class: ", szClass);
	return NULL;
}

// ---------------------------------------------------------------

GFilter::GFilter(GSupervisedLearner* pLearner, bool ownLearner)
: GIncrementalLearner(), m_pLearner(pLearner), m_pIncrementalLearner(NULL), m_ownLearner(ownLearner)
{
	if(pLearner->canTrainIncrementally())
		m_pIncrementalLearner = (GIncrementalLearner*)pLearner;
}

GFilter::GFilter(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalLearner(pNode, ll), m_pIncrementalLearner(NULL), m_ownLearner(true)
{
	m_pLearner = ll.loadLearner(pNode->field("learner"));
	if(m_pLearner->canTrainIncrementally())
		m_pIncrementalLearner = (GIncrementalLearner*)m_pLearner;
}

// virtual
GFilter::~GFilter()
{
	if(m_ownLearner)
		delete(m_pLearner);
}

// virtual
void GFilter::clear()
{
	m_pLearner->clear();
}

void GFilter::initShellOnly(const GRelation& featureRel, const GRelation& labelRel)
{
	delete(m_pRelFeatures);
	m_pRelFeatures = featureRel.cloneMinimal();
	delete(m_pRelLabels);
	m_pRelLabels = labelRel.cloneMinimal();
}

void GFilter::discardIntermediateFilters()
{
	if(m_pLearner->isFilter())
	{
		GFilter* pIntermediateFilter = (GFilter*)m_pLearner;
		pIntermediateFilter->discardIntermediateFilters();
		GSupervisedLearner* pTemp = pIntermediateFilter->m_pLearner;
		pIntermediateFilter->m_pLearner = NULL;
		pIntermediateFilter->m_pIncrementalLearner = NULL;
		delete(m_pLearner);
		m_pLearner = pTemp;
		if(m_pLearner->canTrainIncrementally())
			m_pIncrementalLearner = (GIncrementalLearner*)m_pLearner;
		else
			m_pIncrementalLearner = NULL;
	}
}

GDomNode* GFilter::domNode(GDom* pDoc, const char* szClassName) const
{
	GDomNode* pNode = baseDomNode(pDoc, szClassName);
	pNode->addField(pDoc, "learner", m_pLearner->serialize(pDoc));
	return pNode;
}

GMatrix* GFilter::prefilterFeatures(const GMatrix& in)
{
	GSupervisedLearner* pInnerLearner = m_pLearner;
	while(pInnerLearner->isFilter())
		pInnerLearner = ((GFilter*)pInnerLearner)->m_pLearner;
	GMatrix* pOut = new GMatrix(pInnerLearner->relFeatures().clone());
	pOut->newRows(in.rows());
	for(size_t i = 0; i < in.rows(); i++)
		GVec::copy(pOut->row(i), prefilterFeatures(in[i]), pOut->cols());
	return pOut;
}

GMatrix* GFilter::prefilterLabels(const GMatrix& in)
{
	GSupervisedLearner* pInnerLearner = m_pLearner;
	while(pInnerLearner->isFilter())
		pInnerLearner = ((GFilter*)pInnerLearner)->m_pLearner;
	GMatrix* pOut = new GMatrix(pInnerLearner->relLabels().clone());
	pOut->newRows(in.rows());
	for(size_t i = 0; i < in.rows(); i++)
		GVec::copy(pOut->row(i), prefilterLabels(in[i]), pOut->cols());
	return pOut;
}

#ifndef MIN_PREDICT
// virtual
void GFilter::trainSparse(GSparseMatrix& features, GMatrix& labels)
{
	throw Ex("Sorry, this method has not been implemented");
}
#endif // MIN_PREDICT

// ---------------------------------------------------------------

GFeatureFilter::GFeatureFilter(GSupervisedLearner* pLearner, GIncrementalTransform* pTransform, bool ownLearner, bool ownTransform)
: GFilter(pLearner, ownLearner), m_pTransform(pTransform), m_ownTransform(ownTransform)
{
}

GFeatureFilter::GFeatureFilter(GDomNode* pNode, GLearnerLoader& ll)
: GFilter(pNode, ll), m_ownTransform(true)
{
	m_pTransform = ll.loadIncrementalTransform(pNode->field("trans"));
}

// virtual
GFeatureFilter::~GFeatureFilter()
{
	if(m_ownTransform)
		delete(m_pTransform);
}

// virtual
GDomNode* GFeatureFilter::serialize(GDom* pDoc) const
{
	GDomNode* pNode = domNode(pDoc, "GFeatureFilter");
	pNode->addField(pDoc, "trans", m_pTransform->serialize(pDoc));
	return pNode;
}

// virtual
void GFeatureFilter::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected features and labels to have the same number of rows");
	m_pTransform->train(features);
	GMatrix temp(m_pTransform->after().clone());
	temp.newRows(features.rows());
	for(size_t i = 0; i < features.rows(); i++)
		m_pTransform->transform(features[i], temp[i]);
	m_pLearner->train(temp, labels);
}

// virtual
void GFeatureFilter::predict(const double* pIn, double* pOut)
{
	m_pTransform->transform(pIn, m_pTransform->innerBuf());
	m_pLearner->predict(m_pTransform->innerBuf(), pOut);
}

// virtual
void GFeatureFilter::predictDistribution(const double* pIn, GPrediction* pOut)
{
	m_pTransform->transform(pIn, m_pTransform->innerBuf());
	m_pLearner->predictDistribution(m_pTransform->innerBuf(), pOut);
}

// virtual
void GFeatureFilter::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	m_pTransform->train(featureRel);
	m_pIncrementalLearner->beginIncrementalLearning(m_pTransform->after(), labelRel);
}

// virtual
void GFeatureFilter::trainIncremental(const double* pIn, const double* pOut)
{
	m_pTransform->transform(pIn, m_pTransform->innerBuf());
	m_pIncrementalLearner->trainIncremental(m_pTransform->innerBuf(), pOut);
}

// virtual
const double* GFeatureFilter::prefilterFeatures(const double* pIn)
{
	m_pTransform->transform(pIn, m_pTransform->innerBuf());
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterFeatures(m_pTransform->innerBuf());
	else
		return m_pTransform->innerBuf();
}

// virtual
const double* GFeatureFilter::prefilterLabels(const double* pIn)
{
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterLabels(pIn);
	else
		return pIn;
}

// ---------------------------------------------------------------

GLabelFilter::GLabelFilter(GSupervisedLearner* pLearner, GIncrementalTransform* pTransform, bool ownLearner, bool ownTransform)
: GFilter(pLearner, ownLearner), m_pTransform(pTransform), m_ownTransform(ownTransform)
{
}

GLabelFilter::GLabelFilter(GDomNode* pNode, GLearnerLoader& ll)
: GFilter(pNode, ll), m_ownTransform(true)
{
	m_pTransform = ll.loadIncrementalTransform(pNode->field("trans"));
}

// virtual
GLabelFilter::~GLabelFilter()
{
	if(m_ownTransform)
		delete(m_pTransform);
}

// virtual
GDomNode* GLabelFilter::serialize(GDom* pDoc) const
{
	GDomNode* pNode = domNode(pDoc, "GLabelFilter");
	pNode->addField(pDoc, "trans", m_pTransform->serialize(pDoc));
	return pNode;
}

// virtual
void GLabelFilter::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected features and labels to have the same number of rows");
	m_pTransform->train(labels);
	GMatrix temp(m_pTransform->after().clone());
	temp.newRows(labels.rows());
	for(size_t i = 0; i < labels.rows(); i++)
		m_pTransform->transform(labels[i], temp[i]);
	m_pLearner->train(features, temp);
}

// virtual
void GLabelFilter::predict(const double* pIn, double* pOut)
{
	m_pLearner->predict(pIn, m_pTransform->innerBuf());
	m_pTransform->untransform(m_pTransform->innerBuf(), pOut);
}

// virtual
void GLabelFilter::predictDistribution(const double* pIn, GPrediction* pOut)
{
	m_pLearner->predict(pIn, m_pTransform->innerBuf());
	m_pTransform->untransformToDistribution(m_pTransform->innerBuf(), pOut);
}

// virtual
void GLabelFilter::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	m_pTransform->train(labelRel);
	m_pIncrementalLearner->beginIncrementalLearning(featureRel, m_pTransform->after());
}

// virtual
void GLabelFilter::trainIncremental(const double* pIn, const double* pOut)
{
	m_pTransform->transform(pOut, m_pTransform->innerBuf());
	m_pIncrementalLearner->trainIncremental(pIn, m_pTransform->innerBuf());
}

// virtual
const double* GLabelFilter::prefilterFeatures(const double* pIn)
{
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterFeatures(pIn);
	else
		return pIn;
}

// virtual
const double* GLabelFilter::prefilterLabels(const double* pIn)
{
	m_pTransform->transform(pIn, m_pTransform->innerBuf());
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterLabels(m_pTransform->innerBuf());
	else
		return m_pTransform->innerBuf();
}

// ---------------------------------------------------------------

GAutoFilter::GAutoFilter(GSupervisedLearner* pLearner, bool ownLearner)
: GFilter(pLearner, ownLearner)
{
}

GAutoFilter::GAutoFilter(GDomNode* pNode, GLearnerLoader& ll)
: GFilter(pNode, ll)
{
}

// virtual
GAutoFilter::~GAutoFilter()
{
}

// virtual
GDomNode* GAutoFilter::serialize(GDom* pDoc) const
{
	GDomNode* pNode = domNode(pDoc, "GAutoFilter");
	return pNode;
}

void GAutoFilter::whatTypesAreNeeded(const GRelation& featureRel, const GRelation& labelRel, bool& hasNominalFeatures, bool& hasContinuousFeatures, bool& hasNominalLabels, bool& hasContinuousLabels)
{
	// Determine what types are present in the feature data
	hasNominalFeatures = false;
	hasContinuousFeatures = false;
	for(size_t i = 0; i < featureRel.size(); i++)
	{
		if(featureRel.valueCount(i) == 0)
		{
			hasContinuousFeatures = true;
			if(hasNominalFeatures)
				break;
		}
		else
		{
			hasNominalFeatures = true;
			if(hasContinuousFeatures)
				break;
		}
	}

	// Determine what types are present in the label data
	hasNominalLabels = false;
	hasContinuousLabels = false;
	for(size_t i = 0; i < labelRel.size(); i++)
	{
		if(labelRel.valueCount(i) == 0)
		{
			hasContinuousLabels = true;
			if(hasNominalLabels)
				break;
		}
		else
		{
			hasNominalLabels = true;
			if(hasContinuousLabels)
				break;
		}
	}
}

void GAutoFilter::setupDataDependentFilters(GSupervisedLearner* pLearner, const GMatrix& features, const GMatrix& labels, bool hasNominalFeatures, bool hasContinuousFeatures, bool hasNominalLabels, bool hasContinuousLabels)
{
	// Impute features if necessary
	if(!pLearner->canImplicitlyHandleMissingFeatures())
	{
		if(features.doesHaveAnyMissingValues())
		{
			GImputeMissingVals* pImputer = new GImputeMissingVals();
			pImputer->setLabels(&labels); // use the labels to help impute missing features
			m_pLearner = new GFeatureFilter(m_pLearner, pImputer);
		}
	}

	// Normalize labels if necessary
	if(pLearner->canImplicitlyHandleContinuousLabels())
	{
		double supportedMin, supportedMax;
		if(!pLearner->supportedLabelRange(&supportedMin, &supportedMax))
		{
			double dataMin = 1e300;
			double dataMax = -1e300;
			for(size_t i = 0; i < labels.cols(); i++)
			{
				if(labels.relation().valueCount(i) != 0)
					continue;
				dataMin = std::min(labels.columnMin(i), dataMin);
				dataMax = std::max(labels.columnMax(i), dataMax);
			}
			if(hasNominalLabels && !canImplicitlyHandleNominalLabels())
			{
				supportedMin = std::min(0.0, supportedMin);
				supportedMax = std::max(1.0, supportedMax);
			}
			bool normalizationIsNeeded = false;
			if(dataMin < supportedMin || dataMax > supportedMax)
				normalizationIsNeeded = true;
			else if((dataMax - dataMin) >= 1e-12 && (dataMax - dataMin) * 4 < supportedMax - supportedMin)
				normalizationIsNeeded = true;
			if(normalizationIsNeeded)
				m_pLearner = new GLabelFilter(m_pLearner, new GNormalize(supportedMin, supportedMax));
		}
	}

	// Normalize features if necessary
	if(pLearner->canImplicitlyHandleContinuousFeatures())
	{
		double supportedMin, supportedMax;
		if(!pLearner->supportedFeatureRange(&supportedMin, &supportedMax))
		{
			double dataMin = 1e300;
			double dataMax = -1e300;
			for(size_t i = 0; i < labels.cols(); i++)
			{
				if(features.relation().valueCount(i) != 0)
					continue;
				dataMin = std::min(labels.columnMin(i), dataMin);
				dataMax = std::max(labels.columnMax(i), dataMax);
			}
			if(hasNominalLabels && !canImplicitlyHandleNominalLabels())
			{
				supportedMin = std::min(0.0, supportedMin);
				supportedMax = std::max(1.0, supportedMax);
			}
			bool normalizationIsNeeded = false;
			if(dataMin < supportedMin || dataMax > supportedMax)
				normalizationIsNeeded = true;
			else if((dataMax - dataMin) >= 1e-12 && (dataMax - dataMin) * 4 < supportedMax - supportedMin)
				normalizationIsNeeded = true;
			if(normalizationIsNeeded)
				m_pLearner = new GFeatureFilter(m_pLearner, new GNormalize(supportedMin, supportedMax));
		}
	}
}

void GAutoFilter::setupBasicFilters(GSupervisedLearner* pLearner, bool hasNominalFeatures, bool hasContinuousFeatures, bool hasNominalLabels, bool hasContinuousLabels)
{
	// Nomcat labels if necessary
	if(hasNominalLabels && !pLearner->canImplicitlyHandleNominalLabels())
	{
		if(!canImplicitlyHandleContinuousLabels())
			throw Ex("Expected a learner that can handle either nominal or continuous labels");
		m_pLearner = new GLabelFilter(m_pLearner, new GNominalToCat(16));
	}

	// Nomcat features if necessary
	if(hasNominalFeatures && !pLearner->canImplicitlyHandleNominalFeatures())
	{
		if(!canImplicitlyHandleContinuousFeatures())
			throw Ex("Expected a learner that can handle either nominal or continuous features");
		m_pLearner = new GFeatureFilter(m_pLearner, new GNominalToCat(16));
	}

	// Discretize labels if necessary
	if(hasContinuousLabels && !pLearner->canImplicitlyHandleContinuousLabels())
	{
		if(!canImplicitlyHandleNominalLabels())
			throw Ex("Expected a learner that can handle either nominal or continuous labels");
		m_pLearner = new GLabelFilter(m_pLearner, new GDiscretize());
	}

	// Discretize features if necessary
	if(hasContinuousFeatures && !pLearner->canImplicitlyHandleContinuousFeatures())
	{
		if(!canImplicitlyHandleNominalFeatures())
			throw Ex("Expected a learner that can handle either nominal or continuous features");
		m_pLearner = new GFeatureFilter(m_pLearner, new GDiscretize());
	}
}

void GAutoFilter::resetFilters(const GMatrix& features, const GMatrix& labels)
{
	discardIntermediateFilters();
	GSupervisedLearner* pLearner = m_pLearner;
	bool hasNominalFeatures, hasContinuousFeatures, hasNominalLabels, hasContinuousLabels;
	whatTypesAreNeeded(features.relation(), labels.relation(), hasNominalFeatures, hasContinuousFeatures, hasNominalLabels, hasContinuousLabels);
	setupDataDependentFilters(pLearner, features, labels, hasNominalFeatures, hasContinuousFeatures, hasNominalLabels, hasContinuousLabels);
	setupBasicFilters(pLearner, hasNominalFeatures, hasContinuousFeatures, hasNominalLabels, hasContinuousLabels);

	if(m_pLearner->canTrainIncrementally())
		m_pIncrementalLearner = (GIncrementalLearner*)m_pLearner;
	else
		m_pIncrementalLearner = NULL;
}

void GAutoFilter::resetFilters(const GRelation& features, const GRelation& labels)
{
	discardIntermediateFilters();
	GSupervisedLearner* pLearner = m_pLearner;
	bool hasNominalFeatures, hasContinuousFeatures, hasNominalLabels, hasContinuousLabels;
	whatTypesAreNeeded(features, labels, hasNominalFeatures, hasContinuousFeatures, hasNominalLabels, hasContinuousLabels);
	setupBasicFilters(pLearner, hasNominalFeatures, hasContinuousFeatures, hasNominalLabels, hasContinuousLabels);

	if(m_pLearner->canTrainIncrementally())
		m_pIncrementalLearner = (GIncrementalLearner*)m_pLearner;
	else
		m_pIncrementalLearner = NULL;
}

// virtual
const double* GAutoFilter::prefilterFeatures(const double* pIn)
{
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterFeatures(pIn);
	else
		return pIn;
}

// virtual
const double* GAutoFilter::prefilterLabels(const double* pIn)
{
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterLabels(pIn);
	else
		return pIn;
}

// virtual
void GAutoFilter::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected features and labels to have the same number of rows");
	resetFilters(features, labels);
	m_pLearner->train(features, labels);
}

// virtual
void GAutoFilter::predict(const double* pIn, double* pOut)
{
	m_pLearner->predict(pIn, pOut);
}

// virtual
void GAutoFilter::predictDistribution(const double* pIn, GPrediction* pOut)
{
	m_pLearner->predictDistribution(pIn, pOut);
}

// virtual
void GAutoFilter::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	resetFilters(featureRel, labelRel);
	m_pIncrementalLearner->beginIncrementalLearning(featureRel, labelRel);
}

// virtual
void GAutoFilter::trainIncremental(const double* pIn, const double* pOut)
{
	m_pIncrementalLearner->trainIncremental(pIn, pOut);
}

#ifndef MIN_PREDICT
void GAutoFilter::test()
{
	// This test trains a neural network (which only handles continuous values)
	// in an incremental manner to implement a simple autoencoder for categorical values.
	// This demonstrates that GAutoFilter picks the right filters, applies them, and
	// works with incremental learning.
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	GAutoFilter af(pNN);
	GUniformRelation rel(1, 3);
	af.beginIncrementalLearning(rel, rel);
	GRand rand(0);
	double pat[1];
	for(size_t i = 0; i < 500; i++)
	{
		pat[0] = (double)rand.next(3);
		af.trainIncremental(pat, pat);
	}
	double pred[1];
	for(size_t i = 0; i < 10; i++)
	{
		pat[0] = (double)rand.next(3);
		af.predict(pat, pred);
		if(std::abs(pred[0] - pat[0]) > 1e-12)
			throw Ex("failed");
	}
}
#endif // MIN_PREDICT



// ---------------------------------------------------------------

GCalibrator::GCalibrator(GSupervisedLearner* pLearner)
: GFilter(pLearner)
{
}

GCalibrator::GCalibrator(GDomNode* pNode, GLearnerLoader& ll)
: GFilter(pNode, ll)
{
}

// virtual
GCalibrator::~GCalibrator()
{
}

// virtual
GDomNode* GCalibrator::serialize(GDom* pDoc) const
{
	GDomNode* pNode = domNode(pDoc, "GCalibrator");
	return pNode;
}

// virtual
void GCalibrator::trainInner(const GMatrix& features, const GMatrix& labels)
{
	// Throw out any existing calibration
	if(m_pCalibrations)
	{
		size_t labelDims = m_pRelLabels->size();
		for(size_t i = 0; i < labelDims; i++)
			delete(m_pCalibrations[i]);
		delete[] m_pCalibrations;
		m_pCalibrations = NULL;
	}

	// Train
	m_pLearner->train(features, labels);

	// Calibrate
	size_t labelDims = m_pRelLabels->size();
	vector<GNeuralNet*> calibrations;
	VectorOfPointersHolder<GNeuralNet> hCalibrations(calibrations);
	size_t neighbors = std::max(size_t(4), std::min(size_t(100), (size_t)sqrt(double(features.rows()))));
	GPrediction* out = new GPrediction[labelDims];
	ArrayHolder<GPrediction> hOut(out);
	for(size_t i = 0; i < labelDims; i++)
	{
		// Gather the predicted (before) distribution values
		size_t vals = labels.relation().valueCount(i);
		GMatrix tmpBefore(features.rows(), std::max(size_t(1), vals));
		if(vals == 0)
		{
			for(size_t j = 0; j < features.rows(); j++)
			{
				predictDistribution(features[j], out);
				tmpBefore[j][0] = out[i].asNormal()->variance();
			}
		}
		else
		{
			for(size_t j = 0; j < features.rows(); j++)
			{
				predictDistribution(features[j], out);
				GVec::copy(tmpBefore[j], out[i].asCategorical()->values(vals), vals);
			}
		}

		// Use a temporary k-NN model to measure the target (after) distribution values
		GKNN knn;
		knn.setNeighborCount(neighbors);
		knn.train(tmpBefore, labels);
		GMatrix tmpAfter(features.rows(), std::max(size_t(1), vals));
		if(vals == 0)
		{
			for(size_t j = 0; j < tmpBefore.rows(); j++)
			{
				knn.predictDistribution(tmpBefore[j], out);
				tmpAfter[j][0] = out[0].asNormal()->variance();
			}
		}
		else
		{
			for(size_t j = 0; j < features.rows(); j++)
			{
				knn.predictDistribution(tmpBefore[j], out);
				GVec::copy(tmpAfter[j], out[0].asCategorical()->values(vals), vals);
			}
		}

		// Train a layer of logistic units to map from the before distribution to the after distribution
		GNeuralNet* pNN = new GNeuralNet();
		pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		calibrations.push_back(pNN);
		pNN->train(tmpBefore, tmpAfter);
	}

	// Store the resulting calibration functions
	GAssert(calibrations.size() == labelDims);
	m_pCalibrations = new GNeuralNet*[labelDims];
	for(size_t i = 0; i < labelDims; i++)
	{
		m_pCalibrations[i] = calibrations[i];
		calibrations[i] = NULL;
	}
}

// virtual
void GCalibrator::predict(const double* pIn, double* pOut)
{
	m_pLearner->predict(pIn, pOut);
}

// virtual
void GCalibrator::predictDistribution(const double* pIn, GPrediction* pOut)
{
	m_pLearner->predictDistribution(pIn, pOut);

	// Adjust the predicted distributions to make them approximate real distributions
	GVecBuf vb;
	if(m_pCalibrations)
	{
		size_t labelDims = m_pRelLabels->size();
		for(size_t i = 0; i < labelDims; i++)
		{
			if(pOut[i].isContinuous())
			{
				GNormalDistribution* pNorm = pOut[i].asNormal();
				double varBefore = pNorm->variance();
				double varAfter;
				m_pCalibrations[i]->predict(&varBefore, &varAfter);
				pNorm->setMeanAndVariance(pNorm->mean(), varAfter);
			}
			else
			{
				GCategoricalDistribution* pCat = pOut[i].asCategorical();
				vb.reserve(pCat->valueCount());
				m_pCalibrations[i]->predict(pCat->values(pCat->valueCount()), vb.m_pBuf);
				GVec::copy(pCat->values(pCat->valueCount()), vb.m_pBuf, pCat->valueCount());
			}
		}
	}
}

// virtual
void GCalibrator::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	throw Ex("Sorry, GCalibrator does not support incremental learning");
}

// virtual
void GCalibrator::trainIncremental(const double* pIn, const double* pOut)
{
	throw Ex("Sorry, GCalibrator does not support incremental learning");
}

// virtual
const double* GCalibrator::prefilterFeatures(const double* pIn)
{
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterFeatures(pIn);
	else
		return pIn;
}

// virtual
const double* GCalibrator::prefilterLabels(const double* pIn)
{
	if(m_pLearner->isFilter())
		return ((GFilter*)m_pLearner)->prefilterLabels(pIn);
	else
		return pIn;
}

// ---------------------------------------------------------------

GBaselineLearner::GBaselineLearner()
: GSupervisedLearner()
{
}

GBaselineLearner::GBaselineLearner(GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode, ll)
{
	m_prediction.clear();
	GDomNode* pPred = pNode->field("pred");
	GDomListIterator it(pPred);
	m_prediction.reserve(it.remaining());
	for(size_t i = 0; it.current(); i++)
	{
		m_prediction.push_back(it.current()->asDouble());
		it.advance();
	}
}

// virtual
GBaselineLearner::~GBaselineLearner()
{
	clear();
}

// virtual
void GBaselineLearner::clear()
{
	m_prediction.clear();
}

// virtual
GDomNode* GBaselineLearner::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBaselineLearner");
	if(m_prediction.size() == 0)
		throw Ex("Attempted to serialize a model that has not been trained");
	GDomNode* pPred = pNode->addField(pDoc, "pred", pDoc->newList());
	for(size_t i = 0; i < m_prediction.size(); i++)
		pPred->addItem(pDoc, pDoc->newDouble(m_prediction[i]));
	return pNode;
}

// virtual
void GBaselineLearner::trainInner(const GMatrix& features, const GMatrix& labels)
{
	clear();
	size_t labelDims = labels.cols();
	m_prediction.reserve(labelDims);
	for(size_t i = 0; i < labelDims; i++)
		m_prediction.push_back(labels.baselineValue(i));
}

// virtual
void GBaselineLearner::predictDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, this learner cannot predict a distribution");
}

// virtual
void GBaselineLearner::predict(const double* pIn, double* pOut)
{
	for(vector<double>::iterator it = m_prediction.begin(); it != m_prediction.end(); it++)
		*(pOut++) = *it;
}

void GBaselineLearner::autoTune(GMatrix& features, GMatrix& labels)
{
	// This model has no parameters to tune
}

#ifndef MIN_PREDICT
// static
void GBaselineLearner::test()
{
	GBaselineLearner bl;
	bl.basicTest(0.33, 0.33);
}
#endif // MIN_PREDICT

// ---------------------------------------------------------------

GIdentityFunction::GIdentityFunction()
: GSupervisedLearner(), m_labelDims(0), m_featureDims(0)
{
}

GIdentityFunction::GIdentityFunction(GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode, ll)
{
	m_labelDims = (size_t)pNode->field("labels")->asInt();
	m_featureDims = (size_t)pNode->field("features")->asInt();
}

// virtual
GIdentityFunction::~GIdentityFunction()
{
}

// virtual
void GIdentityFunction::clear()
{
	m_labelDims = 0;
	m_featureDims = 0;
}

// virtual
GDomNode* GIdentityFunction::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GIdentityFunction");
	pNode->addField(pDoc, "labels", pDoc->newInt(m_labelDims));
	pNode->addField(pDoc, "features", pDoc->newInt(m_featureDims));
	return pNode;
}

// virtual
void GIdentityFunction::trainInner(const GMatrix& features, const GMatrix& labels)
{
	m_labelDims = labels.cols();
	m_featureDims = features.cols();
}

// virtual
void GIdentityFunction::predictDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, not implemented yet");
}

// virtual
void GIdentityFunction::predict(const double* pIn, double* pOut)
{
	if(m_labelDims <= m_featureDims)
		GVec::copy(pOut, pIn, m_labelDims);
	else
	{
		GVec::copy(pOut, pIn, m_featureDims);
		GVec::setAll(pOut + m_featureDims, 0.0, m_labelDims - m_featureDims);
	}
}
#endif // MIN_PREDICT

} // namespace GClasses
