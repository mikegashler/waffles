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

#include "GEnsemble.h"
#include "GVec.h"
#include <stdlib.h>
#include "GDistribution.h"
#include "GNeuralNet.h"
#include "GDom.h"
#include "GRand.h"
#include "GHolders.h"
#include "GThread.h"
#include <memory>

using namespace GClasses;
using std::vector;


GWeightedModel::GWeightedModel(GDomNode* pNode, GLearnerLoader& ll)
{
	m_weight = pNode->field("w")->asDouble();
	m_pModel = ll.loadLearner(pNode->field("m"));
}

GWeightedModel::~GWeightedModel()
{
	delete(m_pModel);
}

GDomNode* GWeightedModel::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "w", pDoc->newDouble(m_weight));
	pNode->addField(pDoc, "m", m_pModel->serialize(pDoc));
	return pNode;
}





GEnsemble::GEnsemble()
: GSupervisedLearner(), m_pLabelRel(NULL), m_workerThreads(1), m_pPredictMaster(NULL)
{
}

GEnsemble::GEnsemble(const GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode), m_pPredictMaster(NULL)
{
	m_pLabelRel = GRelation::deserialize(pNode->field("labelrel"));
	size_t accumulatorDims = (size_t)pNode->field("accum")->asInt();
	m_accumulator.resize(accumulatorDims);
	m_workerThreads = (size_t)pNode->field("threads")->asInt();
	GDomNode* pModels = pNode->field("models");
	GDomListIterator it(pModels);
	size_t modelCount = it.remaining();
	for(size_t i = 0; i < modelCount; i++)
	{
		GWeightedModel* pWM = new GWeightedModel(it.current(), ll);
		m_models.push_back(pWM);
		it.advance();
	}
}

GEnsemble::~GEnsemble()
{
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		delete(*it);
	delete(m_pLabelRel);
	delete(m_pPredictMaster);
}

// virtual
void GEnsemble::serializeBase(GDom* pDoc, GDomNode* pNode) const
{
	pNode->addField(pDoc, "labelrel", m_pLabelRel->serialize(pDoc));
	pNode->addField(pDoc, "accum", pDoc->newInt(m_accumulator.size()));
	pNode->addField(pDoc, "threads", pDoc->newInt(m_workerThreads));
	GDomNode* pModels = pNode->addField(pDoc, "models", pDoc->newList());
	for(size_t i = 0; i < m_models.size(); i++)
		pModels->addItem(pDoc, m_models[i]->serialize(pDoc));
}

void GEnsemble::clearBase()
{
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		(*it)->m_pModel->clear();
	delete(m_pLabelRel);
	m_pLabelRel = NULL;
	m_accumulator.resize(0);
	delete(m_pPredictMaster);
	m_pPredictMaster = NULL;
}

// virtual
void GEnsemble::trainInner(const GMatrix& features, const GMatrix& labels)
{
	delete(m_pLabelRel);
	m_pLabelRel = labels.relation().clone();

	// Make the accumulator buffer
	size_t labelDims = m_pLabelRel->size();
	size_t nAccumulatorDims = 0;
	for(size_t i = 0; i < labelDims; i++)
	{
		size_t nValues = m_pLabelRel->valueCount(i);
		if(nValues > 0)
			nAccumulatorDims += nValues;
		else
			nAccumulatorDims += 2; // mean and variance
	}
	m_accumulator.resize(nAccumulatorDims);

	trainInnerInner(features, labels);
}

void GEnsemble::normalizeWeights()
{
	double sum = 0.0;
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		sum += (*it)->m_weight;
	double f = 1.0 / sum;
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		(*it)->m_weight *= f;
}

void GEnsemble::castVote(double weight, const GVec& out)
{
	size_t labelDims = m_pLabelRel->size();
	size_t pos = 0;
	for(size_t i = 0; i < labelDims; i++)
	{
		size_t nValues = m_pLabelRel->valueCount(i);
		if(nValues > 0)
		{
			int nVal = (int)out[i];
			if(nVal >= 0 && nVal < (int)nValues)
				m_accumulator[pos + nVal] += weight;
			pos += nValues;
		}
		else
		{
			double dVal = out[i];
			m_accumulator[pos] += (weight * dVal);
			pos++;
			m_accumulator[pos] += (weight * (dVal * dVal));
			pos++;
		}
	}
}

void GEnsemble::tally(GPrediction* out)
{
	size_t labelDims = m_pLabelRel->size();
	size_t nDims = 0;
	double mean;
	for(size_t i = 0; i < labelDims; i++)
	{
		size_t nValues = m_pLabelRel->valueCount(i);
		if(nValues > 0)
		{
			out[i].makeCategorical()->setValues(nValues, &m_accumulator[nDims]);
			nDims += nValues;
		}
		else
		{
			mean = m_accumulator[nDims];
			out[i].makeNormal()->setMeanAndVariance(mean, m_accumulator[nDims + 1] - (mean * mean));
			nDims += 2;
		}
	}
	GAssert(nDims == m_accumulator.size()); // invalid dim count
}

void GEnsemble::tally(GVec& out)
{
	size_t labelDims = m_pLabelRel->size();
	size_t nDims = 0;
	for(size_t i = 0; i < labelDims; i++)
	{
		size_t nValues = m_pLabelRel->valueCount(i);
		if(nValues > 0)
		{
			out[i] = (double)m_accumulator.indexOfMax(nDims, nDims + nValues) - nDims;
			nDims += nValues;
		}
		else
		{
			out[i] = m_accumulator[nDims];
			nDims += 2;
		}
	}
	GAssert(nDims == m_accumulator.size()); // invalid dim count
}

class GEnsemblePredictWorker : public GWorkerThread
{
protected:
	GEnsemble* m_pEnsemble;
	GVec m_prediction;

public:
	GEnsemblePredictWorker(GMasterThread& master, GEnsemble* pEnsemble, size_t outDims)
	: GWorkerThread(master), m_pEnsemble(pEnsemble)
	{
		m_prediction.resize(outDims);
	}

	virtual ~GEnsemblePredictWorker()
	{
	}

	virtual void doJob(size_t jobId)
	{
		const GVec& in = *(const GVec*)m_pEnsemble->m_pPredictInput;
		GWeightedModel* pWM = m_pEnsemble->models()[jobId];
		pWM->m_pModel->predict(in, m_prediction);
		GSpinLockHolder lockHolder(m_master.getLock(), "GEnsemblePredictWorker::doJob");
		m_pEnsemble->castVote(pWM->m_weight, m_prediction);
	}
};

// virtual
void GEnsemble::predict(const GVec& in, GVec& out)
{
	m_accumulator.fill(0.0);
	m_pPredictInput = &in;
	if(!m_pPredictMaster)
	{
		m_pPredictMaster = new GMasterThread();
		for(size_t i = 0; i < m_workerThreads; i++)
			m_pPredictMaster->addWorker(new GEnsemblePredictWorker(*m_pPredictMaster, this, m_models[0]->m_pModel->relLabels().size()));
	}
	m_pPredictMaster->doJobs(m_models.size());
	tally(out);
}

// virtual
void GEnsemble::predictDistribution(const GVec& in, GPrediction* out)
{
	GVec tmp(m_pLabelRel->size());
	m_accumulator.fill(0.0);
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
	{
		GWeightedModel* pWM = *it;
		pWM->m_pModel->predict(in, tmp);
		castVote(pWM->m_weight, tmp);
	}
	tally(out);
}







GBag::GBag()
: GEnsemble(), m_pCB(NULL), m_pThis(NULL), m_trainSize(1.0)
{
}

GBag::GBag(const GDomNode* pNode, GLearnerLoader& ll)
: GEnsemble(pNode, ll), m_pCB(NULL), m_pThis(NULL)
{
	m_trainSize = pNode->field("ts")->asDouble();
}

GBag::~GBag()
{
}

// virtual
GDomNode* GBag::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBag");
	serializeBase(pDoc, pNode);
	pNode->addField(pDoc, "ts", pDoc->newDouble(m_trainSize));
	return pNode;
}

void GBag::clear()
{
	clearBase();
}

void GBag::flush()
{
	clear();
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		delete(*it);
	m_models.clear();
}

void GBag::addLearner(GSupervisedLearner* pLearner)
{
	pLearner->rand().setSeed(m_rand.next()); // Ensure that every learner has a different seed
	GWeightedModel* pWM = new GWeightedModel(0.0, pLearner); // The weight will be fixed later
	m_models.push_back(pWM);
}

class GBagTrainWorker : public GWorkerThread
{
protected:
	GBag* m_pBag;
	const GMatrix& m_features;
	const GMatrix& m_labels;
	GMatrix m_drawnFeatures;
	GMatrix m_drawnLabels;
	size_t m_drawSize;
	GRand m_rand;

public:
	GBagTrainWorker(GMasterThread& master, GBag* pBag, const GMatrix& features, const GMatrix& labels, double trainSize, size_t seed)
	: GWorkerThread(master),
	m_pBag(pBag),
	m_features(features),
	m_labels(labels),
	m_drawnFeatures(features.relation().clone()),
	m_drawnLabels(labels.relation().clone()),
	m_rand(seed)
	{
		GAssert(m_features.rows() > 0);
		m_drawSize = size_t(trainSize * features.rows());
		m_drawnFeatures.reserve(m_drawSize);
		m_drawnLabels.reserve(m_drawSize);
	}

	virtual ~GBagTrainWorker()
	{
	}

	virtual void doJob(size_t jobId)
	{
		// Randomly draw some data (with replacement)
		GReleaseDataHolder hDrawnFeatures(&m_drawnFeatures);
		GReleaseDataHolder hDrawnLabels(&m_drawnLabels);
		for(size_t j = 0; j < m_drawSize; j++)
		{
			size_t r = (size_t)m_rand.next(m_features.rows());
			m_drawnFeatures.takeRow((GVec*)&m_features[r]); // This case is only okay because we only use drawFeatures as a const GMatrix
			m_drawnLabels.takeRow((GVec*)&m_labels[r]); // This case is only okay because we only use drawnLabels as a const GMatrix
		}

		// Train the learner with the drawn data
		m_pBag->models()[jobId]->m_pModel->train(m_drawnFeatures, m_drawnLabels);
	}
};

// virtual
void GBag::trainInnerInner(const GMatrix& features, const GMatrix& labels)
{
/*
	// Train all the models
	size_t nLearnerCount = m_models.size();
	size_t nDrawSize = size_t(m_trainSize * features.rows());
	GMatrix drawnFeatures(features.relation().clone());
	GMatrix drawnLabels(labels.relation().clone());
	drawnFeatures.reserve(nDrawSize);
	drawnLabels.reserve(nDrawSize);
	{
		for(size_t i = 0; i < nLearnerCount; i++)
		{
			if(m_pCB)
				m_pCB(m_pThis, i, nLearnerCount);

			// Randomly draw some data (with replacement)
			GReleaseDataHolder hDrawnFeatures(&drawnFeatures);
			GReleaseDataHolder hDrawnLabels(&drawnLabels);
			for(size_t j = 0; j < nDrawSize; j++)
			{
				size_t r = (size_t)m_rand.next(features.rows());
				drawnFeatures.takeRow((double*)features[r]); // This case is only okay because we only use drawFeatures as a const GMatrix
				drawnLabels.takeRow((double*)labels[r]); // This case is only okay because we only use drawnLabels as a const GMatrix
			}

			// Train the learner with the drawn data
			m_models[i]->m_pModel->train(drawnFeatures, drawnLabels);
		}
		if(m_pCB)
			m_pCB(m_pThis, nLearnerCount, nLearnerCount);
	}

	// Determine the weights
	determineWeights(features, labels);
	normalizeWeights();
*/

	GMasterThread trainMaster;
	for(size_t i = 0; i < m_workerThreads; i++)
		trainMaster.addWorker(new GBagTrainWorker(trainMaster, this, features, labels, m_trainSize, (size_t)m_rand.next()));
	trainMaster.doJobs(m_models.size());
	determineWeights(features, labels);
	normalizeWeights();
}

// virtual
void GBag::determineWeights(const GMatrix& features, const GMatrix& labels)
{
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		(*it)->m_weight = 1.0;
}

#ifndef NO_TEST_CODE
#include "GDecisionTree.h"
// static
void GBag::test()
{
	GBag bag;
	for(size_t i = 0; i < 64; i++)
	{
		GDecisionTree* pTree = new GDecisionTree();
		pTree->useRandomDivisions();
		bag.addLearner(pTree);
	}
	bag.basicTest(0.764, 0.93, 0.01);
}
#endif






GBomb::GBomb(const GDomNode* pNode, GLearnerLoader& ll)
: GBag(pNode, ll)
{
	m_samples = (size_t)pNode->field("samps")->asInt();
}

// virtual
void GBomb::determineWeights(const GMatrix& features, const GMatrix& labels)
{
	// Try uniform weights first
	double* pWeights = new double[m_models.size()];
	std::unique_ptr<double[]> hWeights(pWeights);
	double uniform = 1.0 / m_models.size();
	GVec::setAll(pWeights, uniform, m_models.size());
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		(*it)->m_weight = uniform;
	double minErr = sumSquaredError(features, labels);

	// Try random weight combinations
	for(size_t i = 0; i < m_samples; i++)
	{
		// Set weights randomly from a dirichlet distribution with unifrom probabilities
		for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
			(*it)->m_weight = m_rand.exponential();
		normalizeWeights();

		// Evaluate accuracy
		double err = sumSquaredError(features, labels);

		// Remember the best weights yet found
		if(err < minErr)
		{
			minErr = err;
			double* pW = pWeights;
			for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
				*(pW++) = (*it)->m_weight;
		}
	}

	// Restore the best weights yet found
	double* pW = pWeights;
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		(*it)->m_weight = *(pW++);
}

// virtual
GDomNode* GBomb::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBomb");
	serializeBase(pDoc, pNode);
	pNode->addField(pDoc, "ts", pDoc->newDouble(m_trainSize));
	pNode->addField(pDoc, "samps", pDoc->newInt(m_samples));
	return pNode;
}

#ifndef NO_TEST_CODE
// static
void GBomb::test()
{
	GBomb bomb;
	for(size_t i = 0; i < 32; i++)
	{
		GDecisionTree* pTree = new GDecisionTree();
		pTree->useRandomDivisions();
		bomb.addLearner(pTree);
	}
	bomb.basicTest(0.76, 0.765, 0.01);
}
#endif







// virtual
void GBayesianModelAveraging::determineWeights(const GMatrix& features, const GMatrix& labels)
{
	double m = -1e38;
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
	{
		double d = 1.0 - ((*it)->m_pModel->sumSquaredError(features, labels) / labels.rows());
		double logProbHypothGivenData;
		if(d <= 0.0)
			logProbHypothGivenData = -1e38;
		else if(d == 1.0)
			logProbHypothGivenData = 0.0;
		else
			logProbHypothGivenData = features.rows() * (d * log(d) + (1.0 - d) * log(1.0 - d));
		m = std::max(m, logProbHypothGivenData);
		(*it)->m_weight = logProbHypothGivenData;
	}
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
	{
		double logProbHypothGivenData = (*it)->m_weight;
		(*it)->m_weight = exp(logProbHypothGivenData - m);
	}
}

// virtual
GDomNode* GBayesianModelAveraging::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBayesianModelAveraging");
	serializeBase(pDoc, pNode);
	pNode->addField(pDoc, "ts", pDoc->newDouble(m_trainSize));
	return pNode;
}

#ifndef NO_TEST_CODE
// static
void GBayesianModelAveraging::test()
{
	GBayesianModelAveraging bma;
	for(size_t i = 0; i < 32; i++)
	{
		GDecisionTree* pTree = new GDecisionTree();
		pTree->useRandomDivisions();
		bma.addLearner(pTree);
	}
	bma.basicTest(0.755, 0.928, 0.01);
}
#endif







GBayesianModelCombination::GBayesianModelCombination(const GDomNode* pNode, GLearnerLoader& ll)
: GBag(pNode, ll)
{
	m_samples = (size_t)pNode->field("samps")->asInt();
}

// virtual
void GBayesianModelCombination::determineWeights(const GMatrix& features, const GMatrix& labels)
{
	double* pWeights = new double[m_models.size()];
	std::unique_ptr<double[]> hWeights(pWeights);
	GVec::setAll(pWeights, 0.0, m_models.size());
	double sumWeight = 0.0;
	double maxLogProb = -1e38;
	for(size_t i = 0; i < m_samples; i++)
	{
		// Set weights randomly from a dirichlet distribution with unifrom probabilities
		for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
			(*it)->m_weight = m_rand.exponential();
		normalizeWeights();

		// Evaluate accuracy
		double d = 1.0 - (sumSquaredError(features, labels) / labels.rows());
		double logProbEnsembleGivenData;
		if(d <= 0.0)
			logProbEnsembleGivenData = -1e38;
		else if(d == 1.0)
			logProbEnsembleGivenData = 0.0;
		else
			logProbEnsembleGivenData = features.rows() * (d * log(d) + (1.0 - d) * log(1.0 - d));

		// Update the weights
		if(logProbEnsembleGivenData > maxLogProb)
		{
			GVec::multiply(pWeights, exp(maxLogProb - logProbEnsembleGivenData), m_models.size());
			maxLogProb = logProbEnsembleGivenData;
		}
		double w = exp(logProbEnsembleGivenData - maxLogProb);
		GVec::multiply(pWeights, sumWeight / (sumWeight + w), m_models.size());
		double* pW = pWeights;
		for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
			*(pW++) += w * (*it)->m_weight;
		sumWeight += w;
	}
	double* pW = pWeights;
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		(*it)->m_weight = *(pW++);
}

// virtual
GDomNode* GBayesianModelCombination::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBayesianModelCombination");
	serializeBase(pDoc, pNode);
	pNode->addField(pDoc, "ts", pDoc->newDouble(m_trainSize));
	pNode->addField(pDoc, "samps", pDoc->newInt(m_samples));
	return pNode;
}

#ifndef NO_TEST_CODE
// static
void GBayesianModelCombination::test()
{
	GBayesianModelCombination bmc;
	for(size_t i = 0; i < 32; i++)
	{
		GDecisionTree* pTree = new GDecisionTree();
		pTree->useRandomDivisions();
		bmc.addLearner(pTree);
	}
	bmc.basicTest(0.76, 0.928, 0.01);
}
#endif





GResamplingAdaBoost::GResamplingAdaBoost(GSupervisedLearner* pLearner, bool ownLearner, GLearnerLoader* pLoader)
: GEnsemble(), m_pLearner(pLearner), m_ownLearner(ownLearner), m_pLoader(pLoader), m_trainSize(1.0), m_ensembleSize(30)
{
}

GResamplingAdaBoost::GResamplingAdaBoost(const GDomNode* pNode, GLearnerLoader& ll)
: GEnsemble(pNode, ll), m_pLearner(NULL), m_ownLearner(false), m_pLoader(NULL)
{
	m_trainSize = pNode->field("ts")->asDouble();
	m_ensembleSize = (size_t)pNode->field("es")->asInt();
}

// virtual
GResamplingAdaBoost::~GResamplingAdaBoost()
{
	clear();
	if(m_ownLearner)
		delete(m_pLearner);
	delete(m_pLoader);
}

// virtual
GDomNode* GResamplingAdaBoost::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GResamplingAdaBoost");
	serializeBase(pDoc, pNode);
	pNode->addField(pDoc, "es", pDoc->newInt(m_ensembleSize));
	pNode->addField(pDoc, "ts", pDoc->newDouble(m_trainSize));
	return pNode;
}

// virtual
void GResamplingAdaBoost::clear()
{
	for(vector<GWeightedModel*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		delete(*it);
	m_models.clear();
	if(m_pLearner)
		m_pLearner->clear();
}

// virtual
void GResamplingAdaBoost::trainInnerInner(const GMatrix& features, const GMatrix& labels)
{
	clear();

	// Initialize all instances with uniform weights
	GVec pDistribution(features.rows());
	pDistribution.fill(1.0 / features.rows());
	size_t drawRows = size_t(m_trainSize * features.rows());
	size_t* pDrawnIndexes = new size_t[drawRows];
	std::unique_ptr<size_t[]> hDrawnIndexes(pDrawnIndexes);

	// Train the ensemble
	size_t labelDims = labels.cols();
	double penalty = 1.0 / labelDims;
	GVec prediction(labelDims);
	for(size_t es = 0; es < m_ensembleSize; es++)
	{
		// Draw a training set from the distribution
		GCategoricalSamplerBatch csb(features.rows(), pDistribution, m_rand);
		csb.draw(drawRows, pDrawnIndexes);
		GMatrix drawnFeatures(features.relation().clone());
		GReleaseDataHolder hDrawnFeatures(&drawnFeatures);
		GMatrix drawnLabels(labels.relation().clone());
		GReleaseDataHolder hDrawnLabels(&drawnLabels);
		size_t* pIndex = pDrawnIndexes;
		for(size_t i = 0; i < drawRows; i++)
		{
			drawnFeatures.takeRow((GVec*)&features[*pIndex]);
			drawnLabels.takeRow((GVec*)&labels[*pIndex]);
			pIndex++;
		}

		// Train an instance of the model and store a clone of it
		m_pLearner->train(drawnFeatures, drawnLabels);
		GDom doc;
		GSupervisedLearner* pClone = m_pLoader->loadLearner(m_pLearner->serialize(&doc));

		// Compute model weight
		double err = 0.5;
		for(size_t i = 0; i < features.rows(); i++)
		{
			pClone->predict(features[i], prediction);
			const GVec& target = labels[i];
			for(size_t j = 0; j < labelDims; j++)
			{
				if((int)target[j] != (int)prediction[j])
					err += penalty;
			}
		}
		err /= features.rows();
		if(err >= 0.5)
		{
			delete(pClone);
			break;
		}
		double weight = 0.5 * log((1.0 - err) / err);
		m_models.push_back(new GWeightedModel(weight, pClone));

		// Update the distribution to favor mis-classified instances
		for(size_t i = 0; i < features.rows(); i++)
		{
			err = 0.0;
			pClone->predict(features[i], prediction);
			const GVec& target = labels[i];
			for(size_t j = 0; j < labelDims; j++)
			{
				if((int)target[j] != (int)prediction[j])
					err += penalty;
			}
			err /= labelDims;
			pDistribution[i] *= exp(weight * (err * 2.0 - 1.0));
		}
		pDistribution.sumToOne();
	}
	normalizeWeights();
}

#ifndef NO_TEST_CODE
// static
void GResamplingAdaBoost::test()
{
	GDecisionTree* pLearner = new GDecisionTree();
	pLearner->useRandomDivisions();
	GResamplingAdaBoost boost(pLearner, true, new GLearnerLoader());
	boost.basicTest(0.753, 0.92);
}
#endif








GWag::GWag(size_t size)
: GSupervisedLearner(), m_noAlign(false)
{
	m_pNN = new GNeuralNet();
}

GWag::GWag(const GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode)
{
	m_pNN = new GNeuralNet(pNode->field("nn"));
	m_models = (size_t)pNode->field("models")->asInt();
	m_noAlign = pNode->field("na")->asBool();
}

// virtual
GWag::~GWag()
{
	delete(m_pNN);
}

// virtual
GDomNode* GWag::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GWag");
	pNode->addField(pDoc, "models", pDoc->newInt(m_models));
	pNode->addField(pDoc, "nn", m_pNN->serialize(pDoc));
	pNode->addField(pDoc, "na", pDoc->newBool(m_noAlign));
	return pNode;
}

// virtual
void GWag::clear()
{
	m_pNN->clear();
}

// virtual
void GWag::trainInner(const GMatrix& features, const GMatrix& labels)
{
	GNeuralNet* pTemp = NULL;
	std::unique_ptr<GNeuralNet> hTemp;
	size_t weights = 0;
	double* pWeightBuf = NULL;
	double* pWeightBuf2 = NULL;
	std::unique_ptr<double[]> hWeightBuf;
	for(size_t i = 0; i < m_models; i++)
	{
		m_pNN->train(features, labels);
		if(pTemp)
		{
			// Average m_pNN with pTemp
			if(!m_noAlign)
				m_pNN->align(*pTemp);
			pTemp->weights(pWeightBuf);
			m_pNN->weights(pWeightBuf2);
			GVec::multiply(pWeightBuf, double(i) / (i + 1), weights);
			GVec::addScaled(pWeightBuf, 1.0 / (i + 1), pWeightBuf2, weights);
			pTemp->setWeights(pWeightBuf);
		}
		else
		{
			// Copy the m_pNN
			GDom doc;
			GDomNode* pNode = m_pNN->serialize(&doc);
			GLearnerLoader ll;
			pTemp = new GNeuralNet(pNode);
			hTemp.reset(pTemp);
			weights = pTemp->countWeights();
			pWeightBuf = new double[2 * weights];
			hWeightBuf.reset(pWeightBuf);
			pWeightBuf2 = pWeightBuf + weights;
		}
	}
	pTemp->weights(pWeightBuf);
	m_pNN->setWeights(pWeightBuf);
}

// virtual
void GWag::predict(const GVec& in, GVec& out)
{
	m_pNN->predict(in, out);
}

// virtual
void GWag::predictDistribution(const GVec& in, GPrediction* out)
{
	m_pNN->predictDistribution(in, out);
}








GBucket::GBucket()
: GSupervisedLearner()
{
	m_nBestLearner = INVALID_INDEX;
}

GBucket::GBucket(const GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode)
{
	GDomNode* pModels = pNode->field("models");
	GDomListIterator it(pModels);
	size_t modelCount = it.remaining();
	for(size_t i = 0; i < modelCount; i++)
	{
		m_models.push_back(ll.loadLearner(it.current()));
		it.advance();
	}
	m_nBestLearner = (size_t)pNode->field("best")->asInt();
}

GBucket::~GBucket()
{
	for(vector<GSupervisedLearner*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		delete(*it);
}

// virtual
GDomNode* GBucket::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBucket");
	GDomNode* pModels = pNode->addField(pDoc, "models", pDoc->newList());
	pModels->addItem(pDoc, m_models[m_nBestLearner]->serialize(pDoc));
	pNode->addField(pDoc, "best", pDoc->newInt(0));
	return pNode;
}

void GBucket::clear()
{
	for(vector<GSupervisedLearner*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		(*it)->clear();
}

void GBucket::flush()
{
	for(vector<GSupervisedLearner*>::iterator it = m_models.begin(); it != m_models.end(); it++)
		delete(*it);
	m_models.clear();
}

void GBucket::addLearner(GSupervisedLearner* pLearner)
{
	pLearner->rand().setSeed(m_rand.next()); // Ensure that every learner has a different seed
	m_models.push_back(pLearner);
}

// virtual
void GBucket::trainInner(const GMatrix& features, const GMatrix& labels)
{
	size_t nLearnerCount = m_models.size();
	double dBestError = 1e200;
	GSupervisedLearner* pLearner;
	m_nBestLearner = (size_t)m_rand.next(nLearnerCount);
	double err;
	for(size_t i = 0; i < nLearnerCount; i++)
	{
		pLearner = m_models[i];
		err = pLearner->crossValidate(features, labels, 2);
		if(err < dBestError)
		{
			dBestError = err;
			m_nBestLearner = i;
		}
		pLearner->clear();
	}
	pLearner = m_models[m_nBestLearner];
	pLearner->train(features, labels);
}

GSupervisedLearner* GBucket::releaseBestModeler()
{
	if(m_nBestLearner == INVALID_INDEX)
		throw Ex("Not trained yet");
	GSupervisedLearner* pModeler = m_models[m_nBestLearner];
	m_models[m_nBestLearner] = m_models[m_models.size() - 1];
	m_models.pop_back();
	m_nBestLearner = -1;
	return pModeler;
}

// virtual
void GBucket::predict(const GVec& in, GVec& out)
{
	if(m_nBestLearner == INVALID_INDEX)
		throw Ex("not trained yet");
	m_models[m_nBestLearner]->predict(in, out);
}

// virtual
void GBucket::predictDistribution(const GVec& in, GPrediction* out)
{
	if(m_nBestLearner == INVALID_INDEX)
		throw Ex("not trained yet");
	m_models[m_nBestLearner]->predictDistribution(in, out);
}

#ifndef NO_TEST_CODE
#include "GDecisionTree.h"
// static
void GBucket::test()
{
	GBucket bucket;
	bucket.addLearner(new GBaselineLearner());
	bucket.addLearner(new GDecisionTree());
	bucket.addLearner(new GAutoFilter(new GMeanMarginsTree()));
	bucket.basicTest(0.695, 0.918);
}
#endif
