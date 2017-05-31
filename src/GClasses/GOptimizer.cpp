/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Luke B. Godfrey,
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

#include "GOptimizer.h"
#include "GNeuralNet.h"
#include "GVec.h"
#include "GRand.h"
#include <string.h>
#include <math.h>

namespace GClasses {

using std::vector;

void GSquaredError::evaluate(const GVec &prediction, const GVec &label, GVec &loss)
{
	double err;
	for(size_t i = 0; i < prediction.size(); ++i)
	{
		if(label[i] == UNKNOWN_REAL_VALUE)
			loss[i] = 0.0;
		else
		{
			err = label[i] - prediction[i];
			loss[i] = err * err;
		}
	}
}

// the mathematically correct multiplication by 2 is omitted intentionally
void GSquaredError::calculateOutputLayerBlame(const GVec &prediction, const GVec &label, GVec &blame)
{
	if(m_useSlack)
	{
		GAssert(m_slack.size() == prediction.size(), "Slack is not the correct size!");
		for(size_t i = 0; i < prediction.size(); i++)
		{
			if(label[i] == UNKNOWN_REAL_VALUE)
				blame[i] = 0.0;
			else
			{
				if(label[i] > prediction[i] + m_slack[i])
					blame[i] = (label[i] - prediction[i] - m_slack[i]);
				else if(label[i] < prediction[i] - m_slack[i])
					blame[i] = (label[i] - prediction[i] + m_slack[i]);
				else
					blame[i] = 0.0;
			}
		}
	}
	else
	{
		for(size_t i = 0; i < prediction.size(); ++i)
		{
			if(label[i] == UNKNOWN_REAL_VALUE)
				blame[i] = 0.0;
			else
				blame[i] = label[i] - prediction[i];
		}
	}
}

#ifdef GCUDA
void GSquaredError::calculateOutputLayerBlameCuda(GCudaEngine& e, const GCudaVector& prediction, const GCudaVector& label, GCudaVector& blame)
{
	GAssert(!m_useSlack); // Sorry, slack is not implemented for the GPU yet
	blame.copy(e, label);
	blame.add(e, pred, -1.0);
}
#endif // GUCDA






GNeuralNetOptimizer::GNeuralNetOptimizer(GNeuralNet& model, GRand& rand, const GMatrix* pTrainingFeatures, const GMatrix* pTrainingLabels, GObjective* objective)
: m_objective(objective != NULL ? objective : new GSquaredError()),
  m_model(model),
  m_pContext(nullptr),
  m_pTrainingFeatures(pTrainingFeatures),
  m_pTrainingLabels(pTrainingLabels),
#ifdef GCUDA
  m_pTrainingFeaturesCuda(nullptr);
  m_pTrainingLabelsCuda(nullptr);
#endif // GCUDA
  m_rand(rand),
  m_batchSize(1), m_batchesPerEpoch(INVALID_INDEX), m_epochs(100), m_windowSize(100), m_minImprovement(0.002), m_learningRate(0.05),
  m_pII(nullptr)
{
	if(m_pTrainingFeatures && m_pTrainingLabels && m_pTrainingFeatures->rows() != m_pTrainingLabels->rows())
		throw Ex("Mismatching numbers of training features and labels");
}

GNeuralNetOptimizer::~GNeuralNetOptimizer()
{
	delete(m_pContext);
	delete(m_objective);
	delete(m_pII);
#ifdef GCUDA
	delete(m_pTrainingFeaturesCuda);
	delete(m_pTrainingLabelsCuda);
#endif // GCUDA
}

GContextNeuralNet& GNeuralNetOptimizer::context()
{
	if(!m_pContext)
	{
		m_pContext = m_model.newContext(m_rand);
		prepareForOptimizing();
	}
	return *m_pContext;
}

void GNeuralNetOptimizer::resetState()
{
	context().resetState();
}

void GNeuralNetOptimizer::optimizeIncremental(const GVec& feat, const GVec& lab)
{
	GAssert(feat.size() == m_model.layer(0).inputs() && lab.size() == m_model.outputLayer().outputs(), "Features/labels size mismatch!");
	GAssert(feat.size() != 0 && lab.size() != 0, "Features/labels are empty!");
	computeGradient(feat, lab);
	descendGradient(m_learningRate);
}

#ifdef GCUDA
void GNeuralNetOptimizer::optimizeIncrementalCuda(const GCudaVector& feat, const GCudaVector& lab)
{
	GAssert(feat.size() == m_model.layer(0).inputs() && lab.size() == m_model.outputLayer().outputs(), "Features/labels size mismatch!");
	GAssert(feat.size() != 0 && lab.size() != 0, "Features/labels are empty!");
	computeGradientCuda(feat, lab);
	descendGradientCuda(m_learningRate);
}
#endif // GCUDA

void GNeuralNetOptimizer::optimizeEpoch()
{
	if(!m_pII || m_pII->length() != m_pTrainingFeatures->rows())
	{
		delete(m_pII);
		m_pII = new GRandomIndexIterator(m_pTrainingFeatures->rows(), m_rand);
	}
	m_pII->reset();
	size_t index;
#ifdef GCUDA
	if(true)
	{
		if(!m_pTrainingFeaturesCuda)
		{
			GAssert(!m_pTrainingLabelsCuda);
			m_pTrainingFeaturesCuda = new GCudaMatrix();
			m_pTrainingFeaturesCuda->upload(*m_pTrainingFeatures);
			m_pTrainingLabelsCuda = new GCudaMatrix();
			m_pTrainingLabelsCuda->upload(*m_pTrainingLabels);
		}
		while(m_pII->next(index))
			optimizeIncrementalCuda((*m_pTrainingFeaturesCuda)[index], (*m_pTrainingLabelsCuda)[index]);
	}
	else
	{
#endif // GCUDA
		while(m_pII->next(index))
			optimizeIncremental((*m_pTrainingFeatures)[index], (*m_pTrainingLabels)[index]);
#ifdef GCUDA
	}
#endif // GCUDA
}

void GNeuralNetOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, size_t start, size_t batchSize)
{
	GAssert(features.cols() == m_model.layer(0).inputs() && labels.cols() == m_model.outputLayer().outputs(), "Features/labels size mismatch!");
	for(size_t i = 0; i < batchSize; ++i)
		computeGradient(features[start + i], labels[start + i]);
	descendGradient(m_learningRate / batchSize);
}

void GNeuralNetOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, size_t start)
{
	optimizeBatch(features, labels, start, m_batchSize);
}

void GNeuralNetOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, GRandomIndexIterator &ii, size_t batchSize)
{
	GAssert(features.cols() == m_model.layer(0).inputs() && labels.cols() == m_model.outputLayer().outputs(), "Features/labels size mismatch!");
	size_t j;
	for(size_t i = 0; i < batchSize; ++i)
	{
		if(!ii.next(j)) ii.reset(), ii.next(j);
		computeGradient(features[j], labels[j]);
	}
	descendGradient(m_learningRate / batchSize);
}

void GNeuralNetOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, GRandomIndexIterator &ii)
{
	optimizeBatch(features, labels, ii, m_batchSize);
}

void GNeuralNetOptimizer::optimize(const GMatrix &features, const GMatrix &labels)
{
	GAssert(features.cols() == m_model.layer(0).inputs() && labels.cols() == m_model.outputLayer().outputs(), "Features/labels size mismatch!");

	size_t batchesPerEpoch = m_batchesPerEpoch;
	if(m_batchesPerEpoch > features.rows())
		batchesPerEpoch = features.rows();

	GRandomIndexIterator ii(features.rows(), m_rand);
	for(size_t i = 0; i < m_epochs; ++i)
		for(size_t j = 0; j < batchesPerEpoch; ++j)
			optimizeBatch(features, labels, ii, m_batchSize);
}

void GNeuralNetOptimizer::optimizeWithValidation(const GMatrix &features, const GMatrix &labels, const GMatrix &validationFeat, const GMatrix &validationLab)
{
	size_t batchesPerEpoch = m_batchesPerEpoch;
	if(m_batchesPerEpoch > features.rows())
		batchesPerEpoch = features.rows();

	double bestError = 1e308, currentError;
	size_t k = 0;
	GRandomIndexIterator ii(features.rows(), m_rand);
	for(size_t i = 0;; ++i, ++k)
	{
		for(size_t j = 0; j < batchesPerEpoch; ++j)
			optimizeBatch(features, labels, ii, m_batchSize);
		if(k >= m_windowSize)
		{
			k = 0;
			currentError = sumLoss(validationFeat, validationLab);
			if(1.0 - currentError / bestError >= m_minImprovement)
			{
				if(currentError < bestError)
				{
					if(currentError == 0.0)
						break;
					bestError = currentError;
				}
			}
			else
				break;
		}
	}
}

void GNeuralNetOptimizer::optimizeWithValidation(const GMatrix &features, const GMatrix &labels, double validationPortion)
{
	size_t validationRows = validationPortion * features.rows();
	size_t trainRows = features.rows() - validationRows;
	if(validationRows > 0)
	{
		GDataRowSplitter splitter(features, labels, m_rand, trainRows);
		optimizeWithValidation(splitter.features1(), splitter.labels1(), splitter.features2(), splitter.labels2());
	}
	else
		optimizeWithValidation(features, labels, features, labels);
}

double GNeuralNetOptimizer::sumLoss(const GMatrix &features, const GMatrix &labels)
{
	GVec loss(labels.cols());
	double sum = 0.0;
	for(size_t i = 0; i < features.rows(); ++i)
	{
		GVec& pred = m_pContext->forwardProp(features[i]);
		m_objective->evaluate(pred, labels[i], loss);
		sum += loss.sum();
	}
	return sum;
}









GSGDOptimizer::GSGDOptimizer(GNeuralNet& model, GRand& rand, const GMatrix* pTrainingFeatures, const GMatrix* pTrainingLabels, GObjective* objective)
: GNeuralNetOptimizer(model, rand, pTrainingFeatures, pTrainingLabels, objective), m_momentum(0)
{
}

void GSGDOptimizer::prepareForOptimizing()
{
	m_gradient.resize(m_model.weightCount());
	m_gradient.fill(0.0);
#ifdef GCUDA
	m_gradientCuda.resize(m_model.weightCount());
	GContextNeuralNet& ctx = context();
	m_gradientCuda.fill(ctx.cudaEngine(), 0.0);
#endif
}

void GSGDOptimizer::computeGradient(const GVec& feat, const GVec& lab)
{
	GContextNeuralNet& ctx = context();
	GVec& pred = ctx.forwardProp_training(feat);
	m_objective->calculateOutputLayerBlame(pred, lab, ctx.blame());
	ctx.backProp();
	m_gradient *= m_momentum;
	ctx.updateGradient(feat, m_gradient);
}

void GSGDOptimizer::descendGradient(double learningRate)
{
	m_model.step(learningRate, m_gradient);
}

#ifdef GCUDA
void GSGDOptimizer::computeGradientCuda(const GCudaVector& feat, const GCudaVector& lab)
{
	GContextNeuralNet& ctx = context();
	GCudaVector& pred = ctx.forwardPropCuda(feat);
	m_objective->calculateOutputLayerBlameCuda(ctx.cudaEngine(), pred, lab, ctx.blameCuda());
	ctx.backPropCuda();
	m_gradientCuda.scale(ctx.cudaEngine(), momentum);
	ctx.updateGradientCuda(feat, m_gradientCuda);

}

void GSGDOptimizer::descendGradientCuda(double learningRate)
{
	m_model.stepCuda(context(), learningRate, m_gradientCuda);
}
#endif // GCUDA











GAdamOptimizer::GAdamOptimizer(GNeuralNet& model, GRand& rand, const GMatrix* pTrainingFeatures, const GMatrix* pTrainingLabels, GObjective* objective)
: GNeuralNetOptimizer(model, rand, pTrainingFeatures, pTrainingLabels, objective), m_correct1(1.0), m_correct2(1.0), m_beta1(0.9), m_beta2(0.999), m_epsilon(1e-8)
{
	m_learningRate = 0.001;
}

void GAdamOptimizer::prepareForOptimizing()
{
	m_gradient.resize(m_model.weightCount());
	m_deltas.resize(m_gradient.size());
	m_sqdeltas.resize(m_gradient.size());
	m_gradient.fill(0.0);
	m_deltas.fill(0.0);
	m_sqdeltas.fill(0.0);
}

void GAdamOptimizer::computeGradient(const GVec& feat, const GVec& lab)
{
	GContextNeuralNet& ctx = context();
	GVec& pred = ctx.forwardProp_training(feat);
	m_objective->calculateOutputLayerBlame(pred, lab, ctx.blame());
	ctx.backProp();
	m_gradient.fill(0.0);
	ctx.updateGradient(feat, m_gradient);
	m_correct1 *= m_beta1;
	m_correct2 *= m_beta2;
	for(size_t i = 0; i < m_gradient.size(); i++)
	{
		m_deltas[i] *= m_beta1;
		m_deltas[i] += (1.0 - m_beta1) * m_gradient[i];
		m_sqdeltas[i] *= m_beta2;
		m_sqdeltas[i] += (1.0 - m_beta2) * (m_gradient[i] * m_gradient[i]);
	}
}

void GAdamOptimizer::descendGradient(double learningRate)
{
	double alpha1 = 1.0 / (1.0 - m_correct1);
	double alpha2 = 1.0 / (1.0 - m_correct2);
	for(size_t i = 0; i < m_gradient.size(); i++)
		m_gradient[i] = alpha1 * m_deltas[i] / (std::sqrt(alpha2 * m_sqdeltas[i]) + m_epsilon);
	m_model.step(learningRate, m_gradient);
}

#ifdef GCUDA
void GAdamOptimizer::computeGradientCuda(const GCudaVector& feat, const GCudaVector& lab)
{
	throw Ex("Sorry, not implemented yet");
}

void GAdamOptimizer::descendGradientCuda(double learningRate)
{
	throw Ex("Sorry, not implemented yet");
}
#endif // GCUDA











GRMSPropOptimizer::GRMSPropOptimizer(GNeuralNet& model, GRand& rand, const GMatrix* pTrainingFeatures, const GMatrix* pTrainingLabels, GObjective* objective)
: GNeuralNetOptimizer(model, rand, pTrainingFeatures, pTrainingLabels, objective), m_momentum(0), m_gamma(0.9), m_epsilon(1e-6)
{
}

void GRMSPropOptimizer::prepareForOptimizing()
{
	m_gradient.resize(m_model.weightCount());
	m_meanSquare.resize(m_gradient.size());
	m_gradient.fill(0.0);
	m_meanSquare.fill(0.0);
}

void GRMSPropOptimizer::computeGradient(const GVec& feat, const GVec& lab)
{
	GContextNeuralNet& ctx = context();
	GVec& pred = ctx.forwardProp_training(feat);
	m_objective->calculateOutputLayerBlame(pred, lab, ctx.blame());
	ctx.backProp();
	m_gradient *= m_momentum;
	ctx.updateGradient(feat, m_gradient);
}

void GRMSPropOptimizer::descendGradient(double learningRate)
{
	for(size_t i = 0; i < m_meanSquare.size(); ++i)
	{
		m_meanSquare[i] *= m_gamma;
		m_meanSquare[i] += (1.0 - m_gamma) * m_gradient[i] * m_gradient[i];
		m_gradient[i] /= sqrt(m_meanSquare[i]) + m_epsilon;
	}
	m_model.step(learningRate, m_gradient);
}

#ifdef GCUDA
void GRMSPropOptimizer::computeGradientCuda(const GCudaVector& feat, const GCudaVector& lab)
{
	throw Ex("Sorry, not implemented yet");
}

void GRMSPropOptimizer::descendGradientCuda(double learningRate)
{
	throw Ex("Sorry, not implemented yet");
}
#endif // GCUDA














GTargetFunction::GTargetFunction(size_t dims)
{
	m_pRelation = new GUniformRelation(dims, 0);
}

// virtual
GTargetFunction::~GTargetFunction()
{
	delete(m_pRelation);
}

// virtual
void GTargetFunction::initVector(GVec& pVector)
{
	pVector.fill(0.0);
}










// virtual
double GOptimizerBasicTestTargetFunction::computeError(const GVec& pVector)
{
	double a = pVector[0] - 0.123456789;
	double b = pVector[1] + 9.876543210;
	double c = pVector[2] - 3.333333333;
	return sqrt(a * a + b * b + c * c);
}










GOptimizer::GOptimizer(GTargetFunction* pCritic)
: m_pCritic(pCritic)
{
}

// virtual
GOptimizer::~GOptimizer()
{
}

double GOptimizer::searchUntil(size_t nBurnInIterations, size_t nIterations, double dImprovement)
{
	for(size_t i = 0; i < nBurnInIterations; i++)
		iterate();
	double dPrevError;
	double dError = iterate();
	while(true)
	{
		dPrevError = dError;
		for(size_t i = 0; i < nIterations; i++)
			iterate();
		dError = iterate();
		if((dPrevError - dError) / dPrevError >= dImprovement && dError > 0.0)
		{
		}
		else
			break;
	}
	return dError;
}

#ifndef MIN_PREDICT
void GOptimizer::basicTest(double minAccuracy, double warnRange)
{
	double d = searchUntil(5, 100, 0.001);
	if(d > minAccuracy)
		throw Ex("Optimizer accuracy has regressed. Expected ", to_str(minAccuracy), ". Got ", to_str(d));
	if(d < minAccuracy - warnRange)
		std::cout << "Accuracy is much better than expected. Expected " << to_str(minAccuracy) << ". Got " << to_str(d) << ". Please tighten the expected accuracy for this test.\n";
}
#endif








GParallelOptimizers::GParallelOptimizers(size_t dims)
{
	if(dims > 0)
		m_pRelation = new GUniformRelation(dims, 0);
}

GParallelOptimizers::~GParallelOptimizers()
{
	for(vector<GOptimizer*>::iterator it = m_optimizers.begin(); it != m_optimizers.end(); it++)
		delete(*it);
	for(vector<GTargetFunction*>::iterator it = m_targetFunctions.begin(); it != m_targetFunctions.end(); it++)
		delete(*it);
}

void GParallelOptimizers::add(GTargetFunction* pTargetFunction, GOptimizer* pOptimizer)
{
	m_targetFunctions.push_back(pTargetFunction);
	m_optimizers.push_back(pOptimizer);
}

double GParallelOptimizers::iterateAll()
{
	double err = 0;
	for(vector<GOptimizer*>::iterator it = m_optimizers.begin(); it != m_optimizers.end(); it++)
		err += (*it)->iterate();
	return err;
}

double GParallelOptimizers::searchUntil(size_t nBurnInIterations, size_t nIterations, double dImprovement)
{
	for(size_t i = 1; i < nBurnInIterations; i++)
		iterateAll();
	double dPrevError;
	double dError = iterateAll();
	while(true)
	{
		dPrevError = dError;
		for(size_t i = 1; i < nIterations; i++)
			iterateAll();
		dError = iterateAll();
		if((dPrevError - dError) / dPrevError < dImprovement)
			break;
	}
	return dError;
}









class GAction
{
protected:
        int m_nAction;
        GAction* m_pPrev;
        unsigned int m_nRefs;

        ~GAction()
        {
                if(m_pPrev)
                        m_pPrev->Release(); // todo: this could overflow the stack with recursion
        }
public:
        GAction(int nAction, GAction* pPrev)
         : m_nAction(nAction), m_nRefs(0)
        {
                m_pPrev = pPrev;
                if(pPrev)
                        pPrev->AddRef();
        }

        void AddRef()
        {
                m_nRefs++;
        }

        void Release()
        {
                if(--m_nRefs == 0)
                        delete(this);
        }

        GAction* GetPrev() { return m_pPrev; }
        int GetAction() { return m_nAction; }
};












GActionPath::GActionPath(GActionPathState* pState) : m_pLastAction(NULL), m_nPathLen(0)
{
        m_pHeadState = pState;
}

GActionPath::~GActionPath()
{
        if(m_pLastAction)
                m_pLastAction->Release();
        delete(m_pHeadState);
}

void GActionPath::doAction(size_t nAction)
{
        GAction* pPrevAction = m_pLastAction;
        m_pLastAction = new GAction((int)nAction, pPrevAction);
        m_pLastAction->AddRef(); // referenced by m_pLastAction
        if(pPrevAction)
                pPrevAction->Release(); // no longer referenced by m_pLastAction
        m_nPathLen++;
        m_pHeadState->performAction(nAction);
}

GActionPath* GActionPath::fork()
{
        GActionPath* pNewPath = new GActionPath(m_pHeadState->copy());
        pNewPath->m_nPathLen = m_nPathLen;
        pNewPath->m_pLastAction = m_pLastAction;
        if(m_pLastAction)
                m_pLastAction->AddRef();
        return pNewPath;
}

void GActionPath::path(size_t nCount, size_t* pOutBuf)
{
        while(nCount > m_nPathLen)
                pOutBuf[--nCount] = -1;
        size_t i = m_nPathLen;
        GAction* pAction = m_pLastAction;
        while(i > nCount)
        {
                i--;
                pAction = pAction->GetPrev();
        }
        while(i > 0)
        {
                pOutBuf[--i] = pAction->GetAction();
                pAction = pAction->GetPrev();
        }
}

double GActionPath::critique()
{
        return m_pHeadState->critiquePath(m_nPathLen, m_pLastAction);
}

} // namespace GClasses
