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
void GSquaredError::calculateDerivative(const GVec &prediction, const GVec &label, GVec &blame)
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

GNeuralNetFunction::GNeuralNetFunction(GNeuralNet &nn) : m_nn(nn)
{}

void GNeuralNetFunction::evaluate(const GVec &x, GVec &y)
{
	GAssert(y.size() == m_nn.outputLayer().outputs(), "Can't calculate output; not enough space in y!");
	m_nn.predict(x, y);
}

void GNeuralNetFunction::updateDeltas(const GVec &x, const GVec &blame, GVec &deltas)
{
	GAssert(deltas.size() == countParameters(), "Can't update gradient; not enough space in deltas!");
	m_nn.backpropagate(blame);
	const GVec *in = &x;
	GVecWrapper out(deltas.data(), 0);
	for(size_t i = 0; i < m_nn.layerCount(); ++i)
	{
		size_t count = m_nn.layer(i).countWeights();
		out.setSize(count);
		if(count > 0)
			((GParameterizedLayer*)&m_nn.layer(i))->updateDeltas(*in, out.vec());
		in = &m_nn.layer(i).activation();
		out.setData(out.vec().data() + count);
	}
}

void GNeuralNetFunction::applyDeltas(const GVec &deltas)
{
	GConstVecWrapper delta(deltas.data(), 0);
	for(size_t i = 0; i < m_nn.layerCount(); ++i)
	{
		size_t count = m_nn.layer(i).countWeights();
		delta.setSize(count);
		if(count > 0)
			((GParameterizedLayer*)&m_nn.layer(i))->applyDeltas(delta.vec());
		delta.setData(delta.vec().data() + count);
	}
}

size_t GNeuralNetFunction::inputs() const
{
	return m_nn.layer(0).inputs();
}

size_t GNeuralNetFunction::outputs() const
{
	return m_nn.outputLayer().outputs();
}

size_t GNeuralNetFunction::countParameters() const
{
	return m_nn.countWeights();
}

GDifferentiableOptimizer::GDifferentiableOptimizer(GDifferentiable *target, GObjective *objective)
: m_target(target), m_objective(objective != NULL ? objective : new GSquaredError()),
  m_rand(new GRand(0)), m_ownsRand(true),
  m_batchSize(1), m_batchesPerEpoch(INVALID_INDEX), m_epochs(100), m_windowSize(100), m_minImprovement(0.002)
{}

GDifferentiableOptimizer::~GDifferentiableOptimizer()
{
	if(m_ownsRand)
		delete(m_rand);
	delete m_target;
	delete m_objective;
}

void GDifferentiableOptimizer::optimizeIncremental(const GVec &feat, const GVec &lab)
{
	GAssert(m_target != NULL, "Target must be set before optimization!");
	GAssert(feat.size() == m_target->inputs() && lab.size() == m_target->outputs(), "Features/labels size mismatch!");
	GAssert(feat.size() != 0 && lab.size() != 0, "Features/labels are empty!");
	updateDeltas(feat, lab);
	applyDeltas();
}

void GDifferentiableOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, size_t start, size_t batchSize)
{
	GAssert(m_target != NULL, "Target must be set before optimization!");
	GAssert(features.cols() == m_target->inputs() && labels.cols() == m_target->outputs(), "Features/labels size mismatch!");
	for(size_t i = 0; i < batchSize; ++i)
		updateDeltas(features[start + i], labels[start + i]);
	scaleDeltas(1.0 / batchSize);
	applyDeltas();
}

void GDifferentiableOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, size_t start)
{
	optimizeBatch(features, labels, start, m_batchSize);
}

void GDifferentiableOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, GRandomIndexIterator &ii, size_t batchSize)
{
	GAssert(m_target != NULL, "Target must be set before optimization!");
	GAssert(features.cols() == m_target->inputs() && labels.cols() == m_target->outputs(), "Features/labels size mismatch!");
	size_t j;
	for(size_t i = 0; i < batchSize; ++i)
	{
		if(!ii.next(j)) ii.reset(), ii.next(j);
		updateDeltas(features[j], labels[j]);
	}
	scaleDeltas(1.0 / batchSize);
	applyDeltas();
}

void GDifferentiableOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, GRandomIndexIterator &ii)
{
	optimizeBatch(features, labels, ii, m_batchSize);
}

void GDifferentiableOptimizer::optimize(const GMatrix &features, const GMatrix &labels)
{
	GAssert(m_target != NULL, "Target must be set before optimization!");
	GAssert(features.cols() == m_target->inputs() && labels.cols() == m_target->outputs(), "Features/labels size mismatch!");

	size_t batchesPerEpoch = m_batchesPerEpoch;
	if(m_batchesPerEpoch > features.rows())
		batchesPerEpoch = features.rows();

	GRandomIndexIterator ii(features.rows(), *m_rand);
	for(size_t i = 0; i < m_epochs; ++i)
		for(size_t j = 0; j < batchesPerEpoch; ++j)
			optimizeBatch(features, labels, ii, m_batchSize);
}

void GDifferentiableOptimizer::optimizeWithValidation(const GMatrix &features, const GMatrix &labels, const GMatrix &validationFeat, const GMatrix &validationLab)
{
	GAssert(m_target != NULL, "Target must be set before optimization!");

	size_t batchesPerEpoch = m_batchesPerEpoch;
	if(m_batchesPerEpoch > features.rows())
		batchesPerEpoch = features.rows();

	double bestError = 1e308, currentError;
	size_t k = 0;
	GRandomIndexIterator ii(features.rows(), *m_rand);
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

void GDifferentiableOptimizer::optimizeWithValidation(const GMatrix &features, const GMatrix &labels, double validationPortion)
{
	size_t validationRows = validationPortion * features.rows();
	size_t trainRows = features.rows() - validationRows;
	if(validationRows > 0)
	{
		GDataRowSplitter splitter(features, labels, *m_rand, trainRows);
		optimizeWithValidation(splitter.features1(), splitter.labels1(), splitter.features2(), splitter.labels2());
	}
	else
		optimizeWithValidation(features, labels, features, labels);
}

double GDifferentiableOptimizer::sumLoss(const GMatrix &features, const GMatrix &labels)
{
	GVec pred(labels.cols()), loss(labels.cols());
	double sum = 0.0;
	for(size_t i = 0; i < features.rows(); ++i)
	{
		m_target->evaluate(features[i], pred);
		m_objective->evaluate(pred, labels[i], loss);
		sum += loss.sum();
	}
	return sum;
}

GSGDOptimizer::GSGDOptimizer(GDifferentiable *target, GObjective *objective)
: GDifferentiableOptimizer(target, objective), m_learningRate(1e-1), m_momentum(0)
{
	if(target != NULL)
		prepareForOptimizing();
}

void GSGDOptimizer::prepareForOptimizing()
{
	GAssert(m_target != NULL, "Target function must be set before preparing to optimize!");
	m_pred.resize(m_target->outputs());
	m_blame.resize(m_target->outputs());
	m_gradient.resize(m_target->countParameters());
	m_deltas.resize(m_target->countParameters());
	m_gradient.fill(0.0);
}

void GSGDOptimizer::updateDeltas(const GVec &feat, const GVec &lab)
{
	m_blame.fill(0.0);
	m_target->evaluate(feat, m_pred);
	m_objective->calculateDerivative(m_pred, lab, m_blame);
	m_target->updateDeltas(feat, m_blame, m_gradient);
}

void GSGDOptimizer::scaleDeltas(double scale)
{
	m_gradient *= scale;
}

void GSGDOptimizer::applyDeltas()
{
	m_deltas.fill(0.0);
	m_deltas.addScaled(m_learningRate, m_gradient);
	m_target->applyDeltas(m_deltas);
	m_gradient *= m_momentum;
}

GRMSPropOptimizer::GRMSPropOptimizer(GDifferentiable *target, GObjective *objective)
: GDifferentiableOptimizer(target, objective), m_learningRate(1e-1), m_momentum(0), m_gamma(0.9), m_epsilon(1e-6)
{
	if(target != NULL)
		prepareForOptimizing();
}

void GRMSPropOptimizer::prepareForOptimizing()
{
	m_pred.resize(m_target->outputs());
	m_blame.resize(m_target->outputs());
	m_gradient.resize(m_target->countParameters());
	m_deltas.resize(m_target->countParameters());
	m_meanSquare.resize(m_target->countParameters());
	m_gradient.fill(0.0);
	m_meanSquare.fill(0.0);
}

void GRMSPropOptimizer::updateDeltas(const GVec &feat, const GVec &lab)
{
	m_blame.fill(0.0);
	m_target->evaluate(feat, m_pred);
	m_objective->calculateDerivative(m_pred, lab, m_blame);
	m_target->updateDeltas(feat, m_blame, m_gradient);
}

void GRMSPropOptimizer::scaleDeltas(double scale)
{
	m_gradient *= scale;
}

void GRMSPropOptimizer::applyDeltas()
{
	for(size_t i = 0; i < m_meanSquare.size(); ++i)
	{
		m_meanSquare[i] *= m_gamma;
		m_meanSquare[i] += (1.0 - m_gamma) * m_gradient[i] * m_gradient[i];
		m_gradient[i] /= sqrt(m_meanSquare[i]) + m_epsilon;
	}

	m_deltas.fill(0.0);
	m_deltas.addScaled(m_learningRate, m_gradient);
	m_target->applyDeltas(m_deltas);
	m_gradient *= m_momentum;
}


















// MARK: Old GOptimizer.cpp below

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

// -------------------------------------------------------

// virtual
double GOptimizerBasicTestTargetFunction::computeError(const GVec& pVector)
{
	double a = pVector[0] - 0.123456789;
	double b = pVector[1] + 9.876543210;
	double c = pVector[2] - 3.333333333;
	return sqrt(a * a + b * b + c * c);
}

// -------------------------------------------------------


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
// -------------------------------------------------------

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

// -------------------------------------------------------

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

// -------------------------------------------------------

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
