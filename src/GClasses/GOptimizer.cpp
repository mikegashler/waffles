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
	for(size_t i = 0; i < prediction.size(); ++i)
	{
		if(label[i] == UNKNOWN_REAL_VALUE)
			blame[i] = 0.0;
		else
			blame[i] = label[i] - prediction[i];
	}
}

GNeuralNetFunction::GNeuralNetFunction(GNeuralNet &nn) : m_nn(nn)
{}

void GNeuralNetFunction::evaluate(const GVec &x, GVec &y)
{
	GAssert(y.size() == m_nn.outputLayer().outputs(), "Can't calculate output; not enough space in y!");
	m_nn.predict(x, y);
}

/// \todo make GNeuralNet have a backpropagate that takes in blame; don't store blame in GNeuralNet
void GNeuralNetFunction::updateDeltas(const GVec &x, const GVec &blame, GVec &deltas)
{
	GAssert(deltas.size() == countParameters(), "Can't update gradient; not enough space in deltas!");
	m_nn.outputLayer().error().put(0, blame);
	m_nn.backpropagateErrorAlreadySet();
	const GVec *in = &x;
	GVecWrapper out(deltas.data(), 0);
	for(size_t i = 0; i < m_nn.layerCount(); ++i)
	{
		size_t count = m_nn.layer(i).countWeights();
		out.setSize(count);
		m_nn.layer(i).updateDeltas(*in, out.vec());
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
		m_nn.layer(i).applyDeltas(delta.vec());
		delta.setData(delta.vec().data() + count);
	}
}

size_t GNeuralNetFunction::countParameters() const
{
	return m_nn.countWeights();
}

GDifferentiableOptimizer::GDifferentiableOptimizer(GDifferentiable *target, GObjective *objective)
: m_target(target), m_objective(objective != NULL ? objective : new GSquaredError())
{}

GDifferentiableOptimizer::~GDifferentiableOptimizer()
{
	delete m_target;
	delete m_objective;
}

void GDifferentiableOptimizer::setTarget(GDifferentiable *target)
{
	delete m_target;
	m_target = target;
}

GDifferentiable *GDifferentiableOptimizer::target()
{
	return m_target;
}

void GDifferentiableOptimizer::setObjective(GObjective *objective)
{
	delete m_objective;
	m_objective = objective;
}

GObjective *GDifferentiableOptimizer::objective()
{
	return m_objective;
}

void GDifferentiableOptimizer::optimizeIncremental(const GVec &feat, const GVec &lab)
{
	updateDeltas(feat, lab);
	applyDeltas();
}

void GDifferentiableOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, size_t start, size_t batchSize)
{
	for(size_t i = 0; i < batchSize; ++i)
		updateDeltas(features[start + i], labels[start + i]);
	scaleDeltas(1.0 / batchSize);
	applyDeltas();
}

void GDifferentiableOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, GRandomIndexIterator &ii, size_t batchSize)
{
	size_t j;
	for(size_t i = 0; i < batchSize; ++i)
	{
		if(!ii.next(j)) ii.reset(), ii.next(j);
		updateDeltas(features[j], labels[j]);
	}
	scaleDeltas(1.0 / batchSize);
	applyDeltas();
}

void GDifferentiableOptimizer::optimize(const GMatrix &features, const GMatrix &labels, size_t epochs, size_t batchesPerEpoch, size_t batchSize, GRand *rand)
{
	bool ownsRand = false;
	if(rand == NULL)
	{
		rand = new GRand(time(NULL));
		ownsRand = true;
	}
	
	GRandomIndexIterator ii(features.rows(), *rand);
	for(size_t i = 0; i < epochs; ++i)
		for(size_t j = 0; j < batchesPerEpoch; ++j)
			optimizeBatch(features, labels, ii, batchSize);
	
	if(ownsRand)
		delete rand;
}

GSGDOptimizer::GSGDOptimizer(GDifferentiable *target, GObjective *objective)
: GDifferentiableOptimizer(target, objective), m_learningRate(1e-3), m_momentum(0), m_ready(false)
{}

void GSGDOptimizer::beginOptimizing(size_t featSize, size_t labSize)
{
	m_pred.resize(labSize);
	m_blame.resize(labSize);
	m_gradient.resize(m_target->countParameters());
	m_deltas.resize(m_target->countParameters());
	m_gradient.fill(0.0);
	m_ready = true;
}

void GSGDOptimizer::updateDeltas(const GVec &feat, const GVec &lab)
{
	GAssert(m_ready, "GFunctionOptimizer::beginOptimizing must be called before attempting to optimize!");
	m_blame.fill(0.0);
	m_target->evaluate(feat, m_pred);
	m_objective->calculateDerivative(m_pred, lab, m_blame);
	m_target->updateDeltas(feat, m_blame, m_gradient);
}

void GSGDOptimizer::scaleDeltas(double scale)
{
	GAssert(m_ready, "GFunctionOptimizer::beginOptimizing must be called before attempting to optimize!");
	m_gradient *= scale;
}

void GSGDOptimizer::applyDeltas()
{
	GAssert(m_ready, "GFunctionOptimizer::beginOptimizing must be called before attempting to optimize!");
	m_deltas.fill(0.0);
	m_deltas.addScaled(m_learningRate, m_gradient);
	m_target->applyDeltas(m_deltas);
	m_gradient *= m_momentum;
}

GRMSPropOptimizer::GRMSPropOptimizer(GDifferentiable *target, GObjective *objective)
: GDifferentiableOptimizer(target, objective), m_learningRate(1e-3), m_momentum(0), m_gamma(0.9), m_epsilon(1e-6), m_ready(false)
{}

void GRMSPropOptimizer::beginOptimizing(size_t featSize, size_t labSize)
{
	m_pred.resize(labSize);
	m_blame.resize(labSize);
	m_gradient.resize(m_target->countParameters());
	m_deltas.resize(m_target->countParameters());
	m_meanSquare.resize(m_target->countParameters());
	m_gradient.fill(0.0);
	m_meanSquare.fill(0.0);
	m_ready = true;
}

void GRMSPropOptimizer::updateDeltas(const GVec &feat, const GVec &lab)
{
	GAssert(m_ready, "GFunctionOptimizer::beginOptimizing must be called before attempting to optimize!");
	m_blame.fill(0.0);
	m_target->evaluate(feat, m_pred);
	m_objective->calculateDerivative(m_pred, lab, m_blame);
	m_target->updateDeltas(feat, m_blame, m_gradient);
}

void GRMSPropOptimizer::scaleDeltas(double scale)
{
	GAssert(m_ready, "GFunctionOptimizer::beginOptimizing must be called before attempting to optimize!");
	m_gradient *= scale;
}

void GRMSPropOptimizer::applyDeltas()
{
	GAssert(m_ready, "GFunctionOptimizer::beginOptimizing must be called before attempting to optimize!");
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

