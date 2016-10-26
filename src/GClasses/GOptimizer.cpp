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

void GSumSquaredErrorFunction::calculateOutput(const GVec &x, GVec &y)
{
	for(size_t i = 0; i < x.size(); ++i)
		y[i] = x[i] * x[i];
}

void GSumSquaredErrorFunction::updateGradient(const GVec &x, const GVec &err, GVec &gradient)
{
	for(size_t i = 0; i < x.size(); ++i)
		gradient[i] += err[i];
}

void GNeuralNetFunction::calculateOutput(const GVec &x, GVec &y)
{
	m_nn.predict(x, y);
}
void GNeuralNetFunction::updateGradient(const GVec &x, const GVec &err, GVec &dy)
{
	m_nn.outputLayer().error().put(0, err);
	m_nn.backpropagateErrorAlreadySet();
	const GVec *in = &x;
	GVecWrapper out(dy.data(), 0);
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

GFunctionOptimizer::GFunctionOptimizer(GOptimizableFunction *function, GDifferentiableFunction *error)
: m_function(function), m_error(error != NULL ? error : new GSumSquaredErrorFunction())
{}

GFunctionOptimizer::~GFunctionOptimizer()
{
	delete m_function;
	delete m_error;
}

void GFunctionOptimizer::optimizeIncremental(const GVec &feat, const GVec &lab)
{
	updateGradient(feat, lab);
	applyGradient();
}

void GFunctionOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, size_t start, size_t batchSize)
{
	for(size_t i = 0; i < batchSize; ++i)
		updateGradient(features[start + i], labels[start + i]);
	scaleGradient(1.0 / batchSize);
	applyGradient();
}

void GFunctionOptimizer::optimizeBatch(const GMatrix &features, const GMatrix &labels, GRandomIndexIterator &ii, size_t batchSize)
{
	size_t j;
	for(size_t i = 0; i < batchSize; ++i)
	{
		if(!ii.next(j)) ii.reset(), ii.next(j);
		updateGradient(features[j], labels[j]);
	}
	scaleGradient(1.0 / batchSize);
	applyGradient();
}

void GFunctionOptimizer::optimize(const GMatrix &features, const GMatrix &labels, size_t epochs, size_t batchesPerEpoch, size_t batchSize, GRand *rand)
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

GSGDOptimizer::GSGDOptimizer(GOptimizableFunction *function, GDifferentiableFunction *error)
: GFunctionOptimizer(function, error), m_learningRate(1e-3), m_momentum(0)
{}

void GSGDOptimizer::beginOptimizing(size_t featSize, size_t labSize)
{
	m_pred.resize(labSize);
	m_blame.resize(labSize);
	m_gradient.resize(m_function->countParameters());
	m_deltas.resize(m_function->countParameters());
	m_gradient.fill(0.0);
}

void GSGDOptimizer::updateGradient(const GVec &feat, const GVec &lab)
{
	m_blame.fill(0.0);
	m_function->calculateOutput(feat, m_pred);
	m_error->updateGradient(m_pred, lab - m_pred, m_blame);
	m_function->updateGradient(feat, m_blame, m_gradient);
}

void GSGDOptimizer::scaleGradient(double scale)
{
	m_gradient *= scale;
}

void GSGDOptimizer::applyGradient()
{
	m_deltas.fill(0.0);
	m_deltas.addScaled(m_learningRate, m_gradient);
	m_function->applyDeltas(m_deltas);
	m_gradient *= m_momentum;
}

GRMSPropOptimizer::GRMSPropOptimizer(GOptimizableFunction *function, GDifferentiableFunction *error)
: GFunctionOptimizer(function, error), m_learningRate(1e-3), m_momentum(0), m_gamma(0.9), m_epsilon(1e-6)
{}

void GRMSPropOptimizer::beginOptimizing(size_t featSize, size_t labSize)
{
	m_pred.resize(labSize);
	m_blame.resize(labSize);
	m_gradient.resize(m_function->countParameters());
	m_deltas.resize(m_function->countParameters());
	m_meanSquare.resize(m_function->countParameters());
	m_gradient.fill(0.0);
	m_meanSquare.fill(0.0);
}

void GRMSPropOptimizer::updateGradient(const GVec &feat, const GVec &lab)
{
	m_blame.fill(0.0);
	m_function->calculateOutput(feat, m_pred);
	m_error->updateGradient(m_pred, lab - m_pred, m_blame);
	m_function->updateGradient(feat, m_blame, m_gradient);
}

void GRMSPropOptimizer::scaleGradient(double scale)
{
	m_gradient *= scale;
}

void GRMSPropOptimizer::applyGradient()
{
	for(size_t i = 0; i < m_meanSquare.size(); ++i)
	{
		m_meanSquare[i] *= m_gamma;
		m_meanSquare[i] += (1.0 - m_gamma) * m_gradient[i] * m_gradient[i];
		m_gradient[i] /= sqrt(m_meanSquare[i]) + m_epsilon;
	}
	
	m_deltas.fill(0.0);
	m_deltas.addScaled(m_learningRate, m_gradient);
	m_function->applyDeltas(m_deltas);
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

