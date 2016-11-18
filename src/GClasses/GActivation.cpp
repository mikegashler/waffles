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

#include "GActivation.h"
#ifndef MIN_PREDICT
#include "GMath.h"
#endif // MIN_PREDICT
#include "GDom.h"
#include "GRand.h"
#include "GTime.h"
#include "GNeuralNet.h"
#include "GVec.h"

namespace GClasses {



/*
GActivationHinge::GActivationHinge()
: GActivationFunction(), m_units(0), m_error(0), m_hinges(0), m_delta(0), m_rates(0)
{
}

GActivationHinge::GActivationHinge(GDomNode* pNode)
: m_hinges(pNode->field("hinges"))
{
	m_units = m_hinges.size();
	m_error.resize(m_units);
	m_delta.resize(m_units);
	m_delta.fill(0.0);
	m_rates.resize(m_units);
	m_rates.fill(0.01);
}

// virtual
GDomNode* GActivationHinge::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "name", pDoc->newString(name()));
	pNode->addField(pDoc, "hinges", m_hinges.serialize(pDoc));
	return pNode;
}

// virtual
void GActivationHinge::resize(size_t units)
{
	m_units = units;
	m_error.resize(units);
	m_hinges.resize(units);
	m_hinges.fill(0.0);
	m_delta.resize(units);
	m_delta.fill(0.0);
	m_rates.resize(m_units);
	m_rates.fill(0.01);
}

// virtual
void GActivationHinge::setError(const GVec& error)
{
	m_error.copy(error);
}

// virtual
void GActivationHinge::updateDeltas(const GVec& net, const GVec& activation, double momentum)
{
	for(size_t i = 0; i < m_units; i++)
	{
		m_delta[i] *= momentum;
		m_delta[i] += m_error[i] * (sqrt(net[i] * net[i] + BEND_SIZE * BEND_SIZE) - BEND_SIZE);
	}
}

void GActivationHinge::updateDeltas(const GVec &net, const GVec &activation, GVec &deltas)
{
	for(size_t i = 0; i < m_units; ++i)
		deltas[i] += m_error[i] * (sqrt(net[i] * net[i] + BEND_SIZE * BEND_SIZE) - BEND_SIZE);
}

// virtual
void GActivationHinge::applyDeltas(double learningRate)
{
	double* pD = m_delta.data();
	double* pHinge = m_hinges.data();
	for(size_t i = 0; i < m_units; i++)
	{
		*pHinge = *pHinge + learningRate * *pD;
		if(*pHinge < -1.0)
		{
			*pHinge = -1.0;
			*pD = 0.0;
		}
		if(*pHinge > 1.0)
		{
			*pHinge = 1.0;
			*pD = 0.0;
		}
		pHinge++;
		pD++;
	}
}

void GActivationHinge::applyDeltas(double learningRate, const GVec &deltas)
{
	const double* pD = deltas.data();
	double* pHinge = m_hinges.data();
	for(size_t i = 0; i < m_units; i++)
	{
		*pHinge = *pHinge + learningRate * *pD;
		if(*pHinge < -1.0)
			*pHinge = -1.0;
		if(*pHinge > 1.0)
			*pHinge = 1.0;
		pHinge++;
		pD++;
	}
}

// virtual
void GActivationHinge::applyAdaptive()
{
	// Adapt the learning rates
	double* pD = m_delta.data();
	double* pR = m_rates.data();
	for(size_t i = 0; i < m_units; i++)
	{
		if(std::signbit(*pD) == std::signbit(*pR))
		{
			if(std::abs(*pR) < 1e3)
				(*pR) *= 1.2;
		}
		else
		{
			if(std::abs(*pR) > 1e-8)
				(*pR) *= -0.2;
			else
				(*pR) *= -1.1;
		}
		pD++;
		pR++;
	}

	// Update the parameters
	pR = m_rates.data();
	double* pHinge = m_hinges.data();
	for(size_t i = 0; i < m_units; i++)
	{
		*pHinge = *pHinge + *pR;
		*pHinge = std::max(-1.0, std::min(1.0, *pHinge));
		pHinge++;
		pR++;
	}
}

// virtual
void GActivationHinge::regularize(double lambda)
{
	double* pHinge = m_hinges.data();
	for(size_t i = 0; i < m_units; i++)
	{
		if(*pHinge >= 0.0)
			*pHinge = std::max(0.0, *pHinge - lambda);
		else
			*pHinge = std::min(0.0, *pHinge + lambda);
	}
}

// virtual
GActivationFunction* GActivationHinge::clone()
{
	GActivationHinge* pClone = new GActivationHinge();
	pClone->resize(m_units);
	pClone->m_hinges.copy(m_hinges);
	pClone->m_rates.copy(m_rates);
	return pClone;
}

// virtual
size_t GActivationHinge::countWeights()
{
	return m_units;
}

// virtual
size_t GActivationHinge::weightsToVector(double* pOutVector)
{
	memcpy(pOutVector, m_hinges.data(), sizeof(double) * m_units);
	return m_units;
}

// virtual
size_t GActivationHinge::vectorToWeights(const double* pVector)
{
	m_hinges.set(pVector, m_units);
	return m_units;
}

// virtual
void GActivationHinge::copyWeights(const GActivationFunction* pOther)
{
	m_hinges.copy(((GActivationHinge*)pOther)->m_hinges);
}

#ifndef MIN_PREDICT
// static
void GActivationHinge::test()
{
	// Make a neural network
	GNeuralNet nn;
	GLayerLinear* pLay1 = new GLayerLinear(2, 3);
	GActivationHinge* pAct1 = new GActivationHinge(); pAct1->resize(3);
	GLayerLinear* pLay2 = new GLayerLinear(3, 2);
	GActivationHinge* pAct2 = new GActivationHinge(); pAct2->resize(2);
	GLayerLinear* pLay3 = new GLayerLinear(2, 2);
	GActivationHinge* pAct3 = new GActivationHinge(); pAct3->resize(3);
	nn.addLayers(pLay1, pAct1, pLay2, pAct2, pLay3, pAct3);
	
	GUniformRelation rel(2);
	
	GSGDOptimizer optimizer(nn);
	optimizer.setLearningRate(0.1);
	optimizer.setMomentum(0.0);
	nn.beginIncrementalLearning(rel, rel);
	
	pLay1->perturbWeights(nn.rand(), 0.03);
	pLay2->perturbWeights(nn.rand(), 0.1);
	pLay3->perturbWeights(nn.rand(), 0.3);
	GVec::perturb(pAct1->alphas().data(), 0.1, pLay1->outputs(), nn.rand());
	GVec::perturb(pAct2->alphas().data(), 0.1, pLay2->outputs(), nn.rand());
	GVec::perturb(pAct3->alphas().data(), 0.1, pLay3->outputs(), nn.rand());
	GVec in(2);
	GVec out(2);
	in.fillNormal(nn.rand());
	in.normalize();
	out.fillNormal(nn.rand());
	out.normalize();

	// Measure baseline error
	nn.forwardProp(in);
	double errBase = out.squaredDistance(nn.outputLayer().activation());
	double epsilon = 1e-6;

	// Empirically measure gradient of a weight
	double beforeWeight = pLay2->weights()[1][1];
	pLay2->weights()[1][1] += epsilon;
	nn.forwardProp(in);
	double errWeight = out.squaredDistance(nn.outputLayer().activation());
	pLay2->weights()[1][1] = beforeWeight;
	
	// Empirically measure gradient of a bias
	double beforeBias = pLay2->bias()[1];
	pLay2->bias()[1] += epsilon;
	nn.forwardProp(in);
	double errBias = out.squaredDistance(nn.outputLayer().activation());
	pLay2->bias()[1] = beforeBias;

	// Empirically measure gradient of an alpha
	double beforeAlpha = pAct2->alphas()[1];
	pAct2->alphas()[1] += epsilon;
	nn.forwardProp(in);
	//double errAlpha = out.squaredDistance(nn.outputLayer().activation());
	pAct2->alphas()[1] = beforeAlpha;

	// Update the weights by gradient descent
	optimizer.optimizeIncremental(in, out);

	// Check the result
	double empiricalGradientWeight = (errWeight - errBase) / epsilon;
	double computedGradientWeight = -2.0 * (pLay2->weights()[1][1] - beforeWeight) / optimizer.learningRate();
	double empiricalGradientBias = (errBias - errBase) / epsilon;
	double computedGradientBias = -2.0 * (pLay2->bias()[1] - beforeBias) / optimizer.learningRate();
	//double empiricalGradientAlpha = (errAlpha - errBase) / epsilon;
	//double computedGradientAlpha = -2.0 * (pAct2->alphas()[1] - beforeAlpha) / optimizer.learningRate();
	if(std::abs(empiricalGradientWeight - computedGradientWeight) > 1e-5)
		throw Ex("failed");
	if(std::abs(empiricalGradientBias - computedGradientBias) > 1e-5)
		throw Ex("failed");
	//if(std::abs(empiricalGradientAlpha - computedGradientAlpha) > 1e-5)
	//	throw Ex("failed");
}
#endif







GActivationSoftExponential::GActivationSoftExponential()
: GActivationFunction(), m_units(0), m_error(0), m_alphas(0), m_delta(0), m_rates(0)
{
}

GActivationSoftExponential::GActivationSoftExponential(GDomNode* pNode)
: m_alphas(pNode->field("alphas"))
{
	m_units = m_alphas.size();
	m_error.resize(m_units);
	m_delta.resize(m_units);
	m_delta.fill(0.0);
	m_rates.resize(m_units);
	m_rates.fill(0.01);
}

// virtual
GDomNode* GActivationSoftExponential::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "name", pDoc->newString(name()));
	pNode->addField(pDoc, "alphas", m_alphas.serialize(pDoc));
	return pNode;
}

// virtual
void GActivationSoftExponential::resize(size_t units)
{
	m_units = units;
	m_error.resize(units);
	m_alphas.resize(units);
	m_alphas.fill(0.0);
	m_delta.resize(units);
	m_delta.fill(0.0);
	m_rates.resize(m_units);
	m_rates.fill(0.01);
}

// virtual
void GActivationSoftExponential::setError(const GVec& error)
{
	m_error.copy(error);
}

// virtual
void GActivationSoftExponential::updateDeltas(const GVec& net, const GVec& activation, double momentum)
{
	double* pAlpha = m_alphas.data();
	const double* pErr = m_error.data();
	const double* pN = net.data();
	const double* pAct = activation.data();
	double* pD = m_delta.data();
	for(size_t i = 0; i < m_units; i++)
	{
		*pD *= momentum;
		double t1 = (*pAlpha * *pAlpha);
		double t2 = (*pAlpha * *pN);
		double delta;
		if(*pAlpha < 1e-8)
			delta = (log(std::max(1e-12, 1.0 - (t1 + t2))) - (t1 + t1 + t2) / (t1 + t2 - 1.0)) / t1;
		else if(*pAlpha > 1e-8)
			delta = (t1 + (t2 - 1.0) * exp(std::min(300.0, t2)) + 1.0) / t1;
		else
			delta = 0.5 * (*pN) * (*pN) + 1.0;
		*pD += *pErr * delta;
		pN++;
		pAct++;
		pErr++;
		pAlpha++;
		pD++;
	}
}

void GActivationSoftExponential::updateDeltas(const GVec& net, const GVec& activation, GVec &deltas)
{
	double* pAlpha = m_alphas.data();
	const double* pErr = m_error.data();
	const double* pN = net.data();
	const double* pAct = activation.data();
	double* pD = deltas.data();
	for(size_t i = 0; i < m_units; i++)
	{
		double t1 = (*pAlpha * *pAlpha);
		double t2 = (*pAlpha * *pN);
		double delta;
		if(*pAlpha < 1e-8)
			delta = (log(std::max(1e-12, 1.0 - (t1 + t2))) - (t1 + t1 + t2) / (t1 + t2 - 1.0)) / t1;
		else if(*pAlpha > 1e-8)
			delta = (t1 + (t2 - 1.0) * exp(std::min(300.0, t2)) + 1.0) / t1;
		else
			delta = 0.5 * (*pN) * (*pN) + 1.0;
		*pD += *pErr * delta;
		pN++;
		pAct++;
		pErr++;
		pAlpha++;
		pD++;
	}
}

// virtual
void GActivationSoftExponential::applyDeltas(double learningRate)
{
	double* pD = m_delta.data();
	double* pAlpha = m_alphas.data();
	for(size_t i = 0; i < m_units; i++)
	{
		*pAlpha = *pAlpha + learningRate * *pD;
		if(*pAlpha < -1.0)
		{
			*pAlpha = -1.0;
			*pD = 0.0;
		}
		if(*pAlpha > 1.0)
		{
			*pAlpha = 1.0;
			*pD = 0.0;
		}
		pAlpha++;
		pD++;
	}
}

void GActivationSoftExponential::applyDeltas(double learningRate, const GVec &deltas)
{
	const double* pD = deltas.data();
	double* pAlpha = m_alphas.data();
	for(size_t i = 0; i < m_units; i++)
	{
		*pAlpha = *pAlpha + learningRate * *pD;
		if(*pAlpha < -1.0)
			*pAlpha = -1.0;
		if(*pAlpha > 1.0)
			*pAlpha = 1.0;
		pAlpha++;
		pD++;
	}
}

// virtual
void GActivationSoftExponential::applyAdaptive()
{
	// Adapt the learning rates
	double* pD = m_delta.data();
	double* pR = m_rates.data();
	for(size_t i = 0; i < m_units; i++)
	{
		if(std::signbit(*pD) == std::signbit(*pR))
		{
			if(std::abs(*pR) < 1e3)
				(*pR) *= 1.2;
		}
		else
		{
			if(std::abs(*pR) > 1e-8)
				(*pR) *= -0.2;
			else
				(*pR) *= -1.1;
		}
		pD++;
		pR++;
	}

	// Update the parameters
	pR = m_rates.data();
	double* pAlpha = m_alphas.data();
	for(size_t i = 0; i < m_units; i++)
	{
		*pAlpha = *pAlpha + *pR;
		*pAlpha = std::max(-1.0, std::min(1.0, *pAlpha));
		pAlpha++;
		pR++;
	}
}

// virtual
void GActivationSoftExponential::regularize(double lambda)
{
	double* pAlpha = m_alphas.data();
	for(size_t i = 0; i < m_units; i++)
	{
		if(*pAlpha >= 0.0)
			*pAlpha = std::max(0.0, *pAlpha - lambda);
		else
			*pAlpha = std::min(0.0, *pAlpha + lambda);
	}
}

// virtual
GActivationFunction* GActivationSoftExponential::clone()
{
	GActivationSoftExponential* pClone = new GActivationSoftExponential();
	pClone->resize(m_units);
	pClone->m_alphas.copy(m_alphas);
	pClone->m_rates.copy(m_rates);
	return pClone;
}

// virtual
size_t GActivationSoftExponential::countWeights()
{
	return m_units;
}

// virtual
size_t GActivationSoftExponential::weightsToVector(double* pOutVector)
{
	memcpy(pOutVector, m_alphas.data(), sizeof(double) * m_units);
	return m_units;
}

// virtual
size_t GActivationSoftExponential::vectorToWeights(const double* pVector)
{
	m_alphas.set(pVector, m_units);
	return m_units;
}

// virtual
void GActivationSoftExponential::copyWeights(const GActivationFunction* pOther)
{
	m_alphas.copy(((GActivationSoftExponential*)pOther)->m_alphas);
}

#ifndef MIN_PREDICT
// static
void GActivationSoftExponential::test()
{
	// Make a neural network
	GNeuralNet nn;
	GLayerLinear* pLay1 = new GLayerLinear(2, 3);
	GActivationSoftExponential* pAct1 = new GActivationSoftExponential(); pAct1->resize(3);
	GLayerLinear* pLay2 = new GLayerLinear(3, 2);
	GActivationSoftExponential* pAct2 = new GActivationSoftExponential(); pAct2->resize(2);
	GLayerLinear* pLay3 = new GLayerLinear(2, 2);
	GActivationSoftExponential* pAct3 = new GActivationSoftExponential(); pAct3->resize(2);
	nn.addLayers(pLay1, pAct1, pLay2, pAct2, pLay3, pAct3);
	
	GUniformRelation rel(2);
	
	GSGDOptimizer optimizer(nn);
	optimizer.setLearningRate(0.1);
	optimizer.setMomentum(0.0);
	nn.beginIncrementalLearning(rel, rel);
	
	nn.beginIncrementalLearning(rel, rel);
	
	pLay1->perturbWeights(nn.rand(), 0.03);
	pLay2->perturbWeights(nn.rand(), 0.1);
	pLay3->perturbWeights(nn.rand(), 0.3);
	GVec::perturb(pAct1->alphas().data(), 0.1, pLay1->outputs(), nn.rand());
	GVec::perturb(pAct2->alphas().data(), 0.1, pLay2->outputs(), nn.rand());
	GVec::perturb(pAct3->alphas().data(), 0.1, pLay3->outputs(), nn.rand());
	GVec in(2);
	GVec out(2);
	in.fillNormal(nn.rand());
	in.normalize();
	out.fillNormal(nn.rand());
	out.normalize();

	// Measure baseline error
	nn.forwardProp(in);
	double errBase = out.squaredDistance(nn.outputLayer().activation());
	double epsilon = 1e-6;

	// Empirically measure gradient of a weight
	double beforeWeight = pLay2->weights()[1][1];
	pLay2->weights()[1][1] += epsilon;
	nn.forwardProp(in);
	double errWeight = out.squaredDistance(nn.outputLayer().activation());
	pLay2->weights()[1][1] = beforeWeight;
	
	// Empirically measure gradient of a bias
	double beforeBias = pLay2->bias()[1];
	pLay2->bias()[1] += epsilon;
	nn.forwardProp(in);
	double errBias = out.squaredDistance(nn.outputLayer().activation());
	pLay2->bias()[1] = beforeBias;

	// Empirically measure gradient of an alpha
	double beforeAlpha = pAct2->alphas()[1];
	pAct2->alphas()[1] += epsilon;
	nn.forwardProp(in);
	//double errAlpha = out.squaredDistance(nn.outputLayer().activation());
	pAct2->alphas()[1] = beforeAlpha;

	// Update the weights by gradient descent
	optimizer.optimizeIncremental(in, out);

	// Check the result
	double empiricalGradientWeight = (errWeight - errBase) / epsilon;
	double computedGradientWeight = -2.0 * (pLay2->weights()[1][1] - beforeWeight) / optimizer.learningRate();
	double empiricalGradientBias = (errBias - errBase) / epsilon;
	double computedGradientBias = -2.0 * (pLay2->bias()[1] - beforeBias) / optimizer.learningRate();
	//double empiricalGradientAlpha = (errAlpha - errBase) / epsilon;
	//double computedGradientAlpha = -2.0 * (pAct2->alphas()[1] - beforeAlpha) / optimizer.learningRate();
	if(std::abs(empiricalGradientWeight - computedGradientWeight) > epsilon)
		throw Ex("GActivation::test failed; weight gradient incorrect");
	if(std::abs(empiricalGradientBias - computedGradientBias) > epsilon)
		throw Ex("GActivation::test failed; bias gradient incorrect");
	//if(std::abs(empiricalGradientAlpha - computedGradientAlpha) > epsilon)
	//	throw Ex("GActivation::test failed; alpha gradient incorrect; expected " + to_str(empiricalGradientAlpha) + ", got " + to_str(computedGradientAlpha));
}
#endif
*/
} // namespace GClasses

