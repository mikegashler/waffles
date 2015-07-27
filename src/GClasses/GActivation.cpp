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

// virtual
GDomNode* GActivationFunction::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "name", pDoc->newString(name()));
	return pNode;
}

// static
GActivationFunction* GActivationFunction::deserialize(GDomNode* pNode)
{
	const char* szName = pNode->field("name")->asString();
	if(*szName < 'm')
	{
		if(strcmp(szName, "logistic") == 0)
			return new GActivationLogistic();
		else if(strcmp(szName, "identity") == 0)
			return new GActivationIdentity();
		else if(strcmp(szName, "arctan") == 0)
			return new GActivationArcTan();
		else if(strcmp(szName, "algebraic") == 0)
			return new GActivationAlgebraic();
		else if(strcmp(szName, "gaussian") == 0)
			return new GActivationGaussian();
		else if(strcmp(szName, "bidir") == 0)
			return new GActivationBiDir();
		else if(strcmp(szName, "bend") == 0)
			return new GActivationBend();
		else if(strcmp(szName, "hinge") == 0)
			return new GActivationHinge(pNode);
		else if(strcmp(szName, "logexp") == 0)
			return new GActivationLogExp(pNode);
		else if(strcmp(szName, "logisticderiv") == 0)
			return new GActivationLogisticDerivative();
		else
			throw Ex("Unrecognized activation function: ", szName);
	}
	else
	{
		if(strcmp(szName, "tanh") == 0)
			return new GActivationTanH();
		else if(strcmp(szName, "relu") == 0)
			return new GActivationRectifiedLinear();
		else if(strcmp(szName, "softplus") == 0)
			return new GActivationSoftPlus();
		else if(strcmp(szName, "sin") == 0)
			return new GActivationSin();
		else if(strcmp(szName, "sinc") == 0)
			return new GActivationSinc();
		else if(strcmp(szName, "softplus2") == 0)
			return new GActivationSoftPlus2();
		else
			throw Ex("Unrecognized activation function: ", szName);
	}
	return NULL;
}
/*
double GActivationFunction::measureWeightScale(size_t width, size_t depth, size_t seed)
{
	GUniformRelation rel(width);
	GNeuralNet nn;
	nn.rand().setSeed(seed);
	for(size_t i = 0; i < depth; i++)
		nn.addLayer(new GLayerClassic(width, width, clone()));
	GLayerClassic scratch(0, width, clone());
	nn.beginIncrementalLearning(rel, rel);
	GRand& rand = nn.rand();
	double step = 0.5;
	double scale = 1.0;
	double recipWid = 1.0 / sqrt((double)width);
for(scale = 0.95; scale < 1.05; scale += 0.01)
//	for(size_t iters = 0; iters < 1000; iters++)
	{
double t0 = 0.0;
double t1 = 0.0;
for(size_t q = 0; q < 400; q++)
{
		// Re-initialize the weights with the candidate scale
		for(size_t i = 0; i < depth; i++)
		{
			GLayerClassic* pLayer = (GLayerClassic*)&nn.layer(i);
			GVec::setAll(pLayer->bias(), 0.0, width);
			GMatrix& w = pLayer->weights();
			for(size_t j = 0; j < width; j++)
			{
				double* pRow = w[j];
rand.spherical(pRow, width);
GVec::multiply(pRow, scale, width);
//				for(size_t k = 0; k < width; k++)
//					*(pRow++) = scale * recipWid * rand.normal();
			}
		}

		// Feed a in a random vector and target a random vector
		double* pScratch = scratch.activation();
		rand.spherical(pScratch, width);
		nn.forwardProp(pScratch);
double mag0 = GVec::squaredMagnitude(nn.outputLayer().activation(), width);
t0 += sqrt(mag0);
		rand.spherical(pScratch, width);
		size_t i = nn.layerCount() - 1;
		GNeuralNetLayer* pLay = &nn.layer(i);
		pLay->computeError(pScratch);
		pLay->deactivateError();
		GVec::normalize(pLay->error(), width);
		while(i > 0)
		{
			GNeuralNetLayer* pUpStream = &nn.layer(i - 1);
			pLay->backPropError(pUpStream);
			pUpStream->deactivateError();
			pLay = pUpStream;
			i--;
		}

		// Adjust the scale to make the magnitude of the error on layer 1 approach 1
		double mag = GVec::squaredMagnitude(nn.layer(0).error(), width);
//		if(mag < 1.0)
//			scale += step;
//		else
//			scale -= step;
//		step *= 0.9863;

t1 += sqrt(mag);
}

std::cout << to_str(scale) << "," << to_str(log(t0 / 400.0) * M_LOG10E) << "," << to_str(log(t1 / 400.0) * M_LOG10E) << "\n";
	}
	return scale;
}
*/





GActivationHinge::GActivationHinge()
: GActivationFunction(), m_units(0), m_error(0), m_hinges(0), m_delta(0)
{
}

GActivationHinge::GActivationHinge(GDomNode* pNode)
{
	GDomListIterator it(pNode->field("hinges"));
	m_units = it.remaining();
	m_error.resize(m_units);
	m_hinges.resize(m_units);
	GVec::deserialize(m_hinges.v, it);
	m_delta.resize(m_units);
	GVec::setAll(m_delta.v, 0.0, m_units);
}

// virtual
GDomNode* GActivationHinge::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "name", pDoc->newString(name()));
	pNode->addField(pDoc, "hinges", GVec::serialize(pDoc, m_hinges.v, m_units));
	return pNode;
}

// virtual
void GActivationHinge::resize(size_t units)
{
	m_units = units;
	m_error.resize(units);
	m_hinges.resize(units);
	GVec::setAll(m_hinges.v, 0.5, units);
	m_delta.resize(units);
	GVec::setAll(m_delta.v, 0.0, units);
}

// virtual
void GActivationHinge::setError(const double* pError)
{
	GVec::copy(m_error.v, pError, m_units);
}

// virtual
void GActivationHinge::updateDeltas(const double* pNet, const double* pActivation, double momentum)
{
	const double* pErr = m_error.v;
	const double* pN = pNet;
	double* pD = m_delta.v;
	for(size_t i = 0; i < m_units; i++)
	{
		*pD *= momentum;
		*pD += (*pErr) * (sqrt(*pN * *pN + BEND_SIZE * BEND_SIZE) - BEND_SIZE);
		pN++;
		pErr++;
		pD++;
	}
}

// virtual
void GActivationHinge::applyDeltas(double learningRate)
{
	double* pD = m_delta.v;
	double* pHinge = m_hinges.v;
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

// virtual
void GActivationHinge::regularize(double lambda)
{
	double* pHinge = m_hinges.v;
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
	GVec::copy(pClone->m_hinges.v, m_hinges.v, m_units);
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
	GVec::copy(pOutVector, m_hinges.v, m_units);
	return m_units;
}

// virtual
size_t GActivationHinge::vectorToWeights(const double* pVector)
{
	GVec::copy(m_hinges.v, pVector, m_units);
	return m_units;
}

// virtual
void GActivationHinge::copyWeights(const GActivationFunction* pOther)
{
	GVec::copy(m_hinges.v, ((GActivationHinge*)pOther)->m_hinges.v, m_units);
}

#ifndef MIN_PREDICT
// static
void GActivationHinge::test()
{
	// Make a neural network
	GNeuralNet nn;
	GActivationHinge* pAct1 = new GActivationHinge();
	GLayerClassic* pLay1 = new GLayerClassic(2, 3, pAct1);
	GActivationHinge* pAct2 = new GActivationHinge();
	GLayerClassic* pLay2 = new GLayerClassic(3, 2, pAct2);
	GActivationHinge* pAct3 = new GActivationHinge();
	GLayerClassic* pLay3 = new GLayerClassic(2, 2, pAct3);
	nn.addLayer(pLay1);
	nn.addLayer(pLay2);
	nn.addLayer(pLay3);
	nn.setLearningRate(0.1);
	nn.setMomentum(0.0);
	GUniformRelation rel(2);
	nn.beginIncrementalLearning(rel, rel);
	pLay1->perturbWeights(nn.rand(), 0.03);
	pLay2->perturbWeights(nn.rand(), 0.1);
	pLay3->perturbWeights(nn.rand(), 0.3);
	GVec::perturb(pAct1->alphas(), 0.1, pLay1->outputs(), nn.rand());
	GVec::perturb(pAct2->alphas(), 0.1, pLay2->outputs(), nn.rand());
	GVec::perturb(pAct3->alphas(), 0.1, pLay3->outputs(), nn.rand());
	double in[2];
	double out[2];
	nn.rand().spherical(in, 2);
	nn.rand().spherical(out, 2);

	// Measure baseline error
	nn.forwardProp(in);
	double errBase = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	double epsilon = 1e-6;

	// Empirically measure gradient of a weight
	double beforeWeight = pLay2->weights()[1][1];
	pLay2->weights()[1][1] += epsilon;
	nn.forwardProp(in);
	double errWeight = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	pLay2->weights()[1][1] = beforeWeight;
	
	// Empirically measure gradient of a bias
	double beforeBias = pLay2->bias()[1];
	pLay2->bias()[1] += epsilon;
	nn.forwardProp(in);
	double errBias = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	pLay2->bias()[1] = beforeBias;

	// Empirically measure gradient of an alpha
	double beforeAlpha = pAct2->alphas()[1];
	pAct2->alphas()[1] += epsilon;
	nn.forwardProp(in);
	double errAlpha = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	pAct2->alphas()[1] = beforeAlpha;

	// Update the weights by gradient descent
	nn.trainIncremental(in, out);

	// Check the result
	double empiricalGradientWeight = (errWeight - errBase) / epsilon;
	double computedGradientWeight = -2.0 * (pLay2->weights()[1][1] - beforeWeight) / nn.learningRate();
	double empiricalGradientBias = (errBias - errBase) / epsilon;
	double computedGradientBias = -2.0 * (pLay2->bias()[1] - beforeBias) / nn.learningRate();
	double empiricalGradientAlpha = (errAlpha - errBase) / epsilon;
	double computedGradientAlpha = -2.0 * (pAct2->alphas()[1] - beforeAlpha) / nn.learningRate();
	if(std::abs(empiricalGradientWeight - computedGradientWeight) > 1e-5)
		throw Ex("failed");
	if(std::abs(empiricalGradientBias - computedGradientBias) > 1e-5)
		throw Ex("failed");
	if(std::abs(empiricalGradientAlpha - computedGradientAlpha) > 1e-5)
		throw Ex("failed");
}
#endif







GActivationLogExp::GActivationLogExp()
: GActivationFunction(), m_units(0), m_error(0), m_alphas(0), m_delta(0)
{
}

GActivationLogExp::GActivationLogExp(GDomNode* pNode)
{
	GDomListIterator it(pNode->field("alphas"));
	m_units = it.remaining();
	m_error.resize(m_units);
	m_alphas.resize(m_units);
	GVec::deserialize(m_alphas.v, it);
	m_delta.resize(m_units);
	GVec::setAll(m_delta.v, 0.0, m_units);
}

// virtual
GDomNode* GActivationLogExp::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "name", pDoc->newString(name()));
	pNode->addField(pDoc, "alphas", GVec::serialize(pDoc, m_alphas.v, m_units));
	return pNode;
}

// virtual
void GActivationLogExp::resize(size_t units)
{
	m_units = units;
	m_error.resize(units);
	m_alphas.resize(units);
	GVec::setAll(m_alphas.v, 0.0, units);
	m_delta.resize(units);
	GVec::setAll(m_delta.v, 0.0, units);
}

// virtual
void GActivationLogExp::setError(const double* pError)
{
	GVec::copy(m_error.v, pError, m_units);
}

// virtual
void GActivationLogExp::updateDeltas(const double* pNet, const double* pActivation, double momentum)
{
	double* pAlpha = m_alphas.v;
	const double* pErr = m_error.v;
	const double* pN = pNet;
	const double* pAct = pActivation;
	double* pD = m_delta.v;
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

// virtual
void GActivationLogExp::applyDeltas(double learningRate)
{
	double* pD = m_delta.v;
	double* pAlpha = m_alphas.v;
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

// virtual
void GActivationLogExp::regularize(double lambda)
{
	double* pAlpha = m_alphas.v;
	for(size_t i = 0; i < m_units; i++)
	{
		if(*pAlpha >= 0.0)
			*pAlpha = std::max(0.0, *pAlpha - lambda);
		else
			*pAlpha = std::min(0.0, *pAlpha + lambda);
	}
}

// virtual
GActivationFunction* GActivationLogExp::clone()
{
	GActivationLogExp* pClone = new GActivationLogExp();
	pClone->resize(m_units);
	GVec::copy(pClone->m_alphas.v, m_alphas.v, m_units);
	return pClone;
}

// virtual
size_t GActivationLogExp::countWeights()
{
	return m_units;
}

// virtual
size_t GActivationLogExp::weightsToVector(double* pOutVector)
{
	GVec::copy(pOutVector, m_alphas.v, m_units);
	return m_units;
}

// virtual
size_t GActivationLogExp::vectorToWeights(const double* pVector)
{
	GVec::copy(m_alphas.v, pVector, m_units);
	return m_units;
}

// virtual
void GActivationLogExp::copyWeights(const GActivationFunction* pOther)
{
	GVec::copy(m_alphas.v, ((GActivationLogExp*)pOther)->m_alphas.v, m_units);
}

#ifndef MIN_PREDICT
// static
void GActivationLogExp::test()
{
	// Make a neural network
	GNeuralNet nn;
	GActivationLogExp* pAct1 = new GActivationLogExp();
	GLayerClassic* pLay1 = new GLayerClassic(2, 3, pAct1);
	GActivationLogExp* pAct2 = new GActivationLogExp();
	GLayerClassic* pLay2 = new GLayerClassic(3, 2, pAct2);
	GActivationLogExp* pAct3 = new GActivationLogExp();
	GLayerClassic* pLay3 = new GLayerClassic(2, 2, pAct3);
	nn.addLayer(pLay1);
	nn.addLayer(pLay2);
	nn.addLayer(pLay3);
	nn.setLearningRate(0.1);
	nn.setMomentum(0.0);
	GUniformRelation rel(2);
	nn.beginIncrementalLearning(rel, rel);
	pLay1->perturbWeights(nn.rand(), 0.03);
	pLay2->perturbWeights(nn.rand(), 0.1);
	pLay3->perturbWeights(nn.rand(), 0.3);
	GVec::perturb(pAct1->alphas(), 0.1, pLay1->outputs(), nn.rand());
	GVec::perturb(pAct2->alphas(), 0.1, pLay2->outputs(), nn.rand());
	GVec::perturb(pAct3->alphas(), 0.1, pLay3->outputs(), nn.rand());
	double in[2];
	double out[2];
	nn.rand().spherical(in, 2);
	nn.rand().spherical(out, 2);

	// Measure baseline error
	nn.forwardProp(in);
	double errBase = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	double epsilon = 1e-6;

	// Empirically measure gradient of a weight
	double beforeWeight = pLay2->weights()[1][1];
	pLay2->weights()[1][1] += epsilon;
	nn.forwardProp(in);
	double errWeight = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	pLay2->weights()[1][1] = beforeWeight;
	
	// Empirically measure gradient of a bias
	double beforeBias = pLay2->bias()[1];
	pLay2->bias()[1] += epsilon;
	nn.forwardProp(in);
	double errBias = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	pLay2->bias()[1] = beforeBias;

	// Empirically measure gradient of an alpha
	double beforeAlpha = pAct2->alphas()[1];
	pAct2->alphas()[1] += epsilon;
	nn.forwardProp(in);
	double errAlpha = GVec::squaredDistance(nn.outputLayer().activation(), out, 2);
	pAct2->alphas()[1] = beforeAlpha;

	// Update the weights by gradient descent
	nn.trainIncremental(in, out);

	// Check the result
	double empiricalGradientWeight = (errWeight - errBase) / epsilon;
	double computedGradientWeight = -2.0 * (pLay2->weights()[1][1] - beforeWeight) / nn.learningRate();
	double empiricalGradientBias = (errBias - errBase) / epsilon;
	double computedGradientBias = -2.0 * (pLay2->bias()[1] - beforeBias) / nn.learningRate();
	double empiricalGradientAlpha = (errAlpha - errBase) / epsilon;
	double computedGradientAlpha = -2.0 * (pAct2->alphas()[1] - beforeAlpha) / nn.learningRate();
	if(std::abs(empiricalGradientWeight - computedGradientWeight) > 1e-6)
		throw Ex("failed");
	if(std::abs(empiricalGradientBias - computedGradientBias) > 1e-6)
		throw Ex("failed");
	if(std::abs(empiricalGradientAlpha - computedGradientAlpha) > 1e-6)
		throw Ex("failed");
}
#endif

} // namespace GClasses

