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
: GActivationFunction(), m_units(0), m_hinges(0)
{
}

GActivationHinge::GActivationHinge(GDomNode* pNode)
{
	GDomListIterator it(pNode->field("hinges"));
	GVec::deserialize(m_hinges.v, it);
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
	m_hinges.resize(units);
	GVec::setAll(m_hinges.v, 0.0, units);
}

// virtual
void GActivationHinge::refine(const double* pNet, const double* pActivation, const double* pError, double learningRate)
{
	double* pHinge = m_hinges.v;
	const double* pErr = pError;
	const double* pN = pNet;
	for(size_t i = 0; i < m_units; i++)
	{
		*pHinge = std::max(-1.0, std::min(1.0, *pHinge + learningRate * (*pErr) * (sqrt(*pN * *pN + 1.0) - 1.0)));
		pN++;
		pErr++;
		pHinge++;
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







GActivationLogExp::GActivationLogExp()
: GActivationFunction(), m_units(0), m_alphas(0)
{
}

GActivationLogExp::GActivationLogExp(GDomNode* pNode)
{
	GDomListIterator it(pNode->field("alphas"));
	GVec::deserialize(m_alphas.v, it);
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
	m_alphas.resize(units);
	GVec::setAll(m_alphas.v, 0.0, units);
}

// virtual
void GActivationLogExp::refine(const double* pNet, const double* pActivation, const double* pError, double learningRate)
{
	double* pAlpha = m_alphas.v;
	const double* pErr = pError;
	const double* pN = pNet;
	const double* pAct = pActivation;
	for(size_t i = 0; i < m_units; i++)
	{
		if(*pAlpha >= 0)
			*pAlpha = std::max(-1.0, std::min(1.0, *pAlpha + learningRate * (*pErr) * ((*pN) * exp(*pAlpha * (*pN)) - (*pN))));
		else
			*pAlpha = std::max(-1.0, std::min(1.0, *pAlpha + learningRate * 1.0 / (*pErr) * ((*pAct) * exp(std::min(300.0, -(*pAlpha) * (*pAct))) - (*pAct))));
		pN++;
		pAct++;
		pErr++;
		pAlpha++;
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


} // namespace GClasses

