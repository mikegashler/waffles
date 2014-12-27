/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Michael R. Smith,
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

#include "GNeuralNet.h"
#ifndef MIN_PREDICT
#include "GMath.h"
#endif // MIN_PREDICT
#include "GActivation.h"
#ifndef MIN_PREDICT
#include "GDistribution.h"
#endif // MIN_PREDICT
#include "GError.h"
#include "GRand.h"
#include "GVec.h"
#include "GDom.h"
#ifndef MIN_PREDICT
#include "GHillClimber.h"
#endif  // MIN_PREDICT
#include "GTransform.h"
#ifndef MIN_PREDICT
#include "GSparseMatrix.h"
#include "GDistance.h"
#include "GAssignment.h"
#endif // MIN_PREDICT
#include "GHolders.h"
#include "GBits.h"
#include "GFourier.h"

using std::vector;

namespace GClasses {

GDomNode* GNeuralNetLayer::baseDomNode(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "type", pDoc->newString(type()));
	return pNode;
}

GNeuralNetLayer* GNeuralNetLayer::deserialize(GDomNode* pNode)
{
	const char* szType = pNode->field("type")->asString();
	if(strcmp(szType, "classic") == 0)
		return new GLayerClassic(pNode);
	if(strcmp(szType, "mixed") == 0)
		return new GLayerMixed(pNode);
	if(strcmp(szType, "rbm") == 0)
		return new GLayerRestrictedBoltzmannMachine(pNode);
	if(strcmp(szType, "softmax") == 0)
		return new GLayerSoftMax(pNode);
	if(strcmp(szType, "conv1") == 0)
		return new GLayerConvolutional1D(pNode);
	else
		throw Ex("Unrecognized neural network layer type: ", szType);
}

GMatrix* GNeuralNetLayer::feedThrough(const GMatrix& data)
{
	size_t outputCount = outputs();
	GMatrix* pResults = new GMatrix(0, outputCount);
	for(size_t i = 0; i < data.rows(); i++)
	{
		feedForward(data[i]);
		GVec::copy(pResults->newRow(), activation(), outputCount);
	}
	return pResults;
}

void GNeuralNetLayer::feedForward(const double* pIn)
{
	copyBiasToNet();
	feedIn(pIn, 0, inputs());
	activate();
}









GLayerClassic::GLayerClassic(size_t inputs, size_t outputs, GActivationFunction* pActivationFunction)
{
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationTanH();
	resize(inputs, outputs, NULL);
}

GLayerClassic::GLayerClassic(GDomNode* pNode)
: m_weights(pNode->field("weights")), m_delta(m_weights.rows(), m_weights.cols()), m_bias(6, m_weights.cols())
{
	GDomListIterator it(pNode->field("bias"));
	GVec::deserialize(bias(), it);
	GDomListIterator itSlack(pNode->field("slack"));
	GVec::deserialize(slack(), itSlack);
	m_pActivationFunction = GActivationFunction::deserialize(pNode->field("act_func"));
	m_delta.setAll(0.0);
	GVec::setAll(biasDelta(), 0.0, m_weights.cols());
}

GLayerClassic::~GLayerClassic()
{
	delete(m_pActivationFunction);
}

GDomNode* GLayerClassic::serialize(GDom* pDoc)
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "weights", m_weights.serialize(pDoc));
	pNode->addField(pDoc, "bias", GVec::serialize(pDoc, bias(), outputs()));
	pNode->addField(pDoc, "slack", GVec::serialize(pDoc, slack(), outputs()));
	pNode->addField(pDoc, "act_func", m_pActivationFunction->serialize(pDoc));
	return pNode;
}

void GLayerClassic::resize(size_t inputCount, size_t outputCount, GRand* pRand, double deviation)
{
	if(inputCount == inputs() && outputCount == outputs())
		return;
	size_t oldInputs = inputs();
	size_t oldOutputs = outputs();
	size_t fewerInputs = std::min(oldInputs, inputCount);
	size_t fewerOutputs = std::min(oldOutputs, outputCount);

	// Weights
	m_weights.resizePreserve(inputCount, outputCount);
	m_delta.resizePreserve(inputCount, outputCount);
	m_delta.setAll(0.0);
	double dev = deviation;
	if(pRand)
	{
		if(fewerInputs * fewerOutputs >= 8)
		{
			double d = 0.0;
			for(size_t i = 0; i < fewerInputs; i++)
			{
				double* pRow = m_weights[i];
				for(size_t j = 0; j < fewerOutputs; j++)
					d += (*pRow) * (*pRow);
			}
			dev *= sqrt(d / (fewerInputs * fewerOutputs));
			if(inputCount * outputCount - fewerInputs * fewerOutputs > fewerInputs * fewerOutputs)
				dev *= fewerInputs * fewerOutputs / (inputCount * outputCount - fewerInputs * fewerOutputs);
		}
		for(size_t i = 0; i < fewerInputs; i++)
		{
			double* pRow = m_weights[i] + fewerOutputs;
			for(size_t j = fewerOutputs; j < outputCount; j++)
				*(pRow++) = dev * pRand->normal();
		}
		for(size_t i = fewerInputs; i < inputCount; i++)
		{
			double* pRow = m_weights[i];
			for(size_t j = 0; j < outputCount; j++)
				*(pRow++) = dev * pRand->normal();
		}
	}

	// Bias
	m_bias.resizePreserve(6, outputCount);
	GVec::setAll(biasDelta(), 0.0, outputCount);
	if(pRand)
	{
		double* pB = bias() + fewerOutputs;
		for(size_t j = fewerOutputs; j < outputCount; j++)
			*(pB++) = dev * pRand->normal();
	}

	// Slack
	double* pS = slack() + fewerOutputs;
	for(size_t j = fewerOutputs; j < outputCount; j++)
		*(pS++) = 0.0;
}

// virtual
void GLayerClassic::resetWeights(GRand& rand)
{
	size_t outputCount = outputs();
	size_t inputCount = inputs();
	double mag = std::max(0.03, 1.0 / inputCount); // maxing with 0.03 helps to prevent the gradient from vanishing beyond the precision of doubles in deep networks
	for(size_t i = 0; i < inputCount; i++)
	{
		double* pW = m_weights[i];
		for(size_t j = 0; j < outputCount; j++)
			*(pW++) = rand.normal() * mag;
	}
	m_delta.setAll(0.0);
	double* pB = bias();
	for(size_t i = 0; i < outputCount; i++)
		*(pB++) = rand.normal() * mag;
	GVec::setAll(biasDelta(), 0.0, outputCount);
}

// virtual
void GLayerClassic::copyBiasToNet()
{
	GAssert(bias()[outputs() - 1] > -1e100 && bias()[outputs() - 1] < 1e100);
	GVec::copy(net(), bias(), outputs());
}

// virtual
void GLayerClassic::feedIn(const double* pIn, size_t inputStart, size_t inputCount)
{
	inputCount += inputStart;
	GAssert(inputCount <= m_weights.rows());
	size_t outputCount = outputs();
	double* pNet = net();
	GAssert(pNet[outputCount - 1] > -1e100 && pNet[outputCount - 1] < 1e100);
	for(size_t i = inputStart; i < inputCount; i++)
	{
		GAssert(m_weights.row(i)[outputCount - 1] > -1e100 && m_weights.row(i)[outputCount - 1] < 1e100);
		GVec::addScaled(pNet, *(pIn++), m_weights.row(i), outputCount);
	}
}

// virtual
void GLayerClassic::activate()
{
	double* pAct = activation();
	size_t outputCount = outputs();
	double* pNet = net();
	for(size_t i = 0; i < outputCount; i++)
	{
		GAssert(*pNet < 1e100 && *pNet > -1e100);
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
	}
}

// virtual
void GLayerClassic::dropOut(GRand& rand, double probOfDrop)
{
	double* pAct = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			pAct[i] = 0.0;
	}
}

// virtual
void GLayerClassic::dropConnect(GRand& rand, double probOfDrop)
{
	size_t outputCount = m_weights.cols();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		double* pW = m_weights[i];
		double* pD = m_delta[i];
		for(size_t j = 0; j < outputCount; j++)
		{
			if(rand.uniform() < probOfDrop)
			{
				*pD = *pW;
				*pW = 0.0;
			}
			else
			{
				*pD = UNKNOWN_REAL_VALUE;
			}
			pD++;
			pW++;
		}
	}
}

void GLayerClassic::feedForwardWithInputBias(const double* pIn)
{
	size_t outputCount = outputs();
	double* pNet = net();
	GVec::setAll(pNet, *(pIn++), outputCount);
	for(size_t i = 0; i < m_weights.rows(); i++)
		GVec::addScaled(pNet, *(pIn++), m_weights.row(i), outputCount);
	GVec::add(pNet, bias(), outputCount);

	// Apply the activation function
	double* pAct = activation();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
}

void GLayerClassic::feedForwardToOneOutput(const double* pIn, size_t output, bool inputBias)
{
	// Compute net = pIn * m_weights + bias
	GAssert(output < outputs());
	double* pNet = net() + output;
	*pNet = inputBias ? *(pIn++) : 0.0;
	for(size_t i = 0; i < m_weights.rows(); i++)
		*pNet += *(pIn++) * m_weights[i][output];
	*pNet += bias()[output];

	// Apply the activation function
	double* pAct = activation() + output;
	*pAct = m_pActivationFunction->squash(*pNet);
}

void GLayerClassic::computeError(const double* pTarget)
{
	size_t outputUnits = outputs();
	double* pAct = activation();
	double* pSlack = slack();
	double* pErr = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(*pTarget == UNKNOWN_REAL_VALUE)
			*pErr = 0.0;
		else
		{
			if(*pTarget > *pAct + *pSlack)
				*pErr = (*pTarget - *pAct - *pSlack);
			else if(*pTarget < *pAct - *pSlack)
				*pErr = (*pTarget - *pAct + *pSlack);
			else
				*pErr = 0.0;
			GAssert(std::abs(*pErr) < m_pActivationFunction->halfRange() * 2.0);
		}
		pTarget++;
		pSlack++;
		pAct++;
		pErr++;
	}
}

void GLayerClassic::computeErrorSingleOutput(double target, size_t output)
{
	double sla = slack()[output];
	double* pErr = &error()[output];
	double act = activation()[output];
	if(target > act + sla)
		*pErr = (target - act - sla);
	else if(target < act - sla)
		*pErr = (target - act + sla);
	else
		*pErr = 0.0;
}

void GLayerClassic::deactivateError()
{
	size_t outputUnits = outputs();
	double* pErr = error();
	double* pNet = net();
	double* pAct = activation();
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct);
		pNet++;
		pAct++;
		pErr++;
	}
}

void GLayerClassic::deactivateErrorSingleOutput(size_t output)
{
	double* pErr = &error()[output];
	double netVal = net()[output];
	double act = activation()[output];
	(*pErr) *= m_pActivationFunction->derivativeOfNet(netVal, act);
}

void GLayerClassic::backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
{
	double* pUpStreamError = pUpStreamLayer->error();
	size_t inputCount = pUpStreamLayer->outputs();
	GAssert(inputStart + inputCount <= m_weights.rows());
	size_t outputCount = outputs();
	const double* pSource = error();
	for(size_t i = 0; i < inputCount; i++)
	{
		*pUpStreamError = GVec::dotProduct(pSource, m_weights[inputStart + i], outputCount);
		pUpStreamError++;
	}
}

void GLayerClassic::backPropErrorSingleOutput(size_t outputNode, double* pUpStreamError)
{
	GAssert(outputNode < outputs());
	double in = error()[outputNode];
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		*pUpStreamError = in * m_weights[i][outputNode];
		pUpStreamError++;
	}
}

void GLayerClassic::updateBias(double learningRate, double momentum)
{
	double* pB = error();
	double* pD = biasDelta();
	double* pW = bias();
	size_t outputCount = outputs();
	for(size_t down = 0; down < outputCount; down++)
	{
		*pD *= momentum;
		*pD += (*(pB++) * learningRate);
		*(pW++) += *(pD++);
	}
}

void GLayerClassic::updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	double* pErr = error();
	size_t outputCount = outputs();
	size_t inputEnd = inputStart + inputCount;
	for(size_t up = inputStart; up < inputEnd; up++)
	{
		double* pB = pErr;
		double* pD = m_delta[up];
		double* pW = m_weights[up];
		double act = *(pUpStreamActivation++);
		for(size_t down = 0; down < outputCount; down++)
		{
			*pD *= momentum;
			*pD += (*(pB++) * learningRate * act);
			*(pW++) += *(pD++);
		}
	}
}

void GLayerClassic::updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	double* pErr = error();
	size_t outputCount = outputs();
	size_t inputEnd = inputStart + inputCount;
	for(size_t up = inputStart; up < inputEnd; up++)
	{
		double* pB = pErr;
		double* pD = m_delta[up];
		double* pW = m_weights[up];
		double act = *(pUpStreamActivation++);
		for(size_t down = 0; down < outputCount; down++)
		{
			if(*pD == UNKNOWN_REAL_VALUE)
				*pW += (*(pB++) * learningRate * act);
			else
				*pW = *pD;
			pW++;
			pD++;
		}
	}
}

void GLayerClassic::copySingleNeuronWeights(size_t source, size_t dest)
{
	for(size_t up = 0; up < m_weights.rows(); up++)
	{
		m_weights[up][dest] = m_weights[up][source];
	}
	bias()[dest] = bias()[source];
}

void GLayerClassic::setWeightsSingleNeuron(size_t outputNode, const double* weights)
{
	for(size_t up = 0; up < m_weights.rows(); up++)
	{
		m_weights[up][outputNode] = weights[up];
	}
	bias()[outputNode] = weights[m_weights.rows()];
}

void GLayerClassic::getWeightsSingleNeuron(size_t outputNode, double*& weights)
{
//	weights = new double(m_weights.rows() + 1);
	for(size_t up = 0; up < m_weights.rows(); up++)
	{
		weights[up] = m_weights[up][outputNode];
	}
	weights[m_weights.rows()] = bias()[outputNode];
}

void GLayerClassic::updateWeightsSingleNeuron(size_t outputNode, const double* pUpStreamActivation, double learningRate, double momentum)
{
	// Adjust the weights
	double err = error()[outputNode];
	for(size_t up = 0; up < m_weights.rows(); up++)
	{
		double* pD = &m_delta[up][outputNode];
		double* pW = &m_weights[up][outputNode];
		double act = *(pUpStreamActivation++);
		*pD *= momentum;
		*pD += (err * learningRate * act);
		*pW = std::max(-1e12, std::min(1e12, *pW + *pD));
	}

	// Adjust the bias
	double* pD = &biasDelta()[outputNode];
	double* pW = &bias()[outputNode];
	*pD *= momentum;
	*pD += (err * learningRate);
	*pW = std::max(-1e12, std::min(1e12, *pW + *pD));
}

void GLayerClassic::scaleWeights(double factor, bool scaleBiases)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < m_weights.rows(); i++)
		GVec::multiply(m_weights[i], factor, outputCount);
	if(scaleBiases)
		GVec::multiply(bias(), factor, outputCount);
}

void GLayerClassic::diminishWeights(double amount, bool regularizeBiases)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < m_weights.rows(); i++)
		GVec::regularize_1(m_weights[i], amount, outputCount);
	if(regularizeBiases)
		GVec::regularize_1(bias(), amount, outputCount);
}

void GLayerClassic::contractWeights(double factor, bool contractBiases)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		double* pRow = m_weights[i];
		double* pErr = error();
		for(size_t j = 0; j < outputCount; j++)
			*(pRow++) *= (1.0 - factor * *(pErr++));
	}
	if(contractBiases)
	{
		double* pRow = bias();
		double* pErr = error();
		for(size_t j = 0; j < outputCount; j++)
			*(pRow++) *= (1.0 - factor * *(pErr++));
	}
}

void GLayerClassic::regularizeWeights(double factor, double power)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		double* pW = m_weights[i];
		for(size_t j = 0; j < outputCount; j++)
		{
			*pW -= GBits::sign(*pW) * factor * pow(std::abs(*pW), power);
			pW++;
		}
	}
	double* pW = bias();
	for(size_t j = 0; j < outputCount; j++)
	{
		*pW -= GBits::sign(*pW) * factor * pow(std::abs(*pW), power);
		pW++;
	}
}

void GLayerClassic::transformWeights(GMatrix& transform, const double* pOffset)
{
	if(transform.rows() != inputs())
		throw Ex("Transformation matrix not suitable size for this layer");
	if(transform.rows() != transform.cols())
		throw Ex("Expected a square transformation matrix.");
	size_t outputCount = outputs();
	GMatrix* pNewWeights = GMatrix::multiply(transform, m_weights, true, false);
	Holder<GMatrix> hNewWeights(pNewWeights);
	m_weights.copyBlock(*pNewWeights, 0, 0, pNewWeights->rows(), outputCount, 0, 0, false);
	double* pNet = net();
	GVec::setAll(pNet, 0.0, outputCount);
	for(size_t i = 0; i < m_weights.rows(); i++)
		GVec::addScaled(pNet, *(pOffset++), m_weights.row(i), outputCount);
	GVec::add(bias(), pNet, outputCount);
}

void GLayerClassic::setWeightsToIdentity(size_t start, size_t count)
{
	size_t end = std::min(start + count, outputs());
	for(size_t i = start; i < end; i++)
	{
		bias()[i] = 0.0;
		for(size_t j = 0; j < inputs(); j++)
		{
			if(j == i)
			{
				m_weights[j][i] = m_pActivationFunction->identityDiag();
				bias()[i] = m_pActivationFunction->identityBias();
			}
			else
				m_weights[j][i] = 0.0;
		}
	}
}

// virtual
void GLayerClassic::maxNorm(double max)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		double squaredMag = 0;
		for(size_t j = 0; j < m_weights.rows(); j++)
		{
			double d = m_weights[j][i];
			squaredMag += (d * d);
		}
		if(squaredMag > max * max)
		{
			double scal = max / sqrt(squaredMag);
			for(size_t j = 0; j < m_weights.rows(); j++)
				m_weights[j][i] *= scal;
		}
	}
}

// virtual
double GLayerClassic::unitIncomingWeightsL1Norm(size_t unit)
{
	double norm = 0.0;
	size_t inputCount = inputs();
	for(size_t i = 0; i < inputCount; i++)
		norm += std::abs(m_weights[i][unit]);
	return norm;
}

// virtual
double GLayerClassic::unitIncomingWeightsL2Norm(size_t unit)
{
	double norm = 0.0;
	size_t inputCount = inputs();
	for(size_t i = 0; i < inputCount; i++)
		norm += m_weights[i][unit] * m_weights[i][unit];
	return norm;
}

// virtual
double GLayerClassic::unitOutgoingWeightsL1Norm(size_t input)
{
	return GVec::sumAbsoluteValues(m_weights[input], outputs());
}

// virtual
double GLayerClassic::unitOutgoingWeightsL2Norm(size_t input)
{
	return GVec::squaredMagnitude(m_weights[input], outputs());
}

// virtual
void GLayerClassic::scaleUnitIncomingWeights(size_t unit, double scalar)
{
	size_t inputCount = inputs();
	for(size_t i = 0; i < inputCount; i++)
		m_weights[i][unit] *= scalar;
}

// virtual
void GLayerClassic::scaleUnitOutgoingWeights(size_t input, double scalar)
{
	GVec::multiply(m_weights[input], scalar, outputs());
}

// virtual
size_t GLayerClassic::countWeights()
{
	return (inputs() + 1) * outputs();
}

// virtual
size_t GLayerClassic::weightsToVector(double* pOutVector)
{
	GVec::copy(pOutVector, bias(), outputs());
	pOutVector += outputs();
	m_weights.toVector(pOutVector);
	return (inputs() + 1) * outputs();
}

// virtual
size_t GLayerClassic::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, outputs());
	pVector += outputs();
	m_weights.fromVector(pVector, inputs());
	return (inputs() + 1) * outputs();
}

// virtual
void GLayerClassic::copyWeights(GNeuralNetLayer* pSource)
{
	GLayerClassic* src = (GLayerClassic*)pSource;
	m_weights.copyBlock(src->m_weights, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	GVec::copy(bias(), src->bias(), src->outputs());
}

// virtual
void GLayerClassic::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	size_t n = std::min(outputs() - start, count);
	for(size_t j = 0; j < m_weights.rows(); j++)
		GVec::perturb(m_weights[j] + start, deviation, n, rand);
	GVec::perturb(bias() + start, deviation, n, rand);
}

// virtual
void GLayerClassic::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	size_t outputCount = outputs();
	double* pW = m_weights[input];
	double* pB = bias();
	double f = (oldMax - oldMin) / (newMax - newMin);
	double g = (oldMin - newMin * f);
	for(size_t i = 0; i < outputCount; i++)
	{
		*pB += (*pW * g);
		*pW *= f;
		pW++;
		pB++;
	}
}








GLayerSoftMax::GLayerSoftMax(size_t inputs, size_t outputs)
: GLayerClassic(inputs, outputs, new GActivationLogistic())
{
}

GLayerSoftMax::GLayerSoftMax(GDomNode* pNode)
: GLayerClassic(pNode)
{
}

// virtual
void GLayerSoftMax::activate()
{
	double* pAct = activation();
	size_t outputCount = outputs();
	double* pNet = net();
	double sum = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		double d = m_pActivationFunction->squash(*(pNet++));
		sum += d;
		*(pAct++) = d;
	}
	if(sum > 1e-12)
	{
		double fac = 1.0 / sum;
		m_weights.multiply(fac);
		GVec::multiply(bias(), fac, outputCount);
		GVec::multiply(activation(), fac, outputCount);
	}
}







GLayerMixed::GLayerMixed()
{
}

GLayerMixed::GLayerMixed(GDomNode* pNode)
{
	GDomListIterator it(pNode->field("comps"));
	while(it.remaining() > 0)
	{
		m_components.push_back(GNeuralNetLayer::deserialize(it.current()));
		it.advance();
	}
}

GLayerMixed::~GLayerMixed()
{
	for(size_t i = 0; i < m_components.size(); i++)
		delete(m_components[i]);
}

// virtual
GDomNode* GLayerMixed::serialize(GDom* pDoc)
{
	GDomNode* pNode = baseDomNode(pDoc);
	GDomNode* pList = pNode->addField(pDoc, "comps", pDoc->newList());
	for(size_t i = 0; i < m_components.size(); i++)
		pList->addItem(pDoc, m_components[i]->serialize(pDoc));
	return pNode;
}

void GLayerMixed::addComponent(GNeuralNetLayer* pComponent)
{
	if(m_activation.cols() > 0)
		throw Ex("Cannot add a component to GLayerMixed after it has been used");
	if(m_inputError.cols() == 0)
		m_inputError.resize(1, pComponent->inputs());
	else if(m_inputError.cols() != pComponent->inputs())
		throw Ex("This component expects ", to_str(pComponent->inputs()), ", inputs, which conflicts with a previous component that expects ", to_str(m_inputError.cols()), " inputs");
	m_components.push_back(pComponent);
}

// virtual
size_t GLayerMixed::inputs()
{
	return m_inputError.cols();
}

// virtual
size_t GLayerMixed::outputs()
{
	size_t outs = m_activation.cols();
	if(outs == 0)
	{
		if(m_components.size() < 2)
			throw Ex("GLayerMixed requires at least 2 components to be added before it is used");
		for(size_t i = 0; i < m_components.size(); i++)
			outs += m_components[i]->outputs();
		m_activation.resize(2, outs);
	}
	return outs;
}

// virtual
void GLayerMixed::resize(size_t inputs, size_t outputs, GRand* pRand, double deviation)
{
	if(outputs != m_activation.cols())
		throw Ex("Sorry, GLayerMixed does not support resizing the number of outputs");
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->resize(inputs, m_components[i]->outputs(), pRand, deviation);
}

// virtual
void GLayerMixed::copyBiasToNet()
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->copyBiasToNet();
}

// virtual
void GLayerMixed::feedIn(const double* pIn, size_t inputStart, size_t inputCount)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->feedIn(pIn, inputStart, inputCount);
}

// virtual
void GLayerMixed::activate()
{
	double* pAct = m_activation[0];
	for(size_t i = 0; i < m_components.size(); i++)
	{
		m_components[i]->activate();
		GVec::copy(pAct, m_components[i]->activation(), m_components[i]->outputs());
		pAct += m_components[i]->outputs();
	}
}

// virtual
void GLayerMixed::dropOut(GRand& rand, double probOfDrop)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->dropOut(rand, probOfDrop);
}

// virtual
void GLayerMixed::dropConnect(GRand& rand, double probOfDrop)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->dropConnect(rand, probOfDrop);
}

// virtual
void GLayerMixed::computeError(const double* pTarget)
{
	size_t outputUnits = outputs();
	double* pAct = activation();
	double* pErr = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(*pTarget == UNKNOWN_REAL_VALUE)
			*pErr = 0.0;
		else
			*pErr = *pTarget - *pAct;
		pTarget++;
		pAct++;
		pErr++;
	}
}

// virtual
void GLayerMixed::deactivateError()
{
	double* pErr = error();
	for(size_t i = 0; i < m_components.size(); i++)
	{
		GVec::copy(m_components[i]->error(), pErr, m_components[i]->outputs());
		m_components[i]->deactivateError();
		pErr += m_components[i]->outputs();
	}
}

// virtual
void GLayerMixed::backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
{
	double* pBuf = m_inputError[0];
	size_t inps = pUpStreamLayer->outputs();
	GVec::setAll(pBuf, 0.0, inps);
	for(size_t i = 0; i < m_components.size(); i++)
	{
		m_components[i]->backPropError(pUpStreamLayer, inputStart);
		GVec::add(pBuf, pUpStreamLayer->error(), inps);
	}
	GVec::copy(pUpStreamLayer->error(), pBuf, inps);
}

// virtual
void GLayerMixed::updateBias(double learningRate, double momentum)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->updateBias(learningRate, momentum);
}

// virtual
void GLayerMixed::updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->updateWeights(pUpStreamActivation, inputStart, inputCount, learningRate, momentum);
}

void GLayerMixed::updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->updateWeightsAndRestoreDroppedOnes(pUpStreamActivation, inputStart, inputCount, learningRate, momentum);
}

// virtual
void GLayerMixed::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->scaleWeights(factor, scaleBiases);
}

// virtual
void GLayerMixed::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->diminishWeights(amount, diminishBiases);
}

// virtual
size_t GLayerMixed::countWeights()
{
	size_t sum = 0;
	for(size_t i = 0; i < m_components.size(); i++)
		sum += m_components[i]->countWeights();
	return sum;
}

// virtual
size_t GLayerMixed::weightsToVector(double* pOutVector)
{
	size_t sum = 0;
	for(size_t i = 0; i < m_components.size(); i++)
	{
		size_t s = m_components[i]->weightsToVector(pOutVector);
		sum += s;
		pOutVector += s;
	}
	return sum;
}

// virtual
size_t GLayerMixed::vectorToWeights(const double* pVector)
{
	size_t sum = 0;
	for(size_t i = 0; i < m_components.size(); i++)
	{
		size_t s = m_components[i]->vectorToWeights(pVector);
		sum += s;
		pVector += s;
	}
	return sum;
}

// virtual
void GLayerMixed::copyWeights(GNeuralNetLayer* pSource)
{
	GLayerMixed* pThat = (GLayerMixed*)pSource;
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->copyWeights(pThat->m_components[i]);
}

// virtual
void GLayerMixed::resetWeights(GRand& rand)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->resetWeights(rand);
}

// virtual
void GLayerMixed::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->perturbWeights(rand, deviation, start, count);
}

// virtual
void GLayerMixed::maxNorm(double max)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->maxNorm(max);
}

// virtual
double GLayerMixed::unitIncomingWeightsL1Norm(size_t unit)
{
	size_t n = 0;
	for(size_t i = 0; i < m_components.size(); i++)
	{
		if(n + m_components[i]->outputs() > unit)
			return m_components[i]->unitIncomingWeightsL1Norm(unit - n);
		n += m_components[i]->outputs();
	}
	throw Ex("out or range");
}

// virtual
double GLayerMixed::unitIncomingWeightsL2Norm(size_t unit)
{
	size_t n = 0;
	for(size_t i = 0; i < m_components.size(); i++)
	{
		if(n + m_components[i]->outputs() > unit)
			return m_components[i]->unitIncomingWeightsL2Norm(unit - n);
		n += m_components[i]->outputs();
	}
	throw Ex("out or range");
}

// virtual
double GLayerMixed::unitOutgoingWeightsL1Norm(size_t input)
{
	double norm = 0.0;
	for(size_t i = 0; i < m_components.size(); i++)
		norm += m_components[i]->unitOutgoingWeightsL1Norm(input);
	return norm;
}

// virtual
double GLayerMixed::unitOutgoingWeightsL2Norm(size_t input)
{
	double norm = 0.0;
	for(size_t i = 0; i < m_components.size(); i++)
		norm += m_components[i]->unitOutgoingWeightsL2Norm(input);
	return norm;
}

// virtual
void GLayerMixed::scaleUnitIncomingWeights(size_t unit, double scalar)
{
	size_t n = 0;
	for(size_t i = 0; i < m_components.size(); i++)
	{
		if(n + m_components[i]->outputs() > unit)
		{
			m_components[i]->scaleUnitIncomingWeights(unit - n, scalar);
			return;
		}
		n += m_components[i]->outputs();
	}
	throw Ex("out or range");
}

// virtual
void GLayerMixed::scaleUnitOutgoingWeights(size_t input, double scalar)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->scaleUnitOutgoingWeights(input, scalar);
}

// virtual
void GLayerMixed::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->renormalizeInput(input, oldMin, oldMax, newMin, newMax);
}











GLayerRestrictedBoltzmannMachine::GLayerRestrictedBoltzmannMachine(size_t inputs, size_t outputs, GActivationFunction* pActivationFunction)
{
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationLogistic();
	resize(inputs, outputs, NULL);
}

GLayerRestrictedBoltzmannMachine::GLayerRestrictedBoltzmannMachine(GDomNode* pNode)
: m_weights(pNode->field("weights")), m_bias(4, m_weights.rows()), m_biasReverse(4, m_weights.cols())
{
	GDomListIterator it(pNode->field("bias"));
	GVec::deserialize(bias(), it);
	GDomListIterator itRev(pNode->field("biasRev"));
	GVec::deserialize(biasReverse(), itRev);

	// Unmarshall the activation function
	m_pActivationFunction = GActivationFunction::deserialize(pNode->field("act_func"));
}

GLayerRestrictedBoltzmannMachine::~GLayerRestrictedBoltzmannMachine()
{
	delete(m_pActivationFunction);
}

GDomNode* GLayerRestrictedBoltzmannMachine::serialize(GDom* pDoc)
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "weights", m_weights.serialize(pDoc));
	pNode->addField(pDoc, "bias", GVec::serialize(pDoc, bias(), outputs()));
	pNode->addField(pDoc, "biasRev", GVec::serialize(pDoc, biasReverse(), inputs()));

	// Marshall the activation function
	pNode->addField(pDoc, "act_func", m_pActivationFunction->serialize(pDoc));

	return pNode;
}

void GLayerRestrictedBoltzmannMachine::resize(size_t inputCount, size_t outputCount, GRand* pRand, double deviation)
{
	if(inputCount == inputs() && outputCount == outputs())
		return;
	size_t oldInputs = inputs();
	size_t oldOutputs = outputs();
	size_t fewerInputs = std::min(oldInputs, inputCount);
	size_t fewerOutputs = std::min(oldOutputs, outputCount);

	// Weights
	m_weights.resizePreserve(outputCount, inputCount);
	if(pRand)
	{
		for(size_t i = 0; i < fewerOutputs; i++)
		{
			double* pRow = m_weights[i] + fewerInputs;
			for(size_t j = fewerInputs; j < inputCount; j++)
				*(pRow++) = deviation * pRand->normal();
		}
		for(size_t i = fewerOutputs; i < outputCount; i++)
		{
			double* pRow = m_weights[i];
			for(size_t j = 0; j < inputCount; j++)
				*(pRow++) = deviation * pRand->normal();
		}
	}
	m_delta.resize(outputCount, inputCount);
	m_delta.setAll(0.0);

	// Bias
	m_bias.resizePreserve(5, outputCount);
	GVec::setAll(biasDelta(), 0.0, outputCount);
	if(pRand)
	{
		double* pB = bias() + fewerOutputs;
		for(size_t j = fewerOutputs; j < outputCount; j++)
			*(pB++) = deviation * pRand->normal();
	}

	// BiasReverse
	m_biasReverse.resizePreserve(5, inputCount);
	if(pRand)
	{
		double* pB = biasReverse() + fewerInputs;
		for(size_t j = fewerInputs; j < inputCount; j++)
			*(pB++) = deviation * pRand->normal();
	}
}

// virtual
void GLayerRestrictedBoltzmannMachine::resetWeights(GRand& rand)
{
	size_t outputCount = outputs();
	size_t inputCount = inputs();
	double mag = std::max(0.03, 1.0 / inputCount);
	double* pB = bias();
	for(size_t i = 0; i < outputCount; i++)
	{
		*pB = rand.normal() * mag;
		for(size_t j = 0; j < inputCount; j++)
			m_weights[i][j] = rand.normal() * mag;
		pB++;
	}
	pB = biasReverse();
	for(size_t i = 0; i < inputCount; i++)
		*(pB++) = rand.normal() * mag;
	m_delta.setAll(0.0);
	GVec::setAll(biasDelta(), 0.0, outputCount);
	GVec::setAll(biasReverseDelta(), 0.0, inputCount);
}

// virtual
void GLayerRestrictedBoltzmannMachine::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	size_t n = std::min(outputs() - start, count);
	for(size_t i = start; i < n; i++)
		GVec::perturb(m_weights[i], deviation, inputs(), rand);
	GVec::perturb(bias() + start, deviation, n, rand);
}

// virtual
void GLayerRestrictedBoltzmannMachine::copyBiasToNet()
{
	GVec::copy(net(), bias(), outputs());
}

// virtual
void GLayerRestrictedBoltzmannMachine::feedIn(const double* pIn, size_t inputStart, size_t inputCount)
{
	size_t outputCount = outputs();
	double* pNet = net();
	for(size_t i = 0; i < outputCount; i++)
		*(pNet++) += GVec::dotProduct(pIn, m_weights[i] + inputStart, inputCount);
}

// virtual
void GLayerRestrictedBoltzmannMachine::activate()
{
	double* pAct = activation();
	size_t outputCount = outputs();
	double* pNet = net();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
}

// virtual
void GLayerRestrictedBoltzmannMachine::dropOut(GRand& rand, double probOfDrop)
{
	double* pAct = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			pAct[i] = 0.0;
	}
}

// virtual
void GLayerRestrictedBoltzmannMachine::dropConnect(GRand& rand, double probOfDrop)
{
	size_t cols = m_weights.cols();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		double* pW = m_weights[i];
		double* pD = m_delta[i];
		for(size_t j = 0; j < cols; j++)
		{
			if(rand.uniform() < probOfDrop)
			{
				*pD = *pW;
				*pW = 0.0;
			}
			else
			{
				*pD = UNKNOWN_REAL_VALUE;
			}
			pD++;
			pW++;
		}
	}
}

/*
void GLayerRestrictedBoltzmannMachine::feedForward(const double* pIn)
{
	// Feed through the weights
	double* pNet = net();
	m_weights.multiply(pIn, pNet);
	size_t outputCount = outputs();
	GVec::add(pNet, bias(), outputCount);

	// Squash it
	double* pAct = activation();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
}
*/
void GLayerRestrictedBoltzmannMachine::feedBackward(const double* pIn)
{
	// Feed through the weights
	double* pNet = netReverse();
	m_weights.multiply(pIn, pNet, true);
	size_t inputCount = inputs();
	GVec::add(pNet, biasReverse(), inputCount);

	// Squash it
	double* pAct = activationReverse();
	for(size_t i = 0; i < inputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
}

void GLayerRestrictedBoltzmannMachine::resampleHidden(GRand& rand)
{
	double* pH = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		*pH = rand.uniform() < *pH ? 1.0 : 0.0;
		pH++;
	}
}

void GLayerRestrictedBoltzmannMachine::resampleVisible(GRand& rand)
{
	double* pV = activationReverse();
	size_t inputCount = inputs();
	for(size_t i = 0; i < inputCount; i++)
	{
		*pV = rand.uniform() < *pV ? 1.0 : 0.0;
		pV++;
	}
}

void GLayerRestrictedBoltzmannMachine::drawSample(GRand& rand, size_t iters)
{
	double* pH = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		*pH = ((rand.next() & 1) == 0 ? 0.0 : 1.0);
		pH++;
	}
	for(size_t i = 0; i < iters; i++)
	{
		feedBackward(activation());
		feedForward(activationReverse());
		resampleHidden(rand);
	}
	feedBackward(activation());
}

double GLayerRestrictedBoltzmannMachine::freeEnergy(const double* pVisibleSample)
{
	feedForward(pVisibleSample);
	double* pBuf = error();
	m_weights.multiply(activationReverse(), pBuf, false);
	return -GVec::dotProduct(activation(), pBuf, outputs()) -
		GVec::dotProduct(biasReverse(), activationReverse(), inputs()) -
		GVec::dotProduct(bias(), activation(), outputs());
}

void GLayerRestrictedBoltzmannMachine::contrastiveDivergence(GRand& rand, const double* pVisibleSample, double learningRate, size_t gibbsSamples)
{
	// Details of this implementation were guided by http://axon.cs.byu.edu/~martinez/classes/678/Papers/guideTR.pdf, particularly Sections 3.2 and 3.3.

	// Sample hidden vector
	feedForward(pVisibleSample);

	// Compute positive gradient
	size_t outputCount = outputs();
	size_t inputCount = inputs();
	for(size_t i = 0; i < outputCount; i++)
		GVec::addScaled(m_weights[i], activation()[i], pVisibleSample, inputCount);

	// Add positive gradient to the biases
	GVec::addScaled(biasReverse(), learningRate, pVisibleSample, inputCount);
	GVec::addScaled(bias(), learningRate, activation(), outputCount);

	// Resample
	for(size_t i = 1; i < gibbsSamples; i++)
	{
		feedBackward(activation());
		feedForward(activationReverse());
		resampleHidden(rand);
	}
	feedBackward(activation());
	feedForward(activationReverse());

	// Compute negative gradient
	for(size_t i = 0; i < outputCount; i++)
		GVec::addScaled(m_weights[i], activation()[i], activationReverse(), inputCount);

	// Subtract negative gradient from biases
	GVec::addScaled(biasReverse(), -learningRate, activationReverse(), inputCount);
	GVec::addScaled(bias(), -learningRate, activation(), outputCount);
}

void GLayerRestrictedBoltzmannMachine::computeError(const double* pTarget)
{
	size_t outputUnits = outputs();
	double* pAct = activation();
	double* pErr = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(*pTarget == UNKNOWN_REAL_VALUE)
			*pErr = 0.0;
		else
			*pErr = *pTarget - *pAct;
		pTarget++;
		pAct++;
		pErr++;
	}
}

void GLayerRestrictedBoltzmannMachine::deactivateError()
{
	size_t outputUnits = outputs();
	double* pErr = error();
	double* pNet = net();
	double* pAct = activation();
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct);
		pNet++;
		pAct++;
		pErr++;
	}
}

void GLayerRestrictedBoltzmannMachine::backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
{
	double* pDownStreamError = error();
	double* pUpStreamError = pUpStreamLayer->error();
	m_weights.multiply(pDownStreamError, pUpStreamError, true);
}

void GLayerRestrictedBoltzmannMachine::updateBias(double learningRate, double momentum)
{
	size_t outputCount = outputs();
	double* pErr = error();
	double* pBias = bias();
	for(size_t i = 0; i < outputCount; i++)
	{
		*(pBias++) += learningRate * (*pErr);
		pErr++;
	}
}

void GLayerRestrictedBoltzmannMachine::updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	if(inputStart != 0 || inputCount != inputs())
		throw Ex("Sorry, updating only a portion of the weights is not yet implemented");
	size_t outputCount = outputs();
	double* pErr = error();
	for(size_t i = 0; i < outputCount; i++)
	{
		GVec::addScaled(m_weights[i], learningRate * (*pErr), pUpStreamActivation, inputCount);
		pErr++;
	}
}

void GLayerRestrictedBoltzmannMachine::updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	if(inputStart != 0 || inputCount != inputs())
		throw Ex("Sorry, updating only a portion of the weights is not yet implemented");
	size_t inpCount = inputs();
	size_t outputCount = outputs();
	double* pErr = error();
	for(size_t i = 0; i < outputCount; i++)
	{
		double* pW = m_weights[i];
		double* pD = m_delta[i];
		const double* pA = pUpStreamActivation;
		for(size_t j = 0; j < inpCount; i++)
		{
			if(*pD == UNKNOWN_REAL_VALUE)
				*pW += learningRate * (*pErr) * *pA;
			else
				*pW = *pD;
			pW++;
			pD++;
			pA++;
		}
		pErr++;
	}
}

void GLayerRestrictedBoltzmannMachine::scaleWeights(double factor, bool scaleBiases)
{
	size_t inputCount = inputs();
	for(size_t i = 0; i < m_weights.rows(); i++)
		GVec::multiply(m_weights[i], factor, inputCount);
	if(scaleBiases)
		GVec::multiply(bias(), factor, outputs());
}

void GLayerRestrictedBoltzmannMachine::diminishWeights(double amount, bool diminishBiases)
{
	size_t inputCount = outputs();
	for(size_t i = 0; i < m_weights.rows(); i++)
		GVec::regularize_1(m_weights[i], amount, inputCount);
	if(diminishBiases)
		GVec::regularize_1(bias(), amount, outputs());
}

// virtual
void GLayerRestrictedBoltzmannMachine::maxNorm(double max)
{
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		double squaredMag = GVec::squaredMagnitude(m_weights[i], inputCount);
		if(squaredMag > max * max)
		{
			double scal = max / sqrt(squaredMag);
			GVec::multiply(m_weights[i], scal, inputCount);
		}
	}
}

// virtual
size_t GLayerRestrictedBoltzmannMachine::countWeights()
{
	return (inputs() + 1) * outputs();
}

// virtual
size_t GLayerRestrictedBoltzmannMachine::weightsToVector(double* pOutVector)
{
	GVec::copy(pOutVector, bias(), outputs());
	pOutVector += outputs();
	m_weights.toVector(pOutVector);
	return (inputs() + 1) * outputs();
}

// virtual
size_t GLayerRestrictedBoltzmannMachine::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, outputs());
	pVector += outputs();
	m_weights.fromVector(pVector, inputs());
	return (inputs() + 1) * outputs();
}

// virtual
void GLayerRestrictedBoltzmannMachine::copyWeights(GNeuralNetLayer* pSource)
{
	GLayerRestrictedBoltzmannMachine* src = (GLayerRestrictedBoltzmannMachine*)pSource;
	m_weights.copyBlock(src->m_weights, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	GVec::copy(bias(), src->bias(), src->outputs());
}

// virtual
double GLayerRestrictedBoltzmannMachine::unitIncomingWeightsL1Norm(size_t unit)
{
	return GVec::sumAbsoluteValues(m_weights[unit], inputs());
}

// virtual
double GLayerRestrictedBoltzmannMachine::unitIncomingWeightsL2Norm(size_t unit)
{
	return GVec::squaredMagnitude(m_weights[unit], inputs());
}

// virtual
double GLayerRestrictedBoltzmannMachine::unitOutgoingWeightsL1Norm(size_t input)
{
	double norm = 0.0;
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
		norm += std::abs(m_weights[i][input]);
	return norm;
}

// virtual
double GLayerRestrictedBoltzmannMachine::unitOutgoingWeightsL2Norm(size_t input)
{
	double norm = 0.0;
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
		norm += m_weights[i][input] * m_weights[i][input];
	return norm;
}

// virtual
void GLayerRestrictedBoltzmannMachine::scaleUnitIncomingWeights(size_t unit, double scalar)
{
	GVec::multiply(m_weights[unit], scalar, outputs());
}

// virtual
void GLayerRestrictedBoltzmannMachine::scaleUnitOutgoingWeights(size_t input, double scalar)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
		m_weights[i][input] *= scalar;
}

// virtual
void GLayerRestrictedBoltzmannMachine::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	size_t outputCount = outputs();
	double* pB = bias();
	double f = (oldMax - oldMin) / (newMax - newMin);
	double g = (oldMin - newMin * f);
	for(size_t i = 0; i < outputCount; i++)
	{
		*pB += m_weights[i][input] * g;
		m_weights[i][input] *= f;
	}
}







GLayerConvolutional1D::GLayerConvolutional1D(size_t inputSamples, size_t inputChannels, size_t kernelSize, size_t kernelsPerChannel, GActivationFunction* pActivationFunction)
: m_inputSamples(inputSamples),
m_inputChannels(inputChannels),
m_outputSamples(inputSamples - kernelSize + 1),
m_kernelsPerChannel(kernelsPerChannel),
m_kernels(inputChannels * kernelsPerChannel, kernelSize),
m_delta(inputChannels * kernelsPerChannel, kernelSize),
m_bias(2, inputChannels * kernelsPerChannel)
{
	if(kernelSize > inputSamples)
		throw Ex("kernelSize must be <= inputSamples");
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationLogistic();
	size_t totalOutputs = inputChannels * kernelsPerChannel * m_outputSamples;
	m_activation.resize(3, totalOutputs);
	m_delta.setAll(0.0);
	GVec::setAll(biasDelta(), 0.0, inputChannels * kernelsPerChannel);
}

GLayerConvolutional1D::GLayerConvolutional1D(GDomNode* pNode)
: m_inputSamples(pNode->field("isam")->asInt()),
m_inputChannels(pNode->field("ichan")->asInt()),
m_outputSamples(pNode->field("osam")->asInt()),
m_kernelsPerChannel(pNode->field("kpc")->asInt()),
m_kernels(pNode->field("kern")),
m_delta(pNode->field("delt")),
m_activation(pNode->field("act")),
m_pActivationFunction(GActivationFunction::deserialize(pNode->field("act_func")))
{
}

GLayerConvolutional1D::~GLayerConvolutional1D()
{
}

// virtual
GDomNode* GLayerConvolutional1D::serialize(GDom* pDoc)
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "isam", pDoc->newInt(m_inputSamples));
	pNode->addField(pDoc, "ichan", pDoc->newInt(m_inputChannels));
	pNode->addField(pDoc, "osam", pDoc->newInt(m_outputSamples));
	pNode->addField(pDoc, "kpc", pDoc->newInt(m_kernelsPerChannel));
	pNode->addField(pDoc, "kern", m_kernels.serialize(pDoc));
	pNode->addField(pDoc, "delt", m_delta.serialize(pDoc));
	pNode->addField(pDoc, "act", m_activation.serialize(pDoc));
	pNode->addField(pDoc, "bias", m_bias.serialize(pDoc));
	pNode->addField(pDoc, "act_func", m_pActivationFunction->serialize(pDoc));
	return pNode;
}

// virtual
void GLayerConvolutional1D::resize(size_t inputs, size_t outputs, GRand* pRand, double deviation)
{
	if(inputs != m_inputSamples * m_inputChannels)
		throw Ex("Changing the size of GLayerConvolutional1D is not supported");
	if(outputs != m_inputChannels * m_kernelsPerChannel * m_outputSamples)
		throw Ex("Changing the size of GLayerConvolutional1D is not supported");
}

// virtual
void GLayerConvolutional1D::resetWeights(GRand& rand)
{
	size_t kernelSize = m_kernels.cols();
	double mag = std::max(0.03, 1.0 / kernelSize);
	for(size_t i = 0; i < m_kernels.rows(); i++)
	{
		double* pW = m_kernels[i];
		for(size_t j = 0; j < kernelSize; j++)
			*(pW++) = rand.normal() * mag;
	}
	m_delta.setAll(0.0);
	double* pB = bias();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		*(pB++) = rand.normal() * mag;
	GVec::setAll(biasDelta(), 0.0, m_kernels.rows());
}

// virtual
void GLayerConvolutional1D::copyBiasToNet()
{
	double* pNet = net();
	for(size_t i = 0; i < m_outputSamples; i++)
	{
		GVec::copy(pNet, bias(), m_kernels.rows());
		pNet += m_kernels.rows();
	}
}

// virtual
void GLayerConvolutional1D::feedIn(const double* pIn, size_t inputStart, size_t inputCount)
{
	GAssert(inputStart == 0);
	if(inputCount != m_inputSamples * m_inputChannels)
		throw Ex("Sorry, partial feeding not supported in GLayerConvolutional1D");
	size_t kernelSize = m_kernels.cols();
	double* pNet = net();
	for(size_t i = 0; i < m_outputSamples; i++) // for each output sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				double* pW = m_kernels[kern++];
				double d = 0.0;
				const double* pInput = pIn;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
				{
					d += *(pW++) * *pInput;
					pInput += m_inputChannels;
				}
				*(pNet++) += d;
			}
			pIn++;
		}
	}
}

// virtual
void GLayerConvolutional1D::activate()
{
	double* pAct = activation();
	size_t outputCount = outputs();
	double* pNet = net();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
}

// virtual
void GLayerConvolutional1D::dropOut(GRand& rand, double probOfDrop)
{
	double* pAct = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			pAct[i] = 0.0;
	}
}

// virtual
void GLayerConvolutional1D::dropConnect(GRand& rand, double probOfDrop)
{
	throw Ex("Sorry, convolutional layers do not support dropConnect");
}

// virtual
void GLayerConvolutional1D::computeError(const double* pTarget)
{
	size_t outputUnits = outputs();
	double* pAct = activation();
	double* pErr = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(*pTarget == UNKNOWN_REAL_VALUE)
			*pErr = 0.0;
		else
			*pErr = *pTarget - *pAct;
		pTarget++;
		pAct++;
		pErr++;
	}
}

// virtual
void GLayerConvolutional1D::deactivateError()
{
	size_t outputUnits = outputs();
	double* pErr = error();
	double* pNet = net();
	double* pAct = activation();
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct);
		pNet++;
		pAct++;
		pErr++;
	}
}

// virtual
void GLayerConvolutional1D::backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
{
	GAssert(inputStart == 0);
	GAssert(pUpStreamLayer->outputs() == inputs());
	double* pUpStreamErr = pUpStreamLayer->error();
	double* pDownStreamErr = error();
	size_t kernelSize = m_kernels.cols();
	GVec::setAll(pUpStreamErr, 0.0, inputs());
	for(size_t i = 0; i < m_outputSamples; i++) // for each sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				double* pW = m_kernels[kern++];
				double* pUp = pUpStreamErr;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
				{
					(*pUp) += *(pW++) * *pDownStreamErr;
					pUp += m_inputChannels;
				}
				pDownStreamErr++;
			}
			pUpStreamErr++;
		}
	}
}

// virtual
void GLayerConvolutional1D::updateBias(double learningRate, double momentum)
{
	double* pErr = error();
	for(size_t i = 0; i < m_outputSamples; i++)
	{
		double* pB = bias();
		for(size_t j = 0; j < m_inputChannels; j++)
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++)
				*(pB++) += learningRate * *(pErr++);
		}
	}
}

// virtual
void GLayerConvolutional1D::updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	GAssert(inputStart == 0);
	GAssert(inputCount == inputs());
	double* pErr = error();
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_outputSamples; i++) // for each sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				double* pW = m_kernels[kern++];
				const double* pIn = pUpStreamActivation;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
				{
					*(pW++) += learningRate * *pErr * *pIn;
					pIn += m_inputChannels;
				}
				pErr ++;
			}
			pUpStreamActivation++;
		}
	}
}

void GLayerConvolutional1D::updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	throw Ex("Sorry, convolutional layers do not support dropConnect");
}

// virtual
void GLayerConvolutional1D::scaleWeights(double factor, bool scaleBiases)
{
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::multiply(m_kernels[i], factor, kernelSize);
	if(scaleBiases)
		GVec::multiply(bias(), factor, m_kernels.rows());
}

// virtual
void GLayerConvolutional1D::diminishWeights(double amount, bool diminishBiases)
{
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::regularize_1(m_kernels[i], amount, kernelSize);
	if(diminishBiases)
		GVec::regularize_1(bias(), amount, m_kernels.rows());
}

// virtual
size_t GLayerConvolutional1D::countWeights()
{
	return (m_kernels.rows() + 1) * m_kernels.cols();
}

// virtual
size_t GLayerConvolutional1D::weightsToVector(double* pOutVector)
{
	GVec::copy(pOutVector, bias(), m_kernels.rows());
	pOutVector += m_kernels.rows();
	m_kernels.toVector(pOutVector);
	return (m_kernels.rows() + 1) * m_kernels.cols();
}

// virtual
size_t GLayerConvolutional1D::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, m_kernels.rows());
	pVector += m_kernels.rows();
	m_kernels.fromVector(pVector, m_kernels.rows());
	return (m_kernels.rows() + 1) * m_kernels.cols();
}

// virtual
void GLayerConvolutional1D::copyWeights(GNeuralNetLayer* pSource)
{
	GLayerConvolutional1D* src = (GLayerConvolutional1D*)pSource;
	m_kernels.copyBlock(src->m_kernels, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	GVec::copy(bias(), src->bias(), src->m_kernels.rows());
}

// virtual
void GLayerConvolutional1D::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	if(start != 0)
		throw Ex("Sorry, convolutional layers do not support perturbing weights for a subset of units");
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::perturb(m_kernels[i], deviation, kernelSize, rand);
	GVec::perturb(bias(), deviation, m_kernels.rows(), rand);
}

// virtual
void GLayerConvolutional1D::maxNorm(double max)
{
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
	{
		GVec::capValues(m_kernels[i], max, kernelSize);
		GVec::floorValues(m_kernels[i], -max, kernelSize);
	}
}

// virtual
double GLayerConvolutional1D::unitIncomingWeightsL1Norm(size_t unit)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
double GLayerConvolutional1D::unitIncomingWeightsL2Norm(size_t unit)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
double GLayerConvolutional1D::unitOutgoingWeightsL1Norm(size_t input)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
double GLayerConvolutional1D::unitOutgoingWeightsL2Norm(size_t input)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
void GLayerConvolutional1D::scaleUnitIncomingWeights(size_t unit, double scalar)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
void GLayerConvolutional1D::scaleUnitOutgoingWeights(size_t input, double scalar)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
void GLayerConvolutional1D::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}





GLayerConvolutional2D::GLayerConvolutional2D(size_t inputCols, size_t inputRows, size_t inputChannels, size_t kernelSize, size_t kernelsPerChannel, GActivationFunction* pActivationFunction)
: m_inputCols(inputCols),
m_inputRows(inputRows),
m_inputChannels(inputChannels),
m_outputCols(inputCols - kernelSize + 1),
m_outputRows(inputRows - kernelSize + 1),
m_kernelsPerChannel(kernelsPerChannel),
m_kernelCount(inputChannels * kernelsPerChannel),
m_kernels(m_kernelCount * kernelSize, kernelSize),
m_delta(m_kernelCount * kernelSize, kernelSize),
m_bias(2, m_kernelCount)
{
	if(kernelSize > inputCols)
		throw Ex("kernelSize must be <= inputCols");
	if(kernelSize > inputRows)
		throw Ex("kernelSize must be <= inputRows");
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationLogistic();
	size_t totalOutputs = m_kernelCount * m_outputCols * m_outputRows;
	m_activation.resize(3, totalOutputs);
	m_delta.setAll(0.0);
	GVec::setAll(biasDelta(), 0.0, m_kernelCount);
}

GLayerConvolutional2D::GLayerConvolutional2D(GDomNode* pNode)
: m_inputCols(pNode->field("icol")->asInt()),
m_inputRows(pNode->field("irow")->asInt()),
m_inputChannels(pNode->field("ichan")->asInt()),
m_outputCols(pNode->field("ocol")->asInt()),
m_outputRows(pNode->field("orow")->asInt()),
m_kernelsPerChannel(pNode->field("kpc")->asInt()),
m_kernelCount(m_inputChannels * m_kernelsPerChannel),
m_kernels(pNode->field("kern")),
m_delta(pNode->field("delt")),
m_activation(pNode->field("act")),
m_pActivationFunction(GActivationFunction::deserialize(pNode->field("act_func")))
{
}

GLayerConvolutional2D::~GLayerConvolutional2D()
{
}

// virtual
GDomNode* GLayerConvolutional2D::serialize(GDom* pDoc)
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "icol", pDoc->newInt(m_inputCols));
	pNode->addField(pDoc, "irow", pDoc->newInt(m_inputRows));
	pNode->addField(pDoc, "ichan", pDoc->newInt(m_inputChannels));
	pNode->addField(pDoc, "ocol", pDoc->newInt(m_outputCols));
	pNode->addField(pDoc, "orow", pDoc->newInt(m_outputRows));
	pNode->addField(pDoc, "kpc", pDoc->newInt(m_kernelsPerChannel));
	pNode->addField(pDoc, "kern", m_kernels.serialize(pDoc));
	pNode->addField(pDoc, "delt", m_delta.serialize(pDoc));
	pNode->addField(pDoc, "act", m_activation.serialize(pDoc));
	pNode->addField(pDoc, "bias", m_bias.serialize(pDoc));
	pNode->addField(pDoc, "act_func", m_pActivationFunction->serialize(pDoc));
	return pNode;
}

// virtual
void GLayerConvolutional2D::resize(size_t inputs, size_t outputs, GRand* pRand, double deviation)
{
	if(inputs != m_inputCols * m_inputRows * m_inputChannels)
		throw Ex("Changing the size of GLayerConvolutional2D is not supported");
	if(outputs != m_inputChannels * m_kernelsPerChannel * m_outputCols * m_outputRows)
		throw Ex("Changing the size of GLayerConvolutional2D is not supported");
}

// virtual
void GLayerConvolutional2D::resetWeights(GRand& rand)
{
	size_t kernelSize = m_kernels.cols();
	double mag = std::max(0.03, 1.0 / (kernelSize * kernelSize));
	for(size_t i = 0; i < m_kernels.rows(); i++)
	{
		double* pW = m_kernels[i];
		for(size_t j = 0; j < kernelSize; j++)
			*(pW++) = rand.normal() * mag;
	}
	m_delta.setAll(0.0);
	double* pB = bias();
	for(size_t i = 0; i < m_kernelCount; i++)
		*(pB++) = rand.normal() * mag;
	GVec::setAll(biasDelta(), 0.0, m_kernelCount);
}

// virtual
void GLayerConvolutional2D::copyBiasToNet()
{
	double* pNet = net();
	double* pBias = bias();
	for(size_t j = 0; j < m_outputRows; j++)
	{
		for(size_t i = 0; i < m_outputCols; i++)
		{
			GVec::copy(pNet, pBias, m_kernelCount);
			pNet += m_kernelCount;
		}
	}
}

// virtual
void GLayerConvolutional2D::feedIn(const double* pIn, size_t inputStart, size_t inputCount)
{
	GAssert(inputStart == 0);
	if(inputCount != m_inputRows * m_inputCols * m_inputChannels)
		throw Ex("Sorry, partial feeding not supported in GLayerConvolutional2D");
	size_t kernelSize = m_kernels.cols();
	double* pNet = net();
	for(size_t h = 0; h < m_outputRows; h++) // for each output row...
	{
		for(size_t i = 0; i < m_outputCols; i++) // for each output column...
		{
			size_t kern = 0;
			for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
				{
					double d = 0.0;
					const double* pInput = pIn;
					for(size_t l = 0; l < kernelSize; l++) // for each kernel row...
					{
						double* pW = m_kernels[kern++];
						for(size_t m = 0; m < kernelSize; m++) // for each kernel column... todo: the cyclomatic complexity of this method is ridiculous!
						{
							d += *(pW++) * *pInput;
							pInput += m_inputChannels;
						}
					}
					*(pNet++) += d;
				}
				pIn++;
			}
		}
	}
}

// virtual
void GLayerConvolutional2D::activate()
{
	double* pAct = activation();
	size_t outputCount = outputs();
	double* pNet = net();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
}

// virtual
void GLayerConvolutional2D::dropOut(GRand& rand, double probOfDrop)
{
	double* pAct = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			pAct[i] = 0.0;
	}
}

// virtual
void GLayerConvolutional2D::dropConnect(GRand& rand, double probOfDrop)
{
	throw Ex("Sorry, convolutional layers do not support dropConnect");
}

// virtual
void GLayerConvolutional2D::computeError(const double* pTarget)
{
	size_t outputUnits = outputs();
	double* pAct = activation();
	double* pErr = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(*pTarget == UNKNOWN_REAL_VALUE)
			*pErr = 0.0;
		else
			*pErr = *pTarget - *pAct;
		pTarget++;
		pAct++;
		pErr++;
	}
}

// virtual
void GLayerConvolutional2D::deactivateError()
{
	size_t outputUnits = outputs();
	double* pErr = error();
	double* pNet = net();
	double* pAct = activation();
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct);
		pNet++;
		pAct++;
		pErr++;
	}
}

// virtual
void GLayerConvolutional2D::backPropError(GNeuralNetLayer* pUpStreamLayer, size_t inputStart)
{
	GAssert(inputStart == 0);
	GAssert(pUpStreamLayer->outputs() == inputs());
	double* pUpStreamErr = pUpStreamLayer->error();
	double* pDownStreamErr = error();
	size_t kernelSize = m_kernels.cols();
	GVec::setAll(pUpStreamErr, 0.0, inputs());
	for(size_t h = 0; h < m_outputRows; h++) // for each output row...
	{
		for(size_t i = 0; i < m_outputCols; i++) // for each output column...
		{
			size_t kern = 0;
			for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
				{
					double* pUp = pUpStreamErr;
					for(size_t l = 0; l < kernelSize; l++) // for each kernel row...
					{
						double* pW = m_kernels[kern++];
						for(size_t m = 0; m < kernelSize; m++) // for each kernel column...
						{
							(*pUp) += *(pW++) * *pDownStreamErr;
							pUp += m_inputChannels;
						}
					}
					pDownStreamErr++;
				}
				pUpStreamErr++;
			}
		}
	}
}

// virtual
void GLayerConvolutional2D::updateBias(double learningRate, double momentum)
{
	double* pErr = error();
	for(size_t h = 0; h < m_outputRows; h++)
	{
		for(size_t i = 0; i < m_outputCols; i++)
		{
			double* pB = bias();
			for(size_t j = 0; j < m_inputChannels; j++)
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++)
					*(pB++) += learningRate * *(pErr++);
			}
		}
	}
}

// virtual
void GLayerConvolutional2D::updateWeights(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	GAssert(inputStart == 0);
	GAssert(inputCount == inputs());
	double* pErr = error();
	size_t kernelSize = m_kernels.cols();
	for(size_t h = 0; h < m_outputRows; h++) // for each sample row...
	{
		for(size_t i = 0; i < m_outputCols; i++) // for each sample column...
		{
			size_t kern = 0;
			for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
				{
					const double* pIn = pUpStreamActivation;
					for(size_t l = 0; l < kernelSize; l++) // for each kernel row...
					{
						double* pW = m_kernels[kern++];
						for(size_t m = 0; m < kernelSize; m++) // for each kernel column...
						{
							*(pW++) += learningRate * *pErr * *pIn;
							pIn += m_inputChannels;
						}
					}
					pErr++;
				}
				pUpStreamActivation++;
			}
		}
	}
}

void GLayerConvolutional2D::updateWeightsAndRestoreDroppedOnes(const double* pUpStreamActivation, size_t inputStart, size_t inputCount, double learningRate, double momentum)
{
	throw Ex("Sorry, convolutional layers do not support dropConnect");
}

// virtual
void GLayerConvolutional2D::scaleWeights(double factor, bool scaleBiases)
{
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::multiply(m_kernels[i], factor, kernelSize);
	if(scaleBiases)
		GVec::multiply(bias(), factor, m_kernelCount);
}

// virtual
void GLayerConvolutional2D::diminishWeights(double amount, bool diminishBiases)
{
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::regularize_1(m_kernels[i], amount, kernelSize);
	if(diminishBiases)
		GVec::regularize_1(bias(), amount, m_kernelCount);
}

// virtual
size_t GLayerConvolutional2D::countWeights()
{
	return m_kernels.rows() * m_kernels.cols() + m_kernelCount;
}

// virtual
size_t GLayerConvolutional2D::weightsToVector(double* pOutVector)
{
	GVec::copy(pOutVector, bias(), m_kernelCount);
	pOutVector += m_kernelCount;
	m_kernels.toVector(pOutVector);
	return m_kernels.rows() * m_kernels.cols() + m_kernelCount;
}

// virtual
size_t GLayerConvolutional2D::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, m_kernelCount);
	pVector += m_kernelCount;
	m_kernels.fromVector(pVector, m_kernels.rows());
	return m_kernels.rows() * m_kernels.cols() + m_kernelCount;
}

// virtual
void GLayerConvolutional2D::copyWeights(GNeuralNetLayer* pSource)
{
	GLayerConvolutional2D* src = (GLayerConvolutional2D*)pSource;
	m_kernels.copyBlock(src->m_kernels, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	GVec::copy(bias(), src->bias(), src->m_kernelCount);
}

// virtual
void GLayerConvolutional2D::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	if(start != 0)
		throw Ex("Sorry, convolutional layers do not support perturbing weights for a subset of units");
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::perturb(m_kernels[i], deviation, kernelSize, rand);
	GVec::perturb(bias(), deviation, m_kernelCount, rand);
}

// virtual
void GLayerConvolutional2D::maxNorm(double max)
{
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
	{
		GVec::capValues(m_kernels[i], max, kernelSize);
		GVec::floorValues(m_kernels[i], -max, kernelSize);
	}
}

// virtual
double GLayerConvolutional2D::unitIncomingWeightsL1Norm(size_t unit)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
double GLayerConvolutional2D::unitIncomingWeightsL2Norm(size_t unit)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
double GLayerConvolutional2D::unitOutgoingWeightsL1Norm(size_t input)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
double GLayerConvolutional2D::unitOutgoingWeightsL2Norm(size_t input)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
void GLayerConvolutional2D::scaleUnitIncomingWeights(size_t unit, double scalar)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
void GLayerConvolutional2D::scaleUnitOutgoingWeights(size_t input, double scalar)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}

// virtual
void GLayerConvolutional2D::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}






GNeuralNet::GNeuralNet()
: GIncrementalLearner(),
m_learningRate(0.1),
m_momentum(0.0),
m_validationPortion(0.35),
m_minImprovement(0.002),
m_epochsPerValidationCheck(100),
m_useInputBias(false)
{
}

GNeuralNet::GNeuralNet(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalLearner(pNode, ll),
m_validationPortion(0.35),
m_minImprovement(0.002),
m_epochsPerValidationCheck(100)
{
	m_learningRate = pNode->field("learningRate")->asDouble();
	m_momentum = pNode->field("momentum")->asDouble();
	m_useInputBias = pNode->field("ib")->asBool();

	// Create the layers
	GDomListIterator it1(pNode->field("layers"));
	while(it1.remaining() > 0)
	{
		m_layers.push_back(GNeuralNetLayer::deserialize(it1.current()));
		it1.advance();
	}
}

GNeuralNet::~GNeuralNet()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
	m_layers.clear();
}

void GNeuralNet::clear()
{
	// Don't delete the layers here, because their topology will affect future training
}

#ifndef MIN_PREDICT
// virtual
GDomNode* GNeuralNet::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNeuralNet");

	// Add the layers
	GDomNode* pLayerList = pNode->addField(pDoc, "layers", pDoc->newList());
	for(size_t i = 0; i < m_layers.size(); i++)
		pLayerList->addItem(pDoc, m_layers[i]->serialize(pDoc));

	// Add other settings
	pNode->addField(pDoc, "learningRate", pDoc->newDouble(m_learningRate));
	pNode->addField(pDoc, "momentum", pDoc->newDouble(m_momentum));
	pNode->addField(pDoc, "ib", pDoc->newBool(m_useInputBias));

	return pNode;
}
#endif // MIN_PREDICT

// virtual
bool GNeuralNet::supportedFeatureRange(double* pOutMin, double* pOutMax)
{
	*pOutMin = -1.0;
	*pOutMax = 1.0;
	return false;
}

// virtual
bool GNeuralNet::supportedLabelRange(double* pOutMin, double* pOutMax)
{
	if(m_layers.size() > 0)
	{
		GActivationFunction* pAct = ((GLayerClassic*)&outputLayer())->m_pActivationFunction; // TODO: This is a HACK
		double hr = pAct->halfRange();
		if(hr >= 1e50)
			return true;
		double c = pAct->center();
		*pOutMin = c - hr;
		*pOutMax = c + hr;
	}
	else
	{
		// Assume the tanh function is the default
		*pOutMin = -1.0;
		*pOutMax = 1.0;
	}
	return false;
}

size_t GNeuralNet::countWeights(size_t layer) const
{
	size_t wc = 0;
	wc += m_layers[layer]->countWeights();
	return wc;
}

size_t GNeuralNet::countWeights() const
{
	size_t wc = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
		wc += m_layers[i]->countWeights();
	return wc;
}

void GNeuralNet::weights(double* pOutWeights, size_t layer) const
{
	pOutWeights += m_layers[layer]->weightsToVector(pOutWeights);
}

void GNeuralNet::weights(double* pOutWeights) const
{
	for(size_t i = 0; i < m_layers.size(); i++)
		pOutWeights += m_layers[i]->weightsToVector(pOutWeights);
}

void GNeuralNet::setWeights(const double* pWeights, size_t layer)
{
	pWeights += m_layers[layer]->vectorToWeights(pWeights);
}

void GNeuralNet::setWeights(const double* pWeights)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		pWeights += m_layers[i]->vectorToWeights(pWeights);
}

void GNeuralNet::copyWeights(GNeuralNet* pOther)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->copyWeights(pOther->m_layers[i]);
}

void GNeuralNet::copyStructure(GNeuralNet* pOther)
{
	clear();
	delete(m_pRelFeatures);
	m_pRelFeatures = pOther->m_pRelFeatures->cloneMinimal();
	delete(m_pRelLabels);
	m_pRelLabels = pOther->m_pRelLabels->cloneMinimal();
	for(size_t i = 0; i < pOther->m_layers.size(); i++)
	{
		// todo: this is not a very efficient way to copy a layer
		GDom doc;
		GDomNode* pNode = pOther->m_layers[i]->serialize(&doc);
		m_layers.push_back(GNeuralNetLayer::deserialize(pNode));
	}
	m_learningRate = pOther->m_learningRate;
	m_momentum = pOther->m_momentum;
	m_validationPortion = pOther->m_validationPortion;
	m_minImprovement = pOther->m_minImprovement;
	m_epochsPerValidationCheck = pOther->m_epochsPerValidationCheck;
	m_useInputBias = pOther->m_useInputBias;
}

void GNeuralNet::perturbAllWeights(double deviation)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->perturbWeights(m_rand, deviation);
}

void GNeuralNet::maxNorm(double max)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->maxNorm(max);
}

void GNeuralNet::invertNode(size_t layer, size_t node)
{
	GLayerClassic& layerUpStream = *(GLayerClassic*)m_layers[layer];
	GMatrix& w = layerUpStream.m_weights;
	for(size_t i = 0; i < w.rows(); i++)
		w[i][node] = -w[i][node];
	layerUpStream.bias()[node] = -layerUpStream.bias()[node];
	if(layer + 1 < m_layers.size())
	{
		GLayerClassic& layerDownStream = *(GLayerClassic*)m_layers[layer + 1];
		GActivationFunction* pActFunc = layerDownStream.m_pActivationFunction;
		size_t downOuts = layerDownStream.outputs();
		double* pW = layerDownStream.m_weights[node];
		double* pB = layerDownStream.bias();
		for(size_t i = 0; i < downOuts; i++)
		{
			pB[i] += 2 * pActFunc->center() * pW[i];
			pW[i] = -pW[i];
		}
	}
}

void GNeuralNet::swapNodes(size_t layer, size_t a, size_t b)
{
	GLayerClassic& layerUpStream = *(GLayerClassic*)m_layers[layer];
	layerUpStream.m_weights.swapColumns(a, b);
	std::swap(layerUpStream.bias()[a], layerUpStream.bias()[b]);
	if(layer + 1 < m_layers.size())
	{
		GLayerClassic& layerDownStream = *(GLayerClassic*)m_layers[layer + 1];
		layerDownStream.m_weights.swapRows(a, b);
	}
}

void GNeuralNet::addLayer(GNeuralNetLayer* pLayer, size_t position)
{
	if(position == INVALID_INDEX)
		position = m_layers.size();
	if(position > m_layers.size())
		throw Ex("Invalid layer position");
	if(position > 0)
	{
		if(m_layers[position - 1]->outputs() != pLayer->inputs())
		{
			if(pLayer->inputs() == FLEXIBLE_SIZE)
			{
				if(m_layers[position - 1]->outputs() == FLEXIBLE_SIZE)
					throw Ex("Two FLEXIBLE_SIZE ends cannot be connected");
				pLayer->resize(m_layers[position - 1]->outputs(), pLayer->outputs());
			}
			else if(m_layers[position - 1]->outputs() == FLEXIBLE_SIZE)
				m_layers[position - 1]->resize(m_layers[position - 1]->inputs(), pLayer->inputs());
			else
				throw Ex("Mismatching layers. The previous layer outputs ", to_str(m_layers[position - 1]->outputs()), " values. The added layer inputs ", to_str(pLayer->inputs()));
		}
	}
	if(position < m_layers.size())
	{
		if(m_layers[position]->inputs() != pLayer->outputs())
		{
			if(pLayer->outputs() == FLEXIBLE_SIZE)
			{
				if(m_layers[position]->inputs() == FLEXIBLE_SIZE)
					throw Ex("Two FLEXIBLE_SIZE ends cannot be connected");
				pLayer->resize(pLayer->inputs(), m_layers[position]->inputs());
			}
			else if(m_layers[position]->inputs() == FLEXIBLE_SIZE)
				m_layers[position]->resize(pLayer->outputs(), m_layers[position]->outputs());
			else
				throw Ex("Mismatching layers. The next layer inputs ", to_str(m_layers[position]->inputs()), " values. The added layer outputs ", to_str(pLayer->outputs()));
		}
	}
	m_layers.insert(m_layers.begin() + position, pLayer);
}

GNeuralNetLayer* GNeuralNet::releaseLayer(size_t index)
{
	GNeuralNetLayer* pLayer = m_layers[index];
	m_layers.erase(m_layers.begin() + index);
	return pLayer;
}

#ifndef MIN_PREDICT
void GNeuralNet::align(const GNeuralNet& that)
{
	if(layerCount() != that.layerCount())
		throw Ex("mismatching number of layers");
	for(size_t i = 0; i + 1 < m_layers.size(); i++)
	{
		// Copy weights into matrices
		GLayerClassic& layerThisCur = *(GLayerClassic*)m_layers[i];

		GLayerClassic& layerThatCur = *(GLayerClassic*)that.m_layers[i];
		if(layerThisCur.outputs() != layerThatCur.outputs())
			throw Ex("mismatching layer size");

		GMatrix costs(layerThisCur.outputs(), layerThatCur.outputs());
		for(size_t k = 0; k < layerThisCur.outputs(); k++)
		{
			for(size_t j = 0; j < layerThatCur.outputs(); j++)
			{
				double d = layerThisCur.bias()[k] - layerThatCur.bias()[j];
				double pos = d * d;
				d = layerThisCur.bias()[k] + layerThatCur.bias()[j];
				double neg = d * d;
				GMatrix& wThis = layerThisCur.m_weights;
				const GMatrix& wThat = layerThatCur.m_weights;
				for(size_t l = 0; l < layerThisCur.inputs(); l++)
				{
					d = wThis[l][k] - wThat[l][j];
					pos += (d * d);
					d = wThis[l][k] + wThat[l][j];
					neg += (d * d);
				}
				costs[j][k] = std::min(pos, neg);
			}
		}
		GSimpleAssignment indexes = linearAssignment(costs);

		// Align this layer with that layer
		for(size_t j = 0; j < layerThisCur.outputs(); j++)
		{
			size_t k = (size_t)indexes((unsigned int)j);
			if(k != j)
			{
				// Fix up the indexes
				size_t m = j + 1;
				for( ; m < layerThisCur.outputs(); m++)
				{
					if((size_t)indexes((unsigned int)m) == j)
						break;
				}
				GAssert(m < layerThisCur.outputs());
				indexes.assign((unsigned int)m, (unsigned int)k);

				// Swap nodes j and k
				swapNodes(i, j, k);
			}

			// Test whether not j needs to be inverted by computing the dot product of the two weight vectors
			double dp = 0.0;
			size_t inputs = layerThisCur.inputs();
			for(size_t k = 0; k < inputs; k++)
				dp += layerThisCur.m_weights[k][j] * layerThatCur.m_weights[k][j];
			dp += layerThisCur.bias()[j] * layerThatCur.bias()[j];
			if(dp < 0)
				invertNode(i, j); // invert it
		}
	}
}

void GNeuralNet::scaleWeights(double factor, bool scaleBiases, size_t startLayer, size_t layerCount)
{
	size_t end = std::min(startLayer + layerCount, m_layers.size());
	for(size_t i = startLayer; i < end; i++)
		m_layers[i]->scaleWeights(factor, scaleBiases);
}

void GNeuralNet::diminishWeights(double amount, bool diminishBiases, size_t startLayer, size_t layerCount)
{
	size_t end = std::min(startLayer + layerCount, m_layers.size());
	for(size_t i = startLayer; i < end; i++)
		m_layers[i]->diminishWeights(amount, diminishBiases);
}

void GNeuralNet::scaleWeightsSingleOutput(size_t output, double factor)
{
	size_t lay = m_layers.size() - 1;
	GMatrix& m = ((GLayerClassic*)m_layers[lay])->m_weights;
	GAssert(output < m.cols());
	for(size_t i = 0; i < m.rows(); i++)
		m[i][output] *= factor;
	((GLayerClassic*)m_layers[lay])->bias()[output] *= factor;
	for(lay--; lay < m_layers.size(); lay--)
	{
		GMatrix& m = ((GLayerClassic*)m_layers[lay])->m_weights;
		size_t outputs = m.cols();
		for(size_t i = 0; i < m.rows(); i++)
			GVec::multiply(m[i], factor, outputs);
		GVec::multiply(((GLayerClassic*)m_layers[lay])->bias(), factor, outputs);
	}
}

void GNeuralNet::contractWeights(double factor, bool contractBiases)
{
	size_t i = m_layers.size() - 1;
	GNeuralNetLayer* pLay = m_layers[i];
	GVec::setAll(pLay->error(), 1.0, pLay->outputs());
	pLay->deactivateError();
	while(i > 0)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		pUpStream->deactivateError();
		((GLayerClassic*)pLay)->contractWeights(factor, contractBiases);
		pLay = pUpStream;
		i--;
	}
	((GLayerClassic*)pLay)->contractWeights(factor, contractBiases);
}
#endif // MIN_PREDICT

void GNeuralNet::bleedWeightsL1(double beta)
{
	for(size_t i = m_layers.size() - 2; i < m_layers.size(); i--)
	{
		size_t layerSize = m_layers[i]->outputs();
		for(size_t j = 0; j < layerSize; j++)
		{
			double sawDownStream = m_layers[i + 1]->unitOutgoingWeightsL1Norm(j);
			double sawUpStream = m_layers[i]->unitIncomingWeightsL1Norm(j);
			double sawAverage = 0.5 * (sawDownStream + sawUpStream);
			double sawNewDownStream = beta * sawAverage + (1.0 - beta) * sawDownStream;
			double sawNewUpStream = beta * sawAverage + (1.0 - beta) * sawUpStream;
			double facDownStream = sawNewDownStream / sawDownStream;
			double facUpStream = sawNewUpStream / sawUpStream;
			m_layers[i + 1]->scaleUnitOutgoingWeights(j, facDownStream);
			m_layers[i]->scaleUnitIncomingWeights(j, facUpStream);
		}
	}
}

void GNeuralNet::bleedWeightsL2(double beta)
{
	for(size_t i = m_layers.size() - 2; i < m_layers.size(); i--)
	{
		size_t layerSize = m_layers[i]->outputs();
		for(size_t j = 0; j < layerSize; j++)
		{
			double sswDownStream = m_layers[i + 1]->unitOutgoingWeightsL2Norm(j);
			double sswUpStream = m_layers[i]->unitIncomingWeightsL2Norm(j);
			double sswAverage = 0.5 * (sswDownStream + sswUpStream);
			double sswNewDownStream = beta * sswAverage + (1.0 - beta) * sswDownStream;
			double sswNewUpStream = beta * sswAverage + (1.0 - beta) * sswUpStream;
			double facDownStream = sqrt(sswNewDownStream) / sqrt(sswDownStream);
			double facUpStream = sqrt(sswNewUpStream) / sqrt(sswUpStream);
			m_layers[i + 1]->scaleUnitOutgoingWeights(j, facDownStream);
			m_layers[i]->scaleUnitIncomingWeights(j, facUpStream);
		}
	}
}

void GNeuralNet::forwardProp(const double* pRow, size_t maxLayers)
{
	GNeuralNetLayer* pLay = m_layers[0];
	if(!pLay)
		throw Ex("No layers have been added to this neural network");
	if(m_useInputBias)
		((GLayerClassic*)pLay)->feedForwardWithInputBias(pRow);
	else
	{
		pLay->copyBiasToNet();
		pLay->feedIn(pRow, 0, pLay->inputs());
		pLay->activate();
	}
	maxLayers = std::min(m_layers.size(), maxLayers);
	for(size_t i = 1; i < maxLayers; i++)
	{
		GNeuralNetLayer* pDS = m_layers[i];
		pDS->copyBiasToNet();
		pDS->feedIn(pLay, 0);
		pDS->activate();
		pLay = pDS;
	}
}

double GNeuralNet::forwardPropSingleOutput(const double* pRow, size_t output)
{
	if(m_layers.size() == 1)
	{
		GLayerClassic& layer = *(GLayerClassic*)m_layers[0];
		layer.feedForwardToOneOutput(pRow, output, m_useInputBias);
		return layer.activation()[output];
	}
	else
	{
		GLayerClassic* pLay = (GLayerClassic*)m_layers[0];
		if(m_useInputBias)
			pLay->feedForwardWithInputBias(pRow);
		else
			pLay->feedForward(pRow);
		for(size_t i = 1; i + 1 < m_layers.size(); i++)
		{
			GLayerClassic* pDS = (GLayerClassic*)m_layers[i];
			pDS->feedForward(pLay->activation());
			pLay = pDS;
		}
		GLayerClassic* pDS = (GLayerClassic*)m_layers[m_layers.size() - 1];
		pDS->feedForwardToOneOutput(pLay->activation(), output, false);
		return pDS->activation()[output];
	}
}

#ifndef MIN_PREDICT
// virtual
void GNeuralNet::predictDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, this model does not predict a distribution");
}
#endif // MIN_PREDICT

void GNeuralNet::copyPrediction(double* pOut)
{
	GNeuralNetLayer& outputLayer = *m_layers[m_layers.size() - 1];
	GVec::copy(pOut, outputLayer.activation(), outputLayer.outputs());
}

double GNeuralNet::sumSquaredPredictionError(const double* pTarget)
{
	GNeuralNetLayer& outputLayer = *m_layers[m_layers.size() - 1];
	return GVec::squaredDistance(pTarget, outputLayer.activation(), outputLayer.outputs());
}

// virtual
void GNeuralNet::predict(const double* pIn, double* pOut)
{
	forwardProp(pIn);
	copyPrediction(pOut);
}

// virtual
void GNeuralNet::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(!features.relation().areContinuous())
		throw Ex("GNeuralNet only supports continuous features. Perhaps you should wrap it in a GAutoFilter.");
	if(!labels.relation().areContinuous())
		throw Ex("GNeuralNet only supports continuous labels. Perhaps you should wrap it in a GAutoFilter.");
	size_t validationRows = (size_t)(m_validationPortion * features.rows());
	size_t trainRows = features.rows() - validationRows;
	if(validationRows > 0)
	{
		GDataRowSplitter splitter(features, labels, m_rand, trainRows);
		trainWithValidation(splitter.features1(), splitter.labels1(), splitter.features2(), splitter.labels2());
	}
	else
		trainWithValidation(features, labels, features, labels);
}

#ifndef MIN_PREDICT
// virtual
void GNeuralNet::trainSparse(GSparseMatrix& features, GMatrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	GUniformRelation featureRel(features.cols());
	beginIncrementalLearning(featureRel, labels.relation());

	GTEMPBUF(size_t, indexes, features.rows());
	GIndexVec::makeIndexVec(indexes, features.rows());
	GTEMPBUF(double, pFullRow, features.cols());
	for(size_t epochs = 0; epochs < 100; epochs++) // todo: need a better stopping criterion
	{
		GIndexVec::shuffle(indexes, features.rows(), &m_rand);
		for(size_t i = 0; i < features.rows(); i++)
		{
			features.fullRow(pFullRow, indexes[i]);
			forwardProp(pFullRow);
			backpropagate(labels.row(indexes[i]));
			descendGradient(pFullRow, m_learningRate, m_momentum);
		}
	}
}
#endif // MIN_PREDICT

double GNeuralNet::validationSquaredError(const GMatrix& features, const GMatrix& labels)
{
	double sse = 0;
	size_t nCount = features.rows();
	for(size_t n = 0; n < nCount; n++)
	{
		forwardProp(features[n]);
		sse += sumSquaredPredictionError(labels[n]);
	}
	return sse;
}

size_t GNeuralNet::trainWithValidation(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& validateFeatures, const GMatrix& validateLabels)
{
	if(trainFeatures.rows() != trainLabels.rows() || validateFeatures.rows() != validateLabels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	if(m_layers.size() == 0)
		throw Ex("At least one layer must be added to a neural network before it can be trained");
	beginIncrementalLearningInner(trainFeatures.relation(), trainLabels.relation());

	// Do the epochs
	size_t nEpochs;
	double dBestError = 1e308;
	size_t nEpochsSinceValidationCheck = 0;
	double dSumSquaredError;
	GRandomIndexIterator ii(trainFeatures.rows(), m_rand);
	for(nEpochs = 0; true; nEpochs++)
	{
		ii.reset();
		size_t index;
		while(ii.next(index))
			trainIncremental(trainFeatures[index], trainLabels[index]);

		// Check for termination condition
		if(nEpochsSinceValidationCheck >= m_epochsPerValidationCheck)
		{
			nEpochsSinceValidationCheck = 0;
			dSumSquaredError = validationSquaredError(validateFeatures, validateLabels);
			if(1.0 - dSumSquaredError / dBestError >= m_minImprovement) // This condition is designed such that if dSumSquaredError is NAN, it will break out of the loop
			{
				if(dSumSquaredError < dBestError)
					dBestError = dSumSquaredError;
			}
			else
				break;
		}
		else
			nEpochsSinceValidationCheck++;
	}

	return nEpochs;
}

// virtual
void GNeuralNet::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	if(labelRel.size() < 1)
		throw Ex("The label relation must have at least 1 attribute");

	// Resize the input and output layers to fit the data
	size_t inputs = featureRel.size() - (m_useInputBias ? 1 : 0);
	size_t outputs = labelRel.size();
	if(m_layers.size() == 1)
		m_layers[0]->resize(inputs, outputs);
	else if(m_layers.size() > 1)
	{
		m_layers[0]->resize(inputs, m_layers[0]->outputs());
		m_layers[m_layers.size() - 1]->resize(m_layers[m_layers.size() - 1]->inputs(), outputs);
	}

	// Reset the weights
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->resetWeights(m_rand);
}

// virtual
void GNeuralNet::trainIncremental(const double* pIn, const double* pOut)
{
	forwardProp(pIn);
	backpropagate(pOut);
	descendGradient(pIn, m_learningRate, m_momentum);
}

void GNeuralNet::trainIncrementalWithDropout(const double* pIn, const double* pOut, double probOfDrop)
{
	if(m_momentum != 0.0)
		throw Ex("Sorry, this implementation is not compatible with momentum");

	// Forward prop with dropout
	GNeuralNetLayer* pLay = m_layers[0];
	pLay->copyBiasToNet();
	pLay->feedIn(pIn, 0, pLay->inputs());
	pLay->activate();
	pLay->dropOut(m_rand, probOfDrop);
	size_t maxLayers = m_layers.size();
	for(size_t i = 1; i < maxLayers; i++)
	{
		GNeuralNetLayer* pDS = m_layers[i];
		pDS->copyBiasToNet();
		pDS->feedIn(pLay, 0);
		pDS->activate();
		if(i + 1 < maxLayers)
		{
			pLay->dropOut(m_rand, probOfDrop);
			pLay = pDS;
		}
	}

	backpropagate(pOut);
	descendGradient(pIn, m_learningRate, 0.0);
}

void GNeuralNet::trainIncrementalWithDropConnect(const double* pIn, const double* pOut, double probOfDrop)
{
	if(m_momentum != 0.0)
		throw Ex("Sorry, this implementation is not compatible with momentum");

	// Forward prop with dropConnect
	GNeuralNetLayer* pLay = m_layers[0];
	pLay->dropConnect(m_rand, probOfDrop);
	pLay->copyBiasToNet();
	pLay->feedIn(pIn, 0, pLay->inputs());
	pLay->activate();
	size_t maxLayers = m_layers.size();
	for(size_t i = 1; i < maxLayers; i++)
	{
		GNeuralNetLayer* pDS = m_layers[i];
		pDS->dropConnect(m_rand, probOfDrop);
		pDS->copyBiasToNet();
		pDS->feedIn(pLay, 0);
		pDS->activate();
		pLay = pDS;
	}

	backpropagate(pOut);

	// Descend gradient with dropConnect
	pLay = m_layers[0];
	pLay->updateWeightsAndRestoreDroppedOnes(pIn, 0, pLay->inputs(), m_learningRate, 0.0);
	pLay->updateBias(m_learningRate, 0.0);
	GNeuralNetLayer* pUpStream = pLay;
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		pLay = m_layers[i];
		pLay->updateWeightsAndRestoreDroppedOnes(pUpStream, 0, m_learningRate, 0.0);
		pLay->updateBias(m_learningRate, 0.0);
		pUpStream = pLay;
	}
}

void GNeuralNet::backpropagate(const double* pTarget, size_t startLayer)
{
	size_t i = std::min(startLayer, m_layers.size() - 1);
	GNeuralNetLayer* pLay = m_layers[i];
	pLay->computeError(pTarget);
	pLay->deactivateError();
	while(i > 0)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		pUpStream->deactivateError();
		pLay = pUpStream;
		i--;
	}
}

void GNeuralNet::backpropagateSingleOutput(size_t outputNode, double target, size_t startLayer)
{
	size_t i = std::min(startLayer, m_layers.size() - 1);
	GLayerClassic* pLay = (GLayerClassic*)m_layers[i];
	pLay->computeErrorSingleOutput(target, outputNode);
	pLay->deactivateErrorSingleOutput(outputNode);
	if(i > 0)
	{
		GLayerClassic* pUpStream = (GLayerClassic*)m_layers[i - 1];
		pLay->backPropErrorSingleOutput(outputNode, pUpStream->error());
		pUpStream->deactivateError();
		pLay = pUpStream;
		i--;
		while(i > 0)
		{
			GLayerClassic* pUpStream = (GLayerClassic*)m_layers[i - 1];
			pLay->backPropError(pUpStream);
			pUpStream->deactivateError();
			pLay = pUpStream;
			i--;
		}
	}
}

void GNeuralNet::descendGradient(const double* pFeatures, double learningRate, double momentum)
{
	GNeuralNetLayer* pLay = m_layers[0];
	pLay->updateWeights(pFeatures + (useInputBias() ? 1 : 0), 0, pLay->inputs(), learningRate, momentum);
	pLay->updateBias(learningRate, momentum);
	GNeuralNetLayer* pUpStream = pLay;
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		pLay = m_layers[i];
		pLay->updateWeights(pUpStream, 0, learningRate, momentum);
		pLay->updateBias(learningRate, momentum);
		pUpStream = pLay;
	}
}

void GNeuralNet::descendGradientSingleOutput(size_t outputNeuron, const double* pFeatures, double learningRate, double momentum)
{
	size_t i = m_layers.size() - 1;
	GLayerClassic* pLay = (GLayerClassic*)m_layers[i];
	if(i == 0)
		pLay->updateWeightsSingleNeuron(outputNeuron, pFeatures, learningRate, momentum);
	else
	{
		GLayerClassic* pUpStream = (GLayerClassic*)m_layers[i - 1];
		pLay->updateWeightsSingleNeuron(outputNeuron, pUpStream->activation(), learningRate, momentum);
		for(i--; i > 0; i--)
		{
			pLay = pUpStream;
			pUpStream = (GLayerClassic*)m_layers[i - 1];
			pLay->updateWeights(pUpStream->activation(), 0, pLay->inputs(), learningRate, momentum);
			pLay->updateBias(learningRate, momentum);
		}
		pLay = (GLayerClassic*)m_layers[0];
		pLay->updateWeights(pFeatures, 0, pLay->inputs(), learningRate, momentum);
		pLay->updateBias(learningRate, momentum);
	}
}

void GNeuralNet::gradientOfInputs(double* pOutGradient)
{
	GMatrix& w = ((GLayerClassic*)m_layers[0])->m_weights;
	size_t outputs = w.cols();
	double* pErr = ((GLayerClassic*)m_layers[0])->error();
	if(useInputBias())
		*(pOutGradient++) = -GVec::sumElements(pErr, outputs);
	for(size_t i = 0; i < w.rows(); i++)
		*(pOutGradient++) = -GVec::dotProduct(w[i], pErr, outputs);
}

void GNeuralNet::gradientOfInputsSingleOutput(size_t outputNeuron, double* pOutGradient)
{
	if(m_layers.size() != 1)
	{
		gradientOfInputs(pOutGradient);
		return;
	}
	GMatrix& w = ((GLayerClassic*)m_layers[0])->m_weights;
	GAssert(outputNeuron < w.cols());

	double* pErr = ((GLayerClassic*)m_layers[0])->error();
	if(useInputBias())
		*(pOutGradient++) = -pErr[outputNeuron];
	for(size_t i = 0; i < w.rows(); i++)
		*(pOutGradient++) = -pErr[outputNeuron] * w[i][outputNeuron];
}

#ifndef MIN_PREDICT
void GNeuralNet::autoTune(GMatrix& features, GMatrix& labels)
{
	// Try a plain-old single-layer network
	size_t hidden = std::max((size_t)4, (features.cols() + 3) / 4);
	Holder<GNeuralNet> hCand0(new GNeuralNet());
	Holder<GNeuralNet> hCand1;
	double scores[2];
	scores[0] = hCand0.get()->crossValidate(features, labels, 2);
	scores[1] = 1e308;

	// Try increasing the number of hidden units until accuracy decreases twice
	size_t failures = 0;
	while(true)
	{
		GNeuralNet* cand = new GNeuralNet();
		cand->addLayer(new GLayerClassic(FLEXIBLE_SIZE, hidden));
		cand->addLayer(new GLayerClassic(hidden, FLEXIBLE_SIZE));
		double d = cand->crossValidate(features, labels, 2);
		if(d < scores[0])
		{
			hCand1.reset(hCand0.release());
			scores[1] = scores[0];
			hCand0.reset(cand);
			scores[0] = d;
		}
		else
		{
			if(d < scores[1])
			{
				hCand1.reset(cand);
				scores[1] = d;
			}
			else
				delete(cand);
			if(++failures >= 2)
				break;
		}
		hidden *= 4;
	}

	// Try narrowing in on the best number of hidden units
	while(true)
	{
		size_t a = hCand0.get()->layerCount() > 1 ? hCand0.get()->layer(0).outputs() : 0;
		size_t b = hCand1.get()->layerCount() > 1 ? hCand1.get()->layer(0).outputs() : 0;
		size_t dif = b < a ? a - b : b - a;
		if(dif <= 1)
			break;
		size_t c = (a + b) / 2;
		GNeuralNet* cand = new GNeuralNet();
		cand->addLayer(new GLayerClassic(FLEXIBLE_SIZE, c));
		cand->addLayer(new GLayerClassic(c, FLEXIBLE_SIZE));
		double d = cand->crossValidate(features, labels, 2);
		if(d < scores[0])
		{
			hCand1.reset(hCand0.release());
			scores[1] = scores[0];
			hCand0.reset(cand);
			scores[0] = d;
		}
		else if(d < scores[1])
		{
			hCand1.reset(cand);
			scores[1] = d;
		}
		else
		{
			delete(cand);
			break;
		}
	}
	hCand1.reset(NULL);

	// Try two hidden layers
	size_t hu1 = hCand0.get()->layerCount() > 1 ? hCand0.get()->layer(0).outputs() : 0;
	size_t hu2 = 0;
	if(hu1 > 12)
	{
		size_t c1 = 16;
		size_t c2 = 16;
		if(labels.cols() < features.cols())
		{
			double d = sqrt(double(features.cols()) / labels.cols());
			c1 = std::max(size_t(9), size_t(double(features.cols()) / d));
			c2 = size_t(labels.cols() * d);
		}
		else
		{
			double d = sqrt(double(labels.cols()) / features.cols());
			c1 = size_t(features.cols() * d);
			c2 = std::max(size_t(9), size_t(double(labels.cols()) / d));
		}
		if(c1 < 16 && c2 < 16)
		{
			c1 = 16;
			c2 = 16;
		}
		GNeuralNet* cand = new GNeuralNet();
		vector<size_t> topology;
		cand->addLayer(new GLayerClassic(FLEXIBLE_SIZE, c1));
		cand->addLayer(new GLayerClassic(c1, c2));
		cand->addLayer(new GLayerClassic(c2, FLEXIBLE_SIZE));
		double d = cand->crossValidate(features, labels, 2);
		if(d < scores[0])
		{
			hCand0.reset(cand);
			scores[0] = d;
			hu1 = c1;
			hu2 = c2;
		}
		else
			delete(cand);
	}

	// Try with momentum
	{
		GNeuralNet* cand = new GNeuralNet();
		vector<size_t> topology;
		if(hu1 > 0) cand->addLayer(new GLayerClassic(FLEXIBLE_SIZE, hu1));
		if(hu2 > 0) cand->addLayer(new GLayerClassic(hu1, hu2));
		cand->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		cand->setMomentum(0.8);
		double d = cand->crossValidate(features, labels, 2);
		if(d < scores[0])
		{
			hCand0.reset(cand);
			scores[0] = d;
		}
		else
			delete(cand);
	}

	copyStructure(hCand0.get());
}
#endif // MIN_PREDICT

void GNeuralNet::pretrainWithAutoencoders(const GMatrix& features, size_t maxLayers)
{
	const GMatrix* pFeat = &features;
	Holder<GMatrix> hFeat(NULL);
	maxLayers = std::min(layerCount(), maxLayers);
	for(size_t i = 0; i < maxLayers; i++)
	{
		GNeuralNet tmp;
		GNeuralNetLayer& encoder = layer(i);
		tmp.addLayer(&encoder);
		GLayerClassic* pDecoder = new GLayerClassic(encoder.outputs(), encoder.inputs());
		tmp.addLayer(pDecoder);
		tmp.setWindowSize(1);
		tmp.setImprovementThresh(0.05);
		tmp.train(*pFeat, *pFeat);
		tmp.releaseLayer(0);
		if(i + 1 < maxLayers)
		{
			pFeat = encoder.feedThrough(*pFeat);
			hFeat.reset((GMatrix*)pFeat);
		}
	}
}

void GNeuralNet::printWeights(std::ostream& stream)
{
	stream.precision(6);
	stream << "Neural Network:\n";
	for(size_t i = layerCount() - 1; i < layerCount(); i--)
	{
		if(i == layerCount() - 1)
			stream << "	Output Layer:\n";
		else
			stream << "	Hidden Layer " << to_str(i) << ":\n";
		GNeuralNetLayer& l = layer(i);
		GLayerClassic& layer = *(GLayerClassic*)&l;
		for(size_t j = 0; j < layer.outputs(); j++)
		{
			stream << "		Unit " << to_str(j) << ":	";
			stream << "(bias: " << to_str(layer.bias()[j]) << ")	";
			for(size_t k = 0; k < layer.inputs(); k++)
			{
				if(k > 0)
					stream << "	";
				stream << to_str(layer.m_weights[k][j]);
			}
			stream << "\n";
		}
	}
}

void GNeuralNet::containIntrinsics(GMatrix& intrinsics)
{
	size_t dims = intrinsics.cols();
	GNeuralNetLayer& lay = layer(0);
	if(lay.inputs() != dims)
		throw Ex("Mismatching number of columns and inputs");
	GTEMPBUF(double, pCentroid, dims);
	intrinsics.centroid(pCentroid);
	double maxDev = 0.0;
	for(size_t i = 0; i < dims; i++)
	{
		double dev = sqrt(intrinsics.columnVariance(i, pCentroid[i]));
		maxDev = std::max(maxDev, dev);
		intrinsics.normalizeColumn(i, pCentroid[i] - dev, pCentroid[i] + dev, -1.0, 1.0);
		lay.renormalizeInput(i, pCentroid[i] - dev, pCentroid[i] + dev, -1.0, 1.0);
	}
}

GMatrix* GNeuralNet::compressFeatures(GMatrix& features)
{
	GLayerClassic& lay = *(GLayerClassic*)&layer(0);
	if(lay.inputs() != features.cols())
		throw Ex("mismatching number of data columns and layer units");
	GPCA pca(lay.inputs());
	pca.train(features);
	GVec off(lay.inputs());
	pca.basis()->multiply(pca.centroid(), off.v);
	GMatrix* pInvTransform = pca.basis()->pseudoInverse();
	Holder<GMatrix> hInvTransform(pInvTransform);
	lay.transformWeights(*pInvTransform, off.v);
	return pca.transformBatch(features);
}

// static
GNeuralNet* GNeuralNet::fourier(GMatrix& series, double period)
{
	if(!GBits::isPowerOfTwo((unsigned int)series.rows()))
		throw Ex("The time series data has ", to_str(series.rows()), " rows. Expected a power of 2.");

	// Make a neural network that combines sine units in the same manner as the Fourier transform
	GNeuralNet* pNN = new GNeuralNet();
	GLayerClassic* pLayerSin = new GLayerClassic(1, series.rows(), new GActivationSin());
	GLayerClassic* pLayerIdent = new GLayerClassic(FLEXIBLE_SIZE, series.cols(), new GActivationIdentity());
	pNN->addLayer(pLayerSin);
	pNN->addLayer(pLayerIdent);
	GUniformRelation relIn(1);
	GUniformRelation relOut(series.cols());
	pNN->beginIncrementalLearning(relIn, relOut);

	// Initialize the weights of the sine units to match the frequencies used by the Fourier transform.
	GMatrix& wSin = pLayerSin->weights();
	double* bSin = pLayerSin->bias();
	for(size_t i = 0; i < series.rows() / 2; i++)
	{
		wSin[0][2 * i] = 2.0 * M_PI * (i + 1) / period;
		bSin[2 * i] = 0.5 * M_PI;
		wSin[0][2 * i + 1] = 2.0 * M_PI * (i + 1) / period;
		bSin[2 * i + 1] = M_PI;
	}

	// Initialize the output layer
	struct ComplexNumber* pFourier = new struct ComplexNumber[series.rows()];
	ArrayHolder<struct ComplexNumber> hIn(pFourier);
	GMatrix& wIdent = pLayerIdent->weights();
	double* bIdent = pLayerIdent->bias();
	for(size_t j = 0; j < series.cols(); j++)
	{
		// Convert column j to the Fourier domain
		struct ComplexNumber* pF = pFourier;
		for(size_t i = 0; i < series.rows(); i++)
		{
			pF->real = series[i][j];
			pF->imag = 0.0;
			pF++;
		}
		GFourier::fft(series.rows(), pFourier, true);

		// Initialize the weights of the identity output units to combine the sine units with the weights
		// specified by the Fourier transform
		for(size_t i = 0; i < series.rows() / 2; i++)
		{
			wIdent[2 * i][j] = pFourier[1 + i].real / (series.rows() / 2);
			wIdent[2 * i + 1][j] = pFourier[1 + i].imag / (series.rows() / 2);
		}
		bIdent[j] = pFourier[0].real / (series.rows());

		// Compensate for the way the FFT doubles-up the values in the last complex element
		wIdent[series.rows() - 2][j] *= 0.5;
		wIdent[series.rows() - 1][j] *= 0.5;
	}

	return pNN;
}


#ifndef MIN_PREDICT
void GNeuralNet_testMath()
{
	GMatrix features(0, 2);
	double* pVec = features.newRow();
	pVec[0] = 0.0;
	pVec[1] = -0.7;
	GMatrix labels(0, 1);
	labels.newRow()[0] = 1.0;

	// Make the Neural Network
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 3));
	nn.addLayer(new GLayerClassic(3, FLEXIBLE_SIZE));
	nn.setLearningRate(0.175);
	nn.setMomentum(0.9);
	nn.beginIncrementalLearning(features.relation(), labels.relation());
	if(nn.countWeights() != 13)
		throw Ex("Wrong number of weights");
	GLayerClassic& layerOut = *(GLayerClassic*)&nn.layer(1);
	layerOut.bias()[0] = 0.02; // w_0
	layerOut.weights()[0][0] = -0.01; // w_1
	layerOut.weights()[1][0] = 0.03; // w_2
	layerOut.weights()[2][0] = 0.02; // w_3
	GLayerClassic& layerHidden = *(GLayerClassic*)&nn.layer(0);
	layerHidden.bias()[0] = -0.01; // w_4
	layerHidden.weights()[0][0] = -0.03; // w_5
	layerHidden.weights()[1][0] = 0.03; // w_6
	layerHidden.bias()[1] = 0.01; // w_7
	layerHidden.weights()[0][1] = 0.04; // w_8
	layerHidden.weights()[1][1] = -0.02; // w_9
	layerHidden.bias()[2] = -0.02; // w_10
	layerHidden.weights()[0][2] = 0.03; // w_11
	layerHidden.weights()[1][2] = 0.02; // w_12

	bool useCrossEntropy = false;

	// Test forward prop
	double tol = 1e-12;
	double pat[3];
	GVec::copy(pat, features[0], 2);
	nn.predict(pat, pat + 2);
	// Here is the math (done by hand) for why these results are expected:
	// Row: {0, -0.7, 1}
	// o_1 = squash(w_4*1+w_5*x+w_6*y) = 1/(1+exp(-(-.01*1-.03*0+.03*(-.7)))) = 0.4922506205862
	// o_2 = squash(w_7*1+w_8*x+w_9*y) = 1/(1+exp(-(.01*1+.04*0-.02*(-.7)))) = 0.50599971201659
	// o_3 = squash(w_10*1+w_11*x+w_12*y) = 1/(1+exp(-(-.02*1+.03*0+.02*(-.7)))) = 0.49150081873869
	// o_0 = squash(w_0*1+w_1*o_1+w_2*o_2+w_3*o_3) = 1/(1+exp(-(.02*1-.01*.4922506205862+.03*.50599971201659+.02*.49150081873869))) = 0.51002053349535
	//if(std::abs(pat[2] - 0.51002053349535) > tol) throw Ex("forward prop problem"); // logistic
	if(std::abs(pat[2] - 0.02034721575641) > tol) throw Ex("forward prop problem"); // tanh

	// Test that the output error is computed properly
	nn.trainIncremental(features[0], labels[0]);
	GNeuralNet* pBP = &nn;
	// Here is the math (done by hand) for why these results are expected:
	// e_0 = output*(1-output)*(target-output) = .51002053349535*(1-.51002053349535)*(1-.51002053349535) = 0.1224456672531
	if(useCrossEntropy)
	{
		// Here is the math for why these results are expected:
		// e_0 = target-output = 1-.51002053349535 = 0.4899794665046473
		if(std::abs(((GLayerClassic*)&pBP->layer(1))->error()[0] - 0.4899794665046473) > tol) throw Ex("problem computing output error");
	}
	else
	{
		// Here is the math for why these results are expected:
		// e_0 = output*(1-output)*(target-output) = .51002053349535*(1-.51002053349535)*(1-.51002053349535) = 0.1224456672531
		//if(std::abs(pBP->layer(1).blame()[0] - 0.1224456672531) > tol) throw Ex("problem computing output error"); // logistic
		if(std::abs(((GLayerClassic*)&pBP->layer(1))->error()[0] - 0.9792471989888) > tol) throw Ex("problem computing output error"); // tanh
	}

	// Test Back Prop
	if(useCrossEntropy)
	{
		if(std::abs(((GLayerClassic*)&pBP->layer(0))->error()[0] + 0.0012246544194742083) > tol) throw Ex("back prop problem");
		// e_2 = o_2*(1-o_2)*(w_2*e_0) = 0.00091821027577176
		if(std::abs(((GLayerClassic*)&pBP->layer(0))->error()[1] - 0.0036743168717579557) > tol) throw Ex("back prop problem");
		// e_3 = o_3*(1-o_3)*(w_3*e_0) = 0.00061205143636003
		if(std::abs(((GLayerClassic*)&pBP->layer(0))->error()[2] - 0.002449189448583718) > tol) throw Ex("back prop problem");
	}
	else
	{
		// e_1 = o_1*(1-o_1)*(w_1*e_0) = .4922506205862*(1-.4922506205862)*(-.01*.1224456672531) = -0.00030604063598154
		//if(std::abs(pBP->layer(0).blame()[0] + 0.00030604063598154) > tol) throw Ex("back prop problem"); // logistic
		if(std::abs(((GLayerClassic*)&pBP->layer(0))->error()[0] + 0.00978306745006032) > tol) throw Ex("back prop problem"); // tanh
		// e_2 = o_2*(1-o_2)*(w_2*e_0) = 0.00091821027577176
		//if(std::abs(pBP->layer(0).blame()[1] - 0.00091821027577176) > tol) throw Ex("back prop problem"); // logistic
		if(std::abs(((GLayerClassic*)&pBP->layer(0))->error()[1] - 0.02936050107376107) > tol) throw Ex("back prop problem"); // tanh
		// e_3 = o_3*(1-o_3)*(w_3*e_0) = 0.00061205143636003
		//if(std::abs(pBP->layer(0).blame()[2] - 0.00061205143636003) > tol) throw Ex("back prop problem"); // logistic
		if(std::abs(((GLayerClassic*)&pBP->layer(0))->error()[2] - 0.01956232122115741) > tol) throw Ex("back prop problem"); // tanh
	}

	// Test weight update
	if(useCrossEntropy)
	{
		if(std::abs(layerOut.weights()[0][0] - 0.10574640663831328) > tol) throw Ex("weight update problem");
		if(std::abs(layerOut.weights()[1][0] - 0.032208721880745944) > tol) throw Ex("weight update problem");
	}
	else
	{
		// d_0 = (d_0*momentum)+(learning_rate*e_0*1) = 0*.9+.175*.1224456672531*1
		// w_0 = w_0 + d_0 = .02+.0214279917693 = 0.041427991769293
		//if(std::abs(layerOut.bias()[0] - 0.041427991769293) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerOut.bias()[0] - 0.191368259823049) > tol) throw Ex("weight update problem"); // tanh
		// d_1 = (d_1*momentum)+(learning_rate*e_0*o_1) = 0*.9+.175*.1224456672531*.4922506205862
		// w_1 = w_1 + d_1 = -.01+.0105479422563 = 0.00054794224635029
		//if(std::abs(layerOut.weights()[0][0] - 0.00054794224635029) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerOut.weights()[0][0] + 0.015310714964467731) > tol) throw Ex("weight update problem"); // tanh
		//if(std::abs(layerOut.weights()[1][0] - 0.040842557664356) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerOut.weights()[1][0] - 0.034112048752708297) > tol) throw Ex("weight update problem"); // tanh
		//if(std::abs(layerOut.weights()[2][0] - 0.030531875498533) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerOut.weights()[2][0] - 0.014175723281037968) > tol) throw Ex("weight update problem"); // tanh
		//if(std::abs(layerHidden.bias()[0] + 0.010053557111297) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerHidden.bias()[0] + 0.011712036803760557) > tol) throw Ex("weight update problem"); // tanh
		if(std::abs(layerHidden.weights()[0][0] + 0.03) > tol) throw Ex("weight update problem"); // logistic & tanh
		//if(std::abs(layerHidden.weights()[1][0] - 0.030037489977908) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerHidden.weights()[1][0] - 0.03119842576263239) > tol) throw Ex("weight update problem"); // tanh
		//if(std::abs(layerHidden.bias()[1] - 0.01016068679826) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerHidden.bias()[1] - 0.015138087687908187) > tol) throw Ex("weight update problem"); // tanh
		if(std::abs(layerHidden.weights()[0][1] - 0.04) > tol) throw Ex("weight update problem"); // logistic & tanh
		//if(std::abs(layerHidden.weights()[1][1] + 0.020112480758782) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerHidden.weights()[1][1] + 0.023596661381535732) > tol) throw Ex("weight update problem"); // tanh
		//if(std::abs(layerHidden.bias()[2] + 0.019892890998637) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerHidden.bias()[2] + 0.016576593786297455) > tol) throw Ex("weight update problem"); // tanh
		if(std::abs(layerHidden.weights()[0][2] - 0.03) > tol) throw Ex("weight update problem"); // logistic & tanh
		//if(std::abs(layerHidden.weights()[1][2] - 0.019925023699046) > tol) throw Ex("weight update problem"); // logistic
		if(std::abs(layerHidden.weights()[1][2] - 0.01760361565040822) > tol) throw Ex("weight update problem"); // tanh
	}
}

void GNeuralNet_testConvolutionalLayerMath()
{
	GLayerConvolutional1D layer(4, 2, 3, 2, new GActivationIdentity());
	layer.bias()[0] = 0.0;
	layer.bias()[1] = -1.0;
	layer.bias()[2] = 2.0;
	layer.bias()[3] = 1.0;
	GMatrix& k = layer.kernels();
	k[0][0] = 0.0;	k[0][1] = 1.0;	k[0][2] = 2.0;
	k[1][0] = 2.0;	k[1][1] = 1.0;	k[1][2] = 0.0;
	k[2][0] = 0.0;	k[2][1] = 2.0;	k[2][2] = 1.0;
	k[3][0] = 1.0;	k[3][1] = 1.0;	k[3][2] = 1.0;
	const double in[] = { 0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 2.0 };
	double learning_rate = 2.0;
	layer.copyBiasToNet();
	layer.feedIn(in, 0, 8);
	layer.activate();
	const double expected_activation[] = { 2.0, -1.0, 9.0, 7.0, 5.0, 0.0, 10.0, 8.0 };
	if(GVec::squaredDistance(layer.activation(), expected_activation, 8) > 1e-9)
		throw Ex("incorrect activation");
	const double target[] = { 2.0, 0.0, 11.0, 10.0, 9.0, 5.0, 16.0, 15.0 };
	layer.computeError(target);
	const double expected_err[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
	if(GVec::squaredDistance(layer.error(), expected_err, 8) > 1e-9)
		throw Ex("incorrect error");
	layer.deactivateError();
	// Note that this test does not cover backPropError().
	layer.updateWeights(in, 0, 8, learning_rate, 0.0);
	const double expected_k1[] = { 0.0, 9.0, 18.0 };
	const double expected_k2[] = { 2.0, 11.0, 22.0 };
	const double expected_k3[] = { 28.0, 46.0, 37.0 };
	const double expected_k4[] = { 35.0, 55.0, 47.0 };
	if(GVec::squaredDistance(k[0], expected_k1, 3) > 1e-9)
		throw Ex("incorrect weights");
	if(GVec::squaredDistance(k[1], expected_k2, 3) > 1e-9)
		throw Ex("incorrect weights");
	if(GVec::squaredDistance(k[2], expected_k3, 3) > 1e-9)
		throw Ex("incorrect weights");
	if(GVec::squaredDistance(k[3], expected_k4, 3) > 1e-9)
		throw Ex("incorrect weights");
	layer.updateBias(learning_rate, 0.0);
	const double expected_bias[] = { 8.0, 11.0, 18.0, 21.0 };
	if(GVec::squaredDistance(layer.bias(), expected_bias, 4) > 1e-9)
		throw Ex("incorrect bias");
}

void GNeuralNet_testInputGradient(GRand* pRand)
{
	for(int i = 0; i < 20; i++)
	{
		// Make the neural net
		GNeuralNet nn;
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 5));
		nn.addLayer(new GLayerClassic(5, 10));
		nn.addLayer(new GLayerClassic(10, FLEXIBLE_SIZE));
		GUniformRelation featureRel(5);
		GUniformRelation labelRel(10);
		nn.beginIncrementalLearning(featureRel, labelRel);

		// Init with random weights
		size_t weightCount = nn.countWeights();
		double* pWeights = new double[weightCount + 5 + 10 + 10 + 5 + 5];
		ArrayHolder<double> hWeights(pWeights);
		double* pFeatures = pWeights + weightCount;
		double* pTarget = pFeatures + 5;
		double* pOutput = pTarget + 10;
		double* pFeatureGradient = pOutput + 10;
		double* pEmpiricalGradient = pFeatureGradient + 5;
		for(size_t j = 0; j < weightCount; j++)
			pWeights[j] = pRand->normal() * 0.8;
		nn.setWeights(pWeights);

		// Compute target output
		GVec::setAll(pFeatures, 0.0, 5);
		nn.predict(pFeatures, pTarget);

		// Move away from the goal and compute baseline error
		for(int i = 0; i < 5; i++)
			pFeatures[i] += pRand->normal() * 0.1;
		nn.predict(pFeatures, pOutput);
		double sseBaseline = GVec::squaredDistance(pTarget, pOutput, 10);

		// Compute the feature gradient
		nn.forwardProp(pFeatures);
		nn.backpropagate(pTarget);
		nn.gradientOfInputs(pFeatureGradient);
		GVec::multiply(pFeatureGradient, 2.0, 5);

		// Empirically measure gradient
		for(int i = 0; i < 5; i++)
		{
			pFeatures[i] += 0.0001;
			nn.predict(pFeatures, pOutput);
			double sse = GVec::squaredDistance(pTarget, pOutput, 10);
			pEmpiricalGradient[i] = (sse - sseBaseline) / 0.0001;
			pFeatures[i] -= 0.0001;
		}

		// Check it
		double corr = GVec::correlation(pFeatureGradient, pEmpiricalGradient, 5);
		if(corr > 1.0)
			throw Ex("pathological results");
		if(corr < 0.999)
			throw Ex("failed");
	}
}

void GNeuralNet_testBinaryClassification(GRand* pRand)
{
	vector<size_t> vals;
	vals.push_back(2);
	GMatrix features(vals);
	GMatrix labels(vals);
	for(size_t i = 0; i < 100; i++)
	{
		double d = (double)pRand->next(2);
		features.newRow()[0] = d;
		labels.newRow()[0] = 1.0 - d;
	}
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	GAutoFilter af(pNN);
	af.train(features, labels);
	double r = af.sumSquaredError(features, labels);
	if(r > 0.0)
		throw Ex("Failed simple sanity test");
}

#define TEST_INVERT_INPUTS 5
void GNeuralNet_testInvertAndSwap(GRand& rand)
{
	size_t layers = 2;
	size_t layerSize = 5;

	// This test ensures that the GNeuralNet::swapNodes and GNeuralNet::invertNode methods
	// have no net effect on the output of the neural network
	double in[TEST_INVERT_INPUTS];
	double outBefore[TEST_INVERT_INPUTS];
	double outAfter[TEST_INVERT_INPUTS];
	for(size_t i = 0; i < 30; i++)
	{
		GNeuralNet nn;
		for(size_t j = 0; j < layers; j++)
			nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, layerSize));
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		GUniformRelation rel(TEST_INVERT_INPUTS);
		nn.beginIncrementalLearning(rel, rel);
		nn.perturbAllWeights(0.5);
		rand.cubical(in, TEST_INVERT_INPUTS);
		nn.predict(in, outBefore);
		for(size_t j = 0; j < 8; j++)
		{
			if(rand.next(2) == 0)
				nn.swapNodes((size_t)rand.next(layers), (size_t)rand.next(layerSize), (size_t)rand.next(layerSize));
			else
				nn.invertNode((size_t)rand.next(layers), (size_t)rand.next(layerSize));
		}
		nn.predict(in, outAfter);
		if(GVec::squaredDistance(outBefore, outAfter, TEST_INVERT_INPUTS) > 1e-10)
			throw Ex("Failed");
	}

	for(size_t i = 0; i < 30; i++)
	{
		// Generate two identical neural networks
		GNeuralNet nn1;
		GNeuralNet nn2;
		for(size_t j = 0; j < layers; j++)
		{
			nn1.addLayer(new GLayerClassic(FLEXIBLE_SIZE, layerSize));
			nn2.addLayer(new GLayerClassic(FLEXIBLE_SIZE, layerSize));
		}
		nn1.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		nn2.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		GUniformRelation rel(TEST_INVERT_INPUTS);
		nn1.beginIncrementalLearning(rel, rel);
		nn2.beginIncrementalLearning(rel, rel);
		nn1.perturbAllWeights(0.5);
		nn2.copyWeights(&nn1);

		// Predict something
		rand.cubical(in, TEST_INVERT_INPUTS);
		nn1.predict(in, outBefore);

		// Mess with the topology of both networks
		for(size_t j = 0; j < 20; j++)
		{
			if(rand.next(2) == 0)
			{
				if(rand.next(2) == 0)
					nn1.swapNodes((size_t)rand.next(layers), (size_t)rand.next(layerSize), (size_t)rand.next(layerSize));
				else
					nn1.invertNode((size_t)rand.next(layers), (size_t)rand.next(layerSize));
			}
			else
			{
				if(rand.next(2) == 0)
					nn2.swapNodes((size_t)rand.next(layers), (size_t)rand.next(layerSize), (size_t)rand.next(layerSize));
				else
					nn2.invertNode((size_t)rand.next(layers), (size_t)rand.next(layerSize));
			}
		}

		// Align the first network to match the second one
		nn1.align(nn2);

		// Check that predictions match before
		nn2.predict(in, outAfter);
		if(GVec::squaredDistance(outBefore, outAfter, TEST_INVERT_INPUTS) > 1e-10)
			throw Ex("Failed");
		nn1.predict(in, outAfter);
		if(GVec::squaredDistance(outBefore, outAfter, TEST_INVERT_INPUTS) > 1e-10)
			throw Ex("Failed");

		// Check that they have matching weights
		size_t wc = nn1.countWeights();
		double* pW1 = new double[wc];
		ArrayHolder<double> hW1(pW1);
		nn1.weights(pW1);
		double* pW2 = new double[wc];
		ArrayHolder<double> hW2(pW2);
		nn2.weights(pW2);
		for(size_t j = 0; j < wc; j++)
		{
			if(std::abs(*pW1 - *pW2) >= 1e-9)
				throw Ex("Failed");
			pW1++;
			pW2++;
		}
	}
}

void GNeuralNet_testNormalizeInput(GRand& rand)
{
	double in[5];
	for(size_t i = 0; i < 20; i++)
	{
		GNeuralNet nn;
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 5));
		nn.addLayer(new GLayerClassic(5, FLEXIBLE_SIZE));
		GUniformRelation relIn(5);
		GUniformRelation relOut(1);
		nn.beginIncrementalLearning(relIn, relOut);
		nn.perturbAllWeights(1.0);
		rand.spherical(in, 5);
		double before, after;
		nn.predict(in, &before);
		double a = rand.normal();
		double b = rand.normal();
		if(b < a)
			std::swap(a, b);
		double c = rand.normal();
		double d = rand.normal();
		if(d < c)
			std::swap(c, d);
		size_t ind = (size_t)rand.next(5);
		GNeuralNetLayer* pInputLayer = &nn.layer(0);
		pInputLayer->renormalizeInput(ind, a, b, c, d);
		in[ind] = GMatrix::normalizeValue(in[ind], a, b, c, d);
		nn.predict(in, &after);
		if(std::abs(after - before) > 1e-9)
			throw Ex("Failed");
	}
}

void GNeuralNet_testBleedWeights()
{
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 2));
	nn.addLayer(new GLayerClassic(2, 2));
	nn.addLayer(new GLayerClassic(2, FLEXIBLE_SIZE));
	GUniformRelation rel(2, 0);
	nn.beginIncrementalLearning(rel, rel);
	((GLayerClassic*)&nn.layer(2))->weights()[0][0] = 1.0;
	((GLayerClassic*)&nn.layer(2))->weights()[1][0] = 1.0;
	((GLayerClassic*)&nn.layer(2))->weights()[0][1] = 1.0;
	((GLayerClassic*)&nn.layer(2))->weights()[1][1] = 1.0;
	((GLayerClassic*)&nn.layer(1))->weights()[0][0] = 5.0;
	((GLayerClassic*)&nn.layer(1))->weights()[1][0] = 2.0;
	((GLayerClassic*)&nn.layer(1))->weights()[0][1] = 3.0;
	((GLayerClassic*)&nn.layer(1))->weights()[1][1] = 1.0;
	((GLayerClassic*)&nn.layer(0))->weights()[0][0] = 0.5;
	((GLayerClassic*)&nn.layer(0))->weights()[1][0] = 0.2;
	((GLayerClassic*)&nn.layer(0))->weights()[0][1] = 0.3;
	((GLayerClassic*)&nn.layer(0))->weights()[1][1] = 0.1;
	size_t wc = nn.countWeights();
	double* pBefore = new double[wc];
	ArrayHolder<double> hBefore(pBefore);
	double* pAfter = new double[wc];
	ArrayHolder<double> hAfter(pAfter);
	nn.weights(pBefore);
	nn.bleedWeightsL2(0.1);
	nn.weights(pAfter);
	if(std::abs(GVec::squaredMagnitude(pBefore, wc) - GVec::squaredMagnitude(pAfter, wc)) > 0.000001)
		throw Ex("failed");
}

void GNeuralNet_testTransformWeights(GRand& prng)
{
	for(size_t i = 0; i < 10; i++)
	{
		// Set up
		GNeuralNet nn;
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		GUniformRelation in(2);
		GUniformRelation out(3);
		nn.beginIncrementalLearning(in, out);
		nn.perturbAllWeights(1.0);
		double x1[2];
		double x2[2];
		double y1[3];
		double y2[3];
		prng.spherical(x1, 2);

		// Predict normally
		nn.predict(x1, y1);

		// Transform the inputs and weights
		GMatrix transform(2, 2);
		prng.spherical(transform[0], 2);
		prng.spherical(transform[1], 2);
		double offset[2];
		prng.spherical(offset, 2);
		GVec::add(x1, offset, 2);
		transform.multiply(x1, x2, false);

		double tmp[2];
		GVec::multiply(offset, -1.0, 2);
		transform.multiply(offset, tmp);
		GVec::copy(offset, tmp, 2);
		GMatrix* pTransInv = transform.pseudoInverse();
		Holder<GMatrix> hTransInv(pTransInv);
		((GLayerClassic*)&nn.layer(0))->transformWeights(*pTransInv, offset);

		// Predict again
		nn.predict(x2, y2);
		if(GVec::squaredDistance(y1, y2, 3) > 1e-15)
			throw Ex("transformWeights failed");
	}
}

#define NN_TEST_DIMS 5

void GNeuralNet_testCompressFeatures(GRand& prng)
{
	GMatrix feat(50, NN_TEST_DIMS);
	for(size_t i = 0; i < feat.rows(); i++)
		prng.spherical(feat[i], NN_TEST_DIMS);

	// Set up
	GNeuralNet nn1;
	nn1.addLayer(new GLayerClassic(FLEXIBLE_SIZE, NN_TEST_DIMS * 2));
	nn1.beginIncrementalLearning(feat.relation(), feat.relation());
	nn1.perturbAllWeights(1.0);
	GNeuralNet nn2;
	nn2.copyStructure(&nn1);
	nn2.copyWeights(&nn1);

	// Test
	GMatrix* pNewFeat = nn1.compressFeatures(feat);
	Holder<GMatrix> hNewFeat(pNewFeat);
	double out1[NN_TEST_DIMS];
	double out2[NN_TEST_DIMS];
	for(size_t i = 0; i < feat.rows(); i++)
	{
		nn1.predict(pNewFeat->row(i), out1);
		nn2.predict(feat[i], out2);
		if(GVec::squaredDistance(out1, out2, NN_TEST_DIMS) > 1e-14)
			throw Ex("failed");
	}
}

void GNeuralNet_testFourier()
{
	GMatrix m(16, 3);
	m[0][0] = 2.7; m[0][1] = 1.0; m[0][2] = 2.0;
	m[1][0] = 3.1; m[1][1] = 1.3; m[1][2] = 2.0;
	m[2][0] = 0.1; m[2][1] = 1.0; m[2][2] = 6.0;
	m[3][0] = 0.7; m[3][1] = 0.7; m[3][2] = 2.0;
	m[4][0] = 2.4; m[4][1] = 0.4; m[4][2] = 0.0;
	m[5][0] = 3.0; m[5][1] = 0.8; m[5][2] = 2.0;
	m[6][0] = 3.8; m[6][1] = 1.3; m[6][2] = 2.0;
	m[7][0] = 2.9; m[7][1] = 1.2; m[7][2] = 2.0;
	m[8][0] = 2.7; m[8][1] = 1.0; m[8][2] = 3.0;
	m[9][0] = 3.1; m[9][1] = 1.3; m[9][2] = 3.0;
	m[10][0] = 0.1; m[10][1] = 1.0; m[10][2] = 7.0;
	m[11][0] = 0.7; m[11][1] = 0.7; m[11][2] = 3.0;
	m[12][0] = 2.4; m[12][1] = 0.4; m[12][2] = 1.0;
	m[13][0] = 3.0; m[13][1] = 0.8; m[13][2] = 3.0;
	m[14][0] = 3.8; m[14][1] = 1.3; m[14][2] = 3.0;
	m[15][0] = 2.9; m[15][1] = 1.2; m[15][2] = 3.0;
	double period = 3.0;
	GNeuralNet* pNN = GNeuralNet::fourier(m, period);
	Holder<GNeuralNet> hNN(pNN);
	double out[3];
	for(size_t i = 0; i < m.rows(); i++)
	{
		double in = (double)i * period / m.rows();
		pNN->predict(&in, out);
		if(GVec::squaredDistance(out, m[i], m.cols()) > 1e-9)
			throw Ex("failed");
	}
}

// static
void GNeuralNet::test()
{
	GRand prng(0);
	GNeuralNet_testMath();
	GNeuralNet_testBinaryClassification(&prng);
	GNeuralNet_testInputGradient(&prng);
	GNeuralNet_testInvertAndSwap(prng);
	GNeuralNet_testNormalizeInput(prng);
	GNeuralNet_testBleedWeights();
	GNeuralNet_testTransformWeights(prng);
	GNeuralNet_testCompressFeatures(prng);
	GNeuralNet_testConvolutionalLayerMath();
	GNeuralNet_testFourier();

	// Test with no hidden layers (logistic regression)
	{
		GNeuralNet* pNN = new GNeuralNet();
		pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		GAutoFilter af(pNN);
		af.basicTest(0.75, 0.86);
	}

	// Test NN with one hidden layer
	{
		GNeuralNet* pNN = new GNeuralNet();
		pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 3));
		pNN->addLayer(new GLayerClassic(3, FLEXIBLE_SIZE));
		GAutoFilter af(pNN);
		af.basicTest(0.76, 0.92);
	}
}

#endif // MIN_PREDICT









GNeuralNetInverseLayer::~GNeuralNetInverseLayer()
{
	delete(m_pInverseWeights);
	delete(m_pActivationFunction);
}




GNeuralNetPseudoInverse::GNeuralNetPseudoInverse(GNeuralNet* pNN, double padding)
: m_padding(padding)
{
	size_t maxNodes = 0;
	size_t i;
	for(i = 0; i < pNN->layerCount(); i++)
	{
		GLayerClassic& nnLayer = *(GLayerClassic*)&pNN->layer(i);
		maxNodes = std::max(maxNodes, nnLayer.outputs());
		GNeuralNetInverseLayer* pLayer = new GNeuralNetInverseLayer();
		m_layers.push_back(pLayer);
		delete(pLayer->m_pActivationFunction);
		pLayer->m_pActivationFunction = nnLayer.activationFunction()->clone();
		GMatrix weights(nnLayer.outputs(), nnLayer.inputs());
		double* pBias = nnLayer.bias();
		GMatrix& weightsIn = nnLayer.weights();
		for(size_t j = 0; j < nnLayer.outputs(); j++)
		{
			double unbias = -*(pBias++);
			double* pRow = weights.row(j);
			for(size_t k = 0; k < nnLayer.inputs(); k++)
			{
				*(pRow++) = weightsIn[k][j];
				unbias -= nnLayer.activationFunction()->center() * weightsIn[k][j];
			}
			pLayer->m_unbias.push_back(unbias);
		}
		pLayer->m_pInverseWeights = weights.pseudoInverse();
	}
	m_pBuf1 = new double[2 * maxNodes];
	m_pBuf2 = m_pBuf1 + maxNodes;
}

GNeuralNetPseudoInverse::~GNeuralNetPseudoInverse()
{
	for(vector<GNeuralNetInverseLayer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++)
		delete(*it);
	delete[] std::min(m_pBuf1, m_pBuf2);
}

void GNeuralNetPseudoInverse::computeFeatures(const double* pLabels, double* pFeatures)
{
	size_t inCount = 0;
	vector<GNeuralNetInverseLayer*>::iterator it = m_layers.end() - 1;
	GVec::copy(m_pBuf2, pLabels, (*it)->m_pInverseWeights->cols());
	for(; true; it--)
	{
		GNeuralNetInverseLayer* pLayer = *it;
		inCount = pLayer->m_pInverseWeights->rows();
		std::swap(m_pBuf1, m_pBuf2);

		// Invert the layer
		double* pT = m_pBuf1;
		for(vector<double>::iterator ub = pLayer->m_unbias.begin(); ub != pLayer->m_unbias.end(); ub++)
		{
			*pT = pLayer->m_pActivationFunction->inverse(*pT) + *ub;
			pT++;
		}
		pLayer->m_pInverseWeights->multiply(m_pBuf1, m_pBuf2);

		// Clip and uncenter the value
		pLayer = *it;
		double halfRange = pLayer->m_pActivationFunction->halfRange();
		double center = pLayer->m_pActivationFunction->center();
		pT = m_pBuf2;
		for(size_t i = 0; i < inCount; i++)
		{
			*pT = std::max(m_padding - halfRange, std::min(halfRange - m_padding, *pT)) + center;
			pT++;
		}

		if(it == m_layers.begin())
			break;
	}
	GVec::copy(pFeatures, m_pBuf2, inCount);
}

#ifndef MIN_PREDICT
// static
void GNeuralNetPseudoInverse::test()
{
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 5));
	nn.addLayer(new GLayerClassic(5, 7));
	nn.addLayer(new GLayerClassic(7, FLEXIBLE_SIZE));
	GUniformRelation featureRel(3);
	GUniformRelation labelRel(12);
	nn.beginIncrementalLearning(featureRel, labelRel);
	for(size_t i = 0; i < nn.layerCount(); i++)
	{
		((GLayerClassic*)&nn.layer(i))->setWeightsToIdentity();
		((GLayerClassic*)&nn.layer(i))->perturbWeights(nn.rand(), 0.5);
	}
	GNeuralNetPseudoInverse nni(&nn, 0.001);
	double labels[12];
	double features[3];
	double features2[3];
	for(size_t i = 0; i < 20; i++)
	{
		for(size_t j = 0; j < 3; j++)
			features[j] = nn.rand().uniform() * 0.98 + 0.01;
		nn.predict(features, labels);
		nni.computeFeatures(labels, features2);
		if(GVec::squaredDistance(features, features2, 3) > 1e-8)
			throw Ex("failed");
	}
}
#endif // MIN_PREDICT









GReservoirNet::GReservoirNet()
: GIncrementalLearner(), m_pModel(NULL), m_pNN(NULL), m_weightDeviation(0.5), m_augments(64), m_reservoirLayers(2)
{
}

GReservoirNet::GReservoirNet(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalLearner(pNode, ll)
{
	m_pModel = (GIncrementalLearner*)ll.loadLearner(pNode->field("model"));
	m_weightDeviation = pNode->field("wdev")->asDouble();
	m_augments = (size_t)pNode->field("augs")->asInt();
	m_reservoirLayers = (size_t)pNode->field("reslays")->asInt();
}

GReservoirNet::~GReservoirNet()
{
	delete(m_pModel);
}

// virtual
void GReservoirNet::predict(const double* pIn, double* pOut)
{
	m_pModel->predict(pIn, pOut);
}

// virtual
void GReservoirNet::predictDistribution(const double* pIn, GPrediction* pOut)
{
	m_pModel->predictDistribution(pIn, pOut);
}

// virtual
void GReservoirNet::clear()
{
	m_pModel->clear();
}

// virtual
void GReservoirNet::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(!features.relation().areContinuous())
		throw Ex("GReservoirNet only supports continuous features. Perhaps you should wrap it in a GAutoFilter.");
	if(!labels.relation().areContinuous())
		throw Ex("GReservoirNet only supports continuous labels. Perhaps you should wrap it in a GAutoFilter.");

	delete(m_pModel);
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	GReservoir* pRes = new GReservoir(m_weightDeviation, m_augments, m_reservoirLayers);
	GDataAugmenter* pAug = new GDataAugmenter(pRes);
	m_pModel = new GFeatureFilter(pNN, pAug);
	m_pModel->train(features, labels);
}

// virtual
void GReservoirNet::trainIncremental(const double* pIn, const double* pOut)
{
	m_pModel->trainIncremental(pIn, pOut);
}

// virtual
void GReservoirNet::trainSparse(GSparseMatrix& features, GMatrix& labels)
{
	m_pModel->trainSparse(features, labels);
}

// virtual
void GReservoirNet::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	delete(m_pModel);
	m_pNN = new GNeuralNet();
	m_pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	GDataAugmenter* pAug = new GDataAugmenter(new GReservoir(m_weightDeviation, m_augments, m_reservoirLayers));
	m_pModel = new GFeatureFilter(m_pNN, pAug);
	m_pModel->beginIncrementalLearning(featureRel, labelRel);
}

// virtual
bool GReservoirNet::supportedFeatureRange(double* pOutMin, double* pOutMax)
{
	*pOutMin = -1.0;
	*pOutMax = 1.0;
	return false;
}

// virtual
bool GReservoirNet::supportedLabelRange(double* pOutMin, double* pOutMax)
{
	*pOutMin = -1.0;
	*pOutMax = 1.0;
	return false;
}

#ifndef MIN_PREDICT
// virtual
GDomNode* GReservoirNet::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GReservoirNet");
	pNode->addField(pDoc, "model", m_pModel->serialize(pDoc));
	pNode->addField(pDoc, "wdev", pDoc->newDouble(m_weightDeviation));
	pNode->addField(pDoc, "augs", pDoc->newInt(m_augments));
	pNode->addField(pDoc, "reslays", pDoc->newInt(m_reservoirLayers));
	return pNode;
}

// static
void GReservoirNet::test()
{
	GAutoFilter af(new GReservoirNet());
	af.basicTest(0.7, 0.74, 0.001, false, 0.9);
}
#endif // MIN_PREDICT



} // namespace GClasses

