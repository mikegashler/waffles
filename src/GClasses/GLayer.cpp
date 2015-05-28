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

#include "GLayer.h"
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

	// Activation function
	m_pActivationFunction->resize(outputCount);
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
void GLayerClassic::feedForward(const double* pIn)
{
	// Copy the bias to the net
	GAssert(bias()[outputs() - 1] > -1e100 && bias()[outputs() - 1] < 1e100);
	GVec::copy(net(), bias(), outputs());

	// Feed the input through
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	double* pNet = net();
	GAssert(pNet[outputCount - 1] > -1e100 && pNet[outputCount - 1] < 1e100);
	for(size_t i = 0; i < inputCount; i++)
	{
		GAssert(m_weights.row(i)[outputCount - 1] > -1e100 && m_weights.row(i)[outputCount - 1] < 1e100);
		GVec::addScaled(pNet, *(pIn++), m_weights.row(i), outputCount);
	}

	// Activate
	double* pAct = activation();
	for(size_t i = 0; i < outputCount; i++)
	{
		GAssert(*pNet < 1e100 && *pNet > -1e100);
		*(pAct++) = m_pActivationFunction->squash(*(pNet++), i);
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
		*(pAct++) = m_pActivationFunction->squash(*(pNet++), i);
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
	*pAct = m_pActivationFunction->squash(*pNet, output);
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
	m_pActivationFunction->setError(pErr);
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct, i);
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
	(*pErr) *= m_pActivationFunction->derivativeOfNet(netVal, act, output);
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

void GLayerClassic::updateDeltas(const double* pUpStreamActivation, double momentum)
{
	double* pErr = error();
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	for(size_t up = 0; up < inputCount; up++)
	{
		double* pB = pErr;
		double* pD = m_delta[up];
		double act = *(pUpStreamActivation++);
		for(size_t down = 0; down < outputCount; down++)
		{
			*pD *= momentum;
			*pD += (*(pB++) * act);
			pD++;
		}
	}
	double* pB = pErr;
	double* pD = biasDelta();
	for(size_t down = 0; down < outputCount; down++)
	{
		*pD *= momentum;
		*pD += *(pB++);
		pD++;
	}
	m_pActivationFunction->updateDeltas(net(), activation(), momentum);
}

void GLayerClassic::copySingleNeuronWeights(size_t source, size_t dest)
{
	for(size_t up = 0; up < m_weights.rows(); up++)
	{
		m_weights[up][dest] = m_weights[up][source];
	}
	bias()[dest] = bias()[source];
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

// virtual
void GLayerClassic::applyDeltas(double learningRate)
{
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	for(size_t i = 0; i < inputCount; i++)
		GVec::addScaled(m_weights[i], learningRate, m_delta[i], outputCount);
	GVec::addScaled(bias(), learningRate, biasDelta(), outputCount);
	m_pActivationFunction->applyDeltas(learningRate);
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
				m_weights[j][i] = 1.0;
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
void GLayerClassic::regularizeActivationFunction(double lambda)
{
	m_pActivationFunction->regularize(lambda);
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
	pOutVector += (inputs() * outputs());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (inputs() + 1) * outputs() + activationWeights;
}

// virtual
size_t GLayerClassic::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, outputs());
	pVector += outputs();
	m_weights.fromVector(pVector, inputs());
	pVector += (inputs() * outputs());
	size_t activationWeights = m_pActivationFunction->vectorToWeights(pVector);
	return (inputs() + 1) * outputs() + activationWeights;
}

// virtual
void GLayerClassic::copyWeights(const GNeuralNetLayer* pSource)
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
		double d = m_pActivationFunction->squash(*(pNet++), i);
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
	m_inputError.resize(1, m_components[0]->inputs());
	outputs(); // Causes a buffer to be allocated
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
	{
		m_components[i]->resize(inputs, m_components[i]->outputs(), pRand, deviation);
		m_inputError.resize(1, inputs);
	}
}

// virtual
void GLayerMixed::feedForward(const double* pIn)
{
	double* pAct = m_activation[0];
	for(size_t i = 0; i < m_components.size(); i++)
	{
		m_components[i]->feedForward(pIn);
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
void GLayerMixed::updateDeltas(const double* pUpStreamActivation, double momentum)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->updateDeltas(pUpStreamActivation, momentum);
}

// virtual
void GLayerMixed::applyDeltas(double learningRate)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->applyDeltas(learningRate);
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
void GLayerMixed::regularizeActivationFunction(double lambda)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->regularizeActivationFunction(lambda);
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
void GLayerMixed::copyWeights(const GNeuralNetLayer* pSource)
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

	// Activation function
	m_pActivationFunction->resize(outputCount);
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
void GLayerRestrictedBoltzmannMachine::feedForward(const double* pIn)
{
	GVec::copy(net(), bias(), outputs());
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	double* pNet = net();
	for(size_t i = 0; i < outputCount; i++)
		*(pNet++) += GVec::dotProduct(pIn, m_weights[i], inputCount);

	// Activate
	double* pAct = activation();
	pNet = net();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++), i);
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
		*(pAct++) = m_pActivationFunction->squash(*(pNet++), i);
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
	m_pActivationFunction->setError(pErr);
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct, i);
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

void GLayerRestrictedBoltzmannMachine::updateDeltas(const double* pUpStreamActivation, double momentum)
{
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	double* pErr = error();
	double* pBD = biasDelta();
	for(size_t i = 0; i < outputCount; i++)
	{
		GVec::multiply(m_delta[i], momentum, inputCount);
		GVec::addScaled(m_delta[i], (*pErr), pUpStreamActivation, inputCount);
		*pBD *= momentum;
		*(pBD++) += *pErr;
		pErr++;
	}
}

// virtual
void GLayerRestrictedBoltzmannMachine::applyDeltas(double learningRate)
{
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
		GVec::addScaled(m_weights[i], learningRate, m_delta[i], inputCount);
	GVec::addScaled(bias(), learningRate, biasDelta(), outputCount);
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
void GLayerRestrictedBoltzmannMachine::regularizeActivationFunction(double lambda)
{
	m_pActivationFunction->regularize(lambda);
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
	pOutVector += (inputs() * outputs());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (inputs() + 1) * outputs() + activationWeights;
}

// virtual
size_t GLayerRestrictedBoltzmannMachine::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, outputs());
	pVector += outputs();
	m_weights.fromVector(pVector, inputs());
	pVector += (inputs() * outputs());
	size_t activationWeights = m_pActivationFunction->vectorToWeights(pVector);
	return (inputs() + 1) * outputs() + activationWeights;
}

// virtual
void GLayerRestrictedBoltzmannMachine::copyWeights(const GNeuralNetLayer* pSource)
{
	GLayerRestrictedBoltzmannMachine* src = (GLayerRestrictedBoltzmannMachine*)pSource;
	m_weights.copyBlock(src->m_weights, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	GVec::copy(bias(), src->bias(), src->outputs());
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
	m_activation.resize(3, inputChannels * kernelsPerChannel * m_outputSamples);
	m_delta.setAll(0.0);
	GVec::setAll(biasDelta(), 0.0, inputChannels * kernelsPerChannel);
	m_pActivationFunction->resize(m_bias.cols());
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
void GLayerConvolutional1D::feedForward(const double* pIn)
{
	// Copy bias to net
	double* pNet = net();
	for(size_t i = 0; i < m_outputSamples; i++)
	{
		GVec::copy(pNet, bias(), m_kernels.rows());
		pNet += m_kernels.rows();
	}

	// Feed pIn through
	size_t kernelSize = m_kernels.cols();
	pNet = net();
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

	// Activate
	double* pAct = activation();
	pNet = net();
	size_t kernelCount = m_bias.cols();
	for(size_t i = 0; i < m_outputSamples; i++)
	{
		for(size_t j = 0; j < kernelCount; j++)
			*(pAct++) = m_pActivationFunction->squash(*(pNet++), j);
	}
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
	m_pActivationFunction->setError(pErr);
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct, i);
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
void GLayerConvolutional1D::updateDeltas(const double* pUpStreamActivation, double momentum)
{
	m_delta.multiply(momentum);
	GVec::multiply(biasDelta(), momentum, m_inputChannels * m_kernelsPerChannel);
	double* pErr = error();
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_outputSamples; i++) // for each sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				double* pD = m_delta[kern++];
				const double* pIn = pUpStreamActivation;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
				{
					*(pD++) += *pErr * *pIn;
					pIn += m_inputChannels;
				}
				pErr++;
			}
			pUpStreamActivation++;
		}
	}
	pErr = error();
	for(size_t i = 0; i < m_outputSamples; i++)
	{
		double* pD = biasDelta();
		for(size_t j = 0; j < m_inputChannels; j++)
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++)
				*(pD++) += *(pErr++);
		}
	}
}

// virtual
void GLayerConvolutional1D::applyDeltas(double learningRate)
{
	size_t n = m_kernels.rows();
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < n; i++)
		GVec::addScaled(m_kernels[i], learningRate, m_delta[i], kernelSize);
	GVec::addScaled(bias(), learningRate, biasDelta(), m_inputChannels * m_kernelsPerChannel);
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
void GLayerConvolutional1D::regularizeActivationFunction(double lambda)
{
	m_pActivationFunction->regularize(lambda);
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
	pOutVector += (m_kernels.rows() * m_kernels.cols());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (m_kernels.rows() + 1) * m_kernels.cols() + activationWeights;
}

// virtual
size_t GLayerConvolutional1D::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, m_kernels.rows());
	pVector += m_kernels.rows();
	m_kernels.fromVector(pVector, m_kernels.rows());
	pVector += (m_kernels.rows() * m_kernels.cols());
	size_t activationWeights = m_pActivationFunction->vectorToWeights(pVector);
	return (m_kernels.rows() + 1) * m_kernels.cols() + activationWeights;
}

// virtual
void GLayerConvolutional1D::copyWeights(const GNeuralNetLayer* pSource)
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
	m_activation.resize(3, m_kernelCount * m_outputCols * m_outputRows);
	m_delta.setAll(0.0);
	GVec::setAll(biasDelta(), 0.0, m_kernelCount);
	m_pActivationFunction->resize(m_kernelCount);
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
void GLayerConvolutional2D::feedForward(const double* pIn)
{
	// Copy the bias to the net
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

	// Feed the input through
	size_t kernelSize = m_kernels.cols();
	pNet = net();
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

	// Activate
	double* pAct = activation();
	pNet = net();
	for(size_t h = 0; h < m_outputRows; h++) // for each output row...
	{
		for(size_t i = 0; i < m_outputCols; i++) // for each output column...
		{
			for(size_t j = 0; j < m_kernelCount; j++)
				*(pAct++) = m_pActivationFunction->squash(*(pNet++), j);
		}
	}
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
	m_pActivationFunction->setError(pErr);
	for(size_t i = 0; i < outputUnits; i++)
	{
		(*pErr) *= m_pActivationFunction->derivativeOfNet(*pNet, *pAct, i);
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
void GLayerConvolutional2D::updateDeltas(const double* pUpStreamActivation, double momentum)
{
	m_delta.multiply(momentum);
	GVec::multiply(biasDelta(), momentum, m_inputChannels * m_kernelsPerChannel);
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
						double* pD = m_delta[kern++];
						for(size_t m = 0; m < kernelSize; m++) // for each kernel column...
						{
							*(pD++) += *pErr * *pIn;
							pIn += m_inputChannels;
						}
					}
					pErr++;
				}
				pUpStreamActivation++;
			}
		}
	}
	pErr = error();
	for(size_t h = 0; h < m_outputRows; h++)
	{
		for(size_t i = 0; i < m_outputCols; i++)
		{
			double* pD = biasDelta();
			for(size_t j = 0; j < m_inputChannels; j++)
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++)
					*(pD++) += *(pErr++);
			}
		}
	}
}

// virtual
void GLayerConvolutional2D::applyDeltas(double learningRate)
{
	size_t n = m_kernels.rows();
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < n; i++)
		GVec::addScaled(m_kernels[i], learningRate, m_delta[i], kernelSize);
	GVec::addScaled(bias(), learningRate, biasDelta(), m_kernelsPerChannel);
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
void GLayerConvolutional2D::regularizeActivationFunction(double lambda)
{
	m_pActivationFunction->regularize(lambda);
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
	pOutVector += (m_kernels.rows() * m_kernels.cols());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return m_kernels.rows() * m_kernels.cols() + m_kernelCount + activationWeights;
}

// virtual
size_t GLayerConvolutional2D::vectorToWeights(const double* pVector)
{
	GVec::copy(bias(), pVector, m_kernelCount);
	pVector += m_kernelCount;
	m_kernels.fromVector(pVector, m_kernels.rows());
	pVector += (m_kernels.rows() * m_kernels.cols());
	size_t activationWeights = m_pActivationFunction->vectorToWeights(pVector);
	return m_kernels.rows() * m_kernels.cols() + m_kernelCount + activationWeights;
}

// virtual
void GLayerConvolutional2D::copyWeights(const GNeuralNetLayer* pSource)
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
void GLayerConvolutional2D::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}



} // namespace GClasses

