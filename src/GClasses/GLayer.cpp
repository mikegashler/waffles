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
#include <memory>

using std::vector;
using std::ostream;

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
		pResults->newRow().copy(activation());
	}
	return pResults;
}









GLayerClassic::GLayerClassic(size_t inps, size_t outs, GActivationFunction* pActivationFunction)
{
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationTanH();
	resize(inps, outs, NULL);
}

GLayerClassic::GLayerClassic(GDomNode* pNode)
: m_weights(pNode->field("weights")), m_delta(m_weights.rows(), m_weights.cols()), m_bias(6, m_weights.cols())
{
	bias().deserialize(pNode->field("bias"));
	slack().deserialize(pNode->field("slack"));
	m_pActivationFunction = GActivationFunction::deserialize(pNode->field("act_func"));
	m_delta.setAll(0.0);
	biasDelta().fill(0.0);
}

GLayerClassic::~GLayerClassic()
{
	delete(m_pActivationFunction);
}

GDomNode* GLayerClassic::serialize(GDom* pDoc)
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "weights", m_weights.serialize(pDoc));
	pNode->addField(pDoc, "bias", bias().serialize(pDoc));
	pNode->addField(pDoc, "slack", slack().serialize(pDoc));
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
				GVec& row = m_weights[i];
				for(size_t j = 0; j < fewerOutputs; j++)
					d += (row[j] * row[j]);
			}
			dev *= sqrt(d / (fewerInputs * fewerOutputs));
			if(inputCount * outputCount - fewerInputs * fewerOutputs > fewerInputs * fewerOutputs)
				dev *= fewerInputs * fewerOutputs / (inputCount * outputCount - fewerInputs * fewerOutputs);
		}
		for(size_t i = 0; i < fewerInputs; i++)
		{
			GVec& row = m_weights[i];
			for(size_t j = fewerOutputs; j < outputCount; j++)
				row[j] = dev * pRand->normal();
		}
		for(size_t i = fewerInputs; i < inputCount; i++)
		{
			GVec& row = m_weights[i];
			for(size_t j = 0; j < outputCount; j++)
				row[j] = dev * pRand->normal();
		}
	}

	// Bias
	m_bias.resizePreserve(6, outputCount);
	biasDelta().fill(0.0);
	if(pRand)
	{
		GVec& b = bias();
		for(size_t j = fewerOutputs; j < outputCount; j++)
			b[j] = dev * pRand->normal();
	}

	// Slack
	GVec& s = slack();
	for(size_t j = fewerOutputs; j < outputCount; j++)
		s[j] = 0.0;

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
		GVec& w = m_weights[i];
		for(size_t j = 0; j < outputCount; j++)
			w[j] = rand.normal() * mag;
	}
	m_delta.setAll(0.0);
	GVec& b = bias();
	for(size_t i = 0; i < outputCount; i++)
		b[i] = rand.normal() * mag;
	biasDelta().fill(0.0);
}

// virtual
void GLayerClassic::feedForward(const GVec& in)
{
	// Copy the bias to the net
	GAssert(bias()[outputs() - 1] > -1e100 && bias()[outputs() - 1] < 1e100);
	net().copy(bias());

	// Feed the input through
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	GVec& n = net();
	GVec& a = activation();
	for(size_t i = 0; i < inputCount; i++)
		n.addScaled(in[i], m_weights.row(i));

	// Activate
	for(size_t i = 0; i < outputCount; i++)
	{
		GAssert(n[i] < 1e100 && n[i] > -1e100);
		a[i] = m_pActivationFunction->squash(n[i], i);
	}
}

// virtual
void GLayerClassic::dropOut(GRand& rand, double probOfDrop)
{
	GVec& a = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			a[i] = 0.0;
	}
}

void GLayerClassic::feedForwardToOneOutput(const GVec& in, size_t output)
{
	// Compute net = in * m_weights + bias
	GAssert(output < outputs());
	GVec& n = net();
	size_t pos = 0;
	n[output] = 0.0;
	for(size_t i = 0; i < m_weights.rows(); i++)
		n[output] += (in[pos++] * m_weights[i][output]);
	n[output] += bias()[output];

	// Apply the activation function
	GVec& a = activation();
	a[output] = m_pActivationFunction->squash(n[output], output);
}

void GLayerClassic::computeError(const GVec& target)
{
	size_t outputUnits = outputs();
	GVec& a = activation();
	GVec& s = slack();
	GVec& err = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE)
			err[i] = 0.0;
		else
		{
			if(target[i] > a[i] + s[i])
				err[i] = (target[i] - a[i] - s[i]);
			else if(target[i] < a[i] - s[i])
				err[i] = (target[i] - a[i] + s[i]);
			else
				err[i] = 0.0;
		}
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
	GVec& err = error();
	GVec& n = net();
	GVec& a = activation();
	m_pActivationFunction->setError(err);
	for(size_t i = 0; i < outputUnits; i++)
		err[i] *= m_pActivationFunction->derivativeOfNet(n[i], a[i], i);
}

void GLayerClassic::deactivateErrorSingleOutput(size_t output)
{
	double* pErr = &error()[output];
	double netVal = net()[output];
	double act = activation()[output];
	(*pErr) *= m_pActivationFunction->derivativeOfNet(netVal, act, output);
}

void GLayerClassic::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	GVec& upStreamError = pUpStreamLayer->error();
	size_t inputCount = pUpStreamLayer->outputs();
	GAssert(inputCount <= m_weights.rows());
	const GVec& source = error();
	for(size_t i = 0; i < inputCount; i++)
		upStreamError[i] = source.dotProduct(m_weights[i]);
}

void GLayerClassic::backPropErrorSingleOutput(size_t outputNode, GVec& upStreamError)
{
	GAssert(outputNode < outputs());
	double in = error()[outputNode];
	for(size_t i = 0; i < m_weights.rows(); i++)
		upStreamError[i] = in * m_weights[i][outputNode];
}

void GLayerClassic::updateDeltas(const GVec& upStreamActivation, double momentum)
{
	GVec& err = error();
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	for(size_t up = 0; up < inputCount; up++)
	{
		GVec& d = m_delta[up];
		double act = upStreamActivation[up];
		for(size_t down = 0; down < outputCount; down++)
		{
			d[down] *= momentum;
			d[down] += (err[down] * act);
		}
	}
	GVec& d = biasDelta();
	for(size_t down = 0; down < outputCount; down++)
	{
		d[down] *= momentum;
		d[down] += err[down];
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

void GLayerClassic::updateWeightsSingleNeuron(size_t outputNode, const GVec& upStreamActivation, double learningRate, double momentum)
{
	// Adjust the weights
	double err = error()[outputNode];
	for(size_t up = 0; up < m_weights.rows(); up++)
	{
		double* pD = &m_delta[up][outputNode];
		double* pW = &m_weights[up][outputNode];
		double act = upStreamActivation[up];
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
	for(size_t i = 0; i < inputCount; i++)
		m_weights[i].addScaled(learningRate, m_delta[i]);
	bias().addScaled(learningRate, biasDelta());
	m_pActivationFunction->applyDeltas(learningRate);
}

void GLayerClassic::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

void GLayerClassic::diminishWeights(double amount, bool regularizeBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i].regularize_L1(amount);
	if(regularizeBiases)
		bias().regularize_L1(amount);
}

void GLayerClassic::contractWeights(double factor, bool contractBiases)
{
	GVec& n = net();
	GVec& a = activation();
	GVec& b = bias();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		double f = 1.0 - factor * m_pActivationFunction->derivativeOfNet(n[i], a[i], i);
		for(size_t j = 0; j < m_weights.rows(); j++)
			m_weights[j][i] *= f;
		if(contractBiases)
			b[i] *= f;
	}
}

void GLayerClassic::regularizeWeights(double factor, double power)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		GVec& w = m_weights[i];
		for(size_t j = 0; j < outputCount; j++)
			w[j] -= GBits::sign(w[j]) * factor * pow(std::abs(w[j]), power);
	}
	GVec& w = bias();
	for(size_t j = 0; j < outputCount; j++)
		w[j] -= GBits::sign(w[j]) * factor * pow(std::abs(w[j]), power);
}

void GLayerClassic::transformWeights(GMatrix& transform, const GVec& offset)
{
	if(transform.rows() != inputs())
		throw Ex("Transformation matrix not suitable size for this layer");
	if(transform.rows() != transform.cols())
		throw Ex("Expected a square transformation matrix.");
	size_t outputCount = outputs();
	GMatrix* pNewWeights = GMatrix::multiply(transform, m_weights, true, false);
	std::unique_ptr<GMatrix> hNewWeights(pNewWeights);
	m_weights.copyBlock(*pNewWeights, 0, 0, pNewWeights->rows(), outputCount, 0, 0, false);
	GVec& n = net();
	n.fill(0.0);
	for(size_t i = 0; i < m_weights.rows(); i++)
		n.addScaled(offset[i], m_weights.row(i));
	bias() += n;
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
void GLayerClassic::maxNorm(double min, double max)
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
		else if(squaredMag < min * min)
		{
			if(squaredMag == 0.0)
			{
				for(size_t j = 0; j < m_weights.rows(); j++)
					m_weights[j][i] = 1.0;
				squaredMag = (double)m_weights.rows();
			}
			double scal = min / sqrt(squaredMag);
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
	return (inputs() + 1) * outputs() + m_pActivationFunction->countWeights();
}

// virtual
size_t GLayerClassic::weightsToVector(double* pOutVector)
{
	GVec::copy(pOutVector, bias().data(), outputs());
	pOutVector += outputs();
	m_weights.toVector(pOutVector);
	pOutVector += (inputs() * outputs());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (inputs() + 1) * outputs() + activationWeights;
}

// virtual
size_t GLayerClassic::vectorToWeights(const double* pVector)
{
	bias().set(pVector, outputs());
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
	bias().copy(src->bias());
	m_pActivationFunction->copyWeights(src->m_pActivationFunction);
}

// virtual
void GLayerClassic::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	size_t n = std::min(outputs() - start, count);
	for(size_t j = 0; j < m_weights.rows(); j++)
		GVec::perturb(m_weights[j].data() + start, deviation, n, rand);
	GVec::perturb(bias().data() + start, deviation, n, rand);
}

// virtual
void GLayerClassic::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	size_t outputCount = outputs();
	GVec& w = m_weights[input];
	GVec& b = bias();
	double f = (oldMax - oldMin) / (newMax - newMin);
	double g = (oldMin - newMin * f);
	for(size_t i = 0; i < outputCount; i++)
	{
		b[i] += (w[i] * g);
		w[i] *= f;
	}
}

void GLayerClassic::printSummary(ostream& stream)
{
	size_t inps = m_weights.rows();
	size_t outs = m_weights.cols();
	stream << " ( " << to_str(inps) << " -> " << to_str(outs) << " )\n";
	stream << "    Bias Mag: " << std::sqrt(bias().squaredMagnitude()) << "\n";
	double sum = 0.0;
	for(size_t i = 0; i < outs; i++)
		sum += std::sqrt(m_weights.columnVariance(i, 0.0));
	stream << "    Ave Weight Mag: " << to_str(sum / outs) << "\n";
	stream << "    Net Mag: " << std::sqrt(net().squaredMagnitude()) << "\n";
	stream << "    Act Mag: " << std::sqrt(activation().squaredMagnitude()) << "\n";
	stream << "    Err Mag: " << std::sqrt(error().squaredMagnitude()) << "\n";
}








GLayerSoftMax::GLayerSoftMax(size_t inputSize, size_t outputSize)
: GLayerClassic(inputSize, outputSize, new GActivationLogistic())
{
}

GLayerSoftMax::GLayerSoftMax(GDomNode* pNode)
: GLayerClassic(pNode)
{
}

// virtual
void GLayerSoftMax::activate()
{
	double* pAct = activation().data();
	size_t outputCount = outputs();
	double* pNet = net().data();
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
		GVec::multiply(bias().data(), fac, outputCount);
		GVec::multiply(activation().data(), fac, outputCount);
	}
	else
	{
		GVec::setAll(activation().data(), 1.0 / outputCount, outputCount);
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
void GLayerMixed::resize(size_t inputSize, size_t outputSize, GRand* pRand, double deviation)
{
	if(outputSize != m_activation.cols())
		throw Ex("Sorry, GLayerMixed does not support resizing the number of outputs");
	for(size_t i = 0; i < m_components.size(); i++)
	{
		m_components[i]->resize(inputSize, m_components[i]->outputs(), pRand, deviation);
		m_inputError.resize(1, inputSize);
	}
}

// virtual
void GLayerMixed::feedForward(const GVec& in)
{
	double* pAct = m_activation[0].data();
	for(size_t i = 0; i < m_components.size(); i++)
	{
		m_components[i]->feedForward(in);
		GVec::copy(pAct, m_components[i]->activation().data(), m_components[i]->outputs());
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
void GLayerMixed::computeError(const GVec& target)
{
	size_t outputUnits = outputs();
	double* pAct = activation().data();
	double* pErr = error().data();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE)
			*pErr = 0.0;
		else
			*pErr = target[i] - *pAct;
		pAct++;
		pErr++;
	}
}

// virtual
void GLayerMixed::deactivateError()
{
	double* pErr = error().data();
	for(size_t i = 0; i < m_components.size(); i++)
	{
		GVec::copy(m_components[i]->error().data(), pErr, m_components[i]->outputs());
		m_components[i]->deactivateError();
		pErr += m_components[i]->outputs();
	}
}

// virtual
void GLayerMixed::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	double* pBuf = m_inputError[0].data();
	size_t inps = pUpStreamLayer->outputs();
	GVec::setAll(pBuf, 0.0, inps);
	for(size_t i = 0; i < m_components.size(); i++)
	{
		m_components[i]->backPropError(pUpStreamLayer);
		GVec::add(pBuf, pUpStreamLayer->error().data(), inps);
	}
	GVec::copy(pUpStreamLayer->error().data(), pBuf, inps);
}

// virtual
void GLayerMixed::updateDeltas(const GVec& upStreamActivation, double momentum)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->updateDeltas(upStreamActivation, momentum);
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
void GLayerMixed::maxNorm(double min, double max)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->maxNorm(min, max);
}

// virtual
void GLayerMixed::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->renormalizeInput(input, oldMin, oldMax, newMin, newMax);
}











GLayerRestrictedBoltzmannMachine::GLayerRestrictedBoltzmannMachine(size_t inputSize, size_t outputSize, GActivationFunction* pActivationFunction)
{
	m_pActivationFunction = pActivationFunction;
	if(!m_pActivationFunction)
		m_pActivationFunction = new GActivationLogistic();
	resize(inputSize, outputSize, NULL);
}

GLayerRestrictedBoltzmannMachine::GLayerRestrictedBoltzmannMachine(GDomNode* pNode)
: m_weights(pNode->field("weights")), m_bias(4, m_weights.rows()), m_biasReverse(4, m_weights.cols())
{
	bias().deserialize(pNode->field("bias"));
	biasReverse().deserialize(pNode->field("biasRev"));
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
	pNode->addField(pDoc, "bias", bias().serialize(pDoc));
	pNode->addField(pDoc, "biasRev", biasReverse().serialize(pDoc));
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
			double* pRow = m_weights[i].data() + fewerInputs;
			for(size_t j = fewerInputs; j < inputCount; j++)
				*(pRow++) = deviation * pRand->normal();
		}
		for(size_t i = fewerOutputs; i < outputCount; i++)
		{
			double* pRow = m_weights[i].data();
			for(size_t j = 0; j < inputCount; j++)
				*(pRow++) = deviation * pRand->normal();
		}
	}
	m_delta.resize(outputCount, inputCount);
	m_delta.setAll(0.0);

	// Bias
	m_bias.resizePreserve(5, outputCount);
	biasDelta().fill(0.0);
	if(pRand)
	{
		double* pB = bias().data() + fewerOutputs;
		for(size_t j = fewerOutputs; j < outputCount; j++)
			*(pB++) = deviation * pRand->normal();
	}

	// BiasReverse
	m_biasReverse.resizePreserve(5, inputCount);
	if(pRand)
	{
		double* pB = biasReverse().data() + fewerInputs;
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
	double* pB = bias().data();
	for(size_t i = 0; i < outputCount; i++)
	{
		*pB = rand.normal() * mag;
		for(size_t j = 0; j < inputCount; j++)
			m_weights[i][j] = rand.normal() * mag;
		pB++;
	}
	pB = biasReverse().data();
	for(size_t i = 0; i < inputCount; i++)
		*(pB++) = rand.normal() * mag;
	m_delta.setAll(0.0);
	biasDelta().fill(0.0);
	biasReverseDelta().fill(0.0);
}

// virtual
void GLayerRestrictedBoltzmannMachine::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	size_t n = std::min(outputs() - start, count);
	for(size_t i = start; i < n; i++)
		GVec::perturb(m_weights[i].data(), deviation, inputs(), rand);
	GVec::perturb(bias().data() + start, deviation, n, rand);
}

// virtual
void GLayerRestrictedBoltzmannMachine::feedForward(const GVec& in)
{
	net().copy(bias());
	size_t outputCount = outputs();
	double* pNet = net().data();
	for(size_t i = 0; i < outputCount; i++)
		*(pNet++) += in.dotProduct(m_weights[i]);

	// Activate
	double* pAct = activation().data();
	pNet = net().data();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++), i);
}

// virtual
void GLayerRestrictedBoltzmannMachine::dropOut(GRand& rand, double probOfDrop)
{
	double* pAct = activation().data();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			pAct[i] = 0.0;
	}
}

/*
void GLayerRestrictedBoltzmannMachine::feedForward(const double* in)
{
	// Feed through the weights
	double* pNet = net();
	m_weights.multiply(in, pNet);
	size_t outputCount = outputs();
	GVec::add(pNet, bias(), outputCount);

	// Squash it
	double* pAct = activation();
	for(size_t i = 0; i < outputCount; i++)
		*(pAct++) = m_pActivationFunction->squash(*(pNet++));
}
*/
void GLayerRestrictedBoltzmannMachine::feedBackward(const GVec& in)
{
	// Feed through the weights
	GVec& n = netReverse();
	m_weights.multiply(in, n, true);
	size_t inputCount = inputs();
	n += biasReverse();

	// Squash it
	GVec& a = activationReverse();
	for(size_t i = 0; i < inputCount; i++)
		a[i] = m_pActivationFunction->squash(n[i], i);
}

void GLayerRestrictedBoltzmannMachine::resampleHidden(GRand& rand)
{
	double* pH = activation().data();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		*pH = rand.uniform() < *pH ? 1.0 : 0.0;
		pH++;
	}
}

void GLayerRestrictedBoltzmannMachine::resampleVisible(GRand& rand)
{
	double* pV = activationReverse().data();
	size_t inputCount = inputs();
	for(size_t i = 0; i < inputCount; i++)
	{
		*pV = rand.uniform() < *pV ? 1.0 : 0.0;
		pV++;
	}
}

void GLayerRestrictedBoltzmannMachine::drawSample(GRand& rand, size_t iters)
{
	double* pH = activation().data();
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

double GLayerRestrictedBoltzmannMachine::freeEnergy(const GVec& visibleSample)
{
	feedForward(visibleSample);
	GVec& buf = error();
	m_weights.multiply(activationReverse(), buf, false);
	return -activation().dotProduct(buf) -
		biasReverse().dotProduct(activationReverse()) -
		bias().dotProduct(activation());
}

void GLayerRestrictedBoltzmannMachine::contrastiveDivergence(GRand& rand, const GVec& visibleSample, double learningRate, size_t gibbsSamples)
{
	// Details of this implementation were guided by http://axon.cs.byu.edu/~martinez/classes/678/Papers/guideTR.pdf, particularly Sections 3.2 and 3.3.

	// Sample hidden vector
	feedForward(visibleSample);

	// Compute positive gradient
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
		m_weights[i].addScaled(activation()[i], visibleSample);

	// Add positive gradient to the biases
	biasReverse().addScaled(learningRate, visibleSample);
	bias().addScaled(learningRate, activation());

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
		m_weights[i].addScaled(activation()[i], activationReverse());

	// Subtract negative gradient from biases
	biasReverse().addScaled(-learningRate, activationReverse());
	bias().addScaled(-learningRate, activation());
}

void GLayerRestrictedBoltzmannMachine::computeError(const GVec& target)
{
	size_t outputUnits = outputs();
	double* pAct = activation().data();
	double* pErr = error().data();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE)
			*pErr = 0.0;
		else
			*pErr = target[i] - *pAct;
		pAct++;
		pErr++;
	}
}

void GLayerRestrictedBoltzmannMachine::deactivateError()
{
	size_t outputUnits = outputs();
	GVec& err = error();
	GVec& n = net();
	GVec& a = activation();
	m_pActivationFunction->setError(err);
	for(size_t i = 0; i < outputUnits; i++)
		err[i] *= m_pActivationFunction->derivativeOfNet(n[i], a[i], i);
}

void GLayerRestrictedBoltzmannMachine::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	GVec& downStreamError = error();
	GVec& upStreamError = pUpStreamLayer->error();
	m_weights.multiply(downStreamError, upStreamError, true);
}

void GLayerRestrictedBoltzmannMachine::updateDeltas(const GVec& upStreamActivation, double momentum)
{
	size_t outputCount = outputs();
	GVec& err = error();
	GVec& bD = biasDelta();
	for(size_t i = 0; i < outputCount; i++)
	{
		m_delta[i] *= momentum;
		m_delta[i].addScaled(err[i], upStreamActivation);
		bD[i] *= momentum;
		bD[i] += err[i];
	}
}

// virtual
void GLayerRestrictedBoltzmannMachine::applyDeltas(double learningRate)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
		m_weights[i].addScaled(learningRate, m_delta[i]);
	bias().addScaled(learningRate, biasDelta());
}

void GLayerRestrictedBoltzmannMachine::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

void GLayerRestrictedBoltzmannMachine::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i].regularize_L1(amount);
	if(diminishBiases)
		bias().regularize_L1(amount);
}

// virtual
void GLayerRestrictedBoltzmannMachine::maxNorm(double min, double max)
{
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		double squaredMag = m_weights[i].squaredMagnitude();
		if(squaredMag > max * max)
		{
			double scal = max / sqrt(squaredMag);
			m_weights[i] *= scal;
		}
		else if(squaredMag < min * min)
		{
			double scal = min / sqrt(squaredMag);
			m_weights[i] *= scal;
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
	GVec::copy(pOutVector, bias().data(), outputs());
	pOutVector += outputs();
	m_weights.toVector(pOutVector);
	pOutVector += (inputs() * outputs());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (inputs() + 1) * outputs() + activationWeights;
}

// virtual
size_t GLayerRestrictedBoltzmannMachine::vectorToWeights(const double* pVector)
{
	GVec::copy(bias().data(), pVector, outputs());
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
	bias().copy(src->bias());
	m_pActivationFunction->copyWeights(src->m_pActivationFunction);
}

// virtual
void GLayerRestrictedBoltzmannMachine::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	size_t outputCount = outputs();
	double* pB = bias().data();
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
	biasDelta().fill(0.0);
	m_pActivationFunction->resize(m_bias.cols());
}

GLayerConvolutional1D::GLayerConvolutional1D(GDomNode* pNode)
: m_inputSamples((size_t)pNode->field("isam")->asInt()),
m_inputChannels((size_t)pNode->field("ichan")->asInt()),
m_outputSamples((size_t)pNode->field("osam")->asInt()),
m_kernelsPerChannel((size_t)pNode->field("kpc")->asInt()),
m_kernels(pNode->field("kern")),
m_delta(pNode->field("delt")),
m_activation(pNode->field("act")),
m_bias(pNode->field("bias")),
m_pActivationFunction(GActivationFunction::deserialize(pNode->field("act_func")))
{
}

GLayerConvolutional1D::~GLayerConvolutional1D()
{
  delete m_pActivationFunction;
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
void GLayerConvolutional1D::resize(size_t inputSize, size_t outputSize, GRand* pRand, double deviation)
{
	if(inputSize != m_inputSamples * m_inputChannels)
		throw Ex("Changing the size of GLayerConvolutional1D is not supported");
	if(outputSize != m_inputChannels * m_kernelsPerChannel * m_outputSamples)
		throw Ex("Changing the size of GLayerConvolutional1D is not supported");
}

// virtual
void GLayerConvolutional1D::resetWeights(GRand& rand)
{
	size_t kernelSize = m_kernels.cols();
	double mag = std::max(0.03, 1.0 / kernelSize);
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].fillNormal(rand, mag);
	m_delta.setAll(0.0);
	bias().fillNormal(rand, mag);
	biasDelta().fill(0.0);
}

// virtual
void GLayerConvolutional1D::feedForward(const GVec& in)
{
	// Copy bias to net
	for(size_t i = 0; i < m_outputSamples; i++)
		net().put(bias().size() * i, bias());

	// Feed in through
	size_t kernelSize = m_kernels.cols();
	GVec& n = net();
	size_t netPos = 0;
	size_t inPos = 0;
	for(size_t i = 0; i < m_outputSamples; i++) // for each output sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				GVec& w = m_kernels[kern++];
				double d = 0.0;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
					d += w[l] * in[inPos + l * m_inputChannels];
				n[netPos++] += d;
			}
			inPos++;
		}
	}

	// Activate
	GVec& a = activation();
	n.copy(net());
	size_t kernelCount = m_bias.cols();
	size_t pos = 0;
	for(size_t i = 0; i < m_outputSamples; i++)
	{
		for(size_t j = 0; j < kernelCount; j++)
		{
			a[pos] = m_pActivationFunction->squash(n[pos], j);
			pos++;
		}
	}
}

// virtual
void GLayerConvolutional1D::dropOut(GRand& rand, double probOfDrop)
{
	GVec& a = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			a[i] = 0.0;
	}
}

// virtual
void GLayerConvolutional1D::dropConnect(GRand& rand, double probOfDrop)
{
	throw Ex("Sorry, convolutional layers do not support dropConnect");
}

// virtual
void GLayerConvolutional1D::computeError(const GVec& target)
{
	size_t outputUnits = outputs();
	GVec& a = activation();
	GVec& err = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE)
			err[i] = 0.0;
		else
			err[i] = target[i] - a[i];
	}
}

// virtual
void GLayerConvolutional1D::deactivateError()
{
	size_t outputUnits = outputs();
	GVec& err = error();
	GVec& n = net();
	GVec& a = activation();
	m_pActivationFunction->setError(err);
	for(size_t i = 0; i < outputUnits; i++)
		err[i] *= m_pActivationFunction->derivativeOfNet(n[i], a[i], i);
}

// virtual
void GLayerConvolutional1D::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	GAssert(pUpStreamLayer->outputs() == inputs());
	GVec& upStreamErr = pUpStreamLayer->error();
	GVec& downStreamErr = error();
	size_t kernelSize = m_kernels.cols();
	upStreamErr.fill(0.0);
	size_t upPos = 0;
	size_t downPos = 0;
	for(size_t i = 0; i < m_outputSamples; i++) // for each sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				GVec& w = m_kernels[kern++];
				size_t samp = 0;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
				{
					upStreamErr[upPos + samp] += w[l] * downStreamErr[downPos];
					samp += m_inputChannels;
				}
				downPos++;
			}
			upPos++;
		}
	}
}

// virtual
void GLayerConvolutional1D::updateDeltas(const GVec& upStreamActivation, double momentum)
{
	m_delta.multiply(momentum);
	biasDelta() *= momentum;
	GVec& err = error();
	size_t kernelSize = m_kernels.cols();
	size_t errPos = 0;
	size_t upPos = 0;
	for(size_t i = 0; i < m_outputSamples; i++) // for each sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				GVec& d = m_delta[kern++];
				size_t upOfs = 0;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
				{
					d[l] += err[errPos] * upStreamActivation[upPos + upOfs];
					upOfs += m_inputChannels;
				}
				errPos++;
			}
			upPos++;
		}
	}
	errPos = 0;
	for(size_t i = 0; i < m_outputSamples; i++)
	{
		GVec& d = biasDelta();
		size_t bdPos = 0;
		for(size_t j = 0; j < m_inputChannels; j++)
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++)
				d[bdPos++] += err[errPos++];
		}
	}
}

// virtual
void GLayerConvolutional1D::applyDeltas(double learningRate)
{
	size_t n = m_kernels.rows();
	for(size_t i = 0; i < n; i++)
		m_kernels[i].addScaled(learningRate, m_delta[i]);
	bias().addScaled(learningRate, biasDelta());
}

// virtual
void GLayerConvolutional1D::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

// virtual
void GLayerConvolutional1D::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].regularize_L1(amount);
	if(diminishBiases)
		bias().regularize_L1(amount);
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
	GVec::copy(pOutVector, bias().data(), m_kernels.rows());
	pOutVector += m_kernels.rows();
	m_kernels.toVector(pOutVector);
	pOutVector += (m_kernels.rows() * m_kernels.cols());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (m_kernels.rows() + 1) * m_kernels.cols() + activationWeights;
}

// virtual
size_t GLayerConvolutional1D::vectorToWeights(const double* pVector)
{
	GVec::copy(bias().data(), pVector, m_kernels.rows());
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
	bias().copy(src->bias());
	m_pActivationFunction->copyWeights(src->m_pActivationFunction);
}

// virtual
void GLayerConvolutional1D::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	if(start != 0)
		throw Ex("Sorry, convolutional layers do not support perturbing weights for a subset of units");
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::perturb(m_kernels[i].data(), deviation, kernelSize, rand);
	GVec::perturb(bias().data(), deviation, m_kernels.rows(), rand);
}

// virtual
void GLayerConvolutional1D::maxNorm(double min, double max)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].clip(-max, max);
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
	biasDelta().fill(0.0);
	m_pActivationFunction->resize(m_kernelCount);
}

GLayerConvolutional2D::GLayerConvolutional2D(GDomNode* pNode)
: m_inputCols((size_t)pNode->field("icol")->asInt()),
m_inputRows((size_t)pNode->field("irow")->asInt()),
m_inputChannels((size_t)pNode->field("ichan")->asInt()),
m_outputCols((size_t)pNode->field("ocol")->asInt()),
m_outputRows((size_t)pNode->field("orow")->asInt()),
m_kernelsPerChannel((size_t)pNode->field("kpc")->asInt()),
m_kernelCount(m_inputChannels * m_kernelsPerChannel),
m_kernels(pNode->field("kern")),
m_delta(pNode->field("delt")),
m_activation(pNode->field("act")),
m_bias(pNode->field("bias")),
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
void GLayerConvolutional2D::resize(size_t inputSize, size_t outputSize, GRand* pRand, double deviation)
{
	if(inputSize != m_inputCols * m_inputRows * m_inputChannels)
		throw Ex("Changing the size of GLayerConvolutional2D is not supported");
	if(outputSize != m_inputChannels * m_kernelsPerChannel * m_outputCols * m_outputRows)
		throw Ex("Changing the size of GLayerConvolutional2D is not supported");
}

// virtual
void GLayerConvolutional2D::resetWeights(GRand& rand)
{
	size_t kernelSize = m_kernels.cols();
	double mag = std::max(0.03, 1.0 / (kernelSize * kernelSize));
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].fillNormal(rand, mag);
	m_delta.setAll(0.0);
	bias().fillNormal(rand, mag);
	biasDelta().fill(0.0);
}

// virtual
void GLayerConvolutional2D::feedForward(const GVec& in)
{
	// Copy the bias to the net
	GVec& n = net();
	GVec& b = bias();
	size_t netPos = 0;
	for(size_t j = 0; j < m_outputRows; j++)
	{
		for(size_t i = 0; i < m_outputCols; i++)
		{
			n.put(netPos, b);
			netPos += m_kernelCount;
		}
	}

	// Feed the input through
	size_t kernelSize = m_kernels.cols();
	netPos = 0;
	size_t inPos = 0;
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
					size_t inOfs = 0;
					for(size_t l = 0; l < kernelSize; l++) // for each kernel row...
					{
						GVec& w = m_kernels[kern++];
						for(size_t m = 0; m < kernelSize; m++) // for each kernel column... todo: the cyclomatic complexity of this method is ridiculous!
						{
							d += w[m] * in[inPos + inOfs];
							inOfs += m_inputChannels;
						}
					}
					n[netPos++] += d;
				}
				inPos++;
			}
		}
	}

	// Activate
	GVec& a = activation();
	size_t actPos = 0;
	for(size_t h = 0; h < m_outputRows; h++) // for each output row...
	{
		for(size_t i = 0; i < m_outputCols; i++) // for each output column...
		{
			for(size_t j = 0; j < m_kernelCount; j++)
			{
				a[actPos] = m_pActivationFunction->squash(n[actPos], j);
				actPos++;
			}
		}
	}
}

// virtual
void GLayerConvolutional2D::dropOut(GRand& rand, double probOfDrop)
{
	GVec& a = activation();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		if(rand.uniform() < probOfDrop)
			a[i] = 0.0;
	}
}

// virtual
void GLayerConvolutional2D::dropConnect(GRand& rand, double probOfDrop)
{
	throw Ex("Sorry, convolutional layers do not support dropConnect");
}

// virtual
void GLayerConvolutional2D::computeError(const GVec& target)
{
	size_t outputUnits = outputs();
	GVec& a = activation();
	GVec& err = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE)
			err[i] = 0.0;
		else
			err[i] = target[i] - a[i];
	}
}

// virtual
void GLayerConvolutional2D::deactivateError()
{
	size_t outputUnits = outputs();
	GVec& err = error();
	GVec& n = net();
	GVec& a = activation();
	m_pActivationFunction->setError(err);
	for(size_t i = 0; i < outputUnits; i++)
		err[i] *= m_pActivationFunction->derivativeOfNet(n[i], a[i], i);
}

// virtual
void GLayerConvolutional2D::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	GAssert(pUpStreamLayer->outputs() == inputs());
	GVec& upStreamErr = pUpStreamLayer->error();
	GVec& downStreamErr = error();
	size_t kernelSize = m_kernels.cols();
	upStreamErr.fill(0.0);
	size_t upPos = 0;
	size_t downPos = 0;
	for(size_t h = 0; h < m_outputRows; h++) // for each output row...
	{
		for(size_t i = 0; i < m_outputCols; i++) // for each output column...
		{
			size_t kern = 0;
			for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
				{
					size_t upOfs = 0;
					for(size_t l = 0; l < kernelSize; l++) // for each kernel row...
					{
						GVec& w = m_kernels[kern++];
						for(size_t m = 0; m < kernelSize; m++) // for each kernel column...
						{
							upStreamErr[upPos + upOfs] += w[m] * downStreamErr[downPos];
							upOfs += m_inputChannels;
						}
					}
					downPos++;
				}
				upPos++;
			}
		}
	}
}

// virtual
void GLayerConvolutional2D::updateDeltas(const GVec& upStreamActivation, double momentum)
{
	m_delta.multiply(momentum);
	biasDelta() *= momentum;
	GVec& err = error();
	size_t kernelSize = m_kernels.cols();
	size_t errPos = 0;
	size_t upPos = 0;
	for(size_t h = 0; h < m_outputRows; h++) // for each sample row...
	{
		for(size_t i = 0; i < m_outputCols; i++) // for each sample column...
		{
			size_t kern = 0;
			for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
				{
					size_t upOfs = 0;
					for(size_t l = 0; l < kernelSize; l++) // for each kernel row...
					{
						GVec& d = m_delta[kern++];
						for(size_t m = 0; m < kernelSize; m++) // for each kernel column...
						{
							d[m] += err[errPos] * upStreamActivation[upOfs];;
							upOfs += m_inputChannels;
						}
					}
					errPos++;
				}
				upPos++;
			}
		}
	}
	errPos = 0;
	
	for(size_t h = 0; h < m_outputRows; h++)
	{
		for(size_t i = 0; i < m_outputCols; i++)
		{
			GVec& d = biasDelta();
			size_t dPos = 0;
			for(size_t j = 0; j < m_inputChannels; j++)
			{
				for(size_t k = 0; k < m_kernelsPerChannel; k++)
				{
					d[dPos++] += err[errPos++];
				}
			}
		}
	}
}

// virtual
void GLayerConvolutional2D::applyDeltas(double learningRate)
{
	size_t n = m_kernels.rows();
	for(size_t i = 0; i < n; i++)
		m_kernels[i].addScaled(learningRate, m_delta[i]);
	bias().addScaled(learningRate, biasDelta());
}

// virtual
void GLayerConvolutional2D::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

// virtual
void GLayerConvolutional2D::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].regularize_L1(amount);
	if(diminishBiases)
		bias().regularize_L1(amount);
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
	GVec::copy(pOutVector, bias().data(), m_kernelCount);
	pOutVector += m_kernelCount;
	m_kernels.toVector(pOutVector);
	pOutVector += (m_kernels.rows() * m_kernels.cols());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return m_kernels.rows() * m_kernels.cols() + m_kernelCount + activationWeights;
}

// virtual
size_t GLayerConvolutional2D::vectorToWeights(const double* pVector)
{
	GVec::copy(bias().data(), pVector, m_kernelCount);
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
	bias().copy(src->bias());
	m_pActivationFunction->copyWeights(src->m_pActivationFunction);
}

// virtual
void GLayerConvolutional2D::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
	if(start != 0)
		throw Ex("Sorry, convolutional layers do not support perturbing weights for a subset of units");
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::perturb(m_kernels[i].data(), deviation, kernelSize, rand);
	GVec::perturb(bias().data(), deviation, m_kernelCount, rand);
}

// virtual
void GLayerConvolutional2D::maxNorm(double min, double max)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].clip(-max, max);
}

// virtual
void GLayerConvolutional2D::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	throw Ex("Sorry, convolutional layers do not support this method");
}
















GMaxPooling2D::GMaxPooling2D(size_t inputCols, size_t inputRows, size_t inputChannels, size_t regionSize)
: m_inputCols(inputCols),
m_inputRows(inputRows),
m_inputChannels(inputChannels)
{
	if(inputCols % regionSize != 0)
		throw Ex("inputCols is not a multiple of regionSize");
	if(inputRows % regionSize != 0)
		throw Ex("inputRows is not a multiple of regionSize");
	m_activation.resize(2, m_inputRows * m_inputCols * m_inputChannels / (m_regionSize * m_regionSize));
}

GMaxPooling2D::GMaxPooling2D(GDomNode* pNode)
: m_inputCols((size_t)pNode->field("icol")->asInt()),
m_inputRows((size_t)pNode->field("irow")->asInt()),
m_inputChannels((size_t)pNode->field("ichan")->asInt()),
m_regionSize((size_t)pNode->field("size")->asInt())
{
}

GMaxPooling2D::~GMaxPooling2D()
{
}

// virtual
GDomNode* GMaxPooling2D::serialize(GDom* pDoc)
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "icol", pDoc->newInt(m_inputCols));
	pNode->addField(pDoc, "irow", pDoc->newInt(m_inputRows));
	pNode->addField(pDoc, "ichan", pDoc->newInt(m_inputChannels));
	pNode->addField(pDoc, "size", pDoc->newInt(m_regionSize));
	return pNode;
}

// virtual
void GMaxPooling2D::resize(size_t inputSize, size_t outputSize, GRand* pRand, double deviation)
{
	if(inputSize != m_inputCols * m_inputRows * m_inputChannels)
		throw Ex("Changing the size of GMaxPooling2D is not supported");
	if(outputSize != m_inputChannels * m_inputCols * m_inputRows / (m_regionSize * m_regionSize))
		throw Ex("Changing the size of GMaxPooling2D is not supported");
}

// virtual
void GMaxPooling2D::resetWeights(GRand& rand)
{
}

// virtual
void GMaxPooling2D::feedForward(const GVec& in)
{
	GVec& a = activation();
	size_t actPos = 0;
	for(size_t yy = 0; yy < m_inputRows; yy += m_regionSize)
	{
		for(size_t xx = 0; xx < m_inputCols; xx += m_regionSize)
		{
			for(size_t c = 0; c < m_inputChannels; c++)
			{
				double m = -1e100;
				size_t yStep = m_inputCols * m_inputChannels;
				size_t yStart = yy * yStep;
				size_t yEnd = yStart + m_regionSize * yStep;
				for(size_t y = yStart; y < yEnd; y += yStep)
				{
					size_t xStart = yStart + xx * m_inputChannels + c;
					size_t xEnd = xStart + m_regionSize * m_inputChannels + c;
					for(size_t x = xStart; x < xEnd; x += m_inputChannels)
						m = std::max(m, in[x]);
				}
				a[actPos++] = m;
			}
		}
	}
}

// virtual
void GMaxPooling2D::dropOut(GRand& rand, double probOfDrop)
{
}

// virtual
void GMaxPooling2D::dropConnect(GRand& rand, double probOfDrop)
{
}

// virtual
void GMaxPooling2D::computeError(const GVec& target)
{
	size_t outputUnits = outputs();
	GVec& a = activation();
	GVec& err = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE)
			err[i] = 0.0;
		else
			err[i] = target[i] - a[i];
	}
}

// virtual
void GMaxPooling2D::deactivateError()
{
}

// virtual
void GMaxPooling2D::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	GVec& downStreamErr = error();
	GVec& a = pUpStreamLayer->activation();
	GVec& upStreamErr = pUpStreamLayer->error();
	size_t downPos = 0;
	for(size_t yy = 0; yy < m_inputRows; yy += m_regionSize)
	{
		for(size_t xx = 0; xx < m_inputCols; xx += m_regionSize)
		{
			for(size_t c = 0; c < m_inputChannels; c++)
			{
				double m = -1e100;
				size_t maxIndex = 0;
				size_t yStep = m_inputCols * m_inputChannels;
				size_t yStart = yy * yStep;
				size_t yEnd = yStart + m_regionSize * yStep;
				for(size_t y = yStart; y < yEnd; y += yStep)
				{
					size_t xStart = yStart + xx * m_inputChannels + c;
					size_t xEnd = xStart + m_regionSize * m_inputChannels + c;
					for(size_t x = xStart; x < xEnd; x += m_inputChannels)
					{
						if(a[x] > m)
						{
							m = a[x];
							maxIndex = x;
						}
						upStreamErr[x] = 0.0;
					}
				}
				upStreamErr[maxIndex] = downStreamErr[downPos++];
			}
		}
	}
}

// virtual
void GMaxPooling2D::updateDeltas(const GVec& upStreamActivation, double momentum)
{
}

// virtual
void GMaxPooling2D::applyDeltas(double learningRate)
{
}

// virtual
void GMaxPooling2D::scaleWeights(double factor, bool scaleBiases)
{
}

// virtual
void GMaxPooling2D::diminishWeights(double amount, bool diminishBiases)
{
}

// virtual
void GMaxPooling2D::regularizeActivationFunction(double lambda)
{
}

// virtual
size_t GMaxPooling2D::countWeights()
{
	return 0;
}

// virtual
size_t GMaxPooling2D::weightsToVector(double* pOutVector)
{
	return 0;
}

// virtual
size_t GMaxPooling2D::vectorToWeights(const double* pVector)
{
	return 0;
}

// virtual
void GMaxPooling2D::copyWeights(const GNeuralNetLayer* pSource)
{
}

// virtual
void GMaxPooling2D::perturbWeights(GRand& rand, double deviation, size_t start, size_t count)
{
}

// virtual
void GMaxPooling2D::maxNorm(double min, double max)
{
}

// virtual
void GMaxPooling2D::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	throw Ex("Sorry, max poolings layers do not support this method");
}




} // namespace GClasses

