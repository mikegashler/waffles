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
	else if(strcmp(szType, "mixed") == 0)
		return new GLayerMixed(pNode);
	else if(strcmp(szType, "rbm") == 0)
		return new GLayerRestrictedBoltzmannMachine(pNode);
	else if(strcmp(szType, "softmax") == 0)
		return new GLayerSoftMax(pNode);
	else if(strcmp(szType, "conv1d") == 0)
		return new GLayerConvolutional1D(pNode);
	else if(strcmp(szType, "conv2d") == 0)
		return new GLayerConvolutional2D(pNode);
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

// virtual
std::string GLayerClassic::to_str()
{
	std::ostringstream os;
	os << "[GLayerClassic:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "\n";
	os << " Weights: " << GClasses::to_str(m_weights) << "\n";
	os << " Bias: " << GClasses::to_str(m_bias) << "\n";
	os << "]";
	return os.str();
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

// virtual
void GLayerClassic::applyAdaptive()
{
	// Lazily make a place to store adaptive learning rates
	while(m_delta.rows() <= m_weights.rows() + m_weights.rows()) // all the weights + the bias vector
		m_delta.newRow().fill(0.01);

	// Adapt the learning rates
	size_t inputCount = inputs();
	size_t outputCount = outputs();
	for(size_t i = 0; i < inputCount; i++)
	{
		GVec& delta = m_delta[i];
		GVec& rates = m_delta[m_weights.rows() + i];
		for(size_t j = 0; j < outputCount; j++)
		{
			if(std::signbit(delta[j]) == std::signbit(rates[j]))
			{
				if(std::abs(rates[j]) < 1e3)
					rates[j] *= 1.2;
			}
			else
			{
				if(std::abs(rates[j]) > 1e-8)
					rates[j] *= -0.2;
				else
					rates[j] *= -1.1;
			}
		}
	}
	GVec& delta = biasDelta();
	GVec& rates = m_delta[m_weights.rows() + m_weights.rows()];
	for(size_t j = 0; j < outputCount; j++)
	{
		if(std::signbit(delta[j]) == std::signbit(rates[j]))
		{
			if(std::abs(rates[j]) < 1e3)
				rates[j] *= 1.2;
		}
		else
		{
			if(std::abs(rates[j]) > 1e-8)
				rates[j] *= -0.2;
			else
				rates[j] *= -1.1;
		}
	}

	// Update the weights and bias
	for(size_t i = 0; i < inputCount; i++)
		m_weights[i] += m_delta[m_weights.rows() + i];
	bias() += m_delta[m_weights.rows() + m_weights.rows()];
	m_pActivationFunction->applyAdaptive();
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
	memcpy(pOutVector, bias().data(), sizeof(double) * outputs());
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
		bias() *= fac;
		activation() *= fac;
	}
	else
	{
		for(size_t i = 0; i < outputCount; i++)
			activation()[i] = 1.0 / outputCount;
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

// virtual
std::string GLayerMixed::to_str()
{
	std::ostringstream os;
	os << "[GLayerMixed:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "\n";
	for(size_t i = 0; i < m_components.size(); i++)
		os << m_components[i]->to_str() << "\n";
	os << "]";
	return os.str();
}

void GLayerMixed::addComponent(GNeuralNetLayer* pComponent)
{
	if(m_activation.cols() > 0)
		throw Ex("Cannot add a component to GLayerMixed after it has been used");
	if(m_inputError.cols() == 0)
		m_inputError.resize(1, pComponent->inputs());
	else if(m_inputError.cols() != pComponent->inputs())
		throw Ex("This component expects ", GClasses::to_str(pComponent->inputs()), ", inputs, which conflicts with a previous component that expects ", GClasses::to_str(m_inputError.cols()), " inputs");
	m_components.push_back(pComponent);
}

// virtual
size_t GLayerMixed::inputs() const
{
	return m_inputError.cols();
}

// virtual
size_t GLayerMixed::outputs() const
{
	size_t outs = m_activation.cols();
	if(outs == 0)
	{
		if(m_components.size() < 2)
			throw Ex("GLayerMixed requires at least 2 components to be added before it is used");
		for(size_t i = 0; i < m_components.size(); i++)
			outs += m_components[i]->outputs();
		((GMatrix*)&m_activation)->resize(2, outs); // !!!HACK: this circumvents the "const" declaration on this function!
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
		memcpy(pAct, m_components[i]->activation().data(), m_components[i]->outputs() * sizeof(double));
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
		memcpy(m_components[i]->error().data(), pErr, m_components[i]->outputs() * sizeof(double));
		m_components[i]->deactivateError();
		pErr += m_components[i]->outputs();
	}
}

// virtual
void GLayerMixed::backPropError(GNeuralNetLayer* pUpStreamLayer)
{
	double* pBuf = m_inputError[0].data();
	size_t inps = pUpStreamLayer->outputs();
	m_inputError[0].fill(0.0);
	for(size_t i = 0; i < m_components.size(); i++)
	{
		m_components[i]->backPropError(pUpStreamLayer);
		m_inputError[0] += pUpStreamLayer->error();
	}
	memcpy(pUpStreamLayer->error().data(), pBuf, inps * sizeof(double));
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
void GLayerMixed::applyAdaptive()
{
	for(size_t i = 0; i < m_components.size(); i++)
		m_components[i]->applyAdaptive();
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

// virtual
std::string GLayerRestrictedBoltzmannMachine::to_str()
{
	std::ostringstream os;
	os << "[GLayerRestrictedBoltzmannMachine:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "\n";
	os << " Weights: " << GClasses::to_str(m_weights) << "\n";
	os << " Bias: " << GClasses::to_str(bias()) << "\n";
	os << " BiasReverse: " << GClasses::to_str(biasReverse()) << "\n";
	os << "]";
	return os.str();
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

// virtual
void GLayerRestrictedBoltzmannMachine::applyAdaptive()
{
	throw new Ex("Sorry, not implemented yet");
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
	memcpy(pOutVector, bias().data(), outputs() * sizeof(double));
	pOutVector += outputs();
	m_weights.toVector(pOutVector);
	pOutVector += (inputs() * outputs());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (inputs() + 1) * outputs() + activationWeights;
}

// virtual
size_t GLayerRestrictedBoltzmannMachine::vectorToWeights(const double* pVector)
{
	memcpy(bias().data(), pVector, outputs() * sizeof(double));
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
std::string GLayerConvolutional1D::to_str()
{
	std::ostringstream os;
	os << "[GLayerConvolutional1D:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "\n";
	os << " Kernels: " << GClasses::to_str(m_kernels) << "\n";
	os << "]";
	return os.str();
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
void GLayerConvolutional1D::applyAdaptive()
{
	throw new Ex("Sorry, not implemented yet");
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
	memcpy(pOutVector, bias().data(), m_kernels.rows() * sizeof(double));
	pOutVector += m_kernels.rows();
	m_kernels.toVector(pOutVector);
	pOutVector += (m_kernels.rows() * m_kernels.cols());
	size_t activationWeights = m_pActivationFunction->weightsToVector(pOutVector);
	return (m_kernels.rows() + 1) * m_kernels.cols() + activationWeights;
}

// virtual
size_t GLayerConvolutional1D::vectorToWeights(const double* pVector)
{
	memcpy(bias().data(), pVector, m_kernels.rows() * sizeof(double));
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






size_t GLayerConvolutional2D::Image::npos = (size_t) -1;

GLayerConvolutional2D::Image::Image(GVec *_data, size_t _width, size_t _height, size_t _channels)
: data(_data), width(_width), height(_height), channels(_channels), interlaced(true), dx(0), dy(0), dz(0), px(0), py(0), sx(1), sy(1), invertStride(false), flip(false) {}

size_t GLayerConvolutional2D::Image::index(size_t x, size_t y, size_t z) const
{
	z += dz;
	
	if(invertStride)
	{
		if((x + dx) % sx > 0 || (y + dy) % sy > 0)
			return npos;
		x = (x + dx) / sx - px;
		y = (y + dy) / sy - py;
	}
	else
	{
		x += dx * sx - px;
		y += dy * sy - py;
	}
	
	if(flip)
	{
		x = width - x - 1;
		y = height - y - 1;
	}
	
	if(x >= width || y >= height)
		return -npos;
	
	if(interlaced)
		return (y * width + x) * channels + z;
	else
		return (z * height + y) * width + x;
}

double GLayerConvolutional2D::Image::read(size_t x, size_t y, size_t z) const
{
	size_t i = index(x, y, z);
	if(i == npos)
		return 0.0;
	else
		return (*data)[i];
}

double &GLayerConvolutional2D::Image::at(size_t x, size_t y, size_t z)
{
	size_t i = index(x, y, z);
	if(i == npos)
		throw Ex("tried to access invalid image location!");
	else
		return (*data)[i];
}


size_t GLayerConvolutional2D::none = (size_t) -1;

GLayerConvolutional2D::GLayerConvolutional2D(size_t width, size_t height, size_t channels, size_t kWidth, size_t kHeight, size_t kCount, GActivationFunction *pActivationFunction)
: m_width(width), m_height(height), m_channels(channels),
  m_kWidth(kWidth), m_kHeight(kHeight),
  m_outputWidth(width - kWidth + 1), m_outputHeight(height - kHeight + 1),
  m_bias(kCount), m_biasDelta(kCount),
  m_kernels(kCount, kWidth * kHeight * channels), m_deltas(kCount, kWidth * kHeight * channels),
  m_activation(3, m_outputWidth * m_outputHeight * kCount),
  m_pActivationFunction(pActivationFunction ? pActivationFunction : new GActivationTanH()),
  m_kernelImage(NULL, kWidth, kHeight, channels), m_deltaImage(NULL, kWidth, kHeight, channels),
  m_inputImage(NULL, width, height, channels), m_upStreamErrorImage(NULL, width, height, channels),
  m_netImage(&m_activation[0], m_outputWidth, m_outputHeight, kCount), m_actImage(&m_activation[1], m_outputWidth, m_outputHeight, kCount), m_errImage(&m_activation[2], m_outputWidth, m_outputHeight, kCount)
{}

GLayerConvolutional2D::GLayerConvolutional2D(size_t kWidth, size_t kHeight, size_t kCount, GActivationFunction *pActivationFunction)
: m_width(FLEXIBLE_SIZE), m_height(FLEXIBLE_SIZE), m_channels(FLEXIBLE_SIZE),
  m_kWidth(kWidth), m_kHeight(kHeight),
  m_outputWidth(0), m_outputHeight(0),
  m_bias(kCount), m_biasDelta(kCount),
  m_kernels(kCount, 0), m_deltas(kCount, 0),
  m_activation(3, 0),
  m_pActivationFunction(pActivationFunction ? pActivationFunction : new GActivationTanH()),
  m_kernelImage(NULL, kWidth, kHeight, 0), m_deltaImage(NULL, kWidth, kHeight, 0),
  m_inputImage(NULL, 0, 0, 0), m_upStreamErrorImage(NULL, 0, 0, 0),
  m_netImage(&m_activation[0], 0, 0, 0), m_actImage(&m_activation[1], 0, 0, 0), m_errImage(&m_activation[2], 0, 0, 0)
{}

GLayerConvolutional2D::GLayerConvolutional2D(GDomNode* pNode)
: m_width(pNode->field("width")->asInt()), m_height(pNode->field("height")->asInt()), m_channels(pNode->field("channels")->asInt()),
  m_kWidth(pNode->field("kWidth")->asInt()), m_kHeight(pNode->field("kHeight")->asInt()),
  m_outputWidth(pNode->field("outputWidth")->asInt()), m_outputHeight(pNode->field("outputHeight")->asInt()),
  m_bias(pNode->field("bias")), m_biasDelta(m_bias.size()),
  m_kernels(pNode->field("kernels")), m_deltas(m_kernels.rows(), m_kernels.cols()),
  m_activation(3, m_outputWidth * m_outputHeight * m_kernels.rows()),
  m_kernelImage(NULL, m_kWidth, m_kHeight, m_channels), m_deltaImage(NULL, m_kWidth, m_kHeight, m_channels),
  m_inputImage(NULL, m_width, m_height, m_channels), m_upStreamErrorImage(NULL, m_width, m_height, m_channels),
  m_netImage(&m_activation[0], m_outputWidth, m_outputHeight, m_kernels.rows()), m_actImage(&m_activation[1], m_outputWidth, m_outputHeight, m_kernels.rows()), m_errImage(&m_activation[2], m_outputWidth, m_outputHeight, m_kernels.rows())
{
	m_inputImage.sx	= pNode->field("strideX")->asInt();
	m_inputImage.sy	= pNode->field("strideY")->asInt();
	m_inputImage.px	= pNode->field("paddingX")->asInt();
	m_inputImage.py	= pNode->field("paddingY")->asInt();
	m_pActivationFunction = GActivationFunction::deserialize(pNode->field("act_func"));
}

GLayerConvolutional2D::~GLayerConvolutional2D()
{
	delete m_pActivationFunction;
}

GDomNode *GLayerConvolutional2D::serialize(GDom *pDoc)
{
	GDomNode *pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "width", pDoc->newInt(m_width));
	pNode->addField(pDoc, "height", pDoc->newInt(m_height));
	pNode->addField(pDoc, "channels", pDoc->newInt(m_channels));
	pNode->addField(pDoc, "kWidth", pDoc->newInt(m_kWidth));
	pNode->addField(pDoc, "kHeight", pDoc->newInt(m_kHeight));
	pNode->addField(pDoc, "strideX", pDoc->newInt(m_inputImage.sx));
	pNode->addField(pDoc, "strideY", pDoc->newInt(m_inputImage.sy));
	pNode->addField(pDoc, "paddingX", pDoc->newInt(m_inputImage.px));
	pNode->addField(pDoc, "paddingY", pDoc->newInt(m_inputImage.py));
	pNode->addField(pDoc, "outputWidth", pDoc->newInt(m_outputWidth));
	pNode->addField(pDoc, "outputHeight", pDoc->newInt(m_outputHeight));
	pNode->addField(pDoc, "bias", m_bias.serialize(pDoc));
	pNode->addField(pDoc, "kernels", m_kernels.serialize(pDoc));
	pNode->addField(pDoc, "act_func", m_pActivationFunction->serialize(pDoc));
	return pNode;

}

std::string GLayerConvolutional2D::to_str()
{
	std::stringstream ss;
	ss << "[GLayerConvolutional2D:\n"
	   << "    " << m_width << "x" << m_height << "x" << m_channels << " (stride=" << m_inputImage.sx << "," << m_inputImage.sy << "; padding=" << m_inputImage.px << "," << m_inputImage.py << ")\n"
	   << " *  " << m_kWidth << "x" << m_kHeight << "\n"
	   << " -> " << m_outputWidth << "x" << m_outputHeight << "x" << m_kernels.rows() << "\n"
	   << "]";
	return ss.str();
}

void GLayerConvolutional2D::resize(size_t inputSize, size_t outputSize, GRand *pRand, double deviation)
{
	if(inputSize != inputs() || outputSize != outputs())
		throw Ex("GLayerConvolutional2D can only be resized given an upstream convolutional layer!");
}

void GLayerConvolutional2D::resizeInputs(GNeuralNetLayer *pUpStreamLayer, GRand *pRand, double deviation)
{
	if(strcmp(pUpStreamLayer->type(), "conv2d") != 0)
		throw Ex("GLayerConvolutional2D can only be resized given an upstream convolutional layer!");
	
	GLayerConvolutional2D &upstream = *((GLayerConvolutional2D *) pUpStreamLayer);
	
	m_width			= upstream.outputWidth();
	m_height		= upstream.outputHeight();
	m_channels		= upstream.outputChannels();
	
	m_kernels.resize(m_kernels.rows(), m_kWidth * m_kHeight * m_channels);
	m_deltas.resize(m_kernels.rows(), m_kWidth * m_kHeight * m_channels);
	
	m_bias.fill(0.0);
	m_kernels.setAll(0.0);
	
	m_inputImage.width = m_width;
	m_inputImage.height = m_height;
	m_inputImage.channels = m_channels;
	
	m_upStreamErrorImage.width = m_width;
	m_upStreamErrorImage.height = m_height;
	m_upStreamErrorImage.channels = m_channels;
	
	m_kernelImage.channels = m_channels;
	m_deltaImage.channels = m_channels;
	
	updateOutputSize();
	
	if(pRand)
		perturbWeights(*pRand, deviation);
}

void GLayerConvolutional2D::feedForward(const GVec &in)
{
	m_inputImage.data = const_cast<GVec *>(&in);
	
	Image &n = m_netImage;
	Image &a = m_actImage;
	
	n.data->fill(0.0);
	for(n.dz = 0; n.dz < n.channels; ++n.dz)
	{
		m_kernelImage.data = &m_kernels[n.dz];
		convolve(m_inputImage, m_kernelImage, n);
		for(size_t y = 0; y < n.height; ++y)
		{
			for(size_t x = 0; x < n.width; ++x)
			{
				n.at(x, y) += m_bias[n.dz];
				a.at(x, y, n.dz) = m_pActivationFunction->squash(n.at(x, y));
			}
		}
	}
	n.dz = 0;
}

void GLayerConvolutional2D::dropOut(GRand &rand, double probOfDrop)
{
	throw Ex("dropOut not implemented");
}

void GLayerConvolutional2D::dropConnect(GRand &rand, double probOfDrop)
{
	throw Ex("dropConnect not implemented");
}

void GLayerConvolutional2D::computeError(const GVec &target)
{
	size_t outputUnits = outputs();
	GVec &a = activation();
	GVec &err = error();
	for(size_t i = 0; i < outputUnits; i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE || target[i] == a[i])
			err[i] = 0.0;
		else
			err[i] = target[i] - a[i];
	}
}

void GLayerConvolutional2D::deactivateError()
{
	GVec &n = net();
	GVec &act = activation();
	GVec &err = error();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
		err[i] *= m_pActivationFunction->derivativeOfNet(n[i], act[i]);
}

void GLayerConvolutional2D::backPropError(GNeuralNetLayer *pUpStreamLayer)
{
	Image &err = m_errImage;
	Image &upErr = m_upStreamErrorImage;
	
	upErr.data = &pUpStreamLayer->error();
	upErr.data->fill(0.0);
	upErr.px = m_inputImage.px;
	upErr.py = m_inputImage.py;
	
	err.invertStride = true, err.sx = m_inputImage.sx, err.sy = m_inputImage.sy;
	for(upErr.dz = 0; upErr.dz < upErr.channels; ++upErr.dz)
	{
		for(err.dz = 0; err.dz < err.channels; ++err.dz)
		{
			m_kernelImage.data = &m_kernels[err.dz];
			m_kernelImage.flip = true, m_kernelImage.dz = upErr.dz;
			convolveFull(err, m_kernelImage, upErr, 1);
			m_kernelImage.flip = false, m_kernelImage.dz = 0;
		}
	}
	err.sx = err.sy = 1, err.invertStride = false;
	err.dz = 0;
	upErr.dz = 0;
	upErr.px = upErr.py = 0;
}

void GLayerConvolutional2D::updateDeltas(const GVec &upStreamActivation, double momentum)
{
	m_biasDelta *= momentum;
	m_deltas.multiply(momentum);
	
	Image &err = m_errImage;
	Image &in = m_inputImage;
	in.data = const_cast<GVec *>(&upStreamActivation);
	
	for(err.dz = 0; err.dz < err.channels; ++err.dz)
	{
		m_deltaImage.data = &m_deltas[err.dz];
		m_deltaImage.data->fill(0.0);
		for(in.dz = m_deltaImage.dz = 0; in.dz < in.channels; ++in.dz, ++m_deltaImage.dz)
			for(in.dy = 0; in.dy < err.height; ++in.dy)
				for(in.dx = 0; in.dx < err.width; ++in.dx)
					addScaled(in, err.read(in.dx, in.dy), m_deltaImage);
		m_deltaImage.dz = 0;
	}
	in.dz = 0;
}

void GLayerConvolutional2D::applyDeltas(double learningRate)
{
	m_bias.addScaled(learningRate, m_biasDelta);
	for(size_t i = 0; i < m_deltas.rows(); i++)
		m_kernels[i].addScaled(learningRate, m_deltas[i]);
}

void GLayerConvolutional2D::applyAdaptive()
{
	throw Ex("not implemented");
}

void GLayerConvolutional2D::scaleWeights(double factor, bool scaleBiases)
{
	throw Ex("scaleWeights not implemented");
}

void GLayerConvolutional2D::diminishWeights(double amount, bool regularizeBiases)
{
	throw Ex("diminishWeights not implemented");
}

size_t GLayerConvolutional2D::countWeights()
{
	return m_kWidth * m_kHeight * m_channels * m_kernels.rows() + m_kernels.rows();
}

size_t GLayerConvolutional2D::weightsToVector(double *pOutVector)
{
	m_kernels.toVector(pOutVector);
	GVecWrapper(pOutVector + m_kernels.rows() * m_kernels.cols(), m_kernels.rows()).vec().put(0, m_bias);
	return countWeights();
}

size_t GLayerConvolutional2D::vectorToWeights(const double *pVector)
{
	m_kernels.fromVector(pVector, m_kernels.rows());
	m_bias.put(0, GConstVecWrapper(pVector + m_kernels.rows() * m_kernels.cols(), m_kernels.rows()).vec());
	return countWeights();
}

void GLayerConvolutional2D::copyWeights(const GNeuralNetLayer *pSource)
{
	throw Ex("copyWeights not implemented");
}

void GLayerConvolutional2D::resetWeights(GRand &rand)
{
	double mag = std::max(0.03, 1.0 / (m_outputWidth * m_outputHeight * m_kernels.rows()));
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].fillNormal(rand, mag);
	m_bias.fillNormal(rand, mag);
	m_deltas.setAll(0.0);
	m_biasDelta.fill(0.0);
}

void GLayerConvolutional2D::perturbWeights(GRand &rand, double deviation, size_t start, size_t count)
{
	GAssert(start + count < m_kernels.rows());
	size_t n = std::min(m_kernels.rows() - start, count);
	for(size_t j = start; j < n; j++)
		GVec::perturb(m_kernels[j].data(), deviation, m_kernels.cols(), rand);
	GVec::perturb(m_bias.data(), deviation, m_kernels.rows(), rand);
}

void GLayerConvolutional2D::maxNorm(double min, double max)
{
	throw Ex("maxNorm not implemented");
}

void GLayerConvolutional2D::regularizeActivationFunction(double lambda)
{
	throw Ex("regularizeActivationFunction not implemented");
}

void GLayerConvolutional2D::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
{
	throw Ex("renormalizeInput not implemented");
}

void GLayerConvolutional2D::setPadding(size_t px, size_t py)
{
	m_inputImage.px = px;
	m_inputImage.py = py == none ? px : py;
	updateOutputSize();
}

void GLayerConvolutional2D::setStride(size_t sx, size_t sy)
{
	m_inputImage.sx = sx;
	m_inputImage.sy = sy == none ? sx : sy;
	updateOutputSize();
}

void GLayerConvolutional2D::setInterlaced(bool interlaced)
{
	setInputInterlaced(interlaced);
	setKernelsInterlaced(interlaced);
	setOutputInterlaced(interlaced);
}

void GLayerConvolutional2D::setInputInterlaced(bool interlaced)
{
	m_inputImage.interlaced = interlaced;
	m_upStreamErrorImage.interlaced = interlaced;
}

void GLayerConvolutional2D::setKernelsInterlaced(bool interlaced)
{
	m_kernelImage.interlaced = interlaced;
	m_deltaImage.interlaced = interlaced;
}

void GLayerConvolutional2D::setOutputInterlaced(bool interlaced)
{
	m_netImage.interlaced = interlaced;
	m_actImage.interlaced = interlaced;
	m_errImage.interlaced = interlaced;
}

void GLayerConvolutional2D::addKernel()
{
	m_kernels.resizePreserve(m_kernels.rows() + 1, m_kernels.cols());
	m_deltas.resize(m_kernels.rows(), m_kernels.cols());
	
	GVec temp(m_bias);
	m_bias.resize(m_kernels.rows() + 1);
	m_bias.put(0, temp);
	
	m_biasDelta.resize(m_kernels.rows());
	m_netImage.channels = m_kernels.rows();
	m_actImage.channels = m_kernels.rows();
	m_errImage.channels = m_kernels.rows();
	updateOutputSize();
}

void GLayerConvolutional2D::addKernels(size_t n)
{
	for(size_t i = 0; i < n; ++i)
		addKernel();
}

double GLayerConvolutional2D::filterSum(const Image &in, const Image &filter, size_t channels)
{
	double output = 0.0;
	for(size_t z = 0; z < channels; ++z)
		for(size_t y = 0; y < filter.height; ++y)
			for(size_t x = 0; x < filter.width; ++x)
				output += in.read(x, y, z) * filter.read(x, y, z);
	return output;
}

void GLayerConvolutional2D::addScaled(const Image &in, double scalar, Image &out)
{
	for(size_t y = 0; y < out.height; ++y)
		for(size_t x = 0; x < out.width; ++x)
			out.at(x, y) += in.read(x, y) * scalar;
}

void GLayerConvolutional2D::convolve(const Image &in, const Image &filter, Image &out, size_t channels)
{
	size_t x, y;
	if(channels == none)
		channels = filter.channels;
	for(y = 0, in.dy = out.py; y < out.height; ++y, ++in.dy)
		for(x = 0, in.dx = out.px; x < out.width; ++x, ++in.dx)
			out.at(in.dx, in.dy, 0) += filterSum(in, filter, channels);
	in.dx = in.dy = 0;
}

void GLayerConvolutional2D::convolveFull(const Image &in, const Image &filter, Image &out, size_t channels)
{
	size_t px = in.px, py = in.py;
	in.px = (in.px + filter.width - 1) / in.sx, in.py = (in.py + filter.height - 1) / in.sy;
	convolve(in, filter, out, channels);
	in.px = px, in.py = py;
}

void GLayerConvolutional2D::updateOutputSize()
{
	m_outputWidth = (m_width - m_kWidth + 2 * m_inputImage.px) / m_inputImage.sx + 1;
	m_outputHeight = (m_height - m_kHeight + 2 * m_inputImage.py) / m_inputImage.sy + 1;
	m_activation.resize(3, m_outputWidth * m_outputHeight * m_kernels.rows());
	
	m_netImage.data = &m_activation[0];
	m_netImage.width = m_outputWidth;
	m_netImage.height = m_outputHeight;
	
	m_actImage.data = &m_activation[1];
	m_actImage.width = m_outputWidth;
	m_actImage.height = m_outputHeight;
	
	m_errImage.data = &m_activation[2];
	m_errImage.width = m_outputWidth;
	m_errImage.height = m_outputHeight;
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
std::string GMaxPooling2D::to_str()
{
	std::ostringstream os;
	os << "[GMaxPooling2D:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "]";
	return os.str();
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
void GMaxPooling2D::applyAdaptive()
{
	throw new Ex("Sorry, not implemented yet");
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
