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
#include <string>
#include <sstream>
#include "GOptimizer.h"
#include "GString.h"

using std::vector;

namespace GClasses {

GBlock::GBlock(size_t inputs, size_t outputs)
: inputCount(inputs), outputCount(outputs)
{
}

GBlock::GBlock(const GBlock& that)
: inputCount(that.inputCount), outputCount(that.outputCount), m_inPos(that.m_inPos)
{
	input.setData(that.input);
	output.setData(((GBlock*)&that)->output);
	outBlame.setData(((GBlock*)&that)->outBlame);
	inBlame.setData(((GBlock*)&that)->inBlame);
}

GBlock::GBlock(GDomNode* pNode)
{
	m_inPos = pNode->field("inpos")->asInt();
	inputCount = pNode->field("in")->asInt();
	outputCount = pNode->field("out")->asInt();
}

GDomNode* GBlock::baseDomNode(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "type", pDoc->newInt(type()));
	pNode->addField(pDoc, "inpos", pDoc->newInt(m_inPos));
	pNode->addField(pDoc, "in", pDoc->newInt(inputCount));
	pNode->addField(pDoc, "out", pDoc->newInt(outputCount));
	return pNode;
}

GDomNode* GBlock::serialize(GDom* pDoc) const
{
	return baseDomNode(pDoc);
}

GBlock* GBlock::deserialize(GDomNode* pNode)
{
	BlockType e = (BlockType)pNode->field("type")->asInt();
	switch(e)
	{
		case block_tanh: return new GBlockTanh(pNode);
		case block_linear: return new GBlockLinear(pNode);
		default: throw Ex("Unrecognized neural network layer type: ", GClasses::to_str((int)e));
	}
}

void GBlock::setInPos(size_t n, GLayer* pPrevLayer)
{
	m_inPos = n;
	if(pPrevLayer)
	{
		input.setData(pPrevLayer->output, m_inPos, inputCount);
		inBlame.setData(pPrevLayer->outBlame, m_inPos, inputCount);
	}
}

void GBlock::computeBlame(const GVec& target)
{
	GAssert(target.size() == outBlame.size());
	outBlame.copy(target);
	outBlame -= output;
}

std::string GBlock::to_str() const
{
	std::ostringstream os;
	os << "[" << name() << ": ";
	os << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << ", Weights=" << GClasses::to_str(weightCount()) << "]";
	return os.str();
}










GBlockActivation::GBlockActivation(size_t size)
: GBlockWeightless(size, size)
{
}

GBlockActivation::GBlockActivation(GDomNode* pNode)
: GBlockWeightless(pNode)
{}

void GBlockActivation::forwardProp(const GVec& weights)
{
	for(size_t i = 0; i < inputCount; i++)
		output[i] = eval(input[i]);
}

void GBlockActivation::backProp(const GVec& weights)
{
	for(size_t i = 0; i < inputCount; i++)
		inBlame[i] += outBlame[i] * derivative(input[i], output[i]);
}

void GBlockActivation::inverseProp(const GVec& output, GVec& input)
{
	for(size_t i = 0; i < outputCount; i++)
		input[i] = inverse(output[i]);
}








void GBlockSoftMax::forwardProp(const GVec& weights)
{
	// Activate with logistic function
	double sum = 0.0;
	for(size_t i = 0; i < outputCount; i++)
	{
		output[i] = std::exp(input[i]);
		sum += output[i];
	}

	// Normalize
	double scalar = 1.0 / sum;
	for(size_t i = 0; i < outputCount; i++)
		output[i] *= scalar;
}

void GBlockSoftMax::computeBlame(const GVec& target)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		GAssert(target[i] >= 0.0 && target[i] <= 1.0);
		outBlame[i] = target[i] - output[i];
	}
}

void GBlockSoftMax::backProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
		inBlame[i] = outBlame[i];
}








GBlockSpectral::GBlockSpectral(double min_wavelength, double max_wavelength, size_t units, bool linear_spacing)
: GBlockWeightless(1, units)
{
	size_t pairs = units / 2;
	if(pairs * 2 != units)
		throw Ex("Expected an even number of units");
	m_freq_start = 1.0 / max_wavelength;
	double freq_end = 1.0 / min_wavelength;

	if(linear_spacing)
	{
		m_freq_scale = 1.0;
		m_freq_shift = (freq_end - m_freq_start) / (pairs - 1);
	}
	else
	{
		m_freq_scale = pow(freq_end / m_freq_start, 1.0 / (pairs - 1));
		m_freq_shift = 0.0;
	}
}

GBlockSpectral::GBlockSpectral(GDomNode* pNode)
: GBlockWeightless(pNode),
m_freq_start(pNode->field("start")->asDouble()),
m_freq_scale(pNode->field("scale")->asDouble()),
m_freq_shift(pNode->field("shift")->asDouble())
{
}

GDomNode* GBlockSpectral::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "start", pDoc->newDouble(m_freq_start));
	pNode->addField(pDoc, "scale", pDoc->newDouble(m_freq_scale));
	pNode->addField(pDoc, "shift", pDoc->newDouble(m_freq_shift));
	return pNode;
}

void GBlockSpectral::forwardProp(const GVec& weights)
{
	double freq = 2.0 * M_PI * m_freq_start;
	size_t pairs = outputCount / 2;
	for(size_t i = 0; i < pairs; i++)
	{
		output[2 * i] = std::sin(freq * input[0]);
		output[2 * i + 1] = std::cos(freq * input[0]);
		freq *= m_freq_scale;
		freq += m_freq_shift;
	}
}

void GBlockSpectral::backProp(const GVec& weights)
{
	double freq = 2.0 * M_PI * m_freq_start;
	size_t pairs = outputCount / 2;
	for(size_t i = 0; i < pairs; i++)
	{
		inBlame[0] += outBlame[2 * i] * std::cos(freq * input[0]);
		inBlame[0] -= outBlame[2 * i + 1] * std::sin(freq * input[0]);
	}
}










GBlockLinear::GBlockLinear(size_t inputs, size_t outputs)
: GBlock(inputs, outputs)
{
}

GBlockLinear::GBlockLinear(GDomNode* pNode)
: GBlock(pNode)
{}

void GBlockLinear::forwardProp(const GVec& weights)
{
	// Start with the bias
	output.copy(weights, 0, outputCount);

	// Do the weights
	size_t pos = outputCount;
	GAssert(output[outputCount - 1] > -1e100 && output[outputCount - 1] < 1e100);
	for(size_t i = 0; i < inputCount; i++)
	{
		output.addScaled(input[i], weights, pos, outputCount);
		pos += outputCount;
	}
	GAssert(output[outputCount - 1] > -1e100 && output[outputCount - 1] < 1e100);
}

void GBlockLinear::backProp(const GVec& weights)
{
	size_t pos = outputCount; // skip the bias weights
	for(size_t i = 0; i < inputCount; i++)
	{
		const GConstVecWrapper v(weights, pos, outputCount);
		inBlame[i] += outBlame.dotProduct(v);
		pos += outputCount;
	}
}

void GBlockLinear::updateGradient(GVec& weights, GVec& gradient)
{
	size_t pos = 0;
	for(size_t j = 0; j < outputCount; j++)
		gradient[pos++] += outBlame[j];
	for(size_t i = 0; i < inputCount; i++)
	{
		double act = input[i];
		for(size_t j = 0; j < outputCount; j++)
			gradient[pos++] += outBlame[j] * act;
	}
}

size_t GBlockLinear::weightCount() const
{
	return (inputCount + 1) * outputCount;
}

void GBlockLinear::initWeights(GRand& rand, GVec& weights)
{
	size_t wc = weightCount();
	double mag = std::max(0.03, 1.0 / std::max(1ul, inputCount));
	for(size_t i = 0; i < wc; i++)
		weights[i] = rand.normal() * mag;
}







GBlockTemperedLinear::GBlockTemperedLinear(size_t inputs, size_t outputs, double deviation_cap, double forget_rate)
: GBlock(inputs, outputs), deviationCap(deviation_cap), forgetRate(forget_rate), moment1(outputs), moment2(outputs)
{
	moment1.fill(0.0);
	moment2.fill(0.0);
}

GBlockTemperedLinear::GBlockTemperedLinear(const GBlockTemperedLinear& that)
: GBlock(that), deviationCap(that.deviationCap), forgetRate(that.forgetRate), moment1(outputCount), moment2(outputCount)
{
	moment1.copy(that.moment1);
	moment2.copy(that.moment2);
}

GBlockTemperedLinear::GBlockTemperedLinear(GDomNode* pNode)
: GBlock(pNode), deviationCap(pNode->field("dc")->asDouble()), forgetRate(pNode->field("fr")->asDouble()),
moment1(pNode->field("mom1")), moment2(pNode->field("mom2"))
{
}

void GBlockTemperedLinear::forwardProp(const GVec& weights)
{
	// Start with the bias
	output.copy(weights, 0, outputCount);

	// Do the weights
	size_t pos = outputCount;
	GAssert(output[outputCount - 1] > -1e100 && output[outputCount - 1] < 1e100);
	for(size_t i = 0; i < inputCount; i++)
	{
		output.addScaled(input[i], weights, pos, outputCount);
		pos += outputCount;
	}
	GAssert(output[outputCount - 1] > -1e100 && output[outputCount - 1] < 1e100);
}

void GBlockTemperedLinear::backProp(const GVec& weights)
{
	size_t pos = outputCount; // skip the bias weights
	for(size_t i = 0; i < inputCount; i++)
	{
		const GConstVecWrapper v(weights, pos, outputCount);
		inBlame[i] += outBlame.dotProduct(v);
		pos += outputCount;
	}
}

void GBlockTemperedLinear::updateGradient(GVec& weights, GVec& gradient)
{
	// Contain the output deviation
	for(size_t i = 0; i < outputCount; i++)
	{
		moment1[i] *= (1.0 - forgetRate);
		moment1[i] += (forgetRate * output[i]);
		moment2[i] *= (1.0 - forgetRate);
		moment2[i] += (forgetRate * output[i] * output[i]);
		double var = moment2[i] - (moment1[i] * moment1[i]);
		if(var > deviationCap * deviationCap)
		{
			double scalar = deviationCap / std::sqrt(var);
			for(size_t j = 0; j < inputCount; j++)
				weights[outputCount + j * outputCount] *= scalar;
		}
	}

	// Update the gradient
	size_t pos = 0;
	for(size_t j = 0; j < outputCount; j++)
		gradient[pos++] += outBlame[j];
	for(size_t i = 0; i < inputCount; i++)
	{
		double act = input[i];
		for(size_t j = 0; j < outputCount; j++)
			gradient[pos++] += outBlame[j] * act;
	}
}

size_t GBlockTemperedLinear::weightCount() const
{
	return (inputCount + 1) * outputCount;
}

void GBlockTemperedLinear::initWeights(GRand& rand, GVec& weights)
{
	size_t wc = weightCount();
	double mag = std::max(0.03, 1.0 / inputCount);
	for(size_t i = 0; i < wc; i++)
		weights[i] = rand.normal() * mag;
}







GBlockScalarSum::GBlockScalarSum(size_t outputs)
: GBlockWeightless(outputs * 2, outputs)
{
}

GBlockScalarSum::GBlockScalarSum(GDomNode* pNode)
: GBlockWeightless(pNode)
{
}

GBlockScalarSum::~GBlockScalarSum()
{
}

// virtual
void GBlockScalarSum::forwardProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
		output[i] = input[i] + input[outputCount + i];
}

void GBlockScalarSum::backProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		inBlame[i] += outBlame[i];
		inBlame[outputCount + i] += outBlame[i];
	}
}









GBlockScalarProduct::GBlockScalarProduct(size_t outputs)
: GBlockWeightless(outputs * 2, outputs)
{
}

GBlockScalarProduct::GBlockScalarProduct(GDomNode* pNode)
: GBlockWeightless(pNode)
{
}

GBlockScalarProduct::~GBlockScalarProduct()
{
}

// virtual
void GBlockScalarProduct::forwardProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
		output[i] = input[i] * input[outputCount + i];
}

void GBlockScalarProduct::backProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		inBlame[i] += outBlame[i] * input[outputCount + i];
		inBlame[outputCount + i] += outBlame[i] * input[i];
	}
}








GBlockSwitch::GBlockSwitch(size_t outputs)
: GBlockWeightless(outputs * 3, outputs)
{
}

GBlockSwitch::GBlockSwitch(GDomNode* pNode)
: GBlockWeightless(pNode)
{
}

GBlockSwitch::~GBlockSwitch()
{
}

// virtual
void GBlockSwitch::forwardProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
		output[i] = input[i] * input[outputCount + i] + (1.0 - input[i]) * input[outputCount + outputCount + i];
}

void GBlockSwitch::backProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		inBlame[i] += (input[outputCount + i] - input[outputCount + outputCount + i]) * outBlame[i];
		inBlame[outputCount + i] += input[i] * outBlame[i];
		inBlame[outputCount + outputCount + i] += (1.0 - input[i]) * outBlame[i];
	}
}







GBlockHinge::GBlockHinge(size_t size)
: GBlock(size, size)
{
}

GBlockHinge::GBlockHinge(GDomNode* pNode)
: GBlock(pNode)
{}

void GBlockHinge::forwardProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double alpha = std::max(-1.0, std::min(1.0, weights[i]));
		double beta = std::max(0.0, weights[outputCount + i]);
		output[i] = alpha * (std::sqrt(input[i] * input[i] + beta * beta) - beta) + input[i];
	}
}

void GBlockHinge::backProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double alpha = weights[i];
		double beta = weights[outputCount + i];
		inBlame[i] += outBlame[i] * (alpha * input[i] / std::sqrt(input[i] * input[i] + beta * beta) + 1.0);
	}
}

void GBlockHinge::updateGradient(GVec& weights, GVec& gradient)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = std::max(-1.0, std::min(1.0, weights[i]));
		double beta = weights[outputCount + i];
		gradient[i] += outBlame[i] * (std::sqrt(input[i] * input[i] + beta * beta) - beta);
	}
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[outputCount + i] = std::max(0.0, weights[outputCount + i]);
		double alpha = weights[i];
		double beta = weights[outputCount + i];
		gradient[outputCount + i] += outBlame[i] * alpha * (beta / std::sqrt(input[i] * input[i] + beta * beta) - 1.0);
	}
}

size_t GBlockHinge::weightCount() const
{
	return outputCount * 2;
}

void GBlockHinge::initWeights(GRand& rand, GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = 0.0;
		weights[outputCount + i] = 0.1;
	}
}












GBlockSoftExp::GBlockSoftExp(size_t size, double beta)
: GBlock(size, size), m_beta(beta)
{
}

GBlockSoftExp::GBlockSoftExp(GDomNode* pNode)
: GBlock(pNode), m_beta(pNode->field("beta")->asDouble())
{}

GDomNode* GBlockSoftExp::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "beta", pDoc->newDouble(m_beta));
	return pNode;
}

void GBlockSoftExp::forwardProp(const GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double alpha = std::max(-1.0, std::min(1.0, weights[i]));
		if(alpha > 1.0e-7)
			output[i] = (exp(std::min(4.6, alpha * input[i]) - 1.0)) / alpha + alpha * m_beta;
		else if(alpha < -1.0e-7)
			output[i] = -log(std::max(1e-43, alpha * (-m_beta * alpha - input[i]) + 1.0)) / alpha;
		else
			output[i] = input[i];
	}
}

void GBlockSoftExp::backProp(const GVec& weights)
{
	for(size_t i = 0; i < inBlame.size(); i++)
	{
		double alpha = weights[i];
		if(alpha >= 0.0)
			inBlame[i] += (outBlame[i] * exp(std::min(4.6, alpha * input[i])));
		else
			inBlame[i] += (outBlame[i] * 1.0 / std::max(0.01, (1.0 - alpha * (alpha * m_beta + input[i]))));
	}
}

void GBlockSoftExp::updateGradient(GVec& weights, GVec& gradient)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = std::max(-1.0, std::min(1.0, weights[i]));
		double alpha = weights[i];
		if(alpha > 1.0e-6)
			gradient[i] += outBlame[i] * ((alpha * alpha * m_beta + (alpha * input[i] - 1.0) * exp(alpha * input[i]) + 1.0) / (alpha * alpha));
		else if(alpha < -1.0e-6)
			gradient[i] += outBlame[i] * (log(alpha * alpha * (-m_beta) - alpha * input[i] + 1.0) / (alpha * alpha) + (2.0 * alpha * m_beta + input[i]) / (alpha * (alpha * alpha * (-m_beta) - alpha * input[i] + 1.0)));
		else
			gradient[i] += outBlame[i] * (input[i] * input[i] / 2.0 + m_beta);
	}
}

size_t GBlockSoftExp::weightCount() const
{
	return outputCount;
}

void GBlockSoftExp::initWeights(GRand& rand, GVec& weights)
{
	weights.fill(0.0);
}















GBlockPAL::GBlockPAL(size_t inputs, size_t outputs, GRand& rand)
: GBlock(inputs, outputs), m_probs(outputs), m_rand(rand)
{
}

GBlockPAL::GBlockPAL(GDomNode* pNode, GRand& rand)
: GBlock(pNode), m_probs(outputCount), m_rand(rand)
{
}

void GBlockPAL::forwardProp(const GVec& weights)
{
	// Compute probabilities of activating
	size_t pos = 0;
	output.copy(weights, pos, outputCount);
	pos += outputCount;
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);
	for(size_t i = 0; i < inputCount; i++)
	{
		if(input[i] != 0.0)
			output.addScaled(input[i], weights, pos, outputCount);
		pos += outputCount;
	}
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);

	// Compute actual activation values
	size_t biasStart = pos;
	size_t weightsStart = pos + outputCount;
	for(size_t i = 0; i < outputCount; i++)
	{
		double t = tanh(output[i]);
		m_probs[i] = t * t * 0.99 + 0.01;
		if(m_rand.uniform() < m_probs[i])
		{
			// Compute the activation value
			output[i] = weights[biasStart + i];
			for(size_t j = 0; j < input.size(); j++)
				output[i] += input[j] * weights[weightsStart + outputCount * j + i];
		}
		else
			output[i] = 0.0; // Nope, don't activate
	}
}

void GBlockPAL::backProp(const GVec& weights)
{
	size_t pos = outputCount + inputCount * outputCount + outputCount;
	for(size_t i = 0; i < inputCount; i++)
	{
		if(input[i] != 0.0)
		{
			for(size_t j = 0; j < outputCount; j++)
			{
				if(output[j] != 0.0)
					inBlame[i] += outBlame[j] * weights[pos++];
			}
		}
	}
}

void GBlockPAL::updateGradient(GVec& weights, GVec& gradient)
{
	size_t pos = 0;

	// Probability weights
	for(size_t j = 0; j < outputCount; j++)
		gradient[pos++] += outBlame[j] * output[j];
	for(size_t i = 0; i < inputCount; i++)
	{
		double act = input[i];
		if(act != 0.0)
		{
			for(size_t j = 0; j < outputCount; j++)
				gradient[pos++] += outBlame[j] * act * output[j];
		}
		else
			pos += outputCount;
	}

	// Value weights
	for(size_t j = 0; j < outputCount; j++)
		gradient[pos++] += outBlame[j] * m_probs[j];
	for(size_t i = 0; i < inputCount; i++)
	{
		double act = input[i];
		if(act != 0.0)
		{
			for(size_t j = 0; j < outputCount; j++)
				gradient[pos++] += outBlame[j] * act * m_probs[j];
		}
		else
			pos += outputCount;
	}
}


size_t GBlockPAL::weightCount() const
{
	return 2 * (inputCount + 1) * outputCount;
}

void GBlockPAL::initWeights(GRand& rand, GVec& weights)
{
	size_t wc = weightCount();
	double mag = std::max(0.03, 1.0 / inputCount);
	for(size_t i = 0; i < wc; i++)
		weights[i] = mag * rand.normal();
}












GBlockLSTM::GBlockLSTM(size_t inputs, size_t outputs)
: GBlock(inputs, outputs), n(inputs + 2),
f(outputs),
t(outputs),
o(outputs),
c(outputs),
h(outputs),
blame_h(outputs),
blame_c(outputs),
pPrevInstance(this),
pNextInstance(this),
pSpare(nullptr)
{
}

GBlockLSTM::GBlockLSTM(const GBlockLSTM& that)
: GBlock(that),
n(that.inputCount + 2),
f(that.outputCount),
t(that.outputCount),
o(that.outputCount),
c(that.outputCount),
h(that.outputCount),
blame_h(that.outputCount),
blame_c(that.outputCount),
pPrevInstance(this),
pNextInstance(this),
pSpare(nullptr)
{
}

GBlockLSTM::GBlockLSTM(GDomNode* pNode)
: GBlock(pNode), n(inputCount + 2),
f(outputCount),
t(outputCount),
o(outputCount),
c(outputCount),
h(outputCount),
blame_h(outputCount),
blame_c(outputCount),
pPrevInstance(this),
pNextInstance(this),
pSpare(nullptr)
{
}

GBlockLSTM::~GBlockLSTM()
{
	while(pSpare)
	{
		GBlockLSTM* pCondemned = pSpare;
		pSpare = pCondemned->pSpare;
		GAssert(pCondemned->pSpare == nullptr);
		pCondemned->pNextInstance = nullptr;
		delete(pCondemned);
	}
	while(pNextInstance != this)
	{
		GBlockLSTM* pCondemned = pNextInstance;
		pNextInstance = pCondemned->pNextInstance;
		GAssert(pCondemned->pSpare == nullptr);
		pCondemned->pNextInstance = nullptr;
		delete(pCondemned);
	}
}

void GBlockLSTM::forwardProp_instance(const GVec& weights)
{
	size_t pos = 0;

	// Compute f
	for(size_t i = 0; i < outputCount; i++)
	{
		n[inputCount] = h[i];
		n[inputCount + 1] = c[i];
		f[i] = weights[pos++];
		const GConstVecWrapper vw(weights, pos, inputCount + 2);
		pos += inputCount + 2;
		f[i] += n.dotProduct(vw);
		f[i] = 1.0 / (1.0 + exp(-f[i]));
	}

	// Compute t
	for(size_t i = 0; i < outputCount; i++)
	{
		n[inputCount] = h[i];
		n[inputCount + 1] = c[i];
		t[i] = weights[pos++];
		const GConstVecWrapper vw(weights, pos, inputCount + 2);
		pos += inputCount + 2;
		t[i] += n.dotProduct(vw);
		t[i] = tanh(t[i]);
	}

	// Compute o
	for(size_t i = 0; i < outputCount; i++)
	{
		n[inputCount] = h[i];
		n[inputCount + 1] = c[i];
		o[i] = weights[pos++];
		const GConstVecWrapper vw(weights, pos, inputCount + 2);
		pos += inputCount + 2;
		o[i] += n.dotProduct(vw);
		o[i] = 1.0 / (1.0 + exp(-o[i]));
	}
}

void GBlockLSTM::forwardProp(const GVec& weights)
{
	for(GBlockLSTM* pInst = pNextInstance; pInst != this; pInst = pInst->pNextInstance)
	{
		pInst->forwardProp_instance(weights);
		pInst->stepInTime();
	}
	n.copy(0, input);
	forwardProp_instance(weights);

	// Compute output
	for(size_t i = 0; i < outputCount; i++)
		output[i] = tanh(f[i] * c[i] + (1.0 - f[i]) * t[i]) * o[i];
}

void GBlockLSTM::resetState()
{
	// Move the ring of instances into the spare chain
	while(pNextInstance != this)
	{
		GBlockLSTM* pCondemned = pNextInstance;
		pNextInstance = pCondemned->pNextInstance;
		pCondemned->pNextInstance->pPrevInstance = this;
		pCondemned->pSpare = pSpare;
		pSpare = pCondemned;
	}

	// Reset the state in this instance
	c.fill(0.0);
	h.fill(0.0);
}

void GBlockLSTM::stepInTime()
{
	pNextInstance->h.copy(output);
	for(size_t i = 0; i < outputCount; i++)
		pNextInstance->c[i] = f[i] * c[i] + (1.0 - f[i]) * t[i];
}

GBlock* GBlockLSTM::advanceState(size_t unfoldedInstances)
{
	// Count the instances
	size_t count = 0;
	for(GBlockLSTM* pInst = pNextInstance; true; pInst = pInst->pNextInstance)
	{
		count++;
		if(pInst == this)
			break;
	}

	// Destroy superfluous instances
	GAssert(unfoldedInstances > 0);
	while(count > unfoldedInstances)
	{
		GBlockLSTM* pCondemned = pNextInstance;
		pNextInstance = pCondemned->pNextInstance;
		pCondemned->pNextInstance->pPrevInstance = this;
		pCondemned->pNextInstance = nullptr;
		pCondemned->pPrevInstance = nullptr;
		delete(pCondemned);
		count--;
	}

	// Pull in spare instances if necessary
	while(count < unfoldedInstances)
	{
		// find a spare instance
		if(!pSpare)
			pSpare = clone();
		pSpare = pSpare->pNextInstance;

		// Add to the main ring
		pSpare->pNextInstance = pNextInstance;
		pSpare->pPrevInstance = this;
		pSpare->pNextInstance->pPrevInstance = pSpare;
		pNextInstance = pSpare;
		count++;
	}

	// Pass the spares forward
	if(pNextInstance != this)
	{
		GAssert(!pNextInstance->pSpare);
		pNextInstance->pSpare = pSpare;
		pSpare = nullptr;
	}

	// Step in time
	stepInTime();

	return pNextInstance;
}

void GBlockLSTM::backProp_instance(const GVec& weights, bool current)
{
	size_t pos = 0;

	// Blame f
	for(size_t i = 0; i < outputCount; i++)
	{
		double dd = tanh(f[i] * c[i] + (1.0 - f[i]) * t[i]);
		double b = blame_h[i] * o[i];
		b *= (1.0 - (dd * dd)); // derivative of tanh
		b += blame_c[i];
		b *= c[i];
		b *= f[i] * (1.0 - f[i]); // derivative of logistic
		pos++; // skip the bias weight
		if(current)
		{
			for(size_t j = 0; j < inputCount; j++)
				inBlame[j] += b * weights[pos++];
		}
		else
			pos += inputCount;
		pPrevInstance->blame_h[i] += b * weights[pos++];
		pPrevInstance->blame_c[i] += b * weights[pos++];
	}

	// Blame t
	for(size_t i = 0; i < outputCount; i++)
	{
		double dd = tanh(f[i] * c[i] + (1.0 - f[i]) * t[i]);
		double b = blame_h[i] * o[i];
		b *= (1.0 - (dd * dd)); // derivative of tanh
		b += blame_c[i];
		b *= (1.0 - f[i]);
		b *= (1.0 - t[i] * t[i]); // derivative of tanh
		pos++; // skip the bias weight
		if(current)
		{
			for(size_t j = 0; j < inputCount; j++)
				inBlame[j] += b * weights[pos++];
		}
		else
			pos += inputCount;
		pPrevInstance->blame_h[i] += b * weights[pos++];
		pPrevInstance->blame_c[i] += b * weights[pos++];
	}

	// Blame o
	for(size_t i = 0; i < outputCount; i++)
	{
		double b = blame_h[i];
		b *= tanh(f[i] * c[i] + (1.0 - f[i]) * t[i]);
		b *= o[i] * (1.0 - o[i]); // derivative of logistic
		pos++; // skip the bias weight
		if(current)
		{
			for(size_t j = 0; j < inputCount; j++)
				inBlame[j] += b * weights[pos++];
		}
		else
			pos += inputCount;
		pPrevInstance->blame_h[i] += b * weights[pos++];
		pPrevInstance->blame_c[i] += b * weights[pos++];
	}
}

void GBlockLSTM::backProp(const GVec& weights)
{
	for(GBlockLSTM* pInst = pPrevInstance; pInst != this; pInst = pInst->pPrevInstance)
	{
		pInst->blame_c.fill(0.0);
		pInst->blame_h.fill(0.0);
	}
	blame_c.fill(0.0);
	blame_h.copy(outBlame);
	if(pPrevInstance != this)
		backProp_instance(weights, true);
	for(GBlockLSTM* pInst = pPrevInstance; pInst != this && pInst->pPrevInstance != this; pInst = pInst->pPrevInstance)
		pInst->backProp_instance(weights, false);
}

void GBlockLSTM::updateGradient_instance(GVec& weights, GVec& gradient)
{
	size_t pos = 0;

	// Update f
	for(size_t i = 0; i < outputCount; i++)
	{
		double dd = tanh(f[i] * c[i] + (1.0 - f[i]) * t[i]);
		double b = blame_h[i] * o[i];
		b *= (1.0 - (dd * dd)); // derivative of tanh
		b += blame_c[i];
		b *= c[i];
		b *= f[i] * (1.0 - f[i]); // derivative of logistic
		gradient[pos++] += b;
		for(size_t j = 0; j < inputCount; j++)
			gradient[pos++] += b * n[j];
		gradient[pos++] += b * h[i];
		gradient[pos++] += b * c[i];
	}

	// Update t
	for(size_t i = 0; i < outputCount; i++)
	{
		double dd = tanh(f[i] * c[i] + (1.0 - f[i]) * t[i]);
		double b = blame_h[i] * o[i];
		b *= (1.0 - (dd * dd)); // derivative of tanh
		b += blame_c[i];
		b *= (1.0 - f[i]);
		b *= (1.0 - t[i] * t[i]); // derivative of tanh
		gradient[pos++] += b;
		for(size_t j = 0; j < inputCount; j++)
			gradient[pos++] += b * n[j];
		gradient[pos++] += b * h[i];
		gradient[pos++] += b * c[i];
	}

	// Update o
	for(size_t i = 0; i < outputCount; i++)
	{
		double b = blame_h[i];
		b *= tanh(f[i] * c[i] + (1.0 - f[i]) * t[i]);
		b *= o[i] * (1.0 - o[i]); // derivative of logistic
		gradient[pos++] += b;
		for(size_t j = 0; j < inputCount; j++)
			gradient[pos++] += b * n[j];
		gradient[pos++] += b * h[i];
		gradient[pos++] += b * c[i];
	}
}

void GBlockLSTM::updateGradient(GVec& weights, GVec& gradient)
{
	for(GBlockLSTM* pInst = pNextInstance; pInst != this; pInst = pInst->pNextInstance)
		pInst->updateGradient_instance(weights, gradient);
	updateGradient_instance(weights, gradient);
}

size_t GBlockLSTM::weightCount() const
{
	return 3 * outputCount * (1 + inputCount + 2);
}

void GBlockLSTM::initWeights(GRand& rand, GVec& weights)
{
	size_t wc = weightCount();
	double mag = std::max(0.03, 1.0 / (inputCount + 3));
	for(size_t i = 0; i < wc; i++)
		weights[i] = rand.normal() * mag;
}













GLayer::GLayer()
: input_count(0), output_count(0), weight_count(0)
{
}

GLayer::GLayer(const GLayer& that, GLayer* pPrevLayer)
: input_count(that.input_count), output_count(that.output_count), weight_count(that.weight_count), output(that.output.size()), outBlame(that.outBlame.size())
{
	size_t pos = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		GBlock* pB = that.m_blocks[i]->clone();
		if(pPrevLayer)
		{
			pB->input.setData(pPrevLayer->output, pB->inPos(), pB->inputs());
			pB->inBlame.setData(pPrevLayer->outBlame, pB->inPos(), pB->inputs());
		}
		pB->output.setData(output, pos, pB->outputs());
		pB->outBlame.setData(outBlame, pos, pB->outputs());
		pos += pB->outputs();
		m_blocks.push_back(pB);
	}
}

GLayer::GLayer(GDomNode* pNode)
: input_count(pNode->field("inputs")->asInt()),
output_count(pNode->field("outputs")->asInt()),
weight_count(0)
{
	GDomNode* pBlocks = pNode->field("blocks");
	GDomListIterator it(pBlocks);
	while(it.remaining() > 0)
	{
		m_blocks.push_back(GBlock::deserialize(it.current()));
		it.advance();
	}
}

// virtual
GLayer::~GLayer()
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		delete(m_blocks[i]);
}

GDomNode* GLayer::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "inputs", pDoc->newInt(input_count));
	pNode->addField(pDoc, "outputs", pDoc->newInt(output_count));
	GDomNode* pBlocks = pNode->addField(pDoc, "blocks", pDoc->newList());
	for(size_t i = 0; i < m_blocks.size(); i++)
		pBlocks->addItem(pDoc, m_blocks[i]->serialize(pDoc));
	return pNode;
}

void GLayer::add(GBlock* pBlock, GLayer* pPrevLayer, size_t inPos)
{
	pBlock->setInPos(inPos, pPrevLayer);
	m_blocks.push_back(pBlock);
	recount();
	output.resize(output_count);
	outBlame.resize(output_count);
	size_t pos = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		GBlock& b = *m_blocks[i];
		if(pPrevLayer)
		{
			b.input.setData(pPrevLayer->output, b.inPos(), b.inputs());
			b.inBlame.setData(pPrevLayer->outBlame, b.inPos(), b.inputs());
		}
		else
		{
			b.input.setData(nullptr, 0);
			b.inBlame.setData(nullptr, 0);
		}
		b.output.setData(output, pos, b.outputs());
		b.outBlame.setData(outBlame, pos, b.outputs());
		pos += b.outputs();
	}
}

void GLayer::recount()
{
	// Count the inputs and outputs
	input_count = 0;
	output_count = 0;
	weight_count = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		size_t inPos = m_blocks[i]->inPos();
		size_t inSize = m_blocks[i]->inputs();
		size_t outSize = m_blocks[i]->outputs();
		if(outSize == 0)
			throw Ex("Empty block");
		input_count = std::max(input_count, inPos + inSize);
		output_count += outSize;
		weight_count += m_blocks[i]->weightCount();
	}
}

size_t GLayer::inputs() const
{
	if(input_count == 0)
		((GLayer*)this)->recount();
	return input_count;
}

size_t GLayer::outputs() const
{
	if(output_count == 0)
		((GLayer*)this)->recount();
	return output_count;
}

size_t GLayer::weightCount() const
{
	if(weight_count == 0)
		((GLayer*)this)->recount();
	return weight_count;
}

void GLayer::initWeights(GRand& rand, GVec& weights)
{
	if(weights.size() != weightCount())
		throw Ex("Mismatching number of weights. Got ", to_str(weights.size()), ". Expected ", to_str(weightCount()));
	size_t pos = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		size_t n = m_blocks[i]->weightCount();
		GVecWrapper w(weights, pos, n);
		m_blocks[i]->initWeights(rand, w);
		pos += n;
	}
}

void GLayer::setInput(const GVec& in)
{
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		b.input.setData(in, b.inPos(), b.inputs());
	}
}

void GLayer::setInBlame(GVec& inBlame)
{
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		b.inBlame.setData(inBlame, b.inPos(), b.inputs());
	}
}

void GLayer::forwardProp(const GVec& weights)
{
	size_t w = 0;
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		size_t wc = b.weightCount();
		const GConstVecWrapper vw(weights, w, wc);
		w += wc;
		b.forwardProp(vw);
	}
	GAssert(w == weights.size());
}

void GLayer::backProp(const GVec& weights)
{
	size_t pos = weights.size();
	for(size_t i = blockCount() - 1; i < blockCount(); i--)
	{
		GBlock& b = block(i);
		size_t wc = b.weightCount();
		pos -= wc;
		const GConstVecWrapper vw(weights, pos, wc);
		b.backProp(vw);
	}
	GAssert(pos == 0);
}

void GLayer::updateGradient(GVec& weights, GVec& gradient)
{
	size_t pos = 0;
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		size_t wc = b.weightCount();
		GVecWrapper w(weights, pos, wc);
		GVecWrapper g(gradient, pos, wc);
		pos += wc;
		b.updateGradient(w, g);
	}
	GAssert(pos == weights.size());
}

void GLayer::resetState()
{
	for(size_t i = 0; i < blockCount(); i++)
		block(i).resetState();
}

void GLayer::advanceState(size_t unfoldedInstances)
{
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock* pOld = m_blocks[i];
		GBlock* pNew = pOld->advanceState(unfoldedInstances);
		if(pNew != pOld)
		{
			GAssert(pNew->input.data() == pOld->input.data());
			GAssert(pNew->output.data() == pOld->output.data());
			m_blocks[i] = pNew;
		}
	}
}

void GLayer::computeBlame(const GVec& target)
{
	GAssert(target.size() == outBlame.size());
	size_t pos = 0;
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		size_t n = b.outputs();
		const GConstVecWrapper t(target, pos, n);
		pos += n;
		b.computeBlame(t);
	}
}













GNeuralNet::GNeuralNet()
: GBlock(0, 0), m_weightCount(0), m_fullBackProp(true)
{
}

GNeuralNet::GNeuralNet(const GNeuralNet& that)
: GBlock(that), m_weightCount(that.m_weightCount), m_fullBackProp(true)
{
	for(size_t i = 0; i < that.m_layers.size(); i++)
	{
		m_layers.push_back(new GLayer(*that.m_layers[i], i > 0 ? m_layers[i - 1] : nullptr));
	}
}

GNeuralNet::GNeuralNet(GDomNode* pNode)
: GBlock(pNode), m_weightCount(0), m_fullBackProp(true)
{
	GDomNode* pLayers = pNode->field("layers");
	GDomListIterator it(pLayers);
	while(it.remaining() > 0)
	{
		m_layers.push_back(new GLayer(it.current()));
		it.advance();
	}
}

// virtual
GNeuralNet::~GNeuralNet()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
}

GDomNode* GNeuralNet::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	GDomNode* pLayers = pNode->addField(pDoc, "layers", pDoc->newList());
	for(size_t i = 0; i < m_layers.size(); i++)
		pLayers->addItem(pDoc, m_layers[i]->serialize(pDoc));
	return pNode;
}

std::string GNeuralNet::to_str(const std::string& line_prefix) const
{
	std::ostringstream oss;
	oss << line_prefix << "[GNeuralNet: " << inputs() << "->" << outputs() << ", Weights=" << GClasses::to_str(weightCount()) << "\n";
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		oss << line_prefix << "  " << pre_pad(2, ' ', GClasses::to_str(i)) << ") ";
		for(size_t j = 0; j < m_layers[i]->blockCount(); j++)
		{
			GBlock& b = m_layers[i]->block(j);
			if(b.type() == block_neuralnet)
				oss << "[GNeuralNet (contents not shown):" << b.inputs() << "->" << b.outputs() << ", Weights=" << GClasses::to_str(b.weightCount()) << "\n";
			else
				oss << b.to_str();
		}
		oss << "\n";
	}
	oss << line_prefix << "]";
	return oss.str();
}

std::string GNeuralNet::to_str() const
{
	return to_str("");
}

void GNeuralNet::add(GBlock* pBlock)
{
	GAssert(m_weightCount == 0, "weights were counted before all blocks were added");
	GLayer* pPrevLayer = nullptr;
	if(m_layers.size() > 0)
		pPrevLayer = m_layers[m_layers.size() - 1];
	GLayer* pNewLayer = new GLayer();
	m_layers.push_back(pNewLayer);
	pNewLayer->add(pBlock, pPrevLayer, 0);
	if(m_layers.size() == 1)
		inputCount = pNewLayer->inputs();
	outputCount = pNewLayer->outputs();
	outBlame.setData(outputLayer().outBlame);
}

void GNeuralNet::concat(GBlock* pBlock, size_t inPos)
{
	GAssert(m_weightCount == 0, "weights were counted before all blocks were added");
	GLayer* pPrevLayer = nullptr;
	if(m_layers.size() > 1)
		pPrevLayer = m_layers[m_layers.size() - 2];
	GLayer* pLastLayer = m_layers[m_layers.size() - 1];
	pLastLayer->add(pBlock, pPrevLayer, inPos);
	if(m_layers.size() == 1)
		inputCount = pLastLayer->inputs();
	outputCount = pLastLayer->outputs();
	outBlame.setData(outputLayer().outBlame);
}

void GNeuralNet::forwardProp(const GVec& weights)
{
	if(input.size() != m_layers[0]->inputs())
		throw Ex("Expected ", GClasses::to_str(m_layers[0]->inputs()), " input values. Got ", GClasses::to_str(input.size()));
	m_layers[0]->setInput(input);
	output.setData(outputLayer().output);
	size_t pos = 0;
	GConstVecWrapper vw;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = *m_layers[i];
		size_t wc = lay.weightCount();
		vw.setData(weights, pos, wc);
		pos += wc;
		lay.forwardProp(vw);
	}
	GAssert(pos == weights.size());
}

GVec& GNeuralNet::forwardProp(const GVec& weights, const GVec& in)
{
	input.setData(in);
	forwardProp(weights);
	return output;
}

void GNeuralNet::computeBlame(const GVec& target)
{
	outputLayer().computeBlame(target);
}

void GNeuralNet::backProp(const GVec& weights)
{
	if(m_fullBackProp)
		m_layers[0]->setInBlame(inBlame);
	size_t pos = weights.size();
	GConstVecWrapper vw;
	for(size_t i = m_layers.size() - 1; i > 0; i--)
	{
		GLayer& layPrev = *m_layers[i - 1];
		layPrev.outBlame.fill(0.0);
		GLayer& lay = *m_layers[i];
		size_t wc = lay.weightCount();
		pos -= wc;
		vw.setData(weights, pos, wc);
		lay.backProp(vw);
	}
	GAssert(pos == m_layers[0]->weightCount());
	if(m_fullBackProp)
	{
		GLayer& lay = *m_layers[0];
		vw.setData(weights, 0, pos);
		lay.backProp(vw);
	}
}

void GNeuralNet::backpropagate(const GVec& weights, GVec* inputBlame)
{
	if(inputBlame)
	{
		GAssert(inputBlame->size() == m_layers[0]->inputs());
		inputBlame->fill(0.0);
		inBlame.setData(*inputBlame);
	}
	else
		m_fullBackProp = false;
	backProp(weights);
	m_fullBackProp = true;
}

void GNeuralNet::updateGradient(GVec& weights, GVec& gradient)
{
	GAssert(weights.size() == gradient.size());
	size_t pos = 0;
	GVecWrapper w;
	GVecWrapper g;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = *m_layers[i];
		size_t wc = lay.weightCount();
		w.setData(weights, pos, wc);
		g.setData(gradient, pos, wc);
		pos += wc;
		lay.updateGradient(w, g);
	}
	GAssert(pos == weights.size());
}

void GNeuralNet::recount()
{
	m_weightCount = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
		m_weightCount += m_layers[i]->weightCount();
}

size_t GNeuralNet::weightCount() const
{
	if(m_weightCount == 0)
		((GNeuralNet*)this)->recount();
	return m_weightCount;
}

void GNeuralNet::copyStructure(const GNeuralNet* pOther)
{
	m_weightCount = pOther->weightCount();
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
	m_layers.clear();
	for(size_t i = 0; i < pOther->m_layers.size(); i++)
	{
		// todo: this is not a very efficient way to copy a layer
		GDom doc;
		GDomNode* pNode = pOther->m_layers[i]->serialize(&doc);
		m_layers.push_back(new GLayer(pNode));
	}
}

void GNeuralNet::initWeights(GRand& rand, GVec& weights)
{
	size_t pos = 0;
	GVecWrapper vw;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = *m_layers[i];
		size_t wc = lay.weightCount();
		vw.setData(weights, pos, wc);
		pos += wc;
		lay.initWeights(rand, vw);
	}
	GAssert(pos == weights.size());
}

double GNeuralNet::measureLoss(const GVec& weights, const GMatrix& features, const GMatrix& labels, double* pOutSAE)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	if(features.cols() != inputs())
		throw Ex("Mismatching number of inputs. Data has ", GClasses::to_str(features.cols()), ". Neural net expects ", GClasses::to_str(inputs()));
	GRand rand(0);
	double sae = 0.0;
	double sse = 0.0;
	if(labels.cols() == outputs())
	{
		// Regression. Compute SSE and SAE.
		for(size_t i = 0; i < features.rows(); i++)
		{
			GVec& prediction = forwardProp(weights, features[i]);
			const GVec& targ = labels[i];
			for(size_t j = 0; j < prediction.size(); j++)
			{
				if(targ[j] != UNKNOWN_REAL_VALUE)
				{
					double d = targ[j] - prediction[j];
					sse += (d * d);
					sae += std::abs(d);
				}
			}
		}
	}
	else if(labels.cols() == 1 && labels.relation().valueCount(0) == outputs())
	{
		// Classification. Count misclassifications.
		for(size_t i = 0; i < features.rows(); i++)
		{
			GVec& prediction = forwardProp(weights, features[i]);
			const GVec& targ = labels[i];
			if(targ[0] >= 0.0)
			{
				size_t j = prediction.indexOfMax();
				if(j != (size_t)targ[0])
					sse++;
			}
		}
		sae = sse;
	}
	else
	{
		if(labels.cols() == 1)
			throw Ex("Mismatching number of outputs. Data has 1 column with ", GClasses::to_str(labels.relation().valueCount(0)), " categorical values. Neural net outputs ", GClasses::to_str(outputs()));
		else
			throw Ex("Mismatching number of outputs. Data has ", GClasses::to_str(labels.cols()), ". Neural net expects ", GClasses::to_str(outputs()));
	}

	if(pOutSAE)
		*pOutSAE = sae;
	return sse;
}

void GNeuralNet::resetState()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->resetState();
}

GBlock* GNeuralNet::advanceState(size_t unfoldedInstances)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->advanceState(unfoldedInstances);
	return this;
}









/*
#ifdef GCUDA
void GLayer::uploadCuda()
{
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		GBlock& b = *m_blocks[i];
		b.uploadCuda();
	}
}

void GLayer::downloadCuda()
{
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		GBlock& b = *m_blocks[i];
		b.downloadCuda();
	}
}

void GLayer::stepCuda(GContextLayer& ctx, double learningRate, const GCudaVector& gradient)
{
	size_t gradPos = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		GBlock& b = *m_blocks[i];
		size_t wc = b.weightCount();
		GCudaVector grad(*(GCudaVector*)&gradient, gradPos, wc);
		b.stepCuda(ctx, learningRate, grad);
		gradPos += wc;
	}
}

void GContextLayer::forwardPropCuda(const GCudaVector& input, GCudaVector& output)
{
	size_t outPos = 0;
	size_t recurrents = 0;
	size_t comp = 0;
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		const GBlock& b = m_layer.block(i);
		GCudaVector in(*(GCudaVector*)&input, b.inPos(), b.inputs());
		GCudaVector out(output, outPos, b.outputs());
		if(b.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			b.forwardPropCuda(*pCompContext, in, out);
		}
		else if(b.isRecurrent())
		{
			GContextRecurrent* pRecContext = m_recurrents[recurrents++];
			pRecContext->forwardPropCuda(in, out);
		}
		else
			b.forwardPropCuda(*this, in, out);
		outPos += b.outputs();
	}
}

void GContextLayer::forwardProp_trainingCuda(const GCudaVector& input, GCudaVector& output)
{
	GConstVecWrapper vwInput;
	GVecWrapper vwOutput;
	size_t outPos = 0;
	size_t recurrents = 0;
	size_t comp = 0;
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		const GBlock& b = m_layer.block(i);
		GCudaVector in(*(GCudaVector*)&input, b.inPos(), b.inputs());
		GCudaVector out(output, outPos, b.outputs());
		if(b.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			b.forwardPropCuda(*pCompContext, in, out);
		}
		else if(b.isRecurrent())
		{
			GContextRecurrent* pRecContext = m_recurrents[recurrents++];
			pRecContext->forwardPropThroughTimeCuda(in, out);
		}
		else
			b.forwardPropCuda(*this, in, out);
		outPos += b.outputs();
	}
}

void GContextLayer::backPropCuda(GContextLayer& ctx, const GCudaVector& input, const GCudaVector& output, const GCudaVector& outBlame, GCudaVector& inBlame)
{
	GAssert(output.size() == outBlame.size());
	size_t outPos = 0;
	size_t recurrents = 0;
	size_t comp = 0;
	inBlame.fill(cudaEngine(), 0.0);
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
			const GBlock& b = m_layer.block(i);
			GCudaVector in(*(GCudaVector*)&input, b.inPos(), b.inputs());
			GCudaVector out(*(GCudaVector*)&output, outPos, b.outputs());
			GCudaVector outBl(*(GCudaVector*)&outBlame, outPos, b.outputs());
			GCudaVector inBl(inBlame, b.inPos(), b.inputs());
			if(b.type() == GBlock::block_neuralnet)
			{
					GContextNeuralNet* pCompContext = m_components[comp++];
					b.backPropCuda(*pCompContext, in, out, outBl, inBl);
			}
			else if(b.isRecurrent())
			{
					GContextRecurrent* pRecContext = m_recurrents[recurrents++];
					pRecContext->backPropThroughTimeCuda(in, out, outBl, inBl);
			}
			else
					b.backPropCuda(*this, in, out, outBl, inBl);
			outPos += b.outputs();
	}
}

void GContextLayer::updateGradientCuda(const GCudaVector& input, const GCudaVector& outBlame, GCudaVector& gradient)
{
	size_t gradPos = 0;
	size_t outPos = 0;
	size_t recurrents = 0;
	size_t comp = 0;
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		const GBlock& b = m_layer.block(i);
		size_t wc = b.weightCount();
		GCudaVector in(*(GCudaVector*)&input, b.inPos(), b.inputs());
		GCudaVector outBl(*(GCudaVector*)&outBlame, outPos, b.outputs());
		GCudaVector grad(gradient, gradPos, wc);
		if(b.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			b.updateGradientCuda(*pCompContext, in, outBl, grad);
		}
		else if(b.isRecurrent())
		{
			GContextRecurrent* pRecContext = m_recurrents[recurrents++];
			pRecContext->updateGradientCuda(in, outBl, grad);
		}
		else
			b.updateGradientCuda(*this, in, outBl, grad);
		outPos += b.outputs();
		gradPos += wc;
	}
}

#endif // GCUDA

std::string GNeuralNet::to_str(const std::string& line_prefix) const
{
	std::ostringstream oss;
	oss << line_prefix << "[GNeuralNet: " << inputs() << "->" << outputs() << ", Weights=" << GClasses::to_str(weightCount()) << "\n";
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		oss << line_prefix << "  " << pre_pad(2, ' ', GClasses::to_str(i)) << ") ";
		for(size_t j = 0; j < m_layers[i]->blockCount(); j++)
		{
			GBlock& b = m_layers[i]->block(j);
			if(b.type() == block_neuralnet)
				oss << "[GNeuralNet (contents not shown):" << b.inputs() << "->" << b.outputs() << ", Weights=" << GClasses::to_str(b.weightCount()) << "\n";
			else
				oss << b.to_str();
		}
		oss << "\n";
	}
	oss << line_prefix << "]";
	return oss.str();
}

#ifdef GCUDA
void GNeuralNet::uploadCuda()
{
	for(size_t i = 0; i < layerCount(); i++)
	{
		GLayer& lay = layer(i);
		lay.uploadCuda();
	}
}

void GNeuralNet::downloadCuda()
{
	for(size_t i = 0; i < layerCount(); i++)
	{
		GLayer& lay = layer(i);
		lay.downloadCuda();
	}
}

void GNeuralNet::forwardPropCuda(GContext& ctx, const GCudaVector& input, GCudaVector& output) const
{
	GAssert(input.size() == layer(0).inputs());
	GAssert(output.size() == outputLayer().outputs());
	const GCudaVector* pInput = &input;
	GContextNeuralNet* pContext = (GContextNeuralNet*)&ctx;
	GAssert(output.d_vals == pContext->predictionCuda().d_vals);
	GAssert(pContext->layerCount() == layerCount());
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GContextLayer* pLayer = pContext->m_layers[i];
		pLayer->m_activationCuda.resize(pLayer->m_layer.outputs());
		pLayer->forwardPropCuda(*pInput, pLayer->m_activationCuda);
		pInput = &pLayer->m_activationCuda;
	}
}

void GNeuralNet::backPropCuda(GContext& ctx, const GCudaVector& input, const GCudaVector& output, const GCudaVector& outBlame, GCudaVector& inBlame) const
{
	const GCudaVector* pOutput = &output;
	const GCudaVector* pOutBlame = &outBlame;
	GContextNeuralNet* pContext = (GContextNeuralNet*)&ctx;
	for(size_t i = pContext->m_layers.size() - 1; i > 0; i--)
	{
		GContextLayer* pLayer = pContext->m_layers[i];
		GContextLayer* pPrevLayer = pContext->m_layers[i - 1];
		pPrevLayer->m_blameCuda.resize(pPrevLayer->m_blame.size());
		pLayer->backPropCuda(*pLayer, pPrevLayer->m_activationCuda, *pOutput, *pOutBlame, pPrevLayer->m_blameCuda);
		pOutput = &pPrevLayer->m_activationCuda;
		pOutBlame = &pPrevLayer->m_blameCuda;
	}
	if(inBlame.d_vals != outBlame.d_vals)
	{
		GContextLayer* pLayer = pContext->m_layers[0];
		pLayer->backPropCuda(*pLayer, input, pLayer->m_activationCuda, pLayer->m_blameCuda, inBlame);
	}
}

void GNeuralNet::updateGradientCuda(GContext& ctx, const GCudaVector &input, const GCudaVector& outBlame, GCudaVector& gradient) const
{
	const GCudaVector* pInput = &input;
	size_t gradPos = 0;
	GContextNeuralNet* pContext = (GContextNeuralNet*)&ctx;
	for(size_t i = 0; i < pContext->m_layers.size(); i++)
	{
		GContextLayer* pLayer = pContext->m_layers[i];
		size_t wc = pLayer->m_layer.weightCount();
		GCudaVector grad(gradient, gradPos, wc);
		GAssert(gradPos + wc <= gradient.size());
		pLayer->updateGradientCuda(*pInput, pLayer->m_blameCuda, grad);
		pInput = &pLayer->m_activationCuda;
		gradPos += wc;
	}
	GAssert(gradPos == weightCount());
}

void GNeuralNet::stepCuda(GContext& ctx, double learningRate, const GCudaVector& gradient)
{
	size_t gradPos = 0;
	GContextNeuralNet* pContext = (GContextNeuralNet*)&ctx;
	for(size_t i = 0; i < layerCount(); ++i)
	{
		GContextLayer* pLayer = pContext->m_layers[i];
		GLayer& lay = layer(i);
		size_t wc = lay.weightCount();
		GCudaVector grad(*(GCudaVector*)&gradient, gradPos, wc);
		lay.stepCuda(*pLayer, learningRate, grad);
		gradPos += wc;
	}
	GAssert(gradPos == weightCount());
}
#endif // GCUDA

void GNeuralNet::invertNode(size_t lay, size_t node)
{
	GLayer& l = layer(lay);
	if(l.blockCount() > 1)
		throw Ex("This method assumes there is only one block in the layer");
	GBlock& b = l.block(0);
	if(b.type() != GBlock::block_linear)
		throw Ex("I don't know how to invert nodes in this type of layer");
	GBlockLinear& blockThis = *(GBlockLinear*)&b;
#ifdef GCUDA
	if(blockThis.m_biasCuda.size() > 0)
		throw Ex("Sorry, not yet implemented for the GPU");
#endif
	GMatrix& w = blockThis.weights();
	for(size_t i = 0; i < w.rows(); i++)
		w[i][node] = -w[i][node];
	size_t ds = lay + 1;
	while(ds < m_layers.size() && m_layers[ds]->blockCount() == 1 && m_layers[ds]->block(0).elementWise())
		ds++;
	if(ds < m_layers.size())
	{
		if(m_layers[ds]->blockCount() != 1 || m_layers[ds]->block(0).type() != GBlock::block_linear)
			throw Ex("Expected the downstream layer to contain exactly one linear block");
		GBlockLinear& blockDownStream = *(GBlockLinear*)&m_layers[ds]->block(0);
		size_t downOuts = blockDownStream.outputs();
		GVec& ww = blockDownStream.weights()[node];
		for(size_t i = 0; i < downOuts; i++)
			ww[i] = -ww[i];
	}
}

void GNeuralNet::swapNodes(size_t lay, size_t a, size_t b)
{
	GLayer& l = layer(lay);
	if(l.blockCount() != 1)
		throw Ex("Expected only one block in this layer");
	if(l.block(0).type() != GBlock::block_linear)
		throw Ex("I don't know how to swap nodes in this type of layer");
	GBlockLinear& blockThis = *(GBlockLinear*)&l;
#ifdef GCUDA
	if(blockThis.m_biasCuda.size() > 0)
		throw Ex("Sorry, not yet implemented for the GPU");
#endif
	blockThis.weights().swapColumns(a, b);
	size_t ds = lay + 1;
	while(ds < m_layers.size() && m_layers[ds]->blockCount() == 1 && m_layers[ds]->block(0).elementWise())
		ds++;
	if(ds < m_layers.size())
	{
		if(m_layers[ds]->blockCount() != 1 || m_layers[ds]->block(0).type() != GBlock::block_linear)
			throw Ex("Expected the downstream layer to contain exactly one linear block");
		GBlockLinear& blockDownStream = *(GBlockLinear*)m_layers[ds];
		blockDownStream.weights().swapRows(a, b);
	}
}

void GNeuralNet::dropNode(size_t lay, size_t index)
{
	GLayer& l = layer(lay);
	if(l.blockCount() != 1)
		throw Ex("Expected only one block in this layer");
	if(l.block(0).type() != GBlock::block_linear)
		throw Ex("I don't know how to drop nodes in this type of layer");
	GBlockLinear& blockThis = *(GBlockLinear*)&l;
#ifdef GCUDA
	if(blockThis.m_biasCuda.size() > 0)
		throw Ex("Sorry, not yet implemented for the GPU");
#endif
	if(index != blockThis.outputs() - 1)
		swapNodes(lay, index, blockThis.outputs() - 1);
	blockThis.weights().deleteColumns(blockThis.outputs() - 1, 1);
	blockThis.bias().erase(blockThis.outputs() - 1, 1);
	size_t ds = lay + 1;
	while(ds < m_layers.size() && m_layers[ds]->blockCount() == 1 && m_layers[ds]->block(0).elementWise())
	{
		m_layers[ds]->block(0).dropUnit(index);
		ds++;
	}
	if(ds < m_layers.size())
	{
		if(m_layers[ds]->blockCount() != 1 || m_layers[ds]->block(0).type() != GBlock::block_linear)
			throw Ex("Expected the downstream layer to contain exactly one linear block");
		GBlockLinear& blockDownStream = *(GBlockLinear*)m_layers[ds];
		blockDownStream.weights().deleteRow(blockDownStream.weights().rows() - 1);
	}
}

void GNeuralNet::splitNode(size_t lay, size_t index, GRand& rand)
{
	GLayer& l = layer(lay);
	if(l.blockCount() != 1)
		throw Ex("Expected only one block in this layer");
	if(l.block(0).type() != GBlock::block_linear)
		throw Ex("I don't know how to split nodes in this type of layer");
	GBlockLinear& blockThis = *(GBlockLinear*)&l;
#ifdef GCUDA
	if(blockThis.m_biasCuda.size() > 0)
		throw Ex("Sorry, not yet implemented for the GPU");
#endif
	GMatrix& w = blockThis.weights();
	w.newColumns(1);
	w.copyBlock(w, 0, index, w.rows(), 1, 0, w.cols() - 1, false);
	GVec& b = blockThis.bias();
	b.resizePreserve(b.size() + 1);
	b[b.size() - 1] = b[index];
	size_t ds = lay + 1;
	while(ds < m_layers.size() && m_layers[ds]->blockCount() == 1 && m_layers[ds]->block(0).elementWise())
	{
		m_layers[ds]->block(0).cloneUnit(index);
		ds++;
	}
	if(ds < m_layers.size())
	{
		if(m_layers[ds]->blockCount() != 1 || m_layers[ds]->block(0).type() != GBlock::block_linear)
			throw Ex("Expected the downstream layer to contain exactly one linear block");
		GBlockLinear& blockDownStream = *(GBlockLinear*)m_layers[ds];
		GVec& a = blockDownStream.weights()[index];
		GVec& b = blockDownStream.weights().newRow();
		for(size_t i = 0; i < a.size(); i++)
		{
			if(rand.next(2) == 0)
			{
				b[i] = a[i];
				a[i] = 0.0;
			}
			else
				b[i] = 0.0;
		}
	}
}

#ifndef MIN_PREDICT
void GNeuralNet::align(const GNeuralNet& that)
{
	if(layerCount() != that.layerCount())
		throw Ex("mismatching number of layers");
	for(size_t i = 0; i + 1 < m_layers.size(); i++)
	{
		// Copy weights into matrices
		GLayer& lay = layer(i);
		if(lay.blockCount() != 1)
			throw Ex("Expected all layers to have exactly one block");
		GBlock& lThis = lay.block(0);
		if(that.m_layers[i]->blockCount() != 1 || lThis.type() != that.m_layers[i]->block(0).type())
			throw Ex("mismatching layer types");
		if(lThis.type() == GBlock::block_linear)
		{
			GBlockLinear& layerThisCur = *(GBlockLinear*)&lThis;
			GBlockLinear& layerThatCur = *(GBlockLinear*)&that.m_layers[i]->block(0);
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
					GMatrix& wThis = layerThisCur.weights();
					const GMatrix& wThat = layerThatCur.weights();
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
				for(size_t kk = 0; kk < inputs; kk++)
					dp += layerThisCur.weights()[kk][j] * layerThatCur.weights()[kk][j];
				dp += layerThisCur.bias()[j] * layerThatCur.bias()[j];
				if(dp < 0)
					invertNode(i, j); // invert it
			}
		}
		else if(lThis.weightCount() > 0)
			throw Ex("I don't know how to align this type of layer");
	}
}
#endif // MIN_PREDICT

GMatrix* GNeuralNet::compressFeatures(GMatrix& features)
{
	GBlock& llay = layer(0);
	if(llay.type() == GBlock::layer_linear)
	{
		GLayerLinear& lay = *(GLayerLinear*)&llay;
		if(lay.inputs() != features.cols())
			throw Ex("mismatching number of data columns and layer units");
		GPCA pca(lay.inputs());
		pca.train(features);
		GVec off(lay.inputs());
		pca.basis()->multiply(pca.centroid(), off);
		GMatrix* pInvTransform = pca.basis()->pseudoInverse();
		std::unique_ptr<GMatrix> hInvTransform(pInvTransform);
		lay.transformWeights(*pInvTransform, off);
		return pca.transformBatch(features);
	}
	else
		throw Ex("I don't know how to contain this type of layer");
}

// static
GNeuralNet* GNeuralNet::fourier(GMatrix& series, double period)
{
	// Pad until the number of rows in series is a power of 2
	GMatrix* pSeries = &series;
	std::unique_ptr<GMatrix> hSeries;
	if(!GBits::isPowerOfTwo((unsigned int)series.rows()))
	{
		pSeries = new GMatrix(series);
		hSeries.reset(pSeries);
		while(pSeries->rows() & (pSeries->rows() - 1)) // Pad until the number of rows is a power of 2
		{
			GVec& newRow = pSeries->newRow();
			newRow.copy(pSeries->row(0));
		}
		period *= ((double)pSeries->rows() / series.rows());
	}

	// Make a neural network that combines sine units in the same manner as the Fourier transform
	GNeuralNet* pNN = new GNeuralNet();
	GLayerLinear* pLayerFreqs = new GLayerLinear(1, pSeries->rows());
	GLayerSine* pLayerSin = new GLayerSine();
	GLayerLinear* pLayerIdent = new GLayerLinear(FLEXIBLE_SIZE, pSeries->cols());
	pNN->addLayers(pLayerFreqs, pLayerSin, pLayerIdent);
	GUniformRelation relIn(1);
	GUniformRelation relOut(pSeries->cols());
	pNN->beginIncrementalLearning(relIn, relOut);

	// Initialize the weights of the sine units to match the frequencies used by the Fourier transform.
	GMatrix& wSin = pLayerFreqs->weights();
	GVec& bSin = pLayerFreqs->bias();
	for(size_t i = 0; i < pSeries->rows() / 2; i++)
	{
		wSin[0][2 * i] = 2.0 * M_PI * (i + 1) / period;
		bSin[2 * i] = 0.5 * M_PI;
		wSin[0][2 * i + 1] = 2.0 * M_PI * (i + 1) / period;
		bSin[2 * i + 1] = M_PI;
	}

	// Initialize the output layer
	struct ComplexNumber* pFourier = new struct ComplexNumber[pSeries->rows()];
	std::unique_ptr<struct ComplexNumber[]> hIn(pFourier);
	GMatrix& wIdent = pLayerIdent->weights();
	GVec& bIdent = pLayerIdent->bias();
	for(size_t j = 0; j < pSeries->cols(); j++)
	{
		// Convert column j to the Fourier domain
		struct ComplexNumber* pF = pFourier;
		for(size_t i = 0; i < pSeries->rows(); i++)
		{
			pF->real = pSeries->row(i)[j];
			pF->imag = 0.0;
			pF++;
		}
		GFourier::fft(pSeries->rows(), pFourier, true);

		// Initialize the weights of the identity output units to combine the sine units with the weights
		// specified by the Fourier transform
		for(size_t i = 0; i < pSeries->rows() / 2; i++)
		{
			wIdent[2 * i][j] = pFourier[1 + i].real / (pSeries->rows() / 2);
			wIdent[2 * i + 1][j] = pFourier[1 + i].imag / (pSeries->rows() / 2);
		}
		bIdent[j] = pFourier[0].real / (pSeries->rows());

		// Compensate for the way the FFT doubles-up the values in the last complex element
		wIdent[pSeries->rows() - 2][j] *= 0.5;
		wIdent[pSeries->rows() - 1][j] *= 0.5;
	}

	return pNN;
}






GContextNeuralNet::GContextNeuralNet(GRand& rand, const GNeuralNet& nn)
: GContext(rand), m_nn(nn)
#ifdef GCUDA
, m_pEngine(nullptr)
#endif // GCUDA
{
	if(nn.layerCount() < 1)
		throw Ex("No layers have been added to this neural network");
	for(size_t i = 0; i < nn.layerCount(); i++)
	{
		const GLayer& lay = nn.layer(i);
		m_layers.push_back(lay.newContext(rand));
	}
	m_pOutputLayer = m_layers[m_layers.size() - 1];
}

#ifdef GCUDA
GContextNeuralNet::GContextNeuralNet(GRand& rand, const GNeuralNet& nn, GCudaEngine& engine)
: GContext(rand), m_nn(nn), m_pEngine(&engine)
{
	if(nn.layerCount() < 1)
		throw Ex("No layers have been added to this neural network");
	for(size_t i = 0; i < nn.layerCount(); i++)
	{
		const GLayer& lay = nn.layer(i);
		m_layers.push_back(lay.newContext(rand, engine));
	}
	m_pOutputLayer = m_layers[m_layers.size() - 1];
}

GCudaVector& GContextNeuralNet::forwardPropCuda(const GCudaVector& input)
{
	GCudaVector& output = predictionCuda();
	output.resize(m_nn.outputs());
	m_nn.forwardPropCuda(*this, input, output);
	return output;
}

GCudaVector& GContextNeuralNet::forwardProp_trainingCuda(const GCudaVector& input)
{
	GAssert(input.size() == m_nn.layer(0).inputs());
	const GCudaVector* pInput = &input;
	GAssert(layerCount() == m_nn.layerCount());
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GContextLayer* pLayer = m_layers[i];
		pLayer->m_activationCuda.resize(pLayer->m_layer.outputs());
		pLayer->forwardProp_trainingCuda(*pInput, pLayer->m_activationCuda);
		pInput = &pLayer->m_activationCuda;
	}
	return *(GCudaVector*)pInput;
}

void GContextNeuralNet::backPropCuda()
{
	m_nn.backPropCuda(*this, predictionCuda(), predictionCuda(), blameCuda(), blameCuda());
}

void GContextNeuralNet::backPropCuda(const GCudaVector& input, GCudaVector& inBlame)
{
	m_nn.backPropCuda(*this, input, predictionCuda(), blameCuda(), inBlame);
}

void GContextNeuralNet::updateGradientCuda(const GCudaVector &input, GCudaVector& gradient)
{
	m_nn.updateGradientCuda(*this, input, input, gradient);
}
#endif // GCUDA
*/







GNeuralNetLearner::GNeuralNetLearner()
: GIncrementalLearner(), m_pOptimizer(nullptr)
{}

GNeuralNetLearner::GNeuralNetLearner(const GDomNode* pNode)
: GIncrementalLearner(pNode),
m_nn(pNode->field("nn")),
m_pOptimizer(nullptr)
{
}

GNeuralNetLearner::~GNeuralNetLearner()
{
	delete(m_pOptimizer);
}

GNeuralNetOptimizer& GNeuralNetLearner::optimizer()
{
	if(!m_pOptimizer)
		m_pOptimizer = new GSGDOptimizer(m_nn, m_rand);
	return *m_pOptimizer;
}

#ifndef MIN_PREDICT
// virtual
GDomNode* GNeuralNetLearner::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNeuralNetLearner");
	pNode->addField(pDoc, "nn", m_nn.serialize(pDoc));
	return pNode;
}
#endif // MIN_PREDICT

void GNeuralNetLearner::trainIncremental(const GVec &in, const GVec &out)
{
	throw Ex("GNeuralNetLearner::trainIncremental is not implemented (need to use GDifferentiableOptimizer).");
}

void GNeuralNetLearner::trainSparse(GSparseMatrix &features, GMatrix &labels)
{
	throw Ex("GNeuralNetLearner::trainSparse is not implemented (need to use GDifferentiableOptimizer).");
}

void GNeuralNetLearner::trainInner(const GMatrix& features, const GMatrix& labels)
{
	beginIncrementalLearningInner(features.relation(), labels.relation());
	GSGDOptimizer optimizer(m_nn, m_rand);
	optimizer.optimizeWithValidation(features, labels);
}

void GNeuralNetLearner::clear()
{
	// Don't delete the layers here, because their topology will affect future training
}

// virtual
bool GNeuralNetLearner::supportedFeatureRange(double* pOutMin, double* pOutMax)
{
	*pOutMin = -1.0;
	*pOutMax = 1.0;
	return false;
}

// virtual
bool GNeuralNetLearner::supportedLabelRange(double* pOutMin, double* pOutMax)
{
	if(nn().layerCount() > 0)
	{
		*pOutMin = -1.0;
		*pOutMax = 1.0;
	}
	else
	{
		// Assume the tanh function is the default
		*pOutMin = -1.0;
		*pOutMax = 1.0;
	}
	return false;
}

#ifndef MIN_PREDICT
// virtual
void GNeuralNetLearner::predictDistribution(const GVec& in, GPrediction* pOut)
{
	throw Ex("Sorry, this model does not predict a distribution");
}
#endif // MIN_PREDICT

// virtual
void GNeuralNetLearner::predict(const GVec& in, GVec& out)
{
	GNeuralNetOptimizer& opt = optimizer();
	out.copy(m_nn.forwardProp(opt.weights(), in));
}

// virtual
void GNeuralNetLearner::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	if(labelRel.size() < 1)
		throw Ex("The label relation must have at least 1 attribute");
	if(m_nn.inputs() != featureRel.size())
		throw Ex("This data has ", to_str(featureRel.size()), " features, but the model expects ", to_str(m_nn.inputs()));
	if(m_nn.outputs() != labelRel.size())
		throw Ex("This data has ", to_str(labelRel.size()), " labels, but the model expects ", to_str(m_nn.outputs()));
	delete(m_pOptimizer);
	m_pOptimizer = nullptr;
	optimizer();
}


#ifndef MIN_PREDICT
void GNeuralNet_testMath()
{
	GMatrix features(0, 2);
	GVec& vec = features.newRow();
	vec[0] = 0.0;
	vec[1] = -0.7;
	GMatrix labels(0, 1);
	labels.newRow()[0] = 1.0;

	// Make the Neural Network
	GNeuralNetLearner nn;
	GBlockLinear* b0 = new GBlockLinear(2, 3);
	GBlockTanh* b1 = new GBlockTanh(3);
	GBlockLinear* b2 = new GBlockLinear(3, 1);
	GBlockTanh* b3 = new GBlockTanh(1);
	nn.nn().add(b0);
	nn.nn().add(b1);
	nn.nn().add(b2);
	nn.nn().add(b3);
	nn.beginIncrementalLearning(features.relation(), labels.relation());
	GNeuralNetOptimizer& opt = nn.optimizer();
	opt.setLearningRate(0.175);
	if(nn.nn().weightCount() != 13)
		throw Ex("Wrong number of weights");
	GVec& w = opt.weights();

	// Init the weights
	w[0] = -0.01;
	w[1] = 0.01;
	w[2] = -0.02;
	w[3] = -0.03;
	w[4] = 0.04;
	w[5] = 0.03;
	w[6] = 0.03;
	w[7] = -0.02;
	w[8] = 0.02;
	w[9] = 0.02;
	w[10] = -0.01;
	w[11] = 0.03;
	w[12] = 0.02;

	// Present one pattern for training
	double tol = 1e-12;
	GVec pat(2);
	pat.copy(features[0]);
	GVec pred(1);
	opt.optimizeIncremental(features[0], labels[0]);

	// Check forward prop values
	if(std::abs(-0.031 - b0->output[0]) > tol) throw Ex("forward prop problem");
	if(std::abs(-0.030990073482402569 - b1->output[0]) > tol) throw Ex("forward prop problem");
	if(std::abs(0.020350024432229302 - b2->output[0]) > tol) throw Ex("forward prop problem");
	if(std::abs(0.020347215756407563 - b3->output[0]) > tol) throw Ex("forward prop problem");

	// Check blame values
	if(std::abs(0.97965278424359248 - b3->outBlame[0]) > tol) throw Ex("problem computing output blame");
	if(std::abs(0.97924719898884915 - b2->outBlame[0]) > tol) throw Ex("back prop problem");
	if(std::abs(-0.0097924719898884911 - b1->outBlame[0]) > tol) throw Ex("back prop problem");
	if(std::abs(0.029377415969665473 - b1->outBlame[1]) > tol) throw Ex("back prop problem");
	if(std::abs(0.019584943979776982 - b1->outBlame[2]) > tol) throw Ex("back prop problem");	
	if(std::abs(-0.00978306745006032 - b0->outBlame[0]) > tol) throw Ex("back prop problem");
	if(std::abs(0.02936050107376107 - b0->outBlame[1]) > tol) throw Ex("back prop problem");
	if(std::abs(0.01956232122115741 - b0->outBlame[2]) > tol) throw Ex("back prop problem");

	// Check updated weights
	if(std::abs(-0.011712036803760557 - w[0]) > tol) throw Ex("weight update problem");
	if(std::abs(0.015138087687908187 - w[1]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.016576593786297455 - w[2]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.03 - w[3]) > tol) throw Ex("weight update problem");
	if(std::abs(0.04 - w[4]) > tol) throw Ex("weight update problem");
	if(std::abs(0.03 - w[5]) > tol) throw Ex("weight update problem");
	if(std::abs(0.03119842576263239 - w[6]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.023596661381535732 - w[7]) > tol) throw Ex("weight update problem");
	if(std::abs(0.01760361565040822 - w[8]) > tol) throw Ex("weight update problem");
	if(std::abs(0.191368259823049 - w[9]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.015310714964467731 - w[10]) > tol) throw Ex("weight update problem");
	if(std::abs(0.034112048752708297 - w[11]) > tol) throw Ex("weight update problem");
	if(std::abs(0.014175723281037968 - w[12]) > tol) throw Ex("weight update problem");
}


// static
void GNeuralNetLearner::test()
{
	GRand prng(0);
	GNeuralNet_testMath();
}

#endif // MIN_PREDICT





} // namespace GClasses
