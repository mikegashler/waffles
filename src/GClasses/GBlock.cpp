/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Michael R. Smith,
    Stephen Ashmore,
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

#include "GBlock.h"
#include "GBlockWeightless.h"
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

GBlock::GBlock()
{}

GBlock::GBlock(GDomNode* pNode)
{
	m_inPos = pNode->field("inpos")->asInt();
}

GDomNode* GBlock::baseDomNode(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "type", pDoc->newInt(type()));
	pNode->addField(pDoc, "inpos", pDoc->newInt(m_inPos));
	return pNode;
}

GBlock* GBlock::deserialize(GDomNode* pNode)
{
	BlockType e = (BlockType)pNode->field("type")->asInt();
	switch(e)
	{
		case block_tanh: return new GBlockTanh(pNode);
		case block_identity: return new GBlockIdentity(pNode);
		case block_logistic: return new GBlockLogistic(pNode);
		case block_bentidentity: return new GBlockBentIdentity(pNode);
		case block_softroot: return new GBlockSoftRoot(pNode);
		case block_sigexp: return new GBlockSigExp(pNode);
		case block_gaussian: return new GBlockGaussian(pNode);
		case block_sine: return new GBlockSine(pNode);
		case block_rectifier: return new GBlockRectifier(pNode);
		case block_leakyrectifier: return new GBlockLeakyRectifier(pNode);
		case block_softplus: return new GBlockSoftPlus(pNode);
		case block_linear: return new GBlockLinear(pNode);
		//case block_restrictedboltzmannmachine: return new GBlockRestrictedBoltzmannMachine(pNode);
		//case block_convolutional1d: return new GBlockConvolutional1D(pNode);
		//case block_convolutional2d: return new GBlockConvolutional2D(pNode);
		default: throw Ex("Unrecognized neural network layer type: ", GClasses::to_str((int)e));
	}
}











GBlockLinear::GBlockLinear(size_t outputs, size_t inputs)
{
	resize(inputs, outputs);
}

GBlockLinear::GBlockLinear(GDomNode* pNode)
: GBlock(pNode), m_weights(pNode->field("weights"))
{}

GDomNode* GBlockLinear::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "weights", m_weights.serialize(pDoc));
	return pNode;
}

void GBlockLinear::resize(size_t in, size_t out)
{
	if(in == inputs() && out == outputs())
		return;
	m_weights.resize(in + 1, out);
}

/// Makes a string representation of this layer
std::string GBlockLinear::to_str() const
{
	std::ostringstream oss;
	oss << "[GBlockLinear:" << inputs() << "->" << outputs() << "]";
	return oss.str();
}

void GBlockLinear::forwardProp(const GVec& input, GVec& output) const
{
	output.copy(bias());
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);
	for(size_t i = 0; i < input.size(); i++)
		output.addScaled(input[i], m_weights.row(i));
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);
}

void GBlockLinear::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	for(size_t i = 0; i < inBlame.size(); i++)
		inBlame[i] += outBlame.dotProduct(m_weights[i]);
}

void GBlockLinear::updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	GAssert(gradient.size() == weightCount(), "gradient must match the dimensions of weights!");
	double *delta = gradient.data();
	for(size_t i = 0; i < inputs(); ++i)
	{
		double act = input[i];
		for(size_t j = 0; j < outputs(); ++j)
			*delta++ += outBlame[j] * act;
	}
	for(size_t j = 0; j < outputs(); ++j)
		*delta++ += outBlame[j];
}

void GBlockLinear::step(double learningRate, const GVec &gradient)
{
	GAssert(gradient.size() == weightCount(), "gradient must match the dimensions of weights!");
	const double *delta = gradient.data();
	GVec &b = bias();
	for(size_t i = 0; i < inputs(); ++i)
		for(size_t j = 0; j < outputs(); ++j)
			m_weights[i][j] += learningRate * *delta++;
	for(size_t j = 0; j < outputs(); ++j)
		b[j] += learningRate * *delta++;
}

void GBlockLinear::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < inputs(); i++)
		m_weights[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

void GBlockLinear::diminishWeights(double amount, bool regularizeBiases)
{
	for(size_t i = 0; i < inputs(); i++)
		m_weights[i].regularizeL1(amount);
	if(regularizeBiases)
		bias().regularizeL1(amount);
}

void GBlockLinear::contractWeights(double factor, bool contractBiases, const GVec& output)
{
	GVec& b = bias();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		// (Note that the official implementation of contractive regularization multiplies by the
		// derivative of the activation function, but we separate activation functions into separate
		// layers, and I don't think they should depend on each other, so this implementation
		// assumes the tanh activation function for regularization purposes.)
		double activ = tanh(output[i]);
		double aprime = 1.0 - (activ * activ);
		double f = 1.0 - factor * aprime;
		for(size_t j = 0; j < inputs(); j++)
			m_weights[j][i] *= f;
		if(contractBiases)
			b[i] *= f;
	}
}

size_t GBlockLinear::weightCount() const
{
	return (inputs() + 1) * outputs();
}

size_t GBlockLinear::weightsToVector(double* pOutVector) const
{
	m_weights.toVector(pOutVector);
	return weightCount();
}

size_t GBlockLinear::vectorToWeights(const double* pVector)
{
	m_weights.fromVector(pVector, inputs() + 1);
	return weightCount();
}

void GBlockLinear::copyWeights(const GBlock* pSource)
{
	GBlockLinear *src = (GBlockLinear*) pSource;
	m_weights.copyBlock(src->m_weights, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
}

void GBlockLinear::resetWeights(GRand& rand)
{
	size_t outputCount = outputs();
	size_t inputCount = inputs();
	double mag = std::max(0.03, 1.0 / inputCount);
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		GVec& w = m_weights[i];
		for(size_t j = 0; j < outputCount; j++)
			w[j] = rand.normal() * mag;
	}
}

void GBlockLinear::perturbWeights(GRand &rand, double deviation)
{
	for(size_t j = 0; j < m_weights.rows(); j++)
		GVec::perturb(m_weights[j].data(), deviation, m_weights.cols(), rand);
}

void GBlockLinear::maxNorm(double min, double max)
{
	size_t outputCount = outputs();
	size_t inputCount = inputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		double squaredMag = 0;
		for(size_t j = 0; j < inputCount; j++)
		{
			double d = m_weights[j][i];
			squaredMag += (d * d);
		}
		if(squaredMag > max * max)
		{
			double scal = max / sqrt(squaredMag);
			for(size_t j = 0; j < inputCount; j++)
				m_weights[j][i] *= scal;
		}
		else if(squaredMag < min * min)
		{
			if(squaredMag == 0.0)
			{
				for(size_t j = 0; j < inputCount; j++)
					m_weights[j][i] = 1.0;
				squaredMag = (double) inputCount;
			}
			double scal = min / sqrt(squaredMag);
			for(size_t j = 0; j < inputCount; j++)
				m_weights[j][i] *= scal;
		}
	}
}

void GBlockLinear::renormalizeInput(size_t input, double oldMin, double oldMax, double newMin, double newMax)
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

void GBlockLinear::transformWeights(GMatrix& transform, const GVec& offset)
{
	if(transform.rows() != inputs())
		throw Ex("Transformation matrix not suitable size for this layer");
	if(transform.rows() != transform.cols())
		throw Ex("Expected a square transformation matrix.");
	size_t outputCount = outputs();

	GMatrix temp(inputs(), outputs());
	temp.copyBlock(m_weights, 0, 0, inputs(), outputs());

	GMatrix* pNewWeights = GMatrix::multiply(transform, temp, true, false);
	std::unique_ptr<GMatrix> hNewWeights(pNewWeights);
	m_weights.copyBlock(*pNewWeights, 0, 0, pNewWeights->rows(), outputCount, 0, 0, false);
	GVec n(outputs());
	n.fill(0.0);
	for(size_t i = 0; i < inputs(); i++)
		n.addScaled(offset[i], m_weights.row(i));
	bias() += n;
}









/*
GBlockMaxOut::GBlockMaxOut(size_t outputs, size_t inputs)
: GBlock()
{
	resize(inputs, outputs);
}

GBlockMaxOut::GBlockMaxOut(GDomNode* pNode)
: GBlock(pNode)
{}

GBlockMaxOut::~GBlockMaxOut()
{
}

GDomNode* GBlockMaxOut::serialize(GDom* pDoc) const
{
	throw Ex("Sorry, not implemented yet");
}

// virtual
std::string GBlockMaxOut::to_str() const
{
	std::ostringstream os;
	os << "[GBlockMaxOut:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "\n";
	os << " Weights: " << GClasses::to_str(m_weights) << "\n";
	os << " Bias: " << GClasses::to_str(m_bias) << "\n";
	os << "]";
	return os.str();
}

void GBlockMaxOut::resize(size_t inputCount, size_t outputCount)
{
	if(inputCount == inputs() && outputCount == outputs())
		return;
	m_input.setData(nullptr, inputCount);
	m_output.setData(nullptr, outputCount);
	if(inputCount != 0 && outputCount != 0)
	{
		// Weights
		m_weights.resize(inputCount, outputCount);
		m_winners.resize(outputCount);

		// Bias
		m_bias.resize(2, inputCount);
	}
}

// virtual
void GBlockMaxOut::resetWeights(GRand& rand)
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
	GVec& b = bias();
	for(size_t i = 0; i < inputCount; i++)
		b[i] = rand.normal() * mag;
}

// virtual
void GBlockMaxOut::forwardProp(const GVec& input, GVec& output) const
{
	const GVec& b = bias();
	for(size_t i = 0; i < output.size(); i++)
	{
		double best = -1e200;
		for(size_t j = 0; j < input.size(); j++)
		{
			double cand = (input[j] + b[j]) * m_weights[j][i];
			if(cand > best)
			{
				best = cand;
				m_winners[i] = j;
			}
		}
		if(rand() % 10 == 0) // todo: get rid of rand()
		{
			size_t j = rand() % input.size(); // todo: get rid of rand()
			best = (input[j] + b[j]) * m_weights[j][i];
			m_winners[i] = j;
		}
		output[i] = best;
	}
}

void GBlockMaxOut::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	const GVec& source = outBlame(); // source
	GVec& inb = inBlame(); // destination
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		size_t up = m_winners[i];
		GAssert(up < inputs());
		inb[up] += m_weights[up][i] * source[i];
	}
}

void GBlockMaxOut::updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	const GVec& err = outBlame();
	const GVec& in = input();
	size_t outputCount = outputs();
	double *delta = gradient.data();
	for(size_t down = 0; down < outputCount; down++)
	{
		size_t up = m_winners[down];
		*delta++ += err[down]; // bias
		*delta++ += err[down] * in[up]; // weights
	}
}

void GBlockMaxOut::step(double learningRate, const GVec &gradient)
{
	size_t outputCount = outputs();
	GVec& bi = bias();
	const double *delta = gradient.data();
	for(size_t down = 0; down < outputCount; down++)
	{
		size_t up = m_winners[down];
		bi[up] += learningRate * *delta++;
		m_weights[up][down] += learningRate * *delta++;
	}
}

void GBlockMaxOut::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

void GBlockMaxOut::diminishWeights(double amount, bool regularizeBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i].regularizeL1(amount);
	if(regularizeBiases)
		bias().regularizeL1(amount);
}

void GBlockMaxOut::transformWeights(GMatrix& transform, const GVec& offset)
{
	throw Ex("Not implemented");
}

void GBlockMaxOut::setWeightsToIdentity(size_t start, size_t count)
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
void GBlockMaxOut::maxNorm(double min, double max)
{
	throw Ex("Not implemented");
}

// virtual
size_t GBlockMaxOut::weightCount() const
{
	return inputs() * (outputs() + 1);
}

// virtual
size_t GBlockMaxOut::weightsToVector(double* pOutVector) const
{
	memcpy(pOutVector, bias().data(), sizeof(double) * inputs());
	pOutVector += inputs();
	m_weights.toVector(pOutVector);
	pOutVector += (inputs() * outputs());
	return inputs() * (outputs() + 1);
}

// virtual
size_t GBlockMaxOut::vectorToWeights(const double* pVector)
{
	bias().set(pVector, inputs());
	pVector += inputs();
	m_weights.fromVector(pVector, inputs());
	pVector += (inputs() * outputs());
	return inputs() * (outputs() + 1);
}

// virtual
void GBlockMaxOut::copyWeights(const GBlock* pSource)
{
	GBlockMaxOut* src = (GBlockMaxOut*)pSource;
	m_weights.copyBlock(src->m_weights, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	bias().copy(src->bias());
}

// virtual
void GBlockMaxOut::perturbWeights(GRand& rand, double deviation)
{
	for(size_t j = 0; j < m_weights.rows(); j++)
		GVec::perturb(m_weights[j].data(), deviation, m_weights.cols(), rand);
	GVec::perturb(bias().data(), deviation, inputs(), rand);
}
*/









GBlockRestrictedBoltzmannMachine::GBlockRestrictedBoltzmannMachine(size_t outputs, size_t inputs)
: GBlock()
{
	resize(inputs, outputs);
}

GBlockRestrictedBoltzmannMachine::GBlockRestrictedBoltzmannMachine(GDomNode* pNode)
: GBlock(pNode),
m_weights(pNode->field("weights")), m_bias(m_weights.rows()), m_biasReverse(3, m_weights.cols())
{
	bias().deserialize(pNode->field("bias"));
	biasReverse().deserialize(pNode->field("biasRev"));
}

GDomNode* GBlockRestrictedBoltzmannMachine::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "weights", m_weights.serialize(pDoc));
	pNode->addField(pDoc, "bias", bias().serialize(pDoc));
	pNode->addField(pDoc, "biasRev", biasReverse().serialize(pDoc));

	return pNode;
}

// virtual
std::string GBlockRestrictedBoltzmannMachine::to_str() const
{
	std::ostringstream os;
	os << "[GBlockRestrictedBoltzmannMachine:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "\n";
	os << " Weights: " << GClasses::to_str(m_weights) << "\n";
	os << " Bias: " << GClasses::to_str(bias()) << "\n";
	os << " BiasReverse: " << GClasses::to_str(biasReverse()) << "\n";
	os << "]";
	return os.str();
}

void GBlockRestrictedBoltzmannMachine::resize(size_t inputCount, size_t outputCount)
{
	if(inputCount == inputs() && outputCount == outputs())
		return;
	m_weights.resize(outputCount, inputCount);
	m_bias.resize(outputCount);
	m_biasReverse.resize(3, inputCount);
}

// virtual
void GBlockRestrictedBoltzmannMachine::resetWeights(GRand& rand)
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
}

// virtual
void GBlockRestrictedBoltzmannMachine::perturbWeights(GRand& rand, double deviation)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		GVec::perturb(m_weights[i].data(), deviation, m_weights.cols(), rand);
	GVec::perturb(m_bias.data(), deviation, m_bias.size(), rand);
}

// virtual
void GBlockRestrictedBoltzmannMachine::forwardProp(const GVec& input, GVec& output) const
{
	output.copy(bias());
	size_t outputCount = outputs();
	double* pNet = output.data();
	for(size_t i = 0; i < outputCount; i++)
		*(pNet++) += input.dotProduct(m_weights[i]);
}

void GBlockRestrictedBoltzmannMachine::feedBackward(const GVec& output, GVec& input) const
{
	m_weights.multiply(output, input, true);
	input += biasReverse();
}

void GBlockRestrictedBoltzmannMachine::resampleHidden(GRand& rand, GVec& output)
{
	double* pH = output.data();
	size_t outputCount = outputs();
	for(size_t i = 0; i < outputCount; i++)
	{
		*pH = rand.uniform() < *pH ? 1.0 : 0.0;
		pH++;
	}
}

void GBlockRestrictedBoltzmannMachine::resampleVisible(GRand& rand, GVec& input)
{
	double* pV = input.data();
	size_t inputCount = inputs();
	for(size_t i = 0; i < inputCount; i++)
	{
		*pV = rand.uniform() < *pV ? 1.0 : 0.0;
		pV++;
	}
}

void GBlockRestrictedBoltzmannMachine::drawSample(GRand& rand, size_t iters, GVec& output, GVec& input)
{
	for(size_t i = 0; i < output.size(); i++)
		output[i] = ((rand.next() & 1) == 0 ? 0.0 : 1.0);
	for(size_t i = 0; i < iters; i++)
	{
		feedBackward(output, input);
		forwardProp(input, output);
		resampleHidden(rand, output);
	}
	feedBackward(output, input);
}
/*
double GBlockRestrictedBoltzmannMachine::freeEnergy(const GVec& visibleSample, GVec& hiddenBuf, GVec& visibleBuf)
{
	forwardProp(visibleSample, hiddenBuf);
	GVec& buf = error();
	m_weights.multiply(hiddenBuf, buf, false);
	return -activation().dotProduct(buf) -
		biasReverse().dotProduct(activationReverse()) -
		bias().dotProduct(activation());
}

void GBlockRestrictedBoltzmannMachine::contrastiveDivergence(GRand& rand, const GVec& visibleSample, double learningRate, size_t gibbsSamples)
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
*/
void GBlockRestrictedBoltzmannMachine::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	m_weights.multiply(outBlame, inBlame, true);
}

void GBlockRestrictedBoltzmannMachine::updateGradient(const GVec& input, const GVec& outBlame, GVec& gradient) const
{
	size_t outputCount = outputs();
	GVecWrapper delta(gradient.data(), m_weights.cols());
	for(size_t i = 0; i < outputCount; i++)
	{
		delta.vec().addScaled(outBlame[i], input);
		delta.setData(delta.vec().data() + m_weights.cols(), m_weights.cols());
	}
	delta.vec() += outBlame;
}

void GBlockRestrictedBoltzmannMachine::step(double learningRate, const GVec &gradient)
{
	size_t outputCount = outputs();
	GConstVecWrapper delta(gradient.data(), m_weights.cols());
	for(size_t i = 0; i < outputCount; i++)
	{
		m_weights[i].addScaled(learningRate, delta.vec());
		delta.setData(delta.vec().data() + m_weights.cols(), m_weights.cols());
	}
	bias().addScaled(learningRate, delta.vec());
}

void GBlockRestrictedBoltzmannMachine::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

void GBlockRestrictedBoltzmannMachine::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_weights[i].regularizeL1(amount);
	if(diminishBiases)
		bias().regularizeL1(amount);
}

// virtual
void GBlockRestrictedBoltzmannMachine::maxNorm(double min, double max)
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
size_t GBlockRestrictedBoltzmannMachine::weightCount() const
{
	return (inputs() + 1) * outputs();
}

// virtual
size_t GBlockRestrictedBoltzmannMachine::weightsToVector(double* pOutVector) const
{
	memcpy(pOutVector, bias().data(), outputs() * sizeof(double));
	pOutVector += outputs();
	m_weights.toVector(pOutVector);
	return (inputs() + 1) * outputs();
}

// virtual
size_t GBlockRestrictedBoltzmannMachine::vectorToWeights(const double* pVector)
{
	memcpy(bias().data(), pVector, outputs() * sizeof(double));
	pVector += outputs();
	m_weights.fromVector(pVector, inputs());
	return (inputs() + 1) * outputs();
}

// virtual
void GBlockRestrictedBoltzmannMachine::copyWeights(const GBlock* pSource)
{
	GBlockRestrictedBoltzmannMachine* src = (GBlockRestrictedBoltzmannMachine*)pSource;
	m_weights.copyBlock(src->m_weights, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	bias().copy(src->bias());
}













GBlockConvolutional1D::GBlockConvolutional1D(size_t inputSamples, size_t inputChannels, size_t kernelSize, size_t kernelsPerChannel)
: GBlock(),
m_inputSamples(inputSamples),
m_inputChannels(inputChannels),
m_outputSamples(inputSamples - kernelSize + 1),
m_kernelsPerChannel(kernelsPerChannel),
m_kernels(inputChannels * kernelsPerChannel, kernelSize),
m_bias(inputChannels * kernelsPerChannel)
{
	if(kernelSize > inputSamples)
		throw Ex("kernelSize must be <= inputSamples");
}

GBlockConvolutional1D::GBlockConvolutional1D(GDomNode* pNode)
: GBlock(pNode),
m_inputSamples((size_t)pNode->field("isam")->asInt()),
m_inputChannels((size_t)pNode->field("ichan")->asInt()),
m_outputSamples((size_t)pNode->field("osam")->asInt()),
m_kernelsPerChannel((size_t)pNode->field("kpc")->asInt()),
m_kernels(pNode->field("kern")),
m_bias(pNode->field("bias"))
{
}

// virtual
GDomNode* GBlockConvolutional1D::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "isam", pDoc->newInt(m_inputSamples));
	pNode->addField(pDoc, "ichan", pDoc->newInt(m_inputChannels));
	pNode->addField(pDoc, "osam", pDoc->newInt(m_outputSamples));
	pNode->addField(pDoc, "kpc", pDoc->newInt(m_kernelsPerChannel));
	pNode->addField(pDoc, "kern", m_kernels.serialize(pDoc));
	pNode->addField(pDoc, "bias", m_bias.serialize(pDoc));
	return pNode;
}

// virtual
std::string GBlockConvolutional1D::to_str() const
{
	std::ostringstream os;
	os << "[GBlockConvolutional1D:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "\n";
	os << " Kernels: " << GClasses::to_str(m_kernels) << "\n";
	os << "]";
	return os.str();
}

// virtual
void GBlockConvolutional1D::resize(size_t inputSize, size_t outputSize)
{
	if(inputSize != m_inputSamples * m_inputChannels)
		throw Ex("Changing the size of GBlockConvolutional1D is not supported");
	if(outputSize != m_inputChannels * m_kernelsPerChannel * m_outputSamples)
		throw Ex("Changing the size of GBlockConvolutional1D is not supported");
}

// virtual
void GBlockConvolutional1D::resetWeights(GRand& rand)
{
	size_t kernelSize = m_kernels.cols();
	double mag = std::max(0.03, 1.0 / kernelSize);
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].fillNormal(rand, mag);
	bias().fillNormal(rand, mag);
}

// virtual
void GBlockConvolutional1D::forwardProp(const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_inputSamples * m_inputChannels);
	GAssert(output.size() == m_inputChannels * m_kernelsPerChannel * m_outputSamples);

	// Copy bias to net
	for(size_t i = 0; i < m_outputSamples; i++)
		output.put(bias().size() * i, bias());

	// Feed in through
	size_t kernelSize = m_kernels.cols();
	size_t netPos = 0;
	size_t inPos = 0;
	for(size_t i = 0; i < m_outputSamples; i++) // for each output sample...
	{
		size_t kern = 0;
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				const GVec& w = m_kernels[kern++];
				double d = 0.0;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
					d += w[l] * input[inPos + l * m_inputChannels];
				output[netPos++] += d;
			}
			inPos++;
		}
	}
}

// virtual
void GBlockConvolutional1D::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	size_t kernelSize = m_kernels.cols();
	inBlame.fill(0.0);
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
					inBlame[upPos + samp] += w[l] * outBlame[downPos];
					samp += m_inputChannels;
				}
				downPos++;
			}
			upPos++;
		}
	}
}

void GBlockConvolutional1D::updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	size_t kernelSize = m_kernels.cols();
	size_t errPos = 0;
	size_t upPos = 0;
	for(size_t i = 0; i < m_outputSamples; i++) // for each sample...
	{
		double *delta = gradient.data();
		for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
		{
			for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
			{
				size_t upOfs = 0;
				for(size_t l = 0; l < kernelSize; l++) // for each connection...
				{
					*delta++ += outBlame[errPos] * input[upPos + upOfs];
					upOfs += m_inputChannels;
				}
				*delta++ += outBlame[errPos++];
			}
			upPos++;
		}
	}
}

void GBlockConvolutional1D::step(double learningRate, const GVec &gradient)
{
	size_t kernelSize = m_kernels.cols();
	size_t errPos = 0;
	size_t upPos = 0;
	size_t kern = 0;
	const double *delta = gradient.data();
	for(size_t j = 0; j < m_inputChannels; j++) // for each input channel...
	{
		for(size_t k = 0; k < m_kernelsPerChannel; k++) // for each kernel...
		{
			GVec& d = m_kernels[kern];
			size_t upOfs = 0;
			for(size_t l = 0; l < kernelSize; l++) // for each connection...
			{
				d[l] += learningRate * *delta++;
				upOfs += m_inputChannels;
			}
			bias()[kern++] += learningRate * *delta++;
			errPos++;
		}
		upPos++;
	}
}

// virtual
void GBlockConvolutional1D::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i] *= factor;
	if(scaleBiases)
		bias() *= factor;
}

// virtual
void GBlockConvolutional1D::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].regularizeL1(amount);
	if(diminishBiases)
		bias().regularizeL1(amount);
}

// virtual
size_t GBlockConvolutional1D::weightCount() const
{
	return (m_kernels.rows() + 1) * m_kernels.cols();
}

// virtual
size_t GBlockConvolutional1D::weightsToVector(double* pOutVector) const
{
	memcpy(pOutVector, bias().data(), m_kernels.rows() * sizeof(double));
	pOutVector += m_kernels.rows();
	m_kernels.toVector(pOutVector);
	return (m_kernels.rows() + 1) * m_kernels.cols();
}

// virtual
size_t GBlockConvolutional1D::vectorToWeights(const double* pVector)
{
	memcpy(bias().data(), pVector, m_kernels.rows() * sizeof(double));
	pVector += m_kernels.rows();
	m_kernels.fromVector(pVector, m_kernels.rows());
	return (m_kernels.rows() + 1) * m_kernels.cols();
}

// virtual
void GBlockConvolutional1D::copyWeights(const GBlock* pSource)
{
	GBlockConvolutional1D* src = (GBlockConvolutional1D*)pSource;
	m_kernels.copyBlock(src->m_kernels, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
	bias().copy(src->bias());
}

// virtual
void GBlockConvolutional1D::perturbWeights(GRand& rand, double deviation)
{
	size_t kernelSize = m_kernels.cols();
	for(size_t i = 0; i < m_kernels.rows(); i++)
		GVec::perturb(m_kernels[i].data(), deviation, kernelSize, rand);
	GVec::perturb(bias().data(), deviation, m_kernels.rows(), rand);
}

// virtual
void GBlockConvolutional1D::maxNorm(double min, double max)
{
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].clip(-max, max);
}












/*
size_t GBlockConvolutional2D::Image::npos = (size_t) -1;

GBlockConvolutional2D::Image::Image(GVec *_data, size_t _width, size_t _height, size_t _channels)
: data(_data), width(_width), height(_height), channels(_channels), interlaced(true), dx(0), dy(0), dz(0), px(0), py(0), sx(1), sy(1), invertStride(false), flip(false) {}

size_t GBlockConvolutional2D::Image::index(size_t x, size_t y, size_t z) const
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
		return npos;

	if(interlaced)
		return (y * width + x) * channels + z;
	else
		return (z * height + y) * width + x;
}

double GBlockConvolutional2D::Image::read(size_t x, size_t y, size_t z) const
{
	size_t i = index(x, y, z);
	if(i == npos)
		return 0.0;
	else
		return (*data)[i];
}

double &GBlockConvolutional2D::Image::at(size_t x, size_t y, size_t z)
{
	size_t i = index(x, y, z);
	if(i == npos)
		throw Ex("tried to access invalid image location!");
	else
		return (*data)[i];
}


size_t GBlockConvolutional2D::none = (size_t) -1;

GBlockConvolutional2D::GBlockConvolutional2D(size_t width, size_t height, size_t channels, size_t kWidth, size_t kHeight, size_t kCount)
: GBlock(),
  m_width(width), m_height(height), m_channels(channels),
  m_kWidth(kWidth), m_kHeight(kHeight),
  m_outputWidth(width - kWidth + 1), m_outputHeight(height - kHeight + 1),
  m_bias(kCount),
  m_kernels(kCount, kWidth * kHeight * channels),
  m_activation(2, m_outputWidth * m_outputHeight * kCount),
  m_kernelImage(NULL, kWidth, kHeight, channels), m_deltaImage(NULL, kWidth, kHeight, channels),
  m_inputImage(NULL, width, height, channels), m_upStreamErrorImage(NULL, width, height, channels),
  m_actImage(&m_activation[0], m_outputWidth, m_outputHeight, kCount), m_errImage(&m_activation[1], m_outputWidth, m_outputHeight, kCount)
{
	m_input.setData(nullptr, width * height * channels);
	m_output.setData(nullptr, m_outputWidth * m_outputHeight * kCount);	
}


GBlockConvolutional2D::GBlockConvolutional2D(size_t kWidth, size_t kHeight, size_t kCount)
: GBlock(),
  m_width(FLEXIBLE_SIZE), m_height(FLEXIBLE_SIZE), m_channels(FLEXIBLE_SIZE),
  m_kWidth(kWidth), m_kHeight(kHeight),
  m_outputWidth(0), m_outputHeight(0),
  m_bias(kCount),
  m_kernels(kCount, 0),
  m_activation(2, 0),
  m_kernelImage(NULL, kWidth, kHeight, 0), m_deltaImage(NULL, kWidth, kHeight, 0),
  m_inputImage(NULL, 0, 0, 0), m_upStreamErrorImage(NULL, 0, 0, 0),
  m_actImage(&m_activation[0], 0, 0, 0), m_errImage(&m_activation[1], 0, 0, 0)
{}

GBlockConvolutional2D::GBlockConvolutional2D(GDomNode* pNode)
: GBlock(pNode),
  m_width(pNode->field("width")->asInt()), m_height(pNode->field("height")->asInt()), m_channels(pNode->field("channels")->asInt()),
  m_kWidth(pNode->field("kWidth")->asInt()), m_kHeight(pNode->field("kHeight")->asInt()),
  m_outputWidth(pNode->field("outputWidth")->asInt()), m_outputHeight(pNode->field("outputHeight")->asInt()),
  m_bias(pNode->field("bias")),
  m_kernels(pNode->field("kernels")),
  m_activation(2, m_outputWidth * m_outputHeight * m_kernels.rows()),
  m_kernelImage(NULL, m_kWidth, m_kHeight, m_channels), m_deltaImage(NULL, m_kWidth, m_kHeight, m_channels),
  m_inputImage(NULL, m_width, m_height, m_channels), m_upStreamErrorImage(NULL, m_width, m_height, m_channels),
  m_actImage(&m_activation[0], m_outputWidth, m_outputHeight, m_kernels.rows()), m_errImage(&m_activation[1], m_outputWidth, m_outputHeight, m_kernels.rows())
{
	m_inputImage.sx	= pNode->field("strideX")->asInt();
	m_inputImage.sy	= pNode->field("strideY")->asInt();
	m_inputImage.px	= pNode->field("paddingX")->asInt();
	m_inputImage.py	= pNode->field("paddingY")->asInt();

	setInputInterlaced(pNode->field("inputInterlaced")->asBool());
	setKernelsInterlaced(pNode->field("kernelsInterlaced")->asBool());
	setOutputInterlaced(pNode->field("outputInterlaced")->asBool());
}

GDomNode *GBlockConvolutional2D::serialize(GDom *pDoc) const
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
	pNode->addField(pDoc, "inputInterlaced", pDoc->newBool(m_inputImage.interlaced));
	pNode->addField(pDoc, "kernelsInterlaced", pDoc->newBool(m_kernelImage.interlaced));
	pNode->addField(pDoc, "outputInterlaced", pDoc->newBool(m_actImage.interlaced));
	pNode->addField(pDoc, "bias", m_bias.serialize(pDoc));
	pNode->addField(pDoc, "kernels", m_kernels.serialize(pDoc));
	return pNode;

}

std::string GBlockConvolutional2D::to_str() const
{
	std::stringstream ss;
	ss << "[GBlockConvolutional2D:\n"
	   << "    " << m_width << "x" << m_height << "x" << m_channels << " (stride=" << m_inputImage.sx << "," << m_inputImage.sy << "; padding=" << m_inputImage.px << "," << m_inputImage.py << ")\n"
	   << " *  " << m_kWidth << "x" << m_kHeight << "\n"
	   << " -> " << m_outputWidth << "x" << m_outputHeight << "x" << m_kernels.rows() << "\n"
	   << "]";
	return ss.str();
}

void GBlockConvolutional2D::resize(size_t inputSize, size_t outputSize)
{
	if(inputSize != inputs() || outputSize != outputs())
		throw Ex("GBlockConvolutional2D can only be resized given an upstream convolutional layer!");
}

// void GBlockConvolutional2D::resizeInputs(GBlock *pUpStreamLayer)
// {
// 	if(pUpStreamLayer->type() != block_convolutional2d)
// 		throw Ex("GBlockConvolutional2D can only be resized given an upstream convolutional layer!");
// 
// 	GBlockConvolutional2D &upstream = *((GBlockConvolutional2D *) pUpStreamLayer);
// 
// 	m_width			= upstream.outputWidth();
// 	m_height		= upstream.outputHeight();
// 	m_channels		= upstream.outputChannels();
// 
// 	m_kernels.resize(m_kernels.rows(), m_kWidth * m_kHeight * m_channels);
// 
// 	m_bias.fill(0.0);
// 	m_kernels.fill(0.0);
// 
// 	m_inputImage.width = m_width;
// 	m_inputImage.height = m_height;
// 	m_inputImage.channels = m_channels;
// 
// 	m_upStreamErrorImage.width = m_width;
// 	m_upStreamErrorImage.height = m_height;
// 	m_upStreamErrorImage.channels = m_channels;
// 
// 	m_kernelImage.channels = m_channels;
// 	m_deltaImage.channels = m_channels;
// 
// 	updateOutputSize();
// }

void GBlockConvolutional2D::forwardProp(const GVec& input, GVec& output) const
{
	m_inputImage.data = const_cast<GVec *>(&in);

	Image &n = m_actImage;
	n.data->fill(0.0);
	for(n.dz = 0; n.dz < n.channels; ++n.dz)
	{
		m_kernelImage.data = &m_kernels[n.dz];
		convolve(m_inputImage, m_kernelImage, n);
		for(size_t y = 0; y < n.height; ++y)
			for(size_t x = 0; x < n.width; ++x)
				n.at(x, y) += m_bias[n.dz];
	}
	n.dz = 0;
}

void GBlockConvolutional2D::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
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

void GBlockConvolutional2D::updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	Image &err = m_errImage;
	Image &in = m_inputImage;
	in.data = const_cast<GVec *>(&upStreamActivation);
	size_t count = m_kernels.cols();
	GVecWrapper delta(gradient.data(), count);
	m_deltaImage.data = &delta.vec();
	for(err.dz = 0; err.dz < err.channels; ++err.dz)
	{
		double *biasDelta = m_deltaImage.data->data() + count;
		m_deltaImage.data->fill(0.0);
		for(in.dz = m_deltaImage.dz = 0; in.dz < in.channels; ++in.dz, ++m_deltaImage.dz)
			for(in.dy = 0; in.dy < err.height; ++in.dy)
				for(in.dx = 0; in.dx < err.width; ++in.dx)
				{
					addScaled(in, err.read(in.dx, in.dy), m_deltaImage);
					*biasDelta += err.read(in.dx, in.dy);
				}
		m_deltaImage.dz = 0;
		delta.setData(delta.vec().data() + count + 1);
	}
	in.dz = 0;
}

void GBlockConvolutional2D::step(double learningRate, const GVec &gradient)
{
	size_t count = m_kernels.cols();
	GConstVecWrapper delta(gradient.data(), count);
	for(size_t i = 0; i < m_kernels.rows(); i++)
	{
		m_kernels[i].addScaled(learningRate, delta.vec());
		m_bias[i] += learningRate * *(delta.vec().data() + count);
		delta.setData(delta.vec().data() + count + 1);
	}
}

void GBlockConvolutional2D::scaleWeights(double factor, bool scaleBiases)
{
	throw Ex("scaleWeights not implemented");
}

void GBlockConvolutional2D::diminishWeights(double amount, bool regularizeBiases)
{
	throw Ex("diminishWeights not implemented");
}

size_t GBlockConvolutional2D::weightCount() const
{
	return m_kWidth * m_kHeight * m_channels * m_kernels.rows() + m_kernels.rows();
}

size_t GBlockConvolutional2D::weightsToVector(double *pOutVector) const
{
	m_kernels.toVector(pOutVector);
	GVecWrapper(pOutVector + m_kernels.rows() * m_kernels.cols(), m_kernels.rows()).vec().put(0, m_bias);
	return weightCount();
}

size_t GBlockConvolutional2D::vectorToWeights(const double *pVector)
{
	m_kernels.fromVector(pVector, m_kernels.rows());
	m_bias.put(0, GConstVecWrapper(pVector + m_kernels.rows() * m_kernels.cols(), m_kernels.rows()).vec());
	return weightCount();
}

void GBlockConvolutional2D::copyWeights(const GBlock *pSource)
{
	throw Ex("copyWeights not implemented");
}

void GBlockConvolutional2D::resetWeights(GRand &rand)
{
	double mag = std::max(0.03, 1.0 / (m_outputWidth * m_outputHeight * m_kernels.rows()));
	for(size_t i = 0; i < m_kernels.rows(); i++)
		m_kernels[i].fillNormal(rand, mag);
	m_bias.fillNormal(rand, mag);
}

void GBlockConvolutional2D::perturbWeights(GRand &rand, double deviation)
{
	for(size_t j = 0; j < m_kernels.rows(); j++)
		GVec::perturb(m_kernels[j].data(), deviation, m_kernels.cols(), rand);
	GVec::perturb(m_bias.data(), deviation, m_kernels.rows(), rand);
}

void GBlockConvolutional2D::maxNorm(double min, double max)
{
	throw Ex("maxNorm not implemented");
}

void GBlockConvolutional2D::setPadding(size_t px, size_t py)
{
	m_inputImage.px = px;
	m_inputImage.py = (py == none ? px : py);
	updateOutputSize();
}

void GBlockConvolutional2D::setStride(size_t sx, size_t sy)
{
	m_inputImage.sx = sx;
	m_inputImage.sy = (sy == none ? sx : sy);
	updateOutputSize();
}

void GBlockConvolutional2D::setInterlaced(bool interlaced)
{
	setInputInterlaced(interlaced);
	setKernelsInterlaced(interlaced);
	setOutputInterlaced(interlaced);
}

void GBlockConvolutional2D::setInputInterlaced(bool interlaced)
{
	m_inputImage.interlaced = interlaced;
	m_upStreamErrorImage.interlaced = interlaced;
}

void GBlockConvolutional2D::setKernelsInterlaced(bool interlaced)
{
	m_kernelImage.interlaced = interlaced;
	m_deltaImage.interlaced = interlaced;
}

void GBlockConvolutional2D::setOutputInterlaced(bool interlaced)
{
	m_actImage.interlaced = interlaced;
	m_errImage.interlaced = interlaced;
}

void GBlockConvolutional2D::addKernel()
{
	m_kernels.resize(m_kernels.rows() + 1, m_kernels.cols());

	GVec temp(m_bias);
	m_bias.resize(m_kernels.rows() + 1);
	m_bias.put(0, temp);

	m_actImage.channels = m_kernels.rows();
	m_errImage.channels = m_kernels.rows();
	updateOutputSize();
}

void GBlockConvolutional2D::addKernels(size_t n)
{
	for(size_t i = 0; i < n; ++i)
		addKernel();
}

double GBlockConvolutional2D::filterSum(const Image &in, const Image &filter, size_t channels)
{
	double output = 0.0;
	for(size_t z = 0; z < channels; ++z)
		for(size_t y = 0; y < filter.height; ++y)
			for(size_t x = 0; x < filter.width; ++x)
				output += in.read(x, y, z) * filter.read(x, y, z);
	return output;
}

void GBlockConvolutional2D::addScaled(const Image &in, double scalar, Image &out)
{
	for(size_t y = 0; y < out.height; ++y)
		for(size_t x = 0; x < out.width; ++x)
			out.at(x, y) += in.read(x, y) * scalar;
}

void GBlockConvolutional2D::convolve(const Image &in, const Image &filter, Image &out, size_t channels)
{
	size_t x, y;
	if(channels == none)
		channels = filter.channels;
	for(y = 0, in.dy = out.py; y < out.height; ++y, ++in.dy)
		for(x = 0, in.dx = out.px; x < out.width; ++x, ++in.dx)
			out.at(in.dx, in.dy, 0) += filterSum(in, filter, channels);
	in.dx = in.dy = 0;
}

void GBlockConvolutional2D::convolveFull(const Image &in, const Image &filter, Image &out, size_t channels)
{
	size_t px = in.px, py = in.py;
	in.px = (in.px + filter.width - 1) / in.sx, in.py = (in.py + filter.height - 1) / in.sy;
	convolve(in, filter, out, channels);
	in.px = px, in.py = py;
}

void GBlockConvolutional2D::updateOutputSize()
{
	m_outputWidth = (m_width - m_kWidth + 2 * m_inputImage.px) / m_inputImage.sx + 1;
	m_outputHeight = (m_height - m_kHeight + 2 * m_inputImage.py) / m_inputImage.sy + 1;
	m_activation.resize(2, m_outputWidth * m_outputHeight * m_kernels.rows());

	m_actImage.data = &m_activation[0];
	m_actImage.width = m_outputWidth;
	m_actImage.height = m_outputHeight;

	m_errImage.data = &m_activation[1];
	m_errImage.width = m_outputWidth;
	m_errImage.height = m_outputHeight;
}
*/




} // namespace GClasses
