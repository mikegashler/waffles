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
#include "GMath.h"
#include "GDistribution.h"
#include "GError.h"
#include "GRand.h"
#include "GVec.h"
#include "GDom.h"
#include "GHillClimber.h"
#include "GTransform.h"
#include "GSparseMatrix.h"
#include "GDistance.h"
#include "GAssignment.h"
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
: inputCount(inputs), outputCount(outputs), m_inPos(0)
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

GBlock::GBlock(const GDomNode* pNode)
{
	m_inPos = pNode->getInt("inpos");
	inputCount = pNode->getInt("in");
	outputCount = pNode->getInt("out");
}

GBlock::~GBlock()
{
}

GDomNode* GBlock::baseDomNode(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "type", name().c_str());
	pNode->add(pDoc, "inpos", m_inPos);
	pNode->add(pDoc, "in", inputCount);
	pNode->add(pDoc, "out", outputCount);
	return pNode;
}

GDomNode* GBlock::serialize(GDom* pDoc) const
{
	return baseDomNode(pDoc);
}

GBlock* GBlock::deserialize(GDomNode* pNode, GRand& rand)
{
	const char* type = pNode->getString("type");
	if(strcmp(type, "GBlockLinear") == 0) return new GBlockLinear(pNode);
	else if(strcmp(type, "GBlockConv") == 0) return new GBlockConv(pNode);
	else if(strcmp(type, "GBlockEl") == 0) return new GBlockEl(pNode);
	else if(strcmp(type, "GBlockTanh") == 0) return new GBlockTanh(pNode);
	else if(strcmp(type, "GBlockLogistic") == 0) return new GBlockLogistic(pNode);
	else if(strcmp(type, "GBlockLeakyRectifier") == 0) return new GBlockLeakyRectifier(pNode);
	else if(strcmp(type, "GBlockRectifier") == 0) return new GBlockRectifier(pNode);
	else if(strcmp(type, "GBlockIdentity") == 0) return new GBlockIdentity(pNode);
	else if(strcmp(type, "GBlockScaledTanh") == 0) return new GBlockScaledTanh(pNode);
	else if(strcmp(type, "GBlockBentIdentity") == 0) return new GBlockBentIdentity(pNode);
	else if(strcmp(type, "GBlockSigExp") == 0) return new GBlockSigExp(pNode);
	else if(strcmp(type, "GBlockGaussian") == 0) return new GBlockGaussian(pNode);
	else if(strcmp(type, "GBlockSine") == 0) return new GBlockSine(pNode);
	else if(strcmp(type, "GBlockSoftPlus") == 0) return new GBlockSoftPlus(pNode);
	else if(strcmp(type, "GBlockSoftRoot") == 0) return new GBlockSoftRoot(pNode);
	else if(strcmp(type, "GBlockSoftMax") == 0) return new GBlockSoftMax(pNode);
	else if(strcmp(type, "GBlockSpectral") == 0) return new GBlockSpectral(pNode);
	else if(strcmp(type, "GBlockTemperedLinear") == 0) return new GBlockTemperedLinear(pNode);
	else if(strcmp(type, "GBlockMaxPooling2D") == 0) return new GBlockMaxPooling2D(pNode);
	else if(strcmp(type, "GBlockScalarSum") == 0) return new GBlockScalarSum(pNode);
	else if(strcmp(type, "GBlockScalarProduct") == 0) return new GBlockScalarProduct(pNode);
	else if(strcmp(type, "GBlockBiasProduct") == 0) return new GBlockBiasProduct(pNode);
	else if(strcmp(type, "GBlockSwitch") == 0) return new GBlockSwitch(pNode);
	else if(strcmp(type, "GBlockHinge") == 0) return new GBlockHinge(pNode);
	else if(strcmp(type, "GBlockSoftExp") == 0) return new GBlockSoftExp(pNode);
	else if(strcmp(type, "GBlockPAL") == 0) return new GBlockPAL(pNode, rand);
	else if(strcmp(type, "GBlockLSTM") == 0) return new GBlockLSTM(pNode);
	else throw Ex("Unrecognized neural network block type: ", type);
}

void GBlock::bind(const GVec* pInput, GVec* pOutput, GVec* pOutBlame, GVec* pInBlame, GVec* pWeights, GVec* pGradient)
{
	if(pInput)
		input.setData(*pInput);
	if(pOutput)
	{
		if(pOutput != &outputBuf)
			outputBuf.resize(0);
		output.setData(*pOutput);
	}
	else
	{
		outputBuf.resize(outputs());
		output.setData(outputBuf);
	}
	if(pOutBlame)
	{
		if(pOutBlame != &outBlameBuf)
			outBlameBuf.resize(0);
		outBlame.setData(*pOutBlame);
	}
	else
	{
		outBlameBuf.resize(outputs());
		outBlame.setData(outBlameBuf);
	}
	if(pInBlame)
		inBlame.setData(*pInBlame);
	if(pWeights)
	{
		if(pWeights != &weightsBuf)
			weightsBuf.resize(0);
		weights.setData(*pWeights);
	}
	else
	{
		weightsBuf.resize(weightCount());
		weights.setData(weightsBuf);
	}
	if(pGradient)
	{
		if(pGradient != &gradientBuf)
			gradientBuf.resize(0);
		gradient.setData(*pGradient);
	}
	else
	{
		gradientBuf.resize(gradCount());
		gradient.setData(gradientBuf);
	}
}

void GBlock::setInPos(size_t n)
{
	m_inPos = n;
}

double GBlock::computeBlame(const GVec& target)
{
	GAssert(target.size() == outBlame.size());
	double sse = 0.0;
	for(size_t i = 0; i < target.size(); i++)
	{
		if(target[i] == UNKNOWN_REAL_VALUE)
			outBlame[i] = 0.0;
		else
		{
			outBlame[i] = target[i] - output[i];
			sse += outBlame[i] * outBlame[i];
		}
	}
	return sse;
}

std::string GBlock::to_str(bool includeWeights, bool includeActivations) const
{
	std::ostringstream os;
	os << "[" << name() << ": ";
	os << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << ", WeightCount=" << GClasses::to_str(weightCount()) << "]";
	if(includeWeights)
		os << "\n(" << weights.to_str(',', 100) << ")";
	if(includeActivations)
	{
		os << "\n[In: " << input.to_str() << "]";
		os << "\n<InBlame: " << inBlame.to_str() << ">";
		os << "\n[Out: " << output.to_str() << "]";
		os << "\n<OutBlame: " << outBlame.to_str() << ">\n";
	}
	if(includeWeights || includeActivations)
		os << "\n";
	return os.str();
}

void GBlock::init_identity(GVec& weights)
{
	if(weights.size() > 0)
		throw Ex("Sorry, this block does not support the init_identity method");
}

bool GBlock::is_identity(GVec& weights)
{
	return false;
}

size_t GBlock::adjustedWeightCount(int inAdjustment, int outAdjustment)
{
	if(weightCount() == 0)
		return 0;
	throw Ex("Sorry, this block does not support the adjustedWeightCount method");
}

void GBlock::addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand)
{
	if(weightsBef.size() != 0)
		throw Ex("Sorry, this block does not support addUnits");
	inputCount += newInputs;
	outputCount += newOutputs;
}

void GBlock::dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft)
{
	if(weightsBef.size() != 0)
		throw Ex("Sorry, this block does not support addUnit");
	if(input != INVALID_INDEX)
		inputCount--;
	if(output != INVALID_INDEX)
		outputCount--;
}

size_t GBlock::firstIgnoredInput(const GVec& weights)
{
	throw Ex("Sorry, this block does not support firstIgnoredInput");
}

void GBlock::biasMask(GVec& mask)
{
	mask.fill(0.0);
}

void GBlock::step(double learningRate, double momentum)
{
	// Classic Momentum
	GAssert(weights.size() == gradient.size());
	for(size_t i = 0; i < gradient.size(); i++)
	{
		weights[i] += learningRate * gradient[i];
		gradient[i] *= momentum;
	}

/*
	// Nesterov's Accelerated Gradient could be implemented here
	// (See "On the importance of initialization and momentum in deep learning" by Sutskever, Martens, Dahl, and Hinton.)
	// (See https://jlmelville.github.io/mize/nesterov.html)
*/
}

void GBlock::step_jitter(double learningRate, double momentum, double jitter, GRand& rand)
{
	GAssert(weights.size() == gradient.size());
	double dev = jitter * std::sqrt(gradient.squaredMagnitude() / gradient.size());
	for(size_t i = 0; i < gradient.size(); i++)
	{
		weights[i] += learningRate * (gradient[i] + dev * rand.normal());
		gradient[i] *= momentum;
	}
}

void GBlock::finiteDifferencingTest(double tolerance)
{
	// Store the worst offender
#ifdef _DEBUG
	size_t worst_i = INVALID_INDEX;
	size_t worst_j = INVALID_INDEX;
#endif
	bool worst_backprop = true;
	double worst_empirical = UNKNOWN_REAL_VALUE;
	double worst_computed = UNKNOWN_REAL_VALUE;
	double worst_required_tolerance = tolerance;

	GVec in(inputs());
	GVec inBl(inputs());
	bind(&in, nullptr, nullptr, &inBl, nullptr, nullptr);
	GRand rand(0);
	for(size_t i = 0; i < 20; i++)
	{
		// Pick some random initial values
		initWeights(rand);
		in.fillNormal(rand);

		for(size_t j = 0; j < outputs(); j++)
		{
			// Compute gradients
			inBl.fill(0.0);
			gradient.fill(0.0);
			forwardProp();
			outBlame.fill(0.0);
			outBlame[j] = 1.0;
			backProp();
			updateGradient();

			// Test gradient of inputs with central differencing
			double epsilon = 1e-8;
			for(size_t k = 0; k < inputs(); k++)
			{
				in[k] -= epsilon;
				forwardProp();
				double lesser = output[j];
				in[k] += epsilon;
				in[k] += epsilon;
				forwardProp();
				double greater = output[j];
				in[k] -= epsilon;
				double empirical_grad = (greater - lesser) / (epsilon + epsilon);
				//std::cout << GClasses::to_str(inBl[k]) << "	" << GClasses::to_str(empirical_grad) << "\n";
				if(empirical_grad == 0)
				{
					if(std::abs(inBl[k]) <= 1e-8)
					{}
					else
						throw Ex("Problem with backProp");
				}
				else if(std::abs((inBl[k] - empirical_grad) / empirical_grad) <= worst_required_tolerance)
				{} // Done this way so that NANs will fail rather than succeed
				else
				{
#ifdef _DEBUG
					worst_i = i;
					worst_j = j;
#endif
					worst_backprop = true;
					worst_empirical = empirical_grad;
					worst_computed = inBl[k];
					worst_required_tolerance = std::abs((inBl[k] - empirical_grad) / empirical_grad);
				}
			}

			// Test gradient of weights with central differencing
			for(size_t k = 0; k < weightCount(); k++)
			{
				weights[k] -= epsilon;
				forwardProp();
				double lesser = output[j];
				weights[k] += epsilon;
				weights[k] += epsilon;
				forwardProp();
				double greater = output[j];
				weights[k] -= epsilon;
				double empirical_grad = (greater - lesser) / (epsilon + epsilon);
				//std::cout << GClasses::to_str(gradient[k]) << "	" << GClasses::to_str(empirical_grad) << "\n";
				if(empirical_grad == 0)
				{
					if(std::abs(gradient[k]) <= 1e-8)
					{}
					else
						throw Ex("Problem with updateGradient");
				}
				else if(std::abs((gradient[k] - empirical_grad) / empirical_grad) <= worst_required_tolerance)
				{} // Done this way so that NANs will fail rather than succeed
				else
				{
#ifdef _DEBUG
					worst_i = i;
					worst_j = j;
#endif
					worst_backprop = false;
					worst_empirical = empirical_grad;
					worst_computed = gradient[k];
					worst_required_tolerance = std::abs((gradient[k] - empirical_grad) / empirical_grad);
				}
			}
		}
	}

	if(worst_required_tolerance > tolerance)
	{
#ifdef _DEBUG
		std::cerr << "i=" << GClasses::to_str(worst_i) << ", j=" << GClasses::to_str(worst_j) << "\n";
#endif
		throw Ex("Problem with ", (worst_backprop ? "backProp" : "updateGradient"), ". Expected about ", GClasses::to_str(worst_empirical), ". Got ", GClasses::to_str(worst_computed), ". This would require a tolerance of ", GClasses::to_str(worst_required_tolerance));
	}
}











GBlockActivation::GBlockActivation(size_t size)
: GBlockWeightless(size, size)
{
}

GBlockActivation::GBlockActivation(const GDomNode* pNode)
: GBlockWeightless(pNode)
{}

void GBlockActivation::forwardProp()
{
	for(size_t i = 0; i < inputCount; i++)
		output[i] = eval(input[i]);
}

void GBlockActivation::backProp()
{
	for(size_t i = 0; i < inputCount; i++)
		inBlame[i] += outBlame[i] * derivative(input[i], output[i]);
}

void GBlockActivation::inverseProp(const GVec& output, GVec& input)
{
	for(size_t i = 0; i < outputCount; i++)
		input[i] = inverse(output[i]);
}








void GBlockSoftMax::forwardProp()
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

double GBlockSoftMax::computeBlame(const GVec& target)
{
	double sse = 0.0;
	for(size_t i = 0; i < outputCount; i++)
	{
		GAssert(target[i] >= 0.0 && target[i] <= 1.0);
		outBlame[i] = target[i] - output[i];
		sse += outBlame[i] * outBlame[i];
	}
	return sse;
}

void GBlockSoftMax::backProp()
{
	for(size_t i = 0; i < outputCount; i++)
		inBlame[i] += outBlame[i];
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
		m_freq_shift = std::min(1.0, (freq_end - m_freq_start) / (pairs - 1));
	}
	else
	{
		m_freq_scale = pow(freq_end / m_freq_start, 1.0 / (pairs - 1));
		m_freq_shift = 0.0;
	}
}

GBlockSpectral::GBlockSpectral(const GDomNode* pNode)
: GBlockWeightless(pNode),
m_freq_start(pNode->getDouble("start")),
m_freq_scale(pNode->getDouble("scale")),
m_freq_shift(pNode->getDouble("shift"))
{
}

GDomNode* GBlockSpectral::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->add(pDoc, "start", m_freq_start);
	pNode->add(pDoc, "scale", m_freq_scale);
	pNode->add(pDoc, "shift", m_freq_shift);
	return pNode;
}

void GBlockSpectral::forwardProp()
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

void GBlockSpectral::backProp()
{
	double freq = 2.0 * M_PI * m_freq_start;
	size_t pairs = outputCount / 2;
	for(size_t i = 0; i < pairs; i++)
	{
		inBlame[0] += outBlame[2 * i] * std::cos(freq * input[0]);
		inBlame[0] -= outBlame[2 * i + 1] * std::sin(freq * input[0]);
	}
}









void GBlockRepeater::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
		output[i] = input[i % inputCount];
}

void GBlockRepeater::backProp()
{
	for(size_t i = 0; i < outputCount; i++)
		inBlame[i % inputCount] += outBlame[i];
}








GBlockSpreader::GBlockSpreader(size_t units, size_t spread)
: GBlockWeightless(units, units), m_spread(spread), m_forw(units), m_back(units)
{
	size_t j = 0;
	m_back.fill(INVALID_INDEX);
	for(size_t i = 0; i < units; i++)
	{
		if(j >= units)
		{
			j -= units;
			while(m_back[j] != INVALID_INDEX)
				j++;
		}
		m_forw[i] = j;
		m_back[j] = i;
		j += spread;
	}
}

GBlockSpreader::GBlockSpreader(const GBlockSpreader& that)
: GBlockSpreader(that.outputCount, that.m_spread)
{
}

GBlockSpreader::GBlockSpreader(const GDomNode* pNode)
: GBlockSpreader(pNode->getInt("out"), pNode->getInt("spread"))
{
}

void GBlockSpreader::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
		output[m_forw[i]] = input[i];
}

void GBlockSpreader::backProp()
{
	for(size_t i = 0; i < outputCount; i++)
		inBlame[i % inputCount] += outBlame[i];
}








GBlockLinear::GBlockLinear(size_t inputs, size_t outputs)
: GBlock(inputs, outputs)
{
}

GBlockLinear::GBlockLinear(GDomNode* pNode)
: GBlock(pNode)
{}

std::string GBlockLinear::to_str(bool includeWeights, bool includeActivations) const
{
	std::ostringstream os;
	os << "[" << name() << ": ";
	os << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << ", Weights=" << GClasses::to_str(weightCount()) << "]";
	if(includeWeights)
	{
		size_t colWid = 16;
		os << "\n";
		os << "    Bias: ";
		for(size_t i = 0; i < outputCount; i++)
		{
			if(i > 0)
				os << ",";
			os << to_fixed_str(weights[i], colWid - 1, ' ');
		}
		os << "\n";
		os << "          ";
		for(size_t i = 0; i < outputCount; i++)
		{
			std::string s = GClasses::to_str(i);
			os << "Out ";
			os << s;
			os << std::string(colWid - 4 - s.length(), ' ');
		}
		os << "\n";
		size_t pos = outputCount;
		for(size_t i = 0; i < inputCount; i++)
		{
			std::string ss = GClasses::to_str(i);
			os << std::string(5 - ss.length(), ' ');
			os << "In ";
			os << GClasses::to_str(ss);
			os << ": ";
			for(size_t j = 0; j < outputCount; j++)
			{
				if(j > 0)
					os << ",";
				os << to_fixed_str(weights[pos++], colWid - 1, ' ');
			}
			os << "\n";
		}
		os << "\n";
	}
	if(includeActivations)
	{
		os << "[In: " << input.to_str() << "]\n";
		os << "<InBlame: " << inBlame.to_str() << ">\n";
		os << "[Out: " << output.to_str() << "]\n";
		os << "<OutBlame: " << outBlame.to_str() << ">\n";
	}
	if(includeWeights || includeActivations)
		os << "\n";
	return os.str();
}

void GBlockLinear::forwardProp()
{
	// Start with the bias
	output.copy(0, weights, 0, outputCount);

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

void GBlockLinear::backProp()
{
	size_t pos = outputCount; // skip the bias weights
	for(size_t i = 0; i < inputCount; i++)
	{
		const GConstVecWrapper v(weights, pos, outputCount);
		inBlame[i] += outBlame.dotProduct(v);
		pos += outputCount;
	}
}

void GBlockLinear::updateGradient()
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

void GBlockLinear::updateGradientNormalized()
{
	size_t pos = 0;
	for(size_t j = 0; j < outputCount; j++)
		gradient[pos++] += outBlame[j];
	for(size_t i = 0; i < inputCount; i++)
	{
		double act = input[i];
		for(size_t j = 0; j < outputCount; j++)
			gradient[pos++] += outBlame[j] * (std::signbit(act) ? -1.0 : 1.0);
	}
}

size_t GBlockLinear::weightCount() const
{
	return (inputCount + 1) * outputCount;
}

void GBlockLinear::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / std::max((size_t)1, inputCount));
}

void GBlockLinear::ordinaryLeastSquares(const GMatrix& features, const GMatrix& labels, GVec& outWeights)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");

	// Compute centroids
	GVec centroidFeat(features.cols());
	features.centroid(centroidFeat);
	GVec centroidLab(labels.cols());
	labels.centroid(centroidLab);

	// Compute numerator
	GMatrix numerator(labels.cols(), features.cols());
	numerator.fill(0.0);
	for(size_t i = 0; i < features.rows(); i++)
	{
		const GVec& f = features[i];
		const GVec& l = labels[i];
		for(size_t j = 0; j < labels.cols(); j++)
		{
			for(size_t k = 0; k < features.cols(); k++)
				numerator[j][k] += (l[j] - centroidLab[j]) * (f[k] - centroidFeat[k]);
		}
	}

	// Compute denominator
	GMatrix denomPre(features.cols(), features.cols());
	denomPre.fill(0.0);
	for(size_t i = 0; i < features.rows(); i++)
	{
		const GVec& f = features[i];
		for(size_t j = 0; j < features.cols(); j++)
		{
			for(size_t k = 0; k < features.cols(); k++)
				denomPre[j][k] += (f[j] - centroidFeat[j]) * (f[k] - centroidFeat[k]);
		}
	}
	GMatrix* pDenom = denomPre.pseudoInverse();
	Holder<GMatrix> hDenom(pDenom);

	// Compute M and b
	GMatrix* pM = GMatrix::multiply(numerator, *pDenom, false, true/*doesn't matter. it's symmetric about transposition*/);
	Holder<GMatrix> hM(pM);
	GVec b(labels.cols());
	pM->multiply(centroidFeat, b);
	b *= (-1);
	b += centroidLab;

	// Unpack into weights
	size_t pos = 0;
	outWeights.copy(pos, b);
	pos += b.size();
	for(size_t i = 0; i < pM->cols(); i++)
	{
		for(size_t j = 0; j < pM->rows(); j++)
			outWeights[pos++] = (*pM)[j][i];
	}
	GAssert(pos == outWeights.size());
}

void GBlockLinear::adjustInputRange(GVec& weights, size_t inputIndex, double oldMin, double oldMax, double newMin, double newMax)
{
	double scalar = (oldMax - oldMin) / (newMax - newMin);
	for(size_t i = 0; i < outputCount; i++)
	{
		double oldWeight = weights[outputCount + outputCount * inputIndex + i];
		double newWeight = oldWeight * scalar;
		weights[outputCount + outputCount * inputIndex + i] = newWeight;
		weights[i] += (oldMin * oldWeight - newMin * newWeight);
	}
}

void GBlockLinear::init_identity(GVec& weights)
{
	weights.fill(0.0, 0, outputCount);
	size_t pos = outputCount;
	for(size_t i = 0; i < inputCount; i++)
	{
		for(size_t j = 0; j < outputCount; j++)
			weights[pos++] = (i == j ? 1.0 : 0.0);
	}
}

void GBlockLinear::biasMask(GVec& mask)
{
	mask.fill(1.0, 0, outputCount);
	mask.fill(0.0, outputCount, mask.size() - outputCount);
}

size_t GBlockLinear::adjustedWeightCount(int inAdjustment, int outAdjustment)
{
	return (inputCount + inAdjustment + 1) * (outputCount + outAdjustment);
}

void GBlockLinear::addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand)
{
	// Bias
	weightsAft.copy(0, weightsBef, 0, outputCount); // Preserve existing biases
	size_t inpos = outputCount;
	size_t outpos = outputCount;
	for(size_t i = 0; i < newOutputs; i++)
		weightsAft[outpos++] = 0.0; // The new biases are zero

	// weights
	for(size_t i = 0; i < inputCount; i++)
	{
		for(size_t j = 0; j < outputCount; j++)
			weightsAft[outpos++] = weightsBef[inpos++]; // Preserve existing connections
		for(size_t j = 0; j < newOutputs; j++)
			weightsAft[outpos++] = rand.normal() * 1.0 / std::max((size_t)1, inputCount + newInputs); // To new units
	}
	for(size_t i = 0; i < newInputs; i++)
	{
		for(size_t j = 0; j < outputCount; j++)
			weightsAft[outpos++] = 0.0; // Don't change the existing outputs
		for(size_t j = 0; j < newOutputs; j++)
			weightsAft[outpos++] = rand.normal() * 1.0 / std::max((size_t)1, inputCount + newInputs); // It's okay to affect new outputs
	}
	if(inpos != weightsBef.size() || outpos != weightsAft.size())
		throw Ex("Weights size problem");

	// The block
	inputCount += newInputs;
	outputCount += newOutputs;
}

void GBlockLinear::dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft)
{
	// Bias
	size_t inpos = 0;
	size_t outpos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		if(i != output)
			weightsAft[outpos++] = weightsBef[inpos];
		inpos++;
	}

	// Weights
	for(size_t i = 0; i < inputCount; i++)
	{
		if(i == input)
			inpos += outputCount;
		else
		{
			for(size_t j = 0; j < outputCount; j++)
			{
				if(j != output)
					weightsAft[outpos++] = weightsBef[inpos];
				inpos++;
			}
		}
	}
	if(inpos != weightsBef.size() || outpos != weightsAft.size())
		throw Ex("Weights size problem");

	// The block
	if(input != INVALID_INDEX)
		inputCount--;
	if(output != INVALID_INDEX)
		outputCount--;
}

size_t GBlockLinear::firstIgnoredInput(const GVec& weights)
{
	GConstVecWrapper vw;
	for(size_t i = 0; i < inputCount; i++)
	{
		vw.setData(weights, (i + 1) * outputCount, outputCount);
		if(vw.squaredMagnitude() == 0.0)
			return i;
	}
	return INVALID_INDEX;
}

/*static*/
void GBlockLinear::fuseLayers(GBlockLinear& aa, GVec& a, GBlockLinear& bb, GVec& b, bool targetA)
{
	if(aa.outputs() != bb.inputs())
		throw Ex("Incompatible layers");
	if(a.size() != (aa.inputs() + 1) * aa.outputs() || b.size() != (bb.inputs() + 1) * bb.outputs())
		throw Ex("Unexpected weight sizes");
	if(targetA)
	{
		if(bb.inputs() != bb.outputs())
			throw Ex("The non-targeted layer must be square");
	}
	else
	{
		if(aa.inputs() != aa.outputs())
			throw Ex("The non-targeted layer must be square");
	}
	GVec target(targetA ? a.size() : b.size());
	size_t ain = aa.inputs();
	size_t aout = aa.outputs();
	size_t bout = bb.outputs();
	size_t tout = targetA ? aout : bout;
	for(size_t i = 0; i < ain; i++)
	{
		for(size_t j = 0; j < bout; j++)
		{
			double d = 0.0;
			for(size_t k = 0; k < aa.outputs(); k++)
				d += a[aout + i * aout + k] * b[bout + k * bout + j];
			target[tout + i * tout + j] = d;
		}
	}

	// Do the biases
	for(size_t j = 0; j < bout; j++)
	{
		double d = 0.0;
		for(size_t k = 0; k < aa.outputs(); k++)
			d += a[k] * b[bout + k * bout + j];
		target[j] = d;
	}
	if(targetA)
		a.copy(target);
	else
		b.copy(target);
}

void GBlockLinear::regularize_square(double lambda, const GVec* pWeightsNext)
{
	if(inputCount != outputCount)
		throw Ex("This method only works with square layers");
	if(weights.size() != weightCount())
		throw Ex("Unexpected number of weights");
	if(pWeightsNext && pWeightsNext->size() != weights.size())
		throw Ex("The next weights should only be supplied if they have the same size");

	// Push toward anti-symmetric weights
	for(size_t i = 0; i < outputCount; i++)
	{
		for(size_t j = i + 1; j < outputCount; j++)
		{
			size_t ii = outputCount * (i + 1) + j;
			size_t jj = outputCount * (j + 1) + i;
			double iVal = weights[ii];
			double jVal = weights[jj];
			weights[ii] = (1.0 - lambda) * iVal - lambda * jVal;
			weights[jj] = (1.0 - lambda) * jVal - lambda * iVal;
		}
	}

	// Push toward the weights in the next layer
	if(pWeightsNext)
	{
		for(size_t i = outputCount; i < weights.size(); i++)
			weights[i] = (1.0 - lambda) * weights[i] + lambda * (*pWeightsNext)[i];
	}
}









GBlockFanOut::GBlockFanOut(size_t inputs, size_t outputsPerInput)
: GBlock(inputs, outputsPerInput * inputs)
{
}

GBlockFanOut::GBlockFanOut(GDomNode* pNode)
: GBlock(pNode)
{}

void GBlockFanOut::forwardProp()
{
	size_t outputsPerInput = outputCount / inputCount;
	size_t outpos = 0;
	size_t weightpos = 0;
	for(size_t j = 0; j < outputsPerInput; j++)
	{
		for(size_t i = 0; i < inputCount; i++)
		{
			if(input[i] == UNKNOWN_REAL_VALUE)
				output[outpos] = UNKNOWN_REAL_VALUE;
			else
				output[outpos] = weights[weightpos] + weights[weightpos + 1] * input[i];
			++outpos;
			weightpos += 2;
		}
	}
	GAssert(outpos == output.size());
	GAssert(weightpos == weights.size());
}

void GBlockFanOut::backProp()
{
	size_t outputsPerInput = outputCount / inputCount;
	size_t outpos = 0;
	size_t weightpos = 0;
	for(size_t j = 0; j < outputsPerInput; j++)
	{
		for(size_t i = 0; i < inputCount; i++)
		{
			++weightpos; // skip the bias
			if(input[i] != UNKNOWN_REAL_VALUE)
				inBlame[i] += outBlame[outpos] * weights[weightpos];
			++outpos;
			++weightpos;
		}
	}
	GAssert(outpos == output.size());
	GAssert(weightpos == weights.size());
}

void GBlockFanOut::updateGradient()
{
	size_t outputsPerInput = outputCount / inputCount;
	size_t outpos = 0;
	size_t weightpos = 0;
	for(size_t j = 0; j < outputsPerInput; j++)
	{
		for(size_t i = 0; i < inputCount; i++)
		{
			if(input[i] == UNKNOWN_REAL_VALUE)
			{
				++weightpos;
				++weightpos;
				++outpos;
			}
			else
			{
				gradient[weightpos++] += outBlame[outpos];
				gradient[weightpos++] += outBlame[outpos++] * input[i];
			}
		}
	}
	GAssert(outpos == output.size());
	GAssert(weightpos == weights.size());
}

size_t GBlockFanOut::weightCount() const
{
	return 2 * outputCount;
}

void GBlockFanOut::initWeights(GRand& rand)
{
	weights.fillNormal(rand);
}










GBlockRunningNormalizer::GBlockRunningNormalizer(size_t units, double effective_batch_size)
: GBlock(units, units), batch_size(effective_batch_size), inv_bs(1.0 / batch_size), decay_scalar(1.0 - inv_bs), epsilon(1e-8)
{
}

GBlockRunningNormalizer::GBlockRunningNormalizer(GDomNode* pNode)
: GBlock(pNode)
{}

void GBlockRunningNormalizer::forwardProp()
{
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		double running_mean = weights[pos++] * inv_bs;
		double running_var = weights[pos++] * inv_bs - (running_mean * running_mean);
		double x_hat = (input[i] - running_mean) / std::sqrt(running_var + epsilon);
		double gamma = weights[pos++];
		double beta = weights[pos++];
		output[i] = gamma * x_hat + beta;
	}
	GAssert(pos == outputCount * 4);
}

void GBlockRunningNormalizer::backProp()
{
/*
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		double running_mean = weights[pos++] * inv_bs;
		double running_var = weights[pos++] * inv_bs - (running_mean * running_mean);
		double gamma = weights[pos++];
		pos++;
		var_blame += inBlame[i] * gamma * (input[i] - running_mean) * (-0.5) * std::pow(running_var + epsilon, -1.5);
	}
*/


	size_t pos = 0;
	double var_blame = 0.0;
	for(size_t i = 0; i < outputCount; i++)
	{
		double running_mean = weights[pos++] * inv_bs;
		double running_var = weights[pos++] * inv_bs - (running_mean * running_mean);
		double gamma = weights[pos++];
		pos++;
		var_blame += outBlame[i] * gamma * (input[i] - running_mean) * (-0.5) * std::pow(running_var + epsilon, -1.5);
	}
	pos = 0;
	double a = 0.0;
	double b = 0.0;
	for(size_t i = 0; i < outputCount; i++)
	{
		double running_mean = weights[pos++] * inv_bs;
		double running_var = weights[pos++] * inv_bs - (running_mean * running_mean);
		double gamma = weights[pos++];
		pos++;
		a += outBlame[i] * gamma * (-1.0 / std::sqrt(running_var + epsilon));
		b += (input[i] - running_mean);
	}
	pos = 0;
	double mean_blame = a + var_blame * (-2.0 * b / outputCount);
	for(size_t i = 0; i < outputCount; i++)
	{
		double running_mean = weights[pos++] * inv_bs;
		double running_var = weights[pos++] * inv_bs - (running_mean * running_mean);
		double gamma = weights[pos++];
		pos++;
		inBlame[i] = outBlame[i] * gamma / std::sqrt(running_var + epsilon) + (var_blame * 2.0 * (input[i] - running_mean) + mean_blame) / outputCount;
	}
}

void GBlockRunningNormalizer::updateGradient()
{
	if(gradient.size() != outputCount)
		throw Ex("gradient has unexpected size");
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
		gradient[pos++] += outBlame[i];
}

void GBlockRunningNormalizer::step(double learningRate, double momentum)
{
	GAssert(gradient.size() == outputCount);
	GAssert(weights.size() == 4 * outputCount);
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		double running_mean = weights[pos] * inv_bs;
		weights[pos] *= decay_scalar;
		weights[pos++] += input[i];
		double running_var = weights[pos] * inv_bs - (running_mean * running_mean);
		double x_hat = (input[i] - running_mean) / std::sqrt(running_var + epsilon);
		weights[pos] *= decay_scalar;
		weights[pos++] += (input[i] * input[i]);
		weights[pos++] += outBlame[i] * x_hat;
		weights[pos++] += outBlame[i];
	}
}

size_t GBlockRunningNormalizer::weightCount() const
{
	return outputCount * 4;
}

size_t GBlockRunningNormalizer::gradCount() const
{
	return outputCount;
}
void GBlockRunningNormalizer::initWeights(GRand& rand)
{
	GAssert(weights.size() == 4 * outputCount);
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[pos++] = 0.0;
		weights[pos++] = batch_size;
		weights[pos++] = 1.0;
		weights[pos++] = 0.0;
	}
}









GBlockWeightDigester::GBlockWeightDigester(GBlock* pDigestMyWeights, size_t extraInputs, size_t outputs)
: GBlock(extraInputs, outputs), m_pDigestMyWeights(pDigestMyWeights)
{
}

GBlockWeightDigester::GBlockWeightDigester(GDomNode* pNode)
: GBlock(pNode)
{
	throw Ex("Sorry, not implemented yet");
}

void GBlockWeightDigester::forwardProp()
{
	// Start with the bias
	output.copy(0, weights, 0, outputCount);

	// Weights from m_pDigestMyWeights
	GVec& wi = m_pDigestMyWeights->weights;
	size_t pos = outputCount;
	for(size_t i = 0; i < wi.size(); i++)
	{
		output.addScaled(wi[i], weights, pos, outputCount);
		pos += outputCount;
	}

	// Extra inputs
	for(size_t i = 0; i < inputCount; i++)
	{
		output.addScaled(input[i], weights, pos, outputCount);
		pos += outputCount;
	}
}

void GBlockWeightDigester::backProp()
{
	// Skip the bias weights
	size_t pos = outputCount;

	// Skip the incoming weights. We'll backpropagate the blame for these in updateGradient.
	pos += (outputCount * m_pDigestMyWeights->weights.size());

	// Extra inputs
	for(size_t i = 0; i < inputCount; i++)
	{
		const GConstVecWrapper v(weights, pos, outputCount);
		inBlame[i] += outBlame.dotProduct(v);
		pos += outputCount;
	}
}

void GBlockWeightDigester::updateGradient()
{
	// Bias
	size_t pos = 0;
	for(size_t j = 0; j < outputCount; j++)
		gradient[pos++] += outBlame[j];

	// Update gradient for the weights from m_pDigestMyWeights
	GVec& wi = m_pDigestMyWeights->weights;
	GVec& wiGrad = m_pDigestMyWeights->gradient;
	size_t ww = outputCount;
	for(size_t i = 0; i < wi.size(); i++)
	{
		const GConstVecWrapper v(weights, ww, outputCount);
		wiGrad[i] += outBlame.dotProduct(v);
		ww += outputCount;
	}

	// Update the gradient for the weights of this layer
	for(size_t i = 0; i < wi.size(); i++)
	{
		double act = wi[i];
		for(size_t j = 0; j < outputCount; j++)
			gradient[pos++] += outBlame[j] * act;
	}

	// Extra inputs
	for(size_t i = 0; i < inputCount; i++)
	{
		double act = input[i];
		for(size_t j = 0; j < outputCount; j++)
			gradient[pos++] += outBlame[j] * act;
	}
}

size_t GBlockWeightDigester::weightCount() const
{
	return (1 + m_pDigestMyWeights->weightCount() + inputCount) * outputCount;
}

void GBlockWeightDigester::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / std::max((size_t)1, m_pDigestMyWeights->weightCount() + inputCount));
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
: GBlock(pNode), deviationCap(pNode->getDouble("dc")), forgetRate(pNode->getDouble("fr")),
moment1(pNode->get("mom1")), moment2(pNode->get("mom2"))
{
}

void GBlockTemperedLinear::forwardProp()
{
	// Start with the bias
	output.copy(0, weights, 0, outputCount);

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

void GBlockTemperedLinear::backProp()
{
	size_t pos = outputCount; // skip the bias weights
	for(size_t i = 0; i < inputCount; i++)
	{
		const GConstVecWrapper v(weights, pos, outputCount);
		inBlame[i] += outBlame.dotProduct(v);
		pos += outputCount;
	}
}

void GBlockTemperedLinear::updateGradient()
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

void GBlockTemperedLinear::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / std::max((size_t)1, inputCount));
}










size_t GBlockConv_countElements(const std::initializer_list<size_t>& shape)
{
	size_t n = 1;
	for(const size_t* it = begin(shape); it != end(shape); ++it)
		n *= *it;
	return n;
}

GBlockConv::GBlockConv(const std::initializer_list<size_t>& inputShape, const std::initializer_list<size_t>& filterShape, const std::initializer_list<size_t>& outputShape)
: GBlock(GBlockConv_countElements(inputShape), GBlockConv_countElements(outputShape)),
filterSize(GBlockConv_countElements(filterShape)),
tensorInput(inputShape, false),
tensorFilter(filterShape, false),
tensorOutput(outputShape, false)
{
	// Iterate over all of the shape values
	filterCount = 1;
	outputsPerFilter = 1;
	const size_t* itIn = begin(inputShape);
	const size_t* itFilt = begin(filterShape);
	const size_t* itOut = begin(outputShape);
	while(true)
	{
		if(itIn != end(inputShape)) // If there are more input dimensions...
		{
			++itIn;
			++itFilt;
			if(itOut != end(outputShape)) // If there are more output dimensions...
			{
				outputsPerFilter *= *itOut;
				++itOut;
			}
		}
		else
		{
			if(itFilt == end(filterShape)) // If there are more filter dimensions...
				break;
			filterCount *= *itFilt;
			++itFilt;
		}
	}
	filterSize /= filterCount;
	if(filterCount * outputsPerFilter != outputCount)
		throw Ex("output dimensions do not work with input and filter dimensions");
	while(tensorFilter.shape.size() > tensorInput.shape.size())
		tensorFilter.shape.erase(tensorFilter.shape.size() - 1);
	while(tensorOutput.shape.size() > tensorInput.shape.size())
		tensorOutput.shape.erase(tensorOutput.shape.size() - 1);
	if(tensorOutput.shape.size() < tensorFilter.shape.size())
	{
		size_t old = tensorOutput.shape.size();
		tensorOutput.shape.append(tensorFilter.shape.size() - tensorOutput.shape.size());
		for(size_t i = old; i < tensorFilter.shape.size(); i++)
			tensorOutput.shape[i] = 1;
	}
}

GBlockConv::GBlockConv(const GBlockConv& that)
: GBlock(that),
filterSize(that.filterSize),
tensorInput(that.tensorInput),
tensorFilter(that.tensorFilter),
tensorOutput(that.tensorOutput),
filterCount(that.filterCount),
outputsPerFilter(that.outputsPerFilter)
{
}

size_t GBlockConv_countTensorSize(const GIndexVec& shape)
{
	size_t n = 1;
	for(size_t i = 0; i < shape.size(); i++)
		n *= shape[i];
	return n;
}

GBlockConv::GBlockConv(GDomNode* pNode)
: GBlock(pNode),
tensorInput(pNode->get("inputShape")),
tensorFilter(pNode->get("filterShape")),
tensorOutput(pNode->get("outputShape")),
filterCount(pNode->getInt("filterCount")),
outputsPerFilter(outputCount / filterCount)
{
	filterSize = GBlockConv_countTensorSize(tensorFilter.shape) / filterCount;
	if(filterCount * outputsPerFilter != outputCount)
		throw Ex("output dimensions do not work with input and filter dimensions");
}

GDomNode* GBlockConv::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->add(pDoc, "filterCount", filterCount);
	pNode->add(pDoc, "inputShape", tensorInput.shape.serialize(pDoc));
	pNode->add(pDoc, "filterShape", tensorFilter.shape.serialize(pDoc));
	pNode->add(pDoc, "outputShape", tensorOutput.shape.serialize(pDoc));
	return pNode;
}

void GBlockConv::forwardProp()
{
	tensorInput.setData(input);
	size_t weightsPos = 0;
	size_t outPos = 0;
	for(size_t i = 0; i < filterCount; i++)
	{
		tensorOutput.setData(output, outPos, outputsPerFilter);
		tensorOutput.fill(weights[weightsPos++]); // initialize with the bias value
		tensorFilter.setData(*(GVec*)&weights, weightsPos, filterSize);
		GTensor::convolve(tensorInput, tensorFilter, tensorOutput, false, 1);
		weightsPos += filterSize;
		outPos += outputsPerFilter;
	}
	if(weightsPos != weights.size())
		throw Ex("Expected ", GClasses::to_str(weightsPos), " weights. Got ", GClasses::to_str(weights.size()));
}

void GBlockConv::backProp()
{
	tensorInput.setData(inBlame);
	size_t weightsPos = 0;
	size_t outPos = 0;
	for(size_t i = 0; i < filterCount; i++)
	{
		tensorOutput.setData(outBlame, outPos, outputsPerFilter);
		weightsPos++; // skip the bias
		tensorFilter.setData(*(GVec*)&weights, weightsPos, filterSize);
		GTensor::convolve(tensorFilter, tensorOutput, tensorInput, true, 1);
		weightsPos += filterSize;
		outPos += outputsPerFilter;
	}
	GAssert(weightsPos == weights.size());
}

void GBlockConv::updateGradient()
{
	tensorInput.setData(input);
	size_t gradPos = 0;
	size_t outPos = 0;
	for(size_t i = 0; i < filterCount; i++)
	{
		tensorOutput.setData(outBlame, outPos, outputsPerFilter);
		gradient[gradPos++] += tensorOutput.sum(); // the bias gradient
		tensorFilter.setData(gradient, gradPos, filterSize);
		GTensor::convolve(tensorInput, tensorOutput, tensorFilter, false, 1);
		gradPos += filterSize;
		outPos += outputsPerFilter;
	}
	GAssert(gradPos == weights.size());
}

size_t GBlockConv::weightCount() const
{
	return filterCount * (filterSize + 1);
}

void GBlockConv::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / filterSize);
}

void GBlockConv::test()
{
	GNeuralNet nn;
	nn.add(new GBlockConv({4}, {3}, {4}));
	if(nn.weightCount() != 4)
		throw Ex("Unexpected number of weights");
	GVec w({0, 1, 2, 0}); // bias=0, filter={1,2,0}
	nn.init(w);

	// Test forwardprop
	GVec x({2, 1, 0, 3});
	const GVec& pred = nn.forwardProp(x);
	GVec expected({4, 4, 1, 6});
	if(std::sqrt(pred.squaredDistance(expected)) > 1e-10)
		throw Ex("wrong");
/*
	// Test backprop
	GVec y({4, 5, 3, 3});
	nn.computeBlame(y);
	GVec inBlame(4); inBlame.fill(0.0);
	nn.backpropagate(w, &inBlame);
	if(std::abs(1.0 - inBlame[0]) > 1e-8) throw Ex("wrong");
	if(std::abs(4.0 - inBlame[1]) > 1e-8) throw Ex("wrong");
	if(std::abs(4.0 - inBlame[2]) > 1e-8) throw Ex("wrong");
	if(std::abs(0.0 - inBlame[3]) > 1e-8) throw Ex("wrong");

	// Test update
	GVec grad(4); grad.fill(0.0);
	nn.updateGradient(w, grad);
	if(std::abs(3.0 - grad[0]) > 1e-8) throw Ex("wrong");
	if(std::abs(4.0 - grad[1]) > 1e-8) throw Ex("wrong");
	if(std::abs(1.0 - grad[2]) > 1e-8) throw Ex("wrong");
	if(std::abs(6.0 - grad[3]) > 1e-8) throw Ex("wrong");
*/
}






GBlockMaxPooling2D::GBlockMaxPooling2D(size_t _width, size_t _height, size_t _channels)
: GBlockWeightless(_width * _height * _channels, _width * _height / 4 * _channels),
width(_width),
height(_height),
channels(_channels)
{
	if((width % 1) || (height % 1))
		throw Ex("Expected an even width and height");
}

GBlockMaxPooling2D::GBlockMaxPooling2D(GDomNode* pNode)
: GBlockWeightless(pNode)
{
}

GBlockMaxPooling2D::~GBlockMaxPooling2D()
{
}

// virtual
void GBlockMaxPooling2D::forwardProp()
{
	size_t chanSize = width * height;
	size_t chanStart = 0;
	size_t pos = 0;
	for(size_t c = 0; c < channels; c++)
	{
		size_t vertStart = chanStart;
		for(size_t y = 0; y < height; y += 2)
		{
			for(size_t x = 0; x < width; x += 2)
			{
				output[pos++] = std::max(
						std::max(input[vertStart + x], input[vertStart + 1 + x]),
						std::max(input[vertStart + width + x], input[vertStart + width + 1 + x])
					);
			}
			vertStart += (width + width);
		}
		chanStart += chanSize;
	}
}

void GBlockMaxPooling2D::backProp()
{
	size_t chanSize = width * height;
	size_t chanStart = 0;
	size_t pos = 0;
	for(size_t c = 0; c < channels; c++)
	{
		size_t vertStart = chanStart;
		for(size_t y = 0; y < height; y += 2)
		{
			for(size_t x = 0; x < width; x += 2)
			{
				size_t i = vertStart + x;
				double d = input[i];
				double cand = input[i + 1];
				if(cand > d)
				{
					d = cand;
					i++;
				}
				cand = input[vertStart + width + x];
				if(cand > d)
				{
					d = cand;
					i = vertStart + width + x;
				}
				cand = input[vertStart + width + 1 + x];
				if(cand > d)
					i = vertStart + width + 1 + x;
				inBlame[i] += outBlame[pos++];
			}
			vertStart += (width + width);
		}
		chanStart += chanSize;
	}
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
void GBlockScalarSum::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
		output[i] = input[i] + input[outputCount + i];
}

void GBlockScalarSum::backProp()
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
void GBlockScalarProduct::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
		output[i] = input[i] * input[outputCount + i];
}

void GBlockScalarProduct::backProp()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		inBlame[i] += outBlame[i] * input[outputCount + i];
		inBlame[outputCount + i] += outBlame[i] * input[i];
	}
}








GBlockBiasProduct::GBlockBiasProduct(size_t outputs)
: GBlockWeightless(outputs * 2, outputs)
{
}

GBlockBiasProduct::GBlockBiasProduct(GDomNode* pNode)
: GBlockWeightless(pNode)
{
}

GBlockBiasProduct::~GBlockBiasProduct()
{
}

// virtual
void GBlockBiasProduct::forwardProp()
{
	output[0] = input[0] + input[outputCount];
	for(size_t i = 1; i < outputCount; i++)
		output[i] = input[i] * input[outputCount + i];
}

void GBlockBiasProduct::backProp()
{
	inBlame[0] += outBlame[0];
	inBlame[outputCount] += outBlame[0];
	for(size_t i = 1; i < outputCount; i++)
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
void GBlockSwitch::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
		output[i] = input[i] * input[outputCount + i] + (1.0 - input[i]) * input[outputCount + outputCount + i];
}

void GBlockSwitch::backProp()
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

void GBlockHinge::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double alpha = std::max(-1.0, std::min(1.0, weights[i]));
		double beta = std::max(0.0, weights[outputCount + i]);
		output[i] = alpha * (std::sqrt(input[i] * input[i] + beta * beta) - beta) + input[i];
	}
}

void GBlockHinge::backProp()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double alpha = weights[i];
		double beta = weights[outputCount + i];
		inBlame[i] += outBlame[i] * (alpha * input[i] / (std::sqrt(input[i] * input[i] + beta * beta) + 1e-8) + 1.0);
	}
}

void GBlockHinge::updateGradient()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double beta = weights[outputCount + i];
		gradient[i] += outBlame[i] * (std::sqrt(input[i] * input[i] + beta * beta) - beta);
	}
	for(size_t i = 0; i < outputCount; i++)
	{
		double alpha = weights[i];
		double beta = weights[outputCount + i];
		gradient[outputCount + i] += outBlame[i] * alpha * (beta / (std::sqrt(input[i] * input[i] + beta * beta) + 1e-8) - 1.0);
	}
}

void GBlockHinge::step(double learningRate, double momentum)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = std::max(-1.0, std::min(1.0, weights[i] + learningRate * gradient[i]));
		gradient[i] *= momentum;
	}
	for(size_t i = 0; i < outputCount; i++)
	{
		size_t index = outputCount + i;
		weights[index] = std::max(0.0, weights[index] + learningRate * gradient[index]);
		gradient[index] *= momentum;
	}
}

size_t GBlockHinge::weightCount() const
{
	return outputCount * 2;
}

void GBlockHinge::initWeights(GRand& rand)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = 0.0;
		weights[outputCount + i] = 0.1;
	}
}

void GBlockHinge::init_identity(GVec& weights)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = 0.0;
		weights[outputCount + i] = 0.1;
	}
}

bool GBlockHinge::is_identity(GVec& weights)
{
	GVecWrapper vw(weights, 0, outputCount);
	if(vw.squaredMagnitude() == 0.0)
		return true;
	else
		return false;
}

size_t GBlockHinge::adjustedWeightCount(int inAdjustment, int outAdjustment)
{
	return (outputCount + outAdjustment) * 2;
}

void GBlockHinge::addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand)
{
	if(newInputs != newOutputs)
		throw Ex("The number of inputs and outputs must remain equal");
	if(weightsBef.size() != weightCount())
		throw Ex("Wrong bef size");
	if(weightsAft.size() != weightsBef.size() + 2 * newOutputs)
		throw Ex("Wrong aft size");
	weightsAft.copy(0, weightsBef, 0, outputCount);
	weightsAft.fill(0.0, outputCount, newOutputs);
	weightsAft.copy(outputCount + newOutputs, weightsBef, outputCount, outputCount);
	weightsAft.fill(0.1, outputCount + newOutputs + outputCount, newOutputs);
	inputCount += newInputs;
	outputCount += newOutputs;
}

void GBlockHinge::dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft)
{
	if(input != output)
		throw Ex("Cannot drop mismatching inputs and outputs for an activation layer");
	if(output == INVALID_INDEX)
	{
		if(weightsAft.size() != weightsBef.size() || weightsBef.size() != weightCount())
			throw Ex("Wrong weights vector sizes");
		weightsAft.copy(weightsBef);
		return;
	}
	if(weightsAft.size() != weightsBef.size() - 2 || weightsBef.size() != weightCount())
		throw Ex("Wrong weights vector sizes");
	weightsAft.copy(0, weightsBef, 0, output);
	weightsAft.copy(output, weightsBef, output + 1, outputCount - output - 1);
	weightsAft.copy(outputCount - 1, weightsBef, outputCount, output);
	weightsAft.copy(outputCount - 1 + output, weightsBef, outputCount + output + 1, outputCount - output - 1);
	inputCount--;
	outputCount--;
}












GBlockElbow::GBlockElbow(size_t size)
: GBlock(size, size)
{
}

GBlockElbow::GBlockElbow(GDomNode* pNode)
: GBlock(pNode)
{}

void GBlockElbow::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double ax = weights[i] * input[i];
		if(ax < 0)
			output[i] = GBits::sign(weights[i]) * (std::exp(ax) - 1.0 - ax) + input[i];
			//output[i] = (weights[i] == 0.0 ? input[i] : (std::exp(ax) - 1.0) / weights[i]);
		else
			output[i] = input[i];
	}
}

void GBlockElbow::backProp()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double ax = weights[i] * input[i];
		if(ax < 0)
			inBlame[i] += outBlame[i] * (GBits::sign(weights[i]) * (weights[i] * (std::exp(ax) - 1.0)) + 1.0);
			//inBlame[i] += outBlame[i] * std::exp(ax);
		else
			inBlame[i] += outBlame[i];
	}
}

void GBlockElbow::updateGradient()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double ax = weights[i] * input[i];
		if(ax < 0)
			gradient[i] += outBlame[i] * (GBits::sign(weights[i]) * (std::exp(ax) - 1.0) * input[i]);
			//gradient[i] += outBlame[i] * (weights[i] == 0.0 ? 0.5 * input[i] * input[i] + 1.0 : (weights[i] * weights[i] + (ax - 1.0) * std::exp(ax) + 1.0) / (weights[i] * weights[i]));
		else
			gradient[i] += outBlame[i] * input[i];
	}
}

void GBlockElbow::step(double learningRate, double momentum)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = std::max(-1.0, std::min(1.0, weights[i] + learningRate * gradient[i]));
		gradient[i] *= momentum;
	}
}

size_t GBlockElbow::weightCount() const
{
	return outputCount;
}

void GBlockElbow::initWeights(GRand& rand)
{
	weights.fillUniform(rand, -1.0, 1.0);
}

void GBlockElbow::init_identity(GVec& weights)
{
	weights.fill(0.0);
}

bool GBlockElbow::is_identity(GVec& weights)
{
	if(weights.squaredMagnitude() == 0.0)
		return true;
	else
		return false;
}

size_t GBlockElbow::adjustedWeightCount(int inAdjustment, int outAdjustment)
{
	return outputCount + outAdjustment;
}

void GBlockElbow::addUnits(size_t newInputs, size_t newOutputs, const GVec& weightsBef, GVec& weightsAft, GRand& rand)
{
	if(newInputs != newOutputs)
		throw Ex("The number of inputs and outputs must remain equal");
	if(weightsBef.size() != weightCount())
		throw Ex("Wrong bef size");
	if(weightsAft.size() != weightsBef.size() + newOutputs)
		throw Ex("Wrong aft size");
	weightsAft.copy(0, weightsBef, 0, outputCount);
	weightsAft.fill(0.0, outputCount, newOutputs);
	inputCount += newInputs;
	outputCount += newOutputs;
}

void GBlockElbow::dropUnit(size_t input, size_t output, const GVec& weightsBef, GVec& weightsAft)
{
	if(input != output)
		throw Ex("Cannot drop mismatching inputs and outputs for an activation layer");
	if(output == INVALID_INDEX)
	{
		if(weightsAft.size() != weightsBef.size() || weightsBef.size() != weightCount())
			throw Ex("Wrong weights vector sizes");
		weightsAft.copy(weightsBef);
		return;
	}
	if(weightsAft.size() != weightsBef.size() - 1 || weightsBef.size() != weightCount())
		throw Ex("Wrong weights vector sizes");
	weightsAft.copy(0, weightsBef, 0, output);
	weightsAft.copy(output, weightsBef, output + 1, outputCount - output - 1);
	inputCount--;
	outputCount--;
}












GBlockLeakyTanh::GBlockLeakyTanh(size_t size)
: GBlock(size, size)
{
}

GBlockLeakyTanh::GBlockLeakyTanh(GDomNode* pNode)
: GBlock(pNode)
{}

void GBlockLeakyTanh::forwardProp()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		output[i] = (1.0 - weights[i]) * input[i] + weights[i] * tanh(input[i]);
	}
}

void GBlockLeakyTanh::backProp()
{
	for(size_t i = 0; i < outputCount; i++)
	{
		double t = tanh(input[i]);
		inBlame[i] += (1.0 - weights[i]) + weights[i] * (1.0 - t * t);
	}
}

void GBlockLeakyTanh::updateGradient()
{
	for(size_t i = 0; i < outputCount; i++)
		gradient[i] += tanh(input[i]) - input[i];
}

void GBlockLeakyTanh::step(double learningRate, double momentum)
{
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[i] = std::max(0.0, std::min(1.0, weights[i] + learningRate * gradient[i]));
		gradient[i] *= momentum;
	}
}

size_t GBlockLeakyTanh::weightCount() const
{
	return outputCount;
}

void GBlockLeakyTanh::initWeights(GRand& rand)
{
	for(size_t i = 0; i < outputCount; i++)
		weights[i] = 0.0;
}












GBlockSoftExp::GBlockSoftExp(size_t size, double beta)
: GBlock(size, size), m_beta(beta)
{
}

GBlockSoftExp::GBlockSoftExp(GDomNode* pNode)
: GBlock(pNode), m_beta(pNode->getDouble("beta"))
{}

GDomNode* GBlockSoftExp::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->add(pDoc, "beta", m_beta);
	return pNode;
}

void GBlockSoftExp::forwardProp()
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

void GBlockSoftExp::backProp()
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

void GBlockSoftExp::updateGradient()
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

void GBlockSoftExp::initWeights(GRand& rand)
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

void GBlockPAL::forwardProp()
{
	// Compute probabilities of activating
	size_t pos = 0;
	output.copy(0, weights, pos, outputCount);
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

void GBlockPAL::backProp()
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

void GBlockPAL::updateGradient()
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

void GBlockPAL::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / std::max((size_t)1, inputCount));
}









GBlockHypercube::GBlockHypercube(size_t dims, size_t vertexSizeIn, size_t vertexSizeOut)
: GBlock((1 << dims) * vertexSizeIn, (1 << dims) * vertexSizeOut),
	m_dims(dims),
	m_vertexSizeIn(vertexSizeIn),
	m_vertexSizeOut(vertexSizeOut)
{
}

GBlockHypercube::GBlockHypercube(GDomNode* pNode)
: GBlock(pNode)
{
}

void GBlockHypercube::forwardProp()
{
	// Bias values
	output.copy(0, weights, 0, outputCount);

	// Connections
	size_t vertices = 1 << m_dims;
	size_t pos = outputCount;
	for(size_t n = 1; n < vertices; n <<= 1) // For each input vertex
	{
		size_t outPos = 0;
		for(size_t outputVertex = 0; outputVertex < vertices; outputVertex++) // For each output vertex
		{
			size_t inputVertex = outputVertex ^ n; // Toggle one bit
			size_t inPos = inputVertex * m_vertexSizeIn;

			// Fully-connect the vertices
			for(size_t i = 0; i < m_vertexSizeOut; i++)
			{
				double d = 0.0;
				for(size_t j = 0; j < m_vertexSizeIn; j++)
					d += input[inPos + j] * weights[pos++];
				output[outPos + i] += d;
			}
			outPos += m_vertexSizeOut;
		}
	}
	GAssert(pos == weights.size());
}

void GBlockHypercube::backProp()
{
	size_t vertices = 1 << m_dims;
	size_t pos = outputCount;
	for(size_t n = 1; n < vertices; n <<= 1) // For each input vertex
	{
		size_t outPos = 0;
		for(size_t outputVertex = 0; outputVertex < vertices; outputVertex++) // For each output vertex
		{
			size_t inputVertex = outputVertex ^ n; // Toggle one bit
			size_t inPos = inputVertex * m_vertexSizeIn;
			for(size_t i = 0; i < m_vertexSizeOut; i++)
			{
				inBlame.addScaled(inPos, outBlame[outPos + i], weights, pos, m_vertexSizeIn);
				pos += m_vertexSizeIn;
			}
			outPos += m_vertexSizeOut;
		}
	}
	GAssert(pos == weights.size());
}

void GBlockHypercube::updateGradient()
{
	// Bias values
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		gradient[pos] += outBlame[pos];
		pos++;
	}

	// Connections
	size_t vertices = 1 << m_dims;
	for(size_t n = 1; n < vertices; n <<= 1) // For each input vertex
	{
		size_t outPos = 0;
		for(size_t outputVertex = 0; outputVertex < vertices; outputVertex++) // For each output vertex
		{
			size_t inputVertex = outputVertex ^ n; // Toggle one bit
			size_t inPos = inputVertex * m_vertexSizeIn;
			for(size_t i = 0; i < m_vertexSizeOut; i++)
			{
				for(size_t j = 0; j < m_vertexSizeIn; j++)
					gradient[pos++] += outBlame[outPos + i] * input[inPos + j];
			}
			outPos += m_vertexSizeOut;
		}
	}
	GAssert(pos == weights.size());
}

size_t GBlockHypercube::weightCount() const
{
	return outputCount * (1 + m_dims * m_vertexSizeIn);
}

void GBlockHypercube::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / std::max((size_t)1, m_vertexSizeIn * m_dims));
}












GBlockOptional::GBlockOptional(size_t inputs, size_t outputs)
: GBlock(inputs, outputs)
{
}

GBlockOptional::GBlockOptional(GDomNode* pNode)
: GBlock(pNode)
{
}

void GBlockOptional::forwardProp()
{
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		double sumWeightedVals = 0.0;
		double sumWeights = 1e-8;
		for(size_t j = 0; j < inputCount; j++)
		{
			if(input[j] == UNKNOWN_REAL_VALUE)
			{
				pos += 4;
				continue;
			}
			double v = weights[pos + 0] + input[j] * weights[pos + 1];
			double w = weights[pos + 2] + input[j] * weights[pos + 3];
			if(w < 0)
				w = std::exp(w);
			else
				w += 1.0;
			pos += 4;
			sumWeightedVals += (v * w);
			sumWeights += w;
		}
		output[i] = sumWeightedVals / sumWeights;
	}
	GAssert(pos == weights.size());
}

void GBlockOptional::backProp()
{
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		// Redo the forward pass (because we didn't bother to store all the values we would need)
		double sumWeightedVals = 0.0;
		double sumWeights = 1e-8;
		double restoreme = pos;
		for(size_t j = 0; j < inputCount; j++)
		{
			if(input[j] == UNKNOWN_REAL_VALUE)
			{
				pos += 4;
				continue;
			}
			double v = weights[pos + 0] + input[j] * weights[pos + 1];
			double w = weights[pos + 2] + input[j] * weights[pos + 3];
			if(w < 0)
				w = std::exp(w);
			else
				w += 1.0;
			pos += 4;
			sumWeightedVals += (v * w);
			sumWeights += w;
		}
		pos = restoreme;

		// Compute the gradients
		for(size_t j = 0; j < inputCount; j++)
		{
			if(input[j] == UNKNOWN_REAL_VALUE)
			{
				pos += 4;
				continue;
			}
			double v = weights[pos + 0] + input[j] * weights[pos + 1];
			double w = weights[pos + 2] + input[j] * weights[pos + 3];
			if(w < 0)
				w = std::exp(w);
			else
				w += 1.0;
			double vBlame = outBlame[i] * (w / sumWeights) + 0.1 * (output[i] - v);
			inBlame[j] += vBlame * weights[pos + 1];
			double wBlame = outBlame[i] * (w < 1.0 ? w : 1.0) * (v * (sumWeights - w) - (sumWeightedVals - (v * w))) / (sumWeights * sumWeights);
			inBlame[j] += wBlame * weights[pos + 3];
			pos += 4;
		}
	}
	GAssert(pos == weights.size());
}

void GBlockOptional::updateGradient()
{
	size_t pos = 0;
	for(size_t i = 0; i < outputCount; i++)
	{
		// Redo the forward pass (because we didn't bother to store all the values we would need)
		double sumWeightedVals = 0.0;
		double sumWeights = 1e-8;
		double restoreme = pos;
		for(size_t j = 0; j < inputCount; j++)
		{
			if(input[j] == UNKNOWN_REAL_VALUE)
			{
				pos += 4;
				continue;
			}
			double v = weights[pos + 0] + input[j] * weights[pos + 1];
			double w = weights[pos + 2] + input[j] * weights[pos + 3];
			if(w < 0)
				w = std::exp(w);
			else
				w += 1.0;
			pos += 4;
			sumWeightedVals += (v * w);
			sumWeights += w;
		}
		pos = restoreme;

		// Compute the gradients
		for(size_t j = 0; j < inputCount; j++)
		{
			if(input[j] == UNKNOWN_REAL_VALUE)
			{
				pos += 4;
				continue;
			}
			double v = weights[pos + 0] + input[j] * weights[pos + 1];
			double w = weights[pos + 2] + input[j] * weights[pos + 3];
			if(w < 0)
				w = std::exp(w);
			else
				w += 1.0;
			double vBlame = outBlame[i] * (w / sumWeights) + 0.1 * (output[i] - v);
			gradient[pos + 0] += vBlame;
			gradient[pos + 1] += vBlame * input[j];
			double wBlame = outBlame[i] * (w < 1.0 ? w : 1.0) * (v * (sumWeights - w) - (sumWeightedVals - (v * w))) / (sumWeights * sumWeights);
			gradient[pos + 2] += wBlame;
			gradient[pos + 3] += wBlame * input[j];
			pos += 4;
		}
	}
	GAssert(pos == weights.size());
}

size_t GBlockOptional::weightCount() const
{
	return 4 * inputCount * outputCount;
}

void GBlockOptional::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / std::max((size_t)1, 4 * inputCount));
}












GBlockCatIn::GBlockCatIn(size_t valueCount, size_t units)
: GBlock(1, units), m_valueCount(valueCount)
{
}

GBlockCatIn::GBlockCatIn(GDomNode* pNode)
: GBlock(pNode), m_valueCount(pNode->getInt("values"))
{
}

void GBlockCatIn::forwardProp()
{
	size_t i = (size_t)input[0];
	GAssert(i < m_valueCount);
	output.copy(0, weights, outputCount * std::min(m_valueCount - 1, i), outputCount);
}

void GBlockCatIn::backProp()
{
	inBlame[0] = 0.0;
}

void GBlockCatIn::updateGradient()
{
	GAssert(gradient.size() == outBlame.size());
	gradient.copy(outBlame);
}

void GBlockCatIn::step(double learningRate, double momentum)
{
	size_t n = std::min(m_valueCount - 1, (size_t)input[0]);
	size_t start = outputCount * n;
	for(size_t i = 0; i < outputCount; i++)
	{
		weights[start + i] += learningRate * gradient[i];
		gradient[i] = 0.0; // Momentum is ignored for this layer because the gradient is specific to this input
	}
}

size_t GBlockCatIn::weightCount() const
{
	return m_valueCount * outputCount;
}

size_t GBlockCatIn::gradCount() const
{
	return outputCount;
}

void GBlockCatIn::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 0.2);
}










GBlockCatOut::GBlockCatOut(size_t categories, double margin)
: GBlockWeightless(categories, 1), m_margin(margin)
{
}

void GBlockCatOut::forwardProp()
{
	output[0] = (double)input.indexOfMax();
}

double GBlockCatOut::computeBlame(const GVec& target)
{
	// We just store the target. backProp will use this value to compute a meaningful blame for each of the inputs.
	outBlame[0] = target[0];
	return 0.0;
}

void GBlockCatOut::backProp()
{
	size_t targ = (size_t)outBlame[0];
	size_t pred = (size_t)output[0];
	for(size_t i = 0; i < inputCount; i++)
	{
		if(i == targ)
		{
			if(targ == pred)
				inBlame[i] = std::max(0.0, m_margin - input[i]);
			else
				inBlame[i] = 1.0 - input[i];
		}
		else
			inBlame[i] = std::min(0.0, std::max(0.0, input[targ] - m_margin) - input[i]);
	}
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

void GBlockLSTM::forwardProp()
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

void GBlockLSTM::backProp()
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

void GBlockLSTM::updateGradient()
{
	for(GBlockLSTM* pInst = pNextInstance; pInst != this; pInst = pInst->pNextInstance)
		pInst->updateGradient_instance(weights, gradient);
	updateGradient_instance(weights, gradient);
}

size_t GBlockLSTM::weightCount() const
{
	return 3 * outputCount * (1 + inputCount + 2);
}

void GBlockLSTM::initWeights(GRand& rand)
{
	weights.fillNormal(rand, 1.0 / inputCount + 3);
}













GLayer::GLayer()
: input_count(0), output_count(0), weight_count(0)
{
}

GLayer::GLayer(const GLayer& that, GLayer* pPrevLayer)
: input_count(that.input_count), output_count(that.output_count), weight_count(that.weight_count), grad_count(that.grad_count), output(that.output), outBlame(that.outBlame)
{
	size_t pos = 0;
	for(size_t i = 0; i < that.m_blocks.size(); i++)
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

GLayer::GLayer(GDomNode* pNode, GRand& rand)
: input_count(0),
output_count(0),
weight_count(0),
grad_count(0)
{
	GDomNode* pBlocks = pNode->get("blocks");
	GDomListIterator it(pBlocks);
	while(it.remaining() > 0)
	{
		GBlock* pBlock = GBlock::deserialize(it.current(), rand);
		add(pBlock, pBlock->inPos());
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
	GDomNode* pBlocks = pNode->add(pDoc, "blocks", pDoc->newList());
	for(size_t i = 0; i < m_blocks.size(); i++)
		pBlocks->add(pDoc, m_blocks[i]->serialize(pDoc));
	return pNode;
}

void GLayer::add(GBlock* pBlock, size_t inPos)
{
	pBlock->setInPos(inPos);
	m_blocks.push_back(pBlock);
	output_count = 0; // Force a recount
}

void GLayer::recount()
{
	// Count the inputs and outputs
	input_count = 0;
	output_count = 0;
	weight_count = 0;
	grad_count = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		size_t inPos = m_blocks[i]->inPos();
		size_t inSize = m_blocks[i]->inputs();
		size_t outSize = m_blocks[i]->outputs();
		input_count = std::max(input_count, inPos + inSize);
		output_count += outSize;
		weight_count += m_blocks[i]->weightCount();
		grad_count += m_blocks[i]->gradCount();
	}
}

size_t GLayer::inputs() const
{
	if(output_count == 0)
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
	if(output_count == 0)
	{
		((GLayer*)this)->recount();
		GAssert(weight_count < 100000000);
	}
	return weight_count;
}

size_t GLayer::gradCount() const
{
	if(output_count == 0)
		((GLayer*)this)->recount();
	return grad_count;
}

void GLayer::initWeights(GRand& rand)
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		m_blocks[i]->initWeights(rand);
}

void GLayer::bind(const GVec* pInput, GVec* pOutput, GVec* pOutBlame, GVec* pInBlame, GVec& _weights, GVec& _gradient)
{
	// Force a recount of sizes
	output_count = 0;

	// Check and set up buffers for this layer
	if(pInput && pInput->size() != inputs())
		throw Ex("Expected an input vector of size ", to_str(inputs()), ". Got ", to_str(pInput->size()));
	if(pOutput)
	{
		if(pOutput->size() != outputs())
			throw Ex("Expected an output vector of size ", to_str(outputs()), ". Got ", to_str(pOutput->size()));
		outputBuf.resize(0);
		output.setData(*pOutput);
	}
	else
	{
		outputBuf.resize(outputs());
		output.setData(outputBuf);
	}
	if(pOutBlame)
	{
		if(pOutBlame->size() != outputs())
			throw Ex("Expected an outBlame vector of size ", to_str(outputs()), ". Got ", to_str(pOutBlame->size()));
		outBlameBuf.resize(0);
		outBlame.setData(*pOutBlame);
	}
	else
	{
		outBlameBuf.resize(outputs());
		outBlame.setData(outBlameBuf);
	}
	if(pInBlame && pInBlame->size() != inputs())
		throw Ex("Expected an inBlame vector of size ", to_str(inputs()), ". Got ", to_str(pInBlame->size()));

	// Bind all the blocks to the appropriate sub-buffers
	GConstVecWrapper vwInput;
	GVecWrapper vwOutput;
	GVecWrapper vwOutBlame;
	GVecWrapper vwInBlame;
	GVecWrapper vwWeights;
	GVecWrapper vwGradient;
	size_t posOutput = 0;
	size_t posWeights = 0;
	size_t posGradient = 0;
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);

		// Figure out all of the sub-buffers for this block
		if(pInput)
			vwInput.setData(*pInput, b.inPos(), b.inputs());
		vwOutput.setData(output, posOutput, b.outputs());
		vwOutBlame.setData(outBlame, posOutput, b.outputs());
		posOutput += b.outputs();
		if(pInBlame)
			vwInBlame.setData(*pInBlame, b.inPos(), b.inputs());
		size_t nWeights = b.weightCount();
		vwWeights.setData(_weights, posWeights, nWeights);
		posWeights += nWeights;
		size_t nGradient = b.gradCount();
		vwGradient.setData(_gradient, posGradient, nGradient);
		posGradient += nGradient;

		// Bind the block to the relevant sub-buffers
		b.bind(pInput ? &vwInput : nullptr, &vwOutput, &vwOutBlame, pInBlame ? &vwInBlame : nullptr, &vwWeights, &vwGradient);
	}
}

void GLayer::bindInput(const GVec& _input)
{
	GAssert(_input.size() == inputs());
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		b.input.setData(_input, b.inPos(), b.inputs());
	}
}

void GLayer::bindInBlame(GVec& _inBlame)
{
	GAssert(_inBlame.size() == inputs());
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		b.inBlame.setData(_inBlame, b.inPos(), b.inputs());
	}
}

void GLayer::forwardProp()
{
	for(size_t i = 0; i < blockCount(); i++)
		m_blocks[i]->forwardProp();
}

void GLayer::backProp()
{
	for(size_t i = blockCount() - 1; i < blockCount(); i--)
		m_blocks[i]->backProp();
}

void GLayer::updateGradient()
{
	for(size_t i = 0; i < blockCount(); i++)
		m_blocks[i]->updateGradient();
}

void GLayer::updateGradientNormalized()
{
	for(size_t i = 0; i < blockCount(); i++)
		m_blocks[i]->updateGradientNormalized();
}

void GLayer::step(double learningRate, double momentum)
{
	for(size_t i = 0; i < blockCount(); i++)
		m_blocks[i]->step(learningRate, momentum);
}

void GLayer::step_jitter(double learningRate, double momentum, double jitter, GRand& rand)
{
	for(size_t i = 0; i < blockCount(); i++)
		m_blocks[i]->step_jitter(learningRate, momentum, jitter, rand);
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

double GLayer::computeBlame(const GVec& target)
{
	GAssert(target.size() == outBlame.size());
	double sse = 0.0;
	size_t pos = 0;
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		size_t n = b.outputs();
		const GConstVecWrapper t(target, pos, n);
		pos += n;
		sse += b.computeBlame(t);
	}
	return sse;
}

void GLayer::biasMask(GVec& mask)
{
	size_t posWeights = 0;
	for(size_t i = 0; i < blockCount(); i++)
	{
		GBlock& b = block(i);
		size_t wc = b.weightCount();
		GVecWrapper w(mask, posWeights, wc);
		posWeights += wc;
		b.biasMask(w);
	}
	GAssert(posWeights == mask.size());
}











GNeuralNet::GNeuralNet()
: GBlock(0, 0), m_weightCount(0)
{
}

GNeuralNet::GNeuralNet(const GNeuralNet& that)
: GBlock(that), m_weightCount(that.m_weightCount)
{
	for(size_t i = 0; i < that.m_layers.size(); i++)
	{
		m_layers.push_back(new GLayer(*that.m_layers[i], i > 0 ? m_layers[i - 1] : nullptr));
	}
}

GNeuralNet::GNeuralNet(GDomNode* pNode, GRand& rand)
: GBlock(pNode), m_weightCount(0), m_gradCount(0)
{
	deserialize(pNode, rand);
}

// virtual
GNeuralNet::~GNeuralNet()
{
	deleteAllLayers();
}

void GNeuralNet::deleteAllLayers()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
	m_layers.clear();
	m_weightCount = 0;
	m_gradCount = 0;
}

GDomNode* GNeuralNet::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	GDomNode* pLayers = pNode->add(pDoc, "layers", pDoc->newList());
	for(size_t i = 0; i < m_layers.size(); i++)
		pLayers->add(pDoc, m_layers[i]->serialize(pDoc));
	if(weights.size() == 0)
		throw Ex("Attempted to serialize a neural net that was never initialized");
	pNode->add(pDoc, "weights", weights.serialize(pDoc));
	return pNode;
}

void GNeuralNet::deserialize(GDomNode* pNode, GRand& rand)
{
	deleteAllLayers();
	m_weightCount = 0;
	m_gradCount = 0;
	GDomNode* pLayers = pNode->get("layers");
	GDomListIterator it(pLayers);
	while(it.remaining() > 0)
	{
		GLayer* pNewLayer = new GLayer(it.current(), rand);
		m_layers.push_back(pNewLayer);
		it.advance();
	}
	weightsBuf.deserialize(pNode->get("weights"));
	init(weightsBuf);
}

std::string GNeuralNet::to_str(const std::string& line_prefix, bool includeWeights, bool includeActivations) const
{
	std::ostringstream oss;
	oss << line_prefix << "[GNeuralNet: " << inputs() << "->" << outputs() << ", Weights=" << GClasses::to_str(weightCount()) << "\n";
	size_t pos = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		oss << line_prefix << "  " << to_fixed_str(i, 2, ' ') << ") ";
		for(size_t j = 0; j < m_layers[i]->blockCount(); j++)
		{
			GBlock& b = m_layers[i]->block(j);
			size_t wc = b.weightCount();
			pos += wc;
			if(b.type() == block_neuralnet)
				oss << ">>> Nest in\n";
			oss << b.to_str(includeWeights, includeActivations);
			if(b.type() == block_neuralnet)
				oss << "<<< Nest out\n";
		}
		oss << "\n";
	}
	oss << line_prefix << "]";
	return oss.str();
}

std::string GNeuralNet::to_str(bool includeWeights, bool includeActivations) const
{
	return to_str("", includeWeights, includeActivations);
}

void GNeuralNet::bind(const GVec* pInput, GVec* pOutput, GVec* pOutBlame, GVec* pInBlame, GVec* pWeights, GVec* pGradient)
{
	// Set the input, inBlame, weights, and gradient
	if(pInput)
		input.setData(*pInput);
	if(pInBlame)
		inBlame.setData(*pInBlame);
	if(pWeights)
	{
		if(pWeights != &weightsBuf)
			weightsBuf.resize(0);
		weights.setData(*pWeights);
	}
	else
	{
		weightsBuf.resize(weightCount());
		weights.setData(weightsBuf);
	}
	if(pGradient)
	{
		if(pGradient != &gradientBuf)
			gradientBuf.resize(0);
		gradient.setData(*pGradient);
	}
	else
	{
		gradientBuf.resize(gradCount());
		gradient.setData(gradientBuf);
	}

	// Bind all the layers to the right parts of the buffers
	const GVec* pIn = pInput;
	GVec* pInBl = pInBlame;
	size_t posWeights = 0;
	size_t posGradient = 0;
	GVecWrapper w;
	GVecWrapper g;
	GLayer* pLayPrev = nullptr;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = *m_layers[i];
		size_t wc = lay.weightCount();
		size_t gc = lay.gradCount();
		w.setData(weights, posWeights, wc);
		g.setData(gradient, posGradient, gc);
		posWeights += wc;
		posGradient += gc;
		if(pLayPrev)
		{
			pIn = &pLayPrev->output;
			pInBl = &pLayPrev->outBlame;
		}
		if(i + 1 == m_layers.size())
			lay.bind(pIn, pOutput, pOutBlame, pInBl, w, g);
		else
			lay.bind(pIn, nullptr, nullptr, pInBl, w, g);
		pLayPrev = &lay;
	}

	// Set output and outBlame
	GLayer& outLay = *m_layers[m_layers.size() - 1];
	output.setData(outLay.output);
	outBlame.setData(outLay.outBlame);

	// Check that everything lined up
	if(posWeights != weights.size())
		throw Ex("Weights size problem");
	if(posGradient != gradient.size())
		throw Ex("Gradient size problem");
}

GBlock* GNeuralNet::add(GBlock* pBlock)
{
	GAssert(m_weightCount == 0, "weights were counted before all blocks were added");
	GLayer* pNewLayer = new GLayer();
	m_layers.push_back(pNewLayer);
	pNewLayer->add(pBlock, 0);
	if(m_layers.size() == 1)
		inputCount = pNewLayer->inputs();
	outputCount = pNewLayer->outputs();
	return pBlock;
}

GBlock* GNeuralNet::concat(GBlock* pBlock, size_t inPos)
{
	GAssert(m_weightCount == 0, "weights were counted before all blocks were added");
	if(m_layers.size() == 0)
		throw Ex("add must be called before concat");
	GLayer* pLastLayer = m_layers[m_layers.size() - 1];
	pLastLayer->add(pBlock, inPos);
	if(m_layers.size() == 1)
		inputCount = pLastLayer->inputs();
	outputCount = pLastLayer->outputs();
	return pBlock;
}
/*
GVec* GNeuralNet::insert(size_t position, GBlock* pBlock, GVec* pOldWeights, GRand& rand)
{
	// Insert the block with a new layer
	if(pOldWeights->size() != weightCount())
		throw Ex("Old weights have unexpected size");
	size_t layStart = layerStart(position);
	m_weightCount = 0;
	if(position == m_layers.size())
		add(pBlock);
	else
	{
		if(pBlock->inputs() != pBlock->outputs())
			throw Ex("Sorry, inserted blocks must have the same number of inputs and outputs");
		if(pBlock->inputs() != layer(position).inputs())
			throw Ex("this block size does not fit there");
		GLayer* pPrevLayer = nullptr;
		if(position > 0)
			pPrevLayer = m_layers[position - 1];
		GLayer* pNewLayer = new GLayer();
		m_layers.insert(m_layers.begin() + position, pNewLayer);
		pNewLayer->add(pBlock, 0);
		pNewLayer->attach(pPrevLayer);
		GLayer* pNextLayer = m_layers[position + 1];
		pNextLayer->attach(pNewLayer);
	}

	// Make the new weights
	if(pBlock->weightCount() == 0)
		return pOldWeights;
	GVec* pNewWeights = new GVec(weightCount());
	pNewWeights->copy(0, *pOldWeights, 0, layStart); // Everything before the new layer
	GVecWrapper vw(*pNewWeights, layStart, pBlock->weightCount());
	pBlock->init_identity(vw);
	pNewWeights->copy(layStart + pBlock->weightCount(), *pOldWeights, layStart, pOldWeights->size() - layStart);
	if(layStart + pBlock->weightCount() + pOldWeights->size() - layStart != pNewWeights->size())
		throw Ex("weights size problem");
	m_weightCount = 0; // Force a recount
	m_gradCount = 0;
	return pNewWeights;
}

GVec* GNeuralNet::drop(size_t layer, GVec* pOldWeights)
{
	// Drop the layer
	if(layer + 1 >= m_layers.size())
		throw Ex("Cannot drop the last layer");
	if(pOldWeights->size() != weightCount())
		throw Ex("Old weights have unexpected size");
	size_t layStart = layerStart(layer);
	GLayer* pLay = m_layers[layer];
	GLayer* pLayBef = (layer > 0 ? m_layers[layer - 1] : nullptr);
	if(pLay->inputs() != pLay->outputs())
		throw Ex("Only layers with the same number of inputs and outputs can be dropped");
	size_t condemnedWeights = pLay->weightCount();
	m_weightCount = 0;
	m_layers.erase(m_layers.begin() + layer);
	GLayer& layAft = *m_layers[layer];
	layAft.attach(pLayBef);
	delete(pLay);

	// Adjust the weights
	if(condemnedWeights == 0)
		return pOldWeights;
	GVec* pNewWeights = new GVec(pOldWeights->size() - condemnedWeights);
	pNewWeights->copy(0, *pOldWeights, 0, layStart);
	pNewWeights->copy(layStart, *pOldWeights, layStart + condemnedWeights, pNewWeights->size() - layStart);
	m_weightCount = 0; // Force a recount
	m_gradCount = 0;
	return pNewWeights;
}

GVec* GNeuralNet::increaseWidth(size_t newUnitsPerLayer, GVec* pWeights, size_t startLayer, size_t layerCount, GRand& rand)
{
	// Allocate a new weights vector
	size_t weightsCountAft = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = layer(i);
		if(lay.blockCount() > 1)
			throw Ex("Sorry, multiple blocks not yet supported");
		GBlock& b = lay.block(0);
		weightsCountAft += b.adjustedWeightCount(i > startLayer && i <= startLayer + layerCount ? (int)newUnitsPerLayer : 0, i >= startLayer && i < startLayer + layerCount ? (int)newUnitsPerLayer : 0);
	}
	GVec* pWeightsAft = new GVec(weightsCountAft);

	// Fill new weights
	size_t posBef = 0;
	size_t posAft = 0;
	GVecWrapper vwBef;
	GVecWrapper vwAft;
	GLayer* pPrevLayer = nullptr;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = layer(i);
		GBlock& b = lay.block(0);
		size_t wcBef = b.weightCount();
		size_t wcAft = b.adjustedWeightCount(i > startLayer && i <= startLayer + layerCount ? (int)newUnitsPerLayer : 0, i >= startLayer && i < startLayer + layerCount ? (int)newUnitsPerLayer : 0);
		vwBef.setData(*pWeights, posBef, wcBef);
		vwAft.setData(*pWeightsAft, posAft, wcAft);
		posBef += wcBef;
		posAft += wcAft;
		b.addUnits(i > startLayer && i <= startLayer + layerCount ? newUnitsPerLayer : 0, i >= startLayer && i < startLayer + layerCount ? newUnitsPerLayer : 0, vwBef, vwAft, rand);

		// Attach to previous layer
		lay.attach(pPrevLayer);
		pPrevLayer = &lay;
	}
	m_weightCount = 0; // Force a recount
	m_gradCount = 0;
	return pWeightsAft;
}

GVec* GNeuralNet::decrementWidth(GVec* pWeights, size_t startLayer, size_t layerCount)
{
	// Test whether a unit can be removed from each layer
	vector<size_t> ignored_units;
	size_t posBef = 0;
	size_t sizeAft = 0;
	GVecWrapper vwBef;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = layer(i);
		if(lay.blockCount() > 1)
			throw Ex("Sorry, multiple blocks not yet supported");
		GBlock& b = lay.block(0);
		size_t wc = b.weightCount();
		vwBef.setData(*pWeights, posBef, wc);
		posBef += wc;
		if(i > startLayer && i <= startLayer + layerCount)
		{
			if(b.type() == GBlock::block_linear)
			{
				size_t unit = b.firstIgnoredInput(vwBef);
				if(unit == INVALID_INDEX)
					return nullptr;
				ignored_units.push_back(unit);
			}
			else if(b.type() == GBlock::block_hinge) {}
			else if(b.type() == GBlock::block_elbow) {}
			else
				throw Ex("Unsupported block type");
		}
		sizeAft += b.adjustedWeightCount(i > startLayer && i <= startLayer + layerCount ? -1 : 0, i >= startLayer && i < startLayer + layerCount ? -1 : 0);
	}
	ignored_units.push_back(INVALID_INDEX);

	// Remove a unit from each layer
	GVec* pWeightsAft = new GVec(sizeAft);
	GVecWrapper vwAft;
	posBef = 0;
	size_t posAft = 0;
	size_t listPos = 0;
	size_t inCondemned = INVALID_INDEX;
	size_t outCondemned = INVALID_INDEX;
	GLayer* pPrevLayer = nullptr;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = layer(i);
		if(lay.blockCount() > 1)
			throw Ex("Sorry, multiple blocks not yet supported");
		GBlock& b = lay.block(0);
		size_t wcBef = b.weightCount();
		size_t wcAft = b.adjustedWeightCount(i > startLayer && i <= startLayer + layerCount ? -1 : 0, i >= startLayer && i < startLayer + layerCount ? -1 : 0);
		vwBef.setData(*pWeights, posBef, wcBef);
		vwAft.setData(*pWeightsAft, posAft, wcAft);
		posBef += wcBef;
		posAft += wcAft;
		if(i >= startLayer && i <= startLayer + layerCount)
		{
			if(b.type() == GBlock::block_linear)
			{
				inCondemned = outCondemned;
				outCondemned = ignored_units[listPos++];
				b.dropUnit(inCondemned, outCondemned, vwBef, vwAft);
			}
			else if(b.type() == GBlock::block_hinge)
				b.dropUnit(outCondemned, outCondemned, vwBef, vwAft);
			else if(b.type() == GBlock::block_elbow)
				b.dropUnit(outCondemned, outCondemned, vwBef, vwAft);
			else
				throw Ex("Unsupported block type");
		}
		else
			vwAft.copy(vwBef);

		// Attach to previous layer
		lay.attach(pPrevLayer);
		pPrevLayer = &lay;
	}
	m_weightCount = 0; // Force a recount
	m_gradCount = 0;
	return pWeightsAft;
}
*/
void GNeuralNet::regularize_square(double lambda)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = *m_layers[i];
		GBlock& b = lay.block(0);
		if(b.type() != block_linear)
			continue;
		if(b.inputs() != b.outputs())
			continue;
		GBlockLinear* pLin = (GBlockLinear*)&b;
		GVec* pWeightsNext = nullptr;
		if(i + 2 < m_layers.size())
		{
			GBlock& b2 = m_layers[i + 2]->block(0);
			if(b2.type() == block_linear && b2.inputs() == b.inputs() && b2.outputs() == b.outputs())
				pWeightsNext = &b2.weights;
		}
		pLin->regularize_square(lambda, pWeightsNext);
	}
}

void GNeuralNet::forwardProp()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->forwardProp();
}

GVec& GNeuralNet::forwardProp(const GVec& in)
{
	input.setData(in);
	m_layers[0]->bindInput(in);
	forwardProp();
	return output;
}

double GNeuralNet::computeBlame(const GVec& target)
{
	return outputLayer().computeBlame(target);
}

void GNeuralNet::backProp()
{
	double minBlameSqMag = outputLayer().outBlame.squaredMagnitude() * 0.0001;
	for(size_t i = m_layers.size() - 1; i > 0; i--)
	{
		GLayer& layPrev = *m_layers[i - 1];
		layPrev.outBlame.fill(0.0);
		GLayer& lay = *m_layers[i];
		lay.backProp();

		// Ensure that the blame has not diminished into oblivion
		double sqMag = layPrev.outBlame.squaredMagnitude();
		if(sqMag > 0.0 && sqMag < minBlameSqMag)
			layPrev.outBlame *= minBlameSqMag / sqMag;
	}
	m_layers[0]->backProp();
}

void GNeuralNet::backPropFast()
{
	double minBlameSqMag = outputLayer().outBlame.squaredMagnitude() * 0.0001;
	for(size_t i = m_layers.size() - 1; i > 0; i--)
	{
		GLayer& layPrev = *m_layers[i - 1];
		layPrev.outBlame.fill(0.0);
		GLayer& lay = *m_layers[i];
		lay.backProp();

		// Ensure that the blame has not diminished into oblivion
		double sqMag = layPrev.outBlame.squaredMagnitude();
		if(sqMag > 0.0 && sqMag < minBlameSqMag)
			layPrev.outBlame *= minBlameSqMag / sqMag;
	}
}

void GNeuralNet::backpropagate(GVec* inputBlame)
{
	GAssert(weights.size() == weightCount());
	if(inputBlame)
	{
		inBlame.setData(*inputBlame);
		m_layers[0]->bindInBlame(*inputBlame);
		inputBlame->fill(0.0);
		inBlame.setData(*inputBlame);
		backProp();
	}
	else
		backPropFast();
}

void GNeuralNet::updateGradient()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->updateGradient();
}

void GNeuralNet::updateGradientNormalized()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->updateGradientNormalized();
}

void GNeuralNet::step(double learningRate, double momentum)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->step(learningRate, momentum);
}

void GNeuralNet::step_jitter(double learningRate, double momentum, double jitter, GRand& rand)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->step_jitter(learningRate, momentum, jitter, rand);
}

void GNeuralNet::recount()
{
	m_weightCount = 0;
	m_gradCount = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		m_weightCount += m_layers[i]->weightCount();
		m_gradCount += m_layers[i]->gradCount();
	}
	GAssert(m_weightCount < 100000000);
}

size_t GNeuralNet::weightCount() const
{
	if(m_weightCount == 0)
		((GNeuralNet*)this)->recount();
	return m_weightCount;
}

size_t GNeuralNet::gradCount() const
{
	if(m_weightCount == 0)
		((GNeuralNet*)this)->recount();
	return m_gradCount;
}

void GNeuralNet::copyTopology(const GNeuralNet& other)
{
	deleteAllLayers();
	m_weightCount = other.weightCount();
	m_gradCount = other.gradCount();
	GLayer* pPrevLayer = nullptr;
	for(size_t i = 0; i < other.m_layers.size(); i++)
	{
		GLayer* pNewLayer = new GLayer(*other.m_layers[i], pPrevLayer);
		m_layers.push_back(pNewLayer);
		pPrevLayer = pNewLayer;
	}
}

void GNeuralNet::initWeights(GRand& rand)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->initWeights(rand);

	// Check that the layers all line up
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		GLayer& a = *m_layers[i - 1];
		GLayer& b = *m_layers[i];
		if(a.outputs() != b.inputs())
			throw Ex("Layer ", GClasses::to_str(i - 1), " outputs ", GClasses::to_str(a.outputs()), " values, but layer ", GClasses::to_str(i), " expects ", GClasses::to_str(b.inputs()), " inputs");
	}
}

void GNeuralNet::init(GRand& rand, GNeuralNet* pBase)
{
	if(pBase)
		bind(&pBase->output, nullptr, nullptr, &pBase->outBlame, nullptr, nullptr);
	else
		bind(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
	initWeights(rand);
	gradient.fill(0.0);
}

void GNeuralNet::init(GVec& weights)
{
	bind(nullptr, nullptr, nullptr, nullptr, &weights, nullptr);
	gradient.fill(0.0);
}

double GNeuralNet::measureLoss(const GMatrix& features, const GMatrix& labels, double* pOutSAE)
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
		if(labels.cols() == 1 && labels.relation().valueCount(0) > 0)
		{
			// Classification with a GBlockCatOut on the output. Count misclassifications.
			for(size_t i = 0; i < features.rows(); i++)
			{
				GVec& prediction = forwardProp(features[i]);
				const GVec& targ = labels[i];
				if(targ[0] >= 0.0)
				{
					if((size_t)prediction[0] != (size_t)targ[0])
						sse++;
				}
			}
			sae = sse;
		}
		else
		{
			// Regression. Compute SSE and SAE.
			for(size_t i = 0; i < features.rows(); i++)
			{
				GVec& prediction = forwardProp(features[i]);
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
	}
	else if(labels.cols() == 1 && labels.relation().valueCount(0) == outputs())
	{
		// Classification with 1-hot output. Count misclassifications.
		for(size_t i = 0; i < features.rows(); i++)
		{
			GVec& prediction = forwardProp(features[i]);
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

std::string GNeuralNet::toEquation()
{
	std::ostringstream os;
	size_t prev = 1;
	size_t cur = prev + inputs();
	size_t wp = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		if(m_layers[i]->blockCount() > 1)
			throw Ex("Multiple blocks not yet supported");
		GBlock* b = &m_layers[i]->block(0);
		for(size_t j = 0; j < b->outputs(); j++)
		{
			if(i < m_layers.size() - 1)
				os << "h" << GClasses::to_str(cur + j) << " = ";
			else
				os << "y" << GClasses::to_str(1 + j) << " = ";
			if(b->type() == GBlock::block_linear)
			{
				size_t biasStart = wp;
				wp += b->outputs();
				size_t weightStart = wp;
				wp += b->inputs() * b->outputs();
				for(size_t k = 0; k < b->outputs(); k++)
				{
					os << GClasses::to_str(weights[biasStart + k]);
					for(size_t l = 0; l < b->inputs(); l++)
					{
						if(weights[weightStart + b->outputs() * l + k] != 0.0)
							os << " + " << GClasses::to_str(weights[weightStart + b->outputs() * l + k]) << (i == 0 ? " * x" : " * h") << GClasses::to_str(prev + l);
					}
				}
			}
			else if(b->type() == GBlock::block_tanh)
			{
				os << "tanh( " << (i == 0 ? " * x" : " * h") << GClasses::to_str(prev + j);
			}
			else if(b->type() == GBlock::block_scalarproduct)
			{
				os << (i == 0 ? " * x" : " * h") << GClasses::to_str(prev + j) << " * " << (i == 0 ? " * x" : " * h") << GClasses::to_str(prev + b->outputs() + j);
			}
		}
		os << ";\n";
		prev = cur;
		cur += m_layers[i]->outputs();
	}
	return os.str();
}

void GNeuralNet::biasMask(GVec& mask)
{
	size_t posWeights = 0;
	GVecWrapper w;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GLayer& lay = *m_layers[i];
		size_t wc = lay.weightCount();
		w.setData(mask, posWeights, wc);
		posWeights += wc;
		lay.biasMask(w);
	}
	GAssert(posWeights == mask.size());
}

size_t GNeuralNet::layerStart(size_t layer)
{
	size_t posWeights = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
        if(i == layer)
            return posWeights;
		GLayer& lay = *m_layers[i];
		size_t wc = lay.weightCount();
		posWeights += wc;
	}
	throw Ex("Layer index out of range");
}

void GNeuralNet_finiteDifferencingTest(GNeuralNet& nn, GVec& x)
{
	size_t outputCount = nn.outputs();

	// Do it with finite differencing
	double epsilon = 1e-6;
	GMatrix measured(outputCount, nn.weightCount());
	GVec xx;
	GVec predNegCopy;
	GVec& weights = nn.weights;
	for(size_t i = 0; i < nn.weightCount(); i++)
	{
		weights[i] -= epsilon / 2.0;
		predNegCopy.copy(nn.forwardProp(x));
		weights[i] += epsilon;
		GVec& predPos = nn.forwardProp(x);
		weights[i] -= epsilon / 2.0;
		for(size_t j = 0; j < outputCount; j++)
			measured[j][i] = (predPos[j] - predNegCopy[j]) / epsilon;
	}

	// Do it with back-prop
	GMatrix computed(outputCount, nn.gradCount());
	GVec& pred = nn.forwardProp(x);
	GVec targ;
	targ.copy(pred);
	for(size_t i = 0; i < outputCount; i++)
	{
		targ[i] += 1.0;
		nn.computeBlame(targ);
		nn.backpropagate();
		targ[i] -= 1.0;
		nn.gradient.fill(0.0);
		nn.updateGradient();
		computed[i].copy(nn.gradient);
	}

	//std::cout << "\n\nMeasured:\n" << to_str(measured) << "\n";
	//std::cout << "Computed:\n" << to_str(computed) << "\n";

	// Check results
	double sum = 0.0;
	double sum_of_squares = 0.0;
	for(size_t i = 0; i < outputCount; i++)
	{
		for(size_t j = 0; j < nn.weightCount(); j++)
		{
			if(std::abs(measured[i][j] - computed[i][j]) > 1e-5)
				throw Ex("off by too much");
			sum += computed[i][j];
			sum_of_squares += (computed[i][j] * computed[i][j]);
		}
	}
	double ex = sum / (outputCount * nn.weightCount());
	double exx = sum_of_squares / (outputCount * nn.weightCount());
	if(std::sqrt(exx - ex * ex) < 0.01)
		throw Ex("Not enough deviation");
}

void GNeuralNet_testLinearAndTanh()
{
	GNeuralNet nn;
	nn.add(new GBlockLinear(3, 5));
	nn.add(new GBlockTanh(5));
	nn.add(new GBlockLinear(5, 2));

	GRand rand(0);
	GVec weights(nn.weightCount());
	weights.fillNormal(rand);
	nn.init(weights);

	GVec x(nn.inputs());
	x.fillNormal(rand);

	nn.finiteDifferencingTest();
}

void GNeuralNet_testConvolutional1()
{
	GNeuralNet nn;
	nn.add(new GBlockConv({4, 4}, {3, 3}, {4, 4}));

	GRand rand(0);
	GVec weights(nn.weightCount());
	nn.init(weights);

	weights.fillNormal(rand);
	GVec x(nn.inputs());
	x.fillNormal(rand);

	nn.finiteDifferencingTest();
}

void GNeuralNet_testConvolutional3()
{
	GNeuralNet nn;
	nn.add(new GBlockConv({4, 4}, {3, 3, 2}, {4, 4, 2}));
	nn.add(new GBlockLeakyRectifier(16 * 2));
	nn.add(new GBlockConv({4, 4, 2}, {3, 3, 2}, {4, 4}));
	nn.add(new GBlockLeakyRectifier(16));

	GRand rand(123);
	GVec weights(nn.weightCount());
	weights.fillNormal(rand);
	nn.init(weights);
	GVec x(nn.inputs());
	x.fillNormal(rand);

	GNeuralNet_finiteDifferencingTest(nn, x);
}

void GNeuralNet_testSerializationRoundTrip()
{
	// Make a random neural net
	GNeuralNet nn;
	nn.add(new GBlockLinear(1, 3));
	nn.add(new GBlockTanh(3));
	nn.add(new GBlockLinear(3, 2));
	nn.add(new GBlockTanh(2));
	nn.add(new GBlockLinear(2, 1));
	nn.add(new GBlockTanh(1));
	GVec weights(nn.weightCount());
	nn.init(weights);
	GRand rand(0);
	weights.fillNormal(rand, 0.3);

	// Make a prediction
	GVec in(1);
	in[0] = 0.5;
	GVec& pred = nn.forwardProp(in);

	// Roundtrip through serialization
	GDom doc;
	GDomNode* pNode = nn.serialize(&doc);
	GNeuralNet nn2(pNode, rand);

	// Make the same prediction
	GVec& pred2 = nn.forwardProp(in);
	if(std::abs(pred[0] - pred2[0]) > 1e-9)
		throw Ex("failed");
}
/*
void GNeuralNet_test_decrementWidthPositive()
{
	GNeuralNet nn;
	nn.add(new GBlockLinear(2, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 2));
	nn.add(new GBlockHinge(2));
	GVec w({0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11, 12, 13, 14,
		15, 16, 17,
		18, 19, 20,
		0, 0, 0,
		24, 25, 26,
		27, 28, 29, 30, 31, 32,
		33, 34,
		35, 36,
		37, 38,
		0, 0,
		41, 42, 43, 44});
	if(w.size() != nn.weightCount())
		throw Ex("problem with test");

	//std::cout << "Bef:\n" << nn.to_str("", &w) << "\n";

	GVec* pNewWeights = nn.decrementWidth(&w, 0, 4);
	std::unique_ptr<GVec> hNewWeights(pNewWeights);
	//std::cout << "Aft:\n" << nn.to_str("", pNewWeights) << "\n";

	GVec targ({0, 2,
		3, 5,
		6, 8,
		9, 11, 12, 14,
		15, 16,
		18, 19,
		24, 25,
		27, 28, 30, 31,
		33, 34,
		35, 36,
		37, 38,
		41, 42, 43, 44});
	if(pNewWeights->squaredDistance(targ) > 1e-8)
		throw Ex("Failed");
}

void GNeuralNet_test_decrementWidthNegative()
{
	GNeuralNet nn;
	nn.add(new GBlockLinear(2, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 2));
	nn.add(new GBlockHinge(2));
	GVec w({0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11, 12, 13, 14,
		15, 16, 17,
		18, 19, 20,
		0, 22, 0, // <---- The 22 should cause this to be unable to decrement the width
		24, 25, 26,
		27, 28, 29, 30, 31, 32,
		33, 34,
		35, 36,
		37, 38,
		0, 0,
		41, 42, 43, 44});
	if(w.size() != nn.weightCount())
		throw Ex("problem with test");

	//std::cout << "Bef:\n" << nn.to_str("", &w) << "\n";

	GVec* pNewWeights = nn.decrementWidth(&w, 0, 4);
	if(pNewWeights != nullptr)
		throw Ex("Should not have succeeded");
}

void GNeuralNet_test_drop()
{
	GNeuralNet nn;
	nn.add(new GBlockLinear(2, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 2));
	nn.add(new GBlockHinge(2));
	GVec w(nn.weightCount());
	for(size_t i = 0; i < nn.weightCount(); i++)
		w[i] = i;
	//std::cout << "Bef:\n" << nn.to_str("", &w) << "\n";

	GVec* pNewWeights1 = nn.drop(2, &w);
	std::unique_ptr<GVec> hNewWeights1(pNewWeights1);
	GVec* pNewWeights2 = nn.drop(2, pNewWeights1);
	std::unique_ptr<GVec> hNewWeights2(pNewWeights2);
	//std::cout << "Aft:\n" << nn.to_str("", pNewWeights2) << "\n";

	GVec targ({0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11, 12, 13, 14,
		33, 34,
		35, 36,
		37, 38,
		39, 40,
		41, 42, 43, 44});
	if(pNewWeights2->squaredDistance(targ) > 1e-8)
		throw Ex("Failed");
}

void GNeuralNet_test_insert()
{
	GRand rand(0);
	GNeuralNet nn;
	nn.add(new GBlockLinear(2, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 3));
	nn.add(new GBlockHinge(3));
	nn.add(new GBlockLinear(3, 2));
	nn.add(new GBlockHinge(2));
	GVec w(nn.weightCount());
	for(size_t i = 0; i < nn.weightCount(); i++)
		w[i] = i;
	//std::cout << "Bef:\n" << nn.to_str("", &w) << "\n";

	GVec* pNewWeights1 = nn.insert(4, new GBlockLinear(3, 3), &w, rand);
	std::unique_ptr<GVec> hNewWeights1(pNewWeights1);
	GVec* pNewWeights2 = nn.insert(5, new GBlockHinge(3), pNewWeights1, rand);
	std::unique_ptr<GVec> hNewWeights2(pNewWeights2);
	//std::cout << "Aft:\n" << nn.to_str("", pNewWeights2) << "\n";

	GVec targ({0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11, 12, 13, 14,
		15, 16, 17,
		18, 19, 20,
		21, 22, 23,
		24, 25, 26,
		27, 28, 29, 30, 31, 32,
		0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 0, 0.1, 0.1, 0.1,
		33, 34,
		35, 36,
		37, 38,
		39, 40,
		41, 42, 43, 44});
	if(pNewWeights2->squaredDistance(targ) > 1e-8)
		throw Ex("Failed");
}

void GNeuralNet_test_lstm()
{
	size_t data_len = 1000;
	GMatrix feat(data_len, 4);
	GMatrix lab(data_len, 1);
	double lab_val = 0.0;
	for(size_t i = 0; i < data_len; i++)
	{
		feat[i][0]
	}
}
*/
// static
void GNeuralNet::test()
{
	GNeuralNet_testLinearAndTanh();
	GNeuralNet_testConvolutional1();
	GNeuralNet_testConvolutional3();
	GNeuralNet_testSerializationRoundTrip();
/*
	GNeuralNet_test_drop();
	GNeuralNet_test_insert();
	GNeuralNet_test_decrementWidthPositive();
	GNeuralNet_test_decrementWidthNegative();
*/
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
		oss << line_prefix << "  " << to_fixed_str(i, 2, ' ') << ") ";
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

*/







void GBlockResidual::forwardProp()
{
	GNeuralNet::forwardProp();
	size_t j = 0;
	for(size_t i = 0; i < output.size(); i++)
	{
		output[i++] += input[j++];
		if(j >= input.size())
			j = 0;
	}
}

void GBlockResidual::backProp()
{
	GNeuralNet::backProp();
	size_t j = 0;
	for(size_t i = 0; i < outBlame.size(); i++)
	{
		inBlame[i++] += outBlame[j++];
		if(j >= inBlame.size())
			j = 0;
	}
}










GNeuralNetLearner::GNeuralNetLearner()
: GIncrementalLearner(), m_pOptimizer(nullptr)
{}
/*
GNeuralNetLearner::GNeuralNetLearner(const GDomNode* pNode)
: GIncrementalLearner(pNode),
m_nn(pNode->get("nn"), m_rand),
m_pOptimizer(nullptr)
{
}
*/
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

// virtual
GDomNode* GNeuralNetLearner::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNeuralNetLearner");
	pNode->add(pDoc, "nn", m_nn.serialize(pDoc));
	return pNode;
}

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

// virtual
void GNeuralNetLearner::predictDistribution(const GVec& in, GPrediction* pOut)
{
	throw Ex("Sorry, this model does not predict a distribution");
}

// virtual
void GNeuralNetLearner::predict(const GVec& in, GVec& out)
{
	out.copy(m_nn.forwardProp(in));
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
	GVec& w = opt.model().weights;

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
	GNeuralNet_testMath();
}






} // namespace GClasses
