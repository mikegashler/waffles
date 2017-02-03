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
#include "GNeuralNet.h"
#include <iostream>

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
		case block_sparse: return new GBlockSparse(pNode);
		case block_linear: return new GBlockLinear(pNode);
		case block_featureselector: return new GBlockFeatureSelector(pNode);
		case block_allpairings: return new GBlockAllPairings(pNode);
		case block_fuzzy: return new GBlockFuzzy(pNode);
		case block_scalarsum: return new GBlockScalarSum(pNode);
		case block_scalarproduct: return new GBlockScalarProduct(pNode);
		case block_switch: return new GBlockSwitch(pNode);
		case block_restrictedboltzmannmachine: return new GBlockRestrictedBoltzmannMachine(pNode);
		case block_convolutional1d: return new GBlockConvolutional1D(pNode);
		//case block_convolutional2d: return new GBlockConvolutional2D(pNode);
		case block_lstm: return new GBlockLSTM(pNode);
		case block_gru: return new GBlockGRU(pNode);
		default: throw Ex("Unrecognized neural network layer type: ", GClasses::to_str((int)e));
	}
}

std::string GBlock::to_str() const
{
	std::ostringstream os;
	os << "[" << name() << ": ";
	os << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << ", Weights=" << GClasses::to_str(weightCount()) << "]";
	return os.str();
}

void GBlock::basicTest()
{
	// Make a layer
	GRand rand(0);
	GLayer lay;
	GContextLayer* pCtx = lay.newContext(rand);
	std::unique_ptr<GContextLayer> hCtx(pCtx);

	// Exercise forwardProp
	GVec in(inputs());
	GVec out(outputs());
	in.fillNormal(rand);
	double* pBef = out.data();
	forwardProp(*pCtx, in, out);
	double* pAft = out.data();
	if(pAft != pBef)
		throw Ex("forwardProp should not resize the output vector"); // because that would not be compatible with the way GLayer uses GVecWrapper

	// Exercise backProp
	GVec outBlame(outputs());
	GVec inBlame(inputs());
	outBlame.fillNormal(rand);
	pBef = inBlame.data();
	backProp(*pCtx, in, out, outBlame, inBlame);
	pAft = inBlame.data();
	if(pAft != pBef)
		throw Ex("backProp should not resize the inBlame vector"); // because that would not be compatible with the way GLayer uses GVecWrapper

	// Roundtrip through serialization
	GDom doc;
	GDomNode* pNode = serialize(&doc);
	GBlock* pBlock = deserialize(pNode);
	std::unique_ptr<GBlock> hBlock(pBlock);
	if(pBlock->inputs() != inputs())
		throw Ex("serialization problem");
	if(pBlock->outputs() != outputs())
		throw Ex("serialization problem");
	GVec in2;
	in2.copy(in);
	GVec out2(out.size());
	pBlock->forwardProp(*pCtx, in2, out2);
	if(out.squaredDistance(out2) > 1e-6)
		throw Ex("forwardProp different after serialization roundtrip");
	GVec outBlame2;
	outBlame2.copy(outBlame);
	GVec inBlame2(inBlame.size());
	pBlock->backProp(*pCtx, in2, out2, outBlame2, inBlame2);
	if(inBlame2.squaredDistance(inBlame) > 1e-6)
		throw Ex("backProp different after serialization roundtrip");
}







GBlockScalarSum::GBlockScalarSum(size_t outputs)
: GBlockWeightless()
{
	resize(outputs * 2, outputs);
}

GBlockScalarSum::GBlockScalarSum(GDomNode* pNode)
: GBlockWeightless(pNode), m_outputCount(pNode->field("outputs")->asInt())
{
}

GBlockScalarSum::~GBlockScalarSum()
{
}

GDomNode* GBlockScalarSum::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "outputs", pDoc->newInt(m_outputCount));
	return pNode;
}

void GBlockScalarSum::resize(size_t inputCount, size_t outputCount)
{
	if(outputCount * 2 != inputCount)
		throw Ex("inputCount must be 2*outputCount");
	m_outputCount = outputCount;
}

// virtual
void GBlockScalarSum::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_outputCount * 2 && output.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
		output[i] = input[i] + input[m_outputCount + i];
}

void GBlockScalarSum::forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const
{
	GAssert(in1.size() == m_outputCount && in2.size() == m_outputCount && output.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
		output[i] = in1[i] + in2[i];
}

void GBlockScalarSum::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(input.size() == 2 * m_outputCount && output.size() == m_outputCount && outBlame.size() == m_outputCount && inBlame.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
	{
		inBlame[i] += outBlame[i];
		inBlame[m_outputCount + i] += outBlame[i];
	}
}

void GBlockScalarSum::backProp2(const GVec& outBlame, GVec& inBlame1, GVec& inBlame2) const
{
	GAssert(outBlame.size() == m_outputCount && inBlame1.size() == m_outputCount && inBlame2.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
	{
		inBlame1[i] += outBlame[i];
		inBlame2[i] += outBlame[i];
	}
}








GBlockScalarProduct::GBlockScalarProduct(size_t outputs)
: GBlockWeightless()
{
	resize(outputs * 2, outputs);
}

GBlockScalarProduct::GBlockScalarProduct(GDomNode* pNode)
: GBlockWeightless(pNode), m_outputCount(pNode->field("outputs")->asInt())
{
}

GBlockScalarProduct::~GBlockScalarProduct()
{
}

GDomNode* GBlockScalarProduct::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "outputs", pDoc->newInt(m_outputCount));
	return pNode;
}

void GBlockScalarProduct::resize(size_t inputCount, size_t outputCount)
{
	if(outputCount * 2 != inputCount)
		throw Ex("inputCount must be 2*outputCount");
	m_outputCount = outputCount;
}

// virtual
void GBlockScalarProduct::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_outputCount * 2 && output.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
		output[i] = input[i] * input[m_outputCount + i];
}

void GBlockScalarProduct::forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const
{
	GAssert(in1.size() == m_outputCount &&
		in2.size() == m_outputCount &&
		output.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
		output[i] = in1[i] * in2[i];
}

void GBlockScalarProduct::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(input.size() == 2 * m_outputCount &&
		output.size() == m_outputCount &&
		outBlame.size() == m_outputCount &&
		inBlame.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
	{
		inBlame[i] += outBlame[i] * input[m_outputCount + i];
		inBlame[m_outputCount + i] += outBlame[i] * input[i];
	}
}

void GBlockScalarProduct::backProp2(const GVec& input1, const GVec& input2, const GVec& outBlame, GVec& inBlame1, GVec& inBlame2) const
{
	GAssert(input1.size() == m_outputCount &&
		input2.size() == m_outputCount &&
		outBlame.size() == m_outputCount &&
		inBlame1.size() == m_outputCount &&
		inBlame2.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
	{
		inBlame1[i] += outBlame[i] * input2[i];
		inBlame2[i] += outBlame[i] * input1[i];
	}
}







GBlockSwitch::GBlockSwitch(size_t outputs)
: GBlockWeightless()
{
	resize(outputs * 3, outputs);
}

GBlockSwitch::GBlockSwitch(GDomNode* pNode)
: GBlockWeightless(pNode), m_outputCount(pNode->field("outputs")->asInt())
{
}

GBlockSwitch::~GBlockSwitch()
{
}

GDomNode* GBlockSwitch::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "outputs", pDoc->newInt(m_outputCount));
	return pNode;
}

void GBlockSwitch::resize(size_t inputCount, size_t outputCount)
{
	if(outputCount * 3 != inputCount)
		throw Ex("inputCount must be 3*outputCount");
	m_outputCount = outputCount;
}

// virtual
void GBlockSwitch::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_outputCount * 3 && output.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
		output[i] = input[i] * input[m_outputCount + i] + (1.0 - input[i]) * input[m_outputCount + m_outputCount + i];
}

void GBlockSwitch::forwardProp3(const GVec& inA, const GVec& inB, const GVec& inC, GVec& output) const
{
	GAssert(inA.size() == m_outputCount &&
		inB.size() == m_outputCount &&
		inC.size() == m_outputCount &&
		output.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
		output[i] = inA[i] * inB[i] + (1.0 - inA[i]) * inC[i];
}

void GBlockSwitch::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(input.size() == 3 * m_outputCount &&
		output.size() == m_outputCount &&
		outBlame.size() == m_outputCount &&
		inBlame.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
	{
		inBlame[i] += (input[m_outputCount + i] - input[m_outputCount + m_outputCount + i]) * outBlame[i];
		inBlame[m_outputCount + i] += input[i] * outBlame[i];
		inBlame[m_outputCount + m_outputCount + i] += (1.0 - input[i]) * outBlame[i];
	}
}

void GBlockSwitch::backProp3(const GVec& inA, const GVec& inB, const GVec& inC, const GVec& outBlame, GVec& inBlameA, GVec& inBlameB, GVec& inBlameC) const
{
	GAssert(inA.size() == m_outputCount &&
		inB.size() == m_outputCount &&
		inC.size() == m_outputCount &&
		outBlame.size() == m_outputCount &&
		inBlameA.size() == m_outputCount &&
		inBlameB.size() == m_outputCount &&
		inBlameC.size() == m_outputCount);
	for(size_t i = 0; i < m_outputCount; i++)
	{
		inBlameA[i] += (inB[i] - inC[i]) * outBlame[i];
		inBlameB[i] += inA[i] * outBlame[i];
		inBlameC[i] += (1.0 - inA[i]) * outBlame[i];
	}
}







GMaxPooling2D::GMaxPooling2D(size_t inputCols, size_t inputRows, size_t inputChannels, size_t regionSize)
: GBlockWeightless(),
m_inputCols(inputCols),
m_inputRows(inputRows),
m_inputChannels(inputChannels),
m_regionSize(regionSize)
{
	if(inputCols % regionSize != 0)
		throw Ex("inputCols is not a multiple of regionSize");
	if(inputRows % regionSize != 0)
		throw Ex("inputRows is not a multiple of regionSize");
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
GDomNode* GMaxPooling2D::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "icol", pDoc->newInt(m_inputCols));
	pNode->addField(pDoc, "irow", pDoc->newInt(m_inputRows));
	pNode->addField(pDoc, "ichan", pDoc->newInt(m_inputChannels));
	pNode->addField(pDoc, "size", pDoc->newInt(m_regionSize));
	return pNode;
}

// virtual
void GMaxPooling2D::resize(size_t inputSize, size_t outputSize)
{
	if(inputSize != m_inputCols * m_inputRows * m_inputChannels)
		throw Ex("Changing the size of GMaxPooling2D is not supported");
	if(outputSize != m_inputChannels * m_inputCols * m_inputRows / (m_regionSize * m_regionSize))
		throw Ex("Changing the size of GMaxPooling2D is not supported");
}

// virtual
void GMaxPooling2D::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
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
						m = std::max(m, input[x]);
				}
				output[actPos++] = m;
			}
		}
	}
}

// virtual
void GMaxPooling2D::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
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
						if(input[x] > m)
						{
							m = input[x];
							maxIndex = x;
						}
						inBlame[x] = 0.0;
					}
				}
				inBlame[maxIndex] = outBlame[downPos++];
			}
		}
	}
}











GBlockActivation::GBlockActivation(size_t size)
: GBlockWeightless()
{
	resize(size, size);
}

GBlockActivation::GBlockActivation(GDomNode* pNode)
: GBlockWeightless(pNode), m_units(pNode->field("units")->asInt())
{}

GDomNode* GBlockActivation::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "units", pDoc->newInt(m_units));
	return pNode;
}

void GBlockActivation::resize(size_t in, size_t out)
{
	if(in != out)
		throw Ex("GBlockActivation must have the same number of inputs as outputs.");
	m_units = out;
}

void GBlockActivation::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	for(size_t i = 0; i < input.size(); i++)
		output[i] = eval(input[i]);
}

void GBlockActivation::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	for(size_t i = 0; i < inBlame.size(); i++)
		inBlame[i] += outBlame[i] * derivative(input[i], output[i]);
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

void GBlockLinear::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_weights.rows() - 1);
	GAssert(output.size() == m_weights.cols());
	output.copy(bias());
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);
	for(size_t i = 0; i < input.size(); i++)
		output.addScaled(input[i], m_weights.row(i));
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);
}

void GBlockLinear::forwardProp2(const GVec& in1, const GVec& in2, GVec& output) const
{
	GAssert(in1.size() + in2.size() == m_weights.rows() - 1);
	GAssert(output.size() == m_weights.cols());
	output.copy(bias());
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);
	for(size_t i = 0; i < in1.size(); i++)
		output.addScaled(in1[i], m_weights.row(i));
	for(size_t i = 0; i < in2.size(); i++)
		output.addScaled(in2[i], m_weights.row(in1.size() + i));
	GAssert(output[outputs() - 1] > -1e100 && output[outputs() - 1] < 1e100);
}

void GBlockLinear::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(outBlame.size() == m_weights.cols() && inBlame.size() == m_weights.rows() - 1);
	for(size_t i = 0; i < inBlame.size(); i++)
		inBlame[i] += outBlame.dotProduct(m_weights[i]);
}

void GBlockLinear::backProp2(const GVec& outBlame, GVec& inBlame1, GVec& inBlame2) const
{
	GAssert(outBlame.size() == m_weights.cols() && inBlame1.size() + inBlame2.size() == m_weights.rows() - 1);
	for(size_t i = 0; i < inBlame1.size(); i++)
		inBlame1[i] += outBlame.dotProduct(m_weights[i]);
	for(size_t i = 0; i < inBlame2.size(); i++)
		inBlame2[i] += outBlame.dotProduct(m_weights[inBlame1.size() + i]);
}

void GBlockLinear::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const
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

void GBlockLinear::updateGradient2(const GVec& in1, const GVec& in2, const GVec& outBlame, GVec &gradient) const
{
	GAssert(gradient.size() == weightCount(), "gradient must match the dimensions of weights!");
	GAssert(in1.size() + in2.size() == inputs());
	double *delta = gradient.data();
	for(size_t i = 0; i < in1.size(); ++i)
	{
		double act = in1[i];
		for(size_t j = 0; j < outputs(); ++j)
			*delta++ += outBlame[j] * act;
	}
	for(size_t i = 0; i < in2.size(); ++i)
	{
		double act = in2[i];
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
	for(size_t i = 0; i < inputs(); ++i)
	{
		GVec& row = m_weights[i];
		for(size_t j = 0; j < outputs(); ++j)
			row[j] += learningRate * *delta++;
	}
	GVec &b = bias();
	for(size_t j = 0; j < outputs(); ++j)
		b[j] += learningRate * *delta++;
}

size_t GBlockLinear::weightCount() const
{
	return m_weights.rows() * m_weights.cols();
}

size_t GBlockLinear::weightsToVector(double* pOutVector) const
{
	m_weights.toVector(pOutVector);
	return weightCount();
}

size_t GBlockLinear::vectorToWeights(const double* pVector)
{
	m_weights.fromVector(pVector, m_weights.rows());
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

void GBlockLinear::dropInput(size_t input)
{
	m_weights.deleteRowPreserveOrder(input);
}

void GBlockLinear::dropOutput(size_t output)
{
	m_weights.deleteColumns(output, 1);
}










GBlockSparse::GBlockSparse(size_t outputs, size_t inputs, GRand& rand, size_t connections)
: m_weights(inputs, outputs), m_bias(outputs), m_connections(connections)
{
	if(connections > inputs * outputs)
		throw Ex("Too many connections. Room for ", GClasses::to_str(inputs * outputs), ", have ", GClasses::to_str(connections));
	if(inputs >= outputs)
	{
		if(connections < inputs)
			throw Ex("Not enough connections. Require at least ", GClasses::to_str(inputs), ", have only ", GClasses::to_str(connections));
		GIndexVec** ppVecs = new GIndexVec*[inputs];
		for(size_t i = 0; i < inputs; i++)
		{
			ppVecs[i] = new GIndexVec(outputs);
			ppVecs[i]->fillIndexes();
		}
		for(size_t i = 0; i < connections; i++)
		{
			size_t in = i % inputs;
			size_t out = (i < outputs ? i : rand.next(outputs));
			m_weights.set(in, out, 1.0);
		}
		for(size_t i = 0; i < inputs; i++)
			delete(ppVecs[i]);
		delete(ppVecs);
	}
	else
	{
		if(connections < outputs)
			throw Ex("Not enough connections. Require at least ", GClasses::to_str(outputs), ", have only ", GClasses::to_str(connections));
		GIndexVec** ppVecs = new GIndexVec*[outputs];
		for(size_t i = 0; i < outputs; i++)
		{
			ppVecs[i] = new GIndexVec(inputs);
			ppVecs[i]->fillIndexes();
		}
		for(size_t i = 0; i < connections; i++)
		{
			size_t out = i % outputs;
			size_t in = (i < inputs ? i : rand.next(inputs));
			m_weights.set(in, out, 1.0);
		}
		for(size_t i = 0; i < outputs; i++)
			delete(ppVecs[i]);
		delete(ppVecs);
	}
}

GBlockSparse::GBlockSparse(size_t outputs, size_t inputs, GRand& rand, double fillPercentage)
	: GBlockSparse(outputs, inputs, rand, (size_t)(fillPercentage * (inputs * outputs))) {}

GBlockSparse::GBlockSparse(GDomNode* pNode)
: GBlock(pNode), m_weights(pNode->field("weights")), m_bias(pNode->field("bias")), m_connections(pNode->field("connections")->asInt())
{}

GDomNode* GBlockSparse::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "weights", m_weights.serialize(pDoc));
	pNode->addField(pDoc, "bias", m_bias.serialize(pDoc));
	pNode->addField(pDoc, "connections", pDoc->newInt(m_connections));
	return pNode;
}

void GBlockSparse::resize(size_t in, size_t out)
{
	if(in != inputs() || out != outputs())
	{
        m_weights.resize(in, out);
        m_bias.resize(out);
    }
}

void GBlockSparse::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_weights.rows());
	GAssert(output.size() == m_weights.cols());
	output.copy(bias());
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(GSparseMatrix::Iter it = m_weights.rowBegin(i); it != m_weights.rowEnd(i); it++)
			output[it->first] += input[i] * it->second;
	}
}

void GBlockSparse::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(outBlame.size() == m_weights.cols() && inBlame.size() == m_weights.rows());
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(GSparseMatrix::Iter it = m_weights.rowBegin(i); it != m_weights.rowEnd(i); it++)
			inBlame[i] += outBlame[it->first] * it->second;
	}
}

void GBlockSparse::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	GAssert(gradient.size() == m_connections + m_bias.size(), "gradient must match the number of connections plus bias values");
	double *delta = gradient.data();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(GSparseMatrix::Iter it = m_weights.rowBegin(i); it != m_weights.rowEnd(i); it++)
			*delta++ += input[i] * outBlame[it->first];
	}
	for(size_t j = 0; j < outputs(); ++j)
		*delta++ += outBlame[j];
}

void GBlockSparse::step(double learningRate, const GVec &gradient)
{
	GAssert(gradient.size() == m_connections + m_bias.size(), "gradient must match the number of connections plus bias values");
	const double *delta = gradient.data();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(SparseVec::iterator it = m_weights.row(i).begin(); it != m_weights.rowEnd(i); it++)
			it->second += learningRate * *delta++;
	}
	GVec &b = bias();
	for(size_t j = 0; j < outputs(); ++j)
		b[j] += learningRate * *delta++;
}

size_t GBlockSparse::weightCount() const
{
	return m_connections + m_bias.size();
}

size_t GBlockSparse::weightsToVector(double* pOutVector) const
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(GSparseMatrix::Iter it = m_weights.rowBegin(i); it != m_weights.rowEnd(i); it++)
			*(pOutVector++) = it->second;
	}
	return weightCount();
}

size_t GBlockSparse::vectorToWeights(const double* pVector)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(SparseVec::iterator it = m_weights.row(i).begin(); it != m_weights.rowEnd(i); it++)
			it->second = *(pVector++);
	}
	return weightCount();
}

void GBlockSparse::copyWeights(const GBlock* pSource)
{
	GBlockSparse* src = (GBlockSparse*)pSource;
	m_weights.clear();
	m_weights.copyFrom(&src->m_weights);
	m_bias.copy(src->bias());
	m_connections = src->m_connections;
}

void GBlockSparse::resetWeights(GRand& rand)
{
	double mag = std::max(0.03, 1.0 / sqrt((double)m_connections));
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(SparseVec::iterator it = m_weights.row(i).begin(); it != m_weights.rowEnd(i); it++)
			it->second = rand.normal() * mag;
	}
}

void GBlockSparse::perturbWeights(GRand &rand, double deviation)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(SparseVec::iterator it = m_weights.row(i).begin(); it != m_weights.rowEnd(i); it++)
			it->second += deviation * rand.normal();
	}
	for(size_t i = 0; i < outputs(); i++)
		m_bias[i] += deviation * rand.normal();
}

void GBlockSparse::maxNorm(double min, double max)
{
	throw Ex("Sorry, not implemented");
}

void GBlockSparse::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(SparseVec::iterator it = m_weights.row(i).begin(); it != m_weights.rowEnd(i); it++)
			it->second *= factor;
	}
	if(scaleBiases)
	{
		for(size_t i = 0; i < outputs(); i++)
			m_bias[i] *= factor;
	}
}

void GBlockSparse::diminishWeights(double amount, bool regularizeBiases)
{
	throw Ex("Sorry, not implemented");
}











GBlockFeatureSelector::GBlockFeatureSelector(size_t outputs, double lambda, size_t inputs)
: m_lambda(lambda)
{
	resize(inputs, outputs);
}

GBlockFeatureSelector::GBlockFeatureSelector(GDomNode* pNode)
: GBlock(pNode), m_weights(pNode->field("weights")), m_lambda(pNode->field("lambda")->asDouble())
{}

GDomNode* GBlockFeatureSelector::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "weights", m_weights.serialize(pDoc));
	pNode->addField(pDoc, "lambda", pDoc->newDouble(m_lambda));
	return pNode;
}

void GBlockFeatureSelector::resize(size_t in, size_t out)
{
	if(in == inputs() && out == outputs())
		return;
	m_weights.resize(in, out);
}

void GBlockFeatureSelector::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_weights.rows());
	GAssert(output.size() == m_weights.cols());
	for(size_t i = 0; i < input.size(); i++)
		output.addScaled(input[i], m_weights.row(i));
}

void GBlockFeatureSelector::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(outBlame.size() == m_weights.cols() && inBlame.size() == m_weights.rows());
	for(size_t i = 0; i < inBlame.size(); i++)
		inBlame[i] += outBlame.dotProduct(m_weights[i]);
}

void GBlockFeatureSelector::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	GAssert(gradient.size() == weightCount(), "gradient must match the dimensions of weights!");
	double *delta = gradient.data();
	for(size_t i = 0; i < inputs(); ++i)
	{
		double act = input[i];
		for(size_t j = 0; j < outputs(); ++j)
			*delta++ += outBlame[j] * act;
	}
}

void GBlockFeatureSelector::step(double learningRate, const GVec &gradient)
{
	GAssert(gradient.size() == weightCount(), "gradient must match the dimensions of weights!");
	const double *delta = gradient.data();
	for(size_t i = 0; i < inputs(); ++i)
	{
		GVec& row = m_weights[i];
		for(size_t j = 0; j < outputs(); ++j)
			row[j] += (learningRate * (*delta++ - m_lambda));
		row.clip(0.0, 1.0);
	}
	for(size_t j = 0; j < outputs(); j++)
		m_weights.scaleColumn(j, 1.0 / m_weights.columnSum(j));
}

size_t GBlockFeatureSelector::weightCount() const
{
	return m_weights.rows() * m_weights.cols();
}

size_t GBlockFeatureSelector::weightsToVector(double* pOutVector) const
{
	m_weights.toVector(pOutVector);
	return weightCount();
}

size_t GBlockFeatureSelector::vectorToWeights(const double* pVector)
{
	m_weights.fromVector(pVector, m_weights.rows());
	return weightCount();
}

void GBlockFeatureSelector::copyWeights(const GBlock* pSource)
{
	GBlockFeatureSelector* src = (GBlockFeatureSelector*)pSource;
	m_weights.copyBlock(src->m_weights, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
}

void GBlockFeatureSelector::resetWeights(GRand& rand)
{
	double b = 1.0 / m_weights.rows();
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		GVec& w = m_weights[i];
		for(size_t j = 0; j < w.size(); j++)
			w[j] = b;// + 0.1 * b * rand.normal();
		w.clip(0.0, 1.0);
	}
	for(size_t j = 0; j < outputs(); j++)
		m_weights.scaleColumn(j, 1.0 / m_weights.columnSum(j));
}

void GBlockFeatureSelector::perturbWeights(GRand &rand, double deviation)
{
	throw Ex("Not implemented");
}

void GBlockFeatureSelector::maxNorm(double min, double max)
{
	throw Ex("Not implemented");
}

void GBlockFeatureSelector::scaleWeights(double factor, bool scaleBiases)
{
	throw Ex("Not implemented");
}

void GBlockFeatureSelector::diminishWeights(double amount, bool regularizeBiases)
{
	throw Ex("Not implemented");
}

bool GBlockFeatureSelector::hasConverged()
{
	for(size_t i = 0; i < m_weights.cols(); i++)
	{
		if(m_weights.columnMax(i) < 1.0)
			return false;
	}
	return true;
}








GBlockAllPairings::GBlockAllPairings(size_t inputs, double lo, double hi)
: GBlockWeightless(), m_inputCount(inputs), m_lo(lo), m_hi(hi)
{
}

GBlockAllPairings::GBlockAllPairings(GDomNode* pNode)
: GBlockWeightless(pNode),
m_inputCount(pNode->field("inputs")->asInt()),
m_lo(pNode->field("lo")->asDouble()),
m_hi(pNode->field("hi")->asDouble())
{
}

GBlockAllPairings::~GBlockAllPairings()
{
}

GDomNode* GBlockAllPairings::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "inputs", pDoc->newInt(m_inputCount));
	pNode->addField(pDoc, "lo", pDoc->newDouble(m_lo));
	pNode->addField(pDoc, "hi", pDoc->newDouble(m_hi));
	return pNode;
}

void GBlockAllPairings::resize(size_t inputCount, size_t outputCount)
{
	if(outputCount != inputCount * (inputCount - 1) + 4 * inputCount)
		throw Ex("outputCount must be inputCount * (inputCount - 1) + 4 * inputCount");
	m_inputCount = inputCount;
}

size_t GBlockAllPairings::findSource(size_t outputUnit)
{
	size_t p1 = 0;
	size_t p2 = outputs() / 2;
	for(size_t i = 0; i < m_inputCount; i++)
	{
		for(size_t j = i + 1; j < m_inputCount; j++)
		{
			if(p1++ == outputUnit)
				return i;
			if(p2++ == outputUnit)
				return j;
		}
		if(p1++ == outputUnit)
			return i;
		if(p2++ == outputUnit)
			return m_inputCount;
		if(p1++ == outputUnit)
			return i;
		if(p2++ == outputUnit)
			return m_inputCount + 1;
	}
	throw Ex("out of range");
}

// virtual
void GBlockAllPairings::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(input.size() == inputs() && output.size() == outputs());
	size_t p1 = 0;
	size_t p2 = outputs() / 2;
	for(size_t i = 0; i < m_inputCount; i++)
	{
		for(size_t j = i + 1; j < m_inputCount; j++)
		{
			output[p1++] = input[i];
			output[p2++] = input[j];
		}
		output[p1++] = input[i];
		output[p2++] = m_lo;
		output[p1++] = input[i];
		output[p2++] = m_hi;
	}
	GAssert(p1 == outputs() / 2 && p2 == outputs());
}

void GBlockAllPairings::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(input.size() == m_inputCount && output.size() == outputs() && outBlame.size() == outputs() && inBlame.size() == m_inputCount);
	size_t p1 = 0;
	size_t p2 = outputs() / 2;
	for(size_t i = 0; i < m_inputCount; i++)
	{
		for(size_t j = i + 1; j < m_inputCount; j++)
		{
			inBlame[i] += outBlame[p1++];
			inBlame[j] += outBlame[p2++];
		}
		inBlame[i] += outBlame[p1++];
		p2++;
		inBlame[i] += outBlame[p1++];
		p2++;
	}
	GAssert(p1 == outputs() / 2 && p2 == outputs());
}










GBlockFuzzy::GBlockFuzzy(size_t outputs)
: GBlock()
{
	resize(outputs * 2, outputs);
}

GBlockFuzzy::GBlockFuzzy(GDomNode* pNode)
: GBlock(pNode), m_alpha(pNode->field("alpha"))
{}

GDomNode* GBlockFuzzy::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "alpha", m_alpha.serialize(pDoc));
	return pNode;
}

void GBlockFuzzy::resize(size_t in, size_t out)
{
	if(in != out * 2)
		throw Ex("GBlockFuzzy requires two inputs for each output");
	if(in == inputs() && out == outputs())
		return;
	m_alpha.resize(out);
}

void GBlockFuzzy::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	GAssert(m_alpha.size() == output.size());
	GAssert(input.size() == 2 * output.size());
	for(size_t i = 0; i < output.size(); i++)
		output[i] = GMath::fuzzy(input[i], input[m_alpha.size() + i], m_alpha[i]);
}

void GBlockFuzzy::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	GAssert(outBlame.size() == outputs());
	GAssert(inBlame.size() == inputs());
	for(size_t i = 0; i < outBlame.size(); i++)
	{
		double a = m_alpha[i];
		inBlame[i] *= (input[output.size() + i] + a) / (std::abs(a) + 1.0);
		inBlame[outBlame.size() + i] *= (input[i] + a) / (std::abs(a) + 1.0);
	}
}

void GBlockFuzzy::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	GAssert(gradient.size() == m_alpha.size());
	for(size_t i = 0; i < m_alpha.size(); i++)
	{
		double a = m_alpha[i];
		double aa = std::abs(m_alpha[i]);
		double x = input[i];
		double y = input[m_alpha.size() + i];
		if(aa < 0.001)
			a = -a; // This makes it robust to the discontinuity in the derivative that occurs when a=0.
		gradient[i] += outBlame[i] * (aa * (x + y) - a * (x * y + 1.0)) / (aa * (aa + 1.0) * (aa + 1.0));
	}
}

void GBlockFuzzy::step(double learningRate, const GVec &gradient)
{
	m_alpha.addScaled(learningRate, gradient);
	m_alpha.clip(-1.0, 1.0);
}

size_t GBlockFuzzy::weightCount() const
{
	return m_alpha.size();
}

size_t GBlockFuzzy::weightsToVector(double* pOutVector) const
{
	for(size_t i = 0; i < m_alpha.size(); i++)
		*(pOutVector++) = m_alpha[i];
	return m_alpha.size();
}

size_t GBlockFuzzy::vectorToWeights(const double* pVector)
{
	for(size_t i = 0; i < m_alpha.size(); i++)
		m_alpha[i] = *(pVector++);
	return m_alpha.size();
}

void GBlockFuzzy::copyWeights(const GBlock* pSource)
{
	GBlockFuzzy* src = (GBlockFuzzy*) pSource;
	m_alpha.copy(src->m_alpha);
}

void GBlockFuzzy::resetWeights(GRand& rand)
{
	for(size_t i = 0; i < m_alpha.size(); i++)
		m_alpha[i] = rand.uniform() * 2.0 - 1.0;
}

void GBlockFuzzy::perturbWeights(GRand &rand, double deviation)
{
	for(size_t i = 0; i < m_alpha.size(); i++)
		m_alpha[i] += rand.normal() * deviation;
	m_alpha.clip(-1.0, 1.0);
}

void GBlockFuzzy::maxNorm(double min, double max)
{
	throw Ex("Not implemented");
}

void GBlockFuzzy::scaleWeights(double factor, bool scaleBiases)
{
	m_alpha *= factor;
}

void GBlockFuzzy::diminishWeights(double amount, bool regularizeBiases)
{
	m_alpha.regularizeL1(amount);
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
void GBlockMaxOut::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
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

void GBlockMaxOut::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
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

void GBlockMaxOut::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const
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
void GBlockRestrictedBoltzmannMachine::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
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

void GBlockRestrictedBoltzmannMachine::drawSample(GContext& ctx, size_t iters, GVec& output, GVec& input)
{
	for(size_t i = 0; i < output.size(); i++)
		output[i] = ((ctx.m_rand.next() & 1) == 0 ? 0.0 : 1.0);
	for(size_t i = 0; i < iters; i++)
	{
		feedBackward(output, input);
		forwardProp(ctx, input, output);
		resampleHidden(ctx.m_rand, output);
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
void GBlockRestrictedBoltzmannMachine::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	m_weights.multiply(outBlame, inBlame, true);
}

void GBlockRestrictedBoltzmannMachine::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec& gradient) const
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
void GBlockConvolutional1D::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
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
void GBlockConvolutional1D::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
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
				const GVec& w = m_kernels[kern++];
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

void GBlockConvolutional1D::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const
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













size_t GBlockConvolutional2D::Image::npos = (size_t) -1;

GBlockConvolutional2D::Image::Image(size_t _width, size_t _height, size_t _channels)
: data(nullptr), width(_width), height(_height), channels(_channels),
interlaced(true), dx(0), dy(0), dz(0), px(0), py(0), sx(1), sy(1), invertStride(false), flip(false)
{}

GBlockConvolutional2D::Image::Image(GVec* _data, const Image& copyMyParams)
: data(_data), width(copyMyParams.width), height(copyMyParams.height), channels(copyMyParams.channels),
interlaced(copyMyParams.interlaced), dx(copyMyParams.dx), dy(copyMyParams.dy), dz(copyMyParams.dz),
px(copyMyParams.px), py(copyMyParams.py), sx(copyMyParams.sx), sy(copyMyParams.sy),
invertStride(copyMyParams.invertStride), flip(copyMyParams.flip)
{}

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
  m_kernelImage(kWidth, kHeight, channels),
  m_deltaImage(kWidth, kHeight, channels),
  m_inputImage(width, height, channels),
  m_upStreamErrorImage(width, height, channels),
  m_actImage(m_outputWidth, m_outputHeight, kCount),
  m_errImage(m_outputWidth, m_outputHeight, kCount)
{
}


GBlockConvolutional2D::GBlockConvolutional2D(size_t kWidth, size_t kHeight, size_t kCount)
: GBlock(),
  m_width(0), m_height(0), m_channels(0),
  m_kWidth(kWidth), m_kHeight(kHeight),
  m_outputWidth(0), m_outputHeight(0),
  m_bias(kCount),
  m_kernels(kCount, 0),
  m_kernelImage(kWidth, kHeight, 0),
  m_deltaImage(kWidth, kHeight, 0),
  m_inputImage(0, 0, 0),
  m_upStreamErrorImage(0, 0, 0),
  m_actImage(0, 0, 0),
  m_errImage(0, 0, 0)
{}

GBlockConvolutional2D::GBlockConvolutional2D(GDomNode* pNode)
: GBlock(pNode),
  m_width(pNode->field("width")->asInt()), m_height(pNode->field("height")->asInt()), m_channels(pNode->field("channels")->asInt()),
  m_kWidth(pNode->field("kWidth")->asInt()), m_kHeight(pNode->field("kHeight")->asInt()),
  m_outputWidth(pNode->field("outputWidth")->asInt()), m_outputHeight(pNode->field("outputHeight")->asInt()),
  m_bias(pNode->field("bias")),
  m_kernels(pNode->field("kernels")),
  m_kernelImage(m_kWidth, m_kHeight, m_channels),
  m_deltaImage(m_kWidth, m_kHeight, m_channels),
  m_inputImage(m_width, m_height, m_channels),
  m_upStreamErrorImage(m_width, m_height, m_channels),
  m_actImage(m_outputWidth, m_outputHeight, m_kernels.rows()),
  m_errImage(m_outputWidth, m_outputHeight, m_kernels.rows())
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

void GBlockConvolutional2D::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	Image inputImage(const_cast<GVec*>(&input), m_inputImage);
	Image n(&output, m_actImage);
	Image k(nullptr, m_kernelImage);
	output.fill(0.0);
	for(n.dz = 0; n.dz < n.channels; ++n.dz)
	{
		k.data = const_cast<GVec*>(&m_kernels[n.dz]);
		convolve(inputImage, k, n);
		for(size_t y = 0; y < n.height; ++y)
			for(size_t x = 0; x < n.width; ++x)
				n.at(x, y) += m_bias[n.dz];
	}
}

void GBlockConvolutional2D::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	Image err(const_cast<GVec*>(&outBlame), m_errImage);
	Image upErr(&inBlame, m_upStreamErrorImage);
	Image k(nullptr, m_kernelImage);
	inBlame.fill(0.0);
	upErr.px = m_inputImage.px;
	upErr.py = m_inputImage.py;

	err.invertStride = true, err.sx = m_inputImage.sx, err.sy = m_inputImage.sy;
	for(upErr.dz = 0; upErr.dz < upErr.channels; ++upErr.dz)
	{
		for(err.dz = 0; err.dz < err.channels; ++err.dz)
		{
			k.data = const_cast<GVec*>(&m_kernels[err.dz]);
			k.flip = true, k.dz = upErr.dz;
			convolveFull(err, k, upErr, 1);
			k.flip = false, k.dz = 0;
		}
	}
}

void GBlockConvolutional2D::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	Image err(const_cast<GVec*>(&outBlame), m_errImage);
	Image in(const_cast<GVec*>(&input), m_inputImage);
	size_t count = m_kernels.cols();
	GVecWrapper delta(gradient.data(), count);
	Image delt(&delta.vec(), m_deltaImage);
	for(err.dz = 0; err.dz < err.channels; ++err.dz)
	{
		double* biasDelta = delt.data->data() + count;
		delta.vec().fill(0.0);
		for(in.dz = delt.dz = 0; in.dz < in.channels; ++in.dz, ++delt.dz)
			for(in.dy = 0; in.dy < err.height; ++in.dy)
				for(in.dx = 0; in.dx < err.width; ++in.dx)
				{
					addScaled(in, err.read(in.dx, in.dy), delt);
					*biasDelta += err.read(in.dx, in.dy);
				}
		delt.dz = 0;
		delta.setData(delta.vec().data() + count + 1);
	}
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

double GBlockConvolutional2D::filterSum(const Image &in, const Image &filter, size_t channels) const
{
	double output = 0.0;
	for(size_t z = 0; z < channels; ++z)
		for(size_t y = 0; y < filter.height; ++y)
			for(size_t x = 0; x < filter.width; ++x)
				output += in.read(x, y, z) * filter.read(x, y, z);
	return output;
}

void GBlockConvolutional2D::addScaled(const Image &in, double scalar, Image &out) const
{
	for(size_t y = 0; y < out.height; ++y)
		for(size_t x = 0; x < out.width; ++x)
			out.at(x, y) += in.read(x, y) * scalar;
}

void GBlockConvolutional2D::convolve(const Image &in, const Image &filter, Image &out, size_t channels) const
{
	size_t x, y;
	if(channels == none)
		channels = filter.channels;
	for(y = 0, in.dy = out.py; y < out.height; ++y, ++in.dy)
		for(x = 0, in.dx = out.px; x < out.width; ++x, ++in.dx)
			out.at(in.dx, in.dy, 0) += filterSum(in, filter, channels);
	in.dx = in.dy = 0;
}

void GBlockConvolutional2D::convolveFull(const Image &in, const Image &filter, Image &out, size_t channels) const
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

	m_actImage.width = m_outputWidth;
	m_actImage.height = m_outputHeight;

	m_errImage.width = m_outputWidth;
	m_errImage.height = m_outputHeight;
}













// virtual
void GBlockRecurrent::forwardProp(GContext& ctx, const GVec& input, GVec& output) const
{
	throw Ex("This method should not be called. Call GContextRecurrent::forwardProp instead.");
}

// virtual
void GBlockRecurrent::backProp(GContext& ctx, const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	throw Ex("This method should not be called. Call GContextRecurrent::backProp instead.");
}

// virtual
void GBlockRecurrent::updateGradient(GContext& ctx, const GVec& input, const GVec& outBlame, GVec& gradient) const
{
	throw Ex("This method should not be called. Call GContextRecurrent::updateGradient instead.");
}

#ifndef MIN_PREDICT
// static
double GBlockRecurrent::testEngine(GNeuralNet& nn)
{
	// Make some sequential data
	GRand rand(0);
	GMatrix f(3000, 1);
	GMatrix l(f.rows(), 1);
	for(size_t i = 0; i < f.rows(); i++)
		f[i][0] = rand.normal();
	l[0][0] = 0.0;
	l[1][0] = 0.0;
	for(size_t i = 2; i < f.rows(); i++)
		l[i][0] = f[i - 2][0] * f[i - 1][0];

	// Train
	nn.resize(f.cols(), l.cols());
	nn.resetWeights(rand);
	GSGDOptimizer opt(nn);
	opt.setLearningRate(0.01);
	for(size_t epoch = 0; epoch < 5000; epoch++)
	{
		opt.resetState();
		for(size_t i = 0; i < f.rows(); i++)
			opt.optimizeIncremental(f[i], l[i]);
	}

	// Test
	double sse = 0.0;
	size_t testCount = 1;
	GContextNeuralNet* ctx = nn.newContext(rand);
	for(size_t test = 0; test < testCount; test++)
	{
		// Make some test data
		for(size_t i = 0; i < 30; i++)
			f[i][0] = rand.normal();
		for(size_t i = 2; i < 30; i++)
			l[i][0] = f[i - 2][0] * f[i - 1][0];

		// predict
		ctx->resetState();
		for(size_t i = 0; i < 30; i++)
			nn.forwardProp(*ctx, f[i], l[1000 + i]);

		// evaluate
		for(size_t i = 5; i < 30; i++)
		{
			double err = l[i][0] - l[1000 + i][0];
			sse += (err * err);
		}
	}
	return std::sqrt(sse / (25 * testCount));
}
#endif







GContextRecurrent::GContextRecurrent(GRand& rand, GBlockRecurrent& block)
: GContext(rand),
m_block(block),
m_emptyBlame(block.outputs()),
m_bogusBlame(block.inputs()),
m_pos(0)
{
	m_emptyBlame.fill(0.0);
	m_bogusBlame.fill(0.0);
}

GContextRecurrent::~GContextRecurrent()
{
	for(size_t i = 0; i < m_contextHistory.size(); i++)
		delete(m_contextHistory[i]);
	for(size_t i = 0; i < m_inputHistory.size(); i++)
		delete(m_inputHistory[i]);
	for(size_t i = 0; i < m_contextSpares.size(); i++)
		delete(m_contextSpares[i]);
	for(size_t i = 0; i < m_inputSpares.size(); i++)
		delete(m_inputSpares[i]);
}

void GContextRecurrent::resetState()
{
	GAssert(m_contextHistory.size() == m_inputHistory.size());
	while(m_contextHistory.size() > 1)
	{
		m_contextSpares.push_back(m_contextHistory.back());
		m_contextHistory.pop_back();
		m_inputSpares.push_back(m_inputHistory.back());
		m_inputHistory.pop_back();
	}
	if(m_contextHistory.size() < 1)
	{
		m_contextHistory.push_back(m_block.newContext(m_rand));
		m_inputHistory.push_back(new GVec(m_block.inputs()));
	}
	GAssert(m_contextHistory.size() == m_inputHistory.size());
	m_contextHistory[0]->resetState();
	m_pos = 0;
}

void GContextRecurrent::forwardProp(const GVec& input, GVec& output)
{
	if(m_contextHistory.size() != 1)
		throw Ex("With recurrent models, resetState() must be called before each prediction sequence begins");
	GContextRecurrentInstance* pCtx = m_contextHistory[0];
	pCtx->forwardProp(pCtx, input, output);
}

void GContextRecurrent::forwardPropThroughTime(const GVec& input, GVec& output)
{
	// Store the input
	GAssert(m_contextHistory.size() == m_inputHistory.size());
	if(m_contextHistory.size() < m_block.depth() + 1)
	{
		if(m_contextHistory.size() < 1)
			throw Ex("With recurrent models, resetState() must be called before each training sequence begins");
		if(m_contextSpares.size() > 0)
		{
			GAssert(m_inputSpares.size() == m_contextSpares.size());
			m_contextHistory.push_back(m_contextSpares.back());
			m_contextSpares.pop_back();
			m_inputHistory.push_back(m_inputSpares.back());
			m_inputSpares.pop_back();
		}
		else
		{
			m_contextHistory.push_back(m_block.newContext(m_rand));
			m_inputHistory.push_back(new GVec(input.size()));
		}
	}
	else
		m_pos = (m_pos + 1) % m_contextHistory.size();
	size_t lastIndex = (m_pos + m_contextHistory.size() - 1) % m_contextHistory.size();
	(*m_inputHistory[lastIndex]).copy(input);

	// Do the forward propagation
	GContextRecurrentInstance* pPrev = m_contextHistory[m_pos];
	for(size_t i = 1; i < m_contextHistory.size(); i++)
	{
		size_t index = (m_pos + i) % m_contextHistory.size();
		GContextRecurrentInstance* pCur = m_contextHistory[index];
		pCur->forwardProp(pPrev, *m_inputHistory[index], output);
		pPrev = pCur;
	}
}

void GContextRecurrent::backPropThroughTime(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	const GVec* pOutBlame = &outBlame;
	GVec* pInBlame = &inBlame;
	m_contextHistory[(m_pos + m_contextHistory.size() - 1) % m_contextHistory.size()]->clearBlame(); // Clear blame for the current context
	for(size_t i = m_contextHistory.size() - 1; i > 0; i--)
	{
		// Clear the blame in the previous context
		size_t prevIndex = (m_pos + i + m_contextHistory.size() - 1) % m_contextHistory.size();
		GContextRecurrentInstance* pPrev = m_contextHistory[prevIndex];
		pPrev->clearBlame();

		// Backpropagate
		size_t index = (m_pos + i) % m_contextHistory.size();
		m_contextHistory[index]->backProp(pPrev, *pOutBlame, *pInBlame);

		pOutBlame = &m_emptyBlame;
		pInBlame = &m_bogusBlame;
	}
}

void GContextRecurrent::updateGradient(const GVec& input, const GVec& outBlame, GVec& gradient) const
{
	GAssert(gradient.size() == m_block.weightCount(), "gradient size must match the number of weights!");
	GContextRecurrentInstance* pPrev = m_contextHistory[m_pos];
	for(size_t i = 1; i < m_contextHistory.size(); i++)
	{
		size_t index = (m_pos + i) % m_contextHistory.size();
		GContextRecurrentInstance* pCur = m_contextHistory[index];
		pCur->updateGradient(pPrev, *m_inputHistory[index], gradient);
		pPrev = pCur;
	}
}








GBlockLSTM::GBlockLSTM(size_t outputs, size_t inputs)
: GBlockRecurrent(),
m_product(outputs),
m_switch(outputs),
m_logistic(outputs),
m_tanh(outputs),
m_write(outputs, outputs + inputs),
m_val(outputs, outputs + inputs),
m_read(outputs, outputs + inputs)
{
}

GBlockLSTM::GBlockLSTM(GDomNode* pNode)
: GBlockRecurrent(pNode),
m_product(pNode->field("product")),
m_switch(pNode->field("switch")),
m_logistic(pNode->field("logsitic")),
m_tanh(pNode->field("tanh")),
m_write(pNode->field("write")),
m_val(pNode->field("val")),
m_read(pNode->field("read"))
{
	size_t units = m_write.outputs();
	resize(units, units);
}

GBlockLSTM::~GBlockLSTM()
{
}

GContextRecurrentInstance* GBlockLSTM::newContext(GRand& rand)
{
	return new GContextLSTM(rand, *this);
}

// virtual
void GBlockLSTM::resize(size_t inputs, size_t outputs)
{
	if(inputs != outputs)
		throw Ex("Expected the same number of inputs and outputs");
	m_product.resize(outputs * 2, outputs);
	m_switch.resize(outputs * 3, outputs);
	m_logistic.resize(outputs, outputs);
	m_tanh.resize(outputs, outputs);
	m_write.resize(outputs + inputs, outputs);
	m_val.resize(outputs + inputs, outputs);
	m_read.resize(outputs + inputs, outputs);
}

// virtual
GDomNode* GBlockLSTM::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "product", m_product.serialize(pDoc));
	pNode->addField(pDoc, "switch", m_switch.serialize(pDoc));
	pNode->addField(pDoc, "logistic", m_logistic.serialize(pDoc));
	pNode->addField(pDoc, "tanh", m_tanh.serialize(pDoc));
	pNode->addField(pDoc, "write", m_write.serialize(pDoc));
	pNode->addField(pDoc, "val", m_val.serialize(pDoc));
	pNode->addField(pDoc, "read", m_read.serialize(pDoc));
	return pNode;
}

size_t GBlockLSTM::weightCount() const
{
	return m_write.weightCount() + m_val.weightCount() + m_read.weightCount();
}

size_t GBlockLSTM::weightsToVector(double* pOutVector) const
{
	double* pStart = pOutVector;
	pOutVector += m_write.weightsToVector(pOutVector);
	pOutVector += m_val.weightsToVector(pOutVector);
	pOutVector += m_read.weightsToVector(pOutVector);
	return pOutVector - pStart;
}

size_t GBlockLSTM::vectorToWeights(const double* pVector)
{
	const double* pStart = pVector;
	pVector += m_write.vectorToWeights(pVector);
	pVector += m_val.vectorToWeights(pVector);
	pVector += m_read.vectorToWeights(pVector);
	return pVector - pStart;
}

void GBlockLSTM::copyWeights(const GBlock* pSource)
{
	GBlockLSTM* src = (GBlockLSTM*)pSource;
	m_write.copyWeights(&src->m_write);
	m_val.copyWeights(&src->m_val);
	m_read.copyWeights(&src->m_read);
}

void GBlockLSTM::resetWeights(GRand& rand)
{
	m_write.resetWeights(rand);
	m_val.resetWeights(rand);
	m_read.resetWeights(rand);
}

void GBlockLSTM::perturbWeights(GRand &rand, double deviation)
{
	m_write.perturbWeights(rand, deviation);
	m_val.perturbWeights(rand, deviation);
	m_read.perturbWeights(rand, deviation);
}

void GBlockLSTM::maxNorm(double min, double max)
{
	m_write.maxNorm(min, max);
	m_val.maxNorm(min, max);
	m_read.maxNorm(min, max);
}

void GBlockLSTM::scaleWeights(double factor, bool scaleBiases)
{
	m_write.scaleWeights(factor, scaleBiases);
	m_val.scaleWeights(factor, scaleBiases);
	m_read.scaleWeights(factor, scaleBiases);
}

void GBlockLSTM::diminishWeights(double amount, bool regularizeBiases)
{
	m_write.diminishWeights(amount, regularizeBiases);
	m_val.diminishWeights(amount, regularizeBiases);
	m_read.diminishWeights(amount, regularizeBiases);
}

void GBlockLSTM::step(double learningRate, const GVec &gradient)
{
	GAssert(gradient.size() == weightCount(), "gradient must match the dimensions of weights!");
	size_t wcWrite = m_write.weightCount();
	size_t wcVal = m_val.weightCount();
	size_t wcRead = m_read.weightCount();
	GConstVecWrapper g(gradient.data(), wcWrite);
	m_write.step(learningRate, g.vec());
	g.setData(gradient.data() + wcWrite, wcVal);
	m_val.step(learningRate, g.vec());
	g.setData(gradient.data() + wcWrite + wcVal, wcRead);
	m_read.step(learningRate, g.vec());
}



#ifndef MIN_PREDICT
// static
void GBlockLSTM::test()
{
	GNeuralNet nnBaseline;
	nnBaseline.add(new GBlockLinear(4));
	nnBaseline.add(new GBlockTanh());
	nnBaseline.add(new GBlockLinear(4));
	nnBaseline.add(new GBlockTanh());
	nnBaseline.add(new GBlockLinear(1));
	nnBaseline.add(new GBlockTanh());

	GNeuralNet nnLSTM;
	nnLSTM.add(new GBlockLinear(4));
	nnLSTM.add(new GBlockTanh());
	nnLSTM.add(new GBlockLSTM(4));
	nnLSTM.add(new GBlockTanh());
	nnLSTM.add(new GBlockLinear(1));
	nnLSTM.add(new GBlockTanh());

	double rmseBaseline = testEngine(nnBaseline);
std::cout << "Baseline: " << GClasses::to_str(rmseBaseline) << "\n";
	double rmseLSTM = testEngine(nnLSTM);
std::cout << "LSTM: " << GClasses::to_str(rmseLSTM) << "\n";
}
#endif







GContextLSTM::GContextLSTM(GRand& rand, GBlockLSTM& block)
: GContextRecurrentInstance(rand),
m_block(block)
{
	size_t units = block.outputs();
	m_c.resize(units);
	m_h.resize(units);
	m_f.resize(units);
	m_t.resize(units);
	m_o.resize(units);
	m_blamec.resize(units);
	m_blameh.resize(units);
	m_blamef.resize(units);
	m_blamet.resize(units);
	m_blameo.resize(units);
	m_buf1.resize(units);
	m_buf2.resize(units);
}

void GContextLSTM::forwardProp(GContextRecurrentInstance* prev, const GVec& input, GVec& output)
{
	GContextLSTM* pPrev = (GContextLSTM*)prev;

	// Compute how much to write into the memory
	m_block.m_write.forwardProp2(pPrev->m_h, input, m_buf1);
	m_block.m_logistic.forwardProp(*this, m_buf1, m_f);

	// Compute the values to write into memory
	m_block.m_val.forwardProp2(pPrev->m_h, input, m_buf1);
	m_block.m_tanh.forwardProp(*this, m_buf1, m_t);

	// Update the memory
	m_block.m_switch.forwardProp3(m_f, pPrev->m_c, m_t, m_c);

	// Compute how much to read from memory
	m_block.m_read.forwardProp2(pPrev->m_h, input, m_buf1);
	m_block.m_logistic.forwardProp(*this, m_buf1, m_o);

	// Read from memory
	m_block.m_tanh.forwardProp(*this, m_c, m_buf1);
	m_block.m_product.forwardProp2(m_o, m_buf1, m_h);
	if(output.data() != m_h.data())
		output.copy(m_h);
}

void GContextLSTM::resetState()
{
	m_c.fill(0.0);
	m_h.fill(0.0);
}

void GContextLSTM::clearBlame()
{
	m_blamec.fill(0.0);
	m_blameh.fill(0.0);
	m_blamef.fill(0.0);
	m_blamet.fill(0.0);
	m_blameo.fill(0.0);
}

void GContextLSTM::backProp(GContextRecurrentInstance* prev, const GVec& outBlame, GVec& inBlame)
{
	GContextLSTM* pPrev = (GContextLSTM*)prev;

	// Blame the read gate
	m_blameh += outBlame;
	m_buf2.fill(0.0);
	m_block.m_product.backProp2(m_o, m_buf1, m_blameh, m_blameo, m_buf2);
	m_block.m_tanh.backProp(*this, m_c, m_buf1, m_buf2, m_blamec);

	// Blame the amount to read
	m_block.m_logistic.backProp(*this, m_buf2/*ignored bogus value*/, m_o, m_blameo, m_buf1);
	m_block.m_read.backProp2(m_buf1, pPrev->m_blameh, inBlame);

	// Blame the memory
	m_block.m_switch.backProp3(m_f, pPrev->m_c, m_t, m_blamec, m_blamef, pPrev->m_blamec, m_blamet);

	// Blame the value written to memory
	m_buf2.fill(0.0);
	m_block.m_tanh.backProp(*this, m_buf2/*ignored bogus value*/, m_t, m_blamet, m_buf2);
	m_block.m_val.backProp2(m_buf2, pPrev->m_blameh, inBlame);

	// Blame the amount to write to memory
	m_buf2.fill(0.0);
	m_block.m_logistic.backProp(*this, m_buf2/*ignored bogus value*/, m_f, m_blamef, m_buf2);
	m_block.m_write.backProp2(m_buf2, pPrev->m_blameh, inBlame);
}

void GContextLSTM::updateGradient(GContextRecurrentInstance* prev, const GVec& input, GVec& gradient) const
{
	GContextLSTM* pPrev = (GContextLSTM*)prev;
	GAssert(gradient.size() == m_block.weightCount(), "gradient size must match the number of weights!");
	size_t wcWrite = m_block.m_write.weightCount();
	size_t wcVal = m_block.m_val.weightCount();
	size_t wcRead = m_block.m_read.weightCount();
	GVecWrapper g(gradient.data(), wcWrite);
	m_block.m_write.updateGradient2(pPrev->m_h, input, m_blamef, g.vec());
	g.setData(gradient.data() + wcWrite, wcVal);
	m_block.m_val.updateGradient2(pPrev->m_h, input, m_blamet, g.vec());
	g.setData(gradient.data() + wcWrite + wcVal, wcRead);
	m_block.m_read.updateGradient2(pPrev->m_h, input, m_blameo, g.vec());
}









GBlockGRU::GBlockGRU(size_t outputs, size_t inputs)
: GBlockRecurrent(),
m_product(outputs),
m_switch(outputs),
m_logistic(outputs),
m_tanh(outputs),
m_update(outputs, outputs + inputs),
m_remember(outputs, outputs + inputs),
m_val(outputs, outputs + inputs)
{
}

GBlockGRU::GBlockGRU(GDomNode* pNode)
: GBlockRecurrent(pNode),
m_product(pNode->field("product")),
m_switch(pNode->field("switch")),
m_logistic(pNode->field("logistic")),
m_tanh(pNode->field("tanh")),
m_update(pNode->field("update")),
m_remember(pNode->field("remember")),
m_val(pNode->field("val"))
{
}

GBlockGRU::~GBlockGRU()
{
}

GContextRecurrentInstance* GBlockGRU::newContext(GRand& rand)
{
	return new GContextGRU(rand, *this);
}

// virtual
void GBlockGRU::resize(size_t inputs, size_t outputs)
{
	if(inputs != outputs)
		throw Ex("Expected the same number of inputs and outputs");
	m_product.resize(outputs * 2, outputs);
	m_switch.resize(outputs * 3, outputs);
	m_logistic.resize(outputs, outputs);
	m_tanh.resize(outputs, outputs);
	m_update.resize(outputs + inputs, outputs);
	m_remember.resize(outputs + inputs, outputs);
	m_val.resize(outputs + inputs, outputs);
}

// virtual
GDomNode* GBlockGRU::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "product", m_product.serialize(pDoc));
	pNode->addField(pDoc, "switch", m_switch.serialize(pDoc));
	pNode->addField(pDoc, "logistic", m_logistic.serialize(pDoc));
	pNode->addField(pDoc, "tanh", m_tanh.serialize(pDoc));
	pNode->addField(pDoc, "update", m_update.serialize(pDoc));
	pNode->addField(pDoc, "remember", m_remember.serialize(pDoc));
	pNode->addField(pDoc, "val", m_val.serialize(pDoc));
	return pNode;
}

size_t GBlockGRU::weightCount() const
{
	return m_update.weightCount() + m_remember.weightCount() + m_val.weightCount();
}

size_t GBlockGRU::weightsToVector(double* pOutVector) const
{
	double* pStart = pOutVector;
	pOutVector += m_update.weightsToVector(pOutVector);
	pOutVector += m_remember.weightsToVector(pOutVector);
	pOutVector += m_val.weightsToVector(pOutVector);
	return pOutVector - pStart;
}

size_t GBlockGRU::vectorToWeights(const double* pVector)
{
	const double* pStart = pVector;
	pVector += m_update.vectorToWeights(pVector);
	pVector += m_remember.vectorToWeights(pVector);
	pVector += m_val.vectorToWeights(pVector);
	return pVector - pStart;
}

void GBlockGRU::copyWeights(const GBlock* pSource)
{
	GBlockGRU* src = (GBlockGRU*)pSource;
	m_update.copyWeights(&src->m_update);
	m_remember.copyWeights(&src->m_remember);
	m_val.copyWeights(&src->m_val);
}

void GBlockGRU::resetWeights(GRand& rand)
{
	m_update.resetWeights(rand);
	m_remember.resetWeights(rand);
	m_val.resetWeights(rand);
}

void GBlockGRU::perturbWeights(GRand &rand, double deviation)
{
	m_update.perturbWeights(rand, deviation);
	m_remember.perturbWeights(rand, deviation);
	m_val.perturbWeights(rand, deviation);
}

void GBlockGRU::maxNorm(double min, double max)
{
	m_update.maxNorm(min, max);
	m_remember.maxNorm(min, max);
	m_val.maxNorm(min, max);
}

void GBlockGRU::scaleWeights(double factor, bool scaleBiases)
{
	m_update.scaleWeights(factor, scaleBiases);
	m_remember.scaleWeights(factor, scaleBiases);
	m_val.scaleWeights(factor, scaleBiases);
}

void GBlockGRU::diminishWeights(double amount, bool regularizeBiases)
{
	m_update.diminishWeights(amount, regularizeBiases);
	m_remember.diminishWeights(amount, regularizeBiases);
	m_val.diminishWeights(amount, regularizeBiases);
}

void GBlockGRU::step(double learningRate, const GVec& gradient)
{
	GAssert(gradient.size() == weightCount(), "gradient must match the dimensions of weights!");
	size_t wcUpdate = m_update.weightCount();
	size_t wcRemember = m_remember.weightCount();
	size_t wcVal = m_val.weightCount();
	GConstVecWrapper g(gradient.data(), wcUpdate);
	m_update.step(learningRate, g.vec());
	g.setData(gradient.data() + wcUpdate, wcRemember);
	m_remember.step(learningRate, g.vec());
	g.setData(gradient.data() + wcUpdate + wcRemember, wcVal);
	m_val.step(learningRate, g.vec());
}

#ifndef MIN_PREDICT
// static
void GBlockGRU::test()
{
	GNeuralNet nnBaseline;
	nnBaseline.add(new GBlockLinear(4));
	nnBaseline.add(new GBlockTanh());
	nnBaseline.add(new GBlockLinear(4));
	nnBaseline.add(new GBlockTanh());
	nnBaseline.add(new GBlockLinear(1));
	nnBaseline.add(new GBlockTanh());

	GNeuralNet nnGRU;
	nnGRU.add(new GBlockLinear(4));
	nnGRU.add(new GBlockTanh());
	nnGRU.add(new GBlockGRU(4));
	nnGRU.add(new GBlockTanh());
	nnGRU.add(new GBlockLinear(1));
	nnGRU.add(new GBlockTanh());

	double rmseBaseline = testEngine(nnBaseline);
std::cout << "Baseline: " << GClasses::to_str(rmseBaseline) << "\n";
	double rmseGRU = testEngine(nnGRU);
std::cout << "GRU: " << GClasses::to_str(rmseGRU) << "\n";
}
#endif










GContextGRU::GContextGRU(GRand& rand, GBlockGRU& block)
: GContextRecurrentInstance(rand),
m_block(block)
{
	size_t units = block.outputs();
	m_h.resize(units);
	m_z.resize(units);
	m_r.resize(units);
	m_t.resize(units);
	m_blameh.resize(units);
	m_blamez.resize(units);
	m_blamer.resize(units);
	m_blamet.resize(units);
	m_buf1.resize(units);
	m_buf2.resize(units);
}

void GContextGRU::forwardProp(GContextRecurrentInstance* prev, const GVec& input, GVec& output)
{
	GContextGRU* pPrev = (GContextGRU*)prev;

	// Compute the update gate
	m_block.m_update.forwardProp2(pPrev->m_h, input, m_buf1);
	m_block.m_logistic.forwardProp(*this, m_buf1, m_z);

	// Compute the remember gate
	m_block.m_remember.forwardProp2(pPrev->m_h, input, m_buf1);
	m_block.m_logistic.forwardProp(*this, m_buf1, m_r);

	// Compute the value to write to memory
	m_block.m_product.forwardProp2(m_r, pPrev->m_h, m_buf2);
	m_block.m_val.forwardProp2(m_buf2, input, m_buf1);
	m_block.m_tanh.forwardProp(*this, m_buf1, m_t);

	// Compute the output
	m_block.m_switch.forwardProp3(m_z, m_t, pPrev->m_h, m_h);
	if(output.data() != m_h.data())
		output.copy(m_h);
}

void GContextGRU::resetState()
{
	m_h.fill(0.0);
}

void GContextGRU::clearBlame()
{
	m_blameh.fill(0.0);
	m_blamez.fill(0.0);
	m_blamer.fill(0.0);
	m_blamet.fill(0.0);
}

void GContextGRU::backProp(GContextRecurrentInstance* prev, const GVec& outBlame, GVec& inBlame)
{
	GContextGRU* pPrev = (GContextGRU*)prev;

	// Blame the output
	m_block.m_switch.backProp3(m_z, m_t, pPrev->m_h, outBlame, m_blamez, m_blamet, pPrev->m_h);

	// Blame the value written to memory
	m_buf2.fill(0.0);
	m_block.m_tanh.backProp(*this, m_buf1/*ignored bogus value*/, m_t, m_blamet, m_buf2);
	m_buf1.fill(0.0);
	m_block.m_val.backProp2(m_buf2, inBlame, m_buf1);
	m_block.m_product.backProp2(m_r, pPrev->m_h, m_buf1, m_blamer, pPrev->m_blameh);

	// Blame the remember gate
	m_buf2.fill(0.0);
	m_block.m_logistic.backProp(*this, m_buf1/*ignored bogus value*/, m_r, m_blamer, m_buf2);
	m_block.m_remember.backProp2(m_buf2, pPrev->m_blameh, inBlame);

	// Blame the update gate
	m_buf2.fill(0.0);
	m_block.m_logistic.backProp(*this, m_buf1/*ignored bogus value*/, m_z, m_blamez, m_buf2);
	m_block.m_update.backProp2(m_buf2, pPrev->m_blameh, inBlame);
}

void GContextGRU::updateGradient(GContextRecurrentInstance* prev, const GVec& input, GVec& gradient) const
{
	GContextGRU* pPrev = (GContextGRU*)prev;
	GAssert(gradient.size() == m_block.weightCount(), "gradient size must match the number of weights!");
	size_t wcUpdate = m_block.m_update.weightCount();
	size_t wcRemember = m_block.m_remember.weightCount();
	size_t wcVal = m_block.m_val.weightCount();
	GVecWrapper g(gradient.data(), wcUpdate);
	m_block.m_update.updateGradient2(pPrev->m_h, input, m_blamez, g.vec());
	g.setData(gradient.data() + wcUpdate, wcRemember);
	m_block.m_remember.updateGradient2(pPrev->m_h, input, m_blamer, g.vec());
	g.setData(gradient.data() + wcUpdate + wcRemember, wcVal);
	m_block.m_val.updateGradient2(pPrev->m_h, input, m_blamet, g.vec());
}



} // namespace GClasses
