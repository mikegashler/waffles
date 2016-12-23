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
#include <memory>
#include <string>
#include <sstream>
#include "GBlock.h"
#include "GOptimizer.h"

using std::vector;

namespace GClasses {


GLayer::GLayer()
: m_inputs(0), m_outputs(0), m_weightCount(0)
{
}

GLayer::GLayer(GDomNode* pNode)
: m_inputs(pNode->field("inputs")->asInt()),
m_outputs(pNode->field("outputs")->asInt()),
m_weightCount(0)
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
	pNode->addField(pDoc, "inputs", pDoc->newInt(m_inputs));
	pNode->addField(pDoc, "outputs", pDoc->newInt(m_outputs));
	GDomNode* pBlocks = pNode->addField(pDoc, "blocks", pDoc->newList());
	for(size_t i = 0; i < m_blocks.size(); i++)
		pBlocks->addItem(pDoc, m_blocks[i]->serialize(pDoc));
	return pNode;
}

GContextLayer* GLayer::newContext() const
{
	return new GContextLayer(*this);
}

void GLayer::add(GBlock* pBlock, size_t inPos)
{
	m_inputs = 0;
	m_outputs = 0;
	pBlock->setInPos(inPos);
	m_blocks.push_back(pBlock);
}

void GLayer::recount()
{
	// Count the inputs and outputs
	m_inputs = 0;
	m_outputs = 0;
	m_weightCount = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		size_t inPos = m_blocks[i]->inPos();
		size_t inSize = m_blocks[i]->inputs();
		if(inSize == 0)
			throw Ex("The number of inputs for this block was not specified, and could not be determined");
		size_t outSize = m_blocks[i]->outputs();
		if(outSize == 0)
			throw Ex("Empty block");
		m_inputs = std::max(m_inputs, inPos + inSize);
		m_outputs += outSize;
		m_weightCount += m_blocks[i]->weightCount();
	}
}

size_t GLayer::inputs() const
{
	if(m_inputs == 0)
		((GLayer*)this)->recount();
	return m_inputs;
}

size_t GLayer::outputs() const
{
	if(m_outputs == 0)
		((GLayer*)this)->recount();
	return m_outputs;
}

size_t GLayer::weightCount() const
{
	if(m_weightCount == 0)
		((GLayer*)this)->recount();
	return m_weightCount;
}

void GLayer::resetWeights(GRand& rand)
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		m_blocks[i]->resetWeights(rand);
}

size_t GLayer::weightsToVector(double* pOutVector) const
{
	size_t total = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		size_t n = m_blocks[i]->weightsToVector(pOutVector);
		pOutVector += n;
		total += n;
	}
	return total;
}

size_t GLayer::vectorToWeights(const double* pVector)
{
	size_t total = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		size_t n = m_blocks[i]->vectorToWeights(pVector);
		pVector += n;
		total += n;
	}
	return total;
}

void GLayer::copyWeights(const GLayer* pOther)
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		m_blocks[i]->copyWeights(pOther->m_blocks[i]);
}

void GLayer::perturbWeights(GRand& rand, double deviation)
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		m_blocks[i]->perturbWeights(rand, deviation);
}

void GLayer::maxNorm(double min, double max)
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		m_blocks[i]->maxNorm(min, max);
}

void GLayer::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		m_blocks[i]->scaleWeights(factor, scaleBiases);
}

void GLayer::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_blocks.size(); i++)
		m_blocks[i]->diminishWeights(amount, diminishBiases);
}

void GLayer::step(double learningRate, const GVec &gradient)
{
	GConstVecWrapper vwGradient;
	size_t gradPos = 0;
	for(size_t i = 0; i < m_blocks.size(); i++)
	{
		GBlock& b = *m_blocks[i];
		size_t wc = b.weightCount();
		vwGradient.setData(gradient.data() + gradPos, wc);
		b.step(learningRate, vwGradient.vec());
		gradPos += wc;
	}
}










GContextLayer::GContextLayer(const GLayer& layer)
: m_layer(layer), m_activation(layer.outputs()), m_blame(layer.outputs())
{
	for(size_t i = 0; i < layer.blockCount(); i++)
	{
		const GBlock* b = &layer.block(i);
		if(b->type() == GBlock::block_neuralnet)
			m_components.push_back(((GNeuralNet*)b)->newContext());
		else if(b->isRecurrent())
			m_recurrents.push_back(new GContextRecurrent(*(GBlockRecurrent*)b));
	}
}

GContextLayer::~GContextLayer()
{
	for(size_t i = 0; i < m_recurrents.size(); i++)
		delete(m_recurrents[i]);
	for(size_t i = 0; i < m_components.size(); i++)
		delete(m_components[i]);
}

void GContextLayer::resetState()
{
	size_t recurrents = 0;
	size_t comp = 0;
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		const GBlock& block = m_layer.block(i);
		if(block.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			pCompContext->resetState();
		}
		else if(block.isRecurrent())
		{
			GContextRecurrent* pRecContext = m_recurrents[recurrents++];
			pRecContext->resetState();
		}
	}
}

void GContextLayer::forwardProp(const GVec& input, GVec& output)
{
	GConstVecWrapper vwInput;
	GVecWrapper vwOutput;
	size_t outPos = 0;
	size_t recurrents = 0;
	size_t comp = 0;
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		const GBlock& block = m_layer.block(i);
		vwInput.setData(input.data() + block.inPos(), block.inputs());
		vwOutput.setData(output.data() + outPos, block.outputs());
		const GVec& in = vwInput.vec();
		GVec& out = vwOutput.vec();
		if(block.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			pCompContext->forwardProp(in, out);
		}
		else if(block.isRecurrent())
		{
			GContextRecurrent* pRecContext = m_recurrents[recurrents++];
			pRecContext->forwardProp(in, out);
		}
		else
			block.forwardProp(in, out);
		outPos += block.outputs();
	}
}

void GContextLayer::forwardProp_training(const GVec& input, GVec& output)
{
	GConstVecWrapper vwInput;
	GVecWrapper vwOutput;
	size_t outPos = 0;
	size_t recurrents = 0;
	size_t comp = 0;
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		const GBlock& block = m_layer.block(i);
		vwInput.setData(input.data() + block.inPos(), block.inputs());
		vwOutput.setData(output.data() + outPos, block.outputs());
		const GVec& in = vwInput.vec();
		GVec& out = vwOutput.vec();
		if(block.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			pCompContext->forwardProp(in, out);
		}
		else if(block.isRecurrent())
		{
			GContextRecurrent* pRecContext = m_recurrents[recurrents++];
			pRecContext->forwardPropThroughTime(in, out);
		}
		else
			block.forwardProp(in, out);
		outPos += block.outputs();
	}
}

void GContextLayer::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	GConstVecWrapper vwInput;
	GConstVecWrapper vwOutput;
	GConstVecWrapper vwOutBlame;
	GVecWrapper vwInBlame;
	size_t outPos = 0;
	size_t comp = 0;
	inBlame.fill(0.0);
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		GBlock& block = ((GLayer*)&m_layer)->block(i);
		vwInput.setData(input.data() + block.inPos(), block.inputs());
		vwOutput.setData(output.data() + outPos, block.outputs());
		vwOutBlame.setData(outBlame.data() + outPos, block.outputs());
		vwInBlame.setData(inBlame.data() + block.inPos(), block.inputs());
		if(block.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			pCompContext->backProp(vwInput.vec(), vwOutput.vec(), vwOutBlame.vec(), &vwInBlame.vec());
		}
		else
			block.backProp(vwInput.vec(), vwOutput.vec(), vwOutBlame.vec(), vwInBlame.vec());
		outPos += block.outputs();
	}
}

void GContextLayer::updateGradient(const GVec& input, const GVec& outBlame, GVec &gradient) const
{
	GConstVecWrapper vwInput;
	GConstVecWrapper vwOutBlame;
	GVecWrapper vwGradient;
	size_t gradPos = 0;
	size_t outPos = 0;
	size_t comp = 0;
	for(size_t i = 0; i < m_layer.blockCount(); i++)
	{
		const GBlock& block = m_layer.block(i);
		size_t wc = block.weightCount();
		vwInput.setData(input.data() + block.inPos(), block.inputs());
		vwOutBlame.setData(outBlame.data() + outPos, block.outputs());
		vwGradient.setData(gradient.data() + gradPos, wc);
		if(block.type() == GBlock::block_neuralnet)
		{
			GContextNeuralNet* pCompContext = m_components[comp++];
			pCompContext->updateGradient(vwInput.vec(), vwOutBlame.vec(), vwGradient.vec());
		}
		else
			block.updateGradient(vwInput.vec(), vwOutBlame.vec(), vwGradient.vec());
		outPos += block.outputs();
		gradPos += wc;
	}
}






GNeuralNet::GNeuralNet()
: GBlock(), m_weightCount(0)
{
}

GNeuralNet::GNeuralNet(GDomNode* pNode)
: GBlock(pNode), m_weightCount(0)
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

GContextNeuralNet* GNeuralNet::newContext() const
{
	return new GContextNeuralNet(*this);
}

std::string GNeuralNet::to_str() const
{
	std::ostringstream oss;
	oss << "[GBlockLinear:" << inputs() << "->" << outputs() << "]";
	return oss.str();
}

GLayer& GNeuralNet::newLayer()
{
	GAssert(m_weightCount == 0, "in inaccurate weight count was previously given");
	GLayer* pNewLayer = new GLayer();
	m_layers.push_back(pNewLayer);
	return *pNewLayer;
}

void GNeuralNet::resize(size_t inputs, size_t outputs)
{
	// Resize the inputs of the first layer
	if(m_layers[0]->blockCount() == 1)
		m_layers[0]->block(0).resize(inputs, m_layers[0]->block(0).outputs());
	else
		throw Ex("Cannot resize a layer with multiple blocks");

	// Resize the outputs of the last non-elementwise layer, and all subsequent element-wise layers, to fit the outputs
	size_t lay = m_layers.size() - 1;
	while(lay > 0 && m_layers[lay]->blockCount() == 1 && m_layers[lay]->block(0).elementWise())
	{
		m_layers[lay]->block(0).resize(outputs, outputs);
		lay--;
	}
	if(m_layers[lay]->blockCount() == 1)
		m_layers[lay]->block(0).resize(m_layers[lay]->block(0).inputs(), outputs);
	else
		throw Ex("Cannot resize a layer with multiple blocks");

	// Initialize the layers
	size_t inCount = inputs;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		if(m_layers[i]->blockCount() == 1)
		{
			if(m_layers[i]->block(0).elementWise())
				m_layers[i]->block(0).resize(inCount, inCount);
			else
				m_layers[i]->block(0).resize(inCount, m_layers[i]->block(0).outputs());
		}
		m_layers[i]->recount();
		if(m_layers[i]->inputs() != inCount)
			throw Ex(GClasses::to_str(inCount), " values feed into layer ", GClasses::to_str(i), ", but it expects ", GClasses::to_str(m_layers[i]->inputs()));
		inCount = m_layers[i]->outputs();
	}

	m_weightCount = 0;
	if(inCount != outputs)
		throw Ex("The last layer outputs ", GClasses::to_str(inCount), " values, but ", GClasses::to_str(outputs), " were expected");
}

void GNeuralNet::forwardProp(const GVec& input, GVec& output) const
{
	throw Ex("You should call GContextNeuralNet::forwardProp instead of GNeuralNet::forwardProp. (See also GNeuralNet::newContext)");
}

void GNeuralNet::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame) const
{
	throw Ex("You should call GContextNeuralNet::backProp instead of GNeuralNet::backProp. (See also GNeuralNet::newContext)");
}

void GNeuralNet::updateGradient(const GVec &x, const GVec& outBlame, GVec &gradient) const
{
	throw Ex("You should call GContextNeuralNet::updateGradient instead of GNeuralNet::updateGradient. (See also GNeuralNet::newContext)");
}

void GNeuralNet::step(double learningRate, const GVec &gradient)
{
	GConstVecWrapper vwGradient;
	size_t gradPos = 0;
	for(size_t i = 0; i < layerCount(); ++i)
	{
		GLayer& lay = layer(i);
		size_t wc = lay.weightCount();
		vwGradient.setData(gradient.data() + gradPos, wc);
		lay.step(learningRate, vwGradient.vec());
		gradPos += wc;
	}
	GAssert(gradPos == weightCount());
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

size_t GNeuralNet::weightsToVector(double* pOutWeights) const
{
	size_t total = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		size_t n = m_layers[i]->weightsToVector(pOutWeights);
		pOutWeights += n;
		total += n;
	}
	return total;
}

size_t GNeuralNet::vectorToWeights(const double* pWeights)
{
	size_t total = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		size_t n = m_layers[i]->vectorToWeights(pWeights);
		pWeights += n;
		total += n;
	}
	return total;
}

void GNeuralNet::copyWeights(const GBlock* pOther)
{
	GNeuralNet* pThat = (GNeuralNet*)pOther;
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->copyWeights(pThat->m_layers[i]);
}

void GNeuralNet::copyStructure(const GNeuralNet* pOther)
{
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
	// todo: the layers are still not yet hooked up with each other
}

void GNeuralNet::resetWeights(GRand& rand)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->resetWeights(rand);
}

void GNeuralNet::perturbWeights(GRand& rand, double deviation)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->perturbWeights(rand, deviation);
}

void GNeuralNet::maxNorm(double min, double max)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->maxNorm(min, max);
}

void GNeuralNet::scaleWeights(double factor, bool scaleBiases)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->scaleWeights(factor, scaleBiases);
}

void GNeuralNet::diminishWeights(double amount, bool diminishBiases)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->diminishWeights(amount, diminishBiases);
}

void GNeuralNet::invertNode(size_t lay, size_t node)
{
	GLayer& l = layer(lay);
	if(l.blockCount() > 1)
		throw Ex("This method assumes there is only one block in the layer");
	GBlock& b = l.block(0);
	if(b.type() == GBlock::block_linear)
	{
		GBlockLinear& layerUpStream = *(GBlockLinear*)&b;
		GMatrix& w = layerUpStream.weights();
		for(size_t i = 0; i < w.rows(); i++)
			w[i][node] = -w[i][node];
		size_t ds = lay + 1;
		while(ds < m_layers.size() && m_layers[ds]->blockCount() == 1 && m_layers[ds]->block(0).elementWise())
			ds++;
		if(ds < m_layers.size())
		{
			if(m_layers[ds]->blockCount() != 1 || m_layers[ds]->block(0).type() != GBlock::block_linear)
				throw Ex("Expected the downstream layer to contain exactly one linear block");
			GBlockLinear& layerDownStream = *(GBlockLinear*)&m_layers[ds]->block(0);
			size_t downOuts = layerDownStream.outputs();
			GVec& ww = layerDownStream.weights()[node];
			for(size_t i = 0; i < downOuts; i++)
				ww[i] = -ww[i];
		}
	}
	else
		throw Ex("I don't know how to invert nodes in this type of layer");
}

void GNeuralNet::swapNodes(size_t lay, size_t a, size_t b)
{
	GLayer& l = layer(lay);
	if(l.blockCount() != 1)
		throw Ex("Expected only one block in this layer");
	if(l.block(0).type() == GBlock::block_linear)
	{
		GBlockLinear& layerUpStream = *(GBlockLinear*)&l;
		layerUpStream.weights().swapColumns(a, b);
		size_t ds = lay + 1;
		while(ds < m_layers.size() && m_layers[ds]->blockCount() == 1 && m_layers[ds]->block(0).elementWise())
			ds++;
		if(ds < m_layers.size())
		{
			if(m_layers[ds]->blockCount() != 1 || m_layers[ds]->block(0).type() != GBlock::block_linear)
				throw Ex("Expected the downstream layer to contain exactly one linear block");
			GBlockLinear& layerDownStream = *(GBlockLinear*)m_layers[ds];
			layerDownStream.weights().swapRows(a, b);
		}
	}
	else
		throw Ex("I don't know how to swap nodes in this type of layer");
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

double GNeuralNet::measureLoss(const GMatrix& features, const GMatrix& labels, double* pOutSAE)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected the features and labels to have the same number of rows");
	if(features.cols() != inputs())
		throw Ex("Unexpected number of inputs");
	if(labels.cols() != outputs())
		throw Ex("Unexpected number of outputs");
	GContextNeuralNet* pCtx = newContext();
	GVec& prediction = pCtx->predBuf();
	double sae = 0.0;
	double sse = 0.0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		pCtx->forwardProp(features[i], prediction);
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
	if(pOutSAE)
		*pOutSAE = sae;
	return sse;
}

void GNeuralNet::printWeights(std::ostream& stream)
{
	stream.precision(6);
	stream << "NeuralNet:\n";
/*	for(size_t i = layerCount() - 1; i < layerCount(); i--)
	{
		if(i == layerCount() - 1)
			stream << "	Output Layer:\n";
		else
			stream << "	Hidden Layer " << to_str(i) << ":\n";
		GBlock& l = layer(i);
		if(!l.hasWeights())
			stream << "		weightless layer type: " << to_str((size_t)l.type());
		else if(l.type() == GBlock::layer_linear)
		{
			GLayerLinear& lay = *(GLayerLinear*)&l;
			for(size_t j = 0; j < lay.outputs(); j++)
			{
				stream << "		Unit " << to_str(j) << ":	";
				stream << "(bias: " << to_str(lay.bias()[j]) << ")	";
				for(size_t k = 0; k < lay.inputs(); k++)
				{
					if(k > 0)
						stream << "	";
					stream << to_str(lay.weights()[k][j]);
				}
				stream << "\n";
			}
		}
		else
			throw Ex("I don't know how to print the weights in layers of this type");
	}*/
}
/*
void GNeuralNet::containIntrinsics(GMatrix& intrinsics)
{
	size_t dims = intrinsics.cols();
	GBlock& llay = layer(0);
	if(llay.inputs() != dims)
		throw Ex("Mismatching number of columns and inputs");
	if(llay.type() == GBlock::layer_linear)
	{
		GLayerLinear& lay = *(GLayerLinear*)&llay;
		GVec pCentroid(dims);
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
	else
		throw Ex("I don't know how to contain this type of layer");
}

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
*/









GContextNeuralNet::GContextNeuralNet(const GNeuralNet& nn)
: m_nn(nn)
{
	if(nn.layerCount() < 1)
		throw Ex("No layers have been added to this neural network");
	for(size_t i = 0; i < nn.layerCount(); i++)
	{
		const GLayer& lay = nn.layer(i);
		m_layers.push_back(lay.newContext());
	}
	m_pOutputLayer = m_layers[m_layers.size() - 1];
}

GContextNeuralNet::~GContextNeuralNet()
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		delete(m_layers[i]);
	}
}

// virtual
void GContextNeuralNet::resetState()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->resetState();
}

void GContextNeuralNet::forwardProp(const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_nn.layer(0).inputs());
	GAssert(output.size() == m_nn.outputLayer().outputs());
	GAssert(layerCount() == m_nn.layerCount());
	const GVec* pInput = &input;
	size_t lastLayer = m_layers.size() - 1;
	for(size_t i = 0; i < lastLayer; i++)
	{
		GContextLayer* pLayer = m_layers[i];
		pLayer->forwardProp(*pInput, pLayer->m_activation);
		pInput = &pLayer->m_activation;
	}
	m_layers[lastLayer]->forwardProp(*pInput, output);
}

void GContextNeuralNet::forwardProp_training(const GVec& input, GVec& output) const
{
	GAssert(input.size() == m_nn.layer(0).inputs());
	GAssert(output.size() == m_nn.outputLayer().outputs());
	GAssert(layerCount() == m_nn.layerCount());
	const GVec* pInput = &input;
	size_t lastLayer = m_layers.size() - 1;
	for(size_t i = 0; i < lastLayer; i++)
	{
		GContextLayer* pLayer = m_layers[i];
		pLayer->forwardProp_training(*pInput, pLayer->m_activation);
		pInput = &pLayer->m_activation;
	}
	m_layers[lastLayer]->forwardProp(*pInput, output);
}

void GContextNeuralNet::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec* pInBlame)
{
	const GVec* pOutput = &output;
	const GVec* pOutBlame = &outBlame;
	for(size_t i = m_layers.size() - 1; i > 0; i--)
	{
		GContextLayer* pLayer = m_layers[i];
		GContextLayer* pPrevLayer = m_layers[i - 1];
		pLayer->backProp(pPrevLayer->m_activation, *pOutput, *pOutBlame, pPrevLayer->m_blame);
		pOutput = &pPrevLayer->m_activation;
		pOutBlame = &pPrevLayer->m_blame;
	}
	if(pInBlame)
	{
		GContextLayer* pLayer = m_layers[0];
		pLayer->backProp(input, pLayer->m_activation, pLayer->m_blame, *pInBlame);
	}
}

void GContextNeuralNet::updateGradient(const GVec& input, const GVec& outBlame, GVec& gradient) const
{
	const GVec* pInput = &input;
	size_t gradPos = 0;
	GVecWrapper vwGradient;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GContextLayer* pLayer = m_layers[i];
		size_t wc = pLayer->m_layer.weightCount();
		vwGradient.setData(gradient.data() + gradPos, wc);
		GAssert(gradPos + wc <= gradient.size());
		pLayer->updateGradient(*pInput, pLayer->m_blame, vwGradient.vec());
		pInput = &pLayer->m_activation;
		gradPos += wc;
	}
	GAssert(gradPos == m_nn.weightCount());
}










GNeuralNetLearner::GNeuralNetLearner()
: GIncrementalLearner(), m_pOptimizer(nullptr), m_ready(false)
{}

GNeuralNetLearner::GNeuralNetLearner(const GDomNode* pNode)
: GIncrementalLearner(pNode),
m_nn(pNode->field("nn")),
m_pOptimizer(nullptr)
{
}

GNeuralNetLearner::~GNeuralNetLearner()
{
}

GNeuralNetOptimizer& GNeuralNetLearner::optimizer()
{
	if(!m_pOptimizer)
		m_pOptimizer = new GSGDOptimizer(m_nn);
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
	GSGDOptimizer optimizer(m_nn);
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
	opt.context().forwardProp(in, out);
}

// virtual
void GNeuralNetLearner::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	if(labelRel.size() < 1)
		throw Ex("The label relation must have at least 1 attribute");

	m_nn.resize(featureRel.size(), labelRel.size());
	m_nn.resetWeights(m_rand);

	m_ready = true;
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
	GBlockLinear* b0 = new GBlockLinear(3);
	GBlockTanh* b1 = new GBlockTanh();
	GBlockLinear* b2 = new GBlockLinear(1);
	GBlockTanh* b3 = new GBlockTanh();
	nn.newLayer().add(b0);
	nn.newLayer().add(b1);
	nn.newLayer().add(b2);
	nn.newLayer().add(b3);
	nn.beginIncrementalLearning(features.relation(), labels.relation());

	GSGDOptimizer optimizer(nn.nn());
	optimizer.setLearningRate(0.175);
	optimizer.setMomentum(0.9);

	if(nn.nn().weightCount() != 13)
		throw Ex("Wrong number of weights");
	b2->bias()[0] = 0.02; // w_0
	b2->weights()[0][0] = -0.01; // w_1
	b2->weights()[1][0] = 0.03; // w_2
	b2->weights()[2][0] = 0.02; // w_3
	b0->bias()[0] = -0.01; // w_4
	b0->weights()[0][0] = -0.03; // w_5
	b0->weights()[1][0] = 0.03; // w_6
	b0->bias()[1] = 0.01; // w_7
	b0->weights()[0][1] = 0.04; // w_8
	b0->weights()[1][1] = -0.02; // w_9
	b0->bias()[2] = -0.02; // w_10
	b0->weights()[0][2] = 0.03; // w_11
	b0->weights()[1][2] = 0.02; // w_12

	// Test forward prop
	double tol = 1e-12;
	GVec pat(2);
	pat.copy(features[0]);
	GVec pred(1);
	optimizer.optimizeIncremental(features[0], labels[0]);

	GContextLayer& c0 = optimizer.context().layer(0);
	GContextLayer& c1 = optimizer.context().layer(1);
	GContextLayer& c2 = optimizer.context().layer(2);
	GContextLayer& c3 = optimizer.context().layer(3);

	// Test forward prop
	if(std::abs(-0.031 - c0.m_activation[0]) > tol) throw Ex("forward prop problem");
	if(std::abs(-0.030990073482402569 - c1.m_activation[0]) > tol) throw Ex("forward prop problem");
	if(std::abs(0.020350024432229302 - c2.m_activation[0]) > tol) throw Ex("forward prop problem");
	if(std::abs(0.020347215756407563 - c3.m_activation[0]) > tol) throw Ex("forward prop problem");

	// Test output blame
	if(std::abs(0.97965278424359248 - c3.m_blame[0]) > tol) throw Ex("problem computing output blame");

	// Test back prop
	if(std::abs(0.97924719898884915 - c2.m_blame[0]) > tol) throw Ex("back prop problem");
	if(std::abs(-0.0097924719898884911 - c1.m_blame[0]) > tol) throw Ex("back prop problem");
	if(std::abs(0.029377415969665473 - c1.m_blame[1]) > tol) throw Ex("back prop problem");
	if(std::abs(0.019584943979776982 - c1.m_blame[2]) > tol) throw Ex("back prop problem");	
	if(std::abs(-0.00978306745006032 - c0.m_blame[0]) > tol) throw Ex("back prop problem");
	if(std::abs(0.02936050107376107 - c0.m_blame[1]) > tol) throw Ex("back prop problem");
	if(std::abs(0.01956232122115741 - c0.m_blame[2]) > tol) throw Ex("back prop problem");

	// Test bias updates
	if(std::abs(0.191368259823049 - b2->bias()[0]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.011712036803760557 - b0->bias()[0]) > tol) throw Ex("weight update problem");
	if(std::abs(0.015138087687908187 - b0->bias()[1]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.016576593786297455 - b0->bias()[2]) > tol) throw Ex("weight update problem");

	// Test weight updates
	if(std::abs(-0.015310714964467731 - b2->weights()[0][0]) > tol) throw Ex("weight update problem");
	if(std::abs(0.034112048752708297 - b2->weights()[1][0]) > tol) throw Ex("weight update problem");
	if(std::abs(0.014175723281037968 - b2->weights()[2][0]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.03 - b0->weights()[0][0]) > tol) throw Ex("weight update problem");
	if(std::abs(0.03119842576263239 - b0->weights()[1][0]) > tol) throw Ex("weight update problem");
	if(std::abs(0.04 - b0->weights()[0][1]) > tol) throw Ex("weight update problem");
	if(std::abs(-0.023596661381535732 - b0->weights()[1][1]) > tol) throw Ex("weight update problem");
	if(std::abs(0.03 - b0->weights()[0][2]) > tol) throw Ex("weight update problem");
	if(std::abs(0.01760361565040822 - b0->weights()[1][2]) > tol) throw Ex("weight update problem");
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
	GNeuralNetLearner* pNN = new GNeuralNetLearner();
	pNN->newLayer().add(new GBlockLinear(1));
	pNN->newLayer().add(new GBlockTanh());
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

	// This test ensures that the GNeuralNetLearner::swapNodes and GNeuralNetLearner::invertNode methods
	// have no net effect on the output of the neural network
	GVec in(TEST_INVERT_INPUTS);
	GVec outBefore(TEST_INVERT_INPUTS);
	GVec outAfter(TEST_INVERT_INPUTS);
	for(size_t i = 0; i < 30; i++)
	{
		GNeuralNetLearner nn;
		for(size_t j = 0; j < layers; j++)
		{
			nn.newLayer().add(new GBlockLinear(layerSize));
			nn.newLayer().add(new GBlockTanh());
		}
		nn.newLayer().add(new GBlockLinear(TEST_INVERT_INPUTS));
		nn.newLayer().add(new GBlockTanh());
		GUniformRelation rel(TEST_INVERT_INPUTS);
		nn.beginIncrementalLearning(rel, rel);
		nn.nn().perturbWeights(nn.rand(), 0.5);
		in.fillUniform(rand);
		nn.predict(in, outBefore);
		for(size_t j = 0; j < 8; j++)
		{
			if(rand.next(2) == 0)
				nn.nn().swapNodes((size_t)rand.next(layers) * 2, (size_t)rand.next(layerSize), (size_t)rand.next(layerSize));
			else
				nn.nn().invertNode((size_t)rand.next(layers) * 2, (size_t)rand.next(layerSize));
		}
		nn.predict(in, outAfter);
		if(outBefore.squaredDistance(outAfter) > 1e-10)
			throw Ex("GNeuralNetLearner::invertAndSwap failed");
	}

	for(size_t i = 0; i < 30; i++)
	{
		// Generate two identical neural networks
		GNeuralNetLearner nn1;
		GNeuralNetLearner nn2;
		for(size_t j = 0; j < layers; j++)
		{
			nn1.newLayer().add(new GBlockLinear(layerSize));
			nn1.newLayer().add(new GBlockTanh());
			nn2.newLayer().add(new GBlockLinear(layerSize));
			nn2.newLayer().add(new GBlockTanh());
		}
		nn1.newLayer().add(new GBlockLinear(TEST_INVERT_INPUTS));
		nn1.newLayer().add(new GBlockTanh());
		nn2.newLayer().add(new GBlockLinear(TEST_INVERT_INPUTS));
		nn2.newLayer().add(new GBlockTanh());
		GUniformRelation rel(TEST_INVERT_INPUTS);
		nn1.beginIncrementalLearning(rel, rel);
		nn2.beginIncrementalLearning(rel, rel);
		nn1.nn().perturbWeights(nn1.rand(), 0.5);
		nn2.nn().copyWeights(&nn1.nn());

		// Predict something
		in.fillUniform(rand);
		nn1.predict(in, outBefore);

		// Mess with the topology of both networks
		for(size_t j = 0; j < 20; j++)
		{
			if(rand.next(2) == 0)
			{
				if(rand.next(2) == 0)
					nn1.nn().swapNodes((size_t)rand.next(layers) * 2, (size_t)rand.next(layerSize), (size_t)rand.next(layerSize));
				else
					nn1.nn().invertNode((size_t)rand.next(layers) * 2, (size_t)rand.next(layerSize));
			}
			else
			{
				if(rand.next(2) == 0)
					nn2.nn().swapNodes((size_t)rand.next(layers) * 2, (size_t)rand.next(layerSize), (size_t)rand.next(layerSize));
				else
					nn2.nn().invertNode((size_t)rand.next(layers) * 2, (size_t)rand.next(layerSize));
			}
		}

		// Align the first network to match the second one
		nn1.nn().align(nn2.nn());

		// Check that predictions match before
		nn2.predict(in, outAfter);
		if(outBefore.squaredDistance(outAfter) > 1e-10)
			throw Ex("Failed");
		nn1.predict(in, outAfter);
		if(outBefore.squaredDistance(outAfter) > 1e-10)
			throw Ex("Failed");

		// Check that they have matching weights
		size_t wc = nn1.nn().weightCount();
		double* pW1 = new double[wc];
		std::unique_ptr<double[]> hW1(pW1);
		nn1.nn().weightsToVector(pW1);
		double* pW2 = new double[wc];
		std::unique_ptr<double[]> hW2(pW2);
		nn2.nn().weightsToVector(pW2);
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
	GVec in(5);
	for(size_t i = 0; i < 20; i++)
	{
		GNeuralNetLearner nn;
		GBlockLinear* pLayerIn = new GBlockLinear(5);
		nn.newLayer().add(pLayerIn);
		GBlockTanh* pLayerAct = new GBlockTanh();
		nn.newLayer().add(pLayerAct);
		GBlockLinear* pLayerOut = new GBlockLinear(1);
		nn.newLayer().add(pLayerOut);
		GBlockTanh* pLayerOutAct = new GBlockTanh();
		nn.newLayer().add(pLayerOutAct);
		GUniformRelation relIn(5);
		GUniformRelation relOut(1);
		nn.beginIncrementalLearning(relIn, relOut);
		nn.nn().perturbWeights(nn.rand(), 1.0);
		in.fillNormal(rand);
		in.normalize();
		GVec before(1);
		GVec after(1);
		nn.predict(in, before);
		double a = rand.normal();
		double b = rand.normal();
		if(b < a)
			std::swap(a, b);
		double c = rand.normal();
		double d = rand.normal();
		if(d < c)
			std::swap(c, d);
		size_t ind = (size_t)rand.next(5);
		pLayerIn->renormalizeInput(ind, a, b, c, d);
		in[ind] = GMatrix::normalizeValue(in[ind], a, b, c, d);
		nn.predict(in, after);
		if(std::abs(after[0] - before[0]) > 1e-9)
			throw Ex("Failed");
	}
}

void GNeuralNet_testTransformWeights(GRand& prng)
{
	for(size_t i = 0; i < 10; i++)
	{
		// Set up
		GNeuralNetLearner nn;
		GBlockLinear* pLayerInput = new GBlockLinear(3);
		nn.newLayer().add(pLayerInput);
		GBlockTanh* pLayerAct = new GBlockTanh();
		nn.newLayer().add(pLayerAct);
		GUniformRelation in(2);
		GUniformRelation out(3);
		nn.beginIncrementalLearning(in, out);
		nn.nn().perturbWeights(nn.rand(), 1.0);
		GVec x1(2);
		GVec x2(2);
		GVec y1(3);
		GVec y2(3);
		x1.fillSphericalShell(prng);

		// Predict normally
		nn.predict(x1, y1);

		// Transform the inputs and weights
		GMatrix transform(2, 2);
		transform[0].fillSphericalShell(prng);
		transform[1].fillSphericalShell(prng);
		GVec offset(2);
		offset.fillSphericalShell(prng);
		x1 += offset;
		transform.multiply(x1, x2, false);

		GVec tmp(2);
		offset *= -1.0;
		transform.multiply(offset, tmp);
		offset.copy(tmp);
		GMatrix* pTransInv = transform.pseudoInverse();
		std::unique_ptr<GMatrix> hTransInv(pTransInv);
		pLayerInput->transformWeights(*pTransInv, offset);

		// Predict again
		nn.predict(x2, y2);
		if(y1.squaredDistance(y2) > 1e-15)
			throw Ex("transformWeights failed");
	}
}
/*
#define NN_TEST_DIMS 5

void GNeuralNet_testCompressFeatures(GRand& prng)
{
	GMatrix feat(50, NN_TEST_DIMS);
	for(size_t i = 0; i < feat.rows(); i++)
		feat[i].fillSphericalShell(prng);

	// Set up
	GNeuralNetLearner nn1;
	nn1.newLayer().add(new GBlockLinear(NN_TEST_DIMS * 2));
	nn1.newLayer().add(new GBlockTanh());
	nn1.beginIncrementalLearning(feat.relation(), feat.relation());
	nn1.nn().perturbWeights(nn1.rand(), 1.0);
	GNeuralNetLearner nn2;
	nn2.nn().copyStructure(&nn1.nn());
	nn2.nn().copyWeights(&nn1.nn());

	// Test
	GMatrix* pNewFeat = nn1.compressFeatures(feat);
	std::unique_ptr<GMatrix> hNewFeat(pNewFeat);
	GVec out1(NN_TEST_DIMS);
	GVec out2(NN_TEST_DIMS);
	for(size_t i = 0; i < feat.rows(); i++)
	{
		nn1.predict(pNewFeat->row(i), out1);
		nn2.predict(feat[i], out2);
		if(out1.squaredDistance(out2) > 1e-14)
			throw Ex("failed");
	}
}

void GNeuralNet_testConvolutionalLayer2D(GRand &prng)
{
	// a 5x5x3 "image"
	// each channel may be given in deinterlaced order (i.e. red as 5x5x1, then blue as 5x5x1, then green as 5x5x1) or in interlaced order (i.e. rgb for pixel 1, rgb for pixel 2, etc)
	// interlaced is assumed, but deinterlaced may be used if the layer is set to interpret the input as deinterlaced
	GVec feature(5*5*3);
	{
		// std::string data = "1 2 0 1 0 0 2 2 2 0 1 1 1 0 0 1 1 1 0 2 1 0 2 0 2 2 1 2 1 0 2 0 1 2 1 2 1 0 1 2 0 2 1 1 0 2 1 1 0 1 1 0 0 0 0 1 0 2 1 1 0 1 2 1 1 0 0 2 2 0 1 2 0 2 1";
		std::string data = "1 2 1 2 1 0 0 2 0 1 1 0 0 0 0 0 2 1 2 0 0 2 1 2 2 2 1 0 1 1 1 2 0 1 1 1 1 0 2 0 1 1 0 2 1 1 0 0 1 2 0 1 1 2 0 1 2 2 0 0 1 2 1 0 1 2 2 1 0 0 0 2 2 1 1";
		std::istringstream ss(data);
		for(size_t i = 0; i < feature.size(); i++)
			ss >> feature[i];
	}

	// the correct convolution for the image above given the kernels below
	// also encoded as the image above
	GVec label(3*3*2);
	{
		// std::string data = "8 5 0 4 -7 4 1 4 8 -4 3 2 0 4 0 0 -2 -3";
		std::string data = "8 -4 5 3 0 2 4 0 -7 4 4 0 1 0 4 -2 8 -3";
		std::istringstream ss(data);
		for(size_t i = 0; i < label.size(); i++)
			ss >> label[i];
	}
	
	GLayerConvolutional2D *pLayer = new GLayerConvolutional2D(5, 5, 3, 3, 3, 2);
	GLayerConvolutional2D &layer = *pLayer;

	GLayerLinear* pLay1 = new GLayerLinear(1, layer.inputs());
	GLayerLinear &lay1 = *pLay1;
	GLayerTanh* pLay2 = new GLayerTanh();

	GNeuralNetLearner nn;
	nn.addLayers(pLay1, pLay2, pLayer);
	nn.beginIncrementalLearning(GUniformRelation(1), GUniformRelation(layer.outputs()));

	for(size_t i = 0; i < layer.inputs(); i++)
		lay1.weights()[0][i] = feature[i] + prng.normal();
	lay1.bias().fill(0.0);

	layer.setPadding(1);
	layer.setStride(2);
	{
		// std::string data1 = "0 1 -1 -1 1 1 1 0 1 1 0 0 -1 1 0 -1 1 1 1 -1 1 0 -1 -1 1 -1 -1 1";
		// std::string data2 = "-1 0 0 0 0 -1 1 1 0 1 0 1 0 0 0 1 -1 0 -1 1 -1 -1 0 -1 -1 0 1 0";
		std::string data1 = "0 1 1 1 0 -1 -1 0 1 -1 -1 0 1 1 -1 1 0 -1 1 -1 1 0 1 -1 1 1 -1 1";
		std::string data2 = "-1 1 -1 0 0 1 0 1 -1 0 0 -1 0 0 0 -1 0 -1 1 1 -1 1 -1 0 0 0 1 0";
		std::istringstream ss1(data1);
		std::istringstream ss2(data2);

		for(size_t i = 0; i < layer.kernels().cols(); i++)
		{
			ss1 >> layer.kernels()[0][i];
			ss2 >> layer.kernels()[1][i];
		}
		ss1 >> layer.bias()[0];
		ss2 >> layer.bias()[1];
	}

	// test serialization

	GDom doc;
	GDomNode *root = layer.serialize(&doc);
	GLayerConvolutional2D *p_layer = (GLayerConvolutional2D *) GBlock::deserialize(root);
	for(size_t i = 0; i < layer.bias().size(); ++i)
		if(layer.bias()[i] != p_layer->bias()[i])
			throw Ex("GLayerConvolutional2D serialization failed (1)");
	for(size_t i = 0; i < layer.kernels().rows(); ++i)
		for(size_t j = 0; j < layer.kernels().cols(); ++j)
			if(layer.kernels()[i][j] != p_layer->kernels()[i][j])
				throw Ex("GLayerConvolutional2D serialization failed (2)");
	delete p_layer;

	// test forward propagation

	layer.feedForward(feature);
	for(size_t i = 0; i < label.size(); i++)
		if(label[i] != layer.activation()[i])
			throw Ex("GLayerConvolutional2D forward prop failed");

	// test backpropagation (1)
	// -- can we update weights in the previous layer?

	GVec oneVec(1);
	oneVec.fill(1.0);

	GSGDOptimizer optimizer(nn);
	optimizer.setLearningRate(1e-2);

	for(size_t i = 0; i < 200; i++)
		optimizer.optimizeIncremental(oneVec, label);

	nn.forwardProp(oneVec);
	if(layer.activation().squaredDistance(label) > 1e-6)
		throw Ex("GLayerConvolutional2D backpropagation failed (1)");

	// test backpropagation (2)
	// -- can we update weights in the convolutional layer?

	for(size_t c = 0; c < layer.kernels().rows(); c++)
	{
		layer.bias()[c] += prng.normal();
		for(size_t i = 0; i < layer.kernels().cols(); i++)
			layer.kernels()[c][i] += prng.normal();
	}

	for(size_t i = 0; i < 200; i++)
		optimizer.optimizeIncremental(oneVec, label);

	nn.forwardProp(oneVec);
	if(layer.activation().squaredDistance(label) > 1e-6)
		throw Ex("GLayerConvolutional2D backpropagation failed (2)");
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
	GNeuralNetLearner* pNN = GNeuralNetLearner::fourier(m, period);
	std::unique_ptr<GNeuralNetLearner> hNN(pNN);
	GVec out(3);
	for(size_t i = 0; i < m.rows(); i++)
	{
		GVec in(1);
		in[0] = (double)i * period / m.rows();
		pNN->predict(in, out);
		if(out.squaredDistance(m[i]) > 1e-9)
			throw Ex("failed");
	}
}
*/
// static
void GNeuralNetLearner::test()
{
	GRand prng(0);
	GNeuralNet_testMath();
	GNeuralNet_testBinaryClassification(&prng);
	GNeuralNet_testNormalizeInput(prng);
	GNeuralNet_testTransformWeights(prng);
//	GNeuralNet_testConvolutionalLayer2D(prng);
//	GNeuralNet_testInvertAndSwap(prng);
//	GNeuralNet_testCompressFeatures(prng);
//	GNeuralNet_testFourier();

	// Test with no hidden layers (logistic regression)
	{
		GNeuralNetLearner* pNN = new GNeuralNetLearner();
		pNN->newLayer().add(new GBlockLinear((size_t)0));
		pNN->newLayer().add(new GBlockTanh());
		GAutoFilter af(pNN);
		af.basicTest(0.78, 0.895);
	}

	// Test NN with one hidden layer
	{
		GNeuralNetLearner* pNN = new GNeuralNetLearner();
		pNN->newLayer().add(new GBlockLinear(3));
		pNN->newLayer().add(new GBlockTanh());
		pNN->newLayer().add(new GBlockLinear((size_t)0));
		pNN->newLayer().add(new GBlockTanh());
		GAutoFilter af(pNN);
		af.basicTest(0.76, 0.92);
	}
}

#endif // MIN_PREDICT

























GReservoirNet::GReservoirNet()
: GIncrementalLearner(), m_pModel(NULL), m_pNN(NULL), m_weightDeviation(0.5), m_augments(64), m_reservoirLayers(2)
{
}

GReservoirNet::GReservoirNet(const GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalLearner(pNode)
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
void GReservoirNet::predict(const GVec& in, GVec& out)
{
	m_pModel->predict(in, out);
}

// virtual
void GReservoirNet::predictDistribution(const GVec& in, GPrediction* out)
{
	m_pModel->predictDistribution(in, out);
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
	GNeuralNetLearner* pNN = new GNeuralNetLearner();
	pNN->newLayer().add(new GBlockLinear((size_t)0));
	pNN->newLayer().add(new GBlockTanh());
	GReservoir* pRes = new GReservoir(m_weightDeviation, m_augments, m_reservoirLayers);
	GDataAugmenter* pAug = new GDataAugmenter(pRes);
	m_pModel = new GFeatureFilter(pNN, pAug);
	m_pModel->train(features, labels);
}

// virtual
void GReservoirNet::trainIncremental(const GVec& in, const GVec& out)
{
	m_pModel->trainIncremental(in, out);
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
	m_pNN = new GNeuralNetLearner();
	m_pNN->newLayer().add(new GBlockLinear((size_t)0));
	m_pNN->newLayer().add(new GBlockTanh());
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
	af.basicTest(0.69, 0.74, 0.001, false, 0.9);
}
#endif // MIN_PREDICT



} // namespace GClasses
