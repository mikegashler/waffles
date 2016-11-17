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

using std::vector;

namespace GClasses {


GNeuralNet::GNeuralNet() : GIncrementalLearner(), m_ready(false)
{}

GNeuralNet::GNeuralNet(const GDomNode* pNode) : GIncrementalLearner(pNode)
{
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

void GNeuralNet::trainIncremental(const GVec &in, const GVec &out)
{
	throw Ex("GNeuralNet::trainIncremental is not implemented (need to use GDifferentiableOptimizer).");
}

void GNeuralNet::trainSparse(GSparseMatrix &features, GMatrix &labels)
{
	throw Ex("GNeuralNet::trainSparse is not implemented (need to use GDifferentiableOptimizer).");
}

void GNeuralNet::trainInner(const GMatrix& features, const GMatrix& labels)
{
	beginIncrementalLearningInner(features.relation(), labels.relation());
	GSGDOptimizer optimizer(*this);
	optimizer.optimizeWithValidation(features, labels);
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

size_t GNeuralNet::countWeights() const
{
	size_t wc = 0;
	for(size_t i = 0; i < m_layers.size(); i++)
		wc += m_layers[i]->countWeights();
	return wc;
}

void GNeuralNet::weights(double* pOutWeights, size_t lay) const
{
	if(m_layers[lay]->countWeights() > 0)
		pOutWeights += ((GParameterizedLayer*)m_layers[lay])->weightsToVector(pOutWeights);
}

void GNeuralNet::weights(double* pOutWeights) const
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		if(m_layers[i]->countWeights() > 0)
			pOutWeights += ((GParameterizedLayer*)m_layers[i])->weightsToVector(pOutWeights);
	}
}

void GNeuralNet::setWeights(const double* pWeights, size_t lay)
{
	if(m_layers[lay]->countWeights() > 0)
		pWeights += ((GParameterizedLayer*)m_layers[lay])->vectorToWeights(pWeights);
}

void GNeuralNet::setWeights(const double* pWeights)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		if(m_layers[i]->countWeights() > 0)
			pWeights += ((GParameterizedLayer*)m_layers[i])->vectorToWeights(pWeights);
	}
}

void GNeuralNet::copyWeights(const GNeuralNet* pOther)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		if(m_layers[i]->countWeights() > 0)
			((GParameterizedLayer*)m_layers[i])->copyWeights(pOther->m_layers[i]);
	}
}

void GNeuralNet::copyStructure(const GNeuralNet* pOther)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
	m_layers.clear();
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
}

void GNeuralNet::perturbAllWeights(double deviation)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		if(m_layers[i]->countWeights() > 0)
			((GParameterizedLayer*)m_layers[i])->perturbWeights(m_rand, deviation);
	}
}

void GNeuralNet::maxNorm(double min, double max, bool output_layer)
{
	size_t layer_count = m_layers.size();
	if(!output_layer)
		layer_count--;
	for(size_t i = 0; i < layer_count; i++)
	{
		if(m_layers[i]->countWeights() > 0)
			((GParameterizedLayer*)m_layers[i])->maxNorm(min, max);
	}
}

void GNeuralNet::invertNode(size_t lay, size_t node)
{
	GNeuralNetLayer& l = layer(lay);
	if(l.type() == GNeuralNetLayer::layer_classic) // This block is ready to be deleted when we get rid of GLayerClassic
	{
		GLayerClassic& layerUpStream = *(GLayerClassic*)&l;
		GMatrix& w = layerUpStream.m_weights;
		for(size_t i = 0; i < w.rows(); i++)
			w[i][node] = -w[i][node];
		if(lay + 1 < m_layers.size())
		{
			if(m_layers[lay + 1]->type() != GNeuralNetLayer::layer_classic)
				throw Ex("Expected the downstream layer to be a classic layer");
			GLayerClassic& layerDownStream = *(GLayerClassic*)m_layers[lay + 1];
			GActivationFunction* pActFunc = layerDownStream.m_pActivationFunction;
			size_t downOuts = layerDownStream.outputs();
			GVec& ww = layerDownStream.m_weights[node];
			GVec& bb = layerDownStream.bias();
			for(size_t i = 0; i < downOuts; i++)
			{
				bb[i] += 2 * pActFunc->squash(0.0, i) * ww[i];
				ww[i] = -ww[i];
			}
		}
	}
	else if(l.type() == GNeuralNetLayer::layer_linear)
	{
		GLayerLinear& layerUpStream = *(GLayerLinear*)m_layers[lay];
		GMatrix& w = layerUpStream.weights();
		for(size_t i = 0; i < w.rows(); i++)
			w[i][node] = -w[i][node];
		size_t ds = lay + 1;
		while(ds < m_layers.size() && m_layers[ds]->type() == GNeuralNetLayer::layer_activation)
			ds++;
		if(ds < m_layers.size())
		{
			if(m_layers[ds]->type() != GNeuralNetLayer::layer_linear)
				throw Ex("Expected the downstream layer to be a linear layer");
			GLayerLinear& layerDownStream = *(GLayerLinear*)m_layers[ds];
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
	GNeuralNetLayer& l = layer(lay);
	if(l.type() == GNeuralNetLayer::layer_classic) // This block is ready to be deleted when we get rid of GLayerClassic
	{
		GLayerClassic& layerUpStream = *(GLayerClassic*)&l;
		layerUpStream.m_weights.swapColumns(a, b);
		if(lay + 1 < m_layers.size())
		{
			GLayerClassic& layerDownStream = *(GLayerClassic*)m_layers[lay + 1];
			layerDownStream.m_weights.swapRows(a, b);
		}
	}
	else if(l.type() == GNeuralNetLayer::layer_linear)
	{
		GLayerLinear& layerUpStream = *(GLayerLinear*)&l;
		layerUpStream.weights().swapColumns(a, b);
		size_t ds = lay + 1;
		while(ds < m_layers.size() && m_layers[ds]->type() == GNeuralNetLayer::layer_activation)
			ds++;
		if(ds < m_layers.size())
		{
			if(m_layers[ds]->type() != GNeuralNetLayer::layer_linear)
				throw Ex("Expected the downstream layer to be a linear layer");
			GLayerLinear& layerDownStream = *(GLayerLinear*)m_layers[ds];
			layerDownStream.weights().swapRows(a, b);
		}
	}
	else
		throw Ex("I don't know how to swap nodes in this type of layer");
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
				pLayer->resizeInputs(m_layers[position - 1]);
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
		GNeuralNetLayer& lThis = layer(i);
		if(lThis.type() != that.m_layers[i]->type())
			throw Ex("mismatching layer types");
		if(lThis.type() == GNeuralNetLayer::layer_classic) // This block is ready to be deleted when we get rid of GLayerClassic
		{
			GLayerClassic& layerThisCur = *(GLayerClassic*)&lThis;
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
				for(size_t kk = 0; kk < inputs; kk++)
					dp += layerThisCur.m_weights[kk][j] * layerThatCur.m_weights[kk][j];
				dp += layerThisCur.bias()[j] * layerThatCur.bias()[j];
				if(dp < 0)
					invertNode(i, j); // invert it
			}
		}
		else if(lThis.type() == GNeuralNetLayer::layer_linear)
		{
			GLayerLinear& layerThisCur = *(GLayerLinear*)&lThis;
			GLayerLinear& layerThatCur = *(GLayerLinear*)that.m_layers[i];
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
		else if(lThis.countWeights() > 0)
			throw Ex("I don't know how to align this type of layer");
	}
}

void GNeuralNet::scaleWeights(double factor, bool scaleBiases, size_t startLayer, size_t layer_count)
{
	size_t end = std::min(startLayer + layer_count, m_layers.size());
	for(size_t i = startLayer; i < end; i++)
	{
		if(m_layers[i]->countWeights() > 0)
			((GParameterizedLayer*)m_layers[i])->scaleWeights(factor, scaleBiases);
	}
}

void GNeuralNet::diminishWeights(double amount, bool diminishBiases, size_t startLayer, size_t layer_count)
{
	size_t end = std::min(startLayer + layer_count, m_layers.size());
	for(size_t i = startLayer; i < end; i++)
	{
		if(m_layers[i]->countWeights() > 0)
		((GParameterizedLayer*)m_layers[i])->diminishWeights(amount, diminishBiases);
	}
}

void GNeuralNet::regularizeActivationFunctions(double lambda)
{
	throw Ex("regularizeActivationFunctions is temporarily unavailable");
	//for(size_t i = 0; i < m_layers.size(); i++)
	//	m_layers[i]->regularizeActivationFunction(lambda);
}

void GNeuralNet::contractWeights(double factor, bool contractBiases)
{
	size_t i = m_layers.size() - 1;
	GNeuralNetLayer* pLay = m_layers[i];
	pLay->error().fill(1.0);
	while(i > 0)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		((GLayerClassic*)pLay)->contractWeights(factor, contractBiases);
		pLay = pUpStream;
		i--;
	}
	((GLayerClassic*)pLay)->contractWeights(factor, contractBiases);
}
#endif // MIN_PREDICT

void GNeuralNet::forwardProp(const GVec& row)
{
	GNeuralNetLayer* pLay = m_layers[0];
	if(!pLay)
		throw Ex("No layers have been added to this neural network");
	pLay->feedForward(row);
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		GNeuralNetLayer* pDS = m_layers[i];
		pDS->feedForward(pLay);
		pLay = pDS;
	}
}

#ifndef MIN_PREDICT
// virtual
void GNeuralNet::predictDistribution(const GVec& in, GPrediction* pOut)
{
	throw Ex("Sorry, this model does not predict a distribution");
}
#endif // MIN_PREDICT

void GNeuralNet::copyPrediction(GVec& out)
{
	GNeuralNetLayer& outputLay = *m_layers[m_layers.size() - 1];
	out.copy(outputLay.activation());
}

// virtual
void GNeuralNet::predict(const GVec& in, GVec& out)
{
	forwardProp(in);
	copyPrediction(out);
}

// virtual
void GNeuralNet::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	if(labelRel.size() < 1)
		throw Ex("The label relation must have at least 1 attribute");

	// Resize the input and output layers to fit the data
	size_t inputs = featureRel.size();
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
	{
		if(m_layers[i]->countWeights() > 0)
			((GParameterizedLayer*)m_layers[i])->resetWeights(m_rand);
	}

	m_ready = true;
}

void GNeuralNet::backpropagate(const GVec &blame)
{
	size_t i = m_layers.size() - 1;
	GNeuralNetLayer* pLay = m_layers[i];
	pLay->error().put(0, blame);
	while(i > 0)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		pLay = pUpStream;
		i--;
	}
}

void GNeuralNet::backpropagateFromLayer(GNeuralNetLayer* pDownstream)
{
	GNeuralNetLayer* pLay = pDownstream;
	for(size_t i = m_layers.size(); i > 0; i--)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		pLay = pUpStream;
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
		if(l.countWeights() == 0)
			stream << "		weightless layer type: " << to_str((size_t)l.type());
		else if(l.type() == GNeuralNetLayer::layer_classic)
		{
			GLayerClassic& lay = *(GLayerClassic*)&l;
			for(size_t j = 0; j < lay.outputs(); j++)
			{
				stream << "		Unit " << to_str(j) << ":	";
				stream << "(bias: " << to_str(lay.bias()[j]) << ")	";
				for(size_t k = 0; k < lay.inputs(); k++)
				{
					if(k > 0)
						stream << "	";
					stream << to_str(lay.m_weights[k][j]);
				}
				stream << "\n";
			}
		}
		else if(l.type() == GNeuralNetLayer::layer_linear)
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
	}
}

void GNeuralNet::containIntrinsics(GMatrix& intrinsics)
{
	size_t dims = intrinsics.cols();
	GNeuralNetLayer& llay = layer(0);
	if(llay.inputs() != dims)
		throw Ex("Mismatching number of columns and inputs");
	if(llay.type() == GNeuralNetLayer::layer_classic)
	{
		GLayerClassic& lay = *(GLayerClassic*)&llay;
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
	else if(llay.type() == GNeuralNetLayer::layer_linear)
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
	GNeuralNetLayer& llay = layer(0);
	if(llay.type() == GNeuralNetLayer::layer_classic)
	{
		GLayerClassic& lay = *(GLayerClassic*)&llay;
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
	else if(llay.type() == GNeuralNetLayer::layer_linear)
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
	GLayerClassic* pLayerSin = new GLayerClassic(1, pSeries->rows(), new GActivationSin());
	GLayerClassic* pLayerIdent = new GLayerClassic(FLEXIBLE_SIZE, pSeries->cols(), new GActivationIdentity());
	pNN->addLayer(pLayerSin);
	pNN->addLayer(pLayerIdent);
	GUniformRelation relIn(1);
	GUniformRelation relOut(pSeries->cols());
	pNN->beginIncrementalLearning(relIn, relOut);

	// Initialize the weights of the sine units to match the frequencies used by the Fourier transform.
	GMatrix& wSin = pLayerSin->weights();
	GVec& bSin = pLayerSin->bias();
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

void GNeuralNet::updateGradient(const GVec &x, const GVec &blame, GVec &deltas)
{
	GAssert(deltas.size() == countWeights(), "Can't update gradient; not enough space in deltas!");
	backpropagate(blame);
	const GVec *in = &x;
	GVecWrapper out(deltas.data(), 0);
	for(size_t i = 0; i < layerCount(); ++i)
	{
		size_t count = layer(i).countWeights();
		out.setSize(count);
		if(count > 0)
			((GParameterizedLayer*)&layer(i))->updateDeltas(*in, out.vec());
		in = &layer(i).activation();
		out.setData(out.vec().data() + count);
	}
}

void GNeuralNet::step(const GVec &deltas)
{
	GConstVecWrapper delta(deltas.data(), 0);
	for(size_t i = 0; i < layerCount(); ++i)
	{
		size_t count = layer(i).countWeights();
		delta.setSize(count);
		if(count > 0)
			((GParameterizedLayer*)&layer(i))->applyDeltas(delta.vec());
		delta.setData(delta.vec().data() + count);
	}
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
	GNeuralNet nn;
	GLayerLinear* lh = new GLayerLinear(2, 3);
	GLayerActivation* lha = new GLayerActivation();
	GLayerLinear* lo = new GLayerLinear(3, 1);
	GLayerActivation* lho = new GLayerActivation();
	nn.addLayers(lh, lha, lo, lho);
	/*
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 3));
	nn.addLayer(new GLayerClassic(3, FLEXIBLE_SIZE));
	*/
	nn.beginIncrementalLearning(features.relation(), labels.relation());
	
	GSGDOptimizer optimizer(nn);
	optimizer.setLearningRate(0.175);
	optimizer.setMomentum(0.9);
	
	if(nn.countWeights() != 13)
		throw Ex("Wrong number of weights");
	GLayerLinear& layerOut = *lo;//*(GLayerClassic*)&nn.layer(1);
	layerOut.bias()[0] = 0.02; // w_0
	layerOut.weights()[0][0] = -0.01; // w_1
	layerOut.weights()[1][0] = 0.03; // w_2
	layerOut.weights()[2][0] = 0.02; // w_3
	GLayerLinear& layerHidden = *lh;//*(GLayerClassic*)&nn.layer(0);
	layerHidden.bias()[0] = -0.01; // w_4
	layerHidden.weights()[0][0] = -0.03; // w_5
	layerHidden.weights()[1][0] = 0.03; // w_6
	layerHidden.bias()[1] = 0.01; // w_7
	layerHidden.weights()[0][1] = 0.04; // w_8
	layerHidden.weights()[1][1] = -0.02; // w_9
	layerHidden.bias()[2] = -0.02; // w_10
	layerHidden.weights()[0][2] = 0.03; // w_11
	layerHidden.weights()[1][2] = 0.02; // w_12

	// Test forward prop
	double tol = 1e-12;
	GVec pat(2);
	pat.copy(features[0]);
	GVec pred(1);
	nn.predict(pat, pred);
	if(std::abs(pred[0] - 0.02034721575641) > tol) throw Ex("forward prop problem"); // tanh

	// Test that the output error is computed properly
	optimizer.optimizeIncremental(features[0], labels[0]);

	// Here is the math for why these results are expected:
	if(std::abs(lo->error()[0] - 0.9792471989888) > tol) throw Ex("problem computing output error"); // tanh

	// Test Back Prop
	if(std::abs(((GLayerClassic*)&nn.layer(0))->error()[0] + 0.00978306745006032) > tol) throw Ex("back prop problem"); // tanh
	if(std::abs(((GLayerClassic*)&nn.layer(0))->error()[1] - 0.02936050107376107) > tol) throw Ex("back prop problem"); // tanh
	if(std::abs(((GLayerClassic*)&nn.layer(0))->error()[2] - 0.01956232122115741) > tol) throw Ex("back prop problem"); // tanh

	// Test weight update
	if(std::abs(layerOut.bias()[0] - 0.191368259823049) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerOut.weights()[0][0] + 0.015310714964467731) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerOut.weights()[1][0] - 0.034112048752708297) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerOut.weights()[2][0] - 0.014175723281037968) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerHidden.bias()[0] + 0.011712036803760557) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerHidden.weights()[0][0] + 0.03) > tol) throw Ex("weight update problem"); // logistic & tanh
	if(std::abs(layerHidden.weights()[1][0] - 0.03119842576263239) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerHidden.bias()[1] - 0.015138087687908187) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerHidden.weights()[0][1] - 0.04) > tol) throw Ex("weight update problem"); // logistic & tanh
	if(std::abs(layerHidden.weights()[1][1] + 0.023596661381535732) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerHidden.bias()[2] + 0.016576593786297455) > tol) throw Ex("weight update problem"); // tanh
	if(std::abs(layerHidden.weights()[0][2] - 0.03) > tol) throw Ex("weight update problem"); // logistic & tanh
	if(std::abs(layerHidden.weights()[1][2] - 0.01760361565040822) > tol) throw Ex("weight update problem"); // tanh
}

void GNeuralNet_testHingeMath()
{
	GMatrix features(1, 2);
	GMatrix labels(1, 2);
	GNeuralNet nn;
	GActivationHinge* pAct1 = new GActivationHinge();
	nn.addLayer(new GLayerClassic(2, 3, pAct1));
	GActivationHinge* pAct2 = new GActivationHinge();
	nn.addLayer(new GLayerClassic(3, 2, pAct2));
	nn.beginIncrementalLearning(features.relation(), labels.relation());
	
	GSGDOptimizer optimizer(nn);
	optimizer.setLearningRate(0.1);
	
	if(nn.countWeights() != 22)
		throw Ex("Wrong number of weights");
	GLayerClassic& layerHidden = *(GLayerClassic*)&nn.layer(0);
	layerHidden.bias()[0] = 0.1;
	layerHidden.weights()[0][0] = 0.1;
	layerHidden.weights()[1][0] = 0.1;
	layerHidden.bias()[1] = 0.1;
	layerHidden.weights()[0][1] = 0.0;
	layerHidden.weights()[1][1] = 0.0;
	layerHidden.bias()[2] = 0.0;
	layerHidden.weights()[0][2] = 0.1;
	layerHidden.weights()[1][2] = -0.1;
	GLayerClassic& layerOut = *(GLayerClassic*)&nn.layer(1);
	layerOut.bias()[0] = 0.1;
	layerOut.weights()[0][0] = 0.1;
	layerOut.weights()[1][0] = 0.1;
	layerOut.weights()[2][0] = 0.1;
	layerOut.bias()[1] = -0.2;
	layerOut.weights()[0][1] = 0.1;
	layerOut.weights()[1][1] = 0.3;
	layerOut.weights()[2][1] = -0.1;
	features[0][0] = 0.3;
	features[0][1] = -0.2;
	labels[0][0] = 0.1;
	labels[0][1] = 0.0;
	GVec& hinge1 = pAct1->alphas();
	GVec& hinge2 = pAct2->alphas();
	hinge1.fill(0.0);
	hinge2.fill(0.0);
	optimizer.optimizeIncremental(features[0], labels[0]);
	if(std::abs(layerHidden.activation()[0] - 0.11) > 1e-9)
		throw Ex("failed");
	if(std::abs(layerHidden.activation()[1] - 0.1) > 1e-9)
		throw Ex("failed");
	if(std::abs(layerHidden.activation()[2] - 0.05) > 1e-9)
		throw Ex("failed");
	if(std::abs(layerOut.activation()[0] - 0.126) > 1e-9)
		throw Ex("failed");
	if(std::abs(layerOut.activation()[1] + 0.164) > 1e-9)
		throw Ex("failed");
	if(std::abs(layerOut.error()[0] + 0.025999999999999995) > 1e-9)
		throw Ex("failed");
	if(std::abs(layerOut.error()[1] - 0.164) > 1e-9)
		throw Ex("failed");
	if(std::abs(hinge1[0] - 1.6500700636595332E-5) > 1e-9)
		throw Ex("failed");
	if(std::abs(hinge1[1] - 4.614309333423788E-5) > 1e-9)
		throw Ex("failed");
	if(std::abs(hinge1[2] + 4.738184006484504E-6) > 1e-9)
		throw Ex("failed");
	if(std::abs(hinge2[0] + 4.064229382785025E-5) > 1e-9)
		throw Ex("failed");
	if(std::abs(hinge2[1] - 4.2982897628915964E-4) > 1e-9)
		throw Ex("failed");
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
	GVec in(TEST_INVERT_INPUTS);
	GVec outBefore(TEST_INVERT_INPUTS);
	GVec outAfter(TEST_INVERT_INPUTS);
	for(size_t i = 0; i < 30; i++)
	{
		GNeuralNet nn;
		for(size_t j = 0; j < layers; j++)
			nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, layerSize));
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		GUniformRelation rel(TEST_INVERT_INPUTS);
		nn.beginIncrementalLearning(rel, rel);
		nn.perturbAllWeights(0.5);
		in.fillUniform(rand);
		nn.predict(in, outBefore);
		for(size_t j = 0; j < 8; j++)
		{
			if(rand.next(2) == 0)
				nn.swapNodes((size_t)rand.next(layers), (size_t)rand.next(layerSize), (size_t)rand.next(layerSize));
			else
				nn.invertNode((size_t)rand.next(layers), (size_t)rand.next(layerSize));
		}
		nn.predict(in, outAfter);
		if(outBefore.squaredDistance(outAfter) > 1e-10)
			throw Ex("GNeuralNet::invertAndSwap failed");
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
		in.fillUniform(rand);
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
		if(outBefore.squaredDistance(outAfter) > 1e-10)
			throw Ex("Failed");
		nn1.predict(in, outAfter);
		if(outBefore.squaredDistance(outAfter) > 1e-10)
			throw Ex("Failed");

		// Check that they have matching weights
		size_t wc = nn1.countWeights();
		double* pW1 = new double[wc];
		std::unique_ptr<double[]> hW1(pW1);
		nn1.weights(pW1);
		double* pW2 = new double[wc];
		std::unique_ptr<double[]> hW2(pW2);
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
	GVec in(5);
	for(size_t i = 0; i < 20; i++)
	{
		GNeuralNet nn;
		GLayerClassic* pInputLayer = new GLayerClassic(FLEXIBLE_SIZE, 5);
		nn.addLayer(pInputLayer);
		nn.addLayer(new GLayerClassic(5, FLEXIBLE_SIZE));
		GUniformRelation relIn(5);
		GUniformRelation relOut(1);
		nn.beginIncrementalLearning(relIn, relOut);
		nn.perturbAllWeights(1.0);
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
		pInputLayer->renormalizeInput(ind, a, b, c, d);
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
		GNeuralNet nn;
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		GUniformRelation in(2);
		GUniformRelation out(3);
		nn.beginIncrementalLearning(in, out);
		nn.perturbAllWeights(1.0);
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
		((GLayerClassic*)&nn.layer(0))->transformWeights(*pTransInv, offset);

		// Predict again
		nn.predict(x2, y2);
		if(y1.squaredDistance(y2) > 1e-15)
			throw Ex("transformWeights failed");
	}
}

#define NN_TEST_DIMS 5

void GNeuralNet_testCompressFeatures(GRand& prng)
{
	GMatrix feat(50, NN_TEST_DIMS);
	for(size_t i = 0; i < feat.rows(); i++)
		feat[i].fillSphericalShell(prng);

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

	GLayerClassic *pUpstream = new GLayerClassic(1, layer.inputs());
	GLayerClassic &upstream = *pUpstream;

	GNeuralNet nn;
	nn.addLayers(pUpstream, pLayer);
	nn.beginIncrementalLearning(GUniformRelation(1), GUniformRelation(layer.outputs()));

	for(size_t i = 0; i < layer.inputs(); i++)
		upstream.weights()[0][i] = feature[i] + prng.normal();
	upstream.bias().fill(0.0);

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
	GLayerConvolutional2D *p_layer = (GLayerConvolutional2D *) GNeuralNetLayer::deserialize(root);
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
	GNeuralNet* pNN = GNeuralNet::fourier(m, period);
	std::unique_ptr<GNeuralNet> hNN(pNN);
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

// static
void GNeuralNet::test()
{
	GRand prng(0);
	GNeuralNet_testMath();
	GNeuralNet_testHingeMath();
	GNeuralNet_testBinaryClassification(&prng);
	GNeuralNet_testInvertAndSwap(prng);
	GNeuralNet_testNormalizeInput(prng);
	GNeuralNet_testTransformWeights(prng);
	GNeuralNet_testCompressFeatures(prng);
	GNeuralNet_testConvolutionalLayer2D(prng);
	GNeuralNet_testFourier();

	// Test with no hidden layers (logistic regression)
	{
		GNeuralNet* pNN = new GNeuralNet();
		pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		GAutoFilter af(pNN);
		af.basicTest(0.78, 0.895);
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
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
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
