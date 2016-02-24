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

using std::vector;

namespace GClasses {


GNeuralNet::GNeuralNet()
: GIncrementalLearner(),
m_learningRate(0.1),
m_momentum(0.0),
m_validationPortion(0.35),
m_minImprovement(0.002),
m_epochsPerValidationCheck(100)
{
}

GNeuralNet::GNeuralNet(const GDomNode* pNode)
: GIncrementalLearner(pNode),
m_validationPortion(0.35),
m_minImprovement(0.002),
m_epochsPerValidationCheck(100)
{
	m_learningRate = pNode->field("learningRate")->asDouble();
	m_momentum = pNode->field("momentum")->asDouble();

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
		*pOutMin = pAct->squash(-400.0, 0);
		*pOutMax = pAct->squash(400.0, 0);
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
	pOutWeights += m_layers[lay]->weightsToVector(pOutWeights);
}

void GNeuralNet::weights(double* pOutWeights) const
{
	for(size_t i = 0; i < m_layers.size(); i++)
		pOutWeights += m_layers[i]->weightsToVector(pOutWeights);
}

void GNeuralNet::setWeights(const double* pWeights, size_t lay)
{
	pWeights += m_layers[lay]->vectorToWeights(pWeights);
}

void GNeuralNet::setWeights(const double* pWeights)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		pWeights += m_layers[i]->vectorToWeights(pWeights);
}

void GNeuralNet::copyWeights(const GNeuralNet* pOther)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->copyWeights(pOther->m_layers[i]);
}

void GNeuralNet::copyStructure(const GNeuralNet* pOther)
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
}

void GNeuralNet::copyErrors(const GNeuralNet* pOther)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		GNeuralNetLayer* pLay = m_layers[i];
		GNeuralNetLayer* pOth = pOther->m_layers[i];
		pLay->error().copy(pOth->error());
	}
}

void GNeuralNet::perturbAllWeights(double deviation)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->perturbWeights(m_rand, deviation);
}

void GNeuralNet::maxNorm(double min, double max, bool output_layer)
{
	size_t layer_count = m_layers.size();
	if(!output_layer)
		layer_count--;
	for(size_t i = 0; i < layer_count; i++)
		m_layers[i]->maxNorm(min, max);
}

void GNeuralNet::invertNode(size_t lay, size_t node)
{
	GLayerClassic& layerUpStream = *(GLayerClassic*)m_layers[lay];
	GMatrix& w = layerUpStream.m_weights;
	for(size_t i = 0; i < w.rows(); i++)
		w[i][node] = -w[i][node];
	layerUpStream.bias()[node] = -layerUpStream.bias()[node];
	if(lay + 1 < m_layers.size())
	{
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

void GNeuralNet::swapNodes(size_t lay, size_t a, size_t b)
{
	GLayerClassic& layerUpStream = *(GLayerClassic*)m_layers[lay];
	layerUpStream.m_weights.swapColumns(a, b);
	std::swap(layerUpStream.bias()[a], layerUpStream.bias()[b]);
	if(lay + 1 < m_layers.size())
	{
		GLayerClassic& layerDownStream = *(GLayerClassic*)m_layers[lay + 1];
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
			for(size_t kk = 0; kk < inputs; kk++)
				dp += layerThisCur.m_weights[kk][j] * layerThatCur.m_weights[kk][j];
			dp += layerThisCur.bias()[j] * layerThatCur.bias()[j];
			if(dp < 0)
				invertNode(i, j); // invert it
		}
	}
}

void GNeuralNet::scaleWeights(double factor, bool scaleBiases, size_t startLayer, size_t layer_count)
{
	size_t end = std::min(startLayer + layer_count, m_layers.size());
	for(size_t i = startLayer; i < end; i++)
		m_layers[i]->scaleWeights(factor, scaleBiases);
}

void GNeuralNet::diminishWeights(double amount, bool diminishBiases, size_t startLayer, size_t layer_count)
{
	size_t end = std::min(startLayer + layer_count, m_layers.size());
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
		GMatrix& m2 = ((GLayerClassic*)m_layers[lay])->m_weights;
		for(size_t i = 0; i < m2.rows(); i++)
			m2[i] *= factor;
		((GLayerClassic*)m_layers[lay])->bias() *= factor;
	}
}

void GNeuralNet::regularizeActivationFunctions(double lambda)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->regularizeActivationFunction(lambda);
}

void GNeuralNet::contractWeights(double factor, bool contractBiases)
{
	size_t i = m_layers.size() - 1;
	GNeuralNetLayer* pLay = m_layers[i];
	pLay->error().fill(1.0);
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

void GNeuralNet::forwardProp(const GVec& row, size_t maxLayers)
{
	GNeuralNetLayer* pLay = m_layers[0];
	if(!pLay)
		throw Ex("No layers have been added to this neural network");
	pLay->feedForward(row);
	maxLayers = std::min(m_layers.size(), maxLayers);
	for(size_t i = 1; i < maxLayers; i++)
	{
		GNeuralNetLayer* pDS = m_layers[i];
		pDS->feedForward(pLay);
		pLay = pDS;
	}
}

double GNeuralNet::forwardPropSingleOutput(const GVec& row, size_t output)
{
	if(m_layers.size() == 1)
	{
		GLayerClassic& lay = *(GLayerClassic*)m_layers[0];
		lay.feedForwardToOneOutput(row, output);
		return lay.activation()[output];
	}
	else
	{
		GLayerClassic* pLay = (GLayerClassic*)m_layers[0];
		pLay->feedForward(row);
		for(size_t i = 1; i + 1 < m_layers.size(); i++)
		{
			GLayerClassic* pDS = (GLayerClassic*)m_layers[i];
			pDS->feedForward(pLay->activation());
			pLay = pDS;
		}
		GLayerClassic* pDS = (GLayerClassic*)m_layers[m_layers.size() - 1];
		pDS->feedForwardToOneOutput(pLay->activation(), output);
		return pDS->activation()[output];
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

double GNeuralNet::sumSquaredPredictionError(const GVec& target)
{
	GNeuralNetLayer& outputLay = *m_layers[m_layers.size() - 1];
	return target.squaredDistance(outputLay.activation());
}

// virtual
void GNeuralNet::predict(const GVec& in, GVec& out)
{
	forwardProp(in);
	copyPrediction(out);
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
	GVec pFullRow(features.cols());
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
				{
					if(dSumSquaredError == 0.0)
						break;
					dBestError = dSumSquaredError;
				}
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
		m_layers[i]->resetWeights(m_rand);
}

// virtual
void GNeuralNet::trainIncremental(const GVec& in, const GVec& out)
{
	forwardProp(in);
	backpropagate(out);
	descendGradient(in, m_learningRate, m_momentum);
}

void GNeuralNet::trainIncrementalBatch(const GMatrix& features, const GMatrix& labels)
{
	const GVec& feat0 = features[0];
	const GVec& targ0 = labels[0];
	forwardProp(feat0);
	backpropagate(targ0);
	updateDeltas(feat0, 0.0);
	for(size_t i = 1; i < features.rows(); i++)
	{
		const GVec& feat = features[i];
		const GVec& targ = labels[i];
		forwardProp(feat);
		backpropagate(targ);
		updateDeltas(feat, 1.0);
	}
	applyDeltas(m_learningRate / features.rows());
}

void GNeuralNet::trainIncrementalWithDropout(const GVec& in, const GVec& out, double probOfDrop)
{
	if(m_momentum != 0.0)
		throw Ex("Sorry, this implementation is not compatible with momentum");

	// Forward prop with dropout
	GNeuralNetLayer* pLay = m_layers[0];
	pLay->feedForward(in);
	pLay->dropOut(m_rand, probOfDrop);
	size_t maxLayers = m_layers.size();
	for(size_t i = 1; i < maxLayers; i++)
	{
		GNeuralNetLayer* pDS = m_layers[i];
		pDS->feedForward(pLay);
		if(i + 1 < maxLayers)
		{
			pLay->dropOut(m_rand, probOfDrop);
			pLay = pDS;
		}
	}

	backpropagate(out);
	descendGradient(in, m_learningRate, 0.0);
}

void GNeuralNet::backpropagateErrorAlreadySet()
{
	size_t i = m_layers.size() - 1;
	GNeuralNetLayer* pLay = m_layers[i];
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

void GNeuralNet::backpropagate(const GVec& target, size_t startLayer)
{
	size_t i = std::min(startLayer, m_layers.size() - 1);
	GNeuralNetLayer* pLay = m_layers[i];
	pLay->computeError(target);
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

double GNeuralNet::backpropagateAndNormalizeErrors(const GVec& target, double alpha)
{
	size_t i = m_layers.size() - 1;
	GNeuralNetLayer* pLay = m_layers[i];
	pLay->computeError(target);
	pLay->deactivateError();
	double errMag = sqrt(pLay->error().squaredMagnitude());
	while(i > 0)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		pUpStream->deactivateError();
		double mag = std::sqrt(pUpStream->error().GVec::squaredMagnitude());
		pLay->scaleWeights(1.0 - alpha + alpha * errMag / mag, true);
		pLay = pUpStream;
		i--;
	}
	return errMag;
}

void GNeuralNet::backpropagateFromLayer(GNeuralNetLayer* pDownstream)
{
	GNeuralNetLayer* pLay = pDownstream;
	for(size_t i = m_layers.size(); i > 0; i--)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		pUpStream->deactivateError();
		pLay = pUpStream;
	}
}

void GNeuralNet::backpropagateFromLayerAndNormalizeErrors(GNeuralNetLayer* pDownstream, double errMag, double alpha)
{
	GNeuralNetLayer* pLay = pDownstream;
	for(size_t i = m_layers.size(); i > 0; i--)
	{
		GNeuralNetLayer* pUpStream = m_layers[i - 1];
		pLay->backPropError(pUpStream);
		pUpStream->deactivateError();
		double mag = std::sqrt(pUpStream->error().GVec::squaredMagnitude());
		pLay->scaleWeights(1.0 - alpha + alpha * errMag / mag, true);
		pLay = pUpStream;
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
			GLayerClassic* pUpStream2 = (GLayerClassic*)m_layers[i - 1];
			pLay->backPropError(pUpStream2);
			pUpStream2->deactivateError();
			pLay = pUpStream2;
			i--;
		}
	}
}

void GNeuralNet::descendGradient(const GVec& feat, double learning_rate, double momentumTerm)
{
	GNeuralNetLayer* pLay = m_layers[0];
	pLay->updateDeltas(feat, momentumTerm);
	pLay->applyDeltas(learning_rate);
	GNeuralNetLayer* pUpStream = pLay;
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		pLay = m_layers[i];
		pLay->updateDeltas(pUpStream, momentumTerm);
		pLay->applyDeltas(learning_rate);
		pUpStream = pLay;
	}
}

void GNeuralNet::descendGradientSingleOutput(size_t outputNeuron, const GVec& feat, double learning_rate, double momentumTerm)
{
	size_t i = m_layers.size() - 1;
	GLayerClassic* pLay = (GLayerClassic*)m_layers[i];
	if(i == 0)
		pLay->updateWeightsSingleNeuron(outputNeuron, feat, learning_rate, momentumTerm);
	else
	{
		GLayerClassic* pUpStream = (GLayerClassic*)m_layers[i - 1];
		pLay->updateWeightsSingleNeuron(outputNeuron, pUpStream->activation(), learning_rate, momentumTerm);
		for(i--; i > 0; i--)
		{
			pLay = pUpStream;
			pUpStream = (GLayerClassic*)m_layers[i - 1];
			pLay->updateDeltas(pUpStream->activation(), momentumTerm);
			pLay->applyDeltas(learning_rate);
		}
		pLay = (GLayerClassic*)m_layers[0];
		pLay->updateDeltas(feat, momentumTerm);
		pLay->applyDeltas(learning_rate);
	}
}

void GNeuralNet::updateDeltas(const GVec& feat, double momentumTerm)
{
	GNeuralNetLayer* pLay = m_layers[0];
	pLay->updateDeltas(feat, momentumTerm);
	GNeuralNetLayer* pUpStream = pLay;
	for(size_t i = 1; i < m_layers.size(); i++)
	{
		pLay = m_layers[i];
		pLay->updateDeltas(pUpStream, momentumTerm);
		pUpStream = pLay;
	}
}

void GNeuralNet::applyDeltas(double learning_rate)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->applyDeltas(learning_rate);
}

void GNeuralNet::gradientOfInputs(GVec& outGradient)
{
	GMatrix& w = ((GLayerClassic*)m_layers[0])->m_weights;
	GVec& err = ((GLayerClassic*)m_layers[0])->error();
	for(size_t i = 0; i < w.rows(); i++)
		outGradient[i] = -w[i].dotProduct(err);
}

void GNeuralNet::gradientOfInputsSingleOutput(size_t outputNeuron, GVec& outGradient)
{
	if(m_layers.size() != 1)
	{
		gradientOfInputs(outGradient);
		return;
	}
	GMatrix& w = ((GLayerClassic*)m_layers[0])->m_weights;
	GAssert(outputNeuron < w.cols());

	GVec& err = ((GLayerClassic*)m_layers[0])->error();
	for(size_t i = 0; i < w.rows(); i++)
		outGradient[i] = -err[outputNeuron] * w[i][outputNeuron];
}

#ifndef MIN_PREDICT
void GNeuralNet::autoTune(GMatrix& features, GMatrix& labels)
{
	// Try a plain-old single-layer network
	size_t hidden = std::max((size_t)4, (features.cols() + 3) / 4);
	std::unique_ptr<GNeuralNet> hCand0(new GNeuralNet());
	hCand0->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	std::unique_ptr<GNeuralNet> hCand1;
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
	std::unique_ptr<GMatrix> hFeat;
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
}

void GNeuralNet::printSummary(std::ostream& stream)
{
	stream << "Neural Net ( " << to_str(layerCount()) << " layers )\n";
	for(size_t i = 0; i < layerCount(); i++)
	{
		stream << "  Layer " << to_str(i);
		((GLayerClassic*)&layer(i))->printSummary(stream);
	}
}

void GNeuralNet::containIntrinsics(GMatrix& intrinsics)
{
	size_t dims = intrinsics.cols();
	GNeuralNetLayer& lay = layer(0);
	if(lay.inputs() != dims)
		throw Ex("Mismatching number of columns and inputs");
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

GMatrix* GNeuralNet::compressFeatures(GMatrix& features)
{
	GLayerClassic& lay = *(GLayerClassic*)&layer(0);
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
	GVec pat(2);
	pat.copy(features[0]);
	GVec pred(1);
	nn.predict(pat, pred);
	// Here is the math (done by hand) for why these results are expected:
	// Row: {0, -0.7, 1}
	// o_1 = squash(w_4*1+w_5*x+w_6*y) = 1/(1+exp(-(-.01*1-.03*0+.03*(-.7)))) = 0.4922506205862
	// o_2 = squash(w_7*1+w_8*x+w_9*y) = 1/(1+exp(-(.01*1+.04*0-.02*(-.7)))) = 0.50599971201659
	// o_3 = squash(w_10*1+w_11*x+w_12*y) = 1/(1+exp(-(-.02*1+.03*0+.02*(-.7)))) = 0.49150081873869
	// o_0 = squash(w_0*1+w_1*o_1+w_2*o_2+w_3*o_3) = 1/(1+exp(-(.02*1-.01*.4922506205862+.03*.50599971201659+.02*.49150081873869))) = 0.51002053349535
	//if(std::abs(pat[2] - 0.51002053349535) > tol) throw Ex("forward prop problem"); // logistic
	if(std::abs(pred[0] - 0.02034721575641) > tol) throw Ex("forward prop problem"); // tanh

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

void GNeuralNet_testHingeMath()
{
	GMatrix features(1, 2);
	GMatrix labels(1, 2);
	GNeuralNet nn;
	GActivationHinge* pAct1 = new GActivationHinge();
	nn.addLayer(new GLayerClassic(2, 3, pAct1));
	GActivationHinge* pAct2 = new GActivationHinge();
	nn.addLayer(new GLayerClassic(3, 2, pAct2));
	nn.setLearningRate(0.1);
	nn.beginIncrementalLearning(features.relation(), labels.relation());
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
	nn.trainIncremental(features[0], labels[0]);
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
	GVec in(8);
	in[0] = 0.0;
	in[1] = 1.0;
	in[2] = 0.0;
	in[3] = 2.0;
	in[4] = 1.0;
	in[5] = 3.0;
	in[6] = 2.0;
	in[7] = 2.0;
	double learning_rate = 2.0;
	layer.feedForward(in);
	const double expected_activation[] = { 2.0, -1.0, 9.0, 7.0, 5.0, 0.0, 10.0, 8.0 };
	if(GVec::squaredDistance(layer.activation().data(), expected_activation, 8) > 1e-9)
		throw Ex("incorrect activation");
	GVec target(8);
	target[0] = 2.0;
	target[1] = 0.0;
	target[2] = 11.0;
	target[3] = 10.0;
	target[4] = 9.0;
	target[5] = 5.0;
	target[6] = 16.0;
	target[7] = 15.0;
	layer.computeError(target);
	const double expected_err[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
	if(GVec::squaredDistance(layer.error().data(), expected_err, 8) > 1e-9)
		throw Ex("incorrect error");
	layer.deactivateError();
	// Note that this test does not cover backPropError().
	layer.updateDeltas(in, 0.0);
	layer.applyDeltas(learning_rate);
	const double expected_k1[] = { 0.0, 9.0, 18.0 };
	const double expected_k2[] = { 2.0, 11.0, 22.0 };
	const double expected_k3[] = { 28.0, 46.0, 37.0 };
	const double expected_k4[] = { 35.0, 55.0, 47.0 };
	if(GVec::squaredDistance(k[0].data(), expected_k1, 3) > 1e-9)
		throw Ex("incorrect weights");
	if(GVec::squaredDistance(k[1].data(), expected_k2, 3) > 1e-9)
		throw Ex("incorrect weights");
	if(GVec::squaredDistance(k[2].data(), expected_k3, 3) > 1e-9)
		throw Ex("incorrect weights");
	if(GVec::squaredDistance(k[3].data(), expected_k4, 3) > 1e-9)
		throw Ex("incorrect weights");
	const double expected_bias[] = { 8.0, 11.0, 18.0, 21.0 };
	if(GVec::squaredDistance(layer.bias().data(), expected_bias, 4) > 1e-9)
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
		GVec pWeights(weightCount);
		GVec feat(5);
		GVec target(10);
		GVec pOutput(10);
		GVec pFeatureGradient(5);
		GVec pEmpiricalGradient(5);
		for(size_t j = 0; j < weightCount; j++)
			pWeights[j] = pRand->normal() * 0.8;
		nn.setWeights(pWeights.data());

		// Compute target output
		feat.fill(0.0);
		nn.predict(feat, target);

		// Move away from the goal and compute baseline error
		for(int j = 0; j < 5; j++)
			feat[j] += pRand->normal() * 0.1;
		nn.predict(feat, pOutput);
		double sseBaseline = target.squaredDistance(pOutput);

		// Compute the feature gradient
		nn.forwardProp(feat);
		nn.backpropagate(target);
		nn.gradientOfInputs(pFeatureGradient);
		pFeatureGradient *= 2.0;

		// Empirically measure gradient
		for(int j = 0; j < 5; j++)
		{
			feat[j] += 0.0001;
			nn.predict(feat, pOutput);
			double sse = target.squaredDistance(pOutput);
			pEmpiricalGradient[j] = (sse - sseBaseline) / 0.0001;
			feat[j] -= 0.0001;
		}

		// Check it
		double corr = pFeatureGradient.correlation(pEmpiricalGradient);
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
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 5));
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
		GNeuralNetLayer* pInputLayer = &nn.layer(0);
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
	GNeuralNet_testInputGradient(&prng);
	GNeuralNet_testInvertAndSwap(prng);
	GNeuralNet_testNormalizeInput(prng);
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










GBackPropThroughTime::GBackPropThroughTime(GNeuralNet& transition, GNeuralNet& observation, size_t unfoldDepth)
: m_transition(transition), m_observation(observation), m_unfoldDepth(unfoldDepth), m_errorNormalizationTerm(0.0)
{
	/// Check parameters
	if(unfoldDepth < 1)
		throw Ex("unfoldDepth must be > 0");
	if(observation.layer(0).inputs() < transition.outputLayer().outputs())
		throw Ex("The observation function must accept at least as many inputs as the transition function outputs");
	m_obsParamCount = observation.layer(0).inputs() - transition.outputLayer().outputs();

	/// Create the unfolded instances
	for(size_t i = 0; i < m_unfoldDepth; i++)
	{
		GNeuralNet* pNN = new GNeuralNet();
		m_parts.push_back(pNN);
		pNN->copyStructure(&transition);
	}

	// Allocate a buffer
	m_buf = new double[std::max(transition.layer(0).inputs(), observation.layer(0).inputs())];
	m_unfoldReciprocal = 1.0 / m_unfoldDepth;
}

GBackPropThroughTime::~GBackPropThroughTime()
{
	for(size_t i = 0; i < m_parts.size(); i++)
		delete(m_parts[i]);
	delete[] m_buf;
}

void GBackPropThroughTime::trainIncremental(const GVec& initialState, const GMatrix& controls, const GVec& obsParams, const GVec& targetObs)
{
	size_t transInputs = m_transition.layer(0).inputs();
	size_t transOutputs = m_transition.outputLayer().outputs();
	size_t obsInputs = m_observation.layer(0).inputs();
	GAssert(initialState.size() == transOutputs);
	GAssert(initialState.size() + controls.cols() == transInputs);
	GAssert(controls.rows() == m_unfoldDepth);
	GAssert(obsParams.size() == m_obsParamCount);
	GAssert(targetObs.size() == m_observation.outputLayer().outputs());
	GVecWrapper vwTrans(m_buf, transInputs);
	GVecWrapper vwObs(m_buf, obsInputs);

	// Forward Prop
	GVec::copy(m_buf, initialState.data(), transOutputs);
	for(size_t i = 0; i < m_unfoldDepth; i++)
	{
		GVec::copy(m_buf + transOutputs, controls[i].data(), transInputs - transOutputs);
		m_parts[i]->forwardProp(vwTrans.vec());
		GVec::copy(m_buf, m_parts[i]->outputLayer().activation().data(), transOutputs);
	}
	GVec::copy(m_buf + transOutputs, obsParams.data(), obsInputs - transOutputs);
	m_observation.forwardProp(vwObs.vec());

	// Back Prop
	double errMag = m_observation.backpropagateAndNormalizeErrors(targetObs, m_errorNormalizationTerm);
	GNeuralNet* pDownStreamNet = &m_observation;
	for(size_t i = m_unfoldDepth - 1; i < m_unfoldDepth; i--)
	{
		pDownStreamNet->layer(0).backPropError(&m_parts[i]->outputLayer());
		m_parts[i]->backpropagateFromLayerAndNormalizeErrors(&pDownStreamNet->layer(0), errMag, m_errorNormalizationTerm);
		pDownStreamNet = m_parts[i];
	}

	// Update weights
	GVec::copy(m_buf, initialState.data(), transOutputs);
	for(size_t i = 0; i < m_unfoldDepth; i++)
	{
		m_transition.copyErrors(m_parts[i]);
		GVec::copy(m_buf + transOutputs, controls[i].data(), transInputs - transOutputs);
		m_transition.descendGradient(vwTrans.vec(), m_transition.learningRate() * m_unfoldReciprocal, m_transition.momentum());
		GVec::copy(m_buf, m_parts[i]->outputLayer().activation().data(), transOutputs);
	}
	GVec::copy(m_buf + transOutputs, obsParams.data(), obsInputs - transOutputs);
	m_observation.descendGradient(vwObs.vec(), m_observation.learningRate(), m_observation.momentum());
}














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

