/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Luke B. Godfrey,
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

// todo: don't assume single column features/labels

#include "GNeuralDecomposition.h"
#include "GActivation.h"
#include "GDom.h"
#include "GBlock.h"

namespace GClasses {

GNeuralDecomposition::GNeuralDecomposition()
: GIncrementalLearner(),
m_nn(nullptr),
m_pContext(nullptr),
m_pOptimizer(nullptr),
m_regularization(0.01),
m_learningRate(0.001),
m_featureScale(1.0),
m_featureBias(0.0),
m_outputScale(1.0),
m_outputBias(0.0),
m_linearUnits(10),
m_softplusUnits(10),
m_sigmoidUnits(10),
m_sinusoidUnits(100),
m_epochs(1000),
m_filterLogarithm(false),
m_autoFilter(true),
m_lockPairs(false)
{
}

GNeuralDecomposition::GNeuralDecomposition(const GDomNode *pNode)
: GIncrementalLearner(pNode)
{
	m_nn = new GNeuralNet(pNode->field("nn"));
	m_pContext = nullptr;
	m_pOptimizer = nullptr;
	m_regularization = pNode->field("regularization")->asDouble();
	m_learningRate = pNode->field("learningRate")->asDouble();
	m_featureScale = pNode->field("featureScale")->asDouble();
	m_featureBias = pNode->field("featureBias")->asDouble();
	m_outputScale = pNode->field("outputScale")->asDouble();
	m_outputBias = pNode->field("outputBias")->asDouble();
	m_linearUnits = (size_t)pNode->field("linearUnits")->asInt();
	m_sinusoidUnits = (size_t)pNode->field("sinusoidUnits")->asInt();
	m_softplusUnits = (size_t)pNode->field("softplusUnits")->asInt();
	m_sigmoidUnits = (size_t)pNode->field("sigmoidUnits")->asInt();
	m_epochs = (size_t)pNode->field("epochs")->asInt();
	m_filterLogarithm = pNode->field("filterLogarithm")->asBool();
	m_autoFilter = pNode->field("autoFilter")->asBool();
	m_lockPairs = pNode->field("lockPairs")->asBool();
}

GNeuralDecomposition::~GNeuralDecomposition()
{
	delete(m_pOptimizer);
	delete(m_nn);
}

void GNeuralDecomposition::trainOnSeries(const GMatrix &series)
{
	// Generate features as equally spaced values between 0 and 1
	GMatrix features(series.rows(), 1);
	for(size_t i = 0; i < series.rows(); i++)
	{
		features[i][0] = i / (double) (series.rows());
	}

	// Train normally
	train(features, series);
}

GMatrix *GNeuralDecomposition::extrapolate(double start, double length, double step, bool outputFeatures)
{
	// note: this method assumes the network was trained with single-column features

	size_t rows = (size_t) (length / step);
	GVec x(1);
	x[0] = start;

	GMatrix *output = new GMatrix(rows, m_nn->outputLayer().outputs() + (outputFeatures ? 1 : 0));
	GVec tmp(output->cols());

	for(size_t i = 0; i < rows; i++)
	{
		GVec &out = output->row(i);
		if(outputFeatures)
		{
			out[0] = x[0] * m_featureScale + m_featureBias;
		}
		predict(x, tmp);
		out.put(outputFeatures ? 1 : 0, tmp);
		x[0] += step;
	}

	return output;
}

GMatrix *GNeuralDecomposition::extrapolate(const GMatrix &features)
{
	// note: this method assumes the network was trained with single-column features
	// note: this method uses featureBias and featureScale to normalize features

	GMatrix *output = new GMatrix(features.rows(), m_nn->outputLayer().outputs());
	GVec in(1);

	for(size_t i = 0; i < features.rows(); i++)
	{
		in[0] = (features[i][0] - m_featureBias) / m_featureScale;
		GVec& out = output->row(i);
		predict(in, out);
	}

	return output;
}

// MARK: GSupervisedLearner virtual methods

GDomNode *GNeuralDecomposition::serialize(GDom *pDoc) const
{
	GDomNode *pNode = baseDomNode(pDoc, "GNeuralDecomposition");
	pNode->addField(pDoc, "nn", m_nn->serialize(pDoc));
	pNode->addField(pDoc, "regularization", pDoc->newDouble(m_regularization));
	pNode->addField(pDoc, "learningRate", pDoc->newDouble(m_learningRate));
	pNode->addField(pDoc, "featureScale", pDoc->newDouble(m_featureScale));
	pNode->addField(pDoc, "featureBias", pDoc->newDouble(m_featureBias));
	pNode->addField(pDoc, "outputScale", pDoc->newDouble(m_outputScale));
	pNode->addField(pDoc, "outputBias", pDoc->newDouble(m_outputBias));
	pNode->addField(pDoc, "linearUnits", pDoc->newInt(m_linearUnits));
	pNode->addField(pDoc, "sinusoidUnits", pDoc->newInt(m_sinusoidUnits));
	pNode->addField(pDoc, "softplusUnits", pDoc->newInt(m_softplusUnits));
	pNode->addField(pDoc, "sigmoidUnits", pDoc->newInt(m_sigmoidUnits));
	pNode->addField(pDoc, "epochs", pDoc->newInt(m_epochs));
	pNode->addField(pDoc, "filterLogarithm", pDoc->newBool(m_filterLogarithm));
	pNode->addField(pDoc, "autoFilter", pDoc->newBool(m_autoFilter));
	pNode->addField(pDoc, "lockPairs", pDoc->newBool(m_lockPairs));
	return pNode;
}

void GNeuralDecomposition::predict(const GVec& pIn, GVec& pOut)
{
	if(!m_pContext)
		m_pContext = m_nn->newContext();
	m_pContext->forwardProp(pIn, pOut);
	if(m_filterLogarithm)
	{
		pOut[0] = exp((pOut[0] * 0.1 * m_outputScale + m_outputBias) * log(10));
	}
	else
	{
		pOut[0] = pOut[0] * 0.1 * m_outputScale + m_outputBias;
	}
}

void GNeuralDecomposition::predictDistribution(const GVec& pIn, GPrediction *pOut)
{
	throw Ex("Sorry, not implemented");
}

void GNeuralDecomposition::trainInner(const GMatrix &features, const GMatrix &labels)
{
	if(features.cols() != 1)
	{
		throw Ex("Neural decomposition expects single-column input features.");
	}

	if(features.rows() != labels.rows())
	{
		throw Ex("Features and labels must have the same number of rows.");
	}

	if(m_sinusoidUnits == 0)
	{
		m_sinusoidUnits = features.rows();
	}

	if(m_autoFilter)
	{
		m_featureScale	= features.columnMax(0) - features.columnMin(0);
		m_featureBias	= features.columnMin(0);
		m_outputScale	= labels.columnMax(0) - labels.columnMin(0);
		m_outputBias	= labels.columnMin(0);
	}

	if(m_filterLogarithm)
	{
		m_outputScale	= log(m_outputScale) / log(10);
		m_outputBias	= log(m_outputBias) / log(10);
	}

	beginIncrementalLearning(features, labels);

	GRandomIndexIterator ii(labels.rows(), rand());
	size_t i;

	for(size_t epoch = 0; epoch < m_epochs; epoch++)
	{
		ii.reset();
		while(ii.next(i))
		{
			trainIncremental(features[i], labels[i]);
		}
	}
}

// MARK: GIncrementalLearner virtual methods

void GNeuralDecomposition::beginIncrementalLearningInner(const GRelation &featureRel, const GRelation &labelRel)
{
	if(featureRel.size() != 1)
	{
		throw Ex("Neural decomposition expects single-column input features.");
	}

	delete(m_nn);
	m_nn = new GNeuralNet();

	GLayer& l1 = m_nn->newLayer();
	GBlockLinear* b1 = new GBlockLinear(m_sinusoidUnits + m_linearUnits + m_softplusUnits + m_sigmoidUnits);
	l1.add(b1);

	GLayer& l2 = m_nn->newLayer();
	l2.add(new GBlockSine(m_sinusoidUnits), 0);
	l2.add(new GBlockIdentity(m_linearUnits), m_sinusoidUnits);
	l2.add(new GBlockSoftPlus(m_softplusUnits), m_sinusoidUnits + m_linearUnits);
	l2.add(new GBlockTanh(m_sigmoidUnits), m_sinusoidUnits + m_linearUnits + m_softplusUnits);

	GLayer& l3 = m_nn->newLayer();
	GBlockLinear* b3 = new GBlockLinear(labelRel.size());
	l3.add(b3);

	// Prepare for learning
	delete(m_pOptimizer);
	m_pOptimizer = new GSGDOptimizer(*m_nn);
	m_pOptimizer->setLearningRate(m_learningRate);
	m_nn->resize(featureRel.size(), labelRel.size());
	m_nn->resetWeights(rand());

	// Initialize weights

	// sinusoids
	{
		// initialize sinusoid nodes inspired by the DFT
		GVec& bias = b1->bias();
		GMatrix& weights = b1->weights();
		for(size_t i = 0; i < m_sinusoidUnits / 2; i++)
		{
			for(size_t j = 0; j < weights.rows(); j++)
			{
				weights[j][2 * i] = 2.0 * M_PI * (i + 1);
				weights[j][2 * i + 1] = 2.0 * M_PI * (i + 1);
			}
			bias[2 * i] = 0.5 * M_PI;
			bias[2 * i + 1] = M_PI;
		}
	}

	// g(t)
	{
		// initialize g(t) weights near identity
		GVec& bias = b1->bias();
		GMatrix& weights = b1->weights();
		for(size_t j = m_sinusoidUnits; j < m_sinusoidUnits + m_linearUnits + m_softplusUnits; j++)
		{
			bias[j] = 0.0;
			for(size_t i = 0; i < featureRel.size(); i++)
				weights[i][j] = rand().normal() * 0.3;
		}
	}

	// output layer
	{
		// initialize output weights near zero
		GVec& bias = b3->bias();
		GMatrix& weights = b3->weights();
		bias.fill(0.0);
		weights.fillNormal(rand(), 0.001);
	}
}

void GNeuralDecomposition::trainIncremental(const GVec& pIn, const GVec& pOut)
{
	// L1 regularization
	m_nn->outputLayer().diminishWeights(m_learningRate * m_regularization, false);
	
	// Filter input
	GVec in(1);
	in[0] = (pIn[0] - m_featureBias) / m_featureScale;

	// Filter output
	GVec out(1);
	if(m_filterLogarithm)
	{
		out[0] = 10.0 * (log(pOut[0]) / log(10) - m_outputBias) / m_outputScale;
	}
	else
	{
		out[0] = 10.0 * (pOut[0] - m_outputBias) / m_outputScale;
	}

	// Backpropagation
	m_pOptimizer->optimizeIncremental(in, out);

	// Lock pairs
	if(m_lockPairs)
	{
		// initialize sinusoid nodes inspired by the DFT
		GVec& bias = ((GBlockLinear*)&m_nn->layer(0).block(0))->bias();
		GMatrix& weights = ((GBlockLinear*)&m_nn->layer(0).block(0))->weights();
		for(size_t i = 0; i < m_sinusoidUnits / 2; i++)
		{
			for(size_t j = 0; j < weights.rows(); j++)
			{
				double t = 0.5 * (weights[j][2 * i] + weights[j][2 * i + 1]);
				weights[j][2 * i] = t;
				weights[j][2 * i + 1] = t;
			}
			bias[2 * i] = 0.5 * M_PI;
			bias[2 * i + 1] = M_PI;
		}
	}

}

void GNeuralDecomposition::trainSparse(GSparseMatrix &features, GMatrix &labels)
{
	// todo: implement this
	throw Ex("Neural decomposition does not work with trainSparse!");
}

// static
// todo: determine why the threshold has to be so high
void GNeuralDecomposition::test()
{
	double step = 0.02;
	double threshold = 0.95;

	size_t testSize = (size_t)(1.0 / step);

	GMatrix series(testSize, 1), test(testSize, 1);
	for(size_t i = 0; i < testSize * 2; i++)
	{
		double x = i / (double) testSize;
		double y = sin(4.1 * M_PI * x) + x;

		if(i < testSize)
			series[i][0] = y;
		else
			test[i - testSize][0] = y;
	}

	GNeuralDecomposition nd;
	nd.setEpochs(10000);
	nd.trainOnSeries(series);
	GMatrix *out = nd.extrapolate(1.0, 1.0, 1.0 / testSize);

	double rmse = 0.0;
	for(size_t i = 0; i < testSize; i++)
	{
		double err = test[i][0] - out->row(i)[0];
		rmse += err * err;
	}
	rmse = sqrt(rmse / testSize);

	delete out;

	if(rmse > threshold)
	{
		throw Ex("Neural decomposition failed to extrapolate toy problem.");
	}
}

}
