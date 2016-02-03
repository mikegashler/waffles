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

namespace GClasses {

GNeuralDecomposition::GNeuralDecomposition()
: GIncrementalLearner(), m_regularization(0.01), m_learningRate(0.001), m_featureScale(1.0), m_featureBias(0.0), m_outputScale(1.0), m_outputBias(0.0), m_linearUnits(10), m_softplusUnits(10), m_sigmoidUnits(10), m_sinusoidUnits(0), m_epochs(1000), m_filterLogarithm(false), m_autoFilter(true)
{
	m_nn = new GNeuralNet();
}

GNeuralDecomposition::GNeuralDecomposition(const GDomNode *pNode)
: GIncrementalLearner(pNode)
{
	m_nn = new GNeuralNet(pNode->field("nn"));
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
}

GNeuralDecomposition::~GNeuralDecomposition()
{
	delete m_nn;
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
	return pNode;
}

void GNeuralDecomposition::predict(const GVec& pIn, GVec& pOut)
{
	m_nn->predict(pIn, pOut);
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
	m_nn->predictDistribution(pIn, pOut);
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
	
	beginIncrementalLearning(features.relation(), labels.relation());
	
	GRandomIndexIterator ii(labels.rows(), m_nn->rand());
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
	
	if(m_sinusoidUnits == 0)
	{
		throw Ex("You must set the number of sinusoid units before calling beginIncrementalLearning!");
	}
	
	m_nn->setLearningRate(m_learningRate);
	m_nn->beginIncrementalLearning(featureRel, labelRel);
	
	// Layer 1: Sinusoids + g(t)
	GLayerMixed *pMix = new GLayerMixed();
	{
		// sinusoids
		GLayerClassic *pSine = new GLayerClassic(featureRel.size(), m_sinusoidUnits, new GActivationSin());
		{
			// initialize sinusoid nodes inspired by the DFT
			GVec& bias = pSine->bias();
			GMatrix& weights = pSine->weights();
			for(size_t i = 0; i < pSine->outputs() / 2; i++)
			{
				for(size_t j = 0; j < pSine->inputs(); j++)
				{
					weights[j][2 * i] = 2.0 * M_PI * (i + 1);
					weights[j][2 * i + 1] = 2.0 * M_PI * (i + 1);
				}
				bias[2 * i] = 0.5 * M_PI;
				bias[2 * i + 1] = M_PI;
			}
		}
		pMix->addComponent(pSine);
		
		// g(t)
		
		GLayerClassic *pLinear = new GLayerClassic(featureRel.size(), m_linearUnits, new GActivationIdentity());
		{
			// initialize g(t) weights near identity
			pLinear->setWeightsToIdentity();
		}
		pMix->addComponent(pLinear);
		
		GLayerClassic *pSoftplus = new GLayerClassic(featureRel.size(), m_softplusUnits, new GActivationSoftPlus());
		{
			// initialize g(t) weights near identity
			pSoftplus->setWeightsToIdentity();
		}
		pMix->addComponent(pSoftplus);
		
		GLayerClassic *pSigmoid = new GLayerClassic(featureRel.size(), m_sigmoidUnits, new GActivationTanH());
		{
			// initialize g(t) weights near identity
			pSigmoid->setWeightsToIdentity();
		}
		pMix->addComponent(pSigmoid);
	}
	m_nn->addLayer(pMix);
	
	// Layer 2: Output
	GLayerClassic *pOutput = new GLayerClassic(pMix->outputs(), labelRel.size(), new GActivationIdentity());
	{
		// initialize output weights near zero
		pOutput->bias().fill(0.0);
		pOutput->weights().setAll(0.0);
		pOutput->perturbWeights(m_nn->rand(), 0.001);
	}
	m_nn->addLayer(pOutput);
}

void GNeuralDecomposition::trainIncremental(const GVec& pIn, const GVec& pOut)
{
	// L1 regularization
	m_nn->outputLayer().diminishWeights(m_learningRate * m_regularization, true);
	
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
	m_nn->trainIncremental(in, out);
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
	double threshold = 0.5;
	
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
