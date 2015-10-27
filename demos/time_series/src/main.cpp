/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include <exception>
#include <iostream>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GFourier.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GActivation.h>
#include <GClasses/GPlot.h>
#include <GClasses/GVec.h>
#include <GClasses/GHillClimber.h>
#include <math.h>
#include <fstream>
#ifdef WINDOWS
#include	<direct.h>
#endif

using namespace GClasses;
using std::cerr;
using std::cout;
using std::string;

#define SOFTPLUS_NODES 2
#define IDENTITY_NODES 4
#define SOFTPLUS_SHIFT 5
#define PERTURBATION 1e-4
#define TRAINING_EPOCHS 50000
#define REGULARIZATION_TERM 0.01
#define LEARNING_RATE 1e-3

void plot_it(const char* filename, GNeuralNet& nn, GMatrix& trainFeat, GMatrix& trainLab, GMatrix& testFeat, GMatrix& testLab)
{
	GSVG svg(1000, 500);
	double xmin = trainFeat[0][0];
	double xmax = testFeat[testFeat.rows() - 1][0];
	svg.newChart(xmin, std::min(trainLab.columnMin(0), testLab.columnMin(0)), xmax, std::max(trainLab.columnMax(0), testLab.columnMax(0)));
	svg.horizMarks(20);
	svg.vertMarks(20);
	double prevx = xmin;
	double prevy = 0.0;
	double step = (xmax - xmin) / 500.0;
	GVec x(1);
	GVec y(1);
	for(x[0] = prevx; x[0] < xmax; x[0] += step)
	{
		nn.predict(x, y);
		if(prevx != x[0])
			svg.line(prevx, prevy, x[0], y[0], 0.3);
		prevx = x[0];
		prevy = y[0];
	}
	for(size_t i = 0; i < trainLab.rows(); i++)
		svg.dot(trainFeat[i][0], trainLab[i][0], 0.4, 0xff000080);
	for(size_t i = 0; i < testLab.rows(); i++)
		svg.dot(testFeat[i][0], testLab[i][0], 0.4, 0xff800000);

	std::ofstream ofs;
	ofs.open(filename);
	svg.print(ofs);
}


void doit()
{
	// Load the data
	GMatrix trainLab;
	GMatrix testLab;
	if (chdir("../bin") != 0)
	{
	}
	trainLab.loadArff("train.arff");
	testLab.loadArff("test.arff");
	double dataMin = trainLab.columnMin(0);
	double dataMax = trainLab.columnMax(0);
	trainLab.normalizeColumn(0, dataMin, dataMax, -5.0, 5.0);
	testLab.normalizeColumn(0, dataMin, dataMax, -5.0, 5.0);
	GMatrix trainFeat(trainLab.rows(), 1);
	for(size_t i = 0; i < trainLab.rows(); i++)
		trainFeat[i][0] = (double)i / trainLab.rows() - 0.5;
	GMatrix testFeat(testLab.rows(), 1);
	for(size_t i = 0; i < testLab.rows(); i++)
		testFeat[i][0] = (double)(i + trainLab.rows()) / trainLab.rows() - 0.5;

	// Make a neural network
	GNeuralNet nn;
	GUniformRelation relOne(1);
	nn.beginIncrementalLearning(relOne, relOne);

	// Initialize the weights of the sine units to match the frequencies used by the Fourier transform.
	GLayerClassic* pSine2 = new GLayerClassic(1, 64, new GActivationSin());
	GMatrix& wSin = pSine2->weights();
	GVec& bSin = pSine2->bias();
	for(size_t i = 0; i < pSine2->outputs() / 2; i++)
	{
		wSin[0][2 * i] = 2.0 * M_PI * (i + 1);
		bSin[2 * i] = 0.5 * M_PI;
		wSin[0][2 * i + 1] = 2.0 * M_PI * (i + 1);
		bSin[2 * i + 1] = M_PI;
	}

	// Make the hidden layer
	GLayerMixed* pMix2 = new GLayerMixed();
	pSine2->resize(1, pSine2->outputs(), &nn.rand(), PERTURBATION);
	pMix2->addComponent(pSine2);
	GLayerClassic* pSoftPlus2 = new GLayerClassic(1, SOFTPLUS_NODES, new GActivationSoftPlus());
	pMix2->addComponent(pSoftPlus2);
	GLayerClassic* pIdentity2 = new GLayerClassic(1, IDENTITY_NODES, new GActivationIdentity());
	pMix2->addComponent(pIdentity2);
	nn.addLayer(pMix2);

	// Make the output layer
	GLayerClassic* pIdentity3 = new GLayerClassic(FLEXIBLE_SIZE, trainLab.cols(), new GActivationIdentity());
	pIdentity3->resize(pMix2->outputs(), pIdentity3->outputs(), &nn.rand(), PERTURBATION);
	nn.addLayer(pIdentity3);

	// Initialize all the non-periodic nodes to approximate the identity function, then perturb a little bit
	pSoftPlus2->setWeightsToIdentity();
	for(size_t i = 0; i < SOFTPLUS_NODES; i++)
	{
		pSoftPlus2->bias()[i] += SOFTPLUS_SHIFT;
		pIdentity3->renormalizeInput(pSine2->outputs() + i, 0.0, 1.0, SOFTPLUS_SHIFT, SOFTPLUS_SHIFT + 1.0);
	}
	pIdentity2->setWeightsToIdentity();
	pSoftPlus2->perturbWeights(nn.rand(), PERTURBATION);
	pIdentity2->perturbWeights(nn.rand(), PERTURBATION);

	// Randomly initialize the weights on the output layer
	pIdentity3->weights().setAll(0.0);
	pIdentity3->perturbWeights(nn.rand(), PERTURBATION);

	// Open Firefox to view the progress
	GApp::systemCall("firefox ./view.html#progress.svg", false, true);

	// Do some training
	GRandomIndexIterator ii(trainLab.rows(), nn.rand());
	nn.setLearningRate(LEARNING_RATE);
	for(size_t epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
	{
		// Visit each sample in random order
		ii.reset();
		size_t i;
		while(ii.next(i))
		{
			// Regularize
			pIdentity3->scaleWeights(1.0 - nn.learningRate() * REGULARIZATION_TERM, true);
			pIdentity3->diminishWeights(nn.learningRate() * REGULARIZATION_TERM, true);

			// Train
			nn.trainIncremental(trainFeat[i], trainLab[i]); // One iteration of stochastic gradient descent
		}

		// Report progress
		double rmse = sqrt(nn.sumSquaredError(trainFeat, trainLab) / trainLab.rows());
		if(epoch % (TRAINING_EPOCHS / 100) == 0)
		{
			double val = sqrt(nn.sumSquaredError(testFeat, testLab) / testLab.rows());
			cout << "prog=" << to_str((double)epoch * 100.0 / TRAINING_EPOCHS) << "%	rmse=" << to_str(rmse) << "	val=" << to_str(val) << "\n";
			plot_it("progress.svg", nn, trainFeat, trainLab, testFeat, testLab);
		}
	}
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int nRet = 0;
	try
	{
		doit();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}
