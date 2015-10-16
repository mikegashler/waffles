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
#include <GClasses/GHolders.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GActivation.h>
#include <GClasses/GPlot.h>
#include <GClasses/GBits.h>
#include <GClasses/GVec.h>
#include <GClasses/GTime.h>
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

#define TRAIN_SIZE 64 // must be a power of 2
#define TEST_SIZE 64 // does not need not be a power of 2. I just like symmetry.
#define SOFTPLUS_NODES 2
#define IDENTITY_NODES 2
#define SOFTPLUS_SHIFT 10
#define PERTURBATION 1e-4
#define TIGHTNESS_GOOD 0.05
#define TIGHTNESS_BAD 0.25


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
	cout << "\n\nThis demo is described at http://arxiv.org/abs/1405.2262\n\n";
	
	cout << "For efficiency reasons, this demo differs from the paper the following ways:\n";
	cout << " * It uses one fewer layers.\n";
	cout << " * It only uses L1 regularization during the last 20% of training.\n\n";

	// Load the data
	GMatrix trainLab;
	GMatrix testLab;
	if (chdir("../bin") != 0)
	{
	}
	trainLab.loadArff("train.arff");
	testLab.loadArff("test.arff");
	GMatrix trainFeat(trainLab.rows(), 1);
	for(size_t i = 0; i < trainLab.rows(); i++)
		trainFeat[i][0] = (double)i / trainLab.rows();
	GMatrix testFeat(testLab.rows(), 1);
	for(size_t i = 0; i < testLab.rows(); i++)
		testFeat[i][0] = (double)(i + trainLab.rows()) / trainLab.rows();

	// Use the Fourier transform to initialize a neural network
	GNeuralNet* pNN = GNeuralNet::fourier(trainLab);
	Holder<GNeuralNet> hNN(pNN);
	plot_it("fourier.svg", *pNN, trainFeat, trainLab, testFeat, testLab);

	// Test that it fits to the training data
	double fourier_rmse = sqrt(pNN->sumSquaredError(trainFeat, trainLab) / trainLab.rows());
	if(fourier_rmse >= 1e-9)
		throw Ex("Fourier transform failed to fit the training data. (This should never happen.)");

	// Release those layers, so we can put them in a new neural network with some additional layers
	GNeuralNetLayer* pIdentity3 = pNN->releaseLayer(1);
	GNeuralNetLayer* pSine2 = pNN->releaseLayer(0);
	GNeuralNet nn;
	GUniformRelation relOne(1);
	nn.beginIncrementalLearning(relOne, relOne);

	// Make layer 1
	GLayerMixed* pMix1 = new GLayerMixed();
	GLayerClassic* pSoftPlus1 = new GLayerClassic(1, SOFTPLUS_NODES, new GActivationSoftPlus());
	pMix1->addComponent(pSoftPlus1);
	GLayerClassic* pIdentity1 = new GLayerClassic(1, IDENTITY_NODES, new GActivationIdentity());
	pMix1->addComponent(pIdentity1);
	nn.addLayer(pMix1);

	// Make layer 2
	GLayerMixed* pMix2 = new GLayerMixed();
	pSine2->resize(pMix1->outputs(), pSine2->outputs(), &nn.rand(), PERTURBATION);
	pMix2->addComponent(pSine2);
	GLayerClassic* pSoftPlus2 = new GLayerClassic(pMix1->outputs(), SOFTPLUS_NODES, new GActivationSoftPlus());
	pMix2->addComponent(pSoftPlus2);
	GLayerClassic* pIdentity2 = new GLayerClassic(pMix1->outputs(), IDENTITY_NODES, new GActivationIdentity());
	pMix2->addComponent(pIdentity2);
	nn.addLayer(pMix2);

	// Make layer 3
	pIdentity3->resize(pMix2->outputs(), pIdentity3->outputs(), &nn.rand(), PERTURBATION);
	nn.addLayer(pIdentity3);

	// Initialize all the softplus nodes to approximate the identity function
	pSoftPlus1->setWeightsToIdentity();
	for(size_t i = 0; i < SOFTPLUS_NODES; i++)
	{
		pSoftPlus1->bias()[i] += SOFTPLUS_SHIFT;
		pMix2->renormalizeInput(i, 0.0, 1.0, SOFTPLUS_SHIFT, SOFTPLUS_SHIFT + 1.0);
	}
	pSoftPlus2->setWeightsToIdentity();
	for(size_t i = 0; i < SOFTPLUS_NODES; i++)
	{
		pSoftPlus2->bias()[i] += SOFTPLUS_SHIFT;
		pIdentity3->renormalizeInput(pSine2->outputs() + i, 0.0, 1.0, SOFTPLUS_SHIFT, SOFTPLUS_SHIFT + 1.0);
	}

	// Initialize all the identity nodes to approximate the identity function
	pIdentity1->setWeightsToIdentity();
	pIdentity2->setWeightsToIdentity();

	// Perturb the weights a little bit
	pSoftPlus1->perturbWeights(nn.rand(), PERTURBATION);
	pSoftPlus2->perturbWeights(nn.rand(), PERTURBATION);
	pIdentity1->perturbWeights(nn.rand(), PERTURBATION);
	pIdentity2->perturbWeights(nn.rand(), PERTURBATION);

	// Test that we still fit to the training data well enough
	double labMean = trainLab.columnMean(0);
	double labDev = sqrt(trainLab.columnVariance(0, labMean));
	cout << "dev=" << to_str(labDev) << "\n";
	double initial_rmse = sqrt(nn.sumSquaredError(trainFeat, trainLab) / trainLab.rows());
	cout << "initial rmse/dev=" << to_str(initial_rmse / labDev) << "\n";
	if(initial_rmse >= TIGHTNESS_GOOD * labDev)
		throw Ex("Already above threshold on initialization. This probably means PERTURBATION is too high or SOFTPLUS_SHIFT is too low.");

	// Open Firefox to view the progress
	GApp::systemCall("firefox view.html#progress.svg", false, true);

	// Do some training
	GNeuralNet backup;
	backup.copyStructure(&nn);
	GNeuralNet nn2;
	nn2.copyStructure(&nn);
	GRandomIndexIterator ii(trainLab.rows(), nn.rand());
	double learningRate = 1e-8;
	double lambda = 0.001;
	for(size_t epoch = 0; true; epoch++)
	{
		// Regularize until it reaches TIGHTNESS_BAD
		double rmse;
		size_t regularize_count = 0;
		while(true)
		{
			// L2 regularization
			pSoftPlus1->scaleWeights(1.0 - 0.1 * lambda, true);
			pIdentity1->scaleWeights(1.0 - 0.01 * lambda, true);
			pSine2->scaleWeights(1.0 - lambda, true);
			pSoftPlus2->scaleWeights(1.0 - 0.1 * lambda, true);
			pIdentity2->scaleWeights(1.0 - 0.01 * lambda, true);
			pIdentity3->scaleWeights(1.0 - lambda, true);

			// L1 regularization
			pSoftPlus1->diminishWeights(0.1 * lambda, true);
			pIdentity1->diminishWeights(0.01 * lambda, true);
			pSine2->diminishWeights(lambda, true);
			pSoftPlus2->diminishWeights(0.1 * lambda, true);
			pIdentity2->diminishWeights(0.01 * lambda, true);
			pIdentity3->diminishWeights(lambda, true);

			// Test whether we are there yet
			regularize_count++;
			rmse = sqrt(nn.sumSquaredError(trainFeat, trainLab) / trainLab.rows());
			if(rmse / labDev >= TIGHTNESS_BAD)
				break;
		}
		if(regularize_count < 5)
			lambda *= 0.1;
		else
			lambda *= 1.2;
		cout << "rmse/dev=" << to_str(rmse / labDev) << "	lambda=" << to_str(lambda) << "	regularization iters=" << to_str(regularize_count) << "\n";

		// Train until it reaches TIGHTNESS_GOOD
		double timeStart = 0.0;
		while(true)
		{
			nn.setLearningRate(learningRate * 1.25);
			nn2.setLearningRate(learningRate * 0.8);
			backup.copyWeights(&nn);
			nn2.copyWeights(&nn);

			// Visit each sample in random order
			ii.reset();
			size_t i;
			while(ii.next(i))
			{
				nn.trainIncremental(trainFeat[i], trainLab[i]); // One iteration of stochastic gradient descent
				nn2.trainIncremental(trainFeat[i], trainLab[i]); // One iteration of stochastic gradient descent
			}

			// Adjust the learning rate
			double rmse1 = sqrt(nn.sumSquaredError(trainFeat, trainLab) / trainLab.rows());
			double rmse2 = sqrt(nn2.sumSquaredError(trainFeat, trainLab) / trainLab.rows());
			if(rmse1 <= rmse2 && rmse1 <= rmse)
			{
				rmse = rmse1;
				learningRate = nn.learningRate();
			}
			else if(rmse2 <= rmse1 && rmse2 <= rmse)
			{
				rmse = rmse2;
				learningRate = nn2.learningRate();
				nn.copyWeights(&nn2);
			}
			else
			{
				learningRate *= 0.1;
				nn.copyWeights(&backup);
			}

			// Report progress
			double timeNow = GTime::seconds();
			if(timeNow - timeStart >= 0.5)
			{
				timeStart = timeNow;
				double val = sqrt(nn.sumSquaredError(testFeat, testLab) / testLab.rows());
				cout << "rmse/dev=" << to_str(rmse / labDev) << "	eta=" << to_str(learningRate)  << "	val=" << to_str(val) << "\n";
				plot_it("progress.svg", nn, trainFeat, trainLab, testFeat, testLab);
			}

			if(rmse / labDev <= TIGHTNESS_GOOD)
				break;
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
