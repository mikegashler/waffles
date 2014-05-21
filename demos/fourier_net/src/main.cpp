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
#include <GClasses/GHillClimber.h>
#include <math.h>
#include <fstream>

using namespace GClasses;
using std::cerr;
using std::cout;
using std::string;

#define TRAIN_SIZE 64 // must be a power of 2
#define TEST_SIZE 64 // does not need not be a power of 2. I just like symmetry.
#define SOFTPLUS_NODES 12
#define IDENTITY_NODES 12
#define SOFTPLUS_SHIFT 10
#define PERTURBATION 1e-4
#define TRAINING_EPOCHS 10000000 // yeah, this will take several hours
#define TIGHTNESS 0.05


void plot_it(const char* filename, GNeuralNet& nn, GMatrix& trainFeat, GMatrix& trainLab, GMatrix& testFeat, GMatrix& testLab)
{
	GSVG svg(1000, 500);
	svg.newChart(0.0, -1.0, 2.0, 5.0);
	svg.horizMarks(20);
	svg.vertMarks(20);
	double prevx = 0.0;
	double prevy = 0.0;
	for(double x = prevx; x < 2.0; x += 0.01)
	{
		double y;
		nn.predict(&x, &y);
		if(prevx != x)
			svg.line(prevx, prevy, x, y, 0.3);
		prevx = x;
		prevy = y;
	}
	for(size_t i = 0; i < trainLab.rows(); i++)
		svg.dot(trainFeat[i][0], trainLab[i][0], 0.4, 0xff000080);
	for(size_t i = 0; i < TEST_SIZE; i++)
		svg.dot(testFeat[i][0], testLab[i][0], 0.4, 0xff800000);

	std::ofstream ofs;
	ofs.open(filename);
	svg.print(ofs);
}





void doit(GArgReader& args)
{
	// Make some data
	GMatrix trainFeat(TRAIN_SIZE, 1);
	GMatrix trainLab(TRAIN_SIZE, 1);
	GMatrix testFeat(TEST_SIZE, 1);
	GMatrix testLab(TEST_SIZE, 1);
	for(size_t i = 0; i < TRAIN_SIZE; i++)
	{
		double x = i;
		trainFeat[i][0] = x / TRAIN_SIZE;
		trainLab[i][0] = sin(x / 3.0) + x * 0.03;
	}
	for(size_t i = 0; i < TEST_SIZE; i++)
	{
		double x = i + TRAIN_SIZE;
		testFeat[i][0] = x / TRAIN_SIZE;
		testLab[i][0] = sin(x / 3.0) + x * 0.03;
	}

	// Use the Fourier transform to initialize a neural  network
	GNeuralNet* pNN = GNeuralNet::fourier(trainLab);
	Holder<GNeuralNet> hNN(pNN);
	plot_it("fourier.svg", *pNN, trainFeat, trainLab, testFeat, testLab);

	// Ensure that we fit to the training data
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

	// Initialize the softplus and identity nodes to approximate the identity function
	pSoftPlus1->setWeightsToIdentity();
	pSoftPlus1->perturbWeights(nn.rand(), PERTURBATION);
	for(size_t i = 0; i < SOFTPLUS_NODES; i++)
	{
		pSoftPlus1->bias()[i] += SOFTPLUS_SHIFT;
		pMix2->renormalizeInput(i, 0.0, 1.0, SOFTPLUS_SHIFT, SOFTPLUS_SHIFT + 1.0);
	}
	pIdentity1->setWeightsToIdentity();
	pIdentity1->perturbWeights(nn.rand(), PERTURBATION);
	pSoftPlus2->setWeightsToIdentity();
	pSoftPlus2->perturbWeights(nn.rand(), PERTURBATION);
	for(size_t i = 0; i < SOFTPLUS_NODES; i++)
	{
		pSoftPlus2->bias()[i] += SOFTPLUS_SHIFT;
		pIdentity3->renormalizeInput(pSine2->outputs() + i, 0.0, 1.0, SOFTPLUS_SHIFT, SOFTPLUS_SHIFT + 1.0);
	}
	pIdentity2->setWeightsToIdentity();
	pIdentity2->perturbWeights(nn.rand(), PERTURBATION);

	// Ensure that we still fit to the training data well enough
	plot_it("before.svg", nn, trainFeat, trainLab, testFeat, testLab);
	double labMean = trainLab.columnMean(0);
	double labDev = sqrt(trainLab.columnVariance(0, labMean));
	cout << "dev=" << to_str(labDev) << "\n";
	double rmse = sqrt(nn.sumSquaredError(trainFeat, trainLab) / trainLab.rows());
	cout << "initial rmse/dev=" << to_str(rmse / labDev) << "\n";
	if(rmse >= TIGHTNESS * labDev)
		throw Ex("Already above threshold on initialization. This probably means PERTURBATION is too high or SOFTPLUS_SHIFT is too low.");

	// Do some training
	GNeuralNet backup;
	backup.copyStructure(&nn);
	GRandomIndexIterator ii(trainLab.rows(), nn.rand());
	double learningRate = 1e-8;
	double lambda = 1.0;
	for(size_t epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
	{
		nn.setLearningRate(learningRate);

		ii.reset();
		size_t i;
		while(ii.next(i))
		{
			if(epoch < TRAINING_EPOCHS / 2)
			{
				pSoftPlus1->scaleWeights(1.0 - 0.1 * learningRate * lambda, true);
				pIdentity1->scaleWeights(1.0 - 0.01 * learningRate * lambda, true);
				pSine2->scaleWeights(1.0 - learningRate * lambda, true);
				pSoftPlus2->scaleWeights(1.0 - 0.1 * learningRate * lambda, true);
				pIdentity2->scaleWeights(1.0 - 0.01 * learningRate * lambda, true);
				pIdentity3->scaleWeights(1.0 - 0.01 * learningRate * lambda, true);
			}
			else
			{
				pSoftPlus1->diminishWeights(0.1 * learningRate * lambda, true);
				pIdentity1->diminishWeights(0.01 * learningRate * lambda, true);
				pSine2->diminishWeights(learningRate * lambda, true);
				pSoftPlus2->diminishWeights(0.1 * learningRate * lambda, true);
				pIdentity2->diminishWeights(0.01 * learningRate * lambda, true);
				pIdentity3->diminishWeights(0.01 * learningRate * lambda, true);
			}

			nn.trainIncremental(trainFeat[i], trainLab[i]);
		}

		// Report some progress
		double rmse = sqrt(nn.sumSquaredError(trainFeat, trainLab) / trainLab.rows());
		if(epoch % (TRAINING_EPOCHS / 1000) == 0)
		{
			double val = sqrt(nn.sumSquaredError(testFeat, testLab) / testLab.rows());
			cout << "prog=" << to_str((double)epoch * 100.0 / TRAINING_EPOCHS) << "%	rmse/dev=" << to_str(rmse / labDev) << "	val=" << to_str(val) << "	eta=" << to_str(learningRate) << "	lambda=" << to_str(lambda) << "\n";
			if(epoch > 0 && epoch % 1000000 == 0)
			{
				string s = "progress";
				s += epoch / 1000000;
				s += ".svg";
				plot_it(s.c_str(), nn, trainFeat, trainLab, testFeat, testLab);
			}
		}

		// Dynamically adjust the learning rate and regularization term
		learningRate *= 1.01;
		if(rmse < TIGHTNESS * labDev)
			lambda *= 1.001;
		else
		{
			if(rmse < 2 * TIGHTNESS * labDev)
			{
				if(rmse < 1.5 * TIGHTNESS * labDev)
					backup.copyWeights(&nn); // make a backup copy of the weights
				lambda /= 1.001;
			}
			else
			{
				nn.copyWeights(&backup); // restore the weights from backup
				learningRate *= 0.1;
				if(learningRate < 1e-100)
					throw Ex("Repeatedly failing to recover");
			}
		}
	}

	// Plot the final model
	plot_it("after.svg", nn, trainFeat, trainLab, testFeat, testLab);
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int nRet = 0;
	try
	{
		GArgReader args(argc, argv);
		doit(args);
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

