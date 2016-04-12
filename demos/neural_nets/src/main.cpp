/*
  The contents of this file are dedicated by all of its authors, including

    Stephen C. Ashmore,
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
#include <vector>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GDecisionTree.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GActivation.h>
#include <GClasses/GKNN.h>
#include <GClasses/GNaiveBayes.h>
#include <GClasses/GEnsemble.h>

using namespace GClasses;
using std::cerr;
using std::cout;
using std::vector;
using std::endl;

void do_fullyconnected(GMatrix& features, GMatrix& labels, GVec& test_features, GVec& predicted_labels)
{
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 100));
	pNN->addLayer(new GLayerClassic(100, 40));
	pNN->addLayer(new GLayerClassic(40, FLEXIBLE_SIZE));
	pNN->setLearningRate(0.01);
	pNN->setMomentum(0.1);
	GAutoFilter af(pNN);
	af.train(features, labels);
	af.predict(test_features, predicted_labels);
}

void do_convolutional(GMatrix& features, GMatrix& labels, GVec& test_features, GVec& predicted_labels)
{
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerConvolutional2D(28, 28, 1, 8, 4, new GActivationBentIdentity()));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 80, new GActivationRectifiedLinear()));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 40, new GActivationBentIdentity()));
	pNN->setLearningRate(0.01);
	pNN->setMomentum(0.1);
	GAutoFilter af(pNN);
	af.train(features, labels);
	af.predict(test_features, predicted_labels);
}

void do_autoencoder(GMatrix& features, GMatrix& labels, GVec& test_features, size_t& countMissedPixels)
{
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 100));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 40));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 20));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 40));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 100));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));

	pNN->setLearningRate(0.01);
	pNN->setMomentum(0.1);
	GAutoFilter af(pNN);
	GRandomIndexIterator iter(features.rows(), pNN->rand()); size_t index;
	pNN->beginIncrementalLearning(features.relation(), features.relation());
	for (size_t epoch = 0; epoch < 600; epoch++ )
	{
		iter.reset();
		while( iter.next(index) )
		{
			pNN->trainIncremental(features[index], features[index]);
		}
	}
	
	GVec predicted_feature(test_features.size());
	pNN->predict(test_features, predicted_feature);
	
	countMissedPixels = 0;
	for ( size_t i = 0; i < test_features.size(); i++ )
	{
		if ( !((test_features[i] - predicted_feature[i]) < 20 && (test_features[i] - predicted_feature[i]) > -20 ))
			countMissedPixels++;
	}
}

void do_bendactivation(GMatrix& features, GMatrix& labels, GVec& test_features, GVec& predicted_labels)
{
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 100, new GActivationBentIdentity()));
	pNN->addLayer(new GLayerClassic(100, 40, new GActivationBentIdentity()));
	pNN->addLayer(new GLayerClassic(40, FLEXIBLE_SIZE, new GActivationBentIdentity()));
	pNN->setLearningRate(0.01);
	pNN->setMomentum(0.1);
	GAutoFilter af(pNN);
	af.train(features, labels);
	af.predict(test_features, predicted_labels);
}

void do_rprop(GMatrix& features, GMatrix& labels, GVec& test_features, GVec& predicted_labels)
{
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 100));
	pNN->addLayer(new GLayerClassic(100, 40));
	pNN->addLayer(new GLayerClassic(40, FLEXIBLE_SIZE));
	pNN->setLearningRate(0.01);
	GAutoFilter af(pNN);
	
	pNN->beginIncrementalLearning(features.relation(), labels.relation());
	for (size_t epoch = 0; epoch < 100; epoch++ )
	{
		pNN->trainIncrementalAdaptive(features, labels);
	}
	
	pNN->predict(test_features, predicted_labels);
}

void do_dropout(GMatrix& features, GMatrix& labels, GVec& test_features, GVec& predicted_labels)
{
	double dropOutRate = 0.2;
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 100));
	pNN->addLayer(new GLayerClassic(100, 40));
	pNN->addLayer(new GLayerClassic(40, FLEXIBLE_SIZE));
	pNN->setLearningRate(0.01);
	GAutoFilter af(pNN);
	
	GRandomIndexIterator iter(features.rows(), pNN->rand()); size_t index;
	pNN->beginIncrementalLearning(features.relation(), labels.relation());
	for (size_t epoch = 0; epoch < 100; epoch++ )
	{
		iter.reset();
		while( iter.next(index) )
		{
			pNN->trainIncrementalWithDropout(features[index], labels[index], dropOutRate);
			pNN->scaleWeights(1.0 - dropOutRate, false, 1);
		}
	}
	
	pNN->predict(test_features, predicted_labels);
}

void doit()
{
	GMatrix train;
	train.loadArff("../mnist_train.arff");
	GDataColSplitter sTrain(train, 1);
	
	GMatrix& features = sTrain.features();
	GMatrix& labels = sTrain.labels();
	
	GMatrix test;
	test.loadArff("../mnist_test.arff");
	
	// Make a test vector
	GVec test_features(784), predicted_feature(784);
	GVec predicted_labels(1), correctLabel(1);
	test_features.put(0, test[0], 0, test.cols() - 1);
	correctLabel.put(0, test[0], test.cols() - 1, 1);
	
	cout << "This demo trains and tests several different neural network architectures using an extremely small subset of the MNIST hand-written digits dataset. Each feature example is a grey-scale picture of a hand-written digit, like the number 4 etc." << endl;
	
	// Use several models to make predictions
	do_fullyconnected(features, labels, test_features, predicted_labels);
	cout << "A typical neural net consists of layers of fully-connected neurons. This example has "
		<< "a two layer network." << endl;
	cout << "It predicted a: " << predicted_labels[0] << " where " << correctLabel[0] << " was correct." << endl;
	
	do_bendactivation(features, labels, test_features, predicted_labels);
	cout << "You can also replace what activation function is used on the neurons. This allows greater flexibility "
		<< "to fit some data, and can improve accuracy and performance in some cases. Waffles supports a large "
		<< "set of various activation functions. Here we try the Bent Identity function on each of our layers. "
		<< "Note: Waffles supports mixed layers of various activation functions." << endl;
	cout << "It predicted a: " << predicted_labels[0] << " where " << correctLabel[0] << " was correct." << endl;
	
	do_convolutional(features, labels, test_features, predicted_labels);
	cout << "Waffles also supports convolution, which excels at visual tasks. This example is a very "
		<< "simple convolutional network with one layer of convolutional neurons." << endl;
	cout << "It predicted a: " << predicted_labels[0] << " where " << correctLabel[0] << " was correct." << endl;

	// R-Prop
// 	do_rprop(features, labels, test_features, predicted_labels);
// 	cout << "You can also use other training methods instead of classic gradient descent, for example R-Prop." << endl;
// 	cout << "R-Prop predicted a: " << predicted_labels[0] << " where " << correctLabel[0] << " was correct." << endl;
	
	// train with dropout
// 	do_dropout(features,labels, test_features, predicted_labels);
// 	cout << "Another training method is to train with drop out, which removes neurons during training. This should help the model to not over-fit the data." << endl;
// 	cout << "It predicted a: " << predicted_labels[0] << " where " << correctLabel[0] << " was correct." << endl;	
	
	size_t countMissedPixels = 0;
	do_autoencoder(features, labels, test_features, countMissedPixels);
	cout << "Autoencoders are a bit different, rather than try to predict the label autoencoders "
		<< "typically try to reproduce the features by first reducing the feature space to a smaller dimension."
		<< endl;
	cout << "Having only seen 35 example images of a handwritten digit, this simple autoencoder only missed: " << endl;
	cout << countMissedPixels << " out of " << test_features.size() << " possible pixels." << endl;
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
