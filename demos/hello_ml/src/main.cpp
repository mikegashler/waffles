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

void do_decision_tree(GMatrix& features, GMatrix& labels, double* test_features, double* predicted_labels)
{
	GDecisionTree model;
	model.train(features, labels);
	model.predict(test_features, predicted_labels);
}

void do_neural_network(GMatrix& features, GMatrix& labels, double* test_features, double* predicted_labels)
{
	GNeuralNet* pNN = new GNeuralNet();
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 3));
	pNN->addLayer(new GLayerClassic(3, FLEXIBLE_SIZE));
	pNN->setLearningRate(0.1);
	pNN->setMomentum(0.1);
	GAutoFilter af(pNN);
	af.train(features, labels);
	af.predict(test_features, predicted_labels);
}

void do_knn(GMatrix& features, GMatrix& labels, double* test_features, double* predicted_labels)
{
	GKNN model;
	model.setNeighborCount(3); // use the 3-nearest neighbors
	model.setInterpolationMethod(GKNN::Linear); // use linear interpolation
	model.train(features, labels);
	model.predict(test_features, predicted_labels);
}

void do_naivebayes(GMatrix& features, GMatrix& labels, double* test_features, double* predicted_labels)
{
	GAutoFilter model(new GNaiveBayes());
	model.train(features, labels);
	model.predict(test_features, predicted_labels);
}

void do_ensemble(GMatrix& features, GMatrix& labels, double* test_features, double* predicted_labels)
{
	GBag ensemble;
	for(size_t i = 0; i < 50; i++)
	{
		GDecisionTree* pDT = new GDecisionTree();
		pDT->useRandomDivisions(1); // Make random tree
		ensemble.addLearner(pDT);
	}
	ensemble.train(features, labels);
	ensemble.predict(test_features, predicted_labels);
}

void doit()
{
	// Define the feature attributes (or columns)
	vector<size_t> feature_values;
	feature_values.push_back(0); // diameter = continuous
	feature_values.push_back(3); // crust_type = { thin_crust=0, Chicago_style_deep_dish=1, Neapolitan=2 }
	feature_values.push_back(2); // meatiness = { vegan=0, meaty=1 }
	feature_values.push_back(4); // presentation = { dine_in=0, take_out=1, delivery=2, frozen=3 }

	// Define the label attributes (or columns)
	vector<size_t> label_values;
	label_values.push_back(2); // taste = { lousy=0, delicious=1 }
	label_values.push_back(0); // cost = continuous

	// Make some contrived hard-coded training data
	GMatrix features(feature_values);
	GMatrix labels(label_values);
	double* f;
	double* l;
	//                     diameter     crust     meatiness presentation                   taste     cost
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 0; l = labels.newRow(); l[0] = 1; l[1] = 22.95;
	f = features.newRow(); f[0] = 12.0; f[1] = 0; f[2] = 0; f[3] = 3; l = labels.newRow(); l[0] = 0; l[1] = 3.29;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 2; l = labels.newRow(); l[0] = 1; l[1] = 15.49;
	f = features.newRow(); f[0] = 12.0; f[1] = 2; f[2] = 0; f[3] = 0; l = labels.newRow(); l[0] = 1; l[1] = 16.65;
	f = features.newRow(); f[0] = 18.0; f[1] = 1; f[2] = 1; f[3] = 3; l = labels.newRow(); l[0] = 0; l[1] = 9.99;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 0; l = labels.newRow(); l[0] = 1; l[1] = 14.49;
	f = features.newRow(); f[0] = 12.0; f[1] = 2; f[2] = 0; f[3] = 2; l = labels.newRow(); l[0] = 1; l[1] = 19.65;
	f = features.newRow(); f[0] = 14.0; f[1] = 0; f[2] = 1; f[3] = 1; l = labels.newRow(); l[0] = 0; l[1] = 6.99;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 2; l = labels.newRow(); l[0] = 1; l[1] = 19.95;
	f = features.newRow(); f[0] = 14.0; f[1] = 2; f[2] = 0; f[3] = 3; l = labels.newRow(); l[0] = 0; l[1] = 12.99;
	f = features.newRow(); f[0] = 16.0; f[1] = 0; f[2] = 1; f[3] = 0; l = labels.newRow(); l[0] = 0; l[1] = 12.20;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 1; l = labels.newRow(); l[0] = 1; l[1] = 15.01;

	// Make a test vector
	double test_features[4];
	double predicted_labels[2];
	cout << "This demo trains and tests several supervised learning models using some contrived hard-coded training data to predict the tastiness and cost of a pizza.\n\n";
	test_features[0] = 15.0; test_features[1] = 2; test_features[2] = 0; test_features[3] = 0;
	cout << "Predicting labels for a 15 inch pizza with a Neapolitan-style crust, no meat, for dine-in.\n\n";

	// Use several models to make predictions
	cout.precision(4);
	do_decision_tree(features, labels, test_features, predicted_labels);
	cout << "The decision tree predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << ".\n";

	do_neural_network(features, labels, test_features, predicted_labels);
	cout << "The neural network predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << ".\n";

	do_knn(features, labels, test_features, predicted_labels);
	cout << "The knn model predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << ".\n";

	do_naivebayes(features, labels, test_features, predicted_labels);
	cout << "The naive Bayes model predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << ".\n";

	do_ensemble(features, labels, test_features, predicted_labels);
	cout << "Random forest predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << ".\n";
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

