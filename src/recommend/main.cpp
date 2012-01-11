// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "../GClasses/GActivation.h"
#include "../GClasses/GApp.h"
#include "../GClasses/GCluster.h"
#include "../GClasses/GMatrix.h"
#include "../GClasses/GDistance.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GError.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GRecommender.h"
#include "../GClasses/GSparseMatrix.h"
#include "../GClasses/GDom.h"
#include "../wizard/usage.h"
#include <time.h>
#include <iostream>
#ifdef WIN32
#	include <direct.h>
#	include <process.h>
#endif
#include <exception>
#include <string>
#include <vector>
#include <set>

using namespace GClasses;
using std::cout;
using std::cerr;
using std::string;
using std::vector;
using std::set;

GCollaborativeFilter* InstantiateAlgorithm(GRand& rand, GArgReader& args);

GMatrix* loadData(const char* szFilename)
{
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	if(_stricmp(szFilename + pd.extStart, ".sparse") == 0)
	{
		GDom doc;
		doc.loadJson(szFilename);
		GSparseMatrix sm(doc.root());
		GMatrix* pData = new GMatrix(0, 3);
		for(size_t i = 0; i < sm.rows(); i++)
		{
			GSparseMatrix::Iter rowEnd = sm.rowEnd(i);
			for(GSparseMatrix::Iter it = sm.rowBegin(i); it != rowEnd; it++)
			{
				double* pVec = pData->newRow();
				pVec[0] = i;
				pVec[1] = it->first;
				pVec[2] = it->second;
			}
		}
		return pData;
	}
	else if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
		return GMatrix::loadArff(szFilename);
	else
	{
		ThrowError("Unsupported file format: ", szFilename + pd.extStart);
		return NULL;
	}
}

GSparseMatrix* loadSparseData(const char* szFilename)
{
	// Load the dataset by extension
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
	{
		// Convert a 3-column dense ARFF file to a sparse matrix
		GMatrix* pData = GMatrix::loadArff(szFilename);
		if(pData->cols() != 3)
			ThrowError("Expected 3 columns: 0) user or row-index, 1) item or col-index, 2) value or rating");
		double m0, r0, m1, r1;
		pData->minAndRange(0, &m0, &r0);
		pData->minAndRange(1, &m1, &r1);
		if(m0 < 0 || m0 > 1e10 || r0 < 2 || r0 > 1e10)
			ThrowError("Invalid row indexes");
		if(m1 < 0 || m1 > 1e10 || r1 < 2 || r1 > 1e10)
			ThrowError("Invalid col indexes");
		GSparseMatrix* pMatrix = new GSparseMatrix(size_t(m0 + r0) + 1, size_t(m1 + r1) + 1, UNKNOWN_REAL_VALUE);
		Holder<GSparseMatrix> hMatrix(pMatrix);
		for(size_t i = 0; i < pData->rows(); i++)
		{
			double* pRow = pData->row(i);
			pMatrix->set(size_t(pRow[0]), size_t(pRow[1]), pRow[2]);
		}
		return hMatrix.release();
	}
	else if(_stricmp(szFilename + pd.extStart, ".sparse") == 0)
	{
		GDom doc;
		doc.loadJson(szFilename);
		return new GSparseMatrix(doc.root());
	}
	ThrowError("Unsupported file format: ", szFilename + pd.extStart);
	return NULL;
}

GBaselineRecommender* InstantiateBaselineRecommender(GRand& rand, GArgReader& args)
{
	return new GBaselineRecommender(rand);
}

GBagOfRecommenders* InstantiateBagOfRecommenders(GRand& rand, GArgReader& args)
{
	GBagOfRecommenders* pEnsemble = new GBagOfRecommenders(rand);
	while(args.size() > 0)
	{
		if(args.if_pop("end"))
			break;
		int instance_count = args.pop_uint();
		int arg_pos = args.get_pos();
		for(int i = 0; i < instance_count; i++)
		{
			args.set_pos(arg_pos);
			GCollaborativeFilter* pRecommender = InstantiateAlgorithm(rand, args);
			pEnsemble->addRecommender(pRecommender);
		}
	}
	return pEnsemble;
}

GInstanceRecommender* InstantiateInstanceRecommender(GRand& rand, GArgReader& args)
{
	if(args.size() < 1)
		ThrowError("The number of neighbors must be specified for this algorithm");
	int neighborCount = args.pop_uint();
	double regularizer = 0.0;
	bool pearson = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-pearson"))
			pearson = true;
		else if(args.if_pop("-regularize"))
			regularizer = args.pop_double();
		else
			ThrowError("Invalid option: ", args.peek());
	}
	GInstanceRecommender* pModel = new GInstanceRecommender(neighborCount, rand);
	if(pearson)
		pModel->setMetric(new GPearsonCorrelation(), true);
	pModel->metric()->setRegularizer(regularizer);
	return pModel;
}

GDenseClusterRecommender* InstantiateDenseClusterRecommender(GRand& rand, GArgReader& args)
{
	if(args.size() < 1)
		ThrowError("The number of clusters must be specified for this algorithm");
	size_t clusterCount = args.pop_uint();
	double missingPenalty = 1.0;
	double norm = 2.0;
	while(args.next_is_flag())
	{
		if(args.if_pop("-norm"))
			norm = args.pop_double();
		else if(args.if_pop("-missingpenalty"))
			missingPenalty = args.pop_double();
		else
			ThrowError("Invalid option: ", args.peek());
		// todo: allow the user to specify the clustering algorithm. (Currently, it uses k-means, but k-medoids and agglomerativeclusterer should also be an option)
	}
	GDenseClusterRecommender* pModel = new GDenseClusterRecommender(clusterCount, rand);
	if(norm == 2.0)
	{
		if(missingPenalty != 1.0)
		{
			GKMeans* pClusterer = new GKMeans(clusterCount, &rand);
			pModel->setClusterer(pClusterer, true);
			GRowDistance* pMetric = new GRowDistance();
			pClusterer->setMetric(pMetric, true);
			pMetric->setDiffWithUnknown(missingPenalty);
		}
	}
	else
	{
		GKMeans* pClusterer = new GKMeans(clusterCount, &rand);
		pModel->setClusterer(pClusterer, true);
		GLNormDistance* pMetric = new GLNormDistance(norm);
		pClusterer->setMetric(pMetric, true);
		if(missingPenalty != 1.0)
			pMetric->setDiffWithUnknown(missingPenalty);
	}
	return pModel;
}

GSparseClusterRecommender* InstantiateSparseClusterRecommender(GRand& rand, GArgReader& args)
{
	if(args.size() < 1)
		ThrowError("The number of clusters must be specified for this algorithm");
	size_t clusterCount = args.pop_uint();
	bool pearson = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-pearson"))
			pearson = true;
		else
			ThrowError("Invalid option: ", args.peek());
		// todo: allow the user to specify the clustering algorithm. (Currently, it uses k-means, but k-medoids should also be an option)
	}
	GSparseClusterRecommender* pModel = new GSparseClusterRecommender(clusterCount, rand);
	if(pearson)
	{
		GKMeansSparse* pClusterer = new GKMeansSparse(clusterCount, &rand);
		pClusterer->setMetric(new GPearsonCorrelation(), true);
		pModel->setClusterer(pClusterer, true);
	}
	return pModel;
}

GMatrixFactorization* InstantiateMatrixFactorization(GRand& rand, GArgReader& args)
{
	if(args.size() < 1)
		ThrowError("The number of intrinsic dims must be specified for this algorithm");
	size_t intrinsicDims = args.pop_uint();
	GMatrixFactorization* pModel = new GMatrixFactorization(intrinsicDims, rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-regularize"))
			pModel->setRegularizer(args.pop_double());
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GNonlinearPCA* InstantiateNonlinearPCA(GRand& rand, GArgReader& args)
{
	if(args.size() < 1)
		ThrowError("The number of intrinsic dims must be specified for this algorithm");
	size_t intrinsicDims = args.pop_uint();
	GNonlinearPCA* pModel = new GNonlinearPCA(intrinsicDims, rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-addlayer"))
			pModel->model()->addLayer(args.pop_uint());
		else if(args.if_pop("-learningrate"))
			pModel->model()->setLearningRate(args.pop_double());
		else if(args.if_pop("-momentum"))
			pModel->model()->setMomentum(args.pop_double());
		else if(args.if_pop("-windowepochs"))
			pModel->model()->setWindowSize(args.pop_uint());
		else if(args.if_pop("-minwindowimprovement"))
			pModel->model()->setImprovementThresh(args.pop_double());
		else if(args.if_pop("-noinputbias"))
			pModel->noInputBias();
		else if(args.if_pop("-nothreepass"))
			pModel->noThreePass();
		else if(args.if_pop("-activation"))
		{
			const char* szSF = args.pop_string();
			GActivationFunction* pSF = NULL;
			if(strcmp(szSF, "logistic") == 0)
				pSF = new GActivationLogistic();
			else if(strcmp(szSF, "arctan") == 0)
				pSF = new GActivationArcTan();
			else if(strcmp(szSF, "tanh") == 0)
				pSF = new GActivationTanH();
			else if(strcmp(szSF, "algebraic") == 0)
				pSF = new GActivationAlgebraic();
			else if(strcmp(szSF, "identity") == 0)
				pSF = new GActivationIdentity();
			else if(strcmp(szSF, "bend") == 0)
				pSF = new GActivationBend();
			else if(strcmp(szSF, "bidir") == 0)
				pSF = new GActivationBiDir();
			else if(strcmp(szSF, "piecewise") == 0)
				pSF = new GActivationPiecewise();
			else if(strcmp(szSF, "gaussian") == 0)
				pSF = new GActivationGaussian();
			else if(strcmp(szSF, "sinc") == 0)
				pSF = new GActivationSinc();
			else
				ThrowError("Unrecognized activation function: ", szSF);
			pModel->model()->setActivationFunction(pSF, true);
		}
		else if(args.if_pop("-crossentropy"))
			pModel->model()->setBackPropTargetFunction(GNeuralNet::cross_entropy);
		else if(args.if_pop("-sign"))
			pModel->model()->setBackPropTargetFunction(GNeuralNet::sign);
		else if(args.if_pop("-physical"))
			pModel->model()->setBackPropTargetFunction(GNeuralNet::physical);
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

void showInstantiateAlgorithmError(const char* szMessage, GArgReader& args)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	const char* szAlgName = args.peek();
	UsageNode* pAlgTree = makeCollaborativeFilterUsageTree();
	Holder<UsageNode> hAlgTree(pAlgTree);
	if(szAlgName)
	{
		UsageNode* pUsageAlg = pAlgTree->choice(szAlgName);
		if(pUsageAlg)
		{
			cerr << "Partial Usage Information:\n\n";
			pUsageAlg->print(cerr, 0, 3, 76, 1000, true);
		}
		else
		{
			cerr << "\"" << szAlgName << "\" is not a recognized algorithm. Try one of these:\n\n";
			pAlgTree->print(cerr, 0, 3, 76, 1, false);
		}
	}
	else
	{
		cerr << "Expected an algorithm. Here are some choices:\n";
		pAlgTree->print(cerr, 0, 3, 76, 1, false);
	}
	cerr << "\nTo see full usage information, run:\n	waffles_learn usage\n\n";
	cerr << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cerr.flush();
}

GCollaborativeFilter* InstantiateAlgorithm(GRand& rand, GArgReader& args)
{
	int argPos = args.get_pos();
	if(args.size() < 1)
		ThrowError("No algorithm specified.");
	try
	{
		if(args.if_pop("baseline"))
			return InstantiateBaselineRecommender(rand, args);
		else if(args.if_pop("bag"))
			return InstantiateBagOfRecommenders(rand, args);
		else if(args.if_pop("instance"))
			return InstantiateInstanceRecommender(rand, args);
		else if(args.if_pop("clusterdense"))
			return InstantiateDenseClusterRecommender(rand, args);
		else if(args.if_pop("clustersparse"))
			return InstantiateSparseClusterRecommender(rand, args);
		else if(args.if_pop("matrix"))
			return InstantiateMatrixFactorization(rand, args);
		else if(args.if_pop("nlpca"))
			return InstantiateNonlinearPCA(rand, args);
		else
			ThrowError("Unrecognized algorithm name: ", args.peek());
	}
	catch(const std::exception& e)
	{
		args.set_pos(argPos);
		showInstantiateAlgorithmError(e.what(), args);
		ThrowError("nevermind"); // this means "don't display another error message"
	}
	return NULL;
}

void crossValidate(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	size_t folds = 2;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-folds"))
			folds = args.pop_uint();
		else
			ThrowError("Invalid crossvalidate option: ", args.peek());
	}
	if(folds < 2)
		ThrowError("There must be at least 2 folds.");

	// Load the data
	if(args.size() < 1)
		ThrowError("No dataset specified.");
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Instantiate the recommender
	GRand prng(seed);
	GCollaborativeFilter* pModel = InstantiateAlgorithm(prng, args);
	Holder<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Do cross-validation
	double mae;
	double mse = pModel->crossValidate(*pData, folds, &mae);
	cout << "RMSE=" << sqrt(mse) << ", MSE=" << mse << ", MAE=" << mae << "\n";
}

void precisionRecall(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool ideal = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-ideal"))
			ideal = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the data
	if(args.size() < 1)
		ThrowError("No dataset specified.");
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Instantiate the recommender
	GRand prng(seed);
	GCollaborativeFilter* pModel = InstantiateAlgorithm(prng, args);
	Holder<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Generate precision-recall data
	GMatrix* pResults = pModel->precisionRecall(*pData, ideal);
	Holder<GMatrix> hResults(pResults);
	pResults->deleteColumn(2); // we don't need the false-positive rate column
	pResults->print(cout);
}

void ROC(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool ideal = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-ideal"))
			ideal = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the data
	if(args.size() < 1)
		ThrowError("No dataset specified.");
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Instantiate the recommender
	GRand prng(seed);
	GCollaborativeFilter* pModel = InstantiateAlgorithm(prng, args);
	Holder<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Generate ROC data
	GMatrix* pResults = pModel->precisionRecall(*pData, ideal);
	Holder<GMatrix> hResults(pResults);
	double auc = GCollaborativeFilter::areaUnderCurve(*pResults);
	pResults->deleteColumn(1); // we don't need the precision column
	pResults->swapColumns(0, 1);
	cout << "% Area Under the Curve = " << auc << "\n";
	pResults->print(cout);
}

void transacc(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			ThrowError("Invalid crossvalidate option: ", args.peek());
	}

	// Load the data
	if(args.size() < 1)
		ThrowError("No training set specified.");
	GMatrix* pTrain = loadData(args.pop_string());
	Holder<GMatrix> hTrain(pTrain);
	if(args.size() < 1)
		ThrowError("No test set specified.");
	GMatrix* pTest = loadData(args.pop_string());
	Holder<GMatrix> hTest(pTest);

	// Instantiate the recommender
	GRand prng(seed);
	GCollaborativeFilter* pModel = InstantiateAlgorithm(prng, args);
	Holder<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Do cross-validation
	double mae;
	double mse = pModel->trainAndTest(*pTrain, *pTest, &mae);
	cout << "MSE=" << mse << ", MAE=" << mae << "\n";
}

void fillMissingValues(GArgReader& args)
{
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the data and the filter
	GMatrix* pDataOrig = GMatrix::loadArff(args.pop_string());
	Holder<GMatrix> hDataOrig(pDataOrig);
	sp_relation pOrigRel = pDataOrig->relation();
	GRand prng(seed);
	GCollaborativeFilter* pModel = InstantiateAlgorithm(prng, args);
	Holder<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Convert to all normalized real values
	GNominalToCat* pNtc = new GNominalToCat();
	GTwoWayTransformChainer filter(new GNormalize(), pNtc);
	pNtc->preserveUnknowns();
	filter.train(*pDataOrig);
	GMatrix* pData = filter.transformBatch(*pDataOrig);
	Holder<GMatrix> hData(pData);
	hDataOrig.release();
	pDataOrig = NULL;

	// Convert to 3-column form
	GMatrix* pMatrix = new GMatrix(0, 3);
	Holder<GMatrix> hMatrix(pMatrix);
	size_t dims = pData->cols();
	for(size_t i = 0; i < pData->rows(); i++)
	{
		double* pRow = pData->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			if(*pRow != UNKNOWN_REAL_VALUE)
			{
				double* pVec = pMatrix->newRow();
				pVec[0] = i;
				pVec[1] = j;
				pVec[2] = *pRow;
			}
			pRow++;
		}
	}

	// Train the collaborative filter
	pModel->train(*pMatrix);
	hMatrix.release();
	pMatrix = NULL;

	// Predict values for missing elements
	for(size_t i = 0; i < pData->rows(); i++)
	{
		double* pRow = pData->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			if(*pRow == UNKNOWN_REAL_VALUE)
				*pRow = pModel->predict(i, j);
			GAssert(*pRow != UNKNOWN_REAL_VALUE);
			pRow++;
		}
	}

	// Convert the data back to its original form
	GMatrix* pOut = filter.untransformBatch(*pData);
	pOut->setRelation(pOrigRel);
	pOut->print(cout);
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeRecommendUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(cout, 0, 3, 76, 1000, true);
	UsageNode* pUsageTree2 = makeCollaborativeFilterUsageTree();
	Holder<UsageNode> hUsageTree2(pUsageTree2);
	pUsageTree2->print(cout, 0, 3, 76, 1000, true);
	cout.flush();
}

void showError(GArgReader& args, const char* szAppName, const char* szMessage)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	args.set_pos(1);
	const char* szCommand = args.peek();
	UsageNode* pUsageTree = makeRecommendUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << szAppName << " ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, true);
			if(pUsageCommand->findPart("[collab-filter]") >= 0)
			{
				UsageNode* pAlgTree = makeCollaborativeFilterUsageTree();
				Holder<UsageNode> hAlgTree(pAlgTree);
				pAlgTree->print(cerr, 1, 3, 76, 2, false);
			}
		}
		else
		{
			cerr << "Brief Usage Information:\n\n";
			pUsageTree->print(cerr, 0, 3, 76, 1, false);
		}
	}
	else
	{
		pUsageTree->print(cerr, 0, 3, 76, 1, false);
		cerr << "\nFor more specific usage information, enter as much of the command as you know.\n";
	}
	cerr << "\nTo see full usage information, run:\n	" << szAppName << " usage\n\n";
	cerr << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cerr.flush();
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int nRet = 0;
	PathData pd;
	GFile::parsePath(argv[0], &pd);
	const char* appName = argv[0] + pd.fileStart;
	GArgReader args(argc, argv);
	try
	{
		args.pop_string(); // advance past the name of this app
		if(args.size() >= 1)
		{
			if(args.if_pop("usage"))
				ShowUsage(appName);
			else if(args.if_pop("crossvalidate"))
				crossValidate(args);
			else if(args.if_pop("fillmissingvalues"))
				fillMissingValues(args);
			else if(args.if_pop("precisionrecall"))
				precisionRecall(args);
			else if(args.if_pop("roc"))
				ROC(args);
			else if(args.if_pop("transacc"))
				transacc(args);
			else
			{
				nRet = 1;
				string s = args.peek();
				s += " is not a recognized command.";
				showError(args, appName, s.c_str());
			}
		}
		else
		{
			nRet = 1;
			showError(args, appName, "Brief Usage Information:");
		}
	}
	catch(const std::exception& e)
	{
		nRet = 1;
		if(strcmp(e.what(), "nevermind") != 0) // if an error message was not already displayed...
			showError(args, appName, e.what());
	}
	return nRet;
}
