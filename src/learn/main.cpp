// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "../GClasses/GActivation.h"
#include "../GClasses/GApp.h"
#include "../GClasses/GMatrix.h"
#include "../GClasses/GCluster.h"
#include "../GClasses/GDecisionTree.h"
#include "../GClasses/GDistance.h"
#include "../GClasses/GDistribution.h"
#include "../GClasses/GEnsemble.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GImage.h"
#include "../GClasses/GKernelTrick.h"
#include "../GClasses/GKNN.h"
#include "../GClasses/GLinear.h"
#include "../GClasses/GError.h"
#include "../GClasses/GManifold.h"
#include "../GClasses/GNaiveBayes.h"
#include "../GClasses/GNaiveInstance.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GSparseMatrix.h"
#include "../GClasses/GSystemLearner.h"
#include "../GClasses/GTime.h"
#include "../GClasses/GTransform.h"
#include "../GClasses/GDom.h"
#include "../GClasses/GVec.h"
#include "../wizard/usage.h"
#include <time.h>
#include <iostream>
#ifdef WINDOWS
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
using std::ostringstream;

GTransducer* InstantiateAlgorithm(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

size_t getAttrVal(const char* szString, size_t attrCount)
{
	bool fromRight = false;
	if(*szString == '*')
	{
		fromRight = true;
		szString++;
	}
	if(*szString < '0' || *szString > '9')
		ThrowError("Expected a digit while parsing attribute list");
#ifdef WIN32
	size_t val = (size_t)_strtoui64(szString, (char**)NULL, 10);
#else
	size_t val = strtoull(szString, (char**)NULL, 10);
#endif
	if(fromRight)
		val = attrCount - 1 - val;
	return val;
}

void parseAttributeList(vector<size_t>& list, GArgReader& args, size_t attrCount)
{
	const char* szList = args.pop_string();
	set<size_t> attrSet;
	while(true)
	{
		// Skip whitespace
		while(*szList <= ' ' && *szList != '\0')
			szList++;

		// Find the next ',' or the end of string, and
		int i;
		int j = -1;
		for(i = 0; szList[i] != '\0' && szList[i] != ','; i++)
		{
			if(j < 0 && szList[i] == '-')
				j = i;
		}
		if(j >= 0)
		{
			while(szList[j + 1] <= ' ' && szList[j + 1] != '\0')
				j++;
		}

		// Add the attributes to the list
		if(i > 0) // If there is more...
		{
			if(j < 0) // If there is no "-" character in the next value...
			{
				size_t val = getAttrVal(szList, attrCount);
				if(val >= attrCount)
					ThrowError("Invalid column index: ", to_str(val), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
				if(attrSet.find(val) != attrSet.end())
					ThrowError("Columns ", to_str(val), " is listed multiple times");
				attrSet.insert(val);
				list.push_back(val);
			}
			else
			{
				size_t beg = getAttrVal(szList, attrCount);
				if(beg >= attrCount)
					ThrowError("Invalid column index: ", to_str(beg), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
				size_t end = getAttrVal(szList + j + 1, attrCount);
				if(end >= attrCount)
					ThrowError("Invalid column index: ", to_str(end), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
				int step = 1;
				if(end < beg)
					step = -1;
				for(size_t val = beg; true; val += step)
				{
					if(attrSet.find(val) != attrSet.end())
						ThrowError("Column ", to_str(val), " is listed multiple times");
					attrSet.insert(val);
						list.push_back(val);
					if(val == end)
						break;
				}
			}
		}

		// Advance
		szList += i;
		if(*szList == '\0')
			break;
		szList++;
	}
}

void loadData(GArgReader& args, Holder<GMatrix>& hFeaturesOut, Holder<GMatrix>& hLabelsOut)
{
	// Load the dataset by extension
	if(args.size() < 1)
		ThrowError("Expected the filename of a datset. (Found end of arguments.)");
	const char* szFilename = args.pop_string();
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	GMatrix* pData = NULL;
	if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
		pData = GMatrix::loadArff(szFilename);
	else if(_stricmp(szFilename + pd.extStart, ".csv") == 0)
		pData = GMatrix::loadCsv(szFilename, ',', false, false);
	else if(_stricmp(szFilename + pd.extStart, ".dat") == 0)
		pData = GMatrix::loadCsv(szFilename, '\0', false, false);
	else
		ThrowError("Unsupported file format: ", szFilename + pd.extStart);
	Holder<GMatrix> hData(pData);

	// Parse params
	vector<size_t> ignore;
	vector<size_t> labels;
	while(args.next_is_flag())
	{
		if(args.if_pop("-labels"))
			parseAttributeList(labels, args, pData->cols());
		else if(args.if_pop("-ignore"))
			parseAttributeList(ignore, args, pData->cols());
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Throw out the ignored attributes
	std::sort(ignore.begin(), ignore.end());
	for(size_t i = ignore.size() - 1; i < ignore.size(); i--)
	{
		pData->deleteColumn(ignore[i]);
		for(size_t j = 0; j < labels.size(); j++)
		{
			if(labels[j] >= ignore[i])
			{
				if(labels[j] == ignore[i])
					ThrowError("Attribute ", to_str(labels[j]), " is both ignored and used as a label");
				labels[j]--;
			}
		}
	}

	// Swap label columns to the end
	size_t labelDims = std::max((size_t)1, labels.size());
	for(size_t i = 0; i < labels.size(); i++)
	{
		size_t src = labels[i];
		size_t dst = pData->cols() - labelDims + i;
		if(src != dst)
		{
			pData->swapColumns(src, dst);
			for(size_t j = i + 1; j < labels.size(); j++)
			{
				if(labels[j] == dst)
				{
					labels[j] = src;
					break;
				}
			}
		}
	}

	// Split pData into a feature matrix and a label matrix
	GMatrix* pFeatures = pData->cloneSub(0, 0, pData->rows(), pData->cols() - labelDims);
	hFeaturesOut.reset(pFeatures);
	GMatrix* pLabels = pData->cloneSub(0, pData->cols() - labelDims, pData->rows(), labelDims);
	hLabelsOut.reset(pLabels);
}

GAgglomerativeTransducer* InstantiateAgglomerativeTransducer(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GAgglomerativeTransducer* pTransducer = new GAgglomerativeTransducer(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pTransducer->autoTune(*pFeatures, *pLabels);
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pTransducer;
}

GBag* InstantiateBag(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GBag* pEnsemble = new GBag(rand);
	while(args.size() > 0)
	{
		if(args.if_pop("end"))
			break;
		int instance_count = args.pop_uint();
		int arg_pos = args.get_pos();
		for(int i = 0; i < instance_count; i++)
		{
			args.set_pos(arg_pos);
			GTransducer* pLearner = InstantiateAlgorithm(rand, args, pFeatures, pLabels);
			if(!pLearner->canGeneralize())
			{
				delete(pLearner);
				ThrowError("bag does not support algorithms that cannot generalize.");
			}
			pEnsemble->addLearner((GSupervisedLearner*)pLearner);
		}
	}
	return pEnsemble;
}

GBaselineLearner* InstantiateBaseline(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GBaselineLearner* pModel = new GBaselineLearner(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GBayesianModelAveraging* InstantiateBMA(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GBayesianModelAveraging* pEnsemble = new GBayesianModelAveraging(rand);
	while(args.size() > 0)
	{
		if(args.if_pop("end"))
			break;
		int instance_count = args.pop_uint();
		int arg_pos = args.get_pos();
		for(int i = 0; i < instance_count; i++)
		{
			args.set_pos(arg_pos);
			GTransducer* pLearner = InstantiateAlgorithm(rand, args, pFeatures, pLabels);
			if(!pLearner->canGeneralize())
			{
				delete(pLearner);
				ThrowError("BMA does not support algorithms that cannot generalize.");
			}
			pEnsemble->addLearner((GSupervisedLearner*)pLearner);
		}
	}
	return pEnsemble;
}

GBayesianModelCombination* InstantiateBMC(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GBayesianModelCombination* pEnsemble = new GBayesianModelCombination(rand);
	size_t samples = 100;
	while(args.next_is_flag())
	{
		if(args.if_pop("-samples"))
			samples = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}
	pEnsemble->setSamples(samples);
	while(args.size() > 0)
	{
		if(args.if_pop("end"))
			break;
		int instance_count = args.pop_uint();
		int arg_pos = args.get_pos();
		for(int i = 0; i < instance_count; i++)
		{
			args.set_pos(arg_pos);
			GTransducer* pLearner = InstantiateAlgorithm(rand, args, pFeatures, pLabels);
			if(!pLearner->canGeneralize())
			{
				delete(pLearner);
				ThrowError("BMC does not support algorithms that cannot generalize.");
			}
			pEnsemble->addLearner((GSupervisedLearner*)pLearner);
		}
	}
	return pEnsemble;
}

GAdaBoost* InstantiateBoost(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	double trainingSizeRatio = 1;
	size_t ensembleSize = 30;
	
	while(args.size() > 0)
	{
		if(args.if_pop("-trainratio"))
		{
			trainingSizeRatio = args.pop_double();
		}
		else if(args.if_pop("-size"))
		{
			ensembleSize = args.pop_uint();
		}
		else
		{
			break;
		}
	}
	
	GTransducer* pLearner = InstantiateAlgorithm(rand, args, pFeatures, pLabels);
	if(!pLearner->canGeneralize())
		{
			delete(pLearner);
			ThrowError("boost does not support algorithms that cannot generalize.");
		}

	GAdaBoost* pEnsemble = new GAdaBoost((GSupervisedLearner*)pLearner, true, new GLearnerLoader(rand));
	pEnsemble->setTrainSize(trainingSizeRatio);
	pEnsemble->setSize(ensembleSize);

	return pEnsemble;
}

GBucket* InstantiateBucket(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GBucket* pEnsemble = new GBucket(rand);
	while(args.size() > 0)
	{
		if(args.if_pop("end"))
			break;
		GTransducer* pLearner = InstantiateAlgorithm(rand, args, pFeatures, pLabels);
		if(!pLearner->canGeneralize())
		{
			delete(pLearner);
			ThrowError("crossvalidationselector does not support algorithms that cannot generalize.");
		}
		pEnsemble->addLearner((GSupervisedLearner*)pLearner);
	}
	return pEnsemble;
}

GBucket* InstantiateCvdt(GRand& rand, GArgReader& args)
{
	size_t trees = args.pop_uint();
	GBucket* pBucket = new GBucket(rand);
	GBag* pBag1 = new GBag(rand);
	pBucket->addLearner(pBag1);
	for(size_t i = 0; i < trees; i++)
		pBag1->addLearner(new GDecisionTree(rand));
	GBag* pBag2 = new GBag(rand);
	pBucket->addLearner(pBag2);
	for(size_t i = 0; i < trees; i++)
		pBag2->addLearner(new GMeanMarginsTree(rand));
	return pBucket;
}

GDecisionTree* InstantiateDecisionTree(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GDecisionTree* pModel = new GDecisionTree(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-random")){
			pModel->useRandomDivisions(args.pop_uint());
		}else if(args.if_pop("-leafthresh")){
			pModel->setLeafThresh(args.pop_uint());
		}else if(args.if_pop("-maxlevels")){
			pModel->setMaxLevels(args.pop_uint());
		}else{
			ThrowError("Invalid option: ", args.peek());
		}
	}
	return pModel;
}

GGraphCutTransducer* InstantiateGraphCutTransducer(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GGraphCutTransducer* pTransducer = new GGraphCutTransducer(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pTransducer->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-neighbors"))
			pTransducer->setNeighbors(args.pop_uint());
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pTransducer;
}

GKNN* InstantiateKNN(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GKNN* pModel = new GKNN(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-neighbors"))
			pModel->setNeighborCount(args.pop_uint());
		else if(args.if_pop("-equalweight"))
			pModel->setInterpolationMethod(GKNN::Mean);
		else if(args.if_pop("-scalefeatures"))
			pModel->setOptimizeScaleFactors(true);
		else if(args.if_pop("-cosine"))
			pModel->setMetric(new GCosineSimilarity(), true);
		else if(args.if_pop("-pearson"))
			pModel->setMetric(new GPearsonCorrelation(), true);
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GLinearRegressor* InstantiateLinearRegressor(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GLinearRegressor* pModel = new GLinearRegressor(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GMeanMarginsTree* InstantiateMeanMarginsTree(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GMeanMarginsTree* pModel = new GMeanMarginsTree(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GNaiveBayes* InstantiateNaiveBayes(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GNaiveBayes* pModel = new GNaiveBayes(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-ess"))
			pModel->setEquivalentSampleSize(args.pop_double());
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GNaiveInstance* InstantiateNaiveInstance(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GNaiveInstance* pModel = new GNaiveInstance(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-neighbors"))
			pModel->setNeighbors(args.pop_uint());
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GNeighborTransducer* InstantiateNeighborTransducer(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GNeighborTransducer* pTransducer = new GNeighborTransducer(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pTransducer->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-neighbors"))
			pTransducer->setNeighbors(args.pop_uint());
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pTransducer;
}

GNeuralNet* InstantiateNeuralNet(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GNeuralNet* pModel = new GNeuralNet(rand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-addlayer"))
			pModel->addLayer(args.pop_uint());
		else if(args.if_pop("-learningrate"))
			pModel->setLearningRate(args.pop_double());
		else if(args.if_pop("-momentum"))
			pModel->setMomentum(args.pop_double());
		else if(args.if_pop("-windowepochs"))
			pModel->setWindowSize(args.pop_uint());
		else if(args.if_pop("-minwindowimprovement"))
			pModel->setImprovementThresh(args.pop_double());
		else if(args.if_pop("-holdout"))
			pModel->setValidationPortion(args.pop_double());
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
			else if(strcmp(szSF, "gaussian") == 0)
				pSF = new GActivationGaussian();
			else if(strcmp(szSF, "sinc") == 0)
				pSF = new GActivationSinc();
			else if(strcmp(szSF, "bend") == 0)
				pSF = new GActivationBend();
			else if(strcmp(szSF, "bidir") == 0)
				pSF = new GActivationBiDir();
			else if(strcmp(szSF, "piecewise") == 0)
				pSF = new GActivationPiecewise();
			else
				ThrowError("Unrecognized activation function: ", szSF);
			pModel->setActivationFunction(pSF, true);
		}
		else if(args.if_pop("-crossentropy"))
			pModel->setBackPropTargetFunction(GNeuralNet::cross_entropy);
		else if(args.if_pop("-physical"))
			pModel->setBackPropTargetFunction(GNeuralNet::physical);
		else if(args.if_pop("-sign"))
			pModel->setBackPropTargetFunction(GNeuralNet::sign);
		else
			ThrowError("Invalid option: ", args.peek());
	}
	return pModel;
}

GRandomForest* InstantiateRandomForest(GRand& rand, GArgReader& args)
{
	size_t trees = args.pop_uint();
	size_t samples = 1;
	while(args.next_is_flag())
	{
		if(args.if_pop("-samples"))
			samples = args.pop_uint();
		else
			ThrowError("Invalid random forest option: ", args.peek());
	}
	return new GRandomForest(rand, trees, samples);
}

void showInstantiateAlgorithmError(const char* szMessage, GArgReader& args)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	const char* szAlgName = args.peek();
	UsageNode* pAlgTree = makeAlgorithmUsageTree();
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

GWag* InstantiateWag(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	GWag* pWag = new GWag(0, rand);
	GNeuralNet* pModel = pWag->model();
	size_t modelCount = 10;
	while(args.next_is_flag())
	{
		if(args.if_pop("-autotune"))
		{
			if(!pFeatures || !pLabels)
				ThrowError("Insufficient data to support automatic tuning");
			pModel->autoTune(*pFeatures, *pLabels);
		}
		else if(args.if_pop("-addlayer"))
			pModel->addLayer(args.pop_uint());
		else if(args.if_pop("-learningrate"))
			pModel->setLearningRate(args.pop_double());
		else if(args.if_pop("-momentum"))
			pModel->setMomentum(args.pop_double());
		else if(args.if_pop("-models"))
			modelCount = args.pop_uint();
		else if(args.if_pop("-windowepochs"))
			pModel->setWindowSize(args.pop_uint());
		else if(args.if_pop("-minwindowimprovement"))
			pModel->setImprovementThresh(args.pop_double());
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
			else if(strcmp(szSF, "gaussian") == 0)
				pSF = new GActivationGaussian();
			else if(strcmp(szSF, "sinc") == 0)
				pSF = new GActivationSinc();
			else if(strcmp(szSF, "bend") == 0)
				pSF = new GActivationBend();
			else if(strcmp(szSF, "bidir") == 0)
				pSF = new GActivationBiDir();
			else if(strcmp(szSF, "piecewise") == 0)
				pSF = new GActivationPiecewise();
			else
				ThrowError("Unrecognized activation function: ", szSF);
			pModel->setActivationFunction(pSF, true);
		}
		else if(args.if_pop("-crossentropy"))
			pModel->setBackPropTargetFunction(GNeuralNet::cross_entropy);
		else if(args.if_pop("-physical"))
			pModel->setBackPropTargetFunction(GNeuralNet::physical);
		else if(args.if_pop("-sign"))
			pModel->setBackPropTargetFunction(GNeuralNet::sign);
		else
			ThrowError("Invalid option: ", args.peek());
	}
	pWag->setModelCount(modelCount);
	return pWag;
}

GTransducer* InstantiateAlgorithm(GRand& rand, GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels)
{
	int argPos = args.get_pos();
	if(args.size() < 1)
		ThrowError("No algorithm specified.");
	try
	{
		if(args.if_pop("agglomerativetransducer"))
			return InstantiateAgglomerativeTransducer(rand, args, pFeatures, pLabels);
		else if(args.if_pop("bag"))
			return InstantiateBag(rand, args, pFeatures, pLabels);
		else if(args.if_pop("baseline"))
			return InstantiateBaseline(rand, args, pFeatures, pLabels);
		else if(args.if_pop("bma"))
			return InstantiateBMA(rand, args, pFeatures, pLabels);
		else if(args.if_pop("bmc"))
			return InstantiateBMC(rand, args, pFeatures, pLabels);
		else if(args.if_pop("boost"))
			return InstantiateBoost(rand, args, pFeatures, pLabels);
		else if(args.if_pop("bucket"))
			return InstantiateBucket(rand, args, pFeatures, pLabels);
		else if(args.if_pop("cvdt"))
			return InstantiateCvdt(rand, args);
		else if(args.if_pop("decisiontree"))
			return InstantiateDecisionTree(rand, args, pFeatures, pLabels);
		else if(args.if_pop("graphcuttransducer"))
			return InstantiateGraphCutTransducer(rand, args, pFeatures, pLabels);
		else if(args.if_pop("knn"))
			return InstantiateKNN(rand, args, pFeatures, pLabels);
		else if(args.if_pop("linear"))
			return InstantiateLinearRegressor(rand, args, pFeatures, pLabels);
		else if(args.if_pop("meanmarginstree"))
			return InstantiateMeanMarginsTree(rand, args, pFeatures, pLabels);
		else if(args.if_pop("naivebayes"))
			return InstantiateNaiveBayes(rand, args, pFeatures, pLabels);
		else if(args.if_pop("naiveinstance"))
			return InstantiateNaiveInstance(rand, args, pFeatures, pLabels);
		else if(args.if_pop("neighbortransducer"))
			return InstantiateNeighborTransducer(rand, args, pFeatures, pLabels);
		else if(args.if_pop("neuralnet"))
			return InstantiateNeuralNet(rand, args, pFeatures, pLabels);
		else if(args.if_pop("randomforest"))
			return InstantiateRandomForest(rand, args);
		else if(args.if_pop("wag"))
			return InstantiateWag(rand, args, pFeatures, pLabels);
		ThrowError("Unrecognized algorithm name: ", args.peek());
	}
	catch(const std::exception& e)
	{
		args.set_pos(argPos);
		if(strcmp(e.what(), "nevermind") != 0) // if an error message was not already displayed...
			showInstantiateAlgorithmError(e.what(), args);
		ThrowError("nevermind"); // this means "don't display another error message"
	}
	return NULL;
}

void autoTuneDecisionTree(GMatrix& features, GMatrix& labels, GRand& rand)
{
	GDecisionTree dt(rand);
	dt.autoTune(features, labels);
	cout << "decisiontree";
	if(dt.leafThresh() != 1)
		cout << " -leafthresh " << dt.leafThresh();
	cout << "\n";
}

void autoTuneKNN(GMatrix& features, GMatrix& labels, GRand& rand)
{
	GKNN model(rand);
	model.autoTune(features, labels);
	cout << "knn";
	if(model.neighborCount() != 1)
		cout << " -neighbors " << model.neighborCount();
	cout << "\n";
}

void autoTuneNeuralNet(GMatrix& features, GMatrix& labels, GRand& rand)
{
	cout << "Warning: Because neural nets take a long time to train, it could take hours to train with enough parameter variations to determine with confidence which parameters are best. (If possible, I would strongly advise running this as a background process while you do something else, rather than sit around waiting for it to finish.)";
	cout.flush();
	GNeuralNet nn(rand);
	nn.autoTune(features, labels);
	const char* szCurrent = "logistic";
	cout << "neuralnet";
	for(size_t i = 0; i < nn.layerCount(); i++)
	{
		const char* szActivationName = nn.layer(i).m_pActivationFunction->name();
		if(strcmp(szActivationName, szCurrent) != 0)
		{
			cout << " -activation " << szActivationName;
			szCurrent = szActivationName;
		}
		if(i < nn.layerCount() - 1)
			cout << " -addlayer " << nn.layer(i).m_neurons.size();
	}
	if(nn.momentum() > 0.0)
		cout << " -momentum " << nn.momentum();
	cout << "\n";
}

void autoTuneNaiveBayes(GMatrix& features, GMatrix& labels, GRand& rand)
{
	GNaiveBayes model(rand);
	model.autoTune(features, labels);
	cout << "naivebayes";
	cout << " -ess " << model.equivalentSampleSize();
	cout << "\n";
}

void autoTuneNaiveInstance(GMatrix& features, GMatrix& labels, GRand& rand)
{
	GNaiveInstance model(rand);
	model.autoTune(features, labels);
	cout << "naiveinstance";
	cout << " -neighbors " << model.neighbors();
	cout << "\n";
}

void autoTuneGraphCutTransducer(GMatrix& features, GMatrix& labels, GRand& rand)
{
	GGraphCutTransducer transducer(rand);
	transducer.autoTune(features, labels);
	cout << "graphcuttransducer";
	cout << " -neighbors " << transducer.neighbors();
	cout << "\n";
}

void autoTune(GArgReader& args)
{
	// Load the data
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();

	// Load the model name
	GRand rand(0);
	const char* szModel = args.pop_string();
	if(strcmp(szModel, "agglomerativetransducer") == 0)
		cout << "agglomerativetransducer\n"; // no params to tune
	else if(strcmp(szModel, "decisiontree") == 0)
		autoTuneDecisionTree(*pFeatures, *pLabels, rand);
	else if(strcmp(szModel, "graphcuttransducer") == 0)
		autoTuneGraphCutTransducer(*pFeatures, *pLabels, rand);
	else if(strcmp(szModel, "knn") == 0)
		autoTuneKNN(*pFeatures, *pLabels, rand);
	else if(strcmp(szModel, "meanmarginstree") == 0)
		cout << "meanmarginstree\n"; // no params to tune
	else if(strcmp(szModel, "neuralnet") == 0)
		autoTuneNeuralNet(*pFeatures, *pLabels, rand);
	else if(strcmp(szModel, "naivebayes") == 0)
		autoTuneNaiveBayes(*pFeatures, *pLabels, rand);
	else if(strcmp(szModel, "naiveinstance") == 0)
		autoTuneNaiveInstance(*pFeatures, *pLabels, rand);
	else
		ThrowError("Sorry, autotune does not currently support a model named ", szModel, ".");
}

void Train(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool calibrate = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-calibrate"))
			calibrate = true;
		else
			ThrowError("Invalid train option: ", args.peek());
	}

	// Load the data
	GRand prng(seed);
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();

	// Instantiate the modeler
	GTransducer* pSupLearner = InstantiateAlgorithm(prng, args, pFeatures, pLabels);
	Holder<GTransducer> hModel(pSupLearner);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());
	if(!pSupLearner->canGeneralize())
		ThrowError("This algorithm cannot be \"trained\". It can only be used to \"transduce\".");
	GSupervisedLearner* pModel = (GSupervisedLearner*)pSupLearner;

	// Train the modeler
	pModel->train(*pFeatures, *pLabels);
	if(calibrate)
		pModel->calibrate(*pFeatures, *pLabels);

	// Output the trained model
	GDom doc;
	GDomNode* pRoot = pModel->serialize(&doc);
	doc.setRoot(pRoot);
	doc.writeJson(cout);
}

void predict(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			ThrowError("Invalid predict option: ", args.peek());
	}

	// Load the model
	GRand prng(seed);
	GDom doc;
	if(args.size() < 1)
		ThrowError("Model not specified.");
	doc.loadJson(args.pop_string());
	GLearnerLoader ll(prng, true);
	GSupervisedLearner* pModeler = ll.loadSupervisedLearner(doc.root());
	Holder<GSupervisedLearner> hModeler(pModeler);

	// Load the data
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();
	if(pLabels->cols() != pModeler->labelDims())
		ThrowError("The model was trained with ", to_str(pModeler->labelDims()), " label dims, but the specified dataset has ", to_str(pLabels->cols()));
	pLabels->setAll(0.0); // Wipe out the existing labels, just to be absolutely certain that we don't somehow accidentally let them influence the predictions

	// Test
	for(size_t i = 0; i < pFeatures->rows(); i++)
	{
		double* pFeatureVec = pFeatures->row(i);
		double* pLabelVec = pLabels->row(i);
		pModeler->predict(pFeatureVec, pLabelVec);
	}

	// Print results
	pLabels->print(cout);
}

void predictDistribution(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the model
	GRand prng(seed);
	GDom doc;
	if(args.size() < 1)
		ThrowError("Model not specified.");
	doc.loadJson(args.pop_string());
	GLearnerLoader ll(prng, true);
	GSupervisedLearner* pModeler = ll.loadSupervisedLearner(doc.root());
	Holder<GSupervisedLearner> hModeler(pModeler);

	// Load the dataset
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();
	if(pLabels->cols() != (size_t)pModeler->labelDims())
		ThrowError("The model was trained with ", to_str(pModeler->labelDims()), " label dims, but the specified dataset has ", to_str(pLabels->cols()));
	if(pFeatures->relation()->type() != GRelation::ARFF || pLabels->relation()->type() != GRelation::ARFF)
		ThrowError("Expected a dataset with ARFF metadata");
	GArffRelation* pFeatureRel = (GArffRelation*)pFeatures->relation().get();
	GArffRelation* pLabelRel = (GArffRelation*)pLabels->relation().get();

	// Parse the pattern
	size_t featureDims = pModeler->featureDims();
	GTEMPBUF(double, pattern, featureDims);
	for(size_t i = 0; i < featureDims; i++)
		pattern[i] = pFeatureRel->parseValue(i, args.pop_string());

	// Predict
	GPrediction* out = new GPrediction[pModeler->labelDims()];
	ArrayHolder<GPrediction> hOut(out);
	pModeler->predictDistribution(pattern, out);

	// Display the prediction
	cout.precision(8);
	for(size_t i = 0; i < pLabels->cols(); i++)
	{
		if(i > 0)
			cout << ", ";
		if(pLabelRel->valueCount(i) == 0)
			cout << out[i].mode();
		else
			pLabelRel->printAttrValue(cout, i, (int)out[i].mode());
	}
	cout << "\n\n";

	// Display the distribution
	for(size_t i = 0; i < pModeler->labelDims(); i++)
	{
		if(out[i].isContinuous())
		{
			GNormalDistribution* pNorm = out[i].asNormal();
			cout << pLabelRel->attrName(i) << ") Normal: predicted mean=" << pNorm->mean() << " predicted variance=" << pNorm->variance() << "\n";
		}
		else
		{
			GCategoricalDistribution* pCat = out[i].asCategorical();
			cout << pLabelRel->attrName(i) << ") Categorical confidences: {";
			double* pValues = pCat->values(pCat->valueCount());
			for(size_t j = 0; j < pCat->valueCount(); j++)
			{
				if(j > 0)
					cout << ", ";
				pLabelRel->printAttrValue(cout, i, (int)j);
				cout << "=" << pValues[j];
			}
			cout << "}\n";
		}
	}
}

void leftJustifiedString(const char* pIn, char* pOut, size_t outLen)
{
	size_t inLen = std::min(outLen, strlen(pIn));
	memcpy(pOut, pIn, inLen);
	memset(pOut + inLen, ' ', outLen - inLen);
	pOut[outLen] = '\0';
}

void rightJustifiedString(const char* pIn, char* pOut, size_t outLen)
{
	size_t inLen = strlen(pIn);
	size_t spaces = std::max(outLen, inLen) - inLen;
	memset(pOut, ' ', spaces);
	memcpy(pOut + spaces, pIn, outLen - spaces);
	pOut[outLen] = '\0';
}

void printConfusionMatrices(GRelation* pRelation, vector<GMatrix*>& matrixArray)
{
	cout << "\n(Rows=expected values, Cols=predicted values, Elements=number of occurrences)\n\n";
	char buf[41];
	ostringstream oss;
	for(size_t i = 0; i < matrixArray.size(); i++)
	{
		if(matrixArray[i] == NULL)
			continue;
		GMatrix& cm = *matrixArray[i];

		// Print attribute name
		oss.precision(9);
		oss.str("");
		oss << "Confusion matrix for ";
		pRelation->printAttrName(oss, i);
		string s = oss.str();
		leftJustifiedString(s.c_str(), buf, 40);
		cout << buf;

		// Print column numbers
		for(size_t j = 0; j < cm.cols(); j++)
		{
			oss.str("");
			pRelation->printAttrValue(oss, i, (double)j);
			s = oss.str();
			rightJustifiedString(s.c_str(), buf, 12);
			cout << buf;
		}
		cout << "\n";

		// Print the confusion matrix values
		for(size_t k = 0; k < cm.rows(); k++)
		{
			oss.str("");
			pRelation->printAttrValue(oss, i, (double)k);
			s = oss.str();
			rightJustifiedString(s.c_str(), buf, 40);
			cout << buf;
			for(size_t j = 0; j < cm.cols(); j++)
			{
				oss.str("");
				oss << cm[k][j];
				s = oss.str();
				rightJustifiedString(s.c_str(), buf, 12);
				cout << buf;
			}
			cout << "\n";
		}
		cout << "\n";
	}
}

void Test(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool confusion = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		if(args.if_pop("-confusion"))
			confusion = true;
		else
			ThrowError("Invalid test option: ", args.peek());
	}

	// Load the model
	GRand prng(seed);
	GDom doc;
	if(args.size() < 1)
		ThrowError("Model not specified.");
	doc.loadJson(args.pop_string());
	GLearnerLoader ll(prng, true);
	GSupervisedLearner* pModeler = ll.loadSupervisedLearner(doc.root());
	Holder<GSupervisedLearner> hModeler(pModeler);

	// Load the data
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();
	if(pLabels->cols() != pModeler->labelDims())
		ThrowError("The model was trained with ", to_str(pModeler->labelDims()), " label dims, but the specified dataset has ", to_str(pLabels->cols()));

	// Test
	GTEMPBUF(double, results, pLabels->cols());
	vector<GMatrix*> confusionMatrices;
	pModeler->accuracy(*pFeatures, *pLabels, results, confusion ? &confusionMatrices : NULL);
	GVec::print(cout, 14, results, pLabels->cols());
	cout << "\n";

	// Print the confusion matrix
	if(confusion)
		printConfusionMatrices(pLabels->relation().get(), confusionMatrices);
}

void Transduce(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			ThrowError("Invalid transduce option: ", args.peek());
	}

	// Load the data sets
	GRand prng(seed);
	if(args.size() < 1)
		ThrowError("No labeled set specified.");

	// Load the labeled and unlabeled sets
	Holder<GMatrix> hFeatures1, hLabels1, hFeatures2, hLabels2;
	loadData(args, hFeatures1, hLabels1);
	loadData(args, hFeatures2, hLabels2);
	GMatrix* pFeatures1 = hFeatures1.get();
	GMatrix* pLabels1 = hLabels1.get();
	GMatrix* pFeatures2 = hFeatures2.get();
	GMatrix* pLabels2 = hLabels2.get();
	if(pFeatures1->cols() != pFeatures2->cols() || pLabels1->cols() != pLabels2->cols())
		ThrowError("The labeled and unlabeled datasets must have the same number of columns. (The labels in the unlabeled set are just place-holders, and will be overwritten.)");

	// Instantiate the modeler
	GTransducer* pSupLearner = InstantiateAlgorithm(prng, args, pFeatures1, pLabels1);
	Holder<GTransducer> hModel(pSupLearner);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Transduce
	GMatrix* pLabels3 = pSupLearner->transduce(*pFeatures1, *pLabels1, *pFeatures2);
	Holder<GMatrix> hLabels3(pLabels3);

	// Print results
	pLabels3->print(cout);
}

void TransductiveAccuracy(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool confusion = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-confusion"))
			confusion = true;
		else
			ThrowError("Invalid transacc option: ", args.peek());
	}

	// Load the data sets
	GRand prng(seed);
	Holder<GMatrix> hFeatures1, hLabels1, hFeatures2, hLabels2;
	loadData(args, hFeatures1, hLabels1);
	loadData(args, hFeatures2, hLabels2);
	GMatrix* pFeatures1 = hFeatures1.get();
	GMatrix* pLabels1 = hLabels1.get();
	GMatrix* pFeatures2 = hFeatures2.get();
	GMatrix* pLabels2 = hLabels2.get();
	if(pFeatures1->cols() != pFeatures2->cols() || pLabels1->cols() != pLabels2->cols())
		ThrowError("The training and test datasets must have the same number of columns.");

	// Instantiate the modeler
	GTransducer* pSupLearner = InstantiateAlgorithm(prng, args, pFeatures1, pLabels1);
	Holder<GTransducer> hModel(pSupLearner);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Transduce and measure accuracy
	GTEMPBUF(double, results, pLabels1->cols());
	vector<GMatrix*> confusionMatrices;
	pSupLearner->trainAndTest(*pFeatures1, *pLabels1, *pFeatures2, *pLabels2, results, confusion ? &confusionMatrices : NULL);

	// Print results
	GVec::print(cout, 14, results, pLabels1->cols());
	cout << "\n";

	// Print the confusion matrix
	if(confusion)
		printConfusionMatrices(pLabels2->relation().get(), confusionMatrices);
}

void SplitTest(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	double trainRatio = 0.5;
	size_t reps = 1;
	string lastModelFile="";
	bool confusion = false;
	bool should_show_standard_dev = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed")){
			seed = args.pop_uint();
		}else if(args.if_pop("-trainratio")){
			trainRatio = args.pop_double();
		}else if(args.if_pop("-reps")){
			reps = args.pop_uint();
		}else if(args.if_pop("-writelastmodel")){
			lastModelFile = args.pop_string();
		}else if(args.if_pop("-stddev")){
			should_show_standard_dev = true;
		}else if(args.if_pop("-confusion")){
			confusion = true;
		}else{
			ThrowError("Invalid splittest option: ", args.peek());
		}
	}
	if(trainRatio < 0 || trainRatio > 1)
		ThrowError("trainratio must be between 0 and 1");

	// Load the data
	GRand prng(seed);
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();

	// Instantiate the modeler
	GTransducer* pSupLearner = InstantiateAlgorithm(prng, args, pFeatures, pLabels);
	Holder<GTransducer> hModel(pSupLearner);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Ensure that can write if we are required to
	if(!pSupLearner->canGeneralize() && lastModelFile != ""){
	  ThrowError("The learner specified does not have an internal model "
		     "and thus cannot be saved to a file.  Remove the "
		     "-lastModelFile argument.");
	}

	// Do the reps
	size_t trainingPatterns = std::max((size_t)1, std::min(pFeatures->rows() - 1, (size_t)floor(pFeatures->rows() * trainRatio + 0.5)));
	size_t testPatterns = pFeatures->rows() - trainingPatterns;
	// Results is the mean results for all columns (plus some extra
	// storage to make allocation and deallocation easier)
	GTEMPBUF(double, results, 6 * pLabels->cols());
	// The result of a single repetition
	double* repResults = results + pLabels->cols();
	// resultsV (for variance, which quantity it resembles) and
	// oldResults are temporary variables used in calculating the
	// standard deviation incrementally in a way robust to
	// round-off error.  The algorithm is from
	// http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html
	// and they cite Knuth "The Art of Computer Programming,
	// Volume 2: Seminumerical Algorithms", section 4.2.2
	double* resultsV = results + 2 * pLabels->cols();
	double* oldResults = results + 3 * pLabels->cols();
	double* tempR1 = results + 4 * pLabels->cols();
	double* tempR2 = results + 5 * pLabels->cols();
	GVec::setAll(results, 0, 6 * pLabels->cols());
	for(size_t i = 0; i < reps; i++)
	{
		// Shuffle and split the data
		pFeatures->shuffle(prng, pLabels);
		GMatrix testFeatures(pFeatures->relation(), pFeatures->heap());
		GMatrix testLabels(pLabels->relation(), pLabels->heap());
		{
			GMergeDataHolder hFeatures(*pFeatures, testFeatures);
			GMergeDataHolder hLabels(*pLabels, testLabels);
			testFeatures.reserve(testPatterns);
			testLabels.reserve(testPatterns);
			pFeatures->splitBySize(&testFeatures, testPatterns);
			pLabels->splitBySize(&testLabels, testPatterns);

			// Test and print results
			vector<GMatrix*> confusionMatrices;
			pSupLearner->trainAndTest(*pFeatures, *pLabels, testFeatures, testLabels, repResults, confusion ? &confusionMatrices : NULL);

			// Write trained model file on last repetition
			if(lastModelFile != "" && i+1 == reps){
				GSupervisedLearner* pSup = 
				  dynamic_cast<GSupervisedLearner*>
				  (pSupLearner);
				GDom doc;
				GDomNode* pRoot = pSup->serialize(&doc);
				doc.setRoot(pRoot);
				std::ofstream out(lastModelFile.c_str());
				if(out){
					doc.writeJson(out);
				}
			}
			cout << "rep " << i << ") ";
			GVec::print(cout, 14, repResults, pLabels->cols());
			cout << "\n";
			if (i>0){
				GVec::copy(oldResults, results, pLabels->cols());
			}
			double weight = 1.0 / (i + 1);
			GVec::multiply(results, 1.0 - weight, pLabels->cols());
			GVec::addScaled(results, weight, repResults, pLabels->cols());

			//Calculate the recurrence s(k)=s(k-1) + (x(k) - M(k-1)) * (x(k) - M(k))
			//where s = resultsV, x=repResults, M = results, M(k-1) = oldResults
			if (i>0){
				GVec::copy(tempR1, repResults, pLabels->cols());
				GVec::copy(tempR2, repResults, pLabels->cols());
				GVec::subtract(tempR1, oldResults, pLabels->cols());
				GVec::subtract(tempR2, results, pLabels->cols());
				GVec::pairwiseMultiply(tempR1, tempR2, pLabels->cols());
				GVec::add(resultsV, tempR1, pLabels->cols());
			}

			// Print the confusion matrix (if specified)
			if(confusion)
				printConfusionMatrices(pLabels->relation().get(), confusionMatrices);
		}
	}
	if(pLabels->cols() > 1){
	  cout << "-----Means-----\n";
	}else{
	  cout << "-----Mean-----\n";
	}
	GVec::print(cout, 14, results, pLabels->cols());
	cout << "\n";

	if(should_show_standard_dev){
		if(pLabels->cols() > 1){
			cout << "-----Standard Deviations-----\n";
		}else{
			cout << "-----Standard Deviation-----\n";
		}
		
		if(reps > 1){
			GVec::multiply(resultsV, 1.0/(reps-1), pLabels->cols());
		}
		GVec::pow(resultsV, 0.5, pLabels->cols());
		GVec::print(cout, 14, resultsV, pLabels->cols());
		cout << "\n";
	}
}

void CrossValidateCallback(void* pSupLearner, size_t nRep, size_t nFold, size_t labelDims, double* pFoldResults)
{
	cout << "Rep: " << nRep << ", Fold: " << nFold <<", Accuracy: ";
	GVec::print(cout, 14, pFoldResults, labelDims);
	cout << "\n";
}

void CrossValidate(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	int reps = 5;
	int folds = 2;
	bool succinct = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-reps"))
			reps = args.pop_uint();
		else if(args.if_pop("-folds"))
			folds = args.pop_uint();
		else if(args.if_pop("-succinct"))
			succinct = true;
		else
			ThrowError("Invalid crossvalidate option: ", args.peek());
	}
	if(reps < 1)
		ThrowError("There must be at least 1 rep.");
	if(folds < 2)
		ThrowError("There must be at least 2 folds.");

	// Load the data
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();

	// Instantiate the modeler
	GRand prng(seed);
	GTransducer* pSupLearner = InstantiateAlgorithm(prng, args, pFeatures, pLabels);
	Holder<GTransducer> hModel(pSupLearner);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Test
	cout.precision(8);
	GMatrix* pResults = pSupLearner->repValidate(*pFeatures, *pLabels, reps, folds, succinct ? NULL : CrossValidateCallback, pSupLearner);
	Holder<GMatrix> hResults(pResults);
	if(!succinct)
		cout << "-----\n";
	for(size_t i = 0; i < pLabels->cols(); i++)
	{
		double mean = pResults->mean(i);
		double variance = pResults->variance(i, mean);
		if(!succinct)
		{
			cout << "Attr: " << (pFeatures->cols() + i);
			if(pLabels->relation()->valueCount(i) == 0)
				cout << ", Mean squared error: ";
			else
				cout << ", Mean predictive accuracy: ";
		}
		cout << mean;
		if(succinct)
		{
			if(i + 1 < pLabels->cols())
				cout << ", ";
		}
		else
			cout << ", Deviation: " << sqrt(variance) << "\n";
	}
	cout << "\n";
}

void vette(string& s)
{
	for(size_t i = 0; i < s.length(); i++)
	{
		if(s[i] <= ' ' || s[i] == '\'' || s[i] == '"')
			s[i] = '_';
	}
}

void PrecisionRecall(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	int reps = 5;
	int samples = 100;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-reps"))
			reps = args.pop_uint();
		else if(args.if_pop("-samples"))
			samples = args.pop_uint();
		else
			ThrowError("Invalid precisionrecall option: ", args.peek());
	}
	if(reps < 1)
		ThrowError("There must be at least 1 rep.");
	if(samples < 2)
		ThrowError("There must be at least 2 samples.");

	// Load the data
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();

	// Instantiate the modeler
	GRand prng(seed);
	GTransducer* pSupLearner = InstantiateAlgorithm(prng, args, pFeatures, pLabels);
	Holder<GTransducer> hModel(pSupLearner);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());
	if(!pSupLearner->canGeneralize())
		ThrowError("This algorithm cannot be \"trained\". It can only be used to \"transduce\".");
	GSupervisedLearner* pModel = (GSupervisedLearner*)pSupLearner;

	// Build the relation for the results
	sp_relation pRelation;
	pRelation = new GArffRelation();
	((GArffRelation*)pRelation.get())->setName("untitled");
	GArffRelation* pRel = (GArffRelation*)pRelation.get();
	pRel->addAttribute("recall", 0, NULL);
	for(size_t i = 0; i < pLabels->cols(); i++)
	{
		size_t valCount = std::max((size_t)1, pLabels->relation()->valueCount(i));
		for(int val = 0; val < (int)valCount; val++)
		{
			string s = "precision_";
			if(pLabels->relation()->type() == GRelation::ARFF)
				s += ((GArffRelation*)pLabels->relation().get())->attrName(i);
			else
			{
				s += "attr";
				s += to_str(i);
			}
			if(valCount > 1)
			{
				s += "_";
				ostringstream oss;
				pLabels->relation()->printAttrValue(oss, i, val);
				s += oss.str();
			}
			vette(s);
			pRel->addAttribute(s.c_str(), 0, NULL);
		}
	}

	// Measure precision/recall
	GMatrix results(pRelation);
	results.newRows(samples);
	for(int i = 0; i < samples; i++)
		results.row(i)[0] = (double)i / samples;
	size_t pos = 1;
	for(size_t i = 0; i < pLabels->cols(); i++)
	{
		size_t valCount = std::max((size_t)1, pLabels->relation()->valueCount(i));
		double* precision = new double[valCount * samples];
		ArrayHolder<double> hPrecision(precision);
		pModel->precisionRecall(precision, samples, *pFeatures, *pLabels, i, reps);
		for(size_t j = 0; j < valCount; j++)
			results.setCol(pos++, precision + samples * j);
	}
	GAssert(pos == pRelation->size()); // counting problem
	results.print(cout);
}

void sterilize(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	size_t folds = 10;
	double diffThresh = 0.1;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-folds"))
			folds = args.pop_uint();
		else if(args.if_pop("-diffthresh"))
			diffThresh = args.pop_double();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the data
	Holder<GMatrix> hFeatures, hLabels;
	loadData(args, hFeatures, hLabels);
	GMatrix* pFeatures = hFeatures.get();
	GMatrix* pLabels = hLabels.get();

	// Instantiate the modeler
	GRand prng(seed);
	GTransducer* pTransducer = InstantiateAlgorithm(prng, args, pFeatures, pLabels);
	Holder<GTransducer> hModel(pTransducer);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());

	// Sterilize
	GMatrix sterileFeatures(pFeatures->relation());
	GReleaseDataHolder hSterileFeatures(&sterileFeatures);
	GMatrix sterileLabels(pLabels->relation());
	GReleaseDataHolder hSterileLabels(&sterileLabels);
	for(size_t fold = 0; fold < folds; fold++)
	{
		// Split the data
		GMatrix trainFeatures(pFeatures->relation());
		GReleaseDataHolder hTrainFeatures(&trainFeatures);
		GMatrix trainLabels(pLabels->relation());
		GReleaseDataHolder hTrainLabels(&trainLabels);
		GMatrix testFeatures(pFeatures->relation());
		GReleaseDataHolder hTestFeatures(&testFeatures);
		GMatrix testLabels(pLabels->relation());
		GReleaseDataHolder hTestLabels(&testLabels);
		size_t foldBegin = fold * pFeatures->rows() / folds;
		size_t foldEnd = (fold + 1) * pFeatures->rows() / folds;
		for(size_t i = 0; i < foldBegin; i++)
		{
			trainFeatures.takeRow(pFeatures->row(i));
			trainLabels.takeRow(pLabels->row(i));
		}
		for(size_t i = foldBegin; i < foldEnd; i++)
		{
			testFeatures.takeRow(pFeatures->row(i));
			testLabels.takeRow(pLabels->row(i));
		}
		for(size_t i = foldEnd; i < pFeatures->rows(); i++)
		{
			trainFeatures.takeRow(pFeatures->row(i));
			trainLabels.takeRow(pLabels->row(i));
		}

		// Transduce
		GMatrix* pPredictedLabels = pTransducer->transduce(trainFeatures, trainLabels, testFeatures);
		Holder<GMatrix> hPredictedLabels(pPredictedLabels);

		// Keep only the correct predictions
		for(size_t j = 0; j < testLabels.rows(); j++)
		{
			double* pTarget = testLabels[j];
			double* pPredicted = pPredictedLabels->row(j);
			for(size_t i = 0; i < testLabels.cols(); i++)
			{
				size_t vals = testLabels.relation()->valueCount(i);
				bool goodEnough = false;
				if(vals == 0)
				{
					if(std::abs(*pTarget - *pPredicted) < diffThresh)
						goodEnough = true;
				}
				else
				{
					if(*pTarget == *pPredicted)
						goodEnough = true;
				}
				if(goodEnough)
				{
					sterileFeatures.takeRow(testFeatures[j]);
					sterileLabels.takeRow(testLabels[j]);
				}
				pTarget++;
				pPredicted++;
			}
		}
	}

	// Merge the sterile features and labels
	GMatrix* pSterile = GMatrix::mergeHoriz(&sterileFeatures, &sterileLabels);
	Holder<GMatrix> hSterile(pSterile);
	pSterile->print(cout);
}

class MyRecurrentModel : public GRecurrentModel
{
protected:
	const char* m_stateFilename;
	double m_validateInterval;
	double m_dStart;

public:
	MyRecurrentModel(GSupervisedLearner* pTransition, GSupervisedLearner* pObservation, size_t actionDims, size_t contextDims, size_t obsDims, GRand* pRand, std::vector<size_t>* pParamDims, const char* stateFilename, double validateInterval)
	: GRecurrentModel(pTransition, pObservation, actionDims, contextDims, obsDims, pRand, pParamDims), m_stateFilename(stateFilename), m_validateInterval(validateInterval)
	{
		m_dStart = GTime::seconds();
	}

	virtual ~MyRecurrentModel()
	{
	}

	virtual void onFinishedComputingStateEstimate(GMatrix* pStateEstimate)
	{
		if(m_stateFilename)
			pStateEstimate->saveArff(m_stateFilename);
		cout << "% Computed state estimate in " << GTime::seconds() - m_dStart << " seconds.\n";
		cout.flush();
	}

	virtual void onObtainValidationScore(int timeSlice, double seconds, double squaredError)
	{
		if(m_validateInterval > 0)
		{
			if(squaredError == UNKNOWN_REAL_VALUE)
				cout << (m_validateInterval * timeSlice) << ", ?\n";
			else
				cout << (m_validateInterval * timeSlice) << ", " << sqrt(squaredError) << "\n";
			cout.flush();
		}
	}
};

void trainRecurrent(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	vector<size_t> paramDims;
	const char* stateFilename = NULL;
	double validationInterval = 0;
	vector<string> validationFilenames;
	const char* outFilename = "model.json";
	double trainTime = 60 * 60; // 1 hour
	bool useIsomap = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-paramdims"))
		{
			unsigned int count = args.pop_uint();
			for(unsigned int i = 0; i < count; i++)
				paramDims.push_back(args.pop_uint());
		}
		else if(args.if_pop("-state"))
			stateFilename = args.pop_string();
		else if(args.if_pop("-validate"))
		{
			validationInterval = args.pop_double();
			int count = args.pop_uint();
			for(int i = 0; i < count; i++)
			{
				validationFilenames.push_back(args.pop_string());
				validationFilenames.push_back(args.pop_string());
			}
		}
		else if(args.if_pop("-out"))
			outFilename = args.pop_string();
		else if(args.if_pop("-traintime"))
			trainTime = args.pop_double();
		else if(args.if_pop("-isomap"))
			useIsomap = true;
		else
			ThrowError("Invalid trainRecurrent option: ", args.peek());
	}

	// Parse the algorithm
	const char* alg = args.pop_string();
	int bpttDepth = 0;
	int bpttItersPerGrow = 0;
	double annealDeviation = 0.0;
	double annealDecay = 0.0;
	double annealTimeWindow = 0.0;
	if(strcmp(alg, "moses") == 0)
	{
	}
	else if(strcmp(alg, "aaron") == 0)
	{
	}
	else if(strcmp(alg, "joshua") == 0)
	{
	}
	else if(strcmp(alg, "bptt") == 0)
	{
		bpttDepth = args.pop_uint();
		bpttItersPerGrow = args.pop_uint();
	}
	else if(strcmp(alg, "bpttcal") == 0)
	{
		bpttDepth = args.pop_uint();
		bpttItersPerGrow = args.pop_uint();
	}
	else if(strcmp(alg, "evolutionary") == 0)
	{
	}
	else if(strcmp(alg, "hillclimber") == 0)
	{
	}
	else if(strcmp(alg, "annealing") == 0)
	{
		annealDeviation = args.pop_double();
		annealDecay = args.pop_double();
		annealTimeWindow = args.pop_double();
	}
	else
		ThrowError("Unrecognized recurrent model training algorithm: ", alg);

	// Load the data
	GMatrix* pDataObs = GMatrix::loadArff(args.pop_string());
	Holder<GMatrix> hDataObs(pDataObs);
	GMatrix* pDataAction = GMatrix::loadArff(args.pop_string());
	Holder<GMatrix> hDataAction(pDataAction);

	// Get the number of context dims
	int contextDims = args.pop_uint();

	// Infer remaining values and check that the parts fit together
	size_t pixels = 1;
	for(vector<size_t>::iterator it = paramDims.begin(); it != paramDims.end(); it++)
		pixels *= *it;
	size_t channels = pDataObs->cols() / pixels;
	if((channels * pixels) != pDataObs->cols())
		ThrowError("The number of columns in the observation data must be a multiple of the product of the param dims");

	// Instantiate the recurrent model
	GRand prng(seed);
	GTransducer* pTransitionFunc = InstantiateAlgorithm(prng, args, NULL, NULL);
	Holder<GTransducer> hTransitionFunc(pTransitionFunc);
	if(!pTransitionFunc->canGeneralize())
		ThrowError("The algorithm specified for the transition function cannot be \"trained\". It can only be used to \"transduce\".");
	GTransducer* pObservationFunc = InstantiateAlgorithm(prng, args, NULL, NULL);
	Holder<GTransducer> hObservationFunc(pObservationFunc);
	if(!pObservationFunc->canGeneralize())
		ThrowError("The algorithm specified for the observation function cannot be \"trained\". It can only be used to \"transduce\".");
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());
	MyRecurrentModel model((GSupervisedLearner*)hTransitionFunc.release(), (GSupervisedLearner*)hObservationFunc.release(), pDataAction->cols(), contextDims, pDataObs->cols(), &prng, &paramDims, stateFilename, validationInterval);

	// Set it up to do validation during training if specified
	vector<GMatrix*> validationData;
	VectorOfPointersHolder<GMatrix> hValidationData(validationData);
	if(validationInterval > 0)
	{
		for(size_t i = 0; i < validationFilenames.size(); i++)
			validationData.push_back(GMatrix::loadArff(validationFilenames[i].c_str()));
		model.validateDuringTraining(validationInterval, &validationData);
		cout << "@RELATION validation_scores\n\n@ATTRIBUTE seconds real\n@ATTRIBUTE " << alg << " real\n\n@DATA\n";
	}

	// Set other flags
	model.setTrainingSeconds(trainTime);
	model.setUseIsomap(useIsomap);

	// Do the training
	if(strcmp(alg, "moses") == 0)
		model.trainMoses(pDataAction, pDataObs);
	else if(strcmp(alg, "aaron") == 0)
		model.trainAaron(pDataAction, pDataObs);
	else if(strcmp(alg, "joshua") == 0)
		model.trainJoshua(pDataAction, pDataObs);
	else if(strcmp(alg, "bptt") == 0)
		model.trainBackPropThroughTime(pDataAction, pDataObs, bpttDepth, bpttItersPerGrow);
	else if(strcmp(alg, "evolutionary") == 0)
		model.trainEvolutionary(pDataAction, pDataObs);
	else if(strcmp(alg, "hillclimber") == 0)
		model.trainHillClimber(pDataAction, pDataObs, 0.0, 0.0, 0.0, true, false);
	else if(strcmp(alg, "annealing") == 0)
		model.trainHillClimber(pDataAction, pDataObs, annealDeviation, annealDecay, annealTimeWindow, false, true);
	GDom doc;
	doc.setRoot(model.serialize(&doc));
	doc.saveJson(outFilename);
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeLearnUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(cout, 0, 3, 76, 1000, true);
	UsageNode* pUsageTree2 = makeAlgorithmUsageTree();
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
	UsageNode* pUsageTree = makeLearnUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << szAppName << " ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, true);
			if(pUsageCommand->findPart("[algorithm]") >= 0)
			{
				UsageNode* pAlgTree = makeAlgorithmUsageTree();
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
			else if(args.if_pop("autotune"))
				autoTune(args);
			else if(args.if_pop("train"))
				Train(args);
			else if(args.if_pop("test"))
				Test(args);
			else if(args.if_pop("predict"))
				predict(args);
			else if(args.if_pop("predictdistribution"))
				predictDistribution(args);
			else if(args.if_pop("transduce"))
				Transduce(args);
			else if(args.if_pop("transacc"))
				TransductiveAccuracy(args);
			else if(args.if_pop("splittest"))
				SplitTest(args);
			else if(args.if_pop("crossvalidate"))
				CrossValidate(args);
			else if(args.if_pop("precisionrecall"))
				PrecisionRecall(args);
 			else if(args.if_pop("sterilize"))
 				sterilize(args);
			else if(args.if_pop("trainrecurrent"))
				trainRecurrent(args);
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
