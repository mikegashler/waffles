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
#include "../GClasses/GDistance.h"
#include "../GClasses/GDistribution.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GImage.h"
#include "../GClasses/GKNN.h"
#include "../GClasses/GLinear.h"
#include "../GClasses/GError.h"
#include "../GClasses/GManifold.h"
#include "../GClasses/GNaiveBayes.h"
#include "../GClasses/GNaiveInstance.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GSparseMatrix.h"
#include "../GClasses/GHtml.h"
#include "../GClasses/GText.h"
#include "../GSup/GDirList.h"
#include "../GClasses/GSystemLearner.h"
#include "../GClasses/GTime.h"
#include "../GClasses/GTransform.h"
#include "../GClasses/GTwt.h"
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


GTransducer* InstantiateAlgorithm(GRand* pRand, GArgReader& args);

GBaselineLearner* InstantiateBaseline(GRand* pRand, GArgReader& args)
{
	GBaselineLearner* pModel = new GBaselineLearner();
	return pModel;
}

GKNN* InstantiateKNN(GRand* pRand, GArgReader& args)
{
	if(args.size() < 1)
		ThrowError("The number of neighbors must be specified for knn");
	int neighborCount = args.pop_uint();
	GKNN* pModel = new GKNN(neighborCount, pRand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-equalweight"))
			pModel->setInterpolationMethod(GKNN::Mean);
		else if(args.if_pop("-scalefeatures"))
			pModel->setOptimizeScaleFactors(true);
		else if(args.if_pop("-cosine"))
			pModel->setMetric(new GCosineSimilarity(), true);
		else if(args.if_pop("-pearson"))
			pModel->setMetric(new GPearsonCorrelation(), true);
		else
			ThrowError("Invalid knn option: ", args.peek());
	}
	return pModel;
}

GLinearRegressor* InstantiateLinearRegressor(GRand* pRand, GArgReader& args)
{
	GLinearRegressor* pModel = new GLinearRegressor(pRand);
	return pModel;
}

GNaiveBayes* InstantiateNaiveBayes(GRand* pRand, GArgReader& args)
{
	GNaiveBayes* pModel = new GNaiveBayes(pRand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-ess"))
			pModel->setEquivalentSampleSize(args.pop_double());
		else
			ThrowError("Invalid naivebayes option: ", args.peek());
	}
	return pModel;
}

GNaiveInstance* InstantiateNaiveInstance(GRand* pRand, GArgReader& args)
{
	if(args.size() < 1)
		ThrowError("The number of neighbors must be specified for naiveinstance");
	int neighborCount = args.pop_uint();
	GNaiveInstance* pModel = new GNaiveInstance(neighborCount);
	return pModel;
}

GNeuralNet* InstantiateNeuralNet(GRand* pRand, GArgReader& args)
{
	GNeuralNet* pModel = new GNeuralNet(pRand);
	while(args.next_is_flag())
	{
		if(args.if_pop("-addlayer"))
			pModel->addLayer(args.pop_uint());
		else if(args.if_pop("-learningrate"))
			pModel->setLearningRate(args.pop_double());
		else if(args.if_pop("-momentum"))
			pModel->setMomentum(args.pop_double());
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
		else
			ThrowError("Invalid neuralnet option: ", args.peek());
	}
	return pModel;
}
/*
GNeuralTransducer* InstantiateNeuralTransducer(GRand* pRand, GArgReader& args)
{
	GNeuralTransducer* pTransducer = new GNeuralTransducer(pRand);
	vector<size_t> paramDims;
	while(args.next_is_flag())
	{
		if(args.if_pop("-addlayer"))
			pTransducer->neuralNet()->addLayer(args.pop_uint());
		else if(args.if_pop("-params"))
		{
			size_t count = args.pop_uint();
			for(size_t i = 0; i < count; i++)
				paramDims.push_back(args.pop_uint());
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}
	pTransducer->setParams(paramDims);
	return pTransducer;
}
*/
void showInstantiateAlgorithmError(const char* szMessage, GArgReader& args)
{
	cout << "_________________________________\n";
	cout << szMessage << "\n\n";
	const char* szAlgName = args.peek();
	UsageNode* pAlgTree = makeAlgorithmUsageTree();
	Holder<UsageNode> hAlgTree(pAlgTree);
	if(szAlgName)
	{
		UsageNode* pUsageAlg = pAlgTree->choice(szAlgName);
		if(pUsageAlg)
		{
			cout << "Partial Usage Information:\n\n";
			pUsageAlg->print(0, 3, 76, 1000, true);
		}
		else
		{
			cout << "\"" << szAlgName << "\" is not a recognized algorithm. Try one of these:\n\n";
			pAlgTree->print(0, 3, 76, 1, false);
		}
	}
	else
	{
		cout << "Expected an algorithm. Here are some choices:\n";
		pAlgTree->print(0, 3, 76, 1, false);
	}
	cout << "\nTo see full usage information, run:\n	waffles_learn usage\n\n";
	cout << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cout.flush();
}

GTransducer* InstantiateAlgorithm(GRand* pRand, GArgReader& args)
{
	int argPos = args.get_pos();
	if(args.size() < 1)
		ThrowError("No algorithm specified.");
	try
	{
		if(args.if_pop("baseline"))
			return InstantiateBaseline(pRand, args);
		else if(args.if_pop("knn"))
			return InstantiateKNN(pRand, args);
		else if(args.if_pop("linear"))
			return InstantiateLinearRegressor(pRand, args);
		else if(args.if_pop("naivebayes"))
			return InstantiateNaiveBayes(pRand, args);
//		else if(args.if_pop("naiveinstance"))
//			return InstantiateNaiveInstance(pRand, args);
		else if(args.if_pop("neuralnet"))
			return InstantiateNeuralNet(pRand, args);
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

void train(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			ThrowError("Invalid trainsparse option: ", args.peek());
	}

	// Load the sparse features
	if(args.size() < 1)
		ThrowError("Expected a filename of a sparse matrix.");
	GSparseMatrix* pSparseFeatures;
	Holder<GSparseMatrix> hSparseFeatures(NULL);
	{
		GTwtDoc doc;
		doc.load(args.pop_string());
		pSparseFeatures = new GSparseMatrix(doc.root());
		hSparseFeatures.reset(pSparseFeatures);
	}

	// Load the dense labels
	GMatrix* pLabels = GMatrix::loadArff(args.pop_string());
	Holder<GMatrix> hLabels(pLabels);

	// Instantiate the modeler
	GRand prng(seed);
	GTransducer* pSupLearner = InstantiateAlgorithm(&prng, args);
	Holder<GTransducer> hModel(pSupLearner);
	if(args.size() > 0)
		ThrowError("Superfluous argument: ", args.peek());
	if(!pSupLearner->canTrainIncrementally())
		ThrowError("This algorithm cannot be trained with a sparse matrix. Only incremental learners (such as naivebayes, knn, and neuralnet) support this functionality.");
	GIncrementalLearner* pModel = (GIncrementalLearner*)pSupLearner;

	// Train the modeler
	pModel->trainSparse(*pSparseFeatures, *pLabels);

	// Output the trained model
	GTwtDoc doc;
	GTwtNode* pRoot = pModel->toTwt(&doc);
	doc.setRoot(pRoot);
	doc.write(cout);
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
			ThrowError("Invalid predictsparse option: ", args.peek());
	}

	// Load the model
	GRand prng(seed);
	GTwtDoc doc;
	if(args.size() < 1)
		ThrowError("Model not specified.");
	doc.load(args.pop_string());
	GLearnerLoader ll(true);
	GSupervisedLearner* pModeler = ll.loadModeler(doc.root(), &prng);
	Holder<GSupervisedLearner> hModeler(pModeler);

	// Load the sparse features
	if(args.size() < 1)
		ThrowError("No dataset specified.");
	GSparseMatrix* pData;
	Holder<GSparseMatrix> hData(NULL);
	{
		GTwtDoc doc;
		doc.load(args.pop_string());
		pData = new GSparseMatrix(doc.root());
		hData.reset(pData);
	}

	// Predict labels
	GMatrix labels(pData->rows(), pModeler->labelDims());
	double* pFullRow = new double[pData->cols()];
	ArrayHolder<double> hFullRow(pFullRow);
	for(unsigned int i = 0; i < pData->rows(); i++)
	{
		pData->fullRow(pFullRow, i);
		pModeler->predict(pFullRow, labels[i]);
	}
	labels.print(cout);
}

void test(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			ThrowError("Invalid predictsparse option: ", args.peek());
	}

	// Load the model
	GRand prng(seed);
	GTwtDoc doc;
	if(args.size() < 1)
		ThrowError("Model not specified.");
	doc.load(args.pop_string());
	GLearnerLoader ll(true);
	GSupervisedLearner* pModeler = ll.loadModeler(doc.root(), &prng);
	Holder<GSupervisedLearner> hModeler(pModeler);

	// Load the sparse features
	if(args.size() < 1)
		ThrowError("No dataset specified.");
	GSparseMatrix* pData;
	Holder<GSparseMatrix> hData(NULL);
	{
		GTwtDoc doc;
		doc.load(args.pop_string());
		pData = new GSparseMatrix(doc.root());
		hData.reset(pData);
	}

	// Load the dense labels
	GMatrix* pLabels = GMatrix::loadArff(args.pop_string());
	Holder<GMatrix> hLabels(pLabels);
	if(pLabels->cols() != pModeler->labelDims())
		ThrowError("The model was trained to predict a different number of label dims");

	// Test
	GTEMPBUF(double, prediction, pLabels->cols());
	double* pFullRow = new double[pData->cols()];
	ArrayHolder<double> hFullRow(pFullRow);
	GTEMPBUF(double, results, pLabels->cols());
	GVec::setAll(results, 0.0, pLabels->cols());
	for(size_t i = 0; i < pData->rows(); i++)
	{
		pData->fullRow(pFullRow, i);
		pModeler->predict(pFullRow, prediction);
		double* pTarget = pLabels->row(i);
		for(size_t j = 0; j < pLabels->cols(); j++)
		{
			if(pLabels->relation()->valueCount(j) == 0)
			{
				double d = pTarget[j] - prediction[j];
				results[j] += (d * d);
			}
			else
			{
				if((int)prediction[j] == (int)pTarget[j])
					results[j]++;
			}
		}
	}
	GVec::multiply(results, 1.0 / pData->rows(), pLabels->cols());
	GVec::print(cout, 14, results, pLabels->cols());
}

class MyHtmlParser1 : public GHtml
{
protected:
	GVocabulary* m_pVocab;

public:
	MyHtmlParser1(GVocabulary* pVocab, const char* pDoc, size_t nSize)
	: GHtml(pDoc, nSize), m_pVocab(pVocab)
	{
	}

	virtual ~MyHtmlParser1() {}

	virtual void onTextChunk(const char* pChunk, size_t chunkSize)
	{
		m_pVocab->addWordsFromTextBlock(pChunk, chunkSize);
	}
};

class MyHtmlParser2 : public GHtml
{
protected:
	GSparseMatrix* m_pSM;
	size_t m_row;
	GVocabulary* m_pVocab;
	bool m_binary;

public:
	MyHtmlParser2(const char* pDoc, size_t nSize, GSparseMatrix* pSM, size_t row, GVocabulary* pVocab, bool binary)
	: GHtml(pDoc, nSize), m_pSM(pSM), m_row(row), m_pVocab(pVocab), m_binary(binary)
	{
	}

	virtual ~MyHtmlParser2() {}

	virtual void onTextChunk(const char* pChunk, size_t chunkSize)
	{
		GWordIterator it(pChunk, chunkSize);
		const char* pWord;
		size_t wordLen;
		while(true)
		{
			if(!it.next(&pWord, &wordLen))
				break;
			size_t col = m_pVocab->wordIndex(pWord, wordLen);
			if(col != INVALID_INDEX)
			{
				if(m_binary)
					m_pSM->set(m_row, col, 1.0);
				else
					m_pSM->set(m_row, col, m_pSM->get(m_row, col) + m_pVocab->weight(col));
			}
		}
	}
};

void addWordsToVocabFromHtmlFile(GVocabulary* pVocab, const char* szFilename)
{
	size_t len;
	char* pFile = GFile::loadFile(szFilename, &len);
	ArrayHolder<char> hFile(pFile);
	pVocab->newDoc();
	MyHtmlParser1 parser(pVocab, pFile, len);
	while(true)
	{
		if(!parser.parseSomeMore())
			break;
	}
}

void makeHtmlFileVector(GSparseMatrix* pFeatures, GMatrix* pLabels, int clss, size_t row, GVocabulary* pVocab, const char* szFilename, bool binary)
{
	size_t len;
	char* pFile = GFile::loadFile(szFilename, &len);
	ArrayHolder<char> hFile(pFile);
	MyHtmlParser2 parser(pFile, len, pFeatures, row, pVocab, binary);
	while(true)
	{
		if(!parser.parseSomeMore())
			break;
	}
	if(clss > 0)
		pLabels->row(row)[0] = (double)clss;
}

void addWordsToVocabFromTextFile(GVocabulary* pVocab, const char* szFilename)
{
	size_t len;
	char* pFile = GFile::loadFile(szFilename, &len);
	ArrayHolder<char> hFile(pFile);
	pVocab->newDoc();
	pVocab->addWordsFromTextBlock(pFile, len);
}

void makeTextFileVector(GSparseMatrix* pFeatures, GMatrix* pLabels, int clss, size_t row, GVocabulary* pVocab, const char* szFilename, bool binary)
{
	size_t len;
	char* pFile = GFile::loadFile(szFilename, &len);
	ArrayHolder<char> hFile(pFile);
	GWordIterator it(pFile, len);
	const char* pWord;
	size_t wordLen;
	while(true)
	{
		if(!it.next(&pWord, &wordLen))
			break;
		size_t col = pVocab->wordIndex(pWord, wordLen);
		if(col != INVALID_INDEX)
		{
			if(binary)
				pFeatures->set(row, col, 1.0);
			else
				pFeatures->set(row, col, pFeatures->get(row, col) + pVocab->weight(col));
		}
	}
	if(pLabels)
		pLabels->row(row)[0] = (double)clss;
}

void docsToSparseMatrix(GArgReader& args)
{
	// Parse options
	bool useStemmer = true;
	bool binary = false;
	string featuresFilename = "features.sparse";
	string labelsFilename = "labels.arff";
	string vocabFile = "";
	while(args.next_is_flag())
	{
		if(args.if_pop("-nostem"))
			useStemmer = false;
		else if(args.if_pop("-binary"))
			binary = true;
		else if(args.if_pop("-out"))
		{
			featuresFilename = args.pop_string();
			labelsFilename = args.pop_string();
		}
		else if(args.if_pop("-vocabfile"))
			vocabFile = args.pop_string();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Parse the vocabulary
	GVocabulary vocab(useStemmer);
	vocab.addTypicalStopWords();
	vector<string> folders;
	while(args.size() > 0)
	{
		const char* szFolder = args.pop_string();
		folders.push_back(szFolder);
		char cwd[300];
		if(!getcwd(cwd, 300))
			ThrowError("Failed to get cwd");
		if(chdir(szFolder) != 0)
			ThrowError("Failed to change directory to: ", szFolder, ", from: ", cwd);
		{
			GDirList dl(false/*recurseSubDirs*/, true/*reportFiles*/, false/*reportDirs*/, false/*reportPaths*/);
			while(true)
			{
				const char* filename = dl.GetNext();
				if(!filename)
					break;
				PathData pd;
				GFile::parsePath(filename, &pd);
				if(_stricmp(filename + pd.extStart, ".txt") == 0)
					addWordsToVocabFromTextFile(&vocab, filename);
				else if(_stricmp(filename + pd.extStart, ".html") == 0 || _stricmp(filename + pd.extStart, ".htm") == 0)
					addWordsToVocabFromHtmlFile(&vocab, filename);
				else
					printf("Skipping file: %s. (Only .txt and .html is supported.)\n", filename);
			}
		}
		if(chdir(cwd) != 0)
			ThrowError("failed to change dir");
	}
	if(folders.size() == 0)
		ThrowError("At least one folder name must be specified");
	printf("-----\n");

	// Make the sparse feature matrix and the label matrix
	GSparseMatrix sparseFeatures(vocab.docCount(), vocab.wordCount());
	GMatrix* pLabels = NULL;
	if(folders.size() > 1)
	{
		vector<size_t> classes;
		classes.push_back(folders.size());
		pLabels = new GMatrix(classes);
		pLabels->newRows(vocab.docCount());
	}
	Holder<GMatrix> hLabels(pLabels);
	size_t row = 0;
	for(int clss = 0; clss < (int)folders.size(); clss++)
	{
		const char* szFolder = folders[clss].c_str();
		char cwd[300];
		if(!getcwd(cwd, 300))
			ThrowError("Failed to get cwd");
		if(chdir(szFolder) != 0)
			ThrowError("Failed to change directory to: ", szFolder, ", from: ", cwd);
		{
			GDirList dl(false/*recurseSubDirs*/, true/*reportFiles*/, false/*reportDirs*/, false/*reportPaths*/);
			while(true)
			{
				const char* filename = dl.GetNext();
				if(!filename)
					break;
				PathData pd;
				GFile::parsePath(filename, &pd);
				if(_stricmp(filename + pd.extStart, ".txt") == 0)
				{
					printf("%d) %s\n", (int)row, filename);
					makeTextFileVector(&sparseFeatures, pLabels, clss, row++, &vocab, filename, binary);
				}
				else if(_stricmp(filename + pd.extStart, ".html") == 0 || _stricmp(filename + pd.extStart, ".htm") == 0)
				{
					printf("%d) %s\n", (int)row, filename);
					makeHtmlFileVector(&sparseFeatures, pLabels, clss, row++, &vocab, filename, binary);
				}
			}
		}
		if(chdir(cwd) != 0)
			ThrowError("Failed to change dir");
	}

	// Save the files
	if(vocabFile.length() > 0)
	{
		FILE* pFile = fopen(vocabFile.c_str(), "w");
		FileHolder hFile(pFile);
		for(size_t i = 0; i < vocab.wordCount(); i++)
		{
			const char* szWord = vocab.stats(i).m_szWord;
			fprintf(pFile, "%s\n", szWord);
		}
	}
	GTwtDoc doc;
	doc.setRoot(sparseFeatures.toTwt(&doc));
	doc.save(featuresFilename.c_str());
	if(pLabels)
		pLabels->saveArff(labelsFilename.c_str());
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeSparseUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(0, 3, 76, 1000, true);
	UsageNode* pUsageTree2 = makeAlgorithmUsageTree();
	Holder<UsageNode> hUsageTree2(pUsageTree2);
	pUsageTree2->print(0, 3, 76, 1000, true);
	cout.flush();
}

void showError(GArgReader& args, const char* szAppName, const char* szMessage)
{
	cout << "_________________________________\n";
	cout << szMessage << "\n\n";
	args.set_pos(1);
	const char* szCommand = args.peek();
	UsageNode* pUsageTree = makeSparseUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cout << "Brief Usage Information:\n\n";
			cout << szAppName << " ";
			pUsageCommand->print(0, 3, 76, 1000, true);
			if(pUsageCommand->findPart("[algorithm]") >= 0)
			{
				UsageNode* pAlgTree = makeAlgorithmUsageTree();
				Holder<UsageNode> hAlgTree(pAlgTree);
				pAlgTree->print(1, 3, 76, 2, false);
			}
		}
		else
		{
			cout << "Brief Usage Information:\n\n";
			pUsageTree->print(0, 3, 76, 1, false);
		}
	}
	else
	{
		pUsageTree->print(0, 3, 76, 1, false);
		cout << "\nFor more specific usage information, enter as much of the command as you know.\n";
	}
	cout << "\nTo see full usage information, run:\n	" << szAppName << " usage\n\n";
	cout << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cout.flush();
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
			else if(args.if_pop("docstosparsematrix")) docsToSparseMatrix(args);
			else if(args.if_pop("train")) train(args);
			else if(args.if_pop("predict")) predict(args);
			else if(args.if_pop("test")) test(args);
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
