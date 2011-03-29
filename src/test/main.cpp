// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <stdio.h>
#include <math.h>
#include <wchar.h>
#include <string>
#include <vector>
#ifdef WINDOWS
#	include <direct.h>
#endif
#include "../GClasses/GApp.h"
#include "../GSup/GBezier.h"
#include "../GClasses/GBitTable.h"
#include "../GClasses/GCluster.h"
#include "../GSup/GCrypto.h"
#include "../GSup/GDate.h"
#include "../GClasses/GDecisionTree.h"
#include "../GSup/GDiff.h"
#include "../GClasses/GEnsemble.h"
#include "../GClasses/GError.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GFourier.h"
#include "../GClasses/GGraph.h"
#include "../GClasses/GHashTable.h"
#include "../GClasses/GHiddenMarkovModel.h"
#include "../GClasses/GHillClimber.h"
#include "../GSup/GKeyPair.h"
#include "../GClasses/GKNN.h"
#include "../GClasses/GLinear.h"
#include "../GClasses/GManifold.h"
#include "../GClasses/GMath.h"
#include "../GClasses/GMatrix.h"
#include "../GClasses/GMixtureOfGaussians.h"
#include "../GClasses/GNaiveBayes.h"
#include "../GClasses/GNaiveInstance.h"
#include "../GClasses/GNeighborFinder.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GPolynomial.h"
#include "../GClasses/GPriorityQueue.h"
#include "../GClasses/GRand.h"
#include "../GSup/GRayTrace.h"
#include "../GClasses/GRegion.h"
#include "../GSup/GSocket.h"
#include "../GClasses/GSparseMatrix.h"
#include "../GClasses/GSpinLock.h"
#include "../GClasses/GStabSearch.h"
#include "../GClasses/GThread.h"
#include "../GClasses/GTime.h"
#include "../GClasses/GTransform.h"
#include "../GClasses/GTwt.h"
#include "../GClasses/GVec.h"

using namespace GClasses;
using std::cerr;
using std::cout;
using std::string;
using std::vector;

typedef void (*TestFunc)();


void sysExec(const char* szAppName, const char* szArgs, GPipe* pStdOut = NULL, GPipe* pStdErr = NULL, GPipe* pStdIn = NULL)
{
	string s = szAppName;
//#ifdef _DEBUG
//	s += "dbg";
//#endif
#ifdef WINDOWS
	s += ".exe";
#endif
	s += " ";
	s += szArgs;
	GApp::systemExecute(s.c_str(), true, pStdOut, pStdErr, pStdIn);
}

class TempFileMaker
{
protected:
	const char* m_szFilename;

public:
	TempFileMaker(const char* szFilename, const char* szContents)
	: m_szFilename(szFilename)
	{
		if(szContents)
			GFile::saveFile(szContents, strlen(szContents), szFilename);
	}

	~TempFileMaker()
	{
		GFile::deleteFile(m_szFilename);
	}
};


void test_transform_mergevert()
{
	// Make some input files
	TempFileMaker tempFile1("a.arff",
		"@RELATION test\n"
		"@ATTRIBUTE a1 continuous\n"
		"@ATTRIBUTE a2 { alice, bob }\n"
		"@ATTRIBUTE a3 { true, false }\n"
		"@DATA\n"
		"1.2, alice, true\n"
		"2.3, bob, false\n"
		);
	TempFileMaker tempFile2("b.arff",
		"@RELATION test\n"
		"@ATTRIBUTE a1 continuous\n"
		"@ATTRIBUTE a2 { charlie, bob }\n"
		"@ATTRIBUTE a3 { false, true }\n"
		"@DATA\n"
		"3.4, bob, true\n"
		"4.5, charlie, false\n"
		);

	// Execute the command
	GPipe pipeStdOut;
	sysExec("waffles_transform", "mergevert a.arff b.arff", &pipeStdOut);
	char buf[512];
	size_t len = pipeStdOut.read(buf, 512);
	if(len == 512)
		ThrowError("need a bigger buffer");
	buf[len] = '\0';

	// Check the results
	GMatrix* pOutput = GMatrix::parseArff(buf, strlen(buf));
	Holder<GMatrix> hOutput(pOutput);
	GMatrix& M = *pOutput;
	if(M.rows() != 4 || M.cols() != 3)
		ThrowError("failed");
	if(M.relation()->valueCount(0) != 0)
		ThrowError("failed");
	if(M.relation()->valueCount(1) != 3)
		ThrowError("failed");
	if(M.relation()->valueCount(2) != 2)
		ThrowError("failed");
	std::ostringstream oss;
	GArffRelation* pRel = (GArffRelation*)M.relation().get();
	pRel->printValue(oss, 2.0, 1);
	string s = oss.str();
	if(strcmp(s.c_str(), "charlie") != 0)
		ThrowError("failed");
	if(M[0][0] != 1.2 || M[1][0] != 2.3 || M[2][0] != 3.4 || M[3][0] != 4.5)
		ThrowError("failed");
	if(M[0][1] != 0 || M[1][1] != 1 || M[2][1] != 1 || M[3][1] != 2)
		ThrowError("failed");
	if(M[0][2] != 0 || M[1][2] != 1 || M[2][2] != 0 || M[3][2] != 1)
		ThrowError("failed");
}

void test_recommend_fillmissingvalues()
{
	// Make some input files
	TempFileMaker tempFile1("a.arff",
		"@RELATION test\n"
		"@ATTRIBUTE a1 { a, b, c }\n"
		"@ATTRIBUTE a2 continuous\n"
		"@ATTRIBUTE a3 { d, e, f }\n"
		"@ATTRIBUTE a4 { g, h, i }\n"
		"@DATA\n"
		"a, ?, f, i\n"
		"?, 2, ?, i\n"
		"b, ?, d, ?\n"
		"?, 4, ?, ?\n"
		"?, ?, e, g\n"
		"?, ?, e, ?\n"
		"a, ?, ?, h\n"
		"\n"
		);

	// Execute the command
	GPipe pipeStdOut;
	sysExec("waffles_recommend", "fillmissingvalues a.arff baseline", &pipeStdOut);
	char buf[512];
	size_t len = pipeStdOut.read(buf, 512);
	if(len == 512)
		ThrowError("need a bigger buffer");
	buf[len] = '\0';

	// Check the results
	GMatrix* pOutput = GMatrix::parseArff(buf, strlen(buf));
	Holder<GMatrix> hOutput(pOutput);
	GMatrix& M = *pOutput;
	if(M.rows() != 7 || M.cols() != 4)
		ThrowError("failed");
	if(M[0][0] != 0)
		ThrowError("failed");
	if(M[0][1] != 3)
		ThrowError("failed");
	if(M[1][1] != 2)
		ThrowError("failed");
	if(M[2][1] != 3)
		ThrowError("failed");
	if(M[3][3] != 2)
		ThrowError("failed");
	if(M[4][0] != 0)
		ThrowError("failed");
	if(M[5][1] != 3)
		ThrowError("failed");
	if(M[6][2] != 1)
		ThrowError("failed");
	if(M[6][3] != 1)
		ThrowError("failed");
}

void test_document_classification()
{
	{
		GFile::makeDir("class_spam");
		TempFileMaker tempFile1("class_spam/a.txt", "Buy now! Cheap Viagara. All of your problems are solved. For a limited time, Act now. Click here to for a free sample. You might already be a winner! Hurry, supplies are limited.");
		TempFileMaker tempFile2("class_spam/b.txt", "Congratulations, the Prince of Nigeria has selected you to be the benefactor of the sum of one million dollars! If you act now, he will immediately transfer this amount to your bank account.");
		TempFileMaker tempFile3("class_spam/c.txt", "Winner winner winner! You are the winner of the one million dollar sweepstakes. To collect one million dollars, click here now! Hurry, or it will be given to the runner up.");
		TempFileMaker tempFile4("class_spam/d.txt", "Dear bank customer, we regret to inform you that your account has been compromised. Please confirm your identity so that we may restore the one million dollars that has erroneously been stolen from your account.");
		TempFileMaker tempFile5("class_spam/e.txt", "Buy now! Just one dollar! Lap top computers. Cheap deals! Hurry. Supplies are limited. Mention this add and receive a free laptop with lifetime subscription.");
		TempFileMaker tempFile6("class_spam/f.txt", "Free samples! Great deals! discounts! coupons! Winner! Free money! One million dollars in prizes! Viagara! Hurry! Cheap!");

		GFile::makeDir("class_auto");
		TempFileMaker tempFile7("class_auto/a.txt", "Thank you for registering with our site. To confirm you identity, please follow this link.");
		TempFileMaker tempFile8("class_auto/b.txt", "Thank you for signing up for an account at foo.com. Your automatically generated password is ax58c90s3.");
		TempFileMaker tempFile9("class_auto/c.txt", "You, or someone claiming to be you has signed up for an account at yomamma.com. Please confirm your identity by clicking on the following link.");
		TempFileMaker tempFile10("class_auto/d.txt", "You are now registered with newssite.com. You may begin posting comments using your user name and password. Thank you for visiting us.");
		TempFileMaker tempFile11("class_auto/e.txt", "This is an automatically generated email. You have registered with somesite.com. Your new user name and password is found below. Thank you, and have a nice day.");
		TempFileMaker tempFile12("class_auto/f.txt", "A password has automatically been generated for you. To log in, please click on the link found below. Your password is asdfjkl");

		GFile::makeDir("class_ham");
		TempFileMaker tempFile13("class_ham/a.txt", "Dear Dr. Johnson, I am writing to inquire whether you will be attending the conference on document classification. I am seeking an opportunity to meet with you. Sincerely, Me.");
		TempFileMaker tempFile14("class_ham/b.txt", "Dear Bob, Are you there? I have been looking for you all over. I am on the fourth floor of the computer building. Didn't we agree to meet at 2:30pm?");
		TempFileMaker tempFile15("class_ham/c.txt", "Dear Susan, Thank you for baking me those delicious cinnamon rolls. I have never eaten anything so delicious in my entire life. You should become a professional chef. Sincerely, Me.");
		TempFileMaker tempFile16("class_ham/d.txt", "Bob, I cannot tell you how dissapointed I am that you have chosen to attend Dr. Johnson's conference. He is a poser, and I think you should talk to Susan about it first.");
		TempFileMaker tempFile17("class_ham/e.txt", "What do you mean? Of course I like cinnamon rolls. Everybody likes them! Unfortunately, my diet does not permit me to indulge in such frivoloties at this time. I sincerely hope you will make more when I am done with this.");
		TempFileMaker tempFile18("class_ham/f.txt", "Of course. How else would a slinky become lodged in the center of a giant cube of Jello? Meet me in the computer building five minutes before it starts, and we'll attend the conference together. See you then. --Bob");

		// Generate a sparse feature matrix and a corresponding dense label matrix
		{
			GPipe pipeIgnoreMe;
			sysExec("waffles_generate", "docstosparsematrix class_ham class_auto class_spam", &pipeIgnoreMe);
		}
		TempFileMaker tempFileFeatures("features.sparse", NULL);
		TempFileMaker tempFileLabels("labels.arff", NULL);

		// Shuffle the data
		GPipe pipeStdOut;
		sysExec("waffles_transform", "sparseshuffle features.sparse -seed 0 -labels labels.arff l2.arff", &pipeStdOut);
		pipeStdOut.toFile("f2.sparse");
		TempFileMaker tempF2("f2.sparse", NULL);
		TempFileMaker tempL2("l2.arff", NULL);

		// Make a set of models
		vector<string> models;
		models.push_back("naivebayes");
		models.push_back("knn 3 -cosine");
		//models.push_back("knn 3 -pearson");
		//models.push_back("neuralnet");

		// Do cross-validation
		TempFileMaker tempFileTrainFeatures("train.sparse", NULL);
		TempFileMaker tempFileTestFeatures("test.sparse", NULL);
		TempFileMaker tempFileTrainLabels("train.arff", NULL);
		TempFileMaker tempFileTestLabels("test.arff", NULL);
		TempFileMaker tempFileModel("model.twt", NULL);
		char buf[256];
		GMatrix results(18, models.size());
		for(size_t i = 0; i < 18; i++)
		{
			// Separate the test fold from the rest of the data
			string sArgs1 = "sparsesplitfold f2.sparse ";
			sArgs1 += to_str(i);
			sArgs1 += " 18";
			sysExec("waffles_transform", sArgs1.c_str());
			string sArgs2 = "splitfold l2.arff ";
			sArgs2 += to_str(i);
			sArgs2 += " 18";
			sysExec("waffles_transform", sArgs2.c_str());

			// Train and test each model
			for(size_t j = 0; j < models.size(); j++)
			{
				// Train the model
				string sArgs = "trainsparse -seed 0 train.sparse train.arff ";
				sArgs += models[j];
				GPipe pipeStdOut2;
				sysExec("waffles_learn", sArgs.c_str(), &pipeStdOut2);
				pipeStdOut2.toFile("model.twt");

				// Test the model
				GPipe pipeStdOut3;
				sysExec("waffles_learn", "testsparse -seed 0 model.twt test.sparse test.arff", &pipeStdOut3);
				size_t len = pipeStdOut3.read(buf, 256);
				if(len >= 256)
					ThrowError("Need a bigger buffer");
				buf[len] = '\0';
				double accuracy = atof(buf);
				results[i][j] = accuracy;
			}
		}
		double resultsNaiveBayes = results.mean(0);
		double resultsKnnCosine = results.mean(1);
		//double resultsKnnPearson = results.mean(2);
		if(resultsNaiveBayes < 0.83)
			ThrowError("failed");
		if(resultsKnnCosine < 0.88)
			ThrowError("failed");
		//if(resultsKnnPearson < 0.50)
		//	ThrowError("failed");
	}
#ifdef WINDOWS
	_rmdir("class_ham");
	_rmdir("class_auto");
	_rmdir("class_spam");
#else
	rmdir("class_ham");
	rmdir("class_auto");
	rmdir("class_spam");
#endif
}







bool runTest(const char* szTestName, TestFunc pTest)
{
	cout << szTestName;
	size_t nSpaces = (size_t)70 - strlen(szTestName);
	for( ; nSpaces > 0; nSpaces--)
		cout << " ";
	cout.flush();
	bool bPass = false;
	try
	{
		pTest();
		bPass = true;
	}
	catch(...)
	{
	}
	if(bPass)
		cout << "Passed\n";
	else
		cout << "FAILED!!!\n";

	return bPass;
}

void RunAllTests()
{
	// Class tests
	runTest("GAgglomerativeClusterer", GAgglomerativeClusterer::test);
	runTest("GAtomicCycleFinder", GAtomicCycleFinder::test);
	runTest("GAttributeSelector", GAttributeSelector::test);
	runTest("GBag", GBag::test);
	runTest("GBaselineLearner", GBaselineLearner::test);
	runTest("GBezier", GBezier::test);
	runTest("GBitTable", GBitTable::test);
	runTest("GBrandesBetweenness", GBrandesBetweennessCentrality::test);
	runTest("GBucket", GBucket::test);
	runTest("GCompressor", GCompressor::test);
	runTest("GCoordVectorIterator", GCoordVectorIterator::test);
	runTest("GCrypto", GCrypto::test);
	runTest("GCycleCut", GCycleCut::test);
	runTest("GDate", TestGDate);
	runTest("GDecisionTree", GDecisionTree::test);
	runTest("GDiff", GDiff::test);
	runTest("GDijkstra", GDijkstra::test);
	runTest("GFloydWarshall", GFloydWarshall::test);
	runTest("GFourier", GFourier::test);
	runTest("GGraphCut", GGraphCut::test);
	runTest("GHashTable", GHashTable::test);
	runTest("GHiddenMarkovModel", GHiddenMarkovModel::test);
	runTest("GKdTree", GKdTree::test);
	runTest("GKeyPair", GKeyPair::test);
	runTest("GKNN", GKNN::test);
	runTest("GLinearProgramming", GLinearProgramming::test);
	runTest("GLinearRegressor", GLinearRegressor::test);
	runTest("GMath", GMath::test);
	runTest("GManifold", GManifold::test);
	runTest("GMatrix", GMatrix::test);
	runTest("GMeanMarginsTree", GMeanMarginsTree::test);
	runTest("GMixtureOfGaussians", GMixtureOfGaussians::test);
	runTest("GNaiveBayes", GNaiveBayes::test);
	runTest("GNaiveInstance", GNaiveInstance::test);
	runTest("GNeuralNet", GNeuralNet::test);
	runTest("GNeuralNetPseudoInverse", GNeuralNetPseudoInverse::test);
	runTest("GPCARotateOnly", GPCARotateOnly::test);
	runTest("GPolynomial", GPolynomial::test);
	runTest("GPriorityQueue", GPriorityQueue::test);
	runTest("GProbeSearch", GProbeSearch::test);
	runTest("GRand", GRand::test);
	runTest("GShortcutPruner", GShortcutPruner::test);
	runTest("GSocket", GSocketClient::test);
	runTest("GSparseMatrix", GSparseMatrix::test);
	runTest("GSpinLock", GSpinLock::test);
	runTest("GSubImageFinder", GSubImageFinder::test);
	runTest("GSubImageFinder2", GSubImageFinder2::test);
	runTest("GTwt", GTwtDoc::test);
	runTest("GVec", GVec::test);

	string s = "waffles_learn";
#ifdef WIN32
	s += ".exe";
#endif

#ifdef WINDOWS
	bool runCommandLineTests = true;
#else
#	ifdef __linux__
	bool runCommandLineTests = true;
#	else
	bool runCommandLineTests = false; // I don't have the test-harness for the command-line apps working on OSX yet
#	endif
#endif
	if(runCommandLineTests)
	{
		if(GFile::doesFileExist(s.c_str()))
		{
			// Command-line tests
			runTest("waffles_transform mergevert", test_transform_mergevert);
			runTest("waffles_recommend fillmissingvalues", test_recommend_fillmissingvalues);
#ifndef WINDOWS
			runTest("document classification", test_document_classification);
#endif
		}
		else
			cout << "Skipping the command-line tool tests because the optimized command-line tools have not yet been built.\n";
	}

	cout << "Done.\n";
	cout.flush();
}

int main(int argc, char *argv[])
{
	GApp::enableFloatingPointExceptions();
	int nRet = 0;
	try
	{
		char buf[256];
		if(GApp::appPath(buf, 256, true) == -1)
			ThrowError("Failed to retrieve app path");
		if(chdir(buf) != 0)
			ThrowError("Failed to change the dir to the app folder");
		RunAllTests();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

