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
using std::string;

typedef void (*TestFunc)();


class TempFileMaker
{
protected:
	const char* m_szFilename;

public:
	TempFileMaker(const char* szFilename, const char* szContents)
	: m_szFilename(szFilename)
	{
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
	GPipeHolder hStdOut;
#ifdef WINDOWS
	GApp::systemExecute("waffles_transform.exe mergevert a.arff b.arff", true, &hStdOut);
#else
	GApp::systemExecute("waffles_transform mergevert a.arff b.arff", true, &hStdOut);
#endif
	char buf[512];
	size_t len = hStdOut.read(buf, 512);
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
	GPipeHolder hStdOut;
#ifdef WINDOWS
	GApp::systemExecute("waffles_recommend.exe fillmissingvalues a.arff baseline", true, &hStdOut);
#else
	GApp::systemExecute("waffles_recommend fillmissingvalues a.arff baseline", true, &hStdOut);
#endif
	char buf[512];
	size_t len = hStdOut.read(buf, 512);
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








bool runTest(const char* szTestName, TestFunc pTest)
{
	printf("%s", szTestName);
	size_t nSpaces = (size_t)70 - strlen(szTestName);
	for( ; nSpaces > 0; nSpaces--)
		printf(" ");
	fflush(stdout);
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
		printf("Passed\n");
	else
		printf("FAILED!!!\n");

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

	// Command-line tests
	runTest("waffles_transform mergevert", test_transform_mergevert);
	runTest("waffles_recommend fillmissingvalues", test_recommend_fillmissingvalues);
	printf("Done.\n");
}

int main(int argc, char *argv[])
{
	GApp::enableFloatingPointExceptions();
	int nRet = 0;
	try
	{
		RunAllTests();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

