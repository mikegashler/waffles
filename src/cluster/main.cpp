// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "../GClasses/GActivation.h"
#include "../GClasses/GApp.h"
#include "../GClasses/GBits.h"
#include "../GClasses/GCluster.h"
#include "../GClasses/GError.h"
#include "../GClasses/GMatrix.h"
#include "../GClasses/GImage.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GTransform.h"
#include "../GClasses/GVec.h"
#include "../GClasses/GHashTable.h"
#include "../GClasses/GHillClimber.h"
#include "../GClasses/GManifold.h"
#include "../GClasses/GNeighborFinder.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GHeap.h"
#include "../GClasses/GRect.h"
#include "../GClasses/GSparseMatrix.h"
#include "../GClasses/GMath.h"
#include "../GClasses/GDom.h"
#include "../GClasses/GSelfOrganizingMap.h"
#include <time.h>
#include <iostream>
#include <string>
#include <set>
#include <map>
#ifdef WIN32
#	include <direct.h>
#	include <process.h>
#endif
#include <exception>
#include "../wizard/usage.h"

using namespace GClasses;
using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::set;
using std::map;

GMatrix* loadData(const char* szFilename)
{
	// Load the dataset by extension
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
	return pData;
}

void agglomerativeclusterer(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int clusters = args.pop_uint();

	// Do the clustering
	GAgglomerativeClusterer clusterer(clusters);
	GMatrix* pOut = clusterer.doit(*pData);
	Holder<GMatrix> hOut(pOut);
	pOut->print(cout);
}

void fuzzykmeans(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int clusters = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	double fuzzifier = 1.3;
	size_t reps = 1;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else if(args.if_pop("-fuzzifier"))
			fuzzifier = args.pop_double();
		else if(args.if_pop("-reps"))
			reps = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Do the clustering
	GRand prng(nSeed);
	GFuzzyKMeans clusterer(clusters, &prng);
	clusterer.setFuzzifier(fuzzifier);
	clusterer.setReps(reps);
	GMatrix* pOut = clusterer.doit(*pData);
	Holder<GMatrix> hOut(pOut);
	pOut->print(cout);
}

void kmeans(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int clusters = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	size_t reps = 1;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else if(args.if_pop("-reps"))
			reps = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Do the clustering
	GRand prng(nSeed);
	GKMeans clusterer(clusters, &prng);
	clusterer.setReps(reps);
	GMatrix* pOut = clusterer.doit(*pData);
	Holder<GMatrix> hOut(pOut);
	pOut->print(cout);
}

void kmedoids(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int clusters = args.pop_uint();

	// Do the clustering
	GKMedoids clusterer(clusters);
	GMatrix* pOut = clusterer.doit(*pData);
	Holder<GMatrix> hOut(pOut);
	pOut->print(cout);
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeClusterUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(cout, 0, 3, 76, 1000, true);
	cout.flush();
}

void showError(GArgReader& args, const char* szAppName, const char* szMessage)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	args.set_pos(1);
	const char* szCommand = args.peek();
	UsageNode* pUsageTree = makeClusterUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << szAppName << " ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, true);
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
	int ret = 0;
	PathData pd;
	GFile::parsePath(argv[0], &pd);
	const char* appName = argv[0] + pd.fileStart;
	GArgReader args(argc, argv);
	args.pop_string(); // advance past the app name
	try
	{
		if(args.size() < 1) ThrowError("Expected a command");
		else if(args.if_pop("usage")) ShowUsage(appName);
		else if(args.if_pop("agglomerative")) agglomerativeclusterer(args);
		else if(args.if_pop("fuzzykmeans")) fuzzykmeans(args);
		else if(args.if_pop("kmeans")) kmeans(args);
		else if(args.if_pop("kmedoids")) kmedoids(args);
		else ThrowError("Unrecognized command: ", args.peek());
	}
	catch(const std::exception& e)
	{
		if(strcmp(e.what(), "nevermind") != 0) // if an error message was not already displayed...
			showError(args, appName, e.what());
		ret = 1;
	}

	return ret;
}

