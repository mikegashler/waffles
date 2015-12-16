/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer
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

#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "../GClasses/GActivation.h"
#include "../GClasses/GApp.h"
#include "../GClasses/GBits.h"
#include "../GClasses/GCluster.h"
#include "../GClasses/GDistance.h"
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
#include <memory>
#include <map>
#include <algorithm>
#ifdef WIN32
#	include <direct.h>
#	include <process.h>
#endif
#include <exception>
#include "../GClasses/usage.h"

using namespace GClasses;
using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::set;
using std::map;
using std::shared_ptr;

size_t getAttrVal(const char* szString, size_t attrCount)
{
	bool fromRight = false;
	if(*szString == '*')
	{
		fromRight = true;
		szString++;
	}
	if(*szString < '0' || *szString > '9')
		throw Ex("Expected a digit while parsing attribute list");
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
					throw Ex("Invalid column index: ", to_str(val), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
				if(attrSet.find(val) != attrSet.end())
					throw Ex("Columns ", to_str(val), " is listed multiple times");
				attrSet.insert(val);
				list.push_back(val);
			}
			else
			{
				size_t beg = getAttrVal(szList, attrCount);
				if(beg >= attrCount)
					throw Ex("Invalid column index: ", to_str(beg), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
				size_t end = getAttrVal(szList + j + 1, attrCount);
				if(end >= attrCount)
					throw Ex("Invalid column index: ", to_str(end), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
				int step = 1;
				if(end < beg)
					step = -1;
				for(size_t val = beg; true; val += step)
				{
					if(attrSet.find(val) != attrSet.end())
						throw Ex("Column ", to_str(val), " is listed multiple times");
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

///Return a pointer to newly allocated data read from the command line
///represented by args.
///
///The returned matrix is allocated by new and it is the caller's
///responsibility to deallocate it. The suggested manner is to use a
///Holder<GMatrix*>
///
///In the returned matrix, all of the attributes designated as labels
///have been moved to the end and ignored attributes have been
///removed. The original indices of all the attributes are returned in
///originalIndices.
///
///\param args the command-line arguments
///
///\param pLabelDims (out parameter) the index of the first attribute
///which is designated a label.
///
///\param originalIndices the vector in which to place the original
///indices.  originalIndices[i] is the index in the original data file
///of the attribute currently at index i.
void loadDataWithSwitches(GMatrix& data, GArgReader& args, size_t& pLabelDims,
			      std::vector<size_t>& originalIndices)
{
	// Load the dataset by extension
	const char* szFilename = args.pop_string();
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
		data.loadArff(szFilename);
	else if(_stricmp(szFilename + pd.extStart, ".csv") == 0)
	{
		GCSVParser parser;
		parser.parse(data, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < data.cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else if(_stricmp(szFilename + pd.extStart, ".dat") == 0)
	{
		GCSVParser parser;
		parser.setSeparator('\0');
		parser.parse(data, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < data.cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else
		throw Ex("Unsupported file format: ", szFilename + pd.extStart);

	//Make the initial list of original indices
	originalIndices.resize(data.cols());
	for(std::size_t i = 0; i < originalIndices.size(); ++i){
	  originalIndices.at(i) = i;
	}

	// Parse params
	vector<size_t> ignore;
	vector<size_t> labels;
	while(args.next_is_flag())
	{
		if(args.if_pop("-labels"))
			parseAttributeList(labels, args, data.cols());
		else if(args.if_pop("-ignore"))
			parseAttributeList(ignore, args, data.cols());
		else
			break;
	}

	// Throw out the ignored attributes
	std::sort(ignore.begin(), ignore.end());
	for(size_t i = ignore.size() - 1; i < ignore.size(); i--)
	{
		data.deleteColumns(ignore[i], 1);
		originalIndices.erase(originalIndices.begin()+ignore[i]);
		for(size_t j = 0; j < labels.size(); j++)
		{
			if(labels[j] >= ignore[i])
			{
				if(labels[j] == ignore[i])
					throw Ex("Attribute ", to_str(labels[j]), " is both ignored and used as a label");
				labels[j]--;
			}
		}
	}

	// Swap label columns to the end
	pLabelDims = std::max((size_t)1, labels.size());
	for(size_t i = 0; i < labels.size(); i++)
	{
		size_t src = labels[i];
		size_t dst = data.cols() - pLabelDims + i;
		if(src != dst)
		{
			data.swapColumns(src, dst);
			std::swap(originalIndices.at(src),
				  originalIndices.at(dst));
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
}

GMatrix* loadData(const char* szFilename)
{
	// Load the dataset by extension
	GMatrix* pM = new GMatrix();
	Holder<GMatrix> hM(pM);
	GMatrix& m = *pM;
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
		m.loadArff(szFilename);
	else if(_stricmp(szFilename + pd.extStart, ".csv") == 0)
	{
		GCSVParser parser;
		parser.parse(m, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < m.cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else if(_stricmp(szFilename + pd.extStart, ".dat") == 0)
	{
		GCSVParser parser;
		parser.setSeparator('\0');
		parser.parse(m, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < m.cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else
		throw Ex("Unsupported file format: ", szFilename + pd.extStart);

	return hM.release();
}

void showInstantiateNeighborFinderError(const char* szMessage, GArgReader& args)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	const char* szNFName = args.peek();
	UsageNode* pNFTree = makeNeighborUsageTree();
	Holder<UsageNode> hNFTree(pNFTree);
	if(szNFName)
	{
		UsageNode* pUsageAlg = pNFTree->choice(szNFName);
		if(pUsageAlg)
		{
			cerr << "Partial Usage Information:\n\n";
			pUsageAlg->print(cerr, 0, 3, 76, 1000, true);
		}
		else
		{
			cerr << "\"" << szNFName << "\" is not a recognized neighbor-finding techniqie. Try one of these:\n\n";
			pNFTree->print(cerr, 0, 3, 76, 1, false);
		}
	}
	else
	{
		cerr << "Expected a neighbor-finding technique. Here are some choices:\n";
		pNFTree->print(cerr, 0, 3, 76, 1, false);
	}
	cerr << "\nTo see full usage information, run:\n	waffles_transform usage\n\n";
	cerr << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cerr.flush();
}

GNeighborFinder* instantiateNeighborFinder(GMatrix* pData, GRand* pRand, GArgReader& args)
{
	// Get the algorithm name
	int argPos = args.get_pos();
	GNeighborFinder* pNF = NULL;
	const char* alg = args.pop_string();

	try
	{
		// Parse the options
		int cutCycleLen = 0;
		bool normalize = false;
		while(args.next_is_flag())
		{
			if(args.if_pop("-cyclecut"))
				cutCycleLen = args.pop_uint();
			else if(args.if_pop("-normalize"))
				normalize = true;
			else
				throw Ex("Invalid neighbor finder option: ", args.peek());
		}

		// Parse required algorithms
		if(_stricmp(alg, "bruteforce") == 0)
		{
			int neighbors = args.pop_uint();
			pNF = new GBruteForceNeighborFinder(pData, neighbors, NULL, true);
		}
		else if(_stricmp(alg, "kdtree") == 0)
		{
			int neighbors = args.pop_uint();
			pNF = new GKdTree(pData, neighbors, NULL, true);
		}
		else if(_stricmp(alg, "temporal") == 0)
		{
			GMatrix* pControlData = loadData(args.pop_string());
			Holder<GMatrix> hControlData(pControlData);
			if(pControlData->rows() != pData->rows())
				throw Ex("mismatching number of rows");
			int neighbors = args.pop_uint();
			pNF = new GTemporalNeighborFinder(pData, hControlData.release(), true, neighbors, pRand);
		}
		else
			throw Ex("Unrecognized neighbor finding algorithm: ", alg);

		// Normalize
		if(normalize)
		{
			GNeighborGraph* pNF2 = new GNeighborGraph(pNF, true);
			pNF2->fillCache();
			pNF2->normalizeDistances();
			pNF = pNF2;
		}

		// Apply CycleCut
		if(cutCycleLen > 0)
		{
			GNeighborGraph* pNF2 = new GNeighborGraph(pNF, true);
			pNF2->fillCache();
			pNF2->cutShortcuts(cutCycleLen);
			pNF = pNF2;
		}
	}
	catch(const std::exception& e)
	{
		args.set_pos(argPos);
		showInstantiateNeighborFinderError(e.what(), args);
		throw Ex("nevermind"); // this means "don't display another error message"
	}

	return pNF;
}

void attributeSelector(GArgReader& args)
{
	// Load the data
	size_t labelDims;
	std::vector<size_t> originalIndices;
	GMatrix data;
	loadDataWithSwitches(data, args, labelDims, originalIndices);

	// Parse the options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	int targetFeatures = 1;
	string outFilename = "";
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-out"))
		{
			targetFeatures = args.pop_uint();
			outFilename = args.pop_string();
		}
		else
			throw Ex("Invalid neighbor finder option: ", args.peek());
	}

	// Do the attribute selection
	GAttributeSelector as(labelDims, targetFeatures);
	as.setSeed(seed);
	if(outFilename.length() > 0)
	{
		as.train(data);
		GMatrix* pDataOut = as.transformBatch(data);
		Holder<GMatrix> hDataOut(pDataOut);
		cout << "Reduced data saved to " << outFilename.c_str() << ".\n";
		pDataOut->saveArff(outFilename.c_str());
	}
	else
		as.train(data);
	cout << "\nAttribute rankings from most salient to least salient. (Attributes are zero-indexed.)\n";
	const GArffRelation& rel = (GArffRelation&)data.relation();
	for(size_t i = 0; i < as.ranks().size(); i++)
	  cout << originalIndices.at(as.ranks()[i]) << " " << rel.attrName(as.ranks()[i]) << "\n";
}

void blendEmbeddings(GArgReader& args)
{
	// Load the files and params
	GMatrix* pDataOrig = loadData(args.pop_string());
	Holder<GMatrix> hDataOrig(pDataOrig);
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	GRand prng(seed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pDataOrig, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	GMatrix* pDataA = loadData(args.pop_string());
	Holder<GMatrix> hDataA(pDataA);
	GMatrix* pDataB = loadData(args.pop_string());
	Holder<GMatrix> hDataB(pDataB);
	if(pDataA->rows() != pDataOrig->rows() || pDataB->rows() != pDataOrig->rows())
		throw Ex("mismatching number of rows");
	if(pDataA->cols() != pDataB->cols())
		throw Ex("mismatching number of cols");

	// Parse Options
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Get a neighbor table
	if(!pNF->isCached())
	{
		GNeighborGraph* pNF2 = new GNeighborGraph(hNF.release(), true);
		hNF.reset(pNF2);
		pNF = pNF2;
	}
	((GNeighborGraph*)pNF)->fillCache();
	size_t* pNeighborTable = ((GNeighborGraph*)pNF)->cache();

	// Do the blending
	size_t startPoint = (size_t)prng.next(pDataA->rows());
	double* pRatios = new double[pDataA->rows()];
	ArrayHolder<double> hRatios(pRatios);
	GVec::setAll(pRatios, 0.5, pDataA->rows());
	GMatrix* pDataC = GManifold::blendEmbeddings(pDataA, pRatios, pDataB, pNF->neighborCount(), pNeighborTable, startPoint);
	Holder<GMatrix> hDataC(pDataC);
	pDataC->print(cout);
}

void breadthFirstUnfolding(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pData, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	int targetDims = args.pop_uint();

	// Parse Options
	size_t reps = 1;
	Holder<GMatrix> hControlData(NULL);
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else if(args.if_pop("-reps"))
			reps = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GBreadthFirstUnfolding transform(reps, pNF->neighborCount(), targetDims);
	transform.rand().setSeed(nSeed);
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.reduce(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

void curviness1(GArgReader& args)
{

}

void curviness2(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GNormalize norm;
	GMatrix* pDataNormalized = norm.reduce(*pData);
	Holder<GMatrix> hDataNormalized(pDataNormalized);
	hData.reset();
	pData = NULL;

	// Parse Options
	size_t maxEigs = 10;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	Holder<GMatrix> hControlData(NULL);
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-maxeigs"))
			maxEigs = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	GRand rand(seed);
	size_t targetDims = std::min(maxEigs, pDataNormalized->cols());

	// Do linear PCA
	GNeuroPCA np1(targetDims, &rand);
	np1.setActivation(new GActivationIdentity());
	np1.computeEigVals();
	GMatrix* pResults1 = np1.reduce(*pDataNormalized);
	Holder<GMatrix> hResults1(pResults1);
	double* pEigVals1 = np1.eigVals();
	for(size_t i = 0; i + 1 < targetDims; i++)
		pEigVals1[i] = sqrt(pEigVals1[i]) - sqrt(pEigVals1[i + 1]);
	size_t max1 = GVec::indexOfMax(pEigVals1, targetDims - 1, &rand);
	double v1 = (double)max1;
	if(max1 > 0 && max1 + 2 < targetDims)
		v1 += (pEigVals1[max1 - 1] - pEigVals1[max1 + 1]) / (2.0 * (pEigVals1[max1 - 1] + pEigVals1[max1 + 1] - 2.0 * pEigVals1[max1]));

	// Do non-linear PCA
	GNeuroPCA np2(targetDims, &rand);
	np1.setActivation(new GActivationLogistic());
	np2.computeEigVals();
	GMatrix* pResults2 = np2.reduce(*pDataNormalized);
	Holder<GMatrix> hResults2(pResults2);
	double* pEigVals2 = np2.eigVals();
	for(size_t i = 0; i + 1 < targetDims; i++)
		pEigVals2[i] = sqrt(pEigVals2[i]) - sqrt(pEigVals2[i + 1]);
	size_t max2 = GVec::indexOfMax(pEigVals2, targetDims - 1, &rand);
	double v2 = (double)max2;
	if(max2 > 0 && max2 + 2 < targetDims)
		v2 += (pEigVals2[max2 - 1] - pEigVals2[max2 + 1]) / (2.0 * (pEigVals2[max2 - 1] + pEigVals2[max2 + 1] - 2.0 * pEigVals2[max2]));

	// Compute the difference in where the eigenvalues fall
	cout.precision(14);
	cout << (v1 - v2) << "\n";
}

void isomap(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pData, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	int targetDims = args.pop_uint();

	// Parse Options
	bool tolerant = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else if(args.if_pop("-tolerant"))
			tolerant = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GIsomap transform(pNF->neighborCount(), targetDims, &prng);
	transform.setNeighborFinder(pNF);
	if(tolerant)
		transform.dropDisconnectedPoints();
	GMatrix* pDataAfter = transform.reduce(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

void lle(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pData, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	int targetDims = args.pop_uint();

	// Parse Options
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GLLE transform(pNF->neighborCount(), targetDims, &prng);
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.reduce(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

void ManifoldSculpting(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pData, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	size_t targetDims = args.pop_uint();

	// Parse Options
	const char* szPreprocessedData = NULL;
	double scaleRate = 0.999;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else if(args.if_pop("-continue"))
			szPreprocessedData = args.pop_string();
		else if(args.if_pop("-scalerate"))
			scaleRate = args.pop_double();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Load the hint data
	GMatrix* pDataHint = NULL;
	Holder<GMatrix> hDataHint(NULL);
	if(szPreprocessedData)
	{
		pDataHint = loadData(szPreprocessedData);
		hDataHint.reset(pDataHint);
		if(pDataHint->relation().size() != targetDims)
			throw Ex("Wrong number of dims in the hint data");
		if(pDataHint->rows() != pData->rows())
			throw Ex("Wrong number of patterns in the hint data");
	}

	// Transform the data
	GManifoldSculpting transform(pNF->neighborCount(), targetDims, &prng);
	transform.setSquishingRate(scaleRate);
	if(pDataHint)
		transform.setPreprocessedData(hDataHint.release());
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.reduce(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}
/*
void manifoldSculptingForControl(GArgReader& args)
{
	// Load the file and params
	GMatrix* pDataObs = loadData(args.pop_string());
	Holder<GMatrix> hDataObs(pDataObs);
	GMatrix* pDataControl = loadData(args.pop_string());
	Holder<GMatrix> hDataControl(pDataControl);
	int neighbors = args.pop_uint();
	int targetDims = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	const char* szPreprocessedData = NULL;
	double scaleRate = 0.999;
	double lambda = 0;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else if(args.if_pop("-continue"))
			szPreprocessedData = args.pop_string();
		else if(args.if_pop("-scalerate"))
			scaleRate = args.pop_double();
		else if(args.if_pop("-alignconsequences"))
			lambda = args.pop_double();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Load the hint data
	GMatrix* pDataHint = NULL;
	Holder<GMatrix> hDataHint(NULL);
	if(szPreprocessedData)
	{
		pDataHint = loadData(szPreprocessedData);
		hDataHint.reset(pDataHint);
		if(pDataHint->relation()->size() != targetDims)
			throw Ex("Wrong number of dims in the hint data");
		if(pDataHint->rows() != pDataObs->rows())
			throw Ex("Wrong number of patterns in the hint data");
	}

	// Transform the data
	GRand prng(nSeed);
	GManifoldSculptingForControl transform(neighbors, targetDims, &prng, pDataControl, lambda);
	transform.setSquishingRate(scaleRate);
	if(pDataHint)
		transform.setPreprocessedData(hDataHint.release());

	GNeighborFinder* pNF = new GDynamicSystemNeighborFinder(pDataObs, pDataControl, false, neighbors, &prng);
	Holder<GNeighborFinder> hNF(pNF);
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.reduce(pDataObs);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

void manifoldUnfolder(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pData, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	int targetDims = args.pop_uint();

	// Parse Options
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GManifoldUnfolder transform(pNF->neighborCount(), targetDims, &prng);
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.reduce(pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}
*/

void multiDimensionalScaling(GArgReader& args)
{
	GRand prng(0);
	GMatrix* pDistances = loadData(args.pop_string());
	int targetDims = args.pop_uint();

	// Parse Options
	bool useSquaredDistances = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-squareddistances"))
			useSquaredDistances = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	GMatrix* pResults = GManifold::multiDimensionalScaling(pDistances, targetDims, &prng, useSquaredDistances);
	Holder<GMatrix> hResults(pResults);
	pResults->print(cout);
}

void neuroPCA(GArgReader& args)
{
	// Load the file
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int nTargetDims = args.pop_uint();

	// Parse options
	string roundTrip;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool trainBias = true;
	bool linear = false;
	string eigenvalues = "";
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-clampbias"))
			trainBias = false;
		else if(args.if_pop("-linear"))
			linear = true;
		else if(args.if_pop("-eigenvalues"))
			eigenvalues = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GRand prng(seed);
	GNeuroPCA transform(nTargetDims, &prng);
	if(!trainBias)
		transform.clampBias();
	if(linear)
		transform.setActivation(new GActivationIdentity());
	if(eigenvalues.length() > 0)
		transform.computeEigVals();
	GMatrix* pDataAfter = transform.reduce(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);

	// Save the eigenvalues
	if(eigenvalues.length() > 0)
	{
		GArffRelation* pRelation = new GArffRelation();
		pRelation->addAttribute("eigenvalues", 0, NULL);
		GMatrix dataEigenvalues(pRelation);
		dataEigenvalues.newRows(nTargetDims);
		double* pEigVals = transform.eigVals();
		for(int i = 0; i < nTargetDims; i++)
			dataEigenvalues[i][0] = pEigVals[i];
		dataEigenvalues.saveArff(eigenvalues.c_str());
	}

	// In linear mode, people usually expect normalized eigenvectors, so let's normalize them now
	if(linear)
	{
		GMatrix* pWeights = transform.weights();
		GAssert(pWeights->cols() == pData->cols());
		for(int i = 0; i < nTargetDims; i++)
		{
			double scal = sqrt(pWeights->row(i + 1).squaredMagnitude());
			for(size_t j = 0; j < pDataAfter->rows(); j++)
				pDataAfter->row(j)[i] *= scal;
		}
	}

	pDataAfter->print(cout);
}

void principalComponentAnalysis(GArgReader& args)
{
	// Load the file
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int nTargetDims = args.pop_uint();

	// Parse options
	string roundTrip;
	size_t seed = getpid() * (unsigned int)time(NULL);
	string eigenvalues;
	string components;
	string modelIn;
	string modelOut;
	bool aboutOrigin = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-roundtrip"))
			roundTrip = args.pop_string();
		else if(args.if_pop("-eigenvalues"))
			eigenvalues = args.pop_string();
		else if(args.if_pop("-components"))
			components = args.pop_string();
		else if(args.if_pop("-aboutorigin"))
			aboutOrigin = true;
		else if(args.if_pop("-modelin"))
			modelIn = args.pop_string();
		else if(args.if_pop("-modelout"))
			modelOut = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GPCA* pTransform = NULL;
	if(modelIn.length() > 0)
	{
		GDom doc;
		doc.loadJson(modelIn.c_str());
		pTransform = new GPCA(doc.root());
	}
	else
	{
		pTransform = new GPCA(nTargetDims);
		if(aboutOrigin)
			pTransform->aboutOrigin();
		if(eigenvalues.length() > 0)
			pTransform->computeEigVals();
		pTransform->train(*pData);
	}
	Holder<GPCA> hTransform(pTransform);
	pTransform->rand().setSeed(seed);

	GMatrix* pDataAfter = pTransform->transformBatch(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);

	// Save the eigenvalues
	if(eigenvalues.length() > 0)
	{
		GArffRelation* pRelation = new GArffRelation();
		pRelation->addAttribute("eigenvalues", 0, NULL);
		GMatrix dataEigenvalues(pRelation);
		dataEigenvalues.newRows(nTargetDims);
		GVec& pEigVals = pTransform->eigVals();
		for(int i = 0; i < nTargetDims; i++)
			dataEigenvalues[i][0] = pEigVals[i];
		dataEigenvalues.saveArff(eigenvalues.c_str());
	}

	// Save the components
	if(components.length() > 0)
		pTransform->components()->saveArff(components.c_str());

	// Do the round-trip
	if(roundTrip.size() > 0)
	{
		GMatrix roundTripped(pData->rows(), pData->cols());
		for(size_t i = 0; i < pData->rows(); i++)
			pTransform->untransform(pDataAfter->row(i), roundTripped.row(i));
		roundTripped.saveArff(roundTrip.c_str());
	}

	if(modelOut.length() > 0)
	{
		GDom doc;
		doc.setRoot(pTransform->serialize(&doc));
		doc.saveJson(modelOut.c_str());
	}

	pDataAfter->print(cout);
}

void scalingUnfolder(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pData, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	int targetDims = args.pop_uint();

	// Parse Options
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GScalingUnfolder transform;
	transform.rand().setSeed(nSeed);
	transform.setNeighborCount(pNF->neighborCount());
	transform.setTargetDims(targetDims);
	//transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.reduce(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

void selfOrganizingMap(GArgReader& args){
  // Load the file
  GMatrix* pData = loadData(args.pop_string());
  Holder<GMatrix> hData(pData);

  // Parse arguments
  std::vector<double> netDims;
  unsigned numNodes = 1;
  while(args.next_is_uint()){
    unsigned dim = args.pop_uint();
    netDims.push_back(dim);
    numNodes *= dim;
  }
  if(netDims.size() < 1){
    throw Ex("No dimensions specified for self organizing map.  ",
	       "A map must be at least 1 dimensional.");
  }

  GRand rand(getpid() * (unsigned int)time(NULL));
  Holder<SOM::ReporterChain> reporters(new SOM::ReporterChain);
  Holder<SOM::TrainingAlgorithm> alg(NULL);
  Holder<GDistanceMetric> weightDist(new GRowDistance);
  Holder<GDistanceMetric> nodeDist(new GRowDistance);
  Holder<SOM::NodeLocationInitialization> topology(new SOM::GridTopology);
  Holder<SOM::NodeWeightInitialization> weightInit
    (new SOM::NodeWeightInitializationTrainingSetSample(rand));
  Holder<SOM::NeighborhoodWindowFunction>
    windowFunc(new SOM::GaussianWindowFunction());

  //Loading and saving
  string loadFrom = "";
  string saveTo = "";

  //Parameters for different training algorithms
  string algoName = "batch";
  double startWidth = -1;//Start width - set later if still negative
  double endWidth   = -1;//End width   - set later if still negative
  double startRate = -1;//Start learning rate
  double endRate   = -1;//End learning rate
  unsigned numIter     = 100;//Total iterations
  unsigned numConverge = 1;//#steps for batch to converge

  while(args.next_is_flag()){
    if(args.if_pop("-tofile")){
      saveTo = args.pop_string();
    }else if(args.if_pop("-fromfile")){
      loadFrom = args.pop_string();
    }else if(args.if_pop("-seed")){
      rand.setSeed(args.pop_uint());
    }else if(args.if_pop("-neighborhood")){
      string name = args.pop_string();
      if(name == "gaussian"){
	windowFunc.reset(new SOM::GaussianWindowFunction());
      }else if(name == "uniform"){
	windowFunc.reset(new SOM::UniformWindowFunction());
      }else{
	throw Ex("Only gaussian and uniform are acceptible ",
		   "neighborhood types");
      }
    }else if(args.if_pop("-printMeshEvery")){
      using namespace SOM;
      unsigned interval = args.pop_uint();
      string baseFilename = args.pop_string();
      unsigned xDim = args.pop_uint();
      unsigned yDim = args.pop_uint();
      bool showTrain = false;
      if(args.if_pop("showTrain") || args.if_pop("showtrain")){
	showTrain = true;
      }
      shared_ptr<Reporter> weightReporter
	(new SVG2DWeightReporter(baseFilename, xDim, yDim, showTrain));
      Holder<IterationIntervalReporter> intervalReporter
	(new IterationIntervalReporter(weightReporter, interval));
      reporters->add(intervalReporter.release());
    }else if(args.if_pop("-batchTrain")){
      algoName = "batch";
      startWidth = args.pop_double();
      endWidth = args.pop_double();
      numIter = args.pop_uint();
      numConverge = args.pop_uint();
    }else if(args.if_pop("-stdTrain")){
      algoName = "standard";
      startWidth = args.pop_double();
      endWidth = args.pop_double();
      startRate = args.pop_double();
      endRate = args.pop_double();
      numIter = args.pop_uint();
    }else{
      throw Ex("Invalid option: ", args.peek());
    }
  }

  //Create the training algorithm
  Holder<SOM::TrainingAlgorithm> algo;
  if(algoName == "batch"){
    double netRadius = *std::max_element(netDims.begin(), netDims.end());
    if(startWidth < 0){ startWidth = 2*netRadius; }
    if(endWidth < 0){ endWidth = 1; }
    algo.reset( new SOM::BatchTraining
      (startWidth, endWidth, numIter, numConverge,
       weightInit.release(), windowFunc.release(),
       reporters.release()));
  }else if(algoName == "standard"){
    algo.reset( new SOM::TraditionalTraining
      (startWidth, endWidth, startRate, endRate, numIter,
       weightInit.release(), windowFunc.release(),
       reporters.release()));
  }else{
    throw Ex("Unknown type of training algorithm: \"",
	       algoName, "\"");
  }

  //Create the network & transform the data
  Holder<GSelfOrganizingMap> som;
  Holder<GMatrix> out;

  if(loadFrom == ""){
    //Create map from arguments given
    som.reset(new GSelfOrganizingMap
      (netDims, numNodes, topology.release(), algo.release(),
       weightDist.release(), nodeDist.release()));
    //Train the network and transform the data in place
    out.reset(som->reduce(*pData));
  }else{
    //Create map from file
    GDom source;
    source.loadJson(loadFrom.c_str());
    som.reset(new GSelfOrganizingMap(source.root()));
    //Transform using the loaded network
    out.reset(som->transformBatch(*pData));
  }

  //Save the trained network
  if(saveTo != ""){
    GDom serialized;
    GDomNode* root = som->serialize(&serialized);
    serialized.setRoot(root);
    serialized.saveJson(saveTo.c_str());
  }

  //Print the result
  out->print(cout);
}

void singularValueDecomposition(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse options
	string ufilename = "u.arff";
	string sigmafilename;
	string vfilename = "v.arff";
	int maxIters = 100;
	while(args.size() > 0)
	{
		if(args.if_pop("-ufilename"))
			ufilename = args.pop_string();
		else if(args.if_pop("-sigmafilename"))
			sigmafilename = args.pop_string();
		else if(args.if_pop("-vfilename"))
			vfilename = args.pop_string();
		else if(args.if_pop("-maxiters"))
			maxIters = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	GMatrix* pU;
	double* pDiag;
	GMatrix* pV;
	pData->singularValueDecomposition(&pU, &pDiag, &pV, false, maxIters);
	Holder<GMatrix> hU(pU);
	ArrayHolder<double> hDiag(pDiag);
	Holder<GMatrix> hV(pV);
	pU->saveArff(ufilename.c_str());
	pV->saveArff(vfilename.c_str());
	if(sigmafilename.length() > 0)
	{
		GMatrix sigma(pU->rows(), pV->rows());
		sigma.setAll(0.0);
		size_t m = std::min(sigma.rows(), (size_t)sigma.cols());
		for(size_t i = 0; i < m; i++)
			sigma.row(i)[i] = pDiag[i];
		sigma.saveArff(sigmafilename.c_str());
	}
	else
	{
		GVecWrapper diag(pDiag, std::min(pU->rows(), pV->rows()));
		diag.vec().print(cout);
		cout << "\n";
	}
}
/*
void unsupervisedBackProp(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int targetDims = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GUnsupervisedBackProp* pUBP = new GUnsupervisedBackProp(targetDims, &prng);
	Holder<GUnsupervisedBackProp> hUBP(pUBP);
	vector<size_t> paramRanges;
	string sModelOut;
	string sProgress;
	bool inputBias = true;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else if(args.if_pop("-addlayer"))
			pUBP->neuralNet()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, args.pop_uint()));
		else if(args.if_pop("-params"))
		{
			if(pUBP->jitterer())
				throw Ex("You can't change the params after you add an image jitterer");
			size_t paramDims = args.pop_uint();
			for(size_t i = 0; i < paramDims; i++)
				paramRanges.push_back(args.pop_uint());
		}
		else if(args.if_pop("-modelin"))
		{
			GDom doc;
			doc.loadJson(args.pop_string());
			pUBP = new GUnsupervisedBackProp(doc.root());
			hUBP.reset(pUBP);
		}
		else if(args.if_pop("-modelout"))
			sModelOut = args.pop_string();
		else if(args.if_pop("-intrinsicin"))
		{
			GMatrix* pInt = new GMatrix();
			pInt->loadArff(args.pop_string());
			pUBP->setIntrinsic(pInt);
		}
		else if(args.if_pop("-jitter"))
		{
			if(paramRanges.size() != 2)
				throw Ex("The params must be set to 2 before a tweaker is set");
			size_t channels = args.pop_uint();
			double rot = args.pop_double();
			double trans = args.pop_double();
			double zoom = args.pop_double();
			GImageJitterer* pJitterer = new GImageJitterer(paramRanges[0], paramRanges[1], channels, rot, trans, zoom);
			pUBP->setJitterer(pJitterer);
		}
		else if(args.if_pop("-noinputbias"))
			inputBias = false;
		else if(args.if_pop("-progress"))
		{
			sProgress = args.pop_string();
			pUBP->trackProgress();
		}
		else if(args.if_pop("-onepass"))
			pUBP->onePass();
		else
			throw Ex("Invalid option: ", args.peek());
	}
	pUBP->setParams(paramRanges);
	pUBP->setUseInputBias(inputBias);
	pUBP->neuralNet()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));

	// Transform the data
	GMatrix* pDataAfter = pUBP->reduce(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);

	// Save the model (if requested)
	if(sModelOut.length() > 0)
	{
		GDom doc;
		doc.setRoot(pUBP->serialize(&doc));
		doc.saveJson(sModelOut.c_str());
	}
	if(sProgress.length() > 0)
		pUBP->progress().saveArff(sProgress.c_str());
}
*/
/*
void autoencoder(GArgReader& args)
{
	// Load the file
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int targetDims = args.pop_uint();

	// Parse options
	string roundTrip;
	size_t seed = getpid() * (unsigned int)time(NULL);
	string eigenvalues;
	string components;
	string modelIn;
	string modelOut;
	bool aboutOrigin = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-roundtrip"))
			roundTrip = args.pop_string();
		else if(args.if_pop("-eigenvalues"))
			eigenvalues = args.pop_string();
		else if(args.if_pop("-components"))
			components = args.pop_string();
		else if(args.if_pop("-aboutorigin"))
			aboutOrigin = true;
		else if(args.if_pop("-modelin"))
			modelIn = args.pop_string();
		else if(args.if_pop("-modelout"))
			modelOut = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}


	// Make the encoder
	GNeuralNet encoder;
	encoder.addLayer(new GLayerClassic(FLEXIBLE_SIZE, std::max(pData->cols() / 2, targetDims * 2), new GActivationHinge()));
	encoder.addLayer(new GLayerClassic(FLEXIBLE_SIZE, targetDims * 2, new GActivationHinge()));
	encoder.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE, new GActivationHinge()));
	encoder.setLearningRate(0.01);
	GUniformRelation relObs(pData->cols());
	GUniformRelation relInt(targetDims);
	encoder.beginIncrementalLearning(relObs, relInt);

	// Make the decoder
	GNeuralNet decoder;
	decoder.addLayer(new GLayerClassic(FLEXIBLE_SIZE, targetDims * 2, new GActivationHinge()));
	decoder.addLayer(new GLayerClassic(FLEXIBLE_SIZE, std::max(pData->cols() / 2, targetDims * 2), new GActivationHinge()));
	decoder.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE, new GActivationHinge()));
	decoder.setLearningRate(0.01);
	encoder.beginIncrementalLearning(relInt, relObs);

	// Train
	GRandomIndexIterator ii(pData->rows(), encoder.rand());
	double prevRmse = 1e200;
	while(true)
	{
		double sse = 0.0;
		ii.reset();
		size_t index;
		while(ii.next(index))
		{
			const double* pIn = observations[index];
			encoder.forwardProp(pIn);
			decoder.forwardProp(encoder.outputLayer().activation());
			sse += decoder.sumSquaredPredictionError(pIn);
			decoder.backpropagateAndRefineActivationFunction(pIn, decoder.learningRate());
			encoder.backpropagateFromLayer(&decoder.layer(0), encoder.learningRate());
			encoder.descendGradient(pIn, encoder.learningRate(), 0.0);
			decoder.descendGradient(encoder.outputLayer().activation(), decoder.learningRate(), 0.0);
		}
		double rmse = sqrt(sse / observations.rows());
		cout << "RMSE = " << to_str(rmse) << "\n";
		if(1.0 - (rmse / prevRmse) < 0.001)
			break;
		prevRmse = rmse;
	}
}
*/
void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeDimRedUsageTree();
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
	UsageNode* pUsageTree = makeDimRedUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << szAppName << " ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, true);
			if(pUsageCommand->findPart("[neighbor-finder]") >= 0)
			{
				UsageNode* pNFTree = makeNeighborUsageTree();
				Holder<UsageNode> hNFTree(pNFTree);
				pNFTree->print(cerr, 1, 3, 76, 2, false);
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
	int ret = 0;
	PathData pd;
	GFile::parsePath(argv[0], &pd);
	const char* appName = argv[0] + pd.fileStart;
	GArgReader args(argc, argv);
	args.pop_string(); // advance past the app name
	try
	{
		if(args.size() < 1) throw Ex("Expected a command");
		else if(args.if_pop("usage")) ShowUsage(appName);
		else if(args.if_pop("attributeselector")) attributeSelector(args);
		else if(args.if_pop("blendembeddings")) blendEmbeddings(args);
		else if(args.if_pop("breadthfirstunfolding")) breadthFirstUnfolding(args);
		else if(args.if_pop("curviness2")) curviness2(args);
		else if(args.if_pop("isomap")) isomap(args);
		else if(args.if_pop("lle")) lle(args);
		else if(args.if_pop("manifoldsculpting")) ManifoldSculpting(args);
		else if(args.if_pop("multidimensionalscaling")) multiDimensionalScaling(args);
		else if(args.if_pop("neuropca")) neuroPCA(args);
		else if(args.if_pop("pca")) principalComponentAnalysis(args);
		else if(args.if_pop("scalingunfolder")) scalingUnfolder(args);
		else if(args.if_pop("svd")) singularValueDecomposition(args);
		else if(args.if_pop("som")) selfOrganizingMap(args);
//		else if(args.if_pop("unsupervisedbackprop")) unsupervisedBackProp(args);
//		else if(args.if_pop("autoencoder")) autoencoder(args);
		else throw Ex("Unrecognized command: ", args.peek());
	}
	catch(const std::exception& e)
	{
		if(strcmp(e.what(), "nevermind") != 0) // if an error message was not already displayed...
			showError(args, appName, e.what());
		ret = 1;
	}

	return ret;
}

