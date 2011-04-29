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
#include "../GClasses/GTwt.h"
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
		if(i > 0)
		{
			if(*szList < '0' || *szList > '9')
				ThrowError("Expected a number");
			if(j < 0)
			{
#ifdef WIN32
				size_t val = (size_t)_strtoui64(szList, (char**)NULL, 10);
#else
				size_t val = strtoull(szList, (char**)NULL, 10);
#endif
				if(val >= attrCount)
					ThrowError("Invalid column index: ", to_str(val), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
				if(attrSet.find(val) != attrSet.end())
					ThrowError("Columns ", to_str(val), " is listed multiple times");
				attrSet.insert(val);
				list.push_back(val);
			}
			else
			{
				if(szList[j + 1] < '0' || szList[j + 1] > '9')
					ThrowError("Expected a number");
#ifdef WIN32
				size_t beg = (size_t)_strtoui64(szList, (char**)NULL, 10);
				size_t end = (size_t)_strtoui64(szList + j + 1, (char**)NULL, 10);
#else
				size_t beg = strtoull(szList, (char**)NULL, 10);
				size_t end = strtoull(szList + j + 1, (char**)NULL, 10);
#endif
				if(beg >= attrCount)
					ThrowError("Invalid column index: ", to_str(beg), ". Valid values are from 0 to ", to_str(attrCount - 1), ". (Columns are zero-indexed.)");
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

GMatrix* loadDataWithSwitches(GArgReader& args, size_t* pLabelDims)
{
	// Load the dataset by extension
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
			break;
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
	*pLabelDims = std::max((size_t)1, labels.size());
	for(size_t i = 0; i < labels.size(); i++)
	{
		size_t src = labels[i];
		size_t dst = pData->cols() - *pLabelDims + i;
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

	return hData.release();
}

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
				ThrowError("Invalid neighbor finder option: ", args.peek());
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
		else if(_stricmp(alg, "manifold") == 0)
		{
			size_t medianCands = args.pop_uint();
			size_t neighbors = args.pop_uint();
			size_t tangentSpaceDims = args.pop_uint();
			double thresh = args.pop_double();
			pNF = new GManifoldNeighborFinder(pData, medianCands, neighbors, tangentSpaceDims, thresh, pRand);
		}
		else if(_stricmp(alg, "system") == 0)
		{
			GMatrix* pControlData = loadData(args.pop_string());
			Holder<GMatrix> hControlData(pControlData);
			if(pControlData->rows() != pData->rows())
				ThrowError("mismatching number of rows");
			int neighbors = args.pop_uint();
			pNF = new GDynamicSystemNeighborFinder(pData, hControlData.release(), true, neighbors, pRand);
		}
		else
			ThrowError("Unrecognized neighbor finding algorithm: ", alg);
	
		// Normalize
		if(normalize)
		{
			GNeighborFinderCacheWrapper* pNF2 = new GNeighborFinderCacheWrapper(pNF, true);
			pNF2->fillCache();
			pNF2->normalizeDistances();
			pNF = pNF2;
		}
	
		// Apply CycleCut
		if(cutCycleLen > 0)
		{
			GNeighborFinderCacheWrapper* pNF2 = new GNeighborFinderCacheWrapper(pNF, true);
			pNF2->fillCache();
			pNF2->cutShortcuts(cutCycleLen);
			pNF = pNF2;
		}
	}
	catch(const std::exception& e)
	{
		args.set_pos(argPos);
		showInstantiateNeighborFinderError(e.what(), args);
		ThrowError("nevermind"); // this means "don't display another error message"
	}

	return pNF;
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

void AddIndexAttribute(GArgReader& args)
{
	// Parse args
	const char* filename = args.pop_string();
	double nStartValue = 0.0;
	double nIncrement = 1.0;
	while(args.size() > 0)
	{
		if(args.if_pop("-start"))
			nStartValue = args.pop_double();
		else if(args.if_pop("-increment"))
			nIncrement = args.pop_double();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	GMatrix* pData = loadData(filename);
	Holder<GMatrix> hData(pData);
	GArffRelation* pIndexRelation = new GArffRelation();
	pIndexRelation->addAttribute("index", 0, NULL);
	sp_relation pIndexRel = pIndexRelation;
	GMatrix indexes(pIndexRel);
	indexes.newRows(pData->rows());
	for(size_t i = 0; i < pData->rows(); i++)
		indexes.row(i)[0] = nStartValue + i * nIncrement;
	GMatrix* pUnified = GMatrix::mergeHoriz(&indexes, pData);
	Holder<GMatrix> hUnified(pUnified);
	pUnified->print(cout);
}

void addMatrices(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	GMatrix* pB = loadData(args.pop_string());
	Holder<GMatrix> hB(pB);
	pA->add(pB, false);
	pA->print(cout);
}

void addNoise(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	double dev = args.pop_double();

	// Parse the options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	int excludeLast = 0;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-excludelast"))
			excludeLast = args.pop_uint();
		else
			ThrowError("Invalid neighbor finder option: ", args.peek());
	}

	GRand prng(seed);
	size_t cols = pData->cols() - excludeLast;
	for(size_t r = 0; r < pData->rows(); r++)
	{
		double* pRow = pData->row(r);
		for(size_t c = 0; c < cols; c++)
			*(pRow++) += dev * prng.normal();
	}
	pData->print(cout);
}

void align(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	GMatrix* pB = loadData(args.pop_string());
	Holder<GMatrix> hB(pB);
	GMatrix* pC = GMatrix::align(pA, pB);
	Holder<GMatrix> hC(pC);
	pC->print(cout);
}

void attributeSelector(GArgReader& args)
{
	// Load the data
	size_t labelDims;
	GMatrix* pData = loadDataWithSwitches(args, &labelDims);
	Holder<GMatrix> hData(pData);

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
			ThrowError("Invalid neighbor finder option: ", args.peek());
	}

	// Do the attribute selection
	GRand prng(seed);
	GAttributeSelector as(labelDims, targetFeatures, &prng);
	if(outFilename.length() > 0)
	{
		as.train(*pData);
		GMatrix* pDataOut = as.transformBatch(*pData);
		Holder<GMatrix> hDataOut(pDataOut);
		cout << "Reduced data saved to " << outFilename.c_str() << ".\n";
		pDataOut->saveArff(outFilename.c_str());
	}
	else
		as.train(*pData);
	cout << "\nAttribute rankings from most salient to least salient. (Attributes are zero-indexed.)\n";
	GArffRelation* pRel = (GArffRelation*)pData->relation().get();
	for(size_t i = 0; i < as.ranks().size(); i++)
		cout << as.ranks()[i] << " " << pRel->attrName(as.ranks()[i]) << "\n";
}

void autoCorrelation(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t lag = std::min((size_t)256, pData->rows() / 2);
	size_t dims = pData->cols();
	GTEMPBUF(double, mean, dims);
	pData->centroid(mean);
	GMatrix ac(0, dims + 1);
	for(size_t i = 1; i <= lag; i++)
	{
		double* pRow = ac.newRow();
		*(pRow++) = (double)i;
		for(size_t j = 0; j < dims; j++)
		{
			*pRow = 0;
			size_t k;
			for(k = 0; k + i < pData->rows(); k++)
			{
				double* pA = pData->row(k);
				double* pB = pData->row(k + i);
				*pRow += (pA[j] - mean[j]) * (pB[j] - mean[j]);
			}
			*pRow /= k;
			pRow++;
		}
	}
	ac.print(cout);
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
		ThrowError("mismatching number of rows");
	if(pDataA->cols() != pDataB->cols())
		ThrowError("mismatching number of cols");

	// Parse Options
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Get a neighbor table
	if(!pNF->isCached())
	{
		GNeighborFinderCacheWrapper* pNF2 = new GNeighborFinderCacheWrapper(hNF.release(), true);
		hNF.reset(pNF2);
		pNF = pNF2;
	}
	((GNeighborFinderCacheWrapper*)pNF)->fillCache();
	size_t* pNeighborTable = ((GNeighborFinderCacheWrapper*)pNF)->cache();

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
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GNeighborFinder* pNF = instantiateNeighborFinder(pData, &prng, args);
	Holder<GNeighborFinder> hNF(pNF);
	int targetDims = args.pop_uint();

	// Parse Options
	int reps = 1;
	Holder<GMatrix> hControlData(NULL);
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else if(args.if_pop("-reps"))
			reps = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Transform the data
	GBreadthFirstUnfolding transform(reps, pNF->neighborCount(), targetDims, &prng);
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.doit(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

///TODO: this command should be documented
void center(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	unsigned int r = args.pop_uint();
	size_t cols = pData->cols();
	double* pRow = pData->row(r);
	for(size_t i = 0; i < r; ++i)
		GVec::subtract(pData->row(i), pRow, cols);
	for(size_t i = r + 1; i < pData->rows(); ++i)
		GVec::subtract(pData->row(i), pRow, cols);
	GVec::setAll(pRow, 0.0, cols);
	pData->print(cout);
}

void cholesky(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	pA->cholesky();
	pA->print(cout);
}

void correlation(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	int attr1 = args.pop_uint();
	int attr2 = args.pop_uint();

	// Parse Options
	bool aboutorigin = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-aboutorigin"))
			aboutorigin = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	double m1, m2;
	if(aboutorigin)
	{
		m1 = 0;
		m2 = 0;
	}
	else
	{
		m1 = pA->mean(attr1);
		m2 = pA->mean(attr2);
	}
	double corr = pA->linearCorrelationCoefficient(attr1, m1, attr2, m2);
	cout.precision(14);
	cout << corr << "\n";
}

void cumulativeColumns(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	vector<size_t> cols;
	parseAttributeList(cols, args, pA->cols());
	double* pPrevRow = pA->row(0);
	for(size_t i = 1; i < pA->rows(); i++)
	{
		double* pRow = pA->row(i);
		for(vector<size_t>::iterator it = cols.begin(); it != cols.end(); it++)
			pRow[*it] += pPrevRow[*it];
		pPrevRow = pRow;
	}
	pA->print(cout);
}

void determinant(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	double d = pA->determinant();
	cout.precision(14);
	cout << d << "\n";
}

void Discretize(GArgReader& args)
{
	// Load the file
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse Options
	size_t nFirst = 0;
	size_t nLast = pData->relation()->size() - 1;
	size_t nBuckets = std::max(2, (int)floor(sqrt((double)pData->rows() + 0.5)));
	while(args.size() > 0)
	{
		if(args.if_pop("-buckets"))
			nBuckets = args.pop_uint();
		else if(args.if_pop("-colrange"))
		{
			nFirst = args.pop_uint();
			nLast = args.pop_uint();
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}
	if(nFirst < 0 || nLast >= pData->relation()->size() || nLast < nFirst)
		ThrowError("column index out of range");

	// Discretize the continuous attributes in the specified range
	for(size_t i = nFirst; i <= nLast; i++)
	{
		if(pData->relation()->valueCount(i) != 0)
			continue;
		double min, range;
		pData->minAndRange(i, &min, &range);
		for(size_t j = 0; j < pData->rows(); j++)
		{
			double* pPat = pData->row(j);
			pPat[i] = (double)std::max((size_t)0, std::min(nBuckets - 1, (size_t)floor(((pPat[i] - min) * nBuckets) / range)));
		}
		((GArffRelation*)pData->relation().get())->setAttrValueCount(i, nBuckets);
	}

	// Print results
	pData->print(cout);
}

void dropColumns(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	vector<size_t> colList;
	size_t attrCount = pData->cols();
	parseAttributeList(colList, args, attrCount);
	std::sort(colList.begin(), colList.end());
	std::reverse(colList.begin(), colList.end());
	for(size_t i = 0; i < colList.size(); i++)
		pData->deleteColumn(colList[i]);
	pData->print(cout);
}

void DropMissingValues(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GRelation* pRelation = pData->relation().get();
	size_t dims = pRelation->size();
	for(size_t i = pData->rows() - 1; i < pData->rows(); i--)
	{
		double* pPat = pData->row(i);
		bool drop = false;
		for(size_t j = 0; j < dims; j++)
		{
			if(pRelation->valueCount(j) == 0)
			{
				if(pPat[j] == UNKNOWN_REAL_VALUE)
				{
					drop = true;
					break;
				}
			}
			else
			{
				if(pPat[j] == UNKNOWN_DISCRETE_VALUE)
				{
					drop = true;
					break;
				}
			}
		}
		if(drop)
			pData->deleteRow(i);
	}
	pData->print(cout);
}

void dropRows(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t newSize = args.pop_uint();
	while(pData->rows() > newSize)
		pData->deleteRow(pData->rows() - 1);
	pData->print(cout);
}

void enumerateValues(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t col = args.pop_uint();
	if(pData->relation()->valueCount(col) > 0)
		((GArffRelation*)pData->relation().get())->setAttrValueCount(col, 0);
	else
	{
		size_t n = 0;
		map<double,size_t> themap;
		for(size_t i = 0; i < pData->rows(); i++)
		{
			double* pRow = pData->row(i);
			map<double,size_t>::iterator it = themap.find(pRow[col]);
			if(it == themap.end())
			{
				themap[pRow[col]] = n;
				pRow[col] = (double)n;
				n++;
			}
			else
				pRow[col] = (double)it->second;
		}
	}
	pData->print(cout);
}

void Export(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse options
	const char* separator = ",";
	while(args.size() > 0)
	{
		if(args.if_pop("-tab"))
			separator = "	";
		else if(args.if_pop("-space"))
			separator = " ";
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Print
	for(size_t i = 0; i < pData->rows(); i++)
		pData->relation()->printRow(cout, pData->row(i), separator);
}

void Import(GArgReader& args)
{
	// Load the file
	size_t len;
	const char* filename = args.pop_string();
	char* pFile = GFile::loadFile(filename, &len);
	ArrayHolder<char> hFile(pFile);

	// Parse Options
	char separator = ',';
	bool tolerant = false;
	bool columnNamesInFirstRow = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-tab"))
			separator = '\t';
		else if(args.if_pop("-space"))
			separator = ' ';
		else if(args.if_pop("-whitespace"))
			separator = '\0';
		else if(args.if_pop("-semicolon"))
			separator = ';';
		else if(args.if_pop("-separator"))
			separator = args.pop_string()[0];
		else if(args.if_pop("-tolerant"))
			tolerant = true;
		else if(args.if_pop("-columnnames"))
			columnNamesInFirstRow = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Parse the file
	GMatrix* pData = GMatrix::parseCsv(pFile, len, separator, columnNamesInFirstRow, tolerant);
	Holder<GMatrix> hData(pData);
	((GArffRelation*)pData->relation().get())->setName(filename);

	// Print the data
	pData->print(cout);
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
			ThrowError("Invalid option: ", args.peek());
	}

	// Transform the data
	GIsomap transform(pNF->neighborCount(), targetDims, &prng);
	transform.setNeighborFinder(pNF);
	if(tolerant)
		transform.dropDisconnectedPoints();
	GMatrix* pDataAfter = transform.doit(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

void kmeans(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int clusters = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Do the clustering
	GRand prng(nSeed);
	GKMeans clusterer(clusters, &prng);
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
			ThrowError("Invalid option: ", args.peek());
	}

	// Transform the data
	GLLE transform(pNF->neighborCount(), targetDims, &prng);
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.doit(*pData);
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
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the hint data
	GMatrix* pDataHint = NULL;
	Holder<GMatrix> hDataHint(NULL);
	if(szPreprocessedData)
	{
		pDataHint = loadData(szPreprocessedData);
		hDataHint.reset(pDataHint);
		if(pDataHint->relation()->size() != targetDims)
			ThrowError("Wrong number of dims in the hint data");
		if(pDataHint->rows() != pData->rows())
			ThrowError("Wrong number of patterns in the hint data");
	}

	// Transform the data
	GManifoldSculpting transform(pNF->neighborCount(), targetDims, &prng);
	transform.setSquishingRate(scaleRate);
	if(pDataHint)
		transform.setPreprocessedData(hDataHint.release());
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.doit(*pData);
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
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the hint data
	GMatrix* pDataHint = NULL;
	Holder<GMatrix> hDataHint(NULL);
	if(szPreprocessedData)
	{
		pDataHint = loadData(szPreprocessedData);
		hDataHint.reset(pDataHint);
		if(pDataHint->relation()->size() != targetDims)
			ThrowError("Wrong number of dims in the hint data");
		if(pDataHint->rows() != pDataObs->rows())
			ThrowError("Wrong number of patterns in the hint data");
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
	GMatrix* pDataAfter = transform.doit(pDataObs);
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
			ThrowError("Invalid option: ", args.peek());
	}

	// Transform the data
	GManifoldUnfolder transform(pNF->neighborCount(), targetDims, &prng);
	transform.setNeighborFinder(pNF);
	GMatrix* pDataAfter = transform.doit(pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}
*/
void ComputeMeanSquaredError(GMatrix* pData1, GMatrix* pData2, size_t dims, double* pResults)
{
	GVec::setAll(pResults, 0.0, dims);
	for(size_t i = 0; i < pData1->rows(); i++)
	{
		double* pPat1 = pData1->row(i);
		double* pPat2 = pData2->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			double d = (*pPat1 - *pPat2);
			pResults[j] += (d * d);
			pPat1++;
			pPat2++;
		}
	}
	GVec::multiply(pResults, 1.0 / pData1->rows(), dims);
}

class FitDataCritic : public GTargetFunction
{
protected:
	GMatrix* m_pData1;
	GMatrix* m_pData2;
	size_t m_attrs;
	GMatrix m_transformed;
	GMatrix m_transform;
	double* m_pResults;

public:
	FitDataCritic(GMatrix* pData1, GMatrix* pData2, size_t attrs)
	: GTargetFunction(attrs + attrs * attrs), m_pData1(pData1), m_pData2(pData2), m_attrs(attrs), m_transformed(pData1->rows(), attrs), m_transform(attrs, attrs)
	{
		m_transform.makeIdentity();
		m_pResults = new double[attrs];
	}

	virtual ~FitDataCritic()
	{
		delete[] m_pResults;
	}

	virtual bool isStable() { return true; }
	virtual bool isConstrained() { return false; }

	virtual void initVector(double* pVector)
	{
		GVec::setAll(pVector, 0.0, m_attrs);
		m_transform.toVector(pVector + m_attrs);
	}

	void TransformData(const double* pVector)
	{
		m_transform.fromVector(pVector + m_attrs, m_attrs);
		for(size_t i = 0; i < m_pData2->rows(); i++)
		{
			double* pPatIn = m_pData2->row(i);
			double* pPatOut = m_transformed.row(i);
			m_transform.multiply(pPatIn, pPatOut);
			GVec::add(pPatOut, pVector, m_attrs);
		}
	}

	virtual double computeError(const double* pVector)
	{
		TransformData(pVector);
		ComputeMeanSquaredError(m_pData1, &m_transformed, m_attrs, m_pResults);
		double sum = GVec::sumElements(m_pResults, m_attrs);
		return sum;
	}

	void ShowResults(const double* pVector)
	{
		TransformData(pVector);
		ComputeMeanSquaredError(m_pData1, &m_transformed, m_attrs, m_pResults);
		//GVec::print(stdout, m_pResults, m_attrs);
		cout.precision(14);
		cout << GVec::sumElements(m_pResults, m_attrs) << "\n";
	}

	const double* GetResults() { return m_pResults; }
};

void MeasureMeanSquaredError(GArgReader& args)
{
	// Load the first file
	GMatrix* pData1 = loadData(args.pop_string());
	Holder<GMatrix> hData1(pData1);

	// Load the second file
	GMatrix* pData2 = loadData(args.pop_string());
	Holder<GMatrix> hData2(pData2);

	// check sizes
	if(pData1->relation()->size() != pData2->relation()->size())
		ThrowError("The datasets must have the same number of dims");
	if(pData1->rows() != pData2->rows())
		ThrowError("The datasets must have the same size");

	// Parse Options
	bool fit = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-fit"))
			fit = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	size_t dims = pData1->relation()->size();
	if(fit)
	{
		FitDataCritic critic(pData1, pData2, dims);
		GHillClimber search(&critic);

		double dPrevError;
		double dError = search.iterate();
		cerr.precision(14);
		cerr << dError << "\n";
		cerr.flush();
		while(true)
		{
			dPrevError = dError;
			for(int i = 1; i < 30; i++)
				search.iterate();
			dError = search.iterate();
			cerr << dError << "\n";
			cerr.flush();
			if((dPrevError - dError) / dPrevError < 1e-10)
				break;
		}
		critic.ShowResults(search.currentVector());
	}
	else
	{
		// Compute mean squared error
		GTEMPBUF(double, results, dims);
		ComputeMeanSquaredError(pData1, pData2, dims, results);
		GVec::print(cout, 14, results, dims);
	}
	cout << "\n";
}

void mergeHoriz(GArgReader& args)
{
	GMatrix* pData1 = loadData(args.pop_string());
	Holder<GMatrix> hData1(pData1);
	GMatrix* pMerged = pData1;
	Holder<GMatrix> hMerged(NULL);
	while(args.size() > 0)
	{
		GMatrix* pData2 = loadData(args.pop_string());
		Holder<GMatrix> hData2(pData2);
		if(pMerged->rows() != pData2->rows())
			ThrowError("The datasets must have the same number of rows");
		pMerged = GMatrix::mergeHoriz(pMerged, pData2);
		hMerged.reset(pMerged);
	}
	pMerged->print(cout);
}

void mergeVert(GArgReader& args)
{
	GMatrix* pData1 = loadData(args.pop_string());
	Holder<GMatrix> hData1(pData1);
	GMatrix* pData2 = loadData(args.pop_string());
	Holder<GMatrix> hData2(pData2);
	pData1->mergeVert(pData2);
	pData1->print(cout);
}

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
			ThrowError("Invalid option: ", args.peek());
	}

	GMatrix* pResults = GManifold::multiDimensionalScaling(pDistances, targetDims, &prng, useSquaredDistances);
	Holder<GMatrix> hResults(pResults);
	pResults->print(cout);
}

void multiplyMatrices(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	GMatrix* pB = loadData(args.pop_string());
	Holder<GMatrix> hB(pB);

	// Parse Options
	bool transposeA = false;
	bool transposeB = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-transposea"))
			transposeA = true;
		else if(args.if_pop("-transposeb"))
			transposeB = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	GMatrix* pC = GMatrix::multiply(*pA, *pB, transposeA, transposeB);
	Holder<GMatrix> hC(pC);
	pC->print(cout);
}

void multiplyScalar(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	double scale = args.pop_double();
	pA->multiply(scale);
	pA->print(cout);
}

void zeroMean(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	pA->centerMeanAtOrigin();
	pA->print(cout);
}


void normalize(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	double min = 0.0;
	double max = 1.0;
	while(args.size() > 0)
	{
		if(args.if_pop("-range"))
		{
			min = args.pop_double();
			max = args.pop_double();
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}

	GNormalize transform(min, max);
	transform.train(*pData);
	GMatrix* pOut = transform.transformBatch(*pData);
	Holder<GMatrix> hOut(pOut);
	pOut->print(cout);
}

void neighbors(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int neighborCount = args.pop_uint();

	// Find the neighbors
	GKdTree neighborFinder(pData, neighborCount, NULL, true);
	GTEMPBUF(size_t, neighbors, neighborCount);
	GTEMPBUF(double, distances, neighborCount);
	double sumClosest = 0;
	double sumAll = 0;
	for(size_t i = 0; i < pData->rows(); i++)
	{
		neighborFinder.neighbors(neighbors, distances, i);
		neighborFinder.sortNeighbors(neighbors, distances);
		sumClosest += sqrt(distances[0]);
		for(int j = 0; j < neighborCount; j++)
			sumAll += sqrt(distances[j]);
	}
	cout.precision(14);
	cout << "average closest neighbor distance = " << (sumClosest / pData->rows()) << "\n";
	cout << "average neighbor distance = " << (sumAll / (pData->rows() * neighborCount)) << "\n";
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
			ThrowError("Invalid option: ", args.peek());
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
	GMatrix* pDataAfter = transform.doit(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);

	// Save the eigenvalues
	if(eigenvalues.length() > 0)
	{
		GArffRelation* pRelation = new GArffRelation();
		pRelation->addAttribute("eigenvalues", 0, NULL);
		sp_relation pRel = pRelation;
		GMatrix dataEigenvalues(pRel);
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
			double scal = sqrt(GVec::squaredMagnitude(pWeights->row(i + 1), pWeights->cols()));
			for(size_t j = 0; j < pDataAfter->rows(); j++)
				pDataAfter->row(j)[i] *= scal;
		}
	}

	pDataAfter->print(cout);
}

void nominalToCat(GArgReader& args)
{
	// Load the file
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse Options
	int maxValues = 12;
	while(args.size() > 0)
	{
		if(args.if_pop("-maxvalues"))
			maxValues = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Transform the data
	GNominalToCat transform(maxValues);
	transform.train(*pData);
	GMatrix* pDataNew = transform.transformBatch(*pData);
	Holder<GMatrix> hDataNew(pDataNew);

	// Print results
	pDataNew->print(cout);
}

void powerColumns(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	vector<size_t> cols;
	parseAttributeList(cols, args, pA->cols());
	double exponent = args.pop_double();
	for(size_t i = 0; i < pA->rows(); i++)
	{
		double* pRow = pA->row(i);
		for(vector<size_t>::iterator it = cols.begin(); it != cols.end(); it++)
			pRow[*it] = pow(pRow[*it], exponent);
	}
	pA->print(cout);
}

void principalComponentAnalysis(GArgReader& args)
{
	// Load the file
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int nTargetDims = args.pop_uint();

	// Parse options
	string roundTrip;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	string eigenvalues = "";
	string components = "";
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
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Transform the data
	GRand prng(seed);
	GPCA transform(nTargetDims, &prng);
	if(aboutOrigin)
		transform.aboutOrigin();
	if(eigenvalues.length() > 0)
		transform.computeEigVals();
	transform.train(*pData);
	GMatrix* pDataAfter = transform.transformBatch(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);

	// Save the eigenvalues
	if(eigenvalues.length() > 0)
	{
		GArffRelation* pRelation = new GArffRelation();
		pRelation->addAttribute("eigenvalues", 0, NULL);
		sp_relation pRel = pRelation;
		GMatrix dataEigenvalues(pRel);
		dataEigenvalues.newRows(nTargetDims);
		double* pEigVals = transform.eigVals();
		for(int i = 0; i < nTargetDims; i++)
			dataEigenvalues[i][0] = pEigVals[i];
		dataEigenvalues.saveArff(eigenvalues.c_str());
	}

	// Save the components
	if(components.length() > 0)
		transform.components()->saveArff(components.c_str());

	// Do the round-trip
	if(roundTrip.size() > 0)
	{
		GMatrix roundTripped(pData->rows(), pData->cols());
		for(size_t i = 0; i < pData->rows(); i++)
			transform.untransform(pDataAfter->row(i), roundTripped.row(i));
		roundTripped.saveArff(roundTrip.c_str());
	}
	
	pDataAfter->print(cout);
}

void pseudoInverse(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GMatrix* pInverted = pData->pseudoInverse();
	Holder<GMatrix> hInverted(pInverted);
	pInverted->print(cout);
}

void reducedRowEchelonForm(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	pA->toReducedRowEchelonForm();
	pA->print(cout);
}

void rotate(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	sp_relation relation = pA->relation();
	unsigned colx = args.pop_uint();
	if(colx >= pA->cols()){
	  ThrowError("Rotation first column index (",to_str(colx),") "
		     "should not be greater "
		     "than the largest index, which is ", to_str(pA->cols()-1),
		     ".");
	}
	if(!relation->areContinuous(colx,1)){
	  ThrowError("Rotation first column index (",to_str(colx),") "
		     "should be continuous and it is not.");
		     
	}
	unsigned coly = args.pop_uint();
	if(coly >= pA->cols()){
	  ThrowError("Rotation second column index (",to_str(coly),") "
		     "should not be greater "
		     "than the largest index, which is ", to_str(pA->cols()-1),
		     ".");
	}
	if(!relation->areContinuous(coly,1)){
	  ThrowError("Rotation second column index (",to_str(coly),") "
		     "should be continuous and it is not.");
	}
	
	double angle = args.pop_double();

	angle = angle * M_PI / 180; //Convert from degrees to radians
	double cosAngle = std::cos(angle);
	double sinAngle = std::sin(angle);
	for(std::size_t rowIdx = 0; rowIdx < pA->rows(); ++rowIdx){
	  double* row = (*pA)[rowIdx];
	  double x = row[colx];
	  double y = row[coly];
	  row[colx]=x*cosAngle-y*sinAngle;
	  row[coly]=x*sinAngle+y*cosAngle;
	}
	pA->print(cout);
}

void scaleColumns(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	vector<size_t> cols;
	parseAttributeList(cols, args, pA->cols());
	double scalar = args.pop_double();
	for(size_t i = 0; i < pA->rows(); i++)
	{
		double* pRow = pA->row(i);
		for(vector<size_t>::iterator it = cols.begin(); it != cols.end(); it++)
			pRow[*it] *= scalar;
	}
	pA->print(cout);
}

void shiftColumns(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	vector<size_t> cols;
	parseAttributeList(cols, args, pA->cols());
	double offset = args.pop_double();
	for(size_t i = 0; i < pA->rows(); i++)
	{
		double* pRow = pA->row(i);
		for(vector<size_t>::iterator it = cols.begin(); it != cols.end(); it++)
			pRow[*it] += offset;
	}
	pA->print(cout);
}

void significance(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int attr1 = args.pop_uint();
	int attr2 = args.pop_uint();

	// Parse options
	double tolerance = 0.001;
	while(args.size() > 0)
	{
		if(args.if_pop("-tol"))
			tolerance = args.pop_double();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Perform the significance tests
	cout.precision(8);
	cout << "\n";
	{
		cout << "Means=" << pData->mean(attr1) << ", " << pData->mean(attr2) << "\n";
		size_t v;
		double t;
		pData->pairedTTest(&v, &t, attr1, attr2, false);
		double p = GMath::tTestAlphaValue(v, t);
		cout << "Paired T Test: v=" << v << ", t=" << t << ", p=" << p << "\n";
	}
	{
		size_t v;
		double t;
		pData->pairedTTest(&v, &t, attr1, attr2, true);
		double p = GMath::tTestAlphaValue(v, t);
		cout << "Paired T Test with normalized values: v=" << v << ", t=" << t << ", p=" << p << "\n";
	}
	cout << "\n";
	{
		int less = 0;
		int eq = 0;
		int more = 0;
		for(size_t i = 0; i < pData->rows(); i++)
		{
			double* pRow = pData->row(i);
			if(std::abs(pRow[attr1] - pRow[attr2]) < tolerance)
				eq++;
			else if(pRow[attr1] < pRow[attr2])
				less++;
			else
				more++;
		}
		cout << ((double)less / pData->rows()) << "%% less, " << ((double)eq / pData->rows()) << "%% same, " << ((double)more / pData->rows()) << "%% greater\n";
		double t = pData->wilcoxonSignedRanksTest(attr1, attr2, tolerance);
		int n = (int)pData->rows();
		double a = GMath::wilcoxonAlphaValue(n, t);
		cout << "Wilcoxon Signed Ranks Test: n=" << n << ", t=" << t << ", alpha=" << a << "\n";
	}
	cout << "\n";
}


void selfOrganizingMap(GArgReader& args){
  // Load the file
  GMatrix* pData = loadData(args.pop_string());
  Holder<GMatrix> hData(pData);

  // Parse arguments
  unsigned nDims = args.pop_uint();
  unsigned nodesPerDim = args.pop_uint();

  unsigned int nSeed = getpid() * (unsigned int)time(NULL);
  string toFile=""; //Write the resulting map to the given file
  string fromFile=""; //Read the map from the given file rather than training
  bool printProgress=false;//If true, write progress to stderr
  double focusFactor = 1.0 - (1e-4);
  double learningRate = 1e-2;
  while(args.next_is_flag()){
    if(args.if_pop("-tofile")){
      toFile = args.pop_string();
      ThrowError("Writing maps to a file is not implemented yet, sorry");
    }else if(args.if_pop("-fromfile")){
      fromFile = args.pop_string();
      ThrowError("Reading maps from a file is not implemented yet, sorry");
    }else if(args.if_pop("-seed")){
      nSeed = args.pop_uint();
    }else if(args.if_pop("-focusfactor")){
      focusFactor = args.pop_double();
    }else if(args.if_pop("-learningRate")){
      learningRate = args.pop_double();
    }else if(args.if_pop("-printprogress")){
      printProgress = true;
    }else{
      ThrowError("Invalid option: ", args.peek());
    }
  }

  GRand prng(nSeed);
  GSelfOrganizingMap* som = new GSelfOrganizingMap(nDims, nodesPerDim, &prng);
  som->learningRate(learningRate);
  som->focusFactor(focusFactor);

  //Train the map
  GMatrix* map = som->makeMap(pData,0,printProgress);
  Holder<GMatrix> hMap(map);

  //Transform the data in place
  GMatrix* out = som->doit(pData, map);
  Holder<GMatrix> hOut(out);

  //Print the result
  out->print(cout);
}

void split(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int pats = (int)pData->rows() - args.pop_uint();
	if(pats < 0)
		ThrowError("out of range. The data only has ", to_str(pData->rows()), " rows.");
	const char* szFilename1 = args.pop_string();
	const char* szFilename2 = args.pop_string();

	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	bool shouldShuffle = false;
	while(args.size() > 0){
		if(args.if_pop("-shuffle")){
			shouldShuffle = true;
		}else if(args.if_pop("-seed")){
			nSeed = args.pop_uint();
		}else
			ThrowError("Invalid option: ", args.peek());
	}

	// Shuffle if necessary
	GRand rng(nSeed);
	if(shouldShuffle){
	  pData->shuffle(&rng);
	}

	// Split
	GMatrix other(pData->relation());
	pData->splitBySize(&other, pats);
	pData->saveArff(szFilename1);
	other.saveArff(szFilename2);
}

void splitFold(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t fold = args.pop_uint();
	size_t folds = args.pop_uint();
	if(fold >= folds)
		ThrowError("fold index out of range. It must be less than the total number of folds.");

	// Options
	string filenameTrain = "train.arff";
	string filenameTest = "test.arff";
	while(args.size() > 0)
	{
		if(args.if_pop("-out"))
		{
			filenameTrain = args.pop_string();
			filenameTest = args.pop_string();
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Copy relevant portions of the data
	GMatrix train(pData->relation());
	GMatrix test(pData->relation());
	size_t begin = pData->rows() * fold / folds;
	size_t end = pData->rows() * (fold + 1) / folds;
	for(size_t i = 0; i < begin; i++)
		train.copyRow(pData->row(i));
	for(size_t i = begin; i < end; i++)
		test.copyRow(pData->row(i));
	for(size_t i = end; i < pData->rows(); i++)
		train.copyRow(pData->row(i));
	train.saveArff(filenameTrain.c_str());
	test.saveArff(filenameTest.c_str());
}

void squaredDistance(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	GMatrix* pB = loadData(args.pop_string());
	Holder<GMatrix> hB(pB);
	double d = pA->sumSquaredDifference(*pB, false);
	cout << "Sum squared distance: " << d << "\n";
	cout << "Mean squared distance: " << (d / pA->rows()) << "\n";
	cout << "Root mean squared distance: " << sqrt(d / pA->rows()) << "\n";
}

void fillMissingValues(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	bool random = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else if(args.if_pop("-random"))
			random = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Replace missing values and print
	GRand prng(nSeed);
	if(random)
	{
		for(size_t i = 0; i < pData->relation()->size(); i++)
			pData->replaceMissingValuesRandomly(i, &prng);
	}
	else
	{
		for(size_t i = 0; i < pData->relation()->size(); i++)
			pData->replaceMissingValuesWithBaseline(i);
	}
	pData->print(cout);
}

void Shuffle(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Shuffle and print
	GRand prng(nSeed);
	pData->shuffle(&prng);
	pData->print(cout);
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
			ThrowError("Invalid option: ", args.peek());
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
		GVec::print(cout, 14, pDiag, std::min(pU->rows(), pV->rows()));
		cout << "\n";
	}
}

void SortByAttribute(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t nAttr = args.pop_uint();
	size_t attrCount = pData->relation()->size();
	if(nAttr >= attrCount)
		ThrowError("Index out of range");

	// Parse options
	bool descending = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-descending"))
			descending = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	pData->sort(nAttr);
	if(descending)
		pData->reverseRows();
	pData->print(cout);
}

void SwapAttributes(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t nAttr1 = args.pop_uint();
	size_t nAttr2 = args.pop_uint();
	size_t attrCount = pData->relation()->size();
	if(nAttr1 >= attrCount)
		ThrowError("Index out of range");
	if(nAttr2 >= attrCount)
		ThrowError("Index out of range");
	pData->swapColumns(nAttr1, nAttr2);
	pData->print(cout);
}

void threshold(GArgReader& args){
  GMatrix* pData = loadData(args.pop_string());
  Holder<GMatrix> hData(pData);
  unsigned column=args.pop_uint();
  if(column >= hData->cols()){
    std::stringstream msg;
    if(hData->cols() >= 1){
      msg << "The column to threshold is too large.   It should be in "
	  << "the range [0.." << (hData->cols()-1) << "].";
    }else{
      msg << "This data has no columns to threshold.";
    }
    ThrowError(msg.str());
  }
  if(hData->relation()->valueCount(column) != 0){
    ThrowError("Can only use threshold on continuous attributes.");
  }
  double value = args.pop_double();

  //Do the actual thresholding
  for(size_t i = 0; i < hData->rows(); ++i){
    double& v = hData->row(i)[column];
    if(v <= value){ v = 0;
    }else { v = 1; }
  }

  //Print the data
  hData->print(cout);
}

void Transpose(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GMatrix* pTransposed = pData->transpose();
	Holder<GMatrix> hTransposed(pTransposed);
	pTransposed->print(cout);
}

void transition(GArgReader& args)
{
	// Load the input data
	GMatrix* pActions = loadData(args.pop_string());
	Holder<GMatrix> hActions(pActions);
	GMatrix* pState = loadData(args.pop_string());
	Holder<GMatrix> hState(pState);
	if(pState->rows() != pActions->rows())
		ThrowError("Expected the same number of rows in both datasets");

	// Parse options
	bool delta = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-delta"))
			delta = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Make the output data
	size_t actionDims = pActions->cols();
	size_t stateDims = pState->cols();
	GMixedRelation* pRelation = new GMixedRelation();
	sp_relation pRel = pRelation;
	pRelation->addAttrs(pActions->relation().get());
	pRelation->addAttrs(stateDims + stateDims, 0);
	GMatrix* pTransition = new GMatrix(pRel);
	pTransition->newRows(pActions->rows() - 1);
	for(size_t i = 0; i < pActions->rows() - 1; i++)
	{
		double* pOut = pTransition->row(i);
		GVec::copy(pOut, pActions->row(i), actionDims);
		GVec::copy(pOut + actionDims, pState->row(i), stateDims);
		GVec::copy(pOut + actionDims + stateDims, pState->row(i + 1), stateDims);
		if(delta)
			GVec::subtract(pOut + actionDims + stateDims, pState->row(i), stateDims);
	}
	pTransition->print(cout);
}

void ubpSparse(GArgReader& args)
{
	// Load the sparse matrix
	GTwtDoc doc;
	doc.load(args.pop_string());
	GSparseMatrix* pData = new GSparseMatrix(doc.root());
	Holder<GSparseMatrix> hData(pData);

	// Target dims
	int targetDims = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GUnsupervisedBackProp ubp(targetDims, &prng);
	string modelIn;
	string modelOut;
	bool updateWeights = true;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else if(args.if_pop("-addlayer"))
			ubp.neuralNet()->addLayer(args.pop_uint());
		else if(args.if_pop("-learningrate"))
			ubp.neuralNet()->setLearningRate(args.pop_double());
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
			ubp.neuralNet()->setActivationFunction(pSF, true);
		}
		else if(args.if_pop("-modelin"))
			modelIn = args.pop_string();
		else if(args.if_pop("-modelout"))
			modelOut = args.pop_string();
		else if(args.if_pop("-noupdateweights"))
			updateWeights = false;
		else if(args.if_pop("-windowsize"))
			ubp.neuralNet()->setWindowSize(args.pop_uint());
		else if(args.if_pop("-improvementthresh"))
			ubp.neuralNet()->setImprovementThresh(args.pop_double());
		else if(args.if_pop("-normalize"))
			ubp.normalize(true);
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Load the model
	if(modelIn.length() > 0)
	{
		GTwtDoc doc;
		doc.load(modelIn.c_str());
		GNeuralNet* pNN = new GNeuralNet(doc.root(), &prng);
		ubp.setNeuralNet(pNN);
	}
	ubp.setUpdateWeights(updateWeights);

	// Transform the data
	GMatrix* pDataAfter = ubp.doitSparse(pData);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);

	// Save the model
	if(modelOut.length() > 0)
	{
		GTwtDoc doc;
		doc.setRoot(ubp.neuralNet()->toTwt(&doc));
		doc.save(modelOut.c_str());
	}
}

void ubpSystem(GArgReader& args)
{
	// Load the observations and actions
	GMatrix* pObs = loadData(args.pop_string());
	Holder<GMatrix> hObs(pObs);
	GMatrix* pActions = loadData(args.pop_string());
	Holder<GMatrix> hActions(pActions);

	// Target dims
	int targetDims = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GUnsupervisedBackProp ubp(targetDims, &prng);
	vector<size_t> paramRanges;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else if(args.if_pop("-params"))
		{
			size_t paramDims = args.pop_uint();
			for(size_t i = 0; i < paramDims; i++)
				paramRanges.push_back(args.pop_uint());
		}
		else if(args.if_pop("-addlayer"))
			ubp.neuralNet()->addLayer(args.pop_uint());
		else if(args.if_pop("-learningrate"))
			ubp.neuralNet()->setLearningRate(args.pop_double());
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
			ubp.neuralNet()->setActivationFunction(pSF, true);
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Transform the data
	GMatrix* pDataAfter = ubp.doitCameraSystem(paramRanges, pObs, pActions);
	Holder<GMatrix> hDataAfter(pDataAfter);
	pDataAfter->print(cout);
}

void unsupervisedBackProp(GArgReader& args)
{
	// Load the file and params
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int targetDims = args.pop_uint();

	// Parse Options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	GRand prng(nSeed);
	GUnsupervisedBackProp ubp(targetDims, &prng);
	vector<size_t> paramRanges;
	string roundTrip;
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			prng.setSeed(args.pop_uint());
		else if(args.if_pop("-addlayer"))
			ubp.neuralNet()->addLayer(args.pop_uint());
		else if(args.if_pop("-learningrate"))
			ubp.neuralNet()->setLearningRate(args.pop_double());
		else if(args.if_pop("-rate"))
			ubp.setRate(args.pop_double());
		else if(args.if_pop("-params"))
		{
			size_t paramDims = args.pop_uint();
			for(size_t i = 0; i < paramDims; i++)
				paramRanges.push_back(args.pop_uint());
		}
		else if(args.if_pop("-roundtrip"))
			roundTrip = args.pop_string();
		else
			ThrowError("Invalid option: ", args.peek());
	}
	ubp.setParams(paramRanges);

	// Transform the data
	GMatrix* pDataAfter = ubp.doit(*pData);
	Holder<GMatrix> hDataAfter(pDataAfter);

	// Make round-trip data
	if(roundTrip.size() > 0)
	{
		GMatrix high(pData->rows(), pData->cols());
		for(size_t i = 0; i < pData->rows(); i++)
			ubp.lowToHigh(pDataAfter->row(i), high.row(i));
		high.saveArff(roundTrip.c_str());
	}

	pDataAfter->print(cout);
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeTransformUsageTree();
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
	UsageNode* pUsageTree = makeTransformUsageTree();
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
		if(args.size() < 1) ThrowError("Expected a command");
		else if(args.if_pop("usage")) ShowUsage(appName);
		else if(args.if_pop("add")) addMatrices(args);
		else if(args.if_pop("addindexcolumn")) AddIndexAttribute(args);
		else if(args.if_pop("addnoise")) addNoise(args);
		else if(args.if_pop("agglomerative")) agglomerativeclusterer(args);
		else if(args.if_pop("align")) align(args);
		else if(args.if_pop("attributeselector")) attributeSelector(args);
		else if(args.if_pop("autocorrelation")) autoCorrelation(args);
		else if(args.if_pop("blendembeddings")) blendEmbeddings(args);
		else if(args.if_pop("breadthfirstunfolding")) breadthFirstUnfolding(args);
		else if(args.if_pop("nominaltocat")) nominalToCat(args);
		else if(args.if_pop("center")) center(args);
		else if(args.if_pop("cholesky")) cholesky(args);
		else if(args.if_pop("correlation")) correlation(args);
		else if(args.if_pop("cumulativecolumns")) cumulativeColumns(args);
		else if(args.if_pop("determinant")) determinant(args);
		else if(args.if_pop("discretize")) Discretize(args);
		else if(args.if_pop("dropcolumns")) dropColumns(args);
		else if(args.if_pop("dropmissingvalues")) DropMissingValues(args);
		else if(args.if_pop("droprows")) dropRows(args);
		else if(args.if_pop("enumeratevalues")) enumerateValues(args);
		else if(args.if_pop("export")) Export(args);
		else if(args.if_pop("fillmissingvalues")) fillMissingValues(args);
		else if(args.if_pop("import")) Import(args);
		else if(args.if_pop("isomap")) isomap(args);
		else if(args.if_pop("kmeans")) kmeans(args);
		else if(args.if_pop("kmedoids")) kmedoids(args);
		else if(args.if_pop("lle")) lle(args);
		else if(args.if_pop("manifoldsculpting")) ManifoldSculpting(args);
		else if(args.if_pop("measuremeansquarederror")) MeasureMeanSquaredError(args);
		else if(args.if_pop("mergehoriz")) mergeHoriz(args);
		else if(args.if_pop("mergevert")) mergeVert(args);
		else if(args.if_pop("multidimensionalscaling")) multiDimensionalScaling(args);
		else if(args.if_pop("multiply")) multiplyMatrices(args);
		else if(args.if_pop("multiplyscalar")) multiplyScalar(args);
		else if(args.if_pop("normalize")) normalize(args);
		else if(args.if_pop("neighbors")) neighbors(args);
		else if(args.if_pop("neuropca")) neuroPCA(args);
		else if(args.if_pop("pca")) principalComponentAnalysis(args);
		else if(args.if_pop("powercolumns")) powerColumns(args);
		else if(args.if_pop("pseudoinverse")) pseudoInverse(args);
		else if(args.if_pop("reducedrowechelonform")) reducedRowEchelonForm(args);
		else if(args.if_pop("rotate")) rotate(args);
		else if(args.if_pop("scalecolumns")) scaleColumns(args);
		else if(args.if_pop("shiftcolumns")) shiftColumns(args);
		else if(args.if_pop("shuffle")) Shuffle(args);
		else if(args.if_pop("significance")) significance(args);
		else if(args.if_pop("som")) selfOrganizingMap(args);
		else if(args.if_pop("sortcolumn")) SortByAttribute(args);
		else if(args.if_pop("split")) split(args);
		else if(args.if_pop("splitfold")) splitFold(args);
		else if(args.if_pop("squaredDistance")) squaredDistance(args);
		else if(args.if_pop("svd")) singularValueDecomposition(args);
		else if(args.if_pop("swapcolumns")) SwapAttributes(args);
		else if(args.if_pop("threshold")) threshold(args);
		else if(args.if_pop("transition")) transition(args);
		else if(args.if_pop("transpose")) Transpose(args);
		else if(args.if_pop("ubpsparse")) ubpSparse(args);
		else if(args.if_pop("ubpsystem")) ubpSystem(args);
		else if(args.if_pop("unsupervisedbackprop")) unsupervisedBackProp(args);
		else if(args.if_pop("zeromean")) zeroMean(args);
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

