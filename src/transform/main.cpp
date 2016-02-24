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
#include "../GClasses/GDom.h"
#include "../GClasses/GError.h"
#include "../GClasses/GMatrix.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GFunction.h"
#include "../GClasses/GTransform.h"
#include "../GClasses/GVec.h"
#include "../GClasses/GHashTable.h"
#include "../GClasses/GHolders.h"
#include "../GClasses/GHillClimber.h"
#include "../GClasses/GNeighborFinder.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GHeap.h"
#include "../GClasses/GRect.h"
#include "../GClasses/GSparseMatrix.h"
#include "../GClasses/GMath.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
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

GMatrix* loadData(const char* szFilename)
{
	// Load the dataset by extension
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	GMatrix* pData = new GMatrix();
	Holder<GMatrix> hData(pData);
	if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
		pData->loadArff(szFilename);
	else if(_stricmp(szFilename + pd.extStart, ".csv") == 0)
	{
		GCSVParser parser;
		parser.parse(*pData, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < pData->cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else if(_stricmp(szFilename + pd.extStart, ".dat") == 0)
	{
		GCSVParser parser;
		parser.setSeparator('\0');
		parser.parse(*pData, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < pData->cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else
		throw Ex("Unsupported file format: ", szFilename + pd.extStart);
	return hData.release();
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

void AddIndexAttribute(GArgReader& args)
{
	// Parse args
	const char* filename = args.pop_string();
	double nStartValue = 0.0;
	double nIncrement = 1.0;
	string name = "index";
	while(args.size() > 0)
	{
		if(args.if_pop("-start"))
			nStartValue = args.pop_double();
		else if(args.if_pop("-increment"))
			nIncrement = args.pop_double();
		else if(args.if_pop("-name"))
			name = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	GMatrix* pData = loadData(filename);
	Holder<GMatrix> hData(pData);
	GArffRelation* pIndexRelation = new GArffRelation();
	pIndexRelation->addAttribute(name.c_str(), 0, NULL);
	GMatrix indexes(pIndexRelation);
	indexes.newRows(pData->rows());
	for(size_t i = 0; i < pData->rows(); i++)
		indexes.row(i)[0] = nStartValue + i * nIncrement;
	GMatrix* pUnified = GMatrix::mergeHoriz(&indexes, pData);
	Holder<GMatrix> hUnified(pUnified);
	pUnified->print(cout);
}

void addCategoryColumn(GArgReader& args)
{
	const char* filename = args.pop_string();
	const char* catname = args.pop_string();
	const char* catvalue = args.pop_string();
	GMatrix* pData = loadData(filename);
	Holder<GMatrix> hData(pData);
	GArffRelation* pIndexRelation = new GArffRelation();
	vector<const char*> vals;
	vals.push_back(catvalue);
	pIndexRelation->addAttribute(catname, 1, &vals);
	GMatrix indexes(pIndexRelation);
	indexes.newRows(pData->rows());
	for(size_t i = 0; i < pData->rows(); i++)
		indexes.row(i)[0] = 0.0;
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
			throw Ex("Invalid neighbor finder option: ", args.peek());
	}

	GRand prng(seed);
	size_t cols = pData->cols() - excludeLast;
	for(size_t r = 0; r < pData->rows(); r++)
	{
		GVec& pRow = pData->row(r);
		for(size_t c = 0; c < cols; c++)
			pRow[c] += dev * prng.normal();
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

void autoCorrelation(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t lag = std::min((size_t)256, pData->rows() / 2);
	size_t dims = pData->cols();
	GVec mean(dims);
	pData->centroid(mean);
	GMatrix ac(0, dims + 1);
	for(size_t i = 1; i <= lag; i++)
	{
		GVec& pRow = ac.newRow();
		pRow[0] = (double)i;
		for(size_t j = 0; j < dims; j++)
		{
			pRow[j + 1] = 0;
			size_t k;
			for(k = 0; k + i < pData->rows(); k++)
			{
				GVec& pA = pData->row(k);
				GVec& pB = pData->row(k + i);
				pRow[j + 1] += (pA[j] - mean[j]) * (pB[j] - mean[j]);
			}
			pRow[j + 1] /= k;
		}
	}
	ac.print(cout);
}

///TODO: this command should be documented
void center(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	unsigned int r = args.pop_uint();
	GVec& pRow = pData->row(r);
	for(size_t i = 0; i < r; ++i)
		pData->row(i) -= pRow;
	for(size_t i = r + 1; i < pData->rows(); ++i)
		pData->row(i) -= pRow;
	pRow.fill(0.0);
	pData->print(cout);
}

void cholesky(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	GMatrix* pB = pA->cholesky();
	Holder<GMatrix> hB(pB);
	pB->print(cout);
}

void colstats(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	GMatrix stats(pA->relation().clone());
	stats.newRows(4);
	for(size_t i = 0; i < pA->cols(); i++)
	{
		stats[0][i] = pA->columnMin(i);
		stats[1][i] = pA->columnMax(i);
		stats[2][i] = pA->columnMean(i);
		stats[3][i] = pA->columnMedian(i);
	}
	stats.print(cout);
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
			throw Ex("Invalid option: ", args.peek());
	}

	double m1, m2;
	if(aboutorigin)
	{
		m1 = 0;
		m2 = 0;
	}
	else
	{
		m1 = pA->columnMean(attr1);
		m2 = pA->columnMean(attr2);
	}
	double corr = pA->linearCorrelationCoefficient(attr1, m1, attr2, m2);
	cout.precision(14);
	cout << corr << "\n";
}

void covariance(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	GMatrix* pB = pA->covarianceMatrix();
	Holder<GMatrix> hB(pB);
	pB->print(cout);
}

void cumulativeColumns(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	vector<size_t> cols;
	parseAttributeList(cols, args, pA->cols());
	GVec* pPrevRow = &pA->row(0);
	for(size_t i = 1; i < pA->rows(); i++)
	{
		GVec& pRow = pA->row(i);
		for(vector<size_t>::iterator it = cols.begin(); it != cols.end(); it++)
			pRow[*it] += (*pPrevRow)[*it];
		pPrevRow = &pRow;
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
	size_t nLast = pData->relation().size() - 1;
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
			throw Ex("Invalid option: ", args.peek());
	}
	if(nLast >= pData->relation().size() || nLast < nFirst)
		throw Ex("column index out of range");

	// Discretize the continuous attributes in the specified range
	for(size_t i = nFirst; i <= nLast; i++)
	{
		if(pData->relation().valueCount(i) != 0)
			continue;
		double min = pData->columnMin(i);
		double range = pData->columnMax(i) - min;
		for(size_t j = 0; j < pData->rows(); j++)
		{
			GVec& pPat = pData->row(j);
			pPat[i] = (double)std::max((size_t)0, std::min(nBuckets - 1, (size_t)floor(((pPat[i] - min) * nBuckets) / range)));
		}
		((GArffRelation*)&pData->relation())->setAttrValueCount(i, nBuckets);
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
		pData->deleteColumns(colList[i], 1);
	pData->print(cout);
}

void dropHomogeneousCols(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	vector<size_t> colList;
	size_t attrCount = pData->cols();
	for(size_t i = 0; i < attrCount; i++)
	{
		if(pData->isAttrHomogenous(i))
			colList.push_back(i);
	}
	std::reverse(colList.begin(), colList.end());
	for(size_t i = 0; i < colList.size(); i++)
		pData->deleteColumns(colList[i], 1);
	pData->print(cout);
}

void keepOnlyColumns(GArgReader& args)
{
	//Load the data file and the list of columns to keep
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	vector<size_t> inputColList;
	size_t attrCount = pData->cols();
	parseAttributeList(inputColList, args, attrCount);
	std::set<size_t> colsToKeep(inputColList.begin(), inputColList.end());

	//colsToDel will be a list of the column indices not listed in
	//colsToKeep sorted from largest to smallest
	vector<size_t> colsToDel;
	colsToDel.reserve(attrCount - colsToKeep.size());
	for(size_t i = attrCount-1; i < attrCount; --i){
	  if(colsToKeep.find(i) == colsToKeep.end()){
	    colsToDel.push_back(i);
	  }
	}

	//Delete the columns. Doing it in largest-to-smallest order
	//keeps the column indices for undeleted columns the same even
	//after deletion.
	for(size_t i = 0; i < colsToDel.size(); i++)
		pData->deleteColumns(colsToDel[i], 1);
	pData->print(cout);
}

void DropMissingValues(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	const GRelation* pRelation = &pData->relation();
	size_t dims = pRelation->size();
	for(size_t i = pData->rows() - 1; i < pData->rows(); i--)
	{
		GVec& pPat = pData->row(i);
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

void dropRandomValues(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	double portion = args.pop_double();

	// Parse the options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	GRand rand(seed);
	size_t n = pData->rows() * pData->cols();
	size_t k = size_t(portion * n);
	for(size_t i = 0; i < pData->cols(); i++)
	{
		size_t vals = pData->relation().valueCount(i);
		if(vals == 0)
		{
			for(size_t j = 0; j < pData->rows(); j++)
			{
				if(rand.next(n) < k)
				{
					pData->row(j)[i] = UNKNOWN_REAL_VALUE;
					k--;
				}
				n--;
			}
		}
		else
		{
			for(size_t j = 0; j < pData->rows(); j++)
			{
				if(rand.next(n) < k)
				{
					pData->row(j)[i] = UNKNOWN_DISCRETE_VALUE;
					k--;
				}
				n--;
			}
		}
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

void dropIfTooClose(GArgReader& args)
{
	const char* szDataset = args.pop_string();
	size_t col = args.pop_uint();
	double minGap = args.pop_double();
	GMatrix* pData = loadData(szDataset);
	Holder<GMatrix> hData(pData);
	{
		GMatrix keep(pData->relation().clone());
		GReleaseDataHolder hKeep(&keep);
		keep.takeRow(&pData->row(0));
		GVec& pLastKept = pData->row(0);
		for(size_t i = 1; i < pData->rows(); i++)
		{
			GVec& pCand = pData->row(i);
			if(pCand[col] - pLastKept[col] >= minGap)
			{
				keep.takeRow(&pCand);
				pLastKept.copy(pCand);
			}
		}
		keep.print(cout);
	}
}

void dropUnusedValues(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	for(size_t i = 0; i < pData->cols(); i++)
	{
		size_t valCount = pData->relation().valueCount(i);
		for(size_t j = 0; j < valCount; j++)
		{
			bool gotOne = false;
			for(size_t k = 0; k < pData->rows(); k++)
			{
				if(pData->row(k)[i] == j)
				{
					gotOne = true;
					break;
				}
			}
			if(!gotOne)
			{
				pData->dropValue(i, (int)j);
				j--;
				valCount--;
				GAssert(valCount == pData->relation().valueCount(i));
			}
		}
	}
	pData->print(cout);
}

void enumerateValues(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t col = args.pop_uint();
	if(pData->relation().valueCount(col) > 0)
		((GArffRelation*)&pData->relation())->setAttrValueCount(col, 0);
	else
	{
		size_t n = 0;
		map<double,size_t> themap;
		for(size_t i = 0; i < pData->rows(); i++)
		{
			GVec& pRow = pData->row(i);
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
	const char* missing = "?";
	bool colnames = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-tab"))
			separator = "	";
		else if(args.if_pop("-space"))
			separator = " ";
		else if(args.if_pop("-r"))
			missing = "NA";
		else if(args.if_pop("-columnnames"))
			colnames = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Print column names
	if(colnames)
	{
		size_t c = pData->cols();
		for(size_t i = 0; i < c; i++)
		{
			if(i > 0)
				cout << separator;
			pData->relation().printAttrName(cout, i);
		}
		cout << "\n";
	}

	// Print data
	for(size_t i = 0; i < pData->rows(); i++)
		pData->relation().printRow(cout, pData->row(i).data(), separator, missing);
}

void Import(GArgReader& args)
{
	// Load the file
	size_t len;
	const char* filename = args.pop_string();
	char* pFile = GFile::loadFile(filename, &len);
	ArrayHolder<char> hFile(pFile);

	// Parse Options
	GCSVParser parser;
	char separator = ',';
	bool tolerant = false;
	bool columnNamesInFirstRow = false;
	size_t maxVals = 200;
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
		else if(args.if_pop("-maxvals"))
			maxVals = args.pop_uint();
		else if(args.if_pop("-time"))
		{
			size_t attr = args.pop_uint();
			const char* szFormat = args.pop_string();
			parser.setTimeFormat(attr, szFormat);
		}
		else if(args.if_pop("-nominal"))
		{
			size_t attr = args.pop_uint();
			parser.setNominalAttr(attr);
		}
		else if(args.if_pop("-real"))
		{
			size_t attr = args.pop_uint();
			parser.setRealAttr(attr);
		}
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Parse the file
	GMatrix data;
	parser.setSeparator(separator);
	parser.setMaxVals(maxVals);
	if(tolerant)
		parser.tolerant();
	if(columnNamesInFirstRow)
		parser.columnNamesInFirstRow();
	parser.parse(data, filename);
	cerr << "\nParsing Report:\n";
	for(size_t i = 0; i < data.cols(); i++)
		cerr << to_str(i) << ") " << parser.report(i) << "\n";
	((GArffRelation*)&data.relation())->setName(filename);

	// Print the data
	data.print(cout);
}

void ComputeMeanSquaredError(GMatrix* pData1, GMatrix* pData2, size_t dims, double* pResults)
{
	GVec::setAll(pResults, 0.0, dims);
	for(size_t i = 0; i < pData1->rows(); i++)
	{
		GVec& pPat1 = pData1->row(i);
		GVec& pPat2 = pData2->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			if(pPat1[j] != UNKNOWN_REAL_VALUE && pPat2[j] != UNKNOWN_REAL_VALUE)
			{
				double d = (pPat1[j] - pPat2[j]);
				pResults[j] += (d * d);
			}
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
			GVec& pPatIn = m_pData2->row(i);
			GVec& pPatOut = m_transformed.row(i);
			m_transform.multiply(pPatIn, pPatOut);
			GVec::add(pPatOut.data(), pVector, m_attrs);
		}
	}

	virtual double computeError(const double* pVector)
	{
		TransformData(pVector);
		ComputeMeanSquaredError(m_pData1, &m_transformed, m_attrs, m_pResults);
		double sum = GVec::sumElements(m_pResults, m_attrs);
		return sum;
	}

	void ShowResults(const double* pVector, bool sumOverAttributes)
	{
		TransformData(pVector);
		ComputeMeanSquaredError(m_pData1, &m_transformed, m_attrs, m_pResults);
		cout.precision(14);
		if(sumOverAttributes)
			cout << GVec::sumElements(m_pResults, m_attrs);
		else
		{
			GVecWrapper vw(m_pResults, m_attrs);
			vw.vec().print(cout);
		}
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
	if(pData1->relation().size() != pData2->relation().size())
		throw Ex("The datasets must have the same number of dims");
	if(pData1->rows() != pData2->rows())
		throw Ex("The datasets must have the same size");

	// Parse Options
	bool fit = false;
	bool sumOverAttributes = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-fit"))
			fit = true;
		else if(args.if_pop("-sum"))
			sumOverAttributes = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	size_t dims = pData1->relation().size();
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
		critic.ShowResults(search.currentVector(), sumOverAttributes);
	}
	else
	{
		// Compute mean squared error
		GTEMPBUF(double, results, dims);
		ComputeMeanSquaredError(pData1, pData2, dims, results);
		cout.precision(14);
		if(sumOverAttributes)
			cout << GVec::sumElements(results, dims);
		else
		{
			GVecWrapper vw(results, dims);
			vw.vec().print(cout);
		}
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
			throw Ex("The datasets must have the same number of rows");
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
	
	bool ignoreMismatchingName = false;
	if(args.if_pop("-f"))
	{
		ignoreMismatchingName = true;
	}
	
	pData1->mergeVert(pData2, ignoreMismatchingName);
	pData1->print(cout);
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
			throw Ex("Invalid option: ", args.peek());
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
	if(args.size() > 0)
		throw Ex("Superfluous arg: ", args.pop_string());
	pA->multiply(scale);
	pA->print(cout);
}

void zeroMean(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	if(args.size() > 0)
		throw Ex("Superfluous arg: ", args.pop_string());
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
			throw Ex("Invalid option: ", args.peek());
	}

	GNormalize transform(min, max);
	transform.train(*pData);
	GMatrix* pOut = transform.transformBatch(*pData);
	Holder<GMatrix> hOut(pOut);
	pOut->print(cout);
}

void normalizeMagnitude(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	for(size_t i = 0; i < pData->rows(); i++)
		pData->row(i).normalize();
	pData->print(cout);
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
			throw Ex("Invalid option: ", args.peek());
	}

	// Transform the data
	GNominalToCat transform(maxValues);
	transform.train(*pData);
	GMatrix* pDataNew = transform.transformBatch(*pData);
	Holder<GMatrix> hDataNew(pDataNew);

	// Print results
	pDataNew->print(cout);
}

void obfuscate(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	if(pData->relation().type() != GRelation::ARFF)
		throw Ex("Expected some meta-data");
	GArffRelation* pRel = (GArffRelation*)&pData->relation();
	pRel->setName("Untitled");
	for(size_t i = 0; i < pRel->size(); i++)
	{
		string s = "attr";
		s += to_str(i);
		pRel->setAttrName(i, s.c_str());
		size_t vals = pRel->valueCount(i);
		if(vals > 0)
			pRel->setAttrValueCount(i, vals);
	}
	pData->print(cout);
}

void overlay(GArgReader& args)
{
	GMatrix* pBase = loadData(args.pop_string());
	Holder<GMatrix> hBase(pBase);
	GMatrix* pOver = loadData(args.pop_string());
	Holder<GMatrix> hOver(pOver);
	if(pOver->rows() != pBase->rows() || pOver->cols() != pBase->cols())
		throw Ex("Matrices not the same size");
	size_t dims = pOver->cols();
	const GRelation* pRelOver = &pOver->relation();
	for(size_t i = 0; i < pOver->rows(); i++)
	{
		GVec& pVecBase = pBase->row(i);
		GVec& pVecOver = pOver->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			size_t vals = pRelOver->valueCount(j);
			if(vals == 0)
			{
				if(pVecOver[j] == UNKNOWN_REAL_VALUE)
					pVecOver[j] = pVecBase[j];
			}
			else
			{
				if(pVecOver[j] == UNKNOWN_DISCRETE_VALUE)
					pVecOver[j] = (double)std::max((size_t)0, std::min(vals - 1, (size_t)floor(pVecBase[j] + 0.5)));
			}
		}
	}
	pOver->print(cout);
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
		GVec& pRow = pA->row(i);
		for(vector<size_t>::iterator it = cols.begin(); it != cols.end(); it++)
			pRow[*it] = pow(pRow[*it], exponent);
	}
	pA->print(cout);
}

void prettify(GArgReader& args)
{
	GDom doc;
	doc.loadJson(args.pop_string());
	doc.writeJsonPretty(cout);
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

void reorderColumns(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse and check the list of columns
	vector<size_t> colList;
	size_t attrCount = pData->cols();
	parseAttributeList(colList, args, attrCount);
	for(size_t i = 0; i < colList.size(); i++)
	{
		size_t ind = colList[i];
		if(ind >= pData->cols())
			throw Ex("Column ", to_str(ind), " is out of range.");
	}

	// Make a list of indexes
	vector<size_t> pos_to_col;
	vector<size_t> col_to_pos;
	pos_to_col.reserve(pData->cols());
	col_to_pos.reserve(pData->cols());
	for(size_t i = 0; i < pData->cols(); i++)
	{
		pos_to_col.push_back(i);
		col_to_pos.push_back(i);
	}

	// Do the swapping
	for(size_t i = 0; i < colList.size(); i++)
	{
		if(pos_to_col[i] == colList[i])
			continue;
		size_t j = col_to_pos[colList[i]];
		size_t coli = pos_to_col[i];
		size_t colj = pos_to_col[j];
		pData->swapColumns(i, j);
		pos_to_col[i] = colj;
		pos_to_col[j] = coli;
		col_to_pos[coli] = j;
		col_to_pos[colj] = i;
	}

	// Drop superfluous columns
	if(pData->cols() > colList.size())
		pData->deleteColumns(colList.size(), pData->cols() - colList.size());

	pData->print(cout);
}

void rotate(GArgReader& args)
{
	GMatrix* pA = loadData(args.pop_string());
	Holder<GMatrix> hA(pA);
	const GRelation* relation = &pA->relation();
	unsigned colx = args.pop_uint();
	if(colx >= pA->cols()){
	  throw Ex("Rotation first column index (",to_str(colx),") "
		     "should not be greater "
		     "than the largest index, which is ", to_str(pA->cols()-1),
		     ".");
	}
	if(!relation->areContinuous(colx,1)){
	  throw Ex("Rotation first column index (",to_str(colx),") "
		     "should be continuous and it is not.");

	}
	unsigned coly = args.pop_uint();
	if(coly >= pA->cols()){
	  throw Ex("Rotation second column index (",to_str(coly),") "
		     "should not be greater "
		     "than the largest index, which is ", to_str(pA->cols()-1),
		     ".");
	}
	if(!relation->areContinuous(coly,1)){
	  throw Ex("Rotation second column index (",to_str(coly),") "
		     "should be continuous and it is not.");
	}

	double angle = args.pop_double();

	angle = angle * M_PI / 180; //Convert from degrees to radians
	double cosAngle = std::cos(angle);
	double sinAngle = std::sin(angle);
	for(std::size_t rowIdx = 0; rowIdx < pA->rows(); ++rowIdx){
		GVec& row = (*pA)[rowIdx];
		double x = row[colx];
		double y = row[coly];
		row[colx]=x*cosAngle-y*sinAngle;
		row[coly]=x*sinAngle+y*cosAngle;
	}
	pA->print(cout);
}

#define MAX_LINE_LENGTH (1024 * 1024)

void sampleRows(GArgReader& args)
{
	const char* filename = args.pop_string();
	double portion = args.pop_double();
	if(portion < 0 || portion > 1)
		throw Ex("The portion must be between 0 and 1");
	PathData pd;
	GFile::parsePath(filename, &pd);
	bool arff = false;
	if(_stricmp(filename + pd.extStart, ".arff") == 0)
		arff = true;

	// Parse Options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.size() > 0)
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}
	GRand rand(seed);

	size_t size = 0;
	std::ifstream s;
	s.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		s.open(filename, std::ios::binary);
		s.seekg(0, std::ios::end);
		size = (size_t)s.tellg();
		s.seekg(0, std::ios::beg);
	}
	catch(const std::exception&)
	{
		if(GFile::doesFileExist(filename))
			throw Ex("Error while trying to open the existing file: ", filename);
		else
			throw Ex("File not found: ", filename);
	}
	char* pLine = new char[MAX_LINE_LENGTH];
	ArrayHolder<char> hLine(pLine);
	size_t line = 1;
	while(size > 0)
	{
		s.getline(pLine, std::min(size + 1, size_t(MAX_LINE_LENGTH)));
		size_t linelen = std::min(size, size_t(s.gcount()));
		if(linelen >= MAX_LINE_LENGTH - 1)
			throw Ex("Line ", to_str(line), " is too long"); // todo: just resize the buffer here
		if(arff)
		{
			if(_strnicmp(pLine, "@DATA", 5) == 0)
				arff = false;
			cout << pLine << "\n";
		}
		else if(rand.uniform() < portion)
			cout << pLine << "\n";
		size -= linelen;
		line++;
	}
}

void sampleRowsRegularly(GArgReader& args)
{
	const char* filename = args.pop_string();
	size_t freq = args.pop_uint();
	PathData pd;
	GFile::parsePath(filename, &pd);
	bool arff = false;
	if(_stricmp(filename + pd.extStart, ".arff") == 0)
		arff = true;

	size_t size = 0;
	std::ifstream s;
	s.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		s.open(filename, std::ios::binary);
		s.seekg(0, std::ios::end);
		size = (size_t)s.tellg();
		s.seekg(0, std::ios::beg);
	}
	catch(const std::exception&)
	{
		if(GFile::doesFileExist(filename))
			throw Ex("Error while trying to open the existing file: ", filename);
		else
			throw Ex("File not found: ", filename);
	}
	char* pLine = new char[MAX_LINE_LENGTH];
	ArrayHolder<char> hLine(pLine);
	size_t line = 1;
	while(size > 0)
	{
		s.getline(pLine, std::min(size + 1, size_t(MAX_LINE_LENGTH)));
		size_t linelen = std::min(size, size_t(s.gcount()));
		if(linelen >= MAX_LINE_LENGTH - 1)
			throw Ex("Line ", to_str(line), " is too long"); // todo: just resize the buffer here
		if(arff)
		{
			if(_strnicmp(pLine, "@DATA", 5) == 0)
				arff = false;
			cout << pLine << "\n";
		}
		else if(line % freq == 0)
			cout << pLine << "\n";
		size -= linelen;
		line++;
	}
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
		GVec& pRow = pA->row(i);
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
		GVec& pRow = pA->row(i);
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
			throw Ex("Invalid option: ", args.peek());
	}

	// Print some basic stats
	cout.precision(8);
	{
		cout << "### Some basic stats\n";
		cout << "Medians = " << pData->columnMedian(attr1) << ", " << pData->columnMedian(attr2) << "\n";
		double mean1 = pData->columnMean(attr1);
		double mean2 = pData->columnMean(attr2);
		cout << "Means = " << mean1 << ", " << mean2 << "\n";
		double var1 = pData->columnVariance(attr1, mean1);
		double var2 = pData->columnVariance(attr2, mean2);
		cout << "Standard deviations = " << sqrt(var1) << ", " << sqrt(var2) << "\n";
		int less = 0;
		int eq = 0;
		int more = 0;
		for(size_t i = 0; i < pData->rows(); i++)
		{
			GVec& pRow = pData->row(i);
			if(std::abs(pRow[attr1] - pRow[attr2]) < tolerance)
				eq++;
			else if(pRow[attr1] < pRow[attr2])
				less++;
			else
				more++;
		}
		cout << less << " less, " << eq << " same, " << more << " greater\n";
	}

	// Perform the significance tests
	{
		cout << "\n### Paired T-test\n";
		size_t v;
		double t;
		pData->pairedTTest(&v, &t, attr1, attr2, false);
		double p = GMath::tTestAlphaValue(v, t);
		cout << "v=" << v << ", t=" << t << ", p=" << p << "\n";
	}
	{
		cout << "\n### Paired T-test with normalized values\n";
		size_t v;
		double t;
		pData->pairedTTest(&v, &t, attr1, attr2, true);
		double p = GMath::tTestAlphaValue(v, t);
		cout << "v=" << v << ", t=" << t << ", p=" << p << "\n";
	}
	{
		cout << "\n### Wilcoxon Signed Ranks Test\n";
		int num;
		double wMinus, wPlus;
		pData->wilcoxonSignedRanksTest(attr1, attr2, tolerance, &num, &wMinus, &wPlus);
		cout << "Number of signed ranks: " << num << "\n";
		double w_min = std::min(wMinus, wPlus);
		double w_sum = wPlus - wMinus;
		cout << "W- = " << wMinus << ", W+ = " << wPlus << ", W_min = " << w_min << ", W_sum = " << w_sum << "\n";

		double p_min = 0.5 * GMath::wilcoxonPValue(num, w_min);
		if(num < 10)
			cout << "Because the number of signed ranks is small, you should use a lookup table, rather than rely on the normal approximation for the P-value.\n";
		cout << "One-tailed P-value (for directional comparisons--is A better than B?) computed with a normal approximation using W_min = " << 0.5 * p_min << "\n";
		cout << "Two-tailed P-value (for non-directional comparisons--is A different than B?) computed with a normal approximation using W_min = " << p_min << "\n";
		cout << "To show that something is \"better\" than something else, use the one-tailed P-value.\n";
		cout << "Commonly, a P-value less that 0.05 is considered to be significant.\n";
/*
			double p_sum = GMath::wilcoxonPValue(num, w_sum);
			cout << "Directional (one-tailed) P-value computed with W_sum = " << p_sum << "\n";
*/
	}
}

void aggregateCols(GArgReader& args)
{
	size_t c = args.pop_uint();
	vector<string> files;
	GFile::fileList(files);
	GMatrix* pResults = NULL;
	Holder<GMatrix> hResults;
	size_t i = 0;
	for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
	{
		PathData pd;
		GFile::parsePath(it->c_str(), &pd);
		if(strcmp(it->c_str() + pd.extStart, ".arff") != 0)
			continue;
		GMatrix* pData = loadData(it->c_str());
		Holder<GMatrix> hData(pData);
		if(!pResults)
		{
			pResults = new GMatrix(pData->rows(), files.size());
			hResults.reset(pResults);
		}
		pResults->copyBlock(*pData, 0, c, pData->rows(), 1, 0, i, false);
		i++;
	}
	pResults->print(cout);
}

void aggregateRows(GArgReader& args)
{
	size_t r = args.pop_uint();
	vector<string> files;
	GFile::fileList(files);
	GMatrix* pResults = NULL;
	Holder<GMatrix> hResults;
	for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
	{
		PathData pd;
		GFile::parsePath(it->c_str(), &pd);
		if(strcmp(it->c_str() + pd.extStart, ".arff") != 0)
			continue;
		GMatrix* pData = loadData(it->c_str());
		Holder<GMatrix> hData(pData);
		if(!pResults)
		{
			pResults = new GMatrix(pData->relation().clone());
			hResults.reset(pResults);
		}
		pResults->takeRow(pData->releaseRow(r));
	}
	pResults->print(cout);
}

void split(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	int pats = (int)pData->rows() - args.pop_uint();
	if(pats < 0)
		throw Ex("out of range. The data only has ", to_str(pData->rows()), " rows.");
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
			throw Ex("Invalid option: ", args.peek());
	}

	// Shuffle if necessary
	GRand rng(nSeed);
	if(shouldShuffle){
		pData->shuffle(rng);
	}

	// Split
	GMatrix other(pData->relation().clone());
	pData->splitBySize(other, pats);
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
		throw Ex("fold index out of range. It must be less than the total number of folds.");

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
			throw Ex("Invalid option: ", args.peek());
	}

	// Copy relevant portions of the data
	GMatrix train(pData->relation().clone());
	GMatrix test(pData->relation().clone());
	size_t begin = pData->rows() * fold / folds;
	size_t end = pData->rows() * (fold + 1) / folds;
	for(size_t i = 0; i < begin; i++)
		train.newRow().copy(pData->row(i));
	for(size_t i = begin; i < end; i++)
		test.newRow().copy(pData->row(i));
	for(size_t i = end; i < pData->rows(); i++)
		train.newRow().copy(pData->row(i));
	train.saveArff(filenameTrain.c_str());
	test.saveArff(filenameTest.c_str());
}

void splitClass(GArgReader& args)
{
	const char* filename = args.pop_string();
	GMatrix* pData = loadData(filename);
	Holder<GMatrix> hData(pData);
	size_t classAttr = args.pop_uint();

	bool dropClass = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-dropclass"))
			dropClass = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	for(size_t i = 0; i < pData->relation().valueCount(classAttr); i++)
	{
		GMatrix tmp(pData->relation().clone());
		pData->splitCategoricalKeepIfNotEqual(&tmp, classAttr, (int)i);
		std::ostringstream oss;
		PathData pd;
		GFile::parsePath(filename, &pd);
		string fn;
		fn.assign(filename + pd.fileStart, pd.extStart - pd.fileStart);
		oss << fn << "_";
		pData->relation().printAttrValue(oss, classAttr, (double)i);
		oss << ".arff";
		string s = oss.str();
		if(dropClass)
			tmp.deleteColumns(classAttr, 1);
		tmp.saveArff(s.c_str());
	}
}

void splitVal(GArgReader& args)
{
	const char* filename = args.pop_string();
	GMatrix* pData = loadData(filename);
	Holder<GMatrix> hData(pData);
	size_t col = args.pop_uint();
	double val = args.pop_double();
	GMatrix ge(pData->relation().clone());
	pData->splitByPivot(&ge, col, val);
	pData->saveArff("less_than.arff");
	ge.saveArff("greater_or_equal.arff");
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
			throw Ex("Invalid option: ", args.peek());
	}

	// Replace missing values and print
	GRand prng(nSeed);
	if(random)
	{
		for(size_t i = 0; i < pData->relation().size(); i++)
			pData->replaceMissingValuesRandomly(i, &prng);
	}
	else
	{
		for(size_t i = 0; i < pData->relation().size(); i++)
			pData->replaceMissingValuesWithBaseline(i);
	}
	pData->print(cout);
}

void filterElements(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t attr = args.pop_uint();
	double dmin = args.pop_double();
	double dmax = args.pop_double();

	bool invert = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-invert"))
			invert = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	if(invert)
	{
		for(size_t i = 0; i < pData->rows(); i++)
		{
			double val = pData->row(i)[attr];
			if(val >= dmin && val <= dmax)
				pData->row(i)[attr] = UNKNOWN_REAL_VALUE;
		}
	}
	else
	{
		for(size_t i = 0; i < pData->rows(); i++)
		{
			double val = pData->row(i)[attr];
			if(val < dmin || val > dmax)
				pData->row(i)[attr] = UNKNOWN_REAL_VALUE;
		}
	}
	pData->print(cout);
}

void filterRows(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t attr = args.pop_uint();
	double dmin = args.pop_double();
	double dmax = args.pop_double();

	bool invert = false;
	bool preserveOrder = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-invert"))
			invert = true;
		else if(args.if_pop("-preserveOrder"))
			preserveOrder = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	if(invert)
	{
		for(size_t i = pData->rows() - 1; i < pData->rows(); i--)
		{
			double val = pData->row(i)[attr];
			if(val >= dmin && val <= dmax)
			{
				if(preserveOrder)
					pData->deleteRowPreserveOrder(i);
				else
					pData->deleteRow(i);
			}
		}
	}
	else
	{
		for(size_t i = pData->rows() - 1; i < pData->rows(); i--)
		{
			double val = pData->row(i)[attr];
			if(val == UNKNOWN_REAL_VALUE || val < dmin || val > dmax)
			{
				if(preserveOrder)
					pData->deleteRowPreserveOrder(i);
				else
					pData->deleteRow(i);
			}
		}
	}
	pData->print(cout);
}

void _function(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Accumulate the expression
	string expr;
	while(args.size() > 0)
		expr += args.pop_string();

	// Parse the expression
	GFunctionParser mfp;
	mfp.add(expr.c_str());

	// Count the functions
	char szFuncName[32];
	size_t funcCount = 0;
	for(int i = 1; true; i++)
	{
		// Find the function
		sprintf(szFuncName, "f%d", i);
		GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
		if(!pFunc)
		{
			if(i == 1)
				throw Ex("There is no function named \"f1\". Nothing to do.");
			break;
		}
		if((size_t)pFunc->m_expectedParams > pData->cols())
			throw Ex("The function ", szFuncName, " takes ", to_str(pFunc->m_expectedParams), " parameters, but the input data has only ", to_str(pData->cols()), " columns.");
		funcCount++;
	}

	// Compute the output matrix
	GMatrix out(pData->rows(), funcCount);
	vector<double> params;
	for(int i = 1; true; i++)
	{
		sprintf(szFuncName, "f%d", i);
		GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
		if(!pFunc)
			break;
		params.resize(pFunc->m_expectedParams);
		for(size_t j = 0; j < pData->rows(); j++)
		{
			GVec& pIn = pData->row(j);
			for(size_t k = 0; k < (size_t)pFunc->m_expectedParams; k++)
				params[k] = pIn[k];
			out[j][i - 1] = pFunc->call(params, mfp);
		}
	}
	out.print(cout);
}

void _function2(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Accumulate the expression
	string expr;
	while(args.size() > 0)
		expr += args.pop_string();

	// Parse the expression
	GFunctionParser mfp;
	mfp.add(expr.c_str());

	// Count the functions
	char szFuncName[32];
	size_t fCount = 0;
	for(int i = 1; true; i++)
	{
		// Find the function
		sprintf(szFuncName, "f%d", i);
		GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
		if(!pFunc)
		{
			if(i == 1)
				throw Ex("There is no function named \"f1\". Nothing to do.");
			break;
		}
		if((size_t)pFunc->m_expectedParams > pData->cols())
			throw Ex("The function ", szFuncName, " takes ", to_str(pFunc->m_expectedParams), " parameters, but the input data has only ", to_str(pData->cols()), " columns.");
		fCount++;
	}
	size_t gCount = 0;
	for(int i = 1; true; i++)
	{
		// Find the function
		sprintf(szFuncName, "g%d", i);
		GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
		if(!pFunc)
			break;
		if((size_t)pFunc->m_expectedParams > 2 * pData->cols()) // The first half of the parameters are the current row. The second half are the iterated row.
			throw Ex("The function ", szFuncName, " takes ", to_str(pFunc->m_expectedParams), " parameters, but the input data has only ", to_str(pData->cols()), " columns.");
		gCount++;
	}

	// Compute the output matrix
	char buf[256];
	vector<double> params;
	vector<double> gsum;
	gsum.resize(gCount);
	GMatrix out(pData->rows(), fCount);
	for(size_t j = 0; j < pData->rows(); j++)
	{
		for(size_t i = 0; i < gCount; i++)
			gsum[i] = 0.0;
		GVec& pCur = pData->row(j);

		// Compute the gsum vector, which applies all the "g" functions to every row and sums the returned values
		for(size_t k = 0; k < pData->rows(); k++)
		{
			GVec& pItt = pData->row(k);
			{
				for(int i = 1; true; i++)
				{
					sprintf(szFuncName, "g%d", i);
					GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
					if(!pFunc)
						break;
					params.resize(pFunc->m_expectedParams);
					for(size_t m = 0; m < (size_t)pFunc->m_expectedParams / 2; m++)
						params[m] = pCur[m];
					size_t half = pFunc->m_expectedParams / 2;
					for(size_t m = half; m < (size_t)pFunc->m_expectedParams; m++)
						params[m] = pItt[m - half];
					gsum[i - 1] += pFunc->call(params, mfp);
				}
			}
		}

		// Add "h#" constants to represent the gsum values
		for(int i = 1; true; i++)
		{
			sprintf(szFuncName, "g%d", i);
			GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
			if(!pFunc)
				break;
			sprintf(buf, "h%d=%f", i, gsum[i - 1]);
			mfp.add(buf);
		}

		// Compute the "f" functions for this row
		for(int i = 1; true; i++)
		{
			sprintf(szFuncName, "f%d", i);
			GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
			if(!pFunc)
				break;
			params.resize(pFunc->m_expectedParams);
			for(size_t k = 0; k < (size_t)pFunc->m_expectedParams; k++)
				params[k] = pCur[k];
			out[j][i - 1] = pFunc->call(params, mfp);
		}
	}
	out.print(cout);
}

void geodistance(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t lat1 = args.pop_uint();
	size_t lon1 = args.pop_uint();
	size_t lat2 = args.pop_uint();
	size_t lon2 = args.pop_uint();

	double radius = 6371.0; // Approximate radius of the Earth in kilometers
	while(args.size() > 0)
	{
		if(args.if_pop("-radius"))
			radius = args.pop_double();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	GArffRelation* pRel = new GArffRelation();
	pRel->addAttribute("geodistance", 0, NULL);
	GMatrix tmp(pRel);
	tmp.newRows(pData->rows());

	for(size_t i = 0; i < pData->rows(); i++)
	{
		double degLat1 = pData->row(i)[lat1];
		double degLon1 = pData->row(i)[lon1];
		double degLat2 = pData->row(i)[lat2];
		double degLon2 = pData->row(i)[lon2];
		if(degLat1 == UNKNOWN_REAL_VALUE || degLon1 == UNKNOWN_REAL_VALUE || degLat2 == UNKNOWN_REAL_VALUE || degLon2 == UNKNOWN_REAL_VALUE)
		{
			tmp[i][0] = UNKNOWN_REAL_VALUE;
			continue;
		}
		double psi1 = degLat1 * M_PI / 180.0;
		double psi2 = degLat2 * M_PI / 180.0;
		double deltapsi = (degLat2 - degLat1) * M_PI / 180.0;
		double deltalambda = (degLon2 - degLon1) * M_PI / 180.0;
		double a = sin(deltapsi * 0.5) * sin(deltapsi * 0.5) +
		        cos(psi1) * cos(psi2) *
		        sin(deltalambda * 0.5) * sin(deltalambda * 0.5);
		double c = 2 * atan2(sqrt(a), sqrt(1.0 - a));
		tmp[i][0] = radius * c;
	}
	tmp.print(cout);
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
			throw Ex("Invalid option: ", args.peek());
	}

	// Shuffle and print
	GRand prng(nSeed);
	pData->shuffle(prng);
	pData->print(cout);
}

void SortByAttribute(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	size_t nAttr = args.pop_uint();
	size_t attrCount = pData->relation().size();
	if(nAttr >= attrCount)
		throw Ex("Index out of range");

	// Parse options
	bool descending = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-descending"))
			descending = true;
		else
			throw Ex("Invalid option: ", args.peek());
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
	size_t attrCount = pData->relation().size();
	if(nAttr1 >= attrCount)
		throw Ex("Index out of range");
	if(nAttr2 >= attrCount)
		throw Ex("Index out of range");
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
    throw Ex(msg.str());
  }
  if(hData->relation().valueCount(column) != 0){
    throw Ex("Can only use threshold on continuous attributes.");
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
		throw Ex("Expected the same number of rows in both datasets");

	// Parse options
	bool delta = false;
	while(args.size() > 0)
	{
		if(args.if_pop("-delta"))
			delta = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Make the output data
	size_t actionDims = pActions->cols();
	size_t stateDims = pState->cols();
	GMixedRelation* pRelation = new GMixedRelation();
	pRelation->addAttrs(pActions->relation());
	pRelation->addAttrs(stateDims + stateDims, 0);
	GMatrix* pTransition = new GMatrix(pRelation);
	pTransition->newRows(pActions->rows() - 1);
	for(size_t i = 0; i < pActions->rows() - 1; i++)
	{
		GVec& pOut = pTransition->row(i);
		GVec::copy(pOut.data(), pActions->row(i).data(), actionDims);
		GVec::copy(pOut.data() + actionDims, pState->row(i).data(), stateDims);
		GVec::copy(pOut.data() + actionDims + stateDims, pState->row(i + 1).data(), stateDims);
		if(delta)
			GVec::subtract(pOut.data() + actionDims + stateDims, pState->row(i).data(), stateDims);
	}
	pTransition->print(cout);
}

void uglify(GArgReader& args)
{
	GDom doc;
	doc.loadJson(args.pop_string());
	doc.writeJson(cout);
}

void wilcoxon(GArgReader& args)
{
	size_t n = args.pop_uint();
	double w = args.pop_double();
	double p = GMath::wilcoxonPValue((int)n, w);
	cout << p << "\n";
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
		if(args.size() < 1) throw Ex("Expected a command");
		else if(args.if_pop("usage")) ShowUsage(appName);
		else if(args.if_pop("add")) addMatrices(args);
		else if(args.if_pop("addindexcolumn")) AddIndexAttribute(args);
		else if(args.if_pop("addcategorycolumn")) addCategoryColumn(args);
		else if(args.if_pop("addnoise")) addNoise(args);
		else if(args.if_pop("aggregatecols")) aggregateCols(args);
		else if(args.if_pop("aggregaterows")) aggregateRows(args);
		else if(args.if_pop("align")) align(args);
		else if(args.if_pop("autocorrelation")) autoCorrelation(args);
		else if(args.if_pop("center")) center(args);
		else if(args.if_pop("cholesky")) cholesky(args);
		else if(args.if_pop("colstats")) colstats(args);
		else if(args.if_pop("correlation")) correlation(args);
		else if(args.if_pop("covariance")) covariance(args);
		else if(args.if_pop("cumulativecolumns")) cumulativeColumns(args);
		else if(args.if_pop("determinant")) determinant(args);
		else if(args.if_pop("discretize")) Discretize(args);
		else if(args.if_pop("dropcolumns")) dropColumns(args);
		else if(args.if_pop("drophomogcols")) dropHomogeneousCols(args);
		else if(args.if_pop("dropiftooclose")) dropIfTooClose(args);
		else if(args.if_pop("dropmissingvalues")) DropMissingValues(args);
		else if(args.if_pop("droprandomvalues")) dropRandomValues(args);
		else if(args.if_pop("droprows")) dropRows(args);
		else if(args.if_pop("dropunusedvalues")) dropUnusedValues(args);
		else if(args.if_pop("enumeratevalues")) enumerateValues(args);
		else if(args.if_pop("export")) Export(args);
		else if(args.if_pop("fillmissingvalues")) fillMissingValues(args);
		else if(args.if_pop("filterelements")) filterElements(args);
		else if(args.if_pop("filterrows")) filterRows(args);
		else if(args.if_pop("function")) _function(args);
		else if(args.if_pop("function2")) _function2(args);
		else if(args.if_pop("geodistance")) geodistance(args);
		else if(args.if_pop("import")) Import(args);
		else if(args.if_pop("keeponlycolumns")) keepOnlyColumns(args);
		else if(args.if_pop("measuremeansquarederror")) MeasureMeanSquaredError(args);
		else if(args.if_pop("mergehoriz")) mergeHoriz(args);
		else if(args.if_pop("mergevert")) mergeVert(args);
		else if(args.if_pop("multiply")) multiplyMatrices(args);
		else if(args.if_pop("multiplyscalar")) multiplyScalar(args);
		else if(args.if_pop("nominaltocat")) nominalToCat(args);
		else if(args.if_pop("normalize")) normalize(args);
		else if(args.if_pop("normalizemagnitude")) normalizeMagnitude(args);
		else if(args.if_pop("neighbors")) neighbors(args);
		else if(args.if_pop("obfuscate")) obfuscate(args);
		else if(args.if_pop("overlay")) overlay(args);
		else if(args.if_pop("powercolumns")) powerColumns(args);
		else if(args.if_pop("prettify")) prettify(args);
		else if(args.if_pop("pseudoinverse")) pseudoInverse(args);
		else if(args.if_pop("reducedrowechelonform")) reducedRowEchelonForm(args);
		else if(args.if_pop("reordercolumns")) reorderColumns(args);
		else if(args.if_pop("rotate")) rotate(args);
		else if(args.if_pop("samplerows")) sampleRows(args);
		else if(args.if_pop("samplerowsregularly")) sampleRowsRegularly(args);
		else if(args.if_pop("scalecolumns")) scaleColumns(args);
		else if(args.if_pop("shiftcolumns")) shiftColumns(args);
		else if(args.if_pop("shuffle")) Shuffle(args);
		else if(args.if_pop("significance")) significance(args);
		else if(args.if_pop("sortcolumn")) SortByAttribute(args);
		else if(args.if_pop("split")) split(args);
		else if(args.if_pop("splitclass")) splitClass(args);
		else if(args.if_pop("splitfold")) splitFold(args);
		else if(args.if_pop("splitval")) splitVal(args);
		else if(args.if_pop("squaredDistance")) squaredDistance(args);
		else if(args.if_pop("swapcolumns")) SwapAttributes(args);
		else if(args.if_pop("threshold")) threshold(args);
		else if(args.if_pop("transition")) transition(args);
		else if(args.if_pop("transpose")) Transpose(args);
		else if(args.if_pop("uglify")) uglify(args);
		else if(args.if_pop("wilcoxon")) wilcoxon(args);
		else if(args.if_pop("zeromean")) zeroMean(args);
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

