// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "../GClasses/GApp.h"
#include "../GClasses/GError.h"
#include "../GClasses/GMatrix.h"
#include "../GClasses/GDecisionTree.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GHistogram.h"
#include "../GClasses/GImage.h"
#include "../GClasses/GOptimizer.h"
#include "../GClasses/GHillClimber.h"
#include "../GClasses/G3D.h"
#include "../GClasses/GRayTrace.h"
#include "../GClasses/GVec.h"
#include "../GClasses/GDom.h"
#include "../GClasses/GPlot.h"
#include "../GClasses/GFunction.h"
#include "../GClasses/GLearner.h"
#include "../GClasses/GNeighborFinder.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GManifold.h"
#include "../GClasses/GSystemLearner.h"
#include "../GClasses/GSelfOrganizingMap.h"
#include "plotchart.h"
#include "../wizard/usage.h"
#include <time.h>
#include <iostream>
#ifdef WIN32
#	include <direct.h>
#	include <process.h>
#endif
#include <string>
#include <vector>
#include <exception>
#include <set>

using namespace GClasses;
using std::string;
using std::cout;
using std::cerr;
using std::vector;
using std::set;
using std::ostringstream;

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
		else if(_stricmp(alg, "saffron") == 0)
		{
			size_t medianCands = args.pop_uint();
			size_t neighbors = args.pop_uint();
			size_t tangentSpaceDims = args.pop_uint();
			double thresh = args.pop_double();
			pNF = new GSaffron(pData, medianCands, neighbors, tangentSpaceDims, thresh, pRand);
		}
		else if(_stricmp(alg, "temporal") == 0)
		{
			GMatrix* pControlData = loadData(args.pop_string());
			Holder<GMatrix> hControlData(pControlData);
			if(pControlData->rows() != pData->rows())
				ThrowError("mismatching number of rows");
			int neighbors = args.pop_uint();
			pNF = new GTemporalNeighborFinder(pData, hControlData.release(), true, neighbors, pRand);
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

void PlotBar(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GArffRelation* pRel = (GArffRelation*)pData->relation().get();
	if(pRel->size() != 1 || pRel->areContinuous(0, 1))
		ThrowError("Expected exactly one continuous attribute");
	double* values = new double[pData->rows()];
	ArrayHolder<double> hValues(values);
	for(size_t i = 0; i < pData->rows(); i++)
		values[i] = pData->row(i)[0];

	// Parse options
	bool bLog = false;
	string filename = "plot.png";
	while(args.next_is_flag())
	{
		if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-log"))
			bLog = true;
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Make the chart
	GRand prng(0);
	size_t minIndex = GVec::indexOfMin(values, pData->rows(), &prng);
	size_t maxIndex = GVec::indexOfMax(values, pData->rows(), &prng);
	double dMin = values[minIndex];
	double dMax = std::max(1e-12, values[maxIndex]);
	if(dMin > 0 && !bLog && (dMax - dMin) / dMax > 0.05)
		dMin = 0;
	GImage image;
	image.setSize(800, 800);
	image.clear(0xffffffff);
	double xmin = -0.5;
	double ymin = bLog ? log(dMin * 0.7) : dMin - 0.1 * (dMax - dMin);
	double xmax = (double)pData->rows();
	double ymax = bLog ? log(dMax + 0.5 * (dMax - dMin)) : dMax + 0.1 * (dMax - dMin);
	GPlotWindow pw(&image, xmin, ymin, xmax, ymax);
	pw.gridLines(0, (bLog ? -1 : 30), 0xff808080);
	for(size_t i = 0; i < pData->rows(); i++)
	{
		int x1, y1, x2, y2;
			pw.windowToView((double)i, 0, &x1, &y1);
		if(bLog)
		{
			pw.windowToView(0.5 + i, log(values[i]), &x2, &y2);
			image.boxFill(x1, y2, x2 - x1, std::max(0, (int)image.height() - y2), gAHSV(0xff, (float)i / pData->rows(), 1.0f, 0.5f));
		}
		else
		{
			pw.windowToView(0.5 + i, values[i], &x2, &y2);
			if(y2 < y1)
				std::swap(y1, y2);
			image.boxFill(x1, y1, x2 - x1, y2 - y1, gAHSV(0xff, (float)i / pData->rows(), 1.0f, 0.5f));
		}
	}
	GImage* pLabeledImage = pw.labelAxes(0, (bLog ? -1 : 30), 5/*precision*/, 1/*size*/, 0xff000000/*color*/, 45.0 * (M_PI / 180)/*angle*/);
	Holder<GImage> hLabeledImage(pLabeledImage);
	pLabeledImage->savePng(filename.c_str());
	cout << "Chart saved to " << filename.c_str() << ".\n";
}

class BigOCritic : public GTargetFunction
{
protected:
	GMatrix* m_pData;
	size_t m_attr;

public:
	BigOCritic(GMatrix* pData, size_t attr)
	: GTargetFunction(3), m_pData(pData), m_attr(attr)
	{
	}

	virtual ~BigOCritic()
	{
	}

	virtual bool isStable() { return true; }
	virtual bool isConstrained() { return false; }

	virtual void initVector(double* pVector)
	{
		pVector[0] = 1.0;
		pVector[1] = 1.0;
		pVector[2] = 0.0;
	}

	virtual double computeError(const double* pVector)
	{
		double err = 0;
		for(size_t i = 0; i < m_pData->rows(); i++)
		{
			double* pPat = m_pData->row(i);
			if(pPat[0] != UNKNOWN_REAL_VALUE && pPat[m_attr] != UNKNOWN_REAL_VALUE)
			{
				double d = pVector[0] * (pow(pPat[0], pVector[1]) + pVector[2]) - pPat[m_attr];
				err += (d * d);
			}
		}
		return err;
	}
};

void EstimateBigO(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GArffRelation* pRel = (GArffRelation*)pData->relation().get();
	if(pRel->size() < 2)
		ThrowError("Expected at least two attributes");
	if(!pRel->areContinuous(0, pRel->size()))
		ThrowError("Expected all continuous attributes");

	// Regress t=an^b+c for each algorithm
	cout.precision(8);
	for(size_t i = 1; i < pRel->size(); i++)
	{
		BigOCritic critic(pData, i);
		GMomentumGreedySearch search(&critic);
		search.searchUntil(500, 50, 0.0001);
		double* pVec = search.currentVector();
		cout << pRel->attrName(i) << ": t=" << pVec[0] << " * (n^" << pVec[1] << " + " << pVec[2] << ")\n";
	}
}

void PlotEquation(GArgReader& args)
{
	// Parse options
	string filename = "plot.png";
	int width = 1024;
	int height = 1024;
	double xmin = -10;
	double ymin = -10;
	double xmax = 10;
	double ymax = 10;
	double textSize = 2.0;
	bool grid = true;
	while(args.next_is_flag())
	{
		if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-size"))
		{
			width = args.pop_uint();
			height = args.pop_uint();
		}
		else if(args.if_pop("-nogrid"))
			grid = false;
		else if(args.if_pop("-range"))
		{
			xmin = args.pop_double();
			ymin = args.pop_double();
			xmax = args.pop_double();
			ymax = args.pop_double();
		}
		else if(args.if_pop("-textsize"))
			textSize = args.pop_double();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Accumulate the expression
	string expr;
	while(args.size() > 0)
		expr += args.pop_string();

	// Parse the expression
	GFunctionParser mfp(expr.c_str());

	// Make the chart
	GImage image;
	image.setSize(width, height);
	image.clear(0xffffffff);
	GPlotWindow pw(&image, xmin, ymin, xmax, ymax);
	if(grid)
		pw.gridLines(30, 30, 0xffa0a0a0);

	// Plot all the functions
	char szFuncName[32];
	unsigned int colors[6];
	colors[0] = 0xff000080;
	colors[1] = 0xff800000;
	colors[2] = 0xff008000;
	colors[3] = 0xff808000;
	colors[4] = 0xff800080;
	colors[5] = 0xff008080;
	for(int i = 1; true; i++)
	{
		// Find the function
		sprintf(szFuncName, "f%d", i);
		GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
		if(!pFunc)
		{
			if(i == 1)
				ThrowError("There is no function named \"f1\". Nothing to plot.");
			break;
		}
		if(pFunc->m_expectedParams != 1)
			ThrowError("The function ", szFuncName, " takes ", to_str(pFunc->m_expectedParams), " parameters. Expected a function with 1 parameter");

		// Plot it
		unsigned int col = colors[i % 6];
		double dx = pw.pixelWidth();
		vector<double> params;
		double x = xmin;
		params.push_back(x);
		double y = pFunc->call(params);
		while(x <= xmax)
		{
			double xPrev = x;
			double yPrev = y;
			x += dx;
			params[0] = x;
			y = pFunc->call(params);
			if(y > -1e100 && y < 1e100 && yPrev > -1e100 && yPrev < 1e100)
				pw.line(xPrev, yPrev, x, y, col);
		}
	}

	GImage* pLabeledImage = pw.labelAxes(30, 30, 5/*precision*/, (float)textSize/*size*/, 0xff000000/*color*/, 45.0 * (M_PI / 180)/*angle*/);
	Holder<GImage> hLabeledImage(pLabeledImage);
	pLabeledImage->savePng(filename.c_str());
	cout << "Plot saved to " << filename.c_str() << ".\n";
}

void PlotScatter(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse options
	GRand prng(0);
	PlotChartMaker pcm(pData->relation(), pData, prng);
	string filename = "plot.png";
	bool horizGrid = true;
	bool vertGrid = true;
	bool showLines = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-size"))
		{
			int wid = args.pop_uint();
			int hgt = args.pop_uint();
			pcm.SetSize(wid, hgt);
		}
		else if(args.if_pop("-lines"))
			showLines = true;
		else if(args.if_pop("-logx"))
			pcm.SetLogX();
		else if(args.if_pop("-logy"))
			pcm.SetLogY();
		else if(args.if_pop("-novgrid"))
			vertGrid = false;
		else if(args.if_pop("-nohgrid"))
			horizGrid = false;
		else if(args.if_pop("-nogrid"))
		{
			vertGrid = false;
			horizGrid = false;
		}
		else if(args.if_pop("-pointradius"))
			pcm.SetPointRadius((float)args.pop_double());
		else if(args.if_pop("-textsize"))
			pcm.setTextSize((float)args.pop_double());
		else if(args.if_pop("-linethickness"))
			pcm.SetLineThickness((float)args.pop_double());
		else if(args.if_pop("-maxgridlines"))
		{
			int h = args.pop_uint();
			int v = args.pop_uint();
			pcm.setMaxGridLines(h, v);
		}
		else if(args.if_pop("-mesh"))
			pcm.SetMeshRowSize(args.pop_uint());
		else if(args.if_pop("-aspect"))
			pcm.setAspect();
		else if(args.if_pop("-range"))
		{
			double xmin = args.pop_double();
			double ymin = args.pop_double();
			double xmax = args.pop_double();
			double ymax = args.pop_double();
			pcm.SetCustomRange(xmin, ymin, xmax, ymax);
		}
		else if(args.if_pop("-chartcolors"))
		{
			unsigned int cBackground = hexToRgb(args.pop_string());
			unsigned int cText = hexToRgb(args.pop_string());
			unsigned int cGrid = hexToRgb(args.pop_string());
			pcm.SetChartColors(cBackground, cText, cGrid);
		}
		else if(args.if_pop("-linecolors"))
		{
			unsigned int c1 = hexToRgb(args.pop_string());
			unsigned int c2 = hexToRgb(args.pop_string());
			unsigned int c3 = hexToRgb(args.pop_string());
			unsigned int c4 = hexToRgb(args.pop_string());
			pcm.SetPlotColors(c1, c2, c3, c4);
		}
		else if(args.if_pop("-spectrum"))
			pcm.UseSpectrumColors();
		else if(args.if_pop("-specmod"))
			pcm.UseSpectrumColors(args.pop_uint());
		else if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-neighbors"))
			pcm.showNeighbors(instantiateNeighborFinder(pData, &prng, args));
		else if(args.if_pop("-randomorder"))
			pcm.randomOrder();
		else
			ThrowError("Invalid option: ", args.peek());
	}
	pcm.ShowAxisLabels(vertGrid, horizGrid);
	if(!showLines)
		pcm.noLines();

	// Make the chart
	GImage* pImage = pcm.MakeChart();
	Holder<GImage> hImage(pImage);
	pImage->savePng(filename.c_str());
	cout << "Plot saved to " << filename.c_str() << ".\n";
}


void semanticMap(GArgReader& args){
  string somFile = args.pop_string();
  string dataFile = args.pop_string();

  // Load the data
  Holder<GMatrix> hData(loadData(dataFile.c_str()));
  if(hData->rows() < 1){
    ThrowError("The dataset is empty.  Cannot make a semantic map from "
	       "an empty dataset.");
  }
  if(hData->cols() < 1){
    ThrowError("The dataset has no attributes.  Cannot make a semantic map "
	       "without attributes.");
  }
  // Load the self organizing map
  GDom doc;
  doc.loadJson(somFile.c_str());
  GSelfOrganizingMap som(doc.root());
  // Parse the options
  string outFilename="semantic_map.svg";
  unsigned labelCol = hData->cols()-1;
  //Set to true to use the variance in the label column among rows
  //where a given node is the winner as the label.  Lower variance
  //means a better approximation.
  bool useVarianceAsLabel = false;
  while(args.next_is_flag()){
    if(args.if_pop("-out")){
      outFilename = args.pop_string();
    }else if(args.if_pop("-labels")){
      labelCol = args.pop_uint();
    }else if(args.if_pop("-variance")){
      useVarianceAsLabel = true;
    }else{
      ThrowError("Invalid option: ", args.peek());
    }
  }
  if(labelCol >= hData->cols()){
    ThrowError("Label column index is too large");
  }
  if(som.outputDimensions() > 2){
    ThrowError("Semantic map can only plot one or two dimensional "
	       "self-organizing maps.");
  }
  if(som.inputDimensions() > hData->cols()){
    ThrowError("The input dataset does not have enough attributes for input "
	       "to the semantic map");
  }

  //Write the svg output file
  std::ofstream out(outFilename.c_str());
  if(!out){
    ThrowError("Could not open the file named \"",outFilename,"\"");
  }
  //First, write the header
  vector<double> axes = som.outputAxes();
  double maxDim = *(max_element(axes.begin(), axes.end()));
  double scale = ((100*maxDim) > 800)?100:800/maxDim;
  const double shift = scale/2;
  double ax0 = axes[0]; 
  double ax1 = (axes.size() == 1)?1:axes[1];
  out << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
      << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
      << "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
      << "<svg viewbox=\"0 0 "
      << ax0*scale+shift << " " << ax1*scale+shift
      << "\" preserveAspectRatio=\"xMinYMin\"\n"
      << "     width=\""<< ax0*scale+shift << "px\" height=\"" 
      << ax1*scale+shift << "px\"\n"
      << "     xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n"
      << "<desc>Semantic map of the self-organizing map stored in \n"
      << "      \"" << somFile << "\"\n" 
      << "      using data from \"" << dataFile << "\"\n"
      << "      Original filename: \"" << outFilename << "\"\n";
  if(som.outputDimensions() == 2){
    out << "      Plotting output dimensions " << 0 << " and " 
	<< 1 << " as (x,y) coordinates.\n";
  }else{
    assert(som.outputDimensions() == 1);
    out << "      Plotting output dimension " << 0 << "as x coordinate.\n";
  };
  out << "</desc>\n";

  std::vector<double> labels; labels.reserve(som.nodes().size());
  if(useVarianceAsLabel){
    //winlist[i] holds the pointers to the rows of the dataset where
    //nodes()[i] is the winner
    std::vector<std::list<const double*> > winLists(som.nodes().size());
    for(size_t row = 0; row < hData->rows(); ++row){
      const double* rowVec = hData->row(row);
      size_t bestNode = som.bestMatch(rowVec);
      winLists.at(bestNode).push_back(rowVec);
    }
    //Calculate the variance of the labelCol column for each node
    std::vector<std::list<const double*> >::const_iterator list;
    for(list=winLists.begin(); list != winLists.end(); ++list){
      if(list->size() == 0){
	//No elements in the list, no variance
	labels.push_back(0);
      }else{
	//Copy the appropriate column into a 1-column matrix
	GMatrix m(0,1);
	std::list<const double*>::const_iterator l;
	for(l = list->begin(); l != list->end(); ++l){
	  m.newRow();
	  double& val = *(m.row(m.rows()-1));
	  val = (*l)[labelCol];
	}
	//Use the matrix to calculate the variance
	labels.push_back(m.variance(0, m.mean(0)));
      }
    }
  }else{
    //Find the best data indices using only the first inputDimensions of
    //the input data
    Holder<GMatrix> lessColumns(hData->clone());
    while(lessColumns->cols() > som.inputDimensions()){
      lessColumns->deleteColumn(lessColumns->cols()-1);
    }
    vector<size_t> bestData = som.bestData(lessColumns.get());
  
    std::vector<double> labelColumn(hData->rows());
    hData->col(labelCol, &(labelColumn.at(0)));
    for(size_t node = 0; node < som.nodes().size(); ++node){
      labels.push_back(labelColumn.at(bestData.at(node)));
    }
  }
  //Calculate min and max for color coding
  double minLabel = *std::min_element(labels.begin(), labels.end());
  double maxLabel = *std::max_element(labels.begin(), labels.end());
  double labelRange = maxLabel-minLabel;  
  if(labelRange == 0){ labelRange = 1; }

  //Write the actual map
  for(size_t node = 0; node < som.nodes().size(); ++node){
    vector<double> loc = som.nodes()[node].outputLocation;
    double label = labels.at(node);
    unsigned red = (unsigned int)std::floor(0.5+(255*(label-minLabel)/(labelRange)));
    unsigned blue = (unsigned int)std::floor(0.5+255-(255*(label-minLabel)/(labelRange)));
    if(som.outputDimensions() == 2){
      out << "<text x=\"" << (shift+scale*loc[0]) << "\" y=\"" << (shift+scale*loc[1]) << "\" ";
    }else{
      assert(som.outputDimensions() == 1);
      out << "<text x=\"" << (shift+scale*loc[0]) << "\" y=\"" << (maxDim*scale/2) << "\" ";
    }
    out << "fill=\"rgb(" << red << ",0," << blue << ")\">";
    out << label << "</text>\n";
  }
  
  //Close the svg
  out << "</svg>\n";

  cout << "Output written to \"" << outFilename << "\"\n" ;
}

void makeHistogram(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);

	// Parse options
	int wid = 800;
	int hgt = 800;
	size_t attr = 0;
	string filename = "plot.png";
	double xmin = UNKNOWN_REAL_VALUE;
	double xmax = UNKNOWN_REAL_VALUE;
	double ymax = UNKNOWN_REAL_VALUE;
	while(args.next_is_flag())
	{
		if(args.if_pop("-attr"))
			attr = args.pop_uint();
		else if(args.if_pop("-size"))
		{
			wid = args.pop_uint();
			hgt = args.pop_uint();
		}
		else if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-range"))
		{
			xmin = args.pop_double();
			xmax = args.pop_double();
			ymax = args.pop_double();
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}
	if(attr < 0 || attr >= pData->relation()->size())
		ThrowError("attr out of range");

	// Make the histogram
	GImage image;
	image.setSize(wid, hgt);
	image.clear(0xffffffff);
	if(pData->relation()->valueCount(attr) == 0)
	{
		GHistogram hist(*pData, attr, xmin, xmax, (size_t)image.width());
		double height = (ymax == UNKNOWN_REAL_VALUE ? hist.binLikelihood(hist.modeBin()) * 1.5 : ymax);
		GPlotWindow pw(&image, hist.xmin(), 0.0, hist.xmax(), height);
		for(int i = 0; i < (int)image.width(); i++)
		{
			double x, y;
			pw.viewToWindow(i, 0, &x, &y);
			size_t bin = hist.xToBin(x);
			double likelihood = hist.binLikelihood(bin);
			if(likelihood > 0.0)
				pw.line(x, 0.0, x, likelihood, 0xff000080);
		}

		// Draw the grid
		pw.gridLines(40, 40, 0xff808080);

		// Draw the labels
		GImage* pLabeledImage = pw.labelAxes(30, 30, 5/*precision*/, 1/*size*/, 0xff000000/*color*/, 45.0 * (M_PI / 180)/*angle*/);
		Holder<GImage> hLabeledImage(pLabeledImage);

		// Save the image
		pLabeledImage->savePng(filename.c_str());
		cout << "Histogram saved to " << filename.c_str() << ".\n";
	}
	else
	{
		size_t buckets = pData->relation()->valueCount(attr);
		GTEMPBUF(double, hist, buckets);
		GVec::setAll(hist, 0.0, buckets);
		for(size_t i = 0; i < pData->rows(); i++)
		{
			int b = (int)pData->row(i)[attr];
			if(b >= 0 && (size_t)b < buckets)
				hist[b]++;
		}

		// Plot it
		size_t max = 0;
		for(size_t i = 1; i < buckets; i++)
		{
			if(hist[i] > hist[max])
				max = i;
		}
		for(int i = 0; i < (int)image.width(); i++)
		{
			size_t b = i * buckets / image.width();
			int h = (int)(hist[b] * image.height() / hist[max]);
			image.line(i, image.height(), i, image.height() - h, (((b & 1) == 0) ? 0xff400000 : 0xff008040));
		}
		image.savePng(filename.c_str());
		cout << "Histogram saved to " << filename.c_str() << ".\n";
	}
}

void MakeAttributeSummaryGraph(GRelation* pRelation, GMatrix* pData, GImage* pImage, int attr)
{
	if(pRelation->valueCount(attr) == 0)
	{
		pImage->clear(0xffffffff);
		GHistogram hist(*pData, attr, UNKNOWN_REAL_VALUE, UNKNOWN_REAL_VALUE, (size_t)pImage->width());
		double height = hist.binLikelihood(hist.modeBin());
		GPlotWindow pw(pImage, hist.xmin(), 0.0, hist.xmax(), height);
		for(int i = 0; i < (int)pImage->width(); i++)
		{
			double x, y;
			pw.viewToWindow(i, 0, &x, &y);
			size_t bin = hist.xToBin(x);
			double likelihood = hist.binLikelihood(bin);
			if(likelihood > 0.0)
				pw.line(x, 0.0, x, likelihood, 0xff000080);
		}
	}
	else
	{
		size_t buckets = pRelation->valueCount(attr);
		GTEMPBUF(double, hist, buckets);
		GVec::setAll(hist, 0.0, buckets);
		for(size_t i = 0; i < pData->rows(); i++)
		{
			int b = (int)pData->row(i)[attr];
			if(b >= 0 && (size_t)b < buckets)
				hist[b]++;
		}

		// Plot it
		pImage->clear(0xffffffff);
		size_t max = 0;
		for(size_t i = 1; i < buckets; i++)
		{
			if(hist[i] > hist[max])
				max = i;
		}
		for(int i = 0; i < (int)pImage->width(); i++)
		{
			size_t b = i * buckets / pImage->width();
			int h = (int)(hist[b] * pImage->height() / hist[max]);
			pImage->line(i, pImage->height(), i, pImage->height() - h, (((b & 1) == 0) ? 0xff400000 : 0xff008040));
		}
	}
}

void MakeCorrelationGraph(GRelation* pRelation, GMatrix* pData, GImage* pImage, int attrx, int attry, double jitter, GRand* pRand)
{
	pImage->clear(0xffffffff);
	double xmin, ymin, xmax, ymax;
	bool bothNominal = true;
	if(pRelation->valueCount(attrx) == 0) //Continuous x attribute
	{
		pData->minAndRange(attrx, &xmin, &xmax);
		xmax += xmin;
		bothNominal = false;
	}
	else //Discrete x attribute
	{
		xmin = -0.5;
		xmax = pRelation->valueCount(attrx) - 0.5;
	}
	if(pRelation->valueCount(attry) == 0) //Continuous y attribute
	{
		pData->minAndRange(attry, &ymin, &ymax);
		ymax += ymin;
		bothNominal = false;
	}
	else //Discrete y atrribute
	{
		ymin = -0.5;
		ymax = pRelation->valueCount(attry) - 0.5;
	}
	if(bothNominal)
	{
		GPlotWindow pw(pImage, 0.0, 0.0, 1.0, 1.0);
		double left = 0.0;
		double right = 0.0;
		size_t tot = pData->rows();
		for(size_t i = 0; i < pRelation->valueCount(attrx); i++)
		{
			GMatrix tmp(pData->relation());
			pData->splitByNominalValue(&tmp, attrx, (int)i);
			right += (double)tmp.rows() / tot;
			double bot = 0.0;
			double top = 0.0;
			for(size_t j = 0; j < pRelation->valueCount(attry); j++)
			{
				top += (double)tmp.countValue(attry, (double)j) / tmp.rows();
				int l, b, r, t;
				pw.windowToView(left, bot, &l, &b);
				pw.windowToView(right, top, &r, &t);
				pImage->boxFill(l, t, r - l, b - t, gAHSV(0xff, 0.9f * j / pRelation->valueCount(attry), 0.6f + ((i & 1) ? 0.0f : 0.4f), 0.4f + ((i & 1) ? 0.4f : 0.0f)));
				bot = top;
			}
			pData->mergeVert(&tmp);
			left = right;
		}
	}
	else
	{
		GPlotWindow pw(pImage, xmin, ymin, xmax, ymax);
		size_t samples = 2048;
		for(size_t i = 0; i < samples; i++)
		{
			double* pPat = pData->row(i * pData->rows() / samples);
			pw.point(pPat[attrx] + pRand->normal() * jitter * (xmax - xmin), pPat[attry] + pRand->normal() * jitter * (ymax - ymin), 0xff000080);
		}
	}
}

void MakeCorrelationLabel(GArffRelation* pRelation, GMatrix* pData, GImage* pImage, int attr, unsigned int bgCol)
{
	pImage->clear(bgCol);
	if(pRelation->valueCount(attr) == 0)
	{
		pImage->text(pRelation->attrName(attr), 0, 0, 1.0f, 0xff400000);
		double min, max;
		pData->minAndRange(attr, &min, &max);
		max += min;

		for(int i = 0; i < 2; i++)
		{
			GImage image2;
			image2.setSize(pImage->width() - 16, 16);
			image2.clear(0);
			int xx = 0;
			char szValue[64];
			sprintf(szValue, "%.4lg", (i == 0 ? min : max));
			int eatspace = pImage->width() - 16 - GImage::measureTextWidth(szValue, 1.0f);
			xx += eatspace;
			image2.text(szValue, xx, 0, 1.0f, 0xff400000);
			GImage image3;
			image3.rotateClockwise90(&image2);
			GRect r3(0, 0, image3.width(), image3.height());
			pImage->blitAlpha((i == 0 ? 0 : pImage->width() - 16), 16, &image3, &r3);
		}
	}
	else
	{
		pImage->text(pRelation->attrName(attr), 0, 0, 1.0f, 0xff000040);
		GImage image2;
		image2.setSize(pImage->width() - 16, 16);
		image2.clear(0);
		GRect r2(0, 0, pImage->width() - 16, 16);

		int valueCount = (int)pRelation->valueCount(attr);
		for(int i = 0; i < valueCount; i++)
		{
			GImage image2;
			image2.setSize(pImage->width() - 16, 16);
			image2.clear(0);
			int xx = 0;
			ostringstream oss;
			pRelation->printAttrValue(oss, attr, i);
			string sValue = oss.str();
			int eatspace = pImage->width() - 16 - GImage::measureTextWidth(sValue.c_str(), 1.0f);
			xx += eatspace;
			image2.text(sValue.c_str(), xx, 0, 1.0f, 0xff000040);
			GImage image3;
			image3.rotateClockwise90(&image2);
			GRect r3(0, 0, image3.width(), image3.height());
			int span = pImage->width() / valueCount;
			int start = std::max(0, (span - 16) / 2);
			pImage->blitAlpha(start + span * i, 16, &image3, &r3);
		}
	}
}

void PlotCorrelations(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GArffRelation* pRel = (GArffRelation*)pData->relation().get();

	// Parse options
	string filename = "plot.png";
	int cellsize = 120;
	int bordersize = 4;
	unsigned int bgCol = 0xffd0d0e0;
	size_t maxAttrs = 30;
	double jitter = 0.03;
	while(args.next_is_flag())
	{
		if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-cellsize"))
			cellsize = args.pop_uint();
		else if(args.if_pop("-jitter"))
			jitter = args.pop_double();
		else if(args.if_pop("-maxattrs"))
			maxAttrs = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Make the chart
	GImage imageBig;
	int wid = (int)(std::min(maxAttrs, pRel->size()) + 1) * (cellsize + bordersize);
	imageBig.setSize(wid, wid);
	imageBig.clear(bgCol);
	GRand prng(getpid() * (unsigned int)time(NULL));
	GImage imageCell;
	GImage imageCell2;
	imageCell.setSize(cellsize, cellsize);
	for(size_t i = 0; i < pRel->size() && i < maxAttrs; i++)
	{
		MakeCorrelationLabel(pRel, pData, &imageCell, (int)i, bgCol);
		GRect r(0, 0, cellsize, cellsize);
		imageBig.blit(((int)i + 1) * (cellsize + bordersize), 0, &imageCell, &r);
		imageCell2.rotateCounterClockwise90(&imageCell);
		imageBig.blit(0, ((int)i + 1) * (cellsize + bordersize), &imageCell2, &r);
	}
	for(size_t y = 0; y < pRel->size() && y < maxAttrs; y++)
	{
		for(size_t x = 0; x < pRel->size() && x < maxAttrs; x++)
		{
			if(x == y)
				MakeAttributeSummaryGraph(pRel, pData, &imageCell, (int)x);
			else
				MakeCorrelationGraph(pRel, pData, &imageCell, (int)x, (int)y, jitter, &prng);
			GRect r(0, 0, cellsize, cellsize);
			imageBig.blit(((int)x + 1) * (cellsize + bordersize), ((int)y + 1) * (cellsize + bordersize), &imageCell, &r);
		}
	}
	imageBig.savePng(filename.c_str());
	cout << "Output saved to " << filename.c_str() << ".\n";
}

class Compare3DPointsByDistanceFromCameraFunctor
{
protected:
	GCamera* m_pCamera;

public:
	Compare3DPointsByDistanceFromCameraFunctor(GCamera* pCamera)
	: m_pCamera(pCamera)
	{
	}

	// returns false if pA is closer than pB
	bool operator() (const double* pA, const double* pB) const
	{
		G3DVector a, b, c, d;
		a.m_vals[0] = pA[0];
		a.m_vals[1] = pA[1];
		a.m_vals[2] = pA[2];
		b.m_vals[0] = pB[0];
		b.m_vals[1] = pB[1];
		b.m_vals[2] = pB[2];
		m_pCamera->project(&a, &c);
		m_pCamera->project(&b, &d);
		return (c.m_vals[2] > d.m_vals[2]);
	}
};

void toImageCoords(GImage* pImage, GCamera* pCamera, G3DVector* pIn, G3DVector* pOut)
{
	pCamera->project(pIn, pOut);

	// Flip the Y value, because positive is down in image coordinates
	pOut->m_vals[1] = pImage->height() - 1 - pOut->m_vals[1];
}

void Plot3d(GImage* pImage, GMatrix* pData, unsigned int bgCol, float pointRadius, double cameraDist, G3DVector* pCameraDirection, bool box, bool labels)
{
	GCamera camera(pImage->width(), pImage->height());
	camera.setViewAngle(M_PI / 3);
	G3DVector mean;
	mean.m_vals[0] = pData->mean(0);
	mean.m_vals[1] = pData->mean(1);
	mean.m_vals[2] = pData->mean(2);
	G3DVector min, max, range;
	pData->minAndRangeUnbiased(0, &min.m_vals[0], &range.m_vals[0]);
	pData->minAndRangeUnbiased(1, &min.m_vals[1], &range.m_vals[1]);
	pData->minAndRangeUnbiased(2, &min.m_vals[2], &range.m_vals[2]);
	max.copy(&range);
	max.add(&min);
	G3DReal dist = sqrt(min.squaredDist(&max)) * cameraDist;
	G3DVector* pCameraPos = camera.lookFromPoint();
	pCameraPos->copy(pCameraDirection);
	pCameraPos->multiply(-1);
	pCameraPos->normalize();
	pCameraPos->multiply(dist);
	pCameraPos->add(&mean);
	camera.setDirection(pCameraDirection, 0.0);

	G3DVector point, coords, point2, coords2;
	pImage->clear(bgCol);

	// Draw box
	if(box)
	{
		min.subtract(&mean);
		min.multiply(1.1);
		min.add(&mean);
		max.subtract(&mean);
		max.multiply(1.1);
		max.add(&mean);
		range.multiply(1.1);
		int x, y, z;
		for(z = 0; z < 2; z++)
		{
			for(y = 0; y < 2; y++)
			{
				for(x = 0; x < 2; x++)
				{
					unsigned int col = 0xff808080;
					if(x == 0 && y == 0 && z == 0)
						col = 0xff8080ff;
					if(x == 0)
					{
						point.set(min.m_vals[0], min.m_vals[1] + y * range.m_vals[1], min.m_vals[2] + z * range.m_vals[2]);
						point2.set(max.m_vals[0], min.m_vals[1] + y * range.m_vals[1], min.m_vals[2] + z * range.m_vals[2]);
						toImageCoords(pImage, &camera, &point, &coords);
						toImageCoords(pImage, &camera, &point2, &coords2);
						pImage->line((int)coords.m_vals[0], (int)coords.m_vals[1], (int)coords2.m_vals[0], (int)coords2.m_vals[1], col);
					}
					if(y == 0)
					{
						point.set(min.m_vals[0] + x * range.m_vals[0], min.m_vals[1], min.m_vals[2] + z * range.m_vals[2]);
						point2.set(min.m_vals[0] + x * range.m_vals[0], max.m_vals[1], min.m_vals[2] + z * range.m_vals[2]);
						toImageCoords(pImage, &camera, &point, &coords);
						toImageCoords(pImage, &camera, &point2, &coords2);
						pImage->line((int)coords.m_vals[0], (int)coords.m_vals[1], (int)coords2.m_vals[0], (int)coords2.m_vals[1], col);
					}
					if(z == 0)
					{
						point.set(min.m_vals[0] + x * range.m_vals[0], min.m_vals[1] + y * range.m_vals[1], min.m_vals[2]);
						point2.set(min.m_vals[0] + x * range.m_vals[0], min.m_vals[1] + y * range.m_vals[1], max.m_vals[2]);
						toImageCoords(pImage, &camera, &point, &coords);
						toImageCoords(pImage, &camera, &point2, &coords2);
						pImage->line((int)coords.m_vals[0], (int)coords.m_vals[1], (int)coords2.m_vals[0], (int)coords2.m_vals[1], col);
					}
				}
			}
		}

		// Draw axis labels
		if(labels)
		{
			{
				char tmp[32];
				GPlotLabelSpacer pls(min.m_vals[0], max.m_vals[0], 10);
				for(int i = 0; i < pls.count(); i++)
				{
					point.set(pls.label(i), min.m_vals[1], min.m_vals[2]);
					toImageCoords(pImage, &camera, &point, &coords);
					pImage->dot((float)coords.m_vals[0], (float)coords.m_vals[1], 3.0, 0xff404040, bgCol);
					sprintf(tmp, "%.5lg", pls.label(i));
					pImage->text(tmp, (int)coords.m_vals[0] + 4, (int)coords.m_vals[1] - 4, 1.0f, 0xff404040);
				}
			}
			{
				char tmp[32];
				GPlotLabelSpacer pls(min.m_vals[1], max.m_vals[1], 10);
				for(int i = 0; i < pls.count(); i++)
				{
					point.set(min.m_vals[0], pls.label(i), min.m_vals[2]);
					toImageCoords(pImage, &camera, &point, &coords);
					pImage->dot((float)coords.m_vals[0], (float)coords.m_vals[1], 3.0, 0xff404040, bgCol);
					sprintf(tmp, "%.5lg", pls.label(i));
					pImage->text(tmp, (int)coords.m_vals[0] + 4, (int)coords.m_vals[1] - 4, 1.0f, 0xff404040);
				}
			}
			{
				char tmp[32];
				GPlotLabelSpacer pls(min.m_vals[2], max.m_vals[2], 10);
				for(int i = 0; i < pls.count(); i++)
				{
					point.set(min.m_vals[0], min.m_vals[1], pls.label(i));
					toImageCoords(pImage, &camera, &point, &coords);
					pImage->dot((float)coords.m_vals[0], (float)coords.m_vals[1], 3.0, 0xff404040, bgCol);
					sprintf(tmp, "%.5lg", pls.label(i));
					pImage->text(tmp, (int)coords.m_vals[0] + 4, (int)coords.m_vals[1] - 4, 1.0f, 0xff404040);
				}
			}
		}
	}

	// Plot the points
	Compare3DPointsByDistanceFromCameraFunctor comparator(&camera);
	GMatrix copy(pData->rows(), 4);
	copy.copyColumns(0, pData, 0, 3);
	for(size_t i = 0; i < copy.rows(); i++)
		copy.row(i)[3] = (double)i;
	copy.sort(comparator);
	for(size_t i = 0; i < copy.rows(); i++)
	{
		double* pVec = copy.row(i);
		point.set(pVec[0], pVec[1], pVec[2]);
		toImageCoords(pImage, &camera, &point, &coords);
		float radius = pointRadius / (float)coords.m_vals[2];
		pImage->dot((float)coords.m_vals[0], (float)coords.m_vals[1], radius, gAHSV(0xff, 0.8f * (float)pVec[3] / copy.rows(), 1.0f, 0.5f), bgCol);
	}
}

void Plot3dMulti(GArgReader& args)
{
	// Load
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	GArffRelation* pRel = (GArffRelation*)pData->relation().get();

	// Parse options
	unsigned int nSeed = getpid() * (unsigned int)time(NULL);
	int horizPlots = 1;
	int vertPlots = 1;
	int wid = 1000;
	int hgt = 1000;
	string filename = "plot.png";
	float pointRadius = 40.0f;
	double cameraDistance = 1.5;
	bool box = true;
	bool labels = true;
	unsigned int cBackground = 0xffffffff;
	G3DVector cameraDirection;
	cameraDirection.set(0.6, -0.3, -0.8);
	bool blast = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-blast"))
			blast = true;
		else if(args.if_pop("-seed"))
			nSeed = args.pop_uint();
		else if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-size"))
		{
			wid = args.pop_uint();
			hgt = args.pop_uint();
		}
		else if(args.if_pop("-pointradius"))
			pointRadius = (float)args.pop_double();
		else if(args.if_pop("-bgcolor"))
			cBackground = hexToRgb(args.pop_string());
		else if(args.if_pop("-cameradistance"))
			cameraDistance = args.pop_double();
		else if(args.if_pop("-cameradirection"))
		{
			cameraDirection.m_vals[0] = args.pop_double();
			cameraDirection.m_vals[1] = args.pop_double();
			cameraDirection.m_vals[2] = args.pop_double();
		}
		else if(args.if_pop("-nobox"))
			box = false;
		else if(args.if_pop("-nolabels"))
			labels = false;
		else
			ThrowError("Invalid option: ", args.peek());
	}
	if(blast)
	{
		pointRadius /= 5;
		wid /= 5;
		hgt /= 5;
		horizPlots *= 5;
		vertPlots *= 5;
	}

	// Check values
	if(pRel->size() != 3)
		ThrowError("Sorry, only data with 3 dims is currently supported");
	if(!pRel->areContinuous(0,3))
		ThrowError("Sorry, only continuous attributes are currently supported");

	// Make plots
	GRand prng(nSeed);
	GImage masterImage;
	masterImage.setSize(horizPlots * wid, vertPlots * hgt);
	GImage tmpImage;
	tmpImage.setSize(wid, hgt);
	for(int y = 0; y < vertPlots; y++)
	{
		for(int x = 0; x < horizPlots; x++)
		{
			if(blast)
			{
				cameraDirection.m_vals[0] = prng.normal();
				cameraDirection.m_vals[1] = prng.normal();
				cameraDirection.m_vals[2] = prng.normal();
				cameraDirection.normalize();
				cout << "row " << y << ", col " << x << ", cameradirection " << cameraDirection.m_vals[0] << " " << cameraDirection.m_vals[1] << " " << cameraDirection.m_vals[2] << "\n";
			}
			Plot3d(&tmpImage, pData, cBackground, pointRadius, cameraDistance, &cameraDirection, box, labels);
			GRect r(0, 0, wid, hgt);
			masterImage.blit(wid * x, hgt * y, &tmpImage, &r);
		}
	}
	masterImage.savePng(filename.c_str());
	cout << "Plot saved to " << filename.c_str() << ".\n";
}

void PrintStats(GArgReader& args)
{
	// Load
	const char* szFilename = args.pop_string();
	GMatrix* pData = loadData(szFilename);
	Holder<GMatrix> hData(pData);
	GArffRelation* pRel = (GArffRelation*)pData->relation().get();

	// Print some quick stats
	cout.precision(8);
	cout << "Filename: " << szFilename << "\n";
	cout << "Patterns: " << pData->rows() << "\n";
	int continuousAttrs = 0;
	for(size_t i = 0; i < pRel->size(); i++)
	{
		if(pRel->valueCount(i) == 0)
			continuousAttrs++;
	}
	cout << "Attributes: " << pRel->size() << " (Continuous:" << continuousAttrs << ", Nominal:" << pRel->size() - continuousAttrs << ")\n";
	size_t stepSize = pRel->size() / 10;
	if(stepSize < 4)
		stepSize = 1;
	for(size_t i = 0; i < pRel->size();)
	{
		cout << "  " << i << ") " << pRel->attrName(i) << ", ";
		if(pRel->valueCount(i) == 0)
		{
			cout << "Type: Continuous, ";
			try
			{
				double d1, d2, d3;
				d1 = pData->mean(i);
				d2 = pData->variance(i, d1);
				d3 = pData->median(i);
				cout << "Mean:" << d1 << ", Dev:" << sqrt(d2) << ", Median:" << d3 << ", ";
				pData->minAndRange(i, &d1, &d2);
				cout << "Min:" << d1 << ", Max:" << d1 + d2 << ", ";
			}
			catch(...)
			{
				// If it gets to here, all values are probably missing
			}
			cout << "Missing:" << pData->countValue(i, UNKNOWN_REAL_VALUE) << "\n";
		}
		else
		{
			cout << "Type: Nominal, ";
			cout << "Values:" << pRel->valueCount(i) << ", ";
			cout << "Entropy: " << pData->entropy(i) << ", ";
			cout << "Missing:" << pData->countValue(i, UNKNOWN_DISCRETE_VALUE);
			if(pRel->valueCount(i) < 9)
			{
				cout << "\n";
				for(size_t j = 0; j < pRel->valueCount(i); j++)
				{
					size_t occurrences = pData->countValue(i, (double)j);
					cout << "     " 
							 << ((double)occurrences * 100.0 / (double)pData->rows()) 
							 << "% (" << occurrences << ") ";
					pRel->printAttrValue(cout, i, (double)j);
					cout << "\n";
				}
			}
			else
			{
				cout << ", ";
				int nMostCommonVal = (int)pData->baselineValue(i);
				size_t mostCommonOccurrences = pData->countValue(i, nMostCommonVal);
				cout << "Most Common:";
				pRel->printAttrValue(cout, i, nMostCommonVal);
				cout << " (" << ((double)mostCommonOccurrences * 100.0 / pData->rows()) << "%)\n";
			}
		}
		if(i < 2)
			i++;
		else if(i + stepSize >= pRel->size() - 3)
			i += std::max(1, (int)pRel->size() - 3 - (int)i);
		else
			i += stepSize;
	}
}

void overlay(GArgReader& args)
{
	GImage a, b, c;
	a.loadPng(args.pop_string());
	b.loadPng(args.pop_string());
	if(a.width() != b.width())
		ThrowError("Images have different widths");
	if(a.height() != b.height())
		ThrowError("Images have different heights");
	c.setSize(a.width(), a.height());

	// options
	string filename = "plot.png";
	unsigned int backcolor = 0xffffffff; // white
	int tolerance = 0;
	while(args.next_is_flag())
	{
		if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-backcolor"))
			backcolor = hexToRgb(args.pop_string());
		else if(args.if_pop("-tolerance"))
			tolerance = (int)args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Make the combination image
	for(int y = 0; y < (int)c.height(); y++)
	{
		for(int x = 0; x < (int)c.width(); x++)
		{
			unsigned int pix = b.pixel(x, y);
			if(	std::abs((int)gRed(pix) - (int)gRed(backcolor)) +
				std::abs((int)gGreen(pix) - (int)gGreen(backcolor)) +
				std::abs((int)gBlue(pix) - (int)gBlue(backcolor)) <= tolerance)
				pix = a.pixel(x, y);
			c.setPixel(x, y, pix);
		}
	}
	c.savePng(filename.c_str());
}

void percentSame(GArgReader& args){
  Holder<GMatrix> hData1(loadData(args.pop_string()));
  Holder<GMatrix> hData2(loadData(args.pop_string()));
  const size_t cols = hData1->cols();
  const size_t rows = hData1->rows();
  if(hData1->cols() != hData2->cols()){
    ThrowError("The two files have different numbers of attributes.  Cannot "
	       "compare entries when the number of columns is different");
  }
  if(hData1->rows() != hData2->rows()){
    ThrowError("The two files have different numbers of tuples.  Cannot "
	       "compare entries when the number of rows is different");
  }
  if(rows == 0){
    ThrowError("The files have no rows.  Cannot calculate the percentage of "
	       "identical values for empty files.");
  }
  for(size_t i = 0; i < cols; ++i){
    if(hData1->relation()->valueCount(i) != 
       hData2->relation()->valueCount(i)){
      size_t v1 = hData1->relation()->valueCount(i);
      size_t v2 = hData2->relation()->valueCount(i);
      std::stringstream msg;
      msg << "The two files have different attribute types at "
	  << "attribute index " << i << ".  The first file has ";
      if(v1 == 0){ 
	msg << "a continuous attribute.  ";
      }else{
	msg << "a nominal attribute with " << v1 << " values.  ";
      }
      
      msg << "The second file has ";
      if(v2 == 0){ 
	msg << "a continuous attribute.";
      }else{
	msg << "a nominal attribute with " << v2 << " values.";
      }
      ThrowError(msg.str());
    }
  }
  //Count the same values
  vector<size_t> numSame(cols, 0);
  for(size_t row = 0; row < rows; ++row){
    const double *r1 = hData1->row(row);
    const double *r2 = hData2->row(row);
    for(size_t col = 0; col < cols; ++col){
      if(r1[col] == r2[col]){ ++numSame[col]; }
    }
  }

  //Convert to percents
  vector<double> pctSame;
  pctSame.resize(numSame.size());
  for(size_t col = 0; col < cols; ++col){
    pctSame[col] = 100.0 * (double)numSame[col] / rows;
  }
  
  //Print
  for(size_t col = 0; col < cols; ++col){
    cout << pctSame[col] << " %  ";
  }
  cout << std::endl;
}

void printDecisionTree(GArgReader& args)
{
	// Load the model
	GDom doc;
	if(args.size() < 1)
		ThrowError("Model not specified.");
	doc.loadJson(args.pop_string());
	GRand prng(0);
	GLearnerLoader ll(prng, true);
	if(_stricmp(doc.root()->field("class")->asString(), "GDecisionTree") != 0)
		ThrowError("That model is not a decision tree");
	GSupervisedLearner* pModeler = ll.loadSupervisedLearner(doc.root());
	Holder<GSupervisedLearner> hModeler(pModeler);

	GMatrix* pData = NULL;
	if(args.size() > 0)
	{
		size_t ld;
		pData = loadDataWithSwitches(args, &ld);
		Holder<GMatrix> hData(pData);
		size_t labelDims = pModeler->labelDims();
		if(ld != labelDims)
			ThrowError("Different number of label dims than the model was trained with");
		GArffRelation relFeatures;
		relFeatures.addAttrs(pData->relation().get(), 0, pData->cols() - labelDims);
		GArffRelation relLabels;
		relLabels.addAttrs(pData->relation().get(), pData->cols() - labelDims, labelDims);
		((GDecisionTree*)pModeler)->print(cout, &relFeatures, &relLabels);
	}
	else
		((GDecisionTree*)pModeler)->print(cout);
}

void model(GArgReader& args)
{
	// Load the model
	GDom doc;
	if(args.size() < 1)
		ThrowError("Model not specified");
	doc.loadJson(args.pop_string());
	GRand prng(0);
	GLearnerLoader ll(prng, true);
	GSupervisedLearner* pModeler = ll.loadSupervisedLearner(doc.root());
	Holder<GSupervisedLearner> hModeler(pModeler);

	// Load the data
	if(args.size() < 1)
		ThrowError("Expected the filename of a dataset");
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	if(pData->cols() != pModeler->featureDims() + pModeler->labelDims())
		ThrowError("Model was trained with a different number of attributes than in this data");

	// Get other parameters
	unsigned int attrx = args.pop_uint();
	if(pData->relation()->valueCount(attrx) != 0)
		ThrowError("Sorry, currently only continuous attributes can be plotted");
	unsigned int attry = args.pop_uint();
	if(pData->relation()->valueCount(attry) != 0)
		ThrowError("Sorry, currently only continuous attributes can be plotted");
	size_t featureDims = pModeler->featureDims();
	if(attrx >= (unsigned int)featureDims || attry >= (unsigned int)featureDims)
		ThrowError("feature attribute out of range");

	// Parse options
	int width = 400;
	int height = 400;
	int labelDim = 0;
	float dotRadius = 3.0f;
	string filename = "plot.png";
	while(args.next_is_flag())
	{
		if(args.if_pop("-size"))
		{
			width = args.pop_uint();
			height = args.pop_uint();
		}
		else if(args.if_pop("-pointradius"))
			dotRadius = (float)args.pop_double();
		else if(args.if_pop("-out"))
			filename = args.pop_string();
		else
			ThrowError("Invalid option: ", args.peek());
	}


	// Compute label range
	double labelMin = 0.0;
	double labelRange = (double)pData->relation()->valueCount(featureDims + labelDim);
	if(labelRange == 0.0)
		pData->minAndRangeUnbiased(featureDims + labelDim, &labelMin, &labelRange);

	// Plot the data
	double xmin, xrange, ymin, yrange;
	pData->minAndRangeUnbiased(attrx, &xmin, &xrange);
	pData->minAndRangeUnbiased(attry, &ymin, &yrange);
	GImage image;
	image.setSize(width, height);
	GPlotWindow pw(&image, xmin, ymin, xmin + xrange, ymin + yrange);
	GTEMPBUF(double, features, pData->cols());
	double* labels = features + featureDims;
	unsigned int* pPix = image.pixels();
	size_t step = std::max((size_t)1, pData->rows() / 100);
	double xx, yy;
	for(int y = 0; y < height; y++)
	{
		cout << ((float)y * 100.0f / height) << "%       \r";
		cout.flush();
		for(int x = 0; x < width; x++)
		{
			pw.viewToWindow(x, y, &xx, &yy);
			size_t r = 0;
			size_t g = 0;
			size_t b = 0;
			size_t count = 0;
			for(size_t i = 0; i < pData->rows(); i += step)
			{
				GVec::copy(features, pData->row(i), featureDims);
				features[attrx] = xx;
				features[attry] = yy;
				pModeler->predict(features, labels);
				unsigned int hue = gAHSV(0xff, std::max(0.0f, std::min(1.0f, (float)((labels[labelDim] - labelMin) / labelRange))), 1.0f, 0.5f);
				r += gRed(hue);
				g += gGreen(hue);
				b += gBlue(hue);
				count++;
			}
			r /= count;
			g /= count;
			b /= count;
			*pPix = gARGB(0xff, ClipChan((int)r), ClipChan((int)g), ClipChan((int)b));
			pPix++;
		}
	}
	cout << "                \n";
	cout.flush();

	// Plot the data
	for(size_t i = 0; i < pData->rows(); i++)
	{
		double* pRow = pData->row(i);
		pw.dot(pRow[attrx], pRow[attry], dotRadius, gAHSV(0xff, std::max(0.0f, std::min(1.0f, (float)((pRow[featureDims + labelDim] - labelMin) / labelRange))), 1.0, 1.0), 0xff000000);
	}

	image.savePng(filename.c_str());
	cout << "Output saved to " << filename.c_str() << ".\n";
}

void rayTraceManifoldModel(GArgReader& args)
{
	// Load the model
	GDom doc;
	if(args.size() < 1)
		ThrowError("Model not specified");
	doc.loadJson(args.pop_string());
	GRand prng(0);
	GLearnerLoader ll(prng, true);
	GSupervisedLearner* pModeler = ll.loadSupervisedLearner(doc.root());
	Holder<GSupervisedLearner> hModeler(pModeler);
	if(pModeler->featureDims() != 2 || pModeler->labelDims() != 3)
		ThrowError("The model has ", to_str(pModeler->featureDims()), " inputs and ", to_str(pModeler->labelDims()), " outputs. 2 real inputs and 3 real outputs are expected");

	// Parse options
	int width = 400;
	int height = 400;
	double amin = 0.0;
	double amax = 1.0;
	double bmin = 0.0;
	double bmax = 1.0;
/*	double xmin = 0.0;
	double xmax = 1.0;
	double ymin = 0.0;
	double ymax = 1.0;
	double zmin = 0.0;
	double zmax = 1.0;*/
	size_t granularity = 50;
	string filename = "plot.png";
	double pointRadius = 0.02;
	GMatrix* pPoints = NULL;
	Holder<GMatrix> hPoints;
	while(args.next_is_flag())
	{
		if(args.if_pop("-size"))
		{
			width = args.pop_uint();
			height = args.pop_uint();
		}
		else if(args.if_pop("-out"))
			filename = args.pop_string();
		else if(args.if_pop("-domain"))
		{
			amin = args.pop_double();
			amax = args.pop_double();
			bmin = args.pop_double();
			bmax = args.pop_double();
		}
/*		else if(args.if_pop("-range"))
		{
			xmin = args.pop_double();
			xmax = args.pop_double();
			ymin = args.pop_double();
			ymax = args.pop_double();
			zmin = args.pop_double();
			zmax = args.pop_double();
		}*/
		else if(args.if_pop("-points"))
		{
			delete(pPoints);
			pPoints = GMatrix::loadArff(args.pop_string());
			hPoints.reset(pPoints);
			if(pPoints->cols() != 3)
				ThrowError("Expected 3-dimensional points");
		}
		else if(args.if_pop("-pointradius"))
			pointRadius = args.pop_double();
		else if(args.if_pop("-granularity"))
			granularity = args.pop_uint();
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Set up the scene
	GRayTraceScene scene(&prng);
	scene.setBackgroundColor(1.0, 0.0, 0.0, 0.0);
	scene.setAmbientLight(0.9, 0.9, 0.9);
	scene.addLight(new GRayTraceDirectionalLight(0.1, 0.2, 0.3, // direction
							0.8, 0.8, 1.0, // color
							0.0)); // jitter
	scene.addLight(new GRayTraceDirectionalLight(-0.1, -0.2, 0.3, // direction
							0.8, 1.0, 0.8, // color
							0.0)); // jitter
	scene.addLight(new GRayTraceDirectionalLight(-0.1, 0.9, -0.1, // direction
							0.3, 0.2, 0.2, // color
							0.0)); // jitter
	GRayTraceCamera* pCamera = scene.camera();
	pCamera->setImageSize(width, height);
	pCamera->setViewAngle(M_PI / 3);
	G3DVector mean;
	mean.set(0, 0, 0);
	
	
	G3DVector cameraDirection(-.35, -0.15, -0.5);
//	G3DVector cameraDirection(0.1, -0.25, -0.85);
	G3DReal dist = 2.0;
	G3DVector* pCameraPos = pCamera->lookFromPoint();
	pCameraPos->copy(&cameraDirection);
	pCameraPos->multiply(-1);
	pCameraPos->normalize();
	pCameraPos->multiply(dist);
	pCameraPos->add(&mean);
	pCamera->setDirection(&cameraDirection, 0.0);

	// Make bluish material
	GRayTracePhysicalMaterial* pMat1 = new GRayTracePhysicalMaterial();
	scene.addMaterial(pMat1);
	pMat1->setColor(GRayTraceMaterial::Diffuse, 0.3, 0.4, 0.6);
	pMat1->setColor(GRayTraceMaterial::Specular, 0.4, 0.4, 0.6);
	pMat1->setColor(GRayTraceMaterial::Reflective, 0.2, 0.2, 0.3);
	pMat1->setColor(GRayTraceMaterial::Transmissive, 0.7, 0.7, 0.8);

	// Make yellowish material
	GRayTracePhysicalMaterial* pMat2 = new GRayTracePhysicalMaterial();
	scene.addMaterial(pMat2);
	pMat2->setColor(GRayTraceMaterial::Diffuse, 0.4, 0.4, 0.05);
	pMat2->setColor(GRayTraceMaterial::Specular, 1.0, 1.0, 0.8);
	pMat2->setColor(GRayTraceMaterial::Reflective, 0.5, 0.5, 0.3);

	// Make the surface
	double in[2];
	double astep = (amax - amin) / (std::max((size_t)2, granularity) - 1);
	double bstep = (bmax - bmin) / (std::max((size_t)2, granularity) - 1);
	for(in[1] = bmin; in[1] + bstep <= bmax; in[1] += bstep)
	{
		for(in[0] = amin; in[0] + astep <= amax; )
		{
			// Predict the 4 corners
			G3DVector v1, v2, v3, v4;
			pModeler->predict(in, v1.vals());
			in[1] += bstep;
			pModeler->predict(in, v3.vals());
			in[0] += astep;
			pModeler->predict(in, v4.vals());
			in[1] -= bstep;
			pModeler->predict(in, v2.vals());

			// Add a quad surface
			scene.addMesh(GRayTraceTriMesh::makeQuadSurface(pMat1, &v1, &v3, &v4, &v2));
		}
	}

	// Make the points
	if(pPoints)
	{
		for(size_t i = 0; i < pPoints->rows(); i++)
		{
			double* pVec = pPoints->row(i);
			scene.addObject(new GRayTraceSphere(pMat2, pVec[0], pVec[1], pVec[2], pointRadius));
		}
	}

//	scene.addObject(new GRayTraceSphere(pMat2, .5,.5,.5, 0.02)); // xyzr

	// Ray-trace the scene
	scene.render();
	GImage* pImage = scene.image();
	pImage->savePng(filename.c_str());
	cout << "Output saved to " << filename.c_str() << ".\n";
}

void rowToImage(GArgReader& args)
{
	GMatrix* pData = loadData(args.pop_string());
	Holder<GMatrix> hData(pData);
	unsigned int r = args.pop_uint();
	if(r > pData->rows())
		ThrowError("row index out of range");
	unsigned int width = args.pop_uint();

	string filename = "plot.png";
	int channels = 3;
	double range = 255.0;

	size_t cols = pData->cols();
	if((cols % (channels * width)) != 0)
		ThrowError("The row has ", to_str(cols), " dims, which is not a multiple of ", to_str(channels), " channels times ", to_str(width), " pixels wide");
	double* pRow = pData->row(r);
	unsigned int height = (unsigned int)cols / (unsigned int)(channels * width);
	GImage image;
	GVec::toImage(pRow, &image, width, height, channels, range);
	image.savePng(filename.c_str());
	cout << "Image saved to " << filename.c_str() << ".\n";
}

void systemFrames(GArgReader& args)
{
	GDom doc;
	doc.loadJson(args.pop_string());
	GMatrix* pActions = loadData(args.pop_string());
	Holder<GMatrix> hActions(pActions);
	GMatrix* pObs = NULL;
	Holder<GMatrix> hObs(NULL);

	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool calibrate = false;
	int frameWidth = 256;
	int stepsPerFrame = 1;
	double scalePredictions = 1.0;
	string outFilename = "frames.png";
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-calibrate"))
			calibrate = true;
		else if(args.if_pop("-framewidth"))
			frameWidth = args.pop_uint();
		else if(args.if_pop("-stepsperframe"))
			stepsPerFrame = args.pop_uint();
		else if(args.if_pop("-scalepredictions"))
			scalePredictions = args.pop_double();
		else if(args.if_pop("-out"))
			outFilename = args.pop_string();
		else if(args.if_pop("-observations"))
		{
			pObs = loadData(args.pop_string());
			hObs.reset(pObs);
		}
		else
			ThrowError("Invalid option: ", args.peek());
	}

	// Instantiate the model
	GRand prng(seed);
	GRecurrentModel rm(doc.root(), &prng);
	GImage* pImage = rm.frames(pActions, pObs, calibrate, frameWidth, stepsPerFrame, scalePredictions);
	Holder<GImage> hImage(pImage);
	pImage->savePng(outFilename.c_str());
	cout << "Frames saved to " << outFilename.c_str() << ".\n";
}

void ubpFrames(GArgReader& args)
{
	const char* szModelFilename = args.pop_string();
	size_t imageWid = args.pop_uint();
	size_t imageHgt = args.pop_uint();
	size_t framesHoriz = args.pop_uint();
	size_t framesVert = args.pop_uint();
	const char* outFilename = args.pop_string();

	GDom doc;
	doc.loadJson(szModelFilename);
	GRand rand(0);
	GLearnerLoader ll(rand);
	GUnsupervisedBackProp* pUBP = new GUnsupervisedBackProp(doc.root(), ll);
	Holder<GUnsupervisedBackProp> hUBP(pUBP);

	size_t featureDims = pUBP->featureDims();
	GTEMPBUF(double, pFeatures, featureDims);
	GVec::setAll(pFeatures, 0.5, featureDims);
	size_t labelDims = pUBP->labelDims();
	GTEMPBUF(double, pLabels, labelDims);
	GImage image;
	image.setSize(imageWid * framesHoriz, imageHgt * framesVert);
	size_t yy = 0;
	for(size_t vFrame = 0; vFrame < framesVert; vFrame++)
	{
		size_t xx = 0;
		for(size_t hFrame = 0; hFrame < framesHoriz; hFrame++)
		{
			pFeatures[featureDims - 2] = (double)hFrame / (framesHoriz - 1);
			pFeatures[featureDims - 1] = (double)vFrame / (framesVert - 1);
			pUBP->lowToHi(pFeatures, pLabels);
			GImage tmp;
			GVec::toImage(pLabels, &tmp, imageWid, imageHgt, pUBP->neuralNet()->labelDims(), 256.0);
			image.blit(xx, yy, &tmp);
			xx += imageWid;
		}
		yy += imageHgt;
	}
	image.savePng(outFilename);
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makePlotUsageTree();
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
	UsageNode* pUsageTree = makePlotUsageTree();
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
	PathData pd;
	GFile::parsePath(argv[0], &pd);
	const char* appName = argv[0] + pd.fileStart;
	GArgReader args(argc, argv);
	args.pop_string(); // advance past the app name
	int ret = 0;
	try
	{
		if(args.size() < 1) ThrowError("Expected a command");
		else if(args.if_pop("usage")) ShowUsage(appName);
		else if(args.if_pop("3d")) Plot3dMulti(args);
		else if(args.if_pop("bar")) PlotBar(args);
		else if(args.if_pop("bigo")) EstimateBigO(args);
		else if(args.if_pop("equation")) PlotEquation(args);
		else if(args.if_pop("histogram")) makeHistogram(args);
		else if(args.if_pop("model")) model(args);
		else if(args.if_pop("overview")) PlotCorrelations(args);
		else if(args.if_pop("rowtoimage")) rowToImage(args);
		else if(args.if_pop("overlay")) overlay(args);
		else if(args.if_pop("percentsame")) percentSame(args);
		else if(args.if_pop("printdecisiontree")) printDecisionTree(args);
		else if(args.if_pop("raytracesurface")) rayTraceManifoldModel(args);
		else if(args.if_pop("scatter")) PlotScatter(args);
		else if(args.if_pop("semanticmap")) semanticMap(args);
		else if(args.if_pop("stats")) PrintStats(args);
		else if(args.if_pop("systemframes")) systemFrames(args);
		else if(args.if_pop("ubpframes")) ubpFrames(args);
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

