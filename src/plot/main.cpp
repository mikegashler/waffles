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
#include "../GClasses/GVec.h"
#include "../GClasses/GDom.h"
#include "../GClasses/GPlot.h"
#include "../GClasses/GFunction.h"
#include "../GClasses/GLearner.h"
#include "../GClasses/GNeighborFinder.h"
#include "../GClasses/GNeuralNet.h"
#include "../GClasses/GManifold.h"
#include "../GClasses/GSelfOrganizingMap.h"
#include "../GClasses/usage.h"
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
#include <memory>

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

void loadDataWithSwitches(GMatrix& data, GArgReader& args, size_t* pLabelDims)
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
			throw Ex("Invalid option: ", args.peek());
	}

	// Throw out the ignored attributes
	std::sort(ignore.begin(), ignore.end());
	for(size_t i = ignore.size() - 1; i < ignore.size(); i--)
	{
		data.deleteColumns(ignore[i], 1);
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
	*pLabelDims = std::max((size_t)1, labels.size());
	for(size_t i = 0; i < labels.size(); i++)
	{
		size_t src = labels[i];
		size_t dst = data.cols() - *pLabelDims + i;
		if(src != dst)
		{
			data.swapColumns(src, dst);
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
	std::unique_ptr<GMatrix> hM(pM);
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
	std::unique_ptr<UsageNode> hNFTree(pNFTree);
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
			std::unique_ptr<GMatrix> hControlData(pControlData);
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

void PlotBar(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	std::unique_ptr<GMatrix> hData(pData);
	//if(!pData->relation()->areContinuous())
	//	throw Ex("Expected all attributes to be continuous");

	// Parse options
	double ymin = UNKNOWN_REAL_VALUE;
	double ymax = UNKNOWN_REAL_VALUE;
	double padding = 0.1;
	double thickness = 2.0;
	double spacing = 1.0;
	double textSize = 1.0;
	size_t width = 960;
	size_t height = 540;
	size_t row = 0;
	bool serifs = true;
	size_t marks = 30;
	vector<string> labels;
	while(args.next_is_flag())
	{
		if(args.if_pop("-range"))
		{
			ymin = args.pop_double();
			ymax = args.pop_double();
		}
		else if(args.if_pop("-row"))
			row = args.pop_uint();
		else if(args.if_pop("-pad"))
			padding = args.pop_double();
		else if(args.if_pop("-thickness"))
			thickness = args.pop_double();
		else if(args.if_pop("-spacing"))
			spacing = args.pop_double();
		else if(args.if_pop("-textsize"))
			textSize = args.pop_double();
		else if(args.if_pop("-noserifs"))
			serifs = false;
		else if(args.if_pop("-marks"))
			marks = args.pop_uint();
		else if(args.if_pop("-size"))
		{
			width = args.pop_uint();
			height = args.pop_uint();
		}
		else if(args.if_pop("-labels"))
		{
			for(size_t i = 0; i < pData->cols(); i++)
				labels.push_back(args.pop_string());
		}
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Determine the range
	GVec& pRow = pData->row(row);
	if(ymin == UNKNOWN_REAL_VALUE || ymax == UNKNOWN_REAL_VALUE)
	{
		ymin = pRow[0];
		ymax = pRow[0];
		if(ymin == UNKNOWN_REAL_VALUE)
			throw Ex("Unknown values are not supported");
		for(size_t i = 1; i < pData->cols(); i++)
		{
			if(pRow[i] == UNKNOWN_REAL_VALUE)
				throw Ex("Unknown values are not supported");
			ymin = std::min(ymin, pRow[i]);
			ymax = std::max(ymax, pRow[i]);
		}
		double d = padding * (ymax - ymin);
		ymin -= d;
		ymax += d;
	}

	// Determine the labels
	while(labels.size() < pData->cols())
	{
		if(pData->relation().type() == GRelation::ARFF)
		{
			const GArffRelation* pRel = (GArffRelation*)&pData->relation();
			labels.push_back(pRel->attrName(labels.size()));
		}
		else
		{
			string s = "Attr ";
			s += to_str(labels.size());
			labels.push_back(s);
		}
	}

	// Make the chart
	GSVG svg(width, height);
	svg.newChart(0, ymin, pData->cols() * thickness + (pData->cols() + 1) * spacing, ymax);
	if(marks > 0)
		svg.vertMarks((int)marks);
	double x = spacing;
	double base = std::max(ymin, 0.0);
	for(size_t i = 0; i < pData->cols(); i++)
	{
		// Draw the bar
		unsigned int c = gAHSV(0xff, 0.85f * (float)i / pData->cols(), 1.0f, 0.5f);
		double bot = base;
		double hgt = pRow[i] - base;
		if(hgt < 0.0)
		{
			bot = hgt;
			hgt = -hgt;
		}
		svg.rect(x, bot, thickness, hgt, c);

		// Draw the label
		svg.text(x + 0.5 * thickness, ymin - 10 * svg.vunit(), labels[i].c_str(), textSize, GSVG::End, 0xff000000, 45.0, serifs);

		x += thickness;
		x += spacing;
	}
	svg.print(cout);
}

void PlotEquation(GArgReader& args)
{
	// Parse options
	size_t width = 960;
	size_t height = 540;
	double margin = 100.0;
	size_t maxHorizMarks = 30;
	size_t maxVertMarks = size_t(-1);
	double xmin = -10;
	double ymin = -5;
	double xmax = 10;
	double ymax = 5;
	bool serifs = true;
	bool aspect = false;
	bool horizMarks = true;
	bool vertMarks = true;
	bool text = true;
	double thickness = 1.0;
	while(args.next_is_flag())
	{
		if(args.if_pop("-size"))
		{
			width = args.pop_uint();
			height = args.pop_uint();
		}
		else if(args.if_pop("-margin"))
			margin = args.pop_double();
		else if(args.if_pop("-horizmarks"))
			maxHorizMarks = args.pop_uint();
		else if(args.if_pop("-vertmarks"))
			maxVertMarks = args.pop_uint();
		else if(args.if_pop("-range"))
		{
			xmin = args.pop_double();
			ymin = args.pop_double();
			xmax = args.pop_double();
			ymax = args.pop_double();
		}
		else if(args.if_pop("-nohmarks"))
			horizMarks = false;
		else if(args.if_pop("-novmarks"))
			vertMarks = false;
		else if(args.if_pop("-notext"))
			text = false;
		else if(args.if_pop("-nogrid"))
		{
			horizMarks = false;
			vertMarks = false;
		}
		else if(args.if_pop("-noserifs"))
			serifs = false;
		else if(args.if_pop("-aspect"))
			aspect = true;
		else if(args.if_pop("-thickness"))
			thickness = args.pop_double();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Accumulate the expression
	string expr;
	while(args.size() > 0)
		expr += args.pop_string();

	// Parse the expression
	GFunctionParser mfp;
	mfp.add(expr.c_str());

	// Make the chart
	if(aspect)
	{
		if((xmax - xmin) / width < (ymax - ymin) / height)
		{
			double dif = 0.5 * ((ymax - ymin) * width / height - (xmax - xmin));
			xmin -= dif;
			xmax += dif;
		}
		else
		{
			double dif = 0.5 * ((xmax - xmin) * height / width - (ymax - ymin));
			ymin -= dif;
			ymax += dif;
		}
	}
	GSVG svg(width, height);
	svg.newChart(xmin, ymin, xmax, ymax, 0, 0, margin);
	if(horizMarks)
		svg.horizMarks((int)maxHorizMarks, !text);
	if(vertMarks)
	{
		if(maxVertMarks == INVALID_INDEX)
			maxVertMarks = maxHorizMarks * height / width;
		svg.vertMarks((int)maxVertMarks, !text);
	}

	// Draw the equation as the label under the graph
	if(text)
		svg.text(0.5 * (xmin + xmax), svg.horizLabelPos(), expr.c_str(), 1.5, GSVG::Middle, 0xff000000, 0, serifs);

	// Count the equations
	size_t equationCount = 0;
	char szFuncName[32];
	for(size_t i = 1; true; i++)
	{
		sprintf(szFuncName, "f%d", (int)i);
		GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
		if(!pFunc)
		{
			if(i == 1)
				throw Ex("There is no function named \"f1\". Nothing to plot.");
			break;
		}
		equationCount++;
	}

	// Plot all the functions
	svg.clip();
	for(size_t i = 0; i < equationCount; i++)
	{
		// Find the function
		sprintf(szFuncName, "f%d", (int)i + 1);
		GFunction* pFunc = mfp.getFunctionNoThrow(szFuncName);
		if(!pFunc)
			break;
		if(pFunc->m_expectedParams != 1)
			throw Ex("The function ", szFuncName, " takes ", to_str(pFunc->m_expectedParams), " parameters. Expected a function with 1 parameter");

		// Plot it
		unsigned int col;
		if(equationCount < 6)
			col = gAHSV(0xff, i / 6.0f, 1.0f, 0.5f);
		else
			col = gAHSV(0xff, i / ((float)equationCount * 1.25f), 1.0f, 0.5f);
		double dx = 2.0 * svg.hunit();
		vector<double> params;
		double x = xmin;
		params.push_back(x);
		double y = pFunc->call(params, mfp);
		while(x <= xmax)
		{
			double xPrev = x;
			double yPrev = y;
			x += dx;
			params[0] = x;
			y = pFunc->call(params, mfp);
			if(y > -1e100 && y < 1e100 && yPrev > -1e100 && yPrev < 1e100)
				svg.line(xPrev, yPrev, x, y, thickness, col);
		}
	}

	// output the plot
	svg.print(cout);
}

class ScatterCol
{
public:
	enum ColType
	{
		Fixed,
		Row,
		Attr,
	};

	ColType m_type;
	size_t m_color;
	size_t m_attrX;
	size_t m_attrY;
	double m_radius;
	double m_thickness;
	GFunctionParser* m_pFP;
	GFunction* m_pFunc;

	ScatterCol()
	: m_pFP(NULL), m_pFunc(NULL)
	{
	}

	~ScatterCol()
	{
		delete(m_pFP);
	}

	void parse(GArgReader& args, size_t cols)
	{
		// Parse the color
		m_type = Fixed;
		m_color = 0xff000000;
		if(args.peek()[0] == '#')
		{
			const char* hex = args.pop_string();
			m_color = hexToRgb(hex + 1);
		}
		else if(args.if_pop("row"))
			m_type = Row;
		else if(args.if_pop("red"))
			m_color = 0xff800000;
		else if(args.if_pop("pink"))
			m_color = 0xffffc0c0;
		else if(args.if_pop("peach"))
			m_color = 0xffffc080;
		else if(args.if_pop("orange"))
			m_color = 0xffff8000;
		else if(args.if_pop("brown"))
			m_color = 0xffa06000;
		else if(args.if_pop("yellow"))
			m_color = 0xffd0d000;
		else if(args.if_pop("green"))
			m_color = 0xff008000;
		else if(args.if_pop("cyan"))
			m_color = 0xff008080;
		else if(args.if_pop("blue"))
			m_color = 0xff000080;
		else if(args.if_pop("purple"))
			m_color = 0xff8000ff;
		else if(args.if_pop("magenta"))
			m_color = 0xff800080;
		else if(args.if_pop("black"))
			m_color = 0xff000000;
		else if(args.if_pop("gray"))
			m_color = 0xff808080;
		else
		{
			m_type = Attr;
			m_color = args.pop_uint();
			if(m_color >= cols)
				throw Ex("column index ", to_str(m_color), " out of range. Expected 0-", to_str(cols - 1));
		}

		// Parse the X attribute
		if(args.if_pop("equation"))
		{
			if(m_type != Fixed)
				throw Ex("Sorry, only fixed colors are compatible with equations");
			m_attrX = 0;
			m_attrY = 0;
			m_pFP = new GFunctionParser();
			m_pFP->add(args.pop_string());
			m_pFunc = m_pFP->getFunctionNoThrow("f");
			if(!m_pFunc)
				throw Ex("Expected a function named \"f\"");
		}
		else if(args.if_pop("row"))
			m_attrX = INVALID_INDEX;
		else
		{
			m_attrX = args.pop_uint();
			if(m_attrX >= cols)
				throw Ex("column index ", to_str(m_attrX), " out of range. Expected 0-", to_str(cols - 1));
		}

		// Parse the Y attribute
		if(!m_pFunc)
		{
			if(args.if_pop("row"))
				m_attrY = INVALID_INDEX;
			else
			{
				m_attrY = args.pop_uint();
				if(m_attrY >= cols)
					throw Ex("column index ", to_str(m_attrY), " out of range. Expected 0-", to_str(cols - 1));
			}
		}

		// Parse the color-specific options
		m_radius = 1.0;
		m_thickness = 1.0;
		while(args.next_is_flag())
		{
			if(args.if_pop("-radius"))
				m_radius = args.pop_double();
			else if(args.if_pop("-thickness"))
				m_thickness = args.pop_double();
			else
				throw Ex("unrecognized flag ", args.pop_string());
		}
	}

	string attrName()
	{
		string s;
		s = "col ";
		s += to_str(m_attrY);
		return s;
	}

	static double attrVal(GMatrix* pData, size_t i, size_t attr)
	{
		if(attr < pData->cols())
			return pData->row(i)[attr];
		else
			return (double)i;
	}

	void plot(GSVG& svg, GMatrix* pData, double xmin, double xmax, size_t width)
	{
		svg.add_raw("<g><!-- ");
		svg.add_raw(attrName().c_str());
		svg.add_raw(" -->\n");
		double x, y;
		double xPrev = UNKNOWN_REAL_VALUE;
		double yPrev = UNKNOWN_REAL_VALUE;
		double colorMin = 0.0;
		double colorRange = 1.0;
		if(m_type == Row)
			colorRange = pData->rows() * 1.15;
		if(m_type == Attr)
		{
			colorMin = pData->columnMin(m_color);
			colorRange = pData->columnMax(m_color) - colorMin;
			colorRange *= 1.15;
		}
		if(m_pFunc)
		{
			double dx = (xmax - xmin) / width;
			x = xmin;
			vector<double> params;
			params.push_back(x);
			y = m_pFunc->call(params, *m_pFP);
			while(x <= xmax)
			{
				xPrev = x;
				yPrev = y;
				x += dx;
				params[0] = x;
				y = m_pFunc->call(params, *m_pFP);
				if(y > -1e100 && y < 1e100 && yPrev > -1e100 && yPrev < 1e100)
					svg.line(xPrev, yPrev, x, y, m_thickness, (unsigned int)m_color);
			}
		}
		else
		{
			for(size_t i = 0; i < pData->rows(); i++)
			{
				x = attrVal(pData, i, m_attrX);
				y = attrVal(pData, i, m_attrY);
				size_t col = 0;
				if(m_type == Fixed)
					col = m_color;
				else if(m_type == Row)
					col = gAHSV(0xff, i / (float)colorRange, 1.0f, 0.5f);
				else if(m_type == Attr)
					col = gAHSV(0xff, (float)(pData->row(i)[m_color] - colorMin) / (float)colorRange, 1.0f, 0.5f);
				else
					throw Ex("Unrecognized type");
				if(x != UNKNOWN_REAL_VALUE && y != UNKNOWN_REAL_VALUE)
				{
					if(m_thickness > 0.0 && xPrev != UNKNOWN_REAL_VALUE && yPrev != UNKNOWN_REAL_VALUE)
						svg.line(xPrev, yPrev, x, y, m_thickness, (unsigned int)col);
					if(m_radius > 0.0)
						svg.dot(x, y, m_radius, (unsigned int)col);
				}
				xPrev = x;
				yPrev = y;
			}
			xPrev = x;
			yPrev = y;
		}
		svg.add_raw("</g>\n\n");
	}
};

void determineRange(GMatrix* pData, vector<ScatterCol>& cols, bool logx, bool forcePad, double pad, bool x, double& axisMin, double& axisMax)
{
	// Determine the range
	if(logx)
	{
		if(axisMin == UNKNOWN_REAL_VALUE)
		{
			axisMin = 1e300;
			axisMax = -1e300;
			for(size_t i = 0; i < cols.size(); i++)
			{
				size_t attr = x ? cols[i].m_attrX : cols[i].m_attrY;
				if(attr < pData->cols()) // if attr is an attribute
				{
					axisMin = std::min(axisMin, pData->columnMin(attr));
					axisMax = std::max(axisMax, pData->columnMax(attr));
				}
				else // attr is the row index
				{
					axisMin = 1.0;
					axisMax = std::max(axisMax, (double)(pData->rows() - 1));
				}
			}
		}
		axisMin = std::max(1e-12, axisMin);
		axisMax = std::max(axisMin * 15, axisMax);
		double d = pow(axisMax / axisMin, pad);
		axisMax *= d;
		axisMin /= d;
		axisMin = log(axisMin);
		axisMax = log(axisMax);
	}
	else
	{
		if(axisMin == UNKNOWN_REAL_VALUE)
		{
			axisMin = 1e300;
			axisMax = -1e300;
			for(size_t i = 0; i < cols.size(); i++)
			{
				size_t attr = x ? cols[i].m_attrX : cols[i].m_attrY;
				if(attr < pData->cols()) // if attr is an attribute
				{
					axisMin = std::min(axisMin, pData->columnMin(attr));
					axisMax = std::max(axisMax, pData->columnMax(attr));
				}
				else // attr is the row index
				{
					axisMin = std::min(axisMin, 0.0);
					axisMax = std::max(axisMax, (double)(pData->rows() - 1));
				}
			}
			if(axisMin < -1e200)
				axisMin = 0.0;
			if(axisMax <= axisMin)
			{
				axisMax = axisMin + 1e-9;
				axisMin -= 1e-9;
			}
			double d = pad * (axisMax - axisMin);
			axisMin -= d;
			axisMax += d;
		}
		else if(forcePad)
		{
			double d = pad * (axisMax - axisMin);
			axisMin -= d;
			axisMax += d;
		}
		if(axisMin < -1e200)
			axisMin = 0.0;
		if(axisMax <= axisMin)
		{
			axisMax = axisMin + 1e-9;
			axisMin -= 1e-9;
		}
	}
}

void autolabel(GMatrix* pData, vector<ScatterCol>& cols, bool horiz, double axisMin, double axisMax, GSVG& svg, bool serifs)
{
	// Count the unique attributes
	size_t count = 1;
	for(size_t i = 1; i < cols.size(); i++)
	{
		if((horiz && cols[i].m_attrX != cols[i - 1].m_attrX) || (!horiz && cols[i].m_attrY != cols[i - 1].m_attrY))
			count++;
	}

	// Draw the labels
	size_t pos = 0;
	for(size_t i = 0; i < cols.size(); i++)
	{
		if(i > 0)
		{
			if((horiz && cols[i].m_attrX == cols[i - 1].m_attrX) || (!horiz && cols[i].m_attrY == cols[i - 1].m_attrY))
				continue;
		}

		// Determine the label
		string sLabel;
		size_t attr = horiz ? cols[i].m_attrX : cols[i].m_attrY;
		if(attr < pData->relation().size())
		{
			if(pData->relation().type() == GRelation::ARFF)
				sLabel = ((GArffRelation*)&pData->relation())->attrName(attr);
			else
			{
				sLabel = "Attr ";
				sLabel += to_str(attr);
			}
		}
		else
			sLabel = "index";

		// Determine the color
		unsigned int c = 0xff000000;
		if(cols[i].m_type == ScatterCol::Fixed)
		{
			if(i + 1 == cols.size() || (horiz && cols[i].m_attrX != cols[i + 1].m_attrX) || (!horiz && cols[i].m_attrY != cols[i + 1].m_attrY))
				c = (unsigned int)cols[i].m_color;
		}

		// Draw the label
		double plotPos = (axisMax - axisMin) * (double)(pos + 1) / (double)(count + 1) + axisMin;
		if(horiz)
			svg.text(plotPos, svg.horizLabelPos(), sLabel.c_str(), 1.5, GSVG::Middle, c, 0, serifs);
		else
			svg.text(svg.vertLabelPos(), plotPos, sLabel.c_str(), 1.5, GSVG::Middle, c, 90, serifs);
		pos++;
	}
}

void findGridPattern(GMatrix* pData, size_t attr, size_t& block, size_t& cycle)
{
	// Count reps
	for(block = 1; block < pData->rows(); block++)
	{
		if(pData->row(block)[attr] != pData->row(0)[attr])
			break;
	}

	// Find first repeat
	for(cycle = block; cycle < pData->rows(); cycle++)
	{
		if(pData->row(cycle)[attr] == pData->row(0)[attr])
			break;
	}

	// Test pattern
	if((pData->rows() % cycle) != 0)
		throw Ex("The values in attr ", to_str(attr), " do not follow a pattern amenable to plotting across a grid of charts");
	for(size_t i = 0; i < cycle; i += block)
	{
		if(i > 0 && pData->row(i)[attr] <= pData->row(i - 1)[attr])
			throw Ex("The values in attr ", to_str(attr), " do not follow a pattern amenable to plotting across a grid of charts");
		for(size_t j = 0; j < block; j++)
		{
			for(size_t k = 0; i + j + k < pData->rows(); k += cycle)
			{
				if(pData->row(i + j + k)[attr] != pData->row(i)[attr])
					throw Ex("The values in attr ", to_str(attr), " do not follow a pattern amenable to plotting across a grid of charts");
			}
		}
	}
}

void makeGridDataSubset(GMatrix* pSource, GMatrix* pDest, size_t horizPos, size_t horizAttr, size_t horizBlock, size_t vertPos, size_t vertAttr, size_t vertBlock)
{
	if(horizAttr == INVALID_INDEX)
	{
		double vertValue = pSource->row(vertBlock * vertPos)[vertAttr];
		for(size_t i = 0; i < pSource->rows(); i++)
		{
			if(pSource->row(i)[vertAttr] == vertValue)
				pDest->takeRow(&pSource->row(i));
		}
	}
	else if(vertAttr == INVALID_INDEX)
	{
		double horizValue = pSource->row(horizBlock * horizPos)[horizAttr];
		for(size_t i = 0; i < pSource->rows(); i++)
		{
			if(pSource->row(i)[horizAttr] == horizValue)
				pDest->takeRow(&pSource->row(i));
		}
	}
	else
	{
		double horizValue = pSource->row(horizBlock * horizPos)[horizAttr];
		double vertValue = pSource->row(vertBlock * vertPos)[vertAttr];
		for(size_t i = 0; i < pSource->rows(); i++)
		{
			if(pSource->row(i)[horizAttr] == horizValue && pSource->row(i)[vertAttr] == vertValue)
				pDest->takeRow(&pSource->row(i));
		}
	}
}

void PlotScatter(GArgReader& args)
{
	// Load the data
	GMatrix* pData = loadData(args.pop_string());
	std::unique_ptr<GMatrix> hData(pData);

	// Values pertaining to grids of charts
	size_t horizCharts = 1;
	size_t horizAttr = INVALID_INDEX;
	size_t horizBlock, horizCycle;
	size_t vertCharts = 1;
	size_t vertAttr = INVALID_INDEX;
	size_t vertBlock, vertCycle;

	// Values pertaining to each chart
	size_t width = 960;
	size_t height = 540;
	size_t margin = INVALID_INDEX;
	size_t maxHorizMarks = 30;
	size_t maxVertMarks = size_t(-1);
	double pad = 0.05;
	double xmin = UNKNOWN_REAL_VALUE;
	double ymin = UNKNOWN_REAL_VALUE;
	double xmax = UNKNOWN_REAL_VALUE;
	double ymax = UNKNOWN_REAL_VALUE;
	bool logx = false;
	bool logy = false;
	bool forcePad = false;
	bool horizMarks = true;
	bool vertMarks = true;
	bool serifs = true;
	bool aspect = false;
	string horizLabel;
	string vertLabel;
	GRand prng(0);
	while(args.next_is_flag())
	{
		if(args.if_pop("-size"))
		{
			width = args.pop_uint();
			height = args.pop_uint();
		}
		else if(args.if_pop("-horizattr"))
		{
			horizAttr = args.pop_uint();
			findGridPattern(pData, horizAttr, horizBlock, horizCycle);
			horizCharts = horizCycle / horizBlock;
		}
		else if(args.if_pop("-vertattr"))
		{
			vertAttr = args.pop_uint();
			findGridPattern(pData, vertAttr, vertBlock, vertCycle);
			vertCharts = vertCycle / vertBlock;
		}
		else if(args.if_pop("-margin"))
			margin = args.pop_uint();
		else if(args.if_pop("-horizmarks"))
			maxHorizMarks = args.pop_uint();
		else if(args.if_pop("-vertmarks"))
			maxVertMarks = args.pop_uint();
		else if(args.if_pop("-pad"))
		{
			pad = args.pop_double();
			forcePad = true;
		}
		else if(args.if_pop("-range"))
		{
			xmin = args.pop_double();
			ymin = args.pop_double();
			xmax = args.pop_double();
			ymax = args.pop_double();
		}
		else if(args.if_pop("-logx"))
			logx = true;
		else if(args.if_pop("-logy"))
			logy = true;
		else if(args.if_pop("-nohmarks"))
			horizMarks = false;
		else if(args.if_pop("-novmarks"))
			vertMarks = false;
		else if(args.if_pop("-nogrid"))
		{
			horizMarks = false;
			vertMarks = false;
		}
		else if(args.if_pop("-noserifs"))
			serifs = false;
		else if(args.if_pop("-hlabel"))
			horizLabel = args.pop_string();
		else if(args.if_pop("-vlabel"))
			vertLabel = args.pop_string();
		else if(args.if_pop("-aspect"))
			aspect = true;
		else
			throw Ex("unrecognized flag ", args.pop_string());
	}

	if(margin == INVALID_INDEX)
		margin = (size_t)(std::min(width, height) * 0.2);

	// Parse the colors
	vector<ScatterCol> cols;
	while(args.size() > 0)
	{
		size_t n = cols.size();
		cols.resize(cols.size() + 1);
		cols[n].parse(args, pData->cols());
	}

	// Draw the grid
	determineRange(pData, cols, logx, forcePad, pad, true, xmin, xmax);
	determineRange(pData, cols, logy, forcePad, pad, false, ymin, ymax);
	if(aspect)
	{
		if(logx || logy)
			throw Ex("the \"-aspect\" flag is not compatible with logarithmic scales");
		if((xmax - xmin) / width < (ymax - ymin) / height)
		{
			double dif = 0.5 * ((ymax - ymin) * width / height - (xmax - xmin));
			xmin -= dif;
			xmax += dif;
		}
		else
		{
			double dif = 0.5 * ((xmax - xmin) * height / width - (ymax - ymin));
			ymin -= dif;
			ymax += dif;
		}
	}
	GSVG svg(width, height, horizCharts, vertCharts);
	for(size_t vert = 0; vert < vertCharts; vert++)
	{
		for(size_t horiz = 0; horiz < horizCharts; horiz++)
		{
			svg.newChart(xmin, ymin, xmax, ymax, horiz, vert, (double)margin);
			if(horizMarks)
				svg.horizMarks((int)maxHorizMarks);
			if(vertMarks)
			{
				if(maxVertMarks == INVALID_INDEX)
					maxVertMarks = maxHorizMarks * height / width;
				svg.vertMarks((int)maxVertMarks);
			}

			// Draw the axis labels
			if(horizLabel.length() > 0)
				svg.text(0.5 * (xmin + xmax), svg.horizLabelPos(), horizLabel.c_str(), 1.5, GSVG::Middle, 0xff000000, 0, serifs);
			else
				autolabel(pData, cols, true, xmin, xmax, svg, serifs);
			if(vertLabel.length() > 0)
				svg.text(svg.vertLabelPos(), 0.5 * (ymin + ymax), vertLabel.c_str(), 1.5, GSVG::Middle, 0xff000000, 90, serifs);
			else
				autolabel(pData, cols, false, ymin, ymax, svg, serifs);

			// Draw the colors
			svg.clip();
			if(horizCharts > 1 || vertCharts > 1)
			{
				GMatrix temp(pData->relation().clone());
				GReleaseDataHolder hTemp(&temp);
				makeGridDataSubset(pData, &temp, horiz, horizAttr, horizBlock, vert, vertAttr, vertBlock);
				for(size_t i = 0; i < cols.size(); i++)
					cols[i].plot(svg, &temp, xmin, xmax, width);
			}
			else
			{
				for(size_t i = 0; i < cols.size(); i++)
					cols[i].plot(svg, pData, xmin, xmax, width);
			}
		}
	}

	// output the plot
	svg.print(cout);
}

void semanticMap(GArgReader& args){
  string somFile = args.pop_string();
  string dataFile = args.pop_string();

  // Load the data
  std::unique_ptr<GMatrix> hData(loadData(dataFile.c_str()));
  if(hData->rows() < 1){
    throw Ex("The dataset is empty.  Cannot make a semantic map from "
	       "an empty dataset.");
  }
  if(hData->cols() < 1){
    throw Ex("The dataset has no attributes.  Cannot make a semantic map "
	       "without attributes.");
  }
  // Load the self organizing map
  GDom doc;
  doc.loadJson(somFile.c_str());
  GSelfOrganizingMap som(doc.root());
  // Parse the options
  string outFilename="semantic_map.svg";
  unsigned labelCol = (unsigned int)hData->cols()-1;
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
      throw Ex("Invalid option: ", args.peek());
    }
  }
  if(labelCol >= hData->cols()){
    throw Ex("Label column index is too large");
  }
  if(som.outputDimensions() > 2){
    throw Ex("Semantic map can only plot one or two dimensional "
	       "self-organizing maps.");
  }
  if(som.inputDimensions() > hData->cols()){
    throw Ex("The input dataset does not have enough attributes for input "
	       "to the semantic map");
  }

  //Write the svg output file
  std::ofstream out(outFilename.c_str());
  if(!out){
    throw Ex("Could not open the file named \"",outFilename,"\"");
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
    std::vector<std::list<const GVec*> > winLists(som.nodes().size());
    for(size_t row = 0; row < hData->rows(); ++row){
      const GVec& rowVec = hData->row(row);
      size_t bestNode = som.bestMatch(rowVec);
      winLists.at(bestNode).push_back(&rowVec);
    }
    //Calculate the variance of the labelCol column for each node
    std::vector<std::list<const GVec*> >::const_iterator list;
    for(list=winLists.begin(); list != winLists.end(); ++list){
      if(list->size() == 0){
	//No elements in the list, no variance
	labels.push_back(0);
      }else{
	//Copy the appropriate column into a 1-column matrix
	GMatrix m(0,1);
	std::list<const GVec*>::const_iterator l;
	for(l = list->begin(); l != list->end(); ++l){
	  m.newRow();
	  double& val = (m.row(m.rows()-1))[0];
	  val = (*(*l))[labelCol];
	}
	//Use the matrix to calculate the variance
	labels.push_back(m.columnVariance(0, m.columnMean(0)));
      }
    }
  }else{
    //Find the best data indices using only the first inputDimensions of
    //the input data
    GMatrix* pLessColumns = new GMatrix();
	pLessColumns->copy(hData.get());
    std::unique_ptr<GMatrix> lessColumns(pLessColumns);
    while(lessColumns->cols() > som.inputDimensions()){
      lessColumns->deleteColumns(lessColumns->cols()-1, 1);
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
	std::unique_ptr<GMatrix> hData(pData);

	// Parse options
	size_t wid = 960;
	size_t hgt = 540;
	size_t attr = 0;
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
		else if(args.if_pop("-range"))
		{
			xmin = args.pop_double();
			xmax = args.pop_double();
			ymax = args.pop_double();
		}
		else
			throw Ex("Invalid option: ", args.peek());
	}
	if(attr >= pData->relation().size())
		throw Ex("attr out of range");

	// Drop any rows with missing values in column attr
	for(size_t i = pData->rows() - 1; i < pData->rows(); i--)
	{
		if(pData->row(i)[attr] == UNKNOWN_REAL_VALUE)
			pData->deleteRow(i);
	}

	// Make the histogram
	if(pData->relation().valueCount(attr) == 0)
	{
		bool use_density_estimation = false;
		if(pData->rows() < 10000)
			use_density_estimation = true;
		
		if(use_density_estimation)
		{
			size_t k = std::max((size_t)3, (size_t)sqrt(pData->rows()));

			// Estimate the inverse density at each point
			pData->sort(attr);
			double laplace = 2.0 * (pData->row(pData->rows() - 1)[attr] - pData->row(0)[attr]) / pData->rows();
			GMatrix invDensity(0, 2);
			GVec& pBlock = invDensity.newRow();
			pBlock[0] = pData->row(0)[attr];
			pBlock[1] = 0.0;
			for(size_t i = k - 1; i < pData->rows(); i++)
			{
				double xBegin = pData->row(i + 1 - k)[attr];
				double xEnd = pData->row(i)[attr];
				GVec& pBlock2 = invDensity.newRow();
				pBlock2[0] = 0.5 * (xBegin + xEnd);
				pBlock2[1] = (double)k / ((xEnd - xBegin + laplace) * pData->rows());
			}
			GVec& pBlock3 = invDensity.newRow();
			pBlock3[0] = pData->row(pData->rows() - 1)[attr];
			pBlock3[1] = 0.0;

			// Plot it
			double maxHeight = invDensity.columnMax(1);
			GSVG svg(wid, hgt);
			svg.newChart(invDensity[0][0], 0.0, invDensity[invDensity.rows() - 1][0], maxHeight);
			svg.add_raw("<path d=\"m "); // Start a path
			svg.add_raw(to_str(invDensity[0][0]).c_str());
			svg.add_raw(",0 c "); // Turn on control points
			for(size_t i = 1; i < invDensity.rows(); i++)
			{
				// Add the leaving direction control point
				svg.add_raw(to_str(0.5 * (invDensity[i][0] - invDensity[i - 1][0])).c_str()); // half way to the destination
				svg.add_raw(",0 "); // horizontal

				// Add the arriving direction control point
				svg.add_raw(to_str(0.5 * (invDensity[i][0] - invDensity[i - 1][0])).c_str()); // half way to the destination
				svg.add_raw(",");
				svg.add_raw(to_str(invDensity[i][1] - invDensity[i - 1][1]).c_str()); // horizontal
				svg.add_raw(" ");

				// Add the destination point
				svg.add_raw(to_str(invDensity[i][0] - invDensity[i - 1][0]).c_str());
				svg.add_raw(",");
				svg.add_raw(to_str(invDensity[i][1] - invDensity[i - 1][1]).c_str());
				svg.add_raw(" ");
			}
			svg.add_raw("z\" style=\"fill:#0000ff;fill-opacity:0.2\" />\n");

			// Draw the grid
			svg.horizMarks(30);
			svg.vertMarks(20);

			// Print it
			svg.print(cout);
		}
		else
		{
			GHistogram hist(*pData, attr, xmin, xmax, wid);
			double height = (ymax == UNKNOWN_REAL_VALUE ? hist.binLikelihood(hist.modeBin()) * 1.5 : ymax);
			GSVG svg(wid, hgt);
			svg.newChart(hist.xmin(), 0.0, hist.xmax(), height);
			for(double x = hist.xmin(); x <= hist.xmax(); x += svg.hunit())
			{
				size_t bin = hist.xToBin(x);
				double likelihood = hist.binLikelihood(bin);
				if(likelihood > 0.0)
					svg.rect(x, 0.0, svg.hunit(), likelihood, 0xff000080);
			}

			// Draw the grid
			svg.horizMarks(30);
			svg.vertMarks(20);

			// Print it
			svg.print(cout);
		}
	}
	else
	{
		size_t buckets = pData->relation().valueCount(attr);
		GTEMPBUF(double, hist, buckets);
		GVec::setAll(hist, 0.0, buckets);
		for(size_t i = 0; i < pData->rows(); i++)
		{
			int b = (int)pData->row(i)[attr];
			if(b >= 0 && (size_t)b < buckets)
				hist[b]++;
		}

		// Plot it
		GSVG svg(wid, hgt);
		svg.newChart(0.0, 0.0, (double)buckets, 1.0);
		for(size_t i = 0; i < buckets; i++)
			svg.rect((double)i, 0, 1, hist[i] / pData->rows(), (((i & 1) == 0) ? 0xff400000 : 0xff008040));

		// Draw the grid
		svg.horizMarks(30);
		svg.vertMarks(20);

		// Print it
		svg.print(cout);
	}
}

void PrintStats(GArgReader& args)
{
	// Load
	const char* szFilename = args.pop_string();
	GMatrix* pData = loadData(szFilename);
	std::unique_ptr<GMatrix> hData(pData);
	GArffRelation* pRel = (GArffRelation*)&pData->relation();

	// Options
	bool printAll = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-all"))
			printAll = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

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
	if(printAll)
		stepSize = 1;

	// Print the arity
	GMatrix arity(pRel->size(), 1);
	size_t maxArity = 0;
	size_t sumArity = 0;
	for(size_t i = 0; i < pRel->size(); i++)
	{
		size_t vals = pRel->valueCount(i);
		size_t a = vals < 3 ? 1 : vals;
		arity[i][0] = (double)a;
		maxArity = std::max(maxArity, a);
		sumArity += a;
	}
	cout << "Median arity=" << arity.columnMedian(0) << ", Max arity=" << maxArity << ", Sum arity=" << sumArity << "\n";

	// Print stats about each attribute
	for(size_t i = 0; i < pRel->size();)
	{
		cout << "  " << i << ") " << pRel->attrName(i) << ", ";
		size_t vals = pRel->valueCount(i);
		if(vals == 0) // continuous
		{
			cout << "Type: Continuous, ";
			try
			{
				double d1, d2, d3;
				d1 = pData->columnMean(i);
				d2 = pData->columnVariance(i, d1);
				d3 = pData->columnMedian(i);
				cout << "Mean:" << d1 << ", Dev:" << sqrt(d2) << ", Median:" << d3 << ", ";
				cout << "Min:" << pData->columnMin(i) << ", Max:" << pData->columnMax(i) << ", ";
			}
			catch(...)
			{
				// If it gets to here, all values are probably missing
			}
			cout << "Missing:" << pData->countValue(i, UNKNOWN_REAL_VALUE) << "\n";
		}
		else if(vals < (size_t)-10)
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
		else if(vals == (size_t)-1) // string;
		{
			cout << "Type: String\n";
		}
		else if(vals == (size_t)-2) // date;
		{
			cout << "Type: Date\n";
		}
		else
			throw Ex("Unexpected number of values");
		size_t prevI = i;
		if(i < 2)
			i++;
		else if(i + stepSize >= pRel->size() - 3)
			i += std::max(1, (int)pRel->size() - 3 - (int)i);
		else
			i += stepSize;
		if(i - prevI > 1)
			cout << "	...\n";
	}
}

void calcError(GArgReader& args){
	GMatrix loader;
	loader.loadArff(args.pop_string());
	
	GMatrix output(0, 1);
	
	int SSE = 0;
	int MAPE = 1;
	int RMSE = 2;
	
	int metric = SSE;
	
	if(args.if_pop("-m")){
		if(args.if_pop("SSE"))
			metric = SSE;
		else if(args.if_pop("MAPE"))
			metric = MAPE;
		else if(args.if_pop("RMSE"))
			metric = RMSE;
		else
			throw Ex("Invalid metric.");
	}
	
	while(args.size() > 0){
		GVec& row = output.newRow();
		row[0] = 0.0;
		
		size_t col1 = args.pop_uint();
		size_t col2 = args.pop_uint();
		
		if(col1 >= loader.cols() || col2 >= loader.cols())
			throw Ex("Invalid column.");
		
		size_t dropped = 0;
		
		for(size_t i = 0; i < loader.rows(); i++){
			if(loader[i][col1] == UNKNOWN_REAL_VALUE){
				dropped++;
				continue;
			}
			
			if(metric == SSE || metric == RMSE){
				row[0] += (loader[i][col1] - loader[i][col2]) * (loader[i][col1] - loader[i][col2]);
			}
			else if(metric == MAPE){
				row[0] += fabs((loader[i][col1] - loader[i][col2]) / loader[i][col1]);
			}
		}
		
		if(dropped == loader.rows())
			throw Ex("Invalid data set!");
		
		if(metric == MAPE || metric == RMSE)
			row[0] /= (loader.rows() - dropped);
		
		if(metric == RMSE)
		{
			row[0] = sqrt(row[0]);
		}
	}
	
	output.print(std::cout);
}

void percentSame(GArgReader& args){
  std::unique_ptr<GMatrix> hData1(loadData(args.pop_string()));
  std::unique_ptr<GMatrix> hData2(loadData(args.pop_string()));
  const size_t cols = hData1->cols();
  const size_t rows = hData1->rows();
  if(hData1->cols() != hData2->cols()){
    throw Ex("The two files have different numbers of attributes.  Cannot "
	       "compare entries when the number of columns is different");
  }
  if(hData1->rows() != hData2->rows()){
    throw Ex("The two files have different numbers of tuples.  Cannot "
	       "compare entries when the number of rows is different");
  }
  if(rows == 0){
    throw Ex("The files have no rows.  Cannot calculate the percentage of "
	       "identical values for empty files.");
  }
  for(size_t i = 0; i < cols; ++i){
    if(hData1->relation().valueCount(i) !=
       hData2->relation().valueCount(i)){
      size_t v1 = hData1->relation().valueCount(i);
      size_t v2 = hData2->relation().valueCount(i);
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
      throw Ex(msg.str());
    }
  }
  //Count the same values
  vector<size_t> numSame(cols, 0);
  for(size_t row = 0; row < rows; ++row){
    const GVec& r1 = hData1->row(row);
    const GVec& r2 = hData2->row(row);
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
		throw Ex("Model not specified.");
	doc.loadJson(args.pop_string());
	GLearnerLoader ll(true);
	GSupervisedLearner* pModeler = ll.loadLearner(doc.root());
	std::unique_ptr<GSupervisedLearner> hModeler(pModeler);
	while(pModeler->isFilter())
		pModeler = ((GFilter*)pModeler)->innerLearner();

	if(args.size() > 0)
	{
		GMatrix data;
		size_t ld;
		loadDataWithSwitches(data, args, &ld);
		size_t labelDims = pModeler->relLabels().size();
		if(ld != labelDims)
			throw Ex("Different number of label dims than the model was trained with");
		GArffRelation relFeatures;
		relFeatures.addAttrs(data.relation(), 0, data.cols() - labelDims);
		GArffRelation relLabels;
		relLabels.addAttrs(data.relation(), data.cols() - labelDims, labelDims);
		((GDecisionTree*)pModeler)->print(cout, &relFeatures, &relLabels);
	}
	else
		((GDecisionTree*)pModeler)->print(cout);
}


void printRandomForest(GArgReader& args)
{
	// Load the model
	GDom doc;
	if(args.size() < 1)
		throw Ex("Model not specified.");
	doc.loadJson(args.pop_string());
	GLearnerLoader ll(true);
	GSupervisedLearner* pModeler = ll.loadLearner(doc.root());
	std::unique_ptr<GSupervisedLearner> hModeler(pModeler);
	while(pModeler->isFilter())
		pModeler = ((GFilter*)pModeler)->innerLearner();

	if(args.size() > 0)
	{
		size_t ld;
		GMatrix data;
		loadDataWithSwitches(data, args, &ld);
		size_t labelDims = pModeler->relLabels().size();
		if(ld != labelDims)
			throw Ex("Different number of label dims than the model was trained with");
		GArffRelation relFeatures;
		relFeatures.addAttrs(data.relation(), 0, data.cols() - labelDims);
		GArffRelation relLabels;
		relLabels.addAttrs(data.relation(), data.cols() - labelDims, labelDims);
		((GRandomForest*)pModeler)->print(cout, &relFeatures, &relLabels);
	}
	else
		((GRandomForest*)pModeler)->print(cout);
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makePlotUsageTree();
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
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
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
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
	PathData pd;
	GFile::parsePath(argv[0], &pd);
	const char* appName = argv[0] + pd.fileStart;
	GArgReader args(argc, argv);
	args.pop_string(); // advance past the app name
	int ret = 0;
	try
	{
		if(args.size() < 1) throw Ex("Expected a command");
		else if(args.if_pop("usage")) ShowUsage(appName);
		else if(args.if_pop("bar")) PlotBar(args);
		else if(args.if_pop("equation")) PlotEquation(args);
		else if(args.if_pop("histogram")) makeHistogram(args);
		else if(args.if_pop("percentsame")) percentSame(args);
		else if(args.if_pop("printdecisiontree")) printDecisionTree(args);
		else if(args.if_pop("printrandomforest")) printRandomForest(args);
		else if(args.if_pop("scatter")) PlotScatter(args);
		else if(args.if_pop("semanticmap")) semanticMap(args);
		else if(args.if_pop("stats")) PrintStats(args);
		else if(args.if_pop("calcerror")) calcError(args);
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

