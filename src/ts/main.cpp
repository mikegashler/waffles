/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Luke B. Godfrey,
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

#include <iostream>
#include <vector>
#include "../GClasses/GApp.h"
#include "../GClasses/GDom.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GNeuralDecomposition.h"
#include "../GClasses/usage.h"
#include <memory>

using namespace GClasses;
using std::cout;
using std::cerr;
using std::string;
using std::vector;

void LoadData(GArgReader &args, std::unique_ptr<GMatrix> &hOutput)
{
	// Load the dataset by extension
	if(args.size() < 1)
		throw Ex("Expected the filename of a datset. (Found end of arguments.)");
	const char* szFilename = args.pop_string();
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	GMatrix data;
	vector<size_t> abortedCols;
	vector<size_t> ambiguousCols;
	const char *input_type;
	if (args.next_is_flag() && args.if_pop("-input_type")) {
		input_type = args.pop_string();
	} else { /* deduce it from extension (if any) */
		input_type = szFilename + pd.extStart;
		if (*input_type != '.') /* no extension - assume ARFF */
			input_type = "arff";
		else
			input_type++;
	}
	
	// Now load the data
	if(_stricmp(input_type, "arff") == 0)
	{
		data.loadArff(szFilename);
	}
	else if(_stricmp(input_type, "csv") == 0)
	{
		GCSVParser parser;
		parser.parse(data, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < data.cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else if(_stricmp(input_type, "dat") == 0)
	{
		GCSVParser parser;
		parser.setSeparator('\0');
		parser.parse(data, szFilename);
		cerr << "\nParsing Report:\n";
		for(size_t i = 0; i < data.cols(); i++)
			cerr << to_str(i) << ") " << parser.report(i) << "\n";
	}
	else
	{
		throw Ex("Unsupported file format: ", szFilename + pd.extStart);
	}
	
	// Split data into a feature matrix and a label matrix
	GMatrix* pFeatures = data.cloneSub(0, 0, data.rows(), data.cols());
	hOutput.reset(pFeatures);
}

void ShowUsage(const char *appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode *pUsageTree = makeTimeSeriesUsageTree();
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(cout, 0, 3, 76, 1000, true);
	cout.flush();
}

void Train(GArgReader &args)
{
	// Load series from file
	std::unique_ptr<GMatrix> hSeries, hFeatures;
	LoadData(args, hSeries);
	GMatrix *pSeries = hSeries.get();
	
	// Split features/labels
	if(pSeries->cols() == 2)
	{
		GMatrix *pFeatures = pSeries->cloneSub(0, 0, pSeries->rows(), 1);
		GMatrix *pLabels = pSeries->cloneSub(0, 1, pSeries->rows(), 1);
		hFeatures.reset(pFeatures);
		hSeries.reset(pLabels);
		pSeries = pLabels;
	}
	else if(pSeries->cols() > 2)
	{
		throw Ex("Too many columns!");
	}
	
	// Parse options
	GNeuralDecomposition *nd = new GNeuralDecomposition();
	nd->nn().rand().setSeed(time(NULL));
	
	while(args.next_is_flag())
	{
		if(args.if_pop("-regularization"))
			nd->setRegularization(args.pop_double());
		else if(args.if_pop("-learningRate"))
			nd->setLearningRate(args.pop_double());
		else if(args.if_pop("-linearUnits"))
			nd->setLinearUnits(args.pop_uint());
		else if(args.if_pop("-softplusUnits"))
			nd->setSoftplusUnits(args.pop_uint());
		else if(args.if_pop("-sigmoidUnits"))
			nd->setSigmoidUnits(args.pop_uint());
		else if(args.if_pop("-epochs"))
			nd->setEpochs(args.pop_uint());
		else if(args.if_pop("-features"))
			LoadData(args, hFeatures);
		else if(args.if_pop("-filterLogarithm"))
			nd->setFilterLogarithm(true);
		else if(args.if_pop("-seed"))
			nd->nn().rand().setSeed(args.pop_uint());
		else
			throw Ex("Invalid option: ", args.peek());
	}
	
	if(hFeatures.get() == NULL)
	{
		// Generate features
		GMatrix *pFeatures = new GMatrix(pSeries->rows(), 1);
		for(size_t i = 0; i < pSeries->rows(); i++)
		{
			pFeatures->row(i)[0] = i / (double) pSeries->rows();
		}
		hFeatures.reset(pFeatures);
	}
	
	// Train
	GMatrix *pFeatures = hFeatures.get();
	nd->train(*pFeatures, *pSeries);
	
	// Output the trained model
	GDom doc;
	doc.setRoot(nd->serialize(&doc));
	doc.writeJson(cout);
}

void Extrapolate(GArgReader &args)
{
	// Load the model
	if(args.size() < 1)
	{
		throw Ex("Model not specified.");
	}
	GDom doc;
	doc.loadJson(args.pop_string());
	GLearnerLoader ll(true);
	GSupervisedLearner *pLearner = ll.loadLearner(doc.root());
	std::unique_ptr<GSupervisedLearner> hLearner(pLearner);
	
	// Parse options
	
	double start = 1.0;
	double length = 1.0;
	double step = 0.0002;
	bool useFeatures = false;
	bool outputFeatures = true;
	
	GNeuralDecomposition *nd = (GNeuralDecomposition *) pLearner;
	std::unique_ptr<GMatrix> hFeatures;
	
	while(args.next_is_flag())
	{
		if(args.if_pop("-start"))
		{
			start = args.pop_double();
		}
		else if(args.if_pop("-length"))
		{
			length = args.pop_double();
		}
		else if(args.if_pop("-step"))
		{
			step = args.pop_double();
		}
		else if(args.if_pop("-features"))
		{
			LoadData(args, hFeatures);
			useFeatures = true;
		}
		else if(args.if_pop("-outputFeatures"))
		{
			outputFeatures = true;
		}
		else
		{
			throw Ex("Invalid option: ", args.peek());
		}
	}
	
	// Extrapolate
	GMatrix *pOutput;
	if(useFeatures)
		pOutput = nd->extrapolate(*hFeatures.get());
	else
		pOutput = nd->extrapolate(start, length, step, outputFeatures);
	std::unique_ptr<GMatrix> hOutput(pOutput);
	
	// Output predictions
	pOutput->print(cout);
}

void ShowError(GArgReader &args, const char *szAppName, const char *szMessage)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	args.set_pos(1);
	const char* szCommand = args.peek();
	UsageNode* pUsageTree = makeTimeSeriesUsageTree();
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << szAppName << " ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, false);
		}
		else
		{
			cerr << "Brief Usage Information:\n\n";
			pUsageTree->print(cerr, 0, 3, 76, 1, false);
		}
	}
	else
	{
		pUsageTree->print(cerr, 0, 3, 76, 1, true);
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
			else if(args.if_pop("train"))
				Train(args);
			else if(args.if_pop("extrapolate"))
				Extrapolate(args);
			else
			{
				nRet = 1;
				string s = args.peek();
				s += " is not a recognized command.";
				ShowError(args, appName, s.c_str());
			}
		}
		else
		{
			nRet = 1;
			ShowError(args, appName, "Brief Usage Information:");
		}
	}
	catch(const std::exception& e)
	{
		nRet = 1;
		if(strcmp(e.what(), "nevermind") != 0)
			ShowError(args, appName, e.what());
	}
	return nRet;
}
