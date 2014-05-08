/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    Michael R. Smith,
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

#include <stdio.h>
#include <stdlib.h>
#include "../GClasses/GRecommenderLib.cpp"
#include <time.h>
#include <iostream>
#ifdef WIN32
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
				GRecommenderLib::ShowUsage(appName);
			else if(args.if_pop("crossvalidate"))
				GRecommenderLib::crossValidate(args);
			else if(args.if_pop("fillmissingvalues"))
				GRecommenderLib::fillMissingValues(args);
			else if(args.if_pop("precisionrecall"))
				GRecommenderLib::precisionRecall(args);
			else if(args.if_pop("roc"))
				GRecommenderLib::ROC(args);
			else if(args.if_pop("transacc"))
				GRecommenderLib::transacc(args);
			else
			{
				nRet = 1;
				string s = args.peek();
				s += " is not a recognized command.";
				GRecommenderLib::showError(args, appName, s.c_str());
			}
		}
		else
		{
			nRet = 1;
			GRecommenderLib::showError(args, appName, "Brief Usage Information:");
		}
	}
	catch(const std::exception& e)
	{
		nRet = 1;
		if(strcmp(e.what(), "nevermind") != 0) // if an error message was not already displayed...
			GRecommenderLib::showError(args, appName, e.what());
	}
	return nRet;
}
