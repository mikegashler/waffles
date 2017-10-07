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
#include "GApp.h"
#include "GCluster.h"
#include "GMatrix.h"
#include "GDistance.h"
#include "GFile.h"
#include "GError.h"
#include "GHolders.h"
#include "GNeuralNet.h"
#include "GRand.h"
#include "GRecommender.h"
#include "GSparseMatrix.h"
#include "GDom.h"
#include "usage.h"
#include <time.h>
#include <iostream>
#ifdef WINDOWS
#	include <direct.h>
#	include <process.h>
#endif
#include <exception>
#include <string>
#include <vector>
#include <set>

using std::cout;
using std::cerr;
using std::string;
using std::vector;
using std::set;

namespace GClasses{

class GRecommenderLib
{
public:
	static size_t getAttrVal(const char* szString, size_t attrCount);
	
	static void parseAttributeList(vector<size_t>& list, GArgReader& args, size_t attrCount);
	
	static void loadData(GMatrix& data, const char* szFilename);
	
	static GSparseMatrix* loadSparseData(const char* szFilename);
	
	static GBaselineRecommender* InstantiateBaselineRecommender(GArgReader& args);
	
	static GBagOfRecommenders* InstantiateBagOfRecommenders(GArgReader& args);
	
	static GInstanceRecommender* InstantiateInstanceRecommender(GArgReader& args);
	
	static GDenseClusterRecommender* InstantiateDenseClusterRecommender(GArgReader& args);
	
	static GSparseClusterRecommender* InstantiateSparseClusterRecommender(GArgReader& args);
	
	static GMatrixFactorization* InstantiateMatrixFactorization(GArgReader& args);
	
//	static GNonlinearPCA* InstantiateNonlinearPCA(GArgReader& args);
	
//	static GHybridNonlinearPCA* InstantiateHybridNonlinearPCA(GArgReader& args);
	
	static GContentBasedFilter* InstantiateContentBasedFilter(GArgReader& args);
	
	static GContentBoostedCF* InstantiateContentBoostedCF(GArgReader& args);
	
	static void showInstantiateAlgorithmError(const char* szMessage, GArgReader& args);
	
	static GCollaborativeFilter* InstantiateAlgorithm(GArgReader& args);
	
	static void crossValidate(GArgReader& args);
	
	static void precisionRecall(GArgReader& args);
	
	static void ROC(GArgReader& args);
	
	static void transacc(GArgReader& args);
	
	static void fillMissingValues(GArgReader& args);
	
	static void ShowUsage(const char* appName);
	
	static void showError(GArgReader& args, const char* szAppName, const char* szMessage);
};

} // namespace GClasses
