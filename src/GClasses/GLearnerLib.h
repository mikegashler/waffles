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
#include <fstream>
#include "GApp.h"
#include "GMatrix.h"
#include "GCluster.h"
#include "GDecisionTree.h"
#include "GDistance.h"
#include "GDistribution.h"
#include "GEnsemble.h"
#include "GFile.h"
#include "GFunction.h"
#include "GGaussianProcess.h"
#include "GHillClimber.h"
#include "GHolders.h"
#include "GImage.h"
#include "GKernelTrick.h"
#include "GKNN.h"
#include "GLinear.h"
#include "GError.h"
#include "GManifold.h"
#include "GNaiveBayes.h"
#include "GNaiveInstance.h"
#include "GNeuralNet.h"
#include "GOptimizer.h"
#include "GRand.h"
#include "GSparseMatrix.h"
#include "GTime.h"
#include "GTransform.h"
#include "GDom.h"
#include "GVec.h"
#include "usage.h"
#include <cassert>
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
#include <memory>

namespace GClasses{

using std::cout;
using std::cerr;
using std::string;
using std::vector;
using std::set;
using std::ostringstream;

///Provides some useful functions for instantiating learning algorithms from the command line
class GLearnerLib
{
public:
	static GTransducer* InstantiateAlgorithm(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static size_t getAttrVal(const char* szString, size_t attrCount);

	static void parseAttributeList(vector<size_t>& list, GArgReader& args, size_t attrCount);

	static void loadData(GArgReader& args, std::unique_ptr<GMatrix>& hFeaturesOut, std::unique_ptr<GMatrix>& hLabelsOut, bool requireMetadata = false);

	static GAgglomerativeTransducer* InstantiateAgglomerativeTransducer(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBaselineLearner* InstantiateBaseline(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBayesianModelAveraging* InstantiateBMA(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBayesianModelCombination* InstantiateBMC(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBag* InstantiateBag(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GGradBoost* InstantiateGradBoost(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GResamplingAdaBoost* InstantiateBoost(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBucket* InstantiateBucket(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBucket* InstantiateCvdt(GArgReader& args);

	static GDecisionTree* InstantiateDecisionTree(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GGaussianProcess* InstantiateGaussianProcess(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GGraphCutTransducer* InstantiateGraphCutTransducer(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBayesianModelCombination* InstantiateHodgePodge(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GKNN* InstantiateKNN(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GLinearRegressor* InstantiateLinearRegressor(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GMeanMarginsTree* InstantiateMeanMarginsTree(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GNaiveBayes* InstantiateNaiveBayes(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GNaiveInstance* InstantiateNaiveInstance(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GBlock* instantiateBlock(GArgReader& args);

	static GNeuralNetLearner* InstantiateNeuralNet(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GNeighborTransducer* InstantiateNeighborTransducer(GArgReader& args, GMatrix* pFeatures, GMatrix* pLabels);

	static GRandomForest* InstantiateRandomForest(GArgReader& args);

	static void showInstantiateAlgorithmError(const char* szMessage, GArgReader& args);

	static void autoTuneDecisionTree(GMatrix& features, GMatrix& labels);

	static void autoTuneKNN(GMatrix& features, GMatrix& labels);

	static void autoTuneNeuralNet(GMatrix& features, GMatrix& labels);

	static void autoTuneNaiveBayes(GMatrix& features, GMatrix& labels);

	static void autoTuneNaiveInstance(GMatrix& features, GMatrix& labels);

	static void autoTuneGraphCutTransducer(GMatrix& features, GMatrix& labels);

	static void autoTune(GArgReader& args);

	static void Train(GArgReader& args);

	static void predict(GArgReader& args);

	static void predictDistribution(GArgReader& args);

	static void leftJustifiedString(const char* pIn, char* pOut, size_t outLen);

	static void rightJustifiedString(const char* pIn, char* pOut, size_t outLen);

	///\brief Returns the header for the machine readable confusion matrix
	///for variable \a variable_idx as printed by
	///printMachineReadableConfusionMatrices
	///
	///The header is comma-separated values. The first two entries in the
	///header are "Variable Name","Variable Index". The rest of the
	///entries fit the format "Expected:xxx/Got:yyy" where xxx and yyy are
	///two values that the variable can take on.
	///
	///\param variable_idx the index of the variable in the relation
	///
	///\param pRelation a pointer to the relation from which the
	///                 variable_idx-'th variable is taken. Cannot be null
	static std::string machineReadableConfusionHeader(std::size_t variable_idx, const GRelation* pRelation);

	//\brief Returns the data for the machine readable confusion matrix
	///for variable \a variable_idx as printed by
	///printMachineReadableConfusionMatrices
	///
	///The first entry is the name of the variable. The second entry is
	///the value of variable_idx, The entry (r*numCols+c)+2 where r and c are both in 0..nv-1, nv being the number of values that the variable takes on, is the entry at row r and column c of *pMatrix
	///
	///\param variable_idx the index of the variable in the relation
	///
	///\param pRelation a pointer to the relation from which the
	///                 variable_idx-'th variable is taken. Cannot be NULL.
	///
	///\param pMatrix a pointer to the confusion matrix. (*pMatrix)[r][c]
	///               is the number of times that r was expected and c was
	///               received. Cannot be NULL.
        static std::string machineReadableConfusionData(std::size_t variable_idx, const GRelation* pRelation, GMatrix const * const pMatrix);

	///\brief Prints the confusion matrices as machine-readable csv-like lines.
	///
	///The first line is a header giving the names of the columns for the
	///next line.  The first column is the name of the label variable for
	///which the matrix is being printed.  The rest of the columns are the
	///names of the expected/got values (row/column in the input matrices)
	///
	///\param pRelation the relation for which the confusion matrices are
	///                 given.  Cannot be NULL.
	///
	///\param matrixArray matrixArray[i] is null if there is no matrix to
	///                   be printed. Otherwise matrixArray[i] is the
	///                   confusion matrix for the i'th attribute of
	///                   pRelation. Row r, column c of matrixArray[i] is the
	///                   number of times the value r of the attribute was expected
	///                   and c was encountered.
	static void printMachineReadableConfusionMatrices(const GRelation* pRelation, vector<GMatrix*>& matrixArray);

	static void printConfusionMatrices(const GRelation* pRelation, vector<GMatrix*>& matrixArray);

	static void Test(GArgReader& args);

	static void Transduce(GArgReader& args);

	static void TransductiveAccuracy(GArgReader& args);

	static void SplitTest(GArgReader& args);

	static void CrossValidateCallback(void* pSupLearner, size_t nRep, size_t nFold, double foldSSE, size_t rows);

	static void CrossValidate(GArgReader& args);

	static void vette(string& s);

	static void PrecisionRecall(GArgReader& args);

	static void sterilize(GArgReader& args);

	static void regress(GArgReader& args);

	static void metaData(GArgReader& args);

	static void ShowUsage(const char* appName);

	static void showError(GArgReader& args, const char* szAppName, const char* szMessage);
};


class OptimizerTargetFunc : public GTargetFunction
{
public:
	GMatrix* m_pIn;
	GMatrix* m_pOut;
	GFunction* m_pFunc;
	GFunctionParser* m_pParser;

	OptimizerTargetFunc(GMatrix* pIn, GMatrix* pOut, GFunction* pFunc, GFunctionParser* pParser) : GTargetFunction(pFunc->m_expectedParams - pIn->cols()), m_pIn(pIn), m_pOut(pOut), m_pFunc(pFunc), m_pParser(pParser)
	{
	}

	virtual ~OptimizerTargetFunc()
	{
	}

	virtual bool isStable() { return true; }
	virtual bool isConstrained() { return false; }

	virtual void initVector(GVec& pVector)
	{
		pVector.fill(0.1);
	}

	virtual double computeError(const GVec& pVector)
	{
		double sse = 0.0;
		vector<double> params;
		params.resize(m_pFunc->m_expectedParams);
		size_t inDims = m_pIn->cols();
		for(size_t j = 0; j < m_pRelation->size(); j++)
			params[inDims + j] = pVector[j];
		for(size_t i = 0; i < m_pIn->rows(); i++)
		{
			GVec& pIn = m_pIn->row(i);
			for(size_t j = 0; j < inDims; j++)
				params[j] = pIn[j];
			double pred = m_pFunc->call(params, *m_pParser);
			GVec& pOut = m_pOut->row(i);
			double d = pOut[0] - pred;
			sse += d * d;
		}
		return sse;
	}
};

} // namespace GClasses
