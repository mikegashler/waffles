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

#include "GRecommenderLib.h"
#include <memory>

using namespace GClasses;
using std::cout;
using std::cerr;
using std::string;
using std::vector;
using std::set;


size_t GRecommenderLib::getAttrVal(const char* szString, size_t attrCount)
{
        bool fromRight = false;
        if(*szString == '*')
        {
                fromRight = true;
                szString++;
        }
        if(*szString < '0' || *szString > '9')
                throw Ex("Expected a digit while parsing attribute list");
#ifdef WINDOWS
        size_t val = (size_t)_strtoui64(szString, (char**)NULL, 10);
#else
        size_t val = strtoull(szString, (char**)NULL, 10);
#endif
        if(fromRight)
                val = attrCount - 1 - val;
        return val;
}

void GRecommenderLib::parseAttributeList(vector<size_t>& list, GArgReader& args, size_t attrCount)
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

void GRecommenderLib::loadData(GMatrix& data, const char* szFilename)
{
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	if(_stricmp(szFilename + pd.extStart, ".sparse") == 0)
	{
		GDom doc;
		doc.loadJson(szFilename);
		GSparseMatrix sm(doc.root());
		data.resize(0, 3);
		for(size_t i = 0; i < sm.rows(); i++)
		{
			GSparseMatrix::Iter rowEnd = sm.rowEnd(i);
			for(GSparseMatrix::Iter it = sm.rowBegin(i); it != rowEnd; it++)
			{
				GVec& vec = data.newRow();
				vec[0] = (double)i;
				vec[1] = (double)it->first;
				vec[2] = it->second;
			}
		}
	}
	else if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
		data.loadArff(szFilename);
	else
		throw Ex("Unsupported file format: ", szFilename + pd.extStart);
}

GSparseMatrix* GRecommenderLib::loadSparseData(const char* szFilename)
{
	// Load the dataset by extension
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	if(_stricmp(szFilename + pd.extStart, ".arff") == 0)
	{
		// Convert a 3-column dense ARFF file to a sparse matrix
		GMatrix data;
		data.loadArff(szFilename);
		if(data.cols() != 3)
			throw Ex("Expected 3 columns: 0) user or row-index, 1) item or col-index, 2) value or rating");
		double m0 = data.columnMin(0);
		double r0 = data.columnMax(0) - m0;
		double m1 = data.columnMin(1);
		double r1 = data.columnMax(1) - m1;
		if(m0 < 0 || m0 > 1e10 || r0 < 2 || r0 > 1e10)
			throw Ex("Invalid row indexes");
		if(m1 < 0 || m1 > 1e10 || r1 < 2 || r1 > 1e10)
			throw Ex("Invalid col indexes");
		GSparseMatrix* pMatrix = new GSparseMatrix(size_t(m0 + r0) + 1, size_t(m1 + r1) + 1, UNKNOWN_REAL_VALUE);
		std::unique_ptr<GSparseMatrix> hMatrix(pMatrix);
		for(size_t i = 0; i < data.rows(); i++)
		{
			GVec& row = data.row(i);
			pMatrix->set(size_t(row[0]), size_t(row[1]), row[2]);
		}
		return hMatrix.release();
	}
	else if(_stricmp(szFilename + pd.extStart, ".sparse") == 0)
	{
		GDom doc;
		doc.loadJson(szFilename);
		return new GSparseMatrix(doc.root());
	}
	throw Ex("Unsupported file format: ", szFilename + pd.extStart);
	return NULL;
}

GBaselineRecommender* GRecommenderLib::InstantiateBaselineRecommender(GArgReader& args)
{
	return new GBaselineRecommender();
}

GBagOfRecommenders* GRecommenderLib::InstantiateBagOfRecommenders(GArgReader& args)
{
	GBagOfRecommenders* pEnsemble = new GBagOfRecommenders();
	while(args.size() > 0)
	{
		if(args.if_pop("end"))
			break;
		int instance_count = args.pop_uint();
		int arg_pos = args.get_pos();
		for(int i = 0; i < instance_count; i++)
		{
			args.set_pos(arg_pos);
			GCollaborativeFilter* pRecommender = InstantiateAlgorithm(args);
			pEnsemble->addRecommender(pRecommender);
		}
	}
	return pEnsemble;
}

GInstanceRecommender* GRecommenderLib::InstantiateInstanceRecommender(GArgReader& args)
{
	if(args.size() < 1)
		throw Ex("The number of neighbors must be specified for this algorithm");
	int neighborCount = args.pop_uint();
	double regularizer = 0.0;
	bool pearson = false;
	size_t sig = 0;
	while(args.next_is_flag())
	{
		if(args.if_pop("-pearson"))
			pearson = true;
		else if(args.if_pop("-regularize"))
			regularizer = args.pop_double();
		else if (args.if_pop("-sigWeight"))
			sig = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}
	GInstanceRecommender* pModel = new GInstanceRecommender(neighborCount);
	if(pearson)
		pModel->setMetric(new GPearsonCorrelation(), true);
	pModel->metric()->setRegularizer(regularizer);
	pModel->setSigWeight(sig);
	return pModel;
}

GDenseClusterRecommender* GRecommenderLib::InstantiateDenseClusterRecommender(GArgReader& args)
{
	if(args.size() < 1)
		throw Ex("The number of clusters must be specified for this algorithm");
	size_t clusterCount = args.pop_uint();
	double missingPenalty = 1.0;
	double norm = 2.0;
	double fuzzifier = 1.3;
	while(args.next_is_flag())
	{
		if(args.if_pop("-norm"))
			norm = args.pop_double();
		else if(args.if_pop("-missingpenalty"))
			missingPenalty = args.pop_double();
		else if(args.if_pop("-fuzzifier"))
			fuzzifier = args.pop_double();
		else
			throw Ex("Invalid option: ", args.peek());
		// todo: allow the user to specify the clustering algorithm. (Currently, it uses k-means, but k-medoids and agglomerativeclusterer should also be an option)
	}
	GDenseClusterRecommender* pModel = new GDenseClusterRecommender(clusterCount);
	if(norm == 2.0)
	{
		if(missingPenalty != 1.0)
		{
			GFuzzyKMeans* pClusterer = new GFuzzyKMeans(clusterCount, &pModel->rand());
			pModel->setClusterer(pClusterer, true);
			GRowDistance* pMetric = new GRowDistance();
			pClusterer->setMetric(pMetric, true);
			pMetric->setDiffWithUnknown(missingPenalty);
		}
	}
	else
	{
		GFuzzyKMeans* pClusterer = new GFuzzyKMeans(clusterCount, &pModel->rand());
		pModel->setClusterer(pClusterer, true);
		GLNormDistance* pMetric = new GLNormDistance(norm);
		pClusterer->setMetric(pMetric, true);
		if(missingPenalty != 1.0)
			pMetric->setDiffWithUnknown(missingPenalty);
	}
	pModel->setFuzzifier(fuzzifier);
	return pModel;
}

GSparseClusterRecommender* GRecommenderLib::InstantiateSparseClusterRecommender(GArgReader& args)
{
	if(args.size() < 1)
		throw Ex("The number of clusters must be specified for this algorithm");
	size_t clusterCount = args.pop_uint();
	bool pearson = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-pearson"))
			pearson = true;
		else
			throw Ex("Invalid option: ", args.peek());
		// todo: allow the user to specify the clustering algorithm. (Currently, it uses k-means, but k-medoids should also be an option)
	}
	GSparseClusterRecommender* pModel = new GSparseClusterRecommender(clusterCount);
	if(pearson)
	{
		GKMeansSparse* pClusterer = new GKMeansSparse(clusterCount, &pModel->rand());
		pClusterer->setMetric(new GPearsonCorrelation(), true);
		pModel->setClusterer(pClusterer, true);
	}
	return pModel;
}

GMatrixFactorization* GRecommenderLib::InstantiateMatrixFactorization(GArgReader& args)
{
	if(args.size() < 1)
		throw Ex("The number of intrinsic dims must be specified for this algorithm");
	size_t intrinsicDims = args.pop_uint();
	GMatrixFactorization* pModel = new GMatrixFactorization(intrinsicDims);
	while(args.next_is_flag())
	{
		if(args.if_pop("-regularize"))
			pModel->setRegularizer(args.pop_double());
		else if(args.if_pop("-miniters"))
			pModel->setMinIters(args.pop_uint());
		else if(args.if_pop("-decayrate"))
			pModel->setDecayRate(args.pop_double());
		else if(args.if_pop("-nonneg"))
			pModel->nonNegative();
		else if(args.if_pop("-clampusers"))
		{
			GMatrix tmp;
			tmp.loadArff(args.pop_string());
			size_t offset = args.pop_uint();
			pModel->clampUsers(tmp, offset);
		}
		else if(args.if_pop("-clampitems"))
		{
			GMatrix tmp;
			tmp.loadArff(args.pop_string());
			size_t offset = args.pop_uint();
			pModel->clampItems(tmp, offset);
		}
		else
			throw Ex("Invalid option: ", args.peek());
	}
	return pModel;
}
/*
GNonlinearPCA* GRecommenderLib::InstantiateNonlinearPCA(GArgReader& args)
{
	if(args.size() < 1)
		throw Ex("The number of intrinsic dims must be specified for this algorithm");
	size_t intrinsicDims = args.pop_uint();
	GNonlinearPCA* pModel = new GNonlinearPCA(intrinsicDims);
	while(args.next_is_flag())
	{
		if(args.if_pop("-addlayer"))
			pModel->model()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, args.pop_uint()));
		else if(args.if_pop("-learningrate"))
			pModel->model()->setLearningRate(args.pop_double());
		else if(args.if_pop("-momentum"))
			pModel->model()->setMomentum(args.pop_double());
		else if(args.if_pop("-windowepochs"))
			pModel->model()->setWindowSize(args.pop_uint());
		else if(args.if_pop("-minwindowimprovement"))
			pModel->model()->setImprovementThresh(args.pop_double());
		else if(args.if_pop("-noinputbias"))
			pModel->noInputBias();
		else if(args.if_pop("-nothreepass"))
			pModel->noThreePass();
		else if(args.if_pop("-miniters"))
			pModel->setMinIters(args.pop_uint());
		else if(args.if_pop("-decayrate"))
			pModel->setDecayRate(args.pop_double());
		else if(args.if_pop("-regularize"))
			pModel->setRegularizer(args.pop_double());
		else if(args.if_pop("-dontsquashoutputs"))
			pAF = new GActivationIdentity();
		else if(args.if_pop("-clampusers"))
		{
			GMatrix tmp;
			tmp.loadArff(args.pop_string());
			size_t offset = args.pop_uint();
			pModel->clampUsers(tmp, offset);
		}
		else if(args.if_pop("-clampitems"))
		{
			GMatrix tmp;
			tmp.loadArff(args.pop_string());
			size_t offset = args.pop_uint();
			pModel->clampItems(tmp, offset);
		}
		else
			throw Ex("Invalid option: ", args.peek());
	}
	pModel->model()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE, pAF));
	return pModel;
}
*/
/*
GHybridNonlinearPCA* GRecommenderLib::InstantiateHybridNonlinearPCA(GArgReader& args)
{
	if(args.size() < 2)
		throw Ex("The number of input dims AND the location of the ARFF for the item attributes must be specified for this algorithm");
	size_t intrinsicDims = args.pop_uint();
	GMatrix data;
	loadData(data, args.pop_string());
//	size_t inputDims = args.pop_uint();
	GHybridNonlinearPCA* pModel = new GHybridNonlinearPCA(intrinsicDims);
	pModel->setItemAttributes(data);
	while(args.next_is_flag())
	{
		if(args.if_pop("-addlayer"))
			pModel->model()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, args.pop_uint()));
		else if(args.if_pop("-learningrate"))
			pModel->model()->setLearningRate(args.pop_double());
		else if(args.if_pop("-momentum"))
			pModel->model()->setMomentum(args.pop_double());
		else if(args.if_pop("-windowepochs"))
			pModel->model()->setWindowSize(args.pop_uint());
		else if(args.if_pop("-minwindowimprovement"))
			pModel->model()->setImprovementThresh(args.pop_double());
		else if(args.if_pop("-noinputbias"))
			pModel->noInputBias();
		else if(args.if_pop("-nothreepass"))
			pModel->noThreePass();
		else if(args.if_pop("-miniters"))
			pModel->setMinIters(args.pop_uint());
		else if(args.if_pop("-decayrate"))
			pModel->setDecayRate(args.pop_double());
		else if(args.if_pop("-regularize"))
			pModel->setRegularizer(args.pop_double());
		else if(args.if_pop("-dontsquashoutputs"))
			pAF = new GActivationIdentity();
		else
			throw Ex("Invalid option: ", args.peek());
	}
	pModel->model()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE, pAF));
	return pModel;
}
*/
GContentBasedFilter* GRecommenderLib::InstantiateContentBasedFilter(GArgReader& args)
{
	if(args.size() < 2)
		throw Ex("The location of the ARFF for the item attributes and a learning algorithm must be specified for this algorithm");
	GMatrix data;
	loadData(data, args.pop_string());
	GArgReader copy = args;
	args.clear_args();
	GContentBasedFilter* pModel = new GContentBasedFilter(copy);
	pModel->setItemAttributes(data);

	return pModel;
}


GContentBoostedCF* GRecommenderLib::InstantiateContentBoostedCF(GArgReader& args)
{
	if(args.size() < 3)
		throw Ex("The location of the ARFF for the item attributes and a learning algorithm must be specified for the content-based algorithm and the number of neighbors must be specified for the instance-based CF algorithm");
	GArgReader copy = args;
	args.clear_args();
	GContentBoostedCF* pModel = new GContentBoostedCF(copy);

	return pModel;
}

void GRecommenderLib::showInstantiateAlgorithmError(const char* szMessage, GArgReader& args)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	const char* szAlgName = args.peek();
	UsageNode* pAlgTree = makeCollaborativeFilterUsageTree();
	std::unique_ptr<UsageNode> hAlgTree(pAlgTree);
	if(szAlgName)
	{
		UsageNode* pUsageAlg = pAlgTree->choice(szAlgName);
		if(pUsageAlg)
		{
			cerr << "Partial Usage Information:\n\n";
			pUsageAlg->print(cerr, 0, 3, 76, 1000, true);
		}
		else
		{
			cerr << "\"" << szAlgName << "\" is not a recognized algorithm. Try one of these:\n\n";
			pAlgTree->print(cerr, 0, 3, 76, 1, false);
		}
	}
	else
	{
		cerr << "Expected an algorithm. Here are some choices:\n";
		pAlgTree->print(cerr, 0, 3, 76, 1, false);
	}
	cerr << "\nTo see full usage information, run:\n	waffles_learn usage\n\n";
	cerr << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cerr.flush();
}

GCollaborativeFilter* GRecommenderLib::InstantiateAlgorithm(GArgReader& args)
{
	int argPos = args.get_pos();
	if(args.size() < 1)
		throw Ex("No algorithm specified.");
	try
	{
		if(args.if_pop("baseline"))
			return InstantiateBaselineRecommender(args);
		else if(args.if_pop("bag"))
			return InstantiateBagOfRecommenders(args);
		else if(args.if_pop("instance"))
			return InstantiateInstanceRecommender(args);
		else if(args.if_pop("clusterdense"))
			return InstantiateDenseClusterRecommender(args);
		else if(args.if_pop("clustersparse"))
			return InstantiateSparseClusterRecommender(args);
		else if(args.if_pop("matrix"))
			return InstantiateMatrixFactorization(args);
//		else if(args.if_pop("nlpca"))
//			return InstantiateNonlinearPCA(args);
//		else if(args.if_pop("hybridnlpca"))
//			return InstantiateHybridNonlinearPCA(args);
		else if(args.if_pop("contentbased"))
			return InstantiateContentBasedFilter(args);
		else if(args.if_pop("cbcf"))
			return InstantiateContentBoostedCF(args);
		else
			throw Ex("Unrecognized algorithm name: ", args.peek());
	}
	catch(const std::exception& e)
	{
		args.set_pos(argPos);
		showInstantiateAlgorithmError(e.what(), args);
		throw Ex("nevermind"); // this means "don't display another error message"
	}
	return NULL;
}

void GRecommenderLib::crossValidate(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	size_t folds = 2;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-folds"))
			folds = args.pop_uint();
		else
			throw Ex("Invalid crossvalidate option: ", args.peek());
	}
	if(folds < 2)
		throw Ex("There must be at least 2 folds.");

	// Load the data
	if(args.size() < 1)
		throw Ex("No dataset specified.");
	GMatrix data;
	loadData(data, args.pop_string());

	// Instantiate the recommender
	GCollaborativeFilter* pModel = InstantiateAlgorithm(args);
	std::unique_ptr<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		throw Ex("Superfluous argument: ", args.peek());
	pModel->rand().setSeed(seed);

	// Do cross-validation
	double mae;
	double mse;
	mse = pModel->crossValidate(data, folds, &mae);
	cout << "RMSE=" << sqrt(mse) << ", MSE=" << mse << ", MAE=" << mae << "\n";
}

void GRecommenderLib::precisionRecall(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool ideal = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-ideal"))
			ideal = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Load the data
	if(args.size() < 1)
		throw Ex("No dataset specified.");
	GMatrix data;
	loadData(data, args.pop_string());

	// Instantiate the recommender
	GCollaborativeFilter* pModel = InstantiateAlgorithm(args);
	std::unique_ptr<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		throw Ex("Superfluous argument: ", args.peek());
	pModel->rand().setSeed(seed);

	// Generate precision-recall data
	GMatrix* pResults = pModel->precisionRecall(data, ideal);
	std::unique_ptr<GMatrix> hResults(pResults);
	pResults->deleteColumns(2, 1); // we don't need the false-positive rate column
	pResults->print(cout);
}

void GRecommenderLib::ROC(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool ideal = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-ideal"))
			ideal = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Load the data
	if(args.size() < 1)
		throw Ex("No dataset specified.");
	GMatrix data;
	loadData(data, args.pop_string());

	// Instantiate the recommender
	GCollaborativeFilter* pModel = InstantiateAlgorithm(args);
	std::unique_ptr<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		throw Ex("Superfluous argument: ", args.peek());
	pModel->rand().setSeed(seed);

	// Generate ROC data
	GMatrix* pResults = pModel->precisionRecall(data, ideal);
	std::unique_ptr<GMatrix> hResults(pResults);
	double auc = GCollaborativeFilter::areaUnderCurve(*pResults);
	pResults->deleteColumns(1, 1); // we don't need the precision column
	pResults->swapColumns(0, 1);
	cout << "% Area Under the Curve = " << auc << "\n";
	pResults->print(cout);
}

void GRecommenderLib::transacc(GArgReader& args)
{
	// Parse options
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else
			throw Ex("Invalid crossvalidate option: ", args.peek());
	}

	// Load the data
	if(args.size() < 1)
		throw Ex("No training set specified.");
	GMatrix train;
	loadData(train, args.pop_string());
	if(args.size() < 1)
		throw Ex("No test set specified.");
	GMatrix test;
	loadData(test, args.pop_string());

	// Instantiate the recommender
	GCollaborativeFilter* pModel = InstantiateAlgorithm(args);
	std::unique_ptr<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		throw Ex("Superfluous argument: ", args.peek());
	pModel->rand().setSeed(seed);

	// Do cross-validation
	double mae;
	double mse = pModel->trainAndTest(train, test, &mae);
	cout << "MSE=" << mse << ", MAE=" << mae << "\n";
}

void GRecommenderLib::fillMissingValues(GArgReader& args)
{
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	bool normalize = true;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seed"))
			seed = args.pop_uint();
		else if(args.if_pop("-nonormalize"))
			normalize = false;
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Load the data and the filter
	GMatrix dataOrig;
	dataOrig.loadArff(args.pop_string());

	// Parse params
	vector<size_t> ignore;
	while(args.next_is_flag())
	{
		if(args.if_pop("-ignore"))
			parseAttributeList(ignore, args, dataOrig.cols());
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Throw out the ignored attributes
	std::sort(ignore.begin(), ignore.end());
	for(size_t i = ignore.size() - 1; i < ignore.size(); i--)
		dataOrig.deleteColumns(ignore[i], 1);

	GRelation* pOrigRel = dataOrig.relation().clone();
	std::unique_ptr<GRelation> hOrigRel(pOrigRel);
	GCollaborativeFilter* pModel = InstantiateAlgorithm(args);
	std::unique_ptr<GCollaborativeFilter> hModel(pModel);
	if(args.size() > 0)
		throw Ex("Superfluous argument: ", args.peek());
	pModel->rand().setSeed(seed);

	// Convert to all normalized real values
	GNominalToCat* pNtc = new GNominalToCat();
	GIncrementalTransform* pFilter = pNtc;
	std::unique_ptr<GIncrementalTransformChainer> hChainer;
	if(normalize)
	{
		GIncrementalTransformChainer* pChainer = new GIncrementalTransformChainer(new GNormalize(), pNtc);
		hChainer.reset(pChainer);
		pFilter = pChainer;
	}
	pNtc->preserveUnknowns();
	pFilter->train(dataOrig);
	GMatrix* pData = pFilter->transformBatch(dataOrig);
	std::unique_ptr<GMatrix> hData(pData);

	// Convert to 3-column form
	auto pMatrix = std::unique_ptr<GMatrix>(new GMatrix(0, 3));
	size_t dims = pData->cols();
	for(size_t i = 0; i < pData->rows(); i++)
	{
		GVec& row = pData->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			if(row[j] != UNKNOWN_REAL_VALUE)
			{
				GVec& vec = pMatrix->newRow();
				vec[0] = (double)i;
				vec[1] = (double)j;
				vec[2] = row[j];
			}
		}
	}

	// Train the collaborative filter
	pModel->train(*pMatrix);

	// Predict values for missing elements
	for(size_t i = 0; i < pData->rows(); i++)
	{
		GVec& row = pData->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			if(row[j] == UNKNOWN_REAL_VALUE)
				row[j] = pModel->predict(i, j);
			GAssert(row[j] != UNKNOWN_REAL_VALUE);
		}
	}

	// Convert the data back to its original form
	auto pOut = pFilter->untransformBatch(*pData);
	pOut->setRelation(hOrigRel.release());
	pOut->print(cout);
}

void GRecommenderLib::ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeRecommendUsageTree();
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(cout, 0, 3, 76, 1000, true);
	UsageNode* pUsageTree2 = makeCollaborativeFilterUsageTree();
	std::unique_ptr<UsageNode> hUsageTree2(pUsageTree2);
	pUsageTree2->print(cout, 0, 3, 76, 1000, true);
	cout.flush();
}

void GRecommenderLib::showError(GArgReader& args, const char* szAppName, const char* szMessage)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	args.set_pos(1);
	const char* szCommand = args.peek();
	UsageNode* pUsageTree = makeRecommendUsageTree();
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << szAppName << " ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, true);
			if(pUsageCommand->findPart("[collab-filter]") >= 0)
			{
				UsageNode* pAlgTree = makeCollaborativeFilterUsageTree();
				std::unique_ptr<UsageNode> hAlgTree(pAlgTree);
				pAlgTree->print(cerr, 1, 3, 76, 2, false);
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
