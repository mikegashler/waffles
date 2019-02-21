/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Michael R. Smith,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#ifndef __GRECOMMENDER_H__
#define __GRECOMMENDER_H__

#include "GError.h"
#include "GRand.h"
#include "GLearner.h"
#include "GApp.h"
#include "GNeighborFinder.h"
#include "GVec.h"
#include <vector>
#include <map>

namespace GClasses {

class GSparseMatrix;
class GSparseClusterer;
class GSparseSimilarity;
class GRand;
class GNeuralNet;
class GMatrix;
class Rating;
class GClusterer;
class GDom;
class GDomNode;
class GLearnerLoader;

using std::multimap;

struct ArrayWrapper { size_t values[2]; };


/// The base class for collaborative filtering recommender systems.
class GCollaborativeFilter
{
protected:
	GRand m_rand;

public:
	GCollaborativeFilter();
	GCollaborativeFilter(const GDomNode* pNode, GLearnerLoader& ll);
	virtual ~GCollaborativeFilter() {}

	/// Trains this recommender system. Let R be an m-by-n sparse
	/// matrix of known ratings from m users of n items. pData should
	/// contain 3 columns, and one row for each known element in R.
	/// Column 0 in pData specifies the user index from 0 to m-1, column 1
	/// in pData specifies the item index from 0 to n-1, and column 2
	/// in pData specifies the rating vector for that user-item pair. All
	/// attributes in pData should be continuous.
	virtual void train(GMatrix& data) = 0;

	/// Train from an m-by-n dense matrix, where m is the number of users
	/// and n is the number of items. All attributes must be
	/// continuous. Missing values are indicated with UNKNOWN_REAL_VALUE.
	/// If pLabels is non-NULL, then the labels will be appended as
	/// additional items.
	void trainDenseMatrix(const GMatrix& data, const GMatrix* pLabels = NULL);

	/// This returns a prediction for how the specified user
	/// will rate the specified item. (The model must be trained before
	/// this method is called. Also, some values for that user and
	/// item should have been included in the training set, or else
	/// this method will have no basis to make a good prediction.)
	virtual double predict(size_t user, size_t item) = 0;

	/// pVec should be a vector of n real values, where n is the number of
	/// items/attributes/columns in the data that was used to train the model.
	/// to UNKNOWN_REAL_VALUE. This method will evaluate the known elements
	/// and impute (predict) values for the unknown elements. (The model should
	/// be trained before this method is called. Unlike the predict method,
	/// this method can operate on row-vectors that were not part of the training
	/// data.)
	virtual void impute(GVec& vec, size_t dims) = 0;

	/// Marshal this object into a DOM that can be converted to a variety
	/// of formats. (Implementations of this method should use baseDomNode.)
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// This randomly assigns each rating to one of the folds. Then,
	/// for each fold, it calls train with a dataset that contains
	/// everything except for the ratings in that fold. It predicts
	/// values for the items in the fold, and returns the mean-squared
	/// difference between the predictions and the actual ratings.
	/// If pOutMAE is non-NULL, it will be set to the mean-absolute error.
	double crossValidate(GMatrix& data, size_t folds, double* pOutMAE = NULL);

	/// This trains on the training set, and then tests on the test set.
	/// Returns the mean-squared difference between actual and target predictions.
	double trainAndTest(GMatrix& train, GMatrix& test, double* pOutMAE = NULL);

	/// This divides the data into two equal-size parts. It trains on one part, and
	/// then measures the precision/recall using the other part. It returns a
	/// three-column data set with recall scores in column 0 and corresponding
	/// precision scores in column 1. The false-positive rate is in column 2. (So,
	/// if you want a precision-recall plot, just drop column 2. If you want an
	/// ROC curve, drop column 1 and swap the remaining two columns.) This method
	/// assumes the ratings range from 0 to 1, so be sure to scale the ratings to
	/// fit that range before calling this method. If ideal is true, then it will
	/// ignore your model and report the ideal results as if your model always
	/// predicted the correct rating. (This is useful because it shows the best
	/// possible results.)
	GMatrix* precisionRecall(GMatrix& data, bool ideal = false);

	/// Pass in the data returned by the precisionRecall function (unmodified), and
	/// this will compute the area under the ROC curve.
	static double areaUnderCurve(GMatrix& data);

	/// Returns a reference to the pseudo-random number generator associated with this object.
	GRand& rand() { return m_rand; }

	/// Performs a basic unit test on this collaborative filter
	void basicTest(double minMSE);

protected:
	/// Child classes should use this in their implementation of serialize
	GDomNode* baseDomNode(GDom* pDoc, const char* szClassName) const;
};


/// This class always predicts the average rating for each item, no matter
/// to whom it is making the recommendation. The purpose of this algorithm
/// is to serve as a baseline for comparison
class GBaselineRecommender : public GCollaborativeFilter
{
protected:
	GVec m_ratings;
	size_t m_items;

public:
	/// General-purpose constructor
	GBaselineRecommender();

	/// Deserialization constructor
	GBaselineRecommender(const GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GBaselineRecommender();

	/// See the comment for GCollaborativeFilter::train
	virtual void train(GMatrix& data);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Performs unit tests. Throws if a failure occurs. Returns if successful.
	static void test();
};


/// This class makes recommendations by finding the nearest-neighbors (as
/// determined by evaluating only overlapping ratings), and assuming that
/// the ratings of these neighbors will be predictive of your ratings.
class GInstanceRecommender : public GCollaborativeFilter
{
protected:
	size_t m_neighbors;
	GSparseSimilarity* m_pMetric;
	bool m_ownMetric;
	GSparseMatrix* m_pData;
	GBaselineRecommender* m_pBaseline;
	size_t m_significanceWeight;
	std::map<size_t, std::multimap<double,ArrayWrapper> > m_user_depq;

public:
	GInstanceRecommender(size_t neighbors);
	GInstanceRecommender(const GDomNode* pNode, GLearnerLoader& ll);
	virtual ~GInstanceRecommender();

	/// Sets the similarity metric to use. if own is true, then this object will take care
	/// of deleting it as appropriate.
	void setMetric(GSparseSimilarity* pMetric, bool own);

	/// Returns the current similarity metric. (This might be useful, for example, if you
	/// want to modify the regularization value.)
	GSparseSimilarity* metric() { return m_pMetric; }

	/// See the comment for GCollaborativeFilter::train
	virtual void train(GMatrix& data);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Set the value for significance weighting. The default value is zero. If there
	/// are fewer matching items between users than the specified values, the
	/// similarity is scaled by numMathces/sigWeight
	void setSigWeight(size_t sig){ m_significanceWeight = sig; }

	/// This is the same function as predict except that it returns the
	/// the priority queue for the nearest neigbors. The values are further user
	/// by the content-boosted cf prediction method to combine the content-based
	/// and cf predictions.
	multimap<double,ArrayWrapper> getNeighbors(size_t user, size_t item);

	/// This method clears the priority queue that keeps track of the neighbors for
	/// a user. This will help speed up the search for each neighbor. It is used in
	/// the content-boosted filter since it is a dense matrix and should not change
	/// based on which item a rating is being asked for.
	void clearUserDEPQ(){ m_user_depq.clear(); }

	/// Get the rating of an item for a user
	double getRating(size_t user, size_t item); //{ return m_pData->get(user, item); }

	/// Performs unit tests. Throws if a failure occurs. Returns if successful.
	static void test();
};


/// This class clusters the rows according to a sparse similarity metric,
/// then uses the baseline vector in each cluster to make predictions.
class GSparseClusterRecommender : public GCollaborativeFilter
{
protected:
	size_t m_clusters;
	GMatrix* m_pPredictions;
	GSparseClusterer* m_pClusterer;
	bool m_ownClusterer;
	size_t m_users, m_items;

public:
	GSparseClusterRecommender(size_t clusters);
	virtual ~GSparseClusterRecommender();

	/// Returns the number of clusters
	size_t clusterCount() { return m_clusters; }

	/// Set the clustering algorithm to use
	void setClusterer(GSparseClusterer* pClusterer, bool own);

	/// See the comment for GCollaborativeFilter::train
	virtual void train(GMatrix& data);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Performs unit tests. Throws if a failure occurs. Returns if successful.
	static void test();
};



/// This class clusters the rows according to a dense distance metric,
/// then uses the baseline vector in each cluster to make predictions.
class GDenseClusterRecommender : public GCollaborativeFilter
{
protected:
	size_t m_clusters;
	GMatrix* m_pPredictions;
	GClusterer* m_pClusterer;
	bool m_ownClusterer;
	size_t m_users, m_items;

public:
	GDenseClusterRecommender(size_t clusters);
	virtual ~GDenseClusterRecommender();

	/// Returns the number of clusters
	size_t clusterCount() { return m_clusters; }

	/// Set the clustering algorithm to use
	void setClusterer(GClusterer* pClusterer, bool own);

	/// See the comment for GCollaborativeFilter::train
	virtual void train(GMatrix& data);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// The behavior of this method is only defined if GFuzzyKMeans
	/// is used as the clusterer. (It is the default.) This sets
	/// the fuzzifier value on it.
	void setFuzzifier(double d);

	/// Performs unit tests. Throws if a failure occurs. Returns if successful.
	static void test();
};



/// This factors the sparse matrix of ratings, M, such that M = PQ^T
/// where each row in P gives the principal preferences for the corresponding
/// user, and each row in Q gives the linear combination of those preferences
/// that map to a rating for an item. (Actually, P and Q also contain an extra column
/// added for a bias.) This class is implemented according to the specification on
/// page 631 in Takacs, G., Pilaszy, I., Nemeth, B., and Tikk, D. Scalable collaborative
/// filtering approaches for large recommender systems. The Journal of Machine Learning
/// Research, 10:623–656, 2009. ISSN 1532-4435., except with the addition of learning-rate
/// decay and a different stopping criteria.
class GMatrixFactorization : public GCollaborativeFilter
{
protected:
	size_t m_intrinsicDims;
	double m_regularizer;
	GMatrix* m_pP;
	GMatrix* m_pQ;
	GMatrix* m_pPMask;
	GMatrix* m_pQMask;
	GMatrix* m_pPWeights;
	GMatrix* m_pQWeights;
	bool m_nonNeg;
	size_t m_minIters;
	double m_decayRate;

public:
	/// General-purpose constructor
	GMatrixFactorization(size_t intrinsicDims);

	/// Deserialization constructor
	GMatrixFactorization(const GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GMatrixFactorization();

	/// Set the regularization value
	void setRegularizer(double d) { m_regularizer = d; }

	/// Specify that a certain attribute of a certain user profile has a fixed value.
	/// (Values for attr are from 0 to m_intrinsicDims-1. No mechanism is provided to clamp the bias value.)
	void clampUserElement(size_t user, size_t attr, double val);

	/// Specify that a certain attribute of a certain item profile has a fixed value.
	/// (Values for attr are from 0 to m_intrinsicDims-1. No mechanism is provided to clamp the bias value.)
	void clampItemElement(size_t item, size_t attr, double val);

	/// Assumes that column 0 of data is a user ID, and all other columns specify
	/// profile values to clamp beginning at the specifed profile offset.
	void clampUsers(const GMatrix& data, size_t offset = 0);

	/// Assumes that column 0 of data is an item ID, and all other columns specify
	/// profile values to clamp beginning at the specifed profile offset.
	void clampItems(const GMatrix& data, size_t offset = 0);

	/// Constrain all non-bias weights to be non-negative during training.
	void nonNegative() { m_nonNeg = true; }

	/// See the comment for GCollaborativeFilter::train
	virtual void train(GMatrix& data);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// Returns the matrix of user preference vectors
	GMatrix* getP() { return m_pP; }

	/// Returns the matrix of item weight vectors
	GMatrix* getQ() { return m_pQ; }

	/// Returns the matrix of user preference vectors, and gives ownership to the caller.
	GMatrix* dropP() { GMatrix* tmp = m_pP; m_pP = NULL; return tmp; }

	/// Returns the matrix of item weight vectors, and gives ownership to the caller.
	GMatrix* dropQ() { GMatrix* tmp = m_pQ; m_pQ = NULL; return tmp; }

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Set the min number of iterations to train
	void setMinIters(size_t i) { m_minIters = i; }

	/// Set the rate to decay the learning rate
	void setDecayRate(double d) { m_decayRate = d; }

	/// Performs unit tests. Throws if a failure occurs. Returns if successful.
	static void test();

protected:
	/// Returns the sum-squared error for the specified set of ratings
	double validate(GMatrix& data);

	void clampP(size_t i);
	void clampQ(size_t i);
};


/*
/// This class implements the Unsupervised Backpropagation algorithm, as described in
/// Gashler, Michael S. and Smith, Michael R. and Morris, Richard and Martinez, T. (2014),
/// Missing Value Imputation With Unsupervised Backpropagation. Computational Intelligence. doi: 10.1111/coin.12048.
/// This algorithm is very similar to an earlier algorithm
/// called NonlinearPCA, except with the addition of a three-pass training approach that yields
/// better accuracy. If you call noThreePass() before you call train(), then this class implements
/// NonlinearPCA, as published in Scholz, M. Kaplan, F. Guy, C. L. Kopka, J. Selbig, J., Non-linear PCA: a missing
/// data approach, In Bioinformatics, Vol. 21, Number 20, pp. 3887-3895, Oxford
/// University Press, 2005.
class GNonlinearPCA : public GCollaborativeFilter
{
protected:
	size_t m_intrinsicDims;
	size_t m_items;
	double* m_pMins;
	double* m_pMaxs;
	GNeuralNet* m_pModel;
	GMatrix* m_pUsers;
	GMatrix* m_pUserMask;
	GMatrix* m_pItemMask;
	bool m_useInputBias;
	bool m_useThreePass;
	size_t m_minIters;
	double m_decayRate;
	double m_regularizer;

public:
	/// General-purpose constructor
	GNonlinearPCA(size_t intrinsicDims);

	/// Deserialization constructor
	GNonlinearPCA(const GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GNonlinearPCA();

	/// Returns a pointer to the neural net that is used to model the recommendation space.
	/// You may want to use this method to add layers to the network. (At least one layer
	/// is necessary). You may also use it to set the learning rate, or change
	/// activation functions before the model is trained.
	GNeuralNet* model() { return m_pModel; }

	/// Returns a pointer to the matrix of user preference vectors.
	GMatrix* users() { return m_pUsers; }

	/// Specify that a certain attribute of a certain user profile has a fixed value.
	/// (Values for attr are from 0 to m_intrinsicDims-2. No mechanism is provided to clamp the input bias.)
	void clampUserElement(size_t user, size_t attr, double val);

	/// Specify that a certain attribute of a certain item profile has a fixed value.
	/// (Values for attr are from 0 to m_pModel->outputLayer().inputs()-1. No mechanism is provided to clamp the item bias.)
	void clampItemElement(size_t item, size_t attr, double val);

	/// Assumes that column 0 of data is a user ID, and all other columns specify
	/// profile values to clamp beginning at the specifed profile offset.
	void clampUsers(const GMatrix& data, size_t offset = 0);

	/// Assumes that column 0 of data is an item ID, and all other columns specify
	/// profile values to clamp beginning at the specifed profile offset.
	void clampItems(const GMatrix& data, size_t offset = 0);

	/// See the comment for GCollaborativeFilter::train
	virtual void train(GMatrix& data);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Specify to use no bias value with the inputs
	void noInputBias() { m_useInputBias = false; }

	/// Specify not to use three-pass training. (It will just use one pass instead.)
	void noThreePass() { m_useThreePass = false; }

	/// Sset the min number of iterations to train
	void setMinIters(size_t i) { m_minIters = i; }

	/// Set the rate to decay the learning rate
	void setDecayRate(double d) { m_decayRate = d; }

	/// Set the regularization value
	void setRegularizer(double d) { m_regularizer = d; }

	/// Performs unit tests. Throws if a failure occurs. Returns if successful.
	static void test();

protected:
	/// Returns the sum-squared error for the specified set of ratings
	double validate(GNeuralNet* pNN, GMatrix& data);

	void clampUsersInternal(size_t i);
	void clampItemsInternal(size_t i);
};
*/


/*
/// A collaborative filtering algorithm invented by Mike Smith.
class GHybridNonlinearPCA : public GNonlinearPCA
{
protected:
	GMatrix* m_itemAttrs;
	std::set<size_t> m_itemSet;
	double* m_itemMax;
	double* m_itemMin;
	GKdTree* m_neighbors;
	size_t* m_itemMap;
	size_t m_numNeighbors;
	size_t* m_pRatingCount;

public:
        /// General-purpose constructor
        GHybridNonlinearPCA(size_t intrinsicDims);

//        /// Deserialization constructor
//        GNonlinearPCA(const GDomNode* pNode, GLearnerLoader& ll);

        /// Destructor
        virtual ~GHybridNonlinearPCA();

        /// See the comment for GCollaborativeFilter::train
        virtual void train(GMatrix& data);

        /// See the comment for GCollaborativeFilter::predict
        virtual double predict(size_t user, size_t item);

	void setItemAttributes(GMatrix& itemAttrs);


protected:
        /// Returns the sum-squared error for the specified set of ratings
        double validate(GNeuralNet* pNN, GMatrix& data);

};
*/



/// This class performs bootstrap aggregation with collaborative filtering algorithms.
class GBagOfRecommenders : public GCollaborativeFilter
{
protected:
	std::vector<GCollaborativeFilter*> m_filters;
	size_t m_itemCount;

public:
	/// General-purpose constructor
	GBagOfRecommenders();

	/// Deserialization constructor
	GBagOfRecommenders(const GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GBagOfRecommenders();

	/// Returns the vector of filters
	std::vector<GCollaborativeFilter*>& filters() { return m_filters; }

	/// Add a filter to the bag
	void addRecommender(GCollaborativeFilter* pRecommender);

	/// See the comment for GCollaborativeFilter::train
	virtual void train(GMatrix& data);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// Delete all of the filters
	void clear();

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// Performs unit tests. Throws if a failure occurs. Returns if successful.
	static void test();
};

class GContentBasedFilter : public GCollaborativeFilter
{
protected:
	std::vector<GSupervisedLearner*> m_learners;
	std::map<size_t, size_t> m_itemMap;
	std::map<size_t, size_t> m_userMap;
	std::multimap<size_t, size_t> m_userRatings;
        GMatrix* m_itemAttrs;
	size_t m_items, m_users;
	GArgReader m_args;
	int m_init_pos;

public:
	/// General-purpose constructor
	GContentBasedFilter(GArgReader copy)
	: GCollaborativeFilter(), m_itemAttrs(NULL), m_args(copy)
	{
		m_init_pos = copy.get_pos();
	}

	/// Destructor
	virtual ~GContentBasedFilter();

	virtual void train(GMatrix& data);

	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(GVec& vec, size_t dims);

	/// Delete all of the learners
	void clear();

	void setItemAttributes(GMatrix& itemAttrs);

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const;

	std::map<size_t, size_t> getUserMap(){ return m_userMap; }

	std::map<size_t, size_t> getItemMap(){ return m_itemMap; }

	std::multimap<size_t, size_t> getUserRatings(){ return m_userRatings; }
};

class GContentBoostedCF : public GCollaborativeFilter
{
protected:
	GContentBasedFilter* m_cbf;
	GInstanceRecommender* m_cf;
	std::map<size_t, size_t> m_userMap;
	size_t* m_ratingCounts;
	double* m_pseudoRatingSum;

public:
	//// General-purpose constructor
	GContentBoostedCF(GArgReader copy);

	/// Destructor
	virtual ~GContentBoostedCF();

	virtual void train(GMatrix& data);

	virtual double predict(size_t user, size_t item);

	virtual void impute(GVec& vec, size_t dims);

	/// See the comment for GCollaborativeFilter::serialize
	virtual GDomNode* serialize(GDom* pDoc) const { return NULL; };

};

} // namespace GClasses

#endif // __GRECOMMENDER_H__
