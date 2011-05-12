/*
	Copyright (C) 2010, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GRECOMMENDER_H__
#define __GRECOMMENDER_H__

#include "GError.h"
#include <vector>

namespace GClasses {

class GSparseMatrix;
class GSparseClusterer;
class GSparseSimilarity;
class GRand;
class GNeuralNet;
class GMatrix;
class Rating;
class GClusterer;


/// The base class for collaborative filtering recommender systems.
class GCollaborativeFilter
{
public:
	GCollaborativeFilter() {}
	virtual ~GCollaborativeFilter() {}

	/// Trains this recommender system in batch mode. The columns
	/// in pData represent the items. The rows in pData represent
	/// the users. The elements should all be real (continuous)
	/// values that specify the users expressed opinion about the item.
	/// Elements should be set to UNKNOWN_REAL_VALUE if the user has
	/// not expressed an opinion about the item.
	virtual void trainBatch(GSparseMatrix* pData) = 0;

	/// This returns a prediction for how much the specified user
	/// will like the specified item. (The model must be trained before
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
	virtual void impute(double* pVec) = 0;

	/// This randomly assignes each rating to one of the folds. Then,
	/// for each fold, it calls trainBatch with a dataset that contains
	/// everything except for the ratings in that fold. It predicts
	/// values for the items in the fold, and returns the mean-squared
	/// difference between the predictions and the actual ratings.
	/// If there are more than maxRecommendationsPerRow ratings in the current
	/// fold in a particular row, then only the maxRecommendationsPerRow ratings
	/// with the higest predicted value will be included in the results for that row.
	/// If pOutMAE is non-NULL, it will be set to the mean-absolute error.
	double crossValidate(GSparseMatrix* pData, size_t folds, GRand* pRand, size_t maxRecommendationsPerRow = 1000000, double* pOutMAE = NULL);

	/// This trains on the training set, and then tests on the test set.
	/// Returns the mean-squared difference between actual and target predictions.
	double transduce(GSparseMatrix& train, GSparseMatrix& test, double* pOutMAE = NULL);

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
	GMatrix* precisionRecall(GSparseMatrix* pData, GRand* pRand, bool ideal = false);

	/// Pass in the data returned by the precisionRecall function (unmodified), and
	/// this will compute the area under the ROC curve.
	static double areaUnderCurve(GMatrix* pData);
};


/// This class always predicts the average rating for each item, no matter
/// to whom it is making the recommendation. The purpose of this algorithm
/// is to serve as a baseline for comparison
class GBaselineRecommender : public GCollaborativeFilter
{
protected:
	double* m_pRatings;
	size_t m_items;

public:
	GBaselineRecommender();
	virtual ~GBaselineRecommender();

	/// See the comment for GCollaborativeFilter::trainBatch
	virtual void trainBatch(GSparseMatrix* pData);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(double* pVec);
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
	double* m_pBaseline;

public:
	GInstanceRecommender(size_t neighbors);
	virtual ~GInstanceRecommender();

	/// Sets the similarity metric to use. if own is true, then this object will take care
	/// of deleting it as appropriate.
	void setMetric(GSparseSimilarity* pMetric, bool own);

	/// Returns the current similarity metric. (This might be useful, for example, if you
	/// want to modify the regularization value.)
	GSparseSimilarity* metric() { return m_pMetric; }

	/// See the comment for GCollaborativeFilter::trainBatch
	virtual void trainBatch(GSparseMatrix* pData);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(double* pVec);
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
	GRand* m_pRand;

public:
	GSparseClusterRecommender(size_t clusters, GRand* pRand);
	virtual ~GSparseClusterRecommender();

	/// Returns the number of clusters
	size_t clusterCount() { return m_clusters; }

	/// Set the clustering algorithm to use
	void setClusterer(GSparseClusterer* pClusterer, bool own);

	/// See the comment for GCollaborativeFilter::trainBatch
	virtual void trainBatch(GSparseMatrix* pData);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(double* pVec);
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
	GRand* m_pRand;

public:
	GDenseClusterRecommender(size_t clusters, GRand* pRand);
	virtual ~GDenseClusterRecommender();

	/// Returns the number of clusters
	size_t clusterCount() { return m_clusters; }

	/// Set the clustering algorithm to use
	void setClusterer(GClusterer* pClusterer, bool own);

	/// See the comment for GCollaborativeFilter::trainBatch
	virtual void trainBatch(GSparseMatrix* pData);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(double* pVec);
};



/// This factors the sparse matrix of ratings, M, such that M = PQ^T
/// where each row in P gives the principal preferences for the corresponding
/// user, and each row in Q gives the linear combination of those preferences
/// that map to a rating for an item. (Actually, P and Q also contain an extra column
/// added for a bias.) This class is implemented according to the specification on
/// page 631 in Takacs, G., Pilaszy, I., Nemeth, B., and Tikk, D. Scalable collaborative
/// filtering approaches for large recommender systems. The Journal of Machine Learning
/// Research, 10:623–656, 2009. ISSN 1532-4435., except with the addition of learning-rate
/// decay and a different stopping criteria, I don't regularize the bias weights, and I
/// don't store the superfluous 1's in the matrices.
class GMatrixFactorization : public GCollaborativeFilter
{
protected:
	size_t m_intrinsicDims;
	double m_regularizer;
	GMatrix* m_pP;
	GMatrix* m_pQ;
	GRand& m_rand;

public:
	GMatrixFactorization(size_t intrinsicDims, GRand& rand);
	virtual ~GMatrixFactorization();

	/// Set the regularization value
	void setRegularizer(double d) { m_regularizer = d; }

	/// See the comment for GCollaborativeFilter::trainBatch
	virtual void trainBatch(GSparseMatrix* pData);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(double* pVec);

protected:
	/// Returns the sum-squared error for the specified set of ratings
	double validate(std::vector<Rating*>& data);
};



/// This class trains a generative neural network to fit the sparse matrix
/// of ratings. This may be seen as a non-linear generalization of matrix
/// factorization.
class GNeuralRecommender : public GCollaborativeFilter
{
protected:
	size_t m_intrinsicDims;
	double* m_pMins;
	double* m_pMaxs;
	GRand* m_pRand;
	GNeuralNet* m_pModel;
	GMatrix* m_pUsers;

public:
	GNeuralRecommender(size_t intrinsicDims, GRand* pRand);
	virtual ~GNeuralRecommender();

	/// Returns a pointer to the neural net that is used to model the recommendation space.
	/// You may want to use this method to add hidden layers, set the learning rate, or change
	/// activation functions before the model is trained.
	GNeuralNet* model() { return m_pModel; }

	/// See the comment for GCollaborativeFilter::trainBatch
	virtual void trainBatch(GSparseMatrix* pData);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(double* pVec);

protected:
	/// Returns the sum-squared error for the specified set of ratings
	double validate(std::vector<Rating*>& data);
};



/// This class performs bootstrap aggregation with collaborative filtering algorithms.
class GBagOfRecommenders : public GCollaborativeFilter
{
protected:
	std::vector<GCollaborativeFilter*> m_filters;
	size_t m_itemCount;
	GRand& m_rand;

public:
	GBagOfRecommenders(GRand& rand);
	virtual ~GBagOfRecommenders();

	/// Returns the vector of filters
	std::vector<GCollaborativeFilter*>& filters() { return m_filters; }

	/// Add a filter to the bag
	void addRecommender(GCollaborativeFilter* pRecommender);

	/// See the comment for GCollaborativeFilter::trainBatch
	virtual void trainBatch(GSparseMatrix* pData);

	/// See the comment for GCollaborativeFilter::predict
	virtual double predict(size_t user, size_t item);

	/// See the comment for GCollaborativeFilter::impute
	virtual void impute(double* pVec);

	/// Delete all of the filters
	void clear();
};


} // namespace GClasses

#endif // __GRECOMMENDER_H__
