/*
	Copyright (C) 2010, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GRecommender.h"
#include "GSparseMatrix.h"
#include "GCluster.h"
#include "GMatrix.h"
#include "GActivation.h"
#include "GHeap.h"
#include "GRand.h"
#include "GNeuralNet.h"
#include "GDistance.h"
#include "GVec.h"
#include <math.h>
#include <map>
#include <vector>
#include <cmath>

using std::multimap;
using std::vector;

namespace GClasses {


double GCollaborativeFilter::crossValidate(GSparseMatrix* pData, size_t folds, GRand* pRand, size_t maxRecommendationsPerRow, double* pOutMAE)
{
	if(pData->defaultValue() != UNKNOWN_REAL_VALUE)
		ThrowError("Expected the default value to be UNKNOWN_REAL_VALUE");
	
	// Randomly assign each rating to one of the folds
	size_t users = pData->rows();
	size_t ratings = 0;
	for(size_t i = 0; i < users; i++)
		ratings += pData->rowNonDefValues(i);
	size_t* pFolds = new size_t[ratings];
	for(size_t i = 0; i < ratings; i++)
		pFolds[i] = (size_t)pRand->next(folds);

	// Make a copy of the sparse data
	GSparseMatrix clone(pData->rows(), pData->cols(), UNKNOWN_REAL_VALUE);
	clone.copyFrom(pData);

	// Evaluate accuracy
	double sse = 0.0;
	double se = 0.0;
	size_t hits = 0;
	for(size_t i = 0; i < folds; i++)
	{
		// Make a data set with ratings in the current fold removed
		size_t* pF = pFolds;
		for(size_t y = 0; y < users; y++) // for each user...
		{
			vector<size_t> condemnedCols;
			condemnedCols.reserve(clone.rowNonDefValues(y));
			for(GSparseMatrix::Iter rating = clone.rowBegin(y); rating != clone.rowEnd(y); rating++) // for each item that this user has rated...
			{
				if(*pF == i)
					condemnedCols.push_back(rating->first);
				pF++;
			}
			for(vector<size_t>::iterator it = condemnedCols.begin(); it != condemnedCols.end(); it++)
				clone.set(y, *it, UNKNOWN_REAL_VALUE); // remove the rating
		}

		// Train it
		trainBatch(&clone);

		// Predict the ratings in the current fold
		pF = pFolds;
		multimap<double,double> priQ;
		for(size_t y = 0; y < users; y++)
		{
			// Find the best recommendations for this user
			priQ.clear();
			for(GSparseMatrix::Iter rating = pData->rowBegin(y); rating != pData->rowEnd(y); rating++) // for each item that this user has rated...
			{
				if(*pF == i)
				{
					double prediction = predict(y, rating->first);
					priQ.insert(std::pair<double,double>(prediction, rating->second)); // <predicted-value,target-value>
					if(priQ.size() > maxRecommendationsPerRow)
						priQ.erase(priQ.begin()); // drop the pair with the lowest prediction
					clone.set(y, rating->first, rating->second); // Restore the rating to the cloned set
				}
				pF++;
			}

			// Evaluate them
			for(multimap<double,double>::iterator it = priQ.begin(); it != priQ.end(); it++)
			{
				double err = it->second - it->first; // error = target - prediction
				se += std::abs(err);
				sse += (err * err);
				hits++;
			}
		}
	}

	if(pOutMAE)
		*pOutMAE = se / hits;
	return sse / hits;
}

double GCollaborativeFilter::transduce(GSparseMatrix& train, GSparseMatrix& test, double* pOutMAE)
{
	if(train.defaultValue() != UNKNOWN_REAL_VALUE)
		ThrowError("Expected the default value to be UNKNOWN_REAL_VALUE");
	if(test.defaultValue() != UNKNOWN_REAL_VALUE)
		ThrowError("Expected the default value to be UNKNOWN_REAL_VALUE");
	if(train.rows() < test.rows())
		train.newRows(test.rows() - train.rows());

	// Train it
	trainBatch(&train);

	// Predict the ratings in the current fold
	double sse = 0.0;
	double se = 0.0;
	size_t hits = 0;
	for(size_t y = 0; y < test.rows(); y++)
	{
		for(GSparseMatrix::Iter rating = test.rowBegin(y); rating != test.rowEnd(y); rating++)
		{
			double err = rating->second - predict(y, rating->first); // error = target - prediction
			se += std::abs(err);
			sse += (err * err);
			hits++;
		}
	}

	if(pOutMAE)
		*pOutMAE = se / hits;
	return sse / hits;
}

class TarPredComparator
{
public:
	TarPredComparator() {}

	bool operator() (const std::pair<double,double>& a, const std::pair<double,double>& b) const
	{
		return a.second > b.second;
	}
};

GMatrix* GCollaborativeFilter::precisionRecall(GSparseMatrix* pData, GRand* pRand, bool ideal)
{
	if(pData->defaultValue() != UNKNOWN_REAL_VALUE)
		ThrowError("Expected the default value to be UNKNOWN_REAL_VALUE");
	size_t users = pData->rows();
	size_t ratings = 0;
	for(size_t i = 0; i < users; i++)
		ratings += pData->rowNonDefValues(i);
	size_t halfRatings = ratings / 2;
	size_t* pFolds = new size_t[ratings];
	size_t f0 = ratings - halfRatings;
	size_t f1 = halfRatings;
	for(size_t i = 0; i < ratings; i++)
	{
		if(pRand->next(f0 + f1) < f0)
		{
			pFolds[i] = 0;
			f0--;
		}
		else
		{
			pFolds[i] = 1;
			f1--;
		}
	}

	// Make a vector of target values and corresponding predictions
	vector<std::pair<double,double> > tarPred;
	tarPred.reserve(halfRatings);
	if(ideal)
	{
		// Simulate perfect predictions for all of the ratings in fold 1
		size_t* pF = pFolds;
		for(size_t y = 0; y < users; y++)
		{
			for(GSparseMatrix::Iter rating = pData->rowBegin(y); rating != pData->rowEnd(y); rating++) // for each item that this user has rated...
			{
				if(*pF != 0)
					tarPred.push_back(std::make_pair(rating->second, rating->second));
				pF++;
			}
		}
	}
	else
	{
		// Clone the data
		GSparseMatrix clone(pData->rows(), pData->cols(), UNKNOWN_REAL_VALUE);
		clone.copyFrom(pData);

		// Train with the ratings in fold 0
		size_t* pF = pFolds;
		size_t n = 0;
		for(size_t y = 0; y < users; y++) // for each user...
		{
			vector<size_t> condemnedCols;
			condemnedCols.reserve(clone.rowNonDefValues(y));
			for(GSparseMatrix::Iter rating = clone.rowBegin(y); rating != clone.rowEnd(y); rating++) // for each item that this user has rated...
			{
				GAssert(n < ratings);
				n++;
				if(*pF != 0)
					condemnedCols.push_back(rating->first);
				pF++;
			}
			for(vector<size_t>::iterator it = condemnedCols.begin(); it != condemnedCols.end(); it++)
				clone.set(y, *it, UNKNOWN_REAL_VALUE); // remove the rating
		}
		trainBatch(&clone);
	
		// Predict the ratings in fold 1
		pF = pFolds;
		for(size_t y = 0; y < users; y++)
		{
			for(GSparseMatrix::Iter rating = pData->rowBegin(y); rating != pData->rowEnd(y); rating++) // for each item that this user has rated...
			{
				if(*pF != 0)
				{
					double prediction = predict(y, rating->first);
					if(prediction == UNKNOWN_REAL_VALUE)
						prediction = 0.0;
					tarPred.push_back(std::make_pair(rating->second, prediction));
				}
				pF++;
			}
		}
	}

	// Make precision-recall data
	TarPredComparator comp;
	std::sort(tarPred.begin(), tarPred.end(), comp);
	double totalRelevant = 0.0;
	double totalIrrelevant = 0.0;
	for(vector<std::pair<double,double> >::iterator it = tarPred.begin(); it != tarPred.end(); it++)
	{
		totalRelevant += it->first;
		totalIrrelevant += (1.0 - it->first); // Here we assume that all ratings range from 0 to 1.
	}
	double retrievedRelevant = 0.0;
	double retrievedIrrelevant = 0.0;
	GMatrix* pResults = new GMatrix(0, 3);
	for(vector<std::pair<double,double> >::iterator it = tarPred.begin(); it != tarPred.end(); it++)
	{
		retrievedRelevant += it->first;
		retrievedIrrelevant += (1.0 - it->first); // Here we assume that all ratings range from 0 to 1.
		double precision = retrievedRelevant / (retrievedRelevant + retrievedIrrelevant);
		double recall = retrievedRelevant / totalRelevant; // recall is the same as the truePositiveRate
		double falsePositiveRate = retrievedIrrelevant / totalIrrelevant;
		double* pRow = pResults->newRow();
		pRow[0] = recall;
		pRow[1] = precision;
		pRow[2] = falsePositiveRate;
	}
	return pResults;
}

// static
double GCollaborativeFilter::areaUnderCurve(GMatrix* pData)
{
	double a = 0.0;
	double b = 0.0;
	double prevX = 0.0;
	double prevY = 0.0;
	for(size_t i = 0; i < pData->rows(); i++)
	{
		double* pRow = pData->row(i);
		a += (pRow[2] - prevX) * pRow[0];
		b += (pRow[2] - prevX) * prevY;
		prevX = pRow[2];
		prevY = pRow[0];
	}
	a += 1.0 - prevX;
	b += (1.0 - prevX) * prevY;
	return 0.5 * (a + b);
}






GBaselineRecommender::GBaselineRecommender()
: GCollaborativeFilter(), m_pRatings(NULL)
{
}

// virtual
GBaselineRecommender::~GBaselineRecommender()
{
	delete[] m_pRatings;
}

// virtual
void GBaselineRecommender::trainBatch(GSparseMatrix* pData)
{
	delete[] m_pRatings;
	size_t items = pData->cols();
	m_pRatings = new double[items];
	size_t* pCounts = new size_t[items];
	ArrayHolder<size_t> hCounts(pCounts);
	for(size_t i = 0; i < items; i++)
	{
		pCounts[i] = 0;
		m_pRatings[i] = 0.0;
	}
	size_t users = pData->rows();
	for(size_t y = 0; y < users; y++)
	{
		for(GSparseMatrix::Iter rating = pData->rowBegin(y); rating != pData->rowEnd(y); rating++) // for each item that this user has rated...
		{
			m_pRatings[rating->first] *= ((double)pCounts[rating->first] / (pCounts[rating->first] + 1));
			m_pRatings[rating->first] += (rating->second / (pCounts[rating->first] + 1));
			pCounts[rating->first]++;
		}
	}
}

// virtual
double GBaselineRecommender::predict(size_t user, size_t item)
{
	if(!m_pRatings)
		ThrowError("This model has not been trained");
	return m_pRatings[item];
}







GInstanceRecommender::GInstanceRecommender(size_t neighbors)
: GCollaborativeFilter(), m_neighbors(neighbors), m_ownMetric(true), m_pData(NULL), m_pBaseline(NULL)
{
	m_pMetric = new GCosineSimilarity();
}

// virtual
GInstanceRecommender::~GInstanceRecommender()
{
	delete(m_pData);
	if(m_ownMetric)
		delete(m_pMetric);
	delete[] m_pBaseline;
}

void GInstanceRecommender::setMetric(GSparseSimilarity* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}

// virtual
void GInstanceRecommender::trainBatch(GSparseMatrix* pData)
{
	// Compute the baseline recommendations
	delete[] m_pBaseline;
	size_t items = pData->cols();
	m_pBaseline = new double[items];
	size_t* pCounts = new size_t[items];
	ArrayHolder<size_t> hCounts(pCounts);
	for(size_t i = 0; i < items; i++)
	{
		pCounts[i] = 0;
		m_pBaseline[i] = 0.0;
	}
	size_t users = pData->rows();
	for(size_t y = 0; y < users; y++)
	{
		for(GSparseMatrix::Iter rating = pData->rowBegin(y); rating != pData->rowEnd(y); rating++) // for each item that this user has rated...
		{
			m_pBaseline[rating->first] *= ((double)pCounts[rating->first] / (pCounts[rating->first] + 1));
			m_pBaseline[rating->first] += (rating->second / (pCounts[rating->first] + 1));
			pCounts[rating->first]++;
		}
	}

	// Store the data
	if(pData->defaultValue() != UNKNOWN_REAL_VALUE)
		ThrowError("Expected the default value to be UNKNOWN_REAL_VALUE");
	delete(m_pData);

	// copy the data
	m_pData = new GSparseMatrix(pData->rows(), pData->cols(), UNKNOWN_REAL_VALUE);
	m_pData->copyFrom(pData);
}

// virtual
double GInstanceRecommender::predict(size_t user, size_t item)
{
	if(!m_pData)
		ThrowError("This model has not been trained");
	multimap<double,size_t> depq; // double-ended priority-queue that maps from similarity to user-id
	for(size_t neigh = 0; neigh < m_pData->rows(); neigh++)
	{
		// Only consider other users that have rated this item
		if(neigh == user)
			continue;
		double rating = m_pData->get(neigh, item);
		if(rating == UNKNOWN_REAL_VALUE)
			continue;

		// Compute the similarity
		double similarity = m_pMetric->similarity(m_pData->row(user), m_pData->row(neigh));

		// If the queue is overfull, drop the worst item
		depq.insert(std::make_pair(similarity, neigh));
		if(depq.size() > m_neighbors)
			depq.erase(depq.begin());
	}

	// Combine the ratings of the nearest neighbors to make a prediction
	double sum = 0.0;
	double weighted_sum = 0.0;
	double sum_weight = 0.0;
	for(multimap<double,size_t>::iterator it = depq.begin(); it != depq.end(); it++)
	{
		double weight = std::max(0.0, std::min(1.0, it->first));
		double val = m_pData->get(it->second, item);
		sum += val;
		weighted_sum += weight * val;
		sum_weight += weight;
	}
	if(depq.size() > 0)
	{
		if(sum_weight > 0.0)
			return weighted_sum / sum_weight;
		else
			return m_pBaseline[item];
	}
	else
		return m_pBaseline[item];
}







GClusterRecommender::GClusterRecommender(size_t clusters, GRand* pRand)
: GCollaborativeFilter(), m_clusters(clusters), m_pPredictions(NULL), m_pClusterer(NULL), m_ownClusterer(false), m_pRand(pRand)
{
}

// virtual
GClusterRecommender::~GClusterRecommender()
{
	if(m_ownClusterer)
		delete(m_pClusterer);
	delete(m_pPredictions);
}

void GClusterRecommender::setClusterer(GSparseClusterer* pClusterer, bool own)
{
	if(pClusterer->clusterCount() != m_clusters)
		ThrowError("Mismatching number of clusters");
	if(m_ownClusterer)
		delete(m_pClusterer);
	m_pClusterer = pClusterer;
	m_ownClusterer = own;
}

// virtual
void GClusterRecommender::trainBatch(GSparseMatrix* pData)
{
	if(!m_pClusterer)
		setClusterer(new GKMeansSparse(m_clusters, m_pRand), true);

	// Cluster the data
	m_pClusterer->cluster(pData);

	// Gather the mean predictions in each cluster
	delete(m_pPredictions);
	m_pPredictions = new GMatrix(m_clusters, pData->cols());
	m_pPredictions->setAll(0.0);
	size_t* pCounts = new size_t[pData->cols() * m_clusters];
	ArrayHolder<size_t> hCounts(pCounts);
	memset(pCounts, '\0', sizeof(size_t) * pData->cols() * m_clusters);
	for(size_t i = 0; i < pData->rows(); i++)
	{
		size_t clust = m_pClusterer->whichCluster(i);
		double* pRow = m_pPredictions->row(clust);
		size_t* pRowCounts = pCounts + (pData->cols() * clust);
		for(GSparseMatrix::Iter it = pData->rowBegin(i); it != pData->rowEnd(i); it++)
		{
			pRow[it->first] *= ((double)pRowCounts[it->first] / (pRowCounts[it->first] + 1));
			pRow[it->first] += (it->second / (pRowCounts[it->first] + 1));
			pRowCounts[it->first]++;
		}
	}
}

// virtual
double GClusterRecommender::predict(size_t user, size_t item)
{
	size_t clust = m_pClusterer->whichCluster(user);
	double* pRow = m_pPredictions->row(clust);
	return pRow[item];
}






class Rating
{
public:
	size_t m_user;
	size_t m_item;
	double m_rating;
};

GMatrixFactorization::GMatrixFactorization(size_t intrinsicDims, GRand& rand)
: GCollaborativeFilter(), m_intrinsicDims(intrinsicDims), m_regularizer(0.01), m_pP(NULL), m_pQ(NULL), m_rand(rand)
{
}

// virtual
GMatrixFactorization::~GMatrixFactorization()
{
	delete(m_pQ);
	delete(m_pP);
}

// virtual
void GMatrixFactorization::trainBatch(GSparseMatrix* pData)
{
	// Make a single list of all the ratings
	GHeap heap(2048);
	vector<Rating*> ratings;
	for(size_t user = 0; user < pData->rows(); user++)
	{
		for(GSparseMatrix::Iter it = pData->rowBegin(user); it != pData->rowEnd(user); it++)
		{
			Rating* pRating = (Rating*)heap.allocAligned(sizeof(Rating));
			pRating->m_user = user;
			pRating->m_item = it->first;
			pRating->m_rating = it->second;
			ratings.push_back(pRating);
		}
	}

	// Initialize P with small random values, and Q with zeros
	delete(m_pP);
	m_pP = new GMatrix(pData->rows(),  1 + m_intrinsicDims);
	for(size_t i = 0; i < m_pP->rows(); i++)
	{
		double* pVec = m_pP->row(i);
		for(size_t j = 0; j <= m_intrinsicDims; j++)
			*(pVec++) = 0.02 * m_rand.normal();
	}
	delete(m_pQ);
	m_pQ = new GMatrix(pData->cols(), 1 + m_intrinsicDims);
	for(size_t i = 0; i < m_pQ->rows(); i++)
	{
		double* pVec = m_pQ->row(i);
		for(size_t j = 0; j <= m_intrinsicDims; j++)
			*(pVec++) = 0.02 * m_rand.normal();
	}

	// Train
	double prevErr = 1e308;
	double learningRate = 0.01;
	GTEMPBUF(double, temp_weights, m_intrinsicDims);
	while(learningRate >= 0.001)
	{
		// Shuffle the ratings
		for(size_t n = ratings.size(); n > 0; n--)
			std::swap(ratings[(size_t)m_rand.next(n)], ratings[n - 1]);

		// Do an epoch of training
		double sse = 0;
		for(vector<Rating*>::iterator it = ratings.begin(); it != ratings.end(); it++)
		{
			Rating* pRating = *it;

			// Estimate the prediction
			double* pPref = m_pP->row(pRating->m_user);
			double* pVec = m_pQ->row(pRating->m_item);
			double pred = *(pVec++) + *(pPref++);
			for(size_t i = 0; i < m_intrinsicDims; i++)
				pred += *(pPref++) * (*pVec++);

			// Update Q
			double err = pRating->m_rating - pred;
			sse += (err * err);
			pPref = m_pP->row(pRating->m_user) + 1;
			double* pT = temp_weights;
			pVec = m_pQ->row(pRating->m_item);
			*pVec += learningRate * err;
			pVec++;
			for(size_t i = 0; i < m_intrinsicDims; i++)
			{
				*(pT++) = *pVec;
				*pVec += learningRate * (err * (*pPref) - m_regularizer * (*pVec));
				pPref++;
				pVec++;
			}

			// Update P
			pVec = temp_weights;
			pPref = m_pP->row(pRating->m_user);
			*pPref += learningRate * err;
			pPref++;
			for(size_t i = 0; i < m_intrinsicDims; i++)
			{
				*pPref += learningRate * (err * (*pVec) - m_regularizer * (*pPref));
				pVec++;
				pPref++;
			}
		}

		// Stopping criteria
		double rsse = sqrt(sse);
		if(rsse < 1e-12 || 1.0 - (rsse / prevErr) < 0.001) // If the amount of improvement is less than 0.01%
			learningRate *= 0.8; // decay the learning rate
		prevErr = rsse;
	}
}

// virtual
double GMatrixFactorization::predict(size_t user, size_t item)
{
	double* pVec = m_pQ->row(item);
	double* pPref = m_pP->row(user);
	double pred = *(pVec++) + *(pPref++);
	for(size_t i = 0; i < m_intrinsicDims; i++)
		pred += *(pPref++) * (*pVec++);
//pred = std::max(1.0, std::min(5.0, floor(pred + 0.5))); // todo: formalize this
	return pred;
}








GNeuralRecommender::GNeuralRecommender(size_t intrinsicDims, GRand* pRand)
: GCollaborativeFilter(), m_intrinsicDims(intrinsicDims), m_pRand(pRand)
{
	m_pModel = new GNeuralNet(m_pRand);
	m_pUsers = new GMatrix(0, intrinsicDims);
}

// virtual
GNeuralRecommender::~GNeuralRecommender()
{
	delete(m_pModel);
	delete(m_pUsers);
}

// virtual
void GNeuralRecommender::trainBatch(GSparseMatrix* pData)
{
	if(pData->defaultValue() != UNKNOWN_REAL_VALUE)
		ThrowError("Expected the default value to be UNKNOWN_REAL_VALUE");

	// Initialize the user preference vectors
	m_pUsers->flush();
	m_pUsers->newRows(pData->rows());

	// Prep the model for incremental training
	sp_relation pFeatureRel = new GUniformRelation(m_intrinsicDims);
	sp_relation pLabelRel = new GUniformRelation(pData->cols());
	m_pModel->setUseInputBias(true);
	m_pModel->enableIncrementalLearning(pFeatureRel, pLabelRel);
	GActivationFunction* pAF = m_pModel->layer(0).m_pActivationFunction;
	m_pUsers->setAll(pAF->center());

	// Make a single list of all the ratings
	m_min = 1e200;
	m_max = -1e200;
	GHeap heap(2048);
	vector<Rating*> ratings;
	for(size_t user = 0; user < pData->rows(); user++)
	{
		for(GSparseMatrix::Iter it = pData->rowBegin(user); it != pData->rowEnd(user); it++)
		{
			Rating* pRating = (Rating*)heap.allocAligned(sizeof(Rating));
			pRating->m_user = user;
			pRating->m_item = it->first;
			pRating->m_rating = it->second;
			m_min = std::min(m_min, it->second);
			m_max = std::max(m_max, it->second);
			ratings.push_back(pRating);
		}
	}
	double scale = 1.0 / (m_max - m_min);
	for(vector<Rating*>::iterator it = ratings.begin(); it != ratings.end(); it++)
		(*it)->m_rating = ((*it)->m_rating - m_min) * scale;

	double prevErr = 1e308;
	double floor = std::max(-50.0, pAF->center() - pAF->halfRange());
	double cap = std::min(50.0, pAF->center() + pAF->halfRange());
	double learningRate = 0.2;
	while(learningRate >= 0.01)
	{
		// Shuffle the ratings
		for(size_t n = ratings.size(); n > 0; n--)
			std::swap(ratings[(size_t)m_pRand->next(n)], ratings[n - 1]);

		// Do an epoch of training
		m_pModel->setLearningRate(learningRate);
		double sse = 0;
		for(vector<Rating*>::iterator it = ratings.begin(); it != ratings.end(); it++)
		{
			Rating* pRating = *it;
			double* pUserPreferenceVector = m_pUsers->row(pRating->m_user);
			double predictedRating = m_pModel->forwardPropSingleOutput(pUserPreferenceVector, pRating->m_item);
			double d = pRating->m_rating - predictedRating;
			sse += (d * d);
			m_pModel->setErrorSingleOutput(pRating->m_rating, pRating->m_item, m_pModel->backPropTargetFunction());
			m_pModel->backProp()->backpropagateSingleOutput(pRating->m_item);
			m_pModel->backProp()->descendGradientSingleOutput(pRating->m_item, pUserPreferenceVector, learningRate, m_pModel->momentum(), m_pModel->useInputBias());
			m_pModel->backProp()->adjustFeaturesSingleOutput(pRating->m_item, pUserPreferenceVector, learningRate, m_pModel->useInputBias());
			GVec::floorValues(pUserPreferenceVector, floor, m_intrinsicDims);
			GVec::capValues(pUserPreferenceVector, cap, m_intrinsicDims);
		}

		// Stopping criteria
		double rsse = sqrt(sse);
		if(rsse < 1e-12 || 1.0 - (rsse / prevErr) < 0.0001) // If the amount of improvement is less than 0.01%
			learningRate *= 0.8; // decay the learning rate
		prevErr = rsse;
	}
}

// virtual
double GNeuralRecommender::predict(size_t user, size_t item)
{
	return (m_max - m_min) * m_pModel->forwardPropSingleOutput(m_pUsers->row(user), item) + m_min;
}










GBagOfRecommenders::GBagOfRecommenders(GRand& rand)
: GCollaborativeFilter(), m_rand(rand)
{
}

GBagOfRecommenders::~GBagOfRecommenders()
{
	clear();
}

void GBagOfRecommenders::clear()
{
	for(vector<GCollaborativeFilter*>::iterator it = m_filters.begin(); it != m_filters.end(); it++)
		delete(*it);
	m_filters.clear();
}

void GBagOfRecommenders::addRecommender(GCollaborativeFilter* pRecommender)
{
	m_filters.push_back(pRecommender);
}

// virtual
void GBagOfRecommenders::trainBatch(GSparseMatrix* pData)
{
	for(vector<GCollaborativeFilter*>::iterator it = m_filters.begin(); it != m_filters.end(); it++)
	{
		// Make a matrix that randomly samples about half of the elements in pData
		GSparseMatrix tmp(pData->rows(), pData->cols(), pData->defaultValue());
		for(size_t i = 0; i < pData->rows(); i++)
		{
			GSparseMatrix::Iter end2 = pData->rowEnd(i);
			for(GSparseMatrix::Iter it2 = pData->rowBegin(i); it2 != end2; it2++)
			{
				if(m_rand.next(2) == 0)
					tmp.set(i, it2->first, it2->second);
			}
		}

		// Train with it
		(*it)->trainBatch(&tmp);
	}
}

// virtual
double GBagOfRecommenders::predict(size_t user, size_t item)
{
	double sum = 0.0;
	for(vector<GCollaborativeFilter*>::iterator it = m_filters.begin(); it != m_filters.end(); it++)
		sum += (*it)->predict(user, item);
	return sum / m_filters.size();
}







} // namespace GClasses
