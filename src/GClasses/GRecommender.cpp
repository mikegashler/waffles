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

#include "GRecommender.h"
#include "GRecommenderLib.h"
#include "GSparseMatrix.h"
#include "GCluster.h"
#include "GMatrix.h"
#include "GHeap.h"
#include "GRand.h"
#include "GNeuralNet.h"
#include "GDistance.h"
#include <math.h>
#include <map>
#include <vector>
#include <cmath>
#include "GDom.h"
#include "GTime.h"
#include "GHolders.h"
#include "GApp.h"
#include "GLearner.h"
#include "GLearnerLib.h"
#include "usage.h"
#include <memory>

using std::map;
using std::multimap;
using std::pair;
using std::set;
using std::vector;

namespace GClasses {

void GCollaborativeFilter_dims(GMatrix& data, size_t* pOutUsers, size_t* pOutItems)
{
	double m = data.columnMin(0);
	double r = data.columnMax(0);
	if(m < 0)
		throw Ex("col 0 (user) indexes out of range");
	*pOutUsers = size_t(ceil(r)) + 1;
	m = data.columnMin(1);
	r = data.columnMax(1);
	if(m < 0)
		throw Ex("col 1 (item) indexes out of range");
	*pOutItems = size_t(ceil(r)) + 1;
	if(data.rows() * 8 < *pOutUsers)
		throw Ex("col 0 (user) indexes out of range");
	if(data.rows() * 8 < *pOutItems)
		throw Ex("col 1 (item) indexes out of range");
}

GCollaborativeFilter::GCollaborativeFilter()
: m_rand(0)
{
}

GCollaborativeFilter::GCollaborativeFilter(const GDomNode* pNode, GLearnerLoader& ll)
: m_rand(0)
{
}

void GCollaborativeFilter::trainDenseMatrix(const GMatrix& data, const GMatrix* pLabels)
{
	if(!data.relation().areContinuous())
		throw Ex("GCollaborativeFilter::trainDenseMatrix only supports continuous attributes.");

	// Convert to 3-column form
	GMatrix* pMatrix = new GMatrix(0, 3);
	std::unique_ptr<GMatrix> hMatrix(pMatrix);
	size_t dims = data.cols();
	for(size_t i = 0; i < data.rows(); i++)
	{
		const GVec& row = data.row(i);
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

	if(pLabels)
	{
		size_t labelDims = pLabels->cols();
		for(size_t i = 0; i < pLabels->rows(); i++)
		{
			const GVec& row = pLabels->row(i);
			for(size_t j = 0; j < labelDims; j++)
			{
				if(row[j] != UNKNOWN_REAL_VALUE)
				{
					GVec& vec = pMatrix->newRow();
					vec[0] = (double)i;
					vec[1] = (double)(dims + j);
					vec[2] = row[j];
				}
			}
		}
	}

	// Train
	train(*pMatrix);
}

GDomNode* GCollaborativeFilter::baseDomNode(GDom* pDoc, const char* szClassName) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "class", szClassName);
	return pNode;
}

double GCollaborativeFilter::crossValidate(GMatrix& data, size_t folds, double* pOutMAE)
{
	// Randomly assign each rating to one of the folds
	size_t ratings = data.rows();
	size_t* pFolds = new size_t[ratings];
	std::unique_ptr<size_t[]> hFolds(pFolds);
	for(size_t i = 0; i < ratings; i++)
		pFolds[i] = (size_t)m_rand.next(folds);

	// Evaluate accuracy
	double ssse = 0.0;
	double smae = 0.0;
	for(size_t i = 0; i < folds; i++)
	{
		// Split the data
		GMatrix dataTrain(data.relation().clone());
		GReleaseDataHolder hDataTrain(&dataTrain);
		GMatrix dataTest(data.relation().clone());
		GReleaseDataHolder hDataTest(&dataTest);
		size_t* pF = pFolds;
		for(size_t j = 0; j < data.rows(); j++)
		{
			if(*pF == i)
				dataTest.takeRow(&data[j]);
			else
				dataTrain.takeRow(&data[j]);
			pF++;
		}

		double mae;
		ssse += trainAndTest(dataTrain, dataTest, &mae);
		smae += mae;
	}

	if(pOutMAE)
		*pOutMAE = smae / folds;
	return ssse / folds;
}

double GCollaborativeFilter::trainAndTest(GMatrix& dataTrain, GMatrix& dataTest, double* pOutMAE)
{
	train(dataTrain);
	double sse = 0.0;
	double se = 0.0;
	size_t hits = 0;
	for(size_t j = 0; j < dataTest.rows(); j++)
	{
		GVec& vec = dataTest[j];
		double prediction = predict(size_t(vec[0]), size_t(vec[1]));
		if (prediction < -1e100 || prediction > 1e100)
		{
			throw Ex("Unreasonable prediction");
		}
		double err = vec[2] - prediction;
		se += std::abs(err);
		sse += (err * err);
		hits++;
	}
	if(pOutMAE)
		*pOutMAE = se / dataTest.rows();
	return sse / dataTest.rows();
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

GMatrix* GCollaborativeFilter::precisionRecall(GMatrix& data, bool ideal)
{
	// Divide into two equal-size folds
	size_t ratings = data.rows();
	size_t halfRatings = ratings / 2;
	size_t* pFolds = new size_t[ratings];
	size_t f0 = ratings - halfRatings;
	size_t f1 = halfRatings;
	for(size_t i = 0; i < ratings; i++)
	{
		if(m_rand.next(f0 + f1) < f0)
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

	// Split the data
	GMatrix dataTrain(data.relation().clone());
	GReleaseDataHolder hDataTrain(&dataTrain);
	GMatrix dataTest(data.relation().clone());
	GReleaseDataHolder hDataTest(&dataTest);
	size_t* pF = pFolds;
	for(size_t j = 0; j < data.rows(); j++)
	{
		if(*pF == 0)
			dataTrain.takeRow(&data[j]);
		else
			dataTest.takeRow(&data[j]);
		pF++;
	}

	if(ideal)
	{
		// Simulate perfect predictions
		for(size_t i = 0; i < dataTest.rows(); i++)
		{
			GVec& vec = dataTest[i];
			tarPred.push_back(std::make_pair(vec[2], vec[2]));
		}
	}
	else
	{
		// Train
		train(dataTrain);

		// Predict the ratings in the test data
		for(size_t i = 0; i < dataTest.rows(); i++)
		{
			GVec& vec = dataTest[i];
			double prediction = predict(size_t(vec[0]), size_t(vec[1]));
			GAssert(prediction != UNKNOWN_REAL_VALUE);
			tarPred.push_back(std::make_pair(vec[2], prediction));
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
		GVec& row = pResults->newRow();
		row[0] = recall;
		row[1] = precision;
		row[2] = falsePositiveRate;
	}
	return pResults;
}

// static
double GCollaborativeFilter::areaUnderCurve(GMatrix& data)
{
	double a = 0.0;
	double b = 0.0;
	double prevX = 0.0;
	double prevY = 0.0;
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& row = data[i];
		a += (row[2] - prevX) * row[0];
		b += (row[2] - prevX) * prevY;
		prevX = row[2];
		prevY = row[0];
	}
	a += 1.0 - prevX;
	b += (1.0 - prevX) * prevY;
	return 0.5 * (a + b);
}

void GCF_basicTest_makeData(GMatrix& m, GRand& rand)
{
	// Generate perfectly linear ratings based on random preferences
	// with both item and user bias
	for(size_t i = 0; i < 300; i++)
	{
		double a = rand.uniform();
		double b = rand.normal();
		double c = rand.uniform();
		double userBias = rand.normal();
		GVec& vec1 = m.newRow();
		vec1[0] = (double)i; // user
		vec1[1] = 0; // item
		vec1[2] = a + 0.0 + 0.2 * c + userBias; // rating
		GVec& vec2 = m.newRow();
		vec2[0] = (double)i; // user
		vec2[1] = 1; // item
		vec2[2] = 0.2 * a + 0.2 * b + c * c + 0.2 + userBias; // rating
		GVec& vec3 = m.newRow();
		vec3[0] = (double)i; // user
		vec3[1] = 2; // item
		vec3[2] = 0.6 * a + 0.1 * b + 0.2 * c * c * c - 0.3 + userBias; // rating
		GVec& vec4 = m.newRow();
		vec4[0] = (double)i; // user
		vec4[1] = 3; // item
		vec4[2] = 0.5 * a + 0.5 * b - 0.5 * c + 0.0 + userBias; // rating
		GVec& vec5 = m.newRow();
		vec5[0] = (double)i; // user
		vec5[1] = 4; // item
		vec5[2] = -0.2 * a + 0.4 * b - 0.3 * sin(c) + 0.1 + userBias; // rating
	}
}

void GCollaborativeFilter::basicTest(double maxMSE)
{
	GRand rnd(0);
	GMatrix m(0, 3);
	GCF_basicTest_makeData(m, rnd);
	double mse = crossValidate(m, 2);
	if(mse > maxMSE)
		throw Ex("Failed. Expected MSE=", to_str(maxMSE), ". Actual MSE=", to_str(mse), ".");
	else if(mse + 0.085 < maxMSE)
		std::cerr << "\nTest needs to be tightened. MSE: " << mse << ", maxMSE: " << maxMSE << "\n";
}





GBaselineRecommender::GBaselineRecommender()
: GCollaborativeFilter(), m_items(0)
{
}

GBaselineRecommender::GBaselineRecommender(const GDomNode* pNode, GLearnerLoader& ll)
: GCollaborativeFilter(pNode, ll)
{
	m_ratings.deserialize(pNode->get("ratings"));
}

// virtual
GBaselineRecommender::~GBaselineRecommender()
{
}

// virtual
void GBaselineRecommender::train(GMatrix& data)
{
	// Determine the sizes
	if(data.cols() != 3)
		throw Ex("Expected 3 cols");
//	double m = data.columnMin(1);
	double r = data.columnMax(1);
	m_items = size_t(ceil(r)) + 1;
	if(data.rows() * 8 < m_items)
		throw Ex("column 1 (item) indexes out of range");

	// Allocate space
	m_ratings.resize(m_items);
	size_t* pCounts = new size_t[m_items];
	std::unique_ptr<size_t[]> hCounts(pCounts);
	size_t* pC = pCounts;
	GVec& rr = m_ratings;
	for(size_t i = 0; i < m_items; i++)
	{
		pC[i] = 0;
		rr[i] = 0.0;
	}
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& vec = data[i];
		size_t c = size_t(vec[1]);
		rr[c] *= ((double)pCounts[c] / (pCounts[c] + 1));
		rr[c] += (vec[2] / (pCounts[c] + 1));
		pCounts[c]++;
	}
}

// virtual
double GBaselineRecommender::predict(size_t user, size_t item)
{
	if(item >= m_items)
		return 0.0;
	return m_ratings[item];
}

// virtual
void GBaselineRecommender::impute(GVec& vec, size_t dims)
{
	size_t n = std::min(dims, m_items);
	size_t i;
	for(i = 0; i < n; i++)
	{
		if(vec[i] == UNKNOWN_REAL_VALUE)
			vec[i] = m_ratings[i];
	}
	for( ; i < dims; i++)
	{
		if(vec[i] == UNKNOWN_REAL_VALUE)
			vec[i] = 0.0;
	}
}

// virtual
GDomNode* GBaselineRecommender::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBaselineRecommender");
	pNode->add(pDoc, "ratings", m_ratings.serialize(pDoc));
	return pNode;
}

// static
void GBaselineRecommender::test()
{
	GBaselineRecommender rec;
	rec.basicTest(1.16);
}







GInstanceRecommender::GInstanceRecommender(size_t neighbors)
: GCollaborativeFilter(), m_neighbors(neighbors), m_ownMetric(true), m_pData(NULL), m_pBaseline(NULL), m_significanceWeight(0)
{
	m_pMetric = new GCosineSimilarity();
}

GInstanceRecommender::GInstanceRecommender(const GDomNode* pNode, GLearnerLoader& ll)
: GCollaborativeFilter(pNode, ll)
{
	m_neighbors = (size_t)pNode->getInt("neighbors");
	m_pMetric = GSparseSimilarity::deserialize(pNode->get("metric"));
	m_ownMetric = true;
	m_pData = new GSparseMatrix(pNode->get("data"));
	m_pBaseline = new GBaselineRecommender(pNode->get("bl"), ll);
	m_significanceWeight = (size_t)pNode->getInt("sigWeight");
}

// virtual
GInstanceRecommender::~GInstanceRecommender()
{
	delete(m_pData);
	if(m_ownMetric)
		delete(m_pMetric);
	delete(m_pBaseline);
}

void GInstanceRecommender::setMetric(GSparseSimilarity* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}

// virtual
void GInstanceRecommender::train(GMatrix& data)
{
	if(data.cols() != 3)
		throw Ex("Expected 3 cols");

	// Compute the baseline recommendations
	delete(m_pBaseline);
	m_pBaseline = new GBaselineRecommender();
	m_pBaseline->train(data);

	// Store the data
	size_t users, items;
	GCollaborativeFilter_dims(data, &users, &items);
	delete(m_pData);
	m_pData = new GSparseMatrix(users, items, UNKNOWN_REAL_VALUE);
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& vec = data[i];
		m_pData->set(size_t(vec[0]), size_t(vec[1]), vec[2]);
	}
}

// virtual
double GInstanceRecommender::predict(size_t user, size_t item)
{
		if(!m_pData)
				throw Ex("This model has not been trained");
		if(user >= m_pData->rows() || item >= m_pData->cols())
				return 0.0;

		// Find the k-nearest neighbors
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
				size_t count = GSparseVec::count_matching_elements(m_pData->row(user), m_pData->row(neigh));
				double similarity = m_pMetric->similarity(m_pData->row(user), m_pData->row(neigh));

				if(count < m_significanceWeight)
						similarity *= count / m_significanceWeight;

				// If the queue is overfull, drop the worst item
				depq.insert(std::make_pair(similarity, neigh));
				if(depq.size() > m_neighbors)
						depq.erase(depq.begin());
		}

		// Combine the ratings of the nearest neighbors to make a prediction
		double weighted_sum = 0.0;
		double sum_weight = 0.0;
		for(multimap<double,size_t>::iterator it = depq.begin(); it != depq.end(); it++)
		{
				double weight = std::max(0.0, std::min(1.0, it->first));
				double val = m_pData->get(it->second, item);
				weighted_sum += weight * val;
				sum_weight += weight;
		}
		if(sum_weight > 0.0)
				return weighted_sum / sum_weight;
		else
				return m_pBaseline->predict(user, item);
}

multimap<double,ArrayWrapper> GInstanceRecommender::getNeighbors(size_t user, size_t item)
{
	if(!m_pData)
		throw Ex("This model has not been trained");
	if(user >= m_pData->rows() || item >= m_pData->cols())
		throw Ex("User and/or item not in the provided data set");

	// Find the k-nearest neighbors
		if(m_user_depq.find(user) == m_user_depq.end())
		{
		multimap<double,ArrayWrapper> depq; // double-ended priority-queue that maps from similarity to user-id
		for(size_t neigh = 0; neigh < m_pData->rows(); neigh++)
		{
			// Only consider other users that have rated this item
			if(neigh == user)
				continue;
			double rating = m_pData->get(neigh, item);
			if(rating == UNKNOWN_REAL_VALUE)
				continue;

			// Compute the similarity
			size_t count = GSparseVec::count_matching_elements(m_pData->row(user), m_pData->row(neigh));
			double similarity = m_pMetric->similarity(m_pData->row(user), m_pData->row(neigh));

			if(count < m_significanceWeight)
				similarity *= count / m_significanceWeight;

			// If the queue is overfull, drop the worst item
			ArrayWrapper temp = {{neigh, count}};
			depq.insert(std::make_pair(similarity, temp));
			if(depq.size() > m_neighbors)
				depq.erase(depq.begin());
		}
				m_user_depq[user] = depq;
		}

	return m_user_depq[user];
}

// virtual
void GInstanceRecommender::impute(GVec& vec, size_t dims)
{
	if(!m_pData)
		throw Ex("This model has not been trained");
	if(dims != m_pData->cols())
		throw Ex("The vector has a different size than this model was trained with");

	// Find the k-nearest neighbors
	multimap<double,size_t> depq; // double-ended priority-queue that maps from similarity to user-id
	for(size_t neigh = 0; neigh < m_pData->rows(); neigh++)
	{
		// Compute the similarity
		size_t count = vec.size();
		double similarity = m_pMetric->similarity(m_pData->row(neigh), vec);

		if(count < m_significanceWeight)
			similarity *= count / m_significanceWeight;

		// If the queue is overfull, drop the worst item
		depq.insert(std::make_pair(similarity, neigh));
		if(depq.size() > m_neighbors)
			depq.erase(depq.begin());
	}

	// Impute missing values by combining the ratings from the neighbors
	for(size_t i = 0; i < m_pData->cols(); i++)
	{
		if(vec[i] == UNKNOWN_REAL_VALUE)
		{
			double weighted_sum = 0.0;
			double sum_weight = 0.0;
			for(multimap<double,size_t>::iterator it = depq.begin(); it != depq.end(); it++)
			{
				double val = m_pData->get(it->second, i);
				if(val != UNKNOWN_REAL_VALUE)
				{
					double weight = std::max(0.0, std::min(1.0, it->first));
					weighted_sum += weight * val;
					sum_weight += weight;
				}
			}
			if(sum_weight > 0.0)
				vec[i] = weighted_sum / sum_weight;
			else
				vec[i] = m_pBaseline->predict(0, i); // baseline ignores the user
		}
	}
}

// virtual
GDomNode* GInstanceRecommender::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GInstanceRecommender");
	pNode->add(pDoc, "neighbors", m_neighbors);
	pNode->add(pDoc, "metric", m_pMetric->serialize(pDoc));
	pNode->add(pDoc, "data", m_pData->serialize(pDoc));
	pNode->add(pDoc, "bl", m_pBaseline->serialize(pDoc));
	pNode->add(pDoc, "sigWeight", m_significanceWeight);
	return pNode;
}

double GInstanceRecommender::getRating(size_t user, size_t item)
{
	return m_pData->get(user, item);
}

// static
void GInstanceRecommender::test()
{
	GInstanceRecommender rec(8);
	rec.basicTest(0.63);
}





GSparseClusterRecommender::GSparseClusterRecommender(size_t clusters)
: GCollaborativeFilter(), m_clusters(clusters), m_pPredictions(NULL), m_pClusterer(NULL), m_ownClusterer(false), m_users(0), m_items(0)
{
}

// virtual
GSparseClusterRecommender::~GSparseClusterRecommender()
{
	if(m_ownClusterer)
		delete(m_pClusterer);
	delete(m_pPredictions);
}

void GSparseClusterRecommender::setClusterer(GSparseClusterer* pClusterer, bool own)
{
	if(pClusterer->clusterCount() != m_clusters)
		throw Ex("Mismatching number of clusters");
	if(m_ownClusterer)
		delete(m_pClusterer);
	m_pClusterer = pClusterer;
	m_ownClusterer = own;
}

// virtual
void GSparseClusterRecommender::train(GMatrix& data)
{
	if(data.cols() != 3)
		throw Ex("Expected 3 cols");

	// Convert the data to a sparse matrix
	size_t users, items;
	GCollaborativeFilter_dims(data, &users, &items);
	m_users = users;
	m_items = items;
	GSparseMatrix sm(users, items, UNKNOWN_REAL_VALUE);
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& vec = data.row(i);
		sm.set(size_t(vec[0]), size_t(vec[1]), vec[2]);
	}

	// Make sure we have a clusterer
	if(!m_pClusterer)
		setClusterer(new GKMeansSparse(m_clusters, &m_rand), true);

	// Cluster the data
	m_pClusterer->cluster(&sm);

	// Gather the mean predictions in each cluster
	delete(m_pPredictions);
	m_pPredictions = new GMatrix(m_clusters, sm.cols());
	m_pPredictions->fill(0.0);
	size_t* pCounts = new size_t[sm.cols() * m_clusters];
	std::unique_ptr<size_t[]> hCounts(pCounts);
	memset(pCounts, '\0', sizeof(size_t) * sm.cols() * m_clusters);
	for(size_t i = 0; i < sm.rows(); i++)
	{
		size_t clust = m_pClusterer->whichCluster(i);
		GVec& row = m_pPredictions->row(clust);
		size_t* pRowCounts = pCounts + (sm.cols() * clust);
		for(GSparseMatrix::Iter it = sm.rowBegin(i); it != sm.rowEnd(i); it++)
		{
			row[it->first] *= ((double)pRowCounts[it->first] / (pRowCounts[it->first] + 1));
			row[it->first] += (it->second / (pRowCounts[it->first] + 1));
			pRowCounts[it->first]++;
		}
	}
}

// virtual
double GSparseClusterRecommender::predict(size_t user, size_t item)
{
	size_t clust = m_pClusterer->whichCluster(user);
	GVec& row = m_pPredictions->row(clust);
	return row[item];
}

// virtual
void GSparseClusterRecommender::impute(GVec& vec, size_t dims)
{
	throw Ex("Sorry, GSparseClusterRecommender::impute is not yet implemented");
	// todo: Find the closest centroid, and use it to impute all values
}

// virtual
GDomNode* GSparseClusterRecommender::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GSparseClusterRecommender");
	throw Ex("Sorry, this method has not been implemented yet");
	return pNode;
}

// static
void GSparseClusterRecommender::test()
{
	GSparseClusterRecommender rec(6);
	rec.basicTest(1.31);
}














GDenseClusterRecommender::GDenseClusterRecommender(size_t clusters)
: GCollaborativeFilter(), m_clusters(clusters), m_pPredictions(NULL), m_pClusterer(NULL), m_ownClusterer(false), m_users(0), m_items(0)
{
}

// virtual
GDenseClusterRecommender::~GDenseClusterRecommender()
{
	if(m_ownClusterer)
		delete(m_pClusterer);
	delete(m_pPredictions);
}

void GDenseClusterRecommender::setClusterer(GClusterer* pClusterer, bool own)
{
	if(pClusterer->clusterCount() != m_clusters)
		throw Ex("Mismatching number of clusters");
	if(m_ownClusterer)
		delete(m_pClusterer);
	m_pClusterer = pClusterer;
	m_ownClusterer = own;
}

void GDenseClusterRecommender::setFuzzifier(double d)
{
	if(!m_pClusterer)
		setClusterer(new GFuzzyKMeans(m_clusters, &m_rand), true);
	((GFuzzyKMeans*)m_pClusterer)->setFuzzifier(d);
}

// virtual
void GDenseClusterRecommender::train(GMatrix& data)
{
	if(data.cols() != 3)
		throw Ex("Expected 3 cols");

	if(!m_pClusterer)
		setClusterer(new GFuzzyKMeans(m_clusters, &m_rand), true);

	// Cluster the data
	size_t users, items;
	GCollaborativeFilter_dims(data, &users, &items);
	m_users = users;
	m_items = items;
	{
		GMatrix dense(users, items);
		for(size_t i = 0; i < data.rows(); i++)
		{
			GVec& vec = data.row(i);
			dense[size_t(vec[0])][size_t(vec[1])] = vec[2];
		}
		m_pClusterer->cluster(&dense);
	}

	// Gather the mean predictions in each cluster
	delete(m_pPredictions);
	m_pPredictions = new GMatrix(m_clusters, items);
	m_pPredictions->fill(0.0);
	size_t* pCounts = new size_t[items * m_clusters];
	std::unique_ptr<size_t[]> hCounts(pCounts);
	memset(pCounts, '\0', sizeof(size_t) * items * m_clusters);
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& vec = data.row(i);
		size_t user = size_t(vec[0]);
		size_t item = size_t(vec[1]);
		size_t clust = m_pClusterer->whichCluster(user);
		GVec& row = m_pPredictions->row(clust);
		size_t* pRowCounts = pCounts + (items * clust);
		row[item] *= ((double)pRowCounts[item] / (pRowCounts[item] + 1));
		row[item] += (vec[2] / (pRowCounts[item] + 1));
		pRowCounts[item]++;
	}
}

// virtual
double GDenseClusterRecommender::predict(size_t user, size_t item)
{
	if(user >= m_users || item >= m_items)
		return 0.0;
	size_t clust = m_pClusterer->whichCluster(user);
	GVec& row = m_pPredictions->row(clust);
	return row[item];
}

// virtual
void GDenseClusterRecommender::impute(GVec& vec, size_t dims)
{
	throw Ex("Sorry, GDenseClusterRecommender::impute is not yet implemented");
	// todo: Find the closest centroid, and use it to impute all values
}

// virtual
GDomNode* GDenseClusterRecommender::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GDenseClusterRecommender");
	throw Ex("Sorry, this method has not been implemented yet");
	return pNode;
}

// static
void GDenseClusterRecommender::test()
{
	GDenseClusterRecommender rec(6);
	rec.basicTest(0.0);
}









GMatrixFactorization::GMatrixFactorization(size_t intrinsicDims)
: GCollaborativeFilter(), m_intrinsicDims(intrinsicDims), m_regularizer(0.01), m_pP(NULL), m_pQ(NULL), m_pPMask(NULL), m_pQMask(NULL), m_pPWeights(NULL), m_pQWeights(NULL), m_nonNeg(false), m_minIters(1), m_decayRate(0.97)
{
}

GMatrixFactorization::GMatrixFactorization(const GDomNode* pNode, GLearnerLoader& ll)
: GCollaborativeFilter(pNode, ll)
{
	m_regularizer = pNode->getDouble("reg");
	m_minIters = (size_t)pNode->getInt("mi");
	m_decayRate = pNode->getDouble("dr");
	m_pP = new GMatrix(pNode->get("p"));
	m_pQ = new GMatrix(pNode->get("q"));
	GDomNode* pPMask = pNode->getIfExists("pm");
	if(pPMask)
		m_pPMask = new GMatrix(pPMask);
	else
		m_pPMask = NULL;
	GDomNode* pPWeights = pNode->getIfExists("pw");
	if(pPWeights)
		m_pPWeights = new GMatrix(pPWeights);
	else
		m_pPWeights = NULL;
	GDomNode* pQMask = pNode->getIfExists("qm");
	if(pQMask)
		m_pQMask = new GMatrix(pQMask);
	else
		m_pQMask = NULL;
	GDomNode* pQWeights = pNode->getIfExists("qw");
	if(pQWeights)
		m_pQWeights = new GMatrix(pQWeights);
	else
		m_pQWeights = NULL;
	if(m_pP->cols() != m_pQ->cols())
		throw Ex("Mismatching matrix sizes");
	m_intrinsicDims = m_pP->cols() - 1;
}

// virtual
GMatrixFactorization::~GMatrixFactorization()
{
	delete(m_pQ);
	delete(m_pP);
	delete(m_pPMask);
	delete(m_pQMask);
	delete(m_pPWeights);
	delete(m_pQWeights);
}

// virtual
GDomNode* GMatrixFactorization::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GMatrixFactorization");
	pNode->add(pDoc, "reg", m_regularizer);
	pNode->add(pDoc, "mi", m_minIters);
	pNode->add(pDoc, "dr", m_decayRate);
	pNode->add(pDoc, "p", m_pP->serialize(pDoc));
	pNode->add(pDoc, "q", m_pQ->serialize(pDoc));
	if(m_pPMask)
	{
		pNode->add(pDoc, "pm", m_pPMask->serialize(pDoc));
		pNode->add(pDoc, "pw", m_pPWeights->serialize(pDoc));
	}
	if(m_pQMask)
	{
		pNode->add(pDoc, "qm", m_pQMask->serialize(pDoc));
		pNode->add(pDoc, "qw", m_pQWeights->serialize(pDoc));
	}
	return pNode;
}

void GMatrixFactorization::clampUserElement(size_t user, size_t attr, double val)
{
	if(attr >= m_intrinsicDims)
		throw Ex("out of range");
	if(!m_pPMask)
	{
		m_pPMask = new GMatrix(0, m_intrinsicDims);
		m_pPWeights = new GMatrix(2, m_intrinsicDims);
		m_pPWeights->fill(0.0);
	}
	while(m_pPMask->rows() <= user)
		m_pPMask->newRow().fill(UNKNOWN_REAL_VALUE);
	m_pPMask->row(user)[attr] = val;
}

void GMatrixFactorization::clampItemElement(size_t item, size_t attr, double val)
{
	if(attr >= m_intrinsicDims)
		throw Ex("out of range");
	if(!m_pQMask)
	{
		m_pQMask = new GMatrix(0, m_intrinsicDims);
		m_pQWeights = new GMatrix(2, m_intrinsicDims);
		m_pQWeights->fill(0.0);
	}
	while(m_pQMask->rows() <= item)
		m_pQMask->newRow().fill(UNKNOWN_REAL_VALUE);
	m_pQMask->row(item)[attr] = val;
}

void GMatrixFactorization::clampUsers(const GMatrix& data, size_t offset)
{
	size_t vals = data.cols() - 1;
	for(size_t i = 0; i < data.rows(); i++)
	{
		const GVec& row = data[i];
		size_t index = (size_t)row[0];
		offset--;
		for(size_t j = 1; j <= vals; j++)
			clampUserElement(index, offset + j, row[j]);
	}
}

void GMatrixFactorization::clampItems(const GMatrix& data, size_t offset)
{
	size_t vals = data.cols() - 1;
	for(size_t i = 0; i < data.rows(); i++)
	{
		const GVec& row = data[i];
		size_t index = (size_t)row[0];
		offset--;
		for(size_t j = 1; j <= vals; j++)
			clampItemElement(index, offset + j, row[j]);
	}
}

double GMatrixFactorization::validate(GMatrix& data)
{
	double sse = 0;
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& vec = data[i];
		GVec& pref = m_pP->row(size_t(vec[0]));
		GVec& weights = m_pQ->row(size_t(vec[1]));
		double pred = weights[0] + pref[0];
		for(size_t j = 1; j <= m_intrinsicDims; j++)
			pred += pref[j] * weights[j];
		double err = vec[2] - pred;
		sse += (err * err);
	}
	return sse;
}

void GMatrixFactorization::clampP(size_t i)
{
	GVec& p = m_pP->row(i);
	GVec& mask = m_pPMask->row(i);
	GVec& bias = m_pPWeights->row(0);
	GVec& weights = m_pPWeights->row(1);
	for(size_t j = 0; j < m_intrinsicDims; j++)
	{
		if(mask[j] != UNKNOWN_REAL_VALUE)
			p[j + 1] = bias[j] + weights[j] * mask[j];
	}
}

void GMatrixFactorization::clampQ(size_t i)
{
	GVec& q = m_pQ->row(i);
	GVec& mask = m_pQMask->row(i);
	GVec& bias = m_pQWeights->row(0);
	GVec& weights = m_pQWeights->row(1);
	for(size_t j = 0; j < m_intrinsicDims; j++)
	{
		if(mask[j] != UNKNOWN_REAL_VALUE)
			q[j + 1] = bias[j] + weights[j] * mask[j];
	}
}

void GMatrixFactorization_absValues(double* pVec, size_t dims)
{
	while(true)
	{
		*pVec = std::abs(*pVec);
		if(--dims == 0)
			return;
	}
}

// virtual
void GMatrixFactorization::train(GMatrix& data)
{
	size_t users, items;
	GCollaborativeFilter_dims(data, &users, &items);

	// Initialize P and Q with small random values
	delete(m_pP);
	size_t colsP = 1 + m_intrinsicDims;
	m_pP = new GMatrix(users, colsP);
	for(size_t i = 0; i < m_pP->rows(); i++)
	{
		GVec& vec = m_pP->row(i);
		vec.fillNormal(m_rand, 0.02);
		if(m_nonNeg)
			GMatrixFactorization_absValues(m_pP->row(i).data() + 1, m_intrinsicDims);
	}
	delete(m_pQ);
	m_pQ = new GMatrix(items, 1 + m_intrinsicDims);
	for(size_t i = 0; i < m_pQ->rows(); i++)
	{
		GVec& vec = m_pQ->row(i);
		vec.fillNormal(m_rand, 0.02);
		if(m_nonNeg)
			GMatrixFactorization_absValues(m_pQ->row(i).data() + 1, m_intrinsicDims);
	}

	// Make a shallow copy of the data (so we can shuffle it)
	GMatrix dataCopy(data.relation().clone());
	GReleaseDataHolder hDataCopy(&dataCopy);
	for(size_t i = 0; i < data.rows(); i++)
		dataCopy.takeRow(&data[i]);

	// Train
	double prevErr = 1e10;
	double learningRate = 0.01;
	GVec pT(m_intrinsicDims + 1);
	size_t epochs = 0;
	while(learningRate >= 0.001)
	{
		GMatrix backupP(*m_pP);
		GMatrix backupQ(*m_pQ);
		for(size_t iter = 0; iter < m_minIters; iter++)
		{
			// Shuffle the ratings
			dataCopy.shuffle(m_rand);

			// Do an epoch of training
			for(size_t j = 0; j < dataCopy.rows(); j++)
			{
				GVec& vec = dataCopy[j];
				size_t user = (size_t)vec[0];
				size_t item = (size_t)vec[1];
				if(m_pPMask && user < m_pPMask->rows())
					clampP(user);
				if(m_pQMask && item < m_pQMask->rows())
					clampQ(item);

				// Compute the error for this rating
				GVec& p = m_pP->row(user);
				GVec& q = m_pQ->row(item);
				double pred = q[0] + p[0];
				for(size_t i = 1; i <= m_intrinsicDims; i++)
					pred += p[i] * q[i];
				double err = vec[2] - pred;

				// Update Q
				q[0] += learningRate * (err - m_regularizer * (q[0]));
				for(size_t i = 1; i <= m_intrinsicDims; i++)
				{
					pT[i] = q[i];
					q[i] += learningRate * (err * p[i] - m_regularizer * q[i]);
					if(m_nonNeg)
						q[i] = std::max(0.0, q[i]);
				}
				if(m_pQMask && item < m_pQMask->rows())
				{
					// Update the bias and weights for clamped values
					GVec& mask = m_pQMask->row(item);
					GVec& bb = m_pQWeights->row(0);
					GVec& w = m_pQWeights->row(1);
					for(size_t i = 0; i < m_intrinsicDims; i++)
					{
						if(mask[i] != UNKNOWN_REAL_VALUE)
						{
							bb[i] += 0.1 * learningRate * err * p[i + 1];
							w[i] += 0.1 * learningRate * err * p[i + 1] * mask[i];
						}
					}
				}

				// Update P
				p[0] += learningRate * (err - m_regularizer * p[0]);
				for(size_t i = 1; i <= m_intrinsicDims; i++)
				{
					p[i] += learningRate * (err * pT[i] - m_regularizer * p[i]);
					if(m_nonNeg)
						p[i] = std::max(0.0, p[i]);
				}
				if(m_pPMask && user < m_pPMask->rows())
				{
					// Update the bias and weights for clamped values
					GVec& mask = m_pPMask->row(user);
					GVec& bb = m_pPWeights->row(0);
					GVec& w = m_pPWeights->row(1);
					for(size_t i = 0; i < m_intrinsicDims; i++)
					{
						if(mask[i] != UNKNOWN_REAL_VALUE)
						{
							bb[i] += 0.1 * learningRate * err * pT[i + 1];
							w[i] += 0.1 * learningRate * err * pT[i + 1] * mask[i];
						}
					}
				}
			}
			epochs++;
		}

		// Stopping criteria
		double rsse = sqrt(validate(data));
		if(rsse >= 1e-12 && 1.0 - (rsse / prevErr) >= 0.001) {} else // This awkward if/else structure causes "nan" to be handled in a useful way
		{
			if(rsse <= prevErr) {} else // This awkward if/else structure causes "nan" to be handled in a useful way
			{
				// We didn't even get better, so restore from backup
				m_pP->copyBlock(backupP, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
				m_pQ->copyBlock(backupQ, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
			}
			learningRate *= m_decayRate; // decay the learning rate
		}
		prevErr = rsse;
	}
}

// virtual
double GMatrixFactorization::predict(size_t user, size_t item)
{
	if(!m_pP)
		throw Ex("Not trained yet");
	if(user >= m_pP->rows() || item >= m_pQ->rows())
		return 0.0;
	GVec& q = m_pQ->row(item);
	GVec& p = m_pP->row(user);
	double pred = p[0] + q[0];
	for(size_t i = 1; i <= m_intrinsicDims; i++)
		pred += p[i] * q[i];
	return pred;
}

void GMatrixFactorization_vectorToRatings(const GVec& vec, size_t dims, GMatrix& data)
{
	for(size_t i = 0; i < dims; i++)
	{
		if(vec[i] != UNKNOWN_REAL_VALUE)
		{
			GVec& row = data.newRow();
			row[0] = 0.0;
			row[1] = (double)i;
			row[2] = vec[i];
		}
	}
}

// virtual
void GMatrixFactorization::impute(GVec& vec, size_t dims)
{
	if(!m_pP)
		throw Ex("Not trained yet");

	// Convert the vector to a set of ratings
	GMatrix data(0, 3);
	GMatrixFactorization_vectorToRatings(vec, std::min(dims, m_pQ->rows()), data);

	// Initialize a preference vector
	GVec pP(1 + m_intrinsicDims);
	pP.fillNormal(m_rand, 0.02);

	// Refine the preference vector
	double prevErr = 1e308;
	double learningRate = 0.05;
	while(learningRate >= 0.001)
	{
		// Shuffle the ratings
		data.shuffle(m_rand);

		// Do an epoch of training
		for(size_t i = 0; i < data.rows(); i++)
		{
			// Compute the error for this rating
			GVec& v = data[i];
			GVec& q = m_pQ->row(size_t(v[1]));
			double pred = pP[0] + q[0];
			for(size_t j = 1; j <= m_intrinsicDims; j++)
				pred += pP[j] * q[j];
			double err = v[2] - pred;

			// Update the preference vec
			pP[0] += learningRate * (err - m_regularizer * pP[0]);
			for(size_t j = 1; j <= m_intrinsicDims; j++)
				pP[j] += learningRate * (err * q[j] - m_regularizer * pP[j]);
			pP.clip(-1.8, 1.8);
		}

		// Stopping criteria
		double rsse = sqrt(validate(data));
		if(rsse >= 1e-12 && 1.0 - (rsse / prevErr) >= 0.001) // If the amount of improvement is large
		{
		}
		else
			learningRate *= m_decayRate; // decay the learning rate
		prevErr = rsse;
	}

	// Impute missing values
	size_t n = std::min(dims, m_pQ->rows());
	size_t i;
	for(i = 0; i < n; i++)
	{
		if(vec[i] == UNKNOWN_REAL_VALUE)
		{
			GVec& q = m_pQ->row(i);
			double pred = pP[0] + q[0];
			for(size_t j = 1; j <= m_intrinsicDims; j++)
				pred += pP[j] * q[j];
			vec[i] = pred;
		}
	}
	for( ; i < dims; i++)
	{
		if(vec[i] == UNKNOWN_REAL_VALUE)
			vec[i] = 0.0;
	}
}

// static
void GMatrixFactorization::test()
{
	GMatrixFactorization rec(3);
	rec.setRegularizer(0.002);
	rec.basicTest(0.17);
}






/*
GHybridNonlinearPCA::GHybridNonlinearPCA(size_t intrinsicDims)
: GNonlinearPCA(intrinsicDims), m_itemAttrs(NULL), m_itemMax(NULL), m_itemMin(NULL), m_itemMap(NULL), m_numNeighbors(100), m_pRatingCount(NULL)
{
	m_neighbors = NULL;
}

// virtual
GHybridNonlinearPCA::~GHybridNonlinearPCA()
{
}

void GHybridNonlinearPCA::train(GMatrix& data)
{
	size_t usrs, items;
	GCollaborativeFilter_dims(data, &items, &usrs);
	m_items = items;

	// Copy and normalize the ratings
	GMatrix* pClone = new GMatrix();
	pClone->copy(&data);
	std::unique_ptr<GMatrix> hClone(pClone);
	delete[] m_pMins;
	m_pMins = new double[items];
	delete[] m_pMaxs;
	m_pMaxs = new double[items];
	delete[] m_pRatingCount;
	m_pRatingCount = new size_t[usrs];
	GVec::setAll(m_pMins, 1e200, items);
	GVec::setAll(m_pMaxs, -1e200, items);
	GIndexVec::setAll(m_pRatingCount, 0, usrs);
	for(size_t i = 0; i < pClone->rows(); i++)
	{
		GVec& vec = pClone->row(i);
		m_pMins[size_t(vec[0])] = std::min(m_pMins[size_t(vec[0])], vec[2]);
		m_pMaxs[size_t(vec[0])] = std::max(m_pMaxs[size_t(vec[0])], vec[2]);
	}
	for(size_t i = 0; i < items; i++)
	{
		if(m_pMins[i] >= 1e200)
			m_pMins[i] = 0.0;
		if(m_pMaxs[i] < m_pMins[i] + 1e-12)
			m_pMaxs[i] = m_pMins[i] + 1.0;
	}
	for(size_t i = 0; i < pClone->rows(); i++)
	{
		GVec&  vec = pClone->row(i);
		vec[2] = (vec[2] - m_pMins[size_t(vec[0])]) / (m_pMaxs[size_t(vec[0])] - m_pMins[size_t(vec[0])]);
		m_itemSet.insert((size_t)vec[1]);
		m_pRatingCount[(size_t)vec[1]]++;
	}

	// Prep the model for incremental training
	size_t numAttr = m_itemAttrs->cols() - 1;
	GUniformRelation featureRel(m_intrinsicDims + numAttr);
	GUniformRelation labelRel(items);
	m_pModel->setUseInputBias(m_useInputBias);
	m_pModel->beginIncrementalLearning(featureRel, labelRel);
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	nn.setUseInputBias(m_useInputBias);
	nn.beginIncrementalLearning(featureRel, labelRel);
	double* pPrefGradient = new double[m_intrinsicDims + numAttr];
	std::unique_ptr<double[]> hPrefGradient(pPrefGradient);

	// Train
	int startPass = 0;
	if(!m_useThreePass)
		startPass = 2;
	else if(m_pModel->layerCount() == 1)
		startPass = 2;
	for(int pass = startPass; pass < 3; pass++)
	{
		GNeuralNet* pNN = (pass == 0 ? &nn : m_pModel);
		if(pass == startPass) //-1)
		{
			delete(m_pUsers);
			m_pUsers = new GMatrix(usrs, m_intrinsicDims + numAttr);
			delete[] m_itemMap;
			m_itemMap = new size_t[m_itemAttrs->rows()];
			GIndexVec::setAll(m_itemMap, 0, m_itemAttrs->rows());
			size_t count = 0;
			double* itemVec = m_itemAttrs->row(count);
			for(size_t i = 0; i < usrs; i++)
			{
				double* vec = m_pUsers->row(i);
				GVec::setAll(vec, 0, m_intrinsicDims + numAttr);
				for(size_t j = 0; j < m_intrinsicDims; j++)
					*(vec++) = 0.01 * m_rand.normal();
				if(*itemVec == i)
				{
					m_itemMap[count]=i;
					*(itemVec) = 0;
					itemVec++;
					for(size_t j = 1; j < numAttr+1; j++)
					{
						*(vec++) = *(itemVec++) * 0.01;
					}
					itemVec = m_itemAttrs->row(++count);
				}
			}
		}
		double rateBegin = 0.1;
		double rateEnd = 0.001;
		double prevErr = 1e10;
		for(double learningRate = rateBegin; learningRate > rateEnd; )
		{
			GNeuralNet backupNet;
			backupNet.copyStructure(pNN);
			GMatrix backupUsers(m_pUsers->rows(), m_intrinsicDims);
			backupUsers.copyBlock(*m_pUsers, 0, 0, INVALID_INDEX, m_intrinsicDims, 0, 0, false);
			for(size_t j = 0; j < m_minIters; j++)
			{
				// Shuffle the ratings
				pClone->shuffle(m_rand);

				// Do an epoch of training
				m_pModel->setLearningRate(learningRate);
				for(size_t i = 0; i < pClone->rows(); i++)
				{
					// Forward-prop
					double* vec = pClone->row(i);
					size_t user = size_t(vec[1]);
					size_t item = size_t(vec[0]);
					double* pPrefs = m_pUsers->row(user);
					pNN->forwardPropSingleOutput(pPrefs, item);

					// Update weights
					pNN->backpropagateSingleOutput(item, vec[2]);
					if(pass < 2)
						pNN->scaleWeightsSingleOutput(item, 1.0 - (learningRate * m_regularizer));
					if(pass != 1)
						pNN->gradientOfInputsSingleOutput(item, pPrefGradient);
					pNN->descendGradientSingleOutput(item, pPrefs, learningRate, pNN->momentum());
					if(pass != 1)
					{
						// Update inputs
						if(pass == 0)
							GVec::multiply(pPrefs, 1.0 - (learningRate * m_regularizer), m_intrinsicDims);
						GVec::addScaled(pPrefs, -learningRate, pPrefGradient, m_intrinsicDims);
					}
				}
			}

			// Stopping criteria
			double rmse = sqrt(validate(pNN, *pClone));
			if(rmse >= 1e-12 && 1.0 - (rmse / prevErr) >= 0.001) {} else // this awkward if/else structure causes "nan" values to be handled in a useful way
			{
				if(rmse <= prevErr) {} else // this awkward if/else structure causes "nan" values to be handled in a useful way
				{
					pNN->copyWeights(&backupNet);
					m_pUsers->copyBlock(backupUsers, 0, 0, INVALID_INDEX, m_intrinsicDims, 0, 0, false);
				}
				learningRate *= m_decayRate; // decay the learning rate
			}
			prevErr = rmse;
		}
	}
	//Insert neighbors here after 0 out all of their indexes
	m_neighbors = new GKdTree(m_itemAttrs, m_numNeighbors);
}

// virtual
double GHybridNonlinearPCA::predict(size_t item, size_t user)
{
	//The user and item are reversed (the item is the user and user is the item)
	//If the user is new
	if(user >= m_pUsers->rows() || item >= m_items)
		return 0.0;


	//If the item has not yet been rated but has item features
	if(m_itemSet.find(user) == m_itemSet.end())
	{
		//find closest instances
		double* vec = m_pUsers->row(user);
		GTEMPBUF(double, features, m_pUsers->cols() - m_intrinsicDims + 1);
		features[0] = 0;
		for(size_t i = m_intrinsicDims; i < m_pUsers->cols(); i++)
		{
			features[i-m_intrinsicDims+1] = vec[i] / 0.01;
		}
		GTEMPBUF(size_t, neighbors, m_numNeighbors);
		GTEMPBUF(double, distances, m_numNeighbors);
		m_neighbors->neighbors(neighbors, distances, features);

		//sort the neighbors based on distance
		m_neighbors->sortNeighbors(m_numNeighbors, neighbors, distances);
/// Commentted out portion is for returning a weighted average instead of a weighted mode
//		double sum =0;
//		double denom = 0;

		size_t counts [11];
		GIndexVec::setAll(counts, 0, 11);
		size_t i = 0;
		while(distances[i] == 0 && i < m_numNeighbors)
		{
			size_t neighbor = m_itemMap[neighbors[i]];
			if(m_pRatingCount[neighbor] > 50)
			{
				double prediction = round(((m_pMaxs[item] - m_pMins[item]) * m_pModel->forwardPropSingleOutput(m_pUsers->row(neighbor), item) + m_pMins[item])* 2.0) / 2.0;
//			double prediction = (m_pMaxs[item] - m_pMins[item]) * m_pModel->forwardPropSingleOutput(m_pUsers->row(neighbor), item) + m_pMins[item];
				if(prediction > m_pMaxs[item])
					prediction = m_pMaxs[item];
				if(prediction <  m_pMins[item])
					prediction =  m_pMins[item];
				counts[(size_t) (prediction * 2)] += m_pRatingCount[neighbor];
			}
			i++;

		}
//		return sum / denom;
		size_t predMode = 0;
		for(size_t j = 1; j < 11; j++)
		{
			if(counts[j] > counts[predMode])
				predMode = j;
		}
		return predMode / 2.0;
	}

	return (m_pMaxs[item] - m_pMins[item]) * m_pModel->forwardPropSingleOutput(m_pUsers->row(user), item) + m_pMins[item];
}

double GHybridNonlinearPCA::validate(GNeuralNet* pNN, GMatrix& data)
{
	double sse = 0;
	for(size_t i = 0; i < data.rows(); i++)
	{
		double* vec = data[i];
		double* pPrefs = m_pUsers->row(size_t(vec[1]));
		double predictedRating = pNN->forwardPropSingleOutput(pPrefs, size_t(vec[0]));
		double d = vec[2] - predictedRating;
		sse += (d * d);
	}
	return sse / data.rows();
}

void GHybridNonlinearPCA::setItemAttributes(GMatrix& itemAttrs)
{
	delete(m_itemAttrs);
	m_itemAttrs = new GMatrix();
	m_itemAttrs->copy(&itemAttrs);
	delete(m_itemMax);
	m_itemMax = new double[m_itemAttrs->cols()-1];
	delete(m_itemMin);
	m_itemMin = new double[m_itemAttrs->cols()-1];
	GVec::setAll(m_itemMin, 1e200, m_itemAttrs->cols() - 1);
	GVec::setAll(m_itemMax, -1e200, m_itemAttrs->cols() - 1);

	//Normalize the item attributes
	for(size_t i = 1; i < m_itemAttrs->cols(); i++)
	{
		if (m_itemAttrs->relation().areContinuous(i, 1))
		{
			m_itemMax[i-1] = m_itemAttrs->columnMax(i);
			m_itemMin[i-1] = m_itemAttrs->columnMin(i);
			if (m_itemMax[i-1] > 1 || m_itemMin[i-1] < 0)
				m_itemAttrs->normalizeColumn(i, m_itemMax[i-1], m_itemMin[i-1]);
		}
	}
}

*/




/*
GNonlinearPCA::GNonlinearPCA(size_t intrinsicDims)
: GCollaborativeFilter(), m_intrinsicDims(intrinsicDims), m_items(0), m_pMins(NULL), m_pMaxs(NULL), m_useInputBias(true), m_useThreePass(true), m_minIters(1), m_decayRate(0.97), m_regularizer(0.0001)
{
	m_pModel = new GNeuralNet();
	m_pUserMask = NULL;
	m_pItemMask = NULL;
	m_pUsers = NULL;
}

GNonlinearPCA::GNonlinearPCA(const GDomNode* pNode, GLearnerLoader& ll)
: GCollaborativeFilter(pNode, ll)
{
	m_useInputBias = pNode->getBool("uib");
	m_pUsers = new GMatrix(pNode->get("users"));
	m_pModel = new GNeuralNet(pNode->get("model"), ll);
	GDomNode* pUserMask = pNode->getIfExists("usermask");
	if(pUserMask)
		m_pUserMask = new GMatrix(pUserMask);
	else
		m_pUserMask = NULL;
	GDomNode* pItemMask = pNode->getIfExists("itemmask");
	if(pItemMask)
		m_pItemMask = new GMatrix(pItemMask);
	else
		m_pItemMask = NULL;
	m_items = m_pModel->outputLayer().outputs();
	m_pMins = new double[m_items];
	GDomListIterator it1(pNode->get("mins"));
	if(it1.remaining() != m_items)
		throw Ex("invalid number of elements");
	GVec::deserialize(m_pMins, it1);
	m_pMaxs = new double[m_items];
	GDomListIterator it2(pNode->get("maxs"));
	if(it2.remaining() != m_items)
		throw Ex("invalid number of elements");
	GVec::deserialize(m_pMaxs, it2);
	m_intrinsicDims = m_pModel->layer(0).inputs();
}

// virtual
GNonlinearPCA::~GNonlinearPCA()
{
	delete[] m_pMins;
	delete[] m_pMaxs;
	delete(m_pModel);
	delete(m_pUsers);
	delete(m_pUserMask);
	delete(m_pItemMask);
}

// virtual
GDomNode* GNonlinearPCA::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNonlinearPCA");
	pNode->add(pDoc, "uib", pDoc->newBool(m_useInputBias));
	pNode->add(pDoc, "users", m_pUsers->serialize(pDoc));
	pNode->add(pDoc, "model", m_pModel->serialize(pDoc));
	if(m_pUserMask)
		pNode->add(pDoc, "usermask", m_pUserMask->serialize(pDoc));
	if(m_pItemMask)
		pNode->add(pDoc, "itemmask", m_pItemMask->serialize(pDoc));
	size_t itemCount = m_pModel->outputLayer().outputs();
	pNode->add(pDoc, "mins", GVec::serialize(pDoc, m_pMins, itemCount));
	pNode->add(pDoc, "maxs", GVec::serialize(pDoc, m_pMaxs, itemCount));
	return pNode;
}

double GNonlinearPCA::validate(GNeuralNet* pNN, GMatrix& data)
{
	double sse = 0;
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& vec = data[i];
		GVec& prefs = m_pUsers->row(size_t(vec[0]));
		double predictedRating = pNN->forwardPropSingleOutput(prefs, size_t(vec[1]));
		double d = vec[2] - predictedRating;
		sse += (d * d);
	}
	return sse / data.rows();
}

void GNonlinearPCA::clampUserElement(size_t user, size_t attr, double val)
{
val *= 0.01;
	if(attr >= m_intrinsicDims - (m_useInputBias ? 1 : 0))
		throw Ex("out of range");
	if(!m_pUserMask)
		m_pUserMask = new GMatrix(0, m_intrinsicDims - (m_useInputBias ? 1 : 0));
	while(m_pUserMask->rows() <= user)
		GVec::setAll(m_pUserMask->newRow(), UNKNOWN_REAL_VALUE, m_intrinsicDims - (m_useInputBias ? 1 : 0));
	m_pUserMask->row(user)[attr] = val;
}

void GNonlinearPCA::clampItemElement(size_t item, size_t attr, double val)
{
val *= 0.01;
	if(attr >= m_pModel->outputLayer().inputs())
		throw Ex("out of range");
	if(!m_pItemMask)
		m_pItemMask = new GMatrix(0, m_pModel->outputLayer().inputs());
	while(m_pItemMask->rows() <= item)
		GVec::setAll(m_pItemMask->newRow(), UNKNOWN_REAL_VALUE, m_pModel->outputLayer().inputs());
	m_pItemMask->row(item)[attr] = val;
}

void GNonlinearPCA::clampUsers(const GMatrix& data, size_t offset)
{
	size_t vals = data.cols() - 1;
	for(size_t i = 0; i < data.rows(); i++)
	{
		const GVec& row = data[i];
		size_t index = (size_t)*row;
		row++;
		for(size_t j = 0; j < vals; j++)
			clampUserElement(index, offset + j, *(row++));
	}
}

void GNonlinearPCA::clampItems(const GMatrix& data, size_t offset)
{
	size_t vals = data.cols() - 1;
	for(size_t i = 0; i < data.rows(); i++)
	{
		const GVec& row = data[i];
		size_t index = (size_t)*row;
		row++;
		for(size_t j = 0; j < vals; j++)
			clampItemElement(index, offset + j, *(row++));
	}
}

void GNonlinearPCA::clampUsersInternal(size_t i)
{
	GVec& profile = m_pUsers->row(i) + (m_useInputBias ? 1 : 0);
	const GVec& mask = m_pUserMask->row(i);
	for(size_t k = (m_useInputBias ? 1 : 0); k < m_intrinsicDims; k++)
	{
		if(*mask != UNKNOWN_REAL_VALUE)
			*profile = *mask;
		profile++;
		mask++;
	}
}

void GNonlinearPCA::clampItemsInternal(size_t i)
{
	GMatrix& itemWeights = ((GLayerClassic*)&m_pModel->outputLayer())->weights();
	const GVec& mask = m_pItemMask->row(i);
	size_t dims = m_pModel->outputLayer().inputs();
	for(size_t k = 0; k < dims; k++)
	{
		if(*mask != UNKNOWN_REAL_VALUE)
			itemWeights[k][i] = *mask;
		mask++;
	}
}


// virtual
void GNonlinearPCA::train(GMatrix& data)
{
	size_t usrs, items;
	GCollaborativeFilter_dims(data, &usrs, &items);
	m_items = items;

	// Copy and normalize the ratings
	GMatrix* pClone = new GMatrix();
	pClone->copy(&data);
	std::unique_ptr<GMatrix> hClone(pClone);
	delete[] m_pMins;
	m_pMins = new double[items];
	delete[] m_pMaxs;
	m_pMaxs = new double[items];
	GVec::setAll(m_pMins, 1e200, items);
	GVec::setAll(m_pMaxs, -1e200, items);
	for(size_t i = 0; i < pClone->rows(); i++)
	{
		GVec& vec = pClone->row(i);
		m_pMins[size_t(vec[1])] = std::min(m_pMins[size_t(vec[1])], vec[2]);
		m_pMaxs[size_t(vec[1])] = std::max(m_pMaxs[size_t(vec[1])], vec[2]);
	}
	for(size_t i = 0; i < items; i++)
	{
		if(m_pMins[i] >= 1e200)
			m_pMins[i] = 0.0;
		if(m_pMaxs[i] < m_pMins[i] + 1e-12)
			m_pMaxs[i] = m_pMins[i] + 1.0;
	}
	for(size_t i = 0; i < pClone->rows(); i++)
	{
		GVec&  vec = pClone->row(i);
		vec[2] = (vec[2] - m_pMins[size_t(vec[1])]) / (m_pMaxs[size_t(vec[1])] - m_pMins[size_t(vec[1])]);
	}

	// Prep the model for incremental training
	GUniformRelation featureRel(m_intrinsicDims);
	GUniformRelation labelRel(items);
	m_pModel->setUseInputBias(m_useInputBias);
	m_pModel->beginIncrementalLearning(featureRel, labelRel);
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	nn.setUseInputBias(m_useInputBias);
	nn.beginIncrementalLearning(featureRel, labelRel);
	GVec& prefGradient = new double[m_intrinsicDims];
	std::unique_ptr<double[]> hPrefGradient(prefGradient);

	// Train
	size_t startPass = 0;
	if(!m_useThreePass)
		startPass = 2;
	else if(m_pModel->layerCount() == 1)
		startPass = 2;
	for(size_t pass = startPass; pass < 3; pass++)
	{
		GNeuralNet* pNN = (pass == 0 ? &nn : m_pModel);
		if(pass == startPass)
		{
			// Initialize the user matrix
			delete(m_pUsers);
			m_pUsers = new GMatrix(usrs, m_intrinsicDims);
			for(size_t i = 0; i < usrs; i++)
			{
				GVec& vec = m_pUsers->row(i);
				for(size_t j = 0; j < m_intrinsicDims; j++)
					*(vec++) = 0.01 * m_rand.normal();
				if(m_pUserMask && i < m_pUserMask->rows())
					clampUsersInternal(i);
			}
		}
		if(m_pItemMask)
		{
			for(size_t i = 0; i < m_pItemMask->rows(); i++)
				clampItemsInternal(i);
		}
		double rateBegin = 0.1;
		double rateEnd = 0.001;
		double prevErr = 1e10;
		for(double learningRate = rateBegin; learningRate > rateEnd; )
		{
			GNeuralNet backupNet;
			backupNet.copyStructure(pNN);
			GMatrix backupUsers(*m_pUsers);
			for(size_t j = 0; j < m_minIters; j++)
			{
				// Shuffle the ratings
				pClone->shuffle(m_rand);

				// Do an epoch of training
				m_pModel->setLearningRate(learningRate);
				for(size_t i = 0; i < pClone->rows(); i++)
				{
					// Forward-prop
					GVec& vec = pClone->row(i);
					size_t user = size_t(vec[0]);
					size_t item = size_t(vec[1]);
					GVec& prefs = m_pUsers->row(user);
					pNN->forwardPropSingleOutput(prefs, item);

					// Update weights
					pNN->backpropagateSingleOutput(item, vec[2]);
					if(pass < 2)
						pNN->scaleWeightsSingleOutput(item, 1.0 - (learningRate * m_regularizer));
					if(pass != 1)
						pNN->gradientOfInputsSingleOutput(item, prefGradient);
					pNN->descendGradientSingleOutput(item, prefs, learningRate, pNN->momentum());
					if(m_pItemMask && item < m_pItemMask->rows())
						clampItemsInternal(item);
					if(pass != 1)
					{
						// Update inputs
						if(pass == 0)
							GVec::multiply(prefs, 1.0 - (learningRate * m_regularizer), m_intrinsicDims);
						GVec::addScaled(prefs, -learningRate, prefGradient, m_intrinsicDims);
						if(m_pUserMask && user < m_pUserMask->rows())
							clampUsersInternal(user);
					}
				}
			}

			// Stopping criteria
			double rmse = sqrt(validate(pNN, *pClone));
			if(rmse >= 1e-12 && 1.0 - (rmse / prevErr) >= 0.001) {} else // this awkward if/else structure causes "nan" values to be handled in a useful way
			{
				if(rmse <= prevErr) {} else // this awkward if/else structure causes "nan" values to be handled in a useful way
				{
					pNN->copyWeights(&backupNet);
					m_pUsers->copyBlock(backupUsers, 0, 0, INVALID_INDEX, INVALID_INDEX, 0, 0, false);
				}
				learningRate *= m_decayRate; // decay the learning rate
			}
			prevErr = rmse;
		}
	}
}

// virtual
double GNonlinearPCA::predict(size_t user, size_t item)
{
	if(user >= m_pUsers->rows() || item >= m_items)
		return 0.0;
	else
		return (m_pMaxs[item] - m_pMins[item]) * m_pModel->forwardPropSingleOutput(m_pUsers->row(user), item) + m_pMins[item];
}

// virtual
void GNonlinearPCA::impute(GVec& vec, size_t dims)
{
	throw Ex("Sorry, GNonlinearPCA::impute is not implemented yet");
}

// static
void GNonlinearPCA::test()
{
	GNonlinearPCA rec(3);
	rec.model()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, 3));
	rec.model()->addLayer(new GLayerClassic(3, FLEXIBLE_SIZE));
	rec.basicTest(0.18);
}
*/















GBagOfRecommenders::GBagOfRecommenders()
: GCollaborativeFilter(), m_itemCount(0)
{
}

GBagOfRecommenders::GBagOfRecommenders(const GDomNode* pNode, GLearnerLoader& ll)
: GCollaborativeFilter(pNode, ll)
{
	m_itemCount = (size_t)pNode->getInt("ic");
	for(GDomListIterator it(pNode->get("filters")); it.current(); it.advance())
		m_filters.push_back(ll.loadCollaborativeFilter(it.current()));
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
	pRecommender->rand().setSeed(m_rand.next()); // Ensure that each recommender has a different seed
	m_filters.push_back(pRecommender);
}

// virtual
void GBagOfRecommenders::train(GMatrix& data)
{
	for(vector<GCollaborativeFilter*>::iterator it = m_filters.begin(); it != m_filters.end(); it++)
	{
		// Make a matrix that randomly samples about half of the elements in pData
		GMatrix tmp(data.relation().clone());
		GReleaseDataHolder hTmp(&tmp);
		for(size_t i = 0; i < data.rows(); i++)
		{
			if(m_rand.next(2) == 0)
				tmp.takeRow(&data[i]);
		}

		// Train with it
		(*it)->train(tmp);
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

// virtual
void GBagOfRecommenders::impute(GVec& vec, size_t dims)
{
	throw Ex("Sorry, not implemented yet");
	/*
	size_t n = std::min(m_itemCount, dims);
	GTEMPBUF(double, pBuf1, n);
	GTEMPBUF(double, pBuf2, n);
	GVec::setAll(pBuf2, 0.0, n);
	double count = 0.0;
	for(vector<GCollaborativeFilter*>::iterator it = m_filters.begin(); it != m_filters.end(); it++)
	{
		GVec::copy(pBuf1, vec, n);
		(*it)->impute(pBuf1, dims);
		GVec::multiply(pBuf2, count / (count + 1), n);
		GVec::addScaled(pBuf2, 1.0 / (count + 1), pBuf1, n);
		count++;
	}
	size_t i;
	for(i = 0; i < n; i++)
	{
		if(*vec == UNKNOWN_REAL_VALUE)
			*vec = pBuf2[i];
		vec++;
	}
	for( ; i < dims; i++)
	{
		if(*vec == UNKNOWN_REAL_VALUE)
			*vec = 0.0;
		vec++;
	}
	*/
}

// virtual
GDomNode* GBagOfRecommenders::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GBagOfRecommenders");
	pNode->add(pDoc, "ic", m_itemCount);
	GDomNode* pFilters = pNode->add(pDoc, "filters", pDoc->newList());
	for(vector<GCollaborativeFilter*>::const_iterator it = m_filters.begin(); it != m_filters.end(); it++)
		pFilters->add(pDoc, (*it)->serialize(pDoc));
	return pNode;
}

// static
void GBagOfRecommenders::test()
{
	GBagOfRecommenders rec;
	rec.addRecommender(new GBaselineRecommender());
	rec.addRecommender(new GMatrixFactorization(3));
//	GNonlinearPCA* nlpca = new GNonlinearPCA(3);
//	nlpca->model()->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
//	rec.addRecommender(nlpca);
	rec.basicTest(0.69);
}






//virtual
GContentBasedFilter::~GContentBasedFilter()
{
	delete(m_itemAttrs);
	clear();
}

//virtual
void GContentBasedFilter::train(GMatrix& data)
{
	clear();
	m_userMap.clear();
	m_userRatings.clear();

	size_t users, items;
		GCollaborativeFilter_dims(data, &users, &items);
		m_items = items;
	m_users = users;
	std::set<size_t> userSet;


	if(m_itemAttrs == NULL)
		throw Ex("The items attributes has to be set");

	//create a training set and learning algorithm for each user
	for(size_t i = 0; i < data.rows(); i++)
	{
		GVec& vec = data.row(i);
		m_userRatings.insert(std::make_pair((size_t)vec[0], (size_t)vec[1]));
		userSet.insert((size_t)vec[0]);
	}

	//Loop through the set of users
	for(std::set<size_t>::iterator it = userSet.begin(); it != userSet.end(); ++it)
	{
		m_args.set_pos(m_init_pos);
		pair<multimap<size_t, size_t>::iterator, multimap<size_t, size_t>::iterator> ratedItems;
		ratedItems = m_userRatings.equal_range(*it);

		//create the training data for the user
		GMatrix* trainingData = new GMatrix(m_itemAttrs->relation().clone());
		GRelation* relation = data.relation().cloneSub(data.cols() - 1, 1);
		GMatrix* labels = new GMatrix(relation);
		for(multimap<size_t, size_t>::iterator ratings = ratedItems.first; ratings != ratedItems.second; ++ratings)
		{
			trainingData->newRow().copy(m_itemAttrs->row(m_itemMap[(*ratings).second]));

			GVec& temp = labels->newRow();
			temp[0] = data[(*ratings).second][2];
		}

		//train a learning algorithm for each user
		GSupervisedLearner* pLearn = (GSupervisedLearner*)GLearnerLib::InstantiateAlgorithm(m_args, trainingData, labels);
			if(m_args.size() > 0)
					throw Ex("Superfluous argument: ", m_args.peek());
		pLearn->train(*trainingData, *labels);
		m_userMap[(*it)] = m_learners.size();
		m_learners.push_back(pLearn);
	}
}

//virtual
double GContentBasedFilter::predict(size_t user, size_t item)
{
	if(user >= m_users || item >= m_items)
				return 0.0;
	GVec pOut(1);
	m_learners[m_userMap[user]]->predict(m_itemAttrs->row(m_itemMap[item]), pOut);
	return pOut[0];
}

//virtual
void GContentBasedFilter::impute(GVec& vec, size_t dims)
{
/*
	for(size_t i = 0; i < dims; i++)
	{
		if(*vec == UNKNOWN_REAL_VALUE)
			(*vec) = m_learners[]
	}
*/
	std::cerr << "Not yet implemented\n";
}

//virtual
GDomNode* GContentBasedFilter::serialize(GDom* pDoc) const
{
	return NULL;
}

void GContentBasedFilter::clear()
{
//		for(vector<GSupervisedLearner*>::iterator it = m_learners.begin(); it != m_learners.end(); it++)
//				delete(*it);
		m_learners.clear();
}

void GContentBasedFilter::setItemAttributes(GMatrix& itemAttrs)
{
	delete(m_itemAttrs);
	m_itemAttrs = new GMatrix(itemAttrs);
	for(size_t i = 0; i < m_itemAttrs->rows(); i++)
	{
		GVec& vec = m_itemAttrs->row(i);
		m_itemMap[(size_t)vec[0]] = i;
	}
	m_itemAttrs->swapColumns(0,m_itemAttrs->cols() - 1);
	m_itemAttrs->deleteColumns(m_itemAttrs->cols() - 1, 1);
}






GContentBoostedCF::GContentBoostedCF(GArgReader copy)
: GCollaborativeFilter(), m_ratingCounts(NULL), m_pseudoRatingSum(NULL)
{
	int orig_argc = copy.get_argc();
	int orig_pos = copy.get_pos();
	while(strcmp(copy.pop_string(), "--") != 0)
		if(copy.size() == 0)
			throw Ex("Expecting \"--\" to denote the parameters for the instance-based CF\n");
	int dashLoc = copy.get_pos() - 1;
	copy.set_argc(dashLoc);
	copy.set_pos(orig_pos);
	m_cbf = GRecommenderLib::InstantiateContentBasedFilter(copy);
	copy.set_pos(dashLoc + 1);
	copy.set_argc(orig_argc);
	m_cf = GRecommenderLib::InstantiateInstanceRecommender(copy);
}

GContentBoostedCF::~GContentBoostedCF()
{
	delete(m_cbf);
	delete(m_cf);
	delete[] m_ratingCounts;
	delete[] m_pseudoRatingSum;
}

void GContentBoostedCF::train(GMatrix& data)
{
	//make a copy of the training data
	GMatrix* pClone = new GMatrix(data);
	std::unique_ptr<GMatrix> hClone(pClone);
	m_cbf->train(*pClone);

	//Create the psuedo user-ratings vector for every user
	m_userMap = m_cbf->getUserMap();
	map<size_t, size_t> items = m_cbf->getItemMap();
	multimap<size_t, size_t> userRatings = m_cbf->getUserRatings();
	delete[] m_ratingCounts;
	delete[] m_pseudoRatingSum;
	m_ratingCounts = new size_t[m_userMap.size()];
	GIndexVec::setAll(m_ratingCounts, 0, m_userMap.size());

	m_pseudoRatingSum = new double[m_userMap.size()];
	GVecWrapper vw(m_pseudoRatingSum, m_userMap.size());
	vw.fill(0.0);

	for(size_t i = 0; i < pClone->rows(); i++)
		{
		GVec& vec = pClone->row(i);
		m_pseudoRatingSum[m_userMap[(size_t)vec[0]]] += vec[2];
	}

	//Loop through all of the users
	for(map<size_t, size_t>::iterator user=m_userMap.begin(); user!=m_userMap.end(); ++user)
	{
		pair<multimap<size_t, size_t>::iterator, multimap<size_t, size_t>::iterator> ratings;
		ratings = userRatings.equal_range(user->first);

		//Loop through all of the items
		for(map<size_t, size_t>::iterator item=items.begin(); item!=items.end(); ++item)
		{
			//Check if user has rated item
			bool isRated = false;
			multimap<size_t, size_t>::iterator rating;
			for(rating=ratings.first; rating!=ratings.second; ++rating)
			{
				if(rating->second == item->first)
					isRated = true;
				m_ratingCounts[m_userMap[user->first]]++;
			}
			if(!isRated)
			{
				//make prediction
				GVec& rat = pClone->newRow();
				rat[0] = (double)user->first;
				rat[1] = (double)item->first;
				rat[2] = m_cbf->predict(user->first, item->first);
				GAssert(rat[2] != UNKNOWN_REAL_VALUE);
				m_pseudoRatingSum[m_userMap[user->first]] += rat[2];
			}
		}
	}

	//Train CF on the psuedo user-ratings
	m_cf->train(*pClone);
	m_cf->clearUserDEPQ();
}

double GContentBoostedCF::predict(size_t user, size_t item)
{
	double max = 2;
	multimap<double,ArrayWrapper> neighbors = m_cf->getNeighbors(user, item);

		// Combine the ratings of the nearest neighbors to make a prediction
	size_t num = m_ratingCounts[m_userMap[user]];
	double selfWeight = (num > 50) ? 1.0 : num / 50.0;
		double weighted_sum = max * selfWeight * (m_cbf->predict(user, item)); // - (m_pseudoRatingSum[m_userMap[user]] / m_ratingCounts[m_userMap[user]]));
		double sum_weight = max * selfWeight;
		for(multimap<double,ArrayWrapper>::iterator it = neighbors.begin(); it != neighbors.end(); it++)
		{
				double weight = std::max(0.0, std::min(1.0, it->first));
		size_t neighNum = m_ratingCounts[m_userMap[(size_t)it->first]];
		double neighWeight = (neighNum > 50) ? 1.0 : neighNum / 50.0;
		double sigWeight = (it->second.values[1] > 50) ? 1.0 : it->second.values[1] / 50.0;
		weight *= ((2 * selfWeight * neighWeight) / (selfWeight + neighWeight)) + sigWeight;
				double val = m_cf->getRating(it->second.values[0], item);
				weighted_sum += weight * val;
				sum_weight += weight;
		}

//	return (m_pseudoRatingSum[m_userMap[user]] / m_ratingCounts[m_userMap[user]]) + (weighted_sum / sum_weight);
	return weighted_sum / sum_weight;
}

void GContentBoostedCF::impute(GVec& vec, size_t dims)
{
	std::cerr << "Not yet implemented\n";
}




} // namespace GClasses
