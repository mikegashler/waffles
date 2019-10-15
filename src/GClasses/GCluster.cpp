/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
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

#include "GCluster.h"
#include "GNeighborFinder.h"
#include "GDistance.h"
#include "GBitTable.h"
#include "GHeap.h"
#include "GMath.h"
#include "GVec.h"
#include <math.h>
#include <stdlib.h>
#include "GHillClimber.h"
#include "GBitTable.h"
#include "GSparseMatrix.h"
#include "GKNN.h"
#include "GHolders.h"
#include "GTime.h"
#include "GGraph.h"
#include "GDom.h"
#include <iostream>
#include <map>
#include <memory>

using namespace GClasses;
using std::cout;
using std::vector;
using std::map;

GClusterer::GClusterer(size_t nClusterCount)
: GTransform(), m_clusterCount(nClusterCount), m_pMetric(NULL), m_ownMetric(false)
{
}

// virtual
GClusterer::~GClusterer()
{
	if(m_ownMetric)
		delete(m_pMetric);
}

void GClusterer::setMetric(GDistanceMetric* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}






GSparseClusterer::GSparseClusterer(size_t clusters)
: m_clusterCount(clusters)
{
	m_pMetric = NULL;
	m_ownMetric = false;
}

// virtual
GSparseClusterer::~GSparseClusterer()
{
	if(m_ownMetric)
		delete(m_pMetric);
}

void GSparseClusterer::setMetric(GSparseSimilarity* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}





GAgglomerativeClusterer::GAgglomerativeClusterer(size_t clusters)
: GClusterer(clusters), m_pClusters(NULL)
{
}

GAgglomerativeClusterer::~GAgglomerativeClusterer()
{
	delete[] m_pClusters;
}

// virtual
void GAgglomerativeClusterer::cluster(const GMatrix* pData)
{
	// Init the metric
	if(!m_pMetric)
		setMetric(new GRowDistance(), true);
	m_pMetric->init(&pData->relation(), false);

	// Find enough neighbors to form a connected graph
	GNeighborGraph* pNF = NULL;
	std::unique_ptr<GNeighborGraph> hNF;
	size_t neighbors = 6;
	while(true)
	{
		GKdTree* pKdTree = new GKdTree(pData, m_pMetric, false);
		pNF = new GNeighborGraph(pKdTree, true, neighbors);
		hNF.reset(pNF);
		if(pNF->isConnected())
			break;
		if(neighbors + 1 >= pData->rows())
			throw Ex("internal problem--a graph with so many neighbors must be connected");
		neighbors = std::min((neighbors * 3) / 2, pData->rows() - 1);
	}

	// Sort all the neighbors by their distances
	vector< std::pair<double,size_t> > distNeighs;
	distNeighs.resize(pData->rows() * neighbors);
	vector< std::pair<double,size_t> >::iterator it = distNeighs.begin();
	for(size_t i = 0; i < pData->rows(); i++)
	{
		size_t nc = pNF->findNearest(neighbors, i);
		for(size_t j = 0; j < nc; j++)
		{
			it->first = pNF->distance(j);
			it->second = i * neighbors + j;
			it++;
		}
	}
	std::sort(distNeighs.begin(), it);

	// Assign each row to its own cluster
	delete[] m_pClusters;
	m_pClusters = new size_t[pData->rows()]; // specifies which cluster each row belongs to
	GIndexVec::makeIndexVec(m_pClusters, pData->rows());
	size_t* pSiblings = new size_t[pData->rows()]; // a cyclical linked list of each row in the cluster
	std::unique_ptr<size_t[]> hSiblings(pSiblings);
	GIndexVec::makeIndexVec(pSiblings, pData->rows()); // init such that each row is in a cluster of 1
	size_t currentClusterCount = pData->rows();
	if(currentClusterCount <= m_clusterCount)
		return; // nothing to do

	// Merge until we have the desired number of clusters
	for(vector< std::pair<double,size_t> >::iterator dn = distNeighs.begin(); dn != it; dn++)
	{
		// Get the next two closest points
		size_t a = dn->second / neighbors;
		pNF->findNearest(neighbors, a);
		size_t b = pNF->neighbor(dn->second % neighbors);
		GAssert(a != b && a < pData->rows() && b < pData->rows());
		size_t clustA = m_pClusters[a];
		size_t clustB = m_pClusters[b];

		// Merge the clusters
		if(clustA == clustB)
			continue; // The two points are already in the same cluster
		if(clustB < clustA) // Make sure clustA has the smaller value
		{
			std::swap(a, b);
			std::swap(clustA, clustB);
		}
		for(size_t i = pSiblings[b]; true; i = pSiblings[i]) // Convert every row in clustB to clustA
		{
			m_pClusters[i] = clustA;
			if(i == b)
				break;
		}
		std::swap(pSiblings[a], pSiblings[b]); // This line joins the cyclical linked lists into one big cycle
		if(clustB < m_clusterCount) // Ensure that the first m_clusterCount cluster numbers are always in use
		{
			for(size_t i = 0; i < pData->rows(); i++) // rename another cluster to take the spot of clustB
			{
				if(m_pClusters[i] >= m_clusterCount)
				{
					for(size_t j = pSiblings[i]; true; j = pSiblings[j])
					{
						m_pClusters[j] = clustB;
						if(j == i)
							break;
					}
					break;
				}
			}
		}
		if(--currentClusterCount <= m_clusterCount)
			return;
	}
	throw Ex("internal error--should have found the desired number of clusters before now");
}

// virtual
size_t GAgglomerativeClusterer::whichCluster(size_t nVector)
{
	return m_pClusters[nVector];
}


#define SPIRAL_POINTS 250
#define SPIRAL_HEIGHT 3

#include "GImage.h"
//#include "G3D.h"

// static
void GAgglomerativeClusterer::test()
{
	// Make a 3D data set with 3 entwined spirals
	GMatrix data(0, 3);
	data.reserve(3 * SPIRAL_POINTS);
	double dThirdCircle = M_PI * 2 / 3;
	double rads;
	for(size_t i = 0; i < SPIRAL_POINTS; i += 3)
	{
		rads = (double)i * 2 * M_PI / SPIRAL_POINTS;

		GVec& vector1 = data.newRow();
		vector1[0] = cos(rads);
		vector1[2] = sin(rads);
		vector1[1] = (double)i * SPIRAL_HEIGHT / SPIRAL_POINTS;

		GVec& vector2 = data.newRow();
		vector2[0] = cos(rads + dThirdCircle);
		vector2[2] = sin(rads + dThirdCircle);
		vector2[1] = (double)i * SPIRAL_HEIGHT / SPIRAL_POINTS;

		GVec& vector3 = data.newRow();
		vector3[0] = cos(rads + dThirdCircle + dThirdCircle);
		vector3[2] = sin(rads + dThirdCircle + dThirdCircle);
		vector3[1] = (double)i * SPIRAL_HEIGHT / SPIRAL_POINTS;
	}

	// Cluster the points
	GAgglomerativeClusterer clust(3);
	clust.cluster(&data);

	// Test for correctness
	if(clust.whichCluster(0) == clust.whichCluster(1))
		throw Ex("failed");
	if(clust.whichCluster(1) == clust.whichCluster(2))
		throw Ex("failed");
	if(clust.whichCluster(2) == clust.whichCluster(0))
		throw Ex("failed");
	for(size_t i = 3; i < SPIRAL_POINTS; i += 3)
	{
		if(clust.whichCluster(i) != clust.whichCluster(0))
			throw Ex("Wrong cluster");
		if(clust.whichCluster(i + 1) != clust.whichCluster(1))
			throw Ex("Wrong cluster");
		if(clust.whichCluster(i + 2) != clust.whichCluster(2))
			throw Ex("Wrong cluster");
	}

/*  // Uncomment this to make a spiffy visualization of the entwined spirals

	// Draw the classifications
	GImage image;
	image.SetSize(600, 600);
	image.Clear(0xff000000);
	GCamera camera;
	camera.SetViewAngle(PI / 2);
	camera.GetLookFromPoint()->Set(2, 1.5, 3);
	camera.GetLookDirection()->Set(-2, 0, -3);
	camera.ComputeSideVector();
	camera.SetImageSize(600, 600);
	double* pVec;
	G3DVector point;
	GColor col = 0;
	for(i = 0; i < SPIRAL_POINTS; i++)
	{
		pVec = data.row(i);
		point.Set(pVec[0], pVec[1], pVec[2]);
		switch(clust.whichCluster(i))
		{
			case 0: col = 0xffff0000; break;
			case 1: col = 0xff00ff00; break;
			case 2: col = 0xff0000ff; break;
		}
		image.Draw3DLine(&point, &point, &camera, col);
	}

	// Draw the bounding box
	G3DVector point2;
	size_t x, y, z;
	for(z = 0; z < 2; z++)
	{
		for(y = 0; y < 2; y++)
		{
			for(x = 0; x < 2; x++)
			{
				if(x == 0)
				{
					point.Set(-1, 3 * y, 2 * z - 1);
					point2.Set(1, 3 * y, 2 * z - 1);
					image.Draw3DLine(&point, &point2, &camera, 0xff808080);
				}
				if(y == 0)
				{
					point.Set(2 * x - 1, 0, 2 * z - 1);
					point2.Set(2 * x - 1, 3, 2 * z - 1);
					image.Draw3DLine(&point, &point2, &camera, 0xff808080);
				}
				if(z == 0)
				{
					point.Set(2 * x - 1, 3 * y, -1);
					point2.Set(2 * x - 1, 3 * y, 1);
					image.Draw3DLine(&point, &point2, &camera, 0xff808080);
				}
			}
		}
	}
	image.SavePNGFile("spirals.png");
*/
}

// -----------------------------------------------------------------------------------------

GAgglomerativeTransducer::GAgglomerativeTransducer()
: GTransducer(), m_pMetric(NULL), m_ownMetric(false)
{
}

GAgglomerativeTransducer::~GAgglomerativeTransducer()
{
}

void GAgglomerativeTransducer::setMetric(GDistanceMetric* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}

void GAgglomerativeTransducer::autoTune(GMatrix& features, GMatrix& labels)
{
	// This model has no parameters to tune
}

// virtual
std::unique_ptr<GMatrix> GAgglomerativeTransducer::transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2)
{
	// Init the metric
	if(!m_pMetric)
		setMetric(new GRowDistance(), true);
	m_pMetric->init(&features1.relation(), false);

	// Make a dataset with all featuers
	GMatrix featuresAll(features1.relation().clone());
	featuresAll.newRows(features1.rows() + features2.rows());
	featuresAll.copyBlock(features1, 0, 0, features1.rows(), features1.cols(), 0, 0, false);
	featuresAll.copyBlock(features2, 0, 0, features2.rows(), features2.cols(), features1.rows(), 0, false);

	// Find enough neighbors to form a connected graph
	GNeighborGraph* pNF = NULL;
	size_t neighbors = 6;
	while(true)
	{
		GKdTree* pKdTree = new GKdTree(&featuresAll, m_pMetric, false);
		pNF = new GNeighborGraph(pKdTree, true, neighbors);
		if(pNF->isConnected())
			break;
		if(neighbors + 1 >= featuresAll.rows())
		{
			delete(pNF);
			throw Ex("internal problem--a graph with so many neighbors must be connected");
		}
		neighbors = std::min((neighbors * 3) / 2, featuresAll.rows() - 1);
	}

	// Sort all the neighbors by their distances
	vector< std::pair<double,size_t> > distNeighs;
	distNeighs.resize(featuresAll.rows() * neighbors);
	vector< std::pair<double,size_t> >::iterator it = distNeighs.begin();
	for(size_t i = 0; i < featuresAll.rows(); i++)
	{
		size_t nc = pNF->findNearest(neighbors, i);
		for(size_t j = 0; j < nc; j++)
		{
			it->first = pNF->distance(j);
			it->second = i * neighbors + j;
			it++;
		}
	}
	std::sort(distNeighs.begin(), it);

	// Transduce
	auto pOut = std::unique_ptr<GMatrix>(new GMatrix(labels1.relation().clone()));
	pOut->newRows(features2.rows());
	pOut->fill(-1);
	size_t* pSiblings = new size_t[featuresAll.rows()]; // a cyclical linked list of each row in the cluster
	std::unique_ptr<size_t[]> hSiblings(pSiblings);
	for(size_t lab = 0; lab < labels1.cols(); lab++)
	{
		// Assign each row to its own cluster
		GIndexVec::makeIndexVec(pSiblings, featuresAll.rows()); // init such that each row is in a cluster of 1
		size_t missingLabels = features2.rows();

		// Merge until we have the desired number of clusters
		for(vector< std::pair<double,size_t> >::iterator dn = distNeighs.begin(); dn != it; dn++)
		{
			// Get the next two closest points
			size_t a = dn->second / neighbors;
			pNF->findNearest(neighbors, a);
			size_t b = pNF->neighbor(dn->second % neighbors);
			GAssert(a != b && a < featuresAll.rows() && b < featuresAll.rows());
			int labelA = (a < features1.rows() ? (int)labels1[a][lab] : (int)pOut->row(a - features1.rows())[lab]);
			int labelB = (b < features1.rows() ? (int)labels1[b][lab] : (int)pOut->row(b - features1.rows())[lab]);

			// Merge the clusters
			if(labelA >= 0 && labelB >= 0)
				continue; // Both points are already labeled, so there is no point in merging their clusters
			if(labelA < 0 && labelB >= 0) // Make sure that if one of them has a valid label, it is point a
			{
				std::swap(a, b);
				std::swap(labelA, labelB);
			}
			if(labelA >= 0)
			{
				for(size_t i = pSiblings[b]; true; i = pSiblings[i]) // Label every row in cluster b
				{
					GAssert(i >= features1.rows());
					GAssert(pOut->row(i - features1.rows())[lab] == (double)-1);
					pOut->row(i - features1.rows())[lab] = labelA;
					missingLabels--;
					if(i == b)
						break;
				}
				if(missingLabels <= 0)
					break;
			}
			std::swap(pSiblings[a], pSiblings[b]); // This line joins the cyclical linked lists into one big cycle
		}
	}
	return pOut;
}


// -----------------------------------------------------------------------------------------

GKMeans::GKMeans(size_t clusters, GRand* pRand)
: GClusterer(clusters), m_pCentroids(NULL), m_pClusters(NULL), m_reps(1), m_pRand(pRand)
{
}

GKMeans::~GKMeans()
{
	delete(m_pCentroids);
	delete[] m_pClusters;
}

void GKMeans::init(const GMatrix* pData)
{
	if(!m_pMetric)
		setMetric(new GRowDistance(), true);
	m_pMetric->init(&pData->relation(), false);
	if(pData->rows() < (size_t)m_clusterCount)
		throw Ex("Fewer data point than clusters");

	// Initialize the centroids with random rows. (Note that it is okay if two centroids happen to be initialized with the same row here, because the assignClusters method randomly picks among the best centroids in the event of a tie.)
	delete(m_pCentroids);
	m_pCentroids = new GMatrix(pData->relation().clone());
	m_pCentroids->newRows(m_clusterCount);
	for(size_t i = 0; i < m_clusterCount; i++)
	{
		size_t index = (size_t)m_pRand->next(m_clusterCount);
		m_pCentroids->row(i).copy(pData->row(index));
	}

	// Initialize the clusters
	delete[] m_pClusters;
	m_pClusters = new size_t[pData->rows()];
}

double GKMeans::assignClusters(const GMatrix* pData)
{
	// Assign each row to a cluster
	double sse = 0.0;
	for(size_t i = 0; i < pData->rows(); i++)
	{
		double best = 1e308;
		size_t clust = 0;
		size_t ties = 1;
		for(size_t j = 0; j < m_clusterCount; j++)
		{
			double d = m_pMetric->squaredDistance(pData->row(i), m_pCentroids->row(j));
			if(d < best)
			{
				clust = j;
				best = d;
				ties = 1;
			}
			else if(d == best)
			{
				// Pick randomly among the centroids that tie for the closest
				ties++;
				if(m_pRand->next(ties) == 0)
					clust = j;
			}
		}
		sse += best;
		m_pClusters[i] = clust;
	}
	return sse;
}

void GKMeans::recomputeCentroids(const GMatrix* pData)
{
	for(size_t i = 0; i < m_clusterCount; i++)
	{
		GVec& centroid = m_pCentroids->row(i);
		size_t unknownCount = 0;
		for(size_t j = 0; j < pData->cols(); j++)
		{
			size_t vals = pData->relation().valueCount(j);
			if(vals == 0)
			{
				double sum = 0.0;
				size_t count = 0;
				for(size_t k = 0; k < pData->rows(); k++)
				{
					if(m_pClusters[k] == i)
					{
						double d = pData->row(k)[j];
						if(d != UNKNOWN_REAL_VALUE)
						{
							sum += d;
							count++;
						}
					}
				}
				if(count > 0)
					centroid[j] = sum / count;
				else
				{
					centroid[j] = UNKNOWN_REAL_VALUE;
					unknownCount++;
				}
			}
			else
			{
				size_t* pFreq = new size_t[vals];
				std::unique_ptr<size_t[]> hFreq(pFreq);
				memset(pFreq, 0, sizeof(size_t) * vals);
				for(size_t k = 0; k < pData->rows(); k++)
				{
					if(m_pClusters[k] == i)
					{
						int v = (int)pData->row(k)[j];
						if(v != UNKNOWN_DISCRETE_VALUE && (size_t)v < vals)
							pFreq[v]++;
					}
				}
				size_t index = GIndexVec::indexOfMax(pFreq, vals);
				if(pFreq[index] == 0)
				{
					centroid[j] = UNKNOWN_DISCRETE_VALUE;
					unknownCount++;
				}
				else
					centroid[j] = (double)index;
			}
		}
		if(unknownCount > 0)
		{
			const GVec& row = pData->row((size_t)m_pRand->next(pData->rows()));
			for(size_t j = 0; j < pData->cols(); j++)
			{
				size_t vals = pData->relation().valueCount(j);
				if(vals == 0)
				{
					if(centroid[j] == UNKNOWN_REAL_VALUE)
						centroid[j] = row[j];
				}
				else
				{
					if(centroid[j] == UNKNOWN_DISCRETE_VALUE)
						centroid[j] = row[j];
				}
			}
		}
	}
}

// virtual
void GKMeans::cluster(const GMatrix* pData)
{
	size_t* pBest = NULL;
	double bestErr = 1e308;
	for(size_t i = 0; i < m_reps; i++)
	{
		init(pData);
		double d = 1e308;
		double sse = 1e308;
		for(size_t iters = 0; true; iters++)
		{
			d = assignClusters(pData);
			if(d >= sse && iters > 2)
				break;
			recomputeCentroids(pData);
			sse = d;
		}
		if(d < bestErr)
		{
			bestErr = d;
			delete[] pBest;
			pBest = m_pClusters;
			m_pClusters = NULL;
		}
	}
	if(pBest)
	{
		delete[] m_pClusters;
		m_pClusters = pBest;
	}
}

// virtual
size_t GKMeans::whichCluster(size_t index)
{
	return m_pClusters[index];
}


// -----------------------------------------------------------------------------------------

GFuzzyKMeans::GFuzzyKMeans(size_t clusters, GRand* pRand)
: GClusterer(clusters), m_pCentroids(NULL), m_pWeights(NULL), m_fuzzifier(1.3), m_reps(1), m_pRand(pRand)
{
}

GFuzzyKMeans::~GFuzzyKMeans()
{
	delete(m_pCentroids);
	delete(m_pWeights);
}

void GFuzzyKMeans::init(const GMatrix* pData)
{
	if(!m_pMetric)
		setMetric(new GRowDistance(), true);
	m_pMetric->init(&pData->relation(), false);
	if(pData->rows() < (size_t)m_clusterCount)
		throw Ex("Fewer data point than clusters");

	// Initialize the centroids
	delete(m_pCentroids);
	m_pCentroids = new GMatrix(pData->relation().clone());
	m_pCentroids->newRows(m_clusterCount);
	for(size_t j = 0; j < 10; j++) // Try up to ten times to find a unique set of initial centroids
	{
		// Pick random rows to be the initial centroids
		for(size_t i = 0; i < m_clusterCount; i++)
		{
			size_t index = (size_t)m_pRand->next(m_clusterCount);
			m_pCentroids->row(i).copy(pData->row(index));
		}

		// Test whether the centroids are all unique
		bool unique = true;
		for(size_t i = 1; i < m_clusterCount && unique; i++)
		{
			// Check if this row is already in use
			size_t k;
			for(k = 0; k < i; k++)
			{
				if(m_pCentroids->row(i).squaredDistance(m_pCentroids->row(k)) == 0)
				{
					unique = false;
					break;
				}
			}
		}
		if(unique)
			break;
	}

	// Initialize the weights
	delete(m_pWeights);
	m_pWeights = new GMatrix(pData->rows(), m_clusterCount);
}

double GFuzzyKMeans::recomputeWeights(const GMatrix* pData)
{
	double sumError = 0.0;
	for(size_t i = 0; i < pData->rows(); i++)
	{
		GVec& weights = m_pWeights->row(i);
		for(size_t j = 0; j < m_clusterCount; j++)
		{
			double e = -2.0 / (m_fuzzifier - 1.0);
			double b = std::max(1e-6, sqrt(m_pMetric->squaredDistance(pData->row(i), m_pCentroids->row(j))));
			weights[j] = pow(b, e);
		}
		double scale = 1.0 / weights.sum();
		sumError += scale;
		weights *= scale;
	}
	return sumError;
}

void GFuzzyKMeans::recomputeCentroids(const GMatrix* pData)
{
	for(size_t i = 0; i < m_clusterCount; i++)
	{
		GVec& centroid = m_pCentroids->row(i);
		size_t unknownCount = 0;
		for(size_t j = 0; j < pData->cols(); j++)
		{
			size_t vals = pData->relation().valueCount(j);
			if(vals == 0)
			{
				double sum = 0.0;
				double sumWeight = 0.0;
				for(size_t k = 0; k < pData->rows(); k++)
				{
					double d = pData->row(k)[j];
					if(d != UNKNOWN_REAL_VALUE)
					{
						double weight = m_pWeights->row(k)[i];
						sumWeight += weight;
						sum += weight * d;
					}
				}
				if(sumWeight > 0.0)
					centroid[j] = sum / sumWeight;
				else
				{
					centroid[j] = UNKNOWN_REAL_VALUE;
					unknownCount++;
				}
			}
			else
			{
				GVec pFreq(vals);
				pFreq.fill(0.0);
				for(size_t k = 0; k < pData->rows(); k++)
				{
					int v = (int)pData->row(k)[j];
					if(v != UNKNOWN_DISCRETE_VALUE && (size_t)v < vals)
						pFreq[v] += m_pWeights->row(k)[i];
				}
				size_t index = pFreq.indexOfMax();
				if(pFreq[index] > 0.0)
					centroid[j] = (double)index;
				else
				{
					centroid[j] = UNKNOWN_DISCRETE_VALUE;
					unknownCount++;
				}
			}
		}
		if(unknownCount > 0)
		{
			const GVec& row = pData->row((size_t)m_pRand->next(pData->rows()));
			for(size_t j = 0; j < pData->cols(); j++)
			{
				size_t vals = pData->relation().valueCount(j);
				if(vals == 0)
				{
					if(centroid[j] == UNKNOWN_REAL_VALUE)
						centroid[j] = row[j];
				}
				else
				{
					if(centroid[j] == UNKNOWN_DISCRETE_VALUE)
						centroid[j] = row[j];
				}
			}
		}
	}
}

// virtual
void GFuzzyKMeans::cluster(const GMatrix* pData)
{
	GMatrix* pBest = NULL;
	double bestErr = 1e308;
	for(size_t i = 0; i < m_reps; i++)
	{
		init(pData);
		double d = 1e308;
		double prevErr = 1e308;
		for(size_t iters = 0; true; iters++)
		{
			d = recomputeWeights(pData);
			if(iters > 2 && (d < 1e-12 || 1.0 - (d / prevErr) < 0.000001)) // If it improved by less than 0.0001%
				break;
			recomputeCentroids(pData);
			prevErr = d;
		}
		if(d < bestErr)
		{
			bestErr = d;
			pBest = m_pWeights;
			m_pWeights = NULL;
		}
	}
	if(pBest)
	{
		delete[] m_pWeights;
		m_pWeights = pBest;
	}
}

// virtual
size_t GFuzzyKMeans::whichCluster(size_t index)
{
	return m_pWeights->row(index).indexOfMax();
}


// -----------------------------------------------------------------------------------------

GKMedoids::GKMedoids(size_t clusters)
: GClusterer(clusters), m_pData(NULL)
{
	m_pMedoids = new size_t[clusters];
}

// virtual
GKMedoids::~GKMedoids()
{
	delete[] m_pMedoids;
}

double GKMedoids::curErr(const GMatrix* pData)
{
	double err = 0;
	for(size_t i = 0; i < pData->rows(); i++)
	{
		whichCluster(i);
		err += m_d;
	}
	return err;
}

// virtual
void GKMedoids::cluster(const GMatrix* pData)
{
	m_pData = pData;
	if(!m_pMetric)
		setMetric(new GRowDistance(), true);
	m_pMetric->init(&pData->relation(), false);
	if(pData->rows() < (size_t)m_clusterCount)
		throw Ex("Fewer data point than clusters");
	for(size_t i = 0; i < m_clusterCount; i++)
		m_pMedoids[i] = i;
	double err = curErr(pData);
	while(true)
	{
		bool improved = false;
		for(size_t i = 0; i < pData->rows(); i++)
		{
			// See if it's already a medoid
			size_t j;
			for(j = 0; j < m_clusterCount; j++)
			{
				if(m_pMedoids[j] == i)
					break;
			}
			if(j < m_clusterCount)
				continue;

			// Try this point in place of each medoid
			for(j = 0; j < m_clusterCount; j++)
			{
				size_t old = m_pMedoids[j];
				m_pMedoids[j] = i;
				double cand = curErr(pData);
				if(cand < err)
				{
					err = cand;
					improved = true;
					break;
				}
				else
					m_pMedoids[j] = old;
			}
		}
		if(!improved)
			break;
	}
}

// virtual
size_t GKMedoids::whichCluster(size_t nVector)
{
	const GVec& vec = m_pData->row(nVector);
	size_t clust = 0;
	m_d = m_pMetric->squaredDistance(vec, m_pData->row(m_pMedoids[0]));
	for(size_t i = 1; i < m_clusterCount; i++)
	{
		double d = m_pMetric->squaredDistance(vec, m_pData->row(m_pMedoids[i]));
		if(d < m_d)
		{
			m_d = d;
			clust = i;
		}
	}
	return clust;
}


// -----------------------------------------------------------------------------------------

GKMedoidsSparse::GKMedoidsSparse(size_t clusters)
: GSparseClusterer(clusters)
{
	m_pMedoids = new size_t[clusters];
	m_pData = NULL;
}

// virtual
GKMedoidsSparse::~GKMedoidsSparse()
{
	delete[] m_pMedoids;
}

double GKMedoidsSparse::curGoodness()
{
	double goodness = 0;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		whichCluster(i);
		goodness += m_d;
	}
	return goodness;
}

// virtual
void GKMedoidsSparse::cluster(GSparseMatrix* pData)
{
	m_pData = pData;
	if(!m_pMetric)
		setMetric(new GCosineSimilarity(), true);
	if(pData->rows() < (size_t)m_clusterCount)
		throw Ex("Fewer data point than clusters");
	for(size_t i = 0; i < m_clusterCount; i++)
		m_pMedoids[i] = i;
	double goodness = curGoodness();
	while(true)
	{
		bool improved = false;
		for(size_t i = 0; i < pData->rows(); i++)
		{
			// See if it's already a medoid
			size_t j;
			for(j = 0; j < m_clusterCount; j++)
			{
				if(m_pMedoids[j] == i)
					break;
			}
			if(j < m_clusterCount)
				continue;

			// Try this point in place of each medoid
			for(j = 0; j < m_clusterCount; j++)
			{
				size_t old = m_pMedoids[j];
				m_pMedoids[j] = i;
				double cand = curGoodness();
				if(cand > goodness)
				{
					goodness = cand;
					improved = true;
					break;
				}
				else
					m_pMedoids[j] = old;
			}
		}
		if(!improved)
			break;
	}
}

// virtual
size_t GKMedoidsSparse::whichCluster(size_t nVector)
{
	std::map<size_t,double>& vec = m_pData->row(nVector);
	size_t clust = 0;
	m_d = m_pMetric->similarity(vec, m_pData->row(m_pMedoids[0]));
	for(size_t i = 1; i < m_clusterCount; i++)
	{
		double d = m_pMetric->similarity(vec, m_pData->row(m_pMedoids[i]));
		if(d > m_d)
		{
			m_d = d;
			clust = i;
		}
	}
	return clust;
}


// -----------------------------------------------------------------------------------------

GKMeansSparse::GKMeansSparse(size_t nClusters, GRand* pRand)
: GSparseClusterer(nClusters)
{
	m_pRand = pRand;
	m_nClusters = nClusters;
	m_pClusters = NULL;
}

GKMeansSparse::~GKMeansSparse()
{
	delete[] m_pClusters;
}

// virtual
void GKMeansSparse::cluster(GSparseMatrix* pData)
{
	if(!m_pMetric)
		setMetric(new GCosineSimilarity(), true);

	// Pick the seeds (by randomly picking a known value for each element independently)
	size_t* pCounts = new size_t[pData->cols()];
	std::unique_ptr<size_t[]> hCounts(pCounts);
	GMatrix means(0, pData->cols());
	means.newRows(m_nClusters);
	{
		for(size_t i = 0; i < m_nClusters; i++)
		{
			size_t* pC = pCounts;
			for(size_t j = 0; j < pData->cols(); j++)
				*(pC++) = 0;
			GVec& mean = means.row(i);
			for(size_t k = 0; k < pData->rows(); k++)
			{
				GSparseMatrix::Iter it;
				for(it = pData->rowBegin(k); it != pData->rowEnd(k); it++)
				{
					if(m_pRand->next(pCounts[it->first] + 1) == 0)
					{
						pCounts[it->first]++;
						mean[it->first] = it->second;
					}
				}
			}
		}
	}

	// Do the clustering
	delete[] m_pClusters;
	m_pClusters = new size_t[pData->rows()];
	memset(m_pClusters, 0xff, sizeof(size_t) * pData->rows());
	double bestSim = -1e300;
	size_t patience = 16;
	while(true)
	{
		// Determine the cluster of each point
		bool somethingChanged = false;
		double sumSim = 0.0;
		size_t* pClust = m_pClusters;
		for(size_t i = 0; i < pData->rows(); i++)
		{
			size_t oldClust = *pClust;
			*pClust = 0;
			double maxSimilarity = -1e300;
			map<size_t,double>& sparseRow = pData->row(i);
			for(size_t j = 0; j < m_nClusters; j++)
			{
				double sim = m_pMetric->similarity(sparseRow, means.row(j));
				if(sim > maxSimilarity)
				{
					maxSimilarity = sim;
					*pClust = j;
				}
			}
			if(*pClust != oldClust)
				somethingChanged = true;
			pClust++;
			sumSim += maxSimilarity;
		}
		if(!somethingChanged)
			break;
		if(sumSim > bestSim)
		{
			bestSim = sumSim;
			patience = 16;
		}
		else
		{
			if(--patience == 0)
				break;
		}

		// Update the means
		for(size_t j = 0; j < m_nClusters; j++)
		{
			memset(pCounts, '\0', sizeof(size_t) * pData->cols());
			size_t* pClusts = m_pClusters;
			GVec& mean = means.row(j);
			for(size_t i = 0; i < pData->rows(); i++)
			{
				if(*pClusts != j)
					continue;

				// Update only the mean of the elements that this row specifies
				GSparseMatrix::Iter it;
				for(it = pData->rowBegin(i); it != pData->rowEnd(i); it++)
				{
					mean[it->first] *= (pCounts[it->first] / (pCounts[it->first] + 1));
					mean[it->first] += (1.0 / (pCounts[it->first] + 1) * it->second);
					pCounts[it->first]++;
				}
				pClusts++;
			}
		}
	}
}

// virtual
size_t GKMeansSparse::whichCluster(size_t nVector)
{
	return m_pClusters[nVector];
}


// -----------------------------------------------------------------------------------------
/*
void BlurVector(size_t nDims, double* pInput, double* pOutput, double dAmount)
{
	double dWeight, dSumWeight;
	size_t i, j;
	for(i = 0; i < nDims; i++)
	{
		pOutput[i] = 0;
		dSumWeight = 0;
		for(j = 0; j < nDims; j++)
		{
			dWeight = GMath::gaussian((double)(j - i) / dAmount);
			dSumWeight += dWeight;
			pOutput[i] += dWeight * pInput[j];
		}
		pOutput[i] /= dSumWeight;
	}
}
*/
/*
void MakeHistogramWithGaussianParzenWindow(size_t nElements, double* pInput, double* pOutput, double dBlurAmount)
{
	size_t i, j;
	for(i = 0; i < nElements; i++)
	{
		pOutput[i] = 0;
		for(j = 0; j < nElements; j++)
			pOutput[i] += GMath::gaussian((pOutput[j] - pOutput[i]) / dBlurAmount);
	}
}

size_t CountLocalMaximums(size_t nElements, double* pData)
{
	if(nElements < 2)
		return nElements;
	size_t nCount = 0;
	if(pData[0] > pData[1])
		nCount++;
	size_t i;
	nElements--;
	for(i = 1; i < nElements; i++)
	{
		if(pData[i] > pData[i - 1] && pData[i] > pData[i + 1])
			nCount++;
	}
	if(pData[nElements] > pData[nElements - 1])
		nCount++;
	return nCount;
}
*/

// -----------------------------------------------------------------------------------------

GGraphCutTransducer::GGraphCutTransducer()
: GTransducer(), m_neighborCount(12)
{
}

// virtual
GGraphCutTransducer::~GGraphCutTransducer()
{
}

void GGraphCutTransducer::setNeighbors(size_t k)
{
	m_neighborCount = k;
}

void GGraphCutTransducer::autoTune(GMatrix& features, GMatrix& labels)
{
	// Find the best value for k
	size_t cap = size_t(floor(sqrt(double(features.rows()))));
	size_t bestK = 4;
	double bestErr = 1e308;
	for(size_t i = 4; i < cap; i = size_t(i * 1.5))
	{
		m_neighborCount = i;
		double d = crossValidate(features, labels, 2);
		if(d < bestErr)
		{
			bestErr = d;
			bestK = i;
		}
		else if(i >= 12)
			break;
	}

	// Set the best values
	m_neighborCount = bestK;
}

// virtual
std::unique_ptr<GMatrix> GGraphCutTransducer::transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2)
{
	// Use k-NN to compute a distance metric with good scale factors for prediction
	GKNN knn;
	knn.setNeighborCount(m_neighborCount);
	//knn.setOptimizeScaleFactors(true);
	knn.train(features1, labels1);
	GDistanceMetric* pMetric = knn.metric();

	// Merge the features into one dataset and build a kd-tree
	GMatrix both(features1.relation().clone());
	both.newRows(features1.rows() + features2.rows());
	both.copyBlock(features1, 0, 0, features1.rows(), features1.cols(), 0, 0, false);
	both.copyBlock(features2, 0, 0, features2.rows(), features2.cols(), features1.rows(), 0, false);
	GKdTree neighborFinder(&both, pMetric, false);

	// Transduce
	auto pOut = std::unique_ptr<GMatrix>(new GMatrix(labels1.relation().clone()));
	pOut->newRows(features2.rows());
	pOut->fill(0);
	for(size_t lab = 0; lab < labels1.cols(); lab++)
	{
		// Use max-flow/min-cut graph-cut to separate out each label value
		int valueCount = (int)labels1.relation().valueCount(lab);
		for(int val = 1; val < valueCount; val++)
		{
			// Add neighborhood edges
			GGraphCut gc(features1.rows() + features2.rows() + 2);
			for(size_t i = 0; i < both.rows(); i++)
			{
				size_t nc = neighborFinder.findNearest(m_neighborCount, i);
				for(size_t j = 0; j < nc; j++)
				{
					gc.addEdge(2 + i, 2 + neighborFinder.neighbor(j), (float)(1.0 / std::max(sqrt(neighborFinder.distance(j)), 1e-9))); // connect neighbors
				}
			}

			// Add source and sink edges
			for(size_t i = 0; i < features1.rows(); i++)
			{
				if((int)labels1[i][0] == val)
					gc.addEdge(0, 2 + i, 1e12f); // connect to source
				else
					gc.addEdge(1, 2 + i, 1e12f); // connect to sink
			}

			// Cut
			gc.cut(0, 1);

			// Label the unlabeled rows
			for(size_t i = 0; i < features2.rows(); i++)
			{
				if(gc.isSource(2 + features1.rows() + i))
					pOut->row(i)[lab] = (double)val;
			}
		}
	}
	return pOut;
}
