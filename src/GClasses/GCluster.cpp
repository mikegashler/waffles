/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
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
#include "GNeuralNet.h"
#include "GHillClimber.h"
#include "GBitTable.h"
#include "GSparseMatrix.h"
#include "GKNN.h"
#include "GTime.h"
#include "GGraph.h"
#include "GTwt.h"
#include <iostream>
#include <map>

using namespace GClasses;
using std::cout;
using std::vector;
using std::map;


GAgglomerativeClusterer::GAgglomerativeClusterer(size_t clusters)
: GClusterer(clusters), m_pMetric(NULL), m_ownMetric(false), m_pClusters(NULL)
{
}

GAgglomerativeClusterer::~GAgglomerativeClusterer()
{
	if(m_ownMetric)
		delete(m_pMetric);
	delete[] m_pClusters;
}

void GAgglomerativeClusterer::setMetric(GDissimilarityMetric* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}

// virtual
void GAgglomerativeClusterer::cluster(GMatrix* pData)
{
	// Init the metric
	if(!m_pMetric)
		setMetric(new GRowDistance(), true);
	m_pMetric->init(pData->relation());

	// Find enough neighbors to form a connected graph
	GNeighborFinderCacheWrapper* pNF = NULL;
	size_t neighbors = 6;
	while(true)
	{
		GKdTree* pKdTree = new GKdTree(pData, neighbors, m_pMetric, false);
		pNF = new GNeighborFinderCacheWrapper(pKdTree, true);
		pNF->fillCache();
		if(pNF->isConnected())
			break;
		if(neighbors + 1 >= pData->rows())
		{
			delete(pNF);
			ThrowError("internal problem--a graph with so many neighbors must be connected");
		}
		neighbors = std::min((neighbors * 3) / 2, pData->rows() - 1);
	}

	// Sort all the neighbors by their distances
	size_t count = pData->rows() * neighbors;
	vector< std::pair<double,size_t> > distNeighs;
	distNeighs.resize(count);
	double* pDistances = pNF->squaredDistanceTable();
	size_t* pRows = pNF->cache();
	size_t index = 0;
	vector< std::pair<double,size_t> >::iterator it = distNeighs.begin();
	for(size_t i = 0; i < count; i++)
	{
		if(*pRows < pData->rows())
		{
			it->first = *pDistances;
			it->second = i;
			it++;
		}
		else
			index--;
		pRows++;
		pDistances++;
	}
	std::sort(distNeighs.begin(), it);

	// Assign each row to its own cluster
	delete[] m_pClusters;
	m_pClusters = new size_t[pData->rows()]; // specifies which cluster each row belongs to
	GIndexVec::makeIndexVec(m_pClusters, pData->rows());
	size_t* pSiblings = new size_t[pData->rows()]; // a cyclical linked list of each row in the cluster
	ArrayHolder<size_t> hSiblings(pSiblings);
	GIndexVec::makeIndexVec(pSiblings, pData->rows()); // init such that each row is in a cluster of 1
	size_t currentClusterCount = pData->rows();
	if(currentClusterCount <= m_clusterCount)
		return; // nothing to do

	// Merge until we have the desired number of clusters
	pRows = pNF->cache();
	for(vector< std::pair<double,size_t> >::iterator dn = distNeighs.begin(); dn != it; dn++)
	{
		// Get the next two closest points
		size_t a = dn->second / neighbors;
		size_t b = pRows[dn->second];
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
	ThrowError("internal error--should have found the desired number of clusters before now");
}

// virtual
size_t GAgglomerativeClusterer::whichCluster(size_t nVector)
{
	return m_pClusters[nVector];
}

#ifndef NO_TEST_CODE

#define SPRIAL_POINTS 250
#define SPIRAL_HEIGHT 3

#include "GImage.h"
//#include "G3D.h"

// static
void GAgglomerativeClusterer::test()
{
	// Make a 3D data set with 3 entwined spirals
	GHeap heap(1000);
	GMatrix data(0, 3, &heap);
	double dThirdCircle = M_PI * 2 / 3;
	double* pVector;
	double rads;
	for(size_t i = 0; i < SPRIAL_POINTS; i += 3)
	{
		rads = (double)i * 2 * M_PI / SPRIAL_POINTS;

		pVector = data.newRow();
		pVector[0] = cos(rads);
		pVector[2] = sin(rads);
		pVector[1] = (double)i * SPIRAL_HEIGHT / SPRIAL_POINTS;

		pVector = data.newRow();
		pVector[0] = cos(rads + dThirdCircle);
		pVector[2] = sin(rads + dThirdCircle);
		pVector[1] = (double)i * SPIRAL_HEIGHT / SPRIAL_POINTS;

		pVector = data.newRow();
		pVector[0] = cos(rads + dThirdCircle + dThirdCircle);
		pVector[2] = sin(rads + dThirdCircle + dThirdCircle);
		pVector[1] = (double)i * SPIRAL_HEIGHT / SPRIAL_POINTS;
	}

	// Cluster the points
	GAgglomerativeClusterer clust(3);
	clust.cluster(&data);

	// Test for correctness
	if(clust.whichCluster(0) == clust.whichCluster(1))
		ThrowError("failed");
	if(clust.whichCluster(1) == clust.whichCluster(2))
		ThrowError("failed");
	if(clust.whichCluster(2) == clust.whichCluster(0))
		ThrowError("failed");
	for(size_t i = 3; i < SPRIAL_POINTS; i += 3)
	{
		if(clust.whichCluster(i) != clust.whichCluster(0))
			ThrowError("Wrong cluster");
		if(clust.whichCluster(i + 1) != clust.whichCluster(1))
			ThrowError("Wrong cluster");
		if(clust.whichCluster(i + 2) != clust.whichCluster(2))
			ThrowError("Wrong cluster");
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
	for(i = 0; i < SPRIAL_POINTS; i++)
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
#endif // !NO_TEST_CODE

// -----------------------------------------------------------------------------------------

GAgglomerativeTransducer::GAgglomerativeTransducer()
: GTransducer(), m_pMetric(NULL), m_ownMetric(false)
{
}

GAgglomerativeTransducer::~GAgglomerativeTransducer()
{
}

void GAgglomerativeTransducer::setMetric(GDissimilarityMetric* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}

// virtual
GMatrix* GAgglomerativeTransducer::transduceInner(GMatrix& features1, GMatrix& labels1, GMatrix& features2)
{
	// Init the metric
	if(!m_pMetric)
		setMetric(new GRowDistance(), true);
	m_pMetric->init(features1.relation());

	// Make a dataset with all featuers
	GMatrix featuresAll(features1.relation());
	featuresAll.reserve(features1.rows() + features2.rows());
	GReleaseDataHolder hFeaturesAll(&featuresAll);
	for(size_t i = 0; i < features1.rows(); i++)
		featuresAll.takeRow(features1[i]);
	for(size_t i = 0; i < features2.rows(); i++)
		featuresAll.takeRow(features2[i]);

	// Find enough neighbors to form a connected graph
	GNeighborFinderCacheWrapper* pNF = NULL;
	size_t neighbors = 6;
	while(true)
	{
		GKdTree* pKdTree = new GKdTree(&featuresAll, neighbors, m_pMetric, false);
		pNF = new GNeighborFinderCacheWrapper(pKdTree, true);
		pNF->fillCache();
		if(pNF->isConnected())
			break;
		if(neighbors + 1 >= featuresAll.rows())
		{
			delete(pNF);
			ThrowError("internal problem--a graph with so many neighbors must be connected");
		}
		neighbors = std::min((neighbors * 3) / 2, featuresAll.rows() - 1);
	}

	// Sort all the neighbors by their distances
	size_t count = featuresAll.rows() * neighbors;
	vector< std::pair<double,size_t> > distNeighs;
	distNeighs.resize(count);
	double* pDistances = pNF->squaredDistanceTable();
	size_t* pRows = pNF->cache();
	size_t index = 0;
	vector< std::pair<double,size_t> >::iterator it = distNeighs.begin();
	for(size_t i = 0; i < count; i++)
	{
		if(*pRows < featuresAll.rows())
		{
			it->first = *pDistances;
			it->second = i;
			it++;
		}
		else
			index--;
		pRows++;
		pDistances++;
	}
	std::sort(distNeighs.begin(), it);

	// Transduce
	GMatrix* pOut = new GMatrix(labels1.relation());
	Holder<GMatrix> hOut(pOut);
	pOut->newRows(features2.rows());
	pOut->setAll(-1);
	size_t* pSiblings = new size_t[featuresAll.rows()]; // a cyclical linked list of each row in the cluster
	ArrayHolder<size_t> hSiblings(pSiblings);
	for(size_t lab = 0; lab < labels1.cols(); lab++)
	{
		// Assign each row to its own cluster
		GIndexVec::makeIndexVec(pSiblings, featuresAll.rows()); // init such that each row is in a cluster of 1
		size_t missingLabels = features2.rows();
	
		// Merge until we have the desired number of clusters
		pRows = pNF->cache();
		for(vector< std::pair<double,size_t> >::iterator dn = distNeighs.begin(); dn != it; dn++)
		{
			// Get the next two closest points
			size_t a = dn->second / neighbors;
			size_t b = pRows[dn->second];
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
	return hOut.release();
}


// -----------------------------------------------------------------------------------------

GKMeans::GKMeans(size_t clusters, GRand* pRand)
: GClusterer(clusters), m_pCentroids(NULL), m_pMetric(NULL), m_pData(NULL), m_pClusters(NULL), m_pRand(pRand)
{
}

GKMeans::~GKMeans()
{
	delete(m_pCentroids);
	delete(m_pMetric);
	delete(m_pClusters);
}

void GKMeans::setMetric(GDissimilarityMetric* pMetric)
{
	delete(m_pMetric);
	m_pMetric = pMetric;
}

void GKMeans::init(GMatrix* pData)
{
	m_pData = pData;
	if(!m_pMetric)
		setMetric(new GRowDistance());
	m_pMetric->init(pData->relation());
	if(pData->rows() < (size_t)m_clusterCount)
		ThrowError("Fewer data point than clusters");

	// Initialize the centroids
	delete(m_pCentroids);
	m_pCentroids = new GMatrix(pData->relation());
	m_pCentroids->newRows(m_clusterCount);
	for(size_t i = 0; i < m_clusterCount; i++)
	{
		size_t j;
		for(j = 0; j < 100; j++)
		{
			// Pick a random row to be a centroid
			size_t index = (size_t)m_pRand->next(m_clusterCount);

			// Check if this row is already in use
			size_t k;
			for(k = 0; k < i; k++)
			{
				if(m_pMetric->dissimilarity(pData->row(index), m_pCentroids->row(k)) == 0)
					break;
			}
			if(k == i) // If a unique instance was found...
			{
				GVec::copy(m_pCentroids->row(i), pData->row(index), pData->cols());
				break;
			}
		}
		if(j >= 100)
			ThrowError("A unique instance could not be found for a centroid in 100 random tries.");
	}

	// Initialize the clusters
	delete(m_pClusters);
	m_pClusters = new size_t[pData->rows()];
}

double GKMeans::assignClusters()
{
	// Assign each row to a cluster
	double sse = 0.0;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		double best = 1e308;
		size_t clust = 0;
		for(size_t j = 0; j < m_clusterCount; j++)
		{
			double d = m_pMetric->dissimilarity(m_pData->row(i), m_pCentroids->row(j));
			if(d < best)
			{
				clust = j;
				best = d;
			}
		}
		sse += best;
		m_pClusters[i] = clust;
	}
	return sse;
}

void GKMeans::recomputeCentroids()
{
	// Recompute the centroids
	for(size_t i = 0; i < m_clusterCount; i++)
	{
		for(size_t j = 0; j < m_pData->cols(); j++)
		{
			size_t vals = m_pData->relation()->valueCount(j);
			if(vals == 0)
			{
				double sum = 0.0;
				size_t count = 0;
				for(size_t k = 0; k < m_pData->rows(); k++)
				{
					if(m_pClusters[k] == i)
					{
						double d = m_pData->row(k)[j];
						if(d != UNKNOWN_REAL_VALUE)
						{
							sum += d;
							count++;
						}
					}
				}
				if(count > 0)
					m_pCentroids->row(i)[j] = sum / count;
				else
					m_pCentroids->row(i)[j] = UNKNOWN_REAL_VALUE;
			}
			else
			{
				size_t* pFreq = new size_t[vals];
				ArrayHolder<size_t> hFreq(pFreq);
				memset(pFreq, 0, sizeof(size_t) * vals);
				for(size_t k = 0; k < m_pData->rows(); k++)
				{
					if(m_pClusters[k] == i)
					{
						int v = (int)m_pData->row(k)[j];
						if(v >= 0 && (size_t)v < vals)
							pFreq[v]++;
					}
				}
				size_t index = GIndexVec::indexOfMax(pFreq, vals);
				if(pFreq[index] == 0)
					m_pCentroids->row(i)[j] = UNKNOWN_DISCRETE_VALUE;
				else
					m_pCentroids->row(i)[j] = (double)index;
			}
		}
	}
}

// virtual
void GKMeans::cluster(GMatrix* pData)
{
	init(pData);
	double sse = 1e308;
	for(size_t iters = 0; true; iters++)
	{
		double d = assignClusters();
		if(d >= sse && iters > 2)
			break;
		recomputeCentroids();
		sse = d;
	}
}

// virtual
size_t GKMeans::whichCluster(size_t index)
{
	GAssert(index < m_pData->rows());
	return m_pClusters[index];
}


// -----------------------------------------------------------------------------------------

GKMedoids::GKMedoids(size_t clusters)
: GClusterer(clusters), m_pMetric(NULL), m_pData(NULL)
{
	m_pMedoids = new size_t[clusters];
}

// virtual
GKMedoids::~GKMedoids()
{
	delete[] m_pMedoids;
	delete(m_pMetric);
}

void GKMedoids::setMetric(GDissimilarityMetric* pMetric)
{
	delete(m_pMetric);
	m_pMetric = pMetric;
}

double GKMedoids::curErr()
{
	double err = 0;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		whichCluster(i);
		err += m_d;
	}
	return err;
}

// virtual
void GKMedoids::cluster(GMatrix* pData)
{
	m_pData = pData;
	if(!m_pMetric)
		setMetric(new GRowDistance());
	m_pMetric->init(pData->relation());
	if(pData->rows() < (size_t)m_clusterCount)
		ThrowError("Fewer data point than clusters");
	for(size_t i = 0; i < m_clusterCount; i++)
		m_pMedoids[i] = i;
	double err = curErr();
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
				double cand = curErr();
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
	double* pVec = m_pData->row(nVector);
	size_t cluster = 0;
	m_d = m_pMetric->dissimilarity(pVec, m_pData->row(m_pMedoids[0]));
	for(size_t i = 1; i < m_clusterCount; i++)
	{
		double d = m_pMetric->dissimilarity(pVec, m_pData->row(m_pMedoids[i]));
		if(d < m_d)
		{
			m_d = d;
			cluster = i;
		}
	}
	return cluster;
}


// -----------------------------------------------------------------------------------------

GKMedoidsSparse::GKMedoidsSparse(size_t clusters)
: GSparseClusterer(clusters)
{
	m_pMedoids = new size_t[clusters];
	m_pMetric = NULL;
	m_ownMetric = false;
	m_pData = NULL;
}

// virtual
GKMedoidsSparse::~GKMedoidsSparse()
{
	delete[] m_pMedoids;
	delete(m_pMetric);
}

void GKMedoidsSparse::setMetric(GSparseSimilarity* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
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
		ThrowError("Fewer data point than clusters");
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
	size_t cluster = 0;
	m_d = m_pMetric->similarity(vec, m_pData->row(m_pMedoids[0]));
	for(size_t i = 1; i < m_clusterCount; i++)
	{
		double d = m_pMetric->similarity(vec, m_pData->row(m_pMedoids[i]));
		if(d > m_d)
		{
			m_d = d;
			cluster = i;
		}
	}
	return cluster;
}


// -----------------------------------------------------------------------------------------

GKMeansSparse::GKMeansSparse(size_t nClusters, GRand* pRand)
: GSparseClusterer(nClusters)
{
	m_pRand = pRand;
	m_nClusters = nClusters;
	m_pClusters = NULL;
	m_pMetric = NULL;
	m_ownMetric = false;
}

GKMeansSparse::~GKMeansSparse()
{
	delete[] m_pClusters;
	if(m_ownMetric)
		delete(m_pMetric);
}

void GKMeansSparse::setMetric(GSparseSimilarity* pMetric, bool own)
{
	if(m_ownMetric)
		delete(m_pMetric);
	m_pMetric = pMetric;
	m_ownMetric = own;
}

// virtual
void GKMeansSparse::cluster(GSparseMatrix* pData)
{
	if(!m_pMetric)
		setMetric(new GCosineSimilarity(), true);

	// Pick the seeds (by randomly picking a known value for each element independently)
	size_t* pCounts = new size_t[pData->cols()];
	ArrayHolder<size_t> hCounts(pCounts);
	GMatrix means(0, pData->cols());
	means.newRows(m_nClusters);
	{
		for(size_t i = 0; i < m_nClusters; i++)
		{
			size_t* pC = pCounts;
			for(size_t j = 0; j < pData->cols(); j++)
				*(pC++) = 0;
			double* pMean = means.row(i);
			for(size_t k = 0; k < pData->rows(); k++)
			{
				GSparseMatrix::Iter it;
				for(it = pData->rowBegin(k); it != pData->rowEnd(k); it++)
				{
					if(m_pRand->next(pCounts[it->first] + 1) == 0)
					{
						pCounts[it->first]++;
						pMean[it->first] = it->second;
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
			size_t* pClust = m_pClusters;
			double* pMean = means.row(j);
			for(size_t i = 0; i < pData->rows(); i++)
			{
				if(*pClust != j)
					continue;

				// Update only the mean of the elements that this row specifies
				GSparseMatrix::Iter it;
				for(it = pData->rowBegin(i); it != pData->rowEnd(i); it++)
				{
					pMean[it->first] *= (pCounts[it->first] / (pCounts[it->first] + 1));
					pMean[it->first] += (1.0 / (pCounts[it->first] + 1) * it->second);
					pCounts[it->first]++;
				}
				pClust++;
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

GGraphCutTransducer::GGraphCutTransducer(size_t neighborCount, GRand* pRand)
: GTransducer(), m_neighborCount(neighborCount), m_pRand(pRand)
{
	m_pNeighbors = new size_t[neighborCount];
	m_pDistances = new double[neighborCount];
}

// virtual
GGraphCutTransducer::~GGraphCutTransducer()
{
	delete[] m_pNeighbors;
	delete[] m_pDistances;
}

// virtual
GMatrix* GGraphCutTransducer::transduceInner(GMatrix& features1, GMatrix& labels1, GMatrix& features2)
{
	// Use k-NN to compute a distance metric with good scale factors for prediction
	GKNN knn(m_neighborCount, m_pRand);
	//knn.setOptimizeScaleFactors(true);
	knn.train(features1, labels1);
	GRowDistanceScaled* pMetric = knn.metric();

	// Merge the features into one dataset and build a kd-tree
	GMatrix both(features1.relation());
	GReleaseDataHolder hBoth(&both);
	both.reserve(features1.rows() + features2.rows());
	for(size_t i = 0; i < features1.rows(); i++)
		both.takeRow(features1[i]);
	for(size_t i = 0; i < features2.rows(); i++)
		both.takeRow(features2[i]);
	GRowDistanceScaled metric2;
	GKdTree neighborFinder(&both, m_neighborCount, &metric2, false);
	GVec::copy(metric2.scaleFactors(), pMetric->scaleFactors(), features1.cols());

	// Transduce
	GMatrix* pOut = new GMatrix(labels1.relation());
	Holder<GMatrix> hOut(pOut);
	pOut->newRows(features2.rows());
	pOut->setAll(0);
	for(size_t lab = 0; lab < labels1.cols(); lab++)
	{
		// Use max-flow/min-cut graph-cut to separate out each label value
		int valueCount = (int)labels1.relation()->valueCount(lab);
		for(int val = 1; val < valueCount; val++)
		{
			// Add neighborhood edges
			GGraphCut gc(features1.rows() + features2.rows() + 2);
			for(size_t i = 0; i < both.rows(); i++)
			{
				neighborFinder.neighbors(m_pNeighbors, m_pDistances, i);
				for(size_t j = 0; j < m_neighborCount; j++)
				{
					if(m_pNeighbors[j] >= both.rows())
						continue;
					gc.addEdge(2 + i, 2 + m_pNeighbors[j], (float)(1.0 / std::max(sqrt(m_pDistances[j]), 1e-9))); // connect neighbors
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
	return hOut.release();
}


