/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GCLUSTER_H__
#define __GCLUSTER_H__

#include "GTransform.h"
#include "GLearner.h"
#include <deque>

namespace GClasses {

class GDissimilarityMetric;
class GNeuralNet;
class GSparseSimilarity;

/// The base class for clustering algorithms. Classes that inherit from this
/// class must implement a method named "cluster" which performs clustering, and
/// a method named "whichCluster" which reports which cluster the specified row
/// is determined to be a member of.
class GClusterer : public GTransform
{
protected:
	size_t m_clusterCount;

public:
	GClusterer(size_t nClusterCount)
	: GTransform(), m_clusterCount(nClusterCount)
	{
	}

	virtual ~GClusterer()
	{
	}

	/// Clusters pIn and outputs a dataset with one column that specifies
	/// the cluster number for each row.
	virtual GMatrix* doit(GMatrix* pIn)
	{
		cluster(pIn);
		sp_relation pRel = new GUniformRelation(1, m_clusterCount);
		GMatrix* pOut = new GMatrix(pRel);
		size_t nCount = pIn->rows();
		pOut->newRows(nCount);
		for(size_t i = 0; i < nCount; i++)
			pOut->row(i)[0] = (double)whichCluster(i);
		return pOut;
	}

	/// Performs clustering.
	virtual void cluster(GMatrix* pData) = 0;

	/// Reports which cluster the specified row is a member of.
	virtual size_t whichCluster(size_t nVector) = 0;
};



/// This is a base class for clustering algorithms that operate on sparse matrices
class GSparseClusterer
{
protected:
	size_t m_clusterCount;

public:
	GSparseClusterer(size_t clusterCount)
	: m_clusterCount(clusterCount)
	{
	}

	virtual ~GSparseClusterer()
	{
	}

	/// Return the number of clusters
	size_t clusterCount() { return m_clusterCount; }

	/// Perform clustering.
	virtual void cluster(GSparseMatrix* pData) = 0;

	/// Report which cluster the specified row is a member of.
	virtual size_t whichCluster(size_t nVector) = 0;
};




/// This merges each cluster with its closest neighbor. (The distance between
/// clusters is computed as the distance between the closest members of the
/// clusters times (n^b), where n is the total number of points from both
/// clusters, and b is a balancing factor.
class GAgglomerativeClusterer : public GClusterer
{
protected:
	GDissimilarityMetric* m_pMetric;
	bool m_ownMetric;
	size_t* m_pClusters;

public:
	GAgglomerativeClusterer(size_t nClusterCount);
	virtual ~GAgglomerativeClusterer();

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !NO_TEST_CODE

	/// Performs clustering
	virtual void cluster(GMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);

	/// Specify the metric to use to determine the distance between points.
	/// If own is true, then this object will take care to delete pMetric.
	void setMetric(GDissimilarityMetric* pMetric, bool own);
};


/// This is a semi-supervised agglomerative clusterer. It can only handle
/// one output, and it must be nominal. All inputs must be continuous. Also,
/// it assumes that all output values are represented in the training set.
class GAgglomerativeTransducer : public GTransducer
{
protected:
	GDissimilarityMetric* m_pMetric;
	bool m_ownMetric;

public:
	GAgglomerativeTransducer();
	virtual ~GAgglomerativeTransducer();

	/// Specify the metric to use to determine the distance between points.
	/// If own is true, then this object will take care to delete pMetric.
	void setMetric(GDissimilarityMetric* pMetric, bool own);

	/// See the comment for GTransducer::transduce.
	/// Throws if labels1 has more than one column.
	virtual GMatrix* transduce(GMatrix& features1, GMatrix& labels1, GMatrix& features2);
};


/// An implementation of the K-means clustering algorithm.
class GKMeans : public GClusterer
{
protected:
	size_t m_nDims;
	GMatrix* m_pData;
	size_t m_nClusters;
	size_t* m_pClusters;
	GRand* m_pRand;

public:
	GKMeans(size_t nClusters, GRand* pRand);
	~GKMeans();

	/// Performs clustering
	virtual void cluster(GMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);

protected:
	bool clusterAttempt(size_t nMaxIterations);
	bool selectSeeds(GMatrix* pSeeds);
};


/// An implementation of the K-medoids clustering algorithm
class GKMedoids : public GClusterer
{
protected:
	size_t* m_pMedoids;
	GDissimilarityMetric* m_pMetric;
	GMatrix* m_pData;
	double m_d;

public:
	GKMedoids(size_t clusters);
	virtual ~GKMedoids();

	/// Takes ownership of pMetric
	void setMetric(GDissimilarityMetric* pMetric);

	/// Performs clustering
	virtual void cluster(GMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);

protected:
	double curErr();
};



/// An implementation of the K-medoids clustering algorithm for sparse data
class GKMedoidsSparse : public GSparseClusterer
{
protected:
	size_t* m_pMedoids;
	GSparseSimilarity* m_pMetric;
	bool m_ownMetric;
	GSparseMatrix* m_pData;
	double m_d;

public:
	GKMedoidsSparse(size_t clusters);
	virtual ~GKMedoidsSparse();

	/// If own is true, then this takes ownership of pMetric
	void setMetric(GSparseSimilarity* pMetric, bool own);

	/// Performs clustering
	virtual void cluster(GSparseMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);

protected:
	double curGoodness();
};



/// An implementation of the K-means clustering algorithm.
class GKMeansSparse : public GSparseClusterer
{
protected:
	size_t m_nDims;
	size_t m_nClusters;
	size_t* m_pClusters;
	GRand* m_pRand;
	GSparseSimilarity* m_pMetric;
	bool m_ownMetric;

public:
	GKMeansSparse(size_t nClusters, GRand* pRand);
	~GKMeansSparse();

	/// If own is true, then this takes ownership of pMetric
	void setMetric(GSparseSimilarity* pMetric, bool own);

	/// Performs clustering
	virtual void cluster(GSparseMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);
};



/// A transduction algorithm that uses a max-flow/min-cut graph-cut algorithm
/// to partition the data until each class is in a separate cluster. Unlabeled points
/// are then assigned the label of the cluster in which they fall.
class GGraphCutTransducer : public GTransducer
{
protected:
	size_t m_neighborCount;
	GRand* m_pRand;
	size_t* m_pNeighbors;
	double* m_pDistances;

public:
	GGraphCutTransducer(size_t neighborCount, GRand* pRand);
	virtual ~GGraphCutTransducer();

	/// See the comment for GTransducer::transduce.
	/// Only supports one-dimensional labels.
	virtual GMatrix* transduce(GMatrix& features1, GMatrix& labels1, GMatrix& features2);
};


/*
class GNeuralTransducer : public GTransducer
{
protected:
	GRand* m_pRand;
	GNeuralNet* m_pNN;
	std::vector<size_t> m_paramRanges;

public:
	GNeuralTransducer(GRand* pRand);
	virtual ~GNeuralTransducer();
	GNeuralNet* neuralNet() { return m_pNN; }

	void setParams(std::vector<size_t>& ranges);

	/// See the comment for GTransducer::transduce.
	/// labelDims must be 1
	virtual void transduce(GMatrix* pDataLabeled, GMatrix* pDataUnlabeled, size_t labelDims);
};
*/
} // namespace GClasses

#endif // __GCLUSTER_H__
