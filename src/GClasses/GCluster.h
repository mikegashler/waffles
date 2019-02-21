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

#ifndef __GCLUSTER_H__
#define __GCLUSTER_H__

#include "GTransform.h"
#include "GLearner.h"
#include <deque>

namespace GClasses {

class GDistanceMetric;
class GSparseSimilarity;

/// The base class for clustering algorithms. Classes that inherit from this
/// class must implement a method named "cluster" which performs clustering, and
/// a method named "whichCluster" which reports which cluster the specified row
/// is determined to be a member of.
class GClusterer : public GTransform
{
protected:
	size_t m_clusterCount;
	GDistanceMetric* m_pMetric;
	bool m_ownMetric;

public:
	GClusterer(size_t nClusterCount);
	virtual ~GClusterer();

	/// If own is true, then this object will delete pMetric when it is destroyed.
	void setMetric(GDistanceMetric* pMetric, bool own);

	/// Return the number of clusters
	size_t clusterCount() { return m_clusterCount; }

	/// Clusters pIn and outputs a dataset with one column that specifies
	/// the cluster number for each row.
	virtual GMatrix* reduce(const GMatrix& in)
	{
		cluster(&in);
		GMatrix* pOut = new GMatrix(new GUniformRelation(1, m_clusterCount));
		size_t nCount = in.rows();
		pOut->newRows(nCount);
		for(size_t i = 0; i < nCount; i++)
			pOut->row(i)[0] = (double)whichCluster(i);
		return pOut;
	}

	/// Performs clustering.
	virtual void cluster(const GMatrix* pData) = 0;

	/// Reports which cluster the specified row is a member of.
	virtual size_t whichCluster(size_t nVector) = 0;
};



/// This is a base class for clustering algorithms that operate on sparse matrices
class GSparseClusterer
{
protected:
	size_t m_clusterCount;
	GSparseSimilarity* m_pMetric;
	bool m_ownMetric;

public:
	GSparseClusterer(size_t clusterCount);
	virtual ~GSparseClusterer();

	/// Return the number of clusters
	size_t clusterCount() { return m_clusterCount; }

	/// Perform clustering.
	virtual void cluster(GSparseMatrix* pData) = 0;

	/// Report which cluster the specified row is a member of.
	virtual size_t whichCluster(size_t nVector) = 0;

	/// If own is true, then this takes ownership of pMetric
	void setMetric(GSparseSimilarity* pMetric, bool own);
};




/// This merges each cluster with its closest neighbor. (The distance between
/// clusters is computed as the distance between the closest members of the
/// clusters times (n^b), where n is the total number of points from both
/// clusters, and b is a balancing factor.
class GAgglomerativeClusterer : public GClusterer
{
protected:
	size_t* m_pClusters;

public:
	GAgglomerativeClusterer(size_t nClusterCount);
	virtual ~GAgglomerativeClusterer();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Performs clustering
	virtual void cluster(const GMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);
};


/// This is a semi-supervised agglomerative clusterer. It can only handle
/// one output, and it must be nominal. All inputs must be continuous. Also,
/// it assumes that all output values are represented in the training set.
class GAgglomerativeTransducer : public GTransducer
{
protected:
	GDistanceMetric* m_pMetric;
	bool m_ownMetric;

public:
	GAgglomerativeTransducer();
	virtual ~GAgglomerativeTransducer();

	/// Specify the metric to use to determine the distance between points.
	/// If own is true, then this object will take care to delete pMetric.
	void setMetric(GDistanceMetric* pMetric, bool own);

	/// This model has no parameters to tune, so this method is a noop.
	void autoTune(GMatrix& features, GMatrix& labels);

protected:
	/// See the comment for GTransducer::transduce.
	/// Throws if labels1 has more than one column.
	virtual std::unique_ptr<GMatrix> transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2);

	/// See the comment for GTransducer::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }
};


/// An implementation of the K-means clustering algorithm.
class GKMeans : public GClusterer
{
protected:
	GMatrix* m_pCentroids;
	size_t* m_pClusters;
	size_t m_reps;
	GRand* m_pRand;

public:
	GKMeans(size_t nClusters, GRand* pRand);
	~GKMeans();

	/// Performs clustering
	virtual void cluster(const GMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);

	/// Selects random centroids and initializes internal data structures
	void init(const GMatrix* pData);

	/// Assigns each row to the cluster of the nearest centroid as measured
	/// with the dissimilarity metric. Returns the sum-squared-distance of each row with its centroid.
	double assignClusters(const GMatrix* pData);

	/// Computes new centroids for each cluster.
	void recomputeCentroids(const GMatrix* pData);

	/// Returns a k x d matrix, where each row is one of the k centroids.
	GMatrix* centroids() { return m_pCentroids; }

	/// Specify the number of times to cluster the data. The best clustering (as measured
	/// by the sum-squared-difference between each point and its cluster-centroid) will be kept.
	void setReps(size_t r) { m_reps = r; }

protected:
	bool clusterAttempt(size_t nMaxIterations);
	bool selectSeeds(const GMatrix* pSeeds);
};


/// A K-means clustering algorithm where every point has partial membership in each cluster.
/// This algorithm is specified in Li, D. and Deogun, J. and Spaulding, W. and Shuart, B.,
/// Towards missing data imputation: A study of fuzzy K-means clustering method, In Rough Sets
/// and Current Trends in Computing, Springer, pages 573--579, 2004.
class GFuzzyKMeans : public GClusterer
{
protected:
	GMatrix* m_pCentroids;
	GMatrix* m_pWeights;
	double m_fuzzifier;
	size_t m_reps;
	GRand* m_pRand;

public:
	GFuzzyKMeans(size_t nClusters, GRand* pRand);
	~GFuzzyKMeans();

	/// Performs clustering
	virtual void cluster(const GMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);

	/// Selects random centroids and initializes internal data structures
	void init(const GMatrix* pData);

	/// Assigns each row to partial membership in each cluster, as measured
	/// with the dissimilarity metric. Returns the weighted-sum-distance of each row with the centroids.
	double recomputeWeights(const GMatrix* pData);

	/// Computes new centroids for each cluster.
	void recomputeCentroids(const GMatrix* pData);

	/// Returns a k x d matrix, where each row is one of the k centroids.
	GMatrix* centroids() { return m_pCentroids; }

	/// Specifies how fuzzy the membership in each cluster should be. d should be
	/// greater than 1, and is typically about 1.3.
	void setFuzzifier(double d) { m_fuzzifier = d; }

	/// Specify the number of times to cluster the data. The best clustering (as measured
	/// by the weighted-sum-difference between each point with the centroids) will be kept.
	void setReps(size_t r) { m_reps = r; }

protected:
	bool clusterAttempt(size_t nMaxIterations);
	bool selectSeeds(const GMatrix* pSeeds);
};


/// An implementation of the K-medoids clustering algorithm
class GKMedoids : public GClusterer
{
protected:
	size_t* m_pMedoids;
	double m_d;
	const GMatrix* m_pData;

public:
	GKMedoids(size_t clusters);
	virtual ~GKMedoids();

	/// Performs clustering
	virtual void cluster(const GMatrix* pData);

	/// Identifies the cluster of the specified row
	virtual size_t whichCluster(size_t nVector);

protected:
	double curErr(const GMatrix* pData);
};



/// An implementation of the K-medoids clustering algorithm for sparse data
class GKMedoidsSparse : public GSparseClusterer
{
protected:
	size_t* m_pMedoids;
	double m_d;
	GSparseMatrix* m_pData;

public:
	GKMedoidsSparse(size_t clusters);
	virtual ~GKMedoidsSparse();

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

public:
	GKMeansSparse(size_t nClusters, GRand* pRand);
	~GKMeansSparse();

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

public:
	GGraphCutTransducer();
	virtual ~GGraphCutTransducer();

	/// Sets the number of neighbors to use to form the graph. The default is 12
	void setNeighbors(size_t k);

	/// Returns the number of neighbors to which each point is connected
	size_t neighbors() { return m_neighborCount; }

	/// Uses cross-validation to find a set of parameters that works well with
	/// the provided data.
	void autoTune(GMatrix& features, GMatrix& labels);

protected:
	/// See the comment for GTransducer::transduce.
	/// Only supports one-dimensional labels.
	virtual std::unique_ptr<GMatrix> transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2);

	/// See the comment for GTransducer::canImplicitlyHandleContinuousLabels
	virtual bool canImplicitlyHandleContinuousLabels() { return false; }
};


} // namespace GClasses

#endif // __GCLUSTER_H__
