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

#ifndef __GNEIGHBORFINDER_H__
#define __GNEIGHBORFINDER_H__

#include "GMatrix.h"
#include <vector>
#include <map>

namespace GClasses {

class GMatrix;
class GRelation;
class GRand;
class GKdNode;
class GBallNode;
class GBitTable;
class GDistanceMetric;
class GSupervisedLearner;
class GRandomIndexIterator;
class GSparseMatrix;
class GSparseSimilarity;
class GNeighborFinderGeneralizing;


/// Finds the k-nearest neighbors of any vector in a dataset.
class GNeighborFinder
{
protected:
	const GMatrix* m_pData;

public:
	GNeighborFinder(const GMatrix* pData)
	: m_pData(pData)
	{
	}

	virtual ~GNeighborFinder()
	{
	}

	/// Returns the data passed to the constructor of this object
	const GMatrix* data() { return m_pData; }

	/// Returns true if this neighbor finder can operate on points that
	/// are not in the dataset passed to the constructor
	virtual bool canGeneralize() { return false; }

	/// Returns true iff the neighbors and distances are pre-computed
	virtual bool isCached() { return false; }

	/// Finds the k-nearest neighbors of the specified point index.
	/// Returns the number of neighbors found.
	/// Call "neighbor" or "distance" to obtain the neighbors and distances that were found.
	virtual size_t findNearest(size_t k, size_t pointIndex) = 0;

	/// Finds all neighbors of the specified point index within a specified radius.
	/// Returns the number of neighbors found.
	/// Call "neighbor" or "distance" to obtain the neighbors and distances that were found.
	virtual size_t findWithinRadius(double squaredRadius, size_t pointIndex) = 0;

	/// Returns the point index of the ith neighbor of the last point passed to "findNearest".
	/// (Behavior is undefined if findNearest has not yet been called.)
	virtual size_t neighbor(size_t i) = 0;

	/// Returns the distance to the ith neighbor of the last point passed to "findNearest".
	/// (Behavior is undefined if findNearest has not yet been called.)
	virtual double distance(size_t i) = 0;
};




/// This wraps a neighbor finding algorithm. It caches the queries for neighbors
/// for the purpose of improving runtime performance.
class GNeighborGraph : public GNeighborFinder
{
protected:
	GNeighborFinder* m_pNF;
	bool m_own;
	size_t m_focus;
	std::vector<std::vector<size_t> > m_neighs;
	std::vector<std::vector<double> > m_dists;

public:
	/// Makes a GNeighborGraph that has precomputed the k-nearest neighbors of each point in a dataset.
	/// If own is true, then this will take ownership of pNF
	GNeighborGraph(GNeighborFinder* pNF, bool own, size_t neighbors);

	/// Makes a GNeighborGraph that has precomputed the neighbors of each point in a dataset within a specified radius.
	GNeighborGraph(double squaredRadius, GNeighborFinder* pNF, bool own);

	/// Makes a GNeighborGraph assuming the data represents a sequence of observations.
	/// First, it computes the distance between each point and its previous and next points.
	/// Then, it finds all neighbors within a radius of the maximum of those two distances.
	GNeighborGraph(bool own, GNeighborFinderGeneralizing* pNF);
	
	virtual ~GNeighborGraph();

	/// See the comment for GNeighborFinder::findNearest
	virtual size_t findNearest(size_t k, size_t pointIndex) { m_focus = pointIndex; return m_neighs[pointIndex].size(); }

	/// See the comment for GNeighborFinder::findWithinRadius
	virtual size_t findWithinRadius(double squaredRadius, size_t pointIndex) { m_focus = pointIndex; return m_neighs[pointIndex].size(); }

	/// See the comment for GNeighborFinder::neighbor
	virtual size_t neighbor(size_t i) { GAssert(i < m_neighs[m_focus].size()); return m_neighs[m_focus][i]; }

	/// See the comment for GNeighborFinder::distance
	virtual double distance(size_t i) { GAssert(i < m_dists[m_focus].size()); return m_dists[m_focus][i]; }

	/// See the comment for GNeighborFinder::isCached.
	virtual bool isCached() { return true; }

	/// Returns a pointer to the neighbor finder that this wraps.
	GNeighborFinder* wrappedNeighborFinder() { return m_pNF; }

	/// Uses CycleCut to remove shortcut connections. (Assumes fillCache has already been called.)
	size_t cutShortcuts(size_t cycleLen);

	/// recomputes all neighbor distances using the specified metric.
	void recomputeDistances(GDistanceMetric* pMetric);

	/// Uses pNewData for subsequent calls to recomputeDistances.
	void swapInData(const GMatrix* pNewData);

	/// Returns true iff the neighbors form a connected graph when each neighbor
	/// is evaluated as a bi-directional edge. (Assumes that fillCache has already been called.)
	bool isConnected();

	/// Sets the specified neighbor. (Does not change the distance.)
	/// This method is used by CycleCut. It is probably not useful for any other purpose.
	void set(size_t point, size_t neighbor_number, size_t neighbor);

	/// Drops all neighbors that have been set to INVALID_INDEX.
	/// This method is used by CycleCut. It is probably not useful for any other purpose.
	void dropInvalidNeighbors();

protected:
	/// Fills the cache with the k-nearest neighbors of each point.
	void fillCacheNearest(size_t k);

	/// Fills the cache with the neighbors of each point within a specified radius.
	void fillCacheRadius(double squaredRadius);
};



/// Finds the k-nearest neighbors (in a dataset) of an arbitrary vector (which may or may not
/// be in the dataset).
class GNeighborFinderGeneralizing : public GNeighborFinder
{
protected:
	GDistanceMetric* m_pMetric;
	bool m_ownMetric;
	std::vector<size_t> m_neighs;
	std::vector<double> m_dists;

public:
	/// Create a neighborfinder for finding the neighborCount
	/// nearest neighbors under the given metric.  If ownMetric is
	/// true, then the neighborFinder takes responsibility for
	/// deleting the metric, otherwise it is the caller's
	/// responsibility.
	GNeighborFinderGeneralizing(const GMatrix* pData, GDistanceMetric* pMetric = NULL, bool ownMetric = false);

	virtual ~GNeighborFinderGeneralizing();

	/// Returns true. See the comment for GNeighborFinder::canGeneralize.
	virtual bool canGeneralize() { return true; }

	/// If you make major changes, you can call this to tell it to rebuild
	/// any optimization structures.
	virtual void reoptimize() = 0;

	/// Finds the nearest neighbors of the specified vector.
	/// Returns the number of neighbors found.
	/// Call "neighbor" or "distance" to obtain the neighbors and distances that were found.
	virtual size_t findNearest(size_t k, const GVec& vector) = 0;
	using GNeighborFinder::findNearest;

	/// Finds all neighbors of the specified vector within a specified radius.
	/// Returns the number of neighbors found.
	/// Call "neighbor" or "distance" to obtain the neighbors and distances that were found.
	virtual size_t findWithinRadius(double squaredRadius, const GVec& vector) = 0;
	using GNeighborFinder::findWithinRadius;

	/// See the comment for GNeighborFinder::neighbor
	virtual size_t neighbor(size_t i) { return m_neighs[i]; }

	/// See the comment for GNeighborFinder::distance
	virtual double distance(size_t i) { return m_dists[i]; }

	/// Returns the metric
	GDistanceMetric *metric() { return m_pMetric; }

	/// Uses Quick Sort to sort the neighbors from least to most distant.
	void sortNeighbors(size_t start = 0, size_t end = INVALID_INDEX);

protected:
	/// A helper method used by sortNeighbors when the remaining portion to sort is small.
	void insertionSortNeighbors(size_t start, size_t end);
};


/// Finds neighbors by measuring the distance to all points. This one should work properly even if
/// the distance metric does not support the triangle inequality.
class GBruteForceNeighborFinder : public GNeighborFinderGeneralizing
{
public:
	GBruteForceNeighborFinder(GMatrix* pData, GDistanceMetric* pMetric = NULL, bool ownMetric = false);
	virtual ~GBruteForceNeighborFinder();

	/// This is a no-op method in this class.
	virtual void reoptimize();

	/// See the comment for GNeighborFinder::findNearest
	virtual size_t findNearest(size_t k, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::neighbors
	virtual size_t findNearest(size_t k, const GVec& vector);

	/// See the comment for GNeighborFinder::findWithinRadius
	virtual size_t findWithinRadius(double squaredRadius, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::findWithinRadius
	virtual size_t findWithinRadius(double squaredRadius, const GVec& vector);

protected:
	size_t findNearest(size_t k, const GVec& vec, size_t exclude);
	size_t findWithinRadius(double squaredRadius, const GVec& vec, size_t exclude);
};




/// Finds neighbors by measuring the distance to all points using a sparse distance metric.
class GSparseNeighborFinder : public GNeighborFinderGeneralizing
{
protected:
	GSparseMatrix* m_pData;
	GSparseSimilarity* m_pSparseMetric;
	bool m_ownSparseMetric;

public:
	/// pData is the sparse dataset in which you want to find neighbors.
	/// pBogusData must be a pointer to a valid dense dataset that will be ignored. (Obviously, this is a hack that should be cleaned up.)
	/// neighborCount is the number of neighbors that you want to find.
	/// pMetric is the similarity metric to use in finding neighbors. Higher similarity indicates closer neighbors.
	/// ownMetric specifies whether this object should delete pMetric when it is deleted.
	GSparseNeighborFinder(GSparseMatrix* pData, GMatrix* pBogusData, GSparseSimilarity* pMetric, bool ownMetric = false);
	virtual ~GSparseNeighborFinder();

	/// This is a no-op method in this class.
	virtual void reoptimize();

	/// See the comment for GNeighborFinder::findNearest
	virtual size_t findNearest(size_t k, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::findNearest
	virtual size_t findNearest(size_t k, const GVec& vector);

	/// See the comment for GNeighborFinder::findWithinRadius
	/// Currently, this method just throws an exception because it is not yet implemented.
	virtual size_t findWithinRadius(double squaredRadius, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::findWithinRadius
	/// Currently, this method just throws an exception because it is not yet implemented.
	virtual size_t findWithinRadius(double squaredRadius, const GVec& vector);
};




/// An efficient algorithm for finding neighbors.
class GKdTree : public GNeighborFinderGeneralizing
{
protected:
	size_t m_maxLeafSize;
	size_t m_size;
	GKdNode* m_pRoot;

public:
	GKdTree(const GMatrix* pData, GDistanceMetric* pMetric = NULL, bool ownMetric = false);
	virtual ~GKdTree();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Rebuilds the tree to improve subsequent performance. This should be called after
	/// a significant number of point-vectors are added to or released from the internal set.
	virtual void reoptimize();

	/// See the comment for GNeighborFinder::findNearest
	virtual size_t findNearest(size_t k, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::neighbors
	virtual size_t findNearest(size_t k, const GVec& vector);

	/// See the comment for GNeighborFinder::findWithinRadius
	size_t findWithinRadius(double squaredRadius, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::findWithinRadius
	size_t findWithinRadius(double squaredRadius, const GVec& vector);

	/// Specify the max number of point-vectors to store in each leaf node.
	void setMaxLeafSize(size_t n) { m_maxLeafSize = n; }

	/// Returns the root node of the kd-tree.
	GKdNode* root() { return m_pRoot; }

	/// Build the tree
	GKdNode* buildTree(size_t count, size_t* pIndexes);

	/// Returns true iff the specified point-vector is on the >= side of the specified pivot
	bool isGreaterOrEqual(const double* pPat, size_t attr, double pivot);

	/// Computes the median distance to the n^th closest neighbor of each row in data
	static double medianDistanceToNeighbor(GMatrix& data, size_t n);

protected:
	/// This is a helper method that finds the nearest neighbors
	size_t findNearest(size_t k, const GVec& vec, size_t nExclude);

	/// This is a helper method that finds neighbors within a radius
	size_t findWithinRadius(double squaredRadius, const GVec& vec, size_t nExclude);

	/// Computes a good pivot for the specified attribute, and the goodness of splitting on
	/// that attribute. For continuous attributes, the pivot is the (not scaled) mean and the goodness is
	/// the scaled variance. For nominal attributes, the pivot is the most common value and the
	/// goodness is scaled entropy.
	void computePivotAndGoodness(size_t count, size_t* pIndexes, size_t attr, double* pOutPivot, double* pOutGoodness);

	/// Moves all the indexes that refer to rows that have a value less than pivot in
	/// the specified attribute to the beginning of the list, and the rest to the end. Returns
	/// the number of rows with a value less than the pivot. For nominal values, not-equal
	/// values are moved to the beginning, and equal values are moved to the end.
	size_t splitIndexes(size_t count, size_t* pIndexes, size_t attr, double pivot);
};




/// An efficient algorithm for finding neighbors. Empirically, this class seems to be a little bit slower than GKdTree.
class GBallTree : public GNeighborFinderGeneralizing
{
protected:
	size_t m_maxLeafSize;
	size_t m_size;
	GBallNode* m_pRoot;

public:
	GBallTree(const GMatrix* pData, GDistanceMetric* pMetric = NULL, bool ownMetric = false);
	virtual ~GBallTree();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Rebuilds the tree to improve subsequent performance. This should be called after
	/// a significant number of point-vectors are added to or released from the internal set.
	virtual void reoptimize();

	/// See the comment for GNeighborFinder::findNearest
	virtual size_t findNearest(size_t k, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::neighbors
	virtual size_t findNearest(size_t k, const GVec& vec);

	/// See the comment for GNeighborFinder::findWithinRadius
	virtual size_t findWithinRadius(double squaredRadius, size_t index);

	/// See the comment for GNeighborFinderGeneralizing::findWithinRadius
	virtual size_t findWithinRadius(double squaredRadius, const GVec& vec);

	/// Specify the max number of point-vectors to store in each leaf node.
	void setMaxLeafSize(size_t n) { m_maxLeafSize = n; }

	/// Inserts a new point into this ball tree. This method assumes you have already added a new
	/// row to the dataset that was used to construct this tree. Calling this method informs this structure
	/// that it should also index the new point. Note that this method may reduce the efficiency
	/// of the tree by a small amount, so you might want to call reoptimize after several points are added.
	void insert(size_t index);

	/// Drops the specified index from this ball tree. Throws an exception of the specified index is not found
	/// in the tree. Note that the tree still assumes that the other
	/// indexes still retain their relationship with the points in the dataset that was used to construct
	/// this object, so you should not move the rows in that dataset around. Also note that before
	/// you call reoptimize, you need to delete any rows that were dropped, or else they will then be added back in.
	void drop(size_t index);

	/// Drops all of the leaf point indexes, but retains the interior structure. (This might be useful
	/// if you know in advance which points will be inserted, but you don't want them to be in the tree yet.)
	void dropAll();

protected:
	/// Build the tree
	GBallNode* buildTree(size_t count, size_t* pIndexes);

	/// This is a helper method that finds the nearest neighbors
	size_t findNearest(size_t k, const GVec& vec, size_t nExclude);

	/// This is a helper method that finds the neighbors within a specified radius
	size_t findWithinRadius(double squaredRadius, const GVec& vec, size_t nExclude);
};





/// This uses "betweeenness centrality" to find the shortcuts in a table of neighbors and replaces them with INVALID_INDEX.
class GShortcutPruner
{
protected:
	size_t* m_pNeighborhoods;
	size_t m_n;
	size_t m_k;
	size_t m_cycleThresh;
	size_t m_subGraphRange;
	size_t m_cuts;

public:
	/// pNeighborMap is expected to be an array of size n*k, where n is the
	/// number of points, and k is the number of neighbors.
	GShortcutPruner(size_t* pNeighborhoods, size_t n, size_t k);
	~GShortcutPruner();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Sets the cycle-length threshold. (The default is 14.)
	void setCycleThreshold(size_t cycleThresh) { m_cycleThresh = cycleThresh; }

	/// Sets the sub graph range. (The default is 6.)
	void setSubGraphRange(size_t range) { m_subGraphRange = range; }

	/// Do the pruning. Returns the number of shortcuts that were removed.
	/// Any atomic cycles in the graph (where neighbors are treated as bi-directional)
	/// with a cycle-length of cycleThresh or bigger indicates the existence of a shortcut
	/// that must be cut. To determine which edge in the cycle is the shortcut, it will
	/// make a subgraph containing all nodes withing "subGraphRange" hops of any vertex
	/// in the cycle, and compute the betweenness centrality of every edge in the sub-graph.
	/// The edge on the cycle with the largest betweenness is determed to be the shortcut,
	/// and is replaced with INVALID_INDEX.
	size_t prune();

	/// Internal method
	void onDetectBigAtomicCycle(std::vector<size_t>& cycle);

protected:
	bool isEveryNodeReachable();
};





/// This finds the shortcuts in a table of neighbors and replaces them with INVALID_INDEX.
class GCycleCut
{
protected:
	GNeighborGraph* m_pNeighborGraph;
	const GMatrix* m_pPoints;
	std::map<std::pair<size_t, size_t>, double> m_capacities;
	std::vector<size_t> m_cuts;
	size_t m_k;
	size_t m_cycleThresh;
	double m_aveDist;
	size_t m_cutCount;

public:
	/// pNeighborMap is expected to be an array of size n*k, where n is the
	/// number pPoints->rows(), and k is the number of neighbors.
	GCycleCut(GNeighborGraph* pNeighborGraph, const GMatrix* pPoints, size_t k);
	~GCycleCut();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Sets the cycle-length threshold. (The default is 14.)
	void setCycleThreshold(size_t cycleThresh) { m_cycleThresh = cycleThresh; }

	/// Do the cutting. Returns the number of edges that were removed.
	/// Any atomic cycles in the graph (where neighbors are treated as bi-directional)
	/// with a cycle-length of cycleThresh or bigger will be cut. It will make the
	/// smallest cut that removes all big atomic cycles
	size_t cut();

	/// Internal method
	void onDetectBigAtomicCycle(std::vector<size_t>& cycle);

protected:
	bool doAnyBigAtomicCyclesExist();
};



} // namespace GClasses

#endif // __GNEIGHBORFINDER_H__
