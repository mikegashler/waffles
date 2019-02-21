/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
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

#ifndef __GMANIFOLD_H__
#define __GMANIFOLD_H__

#include "GTransform.h"
#include "GError.h"
#include "GRand.h"
#include <vector>
#include <deque>
#include "GVec.h"

namespace GClasses {

struct GManifoldSculptingNeighbor;
class GNeighborFinder;
class GNeighborFinderGeneralizing;
class GNeuralNetLearner;
class GNeuralNetLayer;
class GNeighborGraph;


/// This class stores static methods that are useful for manifold learning
class GManifold
{
public:
	/// Computes a set of weights for each neighbor to linearly approximate this point
	static void computeNeighborWeights(const GMatrix* pData, size_t point, size_t k, const size_t* pNeighbors, double* pOutWeights);

	/// Aligns and averages two local neighborhoods together. The results will be
	/// centered around the neighborhood mean. The first point will be
	/// the index point, and the rest will be neighborhood points with
	/// an index that is not INVALID_INDEX.
	static GMatrix* blendNeighborhoods(size_t index, GMatrix* pA, double ratio, GMatrix* pB, size_t neighborCount, GNeighborFinder* pNF);

	/// Combines two embeddings to form an "average" embedding. pRatios is an array that specifies how much to weight the
	/// neighborhoods around each point. If the ratio for a point is close to zero, pA will be emphasized more. If
	/// the ratio for the point is close to 1, pB will be emphasized more. "seed" specifies a starting point. It will
	/// blend outward in a breadth-first manner.
	static GMatrix* blendEmbeddings(GMatrix* pA, double* pRatios, GMatrix* pB, size_t neighborCount, GNeighborGraph* pNeighborTable, size_t seed);

	/// Performs classic MDS. pDistances must be a square matrix, but only the upper-triangle is used.
	/// Each row in the results is one of the result points.
	/// If useSquaredDistances is true, then the values in pDistances are assumed to be squared
	/// distances, rather than normal Euclidean distances.
	static GMatrix* multiDimensionalScaling(GMatrix* pDistances, size_t targetDims, GRand* pRand, bool useSquaredDistances);

	static void test();
};



/// Manifold Sculpting. A non-linear dimensionality reduction algorithm.
/// (See Gashler, Michael S. and Ventura, Dan and Martinez, Tony. Iterative
/// non-linear dimensionality reduction with manifold sculpting. In Advances
/// in Neural Information Processing Systems 20, pages 513–520, MIT Press,
/// Cambridge, MA, 2008.)
class GManifoldSculpting : public GTransform
{
protected:
	size_t m_nDimensions;
	size_t m_nNeighbors;
	size_t m_nStuffIndex;
	size_t m_nRecordSize;
	size_t m_nCurrentDimension;
	size_t m_nTargetDims;
	size_t m_nPass;
	double m_scale;
	double m_dAveNeighborDist;
	double m_dSquishingRate;
	double m_dLearningRate;
	double m_minNeighborDist, m_maxNeighborDist;
	std::deque<size_t> m_q;
	GRelation* m_pRelationAfter;
	GRand* m_pRand;
	GMatrix* m_pData;
	unsigned char* m_pMetaData;
	GNeighborFinderGeneralizing* m_pNF;

public:
	GManifoldSculpting(size_t nNeighbors, size_t targetDims, GRand* pRand);
	virtual ~GManifoldSculpting();

	/// Perform NLDR.
	virtual GMatrix* reduce(const GMatrix& in);

	const GRelation& relationAfter() { return *m_pRelationAfter; }

	GMatrix& data() { return *m_pData; }

	/// Call this before calling SquishPass. pRealSpaceData should be a dataset
	/// of all real values.
	void beginTransform(const GMatrix* pRealSpaceData);

	/// Perform one iteration of squishing. Returns a heuristical error value
	double squishPass(size_t nSeedDataPoint);

	/// Set the rate of squishing. (.99 is a good value)
	void setSquishingRate(double d) { m_dSquishingRate = d; }

	/// Returns the current learning rate
	double learningRate() { return m_dLearningRate; }

	/// Returns the average distance between neighbors
	double aveNeighborDist() { return m_dAveNeighborDist; }

	/// Counts the number of times that a point has a neighbor with an
	/// index that is >= nThreshold away from this points index. (If
	/// the manifold is sampled in order such that points are expected
	/// to find neighbors with indexes close to their own, this can serve
	/// to identify when parts of the manifold are too close to other
	/// parts for so many neighbors to be used.)
	size_t countShortcuts(size_t nThreshold);

	/// This will takes ownership of pData. (It will modify pData too.)
	/// Specifies reduced dimensionality data created by another
	/// algorithm, which should be refined using Manifold Sculpting
	void setPreprocessedData(GMatrix* pData);

	/// Neighbors that are closer than min or farther than max will be
	/// ignored. The default is [0, 1e20]. It is common to make nNeighbors
	/// big and max small so that the hyper-sphere will define the neighborhood.
	void setMinAndMaxNeighborDist(double min, double max)
	{
		m_minNeighborDist = min;
		m_maxNeighborDist = max;
	}

	/// Partially supervise the specified point.
	void clampPoint(size_t n);

	/// Specifies to use the neighborhoods determined by the specified neighbor-finder instead of the nearest
	/// Euclidean-distance neighbors. If this method is called, pNF should have the same number of
	/// neighbors and the same dataset as is passed into this class.
	void setNeighborFinder(GNeighborFinderGeneralizing* pNF) { m_pNF = pNF; }

protected:
	inline struct GManifoldSculptingStuff* stuff(size_t n)
	{
		return (struct GManifoldSculptingStuff*)&m_pMetaData[n * m_nRecordSize + m_nStuffIndex];
	}

	inline struct GManifoldSculptingNeighbor* record(size_t n)
	{
		return (struct GManifoldSculptingNeighbor*)&m_pMetaData[n * m_nRecordSize];
	}

	/// You can overload this to add some intelligent supervision to the heuristic
	virtual double supervisedError(size_t nPoint) { return 0; }

	void calculateMetadata(const GMatrix* pData);
	double vectorCorrelation(const double* pdA, const double* pdV, const double* pdB);
	double vectorCorrelation2(double squaredScale, size_t a, size_t vertex, struct GManifoldSculptingNeighbor* pNeighborRec);
	double computeError(size_t nPoint);
	size_t adjustDataPoint(size_t nPoint, double* pError);
	double averageNeighborDistance(size_t nDims);
	void moveMeanToOrigin();
};


/// Isomap is a manifold learning algorithm that uses the Floyd-Warshall algorithm
/// to compute an estimate of the geodesic distance between every pair of points
/// using local neighborhoods, and then uses classic multidimensional scaling to
/// compute a low-dimensional projection.
class GIsomap : public GTransform
{
protected:
	size_t m_neighborCount;
	size_t m_targetDims;
	GNeighborFinder* m_pNF;
	GRand* m_pRand;
	bool m_dropDisconnectedPoints;

public:
	GIsomap(size_t neighborCount, size_t targetDims, GRand* pRand);
	GIsomap(GDomNode* pNode);
	virtual ~GIsomap();

	/// Serializes this object
	GDomNode* serialize(GDom* pDoc) const;

	/// If there are any points that are not connected to the main group, just drop them instead of
	/// throwing an exception. (Note that this may cause the results to contain a different number
	/// of rows than the input.)
	void dropDisconnectedPoints() { m_dropDisconnectedPoints = true; }

	/// Specifies to use the neighborhoods determined by the specified neighbor-finder instead of the nearest
	/// Euclidean-distance neighbors to establish local linearity. If this method is called, it will also
	/// use the number of neighbors and the data associated with pNF, and ignore the number of neighbors
	/// specified to the constructor, and ignore the data passed to the "transform" method.
	void setNeighborFinder(GNeighborFinder* pNF);

	/// Performs NLDR
	virtual GMatrix* reduce(const GMatrix& in);
};


/// Locally Linear Embedding is a manifold learning algorithm that uses
/// sparse matrix techniques to efficiently compute a low-dimensional projection.
class GLLE : public GTransform
{
protected:
	size_t m_neighborCount;
	size_t m_targetDims;
	GNeighborFinder* m_pNF;
	GRand* m_pRand;

public:
	GLLE(size_t neighborCount, size_t targetDims, GRand* pRand);
	GLLE(GDomNode* pNode);
	virtual ~GLLE();

	/// Serialize this object
	GDomNode* serialize(GDom* pDoc) const;

	/// Specifies to use the neighborhoods determined by the specified neighbor-finder instead of the nearest
	/// Euclidean-distance neighbors to establish local linearity. If this method is called, it will also
	/// use the number of neighbors and the data associated with pNF, and ignore the number of neighbors
	/// specified to the constructor, and ignore the data passed to the "transform" method.
	void setNeighborFinder(GNeighborFinder* pNF);

	/// Performs NLDR
	virtual GMatrix* reduce(const GMatrix& in);
};


/// A manifold learning algorithm that reduces dimensionality in local
/// neighborhoods, and then stitches the reduced local neighborhoods together
/// using the Kabsch algorithm.
class GBreadthFirstUnfolding : public GTransform
{
protected:
	size_t m_reps;
	size_t m_neighborCount;
	size_t m_targetDims;
	GNeighborFinder* m_pNF;
	bool m_useMds;
	GRand m_rand;

public:
	/// reps specifies the number of times to compute the embedding, and blend the
	/// results together. If you just want fast results, use reps=1.
	GBreadthFirstUnfolding(size_t reps, size_t neighborCount, size_t targetDims);
	GBreadthFirstUnfolding(GDomNode* pNode);
	virtual ~GBreadthFirstUnfolding();

	/// Serialize this object
	GDomNode* serialize(GDom* pDoc) const;

	/// Specify the neighbor finder to use to pick neighbors for this algorithm
	void setNeighborFinder(GNeighborFinder* pNF);

	/// Perform NLDR
	virtual GMatrix* reduce(const GMatrix& in);

	/// Specify to use multi-dimensional scaling instead of PCA to reduce in local patches.
	void useMds(bool b) { m_useMds = b; }

	/// Returns a reference to the pseudo-random number generator used by this object
	GRand& rand() { return m_rand; }

protected:
	/// Moves "pPoint" so that it is closer to a distance of "distance" from "pNeighbor". "learningRate"
	/// specifies how much to move it (0=not at all, 1=all the way). Returns the squared distance between
	/// pPoint and pNeighbor.
	static double refinePoint(double* pPoint, double* pNeighbor, size_t dims, double distance, double learningRate, GRand* pRand);

	void refineNeighborhood(GMatrix* pLocal, size_t rootIndex, GNeighborGraph* pNeighborGraph);
	GMatrix* reduceNeighborhood(const GMatrix* pIn, size_t index, GNeighborGraph* pNeighborGraph);
	GMatrix* unfold(const GMatrix* pIn, GNeighborGraph* pNeighborGraph, size_t seed, double* pOutWeights);
};





/// This is a nonlinear dimensionality reduction algorithm loosely inspired by
/// Maximum Variance Unfolding. It iteratively scales up the data, then restores
/// distances between neighbors.
class GScalingUnfolder : public GTransform
{
protected:
	size_t m_neighborCount;
	size_t m_targetDims;
	size_t m_passes;
	size_t m_refines_per_scale;
	double m_scaleRate;
	GRand m_rand;

public:
	GScalingUnfolder();
	GScalingUnfolder(GDomNode* pNode);
	virtual ~GScalingUnfolder();

	/// Specify the number of neighbors to use. (The default is 14.)
	void setNeighborCount(size_t n) { m_neighborCount = n; }

	/// Specify the number of dimensions in the output results
	void setTargetDims(size_t n) { m_targetDims = n; }

	/// Specify the number of times to 'scale the data then recover local relationships'.
	void setPasses(size_t n) { m_passes = n; }

	/// Specify the number of times to refine the points after each scaling.
	void setRefinesPerScale(size_t n) { m_refines_per_scale = n; }

	/// Specify the scaling rate. The default is 0.9.
	void setScaleRate(double d) { m_scaleRate = d; }

	/// Returns a reference to the pseudo-random number generator used by this object
	GRand& rand() { return m_rand; }

	/// Reduces the dimensionality of "in".
	virtual GMatrix* reduce(const GMatrix& in);

	/// Unfolds the points in intrinsic, such that distances specified in nf are preserved.
	/// If pVisible is non-NULL, then pEncoder will be incrementally trained to encode pVisible to intrinsic.
	void unfold(GMatrix& intrinsic, GNeighborGraph& nf, size_t encoderTrainIters = 0, GNeuralNetLearner* pEncoder = NULL, GNeuralNetLearner* pDecoder = NULL, const GMatrix* pVisible = NULL);

	/// Perform a single pass over all the edges and attempt to restore local relationships
	static void restore_local_distances_pass(GMatrix& intrinsic, GNeighborGraph& ng, size_t neighborCount, GRand& rand);

	/// A convenience method for iteratively unfolding the manifold sampled by a matrix.
	/// (This method is not used within this class.) It finds the k-nearest neighbors within
	/// the matrix. (If the neighbor-graph is not connected, it doubles k until it is connected.)
	/// Next, this method multiplies the matrix by scaleFactor. Finally, it refines the points to
	/// restore local distances. (It would not typically be efficient to call this in a tight loop,
	/// because that would perform the neighbor-finding step each time.) Returns the number of neighbors
	/// actually used to find a connected graph.
	static size_t unfold_iter(GMatrix& intrinsic, GRand& rand, size_t neighborCount = 12, double scaleFactor = 1.01, size_t refinements = 100);
};



} // namespace GClasses

#endif // __GMANIFOLD_H__
