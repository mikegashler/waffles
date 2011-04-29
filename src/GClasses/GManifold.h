/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
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
class GNeuralNet;
class GNeuralNetLayer;


/// This class stores static methods that are useful for manifold learning
class GManifold
{
public:
	/// Computes a set of weights for each neighbor to linearly approximate this point
	static void computeNeighborWeights(GMatrix* pData, size_t point, size_t k, const size_t* pNeighbors, double* pOutWeights);

	/// Aligns and averages two local neighborhoods together. The results will be
	/// centered around the neighborhood mean. The first point will be
	/// the index point, and the rest will be neighborhood points with
	/// an index that is not INVALID_INDEX.
	static GMatrix* blendNeighborhoods(size_t index, GMatrix* pA, double ratio, GMatrix* pB, size_t neighborCount, size_t* pHood);

	/// Combines two embeddings to form an "average" embedding. pRatios is an array that specifies how much to weight the
	/// neighborhoods around each point. If the ratio for a point is close to zero, pA will be emphasized more. If
	/// the ratio for the point is close to 1, pB will be emphasized more. "seed" specifies a starting point. It will
	/// blend outward in a breadth-first manner.
	static GMatrix* blendEmbeddings(GMatrix* pA, double* pRatios, GMatrix* pB, size_t neighborCount, size_t* pNeighborTable, size_t seed);

	/// Performs classic MDS. pDistances must be a square matrix, but only the upper-triangle is used.
	/// Each row in the results is one of the result points.
	/// If useSquaredDistances is true, then the values in pDistances are assumed to be squared
	/// distances, rather than normal Euclidean distances.
	static GMatrix* multiDimensionalScaling(GMatrix* pDistances, size_t targetDims, GRand* pRand, bool useSquaredDistances);

#ifndef NO_TEST_CODE
	static void test();
#endif
};



/// This is the base class of manifold learning (aka non-linear
/// dimensionality reducing) algorithms.
class GManifoldLearner : public GTransform
{
public:
	GManifoldLearner() : GTransform() {}
	GManifoldLearner(GTwtNode* pNode) : GTransform(pNode) {}
	virtual ~GManifoldLearner() {}
};




/// Manifold Sculpting. A non-linear dimensionality reduction algorithm.
/// (See Gashler, Michael S. and Ventura, Dan and Martinez, Tony. Iterative
/// non-linear dimensionality reduction with manifold sculpting. In Advances
/// in Neural Information Processing Systems 20, pages 513–520, MIT Press,
/// Cambridge, MA, 2008.)
class GManifoldSculpting : public GManifoldLearner
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
	sp_relation m_pRelationAfter;
	GRand* m_pRand;
	GMatrix* m_pData;
	unsigned char* m_pMetaData;
	GNeighborFinder* m_pNF;

public:
	GManifoldSculpting(size_t nNeighbors, size_t targetDims, GRand* pRand);
	virtual ~GManifoldSculpting();

	/// Perform NLDR.
	virtual GMatrix* doit(GMatrix& in);

	virtual sp_relation& relationAfter() { return m_pRelationAfter; }

	GMatrix& data() { return *m_pData; }

	/// Call this before calling SquishPass. pRealSpaceData should be a dataset
	/// of all real values.
	void beginTransform(GMatrix* pRealSpaceData);

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
	void setNeighborFinder(GNeighborFinder* pNF) { m_pNF = pNF; }

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

	void calculateMetadata(GMatrix* pData);
	double vectorCorrelation(double* pdA, double* pdV, double* pdB);
	double vectorCorrelation2(double squaredScale, size_t a, size_t vertex, struct GManifoldSculptingNeighbor* pNeighborRec);
	double computeError(size_t nPoint);
	size_t adjustDataPoint(size_t nPoint, double* pError);
	double averageNeighborDistance(size_t nDims);
	void plotData(float radius);
	void moveMeanToOrigin();
};

/*
/// An experimental algorithm for using Manifold Sculpting to model dynamic systems.
class GManifoldSculptingForControl : public GManifoldSculpting
{
protected:
	GMatrix* m_pControlData;
	double m_squaredLambda;
	bool m_alignConsequences;

public:
	GManifoldSculptingForControl(size_t nNeighbors, size_t nTargetDims, GRand* pRand, GMatrix* pControlData, double lambda)
	: GManifoldSculpting(nNeighbors, nTargetDims, pRand), m_pControlData(pControlData), m_squaredLambda(lambda * lambda)
	{
	}

	virtual ~GManifoldSculptingForControl()
	{
	}

protected:
	virtual double supervisedError(size_t nPoint);
};
*/

/// Isomap. (A well-known manifold learning algorithm.)
class GIsomap : public GManifoldLearner
{
protected:
	size_t m_neighborCount;
	size_t m_targetDims;
	GNeighborFinder* m_pNF;
	GRand* m_pRand;
	bool m_dropDisconnectedPoints;

public:
	GIsomap(size_t neighborCount, size_t targetDims, GRand* pRand);
	GIsomap(GTwtNode* pNode);
	virtual ~GIsomap();

	/// Serializes this object
	GTwtNode* toTwt(GTwtDoc* pDoc);

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
	virtual GMatrix* doit(GMatrix& in);
};


/// Locally Linear Embedding. (A well-known manifold learning algorithm.)
class GLLE : public GManifoldLearner
{
protected:
	size_t m_neighborCount;
	size_t m_targetDims;
	GNeighborFinder* m_pNF;
	GRand* m_pRand;

public:
	GLLE(size_t neighborCount, size_t targetDims, GRand* pRand);
	GLLE(GTwtNode* pNode);
	virtual ~GLLE();
	
	/// Serialize this object
	GTwtNode* toTwt(GTwtDoc* pDoc);

	/// Specifies to use the neighborhoods determined by the specified neighbor-finder instead of the nearest
	/// Euclidean-distance neighbors to establish local linearity. If this method is called, it will also
	/// use the number of neighbors and the data associated with pNF, and ignore the number of neighbors
	/// specified to the constructor, and ignore the data passed to the "transform" method.
	void setNeighborFinder(GNeighborFinder* pNF);

	/// Performs NLDR
	virtual GMatrix* doit(GMatrix& in);
};

/*
/// An experimental manifold learning algorithm. (Doesn't really work yet.)
class GManifoldUnfolder : public GManifoldLearner
{
protected:
	size_t m_neighborCount;
	size_t m_targetDims;
	GNeighborFinder* m_pNF;
	GRand* m_pRand;

public:
	GManifoldUnfolder(size_t neighborCount, size_t targetDims, GRand* pRand);
	GManifoldUnfolder(GTwtNode* pNode);
	virtual ~GManifoldUnfolder();
	GTwtNode* toTwt(GTwtDoc* pDoc);
	void setNeighborFinder(GNeighborFinder* pNF);

	/// Performs NLDR
	virtual GMatrix* doit(GMatrix& in);

protected:
	GMatrix* unfold(GNeighborFinder* pNF, size_t targetDims, GRand* pRand);
};
*/


/// A manifold learning algorithm that reduces dimensionality in local
/// neighborhoods, and then stitches the reduced local neighborhoods together
/// using the Kabsch algorithm.
class GBreadthFirstUnfolding : public GManifoldLearner
{
protected:
	size_t m_reps;
	size_t m_neighborCount;
	size_t m_targetDims;
	GNeighborFinder* m_pNF;
	bool m_useMds;
	GRand* m_pRand;

public:
	/// reps specifies the number of times to compute the embedding, and blend the
	/// results together. If you just want fast results, use reps=1.
	GBreadthFirstUnfolding(size_t reps, size_t neighborCount, size_t targetDims, GRand* pRand);
	GBreadthFirstUnfolding(GTwtNode* pNode, GRand* pRand);
	virtual ~GBreadthFirstUnfolding();

	/// Serialize this object
	GTwtNode* toTwt(GTwtDoc* pDoc);

	/// Specify the neighbor finder to use to pick neighbors for this algorithm
	void setNeighborFinder(GNeighborFinder* pNF);

	/// Perform NLDR
	virtual GMatrix* doit(GMatrix& in);

	/// Specify to use multi-dimensional scaling instead of PCA to reduce in local patches.
	void useMds(bool b) { m_useMds = b; }

protected:
	void refineNeighborhood(GMatrix* pLocal, size_t rootIndex, size_t* pNeighborTable, double* pDistanceTable);
	GMatrix* reduceNeighborhood(GMatrix* pIn, size_t index, size_t* pNeighborhoods, double* pSquaredDistances);
	GMatrix* unfold(GMatrix* pIn, size_t* pNeighborTable, double* pSquaredDistances, size_t seed, double* pOutWeights);
};



/// This class is a generalization of PCA. When the bias is clamped,
/// and the activation function is "identity", it is strictly equivalent
/// to PCA. By default, however, the bias is allowed to drift from the
/// mean, which gives better results. Also, by default, the activation
/// function is "logistic", which enables it to find non-linear
/// components in the data. (GUnsupervisedBackProp is a
/// multi-layer generalization of this algorithm.)
class GNeuroPCA : public GTransform
{
protected:
	size_t m_targetDims;
	GMatrix* m_pWeights;
	double* m_pEigVals;
	GRand* m_pRand;
	GActivationFunction* m_pActivation;
	bool m_updateBias;

public:
	GNeuroPCA(size_t targetDims, GRand* pRand);

	virtual ~GNeuroPCA();

	/// Returns the weight vectors
	GMatrix* weights() { return m_pWeights; }

	/// Specify to compute the eigenvalues during training. This
	/// method must be called before doIt is called.
	void computeEigVals();

	/// Returns the eigenvalues. Returns NULL if computeEigVals was not called.
	double* eigVals() { return m_pEigVals; }

	/// Returns the number of principal components to find.
	size_t targetDims() { return m_targetDims; }

	/// Specify to not update the bias values.
	void clampBias() { m_updateBias = false; }

	/// Sets the activation function. (Takes ownership of pActivation.)
	void setActivation(GActivationFunction* pActivation);

	/// See the comment for GTransform::doIt
	virtual GMatrix* doit(GMatrix& in);

protected:
	void computeComponent(GMatrix* pIn, GMatrix* pOut, size_t col, GMatrix* pPreprocess);
	double computeSumSquaredErr(GMatrix* pIn, GMatrix* pOut, size_t cols);
};



/// This uses graph-cut to divide the data into two clusters.
/// It then trains a linear regression model for each cluster
/// to map from inputs to change-in-state. It then aligns
/// the smaller cluster with the larger one such that the linear
/// models are in agreement (as much as possible).
class GDynamicSystemStateAligner : public GTransform
{
protected:
	size_t m_neighbors;
	size_t m_seedA, m_seedB;
	size_t* m_pNeighbors;
	double* m_pDistances;
	GMatrix& m_inputs;
	GRand& m_rand;

public:
	GDynamicSystemStateAligner(size_t neighbors, GMatrix& inputs, GRand& rand);
	virtual ~GDynamicSystemStateAligner();

	/// Perform the transformation
	virtual GMatrix* doit(GMatrix& in);

	/// Specify the source and sink points for dividing the data into two clusters
	void setSeeds(size_t a, size_t b);

#ifndef NO_TEST_CODE
	static void test();
#endif
};



/// A manifold learning algorithm that uses back-propagation to train a neural net model
/// to map from low-dimensional space to high-dimensional space.
class GUnsupervisedBackProp : public GManifoldLearner
{
protected:
	size_t m_intrinsicDims;
	GNeuralNet* m_pNN;
	GRand* m_pRand;
	size_t m_paramDims;
	size_t* m_pParamRanges;
	GCoordVectorIterator m_cvi;
	double m_rate;
	bool m_updateWeights;
	bool m_normalize;

public:
	GUnsupervisedBackProp(size_t intrinsicDims, GRand* pRand);
	GUnsupervisedBackProp(GTwtNode* pNode, GRand* pRand);
	virtual ~GUnsupervisedBackProp();

	/// Returns a pointer to the neural network used to model the manifold. Typically, this
	/// is used to add layers to the neural network, or set the learning rate (etc.) before
	/// calling doit.
	GNeuralNet* neuralNet() { return m_pNN; }

	/// Takes ownership of pNN. Replaces the internal neural net with the one specified.
	void setNeuralNet(GNeuralNet* pNN);

	/// An experimental feature that allows one to parameterize the output values.
	void setParams(std::vector<size_t>& paramRanges);

	/// Sets the rate at which the threshold is grown. A typical value is 0.1. Smaller
	/// values will take longer, but may yield better results. Larger values will go faster,
	/// but suffer moew from local optima problems.
	void setRate(double d) { m_rate = d; }

	/// Perform NLDR. (This also trains the internal neural network to map from
	/// low-dimensional space to high-dimensional space.)
	virtual GMatrix* doit(GMatrix& in);

	/// Peform NLDR using a sparse matrix as input
	GMatrix* doitSparse(GSparseMatrix* pData);

	/// Learn intrinsic values for a dynamical system that uses a camera for observations
	GMatrix* doitCameraSystem(std::vector<size_t>& paramRanges, GMatrix* pObservations, GMatrix* pActions);

	/// Projects an intrinsic low-dimensional vector to the corresponding high-dimensional
	/// observation vector.
	void lowToHigh(const double* pIntrinsic, double* pObs);

	/// Specify whether or not to update the weights. The default is to update the weights.
	/// Either way, the intrinsic features will still be updated.
	void setUpdateWeights(bool b) { m_updateWeights = b; }

	/// Specify whether or not to normalize the input vectors. (The default is false.)
	void normalize(bool b) { m_normalize = b; }
};




} // namespace GClasses

#endif // __GMANIFOLD_H__
