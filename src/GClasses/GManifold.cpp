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

#include "GManifold.h"
#include "GActivation.h"
#include <stdio.h>
#include <math.h>
#include "GBits.h"
#include "GBitTable.h"
#include "GGraph.h"
#include "GHillClimber.h"
#include "GHeap.h"
#include "GImage.h"
#include "GKNN.h"
#include "GLinear.h"
#include "GMath.h"
#include "GNeighborFinder.h"
#include "GNeuralNet.h"
#include "GPlot.h"
#include "GSparseMatrix.h"
#include "GTime.h"
#include "GTransform.h"
#include "GDom.h"
#include "GVec.h"
#include "GHolders.h"
#include <deque>
#include <set>
#include <map>
#include <iostream>
#include <string>
#include <queue>

namespace GClasses {

using std::string;
using std::cout;
using std::vector;
using std::deque;
using std::set;
using std::map;
using std::make_pair;


#define USE_ANGLES
#define STEP_SIZE_PER_POINT

// static
void GManifold::computeNeighborWeights(const GMatrix* pData, size_t point, size_t k, const size_t* pNeighbors, double* pOutWeights)
{
	// Create a matrix of all the neighbors normalized around the origin
	size_t colCount = pData->cols();
	GMatrix z(pData->relation().clone());
	const double* pRow = pData->row(point);
	for(size_t i = 0; i < k; i++)
	{
		if(*pNeighbors < pData->rows())
		{
			double* pTarget = z.newRow();
			GVec::copy(pTarget, pData->row(*pNeighbors), colCount);
			GVec::subtract(pTarget, pRow, colCount);
			pNeighbors++;
		}
		else
		{
			double* pTarget = z.newRow();
			GVec::setAll(pTarget, 1e12, colCount);
		}
	}

	// Square it
	GMatrix* pSquare = GMatrix::multiply(z, z, false, true);
	Holder<GMatrix> hSquare(pSquare);

	// if the number of neighbors is more than the number of dimensions then the
	// square matrix will not be full rank so we need to regularize it
	if(pSquare->rows() > (size_t)colCount)
	{
		double dReg = pSquare->trace() * 0.001;
		for(size_t i = 0; i < pSquare->rows(); i++)
			pSquare->row(i)[i] += dReg;
	}

	// Compute the weights the direct way (fails for non-invertible matrices)
// 	for(size_t i = 0; i < pSquare->rows(); i++)
// 		pOutWeights[i] = 1;
// 	if(!pSquare->gaussianElimination(pOutWeights))
// 		throw Ex("Failed to find a solution in computeNeighborWeights");

	// Compute the weights the SVD way
	GMatrix* pInv = pSquare->pseudoInverse();
	Holder<GMatrix> hInv(pInv);
	for(size_t i = 0; i < pSquare->rows(); i++)
		pOutWeights[i] = GVec::sumElements(pInv->row(i), pInv->cols());

	// Normalize the weights to sum to one
	GVec::sumToOne(pOutWeights, pSquare->rows());
}

// static
GMatrix* GManifold::blendNeighborhoods(size_t index, GMatrix* pA, double ratio, GMatrix* pB, size_t neighborCount, size_t* pHood)
{
	// Copy the two neighborhoods
	size_t rowCount = pA->rows();
	size_t colCount = pA->cols();
	GMatrix neighborhoodA(pA->relation().clone());
	GVec::copy(neighborhoodA.newRow(), pA->row(index), colCount);
	GMatrix neighborhoodB(pB->relation().clone());
	GVec::copy(neighborhoodB.newRow(), pB->row(index), colCount);
	for(size_t j = 0; j < neighborCount; j++)
	{
		if(pHood[j] >= rowCount)
			continue;
		GVec::copy(neighborhoodA.newRow(), pA->row(pHood[j]), colCount);
		GVec::copy(neighborhoodB.newRow(), pB->row(pHood[j]), colCount);
	}

	// Subtract the means
	GTEMPBUF(double, mean, colCount);
	neighborhoodA.centroid(mean);
	for(size_t i = 0; i < neighborhoodA.rows(); i++)
		GVec::subtract(neighborhoodA.row(i), mean, colCount);
	neighborhoodB.centroid(mean);
	for(size_t i = 0; i < neighborhoodB.rows(); i++)
		GVec::subtract(neighborhoodB.row(i), mean, colCount);

	// Use the kabsch algorithm to compute the optimal rotation
	GMatrix* pKabsch = GMatrix::kabsch(&neighborhoodA, &neighborhoodB);
	Holder<GMatrix> hKabsch(pKabsch);
	GMatrix* pC = GMatrix::multiply(neighborhoodB, *pKabsch, false, false);
	for(size_t i = 0; i < pC->rows(); i++)
	{
		GVec::multiply(pC->row(i), ratio, colCount);
		GVec::addScaled(pC->row(i), 1.0 - ratio, neighborhoodA.row(i), colCount);
		// todo: should we normalize the distance here?
	}
	return pC;
}

// static
GMatrix* GManifold::blendEmbeddings(GMatrix* pA, double* pRatios, GMatrix* pB, size_t neighborCount, size_t* pNeighborTable, size_t seed)
{
	// Check params
	size_t rowCount = pA->rows();
	size_t colCount = pA->cols();
	if(pB->rows() != rowCount || pB->cols() != colCount)
		throw Ex("mismatching sizes");

	// Blend the seed neighborhood
	GMatrix* pC = new GMatrix(rowCount, colCount);
	deque<size_t> q;
	GBitTable visited(rowCount);
	GBitTable established(rowCount);
	{
		size_t* pHood = pNeighborTable + neighborCount * seed;
		GMatrix* pAve = blendNeighborhoods(seed, pA, pRatios[seed], pB, neighborCount, pHood);
		Holder<GMatrix> hAve(pAve);
		GVec::copy(pC->row(seed), pAve->row(0), colCount);
		visited.set(seed);
		established.set(seed);
		size_t i = 1;
		for(size_t j = 0; j < neighborCount; j++)
		{
			if(pHood[j] >= rowCount)
				continue;
			size_t neigh = pHood[j];
			GVec::copy(pC->row(neigh), pAve->row(i), colCount);
			visited.set(neigh);
			q.push_back(neigh);
			i++;
		}
	}

	// Align in a breadth-first manner
	GTEMPBUF(double, mean, colCount);
	while(q.size() > 0)
	{
		size_t par = q.front();
		q.pop_front();

		// Make a blended neighborhood
		size_t* pHood = pNeighborTable + neighborCount * par;
		GMatrix* pD = blendNeighborhoods(par, pA, pRatios[par], pB, neighborCount, pHood);
		Holder<GMatrix> hD(pD);

		// Make sub-neighborhoods that contain only tentatively-placed points
		GMatrix tentativeC(pC->relation().clone());
		GMatrix tentativeD(pD->relation().clone());
		GReleaseDataHolder hTentativeD(&tentativeD);
		GVec::copy(tentativeC.newRow(), pC->row(par), colCount);
		tentativeD.takeRow(pD->row(0));
		size_t i = 1;
		for(size_t j = 0; j < neighborCount; j++)
		{
			if(pHood[j] >= rowCount)
				continue;
			if(visited.bit(pHood[j]))
			{
				GVec::copy(tentativeC.newRow(), pC->row(pHood[j]), colCount);
				tentativeD.takeRow(pD->row(i));
			}
			i++;
		}

		// Subtract the means
		tentativeD.centroid(mean);
		for(size_t i = 0; i < pD->rows(); i++)
			GVec::subtract(pD->row(i), mean, colCount); // (This will affect tentativeD too b/c it refs the same rows)
		tentativeC.centroid(mean);
		for(size_t i = 0; i < tentativeC.rows(); i++)
			GVec::subtract(tentativeC.row(i), mean, colCount);

		// Compute the rotation to align the tentative neighborhoods
		GMatrix* pKabsch = GMatrix::kabsch(&tentativeC, &tentativeD);
		Holder<GMatrix> hKabsch(pKabsch);

		// Compute an aligned version of pD that fits with pC
		GMatrix* pAligned = GMatrix::multiply(*pD, *pKabsch, false, false);
		Holder<GMatrix> hAligned(pAligned);
		for(size_t i = 0; i < pAligned->rows(); i++)
			GVec::add(pAligned->row(i), mean, colCount);

		// Accept the new points
		GVec::copy(pC->row(par), pAligned->row(0), colCount);
		established.set(par);
		i = 1;
		for(size_t j = 0; j < neighborCount; j++)
		{
			if(pHood[j] >= rowCount)
				continue;
			if(!established.bit(pHood[j]))
				GVec::copy(pC->row(pHood[j]), pAligned->row(i), colCount);
			if(!visited.bit(pHood[j]))
			{
				visited.set(pHood[j]);
				q.push_back(pHood[j]);
			}
			i++;
		}
	}
	return pC;
}

// static
GMatrix* GManifold::multiDimensionalScaling(GMatrix* pDistances, size_t targetDims, GRand* pRand, bool useSquaredDistances)
{
	size_t n = pDistances->rows();
	if((size_t)pDistances->cols() != n)
		throw Ex("Expected a square and symmetric distance matrix");

	// Square every element in the distance matrix (unless it's already squared) and ensure symmetry
	GMatrix* pD = new GMatrix(pDistances->relation().clone());
	Holder<GMatrix> hD(pD);
	pD->newRows(n);
	for(size_t i = 0; i < n; i++)
	{
		double* pIn = pDistances->row(i) + i;
		double* pOut = pD->row(i) + i;
		*pOut = 0.0;
		pOut++;
		pIn++;
		if(useSquaredDistances)
		{
			for(size_t j = i + 1; j < n; j++)
			{
				*pOut = *pIn;
				pOut++;
				pIn++;
			}
		}
		else
		{
			for(size_t j = i + 1; j < n; j++)
			{
				*pOut = (*pIn * *pIn);
				pOut++;
				pIn++;
			}
		}
	}
	pD->mirrorTriangle(true);

	// Some algebra
	GTEMPBUF(double, rowsum, n + targetDims);
	double* pEigenVals = rowsum + n;
	for(size_t i = 0; i < n; i++)
		rowsum[i] = GVec::sumElements(pD->row(i), n);
	double z = 1.0 / n;
	double t = z * GVec::sumElements(rowsum, n);
	for(size_t i = 0; i < n; i++)
	{
		double* pRow = pD->row(i);
		for(size_t j = 0; j < n; j++)
		{
			*pRow = -0.5 * (*pRow + (z * (t - rowsum[i] - rowsum[j])));
			pRow++;
		}
	}

	// Compute eigenvectors
	GMatrix* pEigs = pD->eigs(std::min(n, targetDims), pEigenVals, pRand, true);
	if(n < (size_t)targetDims)
	{
		throw Ex("targetDims cannot be larger than the number of rows or columns in the distance matrix");
/*
		for(size_t i = n; i < targetDims; i++)
			pEigenVals[i] = 0.0;
		GMatrix* pEigs2 = new GMatrix(targetDims);
		pEigs2->newRows(targetDims);
		pEigs2->setAll(0.0);
		for(size_t y = 0; y < n; y++)
			GVec::copy(pEigs2->row(y), pEigs->row(y), n);
		delete(pEigs);
		pEigs = pEigs2;
*/
	}
	Holder<GMatrix> hEigs(pEigs);
	for(size_t i = 0; i < targetDims; i++)
		GVec::multiply(pEigs->row(i), sqrt(std::max(0.0, pEigenVals[i])), n);
	GMatrix* pResults = pEigs->transpose();
	return pResults;
}

#ifndef NO_TEST_CODE
#define POINT_COUNT 9
void GManifold_testMultiDimensionalScaling()
{
	// Test MDS
	GRand prng(0);
	GMatrix foo(POINT_COUNT, 2);
	for(size_t i = 0; i < POINT_COUNT; i++)
		prng.cubical(foo[i], 2);

	// Make distance matrix
	GMatrix dst(POINT_COUNT, POINT_COUNT);
	for(size_t i = 0; i < POINT_COUNT; i++)
	{
		double* pRow = dst.row(i);
		for(size_t j = i + 1; j < POINT_COUNT; j++)
			pRow[j] = GVec::squaredDistance(foo.row(i), foo.row(j), 2);
	}

	// Do MDS
	GMatrix* pMDS = GManifold::multiDimensionalScaling(&dst, 2, &prng, true);
	Holder<GMatrix> hMDS(pMDS);
	for(size_t i = 0; i < POINT_COUNT; i++)
	{
		for(size_t j = 0; j < POINT_COUNT; j++)
		{
			double expected = sqrt(GVec::squaredDistance(foo.row(i), foo.row(j), 2));
			double actual = sqrt(GVec::squaredDistance(pMDS->row(i), pMDS->row(j), 2));
			if(std::abs(expected - actual) > 1e-5)
				throw Ex("failed");
		}
	}
}

// static
void GManifold::test()
{
	GManifold_testMultiDimensionalScaling();
}
#endif







struct GManifoldSculptingNeighbor
{
	size_t m_nNeighbor;
	size_t m_nNeighborsNeighborSlot;
	double m_dCosTheta;
	double m_dDistance;
	double m_junkSquaredDist;
	double m_junkDotProd;
};

struct GManifoldSculptingStuff
{
	size_t m_nCycle;
	bool m_bAdjustable;
};

GManifoldSculpting::GManifoldSculpting(size_t nNeighbors, size_t targetDims, GRand* pRand)
: m_pRand(pRand), m_pNF(NULL)
{
	m_pMetaData = NULL;
	m_nDimensions = 0;
	m_nTargetDims = targetDims;
	m_pRelationAfter = new GUniformRelation(m_nTargetDims, 0);
	m_nNeighbors = nNeighbors;

	m_nStuffIndex = sizeof(struct GManifoldSculptingNeighbor) * m_nNeighbors;
	m_nRecordSize = m_nStuffIndex + sizeof(struct GManifoldSculptingStuff);

	m_dSquishingRate = .99;
	m_nPass = 0;
	m_scale = 1.0;
	m_dAveNeighborDist = 0;
	m_pData = NULL;
	m_minNeighborDist = 0;
	m_maxNeighborDist = 1e20;
}

GManifoldSculpting::~GManifoldSculpting()
{
	delete(m_pData);
	delete[] m_pMetaData;
}

void GManifoldSculpting::setPreprocessedData(GMatrix* pData)
{
	delete(m_pData);
	m_pData = pData;
}

// virtual
GMatrix* GManifoldSculpting::reduce(const GMatrix& in)
{
	beginTransform(&in);

	// Do burn-in iterations
	while(m_scale > 0.01)
		squishPass((size_t)m_pRand->next(m_pData->rows()));

	// Squish until it doesn't improve for a while
	double dBestError = 1e308;
	for(size_t nItersSinceImprovement = 0; nItersSinceImprovement < 50; nItersSinceImprovement++)
	{
//PlotData(4.0f);
		double dError = squishPass((size_t)m_pRand->next(m_pData->rows()));
		if(dError < dBestError)
		{
			dBestError = dError;
			nItersSinceImprovement = 0;
		}
	}

	// Produce the output data
	GMatrix* pDataOut = new GMatrix(relationAfter().clone());
	pDataOut->newRows(m_pData->rows());
	pDataOut->copyCols(*m_pData, 0, m_nTargetDims);
	delete(m_pData);
	m_pData = NULL;
	return pDataOut;
}

void GManifoldSculpting::beginTransform(const GMatrix* pRealSpaceData)
{
	m_nDimensions = pRealSpaceData->cols();
	if(!pRealSpaceData->relation().areContinuous(0, m_nDimensions))
		throw Ex("Only continuous values are supported");

	// Calculate metadata
	calculateMetadata(pRealSpaceData);

	// Preprocess the data
	if(m_pData)
	{
		// Check the supplied pre-processed data
		if(m_pData->rows() != pRealSpaceData->rows())
			throw Ex("Preprocessed data has wrong number of points");
		if(m_pData->relation().size() < (size_t)m_nTargetDims)
			throw Ex("Preprocessed data has too few dimensions");
	}
	else
	{
		// Preprocess the data
		if(pRealSpaceData->relation().size() < 30)
			m_pData = GPCARotateOnly::transform(pRealSpaceData->relation().size(), 0, pRealSpaceData, m_nTargetDims, m_pRand);
		else
		{
			size_t preserveDims = m_nTargetDims * 6;
			preserveDims = std::max((size_t)30, preserveDims);
			preserveDims = std::min(pRealSpaceData->relation().size(), preserveDims);
			GPCA pca(preserveDims);
			pca.train(*pRealSpaceData);
			m_pData = pca.transformBatch(*pRealSpaceData);
		}
	}

	// Calculate the junk
	m_q.clear();
	m_nDimensions = m_pData->relation().size();
	m_nPass = 0;
	m_scale = 1.0;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		struct GManifoldSculptingNeighbor* pArrNeighbors = record(i);
		for(size_t j = 0; j < m_nNeighbors; j++)
		{
			size_t neighbor = pArrNeighbors[j].m_nNeighbor;
			if(neighbor < m_pData->rows())
			{
				pArrNeighbors[j].m_junkSquaredDist = GVec::squaredDistance(m_pData->row(i) + m_nTargetDims, m_pData->row(neighbor) + m_nTargetDims, m_nDimensions - m_nTargetDims);
				size_t slot = pArrNeighbors[j].m_nNeighborsNeighborSlot;
				struct GManifoldSculptingNeighbor* pArrNeighborsNeighbors = record(neighbor);
				size_t neighborsNeighbor = pArrNeighborsNeighbors[slot].m_nNeighbor;
				pArrNeighbors[j].m_junkDotProd = GVec::dotProduct(m_pData->row(neighbor) + m_nTargetDims, m_pData->row(i) + m_nTargetDims, m_pData->row(neighbor) + m_nTargetDims, m_pData->row(neighborsNeighbor) + m_nTargetDims, m_nDimensions - m_nTargetDims);
			}
		}
	}
}

void GManifoldSculpting::calculateMetadata(const GMatrix* pData)
{
	delete[] m_pMetaData;
	m_pMetaData = new unsigned char[m_nRecordSize * pData->rows()];

	// Compute the distance to each neighbor
	m_dAveNeighborDist = 0;
	size_t m_goodNeighbors = 0;
	{
		// Get the appropriate neighbor finder
		Holder<GNeighborFinder> hNF(NULL);
		GNeighborFinder* pNF = m_pNF;
		if(pNF)
		{
			if(pNF->data() != pData)
				throw Ex("Data mismatch");
			if(pNF->neighborCount() != (size_t)m_nNeighbors)
				throw Ex("mismatching numbers of neighbors");
		}
		else
		{
			pNF = new GKdTree(pData, m_nNeighbors, NULL, true);
			hNF.reset(pNF);
		}

		// Set up some some data structures that store the neighbors and distances of each point (and some other stuff)
		GTEMPBUF(size_t, pHood, m_nNeighbors);
		GTEMPBUF(double, pDists, m_nNeighbors);
		for(size_t i = 0; i < pData->rows(); i++)
		{
			stuff(i)->m_bAdjustable = true;
				pNF->neighbors(pHood, pDists, i);
			GNeighborFinder::sortNeighbors(m_nNeighbors, pHood, pDists);
			struct GManifoldSculptingNeighbor* pArrNeighbors = record(i);
			for(size_t j = 0; j < m_nNeighbors; j++)
			{
				pArrNeighbors[j].m_nNeighbor = pHood[j];
				if(pHood[j] < pData->rows())
				{
					m_goodNeighbors++;
					pArrNeighbors[j].m_nNeighborsNeighborSlot = INVALID_INDEX;
					pArrNeighbors[j].m_dDistance = sqrt(pDists[j]);
					m_dAveNeighborDist += pArrNeighbors[j].m_dDistance;
				}
			}
		}
	}
	m_dAveNeighborDist /= m_goodNeighbors;
	m_dLearningRate = m_dAveNeighborDist;

	// For each data point, find the most co-linear of each neighbor's neighbors
	struct GManifoldSculptingNeighbor* pPoint;
	struct GManifoldSculptingNeighbor* pVertex;
	double dCosTheta;
	for(size_t n = 0; n < pData->rows(); n++)
	{
		pPoint = record(n);
		stuff(n)->m_nCycle = INVALID_INDEX;
		for(size_t i = 0; i < m_nNeighbors; i++)
		{
			size_t nVertex = pPoint[i].m_nNeighbor;
			if(nVertex < pData->rows())
			{
				pVertex = record(nVertex);
				pPoint[i].m_nNeighborsNeighborSlot = INVALID_INDEX;
#ifdef USE_ANGLES
				pPoint[i].m_dCosTheta = -100.0;
#else
				pPoint[i].m_dCosTheta = 100.0;
#endif
				for(size_t j = 0; j < m_nNeighbors; j++)
				{
					size_t nCandidate = pVertex[j].m_nNeighbor;
					if(nCandidate < pData->rows())
					{
#ifdef USE_ANGLES
						dCosTheta = acos(vectorCorrelation(pData->row(n), pData->row(nVertex), pData->row(nCandidate))) / M_PI;
						if(dCosTheta > pPoint[i].m_dCosTheta)
#else
						dCosTheta = vectorCorrelation(pData->row(n), pData->row(nVertex), pData->row(nCandidate));
						if(dCosTheta < pPoint[i].m_dCosTheta)
#endif
						{
							pPoint[i].m_dCosTheta = dCosTheta;
							pPoint[i].m_nNeighborsNeighborSlot = j;
						}
					}
				}

				// todo: is this really helpful?
				if(m_nTargetDims < 2)
					pPoint[i].m_dCosTheta = GBits::sign(pPoint[i].m_dCosTheta);
			}
		}
	}
}

void GManifoldSculpting::clampPoint(size_t n)
{
	stuff(n)->m_bAdjustable = false;
}

size_t GManifoldSculpting::countShortcuts(size_t nThreshold)
{
	size_t nShortcuts = 0;
	for(size_t n = 0; n < m_pData->rows(); n++)
	{
		struct GManifoldSculptingNeighbor* pPoint = record(n);
		for(size_t i = 0; i < m_nNeighbors; i++)
		{
			if(pPoint[i].m_nNeighbor < m_pData->rows() && std::abs((int)(pPoint[i].m_nNeighbor - n)) >= (int)nThreshold)
			{
				cout << "shortcut: " << n << "," << pPoint[i].m_nNeighbor << "\n";
				nShortcuts++;
			}
		}
	}
	return nShortcuts;
}

double GManifoldSculpting::vectorCorrelation(const double* pdA, const double* pdV, const double* pdB)
{
	double dDotProd = 0;
	double dMagA = 0;
	double dMagB = 0;
	double dA, dB;
	for(size_t n = 0; n < m_nDimensions; n++)
	{
		dA = pdA[n] - pdV[n];
		dB = pdB[n] - pdV[n];
		dDotProd += (dA * dB);
		dMagA += (dA * dA);
		dMagB += (dB * dB);
	}
	if(dDotProd == 0)
		return 0;
	double dCorrelation = dDotProd / (sqrt(dMagA) * sqrt(dMagB));
	GAssert(dCorrelation > -1.001 && dCorrelation < 1.001);
	return std::max(-1.0, std::min(1.0, dCorrelation));
}

double GManifoldSculpting::vectorCorrelation2(double squaredScale, size_t a, size_t vertex, struct GManifoldSculptingNeighbor* pNeighborRec)
{
	size_t slot = pNeighborRec->m_nNeighborsNeighborSlot;
	if(slot >= m_pData->rows())
		return 0.0;
	struct GManifoldSculptingNeighbor* pNeighborsNeighbors = record(vertex);
	size_t b = pNeighborsNeighbors[slot].m_nNeighbor;
	if(b >= m_pData->rows())
		return 0.0;
	double* pdA = m_pData->row(a);
	double* pdV = m_pData->row(vertex);
	double* pdB = m_pData->row(b);
	double dDotProd = 0;
	double dMagA = 0;
	double dMagB = 0;
	double dA, dB;
	for(size_t n = 0; n < m_nTargetDims; n++)
	{
		dA = pdA[n] - pdV[n];
		dB = pdB[n] - pdV[n];
		dDotProd += (dA * dB);
		dMagA += (dA * dA);
		dMagB += (dB * dB);
	}
	dDotProd += squaredScale * pNeighborRec->m_junkDotProd;
	dMagA += squaredScale * pNeighborRec->m_junkSquaredDist;
	dMagB += squaredScale * pNeighborsNeighbors[slot].m_junkSquaredDist;
	if(dDotProd == 0)
		return 0;
	double dCorrelation = dDotProd / (sqrt(dMagA) * sqrt(dMagB));
	GAssert(dCorrelation > -1.001 && dCorrelation < 1.001);
	return std::max(-1.0, std::min(1.0, dCorrelation));
}

double GManifoldSculpting::averageNeighborDistance(size_t nDims)
{
	double dSum = 0;
	size_t goodNeighbors = 0;
	for(size_t nPoint = 0; nPoint < m_pData->rows(); nPoint++)
	{
		struct GManifoldSculptingNeighbor* pPoint = record(nPoint);
		for(size_t n = 0; n < m_nNeighbors; n++)
		{
			if(pPoint[n].m_nNeighbor < m_pData->rows())
			{
				goodNeighbors++;
				dSum += sqrt(GVec::squaredDistance(m_pData->row(nPoint), m_pData->row(pPoint[n].m_nNeighbor), m_nDimensions));
			}
		}
	}
	return dSum / goodNeighbors;
}

double GManifoldSculpting::computeError(size_t nPoint)
{
	double dError = 0;
	double dDist;
	double dTheta;
	struct GManifoldSculptingNeighbor* pPoint = record(nPoint);
	struct GManifoldSculptingStuff* pNeighborStuff;
	size_t n;
	double squaredScale = m_scale * m_scale;
	size_t accepted = 0;
	for(n = 0; n < m_nNeighbors; n++)
	{
		if(pPoint[n].m_dDistance < m_minNeighborDist)
			continue;
		if(pPoint[n].m_dDistance > m_maxNeighborDist && accepted >= m_nTargetDims)
			break;
		accepted++;

		// Angles
		size_t nNeighbor = pPoint[n].m_nNeighbor;
		if(nNeighbor < m_pData->rows())
		{
#ifdef USE_ANGLES
			dTheta = acos(vectorCorrelation2(squaredScale, nPoint, nNeighbor, &pPoint[n])) / M_PI;
			dTheta -= pPoint[n].m_dCosTheta;
			if(dTheta > 0)
				dTheta = 0;
#else
			dTheta = vectorCorrelation2(squaredScale, nPoint, nNeighbor, &pPoint[n]);
			dTheta -= pPoint[n].m_dCosTheta;
			if(dTheta < 0)
				dTheta = 0;
#endif
			pNeighborStuff = stuff(nNeighbor);
			if(pNeighborStuff->m_nCycle != m_nPass && pNeighborStuff->m_bAdjustable)
				dTheta *= 0.4;

			// Distances
			dDist = sqrt(GVec::squaredDistance(m_pData->row(nPoint), m_pData->row(nNeighbor), m_nTargetDims) + squaredScale * pPoint[n].m_junkSquaredDist);
			dDist -= pPoint[n].m_dDistance;
			dDist /= std::max(m_dAveNeighborDist, 1e-10);
			if(pNeighborStuff->m_nCycle != m_nPass && pNeighborStuff->m_bAdjustable)
				dDist *= 0.4;
			dError += dDist * dDist + dTheta * dTheta;
		}
	}

	return dError + supervisedError(nPoint);
}

size_t GManifoldSculpting::adjustDataPoint(size_t nPoint, double* pError)
{
	bool bMadeProgress = true;
	double* pValues = m_pData->row(nPoint);
	double dErrorBase = computeError(nPoint);
	double dError = 0;
	double dStepSize = m_dLearningRate * (m_pRand->uniform() * .4 + .6); // We multiply the learning rate by a random value so that the points can get away from each other
	size_t nSteps;
	for(nSteps = 0; bMadeProgress && nSteps < 30; nSteps++)
	{
		bMadeProgress = false;
		for(size_t n = 0; n < m_nTargetDims; n++)
		{
			pValues[n] += dStepSize;
			dError = computeError(nPoint);
			if(dError >= dErrorBase)
			{
				pValues[n] -= (dStepSize + dStepSize);
				dError = computeError(nPoint);
			}
			if(dError >= dErrorBase)
				pValues[n] += dStepSize;
			else
			{
				dErrorBase = dError;
				bMadeProgress = true;
			}
		}
	}
	*pError = dError;
	return nSteps - 1; // the -1 is to undo the last incrementor
}

void GManifoldSculpting::moveMeanToOrigin()
{
	GTEMPBUF(double, mean, m_nTargetDims);
	GVec::setAll(mean, 0.0, m_nTargetDims);
	for(size_t i = 0; i < m_pData->rows(); i++)
		GVec::add(mean, m_pData->row(i), m_nTargetDims);
	GVec::multiply(mean, -1.0 / m_pData->rows(), m_nTargetDims);
	for(size_t i = 0; i < m_pData->rows(); i++)
		GVec::add(m_pData->row(i), mean, m_nTargetDims);
}

double GManifoldSculpting::squishPass(size_t nSeedDataPoint)
{
	if(!m_pMetaData)
		throw Ex("You must call BeginTransform before calling this method");
	struct GManifoldSculptingNeighbor* pPoint;
	struct GManifoldSculptingStuff* pStuff;

	// Squish the extra dimensions
	if(m_scale > 0.001)
	{
		m_scale *= m_dSquishingRate;
		if(m_scale > 0.001)
		{
			for(size_t n = 0; n < m_pData->rows(); n++)
				GVec::multiply(m_pData->row(n) + m_nTargetDims, m_dSquishingRate, m_nDimensions - m_nTargetDims);
		}
		else
		{
			for(size_t n = 0; n < m_pData->rows(); n++)
				GVec::setAll(m_pData->row(n) + m_nTargetDims, 0.0, m_nDimensions - m_nTargetDims);
			m_scale = 0;
		}
		while(averageNeighborDistance(m_nDimensions) < m_dAveNeighborDist)
		{
			for(size_t n = 0; n < m_pData->rows(); n++)
				GVec::multiply(m_pData->row(n), 1.0 / m_dSquishingRate, m_nTargetDims);
		}
	}

	// Start at the seed point and correct outward in a breadth-first mannner
	m_q.push_back(nSeedDataPoint);
	size_t nVisitedNodes = 0;
	size_t nSteps = 0;
	double dError = 0;
	double dTotalError = 0;
	while(m_q.size() > 0)
	{
		// Check if this one has already been done
		size_t nPoint = m_q.front();
		m_q.pop_front();
		pPoint = record(nPoint);
		pStuff = stuff(nPoint);
		if(pStuff->m_nCycle == m_nPass)
			continue;
		pStuff->m_nCycle = m_nPass;
		nVisitedNodes++;

		// Push all neighbors into the queue
		for(size_t n = 0; n < m_nNeighbors; n++)
		{
			if(pPoint[n].m_nNeighbor < m_pData->rows())
				m_q.push_back(pPoint[n].m_nNeighbor);
		}

		// Adjust this data point
		if(pStuff->m_bAdjustable && nPoint != nSeedDataPoint)
		{
			nSteps += adjustDataPoint(nPoint, &dError);
			dTotalError += dError;
		}
	}
	if(nSteps < m_pData->rows())
		m_dLearningRate *= .87;
	else
		m_dLearningRate /= .91;
//	cout << "[Learning Rate: " << m_dLearningRate << "]\n";

	if(m_nPass % 20 == 0)
		moveMeanToOrigin();

	m_nPass++;
	return dTotalError;
}






GIsomap::GIsomap(size_t neighborCount, size_t targetDims, GRand* pRand) : m_neighborCount(neighborCount), m_targetDims(targetDims), m_pNF(NULL), m_pRand(pRand), m_dropDisconnectedPoints(false)
{
}

GIsomap::GIsomap(GDomNode* pNode, GLearnerLoader& ll)
: GTransform(pNode, ll)
{
	m_targetDims = (size_t)pNode->field("targetDims")->asInt();
}

// virtual
GIsomap::~GIsomap()
{
}

GDomNode* GIsomap::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GIsomap");
	pNode->addField(pDoc, "targetDims", pDoc->newInt(m_targetDims));
	return pNode;
}

void GIsomap::setNeighborFinder(GNeighborFinder* pNF)
{
	m_pNF = pNF;
}

// virtual
GMatrix* GIsomap::reduce(const GMatrix& in)
{
	GNeighborFinder* pNF = m_pNF;
	Holder<GNeighborFinder> hNF(NULL);
	if(!pNF)
	{
		pNF = new GKdTree(&in, m_neighborCount, NULL, true);
		hNF.reset(pNF);
	}

	// Compute the distance matrix using the Floyd Warshall algorithm
	GTEMPBUF(size_t, hood, pNF->neighborCount());
	GTEMPBUF(double, squaredDists, pNF->neighborCount());
	GFloydWarshall graph(in.rows());
	for(size_t i = 0; i < in.rows(); i++)
	{
		pNF->neighbors(hood, squaredDists, i);
		for(size_t j = 0; j < pNF->neighborCount(); j++)
		{
			if(hood[j] >= in.rows())
				continue;
			double d = sqrt(squaredDists[j]);
			graph.addDirectedEdge(i, hood[j], d);
		}
	}
	graph.compute();
	if(!graph.isConnected())
	{
		if(!m_dropDisconnectedPoints)
			throw Ex("The local neighborhoods do not form a connected graph. Increasing the neighbor count may be a good solution. Another solution is to specify to dropDisconnectedPoints.");
		GMatrix* pCM = graph.costMatrix();
		size_t c = pCM->cols();
		while(true)
		{
			size_t worstRow = 0;
			size_t missing_count = 0;
			for(size_t i = 0; i < pCM->rows(); i++)
			{
				double* pRow = pCM->row(i);
				size_t count = 0;
				for(size_t j = 0; j < c; j++)
				{
					if(*(pRow++) >= 1e200)
						count++;
				}
				if(count > missing_count)
				{
					missing_count = count;
					worstRow = i;
				}
			}
			if(missing_count > 0)
			{
				pCM->deleteRow(worstRow);
				pCM->deleteColumn(worstRow);
			}
			else
				break;
		}
	}

	// Do classic MDS on the distance matrix
	return GManifold::multiDimensionalScaling(graph.costMatrix(), m_targetDims, m_pRand, false);
}














#define SPARSE

// Locally Linear Embedding
class GLLEHelper
{
protected:
	const GMatrix* m_pInputData;
	GMatrix* m_pOutputData;
	size_t m_nInputDims;
	size_t m_nTargetDims;
	size_t m_nNeighbors;
	size_t* m_pNeighbors;
#ifdef SPARSE
	GSparseMatrix* m_pWeights;
#else
	GMatrix* m_pWeights;
#endif
	GRand* m_pRand;

	GLLEHelper(const GMatrix* pData, size_t nTargetDims, size_t nNeighbors, GRand* pRand);
public:
	~GLLEHelper();

	// Uses LLE to compute the reduced dimensional embedding of the data
	// associated with pNF.
	static GMatrix* doLLE(GNeighborFinder* pNF, size_t nTargetDims, GRand* pRand);

protected:
	void findNeighbors(GNeighborFinder* pNF);
	void findNeighborsTheSlowWay();
	void computeWeights();
	void computeEmbedding();
	GMatrix* releaseOutputData();
};

GLLEHelper::GLLEHelper(const GMatrix* pData, size_t nTargetDims, size_t nNeighbors, GRand* pRand)
: m_pRand(pRand)
{
	m_pInputData = pData;
	m_nTargetDims = nTargetDims;
	m_nNeighbors = nNeighbors;
	m_pNeighbors = NULL;
	m_pWeights = NULL;
	m_pOutputData = NULL;
}

GLLEHelper::~GLLEHelper()
{
	delete(m_pNeighbors);
	delete(m_pWeights);
	delete(m_pOutputData);
}

GMatrix* GLLEHelper::releaseOutputData()
{
	GMatrix* pData = m_pOutputData;
	m_pOutputData = NULL;
	return pData;
}

// static
GMatrix* GLLEHelper::doLLE(GNeighborFinder* pNF, size_t nTargetDims, GRand* pRand)
{
	GLLEHelper lle(pNF->data(), nTargetDims, pNF->neighborCount(), pRand);
	lle.findNeighbors(pNF);
	lle.computeWeights();
	lle.computeEmbedding();
	return lle.releaseOutputData();
}

void GLLEHelper::findNeighbors(GNeighborFinder* pNF)
{
	delete(m_pNeighbors);
	m_pNeighbors = new size_t[m_nNeighbors * m_pInputData->rows()];
	size_t* pHood = m_pNeighbors;
	for(size_t i = 0; i < m_pInputData->rows(); i++)
	{
		pNF->neighbors(pHood, i);
		pHood += m_nNeighbors;
	}
}

void GLLEHelper::computeWeights()
{
	size_t nRowCount = m_pInputData->rows();
	GTEMPBUF(double, pVec, m_nNeighbors);
#ifdef SPARSE
	m_pWeights = new GSparseMatrix(nRowCount, nRowCount);
#else
	m_pWeights = new GMatrix(nRowCount, nRowCount);
#endif
	for(size_t n = 0; n < nRowCount; n++)
	{
		GManifold::computeNeighborWeights(m_pInputData, n, m_nNeighbors, m_pNeighbors + n * m_nNeighbors, pVec);
		size_t pos = 0;
		size_t* pHood = m_pNeighbors + n * m_nNeighbors;
#ifdef SPARSE
		for(size_t i = 0; i < m_nNeighbors; i++)
		{
			if(pHood[i] < nRowCount)
				m_pWeights->set(n, pHood[i], pVec[pos++]);
			else
				pos++;
		}
#else
		GVec::setAll(m_pWeights->row(n), 0.0, nRowCount);
		for(size_t i = 0; i < m_nNeighbors; i++)
		{
			if(pHood[i] < nRowCount)
				m_pWeights->row(n)[pHood[i]] = pVec[pos++];
			else
				pos++;
		}
#endif
	}
}

void GLLEHelper::computeEmbedding()
{
	//  Subtract the weights from the identity matrix
	size_t row, col;
	m_pWeights->multiply(-1.0);
	size_t nRowCount = m_pInputData->rows();
#ifdef SPARSE
	for(row = 0; row < nRowCount; row++)
		m_pWeights->set(row, row, m_pWeights->get(row, row) + 1.0);
#else
	for(row = 0; row < nRowCount; row++)
		m_pWeights->row(row)[row] += 1.0;
#endif

	// Compute the smallest (m_nTargetDims+1) eigenvectors of (A^T)A, where A is m_pWeights
#ifdef SPARSE
	// The sparse matrix SVD way
	GSparseMatrix* pU;
	double* diag;
	GSparseMatrix* pV;
	m_pWeights->singularValueDecomposition(&pU, &diag, &pV);
	Holder<GSparseMatrix> hU(pU);
	ArrayHolder<double> hDiag(diag);
	Holder<GSparseMatrix> hV(pV);
	GMatrix* pEigVecs = new GMatrix(m_nTargetDims + 1, pV->cols());
	Holder<GMatrix> hEigVecs(pEigVecs);
	for(size_t i = 1; i <= m_nTargetDims; i++)
	{
		size_t rowIn = pV->rows() - 1 - i;
		double* pRow = pEigVecs->row(i);
		for(size_t j = 0; j < pV->cols(); j++)
			pRow[j] = pV->get(rowIn, j);
	}
#else
/*
	// The brute-force way (slow and not very precise)
	GMatrix* pTmp = m_pWeights->clone();
	Holder<GMatrix> hTmp(pTmp);
	GMatrix* pEigVecs = new GMatrix(nRowCount);
	Holder<GMatrix> hEigVecs(pEigVecs);
	pEigVecs->newRows(m_nTargetDims + 1);
	GTEMPBUF(double, mean, nRowCount);
	pTmp->meanVector(mean);
	for(size_t i = 0; i < nRowCount; i++)
	{
		size_t r = std::min(m_nTargetDims, nRowCount - 1 - i);
		pTmp->principalComponent(pEigVecs->row(r), nRowCount, mean, m_pRand);
		pTmp->removeComponent(mean, pEigVecs->row(r), nRowCount);
	}
*/

	// The SVD way (seems to be the fastest and most accurate)
	GMatrix* pU;
	double* diag;
	GMatrix* pV;
	m_pWeights->singularValueDecomposition(&pU, &diag, &pV);
	Holder<GMatrix> hU(pU);
	ArrayHolder<double> hDiag(diag);
	Holder<GMatrix> hV(pV);
	GMatrix* pEigVecs = new GMatrix(pV->relation(), pV->heap());
	Holder<GMatrix> hEigVecs(pEigVecs);
	for(size_t i = 0; i <= m_nTargetDims; i++)
		pEigVecs->takeRow(pV->releaseRow(pV->rows() - 1));

/*
	// The standard way
	GMatrix* m = GMatrix::multiply(*m_pWeights, *m_pWeights, true, false);
	Holder<GMatrix> hM(m);
	GMatrix* pEigVecs = m->eigs(m_nTargetDims + 1, NULL, m_pRand, false);
	Holder<GMatrix> hEigVecs(pEigVecs);
*/
#endif

	// Make the output data
	m_pOutputData = new GMatrix(nRowCount, m_nTargetDims);
	double d = sqrt((double)nRowCount);
	for(row = 0; row < nRowCount; row++)
	{
		double* pRow = m_pOutputData->row(row);
		for(col = 0; col < m_nTargetDims; col++)
			pRow[col] = pEigVecs->row(col + 1)[row] * d;
	}
}


GLLE::GLLE(size_t neighborCount, size_t targetDims, GRand* pRand) : m_neighborCount(neighborCount), m_targetDims(targetDims), m_pNF(NULL), m_pRand(pRand)
{
}

GLLE::GLLE(GDomNode* pNode, GLearnerLoader& ll)
: GTransform(pNode, ll)
{
	m_targetDims = (size_t)pNode->field("targetDims")->asInt();
}

// virtual
GLLE::~GLLE()
{
}

GDomNode* GLLE::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GLLE");
	pNode->addField(pDoc, "targetDims", pDoc->newInt(m_targetDims));
	return pNode;
}

void GLLE::setNeighborFinder(GNeighborFinder* pNF)
{
	m_pNF = pNF;
}

// virtual
GMatrix* GLLE::reduce(const GMatrix& in)
{
	GNeighborFinder* pNF = m_pNF;
	Holder<GNeighborFinder> hNF(NULL);
	if(!pNF)
	{
		pNF = new GKdTree(&in, m_neighborCount, NULL, true);
		hNF.reset(pNF);
	}
	return GLLEHelper::doLLE(pNF, m_targetDims, m_pRand);
}













GBreadthFirstUnfolding::GBreadthFirstUnfolding(size_t reps, size_t neighborCount, size_t targetDims)
: m_reps(reps), m_neighborCount(neighborCount), m_targetDims(targetDims), m_pNF(NULL), m_useMds(true), m_rand(0)
{
}

GBreadthFirstUnfolding::GBreadthFirstUnfolding(GDomNode* pNode, GLearnerLoader& ll)
: m_reps((size_t)pNode->field("reps")->asInt()), m_neighborCount((size_t)pNode->field("neighbors")->asInt()), m_targetDims((size_t)pNode->field("targetDims")->asInt()), m_pNF(NULL), m_useMds(pNode->field("useMds")->asBool()), m_rand(0)
{
}

// virtual
GBreadthFirstUnfolding::~GBreadthFirstUnfolding()
{
}

GDomNode* GBreadthFirstUnfolding::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "reps", pDoc->newInt(m_reps));
	pNode->addField(pDoc, "neighbors", pDoc->newInt(m_neighborCount));
	pNode->addField(pDoc, "targetDims", pDoc->newInt(m_targetDims));
	pNode->addField(pDoc, "useMds", pDoc->newBool(m_useMds));
	if(m_pNF)
		throw Ex("sorry, serializing a neighbor finder is not yet implemented");
	return pNode;
}

void GBreadthFirstUnfolding::setNeighborFinder(GNeighborFinder* pNF)
{
	m_pNF = pNF;
}

// virtual
GMatrix* GBreadthFirstUnfolding::reduce(const GMatrix& in)
{
	// Obtain the neighbor finder
	GNeighborFinder* pNF = m_pNF;
	Holder<GNeighborFinder> hNF(NULL);
	if(!pNF)
	{
		pNF = new GKdTree(&in, m_neighborCount, NULL, true);
		hNF.reset(pNF);
	}

	// Make sure the neighbor finder is cached
	Holder<GNeighborGraph> hNF2(NULL);
	if(!pNF->isCached())
	{
		GNeighborGraph* pNF2 = new GNeighborGraph(pNF, false);
		hNF2.reset(pNF2);
		pNF = pNF2;
	}
	GNeighborGraph* pCachedNF = (GNeighborGraph*)pNF;
	pCachedNF->fillCache();
	size_t* pNeighborTable = pCachedNF->cache();
	double* pSquaredDistances = pCachedNF->squaredDistanceTable();
	const GMatrix* pData = pNF->data();

	// Learn the manifold
	double* pGlobalWeights = new double[in.rows() * 2];
	ArrayHolder<double> hGlobalWeights(pGlobalWeights);
	double* pLocalWeights = pGlobalWeights + in.rows();
	GVec::setAll(pGlobalWeights, 0.0, in.rows());
	Holder<GMatrix> hFinal(NULL);
	for(size_t i = 0; i < m_reps; i++)
	{
		GMatrix* pRep = unfold(pData, pNeighborTable, pSquaredDistances, (size_t)m_rand.next(pData->rows()), pLocalWeights);
		if(hFinal.get())
		{
			GVec::add(pGlobalWeights, pLocalWeights, in.rows());
			GVec::pairwiseDivide(pLocalWeights, pGlobalWeights, in.rows());
			Holder<GMatrix> hRep(pRep);
			hFinal.reset(GManifold::blendEmbeddings(pRep, pLocalWeights, hFinal.get(), m_neighborCount, pNeighborTable, (size_t)m_rand.next(pData->rows())));
		}
		else
		{
			hFinal.reset(pRep);
			std::swap(pGlobalWeights, pLocalWeights);
		}
	}
	return hFinal.release();
}

void GBreadthFirstUnfolding::refineNeighborhood(GMatrix* pLocal, size_t rootIndex, size_t* pNeighborTable, double* pDistanceTable)
{
	// Determine the index of every row in pLocal
	GTEMPBUF(size_t, indexes, pLocal->rows())
	size_t* pRootNeighbors = pNeighborTable + m_neighborCount * rootIndex;
	indexes[0] = rootIndex;
	size_t pos = 1;
	for(size_t i = 0; i < m_neighborCount; i++)
	{
		if(pRootNeighbors[i] != INVALID_INDEX)
			indexes[pos++] = pRootNeighbors[i];
	}

	// Make a pair-wise distance table
	GTEMPBUF(double, distTable, pLocal->rows() * pLocal->rows());
	double* pTablePos = distTable;
	for(size_t i = 0; i < pLocal->rows(); i++)
	{
		// Make a map to the indexes of point i's neighbors
		size_t indexI = indexes[i];
		size_t* pCurNeighbors = pNeighborTable + m_neighborCount * indexI;
		double* pCurDists = pDistanceTable + m_neighborCount * indexI;
		map<size_t,size_t> indexMap;
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			if(pCurNeighbors[j] != INVALID_INDEX)
				indexMap.insert(make_pair(pCurNeighbors[j], j));
		}
		for(size_t j = 0; j < pLocal->rows(); j++)
		{
			size_t indexJ = indexes[j];
			map<size_t,size_t>::iterator it = indexMap.find(indexJ);
			if(it != indexMap.end())
				*pTablePos = sqrt(pCurDists[it->second]);
			else
				*pTablePos = UNKNOWN_REAL_VALUE;
			pTablePos++;
		}
	}

	// Fill holes with symmetric values
	pTablePos = distTable;
	for(size_t i = 0; i < pLocal->rows(); i++)
	{
		for(size_t j = 0; j < pLocal->rows(); j++)
		{
			if(*pTablePos == UNKNOWN_REAL_VALUE)
			{
				if(distTable[pLocal->rows() * j + i] != UNKNOWN_REAL_VALUE)
					*pTablePos = distTable[pLocal->rows() * j + i];
			}
			pTablePos++;
		}
	}

	// Refine the points
	double firstErr = 0;
	for(size_t iters = 0; iters < 30; iters++)
	{
		double err = 0;
		pTablePos = distTable;
		for(size_t j = 0; j < pLocal->rows(); j++)
		{
			for(size_t i = 0; i < pLocal->rows(); i++)
			{
				if(*pTablePos != UNKNOWN_REAL_VALUE)
					err += GVec::refinePoint(pLocal->row(i), pLocal->row(j), m_targetDims, *pTablePos, 0.1, &m_rand);
				pTablePos++;
			}
		}
		if(iters == 0)
			firstErr = err;
		else if(iters == 29 && err > firstErr)
			throw Ex("made it worse");
	}
}

GMatrix* GBreadthFirstUnfolding::reduceNeighborhood(const GMatrix* pIn, size_t index, size_t* pNeighborhoods, double* pSquaredDistances)
{
	GMatrix* pReducedNeighborhood = NULL;
	if(m_useMds)
	{
		// Build index tables
		GTEMPBUF(size_t, indexes, m_neighborCount + 1);
		size_t localSize = 0;
		map<size_t,size_t> revIndexes;
		map<size_t,size_t>::iterator it;
		revIndexes.insert(make_pair(index, localSize));
		indexes[localSize++] = index;
		size_t* pHood = pNeighborhoods + m_neighborCount * index;
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			if(pHood[j] < pIn->rows())
			{
				revIndexes.insert(make_pair(pHood[j], localSize));
				indexes[localSize++] = pHood[j];
			}
		}

		// Build a distance matrix
		GFloydWarshall graph(localSize);
		for(size_t i = 0; i < localSize; i++)
		{
			size_t from = indexes[i];
			pHood = pNeighborhoods + m_neighborCount * from;
			double* pSquaredDists = pSquaredDistances + m_neighborCount * from;
			for(size_t j = 0; j < m_neighborCount; j++)
			{
				size_t to = pHood[j];
				it = revIndexes.find(to);
				if(it == revIndexes.end())
					continue;
				double d = sqrt(pSquaredDists[j]);
				graph.addDirectedEdge(i, it->second, d);
				graph.addDirectedEdge(it->second, i, d);
			}
		}
		graph.compute();
		GAssert(graph.isConnected());

		// Use MDS to reduce the neighborhood
		pReducedNeighborhood = GManifold::multiDimensionalScaling(graph.costMatrix(), m_targetDims, &m_rand, false);
	}
	else
	{
		// Make a local neighborhood
		GMatrix local(pIn->relation().clone());
		GReleaseDataHolder hLocal(&local);
		local.takeRow((double*)pIn->row(index));
		size_t* pHood = pNeighborhoods + m_neighborCount * index;
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			if(pHood[j] < pIn->rows())
				local.takeRow((double*)pIn->row(pHood[j]));
		}

		// Use PCA to reduce the neighborhood
		GPCA pca(m_targetDims);
		pca.train(local);
		pReducedNeighborhood = pca.transformBatch(local);
	}

	return pReducedNeighborhood;
}

GMatrix* GBreadthFirstUnfolding::unfold(const GMatrix* pIn, size_t* pNeighborTable, double* pSquaredDistances, size_t seed, double* pOutWeights)
{
	// Reduce the seed neighborhood
	GMatrix* pOut = new GMatrix(pIn->rows(), m_targetDims);
	Holder<GMatrix> hOut(pOut);
	deque<size_t> q;
	GBitTable visited(pIn->rows());
	GBitTable established(pIn->rows());
	{
		GMatrix* pLocal = reduceNeighborhood(pIn, seed, pNeighborTable, pSquaredDistances);
		Holder<GMatrix> hLocal(pLocal);
		GVec::copy(pOut->row(seed), pLocal->row(0), m_targetDims);
		visited.set(seed);
		established.set(seed);
		size_t* pHood = pNeighborTable + m_neighborCount * seed;
		size_t i = 1;
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			if(pHood[j] >= pIn->rows())
				continue;
			size_t neigh = pHood[j];
			GVec::copy(pOut->row(neigh), pLocal->row(i), m_targetDims);
			visited.set(neigh);
			q.push_back(neigh);
			q.push_back(1);
			i++;
		}
	}
	pOutWeights[seed] = 8.0;

	// Reduce in a breadth-first manner
	GTEMPBUF(double, mean, m_targetDims);
	while(q.size() > 0)
	{
		size_t par = q.front();
		q.pop_front();
		size_t depth = q.front();
		q.pop_front();
		pOutWeights[par] = 1.0 / (double)depth;

		// Make a blended neighborhood
		GMatrix* pLocal = reduceNeighborhood(pIn, par, pNeighborTable, pSquaredDistances);
		Holder<GMatrix> hLocal(pLocal);

		// Make sub-neighborhoods that contain only tentatively-placed points
		GMatrix tentativeC(pOut->relation().clone());
		GMatrix tentativeD(pLocal->relation().clone());
		GReleaseDataHolder hTentativeD(&tentativeD);
		GVec::copy(tentativeC.newRow(), pOut->row(par), m_targetDims);
		tentativeD.takeRow(pLocal->row(0));
		size_t* pHood = pNeighborTable + m_neighborCount * par;
		size_t i = 1;
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			if(pHood[j] >= pIn->rows())
				continue;
			if(visited.bit(pHood[j]))
			{
				GVec::copy(tentativeC.newRow(), pOut->row(pHood[j]), m_targetDims);
				tentativeD.takeRow(pLocal->row(i));
			}
			i++;
		}

		// Subtract the means
		tentativeD.centroid(mean);
		for(size_t i = 0; i < pLocal->rows(); i++)
			GVec::subtract(pLocal->row(i), mean, m_targetDims); // (This will affect tentativeD too b/c it refs the same rows)
		tentativeC.centroid(mean);
		for(size_t i = 0; i < tentativeC.rows(); i++)
			GVec::subtract(tentativeC.row(i), mean, m_targetDims);

		// Compute the rotation to align the tentative neighborhoods
		GMatrix* pKabsch = GMatrix::kabsch(&tentativeC, &tentativeD);
		Holder<GMatrix> hKabsch(pKabsch);

		// Compute an aligned version of pLocal that fits with pOut
		GMatrix* pAligned = GMatrix::multiply(*pLocal, *pKabsch, false, false);
		Holder<GMatrix> hAligned(pAligned);
		for(size_t i = 0; i < pAligned->rows(); i++)
			GVec::add(pAligned->row(i), mean, m_targetDims);

		// Accept the new points
		GVec::copy(pOut->row(par), pAligned->row(0), m_targetDims);
		established.set(par);
		i = 1;
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			if(pHood[j] >= pIn->rows())
				continue;
			if(!established.bit(pHood[j]))
				GVec::copy(pOut->row(pHood[j]), pAligned->row(i), m_targetDims);
			if(!visited.bit(pHood[j]))
			{
				visited.set(pHood[j]);
				q.push_back(pHood[j]);
				q.push_back(depth + 1);
			}
			i++;
		}
	}
	return hOut.release();
}














GNeuroPCA::GNeuroPCA(size_t targetDims, GRand* pRand)
: GTransform(), m_targetDims(targetDims), m_pWeights(NULL), m_pEigVals(NULL), m_pRand(pRand)
{
	m_pActivation = new GActivationLogistic();
}

// virtual
GNeuroPCA::~GNeuroPCA()
{
	delete(m_pWeights);
	delete[] m_pEigVals;
	delete(m_pActivation);
}

void GNeuroPCA::setActivation(GActivationFunction* pActivation)
{
	delete(m_pActivation);
	m_pActivation = pActivation;
}

void GNeuroPCA::computeEigVals()
{
	delete[] m_pEigVals;
	m_pEigVals = new double[m_targetDims];
}

void GNeuroPCA::computeComponent(const GMatrix* pIn, GMatrix* pOut, size_t col, GMatrix* pPreprocess)
{
	size_t dims = (size_t)pIn->cols();
	if(col == 0)
		pPreprocess->setAll(0.0);
	for(size_t i = 0; i < pOut->rows(); i++)
	{
		double* pX = pOut->row(i);
		pX[col] = 0.5;
		if(col > 0)
		{
			double* pPrevWeights = m_pWeights->row(col);
			double* pPre = pPreprocess->row(i);
			for(size_t j = 0; j < dims; j++)
				*(pPre++) += *(pPrevWeights++) * pX[col - 1];
		}
	}
	double* pBiases = m_pWeights->row(0);
	double* pWeights = m_pWeights->row(1 + col);
	for(size_t i = 0; i < dims; i++)
		pWeights[i] = 0.1 * m_pRand->normal();
	size_t* pIndexes = new size_t[pOut->rows()];
	GIndexVec::makeIndexVec(pIndexes, pOut->rows());
	double learningRate = 0.05;
	double prevRsse = 1e100;
//char buf[64];
	while(true)
	{
		double sse;
		for(size_t iter = 0; iter < 1; iter++)
		{
			GIndexVec::shuffle(pIndexes, pOut->rows(), m_pRand);
			sse = 0.0;
			for(size_t i = 0; i < pOut->rows(); i++)
			{
				size_t index = pIndexes[i];
				double* pBias = pBiases;
				double* pPre = pPreprocess->row(index);
				double* pX = pOut->row(index) + col;
				double* pW = pWeights;
				const double* pTar = pIn->row(index);
				for(size_t j = 0; j < dims; j++)
				{
					if(*pTar != UNKNOWN_REAL_VALUE)
					{
						// Compute the predicted output
						double net = *pBias + *pPre + *pW * (*pX);
						double pred = m_pActivation->squash(net);

						// Compute the error (pIn gives the target)
						double err = learningRate * (*pTar - pred) * m_pActivation->derivativeOfNet(net, pred);
						sse += (err * err);

						// Adjust the bias and weight
						if(m_updateBias)
							*pBias += err;
						double w = *pW;
						*pW += err * (*pX);

						// Adjust x
						*pX += err * w;

						// Clip x
//						*pX = std::max(0.0, std::min(1.0, *pX));
					}

					pBias++;
					pPre++;
					pW++;
					pTar++;
				}
			}
		}
		double rsse = sqrt(sse);
//cout << "learningRate=" << learningRate << ", rsse=" << rsse << ", time=" << GTime::asciiTime(buf, 64) << "\n";
		if((prevRsse - rsse) / prevRsse < 0.0001)
		{
			if(learningRate < 0.00005)
				break;
			learningRate *= 0.5;
		}
		prevRsse = rsse;
	}
}

double GNeuroPCA::computeSumSquaredErr(const GMatrix* pIn, GMatrix* pOut, size_t cols)
{
	size_t dims = (size_t)pIn->cols();
	double sse = 0.0;
	for(size_t i = 0; i < pIn->rows(); i++)
	{
		const double* pTar = pIn->row(i);
		double* pBias = m_pWeights->row(0);
		for(size_t j = 0; j < dims; j++)
		{
			double* pX = pOut->row(i);
			double net = *(pBias++);
			for(size_t k = 0; k < cols; k++)
				net += *(pX++) * m_pWeights->row(k + 1)[j];
			double d = *(pTar++) - m_pActivation->squash(net);
			sse += (d * d);
		}
	}
	return sse;
}

// virtual
GMatrix* GNeuroPCA::reduce(const GMatrix& in)
{
	if(!in.relation().areContinuous())
		throw Ex("GNeuroPCA doesn't support nominal values. You should filter with nominaltocat to make them real.");
	delete(m_pWeights);
	m_pWeights = new GMatrix(in.relation().clone());
	m_pWeights->newRows(1 + m_targetDims); // the first row holds the biases

	// Initialize the biases
	size_t dims = (size_t)in.cols();
	{
		double* pBiases = m_pWeights->row(0);
		for(size_t i = 0; i < dims; i++)
		{
			double mean = in.columnMean(i);
			if((mean < m_pActivation->center() - m_pActivation->halfRange()) || (mean > m_pActivation->center() + m_pActivation->halfRange()))
				throw Ex("The data is expected to fall within the range of the activation function");
			*(pBiases++) = m_pActivation->inverse(mean);
		}
	}

	// Make space for the output data
	GMatrix* pOut = new GMatrix(in.rows(), m_targetDims);
	Holder<GMatrix> hOut(pOut);

	// Make a buffer for preprocessed info
	GMatrix preprocess(in.relation().clone());
	preprocess.newRows(in.rows());

	// Compute the principle components
	double sse = 0;
	if(m_pEigVals)
		sse = computeSumSquaredErr(&in, pOut, 0);
	for(size_t i = 0; i < m_targetDims; i++)
	{
		computeComponent(&in, pOut, i, &preprocess);
		if(m_pEigVals)
		{
			double t = computeSumSquaredErr(&in, pOut, i + 1);
			m_pEigVals[i] = (sse - t) / dims;
			sse = t;
		}
	}
	return hOut.release();
}










GDynamicSystemStateAligner::GDynamicSystemStateAligner(size_t neighbors, GMatrix& inputs, GRand& rand)
: GTransform(), m_neighbors(neighbors), m_inputs(inputs), m_rand(rand)
{
	if(!inputs.relation().areContinuous())
		throw Ex("Only continuous attributes are supported");
	m_seedA = (size_t)m_rand.next(inputs.rows());
	m_seedB = (size_t)m_rand.next(inputs.rows() - 1);
	if(m_seedB >= m_seedA)
		m_seedB++;
	m_pNeighbors = new size_t[neighbors];
	m_pDistances = new double[neighbors];
}

// virtual
GDynamicSystemStateAligner::~GDynamicSystemStateAligner()
{
	delete[] m_pNeighbors;
	delete[] m_pDistances;
}

void GDynamicSystemStateAligner::setSeeds(size_t a, size_t b)
{
	m_seedA = a;
	m_seedB = b;
}

// virtual
GMatrix* GDynamicSystemStateAligner::reduce(const GMatrix& in)
{
	if(!in.relation().areContinuous())
		throw Ex("Only continuous attributes are supported");
	if(in.rows() != m_inputs.rows())
		throw Ex("Expected pIn to have the same number of rows as the inputs");
	if(in.rows() < 6)
	{
		GMatrix* pRet = new GMatrix();
		pRet->copy(&in);
		return pRet;
	}

	// Make a graph of local neighborhoods
	GKdTree neighborFinder(&in, m_neighbors, NULL, false);
	GGraphCut gc(in.rows() + 2);
	for(size_t i = 0; i < in.rows(); i++)
	{
		neighborFinder.neighbors(m_pNeighbors, m_pDistances, i);
		size_t* pNeigh = m_pNeighbors;
		double* pDist = m_pDistances;
		for(size_t j = 0; j < m_neighbors; j++)
		{
			if(*pNeigh >= in.rows())
				continue;
			gc.addEdge(i, *pNeigh, (float)(1.0 / std::max(sqrt(*pDist), 1e-9))); // connect neighbors
			pNeigh++;
			pDist++;
		}
	}

	// Divide into two clusters
	gc.cut(m_seedA, m_seedB);

	// Create training data for the linear regressors
	GMatrix aFeatures(m_inputs.relation().clone());
	GReleaseDataHolder hAFeatures(&aFeatures);
	GMatrix bFeatures(m_inputs.relation().clone());
	GReleaseDataHolder hBFeatures(&bFeatures);
	GMatrix aLabels(in.relation().clone()); // Transitions within cluster A
	GMatrix bLabels(in.relation().clone()); // Transitions within cluster B
	GMatrix cLabels(in.relation().clone()); // Transitions between clusters
	for(size_t i = 0; i < in.rows() - 1; i++)
	{
		double* pLabel;
		if(gc.isSource(i))
		{
			if(gc.isSource(i + 1))
			{
				pLabel = aLabels.newRow();
				aFeatures.takeRow(m_inputs[i]);
			}
			else
				pLabel = cLabels.newRow();
		}
		else
		{
			if(gc.isSource(i + 1))
				pLabel = cLabels.newRow();
			else
			{
				pLabel = bLabels.newRow();
				bFeatures.takeRow(m_inputs[i]);
			}
		}
		GVec::copy(pLabel, in.row(i + 1), in.cols());
		GVec::subtract(pLabel, in.row(i), in.cols());
	}

	// Make the output data
	GMatrix* pOut = new GMatrix();
	pOut->copy(&in);
	Holder<GMatrix> hOut(pOut);
	if(aFeatures.rows() < in.cols() || bFeatures.rows() < in.cols() || cLabels.rows() < 1)
	{
		// There are not enough points to avoid being arbitrary, so we will simply not change anything
		return hOut.release();
	}

	// Train the linear regression models
	GLinearRegressor lrA;
	GLinearRegressor lrB;
	lrA.train(aFeatures, aLabels);
	lrB.train(bFeatures, bLabels);

	// Align the perceptrons
	bool alignCluster = true;
	GLinearRegressor* pLrAlign = &lrA;
	GLinearRegressor* pLrBase = &lrB;
	if(bFeatures.rows() < aFeatures.rows())
	{
		std::swap(pLrAlign, pLrBase);
		alignCluster = false;
	}

	GMatrix* pAInv = pLrAlign->beta()->pseudoInverse();
	Holder<GMatrix> hAInv(pAInv);
	GMatrix* pAlign = GMatrix::multiply(*pLrBase->beta(), *pAInv, false, false);
	Holder<GMatrix> hAlign(pAlign);
	GAssert(pAlign->rows() == pAlign->cols());
	GTEMPBUF(double, shift, 2 * in.cols());
	double* pBuf = shift + in.cols();
	GVec::setAll(shift, 0.0, in.cols());
	size_t crossCount = 0;
	for(size_t i = 0; i < in.rows(); i++)
	{
		if(gc.isSource(i) == alignCluster)
		{
			pAlign->multiply(in.row(i), pOut->row(i));
			if(i > 0 && gc.isSource(i - 1) != alignCluster)
			{
				pLrBase->predict(m_inputs[i - 1], pBuf);
				GVec::add(pBuf, in.row(i - 1), in.cols());
				GVec::subtract(pBuf, pOut->row(i), in.cols());
				GVec::add(shift, pBuf, in.cols());
				crossCount++;
			}
			if(i < in.rows() - 1 && gc.isSource(i + 1) != alignCluster)
			{
				pLrBase->predict(m_inputs[i], pBuf);
				GVec::multiply(pBuf, -1.0, in.cols());
				GVec::add(pBuf, in.row(i + 1), in.cols());
				GVec::subtract(pBuf, pOut->row(i), in.cols());
				GVec::add(shift, pBuf, in.cols());
				crossCount++;
			}
		}
	}
	GVec::multiply(shift, 1.0 / crossCount, in.cols());
	for(size_t i = 0; i < in.rows(); i++)
	{
		if(gc.isSource(i) == alignCluster)
			GVec::add(pOut->row(i), shift, in.cols());
	}

	// Pick new seeds, in case this method is called again
	m_seedA = (size_t)m_rand.next(m_inputs.rows());
	m_seedB = (size_t)m_rand.next(m_inputs.rows() - 1);
	if(m_seedB >= m_seedA)
		m_seedB++;

	return hOut.release();
}

#ifndef NO_TEST_CODE
// static
void GDynamicSystemStateAligner::test()
{
	// Make some data suitable for testing it
	GRand prng(0);
	GMatrix inputs(500, 4);
	inputs.setAll(0.0);
	GMatrix state(500, 2);
	bool alt = false;
	double a2 = 0.7;
	double ca2 = cos(a2);
	double sa2 = sin(a2);
	double x1 = 0.0;
	double y1 = 0.0;
	double x2 = 20.0;
	double y2 = 0.0;
	size_t seedA = 0;
	size_t seedB = 0;
	for(size_t i = 0; i < 500; i++)
	{
		if(alt)
		{
			state[i][0] = x2;
			state[i][1] = y2;
			seedB = i;
		}
		else
		{
			state[i][0] = x1;
			state[i][1] = y1;
			seedA = i;
		}
		while(true)
		{
			double dx = 0;
			double dy = 0;
			size_t action = (size_t)prng.next(4);
			if(action == 0 && x1 < 5)
				dx = 1;
			else if(action == 1 && x1 > -5)
				dx = -1;
			else if(action == 2 && y1 < 5)
				dy = 1;
			else if(action == 3 && y1 > -5)
				dy = -1;
			else
				continue;
			x1 += dx;
			y1 += dy;
			x2 += ca2 * dx - sa2 * dy;
			y2 += ca2 * dy + sa2 * dx;
			GVec::setAll(inputs[i], 0.0, 4);
			inputs[i][action] = 1.0;
			break;
		}
		if((size_t)prng.next(6) == 0)
			alt = !alt;
	}

	// Do the transformation it
	GDynamicSystemStateAligner dssa(16, inputs, prng);
	dssa.setSeeds(seedA, seedB);
	GMatrix* pStateOut = dssa.reduce(state);
	Holder<GMatrix> hStateOut(pStateOut);

	// Check results
	alt = pStateOut->row(0)[0] < 10 ? false : true;
	x1 = 0.0;
	y1 = 0.0;
	x2 = 20.0;
	y2 = 0.0;
	for(size_t i = 0; i < 500; i++)
	{
		if(alt)
		{
			if(std::abs(pStateOut->row(i)[0] - x2) > 0.6)
				throw Ex("failed");
			if(std::abs(pStateOut->row(i)[1] - y2) > 0.6)
				throw Ex("failed");
		}
		else
		{
			if(std::abs(pStateOut->row(i)[0] - x1) > 0.6)
				throw Ex("failed");
			if(std::abs(pStateOut->row(i)[1] - y1) > 0.6)
				throw Ex("failed");
		}
		size_t action = GVec::indexOfMax(inputs[i], 4, &prng);
		double dx = 0;
		double dy = 0;
		if(action == 0)
			dx = 1;
		else if(action == 1)
			dx = -1;
		else if(action == 2)
			dy = 1;
		else
			dy = -1;
		x1 += dx;
		y1 += dy;
		x2 += ca2 * dx - sa2 * dy;
		y2 += ca2 * dy + sa2 * dx;
	}
}
#endif // NO_TEST_CODE






GImageJitterer::GImageJitterer(size_t wid, size_t hgt, size_t channels, double rotateDegrees, double translateWidths, double zoomFactor)
: m_wid(wid), m_hgt(hgt), m_channels(channels), m_rotateRads(rotateDegrees * M_PI / 180.0), m_translatePixels(translateWidths * wid), m_zoomFactor(zoomFactor), m_cx(0.5 * double(wid - 1)), m_cy(0.5 * double(hgt - 1))
{
}

GImageJitterer::GImageJitterer(GDomNode* pNode)
{
	m_wid = size_t(pNode->field("wid")->asInt());
	m_hgt = size_t(pNode->field("hgt")->asInt());
	m_channels = size_t(pNode->field("chan")->asInt());
	m_rotateRads = pNode->field("rot")->asDouble();
	m_translatePixels = pNode->field("tp")->asDouble();
	m_zoomFactor = pNode->field("zoom")->asDouble();
	m_cx = pNode->field("cx")->asDouble();
	m_cy = pNode->field("cy")->asDouble();
}

GDomNode* GImageJitterer::serialize(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "wid", pDoc->newInt(m_wid));
	pNode->addField(pDoc, "hgt", pDoc->newInt(m_hgt));
	pNode->addField(pDoc, "chan", pDoc->newInt(m_channels));
	pNode->addField(pDoc, "rot", pDoc->newDouble(m_rotateRads));
	pNode->addField(pDoc, "tp", pDoc->newDouble(m_translatePixels));
	pNode->addField(pDoc, "zoom", pDoc->newDouble(m_zoomFactor));
	pNode->addField(pDoc, "cx", pDoc->newDouble(m_cx));
	pNode->addField(pDoc, "cy", pDoc->newDouble(m_cy));
	return pNode;
}

double* GImageJitterer::pickParams(GRand& rand)
{
	for(size_t i = 0; i < 4; i++)
		m_params[i] = rand.uniform();
	return m_params;
}

void GImageJitterer::transformedPix(const double* pRow, size_t x, size_t y, double* pOut)
{
	double xx = double(x) - m_cx;
	double yy = double(y) - m_cy;
	double* pParam = m_params;
	double radius = sqrt(xx * xx + yy * yy) * pow(m_zoomFactor, 2 * (*pParam) - 1.0);
	pParam++;
	double angle = atan2(yy, xx) + (*pParam - 0.5) * m_rotateRads;
	pParam++;
	xx = m_cx + radius * cos(angle) + (*pParam - 0.5) * m_translatePixels;
	pParam++;
	yy = m_cy + radius * sin(angle) + (*pParam - 0.5) * m_translatePixels;
	interpolate(pRow, xx, yy, pOut);
}

void GImageJitterer::interpolate(const double* pRow, double x, double y, double* pOut)
{
	// Compute coordinates
	size_t xx, yy;
	if(x < 0.0)
	{
		xx = 0;
		x = 0.0;
	}
	else if(x + 1 >= m_wid)
	{
		xx = m_wid - 2;
		x = (double)(m_wid - 1);
	}
	else
		xx = size_t(floor(x));
	if(y < 0.0)
	{
		yy = 0;
		y = 0.0;
	}
	else if(y + 1 >= m_hgt)
	{
		yy = m_hgt - 2;
		y = (double)(m_hgt - 1);
	}
	else
		yy = size_t(floor(y));

	// Linearly interpolate horizontally
	GTEMPBUF(double, pA, m_channels + m_channels);
	double u = x - xx;
	GVec::setAll(pA, 0.0, m_channels + m_channels);
	GVec::addScaled(pA, 1.0 - u, pRow + m_channels * ((m_wid * yy) + xx), m_channels);
	GVec::addScaled(pA, u, pRow + m_channels * ((m_wid * yy) + xx + 1), m_channels);
	GVec::addScaled(pA + m_channels, 1.0 - u, pRow + m_channels * ((m_wid * (yy + 1)) + xx), m_channels);
	GVec::addScaled(pA + m_channels, u, pRow + m_channels * ((m_wid * (yy + 1)) + xx + 1), m_channels);

	// Linearly interpolate vertically
	u = y - yy;
	GVec::setAll(pOut, 0.0, m_channels);
	GVec::addScaled(pOut, 1.0 - u, pA, m_channels);
	GVec::addScaled(pOut, u, pA + m_channels, m_channels);
}

#ifndef NO_TEST_CODE
void GImageJitterer_makeImage(GImage& image, GImageJitterer& jitterer, double* pVec, double* pParams, double zoom, double rot, double htrans, double vtrans, const char* filename)
{
	pParams[0] = zoom;
	pParams[1] = rot;
	pParams[2] = htrans;
	pParams[3] = vtrans;
	double pix[3];
	GImage imageOut;
	imageOut.setSize(image.width(), image.height());
	for(int y = 0; y < (int)image.height(); y++)
	{
		for(int x = 0; x < (int)image.width(); x++)
		{
			jitterer.transformedPix(pVec, x, y, pix);
			imageOut.setPixel(x, y, gARGB(0xff, (int)pix[0], (int)pix[1], (int)pix[2]));
		}
	}
	imageOut.savePpm(filename);
}

// static
void GImageJitterer::test(const char* filename)
{
	GRand rand(0);
	GImage image;
	image.loadPpm(filename);
	double* pVec = new double[image.width() * image.height() * 3];
	ArrayHolder<double> hVec(pVec);
	GVec::fromImage(&image, pVec, image.width(), image.height(), 3, 255.0);
	GImageJitterer jitterer(image.width(), image.height(), 3, 90.0, 0.5, 2.0);
	double* pParams = jitterer.pickParams(rand);
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.0, 0.5, 0.5, 0.5, "zoom1.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.25, 0.5, 0.5, 0.5, "zoom2.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.75, 0.5, 0.5, 0.5, "zoom3.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 1.0, 0.5, 0.5, 0.5, "zoom4.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.0, 0.5, 0.5, "rot1.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.25, 0.5, 0.5, "rot2.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.75, 0.5, 0.5, "rot3.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 1.0, 0.5, 0.5, "rot4.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 0.0, 0.5, "h1.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 0.25, 0.5, "h2.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 0.75, 0.5, "h3.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 1.0, 0.5, "h4.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 0.5, 0.0, "v1.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 0.5, 0.25, "v2.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 0.5, 0.75, "v3.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.5, 0.5, 0.5, 1.0, "v4.ppm");
	GImageJitterer_makeImage(image, jitterer, pVec, pParams, 0.25, 0.25, 0.25, 0.25, "all.ppm");

	// todo: find an automated way to examine the results
}
#endif







GUnsupervisedBackProp::GUnsupervisedBackProp(size_t intrinsicDims, GRand* pRand)
: m_paramDims(0), m_pParamRanges(NULL), m_jitterDims(0), m_intrinsicDims(intrinsicDims), m_cvi(0, NULL), m_useInputBias(true), m_pJitterer(NULL), m_pIntrinsic(NULL), m_pMins(NULL), m_pRanges(NULL), m_pProgress(NULL), m_onePass(false)
{
	m_pNN = new GNeuralNet();
}

GUnsupervisedBackProp::GUnsupervisedBackProp(GDomNode* pNode, GLearnerLoader& ll)
: GTransform(pNode, ll), m_cvi(0, NULL), m_pIntrinsic(NULL), m_pProgress(NULL)
{
	GDomListIterator it(pNode->field("params"));
	m_paramDims = it.remaining();
	m_pParamRanges = new size_t[m_paramDims];
	GIndexVec::deserialize(m_pParamRanges, it);
	m_cvi.reset(m_paramDims, m_pParamRanges);
	m_pNN = new GNeuralNet(pNode->field("nn"), ll);
	m_useInputBias = pNode->field("bias")->asBool();
	GDomNode* pJitterer = pNode->fieldIfExists("jitterer");
	if(pJitterer)
	{
		m_pJitterer = new GImageJitterer(pJitterer);
		m_jitterDims = 4;
	}
	else
	{
		m_pJitterer = NULL;
		m_jitterDims = 0;
	}
	m_intrinsicDims = m_pNN->relFeatures().size() - (m_paramDims + m_jitterDims);
	GDomListIterator itMins(pNode->field("mins"));
	m_pMins = new double[itMins.remaining()];
	GVec::deserialize(m_pMins, itMins);
	GDomListIterator itRanges(pNode->field("ranges"));
	m_pRanges = new double[itRanges.remaining()];
	GVec::deserialize(m_pRanges, itRanges);
	m_onePass = pNode->field("op")->asBool();
}

// virtual
GUnsupervisedBackProp::~GUnsupervisedBackProp()
{
	delete(m_pJitterer);
	delete(m_pNN);
//	delete(m_pRevNN);
	delete[] m_pParamRanges;
	delete(m_pIntrinsic);
	delete[] m_pMins;
	delete[] m_pRanges;
	delete(m_pProgress);
}

GDomNode* GUnsupervisedBackProp::serialize(GDom* pDoc) const
{
	size_t channels = m_pNN->relLabels().size();
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "params", GIndexVec::serialize(pDoc, m_pParamRanges, m_paramDims));
	pNode->addField(pDoc, "nn", m_pNN->serialize(pDoc));
//	pNode->addField(pDoc, "rev", m_pRevNN->serialize(pDoc));
	GAssert(m_paramDims + m_jitterDims + m_intrinsicDims == m_pNN->relFeatures().size());
	pNode->addField(pDoc, "bias", pDoc->newBool(m_useInputBias));
	if(m_pJitterer)
		pNode->addField(pDoc, "jitterer", m_pJitterer->serialize(pDoc));
	pNode->addField(pDoc, "mins", GVec::serialize(pDoc, m_pMins, channels));
	pNode->addField(pDoc, "ranges", GVec::serialize(pDoc, m_pRanges, channels));
	pNode->addField(pDoc, "op", pDoc->newBool(m_onePass));
	return pNode;
}

void GUnsupervisedBackProp::trackProgress()
{
	delete(m_pProgress);
	m_pProgress = new GMatrix(0, 2);
}

void GUnsupervisedBackProp::setNeuralNet(GNeuralNet* pNN)
{
	delete(m_pNN);
	m_pNN = pNN;
}

void GUnsupervisedBackProp::setParams(vector<size_t>& paramRanges)
{
	m_paramDims = paramRanges.size();
	delete[] m_pParamRanges;
	m_pParamRanges = new size_t[m_paramDims];
	for(size_t i = 0; i < m_paramDims; i++)
		m_pParamRanges[i] = paramRanges[i];
	m_cvi.reset(m_paramDims, m_pParamRanges);
}

void GUnsupervisedBackProp::setIntrinsic(GMatrix* pIntrinsic)
{
	if(pIntrinsic->cols() != m_intrinsicDims)
		throw Ex("Expected ", to_str(m_intrinsicDims), " cols. Got ", to_str(pIntrinsic->cols()));
	delete(m_pIntrinsic);
	m_pIntrinsic = pIntrinsic;
}

void GUnsupervisedBackProp::setJitterer(GImageJitterer* pJitterer)
{
	delete(m_pJitterer);
	m_pJitterer = pJitterer;
}

// virtual
GMatrix* GUnsupervisedBackProp::reduce(const GMatrix& in)
{
	// Compute pixels and channels
	size_t pixels = 1;
	for(size_t i = 0; i < m_paramDims; i++)
		pixels *= m_pParamRanges[i];
	size_t channels = in.cols() / pixels;
	if((pixels * channels) != (size_t)in.cols())
		throw Ex("params don't line up");

	// Compute the mins and ranges
	delete[] m_pMins;
	delete[] m_pRanges;
	m_pMins = new double[channels];
	m_pRanges = new double[channels];
	GVec::setAll(m_pMins, 1e308, channels);
	GVec::setAll(m_pRanges, -1e308, channels);
	for(size_t j = 0; j < pixels; j++)
	{
		for(size_t i = 0; i < channels; i++)
		{
			double inMin = in.columnMin(channels * j + i);
			double inMax = in.columnMax(channels * j + i);
			m_pMins[i] = std::min(m_pMins[i], inMin);
			m_pRanges[i] = std::max(m_pRanges[i], inMax);
		}
	}
	for(size_t i = 0; i < channels; i++)
	{
		if(m_pMins[i] > 1e200)
			m_pMins[i] = 0.0;
		if(m_pRanges[i] <= m_pMins[i])
			m_pRanges[i] = m_pMins[i] + 1.0;
		m_pRanges[i] -= m_pMins[i];
		//std::cerr << "Min " << to_str(i) << "=" << to_str(m_pMins[i]) << ", Range " << to_str(i) << "=" << to_str(m_pRanges[i]) << "\n";
	}

	// Init
	m_jitterDims = 0;
	GTEMPBUF(double, pixBuf, channels);
	if(m_pJitterer)
	{
		if(m_paramDims != 2)
			throw Ex("An image jitterer can only be used in conjunction with 2 param dims");
		if(channels != m_pJitterer->channels() || m_pParamRanges[0] != m_pJitterer->wid() || m_pParamRanges[1] != m_pJitterer->hgt())
			throw Ex("The image jitterer was constructed with mismatching parameters");
		m_jitterDims = 4;
	}
	GUniformRelation featureRel(m_paramDims + m_jitterDims + m_intrinsicDims);
	GUniformRelation labelRel(channels);
	m_pNN->setUseInputBias(m_useInputBias);
	m_pNN->beginIncrementalLearning(featureRel, labelRel);
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	nn.setUseInputBias(m_useInputBias);
	nn.beginIncrementalLearning(featureRel, labelRel);

	// Train
	double* pParams = new double[(m_paramDims + m_jitterDims + m_intrinsicDims) * 2];
	ArrayHolder<double> hParams(pParams);
	double* pJitters = pParams + m_paramDims;
	double* pIntrinsic = pJitters + m_jitterDims;
	double* pGradientOfInputs = pIntrinsic + m_intrinsicDims;
	size_t totalPasses = 3;
	if(m_onePass)
		totalPasses = 1;
	if(m_pNN->layerCount() == 1)
		totalPasses = 1;
	for(size_t pass = 0; pass < totalPasses; pass++)
	{
		GNeuralNet* pNN = m_pNN;
		if(pass == 0)
		{
			if(totalPasses > 1)
				pNN = &nn;

			// Initialize the intrinsic matrix
			delete(m_pIntrinsic);
			m_pIntrinsic = new GMatrix(in.rows(), m_intrinsicDims);
			for(size_t i = 0; i < in.rows(); i++)
			{
				double* pVec = m_pIntrinsic->row(i);
				for(size_t j = 0; j < m_intrinsicDims; j++)
					*(pVec++) = 0.1 * nn.rand().normal();
			}
		}

		double learningRate = 0.1;
		double momentum = 0.0;
		double regularizer = 0.0;
		switch(totalPasses - pass)
		{
			case 3: regularizer = 1e-6; break;
			case 2: regularizer = 1e-8; break;
			case 1: break;
			default: throw Ex("Unexpected case");
		}
		pNN->setLearningRate(learningRate);
		pNN->setMomentum(momentum);
		size_t sampleSize = std::max((size_t)1, (size_t)ceil(sqrt((double)m_cvi.coordCount())));
		size_t batchSize = std::max((size_t)ceil(sqrt((double)in.rows())), in.rows() / sampleSize);

//sampleSize = 1;
//batchSize = 100;

		size_t batches = 1000;
		double decay = pow(0.05, 1.0 / batches);
		for(size_t i = 0; i < batches; i++)
		{
			double sse = 0;
			for(size_t j = 0; j < 200 * batchSize; j++)
			{
				// Pick a row
				size_t r = (size_t)nn.rand().next(in.rows());
				const double* pIn = in[r];
				double* pInt = m_pIntrinsic->row(r);
				for(size_t k = 0; k < sampleSize; k++) // use the same row for a few iterations to reduce page swaps
				{
					// Pick a row, pixel, and channel
					m_cvi.setRandom(&nn.rand());
					size_t c = (size_t)nn.rand().next(channels);
					m_cvi.currentNormalized(pParams);

					// Get the pixel
					const double* pPix;
					if(m_pJitterer)
					{
						GVec::copy(pJitters, m_pJitterer->pickParams(nn.rand()), m_jitterDims);
						m_pJitterer->transformedPix(pIn, m_cvi.current()[0], m_cvi.current()[1], pixBuf);
						pPix = pixBuf;
					}
					else
						pPix = pIn + m_cvi.currentIndex() * channels;

					// Do backprop
					GVec::copy(pIntrinsic, pInt, m_intrinsicDims);
					double prediction = pNN->forwardPropSingleOutput(pParams, c);
					double target = (pPix[c] - m_pMins[c]) / m_pRanges[c];
					GAssert(target >= 0.0 && target <= 1.0 && prediction >= 0.0 && prediction <= 1.0);
					double err = target - prediction;
					sse += (err * err);
					pNN->backpropagateSingleOutput(c, target);

					// Calculate the gradient of the inputs
					if(pass != 1)
						pNN->gradientOfInputsSingleOutput(c, pGradientOfInputs);

					// Update weights
					pNN->scaleWeightsSingleOutput(c, 1.0 - (learningRate * regularizer));
					pNN->descendGradientSingleOutput(c, pParams, pNN->learningRate(), pNN->momentum());

					// Update inputs
					if(pass != 1)
					{
						GVec::multiply(pIntrinsic, 1.0 - learningRate * regularizer, m_intrinsicDims);
						GVec::addScaled(pIntrinsic, -learningRate, pGradientOfInputs + m_paramDims + m_jitterDims, m_intrinsicDims);
						GVec::copy(pInt, pIntrinsic, m_intrinsicDims);
					}
				}
			}
			double rmse = sqrt(sse / batchSize);
			if(m_pProgress)
			{
				double* pProg = m_pProgress->newRow();
				pProg[0] = (double)(batches * pass + i);
				pProg[1] = rmse;
			}

			// Print progress
			std::cerr << "Pass " << to_str(pass + 1) << "/" << to_str(totalPasses) << ", Batch " << to_str(i + 1) << "/" << to_str(batches) << "\n";
			learningRate *= decay;
		}
	}

	// Normalize the intrinsic values
	for(size_t i = 0; i < m_intrinsicDims; i++)
	{
		//std::cerr << "Before intrinsic dim=" << to_str(i) << ", min=" << to_str(m_pIntrinsic->columnMin(i)) << ", max=" << to_str(m_pIntrinsic->columnMax(i)) << "\n";
		double _min = m_pIntrinsic->columnMin(i);
		double _max = m_pIntrinsic->columnMax(i);
		if(_max - _min < 1e-12)
			_max = _min + 1e-6;
		GNeuralNetLayer* pInputLayer = &m_pNN->layer(0);
		pInputLayer->renormalizeInput(m_paramDims + m_jitterDims + i, _min, _max);
		m_pIntrinsic->normalizeColumn(i, _min, _max);
		//std::cerr << " After intrinsic dim=" << to_str(i) << ", min=" << to_str(m_pIntrinsic->columnMin(i)) << ", max=" << to_str(m_pIntrinsic->columnMax(i)) << "\n";
	}

	GMatrix* pOut = m_pIntrinsic;
	m_pIntrinsic = NULL;
	return pOut;
}

class GUBPTargetFunc : public GTargetFunction
{
protected:
	GUnsupervisedBackProp* m_pUBP;
	double* m_pInit;
	double* m_pPrediction;
	size_t m_labelDims;
	const double* m_pTarget;

public:
	GUBPTargetFunc(GUnsupervisedBackProp* pUBP, double* pInit, const double* pTarget)
	: GTargetFunction(pUBP->featureDims()), m_pUBP(pUBP), m_pInit(pInit), m_pTarget(pTarget)
	{
		m_labelDims = m_pUBP->labelDims();
		m_pPrediction = new double[m_labelDims];
	}

	virtual ~GUBPTargetFunc()
	{
		delete[] m_pPrediction;
	}

	virtual bool isStable() { return true; }

	virtual bool isConstrained() { return false; }

	virtual void initVector(double* pVector)
	{
		GVec::copy(pVector, m_pInit, m_pUBP->featureDims());
	}

	virtual double computeError(const double* pVector)
	{
		m_pUBP->lowToHi(pVector, m_pPrediction);
		double err = 0.0;
		for(size_t k = 0; k < m_labelDims; k++)
		{
			double d = (m_pTarget[k] - m_pPrediction[k]) * pow(2.58, m_pTarget[k] / 128 - 1.0);
			err += (d * d);
		}
		return err;
	}
};

void GUnsupervisedBackProp::hiToLow(const double* pIn, double* pOut)
{
	throw Ex("Not implemented yet");
/*
	// Compute values
//	size_t channels = m_pNN->labelDims();

	// Init
	if(!m_pNN->hasTrainingBegun())
		throw Ex("Not trained");
	if(m_pNN->relFeatures()->size() != m_paramDims + m_jitterDims + m_intrinsicDims)
		throw Ex("Incorrect number of inputs Expected ", to_str(m_paramDims + m_jitterDims + m_intrinsicDims), ", got ", to_str(m_pNN->relFeatures()->size()));

	// Use the reverse map
	double* pParams = new double[m_paramDims + m_jitterDims + m_intrinsicDims];
	ArrayHolder<double> hParams(pParams);
	double* pJitters = pParams + m_paramDims;
	//GVec::setAll(pJitters, 0.5, m_jitterDims + m_intrinsicDims);
	m_pRevNN->predict(pIn, pJitters);

	// Refine
	double prevErr = 1e308;
	for(double learningRate = 0.0001; learningRate > 0.00005; )
	{
		m_pNN->setLearningRate(learningRate);
		double sse = 0;
		for(size_t i = 0; i < 1e7; i++)
		{
			// Pick a pixel, and channel
			m_cvi.setRandom(m_pRand);
			size_t c = (size_t)m_pRand->next(channels);
			m_cvi.currentNormalized(pParams);

			// Get the pixel
			const double* pPix = pIn + m_cvi.currentIndex() * channels;

			// Do backprop
			double prediction = m_pNN->forwardPropSingleOutput(pParams, c);
			double target = (pPix[c] - m_pMins[c]) / m_pRanges[c];
//			double err = target - prediction;
double err = (target - prediction) * pow(2.58, target + target - 1.0);
			sse += (err * err);
//			m_pNN->setErrorSingleOutput(target, c);
m_pNN->setErrorSingleOutput(prediction + err, c);
			m_pNN->backProp()->backpropagateSingleOutput(c);

			// Update weights and inputs
			m_pNN->backProp()->adjustFeaturesSingleOutput(c, pParams, m_pNN->learningRate(), true);
			GVec::floorValues(pJitters, 0.2, m_jitterDims + m_intrinsicDims);
			GVec::capValues(pJitters, 0.8, m_jitterDims + m_intrinsicDims);
		}
		double rsse = sqrt(sse);
//		if(1.0 - rsse / prevErr < 0.0001)
			learningRate *= 0.5;
		prevErr = rsse;
	}

	GUBPTargetFunc targetFunc(this, pJitters, pIn);
	GHillClimber optimizer(&targetFunc);
	optimizer.searchUntil(500, 50, 0.0001);
	GVec::copy(pJitters, optimizer.currentVector(), m_jitterDims + m_intrinsicDims);


	GVec::copy(pOut, pJitters, m_jitterDims + m_intrinsicDims);
*/
}

void GUnsupervisedBackProp::lowToHi(const double* pIn, double* pOut)
{
	size_t channels = m_pNN->relLabels().size();
	GTEMPBUF(double, pParams, m_paramDims + m_jitterDims + m_intrinsicDims);
	GVec::copy(pParams + m_paramDims, pIn, m_jitterDims + m_intrinsicDims);
	m_cvi.reset();
	while(true)
	{
		m_cvi.currentNormalized(pParams);
		m_pNN->predict(pParams, pOut);
		for(size_t i = 0; i < channels; i++)
		{
			(*pOut) *= m_pRanges[i];
			(*pOut) += m_pMins[i];
			pOut++;
		}
		if(!m_cvi.advance())
			break;
	}
}

double* GUnsupervisedBackProp::mins()
{
	if(!m_pMins)
	{
		if(!m_pNN)
			throw Ex("No neural net has been set");
		m_pMins = new double[m_pNN->relLabels().size()];
	}
	return m_pMins;
}

double* GUnsupervisedBackProp::ranges()
{
	if(!m_pRanges)
	{
		if(!m_pNN)
			throw Ex("No neural net has been set");
		m_pRanges = new double[m_pNN->relLabels().size()];
	}
	return m_pRanges;
}

size_t GUnsupervisedBackProp::labelDims()
{
	return m_cvi.coordCount() * m_pNN->relLabels().size();
}










GScalingUnfolder::GScalingUnfolder()
: m_neighborCount(14),
m_targetDims(2),
m_passes(50),
m_refines_per_scale(100),
m_scaleRate(0.9),
m_rand(0)
{
}

GScalingUnfolder::GScalingUnfolder(GDomNode* pNode, GLearnerLoader& ll)
: GTransform(pNode, ll),
m_rand(0)
{
	throw Ex("Sorry, this method is not implemented yet");
}

// virtual
GScalingUnfolder::~GScalingUnfolder()
{

}

void GScalingUnfolder_adjustPoints(double* pA, double* pB, size_t dims, double curSqDist, double tarSqDist, GRand& rand)
{
	if(curSqDist == 0.0)
	{
		if(tarSqDist > 0.0)
		{
			// Perturb A and B by a small random amount to separate them
			double d = 0.01 * sqrt(tarSqDist);
			for(size_t i = 0; i < dims; i++)
			{
				*(pA++) += d * rand.normal();
				*(pB++) += d * rand.normal();
			}
		}
		return; // It is easier to just return than to recalculate curSqDist. This edge will probably be visited again soon anyway.
	}
	double scal = std::max(0.5, std::min(2.0, sqrt(tarSqDist) / sqrt(curSqDist)));
	for(size_t i = 0; i < dims; i++)
	{
		double t = 0.5 * (*pB * (1.0 + scal) + *pA * (1.0 - scal)) - *pB;
		*(pA++) -= t;
		*(pB++) += t;
	}
}

// static
void GScalingUnfolder::restore_local_distances_pass(GMatrix& intrinsic, GNeighborGraph& ng, GRand& rand)
{
/*
	// Do it in random order
	size_t dims = intrinsic.cols();
	GRandomIndexIterator& ii = ng.randomEdgeIterator(rand);
	ii.reset();
	size_t ind;
	while(ii.next(ind))
	{
		size_t a = ind / ng.neighborCount();
		size_t b = ng.cache()[ind];
		if(b != INVALID_INDEX)
		{
			double dTarget = ng.squaredDistanceTable()[ind];
			double* pA = intrinsic.row(a);
			double* pB = intrinsic.row(b);
			double dCur = GVec::squaredDistance(pA, pB, dims);
			GScalingUnfolder_adjustPoints(pA, pB, dims, dCur, dTarget, rand);
		}
	}
*/

	// Do it in breadth-first order
	size_t dims = intrinsic.cols();
	size_t edgeCount = ng.data()->rows() * ng.neighborCount();
	std::queue<size_t> q;
	GBitTable used(ng.data()->rows());
	size_t seed = INVALID_INDEX;
	while(seed == INVALID_INDEX)
	{
		seed = (size_t)rand.next(edgeCount);
		if(ng.cache()[seed] == INVALID_INDEX)
			seed = INVALID_INDEX;
	}
	q.push(seed);
	while(q.size() > 0)
	{
		// Get the next edge from the queue
		size_t edge = q.front();
		q.pop();
		size_t a = edge / ng.neighborCount();
		size_t b = ng.cache()[edge];
		if(b == INVALID_INDEX)
			continue;

		// add all edges that connect to either end of this edge
		if(!used.bit(a))
		{
			used.set(a);
			size_t ed = ng.neighborCount() * a;
			for(size_t i = 0; i < ng.neighborCount(); i++)
				q.push(ed++);
		}
		if(!used.bit(b))
		{
			used.set(b);
			size_t ed = ng.neighborCount() * b;
			for(size_t i = 0; i < ng.neighborCount(); i++)
				q.push(ed++);
		}

		double dTarget = ng.squaredDistanceTable()[edge];
		double* pA = intrinsic.row(a);
		double* pB = intrinsic.row(b);
		double dCur = GVec::squaredDistance(pA, pB, dims);
		GScalingUnfolder_adjustPoints(pA, pB, dims, dCur, dTarget, rand);
	}

}

void GScalingUnfolder::unfold(GMatrix& intrinsic, GNeighborGraph& nf, size_t encoderTrainIters, GNeuralNet* pEncoder, GNeuralNet* pDecoder, const GMatrix* pVisible)
{
	GRandomIndexIterator* ii = pEncoder ? new GRandomIndexIterator(intrinsic.rows(), m_rand) : NULL;
	Holder<GRandomIndexIterator> hII2(ii);
	for(size_t pass = 0; pass < m_passes; pass++)
	{
		// Scale up the data
		intrinsic.multiply(1.0 / m_scaleRate);

		for(size_t i = 0; i < m_refines_per_scale; i++)
			restore_local_distances_pass(intrinsic, nf, m_rand);

		// Train the encoder
		if(pVisible)
		{
			intrinsic.centerMeanAtOrigin();
			for(size_t i = 0; i < encoderTrainIters; i++)
			{
				ii->reset();
				size_t ind;
				while(ii->next(ind))
				{
					pEncoder->trainIncremental(pVisible->row(ind), intrinsic[ind]);
					pDecoder->trainIncremental(intrinsic[ind], pVisible->row(ind));
				}
			}
		}
	}
}

// static
size_t GScalingUnfolder::unfold_iter(GMatrix& intrinsic, GRand& rand, size_t neighborCount, double scaleFactor, size_t refinements)
{
	while(true)
	{
		// Find neighbors
		GKdTree kdtree(&intrinsic, neighborCount, NULL, false);
		GNeighborGraph ng(&kdtree, false);
		ng.fillCache();
		if(!ng.isConnected())
		{
			if(neighborCount < 1 || neighborCount >= intrinsic.rows() - 1)
				throw Ex("Invalid neighborCount");
			neighborCount = std::min(intrinsic.rows() - 1, std::max(neighborCount + 1, neighborCount * 3 / 2));
			continue;
		}

		// Scale up the data
		intrinsic.multiply(scaleFactor);

		// Refine the points to restore distances in local neighborhoods
		for(size_t i = 0; i < refinements; i++)
			restore_local_distances_pass(intrinsic, ng, rand);
		return neighborCount;
	}
}

// virtual
GMatrix* GScalingUnfolder::reduce(const GMatrix& in)
{
	// Find neighbors
	GKdTree kdtree(&in, m_neighborCount, NULL, false);
	GNeighborGraph nf(&kdtree, false);
	nf.fillCache();

	// Make a copy of the data
	GMatrix intrinsic;
	intrinsic.copy(&in);
	unfold(intrinsic, nf);

	// Shift the variance into the first few dimensions
	GPCA pca(m_targetDims);
	pca.train(intrinsic);
	return pca.transformBatch(intrinsic);
}





} // namespace GClasses
