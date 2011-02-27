/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
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
#include "GMath.h"
#include "GNeighborFinder.h"
#include "GNeuralNet.h"
#include "GPlot.h"
#include "GSparseMatrix.h"
#include "GTime.h"
#include "GTransform.h"
#include "GTwt.h"
#include "GVec.h"
#include <deque>
#include <set>
#include <map>
#include <iostream>
#include <string>

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
void GManifold::computeNeighborWeights(GMatrix* pData, size_t point, int k, const size_t* pNeighbors, double* pOutWeights)
{
	// Create a matrix of all the neighbors normalized around the origin
	size_t colCount = pData->cols();
	GMatrix z(pData->relation());
	double* pRow = pData->row(point);
	for(int i = 0; i < k; i++)
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
// 	for(int i = 0; i < pSquare->rows(); i++)
// 		pOutWeights[i] = 1;
// 	if(!pSquare->gaussianElimination(pOutWeights))
// 		ThrowError("Failed to find a solution in computeNeighborWeights");

	// Compute the weights the SVD way
	GMatrix* pInv = pSquare->pseudoInverse();
	Holder<GMatrix> hInv(pInv);
	for(size_t i = 0; i < pSquare->rows(); i++)
		pOutWeights[i] = GVec::sumElements(pInv->row(i), pInv->cols());

	// Normalize the weights to sum to one
	GVec::sumToOne(pOutWeights, pSquare->rows());
}

// static
GMatrix* GManifold::blendNeighborhoods(size_t index, GMatrix* pA, double ratio, GMatrix* pB, int neighborCount, size_t* pHood)
{
	// Copy the two neighborhoods
	size_t rowCount = pA->rows();
	size_t colCount = pA->cols();
	GMatrix neighborhoodA(pA->relation());
	GVec::copy(neighborhoodA.newRow(), pA->row(index), colCount);
	GMatrix neighborhoodB(pB->relation());
	GVec::copy(neighborhoodB.newRow(), pB->row(index), colCount);
	for(int j = 0; j < neighborCount; j++)
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
GMatrix* GManifold::blendEmbeddings(GMatrix* pA, double* pRatios, GMatrix* pB, int neighborCount, size_t* pNeighborTable, size_t seed)
{
	// Check params
	size_t rowCount = pA->rows();
	size_t colCount = pA->cols();
	if(pB->rows() != rowCount || pB->cols() != colCount)
		ThrowError("mismatching sizes");

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
		int i = 1;
		for(int j = 0; j < neighborCount; j++)
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
		GMatrix tentativeC(pC->relation());
		GMatrix tentativeD(pD->relation());
		GReleaseDataHolder hTentativeD(&tentativeD);
		GVec::copy(tentativeC.newRow(), pC->row(par), colCount);
		tentativeD.takeRow(pD->row(0));
		int i = 1;
		for(int j = 0; j < neighborCount; j++)
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
		for(int j = 0; j < neighborCount; j++)
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
GMatrix* GManifold::multiDimensionalScaling(GMatrix* pDistances, int targetDims, GRand* pRand, bool useSquaredDistances)
{
	size_t n = pDistances->rows();
	if((size_t)pDistances->cols() != n)
		ThrowError("Expected a square and symmetric distance matrix");

	// Square every element in the distance matrix (unless it's already squared) and ensure symmetry
	GMatrix* pD = new GMatrix(pDistances->relation());
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
	GMatrix* pEigs = pD->eigs(std::min((int)n, targetDims), pEigenVals, pRand, true);
	if(n < (size_t)targetDims)
	{
		ThrowError("targetDims cannot be larger than the number of rows or columns in the distance matrix");
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
	for(int i = 0; i < targetDims; i++)
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
	for(int i = 0; i < POINT_COUNT; i++)
		prng.cubical(foo[i], 2);

	// Make distance matrix
	GMatrix dst(POINT_COUNT, POINT_COUNT);
	for(int i = 0; i < POINT_COUNT; i++)
	{
		double* pRow = dst.row(i);
		for(int j = i + 1; j < POINT_COUNT; j++)
			pRow[j] = GVec::squaredDistance(foo.row(i), foo.row(j), 2);
	}

	// Do MDS
	GMatrix* pMDS = GManifold::multiDimensionalScaling(&dst, 2, &prng, true);
	Holder<GMatrix> hMDS(pMDS);
	for(int i = 0; i < POINT_COUNT; i++)
	{
		for(int j = 0; j < POINT_COUNT; j++)
		{
			double expected = sqrt(GVec::squaredDistance(foo.row(i), foo.row(j), 2));
			double actual = sqrt(GVec::squaredDistance(pMDS->row(i), pMDS->row(j), 2));
			if(std::abs(expected - actual) > 1e-5)
				ThrowError("failed");
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
	int m_nCycle;
	bool m_bAdjustable;
};

GManifoldSculpting::GManifoldSculpting(int nNeighbors, int targetDims, GRand* pRand)
: GManifoldLearner(), m_pRand(pRand), m_pNF(NULL)
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

void GManifoldSculpting::plotData(float radius)
{
	if((m_nPass & (m_nPass - 1)) != 0)
		return; // only plot if m_nPass is a power of 2
	GImage image;
	image.setSize(800, 800);
	image.clear(0xff000000);
	double* pVec = m_pData->row(0);
	GDoubleRect r(pVec[0], pVec[1], 1e-12, 1e-12);
	for(size_t i = 1; i < m_pData->rows(); i++)
	{
		pVec = m_pData->row(i);
		r.include(pVec[0], pVec[1]);
	}
	r.makeAspect(800, 800);
	GPlotWindow pw(&image, r.x, r.y, r.x + r.w, r.y + r.h);
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		pVec = m_pData->row(i);
		pw.dot(pVec[0], pVec[1], radius, gAHSV(0xff, (float)i / m_pData->rows(), 1.0f, 1.0f), 0xff000000);
	}
	string filename = "ms";
	filename += to_str(m_nPass);
	filename += ".png";
	image.savePng(filename.c_str());
}

void GManifoldSculpting::setPreprocessedData(GMatrix* pData)
{
	delete(m_pData);
	m_pData = pData;
}

// virtual
GMatrix* GManifoldSculpting::doit(GMatrix* pIn)
{
	beginTransform(pIn);

	// Do burn-in iterations
	while(m_scale > 0.01)
		squishPass((size_t)m_pRand->next(m_pData->rows()));

	// Squish until it doesn't improve for a while
	double dBestError = 1e308;
	for(int nItersSinceImprovement = 0; nItersSinceImprovement < 50; nItersSinceImprovement++)
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
	GMatrix* pDataOut = new GMatrix(relationAfter());
	pDataOut->newRows(m_pData->rows());
	pDataOut->copyColumns(0, m_pData, 0, m_nTargetDims);
	delete(m_pData);
	m_pData = NULL;
	return pDataOut;
}

void GManifoldSculpting::beginTransform(GMatrix* pRealSpaceData)
{
	m_nDimensions = pRealSpaceData->cols();
	if(!pRealSpaceData->relation()->areContinuous(0, m_nDimensions))
		ThrowError("Only continuous values are supported");

	// Calculate metadata
	calculateMetadata(pRealSpaceData);

	// Preprocess the data
	if(m_pData)
	{
		// Check the supplied pre-processed data
		if(m_pData->rows() != pRealSpaceData->rows())
			ThrowError("Preprocessed data has wrong number of points");
		if(m_pData->relation()->size() < (size_t)m_nTargetDims)
			ThrowError("Preprocessed data has too few dimensions");
	}
	else
	{
		// Preprocess the data
		if(pRealSpaceData->relation()->size() < 30)
			m_pData = GPCARotateOnly::transform(pRealSpaceData->relation()->size(), 0, pRealSpaceData, m_nTargetDims, m_pRand);
		else
		{
			size_t preserveDims = m_nTargetDims * 6;
			preserveDims = std::max((size_t)30, preserveDims);
			preserveDims = std::min(pRealSpaceData->relation()->size(), preserveDims);
			GPCA pca(preserveDims, m_pRand);
			pca.train(pRealSpaceData);
			m_pData = pca.transformBatch(pRealSpaceData);
		}
	}

	// Calculate the junk
	m_q.clear();
	m_nDimensions = m_pData->relation()->size();
	m_nPass = 0;
	m_scale = 1.0;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		struct GManifoldSculptingNeighbor* pArrNeighbors = record((int)i);
		for(int j = 0; j < m_nNeighbors; j++)
		{
			size_t neighbor = pArrNeighbors[j].m_nNeighbor;
			if(neighbor < m_pData->rows())
			{
				pArrNeighbors[j].m_junkSquaredDist = GVec::squaredDistance(m_pData->row(i) + m_nTargetDims, m_pData->row(neighbor) + m_nTargetDims, m_nDimensions - m_nTargetDims);
				int slot = pArrNeighbors[j].m_nNeighborsNeighborSlot;
				struct GManifoldSculptingNeighbor* pArrNeighborsNeighbors = record((int)neighbor);
				size_t neighborsNeighbor = pArrNeighborsNeighbors[slot].m_nNeighbor;
				pArrNeighbors[j].m_junkDotProd = GVec::dotProduct(m_pData->row(neighbor) + m_nTargetDims, m_pData->row(i) + m_nTargetDims, m_pData->row(neighbor) + m_nTargetDims, m_pData->row(neighborsNeighbor) + m_nTargetDims, m_nDimensions - m_nTargetDims);
			}
		}
	}
}

void GManifoldSculpting::calculateMetadata(GMatrix* pData)
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
				ThrowError("Data mismatch");
			if(pNF->neighborCount() != (size_t)m_nNeighbors)
				ThrowError("mismatching numbers of neighbors");
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
			struct GManifoldSculptingNeighbor* pArrNeighbors = record((int)i);
			for(int j = 0; j < m_nNeighbors; j++)
			{
				pArrNeighbors[j].m_nNeighbor = pHood[j];
				if(pHood[j] < pData->rows())
				{
					m_goodNeighbors++;
					pArrNeighbors[j].m_nNeighborsNeighborSlot = -1;
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
		pPoint = record((int)n);
		stuff((int)n)->m_nCycle = -1;
		for(int i = 0; i < m_nNeighbors; i++)
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
				for(int j = 0; j < m_nNeighbors; j++)
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
							pPoint[i].m_nNeighborsNeighborSlot = (size_t)j;
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

void GManifoldSculpting::clampPoint(int n)
{
	stuff(n)->m_bAdjustable = false;
}

int GManifoldSculpting::countShortcuts(int nThreshold)
{
	int nShortcuts = 0;
	for(size_t n = 0; n < m_pData->rows(); n++)
	{
		struct GManifoldSculptingNeighbor* pPoint = record(n);
		for(int i = 0; i < m_nNeighbors; i++)
		{
			if(pPoint[i].m_nNeighbor < m_pData->rows() && std::abs((int)pPoint[i].m_nNeighbor - (int)n) >= (int)nThreshold)
			{
				cout << "shortcut: " << (int)n << "," << (int)pPoint[i].m_nNeighbor << "\n";
				nShortcuts++;
			}
		}
	}
	return nShortcuts;
}

double GManifoldSculpting::vectorCorrelation(double* pdA, double* pdV, double* pdB)
{
	double dDotProd = 0;
	double dMagA = 0;
	double dMagB = 0;
	double dA, dB;
	int n;
	for(n = 0; n < m_nDimensions; n++)
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

double GManifoldSculpting::vectorCorrelation2(double squaredScale, int a, int vertex, struct GManifoldSculptingNeighbor* pNeighborRec)
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
	int n;
	for(n = 0; n < m_nTargetDims; n++)
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

double GManifoldSculpting::averageNeighborDistance(int nDims)
{
	double dSum = 0;
	size_t goodNeighbors = 0;
	for(size_t nPoint = 0; nPoint < m_pData->rows(); nPoint++)
	{
		struct GManifoldSculptingNeighbor* pPoint = record(nPoint);
		for(int n = 0; n < m_nNeighbors; n++)
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

double GManifoldSculpting::computeError(int nPoint)
{
	double dError = 0;
	double dDist;
	double dTheta;
	struct GManifoldSculptingNeighbor* pPoint = record(nPoint);
	struct GManifoldSculptingStuff* pNeighborStuff;
	int n;
	double squaredScale = m_scale * m_scale;
	int accepted = 0;
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

int GManifoldSculpting::adjustDataPoint(int nPoint, double* pError)
{
	bool bMadeProgress = true;
	double* pValues = m_pData->row(nPoint);
	double dErrorBase = computeError(nPoint);
	double dError = 0;
	double dStepSize = m_dLearningRate * (m_pRand->uniform() * .4 + .6); // We multiply the learning rate by a random value so that the points can get away from each other
	int nSteps;
	for(nSteps = 0; bMadeProgress && nSteps < 30; nSteps++)
	{
		bMadeProgress = false;
		for(int n = 0; n < m_nTargetDims; n++)
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
		ThrowError("You must call BeginTransform before calling this method");
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
	int nVisitedNodes = 0;
	int nSteps = 0;
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
		for(int n = 0; n < m_nNeighbors; n++)
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
	if(nSteps < (int)m_pData->rows())
		m_dLearningRate *= .87;
	else
		m_dLearningRate /= .91;
//	cout << "[Learning Rate: " << m_dLearningRate << "]\n";
	
	if(m_nPass % 20 == 0)
		moveMeanToOrigin();
	
	m_nPass++;
	return dTotalError;
}

// --------------------------------------------------------------------------
/*
// virtual
double GManifoldSculptingForControl::supervisedError(int nPoint)
{
	if(m_squaredLambda <= 0 || nPoint >= (int)m_pControlData->rows() - 1)
		return 0;
	int action = (int)m_pControlData->row(nPoint)[0];
	GTEMPBUF(double, pSum, m_nTargetDims);
	GVec::setAll(pSum, 0.0, m_nTargetDims);
	struct GManifoldSculptingNeighbor* pPoint = record(nPoint);
	int count = 0;
	for(int n = 0; n < m_nNeighbors; n++)
	{
		int neighbor = pPoint[n].m_nNeighbor;
		if((int)m_pControlData->row(neighbor)[0] == action && neighbor < (int)m_pControlData->rows() - 1)
		{
			GVec::add(pSum, m_pData->row(neighbor + 1), m_nTargetDims);
			GVec::subtract(pSum, m_pData->row(neighbor), m_nTargetDims);
			count++;
		}
	}
	if(count >= 1)
	{
		GVec::multiply(pSum, -1.0 / count, m_nTargetDims);
		GVec::add(pSum, m_pData->row(nPoint + 1), m_nTargetDims);
		GVec::subtract(pSum, m_pData->row(nPoint), m_nTargetDims);
		return (1.0 - m_scale) * m_squaredLambda * GVec::squaredMagnitude(pSum, m_nTargetDims);
	}
	else
		return 0;
}

*/






GIsomap::GIsomap(int neighborCount, int targetDims, GRand* pRand) : GManifoldLearner(), m_neighborCount(neighborCount), m_targetDims(targetDims), m_pNF(NULL), m_pRand(pRand), m_dropDisconnectedPoints(false)
{
}

GIsomap::GIsomap(GTwtNode* pNode)
: GManifoldLearner(pNode)
{
	m_targetDims = (int)pNode->field("targetDims")->asInt();
}

// virtual
GIsomap::~GIsomap()
{
}

GTwtNode* GIsomap::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = baseTwtNode(pDoc, "GIsomap");
	pNode->addField(pDoc, "targetDims", pDoc->newInt(m_targetDims));
	return pNode;
}

void GIsomap::setNeighborFinder(GNeighborFinder* pNF)
{
	m_pNF = pNF;
}

// virtual
GMatrix* GIsomap::doit(GMatrix* pIn)
{
	GNeighborFinder* pNF = m_pNF;
	Holder<GNeighborFinder> hNF(NULL);
	if(!pNF)
	{
		pNF = new GKdTree(pIn, m_neighborCount, NULL, true);
		hNF.reset(pNF);
	}

	// Compute the distance matrix using the Floyd Warshall algorithm
	GTEMPBUF(size_t, hood, pNF->neighborCount());
	GTEMPBUF(double, squaredDists, pNF->neighborCount());
	GFloydWarshall graph(pIn->rows());
	for(size_t i = 0; i < pIn->rows(); i++)
	{
		pNF->neighbors(hood, squaredDists, i);
		for(size_t j = 0; j < pNF->neighborCount(); j++)
		{
			if(hood[j] >= pIn->rows())
				continue;
			double d = sqrt(squaredDists[j]);
			graph.addDirectedEdge(i, hood[j], d);
		}
	}
	graph.compute();
	if(!graph.isConnected())
	{
		if(!m_dropDisconnectedPoints)
			ThrowError("The local neighborhoods do not form a connected graph. Increasing the neighbor count may be a good solution. Another solution is to specify to dropDisconnectedPoints.");
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














//#define SPARSE

// Locally Linear Embedding
class GLLEHelper
{
protected:
	GMatrix* m_pInputData;
	GMatrix* m_pOutputData;
	int m_nInputDims;
	int m_nTargetDims;
	int m_nNeighbors;
	size_t* m_pNeighbors;
#ifdef SPARSE
	GSparseMatrix* m_pWeights;
#else
	GMatrix* m_pWeights;
#endif
	GRand* m_pRand;

	GLLEHelper(GMatrix* pData, int nTargetDims, int nNeighbors, GRand* pRand);
public:
	~GLLEHelper();

	// Uses LLE to compute the reduced dimensional embedding of the data
	// associated with pNF.
	static GMatrix* doLLE(GNeighborFinder* pNF, int nTargetDims, GRand* pRand);

protected:
	void findNeighbors(GNeighborFinder* pNF);
	void findNeighborsTheSlowWay();
	void computeWeights();
	void computeEmbedding();
	GMatrix* releaseOutputData();
};

GLLEHelper::GLLEHelper(GMatrix* pData, int nTargetDims, int nNeighbors, GRand* pRand)
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
GMatrix* GLLEHelper::doLLE(GNeighborFinder* pNF, int nTargetDims, GRand* pRand)
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
	m_pWeights = new GMatrix(nRowCount, nRowCount); // todo: this should be a sparse matrix
#endif
	for(size_t n = 0; n < nRowCount; n++)
	{
		GManifold::computeNeighborWeights(m_pInputData, n, m_nNeighbors, m_pNeighbors + n * m_nNeighbors, pVec);
		int pos = 0;
		size_t* pHood = m_pNeighbors + n * m_nNeighbors;
#ifdef SPARSE
		for(int i = 0; i < m_nNeighbors; i++)
		{
			if(pHood[i] < nRowCount)
				m_pWeights->set(n, pHood[i], pVec[pos++]);
			else
				pos++;
		}
#else
		GVec::setAll(m_pWeights->row(n), 0.0, nRowCount);
		for(int i = 0; i < m_nNeighbors; i++)
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
	int row, col;
	m_pWeights->multiply(-1.0);
	int nRowCount = m_pInputData->rows();
#ifdef SPARSE
	for(row = 0; row < nRowCount; row++)
		m_pWeights->set(row, row, m_pWeights->get(row, row) + 1.0);
#else
	for(row = 0; row < nRowCount; row++)
		m_pWeights->row(row)[row] += 1.0;
#endif

	// Compute the smallest (m_nTargetDims+1) eigenvectors of (A^T)A, where A is m_pWeights
#ifdef SPARSE
	// The sparse matrix SVD way (not yet working)
	GSparseMatrix* pU;
	double* diag;
	GSparseMatrix* pV;
	m_pWeights->singularValueDecomposition(&pU, &diag, &pV);
	Holder<GSparseMatrix> hU(pU);
	ArrayHolder<double> hDiag(diag);
	Holder<GSparseMatrix> hV(pV);
	GMatrix* pEigVecs = new GMatrix(pV->cols());
	Holder<GMatrix> hEigVecs(pEigVecs);
	pEigVecs->newRows(m_nTargetDims + 1);
	for(int i = 1; i <= m_nTargetDims; i++)
	{
		unsigned int rowIn = pV->rows() - 1 - i;
		double* pRow = pEigVecs->row(i);
		for(int j = 0; j < (int)pV->cols(); j++)
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
	for(int i = 0; i < nRowCount; i++)
	{
		int r = std::min(m_nTargetDims, nRowCount - 1 - i);
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
	for(int i = 0; i <= m_nTargetDims; i++)
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


GLLE::GLLE(int neighborCount, int targetDims, GRand* pRand) : GManifoldLearner(), m_neighborCount(neighborCount), m_targetDims(targetDims), m_pNF(NULL), m_pRand(pRand)
{
}

GLLE::GLLE(GTwtNode* pNode)
: GManifoldLearner(pNode)
{
	m_targetDims = (int)pNode->field("targetDims")->asInt();
}

// virtual
GLLE::~GLLE()
{
}

GTwtNode* GLLE::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = baseTwtNode(pDoc, "GLLE");
	pNode->addField(pDoc, "targetDims", pDoc->newInt(m_targetDims));
	return pNode;
}

void GLLE::setNeighborFinder(GNeighborFinder* pNF)
{
	m_pNF = pNF;
}

// virtual
GMatrix* GLLE::doit(GMatrix* pIn)
{
	GNeighborFinder* pNF = m_pNF;
	Holder<GNeighborFinder> hNF(NULL);
	if(!pNF)
	{
		pNF = new GKdTree(pIn, m_neighborCount, NULL, true);
		hNF.reset(pNF);
	}
	return GLLEHelper::doLLE(pNF, m_targetDims, m_pRand);
}












/*
GManifoldUnfolder::GManifoldUnfolder(int neighborCount, int targetDims, GRand* pRand)
: m_neighborCount(neighborCount), m_targetDims(targetDims), m_pNF(NULL), m_pRand(pRand)
{
}

GManifoldUnfolder::GManifoldUnfolder(GTwtNode* pNode)
{
	ThrowError("Not implemented yet");
}

// virtual
GManifoldUnfolder::~GManifoldUnfolder()
{
}

GTwtNode* GManifoldUnfolder::toTwt(GTwtDoc* pDoc)
{
	ThrowError("Not implemented yet");
	return NULL;
}

void GManifoldUnfolder::setNeighborFinder(GNeighborFinder* pNF)
{
	m_pNF = pNF;
}

// virtual
GMatrix* GManifoldUnfolder::doit(GMatrix* pIn)
{
	GNeighborFinder* pNF = m_pNF;
	Holder<GNeighborFinder> hNF(NULL);
	if(!pNF)
	{
		pNF = new GKdTree(pIn, 0, m_neighborCount, NULL, true);
		hNF.reset(pNF);
	}
	return unfold(pNF, m_targetDims, m_pRand);
}

class GManifoldUnfolderTargetFunction : public GTargetFunction
{
protected:
	GMatrix* m_pData;
	int m_rows, m_cols;
	GNeighborFinder* m_pNF;
	size_t* m_pHood;
	double* m_pDists;
	double* m_pMean;

public:
	GManifoldUnfolderTargetFunction(GMatrix* pData, GNeighborFinder* pNF)
	: GTargetFunction(pData->rows() * pData->cols()), m_pData(pData), m_rows(pData->rows()), m_cols(pData->cols()), m_pNF(pNF)
	{
		int k = pNF->neighborCount();
		m_pHood = new size_t[k];
		m_pDists = new double[k];
		m_pMean = new double[m_cols];
	}

	virtual ~GManifoldUnfolderTargetFunction()
	{
		delete[] m_pHood;
		delete[] m_pDists;
		delete[] m_pMean;
	}

	virtual bool isStable() { return true; }

	virtual bool isConstrained() { return false; }

	virtual void initVector(double* pVector)
	{
		m_pData->toVector(pVector);
	}

	virtual double computeError(const double* pVector)
	{
		// Penalize broken neighbor relationships
		int k = m_pNF->neighborCount();
		double err = 0;
		GVec::setAll(m_pMean, 0.0, m_cols);
		for(int i = 0; i < m_rows; i++)
		{
			m_pNF->neighbors(m_pHood, m_pDists, i);
			for(int j = 0; j < k; j++)
			{
				double d = std::max(0.0, sqrt(GVec::squaredDistance(pVector + i * k, pVector + m_pHood[j] * k, m_cols)) - sqrt(m_pDists[j]));
				err += (1e8 * d * d);
			}
			GVec::add(m_pMean, pVector + i * k, m_cols);
		}

		// Reward variance
		for(int i = 0; i < m_rows; i++)
			err -= GVec::squaredDistance(pVector + i * k, m_pMean, m_cols);

		return err;
	}
};

GMatrix* GManifoldUnfolder::unfold(GNeighborFinder* pNF, int targetDims, GRand* pRand)
{
	// Make sure the neighbor finder is cached
	Holder<GNeighborFinderCacheWrapper> hNF(NULL);
	if(!pNF->isCached())
	{
		GNeighborFinderCacheWrapper* pNF2 = new GNeighborFinderCacheWrapper(pNF, false);
		hNF.reset(pNF2);
		pNF = pNF2;
	}
	int rows = pNF->data()->rows();
	int cols = pNF->data()->cols();
	GManifoldUnfolderTargetFunction targetFunc(pNF->data(), pNF);
	GHillClimber optimizer(&targetFunc);
	//GStochasticGreedySearch optimizer(&targetFunc, m_pRand);
	optimizer.searchUntil(100, // nBurnInIterations
				30, // nIterations
				0.01 //dImprovement
				);
	GMatrix* pUnfolded = new GMatrix(cols);
	Holder<GMatrix> hUnfolded(pUnfolded);
	pUnfolded->newRows(rows);
	pUnfolded->fromVector(optimizer.currentVector(), rows);
	GPCA pca(targetDims, m_pRand);
	pca.train(pUnfolded);
	return pca.transformBatch(pUnfolded);
}
*/

















GBreadthFirstUnfolding::GBreadthFirstUnfolding(int reps, int neighborCount, int targetDims, GRand* pRand)
: m_reps(reps), m_neighborCount(neighborCount), m_targetDims(targetDims), m_pNF(NULL), m_useMds(true), m_pRand(pRand)
{
}

GBreadthFirstUnfolding::GBreadthFirstUnfolding(GTwtNode* pNode, GRand* pRand)
: m_reps((int)pNode->field("reps")->asInt()), m_neighborCount((int)pNode->field("neighbors")->asInt()), m_targetDims((int)pNode->field("targetDims")->asInt()), m_pNF(NULL), m_useMds(pNode->field("useMds")->asBool()), m_pRand(pRand)
{
}

// virtual
GBreadthFirstUnfolding::~GBreadthFirstUnfolding()
{
}

GTwtNode* GBreadthFirstUnfolding::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "reps", pDoc->newInt(m_reps));
	pNode->addField(pDoc, "neighbors", pDoc->newInt(m_neighborCount));
	pNode->addField(pDoc, "targetDims", pDoc->newInt(m_targetDims));
	pNode->addField(pDoc, "useMds", pDoc->newBool(m_useMds));
	if(m_pNF)
		ThrowError("sorry, serializing a neighbor finder is not yet implemented");
	return pNode;
}

void GBreadthFirstUnfolding::setNeighborFinder(GNeighborFinder* pNF)
{
	m_pNF = pNF;
}

// virtual
GMatrix* GBreadthFirstUnfolding::doit(GMatrix* pIn)
{
	// Obtain the neighbor finder
	GNeighborFinder* pNF = m_pNF;
	Holder<GNeighborFinder> hNF(NULL);
	if(!pNF)
	{
		pNF = new GKdTree(pIn, m_neighborCount, NULL, true);
		hNF.reset(pNF);
	}

	// Make sure the neighbor finder is cached
	Holder<GNeighborFinderCacheWrapper> hNF2(NULL);
	if(!pNF->isCached())
	{
		GNeighborFinderCacheWrapper* pNF2 = new GNeighborFinderCacheWrapper(pNF, false);
		hNF2.reset(pNF2);
		pNF = pNF2;
	}
	GNeighborFinderCacheWrapper* pCachedNF = (GNeighborFinderCacheWrapper*)pNF;
	pCachedNF->fillCache();
	size_t* pNeighborTable = pCachedNF->cache();
	double* pSquaredDistances = pCachedNF->squaredDistanceTable();
	GMatrix* pData = pNF->data();

	// Learn the manifold
	double* pGlobalWeights = new double[pIn->rows() * 2];
	ArrayHolder<double> hGlobalWeights(pGlobalWeights);
	double* pLocalWeights = pGlobalWeights + pIn->rows();
	GVec::setAll(pGlobalWeights, 0.0, pIn->rows());
	Holder<GMatrix> hFinal(NULL);
	for(int i = 0; i < m_reps; i++)
	{
		GMatrix* pRep = unfold(pData, pNeighborTable, pSquaredDistances, (size_t)m_pRand->next(pData->rows()), pLocalWeights);
		if(hFinal.get())
		{
			GVec::add(pGlobalWeights, pLocalWeights, pIn->rows());
			GVec::pairwiseDivide(pLocalWeights, pGlobalWeights, pIn->rows());
			Holder<GMatrix> hRep(pRep);
			hFinal.reset(GManifold::blendEmbeddings(pRep, pLocalWeights, hFinal.get(), m_neighborCount, pNeighborTable, (size_t)m_pRand->next(pData->rows())));
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
	int pos = 1;	
	for(int i = 0; i < m_neighborCount; i++)
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
		map<size_t,int> indexMap;
		for(int j = 0; j < m_neighborCount; j++)
		{
			if(pCurNeighbors[j] != INVALID_INDEX)
				indexMap.insert(make_pair(pCurNeighbors[j], j));
		}
		for(size_t j = 0; j < pLocal->rows(); j++)
		{
			size_t indexJ = indexes[j];
			map<size_t,int>::iterator it = indexMap.find(indexJ);
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
	for(int iters = 0; iters < 30; iters++)
	{
		double err = 0;
		pTablePos = distTable;
		for(size_t j = 0; j < pLocal->rows(); j++)
		{
			for(size_t i = 0; i < pLocal->rows(); i++)
			{
				if(*pTablePos != UNKNOWN_REAL_VALUE)
					err += GVec::refinePoint(pLocal->row(i), pLocal->row(j), m_targetDims, *pTablePos, 0.1, m_pRand);
				pTablePos++;
			}
		}
		if(iters == 0)
			firstErr = err;
		else if(iters == 29 && err > firstErr)
			ThrowError("made it worse");
	}
}

GMatrix* GBreadthFirstUnfolding::reduceNeighborhood(GMatrix* pIn, size_t index, size_t* pNeighborhoods, double* pSquaredDistances)
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
		for(int j = 0; j < m_neighborCount; j++)
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
			for(int j = 0; j < m_neighborCount; j++)
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
		pReducedNeighborhood = GManifold::multiDimensionalScaling(graph.costMatrix(), m_targetDims, m_pRand, false);
	}
	else
	{
		// Make a local neighborhood
		GMatrix local(pIn->relation());
		GReleaseDataHolder hLocal(&local);
		local.takeRow(pIn->row(index));
		size_t* pHood = pNeighborhoods + m_neighborCount * index;
		for(int j = 0; j < m_neighborCount; j++)
		{
			if(pHood[j] < pIn->rows())
				local.takeRow(pIn->row(pHood[j]));
		}

		// Use PCA to reduce the neighborhood
		GPCA pca(m_targetDims, m_pRand);
		pca.train(&local);
		pReducedNeighborhood = pca.transformBatch(&local);
	}

	return pReducedNeighborhood;
}

GMatrix* GBreadthFirstUnfolding::unfold(GMatrix* pIn, size_t* pNeighborTable, double* pSquaredDistances, size_t seed, double* pOutWeights)
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
		int i = 1;
		for(int j = 0; j < m_neighborCount; j++)
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
		GMatrix tentativeC(pOut->relation());
		GMatrix tentativeD(pLocal->relation());
		GReleaseDataHolder hTentativeD(&tentativeD);
		GVec::copy(tentativeC.newRow(), pOut->row(par), m_targetDims);
		tentativeD.takeRow(pLocal->row(0));
		size_t* pHood = pNeighborTable + m_neighborCount * par;
		int i = 1;
		for(int j = 0; j < m_neighborCount; j++)
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
		for(int j = 0; j < m_neighborCount; j++)
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














GNeuroPCA::GNeuroPCA(int targetDims, GRand* pRand)
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

void GNeuroPCA::computeComponent(GMatrix* pIn, GMatrix* pOut, size_t col, GMatrix* pPreprocess)
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
	double learningRate = 0.1;
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
				double* pTar = pIn->row(index);
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
			if(learningRate < 0.0001)
				break;
			learningRate *= 0.5;
		}
		prevRsse = rsse;
	}
}

double GNeuroPCA::computeSumSquaredErr(GMatrix* pIn, GMatrix* pOut, size_t cols)
{
	size_t dims = (size_t)pIn->cols();
	double sse = 0.0;
	for(size_t i = 0; i < pIn->rows(); i++)
	{
		double* pTar = pIn->row(i);
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
GMatrix* GNeuroPCA::doit(GMatrix* pIn)
{
	if(!pIn->relation()->areContinuous(0, pIn->cols()))
		ThrowError("GNeuroPCA doesn't support nominal values. You should filter with nominaltocat to make them real.");
	delete(m_pWeights);
	m_pWeights = new GMatrix(pIn->relation());
	m_pWeights->newRows(1 + m_targetDims); // the first row holds the biases

	// Initialize the biases
	size_t dims = (size_t)pIn->cols();
	{
		double* pBiases = m_pWeights->row(0);
		for(size_t i = 0; i < dims; i++)
		{
			double mean = pIn->mean(i);
			if((mean < m_pActivation->center() - m_pActivation->halfRange()) || (mean > m_pActivation->center() + m_pActivation->halfRange()))
				ThrowError("The data is expected to fall within the range of the activation function");
			*(pBiases++) = m_pActivation->inverse(mean);
		}
	}

	// Make space for the output data
	GMatrix* pOut = new GMatrix(pIn->rows(), m_targetDims);
	Holder<GMatrix> hOut(pOut);

	// Make a buffer for preprocessed info
	GMatrix preprocess(pIn->relation());
	preprocess.newRows(pIn->rows());

	// Compute the principle components
	double sse = 0;
	if(m_pEigVals)
		sse = computeSumSquaredErr(pIn, pOut, 0);
	for(size_t i = 0; i < m_targetDims; i++)
	{
		computeComponent(pIn, pOut, i, &preprocess);
		if(m_pEigVals)
		{
			double t = computeSumSquaredErr(pIn, pOut, i + 1);
			m_pEigVals[i] = (sse - t) / dims;
			sse = t;
		}
	}
	return hOut.release();
}















GUnsupervisedBackProp::GUnsupervisedBackProp(size_t intrinsicDims, GRand* pRand)
: m_intrinsicDims(intrinsicDims), m_pRand(pRand), m_paramDims(0), m_pParamRanges(NULL), m_cvi(0, NULL), m_rate(0.2)
{
	m_pNN = new GNeuralNet(m_pRand);
	//m_pNN->setActivationFunction(new GActivationIdentity(), true);
	//m_pNN->setLearningRate(0.03);
}

GUnsupervisedBackProp::GUnsupervisedBackProp(GTwtNode* pNode, GRand* pRand)
: m_pRand(pRand), m_cvi(0, NULL)
{
	ThrowError("not implemented yet");
}

// virtual
GUnsupervisedBackProp::~GUnsupervisedBackProp()
{
	delete(m_pNN);
	delete[] m_pParamRanges;
}

void GUnsupervisedBackProp::setParams(vector<size_t>& paramRanges)
{
	m_paramDims = paramRanges.size();
	m_pParamRanges = new size_t[m_paramDims];
	for(size_t i = 0; i < m_paramDims; i++)
		m_pParamRanges[i] = paramRanges[i];
	m_cvi.reset(m_paramDims, m_pParamRanges);
}

void GUnsupervisedBackProp::lowToHigh(const double* pIntrinsic, double* pObs)
{
	m_cvi.reset();
	while(true)
	{
		m_pNN->predict(pIntrinsic, pObs);
		if(!m_cvi.advance())
			break;
		pObs += m_pNN->labelDims();
	}
}

/*
double GUnsupervisedBackProp::measureError(double* pIntrinsic, double* pImage, size_t channels)
{
	double err = 0;
	GVec::copy(m_pBuf + m_paramDims, pIntrinsic, m_intrinsicDims);
	double* pPrediction = m_pBuf + m_paramDims + m_intrinsicDims;
	m_cvi.reset();
	while(true)
	{
		m_cvi.currentNormalized(m_pBuf);
		m_pNN->predict(m_pBuf, pPrediction);
		err += GVec::squaredDistance(pImage, pPrediction, channels);
		pImage += channels;
		if(!m_cvi.advance())
			break;
	}
	return err;
}
*/
// virtual
GMatrix* GUnsupervisedBackProp::doit(GMatrix* pIn)
{
	// Compute values
	size_t pixels = 1;
	for(size_t i = 0; i < m_paramDims; i++)
		pixels *= m_pParamRanges[i];
	size_t channels = pIn->cols() / pixels;
	if((pixels * channels) != (size_t)pIn->cols())
		ThrowError("params don't line up");

	// Init
	{
		sp_relation pFeatureRel = new GUniformRelation(m_paramDims + m_intrinsicDims);
		sp_relation pLabelRel = new GUniformRelation(channels);
		m_pNN->enableIncrementalLearning(pFeatureRel, pLabelRel);
	}
	GMatrix* pOut = new GMatrix(pIn->rows(), m_intrinsicDims);
	Holder<GMatrix> hOut(pOut);
	pOut->setAll(0.5);

	GBackProp* pBP = m_pNN->backProp();
	GBackPropLayer& bpLayer = pBP->layer(m_pNN->layerCount() - 1);
	double errorThresh = 0.0;

	// Learn
	double* pBuf = new double[m_paramDims + m_intrinsicDims];
	ArrayHolder<double> hBuf(pBuf);
	size_t* pIndexes = new size_t[pIn->rows()];
	ArrayHolder<size_t> hIndexes(pIndexes);
	GIndexVec::makeIndexVec(pIndexes, pIn->rows());
	size_t nextSpotlight = 0;
	for(double targetActive = 0.0; targetActive < pIn->rows() * 2; targetActive += m_rate)
	{
		// Refine
		double* pSpotlightTarget = pIn->row(nextSpotlight);
		double* pSpotlightContext = pOut->row(nextSpotlight);
		double spotlightErr = 0.0;
		if(m_paramDims == 0)
		{
			m_pNN->forwardProp(pSpotlightContext);
			spotlightErr = m_pNN->sumSquaredPredictionError(pSpotlightTarget);
		}
		double maxBelowThresh = 0.0;
		double minAboveThresh = 1e308;
		size_t activeCount = 0;
		double cumErr = 0.0;
		GIndexVec::shuffle(pIndexes, pIn->rows(), m_pRand);
		size_t* pInd = pIndexes;
		for(size_t i = 0; i < pIn->rows(); i++)
		{
			size_t index = *(pInd++);
			double* pObs = pIn->row(index);
			double* pContext = pOut->row(index);
			m_cvi.setRandom(m_pRand);
			m_cvi.currentNormalized(pBuf);
			GVec::copy(pBuf + m_paramDims, pContext, m_intrinsicDims);
			m_pNN->forwardProp(pBuf);
			m_pNN->setErrorOnOutputLayer(pObs + channels * m_cvi.currentIndex());

			double err = 0.0;
			for(vector<GBackPropNeuron>::iterator it = bpLayer.m_neurons.begin(); it != bpLayer.m_neurons.end(); it++)
				err += it->m_error * it->m_error;

			if(err < errorThresh)
			{
				pBP->backpropagate();
				pBP->descendGradient(pContext, m_pNN->learningRate(), 0.0);
				pBP->adjustFeatures(pContext, m_pNN->learningRate(), m_paramDims);
				activeCount++;
				maxBelowThresh = std::max(maxBelowThresh, err);

				// See if we can improve the spotlight point
				if(m_paramDims == 0)
				{
					double sse = m_pNN->sumSquaredPredictionError(pObs);
					cumErr += sse;
					if(m_pRand->uniform() * cumErr < sse)
						nextSpotlight = index;
					double err2 = m_pNN->sumSquaredPredictionError(pSpotlightTarget);
					if(err2 < spotlightErr)
					{
						spotlightErr = err2;
						GVec::copy(pSpotlightContext, pContext, m_intrinsicDims);
					}
				}
			}
			else
			{
				minAboveThresh = std::min(minAboveThresh, err);
				GVec::copy(pContext, pOut->row((size_t)m_pRand->next(pOut->rows())), m_intrinsicDims);
			}
		}
		if(activeCount <= (size_t)floor(targetActive))
			errorThresh = minAboveThresh + 1e-9;
		else
			errorThresh = maxBelowThresh;
	}
GTwtDoc doc;
doc.setRoot(m_pNN->toTwt(&doc));
doc.save("ubp.twt");
	return hOut.release();
}





} // namespace GClasses
