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

#include "GNeighborFinder.h"
#include "GVec.h"
#include "GRand.h"
#include "GPlot.h"
#include <stdlib.h>
#include <vector>
#include <queue>
#include <set>
#include "GOptimizer.h"
#include "GHillClimber.h"
#include <string.h>
#include "GGraph.h"
#include "GBitTable.h"
#include <deque>
#include "GDom.h"
#include "GKNN.h"
#include "GHolders.h"
#include "GTransform.h"
#include <sstream>
#include <string>
#include <iostream>
#include "GDistance.h"
#include <cmath>
#include <map>
#include "GPriorityQueue.h"
#include <memory>

namespace GClasses {

//using std::cerr;
using std::vector;
using std::priority_queue;
using std::set;
using std::deque;
using std::make_pair;
using std::pair;
using std::string;
using std::multimap;


void GNeighborFinder_InsertionSortNeighbors(size_t neighborCount, size_t* pNeighbors, double* pDistances)
{
	size_t tt;
	double t;
	for(size_t i = 1; i < neighborCount; i++)
	{
		for(size_t j = i; j > 0; j--)
		{
			if(pNeighbors[j] == INVALID_INDEX)
				break;
			if(pNeighbors[j - 1] != INVALID_INDEX && pDistances[j] >= pDistances[j - 1])
				break;

			// Swap
			tt = pNeighbors[j - 1];
			pNeighbors[j - 1] = pNeighbors[j];
			pNeighbors[j] = tt;
			t = pDistances[j - 1];
			pDistances[j - 1] = pDistances[j];
			pDistances[j] = t;
		}
	}
}

void GNeighborFinder::sortNeighbors(size_t neighborCount, size_t* pNeighbors, double* pDistances)
{
	// Use insertion sort if the list is small
	if(neighborCount < 7)
	{
		GNeighborFinder_InsertionSortNeighbors(neighborCount, pNeighbors, pDistances);
		return;
	}
	double t;
	size_t tt;
	size_t beg = 0;
	size_t end = neighborCount - 1;

	// Pick a pivot (using the median of 3 technique)
	double pivA = pDistances[0];
	double pivB = pDistances[neighborCount / 2];
	double pivC = pDistances[neighborCount - 1];
	double pivot;
	if(pivA < pivB)
	{
		if(pivB < pivC)
			pivot = pivB;
		else if(pivA < pivC)
			pivot = pivC;
		else
			pivot = pivA;
	}
	else
	{
		if(pivA < pivC)
			pivot = pivA;
		else if(pivB < pivC)
			pivot = pivC;
		else
			pivot = pivB;
	}

	// Do Quick Sort
	while(true)
	{
		while(beg < end && pNeighbors[beg] != INVALID_INDEX && pDistances[beg] < pivot)
			beg++;
		while(end > beg && (pNeighbors[end] == INVALID_INDEX || pDistances[end] > pivot))
			end--;
		if(beg >= end)
			break;
		t = pDistances[beg];
		pDistances[beg] = pDistances[end];
		pDistances[end] = t;
		tt = pNeighbors[beg];
		pNeighbors[beg] = pNeighbors[end];
		pNeighbors[end] = tt;
		beg++;
		end--;
	}

	// Recurse
	if(pNeighbors[beg] != INVALID_INDEX && pDistances[beg] < pivot)
		beg++;
	else if(beg == 0) // This could happen if they're all -1 (bad neighbors)
	{
		GNeighborFinder_InsertionSortNeighbors(neighborCount, pNeighbors, pDistances);
		return;
	}
	GNeighborFinder::sortNeighbors(beg, pNeighbors, pDistances);
	GNeighborFinder::sortNeighbors(neighborCount - beg, pNeighbors + beg, pDistances + beg);
}

void GNeighborFinder::sortNeighbors(size_t* pNeighbors, double* pDistances)
{
	GNeighborFinder::sortNeighbors(m_neighborCount, pNeighbors, pDistances);
}









GNeighborGraph::GNeighborGraph(GNeighborFinder* pNF, bool own)
: GNeighborFinder(pNF->data(), pNF->neighborCount()), m_pNF(pNF), m_own(own), m_pRandomEdgeIterator(NULL)
{
	m_pCache = new size_t[m_pData->rows() * m_neighborCount];
	m_pDissims = new double[m_pData->rows() * m_neighborCount];
	for(size_t i = 0; i < m_pData->rows(); i++)
		m_pCache[i * m_neighborCount] = m_pData->rows();
}

// virtual
GNeighborGraph::~GNeighborGraph()
{
	delete[] m_pCache;
	delete[] m_pDissims;
	if(m_own)
		delete(m_pNF);
	delete(m_pRandomEdgeIterator);
}

GRandomIndexIterator& GNeighborGraph::randomEdgeIterator(GRand& rand)
{
	if(!m_pRandomEdgeIterator)
		m_pRandomEdgeIterator = new GRandomIndexIterator(data()->rows() * neighborCount(), rand);
	return *m_pRandomEdgeIterator;
}

// virtual
void GNeighborGraph::neighbors(size_t* pOutNeighbors, size_t index)
{
	size_t* pCache = m_pCache + m_neighborCount * index;
	if(*pCache == m_pData->rows())
	{
		double* pDissims = m_pDissims + m_neighborCount * index;
		((GNeighborFinder*)m_pNF)->neighbors(pCache, pDissims, index);
	}
	memcpy(pOutNeighbors, pCache, sizeof(size_t) * m_neighborCount);
}

// virtual
void GNeighborGraph::neighbors(size_t* pOutNeighbors, double* pOutDistances, size_t index)
{
	size_t* pCache = m_pCache + m_neighborCount * index;
	double* pDissims = m_pDissims + m_neighborCount * index;
	if(*pCache == m_pData->rows())
		((GNeighborFinder*)m_pNF)->neighbors(pCache, pDissims, index);
	memcpy(pOutNeighbors, pCache, sizeof(size_t) * m_neighborCount);
	memcpy(pOutDistances, pDissims, sizeof(double) * m_neighborCount);
}

void GNeighborGraph::fillCache()
{
	size_t rowCount = m_pData->rows();
	size_t* pCache = m_pCache;
	double* pDissims = m_pDissims;
	for(size_t i = 0; i < rowCount; i++)
	{
		if(*pCache == m_pData->rows())
			((GNeighborFinder*)m_pNF)->neighbors(pCache, pDissims, i);
		pCache += m_neighborCount;
		pDissims += m_neighborCount;
	}
}

void GNeighborGraph::fillDistances(GDistanceMetric* pMetric)
{
	pMetric->init(&m_pData->relation(), false);
	double* pDissim = m_pDissims;
	size_t* pHood = m_pCache;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		const GVec& pA = m_pData->row(i);
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			const GVec& pB = m_pData->row(pHood[j]);
			*pDissim = pMetric->squaredDistance(pA, pB);
			pDissim++;
		}
		pHood += m_neighborCount;
	}
}

size_t GNeighborGraph::cutShortcuts(size_t cycleLen)
{
	GCycleCut cc(m_pCache, m_pData, m_neighborCount);
	cc.setCycleThreshold(cycleLen);
	return cc.cut();
}

void GNeighborGraph::patchMissingSpots(GRand* pRand)
{
	size_t rowCount = m_pData->rows();
	size_t* pCache = m_pCache;
	double* pDissims = m_pDissims;
	for(size_t i = 0; i < rowCount; i++)
	{
		if(*pCache == m_pData->rows())
			throw Ex("cache not filled out");
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			if(pCache[j] >= m_pData->rows())
			{
				size_t k = (size_t)pRand->next(m_neighborCount);
				size_t l;
				for(l = k; l < m_neighborCount; l++)
				{
					if(pCache[l] < m_pData->rows())
						break;
				}
				if(l >= m_neighborCount)
				{
					for(l = 0; l < k; l++)
					{
						if(pCache[l] < m_pData->rows())
							break;
					}
				}
				if(pCache[l] >= m_pData->rows())
					throw Ex("row has zero valid neighbors");
				if(pDissims)
					pDissims[j] = pDissims[l];
				pCache[j] = pCache[l];
			}
		}
		pCache += m_neighborCount;
		pDissims += m_neighborCount;
	}
}

void GNeighborGraph::normalizeDistances()
{
	size_t rowCount = m_pData->rows();
	size_t* pCache = m_pCache;
	double* pDissims = m_pDissims;
	double total = 0.0;
	for(size_t i = 0; i < rowCount; i++)
	{
		if(*pCache == m_pData->rows())
			throw Ex("cache not filled out");
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			pDissims[j] = sqrt(pDissims[j]);
			total += pDissims[j];
		}
		pCache += m_neighborCount;
		pDissims += m_neighborCount;
	}
	pDissims = m_pDissims;
	for(size_t i = 0; i < rowCount; i++)
	{
		double s = 0;
		for(size_t j = 0; j < m_neighborCount; j++)
			s += pDissims[j];
		s = 1.0 / s;
		for(size_t j = 0; j < m_neighborCount; j++)
			pDissims[j] *= s;
		pDissims += m_neighborCount;
	}
	total /= rowCount;
	pDissims = m_pDissims;
	for(size_t i = 0; i < rowCount; i++)
	{
		for(size_t j = 0; j < m_neighborCount; j++)
		{
			double d = pDissims[j] * total;
			pDissims[j] = (d * d);
		}
		pDissims += m_neighborCount;
	}
}

bool GNeighborGraph::isConnected()
{
	// Make a table containing bi-directional neighbor connections
	vector< vector<size_t> > bidirTable;
	bidirTable.resize(m_pData->rows());
	size_t* pHood = m_pCache;
	for(size_t i = 0; i < m_pData->rows(); i++)
		bidirTable[i].reserve(m_neighborCount * 2);
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		vector<size_t>& row = bidirTable[i];
		for(size_t j = 0; j < (size_t)m_neighborCount; j++)
		{
			if(*pHood < m_pData->rows())
			{
				row.push_back(*pHood);
				bidirTable[*pHood].push_back(i);
			}
			pHood++;
		}
	}

	// Use a breadth-first search to determine if the graph is fully connected
	GBitTable bt(m_pData->rows());
	deque<size_t> q;
	bt.set(0);
	q.push_back(0);
	size_t count = 1;
	while(q.size() > 0)
	{
		size_t n = q.front();
		q.pop_front();
		vector<size_t>& hood = bidirTable[n];
		for(vector<size_t>::iterator it = hood.begin(); it != hood.end(); it++)
		{
			if(!bt.bit(*it))
			{
				bt.set(*it);
				count++;
				if(count >= m_pData->rows())
					return true;
				q.push_back(*it);
			}
		}
	}
	return false;
}

// --------------------------------------------------------------------

/// This helper class keeps neighbors sorted as a binary heap, such that the most dissimilar
/// of the k-current-neighbors is always at the front of the heap.
class GClosestNeighborFindingHelper
{
protected:
	size_t m_found;
	size_t m_neighbors;
	size_t* m_pNeighbors;
	double* m_pDistances;

public:
	GClosestNeighborFindingHelper(size_t neighbors, size_t* pNeighbors, double* pDistances)
	: m_found(0), m_neighbors(neighbors), m_pNeighbors(pNeighbors), m_pDistances(pDistances)
	{
		GAssert(m_neighbors >= 1);
		for(size_t i = 0; i < m_neighbors; i++)
		{
			m_pNeighbors[i] = size_t(-1);
			m_pDistances[i] = 1e308;
		}
	}

	~GClosestNeighborFindingHelper()
	{
	}

	// Adds a point to the set of current neighbors if it is closer than the
	// most dissimilar of the k-current-neighbors
	void TryPoint(size_t index, double distance)
	{
		double* pHeapDist = m_pDistances - 1;
		size_t* pHeapNeigh = m_pNeighbors - 1;
		size_t heapPos;
		if(m_found < m_neighbors)
			heapPos = ++m_found;
		else
		{
			// Compare with the front of the heap, which holds the most dissimilar of the k-current-neighbors
			if(distance >= m_pDistances[0])
				return;

			// Release the most dissimilar of the k-current neighbors
			heapPos = 1;
			while(2 * heapPos <= m_neighbors)
			{
				if(2 * heapPos == m_neighbors || pHeapDist[2 * heapPos] > pHeapDist[2 * heapPos + 1])
				{
					pHeapDist[heapPos] = pHeapDist[2 * heapPos];
					pHeapNeigh[heapPos] = pHeapNeigh[2 * heapPos];
					heapPos = 2 * heapPos;
				}
				else
				{
					pHeapDist[heapPos] = pHeapDist[2 * heapPos + 1];
					pHeapNeigh[heapPos] = pHeapNeigh[2 * heapPos + 1];
					heapPos = 2 * heapPos + 1;
				}
			}
		}

		// Insert into heap
		pHeapDist[heapPos] = distance;
		pHeapNeigh[heapPos] = index;
		while(heapPos > 1 && pHeapDist[heapPos / 2] < pHeapDist[heapPos])
		{
			std::swap(pHeapDist[heapPos / 2], pHeapDist[heapPos]);
			std::swap(pHeapNeigh[heapPos / 2], pHeapNeigh[heapPos]);
			heapPos /= 2;
		}
	}

	double GetWorstDist()
	{
		return m_found >= m_neighbors ? m_pDistances[0] : 1e308;
	}

#ifndef NO_TEST_CODE
#	define TEST_NEIGHBOR_COUNT 33
	static void test()
	{
		size_t neighbors[TEST_NEIGHBOR_COUNT];
		double distances[TEST_NEIGHBOR_COUNT];
		GMatrix values1(0, 1);
		GMatrix values2(0, 1);
		GClosestNeighborFindingHelper ob(TEST_NEIGHBOR_COUNT, neighbors, distances);
		GRand prng(0);
		for(size_t i = 0; i < 300; i++)
		{
			double d = prng.uniform();
			ob.TryPoint(i, d);
			values1.newRow()[0] = d;
			values1.sort(0);
			values2.flush();
			for(size_t j = 0; j < std::min((size_t)TEST_NEIGHBOR_COUNT, values1.rows()); j++)
				values2.newRow()[0] = distances[j];
			values2.sort(0);
			for(size_t j = 0; j < std::min((size_t)TEST_NEIGHBOR_COUNT, values1.rows()); j++)
			{
				if(std::abs(values1[j][0] - values2[j][0]) > 1e-12)
					throw Ex("something is wrong");
			}
		}
	}
#endif
};

// --------------------------------------------------------------------------------

GNeighborFinderGeneralizing::GNeighborFinderGeneralizing(const GMatrix* pData, size_t neighbor_count, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinder(pData, neighbor_count), m_pMetric(pMetric), m_ownMetric(ownMetric)
{
	if(!m_pMetric)
	{
		m_pMetric = new GRowDistance();
		m_ownMetric = true;
	}
	m_pMetric->init(&pData->relation(), false);
}

// virtual
GNeighborFinderGeneralizing::~GNeighborFinderGeneralizing()
{
	if(m_ownMetric)
		delete(m_pMetric);
}

// --------------------------------------------------------------------------------

GBruteForceNeighborFinder::GBruteForceNeighborFinder(GMatrix* pData, size_t neighbor_count, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinderGeneralizing(pData, neighbor_count, pMetric, ownMetric)
{
}

GBruteForceNeighborFinder::~GBruteForceNeighborFinder()
{
}

// virtual
void GBruteForceNeighborFinder::reoptimize()
{
}

// virtual
void GBruteForceNeighborFinder::neighbors(size_t* pOutNeighbors, size_t index)
{
	GTEMPBUF(double, distances, m_neighborCount);
	neighbors(pOutNeighbors, distances, index);
}

// virtual
void GBruteForceNeighborFinder::neighbors(size_t* pOutNeighbors, double* pOutDistances, size_t index)
{
	GClosestNeighborFindingHelper helper(m_neighborCount, pOutNeighbors, pOutDistances);
	double dist;
	const GVec& pInputVector = m_pData->row(index);
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		if(i == index)
			continue;
		const GVec& pCand = m_pData->row(i);
		dist = m_pMetric->squaredDistance(pInputVector, pCand);
		helper.TryPoint(i, dist);
	}
}

// virtual
void GBruteForceNeighborFinder::neighbors(size_t* pOutNeighbors, double* pOutDistances, const GVec& pInputVector)
{
	GClosestNeighborFindingHelper helper(m_neighborCount, pOutNeighbors, pOutDistances);
	double dist;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		const GVec& pCand = m_pData->row(i);
		dist = m_pMetric->squaredDistance(pInputVector, pCand);
		helper.TryPoint(i, dist);
	}
}

// --------------------------------------------------------------------------------

class GKdNode
{
protected:
	double m_minDist;
	double* m_pOffset;
	size_t m_dims;

public:
	GKdNode(size_t dims)
	{
		m_dims = dims;
		m_pOffset = new double[dims];
		GVec::setAll(m_pOffset, 0.0, dims);
		m_minDist = 0;
	}

	virtual ~GKdNode()
	{
		delete[] m_pOffset;
	}

	virtual bool IsLeaf() = 0;

	// Builds an array of all the indexes in all of the leaf nodes that descend from me
	virtual size_t Gather(size_t* pOutIndexes) = 0;

	virtual void Insert(GKdTree* pTree, size_t index, double* pRow) = 0;

	virtual void Remove(GKdTree* pTree, size_t index, double* pRow) = 0;

	virtual void Rename(GKdTree* pTree, size_t oldIndex, size_t newIndex, double* pRow) = 0;

	double GetMinDist()
	{
		return m_minDist;
	}

	size_t GetDims()
	{
		return m_dims;
	}

	void CopyOffset(GKdNode* pParent)
	{
		GVec::copy(m_pOffset, pParent->m_pOffset, m_dims);
		m_minDist = pParent->m_minDist;
	}

	void AdjustOffset(size_t attr, double offset, const GVec& scaleFactors)
	{
		if(offset > m_pOffset[attr])
		{
			m_minDist -= (m_pOffset[attr] * m_pOffset[attr] * scaleFactors[attr] * scaleFactors[attr]);
			m_pOffset[attr] = offset;
			m_minDist += (m_pOffset[attr] * m_pOffset[attr] * scaleFactors[attr] * scaleFactors[attr]);
		}
	}
};


class GKdInteriorNode : public GKdNode
{
protected:
	GKdNode* m_pLess;
	GKdNode* m_pGreaterOrEqual;
	size_t m_size;
	size_t m_attr;
	double m_pivot;

public:
	size_t m_timeLeft;

	GKdInteriorNode(size_t dims, GKdNode* pLess, GKdNode* pGreaterOrEqual, size_t size, size_t attr, double pivot)
	 : GKdNode(dims), m_pLess(pLess), m_pGreaterOrEqual(pGreaterOrEqual), m_size(size), m_attr(attr), m_pivot(pivot)
	{
		m_timeLeft = (size_t)std::min((double)0x7fffffff, ((double)size * size) / 36 + 6);
	}

	virtual ~GKdInteriorNode()
	{
		delete(m_pLess);
		delete(m_pGreaterOrEqual);
	}

	virtual bool IsLeaf() { return false; }

	GKdNode* Rebuild(GKdTree* pTree)
	{
		size_t* pIndexes = new size_t[m_size];
		std::unique_ptr<size_t[]> hIndexes(pIndexes);
		size_t used = Gather(pIndexes);
		GAssert(used == m_size); // m_size is wrong. This may corrupt memory.
		return pTree->buildTree(used, pIndexes);
	}

	virtual void Insert(GKdTree* pTree, size_t index, double* pRow)
	{
		m_timeLeft--;
		if(pTree->isGreaterOrEqual(pRow, m_attr, m_pivot))
		{
			m_pGreaterOrEqual->Insert(pTree, index, pRow);
			m_size++;
			if(!m_pGreaterOrEqual->IsLeaf() && ((GKdInteriorNode*)m_pGreaterOrEqual)->m_timeLeft == 0 && m_timeLeft >= m_size / 4)
			{
				GKdNode* pNewNode = ((GKdInteriorNode*)m_pGreaterOrEqual)->Rebuild(pTree);
				delete(m_pGreaterOrEqual);
				m_pGreaterOrEqual = pNewNode;
			}
		}
		else
		{
			m_pLess->Insert(pTree, index, pRow);
			m_size++;
			if(!m_pLess->IsLeaf() && ((GKdInteriorNode*)m_pLess)->m_timeLeft == 0 && m_timeLeft >= m_size / 4)
			{
				GKdNode* pNewNode = ((GKdInteriorNode*)m_pLess)->Rebuild(pTree);
				delete(m_pLess);
				m_pLess = pNewNode;
			}
		}
	}

	virtual void Remove(GKdTree* pTree, size_t index, double* pRow)
	{
		m_timeLeft--;
		if(pTree->isGreaterOrEqual(pRow, m_attr, m_pivot))
			m_pGreaterOrEqual->Remove(pTree, index, pRow);
		else
			m_pLess->Remove(pTree, index, pRow);
		m_size--;
	}

	virtual void Rename(GKdTree* pTree, size_t oldIndex, size_t newIndex, double* pRow)
	{
		if(pTree->isGreaterOrEqual(pRow, m_attr, m_pivot))
			m_pGreaterOrEqual->Rename(pTree, oldIndex, newIndex, pRow);
		else
			m_pLess->Rename(pTree, oldIndex, newIndex, pRow);
	}

	GKdNode* GetLess() { return m_pLess; }
	GKdNode* GetGreaterOrEqual() { return m_pGreaterOrEqual; }
	size_t GetSize() { return m_size; }

	void GetDivision(size_t* pAttr, double* pPivot)
	{
		*pAttr = m_attr;
		*pPivot = m_pivot;
	}

	size_t Gather(size_t* pOutIndexes)
	{
		size_t n = m_pLess->Gather(pOutIndexes);
		return m_pGreaterOrEqual->Gather(pOutIndexes + n) + n;
	}
};


class GKdLeafNode : public GKdNode
{
protected:
	vector<size_t> m_indexes;

public:
	GKdLeafNode(size_t count, size_t* pIndexes, size_t dims, size_t maxLeafSize)
	 : GKdNode(dims)
	{
		m_indexes.reserve(std::max(count, maxLeafSize));
		for(size_t i = 0; i < count; i++)
			m_indexes.push_back(pIndexes[i]);
	}

	virtual ~GKdLeafNode()
	{
	}

	virtual bool IsLeaf() { return true; }

	size_t GetSize()
	{
		return m_indexes.size();
	}

	virtual void Insert(GKdTree* pTree, size_t index, double* pRow)
	{
		m_indexes.push_back(index);
	}

	virtual void Remove(GKdTree* pTree, size_t index, double* pRow)
	{
		size_t count = m_indexes.size();
		for(size_t i = 0; i < count; i++)
		{
			if(m_indexes[i] == index)
			{
				m_indexes[i] = m_indexes[count - 1];
				m_indexes.pop_back();
				return;
			}
		}
		GAssert(false); // failed to find index. Did the row change?
	}

	virtual void Rename(GKdTree* pTree, size_t oldIndex, size_t newIndex, double* pRow)
	{
		size_t count = m_indexes.size();
		for(size_t i = 0; i < count; i++)
		{
			if(m_indexes[i] == oldIndex)
			{
				m_indexes[i] = newIndex;
				return;
			}
		}
		GAssert(false); // failed to find index. Did the row change?
	}

	vector<size_t>* GetIndexes() { return &m_indexes; }

	size_t Gather(size_t* pOutIndexes)
	{
		for(vector<size_t>::iterator i = m_indexes.begin(); i < m_indexes.end(); i++)
		{
			*pOutIndexes = *i;
			pOutIndexes++;
		}
		return m_indexes.size();
	}
};


// --------------------------------------------------------------------------------------------------------

GKdTree::GKdTree(const GMatrix* pData, size_t neighbor_count, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinderGeneralizing(pData, neighbor_count, pMetric, ownMetric)
{
	m_maxLeafSize = 6;
	size_t count = pData->rows();
	GTEMPBUF(size_t, tmp, count);
	for(size_t i = 0; i < count; i++)
		tmp[i] = i;
	m_pRoot = buildTree(count, tmp);
}

// virtual
GKdTree::~GKdTree()
{
	delete(m_pRoot);
}

void GKdTree::computePivotAndGoodness(size_t count, size_t* pIndexes, size_t attr, double* pOutPivot, double* pOutGoodness)
{
	size_t valueCount = m_pMetric->relation()->valueCount(attr);
	if(valueCount > 0)
	{
		// Count the ocurrences of each value
		GTEMPBUF(size_t, counts, valueCount);
		memset(counts, '\0', sizeof(size_t) * valueCount);
		for(size_t i = 0; i < count; i++)
		{
			const GVec& pPat = m_pData->row(pIndexes[i]);
			if((int)pPat[attr] >= 0)
			{
				GAssert((unsigned int)pPat[attr] < (unsigned int)valueCount); // out of range
				if((unsigned int)pPat[attr] < (unsigned int)valueCount)
					counts[(int)pPat[attr]]++;
			}
		}

		// Total up the entropy
		size_t max = 0;
		size_t maxcount = 0;
		double entropy = 0;
		double ratio;
		for(size_t i = 0; i < valueCount; i++)
		{
			if(counts[i] > maxcount || i == 0)
			{
				maxcount = counts[i];
				max = i;
			}
			if(counts[i] > 0)
			{
				ratio = (double)counts[i] / count;
				entropy -= ratio * log(ratio);
			}
		}
		const GVec& pScaleFactors = m_pMetric->scaleFactors();
		entropy *= (pScaleFactors[attr] * pScaleFactors[attr]);

		*pOutPivot = (double)max;
		*pOutGoodness = entropy;
	}
	else
	{
		// Compute the mean
		size_t missing = 0;
		double mean = 0;
		for(size_t i = 0; i < count; i++)
		{
			GAssert(pIndexes[i] < m_pData->rows());
			const GVec& pPat = m_pData->row(pIndexes[i]);
			if(pPat[attr] != UNKNOWN_REAL_VALUE)
				mean += pPat[attr];
			else
				missing++;
		}
		mean /= (count - missing);

		// Compute the scaled variance
		double var = 0;
		double d;
		const GVec& scaleFactors = m_pMetric->scaleFactors();
		for(size_t i = 0; i < count; i++)
		{
			const GVec& pPat = m_pData->row(pIndexes[i]);
			if(pPat[attr] != UNKNOWN_REAL_VALUE)
			{
				d = (pPat[attr] - mean) * scaleFactors[attr];
				var += (d * d);
			}
		}
		var /= (count - missing); // (the biased estimator of variance is better for this purpose)

		*pOutPivot = mean;
		*pOutGoodness = var;
	}
}

size_t GKdTree::splitIndexes(size_t count, size_t* pIndexes, size_t attr, double pivot)
{
	size_t t;
	size_t beg = 0;
	size_t end = count - 1;
	if(m_pMetric->relation()->valueCount(attr) == 0)
	{
		while(end >= beg && end < count)
		{
			const GVec& pPat = m_pData->row(pIndexes[beg]);
			if(pPat[attr] >= pivot)
			{
				t = pIndexes[beg];
				pIndexes[beg] = pIndexes[end];
				pIndexes[end] = t;
				end--;
			}
			else
				beg++;
		}
	}
	else
	{
		while(end >= beg && end < count)
		{
			const GVec& pPat = m_pData->row(pIndexes[beg]);
			if((int)pPat[attr] == (int)pivot)
			{
				t = pIndexes[beg];
				pIndexes[beg] = pIndexes[end];
				pIndexes[end] = t;
				end--;
			}
			else
				beg++;
		}
	}
	return beg;
}

// static
bool GKdTree::isGreaterOrEqual(const double* pPat, size_t attr, double pivot)
{
	if(m_pMetric->relation()->valueCount(attr) == 0)
		return (pPat[attr] >= pivot);
	else
		return ((int)pPat[attr] == (int)pivot);
}

GKdNode* GKdTree::buildTree(size_t count, size_t* pIndexes)
{
	size_t dims = m_pMetric->relation()->size();
	if(count <= (size_t)m_maxLeafSize)
		return new GKdLeafNode(count, pIndexes, dims, m_maxLeafSize);

	// Find a good place to split
	double pivot, goodness, p, g;
	size_t attr = 0;
	computePivotAndGoodness(count, pIndexes, 0, &pivot, &goodness);
	for(size_t i = 1; i < dims; i++)
	{
		computePivotAndGoodness(count, pIndexes, i, &p, &g);
		if(g > goodness)
		{
			pivot = p;
			goodness = g;
			attr = i;
		}
	}

	// Split the data
	size_t lessCount = splitIndexes(count, pIndexes, attr, pivot);
	size_t greaterOrEqualCount = count - lessCount;
	if(lessCount == 0 || greaterOrEqualCount == 0)
		return new GKdLeafNode(count, pIndexes, dims, m_maxLeafSize);

	// Make an interior node
	GKdNode* pLess = buildTree(lessCount, pIndexes);
	GKdNode* greaterOrEqual = buildTree(greaterOrEqualCount, pIndexes + lessCount);
	return new GKdInteriorNode(dims, pLess, greaterOrEqual, count, attr, pivot);
}

class KdTree_Compare_Nodes_Functor
{
public:
	bool operator() (GKdNode* pA, GKdNode* pB) const
	{
		double a = pA->GetMinDist();
		double b = pB->GetMinDist();
		return (a > b);
	}
};

void GKdTree::findNeighbors(size_t* pOutNeighbors, double* pOutSquaredDistances, const GVec& pInputVector, size_t nExclude)
{
	GClosestNeighborFindingHelper helper(m_neighborCount, pOutNeighbors, pOutSquaredDistances);
	KdTree_Compare_Nodes_Functor comparator;
	priority_queue< GKdNode*, vector<GKdNode*>, KdTree_Compare_Nodes_Functor > q(comparator);
	q.push(m_pRoot);
	while(q.size() > 0)
	{
		GKdNode* pNode = q.top();
		q.pop();
		if(pNode->GetMinDist() >= helper.GetWorstDist())
			break;
		if(pNode->IsLeaf())
		{
			double squaredDist;
			vector<size_t>* pIndexes = ((GKdLeafNode*)pNode)->GetIndexes();
			size_t count = pIndexes->size();
			for(size_t i = 0; i < count; i++)
			{
				size_t index = (*pIndexes)[i];
				if(index == nExclude)
					continue;
				const GVec& pCand = m_pData->row(index);
				squaredDist = m_pMetric->squaredDistance(pInputVector, pCand);
				helper.TryPoint(index, squaredDist);
			}
		}
		else
		{
			size_t attr;
			double pivot;
			GKdInteriorNode* pParent = (GKdInteriorNode*)pNode;
			pParent->GetDivision(&attr, &pivot);
			GKdNode* pLess = pParent->GetLess();
			pLess->CopyOffset(pParent);
			GKdNode* pGreaterOrEqual = pParent->GetGreaterOrEqual();
			pGreaterOrEqual->CopyOffset(pParent);
			if(isGreaterOrEqual(pInputVector.data(), attr, pivot))
				pLess->AdjustOffset(attr, pInputVector[attr] - pivot, m_pMetric->scaleFactors());
			else
				pGreaterOrEqual->AdjustOffset(attr, pivot - pInputVector[attr], m_pMetric->scaleFactors());
			q.push(pLess);
			q.push(pGreaterOrEqual);
		}
	}
}

// virtual
void GKdTree::neighbors(size_t* pOutNeighbors, size_t index)
{
	GTEMPBUF(double, distances, m_neighborCount);
	neighbors(pOutNeighbors, distances, index);
}

// virtual
void GKdTree::neighbors(size_t* pOutNeighbors, double* pOutDistances, size_t index)
{
	findNeighbors(pOutNeighbors, pOutDistances, m_pData->row(index), index);
}

// virtual
void GKdTree::neighbors(size_t* pOutNeighbors, double* pOutDistances, const GVec& pInputVector)
{
	findNeighbors(pOutNeighbors, pOutDistances, pInputVector, INVALID_INDEX);
}

// virtual
void GKdTree::reoptimize()
{
	if(!m_pRoot->IsLeaf())
	{
		GKdNode* pNewNode = ((GKdInteriorNode*)m_pRoot)->Rebuild(this);
		delete(m_pRoot);
		m_pRoot = pNewNode;
	}
}

// static
double GKdTree::medianDistanceToNeighbor(GMatrix& data, size_t n)
{
	if(n < 1)
		return 0.0; // 0 is the point itself

	// Fill a vector with the distances to the n^th neighbor of each point
	GKdTree kdtree(&data, n, NULL, false);
	vector<double> vals;
	vals.reserve(data.rows());
	GTEMPBUF(double, distances, n);
	GTEMPBUF(size_t, indexes, n);
	for(size_t i = 0; i < data.rows(); i++)
	{
		kdtree.neighbors(indexes, distances, i);
		GNeighborFinder::sortNeighbors(n, indexes, distances);
		if(indexes[n - 1] < data.rows())
			vals.push_back(sqrt(distances[n - 1]));
	}

	// Find the median value
	if(vals.size() < 1)
		throw Ex("at least one value is required to compute a median");
	if(vals.size() & 1)
	{
		vector<double>::iterator med = vals.begin() + (vals.size() / 2);
		std::nth_element(vals.begin(), med, vals.end());
		return *med;
	}
	else
	{
		vector<double>::iterator a = vals.begin() + (vals.size() / 2 - 1);
		std::nth_element(vals.begin(), a, vals.end());
		vector<double>::iterator b = std::min_element(a + 1, vals.end());
		return 0.5 * (*a + *b);
	}
}

#ifndef NO_TEST_CODE
#	include "GImage.h"
#	include "GHeap.h"

void MeasureBounds(GMatrix* pData, GKdNode* pNode, size_t attr, double* pMin, double* pMax)
{
	if(pNode->IsLeaf())
	{
		double min = 1e200;
		double max = -1e200;
		vector<size_t>* pIndexes = ((GKdLeafNode*)pNode)->GetIndexes();
		for(size_t i = 0; i < pIndexes->size(); i++)
		{
			GVec& pPat = pData->row((*pIndexes)[i]);
			min = std::min(pPat[attr], min);
			max = std::max(pPat[attr], max);
		}
		*pMin = min;
		*pMax = max;
	}
	else
	{
		double min1, min2, max1, max2;
		GKdNode* pChild = ((GKdInteriorNode*)pNode)->GetLess();
		MeasureBounds(pData, pChild, attr, &min1, &max1);
		pChild = ((GKdInteriorNode*)pNode)->GetGreaterOrEqual();
		MeasureBounds(pData, pChild, attr, &min2, &max2);
		*pMin = std::min(min1, min2);
		*pMax = std::max(max1, max2);
	}
}

void DrawKdNode(GPlotWindow* pw, GKdNode* pNode, GMatrix* pData)
{
	if(pNode->IsLeaf())
	{
		vector<size_t>* pIndexes = ((GKdLeafNode*)pNode)->GetIndexes();
		for(size_t i = 0; i < pIndexes->size(); i++)
		{
			GVec& pPat = pData->row((*pIndexes)[i]);
			pw->dot(pPat[0], pPat[1], 5, 0xff00ff00, 0xff000000);
			std::ostringstream os;
			os << (*pIndexes)[i];
			string tmp = os.str();
			pw->label(pPat[0], pPat[1], tmp.c_str(), 1.0f, 0xffffffff);
		}
	}
	else
	{
		size_t attr;
		double pivot, min, max;
		((GKdInteriorNode*)pNode)->GetDivision(&attr, &pivot);
		if(attr == 0)
		{
			MeasureBounds(pData, pNode, 1, &min, &max);
			pw->line(pivot, min, pivot, max, 0xffff0000);
		}
		else
		{
			GAssert(attr == 1); // unsupported value
			MeasureBounds(pData, pNode, 0, &min, &max);
			pw->line(min, pivot, max, pivot, 0xffff0000);
		}
		GKdNode* pChild = ((GKdInteriorNode*)pNode)->GetLess();
		DrawKdNode(pw, pChild, pData);
		pChild = ((GKdInteriorNode*)pNode)->GetGreaterOrEqual();
		DrawKdNode(pw, pChild, pData);
	}
}

class GDontGoFarMetric : public GDistanceMetric
{
public:
	double m_squaredMaxDist;

	GDontGoFarMetric(double maxDist)
	: GDistanceMetric(), m_squaredMaxDist(maxDist * maxDist)
	{
	}

	virtual ~GDontGoFarMetric()
	{
	}

	virtual const char* name() const { return "GDontGoFarMetric"; }

	virtual GDomNode* serialize(GDom* pDoc) const
	{
		throw Ex("not implemented");
		return NULL;
	}

	virtual void init(const GRelation* pRelation, bool own)
	{
		setRelation(pRelation, own);
	}

	virtual double squaredDistance(const GVec& pA, const GVec& pB) const
	{
		double squaredDist = pA.squaredDistance(pB);
		if(squaredDist > m_squaredMaxDist)
			throw Ex("a kd-tree shouldn't have to look this far away");
		return squaredDist;
	}
};

void GKdTree_testThatItDoesntLookFar()
{
	GRand prng(0);
	GMatrix tmp(100000, 2);
	for(size_t i = 0; i < tmp.rows(); i++)
	{
		GVec& pRow = tmp[i];
		pRow[0] = prng.uniform();
		pRow[1] = prng.uniform();
	}
	GDontGoFarMetric metric(0.05);
	GKdTree kdTree(&tmp, 5, &metric, false);
	GVec row(2);
	size_t neighs[5];
	double dists[5];
	for(size_t i = 0; i < 100; i++)
	{
		row[0] = prng.uniform();
		row[1] = prng.uniform();
		kdTree.neighbors(neighs, dists, row);
	}
}

#	define TEST_DIMS 4
#	define TEST_PATTERNS 1000
#	define TEST_NEIGHBORS 24
// static
void GKdTree::test()
{
	GClosestNeighborFindingHelper::test();
	GKdTree_testThatItDoesntLookFar();

	GMatrix data(new GUniformRelation(TEST_DIMS, 0));
	data.reserve(TEST_PATTERNS);
	GRand prng(0);
	for(size_t i = 0; i < TEST_PATTERNS; i++)
	{
		GVec& pPat = data.newRow();
		pPat.fillNormal(prng);
		pPat.normalize();
	}
	GBruteForceNeighborFinder bf(&data, TEST_NEIGHBORS, NULL, true);
	GKdTree kd(&data, TEST_NEIGHBORS, NULL, true);
/*
	GAssert(TEST_DIMS == 2); // You must change TEST_DIMS to 2 if you're going to plot the tree
	GImage image;
	image.SetSize(1000, 1000);
	image.Clear(0xff000000);
	GPlotWindow pw(&image, -1.1, -1.1, 1.1, 1.1);
	DrawKdNode(&pw, kd.GetRoot(), &data);
	image.SavePNGFile("kdtree.png");
*/
	size_t bfNeighbors[TEST_NEIGHBORS];
	size_t kdNeighbors[TEST_NEIGHBORS];
	double bfDistances[TEST_NEIGHBORS];
	double kdDistances[TEST_NEIGHBORS];
	for(size_t i = 0; i < TEST_PATTERNS; i++)
	{
		bf.neighbors(bfNeighbors, bfDistances, i);
		bf.sortNeighbors(bfNeighbors, bfDistances);
		kd.neighbors(kdNeighbors, kdDistances, i);
		kd.sortNeighbors(kdNeighbors, kdDistances);
		for(size_t j = 0; j < TEST_DIMS; j++)
		{
			if(bfNeighbors[j] != kdNeighbors[j])
				throw Ex("wrong answer!");
			if(kdNeighbors[j] != INVALID_INDEX && j > 0 && kdDistances[j] < kdDistances[j - 1])
				throw Ex("Neighbors out of order");
		}
	}
}
#endif // !NO_TEST_CODE

// --------------------------------------------------------------------------------------------------------









class GBallNode
{
public:
	GVec m_center;
	double m_radius;

	GBallNode(size_t count, size_t* pIndexes, const GMatrix* pData, GDistanceMetric* pMetric)
	: m_center(pData->cols())
	{
		m_radius = sqrt(pData->boundingSphere(m_center, pIndexes, count, pMetric));
	}

	virtual ~GBallNode()
	{
	}

	virtual bool isLeaf() = 0;

	double distance(GDistanceMetric* pMetric, const GVec& pVec)
	{
		return sqrt(pMetric->squaredDistance(m_center, pVec)) - m_radius;
	}

	/// Move the center and radius just enough to enclose both pVec and the previous ball
	void enclose(GDistanceMetric* pMetric, const GVec& pVec)
	{
		double d = distance(pMetric, pVec) + 1e-9;
		if(d > 0)
		{
			size_t dims = pMetric->relation()->size();
			double s = 0.5 * d / (m_radius + d);
			for(size_t i = 0; i < dims; i++)
				m_center[i] += s * (pVec[i] - m_center[i]);
			m_radius += 0.5 * d;
		}
	}

	virtual void insert(size_t index, GDistanceMetric* pMetric, const GVec& pVec) = 0;

	virtual bool drop(size_t index, GDistanceMetric* pMetric, const GVec& pVec) = 0;

	virtual void dropAll() = 0;

	virtual void print(std::ostream& stream, size_t depth, GDistanceMetric* pMetric, const GVec& pVec, const GMatrix* pData) = 0;
};

class GBallInterior : public GBallNode
{
public:
	GBallNode* m_pLeft;
	GBallNode* m_pRight;

	GBallInterior(size_t count, size_t* pIndexes, const GMatrix* pData, GDistanceMetric* pMetric)
	: GBallNode(count, pIndexes, pData, pMetric), m_pLeft(NULL), m_pRight(NULL)
	{
	}

	virtual ~GBallInterior()
	{
		delete(m_pLeft);
		delete(m_pRight);
	}

	virtual bool isLeaf() { return false; }

	virtual void insert(size_t index, GDistanceMetric* pMetric, const GVec& pVec)
	{
		enclose(pMetric, pVec);
		if(m_pLeft->distance(pMetric, pVec) < m_pRight->distance(pMetric, pVec))
			m_pLeft->insert(index, pMetric, pVec);
		else
			m_pRight->insert(index, pMetric, pVec);
	}

	virtual bool drop(size_t index, GDistanceMetric* pMetric, const GVec& pVec)
	{
		if(m_pLeft->distance(pMetric, pVec) <= 0.0)
		{
			if(m_pLeft->drop(index, pMetric, pVec))
				return true;
		}
		if(m_pRight->distance(pMetric, pVec) <= 0.0)
		{
			if(m_pRight->drop(index, pMetric, pVec))
				return true;
		}
		return false;
	}

	virtual void dropAll()
	{
		m_pLeft->dropAll();
		m_pRight->dropAll();
	}

	virtual void print(std::ostream& stream, size_t depth, GDistanceMetric* pMetric, const GVec& pVec, const GMatrix* pData)
	{
		m_pRight->print(stream, depth + 1, pMetric, pVec, pData);
		for(size_t i = 0; i < depth; i++)
			stream << "  ";
		stream << to_str(distance(pMetric, pVec));
		stream << "\n";
		m_pLeft->print(stream, depth + 1, pMetric, pVec, pData);
	}
};

class GBallLeaf : public GBallNode
{
public:
	std::vector<size_t> m_indexes;

	GBallLeaf(size_t count, size_t* pIndexes, const GMatrix* pData, GDistanceMetric* pMetric)
	: GBallNode(count, pIndexes, pData, pMetric)
	{
		m_indexes.reserve(count);
		for(size_t i = 0; i < count; i++)
			m_indexes.push_back(pIndexes[i]);
	}

	virtual ~GBallLeaf()
	{
	}

	virtual bool isLeaf() { return true; }

	virtual void insert(size_t index, GDistanceMetric* pMetric, const GVec& pVec)
	{
		enclose(pMetric, pVec);
		m_indexes.push_back(index);
	}

	virtual bool drop(size_t index, GDistanceMetric* pMetric, const GVec& pVec)
	{
		for(size_t i = 0; i < m_indexes.size(); i++)
		{
			if(m_indexes[i] == index)
			{
				std::swap(m_indexes[i], m_indexes[m_indexes.size() - 1]);
				m_indexes.pop_back();
				return true;
			}
		}
		return false;
	}

	virtual void dropAll()
	{
		m_indexes.clear();
	}

	virtual void print(std::ostream& stream, size_t depth, GDistanceMetric* pMetric, const GVec& pVec, const GMatrix* pData)
	{
		for(size_t i = 0; i < depth; i++)
			stream << "  ";
		stream << to_str(distance(pMetric, pVec)) << "\n";
		for(size_t j = 0; j < m_indexes.size(); j++)
		{
			for(size_t i = 0; i <= depth; i++)
				stream << "  ";
			size_t index = m_indexes[j];
			double val = pData->row(index)[0];
			stream << to_str(index) << " (" << to_str(val) << ")\n";
		}
	}
};


GBallTree::GBallTree(const GMatrix* pData, size_t neighbor_count, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinderGeneralizing(pData, neighbor_count, pMetric, ownMetric),
m_maxLeafSize(6),
m_pRoot(NULL)
{
	reoptimize();
}

// virtual
GBallTree::~GBallTree()
{
	delete(m_pRoot);
}

void GBallTree::reoptimize()
{
	if(m_pRoot && m_pRoot->isLeaf())
		return;
	delete(m_pRoot);
	GIndexVec indexes(m_pData->rows());
	GIndexVec::makeIndexVec(indexes.v, m_pData->rows());
	m_pRoot = buildTree(m_pData->rows(), indexes.v);
	m_size = m_pData->rows();
}

// virtual
void GBallTree::neighbors(size_t* pOutNeighbors, size_t index)
{
	GTEMPBUF(double, distances, m_neighborCount);
	neighbors(pOutNeighbors, distances, index);
}

// virtual
void GBallTree::neighbors(size_t* pOutNeighbors, double* pOutDistances, size_t index)
{
	findNeighbors(pOutNeighbors, pOutDistances, m_pData->row(index), index);
}

// virtual
void GBallTree::neighbors(size_t* pOutNeighbors, double* pOutDistances, const GVec& pInputVector)
{
	findNeighbors(pOutNeighbors, pOutDistances, pInputVector, INVALID_INDEX);
}

GBallNode* GBallTree::buildTree(size_t count, size_t* pIndexes)
{
	if(count > m_maxLeafSize)
	{
		// Find the two farthest points
		const GVec& pA = m_pData->row(pIndexes[0]);
		size_t b = 1;
		double sdist = m_pMetric->squaredDistance(pA, m_pData->row(pIndexes[b]));
		for(size_t i = 2; i < count; i++)
		{
			double cand = m_pMetric->squaredDistance(pA, m_pData->row(pIndexes[i]));
			if(cand > sdist)
			{
				sdist = cand;
				b = i;
			}
		}
		const GVec& pB = m_pData->row(pIndexes[b]);
		size_t c = 0;
		sdist = m_pMetric->squaredDistance(pB, m_pData->row(pIndexes[c]));
		for(size_t i = 1; i < count; i++)
		{
			if(i == b)
				continue;
			double cand = m_pMetric->squaredDistance(pB, m_pData->row(pIndexes[i]));
			if(cand > sdist)
			{
				sdist = cand;
				c = i;
			}
		}
		const GVec& pC = m_pData->row(pIndexes[c]);

		// Split based on closeness to b or c
		size_t leftCount = 0;
		for(size_t i = 0; i < count; i++)
		{
			double dB = m_pMetric->squaredDistance(m_pData->row(pIndexes[i]), pB);
			double dC = m_pMetric->squaredDistance(m_pData->row(pIndexes[i]), pC);
			if(dB < dC)
			{
				std::swap(pIndexes[leftCount], pIndexes[i]);
				leftCount++;
			}
		}
		if(leftCount == 0 || leftCount == count) // If we could not separate any of the points (which may occur if they are all the same point)...
			return new GBallLeaf(count, pIndexes, m_pData, m_pMetric);
		GBallInterior* pInterior = new GBallInterior(count, pIndexes, m_pData, m_pMetric);
		std::unique_ptr<GBallInterior> hInterior(pInterior);
		pInterior->m_pLeft = buildTree(leftCount, pIndexes);
		GAssert(pInterior->m_pLeft);
		pInterior->m_pRight = buildTree(count - leftCount, pIndexes + leftCount);
		GAssert(pInterior->m_pRight);
		return hInterior.release();
	}
	else
		return new GBallLeaf(count, pIndexes, m_pData, m_pMetric);
}


void GBallTree::findNeighbors(size_t* pOutNeighbors, double* pOutDistances, const GVec& pInputVector, size_t nExclude)
{
	GClosestNeighborFindingHelper helper(m_neighborCount, pOutNeighbors, pOutDistances);
	GSimplePriorityQueue<GBallNode*> q;
	q.insert(m_pRoot, m_pRoot->distance(m_pMetric, pInputVector));
	while(q.size() > 0)
	{
		double dist = q.peekValue();
		if(helper.GetWorstDist() < dist * dist)
			break;
		GBallNode* pBall = q.peekObject();
		q.pop();
		if(pBall->isLeaf())
		{
			GBallLeaf* pLeaf = (GBallLeaf*)pBall;
			for(size_t i = 0; i < pLeaf->m_indexes.size(); i++)
			{
				size_t index = pLeaf->m_indexes[i];
				if(index != nExclude)
					helper.TryPoint(pLeaf->m_indexes[i], m_pMetric->squaredDistance(m_pData->row(index), pInputVector));
			}
		}
		else
		{
			GBallInterior* pInt = (GBallInterior*)pBall;
			q.insert(pInt->m_pLeft, pInt->m_pLeft->distance(m_pMetric, pInputVector));
			q.insert(pInt->m_pRight, pInt->m_pRight->distance(m_pMetric, pInputVector));
		}
	}
}

void GBallTree::insert(size_t index)
{
	m_pRoot->insert(index, m_pMetric, m_pData->row(index));
	m_size++;
}

void GBallTree::drop(size_t index)
{
	if(!m_pRoot->drop(index, m_pMetric, m_pData->row(index)))
		throw Ex("Could not find the specified index in this structure. (This could happen if the corresponding point in the dataset has changed.)");
	m_size--;
}

void GBallTree::dropAll()
{
	m_pRoot->dropAll();
	m_size = 0;
}

#ifndef NO_TEST_CODE
#define TEST_BALLTREE_ITERS 100
#define TEST_BALLTREE_ROWS 200
#define TEST_BALLTREE_DIMS 7
#define TEST_BALLTREE_NEIGHBORS 17
// static
void GBallTree::test()
{
	double kdDist[TEST_BALLTREE_NEIGHBORS];
	size_t kdInd[TEST_BALLTREE_NEIGHBORS];
	double ballDist[TEST_BALLTREE_NEIGHBORS];
	size_t ballInd[TEST_BALLTREE_NEIGHBORS];
	GRand r(0);
	for(size_t i = 0; i < TEST_BALLTREE_ITERS; i++)
	{
		GMatrix m(TEST_BALLTREE_ROWS, TEST_BALLTREE_DIMS);
		for(size_t j = 0; j < TEST_BALLTREE_ROWS; j++)
			m[j].fillUniform(r);
		GKdTree kd(&m, TEST_BALLTREE_NEIGHBORS);
		GBallTree ball(&m, TEST_BALLTREE_NEIGHBORS);
		kd.neighbors(kdInd, kdDist, (size_t)0);
		kd.sortNeighbors(kdInd, kdDist);
		ball.neighbors(ballInd, ballDist, (size_t)0);
		ball.sortNeighbors(ballInd, ballDist);
		for(size_t j = 0; j < TEST_BALLTREE_NEIGHBORS; j++)
		{
			if(kdInd[j] != ballInd[j])
			{
				/*
				std::cout << "\nIter: " << to_str(i) << "\n";
				std::cout << "Matrix:\n";
				m.print(std::cout);
				std::cout << "kdTree:\n";
				GIndexVec::print(std::cout, kdInd, TEST_BALLTREE_NEIGHBORS);
				std::cout << "\nballTree:\n";
				GIndexVec::print(std::cout, ballInd, TEST_BALLTREE_NEIGHBORS);
				std::cout << "\nkdTree distances:\n";
				GVec::print(std::cout, 8, kdDist, TEST_BALLTREE_NEIGHBORS);
				std::cout << "\nballTree distances:\n";
				GVec::print(std::cout, 8, ballDist, TEST_BALLTREE_NEIGHBORS);
				std::cout << "\nTree:\n";
				ball.m_pRoot->print(std::cout, 0, ball.m_pMetric, m[0], &m);
				*/
				throw Ex("indexes differ");
			}
			if(kdDist[j] != ballDist[j])
				throw Ex("distances differ");
		}
	}
}
#endif












// --------------------------------------------------------------------------------------------------------






















class GShortcutPrunerAtomicCycleDetector : public GAtomicCycleFinder
{
protected:
	GShortcutPruner* m_pThis;
	size_t m_thresh;

public:
	GShortcutPrunerAtomicCycleDetector(size_t nodes, GShortcutPruner* pThis, size_t thresh) : GAtomicCycleFinder(nodes), m_pThis(pThis), m_thresh(thresh)
	{
	}

	virtual ~GShortcutPrunerAtomicCycleDetector()
	{
	}

	virtual bool onDetectAtomicCycle(vector<size_t>& cycle)
	{
		if(cycle.size() >= (size_t)m_thresh)
		{
			m_pThis->onDetectBigAtomicCycle(cycle);
			return false;
		}
		else
			return true;
	}
};

GShortcutPruner::GShortcutPruner(size_t* pNeighborhoods, size_t n, size_t k)
: m_pNeighborhoods(pNeighborhoods), m_n(n), m_k(k), m_cycleThresh(10), m_subGraphRange(6), m_cuts(0)
{
}

GShortcutPruner::~GShortcutPruner()
{
}

bool GShortcutPruner::isEveryNodeReachable()
{
	GBitTable visited(m_n);
	deque<size_t> q;
	visited.set(0);
	q.push_back(0);
	while(q.size() > 0)
	{
		size_t cur = q.front();
		q.pop_front();
		for(size_t j = 0; j < m_k; j++)
		{
			size_t neigh = m_pNeighborhoods[m_k * cur + j];
			if(neigh < m_n && !visited.bit(neigh))
			{
				visited.set(neigh);
				q.push_back(neigh);
			}
		}
	}
	for(size_t i = 0; i < m_n; i++)
	{
		if(!visited.bit(i))
			return false;
	}
	return true;
}

size_t GShortcutPruner::prune()
{
	while(true)
	{
		bool everyNodeReachable = isEveryNodeReachable();
		GShortcutPrunerAtomicCycleDetector g(m_n, this, m_cycleThresh);
		size_t* pHood = m_pNeighborhoods;
		for(size_t i = 0; i < m_n; i++)
		{
			for(size_t j = 0; j < m_k; j++)
			{
				if(pHood[j] < m_n)
					g.addEdgeIfNotDupe(i, pHood[j]);
			}
			pHood += m_k;
		}
		size_t oldCuts = m_cuts;
		g.compute();
		if(everyNodeReachable && !isEveryNodeReachable())
			throw Ex("Cutting shortcuts should not segment the graph");
		if(m_cuts == oldCuts)
			break;
	}
	return m_cuts;
}

void GShortcutPruner::onDetectBigAtomicCycle(vector<size_t>& cycle)
{
	// Make a subgraph containing only nodes close to the cycle
	size_t* mapIn = new size_t[m_n];
	std::unique_ptr<size_t[]> hMapIn(mapIn);
	vector<size_t> mapOut;
	GBitTable visited(m_n);
	deque<size_t> q;
	for(vector<size_t>::iterator it = cycle.begin(); it != cycle.end(); it++)
	{
		q.push_back(*it);
		q.push_back(1);
	}
	while(q.size() > 0)
	{
		size_t cur = q.front();
		q.pop_front();
		size_t depth = q.front();
		q.pop_front();
		mapIn[cur] = mapOut.size();
		mapOut.push_back(cur);
		if(depth <= (size_t)m_subGraphRange)
		{
			for(size_t j = 0; j < m_k; j++)
			{
				size_t neigh = m_pNeighborhoods[cur * m_k + j];
				if(neigh < m_n && !visited.bit(neigh))
				{
					visited.set(neigh);
					q.push_back(neigh);
					q.push_back(depth + 1);
				}
			}
		}
	}

	// Compute betweenness of all edges
	GBrandesBetweennessCentrality g(mapOut.size());
	for(size_t i = 0; i < mapOut.size(); i++)
	{
		size_t* pHood = m_pNeighborhoods + mapOut[i] * m_k;
		for(size_t j = 0; j < m_k; j++)
		{
			size_t neigh = pHood[j];
			if(neigh < m_n && visited.bit(neigh))
			{
				g.addDirectedEdgeIfNotDupe(i, mapIn[neigh]);
				g.addDirectedEdgeIfNotDupe(mapIn[neigh], i);
			}
		}
	}
	g.compute();

	// Find the edge on the cycle with the largest betweenness
	size_t shortcutFrom = 0;
	size_t shortcutTo = 0;
	double shortcutBetweenness = 0;
	for(size_t i = 0; i < cycle.size(); i++)
	{
		size_t from = cycle[i];
		size_t to = cycle[(i + 1) % cycle.size()];
		size_t forwIndex = g.neighborIndex(mapIn[from], mapIn[to]);
		size_t revIndex = g.neighborIndex(mapIn[to], mapIn[from]);
		double d = g.edgeBetweennessByNeighbor(mapIn[from], forwIndex) + g.edgeBetweennessByNeighbor(mapIn[to], revIndex);
		if(i == 0 || d > shortcutBetweenness)
		{
			shortcutBetweenness = d;
			shortcutFrom = from;
			shortcutTo = to;
		}
	}

	// Cut the shortcut
	bool cutForward = false;
	for(size_t j = 0; j < m_k; j++)
	{
		if(m_pNeighborhoods[shortcutFrom * m_k + j] == shortcutTo)
		{
			m_pNeighborhoods[shortcutFrom * m_k + j] = INVALID_INDEX;
			cutForward = true;
			m_cuts++;
			break;
		}
	}
	bool cutReverse = false;
	for(size_t j = 0; j < m_k; j++)
	{
		if(m_pNeighborhoods[shortcutTo * m_k + j] == shortcutFrom)
		{
			m_pNeighborhoods[shortcutTo * m_k + j] = INVALID_INDEX;
			cutReverse = true;
			m_cuts++;
			break;
		}
	}
	if(!cutForward && !cutReverse)
		throw Ex("Failed to find the offending edge");
}

#ifndef NO_TEST_CODE
// static
void GShortcutPruner::test()
{
	// Make a fully-connected grid
	size_t w = 6;
	size_t h = 6;
	size_t n = w * h;
	size_t k = 4;
	size_t* pNeighbors = new size_t[n * k];
	std::unique_ptr<size_t[]> hNeighbors(pNeighbors);
	size_t i = 0;
	size_t* pHood = pNeighbors;
	for(size_t y = 0; y < h; y++)
	{
		for(size_t x = 0; x < w; x++)
		{
			size_t j = 0;
			pHood[j++] = (x > 0 ? i - 1 : INVALID_INDEX);
			pHood[j++] = (x < w - 1 ? i + 1 : INVALID_INDEX);
			pHood[j++] = (y > 0 ? i - w : INVALID_INDEX);
			pHood[j++] = (y < h - 1 ? i + w : INVALID_INDEX);
			pHood += k;
			i++;
		}
	}

	// Add 3 shortcuts
	pNeighbors[(0 * w + 0) * k + 0] = n - 1; // connect (0,0) to (w-1, h-1)
	pNeighbors[(0 * w + (w - 1)) * k + 1] = n - 1; // connect (w-1,0) to (w-1,h-1)
	pNeighbors[((h - 1) * w + (w - 1)) * k + 0] = w - 1; // connect (w-1,h-1) to (w-1,0)

	// Cut the shortcuts
	GShortcutPruner pruner(pNeighbors, n, k);
	pruner.setCycleThreshold(h);
	pruner.setSubGraphRange(3);
	size_t cuts = pruner.prune();
	if(pNeighbors[(0 * w + 0) * k + 0] != INVALID_INDEX)
		throw Ex("missed a shortcut");
	if(pNeighbors[(0 * w + (w - 1)) * k + 1] != INVALID_INDEX)
		throw Ex("missed a shortcut");
	if(pNeighbors[((h - 1) * w + (w - 1)) * k + 0] != INVALID_INDEX)
		throw Ex("missed a shortcut");
	if(cuts != 3)
		throw Ex("wrong number of cuts");
}
#endif // NO_TEST_CODE











class GCycleCutAtomicCycleDetector : public GAtomicCycleFinder
{
protected:
	GCycleCut* m_pThis;
	size_t m_thresh;
	bool m_restore;
	bool m_gotOne;

public:
	GCycleCutAtomicCycleDetector(size_t nodes, GCycleCut* pThis, size_t thresh, bool restore) : GAtomicCycleFinder(nodes), m_pThis(pThis), m_thresh(thresh), m_restore(restore), m_gotOne(false)
	{
	}

	virtual ~GCycleCutAtomicCycleDetector()
	{
	}

	bool gotOne() { return m_gotOne; }

	virtual bool onDetectAtomicCycle(vector<size_t>& cycle)
	{
		if(cycle.size() >= (size_t)m_thresh)
		{
			if(m_restore)
				m_gotOne = true;
			else
				m_pThis->onDetectBigAtomicCycle(cycle);
			return false;
		}
		else
			return true;
	}
};

GCycleCut::GCycleCut(size_t* pNeighborhoods, const GMatrix* pPoints, size_t k)
: m_pNeighborhoods(pNeighborhoods), m_pPoints(pPoints), m_k(k), m_cycleThresh(10), m_cutCount(0)
{
	// Compute the mean neighbor distance
	size_t* pNeigh = m_pNeighborhoods;
	size_t count = 0;
	double sum = 0;
	for(size_t i = 0; i < m_pPoints->rows(); i++)
	{
		for(size_t j = 0; j < k; j++)
		{
			if(*pNeigh < m_pPoints->rows())
			{
				sum += sqrt(m_pPoints->row(i).squaredDistance(m_pPoints->row(*pNeigh)));
				count++;
			}
			pNeigh++;
		}
	}
	m_aveDist = sum / count;

	// Compute the capacities
	pNeigh = m_pNeighborhoods;
	for(size_t i = 0; i < m_pPoints->rows(); i++)
	{
		for(size_t j = 0; j < k; j++)
		{
			if(*pNeigh < m_pPoints->rows())
			{
				double cap = 1.0 / (m_aveDist + sqrt(m_pPoints->row(i).squaredDistance(m_pPoints->row(*pNeigh))));
				m_capacities[make_pair(i, *pNeigh)] = cap;
				m_capacities[make_pair(*pNeigh, i)] = cap;
/*
				m_capacities[make_pair(i, *pNeigh)] = 1.0;
				m_capacities[make_pair(*pNeigh, i)] = 1.0;
*/
			}
			pNeigh++;
		}
	}
}

GCycleCut::~GCycleCut()
{
}

bool GCycleCut::doAnyBigAtomicCyclesExist()
{
	// Make the graph
	GCycleCutAtomicCycleDetector g(m_pPoints->rows(), this, m_cycleThresh, true);
	size_t* pHood = m_pNeighborhoods;
	for(size_t i = 0; i < m_pPoints->rows(); i++)
	{
		for(size_t j = 0; j < m_k; j++)
		{
			if(pHood[j] < m_pPoints->rows())
				g.addEdgeIfNotDupe(i, pHood[j]);
		}
		pHood += m_k;
	}

	// Find a large atomic cycle (calls onDetectBigAtomicCycle when found)
	g.compute();
	return g.gotOne();
}

size_t GCycleCut::cut()
{
	m_cuts.clear();

	// Cut the graph
	while(true)
	{
		// Make the graph
		GCycleCutAtomicCycleDetector g(m_pPoints->rows(), this, m_cycleThresh, false);
		size_t* pHood = m_pNeighborhoods;
		for(size_t i = 0; i < m_pPoints->rows(); i++)
		{
			for(size_t j = 0; j < m_k; j++)
			{
				if(pHood[j] < m_pPoints->rows())
					g.addEdgeIfNotDupe(i, pHood[j]);
			}
			pHood += m_k;
		}

		// Find a large atomic cycle (calls onDetectBigAtomicCycle when found)
		size_t oldCuts = m_cutCount;
		g.compute();
		if(m_cutCount == oldCuts)
			break;
	}

	// Restore superfluous cuts
	for(vector<size_t>::iterator it = m_cuts.begin(); it != m_cuts.end(); )
	{
		size_t point = *it;
		it++;
		GAssert(it != m_cuts.end()); // broken cuts list
		size_t neigh = *it;
		it++;
		GAssert(it != m_cuts.end()); // broken cuts list
		size_t other = *it;
		it++;

		// Restore the edge if it doesn't create a big atomic cycle
		m_pNeighborhoods[point * m_k + neigh] = other;
		if(!doAnyBigAtomicCyclesExist())
			m_cutCount--;
		else
			m_pNeighborhoods[point * m_k + neigh] = INVALID_INDEX;
	}
//cerr << "cuts: " << m_cutCount << "\n";
	return m_cutCount;
}

void GCycleCut::onDetectBigAtomicCycle(vector<size_t>& cycle)
{
	// Find the bottleneck
	double bottleneck = 1e308;
	for(size_t i = 0; i < cycle.size(); i++)
	{
		size_t from = cycle[i];
		size_t to = cycle[(i + 1) % cycle.size()];
		pair<size_t, size_t> p = make_pair(from, to);
		double d = m_capacities[p];
		if(i == 0 || d < bottleneck)
			bottleneck = d;
	}
	GAssert(bottleneck > 0); // all capacities should be greater than zero

	// Reduce every edge in the cycle by the bottleneck's capacity
	for(size_t i = 0; i < cycle.size(); i++)
	{
		size_t from = cycle[i];
		size_t to = cycle[(i + 1) % cycle.size()];
		pair<size_t, size_t> p1 = make_pair(from, to);
		pair<size_t, size_t> p2 = make_pair(to, from);
		double d = m_capacities[p1];
		if(d - bottleneck > 1e-12)
		{
			// Reduce the capacity
			m_capacities[p1] = d - bottleneck;
			m_capacities[p2] = d - bottleneck;
		}
		else
		{
			// Remove the edge
			m_capacities.erase(p1);
			m_capacities.erase(p2);
			size_t forw = INVALID_INDEX;
			size_t* pHood = m_pNeighborhoods + from * m_k;
			for(size_t j = 0; j < m_k; j++)
			{
				if(pHood[j] == to)
				{
					forw = j;
					break;
				}
			}
			size_t rev = INVALID_INDEX;
			pHood = m_pNeighborhoods + to * m_k;
			for(size_t j = 0; j < m_k; j++)
			{
				if(pHood[j] == from)
				{
					rev = j;
					break;
				}
			}
			GAssert(rev != INVALID_INDEX || forw != INVALID_INDEX); // couldn't find the edge
			if(forw != INVALID_INDEX)
			{
				m_pNeighborhoods[from * m_k + forw] = INVALID_INDEX;
				m_cuts.push_back(from);
				m_cuts.push_back(forw);
				m_cuts.push_back(to);
				m_cutCount++;
			}
			if(rev != INVALID_INDEX)
			{
				m_pNeighborhoods[to * m_k + rev] = INVALID_INDEX;
				m_cuts.push_back(to);
				m_cuts.push_back(rev);
				m_cuts.push_back(from);
				m_cutCount++;
			}
		}
	}
}

#ifndef NO_TEST_CODE
// static
void GCycleCut::test()
{
	// Make a fully-connected grid
	size_t w = 6;
	size_t h = 6;
	size_t n = w * h;
	size_t k = 4;
	size_t* pNeighbors = new size_t[n * k];
	std::unique_ptr<size_t[]> hNeighbors(pNeighbors);
	size_t i = 0;
	size_t* pHood = pNeighbors;
	for(size_t y = 0; y < h; y++)
	{
		for(size_t x = 0; x < w; x++)
		{
			size_t j = 0;
			pHood[j++] = (x > 0 ? i - 1 : INVALID_INDEX);
			pHood[j++] = (x < w - 1 ? i + 1 : INVALID_INDEX);
			pHood[j++] = (y > 0 ? i - w : INVALID_INDEX);
			pHood[j++] = (y < h - 1 ? i + w : INVALID_INDEX);
			pHood += k;
			i++;
		}
	}

	// Add 3 shortcuts
	pNeighbors[(0 * w + 0) * k + 0] = n - 1; // connect (0,0) to (w-1, h-1)
	pNeighbors[(0 * w + (w - 1)) * k + 1] = n - 1; // connect (w-1,0) to (w-1,h-1)
	pNeighbors[((h - 1) * w + (w - 1)) * k + 0] = w - 1; // connect (w-1,h-1) to (w-1,0)

	// Make some random data
	GMatrix data(0, 5);
	GRand prng(0);
	for(size_t ii = 0; ii < n; ii++)
	{
		GVec& pRow = data.newRow();
		pRow.fillNormal(prng);
		pRow.normalize();
	}

	// Cut the shortcuts
	GCycleCut pruner(pNeighbors, &data, k);
	pruner.setCycleThreshold(h);
	size_t cuts = pruner.cut();
	if(pNeighbors[(0 * w + 0) * k + 0] != INVALID_INDEX)
		throw Ex("missed a shortcut");
	if(pNeighbors[(0 * w + (w - 1)) * k + 1] != INVALID_INDEX)
		throw Ex("missed a shortcut");
	if(pNeighbors[((h - 1) * w + (w - 1)) * k + 0] != INVALID_INDEX)
		throw Ex("missed a shortcut");
	if(cuts != 3)
		throw Ex("wrong number of cuts");
}
#endif // NO_TEST_CODE

GTemporalNeighborFinder::GTemporalNeighborFinder(GMatrix* pObservations, GMatrix* pActions, bool ownActionsData, size_t neighbor_count, GRand* pRand, size_t maxDims)
: GNeighborFinder(preprocessObservations(pObservations, maxDims), neighbor_count),
m_pActions(pActions),
m_ownActionsData(ownActionsData),
m_pRand(pRand)
{
	if(m_pData->rows() != pActions->rows())
		throw Ex("Expected the same number of observations as control vectors");
	if(pActions->cols() != 1)
		throw Ex("Sorry, only one action dim is currently supported");
	int actionValues = (int)m_pActions->relation().valueCount(0);
	if(actionValues < 2)
		throw Ex("Sorry, only nominal actions are currently supported");

	// Train the consequence maps
	size_t obsDims = m_pData->cols();
	for(int j = 0; j < actionValues; j++)
	{
		GMatrix before(0, obsDims);
		GMatrix delta(0, obsDims);
		for(size_t i = 0; i < m_pData->rows() - 1; i++)
		{
			if((int)pActions->row(i)[0] == (int)j)
			{
				GVec::copy(before.newRow().data(), m_pData->row(i).data(), obsDims);
				GVec& pDelta = delta.newRow();
				GVec::copy(pDelta.data(), m_pData->row(i + 1).data(), obsDims);
				GVec::subtract(pDelta.data(), m_pData->row(i).data(), obsDims);
			}
		}
		GAssert(before.rows() > 20); // not much data
		GSupervisedLearner* pMap = new GFeatureFilter(new GKNN(), new GPCA(12));
		m_consequenceMaps.push_back(pMap);
		pMap->train(before, delta);
	}
}

// virtual
GTemporalNeighborFinder::~GTemporalNeighborFinder()
{
	if(m_ownActionsData)
		delete(m_pActions);
	for(vector<GSupervisedLearner*>::iterator it = m_consequenceMaps.begin(); it != m_consequenceMaps.end(); it++)
		delete(*it);
	delete(m_pPreprocessed);
}

GMatrix* GTemporalNeighborFinder::preprocessObservations(GMatrix* pObs, size_t maxDims)
{
	if(pObs->cols() > maxDims)
	{
		GPCA pca(maxDims);
		pca.train(*pObs);
		m_pPreprocessed = pca.transformBatch(*pObs);
		return m_pPreprocessed;
	}
	else
	{
		m_pPreprocessed = NULL;
		return pObs;
	}
}

bool GTemporalNeighborFinder::findPath(size_t from, size_t to, double* path, double maxDist)
{
	// Find the path
	int actionValues = (int)m_pActions->relation().valueCount(0);
	const GVec& pStart = m_pData->row(from);
	const GVec& pGoal = m_pData->row(to);
	size_t dims = m_pData->cols();
	double origSquaredDist = pStart.squaredDistance(pGoal);
	GVec pObs(dims);
	GVec pDelta(dims);
	GVec pRemaining(dims);
	pObs.copy(pStart);
	GBitTable usedActions(actionValues);
	GVec::setAll(path, 0.0, actionValues);
	while(true)
	{
		pRemaining.copy(pGoal);
		pRemaining -= pObs;
		if(pRemaining.squaredMagnitude() < 1e-9)
			break; // We have arrived at the destination
		double biggestCorr = 1e-6;
		int bestAction = -1;
		double stepSize = 0.0;
		int lastPredicted = -1;
		for(int i = 0; i < actionValues; i++)
		{
			if(usedActions.bit(i))
				continue;
			m_consequenceMaps[i]->predict(pObs, pDelta);
			lastPredicted = i;
			double d = pDelta.correlation(pRemaining);
			if(d <= 0)
				usedActions.set(i);
			else if(d > biggestCorr)
			{
				biggestCorr = d;
				bestAction = i;
				stepSize = std::min(1.0, pDelta.dotProduct(pRemaining) / pDelta.squaredMagnitude());
			}
		}
		if(bestAction < 0)
			break; // There are no good actions, so we're done
		if(stepSize < 1.0)
			usedActions.set(bestAction); // let's not do microscopic zig-zagging

		// Advance the current observation
		if(bestAction != lastPredicted)
			m_consequenceMaps[bestAction]->predict(pObs, pDelta);
		pObs.addScaled(stepSize, pDelta);
		path[bestAction] += stepSize;
		if(GVec::squaredMagnitude(path, actionValues) > maxDist * maxDist)
			return false;
	}
	if(pRemaining.squaredMagnitude() >= 0.2 * 0.2 * origSquaredDist)
		return false; // Too imprecise. Throw this one out.
	return true;
}

// virtual
void GTemporalNeighborFinder::neighbors(size_t* pOutNeighbors, size_t index)
{
	GTEMPBUF(double, dissims, m_neighborCount);
	neighbors(pOutNeighbors, dissims, index);
}

// virtual
void GTemporalNeighborFinder::neighbors(size_t* pOutNeighbors, double* pOutDistances, size_t index)
{
	int valueCount = (int)m_pActions->relation().valueCount(0);
	if(m_pActions->cols() > 1 || valueCount == 0)
		throw Ex("continuous and multi-dim actions not supported yet");
	size_t actionValues = m_pActions->relation().valueCount(0);
	size_t pos = 0;
	GTEMPBUF(double, path, actionValues);
	for(size_t i = 0; pos < m_neighborCount && i < m_pData->rows(); i++)
	{
		if(index == i)
			continue;
		if(!findPath(index, i, path, 2.0 //distCap
			))
		{
			if(index + 1 == i)
			{
				pOutNeighbors[pos] = i;
				pOutDistances[pos] = 1.0;
			}
			else if(index == i + 1)
			{
				pOutNeighbors[pos] = i;
				pOutDistances[pos] = 1.0;
			}
			else
				continue;
		}
		pOutNeighbors[pos] = i;
		pOutDistances[pos] = GVec::squaredMagnitude(path, actionValues);
		//GAssert(ABS(pOutDistances[pos]) < 0.001 || ABS(pOutDistances[pos] - 1.0) < 0.001 || ABS(pOutDistances[pos] - 1.4142) < 0.001 || ABS(pOutDistances[pos] - 2.0) < 0.001); // Noisy result. Does the transition function have noise? If so, then this is expected, so comment me out.
		pos++;
	}

	// Fill the remaining slots with nothing
	while(pos < m_neighborCount)
	{
		pOutNeighbors[pos] = INVALID_INDEX;
		pOutDistances[pos] = 0.0;
		pos++;
	}
}








GSequenceNeighborFinder::GSequenceNeighborFinder(GMatrix* pData, int neighbor_count)
: GNeighborFinder(pData, neighbor_count)
{
}

// virtual
GSequenceNeighborFinder::~GSequenceNeighborFinder()
{
}

// virtual
void GSequenceNeighborFinder::neighbors(size_t* pOutNeighbors, size_t index)
{
	return neighbors(pOutNeighbors, NULL, index);
}

// virtual
void GSequenceNeighborFinder::neighbors(size_t* pOutNeighbors, double* pOutDistances, size_t index)
{
	size_t prevPos = -1;
	size_t pos = 0;
	size_t i = 1;
	while(true)
	{
		if(pos == prevPos)
		{
			while(pos < m_neighborCount)
				pOutNeighbors[pos++] = INVALID_INDEX;
			break;
		}
		prevPos = pos;
		if(index - i < m_pData->rows())
		{
			pOutNeighbors[pos] = index - i;
			if(++pos >= m_neighborCount)
				break;
		}
		if(index + i < m_pData->rows())
		{
			pOutNeighbors[pos] = index + i;
			if(++pos >= m_neighborCount)
				break;
		}
		i++;
	}
	if(pOutDistances)
	{
		for(size_t ii = 0; ii < m_neighborCount; ii++)
			pOutDistances[ii] = (double)((ii + 2) / 2);
	}
}



} // namespace GClasses


