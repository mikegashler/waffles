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
#include "GSparseMatrix.h"


//using std::cerr;
using std::vector;
using std::priority_queue;
using std::set;
using std::deque;
using std::make_pair;
using std::pair;
using std::string;
using std::map;
using std::multimap;

namespace GClasses {


GNeighborGraph::GNeighborGraph(GNeighborFinder* pNF, bool own, size_t neighbors)
: GNeighborFinder(pNF->data()), m_pNF(pNF), m_own(own)
{
	m_neighs.resize(m_pData->rows());
	m_dists.resize(m_pData->rows());
	fillCacheNearest(neighbors);
}

GNeighborGraph::GNeighborGraph(double squaredRadius, GNeighborFinder* pNF, bool own)
: GNeighborFinder(pNF->data()), m_pNF(pNF), m_own(own)
{
	m_neighs.resize(m_pData->rows());
	m_dists.resize(m_pData->rows());
	fillCacheRadius(squaredRadius);
}

GNeighborGraph::GNeighborGraph(bool own, GNeighborFinderGeneralizing* pNF)
: GNeighborFinder(pNF->data()), m_pNF(pNF), m_own(own)
{
	m_neighs.resize(m_pData->rows());
	m_dists.resize(m_pData->rows());
	GDistanceMetric* pMetric = pNF->metric();
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		double prev = 0.0;
		if(i > 0)
			prev = pMetric->squaredDistance(pNF->data()->row(i), pNF->data()->row(i - 1));
		double next = 0.0;
		if(i + 1 < m_pData->rows())
			next = pMetric->squaredDistance(pNF->data()->row(i), pNF->data()->row(i + 1));
		double maxSqRad = std::max(prev, next);
		size_t neigh_count = m_pNF->findWithinRadius(maxSqRad, i);
		m_neighs[i].clear();
		m_dists[i].clear();
		for(size_t j = 0; j < neigh_count; j++)
		{
			m_neighs[i].push_back(m_pNF->neighbor(j));
			m_dists[i].push_back(m_pNF->distance(j));
		}
	}
}

// virtual
GNeighborGraph::~GNeighborGraph()
{
	if(m_own)
		delete(m_pNF);
}

void GNeighborGraph::fillCacheNearest(size_t k)
{
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		size_t neigh_count = m_pNF->findNearest(k, i);
		m_neighs[i].clear();
		m_dists[i].clear();
		for(size_t j = 0; j < neigh_count; j++)
		{
			m_neighs[i].push_back(m_pNF->neighbor(j));
			m_dists[i].push_back(m_pNF->distance(j));
		}
	}
}

void GNeighborGraph::fillCacheRadius(double squaredRadius)
{
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		size_t neigh_count = m_pNF->findWithinRadius(squaredRadius, i);
		m_neighs[i].clear();
		m_dists[i].clear();
		for(size_t j = 0; j < neigh_count; j++)
		{
			m_neighs[i].push_back(m_pNF->neighbor(j));
			m_dists[i].push_back(m_pNF->distance(j));
		}
	}
}

void GNeighborGraph::recomputeDistances(GDistanceMetric* pMetric)
{
	pMetric->init(&m_pData->relation(), false);
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		vector<size_t>& neighs = m_neighs[i];
		vector<double>& dists = m_dists[i];
		for(size_t j = 0; j < neighs.size(); j++)
		{
			dists[j] = pMetric->squaredDistance(m_pData->row(i), m_pData->row(neighs[j]));
		}
	}
}

void GNeighborGraph::swapInData(const GMatrix* pNewData)
{
	m_pData = pNewData;
}

size_t GNeighborGraph::cutShortcuts(size_t cycleLen)
{
	GCycleCut cc(this, m_pData, m_neighs[0].size());
	cc.setCycleThreshold(cycleLen);
	return cc.cut();
}

bool GNeighborGraph::isConnected()
{
	// Make a table containing bi-directional neighbor connections
	vector< vector<size_t> > bidirTable;
	bidirTable.resize(m_pData->rows());
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		size_t nc = findNearest(0, i);
		vector<size_t>& row = bidirTable[i];
		for(size_t j = 0; j < nc; j++)
		{
			row.push_back(neighbor(j));
			bidirTable[neighbor(j)].push_back(i);
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

void GNeighborGraph::set(size_t point, size_t neighbor_number, size_t neigh)
{
	m_neighs[point][neighbor_number] = neigh;
}

void GNeighborGraph::dropInvalidNeighbors()
{
	for(size_t i = 0; i < m_neighs.size(); i++)
	{
		for(size_t j = m_neighs[i].size() - 1; j < m_neighs[i].size(); j--)
		{
			if(m_neighs[i][j] == INVALID_INDEX)
			{
				m_neighs[i].erase(m_neighs[i].begin() + j);
				m_dists[i].erase(m_dists[i].begin() + j);
			}
		}
	}
}

// --------------------------------------------------------------------

/// This helper class keeps neighbors sorted as a binary heap, such that the most dissimilar
/// of the k-current-neighbors is always at the front of the heap.
class GClosestNeighborFindingHelper
{
protected:
	size_t m_neighbors;
	std::vector<size_t>& m_neighs;
	std::vector<double>& m_dists;

public:
	GClosestNeighborFindingHelper(size_t neighbors, std::vector<size_t>& neighs, std::vector<double>& dists)
	: m_neighbors(neighbors), m_neighs(neighs), m_dists(dists)
	{
		GAssert(m_neighbors >= 1);
		neighs.clear();
		dists.clear();
	}

	~GClosestNeighborFindingHelper()
	{
	}

	// Adds a point to the set of current neighbors if it is closer than the
	// most dissimilar of the k-current-neighbors
	void TryPoint(size_t index, double distance)
	{
		size_t heapPos;
		if(m_neighs.size() < m_neighbors)
		{
			m_neighs.push_back(index);
			m_dists.push_back(distance);
			heapPos = m_neighs.size();
		}
		else
		{
			// Compare with the front of the heap, which holds the most dissimilar of the k-current-neighbors
			if(distance >= m_dists[0])
				return;

			// Release the most dissimilar of the k-current neighbors
			heapPos = 1;
			while(2 * heapPos <= m_neighbors)
			{
				if(2 * heapPos == m_neighbors || m_dists[2 * heapPos - 1] > m_dists[2 * heapPos])
				{
					m_dists[heapPos - 1] = m_dists[2 * heapPos - 1];
					m_neighs[heapPos - 1] = m_neighs[2 * heapPos - 1];
					heapPos = 2 * heapPos;
				}
				else
				{
					m_dists[heapPos - 1] = m_dists[2 * heapPos];
					m_neighs[heapPos - 1] = m_neighs[2 * heapPos];
					heapPos = 2 * heapPos + 1;
				}
			}
			m_dists[heapPos - 1] = distance;
			m_neighs[heapPos - 1] = index;
		}

		// Insert into heap
		while(heapPos > 1 && m_dists[heapPos / 2 - 1] < m_dists[heapPos - 1])
		{
			std::swap(m_dists[heapPos / 2 - 1], m_dists[heapPos - 1]);
			std::swap(m_neighs[heapPos / 2 - 1], m_neighs[heapPos - 1]);
			heapPos /= 2;
		}
	}

	double GetWorstDist()
	{
		return m_dists.size() >= m_neighbors ? m_dists[0] : 1e308;
	}

#	define TEST_NEIGHBOR_COUNT 33
	static void test()
	{
		vector<size_t> neighbors;
		vector<double> distances;
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
};

// --------------------------------------------------------------------------------

GNeighborFinderGeneralizing::GNeighborFinderGeneralizing(const GMatrix* pData, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinder(pData), m_pMetric(pMetric), m_ownMetric(ownMetric)
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

void GNeighborFinderGeneralizing::insertionSortNeighbors(size_t start, size_t end)
{
	for(size_t i = start + 1; i < end; i++)
	{
		for(size_t j = i; j > start; j--)
		{
			if(distance(j - 1) < distance(j))
				break;
			std::swap(m_neighs[j - 1], m_neighs[j]);
			std::swap(m_dists[j - 1], m_dists[j]);
		}
	}
}

void GNeighborFinderGeneralizing::sortNeighbors(size_t beg, size_t end)
{
	end = std::min(end, m_neighs.size());
	if(beg + 6 >= end)
	{
		insertionSortNeighbors(beg, end);
		return;
	}
	size_t initial_beg = beg;
	size_t initial_end = end;

	// Pick a pivot (using the median of 3 technique)
	double pivA = distance(0);
	double pivB = distance((beg + end) / 2);
	double pivC = distance(end - 1);
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
		while(beg + 1 < end && distance(beg) < pivot)
			beg++;
		while(end > beg + 1 && distance(end - 1) > pivot)
			end--;
		if(beg + 1 >= end)
			break;
		std::swap(m_dists[beg], m_dists[end - 1]);
		std::swap(m_neighs[beg], m_neighs[end - 1]);
		beg++;
		end--;
	}

	// Recurse
	if(distance(beg) < pivot)
		beg++;
	if(beg == initial_beg || end == initial_end)
	{
		insertionSortNeighbors(initial_beg, initial_end);
		return;
	}
	sortNeighbors(initial_beg, beg);
	sortNeighbors(beg, initial_end);
}






// --------------------------------------------------------------------------------

GBruteForceNeighborFinder::GBruteForceNeighborFinder(GMatrix* pData, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinderGeneralizing(pData, pMetric, ownMetric)
{
}

GBruteForceNeighborFinder::~GBruteForceNeighborFinder()
{
}

// virtual
void GBruteForceNeighborFinder::reoptimize()
{
}

size_t GBruteForceNeighborFinder::findNearest(size_t k, const GVec& vec, size_t exclude)
{
	GClosestNeighborFindingHelper helper(k, m_neighs, m_dists);
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		if(i == exclude)
			continue;
		helper.TryPoint(i, m_pMetric->squaredDistance(vec, m_pData->row(i)));
	}
	return m_neighs.size();
}

size_t GBruteForceNeighborFinder::findWithinRadius(double squaredRadius, const GVec& vec, size_t exclude)
{
	m_neighs.clear();
	m_dists.clear();
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		if(i == exclude)
			continue;
		double d = m_pMetric->squaredDistance(vec, m_pData->row(i));
		if(d <= squaredRadius)
		{
			m_neighs.push_back(i);
			m_dists.push_back(d);
		}
	}
	return m_neighs.size();
}

// virtual
size_t GBruteForceNeighborFinder::findNearest(size_t k, const GVec& vec)
{
	return findNearest(k, vec, INVALID_INDEX);
}

// virtual
size_t GBruteForceNeighborFinder::findNearest(size_t k, size_t index)
{
	return findNearest(k, m_pData->row(index), index);
}

// virtual
size_t GBruteForceNeighborFinder::findWithinRadius(double squaredRadius, const GVec& vec)
{
	return findWithinRadius(squaredRadius, vec, INVALID_INDEX);
}

// virtual
size_t GBruteForceNeighborFinder::findWithinRadius(double squaredRadius, size_t index)
{
	return findWithinRadius(squaredRadius, m_pData->row(index), index);
}

// --------------------------------------------------------------------------------

GSparseNeighborFinder::GSparseNeighborFinder(GSparseMatrix* pData, GMatrix* pBogusData, GSparseSimilarity* pMetric, bool ownMetric)
: GNeighborFinderGeneralizing(pBogusData, new GRowDistance(), true),
m_pData(pData),
m_pSparseMetric(pMetric),
m_ownSparseMetric(ownMetric)
{
}

GSparseNeighborFinder::~GSparseNeighborFinder()
{
	if(m_ownSparseMetric)
		delete(m_pSparseMetric);
}

// virtual
void GSparseNeighborFinder::reoptimize()
{
}

// virtual
size_t GSparseNeighborFinder::findNearest(size_t k, const GVec& vec)
{
	m_neighs.clear();
	m_dists.clear();
	multimap<double,size_t> priority_queue;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		map<size_t,double>& row = m_pData->row(i);
		double similarity = m_pSparseMetric->similarity(row, vec);
		priority_queue.insert(pair<double,size_t>(similarity, i));
		if(priority_queue.size() > k)
			priority_queue.erase(priority_queue.begin());
	}
	for(multimap<double,size_t>::iterator it = priority_queue.begin(); it != priority_queue.end(); it++)
	{
		m_neighs.push_back(it->second);
		m_dists.push_back(1.0 / (it->first + 1e-6));
	}
	return m_neighs.size();
}

// virtual
size_t GSparseNeighborFinder::findNearest(size_t k, size_t index)
{
	map<size_t,double>& vec = m_pData->row(index);
	m_neighs.clear();
	m_dists.clear();
	multimap<double,size_t> priority_queue;
	for(size_t i = 0; i < m_pData->rows(); i++)
	{
		if(i == index)
			continue;
		map<size_t,double>& row = m_pData->row(i);
		double similarity = m_pSparseMetric->similarity(row, vec);
		priority_queue.insert(pair<double,size_t>(similarity, i));
		if(priority_queue.size() > k)
			priority_queue.erase(priority_queue.begin());
	}
	for(multimap<double,size_t>::iterator it = priority_queue.begin(); it != priority_queue.end(); it++)
	{
		m_neighs.push_back(it->second);
		m_dists.push_back(1.0 / (it->first + 1e-6));
	}
	return m_neighs.size();
}

// virtual
size_t GSparseNeighborFinder::findWithinRadius(double squaredRadius, const GVec& vec)
{
	throw new Ex("Sorry, not implemented yet");
}

// virtual
size_t GSparseNeighborFinder::findWithinRadius(double squaredRadius, size_t index)
{
	throw new Ex("Sorry, not implemented yet");
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
		GVecWrapper vw(m_pOffset, dims);
		vw.fill(0.0);
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
		memcpy(m_pOffset, pParent->m_pOffset, sizeof(double) * m_dims);
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

GKdTree::GKdTree(const GMatrix* pData, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinderGeneralizing(pData, pMetric, ownMetric)
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

size_t GKdTree::findNearest(size_t k, const GVec& vec, size_t nExclude)
{
	GClosestNeighborFindingHelper helper(k, m_neighs, m_dists);
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
				squaredDist = m_pMetric->squaredDistance(vec, pCand);
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
			if(isGreaterOrEqual(vec.data(), attr, pivot))
				pLess->AdjustOffset(attr, vec[attr] - pivot, m_pMetric->scaleFactors());
			else
				pGreaterOrEqual->AdjustOffset(attr, pivot - vec[attr], m_pMetric->scaleFactors());
			q.push(pLess);
			q.push(pGreaterOrEqual);
		}
	}
	return m_neighs.size();
}

size_t GKdTree::findWithinRadius(double squaredRadius, const GVec& vec, size_t nExclude)
{
	m_neighs.clear();
	m_dists.clear();
	KdTree_Compare_Nodes_Functor comparator;
	priority_queue< GKdNode*, vector<GKdNode*>, KdTree_Compare_Nodes_Functor > q(comparator);
	q.push(m_pRoot);
	while(q.size() > 0)
	{
		GKdNode* pNode = q.top();
		q.pop();
		if(pNode->GetMinDist() > squaredRadius)
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
				squaredDist = m_pMetric->squaredDistance(vec, pCand);
				if(squaredDist <= squaredRadius)
				{
					m_neighs.push_back(index);
					m_dists.push_back(squaredDist);
				}
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
			if(isGreaterOrEqual(vec.data(), attr, pivot))
				pLess->AdjustOffset(attr, vec[attr] - pivot, m_pMetric->scaleFactors());
			else
				pGreaterOrEqual->AdjustOffset(attr, pivot - vec[attr], m_pMetric->scaleFactors());
			q.push(pLess);
			q.push(pGreaterOrEqual);
		}
	}
	return m_neighs.size();
}

// virtual
size_t GKdTree::findNearest(size_t k, size_t index)
{
	return findNearest(k, m_pData->row(index), index);
}

// virtual
size_t GKdTree::findNearest(size_t k, const GVec& vec)
{
	return findNearest(k, vec, INVALID_INDEX);
}

// virtual
size_t GKdTree::findWithinRadius(double squaredRadius, size_t index)
{
	return findWithinRadius(squaredRadius, m_pData->row(index), index);
}

// virtual
size_t GKdTree::findWithinRadius(double squaredRadius, const GVec& vector)
{
	return findWithinRadius(squaredRadius, vector, INVALID_INDEX);
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
	GKdTree kdtree(&data, NULL, false);
	vector<double> vals;
	vals.reserve(data.rows());
	for(size_t i = 0; i < data.rows(); i++)
	{
		size_t nc = kdtree.findNearest(n, i);
		kdtree.sortNeighbors();
		vals.push_back(sqrt(kdtree.distance(nc - 1)));
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


//#	include "GImage.h"

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
	GKdTree kdTree(&tmp, &metric, false);
	GVec row(2);
	for(size_t i = 0; i < 100; i++)
	{
		row[0] = prng.uniform();
		row[1] = prng.uniform();
		kdTree.findNearest(5, row);
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

	// Test neighbor sorting
	{
		GRand rand(0);
		GMatrix data(256, 1);
		for(size_t i = 0; i < data.rows(); i++)
			data[i][0] = rand.uniform();
		data.sort(0);
		GRandomIndexIterator ii(data.rows(), rand);
		GKdTree kd(&data);
		size_t index;
		kd.m_neighs.clear();
		kd.m_dists.clear();
		ii.reset();
		while(ii.next(index))
		{
			kd.m_neighs.push_back(index);
			kd.m_dists.push_back(data[index][0]);
		}
		kd.sortNeighbors();
		for(size_t i = 0; i < data.rows(); i++)
		{
			if(kd.m_neighs[i] != i || kd.m_dists[i] != data[i][0])
				throw Ex("wrong answer");
		}
	}


	GMatrix data(new GUniformRelation(TEST_DIMS, 0));
	data.reserve(TEST_PATTERNS);
	GRand prng(0);
	for(size_t i = 0; i < TEST_PATTERNS; i++)
	{
		GVec& pPat = data.newRow();
		pPat.fillNormal(prng);
		pPat.normalize();
	}
	GBruteForceNeighborFinder bf(&data);
	GKdTree kd(&data);
/*
	GAssert(TEST_DIMS == 2); // You must change TEST_DIMS to 2 if you're going to plot the tree
	GImage image;
	image.SetSize(1000, 1000);
	image.Clear(0xff000000);
	GPlotWindow pw(&image, -1.1, -1.1, 1.1, 1.1);
	DrawKdNode(&pw, kd.GetRoot(), &data);
	image.SavePNGFile("kdtree.png");
*/
	for(size_t i = 0; i < TEST_PATTERNS; i++)
	{
		size_t ncbf = bf.findNearest(TEST_NEIGHBORS, i);
		bf.sortNeighbors();
		size_t nckd = kd.findNearest(TEST_NEIGHBORS, i);
		kd.sortNeighbors();
		if(ncbf != nckd)
			throw Ex("found different number of neighbors");
		if(ncbf != TEST_NEIGHBORS)
			throw Ex("found unexpected number of neighbors");
		for(size_t j = 0; j < TEST_DIMS; j++)
		{
			if(bf.neighbor(j) != kd.neighbor(j))
				throw Ex("wrong answer!");
			if(j > 0 && kd.distance(j) < kd.distance(j - 1))
				throw Ex("Neighbors out of order");
		}
	}
}

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


GBallTree::GBallTree(const GMatrix* pData, GDistanceMetric* pMetric, bool ownMetric)
: GNeighborFinderGeneralizing(pData, pMetric, ownMetric),
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
	GIndexVec::makeIndexVec(indexes.m_data, m_pData->rows());
	m_pRoot = buildTree(m_pData->rows(), indexes.m_data);
	m_size = m_pData->rows();
}

// virtual
size_t GBallTree::findNearest(size_t k, size_t index)
{
	return findNearest(k, m_pData->row(index), index);
}

// virtual
size_t GBallTree::findNearest(size_t k, const GVec& vec)
{
	return findNearest(k, vec, INVALID_INDEX);
}

// virtual
size_t GBallTree::findWithinRadius(double squaredRadius, size_t index)
{
	return findWithinRadius(squaredRadius, m_pData->row(index), index);
}

// virtual
size_t GBallTree::findWithinRadius(double squaredRadius, const GVec& vec)
{
	return findWithinRadius(squaredRadius, vec, INVALID_INDEX);
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


size_t GBallTree::findNearest(size_t k, const GVec& vec, size_t nExclude)
{
	GClosestNeighborFindingHelper helper(k, m_neighs, m_dists);
	GSimplePriorityQueue<GBallNode*> q;
	q.insert(m_pRoot, m_pRoot->distance(m_pMetric, vec));
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
					helper.TryPoint(pLeaf->m_indexes[i], m_pMetric->squaredDistance(m_pData->row(index), vec));
			}
		}
		else
		{
			GBallInterior* pInt = (GBallInterior*)pBall;
			q.insert(pInt->m_pLeft, pInt->m_pLeft->distance(m_pMetric, vec));
			q.insert(pInt->m_pRight, pInt->m_pRight->distance(m_pMetric, vec));
		}
	}
	return m_neighs.size();
}

size_t GBallTree::findWithinRadius(double squaredRadius, const GVec& vec, size_t nExclude)
{
	m_neighs.clear();
	m_dists.clear();
	GSimplePriorityQueue<GBallNode*> q;
	q.insert(m_pRoot, m_pRoot->distance(m_pMetric, vec));
	while(q.size() > 0)
	{
		double dist = q.peekValue();
		if(dist * dist > squaredRadius)
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
				{
					double d = m_pMetric->squaredDistance(m_pData->row(index), vec);
					if(d <= squaredRadius)
					{
						m_neighs.push_back(pLeaf->m_indexes[i]);
						m_dists.push_back(d);
					}
				}
			}
		}
		else
		{
			GBallInterior* pInt = (GBallInterior*)pBall;
			q.insert(pInt->m_pLeft, pInt->m_pLeft->distance(m_pMetric, vec));
			q.insert(pInt->m_pRight, pInt->m_pRight->distance(m_pMetric, vec));
		}
	}
	return m_neighs.size();
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

#define TEST_BALLTREE_ITERS 100
#define TEST_BALLTREE_ROWS 200
#define TEST_BALLTREE_DIMS 7
#define TEST_BALLTREE_NEIGHBORS 17
// static
void GBallTree::test()
{
	GRand r(0);
	for(size_t i = 0; i < TEST_BALLTREE_ITERS; i++)
	{
		GMatrix m(TEST_BALLTREE_ROWS, TEST_BALLTREE_DIMS);
		for(size_t j = 0; j < TEST_BALLTREE_ROWS; j++)
			m[j].fillUniform(r);
		GKdTree kd(&m);
		GBallTree ball(&m);
		size_t nckd = kd.findNearest(TEST_BALLTREE_NEIGHBORS, 0);
		kd.sortNeighbors();
		size_t ncbt = ball.findNearest(TEST_BALLTREE_NEIGHBORS, 0);
		ball.sortNeighbors();
		if(nckd != ncbt)
			throw Ex("found different number of neighbors");
		for(size_t j = 0; j < TEST_BALLTREE_NEIGHBORS; j++)
		{
			if(kd.neighbor(j) != ball.neighbor(j))
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
			if(kd.distance(j) != ball.distance(j))
				throw Ex("distances differ");
		}
	}
}













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

GCycleCut::GCycleCut(GNeighborGraph* pNeighborGraph, const GMatrix* pPoints, size_t k)
: m_pNeighborGraph(pNeighborGraph), m_pPoints(pPoints), m_k(k), m_cycleThresh(10), m_cutCount(0)
{
	// Compute the mean neighbor distance
	size_t count = 0;
	double sum = 0;
	for(size_t i = 0; i < m_pPoints->rows(); i++)
	{
		size_t nc = pNeighborGraph->findNearest(m_k, i);
		for(size_t j = 0; j < nc; j++)
			sum += sqrt(pNeighborGraph->distance(j));
		count += nc;
	}
	m_aveDist = sum / count;

	// Compute the capacities
	for(size_t i = 0; i < m_pPoints->rows(); i++)
	{
		size_t nc = pNeighborGraph->findNearest(m_k, i);
		for(size_t j = 0; j < nc; j++)
		{
			double cap = 1.0 / (m_aveDist + sqrt(pNeighborGraph->distance(j)));
			m_capacities[make_pair(i, pNeighborGraph->neighbor(j))] = cap;
			m_capacities[make_pair(pNeighborGraph->neighbor(j), i)] = cap;
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
	for(size_t i = 0; i < m_pPoints->rows(); i++)
	{
		size_t nc = m_pNeighborGraph->findNearest(m_k, i);
		for(size_t j = 0; j < nc; j++)
			g.addEdgeIfNotDupe(i, m_pNeighborGraph->neighbor(j));
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
		for(size_t i = 0; i < m_pPoints->rows(); i++)
		{
			size_t nc = m_pNeighborGraph->findNearest(m_k, i);
			for(size_t j = 0; j < nc; j++)
				g.addEdgeIfNotDupe(i, m_pNeighborGraph->neighbor(j));
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
		m_pNeighborGraph->set(point, neigh, other);
		if(!doAnyBigAtomicCyclesExist())
			m_cutCount--;
		else
			m_pNeighborGraph->set(point, neigh, INVALID_INDEX);
	}
//cerr << "cuts: " << m_cutCount << "\n";

	m_pNeighborGraph->dropInvalidNeighbors();
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
			size_t nc = m_pNeighborGraph->findNearest(m_k, from);
			for(size_t j = 0; j < nc; j++)
			{
				if(m_pNeighborGraph->neighbor(j) == to)
				{
					forw = j;
					break;
				}
			}
			size_t rev = INVALID_INDEX;
			nc = m_pNeighborGraph->findNearest(m_k, to);
			for(size_t j = 0; j < nc; j++)
			{
				if(m_pNeighborGraph->neighbor(j) == from)
				{
					rev = j;
					break;
				}
			}
			GAssert(rev != INVALID_INDEX || forw != INVALID_INDEX); // couldn't find the edge
			if(forw != INVALID_INDEX)
			{
				m_pNeighborGraph->set(from, forw, INVALID_INDEX);
				m_cuts.push_back(from);
				m_cuts.push_back(forw);
				m_cuts.push_back(to);
				m_cutCount++;
			}
			if(rev != INVALID_INDEX)
			{
				m_pNeighborGraph->set(to, rev, INVALID_INDEX);
				m_cuts.push_back(to);
				m_cuts.push_back(rev);
				m_cuts.push_back(from);
				m_cutCount++;
			}
		}
	}
}

// static
void GCycleCut::test()
{
	// todo: This test was removed because I didn't want to port it when I made some API changes.
	// I believe GCycleCut still works, but I really should write a new test to exercise it.
}




} // namespace GClasses


