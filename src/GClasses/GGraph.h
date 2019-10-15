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

#ifndef __GGRAPHCUT_H__
#define __GGRAPHCUT_H__

#include "GOptimizer.h"
#include <stdlib.h>
#include <deque>

namespace GClasses {

struct GGraphCutNode;
struct GGraphCutEdge;
class GHeap;
class GRegionAjacencyGraph;
class GGraphEdgeIterator;
class GRand;


/// This implements an optimized max-flow/min-cut algorithm described in
/// "An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision"
/// by Boykov, Y. and Kolmogorov, V.
/// This implementation assumes that edges are undirected
class GGraphCut
{
friend class GGraphEdgeIterator;
protected:
	std::deque<size_t> m_q;
	GHeap* m_pHeap;
	size_t m_nNodes;
	struct GGraphCutNode* m_pNodes;
	struct GGraphCutNode* m_pSource;
	struct GGraphCutNode* m_pSink;

public:
	GGraphCut(size_t nNodes);
	~GGraphCut();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Returns the number of nodes in the graph
	size_t nodeCount() { return m_nNodes; }

	/// Adds an edge to the graph. You must add all the edges
	/// before calling "Cut". The edge will be stored internally
	/// as a directed edge (from nNode1 to nNode2), but the
	/// Cut method will treat them as undirected edges.
	void addEdge(size_t nNode1, size_t nNode2, float fCapacity);

	/// Creates an edge from the node that represents each region
	/// to the node for each of its neighbor regions.
	void getEdgesFromRegionList(GRegionAjacencyGraph* pRegionList);

	/// This computes the cut. nSourceNode is the node that
	/// represents the source, and nSinkNode is the node that
	/// represents the sink.
	void cut(size_t nSourceNode, size_t nSinkNode);

	/// Determine whether the specified node is on the source-side
	/// or the sink-side of the cut. (You must call "Cut" before
	/// calling this method.)
	bool isSource(size_t nNode);

	/// Returns true if the specified node borders the cut
	bool doesBorderTheCut(size_t nNode);

protected:
	void growNode(size_t nNode);
	void augmentPath(struct GGraphCutEdge* pEdge);
	void recycleTree(size_t nChild, size_t nParent);
	void findAHome(size_t nNode);
};


/// Iterates over the edges that connect to the
/// specified node
class GGraphEdgeIterator
{
protected:
	GGraphCut* m_pGraph;
	size_t m_nNode;
	struct GGraphCutEdge* m_pCurrentEdge;

public:
	GGraphEdgeIterator(GGraphCut* pGraph, size_t nNode);
	~GGraphEdgeIterator();

	/// Starts over with a new node
	void reset(size_t nNode);

	/// Gets the next edge. Returns false if there isn't
	/// one left to get. If it returns true, pNode, pEdgeWeight, and pOutgoing
	/// will contain the values of the edge. (Note that pOutgoing tells you
	/// which direction the edge is going, even though my implementation of
	/// graph-cut treats them as undirected edges.)
	bool next(size_t* pNode, float* pEdgeWeight, bool* pOutgoing);
};


/// Computes the shortest-cost path between all pairs of
/// vertices in a graph. Takes O(n^3) time.
class GFloydWarshall
{
protected:
	size_t m_nodes;
	GMatrix* m_pCosts;
	size_t* m_pPaths;

public:
	/// nodes specifies the number of nodes in the graph.
	GFloydWarshall(size_t nodes);
	~GFloydWarshall();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Returns the number of nodes in the graph
	size_t nodeCount() { return m_nodes; }

	/// Adds a directed edge to the graph. (You must call this to add all
	/// the edges before calling compute.)
	void addDirectedEdge(size_t from, size_t to, double cost);

	/// Computes the shortest-cost path between every pair of points
	/// (You must add all the edges before you call compute.)
	void compute();

	/// Returns the smallest cost to get from node "from" to node "to".
	/// (You must call compute before calling this)
	double cost(size_t from, size_t to);

	/// Returns the next node on the shortest path from node "from"
	/// to get to node "goal"
	/// (You must call compute before calling this)
	size_t next(size_t from, size_t goal);

	/// Returns a pointer to the cost matrix
	GMatrix* costMatrix() { return m_pCosts; }

	/// Returns the cost matrix. You are responsible to delete it.
	/// (This class should not be used after this method is called.)
	GMatrix* releaseCostMatrix();

	/// Scans the entire cost matrix, and returns false if any node
	/// is unreachable from any other node. (Assumes compute has already
	/// been called.)
	bool isConnected();
};


/// Finds the shortest path from an origin vertex to all other vertices.
/// Implemented with a binary-heap priority-queue. If the graph is sparse
/// on edges, it will run in about O(n log(n)) time. If the graph is dense,
/// it runs in about O(n^2 log(n))
class GDijkstra
{
protected:
	size_t m_nodes;
	double* m_pCosts;
	size_t* m_pPrevious;
	std::vector<size_t>* m_pNeighbors;
	std::vector<double>* m_pEdgeCosts;

public:
	GDijkstra(size_t nodes);
	~GDijkstra();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Returns the number of nodes in the graph
	size_t nodeCount() { return m_nodes; }

	/// Adds a directed edge to the graph. (You must call this to add all
	/// the edges before calling compute.)
	void addDirectedEdge(size_t from, size_t to, double cost);

	/// Finds the shortest-cost path from the specified origin to every
	/// other point in the graph
	void compute(size_t origin);

	/// Returns the total cost to travel from the origin to the specified target node
	double cost(size_t target);

	/// Returns the previous node on the shortest path from the origin
	/// to the specified vertex
	size_t previous(size_t vertex);
};


/// Computes the number of times that the shortest-path between
/// every pair of points passes over each edge and vertex
class GBrandesBetweennessCentrality
{
protected:
	size_t m_nodeCount;
	std::vector<size_t>* m_pNeighbors;
	double* m_pVertexBetweenness;
	std::vector<double>* m_pEdgeBetweenness;

public:
	GBrandesBetweennessCentrality(size_t nodes);
	~GBrandesBetweennessCentrality();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Returns the number of nodes in the graph
	size_t nodeCount();

	/// Adds a directed edge to the graph. (You must call this to add all
	/// the edges before calling compute.)
	void addDirectedEdge(size_t from, size_t to);

	/// Adds a directed edge if the specified edge does not already
	/// exist. (This method is inefficient if there are a lot of edges.
	/// If you want efficiency, keep track of the edges yourself.)
	void addDirectedEdgeIfNotDupe(size_t from, size_t to);

	/// Computes the betweenness for all nodes. (You must add all the edges
	/// before calling this.) This takes O(VE) time, where V is the number of
	/// vertices, and E is the number of edges.
	void compute();

	/// Returns the betweenness of the specified vertex. (You must call compute
	/// before calling this.) Note that for undirected graphs, you should divide
	/// this value by 2.
	double vertexBetweenness(size_t vertex);

	/// Returns the betweenness of the edge specified by two vertices.
	/// "compute" must be called before this method is called.
	/// Note that for undirected graphs, you should divide the value this returns
	/// by 2, since every edge will be counted in both directions.
	/// Throws an exception if there is no edge between vertex1 and vertex2.
	/// Note that this method is not as efficient as "edgeBetweennessByNeighbor".
	double edgeBetweennessByVertex(size_t vertex1, size_t vertex2);

	/// Returns the betweenness of the edge specified by vertex and neighborIndex.
	/// Note that neighborIndex is not the same as the other vertex. It is the
	/// enumeration value of each neighbor of the first vertex. To obtain the value
	/// for neighborIndex, you should call "neighborIndex" by passing in both vertices.
	/// "compute" must be called before this method is called.
	/// Note that for undirected graphs, you should divide the value this returns
	/// by 2, since every edge will be counted in both directions.
	double edgeBetweennessByNeighbor(size_t vertex, size_t neighborIndex);

	/// Returns the index of the specified neighbor "to" (by iterating over all the
	/// neighbors of "from" until it finds "to"). Returns INVALID_INDEX if not found.
	size_t neighborIndex(size_t from, size_t to);
};


/// This finds all of the atomic cycles (cycles that cannot be divided
/// into two smaller cycles) in a graph
class GAtomicCycleFinder
{
protected:
	size_t m_nodeCount;
	std::vector<size_t>* m_pNeighbors;
	std::vector<size_t>* m_pMirrors;

public:
	GAtomicCycleFinder(size_t nodes);
	virtual ~GAtomicCycleFinder();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Returns the number of nodes in the graph
	size_t nodeCount();

	/// Adds an undirected edge to the graph. (You must call this to add all
	/// the edges before calling compute.)
	void addEdge(size_t a, size_t b);

	/// Adds an undirected edge to the graph if a duplicate edge does not
	/// already exist. (This method is inefficient if there are a lot of edges.
	/// If you want efficiency, keep track of the edges yourself.)
	void addEdgeIfNotDupe(size_t a, size_t b);

	/// Finds all the atomic cycles in the graph, and calls onDetectAtomicCycle
	/// for each one
	void compute();

	/// You must overload this method to receive the cycles as they are detected
	/// (The edges are every torroidally adjacent pair of vertices in cycle.)
	/// If true is returned, it will continue to find more atomic cycles.
	/// If false is returned, it will stop immediately.
	virtual bool onDetectAtomicCycle(std::vector<size_t>& cycle) = 0;
};

} // namespace GClasses

#endif // __GGRAPHCUT_H__
