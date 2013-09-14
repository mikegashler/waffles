/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GPGM_H__
#define __GPGM_H__

#include <vector>
#include <cstddef>
#include "GHeap.h"
#include "GRand.h"

namespace GClasses {

class GRand;
class GPGMCategorical;
class GPGMVariable;


/// The base class of all nodes in a probabilistic graphical model.
class GPGMNode
{
protected:
	bool m_observed;
	double m_observedValue;

public:
	GPGMNode();
	virtual ~GPGMNode();

	/// Returns the current value of this node. If this node represents a
	/// random variable, then it returns the value most-recently sampled
	/// from its distribution.
	virtual double currentValue() = 0;

	/// This method is called when this node is added as the parent to pChild.
	/// It gives this node a chance to link back to the child if needed.
	virtual void onNewChild(GPGMVariable* pChild) = 0;
};


/// A node in a probabilistic graphical model that represents a constant value.
class GPGMConstant : public GPGMNode
{
public:
	/// General-purpose constructor
	GPGMConstant(double val) : GPGMNode() { m_observed = true; m_observedValue = val; }
	virtual ~GPGMConstant() {}

	/// Returns the constant value
	virtual double currentValue() { return m_observedValue; }

	/// This method is a no-op.
	virtual void onNewChild(GPGMVariable* pChild) {}
};


/// The base class of nodes in a probabilistic graphical model that represent variable values.
class GPGMVariable : public GPGMNode
{
protected:
	std::vector<GPGMCategorical*> m_catParents;
	std::vector<GPGMVariable*> m_children;

public:
	GPGMVariable();
	virtual ~GPGMVariable();

	/// Returns all nodes that are known to depend on this one.
	const std::vector<GPGMVariable*>& children() { return m_children; }

	/// Adds a categorical node as a parent of this node. Calling this method will cause
	/// This node to resize its table of distribution parameters, so a default value is
	/// required to fill in new elements. Typically, you will set these new elements to
	/// something more meaningful after you call this method.
	virtual void addCatParent(GPGMCategorical* pNode, GPGMNode* pDefaultVal) = 0;

	/// Compute the likelihood of drawing the value "x" from the distribution represented
	/// in this node given the current values in all of the parent nodes. If "x" is not
	/// a supported value, then this should return 0.0;
	virtual double likelihood(double x) = 0;

	/// Return the most recent value sampled from this node.
	virtual double currentValue() = 0;

	/// Draw a new value for this node given the current values of all other nodes
	/// it its Markov blanket.
	virtual void sample(GRand* pRand) = 0;

	/// Set this node to an observed value. After calling this, subsequent calls to
	/// sample will not change its value.
	void setObserved(double value) { m_observed = true; m_observedValue = value; }

	/// Set this node to an unobserved value. After calling this, subsequent calls to
	/// sample will draw new values for this node.
	void setUnobserved() { m_observed = false; }

	/// Returns the total number of combinations of categorical values supported by
	/// the categorical parents of this node. This node is expected to specify a
	/// distribution for each of these categorical combinations.
	size_t catCount();

	/// Queries the categorical parent nodes to determine their current values, and
	/// combines them to produce a single index value that is unique for this combination
	/// of categorical values. The value this returns will be from 0 to catCount()-1.
	/// The values are organized in little-endian manner. That is, it cycles through
	/// all values supported by the first categorical parent before moving on to the
	/// next value of the next parent.
	size_t currentCatIndex();

	/// This method links back to the child node that just added this node as a parent.
	virtual void onNewChild(GPGMVariable* pChild);
};




/// A node in a probabilistic graphical model that represents a categorical distribution.
/// Instances of this class can serve as parent-nodes to any GPGMVariable node. The child
/// node must specify distribution parameters for every category that the parent node
/// supports (or for every combination of categories if there are multiply categorical parents).
class GPGMCategorical : public GPGMVariable
{
protected:
	size_t m_categories;
	size_t m_val;
	std::vector<GPGMNode*> m_weights;

public:
	/// General-purpose constructor. All of the categories will initially be given a weight
	/// of pDefaultWeight. Typically, you will want to change these (by calling setWeights)
	/// after you construct a node.
	GPGMCategorical(size_t categories, GPGMNode* pDefaultWeight);
	virtual ~GPGMCategorical() {}

	/// Returns the number of categories supported by this distribution.
	size_t categories() { return m_categories; }

	/// Returns the most recent value sampled from this node.
	virtual double currentValue();

	/// Adds a categorical node as a parent of this node. Calling this method will cause
	/// This node to resize its table of distribution parameters, so a default value is
	/// required to fill in new elements. Typically, you will set these new elements to
	/// something more meaningful (by calling setWeights) after you call this method.
	virtual void addCatParent(GPGMCategorical* pNode, GPGMNode* pDefaultVal);

	/// Set the weights for one of the categorical distributions of this node.
	/// If there are n categories in this node, then the first n parameters should be non-NULL.
	/// "cat" specifies the index of the combination of values (in little-endian order) in the
	/// categorical parent distributions for which this categorical distribution is being specified.
	void setWeights(size_t cat, GPGMNode* pW1, GPGMNode* pW2, GPGMNode* pW3 = NULL, GPGMNode* pW4 = NULL, GPGMNode* pW5 = NULL, GPGMNode* pW6 = NULL, GPGMNode* pW7 = NULL, GPGMNode* pW8 = NULL);

	/// Draws a new value for this node given the current values of all other nodes
	/// it its Markov blanket.
	virtual void sample(GRand* pRand);

	/// Computes the likelihood that the specified value (after being truncated to an integer)
	/// would be drawn from this categorical distribution given the current values of all the
	/// parent nodes of this class.
	virtual double likelihood(double x);
};



/// This is the base class for nodes in a probabilistic graphical model that are sampled using
/// the Metropolis algorithm.
class GPGMMetropolisNode : public GPGMVariable
{
protected:
	double m_currentMean, m_currentDeviation;
	unsigned int m_nSamples;
	unsigned int m_nNewValues;
	double m_sumOfValues, m_sumOfSquaredValues;

public:
	/// General-purpose constructor. The parameters priorMean and priorDeviation
	/// are for the normal distribution used by the Metropolis algorithm to draw
	/// candidate values.
	GPGMMetropolisNode(double priorMean, double priorDeviation);
	virtual ~GPGMMetropolisNode() {}

	/// Returns the most recent value sampled from this node.
	virtual double currentValue();

	/// Draws a new value for this node given the current values of all other nodes
	/// it its Markov blanket. Uses the Metropolis algorithm to do so.
	void sample(GRand* pRand);

	/// This should return true iff this node supports only discrete values
	virtual bool isDiscrete() = 0;

protected:
	/// Computes the log-probability of x (as a value for this node) given
	/// the current values for the entire rest of the network (aka the
	/// complete conditional), which according to Gibbs, is equal to
	/// the log-probability of x given the Markov-Blanket of this node,
	/// which we can compute efficiently.
	double gibbs(double x);

	/// Sample the network in a manner that can be proven to converge to a
	/// true joint distribution for the network. Returns true if the new candidate
	/// value is selected.
	bool metropolis(GRand* pRand);
};



/// A node in a probabilistic graphical model that is distributed according to a Gaussian,
/// or normal, distribution.
class GPGMNormal : public GPGMMetropolisNode
{
protected:
	std::vector<GPGMNode*> m_meanAndDev;

public:
	/// General-purpose constructor. The prior mean and deviation parameters are
	/// for the Metropolis algorithm. pDefaultVal specifies a bogus value to be
	/// used for the mean and deviation of this distribution. Typically, you will
	/// change these values (by calling setMeanAndDev) after constructing this node.
	GPGMNormal(double priorMean, double priorDeviation, GPGMNode* pDefaultVal);
	~GPGMNormal() {}

	/// Adds a categorical node as a parent of this node. Calling this method will cause
	/// This node to resize its table of distribution parameters, so a default value is
	/// required to fill in new elements. Typically, you will set these new elements to
	/// something more meaningful (by calling setMeanAndDev) after you call this method.
	virtual void addCatParent(GPGMCategorical* pNode, GPGMNode* pDefaultVal);

	/// Set the mean and deviation for one of the normal distributions of this node.
	/// "cat" specifies the index of the combination of values (in little-endian order) in the
	/// categorical parent distributions for which this normal distribution is being specified.
	void setMeanAndDev(size_t cat, GPGMNode* pMean, GPGMNode* pDeviation);

	/// Returns false.
	virtual bool isDiscrete() { return false; }

	/// Returns true for all values.
	virtual bool isSupported(double val) { return true; }

	/// Returns the likelihood that the value x would be drawn from this distribution given
	/// the current values of all the parent nodes.
	virtual double likelihood(double x);
};




/// This class provides a platform for probabilistic graphical models.
/// (Perhaps, it would more-properly be called GProbabilisticGraphicalModel, but
/// that just doesn't seem to have as nice of a ring to it, so I called it GBayesNet instead.)
/// It allocates nodes in its own help using placement new, so you don't have to worry about deleting the nodes.
/// You can allocate your nodes manually and use them separately from this class if you want, but it is a lot
/// easier if you use this class to do it all.
class GBayesNet
{
protected:
	GHeap m_heap;
	std::vector<GPGMVariable*> m_sampleNodes;
	GRand m_rand;
	GPGMConstant* m_pConstOne;

public:
	/// General-purpose constructor
	GBayesNet(size_t seed = 0);
	~GBayesNet();

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if any tests fail.
	static void test();
#endif
	/// Return a constant value that can be used as a default parameter for various nodes.
	GPGMConstant* def() { return m_pConstOne; }

	/// Return a pointer to a node that represents a constant value.
	/// This node is allocated using placement new in an internal heap,
	/// so you do not need to worry about deleting it.
	GPGMConstant* newConst(double val);

	/// Return a pointer to a node that represents a categorical distribution.
	/// This node is allocated using placement new in an internal heap,
	/// so you do not need to worry about deleting it.
	GPGMCategorical* newCat(size_t categories);

	/// Return a pointer to a node that represents a normal distribution.
	/// This node is allocated using placement new in an internal heap,
	/// so you do not need to worry about deleting it.
	GPGMNormal* newNormal(double priorMean, double priorDev);

	/// Sample each node in the graph one time
	void sample();
};



} // namespace GClasses

#endif // __GPGM_H__
