#ifndef __GMATRIXOPTIMIZER_H__
#define __GMATRIXOPTIMIZER_H__

#include "GMatrix.h"
#include "GNeuralNet.h"
#include <vector>

namespace GClasses
{

/// A way to optimize a GMatrix given a GMatrix of its gradient.
/// This can be used, for example, to optimize a GNeuralNet.
class GMatrixOptimizer
{
public:
	virtual ~GMatrixOptimizer() {}
	
	/// Get a reference to the deltas (for updating them, i.e. by backpropagation)
	GMatrix &deltas()				{ return m_deltas; }
	
	/// Get a const reference to the deltas
	const GMatrix &deltas() const	{ return m_deltas; }
	
	/// Prepare for optimizing the given GMatrix of weights
	void prepare(const GMatrix &weights);
	
	/// Apply deltas to parameters; required
	virtual void applyDeltas(GMatrix &weights, size_t batchSize = 1) = 0;

protected:
	GMatrix m_deltas;
};

/// Stochastic gradient descent
class GSGDMatrixOptimizer : public GMatrixOptimizer
{
public:
	GSGDMatrixOptimizer();
	virtual void applyDeltas(GMatrix &weights, size_t batchSize = 1);
private:
	double m_learningRate, m_momentum;
};

/// Convenience template class for optimizing a GNeuralNet
template <typename T>
class GNeuralNetOptimizer
{
public:
	GNeuralNetOptimizer(GNeuralNet *nn = NULL)
	{
		setNeuralNet(nn);
	}
	
	~GNeuralNetOptimizer()
	{
		clear();
	}
	
	/// (Re)set target network
	void setNeuralNet(GNeuralNet *nn)
	{
		clear();
		m_nn = nn;
		if(m_nn != NULL)
		{
			m_optimizers.reserve(m_nn->layerCount());
			for(size_t i = 0; i < m_nn->layerCount(); ++i)
				m_optimizers.push_back(new T());
		}
	}
	
	/// Clear optimizers and neural net
	void clear()
	{
		m_nn = NULL;
		for(size_t i = 0; i < m_optimizers.size(); ++i)
			delete m_optimizers[i];
		m_optimizers.clear();
	}
	
	/// Prepare for learning
	void prepare()
	{
		GAssert(m_nn != NULL, "Neural network must be set before preparing to train it!");
		for(size_t i = 0; i < m_optimizers.size(); ++i)
			m_optimizers[i]->prepare(m_nn->layer(i).weights());
	}
	
	/// Learn from a single traning sample
	void train(const GVec &feat, const GVec &lab)
	{
		GAssert(m_nn != NULL, "Neural network must be set before training it!");
		updateDeltas(feat, lab);
		applyDeltas(feat);
	}
private:
	/// Update the gradient given a single training sample
	void updateDeltas(const GVec &feat, const GVec &lab)
	{
		m_nn->forwardProp(feat);
		m_nn->backpropagate(lab);
		const GVec *input = &feat;
		for(size_t i = 0; i < m_optimizers.size(); ++i)
		{
			m_nn->layer(i).updateDeltas(*input, m_optimizers[i]->deltas());
			input = &m_nn->layer(i).activation();
		}
	}
	
	/// Apply the gradient to the network
	void applyDeltas(const GVec &feat)
	{
		for(size_t i = 0; i < m_optimizers.size(); ++i)
			m_optimizers[i]->applyDeltas(m_nn->layer(i).weights());
	}
	
	GNeuralNet *m_nn;
	std::vector<T *> m_optimizers;
};

}

#endif
