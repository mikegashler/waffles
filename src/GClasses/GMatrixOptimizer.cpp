#include "GMatrixOptimizer.h"
using namespace GClasses;

void GMatrixOptimizer::prepare(const GMatrix &weights)
{
	m_deltas.resize(weights.rows(), weights.cols());
}

GSGDMatrixOptimizer::GSGDMatrixOptimizer() : m_learningRate(1e-3), m_momentum(0)
{}

void GSGDMatrixOptimizer::applyDeltas(GMatrix &weights, size_t batchSize)
{
	double factor = m_learningRate / batchSize;
	for(size_t i = 0; i < m_deltas.rows(); ++i)
		weights[i].addScaled(factor, m_deltas[i]);
	m_deltas.multiply(m_momentum);
}
