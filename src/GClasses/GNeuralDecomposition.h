/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Luke B. Godfrey,
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

#ifndef __GNEURALDECOMPOSITON_H__
#define __GNEURALDECOMPOSITON_H__

#include "GLearner.h"
#include "GNeuralNet.h"
#include "GMatrix.h"

namespace GClasses {

class GNeuralDecomposition: public GIncrementalLearner
{
	public:
		GNeuralDecomposition();
		GNeuralDecomposition(GDomNode *pNode, GLearnerLoader &ll);
		virtual ~GNeuralDecomposition();
		
		virtual void trainOnSeries(const GMatrix &series);
		virtual GMatrix *extrapolate(double start = 1.0, double length = 1.0, double step = 0.0002);
		virtual GMatrix *extrapolate(const GMatrix &features);
		
		void setRegularization(double regularization) { m_regularization = regularization; }
		void setLearningRate(double learningRate) { m_learningRate = learningRate; }
		void setFeatureScale(double featureScale) { m_featureScale = featureScale; }
		void setFeatureBias(double featureBias) { m_featureBias = featureBias; }
		void setLinearUnits(size_t linearUnits) { m_linearUnits = linearUnits; }
		void setSinusoidUnits(size_t sinusoidUnits) { m_sinusoidUnits = sinusoidUnits; }
		void setEpochs(size_t epochs) { m_epochs = epochs; }
		
		GNeuralNet &nn() const { return *m_nn; }
		double regularization() const { return m_regularization; }
		double learningRate() const { return m_learningRate; }
		double featureScale() const { return m_featureScale; }
		double featureBias() const { return m_featureBias; }
		size_t epochs() const { return m_epochs; }
		
		// GSupervisedLearner
		virtual GDomNode *serialize(GDom *pDoc) const;
		virtual void predict(const double *pIn, double *pOut);
		virtual void predictDistribution(const double *pIn, GPrediction *pOut);
		virtual void clear();
		
		// GIncrementalLearner
		virtual void trainIncremental(const double *pIn, const double *pOut);
		virtual void trainSparse(GSparseMatrix &features, GMatrix &labels);
	
	protected:
		virtual void trainInner(const GMatrix &features, const GMatrix &labels);
		virtual void beginIncrementalLearningInner(const GRelation &featureRel, const GRelation &labelRel);
	
	private:
		GNeuralNet *m_nn;
		double m_regularization, m_learningRate, m_featureScale, m_featureBias;
		size_t m_linearUnits, m_sinusoidUnits, m_epochs;
};

} // namespace GClasses

#endif
