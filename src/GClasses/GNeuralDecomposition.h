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
		GNeuralDecomposition(const GDomNode *pNode);
		virtual ~GNeuralDecomposition();
		
		virtual void trainOnSeries(const GMatrix &series);
		virtual GMatrix *extrapolate(double start = 1.0, double length = 1.0, double step = 0.0002, bool outputFeatures = false);
		virtual GMatrix *extrapolate(const GMatrix &features);
		
		void setRegularization(double newregularization) { m_regularization = newregularization; }
		void setLearningRate(double newlearningRate) { m_learningRate = newlearningRate; }
		void setFeatureScale(double newfeatureScale) { m_featureScale = newfeatureScale; }
		void setFeatureBias(double newfeatureBias) { m_featureBias = newfeatureBias; }
		void setOutputScale(double newoutputScale) { m_outputScale = newoutputScale; }
		void setOutputBias(double newoutputBias) { m_outputBias = newoutputBias; }
		void setLinearUnits(size_t linearUnits) { m_linearUnits = linearUnits; }
		void setSoftplusUnits(size_t softplusUnits) { m_softplusUnits = softplusUnits; }
		void setSigmoidUnits(size_t sigmoidUnits) { m_sigmoidUnits = sigmoidUnits; }
		void setSinusoidUnits(size_t sinusoidUnits) { m_sinusoidUnits = sinusoidUnits; }
		void setEpochs(size_t newepochs) { m_epochs = newepochs; }
		void setFilterLogarithm(bool filter_Logarithm) { m_filterLogarithm = filter_Logarithm; }
		void setAutoFilter(bool auto_Filter) { m_autoFilter = auto_Filter; }
		
		GNeuralNet &nn() const { return *m_nn; }
		double regularization() const { return m_regularization; }
		double learningRate() const { return m_learningRate; }
		double featureScale() const { return m_featureScale; }
		double featureBias() const { return m_featureBias; }
		double outputScale() const { return m_outputScale; }
		double outputBias() const { return m_outputBias; }
		size_t epochs() const { return m_epochs; }
		bool filterLogarithm() const { return m_filterLogarithm; }
		bool autoFilter() const { return m_autoFilter; }
		
		// GSupervisedLearner
		virtual GDomNode* serialize(GDom* pDoc) const;
		virtual void predict(const GVec& in, GVec& out);
		virtual void predictDistribution(const GVec& in, GPrediction *pOut);
		virtual void clear() {}
		
		// GIncrementalLearner
		virtual void trainIncremental(const GVec& in, const GVec&pOut);
		virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);
		
		static void test();
	
	protected:
		virtual void trainInner(const GMatrix& features, const GMatrix& labels);
		virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);
	
	private:
		GNeuralNet *m_nn;
		double m_regularization, m_learningRate, m_featureScale, m_featureBias, m_outputScale, m_outputBias;
		size_t m_linearUnits, m_softplusUnits, m_sigmoidUnits, m_sinusoidUnits, m_epochs;
		bool m_filterLogarithm, m_autoFilter;
};

} // namespace GClasses

#endif
