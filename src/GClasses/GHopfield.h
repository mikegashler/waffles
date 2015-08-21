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

#ifndef __GHOPFIELD_H__
#define __GHOPFIELD_H__

#include "GLearner.h"
#include <vector>
#include "GTree.h"

#define MAX_NODES_PER_LAYER ((size_t)1024 * (size_t)1024 * (size_t)1024)

namespace GClasses {

class GActivationFunction;
class GHopfield;


/// A class used by GHopfield
class GHopfieldLayer
{
public:
	GActivationFunction* m_pActivationFunction;
	GMatrix m_forw; // rows = upstream layer, cols = this layer
	GMatrix m_back; // rows = downstream layer, cols = this layer
	GMatrix m_bias; // row 0 is bias, row 1 is net, row 2 is activation, row 3 is delta

public:
	GHopfieldLayer(size_t bef, size_t cur, size_t aft, GActivationFunction* pActivationFunction = NULL);

	~GHopfieldLayer()
	{
	}

	size_t units() { return m_forw.cols(); }
	double* bias() { return m_bias[0]; }
	double* net() { return m_bias[1]; }
	double* activation() { return m_bias[2]; }
	double* delta() { return m_bias[3]; }

	void init(GRand& rand);
};



// Helper class used by GHopfiled. The user should not need to use this class
// Col 0 = squared delta, larger first
// Col 1 = node {layer,index}
class GHopfieldNodeComparer
{
protected:
	GHopfield* m_pThat;

public:
	GHopfieldNodeComparer(GHopfield* pThat)
	: m_pThat(pThat)
	{
	}

	size_t cols() const { return 2; }

	bool operator ()(size_t a, size_t b, size_t col) const;
};




/// A class that implements bi-directional Hopfiled-like neural networks.
/// This class seeks to be more general than Hopfield's architecture,
/// so it also supports multiple layers, separate weights in each direction, and novel training methods.
class GHopfield : public GIncrementalLearner
{
friend class GHopfieldNodeComparer;
protected:
	std::vector<GHopfieldLayer*> m_layers;
	GHopfieldNodeComparer m_comp;
	GRelationalTable<size_t,GHopfieldNodeComparer> m_table;

public:
	GHopfield();

	/// Load from a text-format
	GHopfield(GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GHopfield();

	void addLayer(GHopfieldLayer* pLayer);

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Saves the model to a text file.
	virtual GDomNode* serialize(GDom* pDoc) const;
#endif // MIN_PREDICT

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const double* pIn, const double* pOut);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const double* pIn, double* pOut);

	/// See the comment for GTransducer::canImplicitlyHandleNominalFeatures
	virtual bool canImplicitlyHandleNominalFeatures() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedFeatureRange(double* pOutMin, double* pOutMax);

	/// See the comment for GTransducer::canImplicitlyHandleMissingFeatures
	virtual bool canImplicitlyHandleMissingFeatures() { return false; }

	/// See the comment for GTransducer::canImplicitlyHandleNominalLabels
	virtual bool canImplicitlyHandleNominalLabels() { return false; }

	/// See the comment for GTransducer::supportedFeatureRange
	virtual bool supportedLabelRange(double* pOutMin, double* pOutMax);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	double relaxPush();
	void relaxPull(size_t lay);
	void setActivation(GHopfieldLayer* pLayer, size_t layer, size_t index, double newActivation);
	void relax();
};


} // namespace GClasses

#endif // __GHOPFIELD_H__

