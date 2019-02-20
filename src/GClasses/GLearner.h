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

#ifndef __GLEARNER_H__
#define __GLEARNER_H__

#include "GMatrix.h"
#include "GRand.h"

#include <memory>

namespace GClasses {

class GDom;
class GDomNode;
class GDistribution;
class GCategoricalDistribution;
class GNormalDistribution;
class GUniformDistribution;
class GUnivariateDistribution;
class GIncrementalTransform;
class GSparseMatrix;
class GCollaborativeFilter;
class GNeuralNetLearner;
class GLearnerLoader;

/// This class is used to represent the predicted distribution made by a supervised learning algorithm.
/// (It is just a shallow wrapper around GDistribution.) It is used in conjunction with calls
/// to GSupervisedLearner::predictDistribution. The predicted distributions will be either
/// categorical distributions (for nominal values) or Normal distributions (for continuous values).
class GPrediction
{
protected:
	GUnivariateDistribution* m_pDistribution;

public:
	GPrediction() : m_pDistribution(NULL)
	{
	}

	~GPrediction();

	/// Converts an array of prediction objects to a vector of most-likely values.
	static void predictionArrayToVector(size_t nOutputCount, GPrediction* pOutputs, GVec& vec);

	/// Converts an array of values to an array of predictions. There's not really
	/// enough information for this conversion, so it simply fabricates the variance
	/// and class-probability information as needed. Only the mean (for normal distributions)
	/// and the most-likely class (for categorical distributions) is reliable after this
	/// conversion.
	static void vectorToPredictionArray(GRelation* pRelation, size_t nOutputCount, GVec& vec, GPrediction* pOutputs);

	/// Returns true if this wraps a normal distribution, false otherwise
	bool isContinuous();

	/// Returns the mode (most likely value). For the Normal distribution, this is the same as the mean.
	double mode();

	/// If the current distribution is not a categorical distribution, then it
	/// replaces it with a new categorical distribution. Then it returns the
	/// current (categorical) distribution.
	GCategoricalDistribution* makeCategorical();

	/// If the current distribution is not a normal distribution, then it
	/// replaces it with a new normal distribution. Then it returns the
	/// current (normal) distribution.
	GNormalDistribution* makeNormal();

	/// Returns the current distribution. Throws if it is not a categorical distribution
	GCategoricalDistribution* asCategorical();

	/// Returns the current distribution. Throws if it is not a normal distribution
	GNormalDistribution* asNormal();
};



// nRep and nFold are zero-indexed
typedef void (*RepValidateCallback)(void* pThis, size_t nRep, size_t nFold, double foldSSE, size_t rows);


/// This is the base class of supervised learning algorithms (that may or may not
/// have an internal model allowing them to generalize rows that were not available
/// at training time). Note that the literature typically refers to supervised learning
/// algorithms that can't generalize (because they lack an internal hypothesis model)
/// as "Semi-supervised". (You cannot generalize with a semi-supervised algorithm--you have to
/// train again with the new rows.)
class GTransducer
{
protected:
	GRand m_rand;

public:
	/// General-purpose constructor.
	GTransducer();

	/// Copy-constructor. Throws an exception to prevent models from being copied by value.
	GTransducer(const GTransducer& that) : m_rand(0)
	{
		throw Ex("This object is not intended to be copied by value");
	}

	virtual ~GTransducer();

	/// Throws an exception to prevent models from being copied by value.
	GTransducer& operator=(const GTransducer& other)
	{
		throw Ex("This object is not intended to be copied by value");
	}


	/// Returns false because semi-supervised learners have no internal
	/// model, so they can't evaluate previously unseen rows.
	virtual bool canGeneralize() { return false; }

	/// Returns false because semi-supervised learners cannot be trained incrementally.
	virtual bool canTrainIncrementally() { return false; }

	/// Predicts a set of labels to correspond with features2, such that these
	/// labels will be consistent with the patterns exhibited by features1 and labels1.
	std::unique_ptr<GMatrix> transduce(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2);

	/// Trains and tests this learner. Returns the sum-squared-error.
	/// if pOutSAE is not NULL, the sum absolute error will be placed there.
	virtual double trainAndTest(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& testFeatures, const GMatrix& testLabels, double* pOutSAE = NULL);

	/// Makes a confusion matrix for a transduction algorithm
	void transductiveConfusionMatrix(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& testFeatures, const GMatrix& testLabels, std::vector<GMatrix*>& stats);

	/// Perform n-fold cross validation on pData. Returns sum-squared error.
	/// Uses trainAndTest for each fold. pCB is an optional callback method for reporting
	/// intermediate stats. It can be NULL if you don't want intermediate reporting.
	/// nRep is just the rep number that will be passed to the callback.
	/// pThis is just a pointer that will be passed to the callback for you
	/// to use however you want. It doesn't affect this method.
	/// if pOutSAE is not NULL, the sum absolute error will be placed there.
	double crossValidate(const GMatrix& features, const GMatrix& labels, size_t nFolds, double* pOutSAE = NULL, RepValidateCallback pCB = NULL, size_t nRep = 0, void* pThis = NULL);

	/// Perform cross validation "nReps" times and return the
	/// average score. pCB is an optional callback method for reporting intermediate stats
	/// It can be NULL if you don't want intermediate reporting.
	/// pThis is just a pointer that will be passed to the callback for you
	/// to use however you want. It doesn't affect this method.
	/// if pOutSAE is not NULL, the sum absolute error will be placed there.
	double repValidate(const GMatrix& features, const GMatrix& labels, size_t reps, size_t nFolds, double* pOutSAE = NULL, RepValidateCallback pCB = NULL, void* pThis = NULL);

	/// Returns a reference to the random number generator associated with this object.
	/// For example, you could use it to change the random seed, to make this algorithm behave differently.
	/// This might be important, for example, in an ensemble of learners.
	GRand& rand() { return m_rand; }

	/// Returns true iff this algorithm can implicitly handle nominal features. If it
	/// cannot, then the GNominalToCat transform will be used to convert nominal
	/// features to continuous values before passing them to it.
	virtual bool canImplicitlyHandleNominalFeatures() { return true; }

	/// Returns true iff this algorithm can implicitly handle continuous features. If it
	/// cannot, then the GDiscretize transform will be used to convert continuous
	/// features to nominal values before passing them to it.
	virtual bool canImplicitlyHandleContinuousFeatures() { return true; }

	/// Returns true if this algorithm supports any feature value, or if it does not
	/// implicitly handle continuous features. If a limited range of continuous values is
	/// supported, returns false and sets pOutMin and pOutMax to specify the range.
	virtual bool supportedFeatureRange(double* pOutMin, double* pOutMax) { return true; }

	/// Returns true iff this algorithm supports missing feature values. If it cannot,
	/// then an imputation filter will be used to predict missing values before any
	/// feature-vectors are passed to the algorithm.
	virtual bool canImplicitlyHandleMissingFeatures() { return true; }

	/// Returns true iff this algorithm can implicitly handle nominal labels (a.k.a.
	/// classification). If it cannot, then the GNominalToCat transform will be
	/// used during training to convert nominal labels to continuous values, and
	/// to convert categorical predictions back to nominal labels.
	virtual bool canImplicitlyHandleNominalLabels() { return true; }

	/// Returns true iff this algorithm can implicitly handle continuous labels (a.k.a.
	/// regression). If it cannot, then the GDiscretize transform will be
	/// used during training to convert nominal labels to continuous values, and
	/// to convert nominal predictions back to continuous labels.
	virtual bool canImplicitlyHandleContinuousLabels() { return true; }

	/// Returns true if this algorithm supports any label value, or if it does not
	/// implicitly handle continuous labels. If a limited range of continuous values is
	/// supported, returns false and sets pOutMin and pOutMax to specify the range.
	virtual bool supportedLabelRange(double* pOutMin, double* pOutMax) { return true; }

protected:
	/// This is the algorithm's implementation of transduction. (It is called by the transduce method.)
	virtual std::unique_ptr<GMatrix> transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2) = 0;
};


/// This is the base class of algorithms that learn with supervision and
/// have an internal hypothesis model that allows them to generalize
/// rows that were not available at training time.
class GSupervisedLearner : public GTransducer
{
protected:
	GRelation* m_pRelFeatures;
	GRelation* m_pRelLabels;

public:
	/// General-purpose constructor.
	GSupervisedLearner();

	/// Deserialization constructor
	GSupervisedLearner(const GDomNode* pNode);

	/// Destructor
	virtual ~GSupervisedLearner();

	/// Marshal this object into a DOM that can be converted to a variety
	/// of formats. (Implementations of this method should use baseDomNode.)
	virtual GDomNode* serialize(GDom* pDoc) const = 0;

	/// Returns true because fully supervised learners have an internal
	/// model that allows them to generalize previously unseen rows.
	virtual bool canGeneralize() { return true; }

	/// Returns a reference to the feature relation (meta-data about the input attributes).
	const GRelation& relFeatures();

	/// Returns a reference to the label relation (meta-data about the output attributes).
	const GRelation& relLabels();

	/// Returns false
	virtual bool isFilter() { return false; }

	/// Call this method to train the model.
	void train(const GMatrix& features, const GMatrix& labels);


	/// Evaluate pIn to compute a prediction for pOut. The model must be trained
	/// (by calling train) before the first time that this method is called.
	/// pIn and pOut should point to arrays of doubles of the same size as the
	/// number of columns in the training matrices that were passed to the train
	/// method.
	virtual void predict(const GVec& in, GVec& out) = 0;

	/// Evaluate pIn and compute a prediction for pOut. pOut is expected
	/// to point to an array of GPrediction objects which have already been
	/// allocated. There should be labelDims() elements in this array.
	/// The distributions will be more accurate if the model is calibrated
	/// before the first time that this method is called.
	virtual void predictDistribution(const GVec& in, GPrediction* pOut) = 0;

	/// Discards all training for the purpose of freeing memory.
	/// If you call this method, you must train before making any predictions.
	/// No settings or options are discarded, so you should be able to
	/// train again without specifying any other parameters and still get
	/// a comparable model.
	virtual void clear() = 0;

	/// Generates a confusion matrix containing the total counts of the number of times
	/// each value was expected and predicted. (Rows represent target values, and columns
	/// represent predicted values.) stats should be an empty vector. This method will
	/// resize stats to the number of dimensions in the label vector. The caller is
	/// responsible to delete all of the matrices that it puts in this vector. For
	/// continuous labels, the value will be NULL.
	void confusion(GMatrix& features, GMatrix& labels, std::vector<GMatrix*>& stats);

	/// Computes the sum-squared-error for predicting the labels from the features.
	/// For categorical labels, Hamming distance is used.
	double sumSquaredError(const GMatrix& features, const GMatrix& labels, double* pOutSAE = NULL);

	/// label specifies which output to measure. (It should be 0 if there is only one label dimension.)
	/// The measurement will be performed "nReps" times and results averaged together
	/// nPrecisionSize specifies the number of points at which the function is sampled
	/// pOutPrecision should be an array big enough to hold nPrecisionSize elements for every possible
	/// label value. (If the attribute is continuous, it should just be big enough to hold nPrecisionSize elements.)
	/// If bLocal is true, it computes the local precision instead of the global precision.
	void precisionRecall(double* pOutPrecision, size_t nPrecisionSize, GMatrix& features, GMatrix& labels, size_t label, size_t nReps);

	/// Trains and tests this learner. Returns sum-squared-error.
	virtual double trainAndTest(const GMatrix& trainFeatures, const GMatrix& trainLabels, const GMatrix& testFeatures, const GMatrix& testLabels, double* pOutSAE = NULL);

	/// This is a helper method used by the unit tests of several model learners
	void basicTest(double minAccuracy1, double minAccuracy2, double deviation = 1e-6, bool printAccuracy = false, double warnRange = 0.035);

	/// Runs some unit tests related to supervised learning. Throws an exception if any problems are found.
	static void test();
protected:
	/// Adds the function pIn to pOut after interpolating pIn to be the same size as pOut.
	/// (This is a helper-function used by precisionRecall.)
	static void addInterpolatedFunction(double* pOut, size_t nOutVals, double* pIn, size_t nInVals);

	/// This is the implementation of the model's training algorithm. (This method is called by train).
	virtual void trainInner(const GMatrix& features, const GMatrix& labels) = 0;

	/// See GTransducer::transduce
	virtual std::unique_ptr<GMatrix> transduceInner(const GMatrix& features1, const GMatrix& labels1, const GMatrix& features2);

	/// This method determines which data filters (normalize, discretize,
	/// and/or nominal-to-cat) are needed and trains them.
	void setupFilters(const GMatrix& features, const GMatrix& labels);

	/// This is a helper method used by precisionRecall.
	size_t precisionRecallNominal(GPrediction* pOutput, double* pFunc, GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, size_t label, int value);

	/// This is a helper method used by precisionRecall.
	size_t precisionRecallContinuous(GPrediction* pOutput, double* pFunc, GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, size_t label);

	/// Child classes should use this in their implementation of serialize
	GDomNode* baseDomNode(GDom* pDoc, const char* szClassName) const;
};

///\brief Converts a GSupervisedLearner to a string
std::string to_str(const GSupervisedLearner& learner);





/// This is the base class of supervised learning algorithms that
/// can learn one row at a time.
class GIncrementalLearner : public GSupervisedLearner
{
public:
	/// General-purpose constructor.
	GIncrementalLearner()
	: GSupervisedLearner()
	{
	}

	/// Deserialization constructor
	GIncrementalLearner(const GDomNode* pNode)
	: GSupervisedLearner(pNode)
	{
	}

	/// Destructor
	virtual ~GIncrementalLearner()
	{
	}

	/// Only the GFilter class should return true to this method
	virtual bool isFilter() { return false; }

	/// Returns true
	virtual bool canTrainIncrementally() { return true; }

	/// You must call this method before you call trainIncremental.
	void beginIncrementalLearning(const GRelation& featureRel, const GRelation& labelRel);

	/// A version of beginIncrementalLearning that supports data-dependent filters.
	void beginIncrementalLearning(const GMatrix& features, const GMatrix& labels);

	/// Pass a single input row and the corresponding label to incrementally train this model.
	virtual void trainIncremental(const GVec& in, const GVec& out) = 0;

	/// Train using a sparse feature matrix. (A Typical implementation of this
	/// method will first call beginIncrementalLearning, then it will
	/// iterate over all of the feature rows, and for each row it
	/// will convert the sparse row to a dense row, call trainIncremental
	/// using the dense row, then discard the dense row and proceed to the next row.)
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels) = 0;

protected:
	/// Prepare the model for incremental learning.
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel) = 0;

	virtual void beginIncrementalLearningInner(const GMatrix& features, const GMatrix& labels)
	{
		beginIncrementalLearningInner(features.relation(), labels.relation());
	}
};



/// This class is for loading various learning algorithms from a DOM. When any
/// learning algorithm is saved, it calls baseDomNode, which creates (among other
/// things) a field named "class" which specifies the class name of the algorithm.
/// This class contains methods that will recognize any of the classes in this library
/// and load them. If it doesn't recognize a class, it will either return NULL or throw
/// and exception, depending on the flags you pass to the constructor.
/// Obviously this loader won't recognize any classes that you make. Therefore, you should
/// overload the corresponding method in this class with a new method that will first
/// recognize and load your classes, and then call these methods to handle other types.
class GLearnerLoader
{
protected:
	bool m_throwIfClassNotFound;

public:
	/// Constructor. If throwIfClassNotFound is true, then all of the methods in this
	/// class will throw an exception of the DOM refers to an unrecognized class.
	/// If throwIfClassNotFound is false, then NULL will be returned if the class
	/// is not recognized.
	GLearnerLoader(bool throwIfClassNotFound = true)
	: m_throwIfClassNotFound(throwIfClassNotFound)
	{
	}

	virtual ~GLearnerLoader() {}

	/// Loads an incremental transform (or a two-way incremental transform) from a DOM.
	virtual GIncrementalTransform* loadIncrementalTransform(const GDomNode* pNode);

	/// Loads a learning algorithm from a DOM.
	virtual GSupervisedLearner* loadLearner(const GDomNode* pNode);

	/// Loads a collaborative filtering algorithm from a DOM.
	virtual GCollaborativeFilter* loadCollaborativeFilter(const GDomNode* pNode);
};



class GFilter : public GIncrementalLearner
{
protected:
	GSupervisedLearner* m_pLearner;
	GIncrementalLearner* m_pIncrementalLearner;
	GSupervisedLearner* m_pOriginal;
	bool m_ownLearner;


	GFilter(GSupervisedLearner* pLearner, bool ownLearner = true);

	/// Deserialization constructor
	GFilter(const GDomNode* pNode, GLearnerLoader& ll);

	virtual ~GFilter();

	virtual bool canTrainIncrementally() { return m_pLearner->canTrainIncrementally(); }

	/// Discards any filters between this filter and the base learner
	void discardIntermediateFilters();

	/// Recursive method used by discardIntermediateFilters
	void discardIntermediateFilters_helper(GSupervisedLearner* pOriginal);

	/// Helper function for serialization
	GDomNode* domNode(GDom* pDoc, const char* szClassName) const;

public:
	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// Returns true
	virtual bool isFilter() { return true; }

	/// Initialize (or train) this filter without calling train on any of the
	/// interior components. (This might be used when filtering a learner that
	/// has already been trained with a transform that has also already been trained.)
	void initShellOnly(const GRelation& featureRel, const GRelation& labelRel);

	/// Transform a feature vector to the form for presenting to the inner learner
	virtual const GVec& prefilterFeatures(const GVec& in) = 0;

	/// Transform a label vector to the form for presenting to the inner learner
	virtual const GVec& prefilterLabels(const GVec& in) = 0;

	/// Transform a feature matrix to the form for presenting to the inner learner
	GMatrix* prefilterFeatures(const GMatrix& in);

	/// Transform a label matrix to the form for presenting to the inner learner
	GMatrix* prefilterLabels(const GMatrix& in);

	/// Returns a pointer to the inner learner (could be another GFilter)
	GSupervisedLearner* innerLearner() { return m_pLearner; }

	/// Returns a pointer to the base larner
	GSupervisedLearner* baseLearner() { return m_pOriginal; }
	/// Throws an exception
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels);
};


class GFeatureFilter : public GFilter
{
protected:
	GIncrementalTransform* m_pTransform;
	bool m_ownTransform;

public:
using GFilter::prefilterFeatures;
using GFilter::prefilterLabels;
	/// This takes ownership of pLearner and pTransform.
	GFeatureFilter(GSupervisedLearner* pLearner, GIncrementalTransform* pTransform, bool ownLearner = true, bool ownTransform = true);

	/// Deserialization constructor
	GFeatureFilter(const GDomNode* pNode, GLearnerLoader& ll);

	/// Deletes the supervised learner and the transform
	virtual ~GFeatureFilter();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const GVec& in, const GVec& out);

	/// Transform a feature vector to the form for presenting to the inner learner
	virtual const GVec& prefilterFeatures(const GVec& in);

	/// Transform a label vector to the form for presenting to the inner learner
	virtual const GVec& prefilterLabels(const GVec& in);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GMatrix& features, const GMatrix& labels);
};



class GLabelFilter : public GFilter
{
protected:
	GIncrementalTransform* m_pTransform;
	bool m_ownTransform;

public:
using GFilter::prefilterFeatures;
using GFilter::prefilterLabels;
	/// This takes ownership of pLearner and pTransform.
	GLabelFilter(GSupervisedLearner* pLearner, GIncrementalTransform* pTransform, bool ownLearner = true, bool ownTransform = true);

	/// Deserialization constructor
	GLabelFilter(const GDomNode* pNode, GLearnerLoader& ll);

	/// Deletes the supervised learner and the transform
	virtual ~GLabelFilter();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const GVec& in, const GVec& out);

	/// Transform a feature vector to the form for presenting to the inner learner
	virtual const GVec& prefilterFeatures(const GVec& in);

	/// Transform a label vector to the form for presenting to the inner learner
	virtual const GVec& prefilterLabels(const GVec& in);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GMatrix& features, const GMatrix& labels);
};




class GAutoFilter : public GFilter
{
protected:
	std::vector<GMatrix*> m_prefilteredData;

public:
using GFilter::prefilterFeatures;
using GFilter::prefilterLabels;

	/// General-purpose constructor. If ownLearner is true, then this will delete pLearner when this object is deleted.
	GAutoFilter(GSupervisedLearner* pLearner, bool ownLearner = true);

	/// Deserialization constructor
	GAutoFilter(const GDomNode* pNode, GLearnerLoader& ll);

	/// Deletes the supervised learner and the transform
	virtual ~GAutoFilter();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

	/// See the comment for GIncrementalLearner::trainIncremental
	virtual void trainIncremental(const GVec& in, const GVec& out);

	/// Transform a feature vector to the form for presenting to the inner learner
	virtual const GVec& prefilterFeatures(const GVec& in);

	/// Transform a label vector to the form for presenting to the inner learner
	virtual const GVec& prefilterLabels(const GVec& in);

	/// Prefilters multiple datasets and stores the results in the same order in a vector that can be retrieved by calling data().
	void prefilterData(const GMatrix* pFeatures1, const GMatrix* pLabels1, const GMatrix* pFeatures2 = nullptr, const GMatrix* pLabels2 = nullptr, const GMatrix* pFeatures3 = nullptr, const GMatrix* pLabels3 = nullptr, const GMatrix* pFeatures4 = nullptr, const GMatrix* pLabels4 = nullptr);

	/// Returns a reference to a vector of prefiltered datasets. (This vector is populated by calling prefilterData.)
	std::vector<GMatrix*>& data() { return m_prefilteredData; }

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel);

	/// See the comment for GIncrementalLearner::beginIncrementalLearningInner
	virtual void beginIncrementalLearningInner(const GMatrix& features, const GMatrix& labels);

	void whatTypesAreNeeded(const GRelation& featureRel, const GRelation& labelRel, bool& hasNominalFeatures, bool& hasContinuousFeatures, bool& hasNominalLabels, bool& hasContinuousLabels);
	void setupDataDependentFilters(GSupervisedLearner* pLearner, const GMatrix& features, const GMatrix& labels, bool hasNominalFeatures, bool hasContinuousFeatures, bool hasNominalLabels, bool hasContinuousLabels);
	void setupBasicFilters(GSupervisedLearner* pLearner, bool hasNominalFeatures, bool hasContinuousFeatures, bool hasNominalLabels, bool hasContinuousLabels);
	void resetFilters(const GMatrix& features, const GMatrix& labels);
	void resetFilters(const GRelation& features, const GRelation& labels);
};




/// Always outputs the label mean (for continuous labels) and the most common
/// class (for nominal labels).
class GBaselineLearner : public GSupervisedLearner
{
protected:
	std::vector<double> m_prediction;

public:
	/// General-purpose constructor
	GBaselineLearner();

	/// Deserialization constructor
	GBaselineLearner(const GDomNode* pNode);

	/// Destructor
	virtual ~GBaselineLearner();

	/// Run unit tests for this class.
	static void test();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// This model has no parameters to tune, so this method is a noop.
	void autoTune(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);
};


/// This is an implementation of the identity function. It might be
/// useful, for example, as the observation function in a GRecurrentModel
/// if you want to create a Jordan network.
class GIdentityFunction : public GSupervisedLearner
{
protected:
	size_t m_labelDims;
	size_t m_featureDims;

public:
	/// General-purpose constructor
	GIdentityFunction();

	/// Deserialization constructor
	GIdentityFunction(const GDomNode* pNode);

	/// Destructor
	virtual ~GIdentityFunction();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// See the comment for GSupervisedLearner::predict
	virtual void predict(const GVec& in, GVec& out);

	/// See the comment for GSupervisedLearner::predictDistribution
	virtual void predictDistribution(const GVec& in, GPrediction* pOut);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(const GMatrix& features, const GMatrix& labels);
};


} // namespace GClasses

#endif // __GLEARNER_H__

