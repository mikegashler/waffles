/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GLEARNER_H__
#define __GLEARNER_H__

#include "GMatrix.h"

namespace GClasses {

class GNeuralNet;


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
class GNeuralNet;
class GLearnerLoader;

#ifndef MIN_PREDICT
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
	static void predictionArrayToVector(size_t nOutputCount, GPrediction* pOutputs, double* pVector);

	/// Converts an array of values to an array of predictions. There's not really
	/// enough information for this conversion, so it simply fabricates the variance
	/// and class-probability information as needed. Only the mean (for normal distributions)
	/// and the most-likely class (for categorical distributions) is reliable after this
	/// conversion.
	static void vectorToPredictionArray(GRelation* pRelation, size_t nOutputCount, double* pVector, GPrediction* pOutputs);

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
#endif // MIN_PREDICT



// nRep and nFold are zero-indexed
typedef void (*RepValidateCallback)(void* pThis, size_t nRep, size_t nFold, size_t labelDims, double* pFoldResults);


/// This is the base class of supervised learning algorithms (that may or may not
/// have an internal model allowing them to generalize rows that were not available
/// at training time). Note that the literature typically refers to supervised learning
/// algorithms that can't generalize (because they lack an internal hypothesis model)
/// as "Semi-supervised". (You cannot generalize with a semi-supervised algorithm--you have to
/// train again with the new rows.)
class GTransducer
{
protected:
	GRand& m_rand;

public:
	GTransducer(GRand& rand);
	virtual ~GTransducer();
#ifndef MIN_PREDICT

	/// Returns false because semi-supervised learners have no internal
	/// model, so they can't evaluate previously unseen rows.
	virtual bool canGeneralize() { return false; }

	/// Returns false because semi-supervised learners cannot be trained incrementally.
	virtual bool canTrainIncrementally() { return false; }

	/// Predicts a set of labels to correspond with features2, such that these
	/// labels will be consistent with the patterns exhibited by features1 and labels1.
	GMatrix* transduce(GMatrix& features1, GMatrix& labels1, GMatrix& features2);

	/// Trains and tests this learner. pOutResults should have
	/// an element for each label dim.
	virtual void trainAndTest(GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, double* pOutResults, std::vector<GMatrix*>* pNominalLabelStats = NULL);

	/// Perform n-fold cross validation on pData. Uses trainAndTest
	/// for each fold. pCB is an optional callback method for reporting
	/// intermediate stats. It can be NULL if you don't want intermediate reporting.
	/// nRep is just the rep number that will be passed to the callback.
	/// pThis is just a pointer that will be passed to the callback for you
	/// to use however you want. It doesn't affect this method.
	/// The results of each fold is returned in a dataset.
	GMatrix* crossValidate(GMatrix& features, GMatrix& labels, size_t nFolds, RepValidateCallback pCB = NULL, size_t nRep = 0, void* pThis = NULL);

	/// Perform cross validation "nReps" times and return the
	/// average score. (5 reps with 2 folds is preferred over 10-fold cross
	/// validation because it yields less type 1 error.)
	/// pCB is an optional callback method for reporting intermediate stats
	/// It can be NULL if you don't want intermediate reporting.
	/// pThis is just a pointer that will be passed to the callback for you
	/// to use however you want. It doesn't affect this method.
	/// The results of each fold is returned in a dataset.
	GMatrix* repValidate(GMatrix& features, GMatrix& labels, size_t reps, size_t nFolds, RepValidateCallback pCB = NULL, void* pThis = NULL);

	/// This performs two-fold cross-validation on a shuffled
	/// non-uniform split of the data, and returns an error value that
	/// represents the results of all labels combined.
	double heuristicValidate(GMatrix& features, GMatrix& labels);
#endif // MIN_PREDICT

	/// Returns a reference to the random number generator associated with this object.
	GRand& rand() { return m_rand; }

protected:
#ifndef MIN_PREDICT
	/// This is the algorithm's implementation of transduction. (It is called by the transduce method.)
	virtual GMatrix* transduceInner(GMatrix& features1, GMatrix& labels1, GMatrix& features2) = 0;
#endif // MIN_PREDICT

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
};


/// This is the base class of algorithms that learn with supervision and
/// have an internal hypothesis model that allows them to generalize
/// rows that were not available at training time.
class GSupervisedLearner : public GTransducer
{
protected:
	GIncrementalTransform* m_pFilterFeatures;
	GIncrementalTransform* m_pFilterLabels;
	sp_relation m_pRelFeatures;
	sp_relation m_pRelLabels;
	GNeuralNet** m_pCalibrations;

public:
	/// General-purpose constructor
	GSupervisedLearner(GRand& rand);

	/// Deserialization constructor
	GSupervisedLearner(GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GSupervisedLearner();

#ifndef MIN_PREDICT
	/// Marshal this object into a DOM that can be converted to a variety
	/// of formats. (Implementations of this method should use baseDomNode.)
	virtual GDomNode* serialize(GDom* pDoc) const = 0;
#endif // MIN_PREDICT

	/// Returns true because fully supervised learners have an internal
	/// model that allows them to generalize previously unseen rows.
	virtual bool canGeneralize() { return true; }

	/// Returns a smart-pointer to the feature relation (meta-data about the input attributes).
	/// (Note that this relation describes outer data, and may contain types that are not
	/// supported by the inner algorithm.)
	sp_relation relFeatures() { return m_pRelFeatures; }

	/// Returns a smart-pointer to the label relation (meta-data about the output attributes).
	/// (Note that this relation describes outer data, and may contain types that are not
	/// supported by the inner algorithm.)
	sp_relation relLabels() { return m_pRelLabels; }

	/// Returns the current feature filter (or NULL if none has been set).
	GIncrementalTransform* featureFilter() { return m_pFilterFeatures; }

	/// Returns the current label filter (or NULL if none has been set).
	GIncrementalTransform* labelFilter() { return m_pFilterLabels; }

	/// Clears the filter for features.
	virtual void clearFeatureFilter();

	/// Wrap whatever feature filter is currently set with the specified filter
	void wrapFeatures(GIncrementalTransform* pFilter);

	/// Clears the filter for labels.
	virtual void clearLabelFilter();

#ifndef MIN_PREDICT
	/// Wrap whatever label filter is currently set with the specified filter
	void wrapLabels(GIncrementalTransform* pFilter);

	/// Call this method to train the model. It automatically determines which
	/// filters are needed to convert the training features and labels into
	/// a form that the model's training algorithm can handle, and then calls
	/// trainInner to do the actual training.
	void train(GMatrix& features, GMatrix& labels);
#endif // MIN_PREDICT

	/// Evaluate pIn to compute a prediction for pOut. The model must be trained
	/// (by calling train) before the first time that this method is called.
	/// pIn and pOut should point to arrays of doubles of the same size as the
	/// number of columns in the training matrices that were passed to the train
	/// method.
	void predict(const double* pIn, double* pOut);

#ifndef MIN_PREDICT
	/// Calibrate the model to make predicted distributions reflect the training
	/// data. This method should be called after train is called, but before
	/// the first time predictDistribution is called. Typically, the same matrices
	/// passed as parameters to the train method are also passed as parameters
	/// to this method. By default, the mean of
	/// continuous labels is predicted as accurately as possible, but the variance
	/// only reflects a heuristic measure of confidence. If calibrate is called, however,
	/// then logistic regression will be used to map from the heuristic variance estimates
	/// to the actual variance as measured in the training data, such that the
	/// predicted variance becomes more reliable. Likewise with categorical
	/// labels, the mode is predicted as accurately as possible, but the
	/// distribution of probability among the categories may not be a very good
	/// prediction of the actual distribution of probability unless this method
	/// has been called to calibrate them. If you never plan to call predictDistribution,
	/// there is no reason to ever call this method.
	void calibrate(GMatrix& features, GMatrix& labels);

	/// Evaluate pIn and compute a prediction for pOut. pOut is expected
	/// to point to an array of GPrediction objects which have already been
	/// allocated. There should be labelDims() elements in this array.
	/// The distributions will be more accurate if the model is calibrated
	/// before the first time that this method is called.
	void predictDistribution(const double* pIn, GPrediction* pOut);
#endif // MIN_PREDICT

	/// Discards all training for the purpose of freeing memory.
	/// If you call this method, you must train before making any predictions.
	/// No settings or options are discarded, so you should be able to
	/// train again without specifying any other parameters and still get
	/// a comparable model.
	virtual void clear() = 0;

#ifndef MIN_PREDICT
	/// This method assumes that this learner has already been trained.
	/// It computes the predictive accuracy for nominal labels and mean-squared error
	/// for continuous labels. pOutResults should be the size of the number of columns in labels.
	/// If pNominalLabelStats is non-NULL, then it will be filled with confusion-matrix statistics about
	/// predictions for nominal labels. pNominalLabelStats should point to an empty vector when
	/// it is passed in. It will be resized to the number of columns in labels. Elements corresponding
	/// with continuous label attributes will be set to NULL. Elements corresponding to nominal label
	/// attributes will be set to contain an n x n matrix, where n is the number of possible values
	/// in that label column. Each row refers to the expected value, each column refers to the
	/// predicted value, and each element contains the count of the number of times that each
	/// expected/predicted value occurred over the test set. The caller is responsible to delete each
	/// element in pNominalLabelStats.
	void accuracy(GMatrix& features, GMatrix& labels, double* pOutResults, std::vector<GMatrix*>* pNominalLabelStats = NULL);

	/// label specifies which output to measure. (It should be 0 if there is only one label dimension.)
	/// The measurement will be performed "nReps" times and results averaged together
	/// nPrecisionSize specifies the number of points at which the function is sampled
	/// pOutPrecision should be an array big enough to hold nPrecisionSize elements for every possible
	/// label value. (If the attribute is continuous, it should just be big enough to hold nPrecisionSize elements.)
	/// If bLocal is true, it computes the local precision instead of the global precision.
	void precisionRecall(double* pOutPrecision, size_t nPrecisionSize, GMatrix& features, GMatrix& labels, size_t label, size_t nReps);

	/// Trains and tests this learner
	virtual void trainAndTest(GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, double* pOutResults, std::vector<GMatrix*>* pNominalLabelStats = NULL);
#endif // MIN_PREDICT

#ifndef MIN_PREDICT
	/// This is a helper method used by the unit tests of several model learners
	void basicTest(double minAccuracy1, double minAccuracy2, double deviation = 1e-6, bool printAccuracy = false);

	/// Runs some unit tests related to supervised learning. Throws an exception if any problems are found.
	static void test();
#endif // MIN_PREDICT
protected:
	/// This is the implementation of the model's training algorithm. (This method is called by train).
	virtual void trainInner(GMatrix& features, GMatrix& labels) = 0;

	/// This is the implementation of the model's prediction algorithm. (This method is called by predict).
	virtual void predictInner(const double* pIn, double* pOut) = 0;

#ifndef MIN_PREDICT
	/// This is the implementation of the model's prediction algorithm. (This method is called by predictDistribution).
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut) = 0;

	/// See GTransducer::transduce
	virtual GMatrix* transduceInner(GMatrix& features1, GMatrix& labels1, GMatrix& features2);
#endif // MIN_PREDICT

	/// This method determines which data filters (normalize, discretize,
	/// and/or nominal-to-cat) are needed and trains them.
	void setupFilters(GMatrix& features, GMatrix& labels);
#ifndef MIN_PREDICT

	/// This is a helper method used by precisionRecall.
	size_t precisionRecallNominal(GPrediction* pOutput, double* pFunc, GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, size_t label, int value);

	/// This is a helper method used by precisionRecall.
	size_t precisionRecallContinuous(GPrediction* pOutput, double* pFunc, GMatrix& trainFeatures, GMatrix& trainLabels, GMatrix& testFeatures, GMatrix& testLabels, size_t label);

	/// Child classes should use this in their implementation of serialize
	GDomNode* baseDomNode(GDom* pDoc, const char* szClassName) const;
#endif // MIN_PREDICT
};

///\brief Converts a GSupervisedLearner to a string
std::string to_str(const GSupervisedLearner& learner);





/// This is the base class of supervised learning algorithms that
/// can learn one row at a time.
class GIncrementalLearner : public GSupervisedLearner
{
public:
	/// General-purpose constructor
	GIncrementalLearner(GRand& rand)
	: GSupervisedLearner(rand)
	{
	}

	/// Deserialization constructor
	GIncrementalLearner(GDomNode* pNode, GLearnerLoader& ll)
	: GSupervisedLearner(pNode, ll)
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
	/// Unlike "train", this method does not automatically set up any filters (even
	/// if you have automatic filter setup enabled). Rather,
	/// it assumes that you have already set up any filters that you wish to use.
	/// Behavior is undefined if you change the filters after this method is called.
	void beginIncrementalLearning(sp_relation& pFeatureRel, sp_relation& pLabelRel);

	/// Pass a single input row and the corresponding label to
	/// incrementally train this model
	void trainIncremental(const double* pIn, const double* pOut);

#ifndef MIN_PREDICT
	/// Train using a sparse feature matrix. (A Typical implementation of this
	/// method will first call beginIncrementalLearning, then it will
	/// iterate over all of the feature rows, and for each row it
	/// will convert the sparse row to a dense row, call trainIncremental
	/// using the dense row, then discard the dense row and proceed to the next row.)
	virtual void trainSparse(GSparseMatrix& features, GMatrix& labels) = 0;
#endif // MIN_PREDICT

protected:
	/// Prepare the model for incremental learning.
	virtual void beginIncrementalLearningInner(sp_relation& pFeatureRel, sp_relation& pLabelRel) = 0;

	/// Refine the model with the specified pattern.
	virtual void trainIncrementalInner(const double* pIn, const double* pOut) = 0;
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
	GRand& m_rand;

public:
	/// Constructor. If throwIfClassNotFound is true, then all of the methods in this
	/// class will throw an exception of the DOM refers to an unrecognized class.
	/// If throwIfClassNotFound is false, then NULL will be returned if the class
	/// is not recognized.
	GLearnerLoader(GRand& rand, bool throwIfClassNotFound = true)
	: m_throwIfClassNotFound(throwIfClassNotFound), m_rand(rand)
	{
	}

	virtual ~GLearnerLoader() {}

	/// Loads an incremental transform (or a two-way incremental transform) from a DOM.
	virtual GIncrementalTransform* loadIncrementalTransform(GDomNode* pNode);

	/// Loads a supervised learning algorithm (or an incremental learner) from a DOM.
	virtual GSupervisedLearner* loadSupervisedLearner(GDomNode* pNode);

	/// Loads an incremental learner from a DOM.
	virtual GIncrementalLearner* loadIncrementalLearner(GDomNode* pNode);

#ifndef MIN_PREDICT
	/// Loads a collaborative filtering algorithm from a DOM.
	virtual GCollaborativeFilter* loadCollaborativeFilter(GDomNode* pNode);
#endif // MIN_PREDICT

	/// Returns the random number generator associated with this object.
	GRand& rand() { return m_rand; }
};



/// Always outputs the label mean (for continuous labels) and the most common
/// class (for nominal labels).
class GBaselineLearner : public GSupervisedLearner
{
protected:
	std::vector<double> m_prediction;

public:
	/// General-purpose constructor
	GBaselineLearner(GRand& rand);

	/// Deserialization constructor
	GBaselineLearner(GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GBaselineLearner();

#ifndef MIN_PREDICT
	static void test();
#endif // MIN_PREDICT

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

	/// This model has no parameters to tune, so this method is a noop.
	void autoTune(GMatrix& features, GMatrix& labels);

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);
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
	GIdentityFunction(GRand& rand);

	/// Deserialization constructor
	GIdentityFunction(GDomNode* pNode, GLearnerLoader& ll);

	/// Destructor
	virtual ~GIdentityFunction();

	/// Marshal this object into a DOM, which can then be converted to a variety of serial formats.
	virtual GDomNode* serialize(GDom* pDoc) const;

	/// See the comment for GSupervisedLearner::clear
	virtual void clear();

protected:
	/// See the comment for GSupervisedLearner::trainInner
	virtual void trainInner(GMatrix& features, GMatrix& labels);

	/// See the comment for GSupervisedLearner::predictInner
	virtual void predictInner(const double* pIn, double* pOut);

	/// See the comment for GSupervisedLearner::predictDistributionInner
	virtual void predictDistributionInner(const double* pIn, GPrediction* pOut);
};


} // namespace GClasses

#endif // __GLEARNER_H__

