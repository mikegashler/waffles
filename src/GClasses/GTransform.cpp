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

#include "GTransform.h"
#include "GDom.h"
#include "GVec.h"
#include "GDistribution.h"
#include "GRand.h"
#include "GManifold.h"
#include "GCluster.h"
#include "GString.h"
#include "GNeuralNet.h"
#include "GRecommender.h"
#include "GHolders.h"
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string>
#include <cmath>
#include <memory>

namespace GClasses {

using std::string;
using std::vector;
using std::ostringstream;

GTransform::GTransform()
{
}

GTransform::GTransform(const GDomNode* pNode)
{
}

// virtual
GTransform::~GTransform()
{
}

// virtual
GDomNode* GTransform::baseDomNode(GDom* pDoc, const char* szClassName) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "class", szClassName);
	return pNode;
}

// ---------------------------------------------------------------

GIncrementalTransform::GIncrementalTransform(const GDomNode* pNode)
: GTransform(pNode)
{
	m_pRelationBefore = GRelation::deserialize(pNode->get("before"));
	m_pRelationAfter = GRelation::deserialize(pNode->get("after"));
}

// virtual
GIncrementalTransform::~GIncrementalTransform()
{
	delete(m_pRelationBefore);
	delete(m_pRelationAfter);
}

// virtual
GDomNode* GIncrementalTransform::baseDomNode(GDom* pDoc, const char* szClassName) const
{
	if(!m_pRelationAfter)
		throw Ex("train must be called before serialize");
	GDomNode* pNode = GTransform::baseDomNode(pDoc, szClassName);
	pNode->add(pDoc, "before", m_pRelationBefore->serialize(pDoc));
	pNode->add(pDoc, "after", m_pRelationAfter->serialize(pDoc));
	return pNode;
}

void GIncrementalTransform::setBefore(GRelation* pRel)
{
	delete(m_pRelationBefore);
	m_pRelationBefore = pRel;
}

void GIncrementalTransform::setAfter(GRelation* pRel)
{
	delete(m_pRelationAfter);
	m_pRelationAfter = pRel;
}

void GIncrementalTransform::train(const GMatrix& data)
{
	setBefore(data.relation().clone());
	setAfter(trainInner(data));
}

void GIncrementalTransform::train(const GRelation& relation)
{
	setBefore(relation.clone());
	setAfter(trainInner(relation));
}

// virtual
GMatrix* GIncrementalTransform::reduce(const GMatrix& in)
{
	train(in);
	return transformBatch(in);
}

// virtual
GMatrix* GIncrementalTransform::transformBatch(const GMatrix& in)
{
	if(!m_pRelationAfter)
		throw Ex("train has not been called");
	size_t nRows = in.rows();
	GMatrix* pOut = new GMatrix(m_pRelationAfter->clone());
	std::unique_ptr<GMatrix> hOut(pOut);
	pOut->newRows(nRows);
	for(size_t i = 0; i < nRows; i++)
		transform(in.row(i), pOut->row(i));
	return hOut.release();
}

GVec& GIncrementalTransform::innerBuf()
{
	if(!m_pRelationAfter)
		throw Ex("Attempted to use a transform that was not trained");
	m_innerBuf.resize(m_pRelationAfter->size());
	return m_innerBuf;
}

// virtual
std::unique_ptr<GMatrix> GIncrementalTransform::untransformBatch(const GMatrix& in)
{
	if(!m_pRelationBefore)
		throw Ex("train has not been called");
	size_t nRows = in.rows();
	auto pOut = std::unique_ptr<GMatrix>(new GMatrix(before().clone()));
	pOut->newRows(nRows);
	for(size_t i = 0; i < nRows; i++)
		untransform(in.row(i), pOut->row(i));
	return pOut;
}

//static
void GIncrementalTransform::test()
{
	// Make an input matrix
	vector<size_t> valCounts;
	valCounts.push_back(0);
	valCounts.push_back(1);
	valCounts.push_back(2);
	valCounts.push_back(3);
	valCounts.push_back(0);
	GMatrix m(valCounts);
	m.newRows(2);
	m[0][0] = 2.4; m[0][1] = 0; m[0][2] = 0; m[0][3] = 2; m[0][4] = 8.2;
	m[1][0] = 0.0; m[1][1] = 0; m[1][2] = 1; m[1][3] = 0; m[1][4] = 2.2;

	// Make an expected output matrix
	GMatrix e(2, 7);
	e[0][0] = 1; e[0][1] = 0; e[0][2] = 0; e[0][3] = 0; e[0][4] = 0; e[0][5] = 1; e[0][6] = 1;
	e[1][0] = 0; e[1][1] = 0; e[1][2] = 1; e[1][3] = 1; e[1][4] = 0; e[1][5] = 0; e[1][6] = 0;

	// Transform the input matrix and check it
	GIncrementalTransformChainer trans(new GNormalize(), new GNominalToCat());
	trans.train(m);
	GMatrix* pA = trans.transformBatch(m);
	std::unique_ptr<GMatrix> hA(pA);
	if(pA->sumSquaredDifference(e) > 1e-12)
		throw Ex("Expected:\n", to_str(e), "\nGot:\n", to_str(*pA));
	if(!pA->relation().areContinuous())
		throw Ex("failed");
	auto pB = trans.untransformBatch(*pA);
	if(pB->sumSquaredDifference(m) > 1e-12)
		throw Ex("Expected:\n", to_str(m), "\nGot:\n", to_str(*pB));
	if(!pB->relation().isCompatible(m.relation()) || !m.relation().isCompatible(pB->relation()))
		throw Ex("failed");

	// Round-trip it through serialization
	GDom doc;
	GDomNode* pNode = trans.serialize(&doc);
	GRand rand(0);
	GLearnerLoader ll;
	GIncrementalTransform* pTrans = ll.loadIncrementalTransform(pNode);
	std::unique_ptr<GIncrementalTransform> hTrans(pTrans);

	// Transform the input matrix again, and check it
	GMatrix* pC = pTrans->transformBatch(m);
	std::unique_ptr<GMatrix> hC(pC);
	if(pC->sumSquaredDifference(e) > 1e-12)
		throw Ex("Expected:\n", to_str(e), "\nGot:\n", to_str(*pC));
	if(!pC->relation().areContinuous())
		throw Ex("failed");
	auto pD = trans.untransformBatch(*pC);
	if(pD->sumSquaredDifference(m) > 1e-12)
		throw Ex("Expected:\n", to_str(m), "\nGot:\n", to_str(*pD));
	if(!pD->relation().isCompatible(m.relation()) || !m.relation().isCompatible(pD->relation()))
		throw Ex("failed");
}






GDataPreprocessor::GDataPreprocessor(const GMatrix& source, size_t rowStart, size_t colStart, size_t rowCount, size_t colCount, bool allowMissing, bool allowNominal, bool allowContinuous, double minVal, double maxVal)
{
	if(rowStart > 0 || colStart > 0 || rowCount < source.rows() || colCount < source.cols())
	{
		GMatrix* pTemp = new GMatrix(source, rowStart, colStart, rowCount, colCount);
		m_pTransform = autoTrans(*pTemp, allowMissing, allowNominal, allowContinuous, minVal, maxVal);
		if(m_pTransform)
		{
			m_processedData.push_back(m_pTransform->transformBatch(*pTemp));
			delete(pTemp);
		}
		else
			m_processedData.push_back(pTemp);
	}
	else
	{
		m_pTransform = autoTrans(source, allowMissing, allowNominal, allowContinuous, minVal, maxVal);
		if(m_pTransform)
			m_processedData.push_back(m_pTransform->transformBatch(source));
		else
			m_processedData.push_back(new GMatrix(source));
	}
}

void GDataPreprocessor::add(const GMatrix& source, size_t rowStart, size_t colStart, size_t rowCount, size_t colCount)
{
	if(rowStart > 0 || colStart > 0 || rowCount < source.rows() || colCount < source.cols())
	{
		GMatrix* pTemp = new GMatrix(source, rowStart, colStart, rowCount, colCount);
		if(m_pTransform)
		{
			m_processedData.push_back(m_pTransform->transformBatch(*pTemp));
			delete(pTemp);
		}
		else
			m_processedData.push_back(pTemp);
	}
	else
	{
		if(m_pTransform)
			m_processedData.push_back(m_pTransform->transformBatch(source));
		else
			m_processedData.push_back(new GMatrix(source));
	}
}

GIncrementalTransform* GDataPreprocessor::autoTrans(const GMatrix& data, bool allowMissing, bool allowNominal, bool allowContinuous, double minVal, double maxVal)
{
	// Determine what types are present in the data
	bool hasNominal = false;
	bool hasContinuous = false;
	const GRelation& rel = data.relation();
	for(size_t i = 0; i < rel.size(); i++)
	{
		if(rel.valueCount(i) == 0)
		{
			hasContinuous = true;
			if(hasNominal)
				break;
		}
		else
		{
			hasNominal = true;
			if(hasContinuous)
				break;
		}
	}

	GIncrementalTransform* pTrans = nullptr;

	// Impute features if necessary
	if(!allowMissing && data.doesHaveAnyMissingValues())
		pTrans = GIncrementalTransformChainer::chain(new GImputeMissingVals(), pTrans);

	// Normalize labels if necessary
	if(allowContinuous)
	{
		// Find the min and max
		double dataMin = 1e300;
		double dataMax = -1e300;
		for(size_t i = 0; i < data.cols(); i++)
		{
			if(data.relation().valueCount(i) != 0)
				continue;
			dataMin = std::min(data.columnMin(i), dataMin);
			dataMax = std::max(data.columnMax(i), dataMax);
		}
		if(hasNominal && !allowNominal)
		{
			minVal = std::min(0.0, minVal);
			maxVal = std::max(1.0, maxVal);
		}
		bool normalizationIsNeeded = false;
		if(dataMin < minVal || dataMax > maxVal)
			normalizationIsNeeded = true;
		else if((dataMax - dataMin) >= 1e-12 && (dataMax - dataMin) * 4 < maxVal - minVal)
			normalizationIsNeeded = true;
		if(normalizationIsNeeded)
			pTrans = GIncrementalTransformChainer::chain(new GNormalize(minVal, maxVal), pTrans);
	}

	// Nomcat if necessary
	if(!allowNominal && hasNominal)
	{
		if(!allowContinuous)
			throw Ex("Either nominal or continuous values must be allowed");
		pTrans = GIncrementalTransformChainer::chain(new GNominalToCat(16), pTrans);
	}

	// Discretize if necessary
	if(!allowContinuous && hasContinuous)
	{
		if(!allowNominal)
			throw Ex("Either nominal or continuous values must be allowed");
		pTrans = GIncrementalTransformChainer::chain(new GDiscretize(), pTrans);
	}
	
	// Train
	if(pTrans)
		pTrans->train(data);

	return pTrans;
}











GIncrementalTransformChainer::GIncrementalTransformChainer(GIncrementalTransform* pFirst, GIncrementalTransform* pSecond)
: GIncrementalTransform(), m_pFirst(pFirst), m_pSecond(pSecond)
{
}

GIncrementalTransformChainer::GIncrementalTransformChainer(const GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode)
{
	m_pFirst = ll.loadIncrementalTransform(pNode->get("first"));
	m_pSecond = ll.loadIncrementalTransform(pNode->get("second"));
}

// virtual
GIncrementalTransformChainer::~GIncrementalTransformChainer()
{
	delete(m_pFirst);
	delete(m_pSecond);
}

// static
GIncrementalTransform* GIncrementalTransformChainer::chain(GIncrementalTransform* pA, GIncrementalTransform* pB)
{
	if(pA == nullptr)
		return pB;
	else if(pB == nullptr)
		return pA;
	else
		return new GIncrementalTransformChainer(pA, pB);
}

// virtual
GDomNode* GIncrementalTransformChainer::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GIncrementalTransformChainer");
	pNode->add(pDoc, "first", m_pFirst->serialize(pDoc));
	pNode->add(pDoc, "second", m_pSecond->serialize(pDoc));
	return pNode;
}

// virtual
GRelation* GIncrementalTransformChainer::trainInner(const GMatrix& data)
{
	m_pFirst->train(data);
	GMatrix* pData2 = m_pFirst->transformBatch(data); // todo: often this step is computation overkill since m_pSecond may not even use it during training. Is there a way to avoid doing it?
	std::unique_ptr<GMatrix> hData2(pData2);
	m_pSecond->train(*pData2);
	return m_pSecond->after().clone();
}

// virtual
GRelation* GIncrementalTransformChainer::trainInner(const GRelation& relation)
{
	m_pFirst->train(relation);
	m_pSecond->train(m_pFirst->after());
	return m_pSecond->after().clone();
}

// virtual
void GIncrementalTransformChainer::transform(const GVec& in, GVec& out)
{
	GVec& buf = m_pFirst->innerBuf();
	m_pFirst->transform(in, buf);
	m_pSecond->transform(buf, out);
}

// virtual
void GIncrementalTransformChainer::untransform(const GVec& in, GVec& out)
{
	GVec& buf = m_pFirst->innerBuf();
	m_pSecond->untransform(in, buf);
	m_pFirst->untransform(buf, out);
}

// virtual
void GIncrementalTransformChainer::untransformToDistribution(const GVec& in, GPrediction* out)
{
	GVec& buf = m_pFirst->innerBuf();
	m_pSecond->untransform(in, buf);
	m_pFirst->untransformToDistribution(buf, out);
}

// ---------------------------------------------------------------

GPCA::GPCA(size_t target_Dims)
: GIncrementalTransform(), m_targetDims(target_Dims), m_pBasisVectors(NULL), m_aboutOrigin(false), m_rand(0)
{
}

GPCA::GPCA(const GDomNode* pNode)
: GIncrementalTransform(pNode), m_rand(0)
{
	m_pBasisVectors = new GMatrix(pNode->get("basis"));
	m_targetDims = m_pBasisVectors->rows();
	m_aboutOrigin = pNode->getBool("aboutOrigin");
}

// virtual
GPCA::~GPCA()
{
	delete(m_pBasisVectors);
}

// virtual
GDomNode* GPCA::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GPCA");
	pNode->add(pDoc, "basis", m_pBasisVectors->serialize(pDoc));
	pNode->add(pDoc, "aboutOrigin", m_aboutOrigin);
	return pNode;
}

void GPCA::computeEigVals()
{
	m_eigVals.resize(m_targetDims);
}

// virtual
GRelation* GPCA::trainInner(const GMatrix& data)
{
	if(!before().areContinuous())
		throw Ex("GPCA doesn't support nominal values. (You could filter with nominaltocat to make them real.)");
	delete(m_pBasisVectors);
	m_pBasisVectors = new GMatrix(1 + m_targetDims, before().size());

	// Compute the mean
	GVec& mean = m_pBasisVectors->row(0);
	if(m_aboutOrigin)
		mean.fill(0.0);
	else
		data.centroid(mean);

	// Make a copy of the data
	GMatrix tmpData(data.relation().cloneMinimal());
	tmpData.copy(data);

	// Compute the principle components
	double sse = 0;
	if(m_eigVals.size() > 0)
		sse = tmpData.sumSquaredDistance(mean);
	for(size_t i = 0; i < m_targetDims; i++)
	{
		GVec& vec = m_pBasisVectors->row(i + 1);
		tmpData.principalComponentIgnoreUnknowns(vec, mean, &m_rand);
		tmpData.removeComponent(mean, vec);
		if(m_eigVals.size() > 0)
		{
			double t = tmpData.sumSquaredDistance(mean);
			m_eigVals[i] = (sse - t) / (data.rows() - 1);
			sse = t;
		}
	}

	return new GUniformRelation(m_targetDims, 0);
}

// virtual
GRelation* GPCA::trainInner(const GRelation& relation)
{
	throw Ex("This transform cannot be trained without data");
	return before().clone();
}

// virtual
void GPCA::transform(const GVec& in, GVec& out)
{
	GVec& c = m_pBasisVectors->row(0);
	for(size_t i = 0; i < m_targetDims; i++)
	{
		GVec& basisVector = m_pBasisVectors->row(1 + i);
		out[i] = basisVector.dotProductIgnoringUnknowns(c, in);
	}
}

// virtual
void GPCA::untransform(const GVec& in, GVec& out)
{
	throw Ex("Sorry, not implemented yet");
/* todo: need invert the basis matrix
	out.copy(m_pBasisVectors->row(0));
	for(size_t i = 0; i < m_targetDims; i++)
		out.addScaled(in[i], m_pBasisVectors->row(1 + i));
*/
}

// virtual
void GPCA::untransformToDistribution(const GVec& in, GPrediction* out)
{
	throw Ex("Sorry, PCA cannot untransform to a distribution");
}

// --------------------------------------------------------------------------

GNoiseGenerator::GNoiseGenerator()
: GIncrementalTransform(), m_rand(0), m_mean(0), m_deviation(1)
{
}

GNoiseGenerator::GNoiseGenerator(const GDomNode* pNode)
: GIncrementalTransform(pNode), m_rand(0)
{
	m_mean = pNode->getDouble("mean");
	m_deviation = pNode->getDouble("dev");
}

GNoiseGenerator::~GNoiseGenerator()
{
}

// virtual
GDomNode* GNoiseGenerator::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNoiseGenerator");
	pNode->add(pDoc, "mean", m_mean);
	pNode->add(pDoc, "dev", m_deviation);
	return pNode;
}

// virtual
GRelation* GNoiseGenerator::trainInner(const GMatrix& data)
{
	return data.relation().clone();
}

// virtual
GRelation* GNoiseGenerator::trainInner(const GRelation& relation)
{
	return relation.clone();
}

// virtual
void GNoiseGenerator::transform(const GVec& in, GVec& out)
{
	size_t nDims = before().size();
	for(size_t i = 0; i < nDims; i++)
	{
		size_t vals = before().valueCount(i);
		if(vals == 0)
			out[i] = m_rand.normal() * m_deviation + m_mean;
		else
			out[i] = (double)m_rand.next(vals);
	}
}

// --------------------------------------------------------------------------

GPairProduct::GPairProduct(size_t nMaxDims)
: GIncrementalTransform(), m_maxDims(nMaxDims)
{
}

GPairProduct::GPairProduct(const GDomNode* pNode)
: GIncrementalTransform(pNode)
{
	m_maxDims = (size_t)pNode->getInt("maxDims");
}

GPairProduct::~GPairProduct()
{
}

// virtual
GDomNode* GPairProduct::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GPairProduct");
	pNode->add(pDoc, "maxDims", m_maxDims);
	return pNode;
}

// virtual
GRelation* GPairProduct::trainInner(const GMatrix& data)
{
	size_t nAttrsIn = before().size();
	size_t nAttrsOut = std::min(m_maxDims, nAttrsIn * (nAttrsIn + 1) / 2);
	return new GUniformRelation(nAttrsOut, 0);
}

// virtual
GRelation* GPairProduct::trainInner(const GRelation& relation)
{
	size_t nAttrsIn = before().size();
	size_t nAttrsOut = std::min(m_maxDims, nAttrsIn * (nAttrsIn + 1) / 2);
	return new GUniformRelation(nAttrsOut, 0);
}

// virtual
void GPairProduct::transform(const GVec& in, GVec& out)
{
	size_t i, j, nAttr;
	size_t nAttrsIn = before().size();
	size_t nAttrsOut = after().size();
	nAttr = 0;
	for(j = 0; j < nAttrsIn && nAttr < nAttrsOut; j++)
	{
		for(i = j; i < nAttrsIn && nAttr < nAttrsOut; i++)
			out[nAttr++] = in[i] * in[j];
	}
	GAssert(nAttr == nAttrsOut);
}

// --------------------------------------------------------------------------

GDataAugmenter::GDataAugmenter(GIncrementalTransform* pTransform)
: GIncrementalTransform(), m_pTransform(pTransform)
{
}

GDataAugmenter::GDataAugmenter(const GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode)
{
	m_pTransform = ll.loadIncrementalTransform(pNode->get("trans"));
}

// virtual
GDataAugmenter::~GDataAugmenter()
{
	delete(m_pTransform);
}

// virtual
GDomNode* GDataAugmenter::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GDataAugmenter");
	pNode->add(pDoc, "trans", m_pTransform->serialize(pDoc));
	return pNode;
}

// virtual
GRelation* GDataAugmenter::trainInner(const GMatrix& data)
{
	m_pTransform->train(data);
	GMixedRelation* pNewRel = new GMixedRelation();
	pNewRel->addAttrs(data.relation());
	pNewRel->addAttrs(m_pTransform->after());
	return pNewRel;
}

// virtual
GRelation* GDataAugmenter::trainInner(const GRelation& relation)
{
	m_pTransform->train(relation);
	GMixedRelation* pNewRel = new GMixedRelation();
	pNewRel->addAttrs(before());
	pNewRel->addAttrs(m_pTransform->after());
	return pNewRel;
}

// virtual
void GDataAugmenter::transform(const GVec& in, GVec& out)
{
	memcpy(out.data(), in.data(), before().size() * sizeof(double));
	m_pTransform->transform(in, m_pTransform->innerBuf());
	out.copy(before().size(), m_pTransform->innerBuf());
}

// virtual
void GDataAugmenter::untransform(const GVec& in, GVec& out)
{
	memcpy(out.data(), in.data(), before().size() * sizeof(double));
}

// virtual
void GDataAugmenter::untransformToDistribution(const GVec& in, GPrediction* out)
{
	throw Ex("Sorry, this method is not implemented yet");
}

// --------------------------------------------------------------------------

GAttributeSelector::GAttributeSelector(const GDomNode* pNode)
: GIncrementalTransform(pNode), m_seed(1234567)
{
	m_labelDims = (size_t)pNode->getInt("labels");
	m_targetFeatures = (size_t)pNode->getInt("target");
	GDomNode* pRanksNode = pNode->get("ranks");
	GDomListIterator it(pRanksNode);
	m_ranks.reserve(it.remaining());
	for( ; it.current(); it.advance())
		m_ranks.push_back((size_t)it.currentInt());
	if(m_ranks.size() + (size_t)m_labelDims != (size_t)before().size())
		throw Ex("invalid attribute selector");
	if(m_targetFeatures > m_ranks.size())
		throw Ex("invalid attribute selector");
}

// virtual
GDomNode* GAttributeSelector::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GAttributeSelector");
	pNode->add(pDoc, "labels", m_labelDims);
	pNode->add(pDoc, "target", m_targetFeatures);
	GDomNode* pRanksNode = pNode->add(pDoc, "ranks", pDoc->newList());
	for(size_t i = 0; i < m_ranks.size(); i++)
		pRanksNode->add(pDoc, m_ranks[i]);
	return pNode;
}

GRelation* GAttributeSelector::setTargetFeatures(size_t n)
{
	if(n > before().size())
		throw Ex("out of range");
	GMixedRelation* pRelAfter;
	if(before().type() == GRelation::ARFF)
		pRelAfter = new GArffRelation();
	else
		pRelAfter = new GMixedRelation();
	for(size_t i = 0; i < m_targetFeatures; i++)
		pRelAfter->copyAttr(&before(), m_ranks[i]);
	if(m_labelDims > before().size())
		throw Ex("label dims out of range");
	size_t featureDims = before().size() - m_labelDims;
	for(size_t i = 0; i < m_labelDims; i++)
		pRelAfter->copyAttr(&before(), featureDims + i);
	return pRelAfter;
}

// virtual
GRelation* GAttributeSelector::trainInner(const GMatrix& data)
{
	// Normalize all the data
	if(m_labelDims > data.cols())
		throw Ex("label dims is greater than the number of columns in the data");
	GNormalize norm;
	norm.train(data);
	GMatrix* pNormData = norm.transformBatch(data);
	std::unique_ptr<GMatrix> hNormData(pNormData);

	// Divide into features and labels
	size_t curDims = data.cols() - m_labelDims;
	m_ranks.resize(curDims);
	GMatrix* pFeatures = new GMatrix(*pNormData, 0, 0, data.rows(), data.cols() - m_labelDims);
	std::unique_ptr<GMatrix> hFeatures(pFeatures);
	GMatrix* pLabels = new GMatrix(*pNormData, 0, data.cols() - m_labelDims, data.rows(), m_labelDims);
	std::unique_ptr<GMatrix> hLabels(pLabels);
	vector<size_t> indexMap;
	for(size_t i = 0; i < curDims; i++)
		indexMap.push_back(i);

	// Produce a ranked attributed ordering by deselecting the weakest attribute each time
	GRand rand(m_seed);
	while(curDims > 1)
	{
		// Convert nominal attributes to a categorical distribution
		GNominalToCat ntc;
		ntc.train(*pFeatures);
		GMatrix* pFeaturesReal = ntc.transformBatch(*pFeatures);
		std::unique_ptr<GMatrix> hFeaturesReal(pFeaturesReal);
		vector<size_t> rmap;
		ntc.reverseAttrMap(rmap);
		GNominalToCat ntc2;
		ntc2.train(*pLabels);
		GMatrix* pLabelsReal = ntc2.transformBatch(*pLabels);
		std::unique_ptr<GMatrix> hLabelsReal(pLabelsReal);

		// Train a single-layer neural network with the normalized remaining data
		GNeuralNet nn;
		GBlockLinear* pB = new GBlockLinear(pFeaturesReal->cols(), pLabelsReal->cols());
		nn.add(pB);
		nn.add(new GBlockTanh(pLabelsReal->cols()));
		GSGDOptimizer optimizer(nn, rand);
		optimizer.setWindowSize(30);
		optimizer.setImprovementThresh(0.002);
		optimizer.optimizeWithValidation(*pFeaturesReal, *pLabelsReal);

		// Identify the weakest attribute
		size_t pos = 0;
		double weakest = 1e308;
		size_t weakestIndex = 0;
		GVec& weights = optimizer.model().weights;
		for(size_t i = 0; i < curDims; i++)
		{
			double w = 0;
			while(pos < pFeaturesReal->cols() && rmap[pos] == i)
			{
				for(size_t neuron = 0; neuron < pB->outputs(); neuron++)
					w = std::max(w, std::abs(weights[pLabelsReal->cols() + pLabelsReal->cols() * pos + neuron]));
				pos++;
			}
			if(w < weakest)
			{
				weakest = w;
				weakestIndex = i;
			}
		}

		// Deselect the weakest attribute
		m_ranks[curDims - 1] = indexMap[weakestIndex];
		indexMap.erase(indexMap.begin() + weakestIndex);
		pFeatures->deleteColumns(weakestIndex, 1);
		curDims--;
		GAssert(pFeatures->cols() == curDims);
	}
	m_ranks[0] = indexMap[0];
	return setTargetFeatures(m_targetFeatures);
}

// virtual
GRelation* GAttributeSelector::trainInner(const GRelation& relation)
{
	throw Ex("This transform cannot be trained without data");
	return before().clone();
}

// virtual
void GAttributeSelector::transform(const GVec& in, GVec& out)
{
	size_t i;
	for(i = 0; i < m_targetFeatures; i++)
		out[i] = in[m_ranks[i]];
	size_t featureDims = before().size() - m_labelDims;
	for(size_t j = 0; j < m_labelDims; j++)
		out[i++] = in[featureDims + j];
}

//static
void GAttributeSelector::test()
{
	GRand prng(0);
	GMatrix data(0, 21);
	for(size_t i = 0; i < 256; i++)
	{
		GVec& vec = data.newRow();
		vec.fillUniform(prng);
		vec[20] = 0.2 * vec[3] * vec[3] * - 7.0 * vec[3] * vec[13] + vec[17];
	}
	GAttributeSelector as(1, 3);
	as.train(data);
	std::vector<size_t>& r = as.ranks();
	if(r[1] == r[0] || r[2] == r[0] || r[2] == r[1])
		throw Ex("bogus rankings");
	if(r[0] != 3 && r[0] != 13 && r[0] != 17)
		throw Ex("failed");
	if(r[1] != 3 && r[1] != 13 && r[1] != 17)
		throw Ex("failed");
	if(r[2] != 3 && r[2] != 13 && r[2] != 17)
		throw Ex("failed");
}

// --------------------------------------------------------------------------

GNominalToCat::GNominalToCat(size_t nValueCap, double lo, double hi)
: GIncrementalTransform(), m_valueCap(nValueCap), m_preserveUnknowns(false), m_lo(lo), m_hi(hi)
{
}

GNominalToCat::GNominalToCat(const GDomNode* pNode)
: GIncrementalTransform(pNode)
{
	m_valueCap = (size_t)pNode->getInt("valueCap");
	m_preserveUnknowns = pNode->getBool("pu");
	m_lo = pNode->getDouble("lo");
	m_hi = pNode->getDouble("hi");
}

GRelation* GNominalToCat::init()
{
	size_t nDims = 0;
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues == 0)
			nDims++;
		else if(nValues < 3 || nValues >= m_valueCap)
			nDims++;
		else if(nValues < m_valueCap)
			nDims += nValues;
		else
			nDims++;
	}
	return new GUniformRelation(nDims);
}

// virtual
GNominalToCat::~GNominalToCat()
{
}

// virtual
GRelation* GNominalToCat::trainInner(const GMatrix& data)
{
	return init();
}

// virtual
GRelation* GNominalToCat::trainInner(const GRelation& relation)
{
	return init();
}

// virtual
GDomNode* GNominalToCat::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNominalToCat");
	pNode->add(pDoc, "valueCap", m_valueCap);
	pNode->add(pDoc, "pu", m_preserveUnknowns);
	pNode->add(pDoc, "lo", m_lo);
	pNode->add(pDoc, "hi", m_hi);
	return pNode;
}

// virtual
void GNominalToCat::transform(const GVec& in, GVec& out)
{
	size_t nInAttrCount = before().size();
	size_t j = 0;
	for(size_t i = 0; i < nInAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues < 3)
		{
			if(nValues == 0)
				out[j++] = in[i];
			else if(nValues == 1)
			{
				if(in[i] == UNKNOWN_DISCRETE_VALUE)
					out[j++] = UNKNOWN_REAL_VALUE;
				else
					out[j++] = m_lo;
			}
			else
			{
				if(in[i] == UNKNOWN_DISCRETE_VALUE)
				{
					if(m_preserveUnknowns)
						out[j++] = UNKNOWN_REAL_VALUE;
					else
						out[j++] = 0.5 * (m_lo + m_hi);
				}
				else
					out[j++] = in[i];
			}
		}
		else if(nValues < m_valueCap)
		{
			if(in[i] >= 0)
			{
				GAssert(in[i] < nValues);
				out.fill(m_lo, j, nValues);
				out[j + (int)in[i]] = m_hi;
			}
			else
			{
				if(m_preserveUnknowns)
					out.fill(UNKNOWN_REAL_VALUE, j, nValues);
				else
					out.fill((m_hi - m_lo) / nValues + m_lo, j, nValues);
			}
			j += nValues;
		}
		else
		{
			if(in[i] == UNKNOWN_DISCRETE_VALUE)
				out[j++] = UNKNOWN_REAL_VALUE;
			else
				out[j++] = in[i];
		}
	}
}

// virtual
void GNominalToCat::untransform(const GVec& in, GVec& out)
{
	size_t nOutAttrCount = before().size();
	size_t j = 0;
	for(size_t i = 0; i < nOutAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues < 3)
		{
			if(nValues == 0)
				out[i] = in[j++];
			else if(nValues == 1)
			{
				if(in[j++] == UNKNOWN_REAL_VALUE)
					out[i] = UNKNOWN_DISCRETE_VALUE;
				else
					out[i] = 0;
			}
			else
			{
				if(in[j] == UNKNOWN_REAL_VALUE)
					out[i] = UNKNOWN_DISCRETE_VALUE;
				else
					out[i] = (in[j] + in[j] < m_lo + m_hi ? 0 : 1);
				j++;
			}
		}
		else if(nValues < m_valueCap)
		{
			double max = in[j++];
			out[i] = 0.0;
			for(size_t k = 1; k < nValues; k++)
			{
				if(in[j] > max)
				{
					max = in[j];
					out[i] = (double)k;
				}
				j++;
			}
		}
		else
		{
			if(in[j] == UNKNOWN_REAL_VALUE)
				out[i] = UNKNOWN_DISCRETE_VALUE;
			else
				out[i] = std::max(0.0, std::min(double(nValues - 1), floor(in[j] + 0.5)));
			j++;
		}
	}
}

// virtual
void GNominalToCat::untransformToDistribution(const GVec& in, GPrediction* out)
{
	size_t nOutAttrCount = before().size();
	size_t j = 0;
	for(size_t i = 0; i < nOutAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues < 3)
		{
			if(nValues == 0)
			{
				GNormalDistribution* pNorm = out->makeNormal();
				pNorm->setMeanAndVariance(in[j], 1.0); // todo: should we throw an exception here since we have no way to estimate the variance?
			}
			else if(nValues == 1)
			{
				GCategoricalDistribution* pCat = out->makeCategorical();
				pCat->setToUniform(1);
			}
			else
			{
				GCategoricalDistribution* pCat = out->makeCategorical();
				if(in[j] == UNKNOWN_REAL_VALUE)
					pCat->setToUniform(2);
				else
				{
					GVec& vals = pCat->values(2);
					double t = (in[j] - m_lo) / (m_hi - m_lo);
					vals[0] = 1.0 - t;
					vals[1] = t;
					pCat->normalize(); // We have to normalize to ensure the values are properly clipped.
				}
			}
			j++;
			out++;
		}
		else if(nValues < m_valueCap)
		{
			GCategoricalDistribution* pCat = out->makeCategorical();
			pCat->setValues(nValues, in.data() + j);
			j += nValues;
			out++;
		}
		else
		{
			GCategoricalDistribution* pCat = out->makeCategorical();
			pCat->setSpike(nValues, std::max(size_t(0), std::min(nValues - 1, size_t(floor(in[j] + 0.5)))), 3);
			j++;
			out++;
		}
	}
}

void GNominalToCat::reverseAttrMap(vector<size_t>& rmap)
{
	rmap.clear();
	size_t nInAttrCount = before().size();
	for(size_t i = 0; i < nInAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues < 3 || nValues >= m_valueCap)
			rmap.push_back(i);
		else
		{
			for(size_t j = 0; j < nValues; j++)
				rmap.push_back(i);
		}
	}
}

// --------------------------------------------------------------------------

GNormalize::GNormalize(double min, double max)
: GIncrementalTransform(), m_min(min), m_max(max)
{
}

GNormalize::GNormalize(const GDomNode* pNode)
: GIncrementalTransform(pNode)
{
	m_min = pNode->getDouble("min");
	m_max = pNode->getDouble("max");
	m_mins.deserialize(pNode->get("mins"));
	m_ranges.deserialize(pNode->get("ranges"));
}

// virtual
GNormalize::~GNormalize()
{
}

// virtual
GDomNode* GNormalize::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNormalize");
	pNode->add(pDoc, "min", m_min);
	pNode->add(pDoc, "max", m_max);
	pNode->add(pDoc, "mins", m_mins.serialize(pDoc));
	pNode->add(pDoc, "ranges", m_ranges.serialize(pDoc));
	return pNode;
}

void GNormalize::setMinsAndRanges(const GRelation& rel, const GVec& mins, const GVec& ranges)
{
	setBefore(rel.clone());
	setAfter(rel.clone());
	m_mins.copy(mins);
	m_ranges.copy(ranges);
}

// virtual
GRelation* GNormalize::trainInner(const GMatrix& data)
{
	size_t nAttrCount = before().size();
	m_mins.resize(nAttrCount);
	m_ranges.resize(nAttrCount);
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			m_mins[i] = data.columnMin(i);
			if(m_mins[i] >= 1e300)
			{
				m_mins[i] = 0.0;
				m_ranges[i] = 1.0;
			}
			else
			{
				m_ranges[i] = data.columnMax(i) - m_mins[i];
				if(m_ranges[i] < 1e-12)
					m_ranges[i] = 1.0;
			}
		}
		else
		{
			m_mins[i] = 0;
			m_ranges[i] = 0;
		}
	}
	return data.relation().clone();
}

// virtual
GRelation* GNormalize::trainInner(const GRelation& relation)
{
	throw Ex("This transform cannot be trained without data");
	return before().clone();
}

// virtual
void GNormalize::transform(const GVec& in, GVec& out)
{
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(in[i] == UNKNOWN_REAL_VALUE)
				out[i] = UNKNOWN_REAL_VALUE;
			else
				out[i] = GMatrix::normalizeValue(in[i], m_mins[i], m_mins[i] + m_ranges[i], m_min, m_max);
		}
		else
			out[i] = in[i];
	}
}

// virtual
void GNormalize::untransform(const GVec& in, GVec& out)
{
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(in[i] == UNKNOWN_REAL_VALUE)
				out[i] = UNKNOWN_REAL_VALUE;
			else
				out[i] = GMatrix::normalizeValue(in[i], m_min, m_max, m_mins[i], m_mins[i] + m_ranges[i]);
		}
		else
			out[i] = in[i];
	}
}

// virtual
void GNormalize::untransformToDistribution(const GVec& in, GPrediction* out)
{
	throw Ex("Sorry, cannot denormalize to a distribution");
}

// --------------------------------------------------------------------------

GDiscretize::GDiscretize(size_t buckets)
: GIncrementalTransform()
{
	m_bucketsIn = buckets;
	m_bucketsOut = -1;
}

GDiscretize::GDiscretize(const GDomNode* pNode)
: GIncrementalTransform(pNode)
{
	m_bucketsIn = (size_t)pNode->getInt("bucketsIn");
	m_bucketsOut = (size_t)pNode->getInt("bucketsOut");
	m_mins.deserialize(pNode->get("mins"));
	m_ranges.deserialize(pNode->get("ranges"));
}

// virtual
GDiscretize::~GDiscretize()
{
}

// virtual
GDomNode* GDiscretize::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GDiscretize");
	pNode->add(pDoc, "bucketsIn", m_bucketsIn);
	pNode->add(pDoc, "bucketsOut", m_bucketsOut);
	pNode->add(pDoc, "mins", m_mins.serialize(pDoc));
	pNode->add(pDoc, "ranges", m_ranges.serialize(pDoc));
	return pNode;
}

// virtual
GRelation* GDiscretize::trainInner(const GMatrix& data)
{
	// Make the relations
	m_bucketsOut = m_bucketsIn;
	if(m_bucketsOut > data.rows())
		m_bucketsOut = std::max((size_t)2, (size_t)sqrt((double)data.rows()));
	size_t nAttrCount = data.cols();
	GMixedRelation* pRelationAfter = new GMixedRelation();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
			pRelationAfter->addAttr(nValues);
		else
			pRelationAfter->addAttr(m_bucketsOut);
	}

	// Determine the boundaries
	m_mins.resize(nAttrCount);
	m_ranges.resize(nAttrCount);
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
		{
			m_mins[i] = 0;
			m_ranges[i] = 0;
		}
		else
		{
			m_mins[i] = data.columnMin(i);
			m_ranges[i] = data.columnMax(i) - m_mins[i];
			m_ranges[i] = std::max(m_ranges[i], 1e-9);
		}
	}
	return pRelationAfter;
}

// virtual
GRelation* GDiscretize::trainInner(const GRelation& relation)
{
	throw Ex("This transform cannot be trained without data");
	return before().clone();
}

// virtual
void GDiscretize::transform(const GVec& in, GVec& out)
{
	if(m_mins.size() == 0)
		throw Ex("Train was not called");
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
			out[i] = in[i];
		else
			out[i] = std::max(0, std::min((int)(m_bucketsOut - 1), (int)(((in[i] - m_mins[i]) * m_bucketsOut) / m_ranges[i])));
	}
}

// virtual
void GDiscretize::untransform(const GVec& in, GVec& out)
{
	if(m_mins.size() == 0)
		throw Ex("Train was not called");
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
			out[i] = in[i];
		else
			out[i] = (((double)in[i] + .5) * m_ranges[i]) / m_bucketsOut + m_mins[i];
	}
}

// virtual
void GDiscretize::untransformToDistribution(const GVec& in, GPrediction* out)
{
	if(m_mins.size() == 0)
		throw Ex("Train was not called");
	size_t attrCount = before().size();
	for(size_t i = 0; i < attrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
			out[i].makeCategorical()->setSpike(nValues, (size_t)in[i], 1);
		else
			out[i].makeNormal()->setMeanAndVariance((((double)in[i] + .5) * m_ranges[i]) / m_bucketsOut + m_mins[i], m_ranges[i] * m_ranges[i]);
	}
}





GImputeMissingVals::GImputeMissingVals()
: m_pCF(NULL), m_pNTC(NULL), m_pLabels(NULL), m_pBatch(NULL)
{
}

GImputeMissingVals::GImputeMissingVals(const GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode), m_pLabels(NULL), m_pBatch(NULL)
{
	m_pCF = ll.loadCollaborativeFilter(pNode->get("cf"));
	GDomNode* pNTC = pNode->getIfExists("ntc");
	if(pNTC)
		m_pNTC = new GNominalToCat(pNTC);
	else
		m_pNTC = NULL;
}

// virtual
GImputeMissingVals::~GImputeMissingVals()
{
	delete(m_pCF);
	delete(m_pNTC);
	delete(m_pBatch);
}

// virtual
GDomNode* GImputeMissingVals::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GImputeMissingVals");
	pNode->add(pDoc, "cf", m_pCF->serialize(pDoc));
	if(m_pNTC)
		pNode->add(pDoc, "ntc", m_pNTC->serialize(pDoc));
	return pNode;
}

void GImputeMissingVals::setCollaborativeFilter(GCollaborativeFilter* pCF)
{
	delete(m_pCF);
	m_pCF = pCF;
}

// virtual
GRelation* GImputeMissingVals::trainInner(const GMatrix& data)
{
	// Train the nominalToCat filter if needed
	if(data.relation().areContinuous())
	{
		delete(m_pNTC);
		m_pNTC = NULL;
	}
	else if(!m_pNTC)
	{
		m_pNTC = new GNominalToCat();
		m_pNTC->preserveUnknowns();
	}
	const GMatrix* pData;
	std::unique_ptr<GMatrix> hData;
	if(m_pNTC)
	{
		m_pNTC->train(data);
		GMatrix* pTemp = m_pNTC->transformBatch(data);
		hData.reset(pTemp);
		pData = pTemp;
	}
	else
		pData = &data;

	// Train the collaborative filter
	if(!m_pCF)
		m_pCF = new GMatrixFactorization(std::max(size_t(2), std::min(size_t(8), data.cols() / 3)));
	m_pCF->trainDenseMatrix(*pData, m_pLabels);
	return before().clone();
}

// virtual
GRelation* GImputeMissingVals::trainInner(const GRelation& relation)
{
	throw Ex("This transform cannot be trained without data");
	return before().clone();
}

// virtual
void GImputeMissingVals::transform(const GVec& in, GVec& out)
{
	// If there are no missing values, just copy it across
	size_t dims = before().size();
	size_t i;
	for(i = 0; i < dims; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(in[i] == UNKNOWN_REAL_VALUE)
				break;
		}
		else
		{
			if(in[i] == UNKNOWN_DISCRETE_VALUE)
				break;
		}
	}
	if(i >= dims)
	{
		out.copy(in);
		return;
	}

	// Convert to all real values if necessary
	GVec* pVec;
	if(m_pNTC)
	{
		m_pNTC->transform(in, m_pNTC->innerBuf());
		pVec = &m_pNTC->innerBuf();
		dims = m_pNTC->after().size();
	}
	else
	{
		out.copy(in);
		pVec = &out;
	}

	// Impute the missing values
	m_pCF->impute(*pVec, dims);

	// Convert back to nominal if necessary
	if(m_pNTC)
		m_pNTC->untransform(*pVec, out);
}

// virtual
void GImputeMissingVals::untransform(const GVec& in, GVec& out)
{
	memcpy(out.data(), in.data(), after().size() * sizeof(double));
}

// virtual
void GImputeMissingVals::untransformToDistribution(const GVec& in, GPrediction* out)
{
	throw Ex("Sorry, cannot unimpute to a distribution");
}

// virtual
GMatrix* GImputeMissingVals::transformBatch(const GMatrix& in)
{
	GMatrix* out = new GMatrix();
	out->copy(in);
	size_t dims = out->cols();
	for(size_t i = 0; i < out->rows(); i++)
	{
		GVec& vec = out->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			if(vec[j] == UNKNOWN_REAL_VALUE)
				vec[j] = m_pCF->predict(i, j);
		}
	}
	return out;
}

void GImputeMissingVals::setLabels(const GMatrix* pLabels)
{
	m_pLabels = pLabels;
}


// --------------------------------------------------------------------------

GLogify::GLogify()
: GIncrementalTransform()
{
}

GLogify::GLogify(const GDomNode* pNode)
: GIncrementalTransform(pNode)
{
}

// virtual
GLogify::~GLogify()
{
}

// virtual
GDomNode* GLogify::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GLogify");
	return pNode;
}

// virtual
GRelation* GLogify::trainInner(const GMatrix& data)
{
	return data.relation().clone();
}

// virtual
GRelation* GLogify::trainInner(const GRelation& relation)
{
	return relation.clone();
}

// virtual
void GLogify::transform(const GVec& in, GVec& out)
{
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(in[i] == UNKNOWN_REAL_VALUE)
				out[i] = UNKNOWN_REAL_VALUE;
			else
				out[i] = log(in[i]);
		}
		else
			out[i] = in[i];
	}
}

// virtual
void GLogify::untransform(const GVec& in, GVec& out)
{
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(in[i] == UNKNOWN_REAL_VALUE)
				out[i] = UNKNOWN_REAL_VALUE;
			else
				out[i] = exp(in[i]);
		}
		else
			out[i] = in[i];
	}
}

// virtual
void GLogify::untransformToDistribution(const GVec& in, GPrediction* out)
{
	throw Ex("Sorry, cannot unlogify to a distribution");
}



} // namespace GClasses

