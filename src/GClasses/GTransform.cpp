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
#ifndef MIN_PREDICT
#include "GDistribution.h"
#endif // MIN_PREDICT
#include "GRand.h"
#ifndef MIN_PREDICT
#include "GManifold.h"
#include "GCluster.h"
#include "GString.h"
#endif // MIN_PREDICT
#include "GNeuralNet.h"
#ifndef MIN_PREDICT
#include "GRecommender.h"
#endif // MIN_PREDICT
#include "GHolders.h"
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string>
#include <cmath>

namespace GClasses {

using std::string;
using std::vector;
using std::ostringstream;

GTransform::GTransform()
{
}

GTransform::GTransform(GDomNode* pNode, GLearnerLoader& ll)
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
	pNode->addField(pDoc, "class", pDoc->newString(szClassName));
	return pNode;
}

// ---------------------------------------------------------------

GIncrementalTransform::GIncrementalTransform(GDomNode* pNode, GLearnerLoader& ll)
: GTransform(pNode, ll), m_pInnerBuf(NULL)
{
	m_pRelationBefore = GRelation::deserialize(pNode->field("before"));
	m_pRelationAfter = GRelation::deserialize(pNode->field("after"));
}

// virtual
GIncrementalTransform::~GIncrementalTransform()
{
	delete(m_pRelationBefore);
	delete(m_pRelationAfter);
	delete[] m_pInnerBuf;
}

// virtual
GDomNode* GIncrementalTransform::baseDomNode(GDom* pDoc, const char* szClassName) const
{
	if(!m_pRelationAfter)
		throw Ex("train must be called before serialize");
	GDomNode* pNode = GTransform::baseDomNode(pDoc, szClassName);
	pNode->addField(pDoc, "before", m_pRelationBefore->serialize(pDoc));
	pNode->addField(pDoc, "after", m_pRelationAfter->serialize(pDoc));
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
	Holder<GMatrix> hOut(pOut);
	pOut->newRows(nRows);
	for(size_t i = 0; i < nRows; i++)
		transform(in.row(i), pOut->row(i));
	return hOut.release();
}

double* GIncrementalTransform::innerBuf()
{
	if(!m_pInnerBuf)
		m_pInnerBuf = new double[m_pRelationAfter->size()];
	return m_pInnerBuf;
}

// virtual
GMatrix* GIncrementalTransform::untransformBatch(const GMatrix& in)
{
	if(!m_pRelationBefore)
		throw Ex("train has not been called");
	size_t nRows = in.rows();
	GMatrix* pOut = new GMatrix(before().clone());
	pOut->newRows(nRows);
	Holder<GMatrix> hOut(pOut);
	for(size_t i = 0; i < nRows; i++)
		untransform(in.row(i), pOut->row(i));
	return hOut.release();
}

#ifndef MIN_PREDICT
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
	Holder<GMatrix> hA(pA);
	if(pA->sumSquaredDifference(e) > 1e-12)
		throw Ex("Expected:\n", to_str(e), "\nGot:\n", to_str(*pA));
	if(!pA->relation().areContinuous())
		throw Ex("failed");
	GMatrix* pB = trans.untransformBatch(*pA);
	Holder<GMatrix> hB(pB);
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
	Holder<GIncrementalTransform> hTrans(pTrans);

	// Transform the input matrix again, and check it
	GMatrix* pC = pTrans->transformBatch(m);
	Holder<GMatrix> hC(pC);
	if(pC->sumSquaredDifference(e) > 1e-12)
		throw Ex("Expected:\n", to_str(e), "\nGot:\n", to_str(*pC));
	if(!pC->relation().areContinuous())
		throw Ex("failed");
	GMatrix* pD = trans.untransformBatch(*pC);
	Holder<GMatrix> hD(pD);
	if(pD->sumSquaredDifference(m) > 1e-12)
		throw Ex("Expected:\n", to_str(m), "\nGot:\n", to_str(*pD));
	if(!pD->relation().isCompatible(m.relation()) || !m.relation().isCompatible(pD->relation()))
		throw Ex("failed");
}
#endif // MIN_PREDICT












GIncrementalTransformChainer::GIncrementalTransformChainer(GIncrementalTransform* pFirst, GIncrementalTransform* pSecond)
: GIncrementalTransform(), m_pFirst(pFirst), m_pSecond(pSecond)
{
}

GIncrementalTransformChainer::GIncrementalTransformChainer(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll)
{
	m_pFirst = ll.loadIncrementalTransform(pNode->field("first"));
	m_pSecond = ll.loadIncrementalTransform(pNode->field("second"));
}

// virtual
GIncrementalTransformChainer::~GIncrementalTransformChainer()
{
	delete(m_pFirst);
	delete(m_pSecond);
}

#ifndef MIN_PREDICT
// virtual
GDomNode* GIncrementalTransformChainer::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GIncrementalTransformChainer");
	pNode->addField(pDoc, "first", m_pFirst->serialize(pDoc));
	pNode->addField(pDoc, "second", m_pSecond->serialize(pDoc));
	return pNode;
}
#endif // MIN_PREDICT

// virtual
GRelation* GIncrementalTransformChainer::trainInner(const GMatrix& data)
{
	m_pFirst->train(data);
	GMatrix* pData2 = m_pFirst->transformBatch(data); // todo: often this step is computation overkill since m_pSecond may not even use it during training. Is there a way to avoid doing it?
	Holder<GMatrix> hData2(pData2);
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
void GIncrementalTransformChainer::transform(const double* pIn, double* pOut)
{
	double* pBuf = m_pFirst->innerBuf();
	m_pFirst->transform(pIn, pBuf);
	m_pSecond->transform(pBuf, pOut);
}

// virtual
void GIncrementalTransformChainer::untransform(const double* pIn, double* pOut)
{
	double* pBuf = m_pFirst->innerBuf();
	m_pSecond->untransform(pIn, pBuf);
	m_pFirst->untransform(pBuf, pOut);
}

#ifndef MIN_PREDICT
// virtual
void GIncrementalTransformChainer::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	double* pBuf = m_pFirst->innerBuf();
	m_pSecond->untransform(pIn, pBuf);
	m_pFirst->untransformToDistribution(pBuf, pOut);
}
#endif // MIN_PREDICT

// ---------------------------------------------------------------

GPCA::GPCA(size_t targetDims)
: GIncrementalTransform(), m_targetDims(targetDims), m_pBasisVectors(NULL), m_pCentroid(NULL), m_pEigVals(NULL), m_aboutOrigin(false), m_rand(0)
{
}

GPCA::GPCA(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll), m_pEigVals(NULL), m_rand(0)
{
	m_targetDims = before().size();
	m_pBasisVectors = new GMatrix(pNode->field("basis"));
	m_pCentroid = new GMatrix(pNode->field("centroid"));
	m_aboutOrigin = pNode->field("aboutOrigin")->asBool();
}

// virtual
GPCA::~GPCA()
{
	delete(m_pBasisVectors);
	delete(m_pCentroid);
	delete[] m_pEigVals;
}

#ifndef MIN_PREDICT
// virtual
GDomNode* GPCA::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GPCA");
	pNode->addField(pDoc, "basis", m_pBasisVectors->serialize(pDoc));
	pNode->addField(pDoc, "centroid", m_pCentroid->serialize(pDoc));
	pNode->addField(pDoc, "aboutOrigin", pDoc->newBool(m_aboutOrigin));
	return pNode;
}
#endif // MIN_PREDICT

void GPCA::computeEigVals()
{
	delete[] m_pEigVals;
	m_pEigVals = new double[m_targetDims];
}

// virtual
GRelation* GPCA::trainInner(const GMatrix& data)
{
	if(!before().areContinuous())
		throw Ex("GPCA doesn't support nominal values. (You could filter with nominaltocat to make them real.)");
	delete(m_pBasisVectors);
	delete(m_pCentroid);
	m_pBasisVectors = new GMatrix(m_targetDims, before().size());
	m_pCentroid = new GMatrix(1, before().size());

	// Compute the mean
	size_t nInputDims = before().size();
	double* pMean = m_pCentroid->row(0);
	if(m_aboutOrigin)
		GVec::setAll(pMean, 0.0, nInputDims);
	else
		data.centroid(pMean);

	// Make a copy of the data
	GMatrix tmpData(data.relation().clone());
	tmpData.copy(&data);

	// Compute the principle components
	double sse = 0;
	if(m_pEigVals)
		sse = tmpData.sumSquaredDistance(pMean);
	for(size_t i = 0; i < m_targetDims; i++)
	{
		double* pVector = m_pBasisVectors->row(i);
		tmpData.principalComponentIgnoreUnknowns(pVector, pMean, &m_rand);
		tmpData.removeComponent(pMean, pVector);
		if(m_pEigVals)
		{
			double t = tmpData.sumSquaredDistance(pMean);
			m_pEigVals[i] = (sse - t) / (data.rows() - 1);
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
void GPCA::transform(const double* pIn, double* pOut)
{
	double* pCentroid = m_pCentroid->row(0);
	size_t nInputDims = before().size();
	for(size_t i = 0; i < m_targetDims; i++)
	{
		double* pBasisVector = m_pBasisVectors->row(i);
		*(pOut++) = GVec::dotProductIgnoringUnknowns(pCentroid, pIn, pBasisVector, nInputDims);
	}
}

// virtual
void GPCA::untransform(const double* pIn, double* pOut)
{
	size_t nInputDims = before().size();
	GVec::copy(pOut, m_pCentroid->row(0), nInputDims);
	for(size_t i = 0; i < m_targetDims; i++)
		GVec::addScaled(pOut, pIn[i], m_pBasisVectors->row(i), nInputDims);
}

// virtual
void GPCA::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, PCA cannot untransform to a distribution");
}

/*
GPCARotateOnly::GPCARotateOnly(GArffRelation* pRelation, GMatrix* pData)
{
	m_pRelation = pRelation;
	m_pInputData = pData;
	m_pOutputData = NULL;
}

GPCARotateOnly::~GPCARotateOnly()
{
	delete(m_pOutputData);
}

// static
GMatrix* GPCARotateOnly::DoPCA(GArffRelation* pRelation, GMatrix* pData)
{
	GPCA pca(pRelation, pData);
	pca.DoPCA();
	return pca.ReleaseOutputData();
}

void GPCARotateOnly::DoPCA()
{
	// Compute the eigenvectors
	GMatrix m;
	m_pInputData->ComputeCovarianceMatrix(&m, m_pRelation);
	GMatrix eigenVectors;
	eigenVectors.ComputeEigenVectors(m.GetColumnCount(), &m);
	m_pOutputData = new GMatrix(m_pInputData->rows());
	int nRowCount = m_pInputData->rows();
	int nInputCount = m_pRelation->GetInputCount();
	int nOutputCount = m_pRelation->GetOutputCount();
	int nAttributeCount = m_pRelation->size();
	double* pInputRow;
	double* pOutputRow;
	int n, i, j, nIndex;

	// Allocate space for the output
	for(n = 0; n < nRowCount; n++)
	{
		pOutputRow = new double[nAttributeCount];
		m_pOutputData->AddRow(pOutputRow);
	}

	// Compute the output
	double* pEigenVector;
	Holder<double> hInputVector(new double[nInputCount]);
	double* pInputVector = hInputVector.Get();
	for(i = 0; i < nInputCount; i++)
	{
		nIndex = m_pRelation->GetInputIndex(i);
		pEigenVector = eigenVectors.row(i);
		for(n = 0; n < nRowCount; n++)
		{
			pInputRow = m_pInputData->row(n);
			for(j = 0; j < nInputCount; j++)
				pInputVector[j] = pInputRow[m_pRelation->GetInputIndex(j)];
			pOutputRow = m_pOutputData->row(n);
			pOutputRow[nIndex] = GVec::dotProduct(pInputVector, pEigenVector, nInputCount);
		}
	}
	for(i = 0; i < nOutputCount; i++)
	{
		for(n = 0; n < nRowCount; n++)
		{
			nIndex = m_pRelation->GetOutputIndex(i);
			pInputRow = m_pInputData->row(n);
			pOutputRow = m_pOutputData->row(n);
			pOutputRow[nIndex] = pInputRow[nIndex];
		}
	}
}

GMatrix* GPCARotateOnly::ReleaseOutputData()
{
	GMatrix* pData = m_pOutputData;
	m_pOutputData = NULL;
	return pData;
}
*/
GMatrix* GPCARotateOnly::transform(size_t nDims, size_t nOutputs, const GMatrix* pData, size_t nComponents, GRand* pRand)
{
	// Init the basis vectors
	size_t nElements = nDims * nDims;
	double* pBasisVectors = new double[nElements + nDims * 4];
	ArrayHolder<double> hBasisVectors(pBasisVectors);
	double* pComponent = &pBasisVectors[nElements];
	double* pA = &pBasisVectors[nElements + nDims];
	double* pB = &pBasisVectors[nElements + 2 * nDims];
	double* pMean = &pBasisVectors[nElements + 3 * nDims];
	size_t j;
	for(size_t i = 0; i < nElements; i++)
		pBasisVectors[i] = 0;
	for(size_t i = 0; i < nDims; i++)
		pBasisVectors[nDims * i + i] = 1;

	// Compute the mean
	for(j = 0; j < nDims; j++)
		pMean[j] = pData->columnMean(j);

	// Make a copy of the data
	GMatrix* pOutData = new GMatrix(pData->relation().clone());
	pOutData->copy(pData);
	Holder<GMatrix> hOutData(pOutData);

	// Rotate the basis vectors
	double dDotProd;
	for(size_t i = 0; i < nComponents; i++)
	{
		// Compute the next principle component
		pOutData->principalComponent(pComponent, pMean, pRand);
		pOutData->removeComponent(pMean, pComponent);

		// Use the current axis as the first plane vector
		GVec::copy(pA, &pBasisVectors[nDims * i], nDims);

		// Use the modified Gram-Schmidt process to compute the other plane vector
		GVec::copy(pB, pComponent, nDims);
		dDotProd = GVec::dotProduct(pB, pA, nDims);
		GVec::addScaled(pB, -dDotProd, pA, nDims);
		double dMag = sqrt(GVec::squaredMagnitude(pB, nDims));
		if(dMag < 1e-6)
			break; // It's already close enough. If we normalized something that small, it would just mess up our data
		GVec::multiply(pB, 1.0 / dMag, nDims);

		// Rotate the remaining basis vectors
		double dAngle = atan2(GVec::dotProduct(pComponent, pB, nDims), dDotProd);
		for(j = i; j < nDims; j++)
		{
			GVec::rotate(&pBasisVectors[nDims * j], nDims, dAngle, pA, pB);
			GAssert(std::abs(GVec::squaredMagnitude(&pBasisVectors[nDims * j], nDims) - 1.0) < 1e-4);
		}
	}

	// Align data with new basis vectors
	const double* pInVector;
	double* pOutVector;
	size_t nCount = pData->rows();
	for(size_t i = 0; i < nCount; i++)
	{
		pInVector = pData->row(i);
		pOutVector = pOutData->row(i);
		for(j = 0; j < nDims; j++)
			pOutVector[j] = GVec::dotProduct(pMean, pInVector, &pBasisVectors[nDims * j], nDims);
	}

	return hOutData.release();
}

#ifndef MIN_PREDICT
//static
void GPCARotateOnly::test()
{
	GRand prng(0);
	GMatrix data(0, 2);
	double* pVec;
	pVec = data.newRow();	pVec[0] = 0;	pVec[1] = 0;
	pVec = data.newRow();	pVec[0] = 10;	pVec[1] = 10;
	pVec = data.newRow();	pVec[0] = 4;	pVec[1] = 6;
	pVec = data.newRow();	pVec[0] = 6;	pVec[1] = 4;
	GMatrix* pOut2 = GPCARotateOnly::transform(2, 0, &data, 2, &prng);
	for(size_t i = 0; i < pOut2->rows(); i++)
	{
		pVec = pOut2->row(i);
		if(std::abs(std::abs(pVec[0]) - 7.071067) < .001)
		{
			if(std::abs(pVec[1]) > .001)
				throw Ex("wrong answer");
		}
		else if(std::abs(pVec[0]) < .001)
		{
			if(std::abs(std::abs(pVec[1]) - 1.414214) > .001)
				throw Ex("wrong answer");
		}
		else
			throw Ex("wrong answer");
	}
	delete(pOut2);
}
#endif // !MIN_PREDICT

// --------------------------------------------------------------------------

GNoiseGenerator::GNoiseGenerator()
: GIncrementalTransform(), m_rand(0), m_mean(0), m_deviation(1)
{
}

GNoiseGenerator::GNoiseGenerator(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll), m_rand(0)
{
	m_mean = pNode->field("mean")->asDouble();
	m_deviation = pNode->field("dev")->asDouble();
}

GNoiseGenerator::~GNoiseGenerator()
{
}

// virtual
GDomNode* GNoiseGenerator::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNoiseGenerator");
	pNode->addField(pDoc, "mean", pDoc->newDouble(m_mean));
	pNode->addField(pDoc, "dev", pDoc->newDouble(m_deviation));
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
void GNoiseGenerator::transform(const double* pIn, double* pOut)
{
	size_t nDims = before().size();
	for(size_t i = 0; i < nDims; i++)
	{
		size_t vals = before().valueCount(i);
		if(vals == 0)
			pOut[i] = m_rand.normal() * m_deviation + m_mean;
		else
			pOut[i] = (double)m_rand.next(vals);
	}
}

// --------------------------------------------------------------------------

GPairProduct::GPairProduct(size_t nMaxDims)
: GIncrementalTransform(), m_maxDims(nMaxDims)
{
}

GPairProduct::GPairProduct(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll)
{
	m_maxDims = (size_t)pNode->field("maxDims")->asInt();
}

GPairProduct::~GPairProduct()
{
}

// virtual
GDomNode* GPairProduct::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GPairProduct");
	pNode->addField(pDoc, "maxDims", pDoc->newInt(m_maxDims));
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
void GPairProduct::transform(const double* pIn, double* pOut)
{
	size_t i, j, nAttr;
	size_t nAttrsIn = before().size();
	size_t nAttrsOut = after().size();
	nAttr = 0;
	for(j = 0; j < nAttrsIn && nAttr < nAttrsOut; j++)
	{
		for(i = j; i < nAttrsIn && nAttr < nAttrsOut; i++)
			pOut[nAttr++] = pIn[i] * pIn[j];
	}
	GAssert(nAttr == nAttrsOut);
}

// --------------------------------------------------------------------------

GReservoir::GReservoir(double weightDeviation, size_t outputs, size_t hiddenLayers)
: GIncrementalTransform(), m_pNN(NULL), m_outputs(outputs), m_deviation(weightDeviation), m_hiddenLayers(hiddenLayers)
{
}

GReservoir::GReservoir(GDomNode* pNode, GLearnerLoader& ll)
{
	m_pNN = new GNeuralNet(pNode->field("nn"), ll);
	m_outputs = m_pNN->relLabels().size();
	m_deviation = pNode->field("dev")->asDouble();
	m_hiddenLayers = (size_t)pNode->field("hl")->asInt();
}

// virtual
GReservoir::~GReservoir()
{
	delete(m_pNN);
}

#ifndef MIN_PREDICT
// virtual
GDomNode* GReservoir::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GReservoir");
	pNode->addField(pDoc, "nn", m_pNN->serialize(pDoc));
	pNode->addField(pDoc, "dev", pDoc->newDouble(m_deviation));
	pNode->addField(pDoc, "hl", pDoc->newInt(m_hiddenLayers));
	return pNode;
}
#endif // MIN_PREDICT


// virtual
GRelation* GReservoir::trainInner(const GMatrix& data)
{
	return trainInner(data.relation());
}

// virtual
GRelation* GReservoir::trainInner(const GRelation& relation)
{
	delete(m_pNN);
	GNeuralNet* pNN = new GNeuralNet();
	for(size_t i = 0; i < m_hiddenLayers; i++)
		pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, m_outputs));
	pNN->addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
	GUniformRelation* pRel = new GUniformRelation(m_outputs);
	m_pNN = pNN;
	if(!relation.areContinuous())
		m_pNN = new GFeatureFilter(m_pNN, new GNominalToCat());
	m_pNN->beginIncrementalLearning(relation, *pRel);
	pNN->perturbAllWeights(m_deviation);
	return pRel;
}

// virtual
void GReservoir::transform(const double* pIn, double* pOut)
{
	m_pNN->predict(pIn, pOut);
}


// --------------------------------------------------------------------------

GDataAugmenter::GDataAugmenter(GIncrementalTransform* pTransform)
: GIncrementalTransform(), m_pTransform(pTransform)
{
}

GDataAugmenter::GDataAugmenter(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll)
{
	m_pTransform = ll.loadIncrementalTransform(pNode->field("trans"));
}

// virtual
GDataAugmenter::~GDataAugmenter()
{
	delete(m_pTransform);
}

#ifndef MIN_PREDICT
// virtual
GDomNode* GDataAugmenter::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GDataAugmenter");
	pNode->addField(pDoc, "trans", m_pTransform->serialize(pDoc));
	return pNode;
}
#endif // MIN_PREDICT

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
void GDataAugmenter::transform(const double* pIn, double* pOut)
{
	GVec::copy(pOut, pIn, before().size());
	m_pTransform->transform(pIn, pOut + before().size());
}

// virtual
void GDataAugmenter::untransform(const double* pIn, double* pOut)
{
	GVec::copy(pOut, pIn, before().size());
}

// virtual
void GDataAugmenter::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, this method is not implemented yet");
}

// --------------------------------------------------------------------------
#ifndef MIN_PREDICT

GAttributeSelector::GAttributeSelector(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll), m_seed(1234567)
{
	m_labelDims = (size_t)pNode->field("labels")->asInt();
	m_targetFeatures = (size_t)pNode->field("target")->asInt();
	GDomNode* pRanksNode = pNode->field("ranks");
	GDomListIterator it(pRanksNode);
	m_ranks.reserve(it.remaining());
	for( ; it.current(); it.advance())
		m_ranks.push_back((size_t)it.current()->asInt());
	if(m_ranks.size() + (size_t)m_labelDims != (size_t)before().size())
		throw Ex("invalid attribute selector");
	if(m_targetFeatures > m_ranks.size())
		throw Ex("invalid attribute selector");
}

// virtual
GDomNode* GAttributeSelector::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GAttributeSelector");
	pNode->addField(pDoc, "labels", pDoc->newInt(m_labelDims));
	pNode->addField(pDoc, "target", pDoc->newInt(m_targetFeatures));
	GDomNode* pRanksNode = pNode->addField(pDoc, "ranks", pDoc->newList());
	for(size_t i = 0; i < m_ranks.size(); i++)
		pRanksNode->addItem(pDoc, pDoc->newInt(m_ranks[i]));
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
	Holder<GMatrix> hNormData(pNormData);

	// Divide into features and labels
	size_t curDims = data.cols() - m_labelDims;
	m_ranks.resize(curDims);
	GMatrix* pFeatures = pNormData->cloneSub(0, 0, data.rows(), data.cols() - m_labelDims);
	Holder<GMatrix> hFeatures(pFeatures);
	GMatrix* pLabels = pNormData->cloneSub(0, data.cols() - m_labelDims, data.rows(), m_labelDims);
	Holder<GMatrix> hLabels(pLabels);
	vector<size_t> indexMap;
	for(size_t i = 0; i < curDims; i++)
		indexMap.push_back(i);

	// Produce a ranked attributed ordering by deselecting the weakest attribute each time
	while(curDims > 1)
	{
		// Convert nominal attributes to a categorical distribution
		GNominalToCat ntc;
		ntc.train(*pFeatures);
		GMatrix* pFeatures2 = ntc.transformBatch(*pFeatures);
		Holder<GMatrix> hFeatures2(pFeatures2);
		vector<size_t> rmap;
		ntc.reverseAttrMap(rmap);
		GNominalToCat ntc2;
		ntc2.train(*pLabels);
		GMatrix* pLabels2 = ntc2.transformBatch(*pLabels);
		Holder<GMatrix> hLabels2(pLabels2);

		// Train a single-layer neural network with the normalized remaining data
		GNeuralNet nn;
		nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, FLEXIBLE_SIZE));
		nn.rand().setSeed(m_seed);
		m_seed += 77152487;
		m_seed *= 37152487;
		nn.setWindowSize(30);
		nn.setImprovementThresh(0.002);
		nn.train(*pFeatures2, *pLabels2);

		// Identify the weakest attribute
		GLayerClassic& layer = *(GLayerClassic*)&nn.layer(nn.layerCount() - 1);
		size_t pos = 0;
		double weakest = 1e308;
		size_t weakestIndex = 0;
		for(size_t i = 0; i < curDims; i++)
		{
			double w = 0;
			while(pos < nn.relFeatures().size() && rmap[pos] == i)
			{
				for(size_t neuron = 0; neuron < layer.outputs(); neuron++)
					w = std::max(w, std::abs(layer.weights()[pos][neuron]));
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
		pFeatures->deleteColumn(weakestIndex);
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
void GAttributeSelector::transform(const double* pIn, double* pOut)
{
	size_t i;
	for(i = 0; i < m_targetFeatures; i++)
		pOut[i] = pIn[m_ranks[i]];
	size_t featureDims = before().size() - m_labelDims;
	for(size_t j = 0; j < m_labelDims; j++)
		pOut[i++] = pIn[featureDims + j];
}

//static
void GAttributeSelector::test()
{
	GRand prng(0);
	GMatrix data(0, 21);
	for(size_t i = 0; i < 256; i++)
	{
		double* pVec = data.newRow();
		prng.cubical(pVec, 20);
		pVec[20] = 0.2 * pVec[3] * pVec[3] * - 7.0 * pVec[3] * pVec[13] + pVec[17];
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
#endif // MIN_PREDICT

// --------------------------------------------------------------------------

GNominalToCat::GNominalToCat(size_t nValueCap)
: GIncrementalTransform(), m_valueCap(nValueCap), m_preserveUnknowns(false)
{
}

GNominalToCat::GNominalToCat(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll)
{
	m_valueCap = (size_t)pNode->field("valueCap")->asInt();
	m_preserveUnknowns = pNode->field("pu")->asBool();
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
	pNode->addField(pDoc, "valueCap", pDoc->newInt(m_valueCap));
	pNode->addField(pDoc, "pu", pDoc->newBool(m_preserveUnknowns));
	return pNode;
}

// virtual
void GNominalToCat::transform(const double* pIn, double* pOut)
{
	size_t nInAttrCount = before().size();
	for(size_t i = 0; i < nInAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues < 3)
		{
			if(nValues == 0)
				*(pOut++) = *(pIn++);
			else if(nValues == 1)
			{
				if(*pIn == UNKNOWN_DISCRETE_VALUE)
					*(pOut++) = UNKNOWN_REAL_VALUE;
				else
					*(pOut++) = 0;
				pIn++;
			}
			else
			{
				if(*pIn == UNKNOWN_DISCRETE_VALUE)
				{
					if(m_preserveUnknowns)
						*(pOut++) = UNKNOWN_REAL_VALUE;
					else
						*(pOut++) = 0.5;
				}
				else
					*(pOut++) = *pIn;
				pIn++;
			}
		}
		else if(nValues < m_valueCap)
		{
			if(*pIn >= 0)
			{
				GAssert(*pIn < nValues);
				GVec::setAll(pOut, 0.0, nValues);
				pOut[(int)*pIn] = 1.0;
			}
			else
			{
				if(m_preserveUnknowns)
					GVec::setAll(pOut, UNKNOWN_REAL_VALUE, nValues);
				else
					GVec::setAll(pOut, 1.0 / nValues, nValues);
			}
			pOut += nValues;
			pIn++;
		}
		else
		{
			if(*pIn == UNKNOWN_DISCRETE_VALUE)
				*(pOut++) = UNKNOWN_REAL_VALUE;
			else
				*(pOut++) = *pIn;
			pIn++;
		}
	}
}

// virtual
void GNominalToCat::untransform(const double* pIn, double* pOut)
{
	size_t nOutAttrCount = before().size();
	for(size_t i = 0; i < nOutAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues < 3)
		{
			if(nValues == 0)
				*(pOut++) = *(pIn++);
			else if(nValues == 1)
			{
				if(*pIn == UNKNOWN_REAL_VALUE)
					*(pOut++) = UNKNOWN_DISCRETE_VALUE;
				else
					*(pOut++) = 0;
				pIn++;
			}
			else
			{
				if(*pIn == UNKNOWN_REAL_VALUE)
				{
					*(pOut++) = UNKNOWN_DISCRETE_VALUE;
					pIn++;
				}
				else
					*(pOut++) = (*(pIn++) < 0.5 ? 0 : 1);
			}
		}
		else if(nValues < m_valueCap)
		{
			double max = *(pIn++);
			*pOut = 0.0;
			for(size_t i = 1; i < nValues; i++)
			{
				if(*pIn > max)
				{
					max = *pIn;
					*pOut = (double)i;
				}
				pIn++;
			}
			pOut++;
		}
		else
		{
			if(*pIn == UNKNOWN_REAL_VALUE)
				*(pOut++) = UNKNOWN_DISCRETE_VALUE;
			else
				*(pOut++) = std::max(0.0, std::min(double(nValues - 1), floor(*pIn + 0.5)));
			pIn++;
		}
	}
}

#ifndef MIN_PREDICT
// virtual
void GNominalToCat::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	size_t nOutAttrCount = before().size();
	for(size_t i = 0; i < nOutAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues < 3)
		{
			if(nValues == 0)
			{
				GNormalDistribution* pNorm = pOut->makeNormal();
				pNorm->setMeanAndVariance(*pIn, 1.0); // todo: should we throw an exception here since we have no way to estimate the variance?
			}
			else if(nValues == 1)
			{
				GCategoricalDistribution* pCat = pOut->makeCategorical();
				pCat->setToUniform(1);
			}
			else
			{
				GCategoricalDistribution* pCat = pOut->makeCategorical();
				if(*pIn == UNKNOWN_REAL_VALUE)
					pCat->setToUniform(2);
				else
				{
					double* pVals = pCat->values(2);
					pVals[0] = 1.0 - *pIn;
					pVals[1] = *pIn;
					pCat->normalize(); // We have to normalize to ensure the values are properly clipped.
				}
			}
			pIn++;
			pOut++;
		}
		else if(nValues < m_valueCap)
		{
			GCategoricalDistribution* pCat = pOut->makeCategorical();
			pCat->setValues(nValues, pIn);
			pIn += nValues;
			pOut++;
		}
		else
		{
			GCategoricalDistribution* pCat = pOut->makeCategorical();
			pCat->setSpike(nValues, std::max(size_t(0), std::min(nValues - 1, size_t(floor(*pIn + 0.5)))), 3);
			pIn++;
			pOut++;
		}
	}
}
#endif // MIN_PREDICT

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
: GIncrementalTransform(), m_min(min), m_max(max), m_pMins(NULL), m_pRanges(NULL)
{
}

GNormalize::GNormalize(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll)
{
	m_min = pNode->field("min")->asDouble();
	m_max = pNode->field("max")->asDouble();
	size_t nAttrCount = before().size();
	m_pMins = new double[2 * nAttrCount];
	m_pRanges = &m_pMins[nAttrCount];
	GDomListIterator it1(pNode->field("mins"));
	if(it1.remaining() != nAttrCount)
		throw Ex("unexpected number of elements");
	GVec::deserialize(m_pMins, it1);
	GDomListIterator it2(pNode->field("ranges"));
	if(it2.remaining() != nAttrCount)
		throw Ex("unexpected number of elements");
	GVec::deserialize(m_pRanges, it2);
}

// virtual
GNormalize::~GNormalize()
{
	delete[] m_pMins;
}

// virtual
GDomNode* GNormalize::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNormalize");
	pNode->addField(pDoc, "min", pDoc->newDouble(m_min));
	pNode->addField(pDoc, "max", pDoc->newDouble(m_max));
	size_t nAttrCount = before().size();
	pNode->addField(pDoc, "mins", GVec::serialize(pDoc, m_pMins, nAttrCount));
	pNode->addField(pDoc, "ranges", GVec::serialize(pDoc, m_pRanges, nAttrCount));
	return pNode;
}

void GNormalize::setMinsAndRanges(const GRelation& rel, const double* pMins, const double* pRanges)
{
	setBefore(rel.clone());
	setAfter(rel.clone());
	size_t nAttrCount = before().size();
	delete[] m_pMins;
	m_pMins = new double[2 * nAttrCount];
	m_pRanges = &m_pMins[nAttrCount];
	GVec::copy(m_pMins, pMins, nAttrCount);
	GVec::copy(m_pRanges, pRanges, nAttrCount);
}

// virtual
GRelation* GNormalize::trainInner(const GMatrix& data)
{
	size_t nAttrCount = before().size();
	delete[] m_pMins;
	m_pMins = new double[2 * nAttrCount];
	m_pRanges = &m_pMins[nAttrCount];
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			m_pMins[i] = data.columnMin(i);
			if(m_pMins[i] >= 1e300)
			{
				m_pMins[i] = 0.0;
				m_pRanges[i] = 1.0;
			}
			else
			{
				m_pRanges[i] = data.columnMax(i) - m_pMins[i];
				if(m_pRanges[i] < 1e-12)
					m_pRanges[i] = 1.0;
			}
		}
		else
		{
			m_pMins[i] = 0;
			m_pRanges[i] = 0;
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
void GNormalize::transform(const double* pIn, double* pOut)
{
	size_t nAttrCount = before().size();
	double* pMins = m_pMins;
	double* pRanges = m_pRanges;
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(*pIn == UNKNOWN_REAL_VALUE)
				*pOut = UNKNOWN_REAL_VALUE;
			else
				*pOut = GMatrix::normalizeValue(*pIn, *pMins, *pMins + *pRanges, m_min, m_max);
		}
		else
			*pOut = *pIn;
		pOut++;
		pIn++;
		pMins++;
		pRanges++;
	}
}

// virtual
void GNormalize::untransform(const double* pIn, double* pOut)
{
	size_t nAttrCount = before().size();
	double* pMins = m_pMins;
	double* pRanges = m_pRanges;
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(*pIn == UNKNOWN_REAL_VALUE)
				*pOut = UNKNOWN_REAL_VALUE;
			else
				*pOut = GMatrix::normalizeValue(*pIn, m_min, m_max, *pMins, *pMins + *pRanges);
		}
		else
			*pOut = *pIn;
GAssert(*pOut < 1e200);
		pOut++;
		pIn++;
		pMins++;
		pRanges++;
	}
}

// virtual
void GNormalize::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, cannot denormalize to a distribution");
}

// --------------------------------------------------------------------------

GDiscretize::GDiscretize(size_t buckets)
: GIncrementalTransform()
{
	m_bucketsIn = buckets;
	m_bucketsOut = -1;
	m_pMins = NULL;
	m_pRanges = NULL;
}

GDiscretize::GDiscretize(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll)
{
	m_bucketsIn = (size_t)pNode->field("bucketsIn")->asInt();
	m_bucketsOut = (size_t)pNode->field("bucketsOut")->asInt();
	size_t nAttrCount = before().size();
	m_pMins = new double[2 * nAttrCount];
	m_pRanges = &m_pMins[nAttrCount];
	GDomListIterator it1(pNode->field("mins"));
	if(it1.remaining() != nAttrCount)
		throw Ex("unexpected number of elements");
	GVec::deserialize(m_pMins, it1);
	GDomListIterator it2(pNode->field("ranges"));
	if(it2.remaining() != nAttrCount)
		throw Ex("unexpected number of elements");
	GVec::deserialize(m_pRanges, it2);
}

// virtual
GDiscretize::~GDiscretize()
{
	delete[] m_pMins;
}

// virtual
GDomNode* GDiscretize::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GDiscretize");
	pNode->addField(pDoc, "bucketsIn", pDoc->newInt(m_bucketsIn));
	pNode->addField(pDoc, "bucketsOut", pDoc->newInt(m_bucketsOut));
	size_t nAttrCount = before().size();
	pNode->addField(pDoc, "mins", GVec::serialize(pDoc, m_pMins, nAttrCount));
	pNode->addField(pDoc, "ranges", GVec::serialize(pDoc, m_pRanges, nAttrCount));
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
	delete[] m_pMins;
	m_pMins = new double[2 * nAttrCount];
	m_pRanges = &m_pMins[nAttrCount];
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
		{
			m_pMins[i] = 0;
			m_pRanges[i] = 0;
		}
		else
		{
			m_pMins[i] = data.columnMin(i);
			m_pRanges[i] = data.columnMax(i) - m_pMins[i];
			m_pRanges[i] = std::max(m_pRanges[i], 1e-9);
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
void GDiscretize::transform(const double* pIn, double* pOut)
{
	if(!m_pMins)
		throw Ex("Train was not called");
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
			pOut[i] = pIn[i];
		else
			pOut[i] = std::max(0, std::min((int)(m_bucketsOut - 1), (int)(((pIn[i] - m_pMins[i]) * m_bucketsOut) / m_pRanges[i])));
	}
}

// virtual
void GDiscretize::untransform(const double* pIn, double* pOut)
{
	if(!m_pMins)
		throw Ex("Train was not called");
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
			pOut[i] = pIn[i];
		else
			pOut[i] = (((double)pIn[i] + .5) * m_pRanges[i]) / m_bucketsOut + m_pMins[i];
	}
}

// virtual
void GDiscretize::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	if(!m_pMins)
		throw Ex("Train was not called");
	size_t attrCount = before().size();
	for(size_t i = 0; i < attrCount; i++)
	{
		size_t nValues = before().valueCount(i);
		if(nValues > 0)
			pOut[i].makeCategorical()->setSpike(nValues, (size_t)pIn[i], 1);
		else
			pOut[i].makeNormal()->setMeanAndVariance((((double)pIn[i] + .5) * m_pRanges[i]) / m_bucketsOut + m_pMins[i], m_pRanges[i] * m_pRanges[i]);
	}
}




#ifndef MIN_PREDICT

GImputeMissingVals::GImputeMissingVals()
: m_pCF(NULL), m_pNTC(NULL), m_pLabels(NULL), m_pBatch(NULL)
{
}

GImputeMissingVals::GImputeMissingVals(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll), m_pLabels(NULL), m_pBatch(NULL)
{
	m_pCF = ll.loadCollaborativeFilter(pNode->field("cf"));
	GDomNode* pNTC = pNode->fieldIfExists("ntc");
	if(pNTC)
		m_pNTC = new GNominalToCat(pNTC, ll);
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
	pNode->addField(pDoc, "cf", m_pCF->serialize(pDoc));
	if(m_pNTC)
		pNode->addField(pDoc, "ntc", m_pNTC->serialize(pDoc));
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
	Holder<GMatrix> hData;
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
void GImputeMissingVals::transform(const double* pIn, double* pOut)
{
	// If there are no missing values, just copy it across
	const double* p = pIn;
	size_t dims = before().size();
	size_t i;
	for(i = 0; i < dims; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(*p == UNKNOWN_REAL_VALUE)
				break;
		}
		else
		{
			if(*p == UNKNOWN_DISCRETE_VALUE)
				break;
		}
		p++;
	}
	if(i >= dims)
	{
		GVec::copy(pOut, pIn, dims);
		return;
	}

	// Convert to all real values if necessary
	double* pVec;
	if(m_pNTC)
	{
		pVec = m_pNTC->innerBuf();
		m_pNTC->transform(pIn, pVec);
		dims = m_pNTC->after().size();
	}
	else
	{
		pVec = pOut;
		GVec::copy(pVec, pIn, dims);
	}

	// Impute the missing values
	m_pCF->impute(pVec, dims);

	// Convert back to nominal if necessary
	if(m_pNTC)
		m_pNTC->untransform(pVec, pOut);
}

// virtual
void GImputeMissingVals::untransform(const double* pIn, double* pOut)
{
	GVec::copy(pOut, pIn, after().size());
}

// virtual
void GImputeMissingVals::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, cannot unimpute to a distribution");
}

// virtual
GMatrix* GImputeMissingVals::transformBatch(const GMatrix& in)
{
	GMatrix* pOut = new GMatrix();
	pOut->copy(&in);
	size_t dims = pOut->cols();
	for(size_t i = 0; i < pOut->rows(); i++)
	{
		double* pVec = pOut->row(i);
		for(size_t j = 0; j < dims; j++)
		{
			if(*pVec == UNKNOWN_REAL_VALUE)
				*pVec = m_pCF->predict(i, j);
			pVec++;
		}
	}
	return pOut;
}

void GImputeMissingVals::setLabels(const GMatrix* pLabels)
{
	m_pLabels = pLabels;
}

#endif // MIN_PREDICT

// --------------------------------------------------------------------------

GLogify::GLogify()
: GIncrementalTransform()
{
}

GLogify::GLogify(GDomNode* pNode, GLearnerLoader& ll)
: GIncrementalTransform(pNode, ll)
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
void GLogify::transform(const double* pIn, double* pOut)
{
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(*pIn == UNKNOWN_REAL_VALUE)
				*pOut = UNKNOWN_REAL_VALUE;
			else
				*pOut = log(*pIn);
		}
		else
			*pOut = *pIn;
		pOut++;
		pIn++;
	}
}

// virtual
void GLogify::untransform(const double* pIn, double* pOut)
{
	size_t nAttrCount = before().size();
	for(size_t i = 0; i < nAttrCount; i++)
	{
		if(before().valueCount(i) == 0)
		{
			if(*pIn == UNKNOWN_REAL_VALUE)
				*pOut = UNKNOWN_REAL_VALUE;
			else
				*pOut = exp(*pIn);
		}
		else
			*pOut = *pIn;
		pOut++;
		pIn++;
	}
}

// virtual
void GLogify::untransformToDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, cannot unlogify to a distribution");
}



} // namespace GClasses

