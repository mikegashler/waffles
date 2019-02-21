/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include "GNaiveInstance.h"
#include "GVec.h"
#include "GDom.h"
#include "GDistribution.h"
#include "GTransform.h"
#include <map>

using std::multimap;
using std::make_pair;

namespace GClasses {

class GNaiveInstanceAttr
{
protected:
	multimap<double,const double*> m_instances;

public:
	GNaiveInstanceAttr() {}

	GNaiveInstanceAttr(GDomNode* pAttr, size_t labelDims, GHeap* pHeap)
	{
		GDomListIterator it(pAttr);
		size_t count = it.remaining() / (1 + labelDims);
		if(count * (1 + labelDims) != it.remaining())
			throw Ex("invalid list size");
		for(size_t i = 0; i < count; i++)
		{
			double d = it.currentDouble();
			it.advance();
			double* pLabel = (double*)pHeap->allocAligned(sizeof(double) * labelDims);
			m_instances.insert(make_pair(d, pLabel));
			for(size_t j = 0; j < labelDims; j++)
			{
				*(pLabel++) = it.currentDouble();
				it.advance();
			}
		}
	}

	virtual ~GNaiveInstanceAttr()
	{
	}

	multimap<double,const double*>& instances() { return m_instances; }

	GDomNode* serialize(GDom* pDoc, size_t labelDims)
	{
		GDomNode* pList = pDoc->newList();
		for(multimap<double,const double*>::iterator it = m_instances.begin(); it != m_instances.end(); it++)
		{
			pList->add(pDoc, it->first);
			for(size_t i = 0; i < labelDims; i++)
				pList->add(pDoc, it->second[i]);
		}
		return pList;
	}

	void addInstance(double dInput, const double* pOutputs)
	{
		m_instances.insert(make_pair(dInput, pOutputs));
	}
};

// -----------------------------------------------------------

GNaiveInstance::GNaiveInstance()
: GIncrementalLearner(), m_pHeap(NULL)
{
	m_nNeighbors = 12;
	m_pAttrs = NULL;
}

GNaiveInstance::GNaiveInstance(const GDomNode* pNode)
: GIncrementalLearner(pNode), m_pHeap(NULL)
{
	m_pAttrs = NULL;
	m_nNeighbors = (size_t)pNode->getInt("neighbors");
	beginIncrementalLearningInner(*m_pRelFeatures, *m_pRelLabels);
	GDomNode* pAttrs = pNode->get("attrs");
	GDomListIterator it(pAttrs);
	if(it.remaining() != m_pRelFeatures->size())
		throw Ex("Expected ", to_str(m_pRelFeatures->size()), " attrs, got ", to_str(it.remaining()), " attrs");
	m_pHeap = new GHeap(1024);
	for(size_t i = 0; i < m_pRelFeatures->size(); i++)
	{
		delete(m_pAttrs[i]);
		m_pAttrs[i] = new GNaiveInstanceAttr(it.current(), m_pRelLabels->size(), m_pHeap);
		it.advance();
	}
}

// virtual
GNaiveInstance::~GNaiveInstance()
{
	clear();
}

void GNaiveInstance::clear()
{
	if(m_pAttrs)
	{
		for(size_t i = 0; i < m_pRelFeatures->size(); i++)
			delete(m_pAttrs[i]);
		delete[] m_pAttrs;
	}
	m_pAttrs = NULL;
	m_pValueSums.resize(0);
	delete(m_pHeap);
	m_pHeap = NULL;
}

// virtual
GDomNode* GNaiveInstance::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GNaiveInstance");
	pNode->add(pDoc, "neighbors", m_nNeighbors);
	GDomNode* pAttrs = pNode->add(pDoc, "attrs", pDoc->newList());
	for(size_t i = 0; i < m_pRelFeatures->size(); i++)
		pAttrs->add(pDoc, m_pAttrs[i]->serialize(pDoc, m_pRelLabels->size()));
	return pNode;
}

void GNaiveInstance::autoTune(GMatrix& features, GMatrix& labels)
{
	// Find the best ess value
	size_t bestK = 0;
	double bestErr = 1e308;
	size_t cap = size_t(floor(sqrt(double(features.rows()))));
	for(size_t i = 2; i < cap; i = size_t(i * 1.5))
	{
		m_nNeighbors = i;
		double d = crossValidate(features, labels, 2);
		if(d < bestErr)
		{
			bestErr = d;
			bestK = i;
		}
		else if(i >= 15)
			break;
	}

	// Set the best values
	m_nNeighbors = bestK;
}

// virtual
void GNaiveInstance::beginIncrementalLearningInner(const GRelation& featureRel, const GRelation& labelRel)
{
	if(!featureRel.areContinuous() || !labelRel.areContinuous())
		throw Ex("Only continuous attributes are supported.");
	clear();
	m_pAttrs = new GNaiveInstanceAttr*[m_pRelFeatures->size()];
	for(size_t i = 0; i < m_pRelFeatures->size(); i++)
		m_pAttrs[i] = new GNaiveInstanceAttr();
	m_pValueSums.resize(m_pRelLabels->size());
	m_pWeightSums.resize(m_pRelLabels->size());
	m_pSumBuffer.resize(m_pRelLabels->size());
	m_pSumOfSquares.resize(m_pRelLabels->size());
}

// virtual
void GNaiveInstance::trainIncremental(const GVec& pIn, const GVec& pOut)
{
	if(!m_pHeap)
		m_pHeap = new GHeap(1024);
	double* pOutputs = (double*)m_pHeap->allocAligned(sizeof(double) * m_pRelLabels->size());
	memcpy(pOutputs, pOut.data(), sizeof(double) * m_pRelLabels->size());
	for(size_t i = 0; i < m_pRelFeatures->size(); i++)
	{
		if(pIn[i] != UNKNOWN_REAL_VALUE)
			m_pAttrs[i]->addInstance(pIn[i], pOutputs);
	}
}

// virtual
void GNaiveInstance::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(!features.relation().areContinuous())
		throw Ex("GNaiveInstance only supports continuous features. Perhaps you should wrap it in a GAutoFilter.");
	if(!labels.relation().areContinuous())
		throw Ex("GNaiveInstance only supports continuous labels. Perhaps you should wrap it in a GAutoFilter.");

	beginIncrementalLearningInner(features.relation(), labels.relation());
	for(size_t i = 0; i < features.rows(); i++)
		trainIncremental(features[i], labels[i]);
}

// virtual
void GNaiveInstance::trainSparse(GSparseMatrix& features, GMatrix& labels)
{
	throw Ex("Sorry, trainSparse is not implemented yet in GNaiveInstance");
}

void GNaiveInstance::evalInput(size_t nInputDim, double dInput)
{
	// Init the accumulators
	m_pSumBuffer.fill(0.0);
	m_pSumOfSquares.fill(0.0);

	// Find the nodes on either side of dInput
	GNaiveInstanceAttr* pAttr = m_pAttrs[nInputDim];
	multimap<double,const double*>& instances = pAttr->instances();
	multimap<double,const double*>::iterator itLeft = instances.lower_bound(dInput);
	multimap<double,const double*>::iterator itRight = itLeft;
	bool leftValid = true;
	if(itLeft == instances.end())
	{
		if(instances.size() > 0)
			itLeft--;
		else
			leftValid = false;
	}
	else
		itRight++;

	// Compute the mean and variance of the values for the k-nearest neighbors
	size_t nNeighbors = 0;
	bool goRight;
	while(true)
	{
		// Pick the closer of the two nodes
		if(!leftValid)
		{
			if(itRight == instances.end())
				break;
			goRight = true;
		}
		else if(itRight == instances.end())
			goRight = false;
		else if(dInput - itLeft->first < itRight->first - dInput)
			goRight = false;
		else
			goRight = true;

		// Accumulate values
		const double* pOutputVec = goRight ? itRight->second : itLeft->second;
		GConstVecWrapper vw(pOutputVec, m_pSumBuffer.size());
		m_pSumBuffer += vw;
		for(size_t j = 0; j < m_pRelLabels->size(); j++)
			m_pSumOfSquares[j] += (pOutputVec[j] * pOutputVec[j]);

		// See if we're done
		if(++nNeighbors >= m_nNeighbors)
			break;

		// Advance
		if(goRight)
			itRight++;
		else
		{
			if(itLeft == instances.begin())
				leftValid = false;
			else
				itLeft--;
		}
	}
	m_pSumBuffer *= (1.0 / nNeighbors);
	m_pSumOfSquares *= (1.0 / nNeighbors);

	// Accumulate the predictions across all dimensions
	int dims = 0;
	double weight;
	for(size_t i = 0; i < m_pRelLabels->size(); i++)
	{
		weight = 1.0 / std::max(m_pSumOfSquares[i] - (m_pSumBuffer[i] * m_pSumBuffer[i]), 1e-5);
		m_pWeightSums[dims] += weight;
		m_pValueSums[dims] += weight * m_pSumBuffer[dims];
		dims++;
	}
}

// virtual
void GNaiveInstance::predictDistribution(const GVec& pIn, GPrediction* pOut)
{
	m_pWeightSums.fill(0.0);
	m_pValueSums.fill(0.0);
	for(size_t i = 0; i < m_pRelFeatures->size(); i++)
		evalInput(i, pIn[i]);
	for(size_t i = 0; i < m_pRelLabels->size(); i++)
	{
		GNormalDistribution* pNorm = pOut[i].makeNormal();
		pNorm->setMeanAndVariance(m_pValueSums[i] / m_pWeightSums[i], 1.0 / m_pWeightSums[i]);
	}
}

// virtual
void GNaiveInstance::predict(const GVec& pIn, GVec& pOut)
{
	m_pWeightSums.fill(0.0);
	m_pValueSums.fill(0.0);
	for(size_t i = 0; i < m_pRelFeatures->size(); i++)
		evalInput(i, pIn[i]);
	for(size_t i = 0; i < m_pRelLabels->size(); i++)
		pOut[i] = m_pValueSums[i] / m_pWeightSums[i];
}

//static
void GNaiveInstance::test()
{

	GNaiveInstance* pNI = new GNaiveInstance();
	GAutoFilter af(pNI);
	pNI->setNeighbors(8);
	af.basicTest(0.72, 0.44, 0.02);
}

} // namespace GClasses
