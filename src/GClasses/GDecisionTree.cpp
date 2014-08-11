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

#include "GDecisionTree.h"
#include "GError.h"
#include <stdlib.h>
#include "GVec.h"
#include "GPolynomial.h"
#include "GHillClimber.h"
#include "GDistribution.h"
#include "GRand.h"
#include "GDom.h"
#include "GTransform.h"
#include "GEnsemble.h"
#include "GHolders.h"
#include <string>
#include <iostream>

using namespace GClasses;
using std::string;
using std::ostream;
using std::ostringstream;
using std::vector;

namespace GClasses {

class GDecisionTreeNode
{
public:
	GDecisionTreeNode()
	{
	}

	virtual ~GDecisionTreeNode()
	{
	}

	virtual bool IsLeaf() = 0;
	virtual size_t GetBranchSize() = 0;
	virtual GDecisionTreeNode* DeepCopy(size_t nOutputCount, GDecisionTreeNode* pInterestingNode, GDecisionTreeNode** ppOutInterestingCopy) = 0;
	virtual void print(GDecisionTree* pTree, ostream& stream, vector<char>& prefix, const char* parentValue) = 0;
	virtual void CountValues(size_t nOutput, size_t* pnCounts) = 0;
	virtual double FindSumOutputValue(size_t nOutput) = 0;
	static GDecisionTreeNode* deserialize(GDomNode* pNode);
	virtual GDomNode* serialize(GDom* pDoc, size_t outputCount) = 0;
};

class GDecisionTreeInteriorNode : public GDecisionTreeNode
{
friend class GDecisionTree;
protected:
	size_t m_nAttribute;
	double m_dPivot;
	size_t m_nChildren;
	size_t m_defaultChild;
	GDecisionTreeNode** m_ppChildren;

public:
	GDecisionTreeInteriorNode(size_t nAttribute, double dPivot, size_t children, size_t defaultChild)
	: GDecisionTreeNode(), m_nAttribute(nAttribute), m_dPivot(dPivot), m_nChildren(children), m_defaultChild(defaultChild)
	{
		m_ppChildren = new GDecisionTreeNode*[children];
		memset(m_ppChildren, '\0', sizeof(GDecisionTreeNode*) * children);
	}

	GDecisionTreeInteriorNode(GDomNode* pNode) : GDecisionTreeNode()
	{
		m_nAttribute = (size_t)pNode->field("attr")->asInt();
		m_dPivot = pNode->field("pivot")->asDouble();
		GDomNode* pChildren = pNode->field("children");
		GDomListIterator it(pChildren);
		m_nChildren = it.remaining();
		m_ppChildren = new GDecisionTreeNode*[m_nChildren];
		for(size_t i = 0 ; it.current(); i++)
		{
			m_ppChildren[i] = GDecisionTreeNode::deserialize(it.current());
			it.advance();
		}
		m_defaultChild = (size_t)pNode->field("def")->asInt();
	}

	virtual ~GDecisionTreeInteriorNode()
	{
		if(m_ppChildren)
		{
			for(size_t n = 0; n < m_nChildren; n++)
				delete(m_ppChildren[n]);
			delete[] m_ppChildren;
		}
	}

	virtual GDomNode* serialize(GDom* pDoc, size_t outputCount)
	{
		GDomNode* pNode = pDoc->newObj();
		pNode->addField(pDoc, "attr", pDoc->newInt(m_nAttribute));
		pNode->addField(pDoc, "pivot", pDoc->newDouble(m_dPivot));
		GDomNode* pChildren = pDoc->newList();
		pNode->addField(pDoc, "children", pChildren);
		for(size_t i = 0; i < m_nChildren; i++)
			pChildren->addItem(pDoc, m_ppChildren[i]->serialize(pDoc, outputCount));
		pNode->addField(pDoc, "def", pDoc->newInt(m_defaultChild));
		return pNode;
	}

	virtual bool IsLeaf() { return false; }

	virtual size_t GetBranchSize()
	{
		size_t size = 1;
		size_t i;
		for(i = 0; i < m_nChildren; i++)
			size += m_ppChildren[i]->GetBranchSize();
		return size;
	}

	virtual GDecisionTreeNode* DeepCopy(size_t nOutputCount, GDecisionTreeNode* pInterestingNode, GDecisionTreeNode** ppOutInterestingCopy)
	{
		GDecisionTreeInteriorNode* pNewNode = new GDecisionTreeInteriorNode(m_nAttribute, m_dPivot, m_nChildren, m_defaultChild);
		for(size_t n = 0; n < m_nChildren; n++)
			pNewNode->m_ppChildren[n] = m_ppChildren[n]->DeepCopy(nOutputCount, pInterestingNode, ppOutInterestingCopy);
		if(this == pInterestingNode)
			*ppOutInterestingCopy = pNewNode;
		return pNewNode;
	}

	virtual void print(GDecisionTree* pTree, ostream& stream, vector<char>& prefix, const char* parentValue)
	{
		for(size_t n = 0; n + 1 < prefix.size(); n++)
			stream << prefix[n];
		stream << "|\n";
		for(size_t n = 0; n + 1 < prefix.size(); n++)
			stream << prefix[n];
		if(parentValue)
			stream << "+" << parentValue << "->";
		if(pTree->relFeatures().valueCount(m_nAttribute) == 0)
		{
			ostringstream oss;
			pTree->relFeatures().printAttrValue(oss, m_nAttribute, m_dPivot);
			string s = oss.str();
			if(pTree->relFeatures().type() == GRelation::ARFF)
				stream << "Is " << ((GArffRelation&)pTree->relFeatures()).attrName(m_nAttribute) << " < " << s.c_str() << "?\n";
			else
				stream << "Is attr " << to_str(m_nAttribute) << " < " << s.c_str() << "?\n";
			if(m_nChildren != 2)
				throw Ex("expected this node to have two child nodes");
			prefix.push_back(' ');
			prefix.push_back(' ');
			prefix.push_back(' ');
			prefix.push_back('|');
			m_ppChildren[0]->print(pTree, stream, prefix, "Yes");
			prefix.pop_back();
			prefix.push_back(' ');
			m_ppChildren[1]->print(pTree, stream, prefix, "No");
			prefix.pop_back();
			prefix.pop_back();
			prefix.pop_back();
			prefix.pop_back();
		}
		else if(pTree->isBinary())
		{
			ostringstream oss;
			pTree->relFeatures().printAttrValue(oss, m_nAttribute, m_dPivot);
			string s = oss.str();
			if(pTree->relFeatures().type() == GRelation::ARFF)
				stream << "Is " << ((GArffRelation&)pTree->relFeatures()).attrName(m_nAttribute) << " == " << s.c_str() << "?\n";
			else
				stream << "Is attr " << to_str(m_nAttribute) << " == " << s.c_str() << "?\n";
			if(m_nChildren != 2)
				throw Ex("expected this node to have two child nodes");
			prefix.push_back(' ');
			prefix.push_back(' ');
			prefix.push_back(' ');
			prefix.push_back('|');
			m_ppChildren[0]->print(pTree, stream, prefix, "Yes");
			prefix.pop_back();
			prefix.push_back(' ');
			m_ppChildren[1]->print(pTree, stream, prefix, "No");
			prefix.pop_back();
			prefix.pop_back();
			prefix.pop_back();
			prefix.pop_back();
		}
		else
		{
			if(pTree->relFeatures().type() == GRelation::ARFF)
				stream << ((GArffRelation&)pTree->relFeatures()).attrName(m_nAttribute) << "?\n";
			else
				stream << "Attr " << to_str(m_nAttribute) << "?\n";
			prefix.push_back(' ');
			prefix.push_back(' ');
			prefix.push_back(' ');
			prefix.push_back('|');
			for(size_t n = 0; n < m_nChildren; n++)
			{
				if(n + 1 == m_nChildren)
				{
					prefix.pop_back();
					prefix.push_back(' ');
				}
				ostringstream oss;
				pTree->relFeatures().printAttrValue(oss, m_nAttribute, (double)n);
				string s = oss.str();
				m_ppChildren[n]->print(pTree, stream, prefix, s.c_str());
			}
			prefix.pop_back();
			prefix.pop_back();
			prefix.pop_back();
			prefix.pop_back();
		}
	}

	// Recursive function that counts the number of times a particular
	// value is found in a particular output in this branch of the tree
	virtual void CountValues(size_t nOutput, size_t* pnCounts)
	{
		for(size_t n = 0; n < m_nChildren; n++)
			m_ppChildren[n]->CountValues(nOutput, pnCounts);
	}

	virtual double FindSumOutputValue(size_t nOutput)
	{
		double dSum = 0;
		for(size_t n = 0; n < m_nChildren; n++)
			dSum += m_ppChildren[n]->FindSumOutputValue(nOutput);
		return dSum;
	}
};

class GDecisionTreeLeafNode : public GDecisionTreeNode
{
public:
	double* m_pOutputValues;
	size_t m_nSampleSize;

public:
	GDecisionTreeLeafNode(double* pOutputValues, size_t nSampleSize) : GDecisionTreeNode()
	{
		m_pOutputValues = pOutputValues;
		m_nSampleSize = nSampleSize;
	}

	GDecisionTreeLeafNode(GDomNode* pNode) : GDecisionTreeNode()
	{
		m_nSampleSize = (size_t)pNode->field("size")->asInt();
		GDomNode* pOut = pNode->field("out");
		GDomListIterator it(pOut);
		size_t count = it.remaining();
		m_pOutputValues = new double[count];
		for(size_t i = 0; it.current(); i++)
		{
			m_pOutputValues[i] = it.current()->asDouble();
			it.advance();
		}
	}

	virtual ~GDecisionTreeLeafNode()
	{
		delete[] m_pOutputValues;
	}

	virtual GDomNode* serialize(GDom* pDoc, size_t outputCount)
	{
		GDomNode* pNode = pDoc->newObj();
		pNode->addField(pDoc, "size", pDoc->newInt(m_nSampleSize));
		GDomNode* pOut = pDoc->newList();
		pNode->addField(pDoc, "out", pOut);
		for(size_t i = 0; i < outputCount; i++)
			pOut->addItem(pDoc, pDoc->newDouble(m_pOutputValues[i]));
		return pNode;
	}

	virtual bool IsLeaf() { return true; }

	virtual size_t GetBranchSize()
	{
		return 1;
	}

	virtual GDecisionTreeNode* DeepCopy(size_t nOutputCount, GDecisionTreeNode* pInterestingNode, GDecisionTreeNode** ppOutInterestingCopy)
	{
		double* pOutputValues = new double[nOutputCount];
		GVec::copy(pOutputValues, m_pOutputValues, nOutputCount);
		GDecisionTreeLeafNode* pNewNode = new GDecisionTreeLeafNode(pOutputValues, m_nSampleSize);
		if(this == pInterestingNode)
			*ppOutInterestingCopy = pNewNode;
		return pNewNode;
	}

	virtual void print(GDecisionTree* pTree, ostream& stream, vector<char>& prefix, const char* parentValue)
	{
		for(size_t n = 0; n + 1 < prefix.size(); n++)
			stream << prefix[n];
		stream << "|\n";
		for(size_t n = 0; n + 1 < prefix.size(); n++)
			stream << prefix[n];
		if(parentValue)
			stream << "+" << parentValue << "->";
		for(size_t n = 0; n < pTree->relLabels().size(); n++)
		{
			if(n > 0)
				stream << ", ";
			ostringstream oss;
			pTree->relLabels().printAttrValue(oss, n, m_pOutputValues[n]);
			string s = oss.str();
			if(pTree->relLabels().type() == GRelation::ARFF)
				stream << ((GArffRelation&)pTree->relLabels()).attrName(n) << "=" << s.c_str();
			else
				stream << s.c_str();
		}
		stream << "\n";
	}

	virtual void CountValues(size_t nOutput, size_t* pnCounts)
	{
		int nVal = (int)m_pOutputValues[nOutput];
		pnCounts[nVal] += m_nSampleSize;
	}

	virtual double FindSumOutputValue(size_t nOutput)
	{
		return m_pOutputValues[nOutput] * m_nSampleSize;
	}
};

}

// static
GDecisionTreeNode* GDecisionTreeNode::deserialize(GDomNode* pNode)
{
	if(pNode->fieldIfExists("children"))
		return new GDecisionTreeInteriorNode(pNode);
	else
		return new GDecisionTreeLeafNode(pNode);
}

// -----------------------------------------------------------------

GDecisionTree::GDecisionTree()
: GSupervisedLearner(), m_leafThresh(1), m_maxLevels(0), m_binaryDivisions(false)
{
	m_pRoot = NULL;
	m_eAlg = GDecisionTree::MINIMIZE_ENTROPY;
}

GDecisionTree::GDecisionTree(GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode, ll), m_leafThresh(1), m_maxLevels(0)
{
	m_eAlg = (DivisionAlgorithm)pNode->field("alg")->asInt();
	m_pRoot = GDecisionTreeNode::deserialize(pNode->field("root"));
	m_binaryDivisions = pNode->field("bin")->asBool();
}

// virtual
GDecisionTree::~GDecisionTree()
{
	clear();
}

// virtual
GDomNode* GDecisionTree::serialize(GDom* pDoc) const
{
	if(!m_pRoot)
		throw Ex("Attempted to serialize a model that has not been trained");
	GDomNode* pNode = baseDomNode(pDoc, "GDecisionTree");
	pNode->addField(pDoc, "alg", pDoc->newInt(m_eAlg));
	pNode->addField(pDoc, "root", m_pRoot->serialize(pDoc, m_pRelLabels->size()));
	pNode->addField(pDoc, "bin", pDoc->newBool(m_binaryDivisions));
	return pNode;
}

size_t GDecisionTree::treeSize()
{
	return m_pRoot->GetBranchSize();
}

void GDecisionTree::useBinaryDivisions()
{
	m_binaryDivisions = true;
	delete(m_pRoot);
	m_pRoot = NULL;
}

void GDecisionTree::print(ostream& stream, GArffRelation* pFeatureRel, GArffRelation* pLabelRel)
{
	if(!m_pRoot)
		throw Ex("not trained yet");
	if(pFeatureRel && pFeatureRel->type() == GRelation::ARFF)
	{
		if(!m_pRelFeatures->isCompatible(*pFeatureRel))
			throw Ex("Feature relation not compatible");
		delete(m_pRelFeatures);
		m_pRelFeatures = pFeatureRel->clone();
	}
	if(pLabelRel && pLabelRel->type() == GRelation::ARFF)
	{
		if(!m_pRelLabels->isCompatible(*pLabelRel))
			throw Ex("Label relation not compatible");
		delete(m_pRelLabels);
		m_pRelLabels = pLabelRel->clone();
	}
	vector<char> prefix;
	m_pRoot->print(this, stream, prefix, NULL);
}

// virtual
void GDecisionTree::trainInner(const GMatrix& features, const GMatrix& labels)
{
	clear();

	// Make a list of available features
	vector<size_t> attrPool;
	attrPool.reserve(m_pRelFeatures->size());
	for(size_t i = 0; i < m_pRelFeatures->size(); i++)
		attrPool.push_back(i);

	// Copy the data
	GMatrix tmpFeatures(m_pRelFeatures->clone());
	tmpFeatures.copy(&features);
	GMatrix tmpLabels(m_pRelLabels->clone());
	tmpLabels.copy(&labels);

	m_pRoot = buildBranch(tmpFeatures, tmpLabels, attrPool, 0/*depth*/, 4/*tolerance*/);
}

void GDecisionTree::autoTune(GMatrix& features, GMatrix& labels)
{
	// Try binary splits
	m_binaryDivisions = false;
	double bestErr = crossValidate(features, labels, 2);
	m_binaryDivisions = true;
	double d = crossValidate(features, labels, 2);
	if(d < bestErr)
		bestErr = d;
	else
		m_binaryDivisions = false;

	// Find the best leaf threshold
	size_t cap = size_t(floor(sqrt(double(features.rows()))));
	size_t bestLeafThresh = 1;
	for(size_t i = 2; i < cap; i = std::max(i + 1, size_t(i * 1.5)))
	{
		m_leafThresh = i;
		double d = crossValidate(features, labels, 2);
		if(d < bestErr)
		{
			bestErr = d;
			bestLeafThresh = i;
		}
		else if(i >= 27)
			break;
	}

	// Set the best values
	m_maxLevels = 0;
	m_leafThresh = bestLeafThresh;
}

double GDecisionTree_measureRealSplitInfo(GMatrix& features, GMatrix& labels, GMatrix& tmpFeatures, GMatrix& tmpLabels, size_t attr, double pivot)
{
	GAssert(tmpFeatures.rows() == 0 && tmpLabels.rows() == 0);
	if(pivot == UNKNOWN_REAL_VALUE)
		return 1e308;
	size_t rowCount = features.rows();
	features.splitByPivot(&tmpFeatures, attr, pivot, &labels, &tmpLabels);
	double dInfo;
	if(features.rows() > 0 && tmpFeatures.rows() > 0)
		dInfo = (labels.measureInfo() * labels.rows() + tmpLabels.measureInfo() * tmpLabels.rows()) / rowCount;
	else
		dInfo = 1e308;
	features.mergeVert(&tmpFeatures);
	labels.mergeVert(&tmpLabels);
	return dInfo;
}

double GDecisionTree_measureBinarySplitInfo(GMatrix& features, GMatrix& labels, GMatrix& tmpFeatures, GMatrix& tmpLabels, size_t attr, int pivot)
{
	GAssert(tmpFeatures.rows() == 0 && tmpLabels.rows() == 0);
	size_t rowCount = features.rows();
	features.splitCategoricalKeepIfEqual(&tmpFeatures, attr, pivot, &labels, &tmpLabels);
	double dInfo;
	if(features.rows() > 0 && tmpFeatures.rows() > 0)
		dInfo = (labels.measureInfo() * labels.rows() + tmpLabels.measureInfo() * tmpLabels.rows()) / rowCount;
	else
		dInfo = 1e308;
	features.mergeVert(&tmpFeatures);
	labels.mergeVert(&tmpLabels);
	return dInfo;
}

double GDecisionTree_pickPivotToReduceInfo(GMatrix& features, GMatrix& labels, GMatrix& tmpFeatures, GMatrix& tmpLabels, double* pPivot, size_t attr, GRand* pRand)
{
	double bestPivot = UNKNOWN_REAL_VALUE;
	double bestInfo = 1e100;
	size_t vals = features.relation().valueCount(attr);
	if(vals == 0)
	{
		size_t attempts = std::min(features.rows() - 1, (features.rows() * features.cols() > 100000 ? (size_t)1 : (size_t)8));
		for(size_t n = 0; n < attempts; n++)
		{
			double* pRow1 = features.row((size_t)pRand->next(features.rows()));
			double* pRow2 = features.row((size_t)pRand->next(features.rows()));
			if(pRow1[attr] == UNKNOWN_REAL_VALUE || pRow2[attr] == UNKNOWN_REAL_VALUE)
				continue;
			double pivot = 0.5 * (pRow1[attr] + pRow2[attr]);
			double info = GDecisionTree_measureRealSplitInfo(features, labels, tmpFeatures, tmpLabels, attr, pivot);
			if(info + 1e-14 < bestInfo) // the small value makes it deterministic across hardware
			{
				bestInfo = info;
				bestPivot = pivot;
			}
		}
	}
	else
	{
		for(int i = 0; (size_t)i < vals; i++)
		{
			double info = GDecisionTree_measureBinarySplitInfo(features, labels, tmpFeatures, tmpLabels, attr, i);
			if(info + 1e-14 < bestInfo) // the small value makes it deterministic across hardware
			{
				bestInfo = info;
				bestPivot = (double)i;
			}
		}
	}
	*pPivot = bestPivot;
	return bestInfo;
}

double GDecisionTree_measureNominalSplitInfo(GMatrix& features, GMatrix& labels, GMatrix& tmpFeatures, GMatrix& tmpLabels, size_t nAttribute)
{
	size_t nRowCount = features.rows() - features.countValue(nAttribute, UNKNOWN_DISCRETE_VALUE);
	int values = (int)features.relation().valueCount(nAttribute);
	double dInfo = 0;
	for(int n = 0; n < values; n++)
	{
		features.splitCategoricalKeepIfEqual(&tmpFeatures, nAttribute, n, &labels, &tmpLabels);
		dInfo += ((double)tmpLabels.rows() / nRowCount) * tmpLabels.measureInfo();
		features.mergeVert(&tmpFeatures);
		labels.mergeVert(&tmpLabels);
	}
	return dInfo;
}

size_t GDecisionTree::pickDivision(GMatrix& features, GMatrix& labels, double* pPivot, vector<size_t>& attrPool, size_t nDepth)
{
	GMatrix tmpFeatures(features.relation().clone());
	tmpFeatures.reserve(features.rows());
	GMatrix tmpLabels(labels.relation().clone());
	tmpLabels.reserve(features.rows());
	if(m_eAlg == MINIMIZE_ENTROPY)
	{
		// Pick the best attribute to divide on
		GAssert(labels.rows() > 0); // Can't work without data
		double bestInfo = 1e100;
		double pivot = 0.0;
		double bestPivot = 0;
		size_t index = 0;
		size_t bestIndex = attrPool.size();
		for(vector<size_t>::iterator it = attrPool.begin(); it != attrPool.end(); it++)
		{
			double info;
			if(m_binaryDivisions || features.relation().valueCount(*it) == 0)
				info = GDecisionTree_pickPivotToReduceInfo(features, labels, tmpFeatures, tmpLabels, &pivot, *it, &m_rand);
			else
				info = GDecisionTree_measureNominalSplitInfo(features, labels, tmpFeatures, tmpLabels, *it);
			if(info + 1e-14 < bestInfo) // the small value makes it deterministic across hardware
			{
				bestInfo = info;
				bestIndex = index;
				bestPivot = pivot;
			}
			index++;
		}
		*pPivot = bestPivot;
		return bestIndex;
	}
	else if(m_eAlg == RANDOM)
	{
		// Pick the best of m_randomDraws random attributes from the attribute pool
		GAssert(features.rows() > 0); // Can't work without data
		double bestInfo = 1e200;
		double bestPivot = 0;
		size_t bestIndex = attrPool.size();
		size_t patience = std::max((size_t)6, m_randomDraws * 2);
		for(size_t i = 0; i < m_randomDraws && patience > 0; i++)
		{
			size_t index = (size_t)m_rand.next(attrPool.size());
			size_t attr = attrPool[index];
			double pivot = 0.0;
			double info;
			if(features.relation().valueCount(attr) == 0)
			{
				double a = features[(size_t)m_rand.next(features.rows())][attr];
				double b = features[(size_t)m_rand.next(features.rows())][attr];
				if(a == UNKNOWN_REAL_VALUE)
				{
					if(b == UNKNOWN_REAL_VALUE)
						pivot = features.columnMedian(attr, false);
					else
						pivot = b;
				}
				else
				{
					if(b == UNKNOWN_REAL_VALUE)
						pivot = a;
					else
						pivot = 0.5 * (a + b);
				}
				if(m_randomDraws > 1)
					info = GDecisionTree_measureRealSplitInfo(features, labels, tmpFeatures, tmpLabels, attr, pivot);
				else
					info = 0.0;
			}
			else if(m_binaryDivisions)
			{
				pivot = features[(size_t)m_rand.next(features.rows())][attr];
				if(pivot == UNKNOWN_DISCRETE_VALUE)
					pivot = features.baselineValue(attr);
				if(m_randomDraws > 1)
					info = GDecisionTree_measureBinarySplitInfo(features, labels, tmpFeatures, tmpLabels, attr, (int)pivot);
				else
					info = 0.0;
			}
			else
			{
				if(m_randomDraws > 1)
					info = GDecisionTree_measureNominalSplitInfo(features, labels, tmpFeatures, tmpLabels, attr);
				else
					info = 0.0;
			}
			if(info + 1e-14 < bestInfo) // the small value makes it deterministic across hardware
			{
				bestInfo = info;
				bestIndex = index;
				bestPivot = pivot;
			}
		}
		if(bestIndex < attrPool.size() && !features.isAttrHomogenous(attrPool[bestIndex]))
		{
			*pPivot = bestPivot;
			return bestIndex;
		}

		// We failed to find a useful attribute with random draws. (This may happen if there is a large
		// ratio of homogeneous attributes.) Now, we need to be a little more systematic about finding a good
		// attribute. (This is not specified in the random forest algorithm, but it can make a big difference
		// with some problems.)
		size_t k = (size_t)m_rand.next(attrPool.size());
		for(size_t i = 0; i < attrPool.size(); i++)
		{
			size_t index = (i + k) % attrPool.size();
			size_t attr = attrPool[index];
			if(features.relation().valueCount(attr) == 0)
			{
				// Find the min
				double m = 1e300;
				for(size_t j = 0; j < features.rows(); j++)
				{
					double d = features[j][attr];
					if(d != UNKNOWN_REAL_VALUE)
						m = std::min(m, d);
				}

				// Randomly pick one of the non-min values
				size_t candidates = 0;
				for(size_t j = 0; j < features.rows(); j++)
				{
					double d = features[j][attr];
					if(d != UNKNOWN_REAL_VALUE && d > m)
					{
						if(m_rand.next(++candidates) == 0)
							*pPivot = d;
					}
				}
				if(candidates == 0)
					continue; // This attribute is worthless
			}
			else
			{
				if(features.isAttrHomogenous(attr))
					continue; // This attribute is worthless
			}
			return index;
		}
	}
	else
		GAssert(false); // unknown division algorithm
	return attrPool.size();
}

double* GDecisionTreeNode_labelVec(GMatrix& labels)
{
	size_t n = labels.cols();
	double* pVec = new double[n];
	for(size_t i = 0; i < n; i++)
	{
		pVec[i] = labels.baselineValue(i);
		if(pVec[i] == UNKNOWN_REAL_VALUE) // This can happen if there are no known values in a real column
			pVec[i] = 0.0;
	}
	return pVec;
}

double* GDecisionTreeNode_copyIfNotTheLast(size_t emptySets, ArrayHolder<double>& hBaselineVec, size_t dims)
{
	if(emptySets > 1)
	{
		double* pCopy = new double[dims];
		GVec::copy(pCopy, hBaselineVec.get(), dims);
		return pCopy;
	}
	else
		return hBaselineVec.release();
}

class GDTAttrPoolHolder
{
protected:
	vector<size_t>& m_attrPool;
	size_t m_attr;

public:
	GDTAttrPoolHolder(vector<size_t>& attrPool)
	: m_attrPool(attrPool), m_attr(INVALID_INDEX)
	{
	}

	void temporarilyRemoveAttribute(size_t index)
	{
		m_attr = m_attrPool[index];
		GAssert(m_attr != INVALID_INDEX);
		std::swap(m_attrPool[index], m_attrPool[m_attrPool.size() - 1]);
		m_attrPool.erase(m_attrPool.end() - 1);
	}

	~GDTAttrPoolHolder()
	{
		// Put the attribute that we temporarily removed back in the pool
		if(m_attr != INVALID_INDEX)
			m_attrPool.push_back(m_attr);
	}
};

// This constructs the decision tree in a recursive depth-first manner
GDecisionTreeNode* GDecisionTree::buildBranch(GMatrix& features, GMatrix& labels, vector<size_t>& attrPool, size_t nDepth, size_t tolerance)
{
	GAssert(features.rows() == labels.rows());

	// Make a leaf if we're out of tolerance or the output is
	// homogenous or there are no attributes left or we have
	// reached the maximum number of levels in the tree
	if(tolerance <= 0 || features.rows() <= m_leafThresh
	   || attrPool.size() == 0 || labels.isHomogenous()
	   || (nDepth+1 == m_maxLevels)){
		return new GDecisionTreeLeafNode(GDecisionTreeNode_labelVec(labels), labels.rows());
	}

	// Pick the division
	double pivot = 0.0;
	size_t bestIndex = pickDivision(features, labels, &pivot, attrPool, nDepth);

	// Make a leaf if there are no good divisions
	if(bestIndex >= attrPool.size()){
		return new GDecisionTreeLeafNode(GDecisionTreeNode_labelVec(labels), labels.rows());
	}
	size_t attr = attrPool[bestIndex];

	// Make sure there aren't any missing values in the decision attribute
	features.replaceMissingValuesWithBaseline(attr);

	// Split the data
	vector<GMatrix*> featureParts;
	VectorOfPointersHolder<GMatrix> hFeatureParts(featureParts);
	vector<GMatrix*> labelParts;
	VectorOfPointersHolder<GMatrix> hLabelParts(labelParts);
	size_t nonEmptyBranchCount = 0;
	GDTAttrPoolHolder hAttrPool(attrPool);
	if(m_pRelFeatures->valueCount(attr) == 0)
	{
		// Split on a continuous attribute
		GMatrix* pOtherFeatures = new GMatrix(m_pRelFeatures->clone());
		featureParts.push_back(pOtherFeatures);
		GMatrix* pOtherLabels = new GMatrix(m_pRelLabels->clone());
		labelParts.push_back(pOtherLabels);
		features.splitByPivot(pOtherFeatures, attr, pivot, &labels, pOtherLabels);
		nonEmptyBranchCount += (features.rows() > 0 ? 1 : 0) + (pOtherFeatures->rows() > 0 ? 1 : 0);
	}
	else if(m_binaryDivisions)
	{
		// Split on a nominal attribute and specific value
		GMatrix* pOtherFeatures = new GMatrix(m_pRelFeatures->clone());
		featureParts.push_back(pOtherFeatures);
		GMatrix* pOtherLabels = new GMatrix(m_pRelLabels->clone());
		labelParts.push_back(pOtherLabels);
		features.splitCategoricalKeepIfEqual(pOtherFeatures, attr, (int)pivot, &labels, pOtherLabels);
		nonEmptyBranchCount += (features.rows() > 0 ? 1 : 0) + (pOtherFeatures->rows() > 0 ? 1 : 0);
	}
	else
	{
		// Split on a nominal attribute
		int valueCount = (int)features.relation().valueCount(attr);
		for(int i = 1; i < valueCount; i++)
		{
			GMatrix* pOtherFeatures = new GMatrix(m_pRelFeatures->clone());
			featureParts.push_back(pOtherFeatures);
			GMatrix* pOtherLabels = new GMatrix(m_pRelLabels->clone());
			labelParts.push_back(pOtherLabels);
			features.splitCategoricalKeepIfNotEqual(pOtherFeatures, attr, i, &labels, pOtherLabels);
			if(pOtherFeatures->rows() > 0)
				nonEmptyBranchCount++;
		}
		if(features.rows() > 0)
			nonEmptyBranchCount++;
		hAttrPool.temporarilyRemoveAttribute(bestIndex);
	}

	// If we didn't actually separate anything
	if(nonEmptyBranchCount < 2)
	{
		size_t setCount = featureParts.size();
		for(size_t i = 0; i < setCount; i++)
		{
			features.mergeVert(featureParts[i]);
			labels.mergeVert(labelParts[i]);
		}
		if(m_eAlg == MINIMIZE_ENTROPY)
			return new GDecisionTreeLeafNode(GDecisionTreeNode_labelVec(labels), labels.rows());
		else
		{
			// Try another division
			GDecisionTreeNode* pNode = buildBranch(features, labels, attrPool, nDepth, tolerance - 1);
			return pNode;
		}
	}

	// Make an interior node
	double* pBaselineVec = NULL;
	size_t emptySets = 0;
	for(size_t i = 0; i < featureParts.size(); i++)
	{
		if(featureParts[i]->rows() == 0)
			emptySets++;
	}
	if(features.rows() == 0)
		emptySets++;
	if(emptySets > 0)
	{
		GMatrix* pB = labelParts[0];
		for(size_t i = 1; i < labelParts.size(); i++)
		{
			if(labelParts[i]->rows() > pB->rows())
				pB = labelParts[i];
		}
		if(labels.rows() > pB->rows())
			pBaselineVec = GDecisionTreeNode_labelVec(labels);
		else
			pBaselineVec = GDecisionTreeNode_labelVec(*pB);
	}
	ArrayHolder<double> hBaselineVec(pBaselineVec);
	GDecisionTreeInteriorNode* pNode = new GDecisionTreeInteriorNode(attr, pivot, featureParts.size() + 1, 0);
	Holder<GDecisionTreeInteriorNode> hNode(pNode);
	size_t biggest = features.rows();
	if(features.rows() > 0){
		pNode->m_ppChildren[0] = buildBranch(features, labels, attrPool, nDepth + 1, tolerance);
	}else{
		pNode->m_ppChildren[0] = new GDecisionTreeLeafNode(GDecisionTreeNode_copyIfNotTheLast(emptySets--, hBaselineVec, labels.cols()), 0);
	}
	for(size_t i = 0; i < featureParts.size(); i++)
	{
		if(featureParts[i]->rows() > 0)
		{
			pNode->m_ppChildren[i + 1] = buildBranch(*featureParts[i], *labelParts[i], attrPool, nDepth + 1, tolerance);
			if(featureParts[i]->rows() > biggest)
			{
				biggest = featureParts[i]->rows();
				pNode->m_defaultChild = i + 1;
			}
		}
		else
			pNode->m_ppChildren[i + 1] = new GDecisionTreeLeafNode(GDecisionTreeNode_copyIfNotTheLast(emptySets--, hBaselineVec, labels.cols()), 0);
	}
	return hNode.release();
}

GDecisionTreeLeafNode* GDecisionTree::findLeaf(const double* pIn, size_t* pDepth)
{
	if(!m_pRoot)
		throw Ex("Not trained yet");
	GDecisionTreeNode* pNode = m_pRoot;
	int nVal;
	size_t nDepth = 1;
	while(!pNode->IsLeaf())
	{
		GDecisionTreeInteriorNode* pInterior = (GDecisionTreeInteriorNode*)pNode;
		if(m_pRelFeatures->valueCount(pInterior->m_nAttribute) == 0)
		{
			if(pIn[pInterior->m_nAttribute] == UNKNOWN_REAL_VALUE)
				pNode = pInterior->m_ppChildren[pInterior->m_defaultChild];
			else if(pIn[pInterior->m_nAttribute] < pInterior->m_dPivot)
				pNode = pInterior->m_ppChildren[0];
			else
				pNode = pInterior->m_ppChildren[1];
		}
		else if(m_binaryDivisions)
		{
			nVal = (int)pIn[pInterior->m_nAttribute];
			if(nVal < 0)
			{
				GAssert(nVal == UNKNOWN_DISCRETE_VALUE); // out of range
				nVal = (int)pInterior->m_defaultChild;
			}
			GAssert((size_t)nVal < m_pRelFeatures->valueCount(pInterior->m_nAttribute)); // value out of range
			if(nVal == (int)pInterior->m_dPivot)
				pNode = pInterior->m_ppChildren[0];
			else
				pNode = pInterior->m_ppChildren[1];
		}
		else
		{
			nVal = (int)pIn[pInterior->m_nAttribute];
			if(nVal < 0)
			{
				GAssert(nVal == UNKNOWN_DISCRETE_VALUE); // out of range
				nVal = (int)pInterior->m_defaultChild;
			}
			GAssert((size_t)nVal < m_pRelFeatures->valueCount(pInterior->m_nAttribute)); // value out of range
			pNode = pInterior->m_ppChildren[nVal];
		}
		nDepth++;
	}
	*pDepth = nDepth;
	return (GDecisionTreeLeafNode*)pNode;
}

// virtual
void GDecisionTree::predict(const double* pIn, double* pOut)
{
	size_t depth;
	GDecisionTreeLeafNode* pLeaf = findLeaf(pIn, &depth);
	GVec::copy(pOut, pLeaf->m_pOutputValues, m_pRelLabels->size());
}

// virtual
void GDecisionTree::predictDistribution(const double* pIn, GPrediction* pOut)
{
	// Copy the output values into the row
	size_t depth;
	GDecisionTreeLeafNode* pLeaf = findLeaf(pIn, &depth);
	size_t n, nValues;
	size_t labelDims = m_pRelLabels->size();
	for(n = 0; n < labelDims; n++)
	{
		nValues = m_pRelLabels->valueCount(n);
		if(nValues == 0)
			pOut[n].makeNormal()->setMeanAndVariance(pLeaf->m_pOutputValues[n], (double)depth);
		else
			pOut[n].makeCategorical()->setSpike(nValues, (int)pLeaf->m_pOutputValues[n], (int)depth);
	}
}

// virtual
void GDecisionTree::clear()
{
	delete(m_pRoot);
	m_pRoot = NULL;
}

#ifndef NO_TEST_CODE
// static
void GDecisionTree::test()
{
	{
		GDecisionTree tree;
		tree.basicTest(0.704, 0.77);
	}
	{
		GDecisionTree ml2Tree;
		ml2Tree.setMaxLevels(2);
		ml2Tree.basicTest(0.57, 0.68);
	}
	{
		GDecisionTree ml1Tree;
		ml1Tree.setMaxLevels(1);
		ml1Tree.basicTest(0.33, 0.33);
	}
}
#endif

// ----------------------------------------------------------------------

namespace GClasses {
class GMeanMarginsTreeNode
{
public:
	GMeanMarginsTreeNode()
	{
	}

	virtual ~GMeanMarginsTreeNode()
	{
	}

	virtual bool IsLeaf() = 0;
	virtual GDomNode* serialize(GDom* pDoc, size_t nInputs, size_t nOutputs) = 0;

	static GMeanMarginsTreeNode* deserialize(GDomNode* pNode);
};


class GMeanMarginsTreeInteriorNode : public GMeanMarginsTreeNode
{
protected:
	double* m_pCenter;
	double* m_pNormal;
	GMeanMarginsTreeNode* m_pLeft;
	GMeanMarginsTreeNode* m_pRight;

public:
	GMeanMarginsTreeInteriorNode(size_t featureDims, const double* pCenter, const double* pNormal)
	: GMeanMarginsTreeNode()
	{
		m_pCenter = new double[featureDims];
		GVec::copy(m_pCenter, pCenter, featureDims);
		m_pNormal = new double[featureDims];
		GVec::copy(m_pNormal, pNormal, featureDims);
		m_pLeft = NULL;
		m_pRight = NULL;
	}

	GMeanMarginsTreeInteriorNode(GDomNode* pNode)
	: GMeanMarginsTreeNode()
	{
		GDomNode* pCenter = pNode->field("center");
		GDomListIterator it(pCenter);
		size_t dims = it.remaining();
		m_pCenter = new double[dims];
		GVec::deserialize(m_pCenter, it);
		GDomListIterator it2(pNode->field("normal"));
		m_pNormal = new double[it2.remaining()];
		GAssert(it2.remaining() == dims);
		GVec::deserialize(m_pNormal, it2);
		m_pLeft = GMeanMarginsTreeNode::deserialize(pNode->field("left"));
		m_pRight = GMeanMarginsTreeNode::deserialize(pNode->field("right"));
	}

	virtual ~GMeanMarginsTreeInteriorNode()
	{
		delete[] m_pCenter;
		delete[] m_pNormal;
		delete(m_pLeft);
		delete(m_pRight);
	}

	virtual GDomNode* serialize(GDom* pDoc, size_t nInputs, size_t nOutputs)
	{
		GDomNode* pNode = pDoc->newObj();
		pNode->addField(pDoc, "center", GVec::serialize(pDoc, m_pCenter, nInputs));
		pNode->addField(pDoc, "normal", GVec::serialize(pDoc, m_pNormal, nInputs));
		pNode->addField(pDoc, "left", m_pLeft->serialize(pDoc, nInputs, nOutputs));
		pNode->addField(pDoc, "right", m_pRight->serialize(pDoc, nInputs, nOutputs));
		return pNode;
	}

	virtual bool IsLeaf()
	{
		return false;
	}

	bool Test(const double* pInputVector, size_t nInputs)
	{
		return GVec::dotProductIgnoringUnknowns(m_pCenter, pInputVector, m_pNormal, nInputs) >= 0;
	}

	void SetLeft(GMeanMarginsTreeNode* pNode)
	{
		m_pLeft = pNode;
	}

	void SetRight(GMeanMarginsTreeNode* pNode)
	{
		m_pRight = pNode;
	}

	GMeanMarginsTreeNode* GetRight()
	{
		return m_pRight;
	}

	GMeanMarginsTreeNode* GetLeft()
	{
		return m_pLeft;
	}
};

class GMeanMarginsTreeLeafNode : public GMeanMarginsTreeNode
{
protected:
	double* m_pOutputs;

public:
	GMeanMarginsTreeLeafNode(size_t nOutputCount, const double* pOutputs)
	: GMeanMarginsTreeNode()
	{
		m_pOutputs = new double[nOutputCount];
		GVec::copy(m_pOutputs, pOutputs, nOutputCount);
	}

	GMeanMarginsTreeLeafNode(GDomNode* pNode)
	: GMeanMarginsTreeNode()
	{
		GDomListIterator it(pNode);
		size_t dims = it.remaining();
		m_pOutputs = new double[dims];
		GVec::deserialize(m_pOutputs, it);
	}

	virtual ~GMeanMarginsTreeLeafNode()
	{
		delete[] m_pOutputs;
	}

	virtual GDomNode* serialize(GDom* pDoc, size_t nInputs, size_t nOutputs)
	{
		return GVec::serialize(pDoc, m_pOutputs, nOutputs);
	}

	virtual bool IsLeaf()
	{
		return true;
	}

	double* GetOutputs()
	{
		return m_pOutputs;
	}
};
}

// static
GMeanMarginsTreeNode* GMeanMarginsTreeNode::deserialize(GDomNode* pNode)
{
	if(pNode->type() == GDomNode::type_list)
		return new GMeanMarginsTreeLeafNode(pNode);
	else
		return new GMeanMarginsTreeInteriorNode(pNode);
}

// ---------------------------------------------------------------

GMeanMarginsTree::GMeanMarginsTree()
: GSupervisedLearner(), m_internalFeatureDims(0), m_internalLabelDims(0), m_pRoot(NULL)
{
}

GMeanMarginsTree::GMeanMarginsTree(GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode, ll)
{
	m_pRoot = GMeanMarginsTreeNode::deserialize(pNode->field("root"));
	m_internalFeatureDims = (size_t)pNode->field("ifd")->asInt();
	m_internalLabelDims = (size_t)pNode->field("ild")->asInt();
}

GMeanMarginsTree::~GMeanMarginsTree()
{
	delete(m_pRoot);
}

// virtual
GDomNode* GMeanMarginsTree::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GMeanMarginsTree");
	pNode->addField(pDoc, "ifd", pDoc->newInt(m_internalFeatureDims));
	pNode->addField(pDoc, "ild", pDoc->newInt(m_internalLabelDims));
	if(!m_pRoot)
		throw Ex("Attempted to serialize a model that has not been trained");
	pNode->addField(pDoc, "root", m_pRoot->serialize(pDoc, m_internalFeatureDims, m_internalLabelDims));
	return pNode;
}

// virtual
void GMeanMarginsTree::trainInner(const GMatrix& features, const GMatrix& labels)
{
	if(!features.relation().areContinuous())
		throw Ex("GMeanMarginsTree only supports continuous features. Perhaps you should wrap it in a GAutoFilter.");
	if(!labels.relation().areContinuous())
		throw Ex("GMeanMarginsTree only supports continuous labels. Perhaps you should wrap it in a GAutoFilter.");
	clear();
	m_internalFeatureDims = features.cols();
	m_internalLabelDims = labels.cols();
	double* pBuf = new double[m_internalLabelDims * 2 + m_internalFeatureDims * 2];
	ArrayHolder<double> hBuf(pBuf);
	size_t* pBuf2 = new size_t[m_internalFeatureDims * 2];
	ArrayHolder<size_t> hBuf2(pBuf2);
	GMatrix fTmp(features);
	GMatrix lTmp(labels);
	m_pRoot = buildNode(fTmp, lTmp, pBuf, pBuf2);
}

void GMeanMarginsTree::autoTune(GMatrix& features, GMatrix& labels)
{
	// This model has no parameters to tune
}

GMeanMarginsTreeNode* GMeanMarginsTree::buildNode(GMatrix& features, GMatrix& labels, double* pBuf, size_t* pBuf2)
{
	// Check for a leaf node
	GAssert(features.rows() == labels.rows());
	size_t nCount = features.rows();
	if(nCount < 2)
	{
		GAssert(nCount > 0); // no data left
		return new GMeanMarginsTreeLeafNode(m_internalLabelDims, labels[0]);
	}

	// Compute the centroid and principal component of the labels
	double* pLabelCentroid = pBuf;
	labels.centroid(pLabelCentroid);
	double* pPrincipalComponent = pLabelCentroid + m_internalLabelDims;
	labels.principalComponentIgnoreUnknowns(pPrincipalComponent, pLabelCentroid, &m_rand);

	// Compute the centroid of each feature cluster in a manner tolerant of unknown values
	double* pFeatureCentroid1 = pPrincipalComponent + m_internalLabelDims;
	double* pFeatureCentroid2 = pFeatureCentroid1 + m_internalFeatureDims;
	GVec::setAll(pFeatureCentroid1, 0.0, m_internalFeatureDims);
	GVec::setAll(pFeatureCentroid2, 0.0, m_internalFeatureDims);
	size_t* pCounts1 = pBuf2;
	size_t* pCounts2 = pCounts1 + m_internalFeatureDims;
	memset(pCounts1, '\0', sizeof(size_t) * m_internalFeatureDims);
	memset(pCounts2, '\0', sizeof(size_t) * m_internalFeatureDims);
	for(size_t i = 0; i < nCount; i++)
	{
		const double* pF = features[i];
		if(GVec::dotProductIgnoringUnknowns(pLabelCentroid, labels[i], pPrincipalComponent, m_internalLabelDims) >= 0)
		{
			double* pM = pFeatureCentroid2;
			size_t* pC = pCounts2;
			for(size_t j = 0; j < m_internalFeatureDims; j++)
			{
				if(*pF != UNKNOWN_REAL_VALUE)
				{
					*pM += *pF;
					(*pC)++;
				}
				pF++;
				pM++;
				pC++;
			}
		}
		else
		{
			double* pM = pFeatureCentroid1;
			size_t* pC = pCounts1;
			for(size_t j = 0; j < m_internalFeatureDims; j++)
			{
				if(*pF != UNKNOWN_REAL_VALUE)
				{
					*pM += *pF;
					(*pC)++;
				}
				pF++;
				pM++;
				pC++;
			}
		}
	}
	size_t* pC1 = pCounts1;
	size_t* pC2 = pCounts2;
	double* pF1 = pFeatureCentroid1;
	double* pF2 = pFeatureCentroid2;
	for(size_t j = 0; j < m_internalFeatureDims; j++)
	{
		if(*pC1 == 0 || *pC2 == 0)
			return new GMeanMarginsTreeLeafNode(m_internalLabelDims, pLabelCentroid);
		*(pF1++) /= *(pC1++);
		*(pF2++) /= *(pC2++);
	}

	// Compute the feature center and normal
	GVec::add(pFeatureCentroid1, pFeatureCentroid2, m_internalFeatureDims);
	GVec::multiply(pFeatureCentroid1, 0.5, m_internalFeatureDims);
	GVec::subtract(pFeatureCentroid2, pFeatureCentroid1, m_internalFeatureDims);
	GVec::safeNormalize(pFeatureCentroid2, m_internalFeatureDims, &m_rand);

	// Make the interior node
	GMeanMarginsTreeInteriorNode* pNode = new GMeanMarginsTreeInteriorNode(m_internalFeatureDims, pFeatureCentroid1, pFeatureCentroid2);
	Holder<GMeanMarginsTreeInteriorNode> hNode(pNode);

	// Divide the data
	GMatrix otherFeatures(features.relation().clone());
	GMatrix otherLabels(labels.relation().clone());
	{
		GMergeDataHolder hFeatures(features, otherFeatures);
		GMergeDataHolder hLabels(labels, otherLabels);
		otherFeatures.reserve(features.rows());
		otherLabels.reserve(labels.rows());
		for(size_t i = features.rows() - 1; i < features.rows(); i--)
		{
			if(pNode->Test(features[i], m_internalFeatureDims))
			{
				otherFeatures.takeRow(features.releaseRow(i));
				otherLabels.takeRow(labels.releaseRow(i));
			}
		}

		// If we couldn't separate anything, just return a leaf node
		if(features.rows() == 0 || otherFeatures.rows() == 0)
			return new GMeanMarginsTreeLeafNode(m_internalLabelDims, pLabelCentroid);

		// Build the child nodes
		pNode->SetLeft(buildNode(features, labels, pBuf, pBuf2));
		pNode->SetRight(buildNode(otherFeatures, otherLabels, pBuf, pBuf2));
	}
	GAssert(otherFeatures.rows() == 0 && otherLabels.rows() == 0);
	return hNode.release();
}

// virtual
void GMeanMarginsTree::predictDistribution(const double* pIn, GPrediction* pOut)
{
	throw Ex("Sorry, this model cannot predict a distribution");
}

// virtual
void GMeanMarginsTree::predict(const double* pIn, double* pOut)
{
	GMeanMarginsTreeNode* pNode = m_pRoot;
	size_t nDepth = 1;
	while(!pNode->IsLeaf())
	{
		if(((GMeanMarginsTreeInteriorNode*)pNode)->Test(pIn, m_internalFeatureDims))
			pNode = ((GMeanMarginsTreeInteriorNode*)pNode)->GetRight();
		else
			pNode = ((GMeanMarginsTreeInteriorNode*)pNode)->GetLeft();
		nDepth++;
	}
	GVec::copy(pOut, ((GMeanMarginsTreeLeafNode*)pNode)->GetOutputs(), m_internalLabelDims);
}

// virtual
void GMeanMarginsTree::clear()
{
	delete(m_pRoot);
	m_pRoot = NULL;
	m_internalFeatureDims = 0;
	m_internalLabelDims = 0;
}

#ifndef NO_TEST_CODE
// static
void GMeanMarginsTree::test()
{
	GAutoFilter af(new GMeanMarginsTree());
	af.basicTest(0.70, 0.9);
}
#endif










GRandomForest::GRandomForest(size_t trees, size_t samples)
: GSupervisedLearner()
{
	m_pEnsemble = new GBag();
	for(size_t i = 0; i < trees; i++)
	{
		GDecisionTree* pTree = new GDecisionTree();
		pTree->useBinaryDivisions();
		pTree->useRandomDivisions(samples);
		m_pEnsemble->addLearner(pTree);
	}
}

GRandomForest::GRandomForest(GDomNode* pNode, GLearnerLoader& ll)
: GSupervisedLearner(pNode, ll)
{
	m_pEnsemble = new GBag(pNode->field("bag"), ll);
}

// virtual
GRandomForest::~GRandomForest()
{
	delete(m_pEnsemble);
}

// virtual
GDomNode* GRandomForest::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GRandomForest");
	pNode->addField(pDoc, "bag", m_pEnsemble->serialize(pDoc));
	return pNode;
}

// virtual
void GRandomForest::clear()
{
	m_pEnsemble->clear();
}

void GRandomForest::print(std::ostream& stream, GArffRelation* pFeatureRel, GArffRelation* pLabelRel)
{
	std::vector<GWeightedModel*>& models = m_pEnsemble->models();
	size_t nModels = models.size();
	for (size_t i = 0; i < nModels; i++)
	{
	    stream << "TREE " << i << ":" << std::endl;
	    ((GDecisionTree*)models[i]->m_pModel)->print(stream, pFeatureRel, pLabelRel);
	}
}

// virtual
void GRandomForest::trainInner(const GMatrix& features, const GMatrix& labels)
{
	m_pEnsemble->train(features, labels);
}

// virtual
void GRandomForest::predict(const double* pIn, double* pOut)
{
	m_pEnsemble->predict(pIn, pOut);
}

// virtual
void GRandomForest::predictDistribution(const double* pIn, GPrediction* pOut)
{
	m_pEnsemble->predictDistribution(pIn, pOut);
}

#ifndef NO_TEST_CODE
// static
void GRandomForest::test()
{
	GRandomForest rf(30);
	rf.basicTest(0.762, 0.925, 0.01);
}
#endif
