/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    Michael R. Smith,
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

#include "GDistance.h"
#include "GDom.h"
#include "GVec.h"
#include <math.h>
#include <cassert>
#include <memory>

using std::map;

namespace GClasses {

GDistanceMetric::GDistanceMetric(GDomNode* pNode)
: m_scaleFactors(pNode->get("scalars"))
{
	m_pRelation = GRelation::deserialize(pNode->get("relation"));
	if(m_pRelation->size() != m_scaleFactors.size())
		throw Ex("Mismatching sizes");
	m_ownRelation = true;
}

// virtual
GDistanceMetric::~GDistanceMetric()
{
	setRelation(NULL, false);
}

void GDistanceMetric::setRelation(const GRelation* pRelation, bool own)
{
	if(m_ownRelation)
		delete((GRelation*)m_pRelation);
	m_pRelation = pRelation;
	m_ownRelation = own;
	if(pRelation)
	{
		m_scaleFactors.resize(pRelation->size());
		m_scaleFactors.fill(1.0);
	}
	else
		m_scaleFactors.resize(0);
}

GDomNode* GDistanceMetric::baseDomNode(GDom* pDoc) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "class", name());
	pNode->add(pDoc, "relation", m_pRelation->serialize(pDoc));
	pNode->add(pDoc, "scalars", m_scaleFactors.serialize(pDoc));
	return pNode;
}

// static
GDistanceMetric* GDistanceMetric::deserialize(GDomNode* pNode)
{
	const char* szClass = pNode->getString("class");
	if(strcmp(szClass, "GRowDistance") == 0)
		return new GRowDistance(pNode);
	if(strcmp(szClass, "GDenseCosineDistance") == 0)
		return new GDenseCosineDistance(pNode);
	if(strcmp(szClass, "GLNormDistance") == 0)
		return new GLNormDistance(pNode);
	if(strcmp(szClass, "GKernelDistance") == 0)
		return new GKernelDistance(pNode);
	throw Ex("Unrecognized class: ", szClass);
	return NULL;
}

void GDistanceMetric_exerciseMetric(GDistanceMetric& metric)
{
	// Make two vectors
	GVec a(2);
	a.fill(1.0);
	GVec b(2);
	b.fill(2.0);
	GUniformRelation rel(2);
	metric.init(&rel, false);

	// Measure distance	
	double sqDist = metric.squaredDistance(a, b);

	// Roundtrip through serialization
	GDom doc;
	GDomNode* pNode = metric.serialize(&doc);
	GDistanceMetric* pRoundtripped = GDistanceMetric::deserialize(pNode);
	std::unique_ptr<GDistanceMetric> hRoundtripped(pRoundtripped);

	// Make sure the roundtripped metric still returns the same result
	double sqDist2 = pRoundtripped->squaredDistance(a, b);
	if(std::abs(sqDist - sqDist2) > 1e-12)
		throw Ex("failed");
}

// static
void GDistanceMetric::test()
{
	GRowDistance d1; GDistanceMetric_exerciseMetric(d1);
	GLNormDistance d2(1.4); GDistanceMetric_exerciseMetric(d2);
	GDenseCosineDistance d3; GDistanceMetric_exerciseMetric(d3);
	GKernelDistance d4(GKernel::kernelComplex1(), true); GDistanceMetric_exerciseMetric(d4);
}

// --------------------------------------------------------------------

GRowDistance::GRowDistance()
: GDistanceMetric(), m_diffWithUnknown(1.0)
{
}

GRowDistance::GRowDistance(GDomNode* pNode)
: GDistanceMetric(pNode)
{
	m_diffWithUnknown = pNode->getDouble("dwu");
}

// virtual
GDomNode* GRowDistance::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->add(pDoc, "dwu", m_diffWithUnknown);
	return pNode;
}

// virtual
void GRowDistance::init(const GRelation* pRelation, bool own)
{
	setRelation(pRelation, own);
}

// virtual
double GRowDistance::squaredDistance(const GVec& a, const GVec& b) const
{
	if(a.size() != m_pRelation->size() || b.size() != m_pRelation->size())
		throw Ex("unexpected size");
	double sum = 0;
	size_t count = m_pRelation->size();
	double d;
	for(size_t i = 0; i < count; i++)
	{
		if(m_pRelation->valueCount(i) == 0)
		{
			if(a[i] == UNKNOWN_REAL_VALUE || b[i] == UNKNOWN_REAL_VALUE)
				d = m_diffWithUnknown;
			else
				d = (b[i] - a[i]) * m_scaleFactors[i];
		}
		else
		{
			if((int)a[i] == UNKNOWN_DISCRETE_VALUE || (int)b[i] == UNKNOWN_DISCRETE_VALUE)
				d = m_diffWithUnknown;
			else
				d = ((int)b[i] == (int)a[i] ? 0 : 1);
		}
		sum += (d * d);
	}
	return sum;
}

// --------------------------------------------------------------------

GLNormDistance::GLNormDistance(double norm)
: GDistanceMetric(), m_norm(norm), m_diffWithUnknown(1.0)
{
}

GLNormDistance::GLNormDistance(GDomNode* pNode)
: GDistanceMetric(pNode), m_norm(pNode->getDouble("norm")), m_diffWithUnknown(pNode->getDouble("dwu"))
{
}

// virtual
GDomNode* GLNormDistance::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->add(pDoc, "norm", m_norm);
	pNode->add(pDoc, "dwu", m_diffWithUnknown);
	return pNode;
}

// virtual
void GLNormDistance::init(const GRelation* pRelation, bool own)
{
	setRelation(pRelation, own);
}

// virtual
double GLNormDistance::squaredDistance(const GVec& a, const GVec& b) const
{
	if(a.size() != m_pRelation->size() || b.size() != m_pRelation->size())
		throw Ex("unexpected size");
	double sum = 0;
	size_t count = m_pRelation->size();
	double d;
	for(size_t i = 0; i < count; i++)
	{
		if(m_pRelation->valueCount(i) == 0)
		{
			if(a[i] == UNKNOWN_REAL_VALUE || b[i] == UNKNOWN_REAL_VALUE)
				d = m_diffWithUnknown;
			else
				d = (b[i] - a[i]) * m_scaleFactors[i];
		}
		else
		{
			if((int)a[i] == UNKNOWN_DISCRETE_VALUE || (int)b[i] == UNKNOWN_DISCRETE_VALUE)
				d = 1;
			else
				d = ((int)b[i] == (int)a[i] ? 0 : 1);
		}
		sum += pow(fabs(d), m_norm);
	}
	d = pow(sum, 1.0 / m_norm);
	return (d * d);
}

// --------------------------------------------------------------------

GDenseCosineDistance::GDenseCosineDistance()
: GDistanceMetric()
{
}

GDenseCosineDistance::GDenseCosineDistance(GDomNode* pNode)
: GDistanceMetric(pNode)
{
}

// virtual
GDomNode* GDenseCosineDistance::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	return pNode;
}

// virtual
void GDenseCosineDistance::init(const GRelation* pRelation, bool own)
{
	setRelation(pRelation, own);
}

// virtual
double GDenseCosineDistance::squaredDistance(const GVec& a, const GVec& b) const
{
	if(a.size() != m_pRelation->size() || b.size() != m_pRelation->size())
		throw Ex("unexpected size");
	double sum_sq_a = 0.0;
	double sum_sq_b = 0.0;
	double sum_co_prod = 0.0;
	for(size_t i = 0; i < a.size(); i++)
	{
		sum_sq_a += (a[i] * a[i]);
		sum_sq_b += (b[i] * b[i]);
		sum_co_prod += (a[i] * b[i]);
	}
	double denom = sqrt(sum_sq_a * sum_sq_b);
	if(denom > 0.0)
		return 1.0 - sum_co_prod / denom;
	else
		return 1.0;

}

// --------------------------------------------------------------------

GKernelDistance::GKernelDistance(GKernel* pKernel, bool own)
: GDistanceMetric(), m_pKernel(pKernel), m_ownKernel(own)
{
}

GKernelDistance::GKernelDistance(GDomNode* pNode)
: GDistanceMetric(pNode), m_ownKernel(true)
{
	m_pKernel = GKernel::deserialize(pNode->get("kernel"));
}

// virtual
GKernelDistance::~GKernelDistance()
{
	if(m_ownKernel)
		delete(m_pKernel);
}

// virtual
GDomNode* GKernelDistance::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->add(pDoc, "kernel", m_pKernel->serialize(pDoc));
	return pNode;
}

// virtual
void GKernelDistance::init(const GRelation* pRelation, bool own)
{
	setRelation(pRelation, own);
}

// virtual
double GKernelDistance::squaredDistance(const GVec& a, const GVec& b) const
{
	return 1.0 - m_pKernel->apply(a, b);
}

// --------------------------------------------------------------------

// static
GSparseSimilarity* GSparseSimilarity::deserialize(GDomNode* pNode)
{
	const char* szClass = pNode->getString("class");
	GSparseSimilarity* pObj = NULL;
	if(strcmp(szClass, "GCosineSimilarity") == 0)
		return new GCosineSimilarity(pNode);
	else if(strcmp(szClass, "GEuclidSimilarity") == 0)
		return new GEuclidSimilarity(pNode);
	else if(strcmp(szClass, "GPearsonCorrelation") == 0)
		return new GPearsonCorrelation(pNode);
	else
		throw Ex("Unrecognized class: ", szClass);
	pObj->m_regularizer = pNode->getDouble("reg");
	return pObj;
}

GDomNode* GSparseSimilarity::baseDomNode(GDom* pDoc, const char* szClassName) const
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "class", szClassName);
	pNode->add(pDoc, "reg", m_regularizer);
	return pNode;
}

// --------------------------------------------------------------------

// virtual
GDomNode* GCosineSimilarity::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GCosineSimilarity");
	return pNode;
}

// virtual
double GCosineSimilarity::similarity(const map<size_t,double>& a, const map<size_t,double>& b)
{
	map<size_t,double>::const_iterator itA = a.begin();
	map<size_t,double>::const_iterator itB = b.begin();
	if(itA == a.end())
		return 0.0;
	if(itB == b.end())
		return 0.0;
	double sum_sq_a = 0.0;
	double sum_sq_b = 0.0;
	double sum_co_prod = 0.0;
	while(true)
	{
		if(itA->first < itB->first)
		{
			if(++itA == a.end())
				break;
		}
		else if(itB->first < itA->first)
		{
			if(++itB == b.end())
				break;
		}
		else
		{
			sum_sq_a += (itA->second * itA->second);
			sum_sq_b += (itB->second * itB->second);
			sum_co_prod += (itA->second * itB->second);
			if(++itA == a.end())
				break;
			if(++itB == b.end())
				break;
		}
	}
	double denom = sqrt(sum_sq_a * sum_sq_b) + m_regularizer;
	if(denom > 0.0)
		return sum_co_prod / denom;
	else
		return 0.0;
}

// virtual
double GCosineSimilarity::similarity(const map<size_t,double>& a, const GVec& b)
{
	map<size_t,double>::const_iterator itA = a.begin();
	if(itA == a.end())
		return 0.0;
	double sum_sq_a = 0.0;
	double sum_sq_b = 0.0;
	double sum_co_prod = 0.0;
	while(itA != a.end())
	{
		sum_sq_a += (itA->second * itA->second);
		sum_sq_b += (b[itA->first] * b[itA->first]);
		sum_co_prod += (itA->second * b[itA->first]);
		itA++;
	}
	double denom = sqrt(sum_sq_a * sum_sq_b) + m_regularizer;
	if(denom > 0.0)
		return sum_co_prod / denom;
	else
		return 0.0;
}

// virtual
double GCosineSimilarity::similarity(const GVec& a, const GVec& b)
{
	if(a.size() != b.size())
		throw Ex("mismatching sizes");
	double sum_sq_a = 0.0;
	double sum_sq_b = 0.0;
	double sum_co_prod = 0.0;
	for(size_t i = 0; i < a.size(); i++)
	{
		sum_sq_a += (a[i] * a[i]);
		sum_sq_b += (b[i] * b[i]);
		sum_co_prod += (a[i] * b[i]);
	}
	double denom = sqrt(sum_sq_a * sum_sq_b) + m_regularizer;
	if(denom > 0.0)
		return sum_co_prod / denom;
	else
		return 0.0;
}

// --------------------------------------------------------------------

// virtual
GDomNode* GPearsonCorrelation::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GPearsonCorrelation");
	return pNode;
}

// virtual
double GPearsonCorrelation::similarity(const map<size_t,double>& a, const map<size_t,double>& b)
{
	// Compute the mean of the overlapping portions
	map<size_t,double>::const_iterator itA = a.begin();
	map<size_t,double>::const_iterator itB = b.begin();
	if(itA == a.end())
		return 0.0;
	if(itB == b.end())
		return 0.0;
	double mean_a = 0.0;
	double mean_b = 0.0;
	size_t count = 0;
	while(true)
	{
		if(itA->first < itB->first)
		{
			if(++itA == a.end())
				break;
		}
		else if(itB->first < itA->first)
		{
			if(++itB == b.end())
				break;
		}
		else
		{
			mean_a += itA->second;
			mean_b += itB->second;
			count++;
			if(++itA == a.end())
				break;
			if(++itB == b.end())
				break;
		}
	}
	double d = count > 0 ? 1.0 / count : 0.0;
	mean_a *= d;
	mean_b *= d;

	// Compute the similarity
	itA = a.begin();
	itB = b.begin();
	double sum = 0.0;
	double sum_of_sq = 0.0;
	while(true)
	{
		if(itA->first < itB->first)
		{
			if(++itA == a.end())
				break;
		}
		else if(itB->first < itA->first)
		{
			if(++itB == b.end())
				break;
		}
		else
		{
			d = (itA->second - mean_a) * (itB->second - mean_b);
			sum += d;
			sum_of_sq += (d * d);
			if(++itA == a.end())
				break;
			if(++itB == b.end())
				break;
		}
	}
	double denom = sqrt(sum_of_sq) + m_regularizer;
	if(denom > 0.0)
		return std::max(-1.0, std::min(1.0, sum / denom));
	else
		return 0.0;
}

// virtual
double GPearsonCorrelation::similarity(const map<size_t,double>& a, const GVec& b)
{
	// Compute the mean of the overlapping portions
	map<size_t,double>::const_iterator itA = a.begin();
	double mean_a = 0.0;
	double mean_b = 0.0;
	size_t count = 0;
	while(itA != a.end())
	{
		mean_a += itA->second;
		mean_b += b[itA->first];
		count++;
		itA++;
	}
	double d = 1.0 / count;
	mean_a *= d;
	mean_b *= d;

	// Compute the similarity
	itA = a.begin();
	double sum = 0.0;
	double sum_of_sq = 0.0;
	while(itA != a.end())
	{
		d = (itA->second - mean_a) * (b[itA->first] - mean_b);
		sum += d;
		sum_of_sq += (d * d);
		itA++;
	}
	double denom = sqrt(sum_of_sq) + m_regularizer;
	if(denom > 0.0)
		return sum / denom;
	else
		return 0.0;
}

// virtual
double GPearsonCorrelation::similarity(const GVec& a, const GVec& b)
{
	if(a.size() != b.size())
		throw Ex("mismatching sizes");

	// Compute the mean of the overlapping portions
	double mean_a = 0.0;
	double mean_b = 0.0;
	for(size_t i = 0; i < a.size(); i++)
	{
		mean_a += a[i];
		mean_b += b[i];
	}
	double d = 1.0 / a.size();
	mean_a *= d;
	mean_b *= d;

	// Compute the similarity
	double sum = 0.0;
	double sum_of_sq = 0.0;
	for(size_t i = 0; i < a.size(); i++)
	{
		d = (a[i] - mean_a) * (b[i] - mean_b);
		sum += d;
		sum_of_sq += (d * d);
	}
	double denom = sqrt(sum_of_sq) + m_regularizer;
	if(denom > 0.0)
		return sum / denom;
	else
		return 0.0;
}

// --------------------------------------------------------------------

// virtual
GDomNode* GEuclidSimilarity::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc, "GEuclidSimilarity");
	return pNode;
}

// virtual
double GEuclidSimilarity::similarity(const map<size_t,double>& a, const map<size_t,double>& b)
{
	map<size_t,double>::const_iterator itA = a.begin();
	map<size_t,double>::const_iterator itB = b.begin();
	if(itA == a.end())
		return 0.0;
	if(itB == b.end())
		return 0.0;
	double sum_sq = 0.0;
	while(true)
	{
		if(itA->first < itB->first)
		{
			if(++itA == a.end())
				break;
		}
		else if(itB->first < itA->first)
		{
			if(++itB == b.end())
				break;
		}
		else
		{	
			double d = (itB->second - itA->second);
			sum_sq += (d * d);
			if(++itA == a.end())
				break;
			if(++itB == b.end())
				break;
		}
	}
	if(sum_sq > 0.0)
		return 1.0 / sum_sq;
	else
		return 1e12;
}

// virtual
double GEuclidSimilarity::similarity(const map<size_t,double>& a, const GVec& b)
{
	map<size_t,double>::const_iterator itA = a.begin();
	if(itA == a.end())
		return 0.0;
	double sum_sq = 0.0;
	while(itA != a.end())
	{
		double d = (b[itA->first] - itA->second);
		sum_sq += (d * d);
		itA++;
	}
	if(sum_sq > 0.0)
		return 1.0 / sum_sq;
	else
		return 1e12;
}

// virtual
double GEuclidSimilarity::similarity(const GVec& a, const GVec& b)
{
	if(a.size() != b.size())
		throw Ex("mismatching sizes");

	double sum_sq = 0.0;
	for(size_t i = 0; i < a.size(); i++)
	{
		double d = (b[i] - a[i]);
		sum_sq += (d * d);
	}
	if(sum_sq > 0.0)
		return 1.0 / sum_sq;
	else
		return 1e12;
}


} // namespace GClasses
