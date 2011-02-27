/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GDistance.h"
#include "GTwt.h"
#include "GVec.h"
#include <math.h>

using std::map;

namespace GClasses {

GDissimilarityMetric::GDissimilarityMetric(GTwtNode* pNode)
{
	m_pRelation = GRelation::fromTwt(pNode->field("relation"));
}

GTwtNode* GDissimilarityMetric::baseTwtNode(GTwtDoc* pDoc, const char* szClassName)
{
	GTwtNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "class", pDoc->newString(szClassName));
	pNode->addField(pDoc, "relation", m_pRelation->toTwt(pDoc));
	return pNode;
}

// static
GDissimilarityMetric* GDissimilarityMetric::fromTwt(GTwtNode* pNode)
{
	const char* szClass = pNode->field("class")->asString();
	if(strcmp(szClass, "GRowDistanceScaled") == 0)
		return new GRowDistanceScaled(pNode);
	if(strcmp(szClass, "GRowDistance") == 0)
		return new GRowDistance(pNode);
	if(strcmp(szClass, "GMinkowskiDistance") == 0)
		return new GMinkowskiDistance(pNode);
	ThrowError("Unrecognized class: ", szClass);
	return NULL;
}

// --------------------------------------------------------------------

GRowDistance::GRowDistance(GTwtNode* pNode)
: GDissimilarityMetric(pNode)
{
}

// virtual
GTwtNode* GRowDistance::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = baseTwtNode(pDoc, "GRowDistance");
	return pNode;
}

// virtual
void GRowDistance::init(sp_relation& pRelation)
{
	m_pRelation = pRelation;
}

// virtual
double GRowDistance::dissimilarity(const double* pA, const double* pB)
{
	double sum = 0;
	size_t count = m_pRelation->size();
	double d;
	for(size_t i = 0; i < count; i++)
	{
		if(m_pRelation->valueCount(i) == 0)
			d = *pB - *pA;
		else
			d = ((int)*pB == (int)*pA ? 0 : 1);
		pA++;
		pB++;
		sum += (d * d);
	}
	return sum;
}

// --------------------------------------------------------------------

GRowDistanceScaled::GRowDistanceScaled(GTwtNode* pNode)
: GDissimilarityMetric(pNode)
{
	GTwtNode* pScaleFactors = pNode->field("scaleFactors");
	size_t dims = m_pRelation->size();
	if(pScaleFactors->itemCount() != dims)
		ThrowError("wrong number of scale factors");
	m_pScaleFactors = new double[dims];
	for(size_t i = 0; i < dims; i++)
		m_pScaleFactors[i] = pScaleFactors->item(i)->asDouble();
}

// virtual
GTwtNode* GRowDistanceScaled::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = baseTwtNode(pDoc, "GRowDistance");
	size_t dims = m_pRelation->size();
	GTwtNode* pScaleFactors = pNode->addField(pDoc, "scaleFactors", pDoc->newList(dims));
	for(size_t i = 0; i < dims; i++)
		pScaleFactors->setItem(i, pDoc->newDouble(m_pScaleFactors[i]));
	return pNode;
}

// virtual
void GRowDistanceScaled::init(sp_relation& pRelation)
{
	m_pRelation = pRelation;
	delete[] m_pScaleFactors;
	m_pScaleFactors = new double[pRelation->size()];
	GVec::setAll(m_pScaleFactors, 1.0, pRelation->size());
}

// virtual
double GRowDistanceScaled::dissimilarity(const double* pA, const double* pB)
{
	double sum = 0;
	size_t count = m_pRelation->size();
	double d;
	const double* pSF = m_pScaleFactors;
	for(size_t i = 0; i < count; i++)
	{
		if(m_pRelation->valueCount(i) == 0)
			d = (*pB - *pA) * (*pSF);
		else
			d = ((int)*pB == (int)*pA ? 0 : *pSF);
		pA++;
		pB++;
		pSF++;
		sum += (d * d);
	}
	return sum;
}

// --------------------------------------------------------------------

GMinkowskiDistance::GMinkowskiDistance(GTwtNode* pNode)
: GDissimilarityMetric(pNode)
{
}

// virtual
GTwtNode* GMinkowskiDistance::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = baseTwtNode(pDoc, "GMinkowskiDistance");
	return pNode;
}

// virtual
void GMinkowskiDistance::init(sp_relation& pRelation)
{
	if(!pRelation->areContinuous(0, pRelation->size()))
		ThrowError("Only continuous attributes are supported");
	m_pRelation = pRelation;
}

// virtual
double GMinkowskiDistance::dissimilarity(const double* pA, const double* pB)
{
	return GVec::minkowskiDistance(m_norm, pA, pB, m_pRelation->size());
}

// --------------------------------------------------------------------

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
double GCosineSimilarity::similarity(const map<size_t,double>& a, const double* pB)
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
		sum_sq_b += (pB[itA->first] * pB[itA->first]);
		sum_co_prod += (itA->second * pB[itA->first]);
		itA++;
	}
	double denom = sqrt(sum_sq_a * sum_sq_b) + m_regularizer;
	if(denom > 0.0)
		return sum_co_prod / denom;
	else
		return 0.0;
}

// --------------------------------------------------------------------

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
double GPearsonCorrelation::similarity(const map<size_t,double>& a, const double* pB)
{
	// Compute the mean of the overlapping portions
	map<size_t,double>::const_iterator itA = a.begin();
	double mean_a = 0.0;
	double mean_b = 0.0;
	size_t count = 0;
	while(itA != a.end())
	{
		mean_a += itA->second;
		mean_b += pB[itA->first];
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
		d = (itA->second - mean_a) * (pB[itA->first] - mean_b);
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

} // namespace GClasses
