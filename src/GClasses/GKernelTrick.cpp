/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GKernelTrick.h"
#include "GHillClimber.h"
#include "GDistribution.h"
#include "GMath.h"

using namespace GClasses;

GDomNode* GKernel::makeBaseNode(GDom* pDoc) const
{
	GDomNode* pObj = pDoc->newObj();
	pObj->addField(pDoc, "name", pDoc->newString(name()));
	return pObj;
}

// static
GKernel* GKernel::deserialize(GDomNode* pNode)
{
	const char* szName = pNode->field("name")->asString();
	if(strcmp(szName, "identity") == 0)
		return new GKernelIdentity(pNode);
	else if(strcmp(szName, "chisquared") == 0)
		return new GKernelChiSquared(pNode);
	else if(strcmp(szName, "polynomial") == 0)
		return new GKernelPolynomial(pNode);
	else if(strcmp(szName, "rbf") == 0)
		return new GKernelGaussianRBF(pNode);
	else if(strcmp(szName, "translate") == 0)
		return new GKernelTranslate(pNode);
	else if(strcmp(szName, "scale") == 0)
		return new GKernelScale(pNode);
	else if(strcmp(szName, "add") == 0)
		return new GKernelAdd(pNode);
	else if(strcmp(szName, "multiply") == 0)
		return new GKernelMultiply(pNode);
	else if(strcmp(szName, "pow") == 0)
		return new GKernelPow(pNode);
	else if(strcmp(szName, "exp") == 0)
		return new GKernelExp(pNode);
	else if(strcmp(szName, "normalize") == 0)
		return new GKernelNormalize(pNode);
	else
		throw Ex("Unrecognized activation function: ", szName);
	return NULL;
}

GKernel* makeKernel()
{
	//return new GKernelIdentity();
	//return new GKernelPolynomial(1, 7);

	GKernel* pK1 = new GKernelPolynomial(0, 3);
	GKernel* pK2 = new GKernelPolynomial(1, 7);
	GKernel* pK3 = new GKernelAdd(pK1, pK2);
	GKernel* pK4 = new GKernelNormalize(pK3);

	GKernel* pK5 = new GKernelGaussianRBF(0.01);
	GKernel* pK6 = new GKernelGaussianRBF(0.1);
	GKernel* pK7 = new GKernelAdd(pK5, pK6);
	GKernel* pK8 = new GKernelNormalize(pK7);

	GKernel* pK9 = new GKernelGaussianRBF(1.0);
	GKernel* pK10 = new GKernelGaussianRBF(10.0);
	GKernel* pK11 = new GKernelMultiply(pK9, pK10);
	GKernel* pK12 = new GKernelNormalize(pK11);

	GKernel* pK13 = new GKernelAdd(pK8, pK12);
	GKernel* pK14 = new GKernelAdd(pK4, pK13);
	return pK14;
}

