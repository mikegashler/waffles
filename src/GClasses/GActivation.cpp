/*
	Copyright (C) 2010, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GActivation.h"
#ifndef MIN_PREDICT
#include "GMath.h"
#endif // MIN_PREDICT
#include "GDom.h"
#include "GRand.h"
#include "GTime.h"

namespace GClasses {

GDomNode* GActivationFunction::serialize(GDom* pDoc) const
{
	return pDoc->newString(name());
}

// static
GActivationFunction* GActivationFunction::deserialize(GDomNode* pNode)
{
	const char* szName = pNode->asString();
	if(strcmp(szName, "logistic") == 0)
		return new GActivationLogistic();
	else if(strcmp(szName, "arctan") == 0)
		return new GActivationArcTan();
	else if(strcmp(szName, "tanh") == 0)
		return new GActivationTanH();
	else if(strcmp(szName, "algebraic") == 0)
		return new GActivationAlgebraic();
	else if(strcmp(szName, "identity") == 0)
		return new GActivationIdentity();
	else if(strcmp(szName, "gaussian") == 0)
		return new GActivationGaussian();
	else if(strcmp(szName, "bidir") == 0)
		return new GActivationBiDir();
	else if(strcmp(szName, "bend") == 0)
		return new GActivationBend();
	else if(strcmp(szName, "sinc") == 0)
		return new GActivationSinc();
	else if(strcmp(szName, "piecewise") == 0)
		return new GActivationPiecewise();
	else if(strcmp(szName, "logisticderiv") == 0)
		return new GActivationLogisticDerivative();
	else
		throw Ex("Unrecognized activation function: ", szName);
	return NULL;
}

/*
double logisticLookup[64] = {
	0.5, 0.51561991572302, 0.53120937337376, 0.54673815198461, 0.5621765008858, 0.57749536518581, 0.59266659995407, 0.60766316983289,
	0.62245933120185, 0.63703079448038, 0.65135486466605, 0.66541055874681, 0.67917869917539, 0.69264198313474, 0.70578502783701, 0.71859439257086,
	0.73105857863, 0.75491498686763, 0.77729986117469, 0.79818677773962, 0.81757447619364, 0.83548353710344, 0.85195280196831, 0.86703575980217,
	0.88079707797788, 0.90465053510089, 0.92414181997876, 0.93991334982599, 0.95257412682243, 0.96267311265587, 0.97068776924864, 0.97702263008997,
	0.98201379003791, 0.98901305736941, 0.99330714907572, 0.9959298622841, 0.99752737684337, 0.99849881774326, 0.9990889488056, 0.99944722136308,
	0.99966464986953, 0.99987660542401, 0.9999546021313, 0.99998329857815, 0.9999938558254, 0.9999977396757, 0.99999916847197, 0.99999969409777,
	0.99999988746484, 0.99999998477002, 0.99999999793885, 0.99999999972105, 0.99999999996225, 0.99999999999489, 0.99999999999931, 0.99999999999991,
	0.99999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
};

// virtual
double GActivationLogisticLookup::squash(double x)
{
	if(x < 0)
	{
		int ex;
		double mantissa = -frexp(x, &ex);
		if(ex < 0)
		{
			mantissa = -x;
			ex = 0;
		}
		else
		{
			mantissa -= 0.5;
			mantissa *= 16;
			ex = std::min(ex + 1, 7);
		}
		double index;
		double rem = modf(mantissa, &index);
		int ind = ex * 8 + (int)index;
		double a = logisticLookup[ind];
		double b = logisticLookup[ind + 1];
		return 1.0 - ((1.0 - rem) * a + rem * b);
	}
	else
	{
		int ex;
		double mantissa = frexp(x, &ex);
		if(ex < 0)
		{
			mantissa = x;
			ex = 0;
		}
		else
		{
			mantissa -= 0.5;
			mantissa *= 16;
			ex = std::min(ex + 1, 7);
		}
		double index;
		double rem = modf(mantissa, &index);
		int ind = ex * 8 + (int)index;
		double a = logisticLookup[ind];
		double b = logisticLookup[ind + 1];
		return (1.0 - rem) * a + rem * b;
	}
}

double logisticDerivativeLookup[64] = {
	0.25, 0.24975601823281, 0.24902597501362, 0.24781554514906, 0.2461340827376, 0.24399446837472, 0.24141290125295, 0.23840864186153,
	0.23500371220159, 0.23122256136407, 0.22709170494192, 0.22263934705507, 0.21789499376181, 0.21288906633392, 0.20765252231812, 0.20221649153658,
	0.19661193324148, 0.18501834947028, 0.1731047869925, 0.16108464558126, 0.14914645207033, 0.13745079633257, 0.12612922518666, 0.11528475102644,
	0.10499358540351, 0.086257944442563, 0.070103716545108, 0.056476244644874, 0.045176659730912, 0.035933590825328, 0.028453023879736, 0.022449410382043,
	0.017662706213291, 0.010866229722225, 0.00664805667079, 0.0040535716948697, 0.0024665092913599, 0.0014989287085691, 0.00091022118012178, 0.00055247307270216,
	0.00033523767075637, 0.00012337934976493, 4.5395807735908e-05, 1.6701142910462e-05, 6.1441368513257e-06, 2.2603191888876e-06, 8.3152733632823e-07, 3.0590213341739e-07,
	1.1253514941371e-07, 1.5229979267812e-08, 2.0611536879194e-09, 2.7894686552833e-10, 3.7751357594114e-11, 5.1090243146939e-12, 6.9144689973607e-13, 9.3480778673429e-14,
	1.2656542480727e-14, 2.2204460492503e-16, 0, 0, 0, 0, 0, 0
};

#ifndef MIN_PREDICT
void GActivationLogisticLookup::test()
{
	GActivationLogistic logistic;
	GActivationLogisticLookup lookup;
	double maxErr = 0.0;
	GRand r(0);
	for(size_t i = 0; i < 1000000; i++)
	{
		double v = 5 * r.normal();
		double a = logistic.squash(v);
		double b = lookup.squash(v);
		maxErr = std::max(maxErr, std::abs(a - b));
	}
	if(maxErr > 0.15)
		throw Ex("Max error ", to_str(maxErr), " too big.");
	double t1 = GTime::seconds();
	r.setSeed(0);
	for(size_t i = 0; i < 1000000; i++)
		logistic.squash(5 * r.normal());
	double t2 = GTime::seconds();
	r.setSeed(0);
	for(size_t i = 0; i < 1000000; i++)
		lookup.squash(5 * r.normal());
	double t3 = GTime::seconds();
	double speedup = (t3 - t2) / (t2 - t1);
	if(speedup < 100)
		throw Ex("Speedup ", to_str(speedup), " insufficient");
}
#endif
*/



// virtual
double GActivationArcTan::halfRange()
{
	return M_PI / 2;
}




// virtual
double GActivationPiecewise::squash(double x)
{
	double d = floor(log(std::max(1.0, std::abs(x))) * 2.46630346237643166848294787835207968);
	double a = pow(1.5, d);
	double b = pow(1.5, d + 1);
	if(b >= 700.0)
		return (x >= 0 ? 1.0 : 0.0);
	double t = (std::abs(x) - a) / (b - a);
	double v = (1.0 - t) / (exp(-a) + 1.0) + t / (exp(-b) + 1.0);
	return (x >= 0 ? v : 1.0 - v);
}




} // namespace GClasses

