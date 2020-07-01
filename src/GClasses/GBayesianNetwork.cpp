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

#include "GBayesianNetwork.h"
#include "GRand.h"
#include "GError.h"
#include "GMath.h"
#include "GHolders.h"
#include "GMatrix.h"
#include <stddef.h>
#include <cmath>

using namespace GClasses;
using std::vector;


#define MIN_LOG_PROB -1e300


GBNNode::GBNNode()
: m_observed(false)
{
}

// virtual
GBNNode::~GBNNode()
{
}





void GBNSum::addParent(GBNNode* pNode)
{
	if(m_gotChildren)
		throw Ex("GBNSum nodes require all parents to be added before any children are added");
	m_parents.push_back(pNode);
}

// virtual
double GBNSum::currentValue()
{
	double sum = 0.0;
	for(vector<GBNNode*>::iterator it = m_parents.begin(); it != m_parents.end(); it++)
		sum += (*it)->currentValue();
	return sum;
}

// virtual
void GBNSum::onNewChild(GBNVariable* pChild)
{
	m_gotChildren = true;
	for(vector<GBNNode*>::iterator it = m_parents.begin(); it != m_parents.end(); it++)
		(*it)->onNewChild(pChild);
}




void GBNProduct::addParent(GBNNode* pNode)
{
	if(m_gotChildren)
		throw Ex("GBNProduct nodes require all parents to be added before any children are added");
	m_parents.push_back(pNode);
}

// virtual
double GBNProduct::currentValue()
{
	double prod = 1.0;
	for(vector<GBNNode*>::iterator it = m_parents.begin(); it != m_parents.end(); it++)
		prod *= (*it)->currentValue();
	return prod;
}

// virtual
void GBNProduct::onNewChild(GBNVariable* pChild)
{
	m_gotChildren = true;
	for(vector<GBNNode*>::iterator it = m_parents.begin(); it != m_parents.end(); it++)
		(*it)->onNewChild(pChild);
}





// virtual
double GBNMath::currentValue()
{
	double val = m_parent->currentValue();
	switch(m_op)
	{
		case NEGATE: return -val;
		case RECIPROCAL: return 1.0 / val;
		case SQUARE_ROOT: return sqrt(val);
		case SQUARE: return val * val;
		case LOG_E: return log(val);
		case EXP: return exp(val);
		case TANH: return tanh(val);
		case GAMMA: return GMath::gamma(val);
		case ABS: return std::abs(val);
		default:
			throw Ex("Unexpected operator");
	}
	return 0.0;
}

// virtual
void GBNMath::onNewChild(GBNVariable* pChild)
{
	m_parent->onNewChild(pChild);
}






GBNVariable::GBNVariable()
: GBNNode()
{

}

// virtual
GBNVariable::~GBNVariable()
{
}

// virtual
void GBNVariable::onNewChild(GBNVariable* pChild)
{
	m_children.push_back(pChild);
}

size_t GBNVariable::catCount()
{
	size_t cats = 1;
	for(vector<GBNCategorical*>::iterator it = m_catParents.begin(); it != m_catParents.end(); it++)
		cats *= (*it)->categories();
	return cats;
}

size_t GBNVariable::currentCatIndex()
{
	size_t mult = 1;
	size_t ind = 0;
	for(vector<GBNCategorical*>::iterator it = m_catParents.begin(); it != m_catParents.end(); it++)
	{
		GBNCategorical* pPar = *it;
		size_t val = (size_t)pPar->currentValue();
		ind += mult * val;
		if(pPar->categories() == 0)
			throw Ex("Categorical parent has no categories. Perhaps addWeights was never called as it ought to have been");
		mult *= pPar->categories();
	}
	GAssert(ind < mult);
	return ind;
}








GBNCategorical::GBNCategorical(size_t categors, GBNNode* pDefaultWeight)
: GBNVariable(), m_categories(categors), m_val(0)
{
	if(categors < 2)
		throw Ex("Expected at least 2 categories. Got ", to_str(categors));
	m_weights.resize(categors, pDefaultWeight);
}

void GBNCategorical::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultWeight)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_weights.resize(m_categories * catCount(), pDefaultWeight);
}

void GBNCategorical::setWeights(size_t cat, GBNNode* pW1, GBNNode* pW2, GBNNode* pW3, GBNNode* pW4, GBNNode* pW5, GBNNode* pW6, GBNNode* pW7, GBNNode* pW8)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = cat * m_categories;
	m_weights[base] = pW1;
	pW1->onNewChild(this);
	m_weights[base + 1] = pW2;
	pW2->onNewChild(this);
	size_t given = 2;
	if(pW3){
		m_weights[base + 2] = pW3;
		pW3->onNewChild(this);
		given++;
		if(pW4){
			m_weights[base + 3] = pW4;
			pW4->onNewChild(this);
			given++;
			if(pW5){
				m_weights[base + 4] = pW5;
				pW5->onNewChild(this);
				given++;
				if(pW6){
					m_weights[base + 5] = pW6;
					pW6->onNewChild(this);
					given++;
					if(pW7){
						m_weights[base + 6] = pW7;
						pW7->onNewChild(this);
						given++;
						if(pW8){
							m_weights[base + 7] = pW8;
							pW8->onNewChild(this);
							given++;
						}
					}
				}
			}
		}
	}
	if(given != m_categories)
		throw Ex("Expected ", to_str(m_categories), " sequential non-NULL nodes. Got ", to_str(given));
}

// virtual
double GBNCategorical::currentValue()
{
	if(m_observed)
		return m_observedValue;
	else
		return (double)m_val;
}

// virtual
void GBNCategorical::sample(GRand* pRand)
{
	if(m_observed)
		return;

	// Compute the sum Markov-blanket probability over each category
	size_t base = m_categories * currentCatIndex();
	double sumProb = 0.0;
	for(size_t i = 0; i < m_categories; i++)
	{
		double catProb = m_weights[base + i]->currentValue();
		for(vector<GBNVariable*>::const_iterator it = children().begin(); it != children().end(); it++)
		{
			GBNVariable* pChildNode = *it;
			size_t oldVal = m_val;
			m_val = i;
			catProb *= pChildNode->likelihood(pChildNode->currentValue());
			m_val = oldVal;
		}
		sumProb += catProb;
	}

	// Pick a category at random according to Markov-blanket probabilities
	double uni = pRand->uniform();
	double sumProb2 = 0.0;
	for(size_t i = 0; i < m_categories; i++)
	{
		double catProb = m_weights[base + i]->currentValue();
		for(vector<GBNVariable*>::const_iterator it = children().begin(); it != children().end(); it++)
		{
			GBNVariable* pChildNode = *it;
			size_t oldVal = m_val;
			m_val = i;
			catProb *= pChildNode->likelihood(pChildNode->currentValue());
			m_val = oldVal;
		}
		m_val = i;
		sumProb2 += catProb / sumProb;
		if(sumProb2 >= uni)
			break;
	}
}

// virtual
double GBNCategorical::likelihood(double x)
{
	size_t base = m_categories * currentCatIndex();
	double sumWeight = 0.0;
	for(size_t i = 0; i < m_categories; i++)
		sumWeight += m_weights[base + i]->currentValue();
	size_t xx = (size_t)x;
	GAssert(xx >= 0 && xx < m_categories);
	GBNNode* pX = m_weights[base + xx];
	double num = pX->currentValue();
	if(num > 0.0 && sumWeight > 0.0)
		return num / sumWeight;
	else
		return 0.0;
}







#define DEV_SAMPLES 100
#define DEV_DECAY ((double)(DEV_SAMPLES - 1) / DEV_SAMPLES)


GBNMetropolisNode::GBNMetropolisNode()
: GBNVariable(), m_currentMean(UNKNOWN_REAL_VALUE), m_sumOfValues(UNKNOWN_REAL_VALUE), m_sumOfSquaredValues(UNKNOWN_REAL_VALUE)
{
}

double GBNMetropolisNode::markovBlanket(double x)
{
	double d;
	double lik = likelihood(x);
	if(lik <= 0)
	{
		if(lik < 0)
			throw Ex("Negative likelihood?");
		return MIN_LOG_PROB;
	}
	double logSum = log(lik);
	if(logSum >= MIN_LOG_PROB)
	{
		const vector<GBNVariable*>& kids = children();
		for(vector<GBNVariable*>::const_iterator it = kids.begin(); it != kids.end(); it++)
		{
			GBNVariable* pChildNode = *it;
			double oldVal = m_currentMean;
			m_currentMean = x;
			double cv = pChildNode->currentValue();
			lik = pChildNode->likelihood(cv);
			if(lik <= 0)
			{
				if(lik < 0)
					throw Ex("Negative likelihood?");
				return MIN_LOG_PROB;
			}
			d = log(lik);
			m_currentMean = oldVal;
			if(d >= MIN_LOG_PROB)
				logSum += d;
			else
				return MIN_LOG_PROB;
		}
		return logSum;
	}
	else
		return MIN_LOG_PROB;
}

void GBNMetropolisNode::metropolis(GRand* pRand)
{
	currentValue(); // (This needs to be called to potentially initialize m_sumOfValues and m_sumOfSquaredValues)
	double sampleMean = m_sumOfValues / DEV_SAMPLES;
	double sampleVariance = ((double)DEV_SAMPLES / (DEV_SAMPLES - 1)) * (m_sumOfSquaredValues / DEV_SAMPLES - (sampleMean * sampleMean));
	double dCandidateValue = pRand->normal() * sqrt(sampleVariance) + m_currentMean;
	if(isDiscrete())
		dCandidateValue = floor(dCandidateValue + 0.5);
	double cand = markovBlanket(dCandidateValue);
	if(cand > MIN_LOG_PROB)
	{
		if(log(pRand->uniform()) < cand - markovBlanket(m_currentMean))
			m_currentMean = dCandidateValue;
	}
}

void GBNMetropolisNode::sample(GRand* pRand)
{
	if(m_observed)
		return;
	metropolis(pRand);
	m_sumOfValues *= DEV_DECAY;
	m_sumOfSquaredValues *= DEV_DECAY;
	m_sumOfValues += m_currentMean;
	m_sumOfSquaredValues += (m_currentMean * m_currentMean);
}

// virtual
double GBNMetropolisNode::currentValue()
{
	if(m_observed)
		return m_observedValue;
	else if(m_currentMean == UNKNOWN_REAL_VALUE)
	{
		m_currentMean = initMean();
		m_sumOfValues = DEV_SAMPLES * m_currentMean;
		m_sumOfSquaredValues = DEV_SAMPLES * (m_currentMean * m_currentMean + 1.0);
		return m_currentMean;
	}
	else
		return m_currentMean;
}







#define SQRT_2PI 2.50662827463

GBNNormal::GBNNormal(GBNNode* pDefaultVal)
: GBNMetropolisNode(), m_devIsVariance(false)
{
	m_meanAndDev.resize(2, pDefaultVal);
}

void GBNNormal::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_meanAndDev.resize(2 * catCount(), pDefaultVal);
}

void GBNNormal::setMeanAndDev(size_t cat, GBNNode* pMean, GBNNode* pDeviation)
{
	if(cat >= catCount())
		throw Ex("out of range");
	m_devIsVariance = false;
	size_t base = 2 * cat;
	m_meanAndDev[base] = pMean;
	pMean->onNewChild(this);
	m_meanAndDev[base + 1] = pDeviation;
	pDeviation->onNewChild(this);
}

void GBNNormal::setMeanAndVariance(size_t cat, GBNNode* pMean, GBNNode* pVariance)
{
	if(cat >= catCount())
		throw Ex("out of range");
	m_devIsVariance = true;
	size_t base = 2 * cat;
	m_meanAndDev[base] = pMean;
	pMean->onNewChild(this);
	m_meanAndDev[base + 1] = pVariance;
	pVariance->onNewChild(this);
}

// virtual
double GBNNormal::likelihood(double x)
{
	size_t base = 2 * currentCatIndex();
	double mean = m_meanAndDev[base]->currentValue();
	double dev = m_meanAndDev[base + 1]->currentValue();
	if(dev < 0)
		throw Ex("Negative deviation?");
	double t = x - mean;
	if(m_devIsVariance)
		return 1.0 / (sqrt(dev * 2.0 * M_PI)) * exp(-(t * t) / (2.0 * dev));
	else
		return 1.0 / (dev * SQRT_2PI) * exp(-(t * t) / (2.0 * dev * dev));
}

double GBNNormal::initMean()
{
	return m_meanAndDev[0]->currentValue();
}







GBNLogNormal::GBNLogNormal(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_meanAndDev.resize(2, pDefaultVal);
}

void GBNLogNormal::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_meanAndDev.resize(2 * catCount(), pDefaultVal);
}

void GBNLogNormal::setMeanAndDev(size_t cat, GBNNode* pMean, GBNNode* pDeviation)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 2 * cat;
	m_meanAndDev[base] = pMean;
	pMean->onNewChild(this);
	m_meanAndDev[base + 1] = pDeviation;
	pDeviation->onNewChild(this);
}

// virtual
double GBNLogNormal::likelihood(double x)
{
	size_t base = 2 * currentCatIndex();
	double mean = m_meanAndDev[base]->currentValue();
	double dev = m_meanAndDev[base + 1]->currentValue();
	double t = log(x - mean);
	return 1.0 / (x * dev * SQRT_2PI) * exp(-(t * t) / (2.0 * dev * dev));
}

double GBNLogNormal::initMean()
{
	return exp(m_meanAndDev[0]->currentValue());
}








GBNPareto::GBNPareto(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_alphaAndM.resize(2, pDefaultVal);
}

void GBNPareto::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_alphaAndM.resize(2 * catCount(), pDefaultVal);
}

void GBNPareto::setAlphaAndM(size_t cat, GBNNode* pAlpha, GBNNode* pM)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 2 * cat;
	m_alphaAndM[base] = pAlpha;
	pAlpha->onNewChild(this);
	m_alphaAndM[base + 1] = pM;
	pM->onNewChild(this);
}

// virtual
double GBNPareto::likelihood(double x)
{
	size_t base = 2 * currentCatIndex();
	double alpha = m_alphaAndM[base]->currentValue();
	double m = m_alphaAndM[base + 1]->currentValue();
	if(x < m)
		return 0;
	return alpha * pow(m, alpha) / pow(x, alpha + 1.0);
}

double GBNPareto::initMean()
{
	return m_alphaAndM[1]->currentValue();
}








GBNUniformDiscrete::GBNUniformDiscrete(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_minAndMax.resize(2, pDefaultVal);
}

void GBNUniformDiscrete::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_minAndMax.resize(2 * catCount(), pDefaultVal);
}

void GBNUniformDiscrete::setMinAndMax(size_t cat, GBNNode* pMin, GBNNode* pMax)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 2 * cat;
	m_minAndMax[base] = pMin;
	pMin->onNewChild(this);
	m_minAndMax[base + 1] = pMax;
	pMax->onNewChild(this);
}

// virtual
double GBNUniformDiscrete::likelihood(double x)
{
	size_t base = 2 * currentCatIndex();
	double a = std::ceil(m_minAndMax[base]->currentValue());
	double b = std::floor(m_minAndMax[base + 1]->currentValue());
	if(x < a)
		return 0.0;
	if(x > b)
		return 0.0;
	return 1.0 / (b - a + 1.0);
}

double GBNUniformDiscrete::initMean()
{
	return std::floor(0.5 * (m_minAndMax[0]->currentValue() + m_minAndMax[1]->currentValue()));
}








GBNUniformContinuous::GBNUniformContinuous(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_minAndMax.resize(2, pDefaultVal);
}

void GBNUniformContinuous::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_minAndMax.resize(2 * catCount(), pDefaultVal);
}

void GBNUniformContinuous::setMinAndMax(size_t cat, GBNNode* pMin, GBNNode* pMax)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 2 * cat;
	m_minAndMax[base] = pMin;
	pMin->onNewChild(this);
	m_minAndMax[base + 1] = pMax;
	pMax->onNewChild(this);
}

// virtual
double GBNUniformContinuous::likelihood(double x)
{
	size_t base = 2 * currentCatIndex();
	double a = m_minAndMax[base]->currentValue();
	double b = m_minAndMax[base + 1]->currentValue();
	if(x < a)
		return 0.0;
	if(x > b)
		return 0.0;
	return 1.0 / (b - a);
}

double GBNUniformContinuous::initMean()
{
	return 0.5 * (m_minAndMax[0]->currentValue() + m_minAndMax[1]->currentValue());
}








GBNPoisson::GBNPoisson(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_lambda.resize(1, pDefaultVal);
}

void GBNPoisson::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_lambda.resize(1 * catCount(), pDefaultVal);
}

void GBNPoisson::setLambda(size_t cat, GBNNode* pLambda)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 1 * cat;
	m_lambda[base] = pLambda;
	pLambda->onNewChild(this);
}

// virtual
double GBNPoisson::likelihood(double x)
{
	size_t base = 1 * currentCatIndex();
	double l = m_lambda[base]->currentValue();
	if(x < 0)
		return 0.0;
	return pow(l, x) * exp(-l) / GMath::gamma(x + 1);
}

double GBNPoisson::initMean()
{
	return m_lambda[0]->currentValue();
}








GBNExponential::GBNExponential(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_lambda.resize(1, pDefaultVal);
}

void GBNExponential::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_lambda.resize(1 * catCount(), pDefaultVal);
}

void GBNExponential::setLambda(size_t cat, GBNNode* pLambda)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 1 * cat;
	m_lambda[base] = pLambda;
	pLambda->onNewChild(this);
}

// virtual
double GBNExponential::likelihood(double x)
{
	size_t base = 1 * currentCatIndex();
	double l = m_lambda[base]->currentValue();
	if(x < 0)
		return 0.0;
	return l * exp(-l * x);
}

double GBNExponential::initMean()
{
	return 0.0;
}








GBNBeta::GBNBeta(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_alphaAndBeta.resize(2, pDefaultVal);
}

void GBNBeta::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_alphaAndBeta.resize(2 * catCount(), pDefaultVal);
}

void GBNBeta::setAlphaAndBeta(size_t cat, GBNNode* pAlpha, GBNNode* pBeta)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 2 * cat;
	m_alphaAndBeta[base] = pAlpha;
	pAlpha->onNewChild(this);
	m_alphaAndBeta[base + 1] = pBeta;
	pBeta->onNewChild(this);
}

// virtual
double GBNBeta::likelihood(double x)
{
	if(x < 0.0 || x > 1.0)
		return 0.0;
	size_t base = 2 * currentCatIndex();
	double alpha = m_alphaAndBeta[base]->currentValue();
	double beta = m_alphaAndBeta[base + 1]->currentValue();
	double denom = GMath::gamma(alpha) * GMath::gamma(beta);
	if(std::abs(denom) < 1e-15)
		denom = (denom < 0 ? -1e-15 : 1e-15);
	return GMath::gamma(alpha + beta) / denom * pow(x, alpha - 1.0) * pow(1.0 - x, beta - 1.0);
}

double GBNBeta::initMean()
{
	double alpha = m_alphaAndBeta[0]->currentValue();
	double beta = m_alphaAndBeta[1]->currentValue();
	return std::max(0.0, std::min(1.0, alpha / std::max(0.01, alpha + beta)));
}








GBNGamma::GBNGamma(GBNNode* pDefaultVal)
: GBNMetropolisNode(), m_betaIsScaleInsteadOfRate(false)
{
	m_alphaAndBeta.resize(2, pDefaultVal);
}

void GBNGamma::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_alphaAndBeta.resize(2 * catCount(), pDefaultVal);
}

void GBNGamma::setAlphaAndBeta(size_t cat, GBNNode* pAlpha, GBNNode* pBeta)
{
	if(cat >= catCount())
		throw Ex("out of range");
	m_betaIsScaleInsteadOfRate = false;
	size_t base = 2 * cat;
	m_alphaAndBeta[base] = pAlpha;
	pAlpha->onNewChild(this);
	m_alphaAndBeta[base + 1] = pBeta;
	pBeta->onNewChild(this);
}

void GBNGamma::setShapeAndScale(size_t cat, GBNNode* pK, GBNNode* pTheta)
{
	if(cat >= catCount())
		throw Ex("out of range");
	m_betaIsScaleInsteadOfRate = true;
	size_t base = 2 * cat;
	m_alphaAndBeta[base] = pK;
	pK->onNewChild(this);
	m_alphaAndBeta[base + 1] = pTheta;
	pTheta->onNewChild(this);
}

// virtual
double GBNGamma::likelihood(double x)
{
	if(x < 0.0)
		return 0.0;
	size_t base = 2 * currentCatIndex();
	double alpha = m_alphaAndBeta[base]->currentValue();
	double beta = m_alphaAndBeta[base + 1]->currentValue();
	if(m_betaIsScaleInsteadOfRate)
		beta = 1.0 / beta;
	return pow(beta, alpha) * pow(x, alpha - 1.0) * exp(-beta * x) / GMath::gamma(alpha);
}

double GBNGamma::initMean()
{
	double alpha = m_alphaAndBeta[0]->currentValue();
	double beta = m_alphaAndBeta[1]->currentValue();
	if(m_betaIsScaleInsteadOfRate)
		beta = 1.0 / beta;
	return std::max(0.0, alpha / std::max(.01, beta));
}








GBNInverseGamma::GBNInverseGamma(GBNNode* pDefaultVal)
: GBNMetropolisNode()
{
	m_alphaAndBeta.resize(2, pDefaultVal);
}

void GBNInverseGamma::addCatParent(GBNCategorical* pNode, GBNNode* pDefaultVal)
{
	m_catParents.push_back(pNode);
	pNode->onNewChild(this);
	m_alphaAndBeta.resize(2 * catCount(), pDefaultVal);
}

void GBNInverseGamma::setAlphaAndBeta(size_t cat, GBNNode* pAlpha, GBNNode* pBeta)
{
	if(cat >= catCount())
		throw Ex("out of range");
	size_t base = 2 * cat;
	m_alphaAndBeta[base] = pAlpha;
	pAlpha->onNewChild(this);
	m_alphaAndBeta[base + 1] = pBeta;
	pBeta->onNewChild(this);
}

// virtual
double GBNInverseGamma::likelihood(double x)
{
	if(x < 0.0)
		return 0.0;
	size_t base = 2 * currentCatIndex();
	double alpha = m_alphaAndBeta[base]->currentValue();
	double beta = m_alphaAndBeta[base + 1]->currentValue();
	return pow(beta, alpha) / GMath::gamma(alpha) * pow(x, -alpha - 1.0) * exp(-beta / x);
}

double GBNInverseGamma::initMean()
{
	double alpha = m_alphaAndBeta[0]->currentValue();
	double beta = m_alphaAndBeta[1]->currentValue();
	double mean = std::max(0.0, beta / (std::max(0.5, alpha - 1.0)));
	if(mean < 0.0)
		throw Ex("Only positive values are supported");
	return mean;
}







GBayesNet::GBayesNet(size_t seed)
: m_heap(2048), m_rand(seed)
{
	m_pConstOne = newConst(1.0);
}

GBayesNet::~GBayesNet()
{
	for(size_t i = 0; i < m_nodes.size(); i++)
		m_nodes[i]->~GBNNode();
}

GBNConstant* GBayesNet::newConst(double val)
{
	char* pDest = m_heap.allocAligned(sizeof(GBNConstant));
	return new (pDest) GBNConstant(val);
}

GBNSum* GBayesNet::newSum()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNSum));
	return new (pDest) GBNSum();
}

GBNProduct* GBayesNet::newProduct()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNProduct));
	GBNProduct* pNode = new (pDest) GBNProduct();
	m_nodes.push_back(pNode);
	return pNode;
}

GBNMath* GBayesNet::newMath(GBNNode* pParent, GBNMath::math_op operation)
{
	char* pDest = m_heap.allocAligned(sizeof(GBNMath));
	GBNMath* pNode = new (pDest) GBNMath(pParent, operation);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNCategorical* GBayesNet::newCat(size_t categories)
{
	char* pDest = m_heap.allocAligned(sizeof(GBNCategorical));
	GBNCategorical* pNode = new (pDest) GBNCategorical(categories, m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNNormal* GBayesNet::newNormal()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNNormal));
	GBNNormal* pNode = new (pDest) GBNNormal(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNLogNormal* GBayesNet::newLogNormal()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNLogNormal));
	GBNLogNormal* pNode = new (pDest) GBNLogNormal(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNPareto* GBayesNet::newPareto()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNPareto));
	GBNPareto* pNode = new (pDest) GBNPareto(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNUniformDiscrete* GBayesNet::newUniformDiscrete()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNUniformDiscrete));
	GBNUniformDiscrete* pNode = new (pDest) GBNUniformDiscrete(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNUniformContinuous* GBayesNet::newUniformContinuous()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNUniformContinuous));
	GBNUniformContinuous* pNode = new (pDest) GBNUniformContinuous(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNPoisson* GBayesNet::newPoisson()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNPoisson));
	GBNPoisson* pNode = new (pDest) GBNPoisson(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNExponential* GBayesNet::newExponential()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNExponential));
	GBNExponential* pNode = new (pDest) GBNExponential(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNBeta* GBayesNet::newBeta()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNBeta));
	GBNBeta* pNode = new (pDest) GBNBeta(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNGamma* GBayesNet::newGamma()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNGamma));
	GBNGamma* pNode = new (pDest) GBNGamma(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

GBNInverseGamma* GBayesNet::newInverseGamma()
{
	char* pDest = m_heap.allocAligned(sizeof(GBNInverseGamma));
	GBNInverseGamma* pNode = new (pDest) GBNInverseGamma(m_pConstOne);
	m_sampleNodes.push_back(pNode);
	m_nodes.push_back(pNode);
	return pNode;
}

void GBayesNet::sample()
{
	// Shuffle the order of the nodes
	for(size_t i = m_sampleNodes.size(); i > 1; i--)
		std::swap(m_sampleNodes[i - 1], m_sampleNodes[(size_t)m_rand.next(i)]);

	// Sample each node
	for(vector<GBNVariable*>::iterator it = m_sampleNodes.begin(); it != m_sampleNodes.end(); it++)
		(*it)->sample(&m_rand);
}

void GBayesNet_simpleTest()
{
	GBayesNet bn;
	GBNCategorical* pPar = bn.newCat(2);
	pPar->setWeights(0, bn.newConst(0.4), bn.newConst(0.6));

	GBNNormal* pChild = bn.newNormal();
	pChild->addCatParent(pPar, bn.def());
	pChild->setMeanAndDev(0, bn.newConst(0.0), bn.newConst(1.0));
	pChild->setMeanAndDev(1, bn.newConst(3.0), bn.newConst(2.0));

	pChild->setObserved(1.0);

	for(size_t burnin = 0; burnin < 10000; burnin++)
		bn.sample();
	size_t parCount = 0;
	size_t sampleCount = 0;
	for(size_t sample = 0; sample < 50000; sample++)
	{
		bn.sample();
		if(pPar->currentValue() == 0.0)
			parCount++;
		sampleCount++;
	}

	if(std::abs((double)parCount / sampleCount - 0.5714286) > 0.003)
		throw Ex("Not close enough");
}

void GBayesNet_threeTest()
{
	GBayesNet bn;
	GBNCategorical* pA = bn.newCat(2);
	pA->setWeights(0, bn.newConst(2.0 / 5.0), bn.newConst(3.0 / 5.0));

	GBNCategorical* pB = bn.newCat(2);
	pB->addCatParent(pA, bn.def());
	pB->setWeights(0, bn.newConst(2.0 / 3.0), bn.newConst(1.0 / 3.0));
	pB->setWeights(1, bn.newConst(3.0 / 7.0), bn.newConst(4.0 / 7.0));

	GBNCategorical* pC = bn.newCat(2);
	pC->addCatParent(pB, bn.def());
	pC->setWeights(0, bn.newConst(1.0 / 2.0), bn.newConst(1.0 / 2.0));
	pC->setWeights(1, bn.newConst(1.0 / 3.0), bn.newConst(2.0 / 3.0));

	pA->setObserved(0.0);
	pC->setObserved(0.0);

	for(size_t burnin = 0; burnin < 10000; burnin++)
		bn.sample();
	size_t bCount = 0;
	size_t sampleCount = 0;
	for(size_t sample = 0; sample < 50000; sample++)
	{
		bn.sample();
		if(pB->currentValue() == 0.0)
			bCount++;
		sampleCount++;
	}

	if(std::abs((double)bCount / sampleCount - 0.75) > 0.005)
		throw Ex("Not close enough");
}

void GBayesNet_alarmTest()
{
	// This example is given in Russell and Norvig page 504. (See also http://www.d.umn.edu/~rmaclin/cs8751/Notes/chapter14a.pdf)
	GBayesNet bn;
	GBNCategorical* pBurglary = bn.newCat(2);
	pBurglary->setWeights(0, bn.newConst(0.001), bn.newConst(0.999));

	GBNCategorical* pEarthquake = bn.newCat(2);
	pEarthquake->setWeights(0, bn.newConst(0.002), bn.newConst(0.998));

	GBNCategorical* pAlarm = bn.newCat(2);
	pAlarm->addCatParent(pBurglary, bn.def());
	pAlarm->addCatParent(pEarthquake, bn.def());
	pAlarm->setWeights(0, bn.newConst(0.95), bn.newConst(0.05));
	pAlarm->setWeights(1, bn.newConst(0.29), bn.newConst(0.71));
	pAlarm->setWeights(2, bn.newConst(0.94), bn.newConst(0.06));
	pAlarm->setWeights(3, bn.newConst(0.001), bn.newConst(0.999));

	GBNCategorical* pJohnCalls = bn.newCat(2);
	pJohnCalls->addCatParent(pAlarm, bn.def());
	pJohnCalls->setWeights(0, bn.newConst(0.9), bn.newConst(0.1));
	pJohnCalls->setWeights(1, bn.newConst(0.05), bn.newConst(0.95));

	GBNCategorical* pMaryCalls = bn.newCat(2);
	pMaryCalls->addCatParent(pAlarm, bn.def());
	pMaryCalls->setWeights(0, bn.newConst(0.7), bn.newConst(0.3));
	pMaryCalls->setWeights(1, bn.newConst(0.01), bn.newConst(0.99));

	pJohnCalls->setObserved(0.0);
	pMaryCalls->setObserved(0.0);

	GRand rand(0);
	for(size_t burnin = 0; burnin < 10000; burnin++)
		bn.sample();
	size_t sampleCount = 0;
	vector<double> probs;
	probs.resize(3);
	for(size_t sample = 0; sample < 50000; sample++)
	{
		bn.sample();
		sampleCount++;
		probs[0] *= (1.0 - 1.0 / sampleCount);
		if(pBurglary->currentValue() == 0.0)
			probs[0] += (1.0 / sampleCount);
		probs[1] *= (1.0 - 1.0 / sampleCount);
		if(pEarthquake->currentValue() == 0.0)
			probs[1] += (1.0 / sampleCount);
		probs[2] *= (1.0 - 1.0 / sampleCount);
		if(pAlarm->currentValue() == 0.0)
			probs[2] += (1.0 / sampleCount);
	}

	if(std::abs(probs[0] - 0.284) > 0.005)
		throw Ex("Not close enough");
	if(std::abs(probs[1] - 0.176) > 0.005)
		throw Ex("Not close enough");
	if(std::abs(probs[2] - 0.761) > 0.005)
		throw Ex("Not close enough");
}

void GBayesNet::test()
{
	GBayesNet_simpleTest();
	GBayesNet_threeTest();
	GBayesNet_alarmTest();
}

