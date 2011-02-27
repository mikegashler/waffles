#include "PolicyLearner.h"
#include <GClasses/GError.h>
#include <GClasses/GOptimizer.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GActivation.h>
#include <GClasses/GRand.h>
#include <GClasses/GTwt.h>
#include <GClasses/GFile.h>
#include <GClasses/GEvolutionary.h>
#include <GClasses/GTime.h>
#include <GClasses/GVec.h>
#include "Body.h"
#include "DemoApplication.h"
#include "btBulletDynamicsCommon.h"

using namespace GClasses;

class PolicyCritic : public GTargetFunction
{
protected:
	GNeuralNet* m_pNN;
	GRand* m_pRand;
	const double* m_pInitialVec;

public:
	PolicyCritic(GNeuralNet* pNN, GRand* pRand, const double* pInitialVec)
	: GTargetFunction(pNN->countWeights()), m_pNN(pNN), m_pRand(pRand), m_pInitialVec(pInitialVec)
	{
	}

	virtual ~PolicyCritic()
	{
	}

	virtual void initVector(double* pVector)
	{
		GVec::copy(pVector, m_pInitialVec, m_pRelation->size());
	}

	virtual bool isStable() { return false; }
	virtual bool isConstrained() { return false; }

protected:
	virtual double computeError(const double* pVector)
	{
		RagdollDemo simulation;
		m_pNN->setWeights(pVector);
		simulation.SetPolicy(m_pNN);
		float height = 0;
		int i;
		for(i = 0; i < 240; i++)
		{
			simulation.advanceTime(1.0f / 60);
			height = std::max(height, simulation.getRagDoll(0)->getLowestPoint());
		}
		return -(double)height;
	}
};



// This is a "seed" jumping policy that I made by hand. It's not very good,
// but we'll use an evolutionary algorithm to refine it.
//                        spine    lhip    lknee   rhip    rknee   lshldr   lelbow   rshldr   relbow    lankle  rankle
const double g_squat[] = { 0.4,    -0.4,    0.4,   -0.4,    0.4,     0.4,     0.4,     0.4,     0.4,    -0.4,    -0.4 };
const double g_jump[] =  { -0.4,    0.4,   -0.4,    0.4,   -0.4,    -0.4,    -0.4,    -0.4,    -0.4,     0.4,     0.4 };
const double g_tuck[] =  { 0.4,    -0.4,    0.4,   -0.4,    0.4,    -0.4,    -0.4,    -0.4,    -0.4,     0.4,     0.4 };

void manualPolicy(const double* pIn, double* pOut)
{
	GAssert(sizeof(g_squat) == LABEL_DIMS * sizeof(double));
	double time = pIn[0];
	if(time < 0.2)
		GVec::copy(pOut, g_squat, LABEL_DIMS);
	else if(time < 0.6)
		GVec::copy(pOut, g_jump, LABEL_DIMS);
	else
		GVec::copy(pOut, g_tuck, LABEL_DIMS);
}

void GenerateSeedTrainingSet(GMatrix* pData, GRand* pRand)
{
	GAssert(FEATURE_DIMS == 3);
	double* pPat;
	double d;
	for(d = 0.0; d < 1.2; d += 0.0001)
	{
		pPat = pData->newRow();
		pPat[0] = d;
		pPat[1] = pRand->normal();
		pPat[2] = pRand->normal();
		manualPolicy(pPat, pPat + FEATURE_DIMS);
	}
}

void Train()
{
	printf("Learning the seed policy...\n");
	GRand prng(0);
	GMixedRelation* pRel = new GMixedRelation;
	sp_relation pRelation = pRel;
	int i;
	for(i = 0; i < LABEL_DIMS + FEATURE_DIMS; i++)
		pRel->addAttr(0);
	GMatrix data(pRelation);
	GenerateSeedTrainingSet(&data, &prng);
	GNeuralNet nn(&prng);
	nn.addLayer(8);
	nn.setActivationFunction(new GActivationBiDir(), true);
	nn.train(&data, LABEL_DIMS);
	nn.clipWeights(36.0); // ensure that network is still somewhat malleable
	int weightCount = nn.countWeights();
	GTEMPBUF(double, hintVec, weightCount);
	nn.weights(hintVec);

	printf("Initializing a population for the evolutionary search...\n");
	PolicyCritic critic(&nn, &prng, hintVec);
	//GAssert(critic.Critique(hintVec) < -0.5, "bad critique");
	//printf("HintVec Score: %lg\n", critic.Critique(hintVec));
	GEvolutionaryOptimizer search(&critic, 15/*population*/, &prng, .93/*moreFitSurvivalProbability*/);

	printf("Evolving better policies...\n");
	char szBuf[64];
	char szTime[64];
	double err;
	for(i = 0; i <= 25000; i++)
	{
		err = search.iterate();
		if(i % 250 == 0)
		{
			GTwtDoc doc;
			nn.setWeights(search.currentVector());
			doc.setRoot(nn.toTwt(&doc));
			sprintf(szBuf, "policy%d.twt", i);
			printf("Saving %s (height=%lg) at %s\n", szBuf, -err, GTime::asciiTime(szTime, 64, false));
			doc.save(szBuf);
		}
	}
}

GNeuralNet* LoadPolicy(const char* szFilename, GRand* pRand)
{
	GTwtDoc doc;
	doc.load(szFilename);
	return new GNeuralNet(doc.root(), pRand);
}

GNeuralNet* TrainPolicy()
{
	GRand* pRand = new GRand(1);
	GMixedRelation* pRel = new GMixedRelation;
	sp_relation pRelation = pRel;
	int i;
	for(i = 0; i < FEATURE_DIMS + LABEL_DIMS; i++)
		pRel->addAttr(0);
	GMatrix data(pRelation);
	GenerateSeedTrainingSet(&data, pRand);
	GNeuralNet* pNN = new GNeuralNet(pRand);
	//pNN->SetMinImprovement(0.01);
	//pNN->SetIterationsPerValidationCheck(600);
	pNN->addLayer(4);
	pNN->train(&data, LABEL_DIMS);
	return pNN;
}

