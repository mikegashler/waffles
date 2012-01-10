// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include "Manifold.h"
#include <GClasses/GManifold.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GBits.h>
#include <GClasses/GFile.h>
#include <GClasses/GTime.h>
#include <GClasses/GThread.h>
#include <GClasses/GVec.h>
#include <GClasses/GMath.h>
#include <GClasses/GHillClimber.h>
#include <GClasses/GHeap.h>
#include <GClasses/GApp.h>
#include <math.h>
#include <time.h>

using namespace GClasses;

#define SWISS_ROLL_POINTS 2000

#define FACE_WIDTH 43
#define FACE_HEIGHT 38
#define FACE_COUNT 50
#define FACE_ROWS 5
#define FACE_NEIGHBORS 6

class ManifoldModel;

class Compute2DErrorCritic : public GTargetFunction
{
protected:
	ManifoldModel* m_pModel;

public:
	Compute2DErrorCritic(ManifoldModel* pModel) : GTargetFunction(5)
	{
		m_pModel = pModel;
	}

	virtual ~Compute2DErrorCritic()
	{
	}

	virtual bool isStable() { return true; }
	virtual bool isConstrained() { return false; }

protected:
	virtual void initVector(double* pVector)
	{
		GVec::setAll(pVector, 0.0, 5);
	}

	virtual double computeError(const double* pVector);
};

class Compute1DErrorCritic : public GTargetFunction
{
protected:
	ManifoldModel* m_pModel;

public:
	Compute1DErrorCritic(ManifoldModel* pModel) : GTargetFunction(2)
	{
		m_pModel = pModel;
	}

	virtual ~Compute1DErrorCritic()
	{
	}

	virtual bool isStable() { return true; }
	virtual bool isConstrained() { return false; }

protected:
	virtual void initVector(double* pVector)
	{
		GVec::setAll(pVector, 0.0, 2);
	}

	virtual double computeError(const double* pVector);
};



class ManifoldModel
{
protected:
	sp_relation m_pRelation;
	GMatrix* m_pData;
	GManifoldSculpting* m_pSculpter;
	double* m_pIdealResults;

public:
	ManifoldModel()
	{
		m_pData = NULL;
		m_pSculpter = NULL;
		m_pIdealResults = NULL;
		srand(0); // arbitrary value so we always get consistent results
	}

	virtual ~ManifoldModel()
	{
		delete(m_pSculpter);
		delete(m_pData);
		delete[] m_pIdealResults;
	}

	double* GetPoint(int n)
	{
		return m_pSculpter->data().row(n);
	}

	int GetPointCount()
	{
		return m_pSculpter->data().rows();
	}

	GManifoldSculpting* GetSculpter()
	{
		return m_pSculpter;
	}

	void ToMatrix(const char* szFilename)
	{
		FILE* pFile = fopen(szFilename, "w");
		if(!pFile)
			ThrowError("failed to create file");
		size_t nDataPoints = m_pData->rows();
		for(size_t dim = 0; dim < m_pRelation->size(); dim++)
		{
			for(size_t n = 0; n < nDataPoints; n++)
			{
				if(n > 0)
					fprintf(pFile, "\t");
				fprintf(pFile, "%f", m_pSculpter->data().row(n)[dim]);
			}
			fprintf(pFile, "\n");
		}
		fclose(pFile);
	}

	void FromMatrix(const char* szFilename, int nTargetDimensions)
	{
		FILE* pFile = fopen(szFilename, "r");
		if(!pFile)
			ThrowError("failed to open file");
		FileHolder hFile(pFile);
		int nDataPoints = m_pData->rows();
		char* pBuf = new char[nDataPoints * 20];
		Holder<char> hLine(pBuf);
		char* pData;
		int d;
		for(d = 0; d < nTargetDimensions; d++)
		{
			pData = fgets(pBuf, nDataPoints * 20, pFile);
			if(!pData)
			{
				GAssert(false);
				break;
			}
			int i;
			for(i = 0; i < nDataPoints; i++)
			{
				while(*pData > '\0' && *pData <= ' ')
					pData++;
				double val = atof(pData);
				m_pSculpter->data().row(i)[d] = val;
				while(*pData > ' ')
					pData++;
			}
		}
		int i;
		for(i = 0; i < nDataPoints; i++)
		{
			m_pSculpter->data().row(i)[2] = 0;
		}
	}

	int MeasureSingleDimOrderErrors()
	{
/*
		int nPos = 0;
		int nNeg = 0;
		int i;
		int nCount = m_pSculpter->data().size();
		double* pVec1;
		double* pVec2;
		for(i = 1; i < nCount; i++)
		{
			pVec1 = m_pSculpter->GetVector(i - 1);
			pVec2 = m_pSculpter->GetVector(i);
			if(pVec1[0] > pVec2[0])
				nPos++;
			else if(pVec1[0] < pVec2[0])
				nNeg++;
		}
		return MIN(nNeg, nPos);
*/

		int nErrors = 0;
		int i;
		int nCount = m_pSculpter->data().rows();
		double* pVec1 = m_pSculpter->data().row(0);
		double* pVec2 = m_pSculpter->data().row(1);
		int nPrevSign = GBits::sign(pVec2[0] - pVec1[0]);
		int nSign;
		for(i = 2; i < nCount; i++)
		{
			pVec1 = m_pSculpter->data().row(i - 1);
			pVec2 = m_pSculpter->data().row(i);
			nSign = GBits::sign(pVec2[0] - pVec1[0]);
			if(nSign != nPrevSign)
			{
				nErrors++;
				nPrevSign = nSign;
			}
		}
		return nErrors;

	}

	void DoLLE(int nNeighbors)
	{
/*m_pRelation = new GRelation();
m_pRelation->AddAttr(0);
m_pRelation->AddAttr(0);
m_pData->releaseAllVectors();
double* pVec;
pVec = new double[2];	pVec[0] = -20;	pVec[1] = -8;	m_pData->AddPattern(pVec);
pVec = new double[2];	pVec[0] = -10;	pVec[1] = -1;	m_pData->AddPattern(pVec);
pVec = new double[2];	pVec[0] = 0.00001;	pVec[1] = 0.00001;	m_pData->AddPattern(pVec);
pVec = new double[2];	pVec[0] = 10;	pVec[1] = 1;	m_pData->AddPattern(pVec);
pVec = new double[2];	pVec[0] = 20;	pVec[1] = 8;	m_pData->AddPattern(pVec);
nNeighbors = 2;*/


/*
		GRand prng(0);
		GMatrix* pOldData = m_pData;
		m_pData = GLLE::DoLLE(pOldData, 3, 2, nNeighbors, &prng);
		delete(pOldData);
		m_pSculpter->SetData(m_pData);
*/
	}

	double Compute2DError(double dRotation, double dXScale, double dYScale, double dX, double dY)
	{
		// Compute the mean
		int i;
		int nPointCount = m_pSculpter->data().rows();
		double* pVec;
		double mean[2];
		mean[0] = 0;
		mean[1] = 0;
		for(i = 0; i < nPointCount; i++)
		{
			pVec = m_pSculpter->data().row(i);
			mean[0] += pVec[0];
			mean[1] += pVec[1];
		}
		mean[0] /= nPointCount;
		mean[1] /= nPointCount;

		// Compute the error
		double* pIdeal;
		double d, t;
		double vec[2];
		double dError = 0;
		for(i = 0; i < nPointCount; i++)
		{
			pVec = m_pSculpter->data().row(i);
			pVec[0] -= mean[0];
			pVec[1] -= mean[1];
			d = sqrt(GVec::squaredMagnitude(pVec, 2));
			t = atan2(pVec[1], pVec[0]);
			pVec[0] += mean[0];
			pVec[1] += mean[1];
			t += dRotation;
			vec[0] = d * cos(t) * dXScale + mean[0] + dX;
			vec[1] = d * sin(t) * dYScale + mean[1] + dY;
			pIdeal = &m_pIdealResults[2 * i];
			dError += GVec::squaredDistance(vec, pIdeal, 2);
		}
		return dError / nPointCount;
	}

	double Measure2DError()
	{
		GAssert(m_pIdealResults); // ideal results were not computed
		Compute2DErrorCritic critic(this);
		GMomentumGreedySearch searcher(&critic);
/*		double dInitVec[5];
		dInitVec[0] = GBits::GetRandomDouble() * 6.283; // rotation
		dInitVec[1] = GBits::GetRandomDouble(); // x-scale
		dInitVec[2] = GBits::GetRandomDouble(); // y-scale
		dInitVec[3] = GBits::GetRandomDouble() - .5; // x-translate
		dInitVec[4] = GBits::GetRandomDouble() - .5; // y-translate
		searcher.SetState(dInitVec);
		searcher.SetAllStepSizes(.0001);*/
		double err = searcher.searchUntil(200, 50, 0.005);
		double* pBestVec = searcher.currentVector();
		printf("Rot=%f, XScale=%f, YScale=%f, X=%f, Y=%f\n", pBestVec[0], pBestVec[1], pBestVec[2], pBestVec[3], pBestVec[4]);
		return err;
	}

	virtual double Compute1DError(double dScale, double dAdd)
	{
		// Compute the error
		int i;
		int nPointCount = m_pSculpter->data().rows();
		double* pVec;
		double dIdeal, d;
		double dError = 0;
		for(i = 0; i < nPointCount; i++)
		{
			dIdeal = m_pIdealResults[i];
			pVec = m_pSculpter->data().row(i);
			d = dIdeal - (pVec[0] * dScale + dAdd);
			dError += (d * d);
		}
		return dError / nPointCount;
	}

	double Measure1DError()
	{
		GAssert(m_pIdealResults); // ideal results were not computed
		Compute1DErrorCritic critic(this);
		GMomentumGreedySearch searcher(&critic);
		double err = searcher.searchUntil(200, 50, 0.005);
		//double* pBestVec = searcher.GetCurrentVector();
		//printf("Scale=%f, Add=%f\n", pBestVec[0], pBestVec[1]);
		//printf("order errors: %d\n", MeasureSingleDimOrderErrors());
		return err;
	}
};



double LengthOfSineFunc(void* pThis, double x)
{
	double d = cos(x);
	return sqrt(d * d + 1.0);
}

double LengthOfSwissRoll(double x)
{
#ifdef WINDOWS
	ThrowError("not implemented yet for Windows");
	return 0;
#else
	return (x * sqrt(x * x + 1) + asinh(x)) / 2;
#endif
}

void MakeSwissRollData(GMatrix* pData, int points, GRand* pRand)
{
	for(int n = 0; n < points; n++)
	{
		double t = ((double)n * 8) / points;
		double* pVector = pData->newRow();
		pVector[0] = ((t + 2) * sin(t)) / 20 + 0.5;
		pVector[1] = pRand->uniform() / 2 + 0.25;
		pVector[2] = ((t + 2) * cos(t)) / 20 + 0.5;
	}
}

class SwissRollModel : public ManifoldModel
{
protected:

public:
	enum ManifoldType
	{
		SWISS_ROLL,
		S_CURVE,
		SPIRALS,
	};

	SwissRollModel(ManifoldType eType, int nPoints, bool bMask, int nNeighbors, double dSquishingRate, bool bComputeIdeal, int nSupervisedPoints, SwissRollModel* pTrainedModel, GRand* prng) : ManifoldModel()
	{
		// Make the relation
		m_pRelation = new GUniformRelation(3, 0);

		// Make the ARFF data
		if(bComputeIdeal)
			m_pIdealResults = new double[nPoints * (eType == SPIRALS ? 1 : 2)];
		double t;
		int n;
		m_pData = new GMatrix(m_pRelation);
		m_pData->reserve(nPoints);
		if(eType == SWISS_ROLL)
		{
			// Load the image mask (if necessary)
			GImage imageMask;
			if(bMask)
				imageMask.loadPng("mask.png");

			for(n = 0; n < nPoints; n++)
			{
				t = ((double)n * 8) / nPoints;
				while(true)
				{
					double* pVector = m_pData->newRow();
					pVector[0] = (t + 2) * sin(t) + 14;
					pVector[1] = prng->uniform() * 12 - 6;
					pVector[2] = (t + 2) * cos(t);
					if(bComputeIdeal)
					{
						m_pIdealResults[2 * n] = pVector[1];
						m_pIdealResults[2 * n + 1] = LengthOfSwissRoll(t + 2);/* - LengthOfSwissRoll(2);*/
					}
					if(bMask)
					{
						int x = (int)(n * imageMask.width() / nPoints);
						int y = (int)((pVector[1] + 6) * imageMask.height() / 12);
						unsigned int c = imageMask.pixelNearest(x, y);
						if(gGreen(c) < 128)
							break;
					}
					else
						break;
				}
			}
		}
		else if(eType == S_CURVE)
		{
			for(n = 0; n < nPoints; n++)
			{
				t = ((double)n * 2.2 * M_PI - .1 * M_PI) / nPoints;
				double* pVector = m_pData->newRow();
				pVector[0] = 1.0 - sin(t);
				pVector[1] = t;
				pVector[2] = prng->uniform() * 2;
				if(bComputeIdeal)
				{
					m_pIdealResults[2 * n] = pVector[2];
					m_pIdealResults[2 * n + 1] = (n > 0 ? GMath::integrate(LengthOfSineFunc, 0, t, n + 30, NULL) : 0);
				}
			}
		}
		else if(eType == SPIRALS)
		{
			double dHeight = 3;
			double dWraps = 1.5;
			double dSpiralLength = sqrt((dWraps * 2.0 * M_PI) * (dWraps * 2.0 * M_PI) + dHeight * dHeight);
			double dTotalLength = 2.0 * (dSpiralLength + 1); // radius = 1
			double d;
			for(n = 0; n < nPoints; n++)
			{
				t = ((double)n * dTotalLength) / nPoints;
				double* pVector = m_pData->newRow();
				if(t < dSpiralLength)
				{
					d = (dSpiralLength - t) * dWraps * 2 * M_PI / dSpiralLength; // d = radians
					pVector[0] = -cos(d);
					pVector[1] = dHeight * t / dSpiralLength;
					pVector[2] = -sin(d);
				}
				else if(t - 2.0 - dSpiralLength >= 0)
				{
					d = (t - 2.0 - dSpiralLength) * dWraps * 2 * M_PI / dSpiralLength; // d = radians
					pVector[0] = cos(d);
					pVector[1] = dHeight * (dSpiralLength - (t - 2.0 - dSpiralLength)) / dSpiralLength;
					pVector[2] = sin(d);
				}
				else
				{
					d = (t - dSpiralLength) / 2.0; // 2 = diameter
					pVector[0] = 2.0 * d - 1.0;
					pVector[1] = dHeight;
					pVector[2] = 0;
				}
				if(bComputeIdeal)
					m_pIdealResults[n] = dTotalLength * n / nPoints;
			}
		}

		// Allocate the sculpter
		m_pSculpter = new GManifoldSculpting(nNeighbors, (eType == SPIRALS ? 1 : 2), prng);

		m_pSculpter->beginTransform(m_pData);
		m_pSculpter->setSquishingRate(dSquishingRate);

		// Set the supervised points
		if(nSupervisedPoints > 0)
		{
			int i;
			for(i = 0; i < nSupervisedPoints; i++)
			{
				size_t nPoint = (size_t)prng->next(nPoints);
				GVec::copy(m_pSculpter->data().row(nPoint), pTrainedModel->GetSculpter()->data().row(nPoint), m_pSculpter->data().relation()->size());
				m_pSculpter->clampPoint(nPoint);
			}
		}
	}

	virtual ~SwissRollModel()
	{
	}

	void DoSemiSupervisedThing()
	{
		GRand prng(0);

		// Make another swiss roll
		int nNeighbors = 24;
		srand(0); // Make sure we always get consistent results
		GMatrix data(m_pRelation);
		data.reserve(SWISS_ROLL_POINTS);
		double t;
		int n;
		for(n = 0; n < SWISS_ROLL_POINTS; n++)
		{
			double* pVector = data.newRow();
			t = ((double)n * 8) / SWISS_ROLL_POINTS;
			pVector[0] = (t + 2) * sin(t) + 14;
			pVector[1] = prng.uniform() * 12 - 6;
			pVector[2] = (t + 2) * cos(t);
		}

		// Init the sculpter
		GManifoldSculpting* pSculpter = new GManifoldSculpting(nNeighbors, 2, &prng);
		m_pSculpter->beginTransform(&data);
		pSculpter->setSquishingRate(.99);

		// Set some supervised points
		for(n = 0; n < SWISS_ROLL_POINTS; n++)
		{
			if(prng.next(20) == 0)
			{
				GVec::copy(pSculpter->data().row(n), m_pSculpter->data().row(n), pSculpter->data().relation()->size());
				pSculpter->clampPoint(n);
			}
		}

		// Swap in the new sculpter
		delete(m_pSculpter);
		m_pSculpter = pSculpter;
	}
};

// virtual
double Compute2DErrorCritic::computeError(const double* pVector)
{
	return m_pModel->Compute2DError(pVector[0], pVector[1], pVector[2], pVector[3], pVector[4]);
}

// virtual
double Compute1DErrorCritic::computeError(const double* pVector)
{
	return m_pModel->Compute1DError(pVector[0], pVector[1]);
}


char* MakeEightDigitInt(int n, char* szBuf)
{
	sprintf(szBuf, "%d", n);
	int len = strlen(szBuf);
	int i;
	for(i = 7 - len; i >= 0; i--)
		szBuf[i] = '0';
	sprintf(szBuf + 8 - len, "%d", n);
	return szBuf;
}

class ImageModel : public ManifoldModel
{
protected:
	GImage* m_pImages;

public:
	ImageModel(int nSkip, int nImageCount, int nImageWidth, int nImageHeight, int nTargetDims, int nNeighbors, double dSquishingRate, const char* szFilenamePrefix, int nSupervisedPoints, ImageModel* pTrainedModel, bool bComputeIdeal, GRand* prng) : ManifoldModel()
	{
//		printf("Image Count=%d, wid=%d, hgt=%d, target dims=%d, neighbors=%d, squishing rate=%f, supervised points=%d\n", nImageCount, nImageWidth, nImageHeight, nTargetDims, nNeighbors, dSquishingRate, nSupervisedPoints);
		int nAttrs;
		m_pImages = new GImage[nImageCount];
		nAttrs = nImageWidth * nImageHeight;

		// Make the relation
		m_pRelation = new GUniformRelation(nAttrs, 0);
		int i, x, y;

		// Load the images into the ARFF data
		m_pData = new GMatrix(m_pRelation);
		m_pData->reserve(nImageCount);
		char szFilename[300];
		GApp::appPath(szFilename, 300, true);
		strcat(szFilename, szFilenamePrefix);
		int nLen = strlen(szFilename);
		double* pVector;
		int nGrayScaleValue;
		unsigned int col;
		nNeighbors = FACE_NEIGHBORS;
		char szTmp[9];
		for(i = 0; i < nImageCount; i++)
		{
			MakeEightDigitInt(i + 1 + nSkip, szTmp);
			strcpy(szFilename + nLen, szTmp);
			strcpy(szFilename + nLen + 8, ".png");
			m_pImages[i].loadPng(szFilename);
			GAssert((int)m_pImages[i].width() == nImageWidth); // unexpected size
			GAssert((int)m_pImages[i].height() == nImageHeight); // unexpected size
			pVector = m_pData->newRow();
			for(y = 0; y < nImageHeight; y++)
			{
				for(x = 0; x < nImageWidth; x++)
				{
					col = m_pImages[i].pixel(x, y);
					nGrayScaleValue = 77 * (int)gRed(col) + 150 * (int)gGreen(col) + 29 * (int)gBlue(col);
					pVector[y * nImageWidth + x]  = nGrayScaleValue;
				}
			}
		}

		if(bComputeIdeal)
		{
			m_pIdealResults = new double[m_pData->rows()];
			m_pIdealResults[0] = 0;
			double* pVec1;
			double* pVec2;
			for(size_t i = 1; i < m_pData->rows(); i++)
			{
				pVec1 = m_pData->row(i - 1);
				pVec2 = m_pData->row(i);
				m_pIdealResults[i] = m_pIdealResults[i - 1] + sqrt(GVec::squaredDistance(pVec1, pVec2, nAttrs));
			}
		}

		// Allocate and init the sculpter
		m_pSculpter = new GManifoldSculpting(nNeighbors, nTargetDims, prng);
		m_pSculpter->beginTransform(m_pData);
		m_pSculpter->setSquishingRate(dSquishingRate);
		//printf("Total shortcuts: %d\n", m_pSculpter->CountShortcuts(nNeighbors * 2));

		// Set the supervised points
		if(nSupervisedPoints > 0)
		{
			ThrowError("not implemented");
		}
	}

	virtual ~ImageModel()
	{
		delete[] m_pImages;
	}

	GImage* GetImage(int i)
	{
		return &m_pImages[i];
	}
};


void Make2DImages(const char* szFilenameIn, const char* szDir)
{
	GImage imageMaster;
	imageMaster.loadPng(szFilenameIn);
	GImage imageSubset;
	imageSubset.setSize(35, 35);
	float x, y;
	int xx, yy;
	char szFilename[256];
	char szTmp[9];
	int nImage = 1;
	for(y = 0; y < imageMaster.height() - 35; y += (float).2)
	{
		for(x = 0; x < imageMaster.width() - 35; x += (float).2)
		{
			for(yy = 0; yy < 35; yy++)
			{
				for(xx = 0; xx < 35; xx++)
					imageSubset.setPixel(xx, yy, imageMaster.interpolatePixel(x + xx, y + yy));
			}
			MakeEightDigitInt(nImage++, szTmp);
			strcpy(szFilename, szDir);
			strcat(szFilename, szTmp);
			strcat(szFilename, ".png");
			imageSubset.savePng(szFilename);
		}
	}
}

void Make2DNoiseImages(const char* szFilenameIn, const char* szDir)
{
	GImage imageMaster;
	imageMaster.loadPng(szFilenameIn);
	GImage imageNoise;
	imageNoise.setSize(imageMaster.width() + 19, imageMaster.height() + 19);
	int x, y;
	GRand prng(0);
	for(y = 0; y < (int)imageNoise.height(); y++)
	{
		for(x = 0; x < (int)imageNoise.width(); x++)
			imageNoise.setPixel(x, y, gARGB(0xff, (int)prng.next(256), (int)prng.next(256), (int)prng.next(256)));
	}
	GImage imageAll;
	imageAll.setSize(imageNoise.width(), imageNoise.height());
	char szFilename[256];
	char szTmp[9];
	int nImage = 1;
	GRect r(0, 0, imageMaster.width(), imageMaster.height());
	for(y = 0; y < 20; y++)
	{
		for(x = 0; x < 20; x++)
		{
			imageAll.copy(&imageNoise);
			imageAll.blit(x, y, &imageMaster, &r);
			MakeEightDigitInt(nImage++, szTmp);
			strcpy(szFilename, szDir);
			strcat(szFilename, szTmp);
			strcat(szFilename, ".png");
			imageAll.savePng(szFilename);
		}
	}
}


// -------------------------------------------------------------------------------

class SwissRollView : public ViewBase
{
protected:
	GFloatRect m_viewRect, m_nextRect;
	ManifoldModel* m_pModel;

public:
	SwissRollView(ManifoldModel* pModel);
	virtual ~SwissRollView();

	void SetModel(ManifoldModel* pModel) { m_pModel = pModel; }

protected:
	virtual void draw(SDL_Surface *pScreen);
	void DrawPoint(SDL_Surface *pScreen, int n);
};

SwissRollView::SwissRollView(ManifoldModel* pModel)
: ViewBase()
{
	m_pModel = pModel;

	// Set the view rect
	m_nextRect.set(-20, -15, 40, 30);
	m_viewRect = m_nextRect;
}

SwissRollView::~SwissRollView()
{
}

void SwissRollView::DrawPoint(SDL_Surface *pScreen, int n)
{
	// Draw the dot
	double* pValues = m_pModel->GetPoint(n);
	int x = (int)((pValues[0] - .3 * pValues[2] - m_viewRect.x) * m_screenRect.w / m_viewRect.w) + m_screenRect.x;
	int y = (int)((pValues[1] + .5 * pValues[2] - m_viewRect.y) * m_screenRect.h / m_viewRect.h) + m_screenRect.y;
	unsigned int col = (unsigned int)gAHSV(0xff, (float)n / SWISS_ROLL_POINTS, 1.0f, 1.0f);
	drawDot(pScreen, x, y, col, /*(n % 200) == 0 ? 20 :*/ 5);

	// Ajust the next rect
	if(pValues[0] < m_nextRect.x)
	{
		m_nextRect.w += (m_nextRect.x - (float)pValues[0]);
		m_nextRect.x = (float)pValues[0];
	}
	else if(pValues[0] > m_nextRect.x + m_nextRect.w)
		m_nextRect.w = (float)pValues[0] - m_nextRect.x;
	if(pValues[1] < m_nextRect.y)
	{
		m_nextRect.h += (m_nextRect.y - (float)pValues[1]);
		m_nextRect.y = (float)pValues[1];
	}
	else if(pValues[1] > m_nextRect.y + m_nextRect.h)
		m_nextRect.h = (float)pValues[1] - m_nextRect.y;
}

/*virtual*/ void SwissRollView::draw(SDL_Surface *pScreen)
{
	// Clear the screen
	SDL_FillRect(pScreen, NULL/*&r*/, 0x000000);
	//m_viewRect = m_nextRect;
	m_viewRect.set((m_viewRect.x * 4 + m_nextRect.x) / 5, (m_viewRect.y * 4 + m_nextRect.y) / 5, (m_viewRect.w * 4 + m_nextRect.w) / 5, (m_viewRect.h * 4 + m_nextRect.h) / 5);

	// Reset the next rect
	double* pValues = m_pModel->GetPoint(0);
	m_nextRect.set((float)pValues[0], (float)pValues[1], (float).000001, (float).000001);

	// Plot the points in approximate order from back to front so it looks like we're
	// clipping properly--this is a cheap hacky way to do clipping, but who cares?
	int n;
/*	for(n = (int)(SWISS_ROLL_POINTS * .225); n < SWISS_ROLL_POINTS / 2; n++)
		DrawPoint(pScreen, n);
	for(n = 0; n < (int)(SWISS_ROLL_POINTS * .225); n++)
		DrawPoint(pScreen, n);
*/
	for(n = 0/*SWISS_ROLL_POINTS / 2*/; n < SWISS_ROLL_POINTS; n++)
		DrawPoint(pScreen, n);

	// Add some border to the next rect and preserve aspect ratio
	m_nextRect.x -= m_nextRect.w / 4;
	m_nextRect.w += m_nextRect.w / 2;
	m_nextRect.y -= m_nextRect.h / 4;
	m_nextRect.h += m_nextRect.h / 2;
	if(m_nextRect.h * 4 > m_nextRect.w * 3)
	{
		m_nextRect.x -= (m_nextRect.h * 4 / 3 - m_nextRect.w) / 2;
		m_nextRect.w = m_nextRect.h * 4 / 3;
	}
	else
	{
		m_nextRect.y -= (m_nextRect.w * 3 / 4 - m_nextRect.h) / 2;
		m_nextRect.h = m_nextRect.w * 3 / 4;
	}
}


// -------------------------------------------------------------------------------

class ImageView : public ViewBase
{
public:
	enum mode
	{
		FACES,
		DOTS_2D,
	};

protected:
	mode m_eMode;
	GFloatRect m_viewRect, m_nextRect;
	ImageModel* m_pModel;

public:
	ImageView(ImageModel* pModel, mode eMode);
	virtual ~ImageView();

protected:
	virtual void draw(SDL_Surface* pScreen);
	void DrawFaceOneDim(SDL_Surface* pScreen, int n);
	void DrawIsomapFace(SDL_Surface* pScreen, int n);
	void Draw2DPoint(SDL_Surface* pScreen, int n);
};

ImageView::ImageView(ImageModel* pModel, mode eMode)
	: ViewBase()
{
	m_eMode = eMode;
	m_nextRect.set(0, 0, 1, 1);
	m_pModel = pModel;
}

ImageView::~ImageView()
{
}

void ImageView::DrawFaceOneDim(SDL_Surface* pScreen, int n)
{
	// Draw the face
	double* pValues = m_pModel->GetPoint(n);
	float rowSize = m_viewRect.w / FACE_ROWS;
	int row = (int)((pValues[0] - m_viewRect.x) / rowSize);
	float column = (float)(pValues[0] - m_viewRect.x) - (row * rowSize);
	int x = (int)(column * (m_screenRect.w - FACE_WIDTH) / rowSize) + m_screenRect.x;
	//unsigned int col = (unsigned int)GetSpectrumColor((float)n / FACE_COUNT);
	int y = m_screenRect.y + FACE_HEIGHT + row * (m_screenRect.h - FACE_HEIGHT) / FACE_ROWS;
	if(x >= 0 && y >= 0 && x < 800 - FACE_WIDTH && y < 600 - FACE_HEIGHT)
		blitImage(pScreen, x, y, m_pModel->GetImage(n));

	// Ajust the next rect
	if(pValues[0] < m_nextRect.x)
	{
		m_nextRect.w += (m_nextRect.x - (float)pValues[0]);
		m_nextRect.x = (float)pValues[0];
	}
	else if(pValues[0] > m_nextRect.x + m_nextRect.w)
		m_nextRect.w = (float)pValues[0] - m_nextRect.x;
}

void ImageView::Draw2DPoint(SDL_Surface* pScreen, int n)
{
	// Draw the point
	double* pValues = m_pModel->GetPoint(n);
	int x = (int)((pValues[0] - m_viewRect.x) * m_screenRect.w / m_viewRect.w) + m_screenRect.x;
	int y = (int)((pValues[1] - m_viewRect.y) * m_screenRect.h / m_viewRect.h) + m_screenRect.y;

/*
int right = n + 1;
int bot = n + 27;
if(bot < 783 && right / 27 == n / 27)
{
	pValues = m_pModel->GetPoint(right);
	int xr = (int)((pValues[0] - m_viewRect.x) * m_screenRect.w / m_viewRect.w) + m_screenRect.x;
	int yr = (int)((pValues[1] - m_viewRect.y) * m_screenRect.h / m_viewRect.h) + m_screenRect.y;
	pValues = m_pModel->GetPoint(bot);
	int xb = (int)((pValues[0] - m_viewRect.x) * m_screenRect.w / m_viewRect.w) + m_screenRect.x;
	int yb = (int)((pValues[1] - m_viewRect.y) * m_screenRect.h / m_viewRect.h) + m_screenRect.y;
	DrawLine(x, y, xr, yr);
	DrawLine(x, y, xb, yb);
}
*/

	drawDot(pScreen, x, y, gAHSV(0xff, (float)n / m_pModel->GetPointCount(), 1.0f, 1.0f), 3);

	// Ajust the next rect
	if(pValues[0] < m_nextRect.x)
	{
		m_nextRect.w += (m_nextRect.x - (float)pValues[0]);
		m_nextRect.x = (float)pValues[0];
	}
	else if(pValues[0] > m_nextRect.x + m_nextRect.w)
		m_nextRect.w = (float)pValues[0] - m_nextRect.x;
	if(pValues[1] < m_nextRect.y)
	{
		m_nextRect.h += (m_nextRect.y - (float)pValues[1]);
		m_nextRect.y = (float)pValues[1];
	}
	else if(pValues[1] > m_nextRect.y + m_nextRect.h)
		m_nextRect.h = (float)pValues[1] - m_nextRect.y;
}

/*virtual*/ void ImageView::draw(SDL_Surface *pScreen)
{
	// Clear the screen
	SDL_FillRect(pScreen, NULL/*&r*/, 0x000000);

	//m_viewRect.Set((m_viewRect.x * 4 + m_nextRect.x) / 5, (m_viewRect.y * 4 + m_nextRect.y) / 5, (m_viewRect.w * 4 + m_nextRect.w) / 5, (m_viewRect.h * 4 + m_nextRect.h) / 5);
	m_viewRect = m_nextRect;

	if(m_eMode == FACES)
	{
		// Reset the next rect
		double* pValues = m_pModel->GetPoint(0);
		m_nextRect.set((float)pValues[0], 0, (float).000001, (float).000001);

		// Plot the points
		int n;
		for(n = 0; n < FACE_COUNT; n++)
			DrawFaceOneDim(pScreen, n);
	}
	else if(m_eMode == DOTS_2D)
	{
		// Reset the next rect
		double* pValues = m_pModel->GetPoint(0);
		m_nextRect.set((float)pValues[0], (float)pValues[1], (float).000001, (float).000001);

		// Plot the points
		int nPointCount = m_pModel->GetPointCount();
		int n;
		for(n = 0; n < nPointCount; n++)
			Draw2DPoint(pScreen, n);
	}
}



// -------------------------------------------------------------------------------


class ManifoldController : public ControllerBase
{
public:
	enum WhichDemo
	{
		MC_SWISS_ROLL,
		MC_S_CURVE,
		MC_SPIRALS,
		MC_SEMI_SUPERVISED,
		MC_FACE,
		MC_LLE_SWISS_ROLL,
	};

protected:
	WhichDemo m_eDemo;
	ManifoldModel* m_pModel;

public:
	ManifoldController(WhichDemo eDemo) : ControllerBase()
	{
		m_eDemo = eDemo;
		m_pModel = NULL;
		m_pView = NULL;
	}

	virtual ~ManifoldController()
	{
		delete(m_pView);
		delete(m_pModel);
	}

	void RunModal()
	{
		if(m_eDemo == MC_SWISS_ROLL)
			DoSwissRollDemo();
		else if(m_eDemo == MC_S_CURVE)
			DoSCurveDemo();
		else if(m_eDemo == MC_SPIRALS)
			DoSpiralsDemo();
		else if(m_eDemo == MC_SEMI_SUPERVISED)
			DoSemiSupervisedDemo();
		else if(m_eDemo == MC_FACE)
			DoFaceDemo();
		else if(m_eDemo == MC_LLE_SWISS_ROLL)
			DoLLESwissRoll();
		else
			GAssert(false); // unrecognized demo
	}

	void DoSwissRollDemo()
	{
		delete(m_pView);
		delete(m_pModel);
		GRand prng(0);
		m_pModel = new SwissRollModel(SwissRollModel::SWISS_ROLL, SWISS_ROLL_POINTS, false, 20/*neighbors*/, .98, false, 0, NULL, &prng);
		m_pView = new SwissRollView(m_pModel);
		double timeOld = GTime::seconds();
		double time;
		m_pView->update();
		GManifoldSculpting* pSculpter = m_pModel->GetSculpter();
		int nDataPoints = pSculpter->data().rows();
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			pSculpter->squishPass((size_t)prng.next(nDataPoints));
			m_pView->update();
			timeOld = time;
			//if(pSculpter->GetLearningRate() / pSculpter->GetAveNeighborDist() < .001)
			//	break;
		}
	}

	void DoSCurveDemo()
	{
		delete(m_pView);
		delete(m_pModel);
		GRand prng(0);
		m_pModel = new SwissRollModel(SwissRollModel::S_CURVE, SWISS_ROLL_POINTS, false, 20/*neighbors*/, .98, false, 0, NULL, &prng);
		m_pView = new SwissRollView(m_pModel);
		double timeOld = GTime::seconds();
		double time;
		m_pView->update();
		GManifoldSculpting* pSculpter = m_pModel->GetSculpter();
		int nDataPoints = pSculpter->data().rows();
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			pSculpter->squishPass((size_t)prng.next(nDataPoints));
			m_pView->update();
			timeOld = time;
		}
	}

	void DoSpiralsDemo()
	{
		delete(m_pView);
		delete(m_pModel);
		GRand prng(0);
		m_pModel = new SwissRollModel(SwissRollModel::SPIRALS, SWISS_ROLL_POINTS, false, 20/*neighbors*/, .98, false, 0, NULL, &prng);
		m_pView = new SwissRollView(m_pModel);
		double timeOld = GTime::seconds();
		double time;
		m_pView->update();
		GManifoldSculpting* pSculpter = m_pModel->GetSculpter();
		int nDataPoints = pSculpter->data().rows();
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			pSculpter->squishPass((size_t)prng.next(nDataPoints));
			m_pView->update();
			timeOld = time;
		}
	}

	void DoSemiSupervisedDemo()
	{
		delete(m_pView);
		delete(m_pModel);
		GRand prng(0);
		m_pModel = new SwissRollModel(SwissRollModel::SWISS_ROLL, SWISS_ROLL_POINTS, false, 40/*neighbors*/, .98, false, 0, NULL, &prng);
		m_pView = new SwissRollView(m_pModel);

		// First learn the points
		int nPass = 0;
		double timeOld = GTime::seconds();
		double time;
		m_pView->update();
		GManifoldSculpting* pSculpter = m_pModel->GetSculpter();
		int nDataPoints = pSculpter->data().rows();
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			pSculpter->squishPass((size_t)prng.next(nDataPoints));
			m_pView->update();
			timeOld = time;
			//printf("Pass %d\n", nPass++);
		}

		// Now do the demo with semi-supervision
		prng.setSeed(0);
		nPass = 0;
		SwissRollModel* pPrevModel = (SwissRollModel*)m_pModel;
		m_pModel = new SwissRollModel(SwissRollModel::SWISS_ROLL, SWISS_ROLL_POINTS, false, 40/*neighbors*/, .7, false, 100, pPrevModel, &prng);
		((SwissRollView*)m_pView)->SetModel(m_pModel);
		delete(pPrevModel);
		pSculpter = m_pModel->GetSculpter();
		nDataPoints = pSculpter->data().rows();
		m_bKeepRunning = true;
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			pSculpter->squishPass((size_t)prng.next(nDataPoints));
			m_pView->update();
			timeOld = time;
			//printf("Pass %d\n", nPass++);
		}
	}

	void DoFaceDemo()
	{
		delete(m_pView);
		delete(m_pModel);

		GRand prng(0);

		// 0 (srand=0, itters=100)
		m_pModel = new ImageModel(0, 50/*count*/, 43/*wid*/, 38/*hgt*/, 1/*target dims*/, 6/*neighbors*/, .995/*squishing rate*/, "faces/", 0, NULL, true, &prng);

		m_pView = new ImageView((ImageModel*)m_pModel, ImageView::FACES);

		double timeOld = GTime::seconds();
		double time;
		m_pView->update();
		GManifoldSculpting* pSculpter = m_pModel->GetSculpter();
		int nDataPoints = pSculpter->data().rows();
//		int nCycle = 0;
		int nIterations = 0;
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			pSculpter->squishPass((size_t)prng.next(nDataPoints));
//			if(++nCycle >= 15)
//			{
				m_pView->update();
//				nCycle = 0;
//			}
			timeOld = time;
			//int nErrors = m_pModel->MeasureSingleDimOrderErrors();
			//printf("Iterations: %d, Ordering Errors: %d\n", nIterations, nErrors);
			//if(nIterations % 50 == 0)
			//{
				//double dMeanSquaredError = m_pModel->Measure1DError();
				int nOrderErrors = m_pModel->MeasureSingleDimOrderErrors();
				printf("Iterations: %d, Order Errors: %d\n", nIterations, nOrderErrors);
				if(nOrderErrors <= 0)
					break;
			//}
			nIterations++;
		}
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			timeOld = time;
			GThread::sleep(15);
		}
	}

	void DoLLESwissRoll()
	{
		delete(m_pModel);
		GRand prng(0);
		m_pModel = new SwissRollModel(SwissRollModel::SWISS_ROLL, 2000, false, 14/*neighbors*/, .99, false, 0, NULL, &prng);
		delete(m_pView);
		m_pView = new SwissRollView((ImageModel*)m_pModel);
		m_pView->update();
		m_pModel->DoLLE(14);
		double timeOld = GTime::seconds();
		double time;
		m_pView->update();
		while(m_bKeepRunning)
		{
			time = GTime::seconds();
			handleEvents(time - timeOld);
			m_pView->update();
			timeOld = time;
			GThread::sleep(100);
		}
	}
};




// -------------------------------------------------------------------------------


#define BACKGROUND_COLOR 0xffeecc77

class ManifoldMenuDialog : public GWidgetDialog
{
protected:
	ManifoldMenuController* m_pController;
	GWidgetTextButton* m_pSwissRollDemoButton;
	GWidgetTextButton* m_pSCurveDemoButton;
	GWidgetTextButton* m_pSpiralsDemoButton;
	GWidgetTextButton* m_pSemiSupervisedDemoButton;
	GWidgetTextButton* m_pFaceDemoButton;
	GWidgetTextButton* m_pLLESwissRollDemoButton;

	int m_nValidationTechnique;
	double m_dTrainingPercent;

public:
	ManifoldMenuDialog(ManifoldMenuController* pController, int w, int h);
	virtual ~ManifoldMenuDialog();

	virtual void onReleaseTextButton(GWidgetTextButton* pButton);
};

ManifoldMenuDialog::ManifoldMenuDialog(ManifoldMenuController* pController, int w, int h)
 : GWidgetDialog(w, h, BACKGROUND_COLOR)
{
	m_pController = pController;

	// Left column
	m_pSwissRollDemoButton = new GWidgetTextButton(this, 100, 100, 200, 24, "Swiss roll demo");
	m_pSemiSupervisedDemoButton = new GWidgetTextButton(this, 100, 150, 200, 24, "Semi-supervised demo");
	m_pSCurveDemoButton = new GWidgetTextButton(this, 100, 200, 200, 24, "S-Curve demo");
	m_pSpiralsDemoButton = new GWidgetTextButton(this, 100, 250, 200, 24, "Entwined spirals demo");
	//m_pFaceDemoButton = new GWidgetTextButton(this, 100, 300, 200, 24, "Face demo");

	// Right column
	//m_pLLESwissRollDemoButton = new GWidgetTextButton(this, 400, 200, 200, 24, "Swiss Roll with LLE");
}

/*virtual*/ ManifoldMenuDialog::~ManifoldMenuDialog()
{
}

/*virtual*/ void ManifoldMenuDialog::onReleaseTextButton(GWidgetTextButton* pButton)
{
	ManifoldController::WhichDemo eDemo;
	if(pButton == m_pSwissRollDemoButton)
		eDemo = ManifoldController::MC_SWISS_ROLL;
	else if(pButton == m_pSCurveDemoButton)
		eDemo = ManifoldController::MC_S_CURVE;
	else if(pButton == m_pSpiralsDemoButton)
		eDemo = ManifoldController::MC_SPIRALS;
	else if(pButton == m_pSemiSupervisedDemoButton)
		eDemo = ManifoldController::MC_SEMI_SUPERVISED;
	else if(pButton == m_pFaceDemoButton)
		eDemo = ManifoldController::MC_FACE;
	else if(pButton == m_pLLESwissRollDemoButton)
		eDemo = ManifoldController::MC_LLE_SWISS_ROLL;
	else
	{
		GAssert(false); // unrecognized button
		return;
	}
	ManifoldController c(eDemo);
	c.RunModal();
}






// -------------------------------------------------------------------------------


class ManifoldMenuView : public ViewBase
{
protected:
	ManifoldMenuDialog* m_pDialog;

public:
	ManifoldMenuView(ManifoldMenuController* pController);
	virtual ~ManifoldMenuView();

	virtual void onChar(char c);
	virtual void onMouseDown(int nButton, int x, int y);
	virtual void onMouseUp(int nButton, int x, int y);
	virtual bool onMousePos(int x, int y);

protected:
	virtual void draw(SDL_Surface *pScreen);
};

ManifoldMenuView::ManifoldMenuView(ManifoldMenuController* pController)
: ViewBase()
{
	m_pDialog = new ManifoldMenuDialog(pController, m_screenRect.w, m_screenRect.h);
}

ManifoldMenuView::~ManifoldMenuView()
{
	delete(m_pDialog);
}

/*virtual*/ void ManifoldMenuView::draw(SDL_Surface *pScreen)
{
	// Clear the screen
	SDL_FillRect(pScreen, NULL/*&r*/, 0x000000);

	// Draw the dialog
	blitImage(pScreen, m_screenRect.x, m_screenRect.y, m_pDialog->image());
}

void ManifoldMenuView::onChar(char c)
{
	m_pDialog->handleChar(c);
}

void ManifoldMenuView::onMouseDown(int nButton, int x, int y)
{
	m_pDialog->pressButton(nButton, x - m_screenRect.x, y - m_screenRect.y);
}

void ManifoldMenuView::onMouseUp(int nButton, int x, int y)
{
	m_pDialog->releaseButton(nButton);
}

bool ManifoldMenuView::onMousePos(int x, int y)
{
	return m_pDialog->handleMousePos(x - m_screenRect.x, y - m_screenRect.y);
}






// -------------------------------------------------------------------------------


ManifoldMenuController::ManifoldMenuController()
: ControllerBase()
{
	m_pView = new ManifoldMenuView(this);
}

ManifoldMenuController::~ManifoldMenuController()
{
	delete(m_pView);
}

void ManifoldMenuController::RunModal()
{
	double timeOld = GTime::seconds();
	double time;
	m_pView->update();
	while(m_bKeepRunning)
	{
		time = GTime::seconds();
		if(handleEvents(time - timeOld)) // HandleEvents returns true if it thinks the view needs to be updated
		{
			m_pView->update();
		}
		else
			GThread::sleep(10);
		timeOld = time;
	}
}

