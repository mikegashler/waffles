/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
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

#include "GFourier.h"
#include <math.h>
#include "GError.h"
#include "GHolders.h"
#include "GMath.h"
#include "GImage.h"
#include "GBits.h"
#include "GVec.h"
#include <cmath>
#include <memory>

using namespace GClasses;

void ComplexNumber::interpolate(ComplexNumber& a, double w, ComplexNumber& b)
{
/*
	// The trigonometric way
	double ta = atan2(a.imag, a.real);
	double tb = atan2(b.imag, b.real);
	if(ta - tb >= M_PI)
		tb += 2 * M_PI;
	else if(tb - ta >= M_PI)
		ta += 2 * M_PI;
	double t = w * tb + (1.0 - w) * ta;
	double ra = sqrt(a.squaredMagnitude());
	double rb = sqrt(b.squaredMagnitude());
	double r = w * rb + (1.0 - w) * ra;
	real = r * cos(t);
	imag = r * sin(t);
*/

	// The algebraic way
	double ma = sqrt(a.squaredMagnitude());
	if(ma < 1e-15)
		ma = 1.0;
	double mb = sqrt(b.squaredMagnitude());
	if(mb < 1e-15)
		mb = 1.0;
	real = w * b.real / mb + (1.0 - w) * a.real / ma;
	imag = w * b.imag / mb + (1.0 - w) * a.imag / ma;
	double m = sqrt(squaredMagnitude());
	double s = m > 1e-15 ? (w * mb + (1.0 - w) * ma) / m : 0.0;
	real *= s;
	imag *= s;

}


inline size_t ReverseBits(size_t nValue, size_t nBits)
{
	size_t n;
	size_t nReversed = 0;
	for(n = 0; n < nBits; n++)
	{
		nReversed = (nReversed << 1) | (nValue & 1);
		nValue >>= 1;
	}
	return nReversed;
}

void GFourier::fft(size_t arraySize, struct ComplexNumber* pComplexNumberArray, bool bForward)
{
	double* pData = (double*)pComplexNumberArray;

	// Make sure arraySize is a power of 2
	if(arraySize == 0 || arraySize & (arraySize - 1))
		throw Ex("Expected the array to be a power of 2");

	// Calculate the Log2 of arraySize and put it in nBits
	size_t nBits = 0;
	{
		size_t n = 1;
		while(n < arraySize)
		{
			n = n << 1;
			nBits++;
		}
	}

	// Move the data to it's reversed-bit position
	size_t totalSize = arraySize << 1;
	{
		GQUICKVEC(pTmp, totalSize);
		for(size_t n = 0; n < arraySize; n++)
		{
			size_t nReversed = ReverseBits(n, nBits);
			pTmp[nReversed << 1] = pData[n << 1];
			pTmp[(nReversed << 1) + 1] = pData[(n << 1) + 1];
		}
		memcpy(pData, pTmp.data(), sizeof(double) * totalSize);
	}

	// Calculate the angle numerator
	double dAngleNumerator;
	if(bForward)
		dAngleNumerator = -2.0 * M_PI;
	else
		dAngleNumerator = 2.0 * M_PI;

	// Do the Fast Forier Transform
	double dR0, dR1, dR2, dR3, dI0, dI1, dI2, dI3;
	for(size_t nHalfBlockSize = 1; nHalfBlockSize < arraySize; nHalfBlockSize = nHalfBlockSize << 1)
	{
		// Calculate angles, sines, and cosines
		double dAngleDelta = dAngleNumerator / ((double)(nHalfBlockSize << 1));
		double dCos1 = cos(-dAngleDelta);
		double d2Cos1 = 2 * dCos1; // So we don't have to calculate this a bunch of times
		double dCos2 = cos(-2 * dAngleDelta);
		double dSin1 = sin(-dAngleDelta);
		double dSin2 = sin(-2 * dAngleDelta);

		// Do each block
		for(size_t nStart = 0; nStart < arraySize; nStart += (nHalfBlockSize << 1))
		{
			dR1 = dCos1;
			dR2 = dCos2;
			dI1 = dSin1;
			dI2 = dSin2;
			size_t nEnd = nStart + nHalfBlockSize;
			for(size_t n = nStart; n < nEnd; n++)
			{
				dR0 = d2Cos1 * dR1 - dR2;
				dR2 = dR1;
				dR1 = dR0;
				dI0 = d2Cos1 * dI1 - dI2;
				dI2 = dI1;
				dI1 = dI0;
				size_t n2 = n + nHalfBlockSize;
				dR3 = dR0 * pData[n2 << 1] - dI0 * pData[(n2 << 1) + 1];
				dI3 = dR0 * pData[(n2 << 1) + 1] + dI0 * pData[n2 << 1];
				pData[n2 << 1] = pData[n << 1] - dR3;
				pData[(n2 << 1) + 1] = pData[(n << 1) + 1] - dI3;
				pData[n << 1] += dR3;
				pData[(n << 1) + 1] += dI3;
			}
		}
	}

	// Normalize output if we're doing the inverse forier transform
	if(!bForward)
	{
		GVecWrapper vw(pData, totalSize);
		vw *= (1.0 / (double)arraySize);
	}
}

void GFourier::fft2d(size_t arrayWidth, size_t arrayHeight, struct ComplexNumber* p2DComplexNumberArray, bool bForward)
{
	double* pData = (double*)p2DComplexNumberArray;

	double* pTmpArray = new double[std::max(arrayWidth, arrayHeight) << 1];
	std::unique_ptr<double[]> hTmpArray(pTmpArray);

	// Horizontal transforms
	for(size_t y = 0; y < arrayHeight; y++)
	{
		for(size_t x = 0; x < arrayWidth; x++)
		{
			pTmpArray[x << 1] = pData[(arrayWidth * y + x) << 1];
			pTmpArray[(x << 1) + 1] = pData[((arrayWidth * y + x) << 1) + 1];
		}
		fft(arrayWidth, (struct ComplexNumber*)pTmpArray, bForward);
		for(size_t x = 0; x < arrayWidth; x++)
		{
			pData[(arrayWidth * y + x) << 1] = pTmpArray[x << 1];
			pData[((arrayWidth * y + x) << 1) + 1] = pTmpArray[(x << 1) + 1];
		}
	}

	// Vertical transforms
	for(size_t x = 0; x < arrayWidth; x++)
	{
		for(size_t y = 0; y < arrayHeight; y++)
		{
			pTmpArray[y << 1] = pData[(arrayWidth * y + x) << 1];
			pTmpArray[(y << 1) + 1] = pData[((arrayWidth * y + x) << 1) + 1];
		}
		fft(arrayHeight, (struct ComplexNumber*)pTmpArray, bForward);
		for(size_t y = 0; y < arrayHeight; y++)
		{
			pData[(arrayWidth * y + x) << 1] = pTmpArray[y << 1];
			pData[((arrayWidth * y + x) << 1) + 1] = pTmpArray[(y << 1) + 1];
		}
	}
}

// static
struct ComplexNumber* GFourier::imageToFftArray(GImage* pImage, int* pWidth, int* pOneThirdHeight)
{
	int x, y;
	int width = pImage->width();
	int height = pImage->height();
	int wid = GBits::boundingPowerOfTwo(width);
	int hgt = GBits::boundingPowerOfTwo(height);
	struct ComplexNumber* pArray = new struct ComplexNumber[3 * wid * hgt];
	std::unique_ptr<struct ComplexNumber[]> hArray(pArray);
	int pos = 0;

	// Red channel
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			pArray[pos].real = gRed(pImage->pixel(x, y));
			pArray[pos].imag = 0;
			pos++;
		}
		for(; x < wid; x++)
		{
			pArray[pos].real = 0;
			pArray[pos].imag = 0;
			pos++;
		}
	}
	for(; y < hgt; y++)
	{
		for(x = 0; x < wid; x++)
		{
			pArray[pos].real = 0;
			pArray[pos].imag = 0;
			pos++;
		}
	}

	// Green channel
	int nGreenStart = pos;
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			pArray[pos].real = gGreen(pImage->pixel(x, y));
			pArray[pos].imag = 0;
			pos++;
		}
		for(; x < wid; x++)
		{
			pArray[pos].real = 0;
			pArray[pos].imag = 0;
			pos++;
		}
	}
	for(; y < hgt; y++)
	{
		for(x = 0; x < wid; x++)
		{
			pArray[pos].real = 0;
			pArray[pos].imag = 0;
			pos++;
		}
	}

	// Blue channel
	int nBlueStart = pos;
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			pArray[pos].real = gBlue(pImage->pixel(x, y));
			pArray[pos].imag = 0;
			pos++;
		}
		for(; x < wid; x++)
		{
			pArray[pos].real = 0;
			pArray[pos].imag = 0;
			pos++;
		}
	}
	for(; y < hgt; y++)
	{
		for(x = 0; x < wid; x++)
		{
			pArray[pos].real = 0;
			pArray[pos].imag = 0;
			pos++;
		}
	}

	// Convert to the Fourier domain
	GFourier::fft2d(wid, hgt, pArray, true);
	GFourier::fft2d(wid, hgt, &pArray[nGreenStart], true);
	GFourier::fft2d(wid, hgt, &pArray[nBlueStart], true);

	*pWidth = wid;
	*pOneThirdHeight = hgt;
	return hArray.release();
}

// static
void GFourier::fftArrayToImage(struct ComplexNumber* pArray, int width, int oneThirdHeight, GImage* pImage, bool normalize)
{
	// Convert to the Spatial domain
	int nGreenStart = width * oneThirdHeight;
	int nBlueStart = nGreenStart + nGreenStart;
	GFourier::fft2d(width, oneThirdHeight, pArray, false);
	GFourier::fft2d(width, oneThirdHeight, &pArray[nGreenStart], false);
	GFourier::fft2d(width, oneThirdHeight, &pArray[nBlueStart], false);

	// Copy the data back into the image
	int nWid = pImage->width();
	int nHgt = pImage->height();
	int x, y;
	double min = 0;
	double max = (double)256;
	if(normalize)
	{
		min = pArray[0].real;
		max = min;
		double d;
		for(y = 0; y < nHgt; y++)
		{
			for(x = 0; x < nWid; x++)
			{
				d = pArray[width * y + x].real;
				if(d < min)
					min = d;
				else if(d > max)
					max = d;
				d = pArray[nGreenStart + width * y + x].real;
				if(d < min)
					min = d;
				else if(d > max)
					max = d;
				d = pArray[nBlueStart + width * y + x].real;
				if(d < min)
					min = d;
				else if(d > max)
					max = d;
			}
		}
	}
	double scale = (double)256 / (max - min);
	for(y = 0; y < nHgt; y++)
	{
		for(x = 0; x < nWid; x++)
		{
			pImage->setPixel(x, y, gARGB(0xff,
						ClipChan((int)((pArray[width * y + x].real - min) * scale)),
						ClipChan((int)((pArray[nGreenStart + width * y + x].real - min) * scale)),
						ClipChan((int)((pArray[nBlueStart + width * y + x].real - min) * scale))
					));
		}
	}
}


void GFourier::test()
{
	struct ComplexNumber cn[4];
	cn[0].real = 1;
	cn[0].imag = 0;
	cn[1].real = 1;
	cn[1].imag = 0;
	cn[2].real = 1;
	cn[2].imag = 0;
	cn[3].real = 1;
	cn[3].imag = 0;
	GFourier::fft(4, cn, true);
	if(std::abs(cn[0].real - 4) > 1e-12)
		throw Ex("wrong answer");
	if(std::abs(cn[0].imag) > 1e-12)
		throw Ex("wrong answer");
	int n;
	for(n = 1; n < 3; n++)
	{
		if(std::abs(cn[n].real) > 1e-12)
			throw Ex("wrong answer");
		if(std::abs(cn[n].imag) > 1e-12)
			throw Ex("wrong answer");
	}
	GFourier::fft(4, cn, false);
	for(n = 0; n < 3; n++)
	{
		if(std::abs(cn[n].real - 1) > 1e-12)
			throw Ex("wrong answer");
		if(std::abs(cn[n].imag) > 1e-12)
			throw Ex("wrong answer");
	}
}





GWavelet::GWavelet()
: m_pBuf(NULL), m_bufSize(0)
{
}

GWavelet::~GWavelet()
{
	delete[] m_pBuf;
}

double* GWavelet::getBuf(size_t n)
{
	if(m_bufSize < n)
	{
		delete[] m_pBuf;
		m_pBuf = new double[n];
	}
	return m_pBuf;
}

void GWavelet::transform(double* x, size_t n)
{
  if(n < 2)
    throw Ex("out of range");
  double a;
  size_t i;
  // Predict 1
  a=-1.586134342;
  for (i=1;i+2<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[n-1]+=2*a*x[n-2];
  // Update 1
  a=-0.05298011854;
  for (i=2;i<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[0]+=2*a*x[1];
  // Predict 2
  a=0.8829110762;
  for (i=1;i+2<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[n-1]+=2*a*x[n-2];
  // Update 2
  a=0.4435068522;
  for (i=2;i<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[0]+=2*a*x[1];
  // Scale
  a=1/1.149604398;
  for (i=0;i<n;i++)
  {
    if (i%2) x[i]*=a;
    else x[i]/=a;
  }
  // Pack
  double* tempbank = getBuf(n);
  for (i=0;i<n;i++)
  {
    if (i%2==0) tempbank[i/2]=x[i];
    else tempbank[n/2+i/2]=x[i];
  }
  for (i=0;i<n;i++) x[i]=tempbank[i];
}

void GWavelet::inverse(double* x, size_t n)
{
  if(n < 2)
    throw Ex("out of range");
  double a;
  size_t i;
  // Unpack
  double* tempbank = getBuf(n);
  for (i=0;i<n/2;i++)
  {
    tempbank[i*2]=x[i];
    tempbank[i*2+1]=x[i+n/2];
  }
  for (i=0;i<n;i++) x[i]=tempbank[i];
  // Undo scale
  a=1.149604398;
  for (i=0;i<n;i++)
  {
    if (i%2) x[i]*=a;
    else x[i]/=a;
  }
  // Undo update 2
  a=-0.4435068522;
  for (i=2;i<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[0]+=2*a*x[1];
  // Undo predict 2
  a=-0.8829110762;
  for (i=1;i+2<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[n-1]+=2*a*x[n-2];
  // Undo update 1
  a=0.05298011854;
  for (i=2;i<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[0]+=2*a*x[1];
  // Undo predict 1
  a=1.586134342;
  for (i=1;i+2<n;i+=2)
  {
    x[i]+=a*(x[i-1]+x[i+1]);
  }
  x[n-1]+=2*a*x[n-2];
}

#include "GMatrix.h"
// static
void GWavelet::test()
{
	GImage im;
	im.loadPgm("/home/mike/tmp/mike.pgm");
	GMatrix m(im.height(), im.width());
	GWavelet w;
	for(size_t y = 0; y < im.height(); y++)
	{
		GVec& pRow = m[y];
		for(size_t x = 0; x < im.width(); x++)
			pRow[x] = gGreen(im.pixel((int)x, (int)y));
		w.transform(pRow.data(), im.width());
	}
	GMatrix* pT = m.transpose();
	std::unique_ptr<GMatrix> hT(pT);
	for(size_t x = 0; x < im.width(); x++)
	{
		GVec& pRow = pT->row(x);
		w.transform(pRow.data(), im.height());
		for(size_t y = 0; y < im.height(); y++)
		{
			int g = ClipChan((int)pRow[y]);
			im.setPixel((int)x, (int)y, gARGB(0xff, g, g, g));
		}
	}
	im.savePgm("/home/mike/tmp/mike2.pgm");

	// Now do the inverse transform
	GMatrix* pM = pT->transpose();
	std::unique_ptr<GMatrix> hM(pM);
	for(size_t y = 0; y < im.height(); y++)
	{
		GVec& pRow = pM->row(y);
		w.inverse(pRow.data(), im.width());
	}
	GMatrix* pTT = pM->transpose();
	std::unique_ptr<GMatrix> hTT(pTT);
	for(size_t x = 0; x < im.width(); x++)
	{
		GVec& pRow = pTT->row(x);
		w.inverse(pRow.data(), im.height());
		for(size_t y = 0; y < im.height(); y++)
		{
			int g = ClipChan((int)pRow[y]);
			im.setPixel((int)x, (int)y, gARGB(0xff, g, g, g));
		}
	}
	im.savePgm("/home/mike/tmp/mike3.pgm");
}

