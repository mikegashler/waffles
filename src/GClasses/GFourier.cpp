/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GFourier.h"
#include <math.h>
#include "GError.h"
#include "GHolders.h"
#include "GMath.h"
#include "GImage.h"
#include "GBits.h"
#include <cmath>

using namespace GClasses;

inline int ReverseBits(int nValue, int nBits)
{
    int n;
	int nReversed = 0;
    for(n = 0; n < nBits; n++)
    {
        nReversed = (nReversed << 1) | (nValue & 1);
        nValue >>= 1;
    }
    return nReversed;
}

bool GFourier::fft(int nArraySize, struct ComplexNumber* pComplexNumberArray, bool bForward)
{
	double* pData = (double*)pComplexNumberArray;

	// Make sure nArraySize is a power of 2
	if(nArraySize & (nArraySize - 1))
	{
		GAssert(false); // Error, nArraySize must be a power of 2
		return false;
	}

	// Calculate the Log2 of nArraySize and put it in nBits
	int n = 1;
	int nBits = 0;
	while(n < nArraySize)
	{
		n = n << 1;
		nBits++;
	}

	// Move the data to it's reversed-bit position
	int nTotalSize = nArraySize << 1;
	double* pTmp = new double[nArraySize << 1];
	int nReversed;
	for(n = 0; n < nArraySize; n++)
	{
		nReversed = ReverseBits(n, nBits);
		pTmp[nReversed << 1] = pData[n << 1];
		pTmp[(nReversed << 1) + 1] = pData[(n << 1) + 1];
	}
	for(n = 0; n < nTotalSize; n++)
		pData[n] = pTmp[n];
	delete[] pTmp;

	// Calculate the angle numerator
	double dAngleNumerator;
	if(bForward)
		dAngleNumerator = -2.0 * M_PI;
	else
		dAngleNumerator = 2.0 * M_PI;

	// Do the Fast Forier Transform
	double dR0, dR1, dR2, dR3, dI0, dI1, dI2, dI3;
	int n2;
	int nStart;
	int nHalfBlockSize;
	for(nHalfBlockSize = 1; nHalfBlockSize < nArraySize; nHalfBlockSize = nHalfBlockSize << 1)
	{
		// Calculate angles, sines, and cosines
		double dAngleDelta = dAngleNumerator / ((double)(nHalfBlockSize << 1));
		double dCos1 = cos(-dAngleDelta);
		double d2Cos1 = 2 * dCos1; // So we don't have to calculate this a bunch of times
		double dCos2 = cos(-2 * dAngleDelta);
		double dSin1 = sin(-dAngleDelta);
		double dSin2 = sin(-2 * dAngleDelta);

		// Do each block
		for(nStart = 0; nStart < nArraySize; nStart += (nHalfBlockSize << 1))
		{
			dR1 = dCos1;
			dR2 = dCos2;
			dI1 = dSin1;
			dI2 = dSin2;
			int nEnd = nStart + nHalfBlockSize;
			for(n = nStart; n < nEnd; n++)
			{
				dR0 = d2Cos1 * dR1 - dR2;
				dR2 = dR1;
				dR1 = dR0;
				dI0 = d2Cos1 * dI1 - dI2;
				dI2 = dI1;
				dI1 = dI0;
				n2 = n + nHalfBlockSize;
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
		for(n = 0; n < nTotalSize; n++)
			pData[n] /= (double)nArraySize;
	}

	return true;
}

bool GFourier::fft2d(int nArrayWidth, int nArrayHeight, struct ComplexNumber* p2DComplexNumberArray, bool bForward)
{
	double* pData = (double*)p2DComplexNumberArray;

	double* pTmpArray = new double[std::max(nArrayWidth, nArrayHeight) << 1];
	ArrayHolder<double> hTmpArray(pTmpArray);
	int x, y;

	// Horizontal transforms
	for(y = 0; y < nArrayHeight; y++)
	{
		for(x = 0; x < nArrayWidth; x++)
		{
			pTmpArray[x << 1] = pData[(nArrayWidth * y + x) << 1];
			pTmpArray[(x << 1) + 1] = pData[((nArrayWidth * y + x) << 1) + 1];
		}
		if(!fft(nArrayWidth, (struct ComplexNumber*)pTmpArray, bForward))
			return false;
		for(x = 0; x < nArrayWidth; x++)
		{
			pData[(nArrayWidth * y + x) << 1] = pTmpArray[x << 1];
			pData[((nArrayWidth * y + x) << 1) + 1] = pTmpArray[(x << 1) + 1];
		}
	}

	// Vertical transforms
	for(x = 0; x < nArrayWidth; x++)
	{
		for(y = 0; y < nArrayHeight; y++)
		{
			pTmpArray[y << 1] = pData[(nArrayWidth * y + x) << 1];
			pTmpArray[(y << 1) + 1] = pData[((nArrayWidth * y + x) << 1) + 1];
		}
		if(!fft(nArrayHeight, (struct ComplexNumber*)pTmpArray, bForward))
			return false;
		for(y = 0; y < nArrayHeight; y++)
		{
			pData[(nArrayWidth * y + x) << 1] = pTmpArray[y << 1];
			pData[((nArrayWidth * y + x) << 1) + 1] = pTmpArray[(y << 1) + 1];
		}
	}
	return true;
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
	if(!GFourier::fft2d(wid, hgt, pArray, true))
	{
		delete[] pArray;
		return NULL;
	}
	if(!GFourier::fft2d(wid, hgt, &pArray[nGreenStart], true))
	{
		delete[] pArray;
		return NULL;
	}
	if(!GFourier::fft2d(wid, hgt, &pArray[nBlueStart], true))
	{
		delete[] pArray;
		return NULL;
	}

	*pWidth = wid;
	*pOneThirdHeight = hgt;
	return pArray;
}

// static
void GFourier::fftArrayToImage(struct ComplexNumber* pArray, int width, int oneThirdHeight, GImage* pImage, bool normalize)
{
	// Convert to the Spatial domain
	int nGreenStart = width * oneThirdHeight;
	int nBlueStart = nGreenStart + nGreenStart;
	if(!GFourier::fft2d(width, oneThirdHeight, pArray, false))
	{
		GAssert(false);
		return;
	}
	if(!GFourier::fft2d(width, oneThirdHeight, &pArray[nGreenStart], false))
	{
		GAssert(false);
		return;
	}
	if(!GFourier::fft2d(width, oneThirdHeight, &pArray[nBlueStart], false))
	{
		GAssert(false);
		return;
	}

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


#ifndef NO_TEST_CODE
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
		throw "wrong answer";
	if(std::abs(cn[0].imag) > 1e-12)
		throw "wrong answer";
	int n;
	for(n = 1; n < 3; n++)
	{
		if(std::abs(cn[n].real) > 1e-12)
			throw "wrong answer";
		if(std::abs(cn[n].imag) > 1e-12)
			throw "wrong answer";
	}
	GFourier::fft(4, cn, false);
	for(n = 0; n < 3; n++)
	{
		if(std::abs(cn[n].real - 1) > 1e-12)
			throw "wrong answer";
		if(std::abs(cn[n].imag) > 1e-12)
			throw "wrong answer";
	}
}
/*
void GFourier::test()
{
	struct ComplexNumber cn[4];
	cn[0].real = 1;
	cn[0].imag = 0;
	cn[1].real = 0;
	cn[1].imag = 0;
	cn[2].real = 0;
	cn[2].imag = 0;
	cn[3].real = 0;
	cn[3].imag = 0;
	GFourier::FFT(4, cn, true);
	int n;
	for(n = 0; n < 4; n++)
	{
		if(cn[n].real != 1)
			throw "wrong answer";
		if(cn[n].imag != 0)
			throw "wrong answer";
	}
	GFourier::FFT(4, cn, false);
	if(cn[0].real != 1)
		throw "wrong answer";
	if(cn[0].imag != 0)
		throw "wrong answer";
	for(n = 1; n < 4; n++)
	{
		if(cn[n].real != 0)
			throw "wrong answer";
		if(cn[n].imag != 0)
			throw "wrong answer";
	}
}
*/
#endif // NO_TEST_CODE
