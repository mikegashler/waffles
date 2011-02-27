/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GVideo.h"
#include "../GClasses/GImage.h"
#include "../GClasses/GHolders.h"

namespace GClasses {

using std::vector;

GVideo::GVideo(int nWidth, int nHeight)
{
	m_nWidth = nWidth;
	m_nHeight = nHeight;
}

GVideo::~GVideo()
{
	for(vector<GImage*>::iterator it = m_frames.begin(); it != m_frames.end(); it++)
		delete(*it);
}

int GVideo::frameCount()
{
	return (int)m_frames.size();
}

GImage* GVideo::frame(int index)
{
	return m_frames[index];
}

bool GVideo::loadFrame(const char* szFilename)
{
	GImage* pNewFrame = new GImage();
	Holder<GImage> hNewFrame(pNewFrame);
	pNewFrame->loadPng(szFilename);
	GAssert((int)pNewFrame->width() == m_nWidth && (int)pNewFrame->height() == m_nHeight);
	m_frames.push_back(hNewFrame.release());
	return true;
}

void GVideo::addBlankFrame()
{
	GImage* pNewFrame = new GImage();
	pNewFrame->setSize(m_nWidth, m_nHeight);
	m_frames.push_back(pNewFrame);
}

void GVideo::setSize(int width, int height)
{
	for(vector<GImage*>::iterator it = m_frames.begin(); it != m_frames.end(); it++)
		delete(*it);
	m_frames.clear();
	m_nWidth = width;
	m_nHeight = height;
}

void GVideo::makeGradientMagnitudeVideo(GVideo* pVideo, bool bForDisplay)
{
	setSize(pVideo->width(), pVideo->height());
	const int sobel[] = { 1, 3, 1, 3, 6, 3, 1, 3, 1 };
	GImage* pFrames[3];
	GImage* pCurrentFrame;
	int x, y, z, i, j, m, c1, c2;
	int sums[9];
	int nFrameCount = pVideo->frameCount();
	for(z = 0; z < pVideo->frameCount(); z++)
	{
		if(frameCount() <= z)
			addBlankFrame();
		pCurrentFrame = frame(z);
		pFrames[0] = pVideo->frame((z + nFrameCount - 1) % nFrameCount);
		pFrames[1] = pVideo->frame(z);
		pFrames[2] = pVideo->frame((z + 1) % nFrameCount);
		for(y = 0; y < m_nHeight; y++)
		{
			for(x = 0; x < m_nWidth; x++)
			{
				memset(sums, '\0', sizeof(int) * 9);
				for(j = -1; j < 2; j++)
				{
					for(i = -1; i < 2; i++)
					{
						m = sobel[3 * j + i + 4];
						c1 = pFrames[j + 1]->pixelNearest(x + 1, y + i);
						c2 = pFrames[j + 1]->pixelNearest(x - 1, y + i);
						sums[0] += m * (gRed(c1) - gRed(c2));
						sums[1] += m * (gGreen(c1) - gGreen(c2));
						sums[2] += m * (gBlue(c1) - gBlue(c2));
						c1 = pFrames[j + 1]->pixelNearest(x + i, y + 1);
						c2 = pFrames[j + 1]->pixelNearest(x + i, y - 1);
						sums[3] += m * (gRed(c1) - gRed(c2));
						sums[4] += m * (gGreen(c1) - gGreen(c2));
						sums[5] += m * (gBlue(c1) - gBlue(c2));
						c1 = pFrames[2]->pixelNearest(x + i, y + j);
						c2 = pFrames[0]->pixelNearest(x + i, y + j);
						sums[6] += m * (gRed(c1) - gRed(c2));
						sums[7] += m * (gGreen(c1) - gGreen(c2));
						sums[8] += m * (gBlue(c1) - gBlue(c2));
					}
				}
				m = std::max(
						sums[0] * sums[0] + sums[3] * sums[3] + sums[6] * sums[6],
						std::max(
							sums[1] * sums[1] + sums[4] * sums[4] + sums[7] * sums[7],
							sums[2] * sums[2] + sums[5] * sums[5] + sums[8] * sums[8])
						);
				if(bForDisplay)
				{
					m = ClipChan(m / 50000);
					pCurrentFrame->setPixel(x, y, gARGB(0xff, m, m, m));
				}
				else
					pCurrentFrame->setPixel(x, y, m);
			}
		}
	}
}

} // namespace GClasses

