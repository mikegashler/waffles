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

#include "GRegion.h"
#include "GImage.h"
#include "GHeap.h"
//#include "GVideo.h"
#include "GFourier.h"
#include "GBits.h"
#include "GRect.h"
#include "GVec.h"
#include "GHolders.h"
#include <math.h>
#include <vector>
#include <algorithm>
#include <memory>

using std::vector;

namespace GClasses {


struct GRegionEdge
{
	size_t m_nRegion1;
	size_t m_nRegion2;
	GRegionEdge* m_pNext1;
	GRegionEdge* m_pNext2;

	size_t GetOther(size_t n)
	{
		if(n == m_nRegion1)
			return m_nRegion2;
		else if(n == m_nRegion2)
			return m_nRegion1;
		else
		{
			GAssert(false); // That region doesn't share this edge
			return INVALID_INDEX;
		}
	}

	struct GRegionEdge* GetNext(size_t nRegion)
	{
		if(nRegion == m_nRegion1)
			return m_pNext1;
		else if(nRegion == m_nRegion2)
			return m_pNext2;
		else
		{
			GAssert(false); // That region doesn't share this edge
			return NULL;
		}
	}
};





GRegionAjacencyGraph::GRegionAjacencyGraph()
{
	m_pHeap = new GHeap(2048);
}

GRegionAjacencyGraph::~GRegionAjacencyGraph()
{
	delete(m_pHeap);
}

size_t GRegionAjacencyGraph::addRegion()
{
	size_t nRegion = m_regions.size();
	struct GRegion* pNewRegion = (struct GRegion*)m_pHeap->allocate(sizeof(struct GRegion));
	pNewRegion->m_nPixels = 0;
	pNewRegion->m_nSumRed = 0;
	pNewRegion->m_nSumGreen = 0;
	pNewRegion->m_nSumBlue = 0;
	pNewRegion->m_pNeighbors = NULL;
	m_regions.push_back(pNewRegion);
	return nRegion;
}

size_t GRegionAjacencyGraph::regionCount()
{
	return m_regions.size();
}

void GRegionAjacencyGraph::averageColor(size_t nRegion, float* pRed, float* pGreen, float* pBlue)
{
	struct GRegion* pRegion = m_regions[nRegion];
	*pRed = (float)pRegion->m_nSumRed / pRegion->m_nPixels;
	*pGreen = (float)pRegion->m_nSumGreen / pRegion->m_nPixels;
	*pBlue = (float)pRegion->m_nSumBlue / pRegion->m_nPixels;
}

bool GRegionAjacencyGraph::areNeighbors(size_t nRegion1, size_t nRegion2)
{
	GAssert(nRegion1 != nRegion2); // same region
	struct GRegion* pRegion1 = m_regions[nRegion1];
	struct GRegionEdge* pEdge;
	for(pEdge = pRegion1->m_pNeighbors; pEdge; pEdge = pEdge->GetNext(nRegion1))
	{
		if(pEdge->GetOther(nRegion1) == nRegion2)
			return true;
	}
	return false;
}

void GRegionAjacencyGraph::makeNeighbors(size_t nRegion1, size_t nRegion2)
{
	if(areNeighbors(nRegion1, nRegion2))
		return;
	struct GRegion* pRegion1 = m_regions[nRegion1];
	struct GRegion* pRegion2 = m_regions[nRegion2];
	struct GRegionEdge* pNewEdge = (struct GRegionEdge*)m_pHeap->allocate(sizeof(struct GRegionEdge));
	pNewEdge->m_nRegion1 = nRegion1;
	pNewEdge->m_nRegion2 = nRegion2;
	pNewEdge->m_pNext1 = pRegion1->m_pNeighbors;
	pNewEdge->m_pNext2 = pRegion2->m_pNeighbors;
	pRegion1->m_pNeighbors = pNewEdge;
	pRegion2->m_pNeighbors = pNewEdge;
	m_neighbors.push_back(pNewEdge);
}

size_t GRegionAjacencyGraph::ajacencyCount()
{
	return m_neighbors.size();
}

void GRegionAjacencyGraph::ajacency(size_t nEdge, size_t* pRegion1, size_t* pRegion2)
{
	struct GRegionEdge* pEdge = m_neighbors[nEdge];
	*pRegion1 = pEdge->m_nRegion1;
	*pRegion2 = pEdge->m_nRegion2;
}

// ------------------------------------------------------------------------------------------

G2DRegionGraph::G2DRegionGraph(int nWidth, int nHeight)
 : GRegionAjacencyGraph()
{
	m_pRegionMask = new GImage();
	m_pRegionMask->setSize(nWidth, nHeight);
	m_pRegionMask->clear(0xffffffff);
}

G2DRegionGraph::~G2DRegionGraph()
{
	delete(m_pRegionMask);
}

void G2DRegionGraph::setMaskPixel(int x, int y, unsigned int c, size_t nRegion)
{
	GAssert(x >= 0 && x < (int)m_pRegionMask->width() && y >= 0 && y < (int)m_pRegionMask->height()); // out of range
	GAssert(m_pRegionMask->pixel(x, y) == 0xffffffff); // This pixel is already set
	m_pRegionMask->setPixel(x, y, (unsigned int)nRegion);
	struct GRegion* pRegion = m_regions[nRegion];
	pRegion->m_nSumRed += gRed(c);
	pRegion->m_nSumGreen += gGreen(c);
	pRegion->m_nSumBlue += gBlue(c);
	pRegion->m_nPixels++;
}

int MaxChan(unsigned int c)
{
	//return MAX(gRed(c), MAX(gGreen(c), gBlue(c)));
	return c;
}

void PickTobogganDirection(GImage* pGradMagImage, int u, int v, int* pdu, int* pdv)
{
	int cand;
	int grad = MaxChan(pGradMagImage->pixel(u, v));
	int du = 0;
	int dv = 0;
	if(u > 0)
	{
		cand = MaxChan(pGradMagImage->pixel(u - 1, v));
		if(cand < grad)
		{
			grad = cand;
			du = -1;
			dv = 0;
		}
	}
	if(v > 0)
	{
		cand = MaxChan(pGradMagImage->pixel(u, v - 1));
		if(cand < grad)
		{
			grad = cand;
			du = 0;
			dv = -1;
		}
	}
	if(u < (int)pGradMagImage->width() - 1)
	{
		cand = MaxChan(pGradMagImage->pixel(u + 1, v));
		if(cand < grad)
		{
			grad = cand;
			du = 1;
			dv = 0;
		}
	}
	if(v < (int)pGradMagImage->height() - 1)
	{
		cand = MaxChan(pGradMagImage->pixel(u, v + 1));
		if(cand < grad)
		{
			grad = cand;
			du = 0;
			dv = 1;
		}
	}
	*pdu = du;
	*pdv = dv;
}

void G2DRegionGraph::makeWatershedRegions(const GImage* pImage)
{
	GImage gradMag;
	gradMag.gradientMagnitudeImage(pImage);
	GImage* pMask = regionMask();
	int x, y, u, v, du, dv;
	size_t region, other;
	for(y = 0; y < (int)pImage->height(); y++)
	{
		for(x = 0; x < (int)pImage->width(); x++)
		{
			u = x;
			v = y;
			do
			{
				region = pMask->pixel(u, v);
				if(region != 0xffffffff)
					break;
				PickTobogganDirection(&gradMag, u, v, &du, &dv);
				u += du;
				v += dv;
			} while(du != 0 || dv != 0);
			if(region == 0xffffffff)
			{
				region = addRegion();
				setMaskPixel(u, v, pImage->pixel(u, v), region);
			}
			u = x;
			v = y;
			do
			{
				if(pMask->pixel(u, v) != 0xffffffff)
					break;
				setMaskPixel(u, v, pImage->pixel(u, v), region);
				PickTobogganDirection(&gradMag, u, v, &du, &dv);
				u += du;
				v += dv;
			} while(du != 0 || dv != 0);
			if(x > 0)
			{
				other = pMask->pixel(x - 1, y);
				if(other != region)
					makeNeighbors(region, other);
			}
			if(y > 0)
			{
				other = pMask->pixel(x, y - 1);
				if(other != region)
					makeNeighbors(region, other);
			}
		}
	}
}

double MeasureRegionDifference(struct GRegion* pA, struct GRegion* pB)
{
	double dSum = 0;
	double d;
	d = (double)pA->m_nSumRed / pA->m_nPixels - (double)pB->m_nSumRed / pB->m_nPixels;
	dSum += (d * d);
	d = (double)pA->m_nSumGreen / pA->m_nPixels - (double)pB->m_nSumGreen / pB->m_nPixels;
	dSum += (d * d);
	d = (double)pA->m_nSumBlue / pA->m_nPixels - (double)pB->m_nSumBlue / pB->m_nPixels;
	dSum += (d * d);
	return dSum;
}

void G2DRegionGraph::makeCoarserRegions(G2DRegionGraph* pFineRegions)
{
	// Find every region's closest neighbor
	GImage* pFineRegionMask = pFineRegions->regionMask();
	GImage* pCoarseRegionMask = regionMask();
	GAssert(pCoarseRegionMask->width() == pFineRegionMask->width() && pCoarseRegionMask->height() == pFineRegionMask->height()); // size mismatch
	int* pBestNeighborMap = new int[pFineRegions->regionCount()];
	std::unique_ptr<int[]> hBestNeighborMap(pBestNeighborMap);
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		struct GRegion* pRegion = pFineRegions->m_regions[i];
		struct GRegionEdge* pEdge;
		double d;
		double dBestDiff = 1e200;
		int nBestNeighbor = -1;
		for(pEdge = pRegion->m_pNeighbors; pEdge; pEdge = pEdge->GetNext(i))
		{
			size_t j = pEdge->GetOther(i);
			struct GRegion* pOtherRegion = pFineRegions->m_regions[j];
			d = MeasureRegionDifference(pRegion, pOtherRegion);
			if(d < dBestDiff)
			{
				dBestDiff = d;
				nBestNeighbor = (int)j;
			}
		}
		GAssert(nBestNeighbor != -1 || pFineRegions->regionCount() == 1); // failed to find a neighbor
		pBestNeighborMap[i] = nBestNeighbor;
	}

	// Create a mapping to new regions numbers
	int* pNewRegionMap = new int[pFineRegions->regionCount()];
	std::unique_ptr<int[]> hNewRegionMap(pNewRegionMap);
	memset(pNewRegionMap, 0xff, sizeof(int) * pFineRegions->regionCount());
	int nNewRegionCount = 0;
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		size_t nNewRegion = -1;
		size_t j = i;
		while(pNewRegionMap[j] == -1)
		{
			pNewRegionMap[j] = -2;
			j = pBestNeighborMap[j];
		}
		if(pNewRegionMap[j] == -2)
			nNewRegion = nNewRegionCount++;
		else
			nNewRegion = pNewRegionMap[j];
		j = i;
		while(pNewRegionMap[j] == -2)
		{
			pNewRegionMap[j] = (int)nNewRegion;
			j = pBestNeighborMap[j];
		}
	}

	// Make the new regions
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		struct GRegion* pRegion = pFineRegions->m_regions[i];
		size_t j = pNewRegionMap[i];
		if(regionCount() <= j)
		{
			GAssert(regionCount() == j); // how'd it get two behind?
			addRegion();
		}
		struct GRegion* pCoarseRegion = m_regions[j];
		pCoarseRegion->m_nSumRed += pRegion->m_nSumRed;
		pCoarseRegion->m_nSumGreen += pRegion->m_nSumGreen;
		pCoarseRegion->m_nSumBlue += pRegion->m_nSumBlue;
		pCoarseRegion->m_nPixels += pRegion->m_nPixels;
	}
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		struct GRegion* pRegion = pFineRegions->m_regions[i];
		size_t j = pNewRegionMap[i];
		struct GRegionEdge* pEdge;
		for(pEdge = pRegion->m_pNeighbors; pEdge; pEdge = pEdge->GetNext(i))
		{
			size_t k = pNewRegionMap[pEdge->GetOther(i)];
			if(j != k)
				makeNeighbors(j, k);
		}
	}

	// Make the fine region mask
	unsigned int nOldRegion;
	int x, y;
	for(y = 0; y < (int)pFineRegionMask->height(); y++)
	{
		for(x = 0; x < (int)pFineRegionMask->width(); x++)
		{
			nOldRegion = pFineRegionMask->pixel(x, y);
			pCoarseRegionMask->setPixel(x, y, pNewRegionMap[nOldRegion]);
		}
	}
}

// ------------------------------------------------------------------------------------------

G3DRegionGraph::G3DRegionGraph(size_t frames, int nWidth, int nHeight)
 : GRegionAjacencyGraph()
{
	for(size_t i = 0; i < frames; i++)
	{
		GImage* pImage = new GImage();
		m_regionMask.push_back(pImage);
		pImage->setSize(nWidth, nHeight);
		pImage->clear(0xffffffff);
	}
}

G3DRegionGraph::~G3DRegionGraph()
{
	for(size_t i = 0; i < m_regionMask.size(); i++)
		delete(m_regionMask[i]);
}

void G3DRegionGraph::setMaskPixel(size_t frame, int x, int y, unsigned int c, size_t nRegion)
{
	GImage* pImage = m_regionMask[frame];
	GAssert(x >= 0 && x < (int)pImage->width() && y >= 0 && y < (int)pImage->height()); // out of range
	GAssert(pImage->pixel(x, y) == 0xffffffff); // This pixel is already set
	pImage->setPixel(x, y, (unsigned int)nRegion);
	struct GRegion* pRegion = m_regions[nRegion];
	pRegion->m_nSumRed += gRed(c);
	pRegion->m_nSumGreen += gGreen(c);
	pRegion->m_nSumBlue += gBlue(c);
	pRegion->m_nPixels++;
}

void G3DRegionGraph_makeGradientMagnitudeVideo(const std::vector<GImage*>& in, std::vector<GImage*>& out)
{
	while(out.size() < in.size())
	{
		GImage* pImage = new GImage();
		out.push_back(pImage);
		pImage->setSize(in[0]->width(), in[0]->height());
	}
	for(int z = 0; z < (int)in.size(); z++)
	{
		GImage* pImage = in[z];
		GImage* pOutImage = out[z];
		for(int y = 0; y < (int)pImage->height(); y++)
		{
			for(int x = 0; x < (int)pImage->width(); x++)
			{
				int rx = 0;	int ry = 0;	int rz = 0;
				int gx = 0; int gy = 0;	int gz = 0;
				int bx = 0; int by = 0;	int bz = 0;
				int r, g, b;
				for(int dz = -1; dz < 2; dz++)
				{
					int zz = std::max(0, (int)std::min((int)in.size() - 1, z + dz));
					GImage* pImg = in[zz];
					for(int dy = -1; dy < 2; dy++)
					{
						int yy = std::max(0, std::min((int)pImg->height() - 1, y + dy));
						for(int dx = -1; dx < 2; dx++)
						{
							int xx = std::max(0, std::min((int)pImg->width() - 1, x + dx));
							unsigned int pix = pImg->pixel(xx, yy);
							r = gRed(pix);
							g = gGreen(pix);
							b = gBlue(pix);
							int x_scalar = dx * (2 - dy * dy) * (2 - dz * dz);
							int y_scalar = (2 - dx * dx) * dy * (2 - dz * dz);
							int z_scalar = (2 - dx * dx) * (2 - dy * dy) * dz;
							rx += r * x_scalar;
							gx += g * x_scalar;
							bx += b * x_scalar;
							ry += r * y_scalar;
							gy += g * y_scalar;
							by += b * y_scalar;
							rz += r * z_scalar;
							gz += g * z_scalar;
							bz += b * z_scalar;
						}
					}
				}
				unsigned int grad = rx * rx + ry * ry + rz * rz;
				grad = std::max((int)grad, gx * gx + gy * gy + gz * gz);
				grad = std::max((int)grad, bx * bx + by * by + bz * bz);
				pOutImage->setPixel(x, y, grad);
			}
		}
	}
}

void G3DRegionGraph_PickTobogganDirection(GImage* pGradMagImage, int u, int v, int* pdu, int* pdv)
{
	unsigned int cand;
	unsigned int grad = pGradMagImage->pixel(u, v);
	int du = 0;
	int dv = 0;
	if(u > 0)
	{
		cand = pGradMagImage->pixel(u - 1, v);
		if(cand < grad)
		{
			grad = cand;
			du = -1;
			dv = 0;
		}
	}
	if(v > 0)
	{
		cand = pGradMagImage->pixel(u, v - 1);
		if(cand < grad)
		{
			grad = cand;
			du = 0;
			dv = -1;
		}
	}
	if(u < (int)pGradMagImage->width() - 1)
	{
		cand = pGradMagImage->pixel(u + 1, v);
		if(cand < grad)
		{
			grad = cand;
			du = 1;
			dv = 0;
		}
	}
	if(v < (int)pGradMagImage->height() - 1)
	{
		cand = pGradMagImage->pixel(u, v + 1);
		if(cand < grad)
		{
			grad = cand;
			du = 0;
			dv = 1;
		}
	}
	*pdu = du;
	*pdv = dv;
}
/*
void G3DRegionGraph::makeWatershedRegions(const std::vector<GImage*>& frames)
{
	std::vector<GImage*> gradMag;
	G3DRegionGraph_makeGradientMagnitudeVideo(frames, gradMag);
	GImage* pMask = regionMask();
	int x, y, u, v, du, dv;
	size_t region, other;
	for(y = 0; y < (int)pImage->height(); y++)
	{
		for(x = 0; x < (int)pImage->width(); x++)
		{
			u = x;
			v = y;
			do
			{
				region = pMask->pixel(u, v);
				if(region != 0xffffffff)
					break;
				G3DRegionGraph_PickTobogganDirection(&gradMag, u, v, &du, &dv);
				u += du;
				v += dv;
			} while(du != 0 || dv != 0);
			if(region == 0xffffffff)
			{
				region = addRegion();
				setMaskPixel(u, v, pImage->pixel(u, v), region);
			}
			u = x;
			v = y;
			do
			{
				if(pMask->pixel(u, v) != 0xffffffff)
					break;
				setMaskPixel(u, v, pImage->pixel(u, v), region);
				G3DRegionGraph_PickTobogganDirection(&gradMag, u, v, &du, &dv);
				u += du;
				v += dv;
			} while(du != 0 || dv != 0);
			if(x > 0)
			{
				other = pMask->pixel(x - 1, y);
				if(other != region)
					makeNeighbors(region, other);
			}
			if(y > 0)
			{
				other = pMask->pixel(x, y - 1);
				if(other != region)
					makeNeighbors(region, other);
			}
		}
	}
}

double MeasureRegionDifference(struct GRegion* pA, struct GRegion* pB)
{
	double dSum = 0;
	double d;
	d = (double)pA->m_nSumRed / pA->m_nPixels - (double)pB->m_nSumRed / pB->m_nPixels;
	dSum += (d * d);
	d = (double)pA->m_nSumGreen / pA->m_nPixels - (double)pB->m_nSumGreen / pB->m_nPixels;
	dSum += (d * d);
	d = (double)pA->m_nSumBlue / pA->m_nPixels - (double)pB->m_nSumBlue / pB->m_nPixels;
	dSum += (d * d);
	return dSum;
}

void G3DRegionGraph::makeCoarserRegions(G2DRegionGraph* pFineRegions)
{
	// Find every region's closest neighbor
	GImage* pFineRegionMask = pFineRegions->regionMask();
	GImage* pCoarseRegionMask = regionMask();
	GAssert(pCoarseRegionMask->width() == pFineRegionMask->width() && pCoarseRegionMask->height() == pFineRegionMask->height()); // size mismatch
	int* pBestNeighborMap = new int[pFineRegions->regionCount()];
	std::unique_ptr<int[]> hBestNeighborMap(pBestNeighborMap);
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		struct GRegion* pRegion = pFineRegions->m_regions[i];
		struct GRegionEdge* pEdge;
		double d;
		double dBestDiff = 1e200;
		int nBestNeighbor = -1;
		for(pEdge = pRegion->m_pNeighbors; pEdge; pEdge = pEdge->GetNext(i))
		{
			size_t j = pEdge->GetOther(i);
			struct GRegion* pOtherRegion = pFineRegions->m_regions[j];
			d = MeasureRegionDifference(pRegion, pOtherRegion);
			if(d < dBestDiff)
			{
				dBestDiff = d;
				nBestNeighbor = (int)j;
			}
		}
		GAssert(nBestNeighbor != -1 || pFineRegions->regionCount() == 1); // failed to find a neighbor
		pBestNeighborMap[i] = nBestNeighbor;
	}

	// Create a mapping to new regions numbers
	int* pNewRegionMap = new int[pFineRegions->regionCount()];
	std::unique_ptr<int[]> hNewRegionMap(pNewRegionMap);
	memset(pNewRegionMap, 0xff, sizeof(int) * pFineRegions->regionCount());
	int nNewRegionCount = 0;
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		size_t nNewRegion = -1;
		size_t j = i;
		while(pNewRegionMap[j] == -1)
		{
			pNewRegionMap[j] = -2;
			j = pBestNeighborMap[j];
		}
		if(pNewRegionMap[j] == -2)
			nNewRegion = nNewRegionCount++;
		else
			nNewRegion = pNewRegionMap[j];
		j = i;
		while(pNewRegionMap[j] == -2)
		{
			pNewRegionMap[j] = (int)nNewRegion;
			j = pBestNeighborMap[j];
		}
	}

	// Make the new regions
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		struct GRegion* pRegion = pFineRegions->m_regions[i];
		size_t j = pNewRegionMap[i];
		if(regionCount() <= j)
		{
			GAssert(regionCount() == j); // how'd it get two behind?
			addRegion();
		}
		struct GRegion* pCoarseRegion = m_regions[j];
		pCoarseRegion->m_nSumRed += pRegion->m_nSumRed;
		pCoarseRegion->m_nSumGreen += pRegion->m_nSumGreen;
		pCoarseRegion->m_nSumBlue += pRegion->m_nSumBlue;
		pCoarseRegion->m_nPixels += pRegion->m_nPixels;
	}
	for(size_t i = 0; i < pFineRegions->regionCount(); i++)
	{
		struct GRegion* pRegion = pFineRegions->m_regions[i];
		size_t j = pNewRegionMap[i];
		struct GRegionEdge* pEdge;
		for(pEdge = pRegion->m_pNeighbors; pEdge; pEdge = pEdge->GetNext(i))
		{
			size_t k = pNewRegionMap[pEdge->GetOther(i)];
			if(j != k)
				makeNeighbors(j, k);
		}
	}

	// Make the fine region mask
	unsigned int nOldRegion;
	int x, y;
	for(y = 0; y < (int)pFineRegionMask->height(); y++)
	{
		for(x = 0; x < (int)pFineRegionMask->width(); x++)
		{
			nOldRegion = pFineRegionMask->pixel(x, y);
			pCoarseRegionMask->setPixel(x, y, pNewRegionMap[nOldRegion]);
		}
	}
}
*/
// ------------------------------------------------------------------------------------------

#define START_DIRECTION 1

GRegionBorderIterator::GRegionBorderIterator(GImage* pImage, int nSampleX, int nSampleY)
{
	m_pImage = pImage;
	m_nRegion = pImage->pixel(nSampleX, nSampleY);
	m_x = nSampleX;
	m_y = nSampleY;
	m_direction = START_DIRECTION;
	while(look())
		leap();
	m_endX = m_x;
	m_endY = m_y;
	m_bOddPass = false;
}

GRegionBorderIterator::~GRegionBorderIterator()
{
}

bool GRegionBorderIterator::look()
{
	switch(m_direction)
	{
		case 0:
			if(m_x < (int)m_pImage->width() - 1)
				return(m_pImage->pixel(m_x + 1, m_y) == m_nRegion);
			else
				return false;
		case 1:
			if(m_y > 0)
				return(m_pImage->pixel(m_x, m_y - 1) == m_nRegion);
			else
				return false;
		case 2:
			if(m_x > 0)
				return(m_pImage->pixel(m_x - 1, m_y) == m_nRegion);
			else
				return false;
		case 3:
			if(m_y < (int)m_pImage->height() - 1)
				return(m_pImage->pixel(m_x, m_y + 1) == m_nRegion);
			else
				return false;
		default:
			GAssert(false); // unexpected direction
	}
	return false;
}

void GRegionBorderIterator::leap()
{
	switch(m_direction)
	{
		case 0:
			m_x++;
			break;
		case 1:
			m_y--;
			break;
		case 2:
			m_x--;
			break;
		case 3:
			m_y++;
			break;
		default:
			GAssert(false); // unexpected direction
	}
}

bool GRegionBorderIterator::next(int* pX, int* pY, int* pDirection)
{
	*pX = m_x;
	*pY = m_y;
	*pDirection = m_direction;
	if(m_x == m_endX && m_y == m_endY && m_direction == START_DIRECTION)
		m_bOddPass = !m_bOddPass;
	if(++m_direction >= 4)
		m_direction = 0;
	if(look())
	{
		leap();
		if(--m_direction < 0)
			m_direction = 3;
		if(look())
		{
			leap();
			if(--m_direction < 0)
				m_direction = 3;
		}
	}
	return m_bOddPass;
}

// ------------------------------------------------------------------------------------------

GRegionAreaIterator::GRegionAreaIterator(GImage* pImage, int nSampleX, int nSampleY)
{
	m_pImage = pImage;
	m_nRegion = pImage->pixel(nSampleX, nSampleY);

	// Find the bounding rectangle
	m_left = pImage->width() - 1;
	m_top = pImage->height() - 1;
	m_right = 0;
	m_bottom = 0;
	GRegionBorderIterator itBorder(pImage, nSampleX, nSampleY);
	int x, y, d;
	while(itBorder.next(&x, &y, &d))
	{
		if(x < m_left)
			m_left = x;
		if(x > m_right)
			m_right = x;
		if(y < m_top)
			m_top = y;
		if(y > m_bottom)
			m_bottom = y;
	}
	m_x = m_left;
	m_y = m_top;
}

GRegionAreaIterator::~GRegionAreaIterator()
{
}

bool GRegionAreaIterator::next(int* pX, int* pY)
{
	while(m_y <= m_bottom && m_pImage->pixel(m_x, m_y) != m_nRegion)
	{
		if(++m_x > m_right)
		{
			m_x = m_left;
			m_y++;
		}
	}
	if(m_y > m_bottom)
		return false;
	*pX = m_x;
	*pY = m_y;
	if(++m_x > m_right)
	{
		m_x = m_left;
		m_y++;
	}
	return true;
}

// ------------------------------------------------------------------------------------------

GSubImageFinder::GSubImageFinder(GImage* pHaystack)
{
	m_nHaystackWidth = GBits::boundingPowerOfTwo(pHaystack->width());
	m_nHaystackHeight = GBits::boundingPowerOfTwo(pHaystack->height());
	m_nHaystackX = (m_nHaystackWidth - pHaystack->width()) / 2;
	m_nHaystackY = (m_nHaystackHeight - pHaystack->height()) / 2;
	int nSize = m_nHaystackWidth * m_nHaystackHeight;
	m_pHaystackRed = new struct ComplexNumber[9 * nSize];
	m_pHaystackGreen = &m_pHaystackRed[nSize];
	m_pHaystackBlue = &m_pHaystackRed[2 * nSize];
	m_pNeedleRed = &m_pHaystackRed[3 * nSize];
	m_pNeedleGreen = &m_pHaystackRed[4 * nSize];
	m_pNeedleBlue = &m_pHaystackRed[5 * nSize];
	m_pCorRed = &m_pHaystackRed[6 * nSize];
	m_pCorGreen = &m_pHaystackRed[7 * nSize];
	m_pCorBlue = &m_pHaystackRed[8 * nSize];
	int x, y, xx, yy;
	unsigned int c;
	int pos = 0;
	for(y = 0; y < m_nHaystackHeight; y++)
	{
		yy = y - m_nHaystackY;
		for(x = 0; x < m_nHaystackWidth; x++)
		{
			xx = x - m_nHaystackX;
			if(xx >= 0 && xx < (int)pHaystack->width() && yy >= 0 && yy < (int)pHaystack->height())
			{
				c = pHaystack->pixel(xx, yy);
				m_pHaystackRed[pos].real = gRed(c) - 128;
				m_pHaystackRed[pos].imag = 0;
				m_pHaystackGreen[pos].real = gGreen(c) - 128;
				m_pHaystackGreen[pos].imag = 0;
				m_pHaystackBlue[pos].real = gBlue(c) - 128;
				m_pHaystackBlue[pos].imag = 0;
			}
			else
			{
				m_pHaystackRed[pos].real = 0;
				m_pHaystackRed[pos].imag = 0;
				m_pHaystackGreen[pos].real = 0;
				m_pHaystackGreen[pos].imag = 0;
				m_pHaystackBlue[pos].real = 0;
				m_pHaystackBlue[pos].imag = 0;
			}
			pos++;
		}
	}
	GFourier::fft2d(m_nHaystackWidth, m_nHaystackHeight, m_pHaystackRed, true);
	GFourier::fft2d(m_nHaystackWidth, m_nHaystackHeight, m_pHaystackGreen, true);
	GFourier::fft2d(m_nHaystackWidth, m_nHaystackHeight, m_pHaystackBlue, true);
}

GSubImageFinder::~GSubImageFinder()
{
	delete[] m_pHaystackRed;
}

void GSubImageFinder::findSubImage(int* pOutX, int* pOutY, GImage* pNeedle, GRect* pNeedleRect, GRect* pHaystackRect)
{
	// Copy into the array of complex numbers
	GAssert(GBits::isPowerOfTwo(pNeedleRect->w) && GBits::isPowerOfTwo(pNeedleRect->h)); // Expected a power of 2
	int x, y;
	int pos = 0;
	unsigned int c;
	for(y = 0; y < pNeedleRect->h; y++)
	{
		for(x = 0; x < pNeedleRect->w; x++)
		{
			c = pNeedle->pixel(pNeedleRect->x + x, pNeedleRect->y + y);
			m_pNeedleRed[pos].real = gRed(c) - 128;
			m_pNeedleRed[pos].imag = 0;
			m_pNeedleGreen[pos].real = gGreen(c) - 128;
			m_pNeedleGreen[pos].imag = 0;
			m_pNeedleBlue[pos].real = gBlue(c) - 128;
			m_pNeedleBlue[pos].imag = 0;
			pos++;
		}
	}

	// Convert to the Fourier domain
	GFourier::fft2d(pNeedleRect->w, pNeedleRect->h, m_pNeedleRed, true);
	GFourier::fft2d(pNeedleRect->w, pNeedleRect->h, m_pNeedleGreen, true);
	GFourier::fft2d(pNeedleRect->w, pNeedleRect->h, m_pNeedleBlue, true);

	// Multiply m_pHaystack with the complex conjugate of m_pNeedle
	double r, i, mag;
	int xx, yy;
	pos = 0;
	for(y = 0; y < m_nHaystackHeight; y++)
	{
		yy = (y * pNeedleRect->h / m_nHaystackHeight) * pNeedleRect->w;
		for(x = 0; x < m_nHaystackWidth; x++)
		{
			xx = x * pNeedleRect->w / m_nHaystackWidth;
			r = m_pNeedleRed[yy + xx].real * m_pHaystackRed[pos].real + m_pNeedleRed[yy + xx].imag * m_pHaystackRed[pos].imag;
			i = m_pNeedleRed[yy + xx].real * m_pHaystackRed[pos].imag - m_pNeedleRed[yy + xx].imag * m_pHaystackRed[pos].real;
			mag = sqrt(r * r + i * i);
			m_pCorRed[pos].real = r / mag;
			m_pCorRed[pos].imag = i / mag;
			r = m_pNeedleGreen[yy + xx].real * m_pHaystackGreen[pos].real + m_pNeedleGreen[yy + xx].imag * m_pHaystackGreen[pos].imag;
			i = m_pNeedleGreen[yy + xx].real * m_pHaystackGreen[pos].imag - m_pNeedleGreen[yy + xx].imag * m_pHaystackGreen[pos].real;
			mag = sqrt(r * r + i * i);
			m_pCorGreen[pos].real = r / mag;
			m_pCorGreen[pos].imag = i / mag;
			r = m_pNeedleBlue[yy + xx].real * m_pHaystackBlue[pos].real + m_pNeedleBlue[yy + xx].imag * m_pHaystackBlue[pos].imag;
			i = m_pNeedleBlue[yy + xx].real * m_pHaystackBlue[pos].imag - m_pNeedleBlue[yy + xx].imag * m_pHaystackBlue[pos].real;
			mag = sqrt(r * r + i * i);
			m_pCorBlue[pos].real = r / mag;
			m_pCorBlue[pos].imag = i / mag;
			pos++;
		}
	}

	// Convert to the Spatial domain
	GFourier::fft2d(m_nHaystackWidth, m_nHaystackHeight, m_pCorRed, false);
	GFourier::fft2d(m_nHaystackWidth, m_nHaystackHeight, m_pCorGreen, false);
	GFourier::fft2d(m_nHaystackWidth, m_nHaystackHeight, m_pCorBlue, false);

	// Find the max
	*pOutX = 0;
	*pOutY = 0;
	double d;
	double dBest = -1e200;
	for(y = 0; y < pHaystackRect->h; y++)
	{
		yy = m_nHaystackY + pHaystackRect->y + y;
		if(yy < 0 || yy >= m_nHaystackHeight) // todo: precompute range instead
			continue;
		yy *= m_nHaystackWidth;
		for(x = 0; x < pHaystackRect->w; x++)
		{
			xx = m_nHaystackX + pHaystackRect->x + x;
			if(xx < 0 || xx >= m_nHaystackWidth) // todo: precompute range instead
				continue;
			xx += yy;
			d = m_pCorRed[xx].real * m_pCorGreen[xx].real * m_pCorBlue[xx].real;
			if(d > dBest)
			{
				dBest = d;
				*pOutX = pHaystackRect->x + x;
				*pOutY = pHaystackRect->y + y;
			}
		}
	}
/*
	// Save correlation image
	GImage corr;
	corr.setSize(m_nHaystackWidth, m_nHaystackHeight);
	pos = 0;
	for(y = 0; y < m_nHaystackHeight; y++)
	{
		for(x = 0; x < m_nHaystackWidth; x++)
		{
			d = m_pCorRed[pos].real * m_pCorGreen[pos].real * m_pCorBlue[pos].real;
			corr.setPixel(x, y, gFromGray(std::max((int)0, (int)(d * 255 * 256 / dBest))));
			pos++;
		}
	}
	corr.savePpm("correlat.ppm");
*/
}

// static
void GSubImageFinder::test()
{
	// Make some random image
	GImage foo;
	foo.setSize(256, 256);
	foo.clear(0xff000000);
	foo.boxFill(13, 8, 12, 17, 0xff808030);
	foo.boxFill(8, 13, 10, 9, 0xff407040);
	foo.boxFill(20, 20, 220, 220, 0xffffffee);

	// Make the finder
	GSubImageFinder finder(&foo);

	// Make a sub-image
	GRect r2(0, 0, 256, 256);
	GRect r;
	r.x = 13;
	r.y = 17;
	r.w = 32;
	r.h = 32;
	GImage bar;
	bar.setSize(32, 32);
	bar.blit(0, 0, &foo, &r);

	// Find the sub-image
	r.x = 0;
	r.y = 0;
	int x, y;
	finder.findSubImage(&x, &y, &bar, &r, &r2);
	if(x != 13 || y != 17)
		throw Ex("wrong answer");
}






class GSIFStats
{
public:
	unsigned int m_x;
	unsigned int m_y;
	unsigned int m_lastPassIter;
	size_t m_sse;
};

class GSIFStatsComparer
{
public:
	bool operator() (const GSIFStats* a, const GSIFStats* b) const
	{
		return a->m_sse < b->m_sse;
	}
};

GSubImageFinder2::GSubImageFinder2(GImage* pHaystack)
: m_pHaystack(pHaystack)
{
}

GSubImageFinder2::~GSubImageFinder2()
{
}

void GSubImageFinder2::findSubImage(int* pOutX, int* pOutY, GImage* pNeedle, GRect* pNeedleRect)
{
	// Fill a vector of candidate offsets with every possible offset
	vector<GSIFStats*> cands;
	cands.reserve((m_pHaystack->height() - pNeedleRect->h) * (m_pHaystack->width() - pNeedleRect->w));
	VectorOfPointersHolder<GSIFStats> hCands(cands);
	for(unsigned int y = 0; y + pNeedleRect->h <= m_pHaystack->height(); y++)
	{
		for(unsigned int x = 0; x + pNeedleRect->w <= m_pHaystack->width(); x++)
		{
			GSIFStats* pStats = new GSIFStats();
			cands.push_back(pStats);
			pStats->m_x = x;
			pStats->m_y = y;
			pStats->m_lastPassIter = 0;
			pStats->m_sse = 0;
		}
	}

	// Measure pixel differences until we can narrow the candidate set down to just one
	GSIFStatsComparer comparer;
	size_t ranges[2];
	ranges[0] = pNeedleRect->w;
	ranges[1] = pNeedleRect->h;
	GCoordVectorIterator cvi(2, ranges); // This chooses which pixel to try next
	for(unsigned int iters = 0; true; iters++)
	{
		// Do another pixel (penalize each candidate for any difference in that pixel)
		size_t best = INVALID_INDEX;
		size_t* pCoords = cvi.current();
		GAssert(pCoords[0] < (size_t)pNeedleRect->w && pCoords[1] < (size_t)pNeedleRect->h);
		unsigned int n = pNeedle->pixel((int)pCoords[0], (int)pCoords[1]);
		for(vector<GSIFStats*>::iterator it = cands.begin(); it != cands.end(); it++)
		{
			GSIFStats* pStats = *it;
			size_t* pCoords2 = cvi.current();
			unsigned int h = m_pHaystack->pixel((int)pStats->m_x + (int)pCoords2[0], (int)pStats->m_y + (int)pCoords2[1]);
			int dif;
			dif = gRed(h) - gRed(n);
			pStats->m_sse += (dif * dif);
			dif = gGreen(h) - gGreen(n);
			pStats->m_sse += (dif * dif);
			dif = gBlue(h) - gBlue(n);
			pStats->m_sse += (dif * dif);
			if(pStats->m_sse < best)
			{
				best = pStats->m_sse;
				*pOutX = pStats->m_x;
				*pOutY = pStats->m_y;
			}
		}

		// Divide into the best and worst halves
		vector<GSIFStats*>::iterator median = cands.begin() + (cands.size() / 2);
		std::nth_element(cands.begin(), median, cands.end(), comparer);

		// Ensure that the best half will survive for a while longer
		for(vector<GSIFStats*>::iterator it = cands.begin(); it != median; it++)
		{
			GSIFStats* pStats = *it;
			pStats->m_lastPassIter = iters;
		}

		// Kill off candidates that have been in the worst half for too long
		for(size_t i = median - cands.begin(); i < cands.size(); i++)
		{
			if(iters - cands[i]->m_lastPassIter >= 32)
			{
				size_t last = cands.size() - 1;
				delete(cands[i]);
				std::swap(cands[i], cands[last]);
				cands.erase(cands.begin() + last);
				i--;
			}
		}

		// Pick the next pixel (using a well-distributed sampling technique)
		if(!cvi.advanceSampling() || cands.size() == 1)
			break;
	}

	// Return the results
	vector<GSIFStats*>::iterator itBest = std::min_element(cands.begin(), cands.end(), comparer);
	*pOutX = (*itBest)->m_x;
	*pOutY = (*itBest)->m_y;
}

// static
void GSubImageFinder2::test()
{
	// Make some random image
	GImage foo;
	foo.setSize(256, 256);
	foo.clear(0xff000000);
	foo.boxFill(13, 8, 12, 17, 0xff808030);
	foo.boxFill(8, 13, 10, 9, 0xff407040);
	foo.boxFill(20, 20, 220, 220, 0xffffffee);

	// Make the finder
	GSubImageFinder2 finder(&foo);

	// Make a sub-image
	GRect r2(0, 0, 256, 256);
	GRect r;
	r.x = 13;
	r.y = 17;
	r.w = 32;
	r.h = 32;
	GImage bar;
	bar.setSize(32, 32);
	bar.blit(0, 0, &foo, &r);

	// Find the sub-image
	r.x = 0;
	r.y = 0;
	int x, y;
	finder.findSubImage(&x, &y, &bar, &r);
	if(x != 13 || y != 17)
		throw Ex("wrong answer");
}



} // namespace GClasses
