// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include "Paint.h"
#ifdef WINDOWS
#	include <windows.h>
#else
#	include <unistd.h>
#endif
#include <GClasses/GBitTable.h>
#include <GClasses/GTime.h>
#include <GClasses/GError.h>
#include <GClasses/GFile.h>
#include <GClasses/GImage.h>
#include <GClasses/GBits.h>
#include <GClasses/GThread.h>
#include <GClasses/GRegion.h>
#include <GClasses/GGraph.h>
#include <GClasses/GKNN.h>
#include <GClasses/GBezier.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GCluster.h>
#include <math.h>
#include <vector>
#include <iostream>

using namespace GClasses;
using std::cerr;
using std::vector;


#define PAINT_TAB_COUNT 2
#define TOOL_AREA_SIZE 150
#define PAINT_AREA_BACKGROUND_COLOR 0xff888888

// todo: remove me
#define OCR_HALF_VEC_SIZE 20


class PaintTool
{
protected:
	GImage* m_pImage;

public:
	PaintTool(GImage* pImage)
	{
		m_pImage = pImage;
	}

	virtual ~PaintTool()
	{
	}

	virtual void onSelect() {}
	virtual void onMouseDown(int nButton, int x, int y) = 0;
	virtual void onMouseUp(int nButton, int x, int y) {}
	virtual void onMouseMove(int x, int y, bool bPressed) {}
};

class PaintToolPen : public PaintTool
{
protected:
	int m_prevX, m_prevY;

public:
	PaintToolPen(GImage* pImage) : PaintTool(pImage) {}
	virtual ~PaintToolPen() {}

	virtual void onMouseDown(int nButton, int x, int y)
	{
		m_prevX = x;
		m_prevY = y;
		m_pImage->setPixelIfInRange(x, y, 0xffff0000);
	}

	virtual void onMouseMove(int x, int y, bool bPressed)
	{
		if(bPressed && (x != m_prevX || y != m_prevY))
		{
			m_pImage->line(m_prevX, m_prevY, x, y, 0xffff0000);
			m_prevX = x;
			m_prevY = y;
		}
	}
};

class PaintToolBorder : public PaintTool
{
protected:

public:
	PaintToolBorder(GImage* pImage) : PaintTool(pImage) {}
	virtual ~PaintToolBorder() {}

	virtual void onMouseDown(int nButton, int x, int y)
	{
		GImage tmp;
		tmp.setSize(m_pImage->width(), m_pImage->height());
		tmp.clear(0xff000000);
		int xx, yy, dd;
		GRegionBorderIterator itt(m_pImage, x, y);
		while(itt.next(&xx, &yy, &dd))
			tmp.setPixel(xx, yy, 0xffffffff);
		m_pImage->swapData(&tmp);
	}
};

class PaintToolStretch : public PaintTool
{
protected:
	GImage m_imageDown;
	int m_downX, m_downY, m_prevX, m_prevY;

public:
	PaintToolStretch(GImage* pImage) : PaintTool(pImage) {}
	virtual ~PaintToolStretch() {}

	virtual void onMouseDown(int nButton, int x, int y)
	{
		m_imageDown.copy(m_pImage);
		m_downX = x;
		m_downY = y;
	}

	virtual void onMouseMove(int x, int y, bool bPressed)
	{
		if(bPressed && (x != m_prevX || y != m_prevY))
		{
			m_pImage->copy(&m_imageDown);
			m_pImage->stretch(m_downX, m_downY, x, y);
			m_prevX = x;
			m_prevY = y;
		}
	}
};

class PaintToolBezier : public PaintTool
{
protected:
	vector<int> m_points;

public:
	PaintToolBezier(GImage* pImage) : PaintTool(pImage), m_points(32) {}
	virtual ~PaintToolBezier() {}

	virtual void onMouseDown(int nButton, int x, int y)
	{
		if(nButton == 1)
		{
			m_points.push_back(x);
			m_points.push_back(y);
			m_pImage->setPixel(x, y, 0xff00ff00);
		}
		else
		{
			// Construct the GBezier object
			G3DVector point;
			GBezier bez(m_points.size() / 2);
			for(size_t i = 0; i < m_points.size(); i += 2)
			{
				point.m_vals[0] = m_points[i];
				point.m_vals[1] = m_points[i + 1];
				bez.setControlPoint(i / 2, &point, 1);
			}
			m_points.clear();

			// Draw the curve
			for(int i = 0; i < 1000; i++)
			{
				bez.point((double)i / 1000, &point);
				m_pImage->setPixelIfInRange((int)point.m_vals[0], (int)point.m_vals[1], 0xff0000ff);
			}
		}
	}
};


class PaintToolSmartSelect : public PaintTool
{
protected:
	G2DRegionGraph* m_pRegions;
	GImage* m_pSelection;
	GBitTable* m_pForeground;
	GBitTable* m_pBackground;
	int m_nCurrentButton;

public:
	PaintToolSmartSelect(GImage* pImage, GImage* pSelection) : PaintTool(pImage)
	{
		m_pRegions = NULL;
		m_pSelection = pSelection;
		m_pForeground = NULL;
		m_pBackground = NULL;
	}

	virtual ~PaintToolSmartSelect()
	{
		delete(m_pRegions);
		delete(m_pForeground);
		delete(m_pBackground);
	}

	virtual void onSelect()
	{
		delete(m_pRegions);
		m_pRegions = new G2DRegionGraph(m_pImage->width(), m_pImage->height());
		m_pRegions->makeWatershedRegions(m_pImage);

		while(m_pRegions->regionCount() > 32000)
		{
			G2DRegionGraph* pNewRegionList = new G2DRegionGraph(m_pImage->width(), m_pImage->height());
			pNewRegionList->makeCoarserRegions(m_pRegions);
			delete(m_pRegions);
			m_pRegions = pNewRegionList;
		}

/*
GImage colorRegion;
colorRegion.CopyImage(m_pRegions->GetRegionMask());
colorRegion.ColorizeRegionMap();
colorRegion.SavePNGFile("/home/mike/regions.png");
*/
		delete(m_pForeground);
		delete(m_pBackground);
		m_pForeground = new GBitTable(m_pRegions->regionCount());
		m_pBackground = new GBitTable(m_pRegions->regionCount());
	}

	virtual void onMouseDown(int nButton, int x, int y)
	{
		m_nCurrentButton = nButton;
		onMouseMove(x, y, true);
	}

	void SelectSourceRegions(GGraphCut* pGC)
	{
		int x, y;
		GImage* pRegionMask = m_pRegions->regionMask();
		unsigned int nRegion;
		for(y = 0; y < (int)m_pSelection->height(); y++)
		{
			for(x = 0;  x < (int)m_pSelection->width(); x++)
			{
				nRegion = pRegionMask->pixel(x, y);
				if(pGC->isSource(nRegion))
					m_pSelection->setPixel(x, y, (rand() << 12) | rand() | 0xff000000);
				else
					m_pSelection->setPixel(x, y, 0);
			}
		}
	}

	void SetEdgeRegionsAsBackground(GGraphCut* pGC, float fCapacity)
	{
		unsigned int nRegion[6];
		int i;
		for(i = 0; i < 6; i++)
			nRegion[i] = (unsigned int)-1;
		int nRegionCount = m_pRegions->regionCount();
		GImage* pRegionMask = m_pRegions->regionMask();
		for(i = 0; i < (int)pRegionMask->width(); i++)
		{
			nRegion[0] = pRegionMask->pixel(i, 0);
			if(nRegion[0] != nRegion[1] && nRegion[0] != nRegion[2])
			{
				pGC->addEdge(nRegion[0], nRegionCount + 1, fCapacity);
				nRegion[2] = nRegion[1];
				nRegion[1] = nRegion[0];
			}
			nRegion[3] = pRegionMask->pixel(i, pRegionMask->height() - 1);
			if(nRegion[3] != nRegion[4] && nRegion[3] != nRegion[5])
			{
				pGC->addEdge(nRegion[3], nRegionCount + 1, fCapacity);
				nRegion[5] = nRegion[4];
				nRegion[4] = nRegion[3];
			}
		}
		for(i = 1; i < (int)pRegionMask->height() - 1; i++)
		{
			nRegion[0] = pRegionMask->pixel(0, i);
			if(nRegion[0] != nRegion[1] && nRegion[0] != nRegion[2])
			{
				pGC->addEdge(nRegion[0], nRegionCount + 1, fCapacity);
				nRegion[2] = nRegion[1];
				nRegion[1] = nRegion[0];
			}
			nRegion[3] = pRegionMask->pixel(pRegionMask->width() - 1, i);
			if(nRegion[3] != nRegion[4] && nRegion[3] != nRegion[5])
			{
				pGC->addEdge(nRegion[3], nRegionCount + 1, fCapacity);
				nRegion[5] = nRegion[4];
				nRegion[4] = nRegion[3];
			}
		}
	}

	virtual void onMouseUp(int nButton, int x, int y)
	{
		// Make a graph of the regions
		int nRegionCount = m_pRegions->regionCount();
		GGraphCut gc(nRegionCount + 2);
		gc.getEdgesFromRegionList(m_pRegions); // (where edges refers to graph edges between vertices)

		// Connect frame border regions to background
		SetEdgeRegionsAsBackground(&gc, 30); // (where edges refers to the borders of the frame)

		// Connect foreground to the source and background to the sink
		int nForeCount = 0;
		int nBackCount = 0;
		int i;
		for(i = 0; i < nRegionCount; i++)
		{
			if(m_pForeground->bit(i))
			{
				gc.addEdge(i, nRegionCount, (float)1e30);
				nForeCount++;
			}
			else if(m_pBackground->bit(i))
			{
				gc.addEdge(i, nRegionCount + 1, (float)1e30);
				nBackCount++;
			}
		}
		//if(nForeCount <= 0 || nBackCount <= 0)
		//	return;

		// Cut the graph and select the foreground
		gc.cut(nRegionCount, nRegionCount + 1);
		SelectSourceRegions(&gc);
	}

	virtual void onMouseMove(int x, int y, bool bPressed)
	{
		if(!bPressed)
			return;
		unsigned int nRegion = m_pRegions->regionMask()->pixel(x, y);
		if(m_nCurrentButton == 1)
		{
			m_pForeground->set(nRegion);
			m_pBackground->unset(nRegion);
		}
		else if(m_nCurrentButton == 3)
		{
			m_pForeground->unset(nRegion);
			m_pBackground->set(nRegion);
		}
	}
};

// ----------------------------------------------------------------------------

class PaintDialog : public GWidgetDialog
{
friend class PaintController;
protected:
	GImage* m_pImage;
	GImage* m_pSelection;
	PaintController* m_pController;
	GWidgetTextTab** m_pTabs;
	GWidgetCanvas* m_pCanvas;
	GWidgetHorizScrollBar* m_pHorizScrollBar;
	GWidgetVertScrollBar* m_pVertScrollBar;
	PaintTool* m_pCurrentTool;

public:
	PaintDialog(PaintController* pController, int w, int h, GImage* pImage, GImage* pSelection)
	: GWidgetDialog(w, h, PAINT_AREA_BACKGROUND_COLOR)
	{
		m_pCurrentTool = NULL;
		m_pImage = pImage;
		m_pSelection = pSelection;
		m_pController = pController;
		m_pTabs = new GWidgetTextTab*[PAINT_TAB_COUNT];
		m_pTabs[0] = new GWidgetTextTab(this, 0, 0, 50, 20, "File", 0xff008800);
		m_pTabs[1] = new GWidgetTextTab(this, 50, 0, 50, 20, "Tools", 0xff008800);
		m_pTabs[0]->setSelected(true);

		m_pCanvas = new GWidgetCanvas(this, 0, 20, w - 16, h - 36, pImage);
		m_pCanvas->setSelectionMask(m_pSelection);
		m_pHorizScrollBar = new GWidgetHorizScrollBar(this, 0, h - 16, w - 16, 16, w - 16, pImage->width());
		m_pVertScrollBar = new GWidgetVertScrollBar(this, w - 16, 0, 16, h, h - 36, pImage->height());
	}

	virtual ~PaintDialog()
	{
		delete[] m_pTabs;
	}

	virtual void onSelectTextTab(GWidgetTextTab* pTab)
	{
		pTab->setSelected(true);
		int i;
		for(i = 0; i < PAINT_TAB_COUNT; i++)
		{
			if(!m_pTabs[i])
				continue;
			if(m_pTabs[i] == pTab)
				m_pController->OnSelectTab(i);
			else
				m_pTabs[i]->setSelected(false);
		}
	}

	void OnLoadImage()
	{
		m_pCanvas->zoomToFit();
		CleanCanvas();
	}

	void CleanCanvas()
	{
		// This is just a cosmetic thing so that the edges of an old
		// larger image don't clutter up the edges around the new smaller image
//		m_pCanvas->GetOutImage()->Clear(PAINT_AREA_BACKGROUND_COLOR);
		m_pHorizScrollBar->setModelSize((int)(m_pCanvas->zoom() * m_pImage->width()));
		m_pVertScrollBar->setModelSize((int)(m_pCanvas->zoom() * m_pImage->height()));
	}

	inline void RedrawCanvas()
	{
		m_pCanvas->setDirty();
	}

	virtual void onHorizScroll(GWidgetHorizScrollBar* pScrollBar)
	{
		m_pCanvas->setHorizScroll(pScrollBar->pos());
		RedrawCanvas();
	}

	virtual void onVertScroll(GWidgetVertScrollBar* pScrollBar)
	{
		m_pCanvas->setVertScroll(pScrollBar->pos());
		RedrawCanvas();
	}

	virtual void onCanvasMouseDown(GWidgetCanvas* pCanvas, int nButton, int x, int y)
	{
		m_pCurrentTool->onMouseDown(nButton, x, y);
		RedrawCanvas();
	}

	virtual void onCanvasMouseUp(GWidgetCanvas* pCanvas, int nButton, int x, int y)
	{
		m_pCurrentTool->onMouseUp(nButton, x, y);
		RedrawCanvas();
	}

	virtual void onCanvasMouseMove(GWidgetCanvas* pCanvas, int x, int y, bool bPressed)
	{
		if(x < (int)m_pImage->width() && y < (int)m_pImage->height())
			m_pCurrentTool->onMouseMove(x, y, bPressed);
		if(bPressed)
			RedrawCanvas();
	}

	void SetCurrentTool(PaintTool* pTool)
	{
		m_pCurrentTool = pTool;
	}

	void ZoomIn()
	{
		m_pCanvas->setZoom(m_pCanvas->zoom() * 2);
		CleanCanvas();
	}

	void ZoomOut()
	{
		m_pCanvas->setZoom(m_pCanvas->zoom() / 2);
		CleanCanvas();
	}
};


// ----------------------------------------------------------------------------

class TabDialog : public GWidgetDialog
{
protected:
	PaintController* m_pController;

public:
	TabDialog(PaintController* pController, int w)
	 : GWidgetDialog(w, TOOL_AREA_SIZE, PAINT_AREA_BACKGROUND_COLOR)
	{
		m_pController = pController;
	}

	virtual ~TabDialog()
	{
	}
};

// ----------------------------------------------------------------------------

class TabOpen : public TabDialog
{
protected:
	GWidgetTextButton* m_pOpen;
	GWidgetTextButton* m_pSave;

public:
	TabOpen(PaintController* pController, int w)
	 : TabDialog(pController, w)
	{
		m_pOpen = new GWidgetTextButton(this, 10, 10, 100, 20, "Open");
		m_pSave = new GWidgetTextButton(this, 10, 40, 100, 20, "Save");
	}

	virtual ~TabOpen()
	{
	}

	virtual void onReleaseTextButton(GWidgetTextButton* pButton)
	{
		if(pButton == m_pOpen)
			m_pController->OpenFile();
		else if(pButton == m_pSave)
			m_pController->SaveFile();
	}
};


// ----------------------------------------------------------------------------

class TabTools : public TabDialog
{
protected:
	GWidgetTextTab* m_pCurrentToolTab;
	GWidgetTextButton* m_pZoomInButton;
	GWidgetTextButton* m_pZoomOutButton;
	GWidgetTextButton* m_pHighPassButton;
	GWidgetTextButton* m_pMedianFilter;
	GWidgetTextButton* m_pThreshold;
	GWidgetTextButton* m_pDialateButton;
	GWidgetTextButton* m_pErodeButton;
	GWidgetTextButton* m_pOpenButton;
	GWidgetTextButton* m_pCloseButton;
	GWidgetTextButton* m_pInvertButton;
	GWidgetTextButton* m_pEqualizeHistogram;
	GWidgetTextButton* m_pShowSelButton;
	GWidgetTextButton* m_pBlurButton;
	GWidgetTextButton* m_pGlowButton;
	GWidgetTextButton* m_pCutButton;
	GWidgetTextButton* m_pGrayToAlphaButton;

	GWidgetTextTab* m_pTabPen;
	PaintToolPen* m_pToolPen;

	GWidgetTextTab* m_pTabStretch;
	PaintToolStretch* m_pToolStretch;

	GWidgetTextTab* m_pTabSmartSelect;
	PaintToolSmartSelect* m_pToolSmartSelect;

	GWidgetTextTab* m_pTabBezier;
	PaintToolBezier* m_pToolBezier;

	GWidgetTextTab* m_pTabBorder;
	PaintToolBorder* m_pToolBorder;

	GWidgetTextBox* m_pTB1;
	GWidgetTextBox* m_pTB2;

public:
	TabTools(PaintController* pController, int w)
	 : TabDialog(pController, w)
	{
		m_pZoomInButton = new GWidgetTextButton(this, 5, 30, 80, 20, "Zoom In");
		m_pZoomOutButton = new GWidgetTextButton(this, 5, 54, 80, 20, "Zoom Out");
		m_pHighPassButton = new GWidgetTextButton(this, 5, 78, 80, 20, "High Pass");
		m_pMedianFilter = new GWidgetTextButton(this, 5, 102, 80, 20, "Med. Filter");
		m_pThreshold = new GWidgetTextButton(this, 5, 126, 80, 20, "Auto Thresh");
		m_pDialateButton = new GWidgetTextButton(this, 90, 30, 80, 20, "Dialate");
		m_pErodeButton = new GWidgetTextButton(this, 90, 54, 80, 20, "Erode");
		m_pOpenButton = new GWidgetTextButton(this, 90, 78, 80, 20, "Morph Open");
		m_pCloseButton = new GWidgetTextButton(this, 90, 102, 80, 20, "Morph Close");
		m_pInvertButton = new GWidgetTextButton(this, 175, 30, 80, 20, "Invert");
		m_pEqualizeHistogram = new GWidgetTextButton(this, 175, 54, 80, 20, "Eq. Hist");
		m_pShowSelButton = new GWidgetTextButton(this, 175, 102, 80, 20, "Show Sel.");
		m_pBlurButton = new GWidgetTextButton(this, 260, 30, 80, 20, "Blur");
		m_pGlowButton = new GWidgetTextButton(this, 260, 54, 80, 20, "Add Border");
		m_pCutButton = new GWidgetTextButton(this, 260, 78, 80, 20, "Cut Sel.");
		m_pGrayToAlphaButton = new GWidgetTextButton(this, 260, 102, 80, 20, "GrayToAlpha");

		m_pTabPen = new GWidgetTextTab(this, 5, 5, 50, 20, "Pen");
		m_pToolPen = new PaintToolPen(pController->GetCanvas());
		m_pCurrentToolTab = m_pTabPen;
		m_pCurrentToolTab->setSelected(true);

		m_pTabStretch = new GWidgetTextTab(this, 55, 5, 50, 20, "Stretch");
		m_pToolStretch = new PaintToolStretch(pController->GetCanvas());

		m_pTabSmartSelect = new GWidgetTextTab(this, 105, 5, 100, 20, "Smart Select");
		m_pToolSmartSelect = new PaintToolSmartSelect(pController->GetCanvas(), pController->GetSelectionMask());

		m_pTabBorder = new GWidgetTextTab(this, 205, 5, 50, 20, "Border");
		m_pToolBorder = new PaintToolBorder(pController->GetCanvas());

		m_pTabBezier = new GWidgetTextTab(this, 255, 5, 50, 20, "Bezier");
		m_pToolBezier = new PaintToolBezier(pController->GetCanvas());

		new GWidgetTextLabel(this, 500, 34, 150, 16, "Eq. Hist. Amount");
		m_pTB1 = new GWidgetTextBox(this, 500, 50, 50, 20);
		m_pTB1->setText(".5");

		new GWidgetTextLabel(this, 500, 84, 150, 16, "Open Amount");
		m_pTB2 = new GWidgetTextBox(this, 500, 100, 50, 20);
		m_pTB2->setText("2");
	}

	virtual ~TabTools()
	{
		delete(m_pToolPen);
		delete(m_pToolStretch);
		delete(m_pToolSmartSelect);
		delete(m_pToolBezier);
	}

	virtual void onReleaseTextButton(GWidgetTextButton* pButton)
	{
		GImage* pImage = m_pController->GetCanvas();
		if(pButton == m_pZoomInButton)
			m_pController->ZoomIn();
		else if(pButton == m_pZoomOutButton)
			m_pController->ZoomOut();
		else if(pButton == m_pHighPassButton)
			pImage->highPassFilter(.1);
		else if(pButton == m_pMedianFilter)
			pImage->medianFilter((float)2);
		else if(pButton == m_pThreshold)
			pImage->threshold(128 * 256);
		else if(pButton == m_pDialateButton)
		{
			GImage se;
			se.gaussianKernel(3, (float)1);
			pImage->dialate(&se);
		}
		else if(pButton == m_pErodeButton)
		{
			GImage se;
			se.gaussianKernel(3, (float)1);
			pImage->erode(&se);
		}
		else if(pButton == m_pOpenButton)
			pImage->open(1);
		else if(pButton == m_pCloseButton)
			pImage->open(-1);
		else if(pButton == m_pInvertButton)
			pImage->invert();
		else if(pButton == m_pEqualizeHistogram)
		{
			float fEqualizeAmount = (float)atof(m_pTB1->text().c_str());
			pImage->locallyEqualizeColorSpread(std::max(pImage->width(), pImage->height()) * 2, fEqualizeAmount);
		}
		else if(pButton == m_pShowSelButton)
		{
			GImage* pSelection = m_pController->GetSelectionMask();
			int x, y, c;
			for(y = 0; y < (int)pImage->height(); y++)
			{
				for(x = 0; x < (int)pImage->width(); x++)
				{
					c = gAlpha(pSelection->pixel(x, y));
					pImage->setPixel(x, y, gARGB(0xff, c, c, c));
				}
			}
			pSelection->clear(0);
			m_pController->CleanCanvas();
		}
		else if(pButton == m_pBlurButton)
		{
			pImage->blur(7);
			//pImage->QuickBlur(5);
		}
		else if(pButton == m_pGlowButton)
		{
			GImage tmp;
			tmp.addBorder(pImage, 0xffffffff, 0xff000000);
			tmp.swapData(pImage);
//pImage->SavePNGFile("foo.png");
		}
		else if(pButton == m_pCutButton)
		{
			GImage* pSelection = m_pController->GetSelectionMask();
			int x, y, c;
			for(y = 0; y < (int)pImage->height(); y++)
			{
				for(x = 0; x < (int)pImage->width(); x++)
				{
					c = gAlpha(pSelection->pixel(x, y));
					if(c == 0)
						pImage->setPixel(x, y, 0);
				}
			}
			pSelection->clear(0);
			m_pController->CleanCanvas();
		}
		else if(pButton == m_pGrayToAlphaButton)
		{
			int x, y, c;
			for(y = 0; y < (int)pImage->height(); y++)
			{
				for(x = 0; x < (int)pImage->width(); x++)
				{
					c = gGray(pImage->pixel(x, y)) / 256;
					pImage->setPixel(x, y, gARGB(c, 0xff, 0xff, 0xff));
				}
			}
		}
		else
			GAssert(false); // unrecognized button

		// Redraw the canvas
		m_pController->RedrawCanvas();
	}

	void SaveVector(double* pVector, const char* szFilename)
	{
		GImage image;
		image.setSize(2 * OCR_HALF_VEC_SIZE, 2 * OCR_HALF_VEC_SIZE);
		image.clear(0xffffffff);
		int x, y;
		for(x = 0; x < 2 * OCR_HALF_VEC_SIZE; x++)
		{
			for(y = 0; y < OCR_HALF_VEC_SIZE * pVector[x] && y < 2 * OCR_HALF_VEC_SIZE; y++)
				image.setPixel(x, 2 * OCR_HALF_VEC_SIZE - 1 - y, x >= OCR_HALF_VEC_SIZE ? 0xff00ff00 : 0xff0000ff);
		}
		image.saveBmp(szFilename);
	}

	double EvaluateOCRVector(double* pVector)
	{
		GAssert((OCR_HALF_VEC_SIZE & 1) == 0); // OCR_HALF_VEC_SIZE should be a multiple of 2
		double a = 0;
		double b = 0;
		double c = 0;
		double d = 0;
		int i;
		for(i = 0; i < OCR_HALF_VEC_SIZE / 2; i++)
		{
			a += pVector[i];
			b += pVector[OCR_HALF_VEC_SIZE / 2 + i];
			c += pVector[OCR_HALF_VEC_SIZE + i];
			d += pVector[OCR_HALF_VEC_SIZE * 3 / 2 + i];
		}
		return 7.1 * (a + b) / (std::max(c + d, .00001)) + 2.5 * (a / std::max(b, .00001)) + (c / std::max(d, .00001));
	}

	void RotateOCRVector(double* pVector)
	{
		GTEMPBUF(double, pTemp, 2 * OCR_HALF_VEC_SIZE);
		memcpy(pTemp, pVector, sizeof(double) * 2 * OCR_HALF_VEC_SIZE);
		int i;
		for(i = 0; i < OCR_HALF_VEC_SIZE; i++)
		{
			pVector[i] = pTemp[OCR_HALF_VEC_SIZE + i];
			pVector[OCR_HALF_VEC_SIZE + i] = pTemp[OCR_HALF_VEC_SIZE - 1 - i];
		}
	}

	void MakeOCRVector(double* pOutVector, GImage* pImage, int nRegion)
	{
		// Find the bounds
		int l = pImage->width() - 1;
		int r = 0;
		int t = pImage->height() - 1;
		int b = 0;
		int x, y, n;
		for(y = 0; y < (int)pImage->height(); y++)
		{
			for(x = 0; x < (int)pImage->width(); x++)
			{
				if((int)pImage->pixel(x, y) == nRegion)
				{
					if(x < l)
						l = x;
					if(x > r)
						r = x;
					if(y < t)
						t = y;
					if(y > b)
						b = y;
				}
			}
		}

		// Compute the vector
		int i = 0;
		int w = r - l + 1;
		int h = b - t + 1;
		int min, max, nCount, nTot;
		for(n = 0; n < OCR_HALF_VEC_SIZE; n++)
		{
			// Vertical profile
			min = n * w / OCR_HALF_VEC_SIZE + l;
			max = (n + 1) * w / OCR_HALF_VEC_SIZE + l;
			if(max == min)
				max++;
			nCount = 0;
			nTot = 0;
			for(x = min; x < max; x++)
			{
				for(y = t; y <= b; y++)
				{
					if((int)pImage->pixel(x, y) == nRegion)
						nCount++;
					nTot++;
				}
			}
			GAssert(nCount > 0); // something's wrong
			pOutVector[n] = (double)nCount /** h*/ / (nTot /** w*/);

			// Horizontal profile
			min = n * h / OCR_HALF_VEC_SIZE + t;
			max = (n + 1) * h / OCR_HALF_VEC_SIZE + t;
			if(max == min)
				max++;
			nCount = 0;
			nTot = 0;
			for(y = min; y < max; y++)
			{
				for(x = l; x <= r; x++)
				{
					if((int)pImage->pixel(x, y) == nRegion)
						nCount++;
					nTot++;
				}
			}
			GAssert(nCount > 0); // something's wrong
			pOutVector[OCR_HALF_VEC_SIZE + n] = (double)nCount /** w*/ / (nTot /** h*/);
		}

		// Find the best rotation
		double dBestScore = 0;
		double d;
		int nBest = -1;
		for(i = 0; i < 4; i++)
		{
			d = EvaluateOCRVector(pOutVector);
			if(d > dBestScore)
			{
				dBestScore = d;
				nBest = i;
			}
			RotateOCRVector(pOutVector);
		}
		GAssert(nBest >= 0); // something's wrong
		for(i = 0; i < nBest; i++)
			RotateOCRVector(pOutVector);
	}


	void EvalRS(GImage* pImage, const char* szMaster, const char* szCand, int x, int y, unsigned int col)
	{
		// Load the master
		char szLine[256];
		FILE* pFile = fopen(szMaster, "r");
		if(!pFile)
			ThrowError("Failed to open file");
		if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
		int nLenMaster = atoi(szLine);
		double* pMaster = new double[nLenMaster];
		Holder<double> hMaster(pMaster);
		if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
		int nStartMaster = atoi(szLine);
		if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
		int i;
		for(i = 0; i < nLenMaster; i++)
		{
			if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
			pMaster[i] = atof(szLine);
			if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
		}
		fclose(pFile);

		// Load the candidate
		pFile = fopen(szCand, "r");
		if(!pFile)
			ThrowError("Failed to open file");
		if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
		int nLenCand = atoi(szLine);
		double* pCand1 = new double[nLenCand];
		double* pCand2 = new double[nLenCand];
		Holder<double> hCand1(pCand1);
		Holder<double> hCand2(pCand2);
		if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
		int nStartCand1 = atoi(szLine);
		if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
		int nStartCand2 = atoi(szLine);
		for(i = 0; i < nLenCand; i++)
		{
			if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
			pCand1[i] = atof(szLine);
			if(!fgets(szLine, 256, pFile)) ThrowError("expected more");
			pCand2[i] = atof(szLine);
		}
		fclose(pFile);

		// Measure the error
		double dError1 = 0;
		double dError2 = 0;
		double d;
		int j;
		for(i = 0; i < nLenMaster; i++)
		{
			j = i * nLenCand / nLenMaster;
			int indexMaster = (i + nStartMaster) % nLenMaster;
			int indexCand1 = (j + nStartCand1) % nLenCand;
			d = pMaster[indexMaster] - pCand1[indexCand1];
			d *= d;
			dError1 += d;

			int indexCand2 = (j + nStartCand2) % nLenCand;
			d = pMaster[indexMaster] - pCand2[indexCand2];
			d *= d;
			dError2 += d;
		}
		double dFinal = std::min(dError1, dError2) / nLenMaster;
		sprintf(szLine, "%f", dFinal);
		pImage->text(szLine, x, y, 3.0f, col);
	}


	virtual void onSelectTextTab(GWidgetTextTab* pTab)
	{
		// Select the new tab
		m_pCurrentToolTab->setSelected(false);
		m_pCurrentToolTab = pTab;
		m_pCurrentToolTab->setSelected(true);

		// Get the new tool
		if(pTab == m_pTabPen)
			m_pController->SetCurrentTool(m_pToolPen);
		else if(pTab == m_pTabStretch)
			m_pController->SetCurrentTool(m_pToolStretch);
		else if(pTab == m_pTabSmartSelect)
			m_pController->SetCurrentTool(m_pToolSmartSelect);
		else if(pTab == m_pTabBorder)
			m_pController->SetCurrentTool(m_pToolBorder);
		else if(pTab == m_pTabBezier)
			m_pController->SetCurrentTool(m_pToolBezier);
		else
			GAssert(false); // unknown tool
	}

	void SelectDefaultTool()
	{
		m_pController->SetCurrentTool(m_pToolPen);
	}
};


// ----------------------------------------------------------------------------

class PaintView : public ViewBase
{
friend class PaintController;
protected:
	GWidgetDialog** m_pTabDialogs;
	int m_nSelectedTab;
	PaintDialog* m_pMainDialog;
	GImage* m_pImage;

public:
	PaintView(PaintController* pController, GImage* pImage, GImage* pSelection);
	virtual ~PaintView();

	void onSelectTab(int i);
	virtual void onChar(char c);
	virtual void onMouseDown(int nButton, int x, int y);
	virtual void onMouseUp(int nButton, int x, int y);
	virtual bool onMousePos(int x, int y);
	void SetImage(GImage* pImage) { m_pImage = pImage; }
	void OnLoadImage();
	void CleanCanvas();
	void RedrawCanvas();
	void ZoomIn();
	void ZoomOut();
	void SelectDefaultTool();
	void SetCurrentTool(PaintTool* pTool);

protected:
	virtual void draw(SDL_Surface *pScreen);
};

PaintView::PaintView(PaintController* pController, GImage* pImage, GImage* pSelection)
: ViewBase()
{
	m_pTabDialogs = new GWidgetDialog*[PAINT_TAB_COUNT];
	m_pTabDialogs[0] = new TabOpen(pController, m_screenRect.w);
	m_pTabDialogs[1] = new TabTools(pController, m_screenRect.w);
	m_nSelectedTab = 0;
	m_pMainDialog = new PaintDialog(pController, m_screenRect.w, m_screenRect.h - TOOL_AREA_SIZE, pImage, pSelection);
	m_pImage = NULL;
}

PaintView::~PaintView()
{
	int i;
	for(i = 0; i < PAINT_TAB_COUNT; i++)
		delete(m_pTabDialogs[i]);
	delete[] m_pTabDialogs;
	delete(m_pMainDialog);
}

/*virtual*/ void PaintView::draw(SDL_Surface *pScreen)
{
	// Draw the tool tab
	GImage* pImage = m_pTabDialogs[m_nSelectedTab]->image();
	blitImage(pScreen, m_screenRect.x, m_screenRect.y, pImage);

	// Draw the canvas dialog
	pImage = m_pMainDialog->image();
	blitImage(pScreen, m_screenRect.x, m_screenRect.y + TOOL_AREA_SIZE, pImage);
}

void PaintView::onSelectTab(int i)
{
	m_nSelectedTab = i;
}

void PaintView::onChar(char c)
{
	m_pTabDialogs[m_nSelectedTab]->handleChar(c);
}

void PaintView::onMouseDown(int nButton, int x, int y)
{
	if(y >= TOOL_AREA_SIZE)
		m_pMainDialog->pressButton(nButton, x - m_screenRect.x, y - m_screenRect.y - TOOL_AREA_SIZE);
	else
		m_pTabDialogs[m_nSelectedTab]->pressButton(nButton, x - m_screenRect.x, y - m_screenRect.y);
}

void PaintView::onMouseUp(int nButton, int x, int y)
{
	m_pMainDialog->releaseButton(nButton);
	m_pTabDialogs[m_nSelectedTab]->releaseButton(nButton);
}

bool PaintView::onMousePos(int x, int y)
{
	x -= m_screenRect.x;
	y -= m_screenRect.y;
	if(y >= TOOL_AREA_SIZE)
		return m_pMainDialog->handleMousePos(x, y - TOOL_AREA_SIZE);
	else
		return m_pTabDialogs[m_nSelectedTab]->handleMousePos(x, y);
}

void PaintView::OnLoadImage()
{
	m_pMainDialog->OnLoadImage();
}

void PaintView::CleanCanvas()
{
	m_pMainDialog->CleanCanvas();
}

void PaintView::RedrawCanvas()
{
	m_pMainDialog->RedrawCanvas();
}

void PaintView::ZoomIn()
{
	m_pMainDialog->ZoomIn();
}

void PaintView::ZoomOut()
{
	m_pMainDialog->ZoomOut();
}

void PaintView::SelectDefaultTool()
{
	((TabTools*)m_pTabDialogs[1])->SelectDefaultTool();
}

void PaintView::SetCurrentTool(PaintTool* pTool)
{
	m_pMainDialog->SetCurrentTool(pTool);
}

// -------------------------------------------------------------------------------

PaintController::PaintController()
: ControllerBase()
{
	m_pImage = new GImage();
	m_pImage->setSize(300, 300);
	m_pImage->clear(0xffffffff);
	m_pSelection = new GImage();
	m_pSelection->setSize(300, 300);
	m_pSelection->clear(0);
	m_pView = new PaintView(this, m_pImage, m_pSelection);
	((PaintView*)m_pView)->SelectDefaultTool();
}

PaintController::~PaintController()
{
	delete(m_pView);
	delete(m_pImage);
	delete(m_pSelection);
}

void PaintController::SetCurrentTool(PaintTool* pTool)
{
	((PaintView*)m_pView)->SetCurrentTool(pTool);
	pTool->onSelect();
}

void PaintController::OpenFile()
{
	GetOpenFilenameDialog dialog("Please select a file to open", ".png");
	RunPopup(&dialog);
	const char* szFilename = dialog.filename();
	if(!szFilename)
		return;
	m_pImage->loadByExtension(szFilename);
	m_pSelection->setSize(m_pImage->width(), m_pImage->height());
	m_pSelection->clear(0);
	((PaintView*)m_pView)->OnLoadImage();
	RedrawCanvas();
}

void PaintController::SaveFile()
{
	GetStringDialog dialog("Save as...");
	RunPopup(&dialog);
	const char* szFilename = dialog.m_pTextBox->text().c_str();
	if(!szFilename)
		return;
	try
	{
		m_pImage->saveByExtension(szFilename);
	}
	catch(const std::exception& e)
	{
		cerr << "Error while saving: " << e.what() << "\n";
#ifdef WINDOWS
		MessageBox(NULL, e.what(), "File not saved!", MB_OK);
#endif
	}
}

void PaintController::CleanCanvas()
{
	((PaintView*)m_pView)->CleanCanvas();
}

void PaintController::RedrawCanvas()
{
	((PaintView*)m_pView)->RedrawCanvas();
}

void PaintController::OnSelectTab(int i)
{
	((PaintView*)m_pView)->onSelectTab(i);
}

void PaintController::RunModal()
{
	double timeOld = GTime::seconds();
	double time;
	double timeUpdate = 0;
	m_pView->update();
	while(m_bKeepRunning)
	{
		time = GTime::seconds();
		if(handleEvents(time - timeOld)) // HandleEvents returns true if it thinks the view needs to be updated
		{
			m_pView->update();
			timeUpdate = time;
		}
		else
		{
			RedrawCanvas();
			m_pView->update();
			GThread::sleep(20);
		}
		timeOld = time;
	}
}

void PaintController::ZoomIn()
{
	((PaintView*)m_pView)->ZoomIn();
}

void PaintController::ZoomOut()
{
	((PaintView*)m_pView)->ZoomOut();
}
