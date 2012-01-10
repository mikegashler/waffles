// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef __CHART_H__
#define __CHART_H__

#include "../GClasses/GImage.h"
#include "../GClasses/GMatrix.h"
#include "../GClasses/GNeighborFinder.h"

class ChartController;
class ChartDialog;
class DoubleRect;
class PlotChartMaker;


class PlotChartMaker
{
protected:
	GClasses::GRand& m_rand;

	// Data
	GClasses::GMatrix* m_pData;
	GClasses::sp_relation m_pRelation;
	int m_nOutputCount;

	// Chart
	double m_dXMin, m_dYMin, m_dXMax, m_dYMax;
	int m_nChartWidth;
	int m_nChartHeight;
	bool m_bCustomRange;
	bool m_aspect;
	bool m_randomOrder;

	// Lines
	bool m_showLines;
	float m_fLineThickness;
	float m_fPointRadius;
	int m_meshSize;

	// Labels
	int m_nVertLabelPrecision;
	int m_nHorizLabelPrecision;
	float m_textSize;
	bool m_showHorizAxisLabels, m_showVertAxisLabels;
	bool m_logx, m_logy;
	int m_maxGridLinesH;
	int m_maxGridLinesV;

	// Colors
	unsigned int m_cBackground;
	unsigned int m_cText;
	unsigned int m_cGrid;
	unsigned int m_cPlot[4];
	bool m_useSpectrumColors;
	int m_spectrumModulus;

	// Neighbors
	GClasses::GNeighborFinder* m_pNeighborFinder;

public:
	PlotChartMaker(GClasses::sp_relation& pRelation, GClasses::GMatrix* pData, GClasses::GRand& rand);
	~PlotChartMaker();

	GClasses::GImage* MakeChart();

	void SetSize(int nWidth, int nHeight)
	{
		m_nChartWidth = nWidth;
		m_nChartHeight = nHeight;
	}

	void ShowAxisLabels(bool horiz, bool vert)
	{
		m_showHorizAxisLabels = horiz;
		m_showVertAxisLabels = vert;
	}

	void SetLogX() { m_logx = true; }
	void SetLogY() { m_logy = true; }

	void noLines()
	{
		m_showLines = false;
	}

	void setTextSize(float f)
	{
		m_textSize = f;
	}

	void SetPointRadius(float f)
	{
		m_fPointRadius = f;
	}

	void SetLineThickness(float f)
	{
		m_fLineThickness = f;
	}

	void SetMeshRowSize(int n)
	{
		m_meshSize = n;
	}

	void setAspect()
	{
		m_aspect = true;
	}

	void SetCustomRange(double xmin, double ymin, double xmax, double ymax)
	{
		m_bCustomRange = true;
		m_dXMin = xmin;
		m_dYMin = ymin;
		m_dXMax = xmax;
		m_dYMax = ymax;
	}

	void SetChartColors(unsigned int cBackground, unsigned int cText, unsigned int cGrid)
	{
		m_cBackground = cBackground;
		m_cText = cText;
		m_cGrid = cGrid;
	}

	void SetPlotColors(unsigned int c1, unsigned int c2, unsigned int c3, unsigned int c4)
	{
		m_cPlot[0] = c1;
		m_cPlot[1] = c2;
		m_cPlot[2] = c3;
		m_cPlot[3] = c4;
	}

	void setMaxGridLines(int h, int v)
	{
		m_maxGridLinesH = h;
		m_maxGridLinesV = v;
	}

	void UseSpectrumColors(int modulus = 0)
	{
		m_useSpectrumColors = true;
		m_spectrumModulus = modulus;
	}

	// Takes ownership of pNF
	void showNeighbors(GClasses::GNeighborFinder* pNF) { m_pNeighborFinder = pNF; }

	void randomOrder() { m_randomOrder = true; }

protected:
	double xval(double x);
	double yval(double y);
	void DrawMeshLines(DoubleRect* pSourceRect, GClasses::GMatrix* pData, GClasses::GImage* pImage, GClasses::GRect* pDestRect, unsigned int color);
	void ComputeChartRange();
	unsigned int GetLineColor(int line, size_t index);
};


#endif // __CHART_H__

