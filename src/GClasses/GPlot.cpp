/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GPlot.h"
#include <stdlib.h>
#include "GError.h"
#include "GVec.h"
#include "GImage.h"
#include "GRand.h"
#include "GMath.h"
#include <string>
#include <sstream>
#include <cmath>

using std::string;
using std::ostringstream;

namespace GClasses {

GPlotLabelSpacer::GPlotLabelSpacer(double min, double max, int maxLabels)
{
	if(maxLabels == 0)
	{
		m_spacing = 0.0;
		m_start = 0;
		m_count = 0;
		return;
	}
	if(max <= min)
		throw Ex("invalid range");
	int p = (int)ceil(log((max - min) / maxLabels) * M_LOG10E);

	// Every 10
	m_spacing = pow(10.0, p);
	m_start = (int)ceil(min / m_spacing);
	m_count = (int)floor(max / m_spacing) - m_start + 1;

	if(m_count * 5 + 4 < maxLabels)
	{
		// Every 2
		m_spacing *= 0.2;
		m_start = (int)ceil(min / m_spacing);
		m_count = (int)floor(max / m_spacing) - m_start + 1;
	}
	else if(m_count * 2 + 1 < maxLabels)
	{
		// Every 5
		m_spacing *= 0.5;
		m_start = (int)ceil(min / m_spacing);
		m_count = (int)floor(max / m_spacing) - m_start + 1;
	}
}

int GPlotLabelSpacer::count()
{
	return m_count;
}

double GPlotLabelSpacer::label(int index)
{
	return (m_start + index) * m_spacing;
}








GPlotLabelSpacerLogarithmic::GPlotLabelSpacerLogarithmic(double log_e_min, double log_e_max)
{
	double min = exp(log_e_min);
	m_max = exp(std::min(500.0, log_e_max));
	m_n = (int)floor(log_e_min * M_LOG10E);
	m_i = 1;
	while(true)
	{
		double p = pow((double)10, m_n);
		if((m_i * p) >= min)
			break;
		m_i++;
		if(m_i >= 10)
		{
			m_i = 0;
			m_n++;
		}
	}
}

bool GPlotLabelSpacerLogarithmic::next(double* pos, bool* primary)
{
	double p = pow((double)10, m_n);
	*pos = p * m_i;
	if(*pos > m_max)
		return false;
	if(m_i == 1)
		*primary = true;
	else
		*primary = false;
	m_i++;
	if(m_i >= 10)
	{
		m_i = 0;
		m_n++;
	}
	return true;
}






GPlotWindow::GPlotWindow(GImage* pImage, double xmin, double ymin, double xmax, double ymax)
{
	if(xmin > -1e300 && xmax < 1e300 && ymin > -1e300 && ymax < 1e300)
	{
	}
	else
		throw Ex("Invalid range");
	if(xmin >= xmax)
		throw Ex("xmin is expected to be smaller then xmax");
	if(ymin >= ymax)
		throw Ex("ymin is expected to be smaller then ymax");
	m_pImage = pImage;
	m_window.set(xmin, ymin, xmax - xmin, ymax - ymin);
	if(m_window.w <= 0)
		throw Ex("xmax must be > xmin");
	if(m_window.h <= 0)
		throw Ex("ymax must be > ymin");
	m_w = pImage->width();
	m_h = pImage->height();
}

GPlotWindow::~GPlotWindow()
{
}

void GPlotWindow::point(double x, double y, unsigned int col)
{
	int x1, y1;
	windowToView(x, y, &x1, &y1);
	m_pImage->setPixelIfInRange(x1, y1, col);
}

void GPlotWindow::dot(double x, double y, float radius, unsigned int colFore, unsigned int colBack)
{
	float x1, y1;
	windowToView(x, y, &x1, &y1);
	m_pImage->dot(x1, y1, radius, colFore, colBack);
}

void GPlotWindow::line(double x1, double y1, double x2, double y2, unsigned int col)
{
	int xx1, yy1, xx2, yy2;
	windowToView(x1, y1, &xx1, &yy1);
	windowToView(x2, y2, &xx2, &yy2);
	m_pImage->line(xx1, yy1, xx2, yy2, col);
}

void GPlotWindow::fatLine(double x1, double y1, double x2, double y2, float thickness, unsigned int col)
{
	float xx1, yy1, xx2, yy2;
	windowToView(x1, y1, &xx1, &yy1);
	windowToView(x2, y2, &xx2, &yy2);
	m_pImage->fatLine(xx1, yy1, xx2, yy2, thickness, col);
}

void GPlotWindow::function(MathFunc pFunc, unsigned int col, void* pThis)
{
	double x, y, xPrev, yPrev;
	xPrev = m_window.x;
	yPrev = pFunc(pThis, xPrev);
	int i;
	for(i = 1; i < m_w; i++)
	{
		x = (double)i * m_window.w / m_w + m_window.x;
		y = pFunc(pThis, x);
		if(xPrev >= -1e200 && xPrev < 1e200 && yPrev >= -1e200 && yPrev < 1e200 &&
			x >= -1e200 && x < 1e200 && y >= -1e200 && y < 1e200)
		line(xPrev, yPrev, x, y, col);
		xPrev = x;
		yPrev = y;
	}
}

void GPlotWindow::label(double x, double y, const char* szLabel, float size, unsigned int col)
{
	int x1, y1;
	windowToView(x, y, &x1, &y1);
	m_pImage->text(szLabel, x1, y1, size, col, 1000, 1000);
}

void GPlotWindow::arrow(double x1, double y1, double x2, double y2, unsigned int col, int headSize)
{
	int xx1, yy1, xx2, yy2;
	windowToView(x1, y1, &xx1, &yy1);
	windowToView(x2, y2, &xx2, &yy2);
	m_pImage->arrow(xx1, yy1, xx2, yy2, col, headSize);
}

// static
void GPlotWindow::stringLabel(GImage* pImage, const char* szText, int x, int y, float size, unsigned int color, double angle)
{
	// Draw the label such that it ends at the center of the temp image
	int width = GImage::measureTextWidth(szText, size);
	int nSize = (int)(std::max((float)width, size * 12) * 2.3);
	GImage tmp;
	tmp.setSize(nSize, nSize);
	tmp.clear(0x0);
	tmp.text(szText, nSize / 2 - width, (int)((nSize - size * 12) / 2), size, color, 1000, 1000);

	// Rotate the label around the center
	GImage tmp2;
	tmp2.rotate(&tmp, nSize / 2, nSize / 2, angle);

	// Blit such that the label ends at the specified point
	GRect r(0, 0, nSize, nSize);
	pImage->blitAlpha(x - nSize / 2, y - nSize / 2, &tmp2, &r);
}

// static
void GPlotWindow::numericLabel(GImage* pImage, double value, int x, int y, int precision, float size, unsigned int color, double angle)
{
	std::ostringstream os;
	os.precision(precision);
	os << value;
	string s = os.str();
	stringLabel(pImage, s.c_str(), x, y, size, color, angle);
}

void GPlotWindow::gridLines(int maxHorizAxisLabels, int maxVertAxisLabels, unsigned int col)
{
	if(maxHorizAxisLabels > 0)
	{
		GPlotLabelSpacer spacer(m_window.x, m_window.x + m_window.w, maxHorizAxisLabels);
		for(int i = 0; i < spacer.count(); i++)
		{
			double pos = spacer.label(i);
			line(pos, m_window.y, pos, m_window.y + m_window.h, col);
		}
	}
	else if(maxHorizAxisLabels == -1)
	{
		GPlotLabelSpacerLogarithmic spacer(m_window.x, m_window.x + m_window.w);
		while(true)
		{
			double pos;
			bool primary;
			if(!spacer.next(&pos, &primary))
				break;
			double x = log(pos);
			line(x, m_window.y, x, m_window.y + m_window.h, col);
		}
	}
	if(maxVertAxisLabels > 0)
	{
		GPlotLabelSpacer spacer(m_window.y, m_window.y + m_window.h, maxVertAxisLabels);
		for(int i = 0; i < spacer.count(); i++)
		{
			double pos = spacer.label(i);
			line(m_window.x, pos, m_window.x + m_window.w, pos, col);
		}
	}
	else if(maxVertAxisLabels == -1)
	{
		GPlotLabelSpacerLogarithmic spacer(m_window.y, m_window.y + m_window.h);
		while(true)
		{
			double pos;
			bool primary;
			if(!spacer.next(&pos, &primary))
				break;
			double y = log(pos);
			line(m_window.x, y, m_window.x + m_window.w, y, col);
		}
	}
}

GImage* GPlotWindow::labelAxes(int maxHorizAxisLabels, int maxVertAxisLabels, int precision, float size, unsigned int color, double angle)
{
	int spacing = 10;
	int horizMargin = 200;
	int vertMargin = 200;
	GImage* pOutImage = new GImage();
	pOutImage->setSize(m_pImage->width() + horizMargin, m_pImage->height() + vertMargin);
	pOutImage->clear(0xffffffff);
	GRect r(0, 0, m_pImage->width(), m_pImage->height());
	pOutImage->blit(horizMargin, 0, m_pImage, &r);
	if(maxHorizAxisLabels > 0)
	{
		GPlotLabelSpacer spacer(m_window.x, m_window.x + m_window.w, maxHorizAxisLabels);
		for(int i = 0; i < spacer.count(); i++)
		{
			double pos = spacer.label(i);
			int x1, y1;
			windowToView(pos, 0, &x1, &y1);
			numericLabel(pOutImage, pos, horizMargin + x1, m_pImage->height() + spacing, precision, size, color, angle);
		}
	}
	else if(maxHorizAxisLabels == -1)
	{
		GPlotLabelSpacerLogarithmic spacer(m_window.x, m_window.x + m_window.w);
		while(true)
		{
			double pos;
			bool primary;
			if(!spacer.next(&pos, &primary))
				break;
			if(primary)
			{
				double x = log(pos);
				int x1, y1;
				windowToView(x, 0, &x1, &y1);
				numericLabel(pOutImage, pos, horizMargin + x1, m_pImage->height() + spacing, precision, size, color, angle);
			}
		}
	}
	if(maxVertAxisLabels > 0)
	{
		GPlotLabelSpacer spacer(m_window.y, m_window.y + m_window.h, maxVertAxisLabels);
		for(int i = 0; i < spacer.count(); i++)
		{
			double pos = spacer.label(i);
			int x1, y1;
			windowToView(0, pos, &x1, &y1);
			numericLabel(pOutImage, pos, horizMargin - spacing, y1, precision, size, color, 0.0);
		}
	}
	else if(maxVertAxisLabels == -1)
	{
		GPlotLabelSpacerLogarithmic spacer(m_window.y, m_window.y + m_window.h);
		while(true)
		{
			double pos;
			bool primary;
			if(!spacer.next(&pos, &primary))
				break;
			if(primary)
			{
				double y = log(pos);
				int x1, y1;
				windowToView(0, y, &x1, &y1);
				numericLabel(pOutImage, pos, horizMargin - spacing, y1, precision, size, color, 0.0);
			}
		}
	}
	return pOutImage;
}







const char* g_hexChars = "0123456789abcdef";

GSVG::GSVG(int width, int height, double xmin, double ymin, double xmax, double ymax, double margin)
: m_hunit((xmax - xmin) / width), m_vunit((ymax - ymin) / height), m_margin(margin), m_xmin(xmin), m_ymin(ymin), m_xmax(xmax), m_ymax(ymax)
{
	m_ss << "<?xml version=\"1.0\"?><svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"";
	m_ss << to_str(width + m_margin) << "\" height=\"" << to_str(height + m_margin) << "\">\n";
	m_ss << "<g transform=\"translate(" << to_str(m_margin) << " " << to_str(height) << ") scale(" << to_str((double)width / (xmax - xmin)) <<
		" " << to_str(-(double)height / (ymax - ymin)) << ") translate(" << to_str(-xmin) << " " << to_str(-ymin) << ") \">\n";
}

GSVG::~GSVG()
{
}

void GSVG::color(unsigned int c)
{
	m_ss << '#' << g_hexChars[(c >> 20) & 0xf] << g_hexChars[(c >> 16) & 0xf];
	m_ss << g_hexChars[(c >> 12) & 0xf] << g_hexChars[(c >> 8) & 0xf];
	m_ss << g_hexChars[(c >> 4) & 0xf] << g_hexChars[c & 0xf];
}

void GSVG::dot(double x, double y, double r, unsigned int col)
{
	m_ss << "<ellipse cx=\"" << to_str(x) << "\" cy=\"" << to_str(y) << "\" rx=\"" << to_str(r * 4 * m_hunit) << "\" ry=\"" << to_str(r * 4 * m_vunit) << "\" fill=\"";
	color(col);
	m_ss << "\" />\n";
}

void GSVG::line(double x1, double y1, double x2, double y2, double thickness, unsigned int col)
{
	m_ss << "<line x1=\"" << to_str(x1) << "\" y1=\"" << to_str(y1) << "\" x2=\"" << to_str(x2) << "\" y2=\"" << to_str(y2) << "\" style=\"stroke:";
	color(col);
	double l = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	double w = thickness * (std::abs(x2 - x1) * m_vunit + std::abs(y2 - y1) * m_hunit) / l;
	m_ss << ";stroke-width:" << to_str(w) << "\"/>\n";
}

void GSVG::rect(double x, double y, double w, double h, unsigned int col)
{
	m_ss << "<rect x=\"" << to_str(x) << "\" y=\"" << to_str(y) << "\" width=\"" << to_str(w) << "\" height=\"" << to_str(h) << "\" style=\"fill:";
	color(col);
	m_ss << "\"/>\n";
}

void GSVG::text(double x, double y, const char* szText, double size, Anchor eAnchor, unsigned int col, double angle, bool serifs)
{
	double xx = x / (m_hunit * size);
	double yy = -y / (m_vunit * size);
	m_ss << "<text x=\"" << to_str(xx) << "\" y=\"" << to_str(yy) << "\" style=\"fill:";
	color(col);
	if(!serifs)
		m_ss << ";font-family:Sans";
	m_ss << "\" transform=\"";
	m_ss << "scale(" << to_str(size * m_hunit) << " " << to_str(-size * m_vunit) << ")";
	if(angle != 0.0)
		m_ss << " rotate(" << to_str(-angle) << " " << to_str(xx) << " " << to_str(yy) << ")";
	m_ss << "\"";
	if(eAnchor == Middle)
		m_ss << " text-anchor=\"middle\"";
	else if(eAnchor == End)
		m_ss << " text-anchor=\"end\"";
	m_ss << ">" << szText << "</text>\n";
}

void GSVG::print(std::ostream& stream)
{
	m_ss << "</g></svg>\n";
	stream << m_ss.str();
}

void GSVG::horizLabels(const char* szAxisLabel, int maxLabels, std::vector<std::string>* pLabels)
{
	m_ss << "\n<!-- Horiz labels -->\n";
	if(strlen(szAxisLabel) > 0)
		text((m_xmin + m_xmax) / 2, m_ymin - m_vunit * ((m_margin / 2) + 10), szAxisLabel, 1.5, Middle, 0x000000, 0.0);
	if(maxLabels >= 0)
	{
		GPlotLabelSpacer spacer(m_xmin, m_xmax, maxLabels);
		int count = spacer.count();
		for(int i = 0; i < count; i++)
		{
			double x = spacer.label(i);
			line(x, m_ymin, x, m_ymax, 0.2, 0xa0a0a0);
			if(pLabels)
			{
				if(pLabels->size() > (size_t)i)
					text(x + 3 * m_hunit, m_ymin - m_vunit, (*pLabels)[i].c_str(), 1, End, 0x000000, 90);
			}
			else
				text(x + 3 * m_hunit, m_ymin - m_vunit, to_str(x).c_str(), 1, End, 0x000000, 90);
		}
	}
	else
	{
		GPlotLabelSpacerLogarithmic spacer(m_xmin, m_xmax);
		double x;
		bool primary;
		while(true)
		{
			if(!spacer.next(&x, &primary))
				break;
			line(log(x), m_ymin, log(x), m_ymax, 0.2, 0xa0a0a0);
			if(primary)
				text(log(x) + 3 * m_hunit, m_ymin - m_vunit, to_str(x).c_str(), 1, End, 0x000000, 90);
		}
	}
	m_ss << "\n";
}

void GSVG::vertLabels(const char* szAxisLabel, int maxLabels, std::vector<std::string>* pLabels)
{
	m_ss << "\n<!-- Vert labels -->\n";
	if(strlen(szAxisLabel) > 0)
		text(m_xmin - m_hunit * ((m_margin / 2) - 10), (m_ymax - m_ymin) / 2, szAxisLabel, 1.5, Middle, 0x000000, 90.0);
	if(maxLabels >= 0)
	{
		GPlotLabelSpacer spacer(m_ymin, m_ymax, maxLabels);
		int count = spacer.count();
		for(int i = 0; i < count; i++)
		{
			double y = spacer.label(i);
			line(m_xmin, y, m_xmax, y, 0.2, 0xa0a0a0);
			if(pLabels)
			{
				if(pLabels->size() > (size_t)i)
					text(m_xmin - m_hunit, y - 3 * m_vunit, (*pLabels)[i].c_str(), 1, End, 0x000000);
			}
			else
				text(m_xmin - m_hunit, y - 3 * m_vunit, to_str(y).c_str(), 1, End, 0x000000);
		}
	}
	else
	{
		GPlotLabelSpacerLogarithmic spacer(m_xmin, m_xmax);
		double y;
		bool primary;
		while(true)
		{
			if(!spacer.next(&y, &primary))
				break;
			line(m_xmin, log(y), m_xmax, log(y), 0.2, 0xa0a0a0);
			if(primary)
				text(m_xmin - m_hunit, log(y) - 3 * m_vunit, to_str(y).c_str(), 1, End, 0x000000);
		}
	}
	m_ss << "\n";
}

} // namespace GClasses

