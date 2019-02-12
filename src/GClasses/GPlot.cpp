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
#include <fstream>

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



std::string svg_to_str(double d)
{
	std::ostringstream os;
	os.setf(std::ios::fixed);
	//os.precision(6);
	os << d;
	return os.str();
}


#define BOGUS_XMIN -1e308
const char* g_hexChars = "0123456789abcdef";

GSVG::GSVG(size_t width, size_t height, size_t hWindows, size_t vWindows)
: m_width(width), m_height(height), m_hWindows(hWindows), m_vWindows(vWindows), m_xmin(BOGUS_XMIN), m_clipping(false)
{
	m_ss << "<?xml version=\"1.0\"?><svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"";
	m_ss << to_str(width) << "\" height=\"" << to_str(height) << "\">\n";
}

void GSVG::closeTags()
{
	// Close the current clipping group
	if(m_clipping)
	{
		m_ss << "</g>";
		m_clipping = false;
	}

	// Close the current chart
	if(m_xmin != BOGUS_XMIN)
		m_ss << "</g>";
	m_ss << "\n\n";
}

void GSVG::newChart(double xmin, double ymin, double xmax, double ymax, size_t hPos, size_t vPos, double margin)
{
	closeTags();
	m_hPos = hPos;
	m_vPos = vPos;
	double chartWidth = (double)m_width / m_hWindows;
	double chartHeight = (double)m_height / m_vWindows;
	margin = std::min(margin, 0.75 * chartWidth);
	margin = std::min(margin, 0.75 * chartHeight);
	m_hunit = ((xmax - xmin)) / (chartWidth - margin);
	m_vunit = ((ymax - ymin)) / (chartHeight - margin);
	m_margin = margin;
	m_xmin = xmin;
	m_ymin = ymin;
	m_xmax = xmax;
	m_ymax = ymax;
	m_ss << "<defs><clipPath id=\"chart" << to_str(hPos) << "-" << to_str(vPos) << "\"><rect x=\"" << svg_to_str(xmin) << "\" y=\"" << svg_to_str(ymin) << "\" width=\"" << svg_to_str(xmax - xmin) << "\" height=\"" << svg_to_str(ymax - ymin) << "\" /></clipPath></defs>\n";
	m_ss << "<g transform=\"translate(" << svg_to_str(chartWidth * hPos + margin) << " "
		<< svg_to_str(chartHeight * (vPos + 1) - margin) << ") scale(" << svg_to_str((chartWidth - margin) / (xmax - xmin)) <<
		" " << svg_to_str(-(chartHeight - margin) / (ymax - ymin)) << ") translate(" << svg_to_str(-xmin) << " " << svg_to_str(-ymin) << ")\""
		<< ">\n";
}

GSVG::~GSVG()
{
}

void GSVG::clip()
{
	m_ss << "\n<!-- Clipped region -->\n";
	m_ss << "<g clip-path=\"url(#chart" << to_str(m_hPos) << "-" << to_str(m_vPos) << ")\">\n";
	m_clipping = true;
}

void GSVG::color(unsigned int c)
{
	m_ss << '#' << g_hexChars[(c >> 20) & 0xf] << g_hexChars[(c >> 16) & 0xf];
	m_ss << g_hexChars[(c >> 12) & 0xf] << g_hexChars[(c >> 8) & 0xf];
	m_ss << g_hexChars[(c >> 4) & 0xf] << g_hexChars[c & 0xf];
}

void GSVG::add_raw(const char* string)
{
	m_ss << string;
}

void GSVG::dot(double x, double y, double r, unsigned int col)
{
	m_ss << "<ellipse cx=\"" << svg_to_str(x) << "\" cy=\"" << svg_to_str(y) << "\" rx=\"" << svg_to_str(r * 4 * m_hunit) << "\" ry=\"" << svg_to_str(r * 4 * m_vunit) << "\" fill=\"";
	color(col);
	m_ss << "\" />\n";
}

void GSVG::line(double x1, double y1, double x2, double y2, double thickness, unsigned int col)
{
	m_ss << "<line x1=\"" << svg_to_str(x1) << "\" y1=\"" << svg_to_str(y1) << "\" x2=\"" << svg_to_str(x2) << "\" y2=\"" << svg_to_str(y2) << "\" style=\"stroke:";
	color(col);
	double l = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	double w = thickness * (std::abs(x2 - x1) * m_vunit + std::abs(y2 - y1) * m_hunit) / std::max(0.000001, l);
	m_ss << ";stroke-width:" << svg_to_str(w) << "\"/>\n";
}

void GSVG::rect(double x, double y, double w, double h, unsigned int col)
{
	m_ss << "<rect x=\"" << svg_to_str(x) << "\" y=\"" << svg_to_str(y) << "\" width=\"" << svg_to_str(w) << "\" height=\"" << svg_to_str(h) << "\" style=\"fill:";
	color(col);
	m_ss << "\"/>\n";
}

void GSVG::arc(double cx, double cy, double r, double astart, double aend, double thickness, unsigned int col)
{
	bool bigarc = false;
	if(aend > astart)
	{
		if(aend - astart > M_PI)
			bigarc = true;
	}
	else
	{
		if(aend + 2 * M_PI - astart > M_PI)
			bigarc = true;
	}
	double sx = cx + r * cos(astart);
	double sy = cy + r * sin(astart);
	double fx = cx + r * cos(aend);
	double fy = cy + r * sin(aend);
	m_ss << "<path d=\"M " << svg_to_str(sx) << " " << svg_to_str(sy) << " A ";
	m_ss << svg_to_str(r) << " " << svg_to_str(r) << " 0 " << (bigarc ? "1" : "0") << " 1 ";
	m_ss << svg_to_str(fx) << " " << svg_to_str(fy) << "\" stroke=\"";
	color(col);
	m_ss << "\" fill=\"none\" stroke-width=\"" << svg_to_str(thickness) << "\" />";
}

void GSVG::text(double x, double y, const char* szText, double size, Anchor eAnchor, unsigned int col, double angle, bool serifs)
{
	double xx = x / (m_hunit * size);
	double yy = -y / (m_vunit * size);
	m_ss << "<text x=\"" << svg_to_str(xx) << "\" y=\"" << svg_to_str(yy) << "\" style=\"fill:";
	color(col);
	if(!serifs)
		m_ss << ";font-family:Sans";
	m_ss << "\" transform=\"";
	m_ss << "scale(" << svg_to_str(size * m_hunit) << " " << svg_to_str(-size * m_vunit) << ")";
	if(angle != 0.0)
		m_ss << " rotate(" << svg_to_str(-angle) << " " << svg_to_str(xx) << " " << svg_to_str(yy) << ")";
	m_ss << "\"";
	if(eAnchor == Middle)
		m_ss << " text-anchor=\"middle\"";
	else if(eAnchor == End)
		m_ss << " text-anchor=\"end\"";
	m_ss << ">" << szText << "</text>\n";
}

void GSVG::print(std::ostream& stream)
{
	closeTags();

	// Close the whole SVG file
	m_ss << "</svg>\n";

	// Print it
	stream << m_ss.str();
}

void GSVG::save(const char* szFilename)
{
	std::ofstream os;
	os.exceptions(std::ios::badbit | std::ios::failbit);
	try
	{
		os.open(szFilename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to create the file, ", szFilename, ". ", strerror(errno));
	}
	print(os);
}

double GSVG::horizLabelPos()
{
	return m_ymin - m_vunit * ((m_margin / 2));
}

double GSVG::vertLabelPos()
{
	return m_xmin - m_hunit * ((m_margin / 2));
}

void GSVG::horizMarks(int maxLabels, bool notext, std::vector<std::string>* pLabels)
{
	m_ss << "\n<!-- Horiz labels -->\n";
	if(maxLabels >= 0)
	{
		GPlotLabelSpacer spacer(m_xmin, m_xmax, maxLabels);
		int count = spacer.count();
		for(int i = 0; i < count; i++)
		{
			double x = spacer.label(i);
			line(x, m_ymin, x, m_ymax, 0.2, (x == 0.0 ? 0x000000 : 0xc0c0c0));
			if(!notext)
			{
				if(pLabels)
				{
					if(pLabels->size() > (size_t)i)
						text(x + 3 * m_hunit, m_ymin - m_vunit, (*pLabels)[i].c_str(), 1, End, 0x000000, 90);
				}
				else
					text(x + 3 * m_hunit, m_ymin - m_vunit, to_str(x).c_str(), 1, End, 0x000000, 90);
			}
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
			line(log(x), m_ymin, log(x), m_ymax, 0.2, (x == 0.0 ? 0x000000 : 0xc0c0c0));
			if(primary && !notext)
				text(log(x) + 3 * m_hunit, m_ymin - m_vunit, to_str(x).c_str(), 1, End, 0x000000, 90);
		}
	}
	m_ss << "\n";
}

void GSVG::vertMarks(int maxLabels, bool notext, std::vector<std::string>* pLabels)
{
	m_ss << "\n<!-- Vert labels -->\n";
	if(maxLabels >= 0)
	{
		GPlotLabelSpacer spacer(m_ymin, m_ymax, maxLabels);
		int count = spacer.count();
		for(int i = 0; i < count; i++)
		{
			double y = spacer.label(i);
			line(m_xmin, y, m_xmax, y, 0.2, (y == 0.0 ? 0x000000 : 0xc0c0c0));
			if(!notext)
			{
				if(pLabels)
				{
					if(pLabels->size() > (size_t)i)
						text(m_xmin - m_hunit, y - 3 * m_vunit, (*pLabels)[i].c_str(), 1, End, 0x000000);
				}
				else
					text(m_xmin - m_hunit, y - 3 * m_vunit, to_str(y).c_str(), 1, End, 0x000000);
			}
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
			line(m_xmin, log(y), m_xmax, log(y), 0.2, (y == 0.0 ? 0x000000 : 0xc0c0c0));
			if(primary && !notext)
				text(m_xmin - m_hunit, log(y) - 3 * m_vunit, to_str(y).c_str(), 1, End, 0x000000);
		}
	}
	m_ss << "\n";
}

} // namespace GClasses

