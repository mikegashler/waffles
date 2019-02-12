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

#ifndef __GPLOT_H__
#define __GPLOT_H__

#include "GRect.h"
#include "GMath.h"
#include <vector>
#include <string>

namespace GClasses {

class GImage;
class GRand;


/// If you need to place grid lines or labels at regular intervals
/// (like 1000, 2000, 3000, 4000... or 20, 25, 30, 35... or 0, 2, 4, 6, 8, 10...)
/// this class will help you pick where to place the labels so that
/// there are a reasonable number of them, and they all land on nice label
/// values.
class GPlotLabelSpacer
{
protected:
	double m_spacing;
	int m_start;
	int m_count;

public:
	/// maxLabels specifies the maximum number of labels that it can ever
	/// decide to use. (It should be just smaller than the number of labels
	/// that would make the graph look too crowded.)
	GPlotLabelSpacer(double min, double max, int maxLabels);

	/// Returns the number of labels that have been picked. It will be a value
	/// smaller than maxLabels.
	int count();

	/// Returns the location of the n'th label (where 0 <= n < count())
	double label(int index);
};


/// Similar to GPlotLabelSpacer, except for logarithmic grids. To plot in
/// logarithmic space, set your plot window to have a range from log_e(min)
/// to log_e(max). When you actually plot things, plot them at log_e(x), where
/// x is the position of the thing you want to plot.
class GPlotLabelSpacerLogarithmic
{
protected:
	double m_max;
	int m_n, m_i;

public:
	/// Pass in the log (base e) of your min and max values. (We make you
	/// pass them in logarithmic form, so you can't use a negative min value.)
	GPlotLabelSpacerLogarithmic(double log_e_min, double log_e_max);

	/// Returns true and sets *pos to the position of the next label.
	/// (You should actually plot it at log_e(*pos) in your plot window.)
	/// Returns false if there are no more (and doesn't set *pos).
	/// primary is set to true if the label is the primary
	/// label for the new scale.
	bool next(double* pos, bool* primary);
};


/// This class makes it easy to plot points and functions on 2D cartesian coordinates.
class GPlotWindow
{
protected:
	GImage* m_pImage;
	GDoubleRect m_window;
	int m_w, m_h;

public:
	/// pImage is the image onto which you wish to plot
	GPlotWindow(GImage* pImage, double xmin, double ymin, double xmax, double ymax);
	~GPlotWindow();

	/// Convert from window (Euclidean space) coordinates to view (image) coordinates
	inline void windowToView(double x, double y, float* pX, float* pY)
	{
		*pX = (float)((x - m_window.x) / m_window.w * m_w);
		*pY = m_h - 1 - (float)((y - m_window.y) / m_window.h * m_h);
	}

	/// Convert from window (Euclidean space) coordinates to view (image) coordinates
	inline void windowToView(double x, double y, int* pX, int* pY)
	{
		*pX = (int)floor((x - m_window.x) / m_window.w * m_w + 0.5);
		*pY = m_h - 1 - (int)floor((y - m_window.y) / m_window.h * m_h + 0.5);
	}

	/// Convert from view (image) coordinates to window (Euclidean space) coordinates
	inline void viewToWindow(int x, int y, double* pX, double* pY)
	{
		*pX = (double)x * m_window.w / m_w + m_window.x;
		*pY = (double)(m_h - 1 - y) * m_window.h / m_h + m_window.y;
	}

	/// Returns the width represented by each pixel
	double pixelWidth() const { return m_window.w / m_w; }

	/// Returns the height represented by each pixel
	double pixelHeight() const { return m_window.h / m_h; }

	/// Returns the image that was passed in to the constructor
	GImage* image() { return m_pImage; }

	/// Returns the rect of the Euclidean space that this image represents
	GDoubleRect* window() { return &m_window; }

	/// Plots a single pixel. (Note that for most applications, PlotDot is a better choice because it
	/// draws a larger dot centered at the sub-pixel location that you specify.)
	void point(double x, double y, unsigned int col);

	/// Plots a dot at the specified location. You must specify both the foreground and background
	/// color so that it can make the dot appear to be centered at the precise sub-pixel location
	/// specified. radius is specified in pixels.
	void dot(double x, double y, float radius, unsigned int colFore, unsigned int colBack);

	/// Plots a line
	void line(double x1, double y1, double x2, double y2, unsigned int col);

	/// Plots a fat line
	void fatLine(double x1, double y1, double x2, double y2, float thickness, unsigned int col);

	/// Plots a function
	void function(MathFunc pFunc, unsigned int col, void* pThis);

	/// Draws a label at the specified location. (A size of 1.0f will be small but legible.)
	void label(double x, double y, const char* szLabel, float size, unsigned int col);

	/// Draws an arrow from (x1, y1) to (x2, y2). headSize is specified in pixels.
	void arrow(double x1, double y1, double x2, double y2, unsigned int col, int headSize);

	/// Draw grid lines
	/// If maxHorizAxisLabels is 0, no grid lines will be drawn for the horizontal axis.
	/// If maxHorizAxisLabels is -1, a logarithmic scale will be used for the horizontal axis.
	/// Same with maxVertAxisLabels.
	void gridLines(int maxHorizAxisLabels, int maxVertAxisLabels, unsigned int col);

	/// Copy the image onto a larger image, and label the axes on the larger image.
	/// If maxHorizAxisLabels is 0, no labels will be drawn for the horizontal axis.
	/// If maxHorizAxisLabels is -1, a logarithmic scale will be used for the horizontal axis.
	/// Same with maxVertAxisLabels.
	GImage* labelAxes(int maxHorizAxisLabels, int maxVertAxisLabels, int precision, float size, unsigned int color, double angle);

protected:
	static void stringLabel(GImage* pImage, const char* szText, int x, int y, float size, unsigned int color, double angle);
	static void numericLabel(GImage* pImage, double value, int x, int y, int precision, float size, unsigned int color, double angle);
};




/// This class simplifies plotting data to an SVG file
class GSVG
{
public:
	enum Anchor
	{
		Start,
		Middle,
		End,
	};

protected:
	std::stringstream m_ss;
	size_t m_width, m_height, m_hWindows, m_vWindows, m_hPos, m_vPos;
	double m_hunit, m_vunit, m_margin;
	double m_xmin, m_ymin, m_xmax, m_ymax;
	bool m_clipping;

public:
	/// This object represents a hWindows-by-vWindows grid of charts. Typically, hWindows and vWindows
	/// are 1, so this is just a single chart, but you may specify larger values to produce a grid of charts.
	/// width and height specify the width and height of the entire grid of charts. An equal portion of
	/// the width and height is given to each chart in the grid.
	GSVG(size_t width, size_t height, size_t hWindows = 1, size_t vWindows = 1);
	~GSVG();

	/// Start a new chart. This method must be called before any drawing operations are performed.
	/// xmin, ymin, xmax, and ymax specify the coordinates in the chart to begin drawing.
	/// If this contains more than a 1x1 grid of charts, then you may call this method again to begin
	/// drawing a different chart in the grid. hPos and vPos specify which chart in the grid to begin drawing.
	void newChart(double xmin, double ymin, double xmax, double ymax, size_t hPos = 0, size_t vPos = 0, double margin = 100);

	/// Returns (xmax - xmin) / width, which is often a useful size.
	double hunit() { return m_hunit; }

	/// Returns (ymax - ymin) / height, which is often a useful size.
	double vunit() { return m_vunit; }

	/// Writes some raw SVG code to the stream
	void add_raw(const char* string);

	/// Draw a dot
	void dot(double x, double y, double r = 1.0, unsigned int col = 0x000080);

	/// Draw a line
	void line(double x1, double y1, double x2, double y2, double thickness = 1.0, unsigned int col = 0x008000);

	/// Draw a rectangle
	void rect(double x, double y, double w, double h, unsigned int col = 0x008080);

	/// Draw an arc
	void arc(double cx, double cy, double r, double astart, double aend, double thickness, unsigned int col = 0x800080);

	/// Draw text
	void text(double x, double y, const char* szText, double size = 1.0, Anchor eAnchor = Start, unsigned int col = 0x000000, double angle = 0.0, bool serifs = true);

	/// Generate an SVG file with all of the components that have been added so far.
	void print(std::ostream& stream);

	/// Write to an SVG file
	void save(const char* szFilename);
    
	/// Label the horizontal axis. If maxLabels is 0, then no grid-lines will be drawn. If maxLabels is -1, then
	/// Logarithmic grid-lines will be drawn. If pLabels is non-NULL, then its values will be used to label
	/// the grid-lines instead of the continuous values.
	void horizMarks(int maxLabels, bool notext = false, std::vector<std::string>* pLabels = NULL);

	/// Label the vertical axis. If maxLabels is 0, then no grid-lines will be drawn. If maxLabels is -1, then
	/// Logarithmic grid-lines will be drawn. If pLabels is non-NULL, then its values will be used to label
	/// the grid-lines instead of the continuous values.
	void vertMarks(int maxLabels, bool notext = false, std::vector<std::string>* pLabels = NULL);

	/// Returns a good y position for the horizontal axis label
	double horizLabelPos();

	/// Returns a good x position for the vertical axis label
	double vertLabelPos();

	/// After calling this method, all draw operations will be clipped to fall within (xmin, ymin)-(xmax, ymax),
	/// until a new chart is started.
	void clip();

protected:
	void color(unsigned int c);
	void closeTags();
};


} // namespace GClasses

#endif // __GPLOT_H__
