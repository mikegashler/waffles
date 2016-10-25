/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
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

#include "GHistogram.h"
#include "GError.h"
#include <math.h>
#include <fstream>
#include <errno.h>
#include <string.h>
#include "GVec.h"

using namespace GClasses;

GHistogram::GHistogram(double xMin, double xMax, size_t bins)
{
	m_min = xMin;
	m_max = xMax;
	m_binCount = bins;
	m_bins = new double[bins];
	for(size_t i = 0; i < bins; i++)
		m_bins[i] = 0.0;
	m_sum = 0.0;
}

GHistogram::GHistogram(GMatrix& data, size_t col, double xMin, double xMax, size_t maxBuckets)
{
	double dataMin = data.columnMin(col);
	double dataRange = data.columnMax(col) - dataMin;
	double mean = data.columnMean(col);
	double median = data.columnMedian(col);
	double dev = sqrt(data.columnVariance(col, mean));
	if(xMin == UNKNOWN_REAL_VALUE)
		m_min = std::max(dataMin, median - 4 * dev);
	else
		m_min = xMin;
	if(xMax == UNKNOWN_REAL_VALUE)
		m_max = std::min(dataMin + dataRange, median + 4 * dev);
	else
		m_max = xMax;
	m_binCount = std::max((size_t)1, std::min(maxBuckets, (size_t)floor(sqrt((double)data.rows()))));
	m_bins = new double[m_binCount];
	for(size_t i = 0; i < m_binCount; i++)
		m_bins[i] = 0.0;
	m_sum = 0.0;

	for(size_t i = 0; i < data.rows(); i++)
		addSample(data[i][col], 1.0);
}

GHistogram::~GHistogram()
{
	delete[] m_bins;
}

void GHistogram::addSample(double x, double weight)
{
	size_t bin = (size_t)floor((x - m_min) * m_binCount / (m_max - m_min));
	if(bin < m_binCount)
		m_bins[bin] += weight;
	m_sum += weight;
}

size_t GHistogram::binCount()
{
	return m_binCount;
}

double GHistogram::binToX(size_t n)
{
	return ((double)n + 0.5) * (m_max - m_min) / m_binCount + m_min;
}

size_t GHistogram::xToBin(double x)
{
	size_t bin = (size_t)floor((x - m_min) * m_binCount / (m_max - m_min));
	if(bin < m_binCount)
		return bin;
	else
		return INVALID_INDEX;
}

double GHistogram::binLikelihood(size_t n)
{
	if(m_max - m_min == 0.0 || m_sum == 0.0)
		return 1.0 / m_binCount;
	return m_bins[n] * m_binCount / ((m_max - m_min) * m_sum);
}

double GHistogram::binProbability(size_t n)
{
	return m_bins[n] / m_sum;
}

size_t GHistogram::modeBin()
{
	size_t mode = 0;
	double modeSum = m_bins[0];
	for(size_t i = 1; i < m_binCount; i++)
	{
		if(m_bins[i] > modeSum)
		{
			modeSum = m_bins[i];
			mode = i;
		}
	}
	return mode;
}

void GHistogram::toFile(const char* filename)
{
	std::ofstream os;
	os.exceptions(std::ios::badbit | std::ios::failbit);
	try
	{
		os.open(filename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to create the file, ", filename, ". ", strerror(errno));
	}
	os.precision(5);
	for(size_t i = 0; i < m_binCount; i++)
		os << binToX(i) << " " << binLikelihood(i) << "\n";
}

double GHistogram::computeRange()
{
	double min = binLikelihood(0);
	double max = min;
	double d;
	for(size_t i = 1; i < m_binCount; i++)
	{
		d = binLikelihood(i);
		if(d < min)
			min = d;
		if(d > max)
			max = d;
	}
	return max - min;
}
