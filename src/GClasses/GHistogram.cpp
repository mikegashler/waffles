/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GHistogram.h"
#include "GError.h"
#include <math.h>
#include <fstream>

using namespace GClasses;

GHistogram::GHistogram(double min, double max, int binCount)
{
	m_min = min;
	m_max = max;
	m_binCount = binCount;
	m_bins = new double[binCount];
	int i;
	for(i = 0; i < binCount; i++)
		m_bins[i] = 0;
	m_sum = 0;
}

GHistogram::~GHistogram()
{
	delete[] m_bins;
}

void GHistogram::addSample(double value, double weight)
{
	int bin = (int)floor((value - m_min) * m_binCount / (m_max - m_min));
	if(bin >= 0 && bin < m_binCount)
		m_bins[bin] += weight;
	m_sum += weight;
}

int GHistogram::binCount()
{
	return m_binCount;
}

double GHistogram::binValue(int n)
{
	return ((double)n + 0.5) * (m_max - m_min) / m_binCount + m_min;
}

double GHistogram::binLiklihood(int n)
{
	return m_bins[n] * m_binCount / ((m_max - m_min) * m_sum);
}

void GHistogram::toFile(const char* filename)
{
	std::ofstream os;
	os.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		os.open(filename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		ThrowError("Error creating file: ", filename);
	}
	os.precision(5);
	for(int i = 0; i < m_binCount; i++)
		os << binValue(i) << " " << binLiklihood(i) << "\n";
}

double GHistogram::computeRange()
{
	double min = binLiklihood(0);
	double max = min;
	double d;
	int i;
	for(i = 1; i < m_binCount; i++)
	{
		d = binLiklihood(i);
		if(d < min)
			min = d;
		if(d > max)
			max = d;
	}
	return max - min;
}
