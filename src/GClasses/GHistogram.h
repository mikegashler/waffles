/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GHISTOGRAM_H__
#define __GHISTOGRAM_H__

namespace GClasses {

/// Gathers values and puts them in bins.
class GHistogram
{
protected:
	double* m_bins;
	double m_sum, m_min, m_max;
	
	int m_binCount;

public:
	/// This creates a histogram normalized to sum to 1. If values
	/// are added that fall outside of the range, they are counted
	/// toward the summing to 1, but they don't show up in any bin
	/// (so they basically make all the bins a little bit smaller).
	GHistogram(double min, double max, int binCount);
	~GHistogram();

	/// Adds another sample to the histogram.
	void addSample(double value, double weight);
	
	/// Returns the number of bins in the histogram
	int binCount();

	/// Returns the total amount of value in the specified bin
	double binValue(int n);
	
	/// Returns the relative likelihood of the specified bin
	double binLiklihood(int n);

	/// Dumps to a file. You can then plot itwith GnuPlot or something similar
	void toFile(const char* filename);

	/// Returns the difference between the max and min values in the histogram
	double computeRange();
};

} // namespace GClasses

#endif // __GHISTOGRAM_H__
