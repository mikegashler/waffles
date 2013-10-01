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

#ifndef __GHISTOGRAM_H__
#define __GHISTOGRAM_H__

#include "GMatrix.h"

namespace GClasses {

/// Gathers values and puts them in bins.
class GHistogram
{
protected:
	double* m_bins;
	double m_sum, m_min, m_max;
	size_t m_binCount;

public:
	/// This creates an empty histogram. You will need to call addSample
	/// To fill it with values.
	GHistogram(double min, double max, size_t binCount);

	/// Creates a histogram and fills it with values in the specified column of data.
	/// If xmin and/or xmax are UNKNOWN_REAL_VALUE, then it will determine a suitable
	/// range automatically. The number of buckets will be computed as
	/// min(maxBuckets,floor(sqrt(data.rows))).
	GHistogram(GMatrix& data, size_t col, double xmin = UNKNOWN_REAL_VALUE, double xmax = UNKNOWN_REAL_VALUE, size_t maxBuckets = 10000000);
	~GHistogram();

	/// Adds a sample to the histogram.
	void addSample(double x, double weight = 1.0);

	/// Returns the number of bins in the histogram
	size_t binCount();

	/// Returns the center (median) x-value represented by the specified binsum value in the specified bin
	double binToX(size_t n);

	/// Returns the bin into which the specified value would fall. returns INVALID_INDEX if
	/// the value falls outside all of the bins.
	size_t xToBin(double x);

	/// Returns the probability that a value falls in the specified bin
	double binProbability(size_t n);

	/// Returns the relative likelihood of the specified bin. (This is typically
	/// plotted as the height, or y-value for the specified bin.)
	double binLikelihood(size_t n);

	/// Returns the index of the bin with the largest sum
	size_t modeBin();

	/// Dumps to a file. You can then plot itwith GnuPlot or something similar
	void toFile(const char* filename);

	/// Returns the difference between the max and min likelihood values in the histogram
	double computeRange();

	/// Returns the minimum x value that will fall into one of the buckets
	double xmin() { return m_min; }

	/// Returns the maximum x value. That is, the value that will fall just beyond that greatest bucket
	double xmax() { return m_max; }
};

} // namespace GClasses

#endif // __GHISTOGRAM_H__
