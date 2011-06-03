/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GWAVE_H__
#define __GWAVE_H__

#include "../GClasses/GError.h"

namespace GClasses {

/// Currently only supports PCM wave format
class GWave
{
public:
	enum WaveType
	{
		wt_signed = 0,
		wt_unsigned,
		wt_float,
	};

protected:
	unsigned int m_size;
	unsigned int m_sampleCount;
	unsigned short m_channels;
	unsigned int m_sampleRate;
	unsigned short m_bitsPerSample;
	WaveType m_type;
	unsigned char* m_pData;

public:
	GWave();
	~GWave();

	void load(const char* szFilename);
	void save(const char* szFilename);
	unsigned char* data() { return m_pData; }
	void setMetaData(int channels, int sampleRate);
	void setData(unsigned char* pData, int bitsPerSample, int sampleCount, WaveType type);
	int sampleCount() { return m_sampleCount; }
	int bitsPerSample() { return m_bitsPerSample; }
	int sampleRate() { return m_sampleRate; }
	unsigned short channels() { return m_channels; }
	WaveType type() { return m_type; }
};


/// This class iterates over the samples in a WAVE file.
/// Regardless of the bits-per-sample, this iterator
/// will convert all samples to doubles with a range from -1 to 1.
class GWaveIterator
{
protected:
	GWave& m_wave;
	size_t m_remaining;
	double* m_pSamples;
	unsigned char* m_pPos;

public:
	GWaveIterator(GWave& wave);
	~GWaveIterator();

	/// Returns the number of samples remaining
	size_t remaining();

	/// Advances to the next sample. Returns false if it
	/// reaches the end of the samples. Returns true otherwise.
	bool advance();

	/// Returns a pointer to a c-dimensional array of doubles,
	/// where c is the number of channels. Each element is
	/// one of the sample values for a channel from -1 to 1.
	double* current();
};

} // namespace GClasses

#endif // __GWAVE_H__
