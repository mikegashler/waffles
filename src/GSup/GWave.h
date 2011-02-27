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

namespace GClasses {

/// Currently only supports PCM wave format
class GWave
{
protected:
	unsigned int m_size;
	unsigned int m_sampleCount;
	unsigned short m_channels;
	unsigned int m_sampleRate;
	unsigned short m_bitsPerSample;
	unsigned char* m_pData;

public:
	GWave();
	~GWave();

	void load(const char* szFilename);
	void save(const char* szFilename);
	unsigned char* data() { return m_pData; }
	void setMetaData(int channels, int sampleRate);
	void setData(unsigned char* pData, int bitsPerSample, int sampleCount);
	int sampleCount() { return m_sampleCount; }
	int bitsPerSample() { return m_bitsPerSample; }
	int sampleRate() { return m_sampleRate; }
};

} // namespace GClasses

#endif // __GWAVE_H__
