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

#include "GWave.h"
#include "GError.h"
#include "GMath.h"
#include <fstream>
#include <errno.h>
#include "math.h"
#include "GFourier.h"

#ifdef WINDOWS
#	pragma pack(1)
#endif

namespace GClasses {

struct WaveHeader
{
	// RIFF header
	unsigned char RIFF[4]; // "RIFF"
	unsigned int dwSize; // Size of data to follow
	unsigned char WAVE[4]; // "WAVE"

	// FMT sub-chunk
	unsigned char fmt_[4]; // "fmt "
	unsigned int dw16; // 16 (the size of the rest of the fmt sub-chunk)
	unsigned short wOne_0; // 1 (means PCM with no compression)
	unsigned short wChnls; // Number of Channels
	unsigned int dwSRate; // Sample Rate
	unsigned int BytesPerSec; // SampleRate * NumChannels * BitsPerSample / 8
	unsigned short wBlkAlign; // NumChannels * BitsPerSample/8
	unsigned short BitsPerSample; // Sample size

	// DATA sub-chunk
	unsigned char DATA[4]; // "DATA"
	unsigned int dwDSize; // Number of bytes of data
};

#ifdef WINDOWS
#	pragma pack()
#endif


GWave::GWave()
: m_sampleCount(0),
m_channels(0),
m_sampleRate(0),
m_bitsPerSample(0),
m_pData(NULL)
{
}

GWave::~GWave()
{
	delete[] m_pData;
}

void GWave::setData(unsigned char* pData, int bps, int sample_Count, int chans, int sample_Rate)
{
	delete[] m_pData;
	m_pData = pData;
	m_sampleCount = sample_Count;
	m_bitsPerSample = bps;
	m_channels = chans;
	m_sampleRate = sample_Rate;
}

void GWave::load(const char* szFilename)
{
	// Read in the wave header
	std::ifstream s;
	s.exceptions(std::ios::badbit | std::ios::failbit);
	try
	{
		s.open(szFilename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to open the file, ", szFilename, ". ", strerror(errno));
	}
	struct WaveHeader waveHeader;
	s.read((char*)&waveHeader, sizeof(struct WaveHeader));

	// Get stats from the wave header
	int size = waveHeader.dwDSize;
	m_channels = waveHeader.wChnls;
	m_sampleRate = waveHeader.dwSRate;
	m_bitsPerSample = waveHeader.BitsPerSample;
	m_sampleCount = size / (m_bitsPerSample / 8);

	// Read the wave data
	m_pData = new unsigned char[size];
	s.read((char*)m_pData, size);
}

void GWave::save(const char* szFilename)
{
	// Write the wave header
	int size = m_sampleCount * m_channels * m_bitsPerSample / 8;
	struct WaveHeader waveHeader;
	waveHeader.RIFF[0] = 'R';
	waveHeader.RIFF[1] = 'I';
	waveHeader.RIFF[2] = 'F';
	waveHeader.RIFF[3] = 'F';
	waveHeader.dwSize = sizeof(WaveHeader) - 8 + size;
	waveHeader.WAVE[0] = 'W';
	waveHeader.WAVE[1] = 'A';
	waveHeader.WAVE[2] = 'V';
	waveHeader.WAVE[3] = 'E';
	waveHeader.fmt_[0] = 'f';
	waveHeader.fmt_[1] = 'm';
	waveHeader.fmt_[2] = 't';
	waveHeader.fmt_[3] = ' ';
	waveHeader.dw16 = 16;
	waveHeader.wOne_0 = 1;
	waveHeader.wChnls = m_channels;
	waveHeader.dwSRate = m_sampleRate;
	waveHeader.BytesPerSec = m_sampleRate * m_channels * m_bitsPerSample / 8;
	waveHeader.wBlkAlign = m_channels * m_bitsPerSample / 8;
	waveHeader.BitsPerSample = m_bitsPerSample;
	waveHeader.DATA[0] = 'd';
	waveHeader.DATA[1] = 'a';
	waveHeader.DATA[2] = 't';
	waveHeader.DATA[3] = 'a';
	waveHeader.dwDSize = size;

	// Make the file
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
	os.write((const char*)&waveHeader, sizeof(struct WaveHeader));
	os.write((const char*)m_pData, size);
}




GWaveIterator::GWaveIterator(GWave& wave)
: m_wave(wave)
{
	m_pPos = wave.data();
	m_remaining = wave.sampleCount();
	m_pSamples = new double[wave.channels()];
}

GWaveIterator::~GWaveIterator()
{
	delete[] m_pSamples;
}

size_t GWaveIterator::remaining()
{
	return m_remaining;
}

bool GWaveIterator::advance(size_t steps)
{
	if(m_remaining < steps)
		return false;
	m_pPos += steps * m_wave.channels() * m_wave.bitsPerSample() / 8;
	m_remaining -= steps;
	return true;
}

void GWaveIterator::copy(GWaveIterator& other)
{
	GAssert(m_wave.data() == other.m_wave.data());
	m_pPos = other.m_pPos;
	m_remaining = other.m_remaining;
}

double* GWaveIterator::current()
{
	unsigned char* pX = m_pPos;
	double* pY = m_pSamples;
	for(unsigned short i = 0; i < m_wave.channels(); i++)
	{
		switch(m_wave.bitsPerSample())
		{
			case 8: *pY = (double(*pX) - 0x80) / 0x80; break; // 8-bit is unsigned
			case 16: *pY = double(*(short*)pX) / 0x8000; break; // 16-bit is signed
			case 32: *pY = double(*(int*)pX) / 0x80000000; break; // 32-bit is signed
		}
		pX += m_wave.bitsPerSample() / 8;
		pY++;
	}
	return m_pSamples;
}

void GWaveIterator::set(double* pSamples)
{
	unsigned char* pX = m_pPos;
	double* pY = pSamples;
	for(unsigned short i = 0; i < m_wave.channels(); i++)
	{
		switch(m_wave.bitsPerSample())
		{
			case 8: *pX = (unsigned char)std::max(0.0, std::min(255.0, floor((*pY + 1) * 0x80 + 0.5))); break;
			case 16: *(short*)pX = (short)std::max(-32768.0, std::min(32767.0, floor(*pY * 0x8000 + 0.5))); break;
			case 32: *(int*)pX = (int)std::max(-2147483648.0, std::min(2147483647.0, floor(*pY * 0x80000000 + 0.5))); break;
		}
		pX += m_wave.bitsPerSample() / 8;
		pY++;
	}
}







GFourierWaveProcessor::GFourierWaveProcessor(size_t blockSize)
: m_blockSize(blockSize)
{
	m_pBufA = new struct ComplexNumber[m_blockSize];
	m_pBufB = new struct ComplexNumber[m_blockSize];
	m_pBufC = new struct ComplexNumber[m_blockSize];
	m_pBufD = new struct ComplexNumber[m_blockSize];
	m_pBufE = new struct ComplexNumber[m_blockSize];
	m_pBufFinal = new struct ComplexNumber[m_blockSize];
}

// virtual
GFourierWaveProcessor::~GFourierWaveProcessor()
{
	delete[] m_pBufA;
	delete[] m_pBufB;
	delete[] m_pBufC;
	delete[] m_pBufD;
	delete[] m_pBufE;
	delete[] m_pBufFinal;
}

void GFourierWaveProcessor::reduce(GWave& signal)
{
	for(unsigned short chan = 0; chan < signal.channels(); chan++)
	{
		// Denoise the signal. Here is an ascii-art representation of how the blocks align.
		// Suppose the signal is 4 blocks long (n=4). i will iterate from 0 to 4.
		//                 _________ _________ _________ _________
		//  0   A    B    C    D    E
		//  1             A    B    C    D    E
		//  2                       A    B    C    D    E
		//  3                                 A    B    C    D    E
		//  4                                           A    B    C    D    E
		GWaveIterator itSignalIn(signal);
		GWaveIterator itSignalOut(signal);
		size_t n = itSignalIn.remaining() / m_blockSize;
		if((m_blockSize * n) < itSignalIn.remaining())
			n++;
		for(size_t i = 0; i <= n; i++)
		{
			// Encode block D (which also covers the latter-half of C and the first-half of E)
			if(i != n)
			{
				encodeBlock(itSignalIn, m_pBufD, chan);
				struct ComplexNumber* pSrc = m_pBufD;
				struct ComplexNumber* pDest = m_pBufC + m_blockSize / 2;
				for(size_t j = 0; j < m_blockSize / 2; j++)
					*(pDest++) = *(pSrc++);
				pDest = m_pBufE;
				for(size_t j = 0; j < m_blockSize / 2; j++)
					*(pDest++) = *(pSrc++);

				// Blocks C and D are fully-encoded, so we can bring them to the Fourier domain now
				if(i != 0)
					GFourier::fft(m_blockSize, m_pBufC, true);
				GFourier::fft(m_blockSize, m_pBufD, true);
			}

			// Process the blocks that are ready-to-go
			if(i != 0)
			{
				// Denoise blocks B and C
				process(m_pBufB);
				GFourier::fft(m_blockSize, m_pBufB, false);
				if(i != n)
				{
					process(m_pBufC);
					GFourier::fft(m_blockSize, m_pBufC, false);
				}

				// Interpolate A, B, and C to produce the final B
				interpolate(i == 1 ? NULL : m_pBufA, m_pBufB, i == n ? NULL : m_pBufC, m_blockSize / 2);
				decodeBlock(itSignalOut, chan);
			}

			// Shift A<-C, B<-D, C<-E
			struct ComplexNumber* pTemp = m_pBufA;
			m_pBufA = m_pBufC;
			m_pBufC = m_pBufE;
			m_pBufE = pTemp;
			pTemp = m_pBufB;
			m_pBufB = m_pBufD;
			m_pBufD = pTemp;
		}
		GAssert(itSignalOut.remaining() == 0);
	}
}

void GFourierWaveProcessor::encodeBlock(GWaveIterator& it, struct ComplexNumber* pBuf, unsigned short chan)
{
	struct ComplexNumber* pCN = pBuf;
	for(size_t i = 0; i < m_blockSize; i++)
	{
		if(it.remaining() > 0)
		{
			pCN->real = it.current()[chan];
			it.advance();
		}
		else
			pCN->real = 0.0;
		pCN->imag = 0.0;
		pCN++;
	}
}

void GFourierWaveProcessor::decodeBlock(GWaveIterator& it, unsigned short chan)
{
	struct ComplexNumber* pCN = m_pBufFinal;
	for(size_t i = 0; i < m_blockSize; i++)
	{
		if(it.remaining() > 0)
		{
			double* pSamples = it.current();
			pSamples[chan] = pCN->real;
			it.set(pSamples);
			it.advance();
		}
		else
			return;
		pCN++;
	}
}

void GFourierWaveProcessor::interpolate(struct ComplexNumber* pPre, struct ComplexNumber* pCur, struct ComplexNumber* pPost, size_t graftSamples)
{
	size_t graftBegin = (m_blockSize / 2 - graftSamples) / 2;
	size_t graftEnd = (m_blockSize / 2 + graftSamples) / 2;
	struct ComplexNumber* pOut = m_pBufFinal;
	if(pPre)
	{
		pPre += m_blockSize / 2;
		for(size_t i = 0; i < m_blockSize / 2; i++)
		{
			double w = i < graftBegin ? 1.0 : i > graftEnd ? 0.0 : 0.5 * (cos((double)(i - graftBegin) * M_PI / graftSamples) + 1);
			pOut->real = (1.0 - w) * pCur->real + w * pPre->real;
			pOut++;
			pPre++;
			pCur++;
		}
	}
	else
	{
		for(size_t i = 0; i < m_blockSize / 2; i++)
		{
			pOut->real = pCur->real;
			pOut++;
			pCur++;
		}
	}
	graftBegin += m_blockSize / 2;
	graftEnd += m_blockSize / 2;
	if(pPost)
	{
		for(size_t i = m_blockSize / 2; i < m_blockSize; i++)
		{
			double w = i < graftBegin ? 1.0 : i > graftEnd ? 0.0 : 0.5 * (cos((double)(i - graftBegin) * M_PI / graftSamples) + 1);
			pOut->real = (1.0 - w) * pPost->real + w * pCur->real;
			pOut++;
			pPost++;
			pCur++;
		}
	}
	else
	{
		for(size_t i = m_blockSize / 2; i < m_blockSize; i++)
		{
			pOut->real = pCur->real;
			pOut++;
			pCur++;
		}
	}
}


} // namespace GClasses

