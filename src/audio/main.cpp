// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <exception>
#include <iostream>
#include <GClasses/GApp.h>
#include <GClasses/GBits.h>
#include <GClasses/GError.h>
#include <GClasses/GFile.h>
#include <GClasses/GHolders.h>
#include <GClasses/GFourier.h>
#include <GClasses/GMath.h>
#include <GClasses/GVec.h>
#include <GClasses/GWave.h>
#include <cmath>
#include "../wizard/usage.h"

using namespace GClasses;
using std::cerr;
using std::cout;

// This is an abstract class that processes a wave file in blocks. Specifically, it
// divides the wave file up into overlapping blocks, converts them into Fourier space,
// calls the abstract "process" method with each block, converts back from Fourier space,
// and then interpolates to create the wave output.
class FourierWaveProcessor
{
protected:
	size_t m_blockSize;
	struct ComplexNumber* m_pBufA;
	struct ComplexNumber* m_pBufB;
	struct ComplexNumber* m_pBufC;
	struct ComplexNumber* m_pBufD;
	struct ComplexNumber* m_pBufE;
	struct ComplexNumber* m_pBufFinal;

public:
	FourierWaveProcessor(size_t blockSize)
	: m_blockSize(blockSize)
	{
		m_pBufA = new struct ComplexNumber[m_blockSize];
		m_pBufB = new struct ComplexNumber[m_blockSize];
		m_pBufC = new struct ComplexNumber[m_blockSize];
		m_pBufD = new struct ComplexNumber[m_blockSize];
		m_pBufE = new struct ComplexNumber[m_blockSize];
		m_pBufFinal = new struct ComplexNumber[m_blockSize];
	}

	~FourierWaveProcessor()
	{
		delete[] m_pBufA;
		delete[] m_pBufB;
		delete[] m_pBufC;
		delete[] m_pBufD;
		delete[] m_pBufE;
		delete[] m_pBufFinal;
	}

	void doit(GWave& signal)
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

protected:
	void encodeBlock(GWaveIterator& it, struct ComplexNumber* pBuf, unsigned short chan)
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

	void decodeBlock(GWaveIterator& it, unsigned short chan)
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

	void interpolate(struct ComplexNumber* pPre, struct ComplexNumber* pCur, struct ComplexNumber* pPost, size_t graftSamples)
	{
		size_t graftBegin = (m_blockSize / 2 - graftSamples) / 2;
		size_t graftEnd = (m_blockSize / 2 + graftSamples) / 2;
		struct ComplexNumber* pOut = m_pBufFinal;
		if(pPre)
		{
			pPre += m_blockSize / 2;
			for(size_t i = 0; i < m_blockSize / 2; i++)
			{
				double w = i < graftBegin ? 1.0 : i > graftEnd ? 0.0 : 0.5 * (cos((i - graftBegin) * M_PI / graftSamples) + 1);
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
				double w = i < graftBegin ? 1.0 : i > graftEnd ? 0.0 : 0.5 * (cos((i - graftBegin) * M_PI / graftSamples) + 1);
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

	virtual void process(struct ComplexNumber* pBuf) = 0;
};

class AmbientNoiseReducer : public FourierWaveProcessor
{
protected:
	double m_deviations;
	size_t m_noiseBlocks;
	struct ComplexNumber* m_pNoise;

public:
	AmbientNoiseReducer(GWave& noise, size_t blockSize, double deviations)
	: FourierWaveProcessor(blockSize), m_deviations(deviations)
	{
		if(noise.channels() != 1)
			ThrowError("Sorry, ", to_str(noise.channels()), "-channel ambient noise files are not yet supported");
		m_pNoise = new struct ComplexNumber[m_blockSize];
		m_noiseBlocks = 0;

		// Analyze the noise
		GWaveIterator itNoise(noise);
		if(itNoise.remaining() < m_blockSize * 8)
			ThrowError("Not enough noise to analyze");
		for(size_t i = 0; i < m_blockSize; i++)
			itNoise.advance(); // skip a little bit from the beginning
		while(itNoise.remaining() > m_blockSize * 2)
		{
			encodeBlock(itNoise, m_pBufA, 0);
			GFourier::fft(m_blockSize, m_pBufA, true);
			analyzeNoise(m_pBufA);
		}
	}

	virtual ~AmbientNoiseReducer()
	{
		delete[] m_pNoise;
	}

protected:
	void analyzeNoise(struct ComplexNumber* pBuf)
	{
		if(m_noiseBlocks == 0)
		{
			struct ComplexNumber* pDest = m_pNoise;
			for(size_t i = 0; i < m_blockSize; i++)
			{
				pDest->real = 0.0;
				pDest->imag = 0.0;
				pDest++;
			}
		}
		m_noiseBlocks++;
		double b = 1.0 / m_noiseBlocks;
		double a = 1.0 - b;
		struct ComplexNumber* pSrc = pBuf;
		struct ComplexNumber* pDest = m_pNoise;
		for(size_t i = 0; i < m_blockSize; i++)
		{
			double mag = sqrt(pSrc->squaredMagnitude());
			pDest->real *= a;
			pDest->imag *= a;
			pDest->real += b * mag;
			pDest->imag += b * mag * mag;
			pSrc++;
			pDest++;
		}
	}

	virtual void process(struct ComplexNumber* pBuf)
	{
		if(m_noiseBlocks < 2)
			ThrowError("Not enough noise blocks to estimate a deviation");
		struct ComplexNumber* pNoise = m_pNoise;
		struct ComplexNumber* pSignal = pBuf;
		for(size_t i = 0; i < m_blockSize; i++)
		{
			double noiseDev = sqrt((double)m_noiseBlocks * (pNoise->imag - (pNoise->real * pNoise->real)) / (m_noiseBlocks - 1));
			double noiseMag = pNoise->real + m_deviations * noiseDev;
			double signalMag = sqrt(pSignal->squaredMagnitude());
			double scalar = signalMag > 1e-9 ? std::max(0.0, signalMag - noiseMag) / signalMag : 0.0;
			pSignal->real *= scalar;
			pSignal->imag *= scalar;
			pNoise++;
			pSignal++;
		}
	}
};

void amplify(GArgReader& args)
{
	const char* inputFilename = args.pop_string();
	double scale = args.pop_double();
	const char* outputFilename = args.pop_string();
	GWave w;
	w.load(inputFilename);
	GWaveIterator it(w);
	size_t samples = it.remaining() * w.channels();
	size_t clipped = 0;
	for( ; it.remaining() > 0; it.advance())
	{
		double* pSample = it.current();
		for(unsigned short c = 0; c < w.channels(); c++)
		{
			pSample[c] *= scale;
			if(std::abs(pSample[c]) > 1.0)
				clipped++;
		}
		it.set(pSample);
	}
	w.save(outputFilename);
	if(clipped > 0)
		cout << "Warning: " << to_str((double)clipped * 100 / samples) << "% of samples were clipped!\n";
}

void makeSilence(GArgReader& args)
{
	double seconds = args.pop_double();
	const char* filename = args.pop_string();

	// Parse params
	int bitsPerSample = 16;
	int channels = 1;
	int sampleRate = 44100;
	while(args.next_is_flag())
	{
		if(args.if_pop("-bitspersample"))
			bitsPerSample = args.pop_uint();
		else if(args.if_pop("-channels"))
			channels = args.pop_uint();
		else if(args.if_pop("-samplerate"))
			sampleRate = args.pop_uint();
		else
			break;
	}
	if(bitsPerSample % 8 != 0)
		ThrowError("The number of bits-per-sample must be a multiple of 8");

	// Generate the silence
	size_t samples = (size_t)(sampleRate * seconds);
	GWave w;
	unsigned char* pData = new unsigned char[channels * samples * bitsPerSample];
	w.setData(pData, bitsPerSample, samples, channels, sampleRate);
	for(GWaveIterator it(w); it.remaining() > 0; it.advance())
	{
		double* pSample = it.current();
		GVec::setAll(pSample, 0.0, channels);
		it.set(pSample);
	}
	w.save(filename);
}

void mix(GArgReader& args)
{
	const char* input1 = args.pop_string();
	double scale1 = args.pop_double();
	const char* input2 = args.pop_string();
	double scale2 = args.pop_double();
	const char* outputFilename = args.pop_string();
	GWave w1;
	w1.load(input1);
	GWave w2;
	w2.load(input2);
	if(w1.channels() != w2.channels())
		ThrowError("Mismatching number of channels");
	if(w1.sampleRate() != w2.sampleRate())
		ThrowError("Mismatching sample rates");
	GWave* pW1 = &w1;
	GWave* pW2 = &w2;
	if(pW1->sampleCount() < pW2->sampleCount())
	{
		std::swap(pW1, pW2);
		std::swap(scale1, scale2);
	}
	GWaveIterator it1(*pW1);
	GWaveIterator it2(*pW2);
	size_t samples = it1.remaining() * pW1->channels();
	size_t clipped = 0;
	for( ; it1.remaining() > 0; it1.advance())
	{
		double* pSamp1 = it1.current();
		if(it2.remaining() > 0)
		{
			double* pSamp2 = it2.current();
			for(unsigned short c = 0; c < w1.channels(); c++)
			{
				pSamp1[c] *= scale1;
				pSamp1[c] += scale2 * pSamp2[c];
				if(std::abs(pSamp1[c]) > 1.0)
					clipped++;
			}
		}
		else
		{
			for(unsigned short c = 0; c < w1.channels(); c++)
			{
				pSamp1[c] *= scale1;
				if(std::abs(pSamp1[c]) > 1.0)
					clipped++;
			}
		}
		it1.set(pSamp1);
		it2.advance();
	}
	pW1->save(outputFilename);
	if(clipped > 0)
		cout << "Warning: " << to_str((double)clipped * 100 / samples) << "% of samples were clipped!\n";
}

void reduceAmbientNoise(GArgReader& args)
{
	const char* noiseFilename = args.pop_string();
	const char* signalFilename = args.pop_string();
	const char* outputFilename = args.pop_string();

	// Parse params
	size_t blockSize = 2048;
	double deviations = 2.5;
	while(args.next_is_flag())
	{
		if(args.if_pop("-blocksize"))
			blockSize = args.pop_uint();
		else if(args.if_pop("-deviations"))
			deviations = args.pop_double();
		else
			break;
	}
	if(!GBits::isPowerOfTwo(blockSize))
		ThrowError("the block size must be a power of 2");

	GWave wNoise;
	wNoise.load(noiseFilename);
	GWave wSignal;
	wSignal.load(signalFilename);
	AmbientNoiseReducer denoiser(wNoise, blockSize, deviations);
	denoiser.doit(wSignal);
	wSignal.save(outputFilename);
}

void ShowUsage(const char* appName)
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeAudioUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(cout, 0, 3, 76, 1000, true);
	cout.flush();
}

void showError(GArgReader& args, const char* szAppName, const char* szMessage)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	args.set_pos(1);
	const char* szCommand = args.peek();
	UsageNode* pUsageTree = makeAudioUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << szAppName << " ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, true);
		}
		else
		{
			cerr << "Brief Usage Information:\n\n";
			pUsageTree->print(cerr, 0, 3, 76, 1, false);
		}
	}
	else
	{
		pUsageTree->print(cerr, 0, 3, 76, 1, false);
		cerr << "\nFor more specific usage information, enter as much of the command as you know.\n";
	}
	cerr << "\nTo see full usage information, run:\n	" << szAppName << " usage\n\n";
	cerr << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cerr.flush();
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int ret = 0;
	PathData pd;
	GFile::parsePath(argv[0], &pd);
	const char* appName = argv[0] + pd.fileStart;
	GArgReader args(argc, argv);
	args.pop_string(); // advance past the app name
	try
	{
		if(args.size() < 1) ThrowError("Expected a command");
		else if(args.if_pop("usage")) ShowUsage(appName);
		else if(args.if_pop("amplify")) amplify(args);
		else if(args.if_pop("makesilence")) makeSilence(args);
		else if(args.if_pop("mix")) mix(args);
		else if(args.if_pop("reduceambientnoise")) reduceAmbientNoise(args);
		else ThrowError("Unrecognized command: ", args.peek());
	}
	catch(const std::exception& e)
	{
		if(strcmp(e.what(), "nevermind") != 0) // if an error message was not already displayed...
			showError(args, appName, e.what());
		ret = 1;
	}

	return ret;
}
