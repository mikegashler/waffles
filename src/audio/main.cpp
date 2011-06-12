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

class AmbientNoiseReducer : public GFourierWaveProcessor
{
protected:
	double m_deviations;
	size_t m_noiseBlocks;
	struct ComplexNumber* m_pNoise;

public:
	AmbientNoiseReducer(GWave& noise, size_t blockSize, double deviations)
	: GFourierWaveProcessor(blockSize), m_deviations(deviations)
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
			ThrowError("Unrecognized option: ", args.pop_string());
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
			ThrowError("Unrecognized option: ", args.pop_string());
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

void sanitize(GArgReader& args)
{
	const char* inFilename = args.pop_string();
	const char* outFilename = args.pop_string();

	// Parse params
	double seconds = 0.1;
	double thresh = 0.15;
	while(args.next_is_flag())
	{
		if(args.if_pop("-seconds"))
			seconds = args.pop_double();
		else if(args.if_pop("-thresh"))
			thresh = args.pop_double();
		else
			ThrowError("Unrecognized option: ", args.pop_string());
	}

	// Sanitize
	GWave w;
	w.load(inFilename);
	size_t samps = (size_t)(seconds * w.sampleRate());
	for(unsigned short chan = 0; chan < w.channels(); chan++)
	{
		GWaveIterator itHead(w);
		GWaveIterator itTail(w);
		while(true)
		{
			if(itHead.remaining() > 0 && std::abs(itHead.current()[chan]) < thresh)
				itHead.advance();
			else
			{
				if(itTail.remaining() - itHead.remaining() >= 2 * samps)
				{
					for(size_t i = 0; i < samps; i++)
					{
						double* pSamp = itTail.current();
						pSamp[chan] *= 0.5 * (1.0 + cos((double)i * M_PI / samps));
						itTail.set(pSamp);
						itTail.advance();
					}
					while(itTail.remaining() - itHead.remaining() > samps)
					{
						double* pSamp = itTail.current();
						pSamp[chan] = 0.0;
						itTail.set(pSamp);
						itTail.advance();
					}
					for(size_t i = 0; i < samps; i++)
					{
						double* pSamp = itTail.current();
						pSamp[chan] *= 0.5 * (1.0 - cos((double)i * M_PI / samps));
						itTail.set(pSamp);
						itTail.advance();
					}
				}
				if(itHead.remaining() == 0)
					break;
				itHead.advance();
				itTail.copy(itHead);
			}
		}
	}
	w.save(outFilename);
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
		else if(args.if_pop("sanitize")) sanitize(args);
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
