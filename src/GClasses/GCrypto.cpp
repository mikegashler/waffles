/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GCrypto.h"
#include "GError.h"
#include "GHolders.h"
#include "GRand.h"
#include "sha1.h"
#include "sha2.h"
#include <algorithm>
#include <iostream>

using std::cout;

namespace GClasses {

#define DIGEST_BYTES 1024

void Xor_buffers(unsigned char* pDest, const unsigned char* pSource, size_t nBytes)
{
	while(nBytes > 0)
	{
		*pDest++ ^= *pSource++;
		nBytes--;
	}
}

void Taint_buffer(unsigned char* pDest, const unsigned char* pSource, size_t nBytes, size_t round)
{
	while(nBytes > 0)
	{
		*pDest++ ^= ((1 << (round & 7)) & *pSource++);
		nBytes--;
		round++;
	}
}

class Mangler
{
protected:
	GRand m_rand;
	unsigned char m_pHash[DIGEST_BYTES];
	unsigned char m_pBuf[DIGEST_BYTES];
	unsigned char* m_pPos;
	unsigned char* m_pTaint;
	size_t m_remaining;
	
public:
	Mangler()
	: m_rand(0)
	{
		reset(NULL);
	}

	void resetSeed()
	{
		m_rand.setSeed(0);
	}

	void reset(unsigned char* pTaint)
	{
		wipe(m_pHash, DIGEST_BYTES);
		m_pPos = m_pBuf;
		m_remaining = DIGEST_BYTES;
		m_pTaint = pTaint;
	}

	void purge()
	{
		resetSeed();
		reset(NULL);
		wipe(m_pBuf, DIGEST_BYTES);
	}

	void add(unsigned char* pBuf, size_t len)
	{
		while(len >= m_remaining)
		{
			memcpy(m_pPos, pBuf, m_remaining);
			pBuf += m_remaining;
			len -= m_remaining;
			digestBuf();
			m_pPos = m_pBuf;
			m_remaining = DIGEST_BYTES;
		}
		memcpy(m_pPos, pBuf, len);
		m_pPos += len;
		m_remaining -= len;
	}

	unsigned char* getHash()
	{
		if(m_remaining < DIGEST_BYTES)
		{
			while(m_remaining > 0)
			{
				*m_pPos = (unsigned char)m_rand.next();
				m_pPos++;
				m_remaining--;
			}
			digestBuf();
			m_remaining = DIGEST_BYTES;
		}
		return m_pHash;
	}

	void shuffleBytes(unsigned char* pBuf, size_t len)
	{
		for(size_t i = len; i > 1; i--)
			std::swap(pBuf[(size_t)m_rand.next(i)], pBuf[i - 1]);
	}

	void wipe(unsigned char* pBuf, size_t len)
	{
		while(len > 0)
		{
			*(pBuf++) = (unsigned char)m_rand.next();
			len--;
		}
	}

protected:
	void digestBuf()
	{
		// Mangle the data
		for(size_t j = 0; j < 3; j++)
		{
			// Taint the bytes
			GAssert(DIGEST_BYTES >= 32);
			if(m_pTaint)
				Taint_buffer(m_pBuf, m_pTaint + (m_rand.next() % (DIGEST_BYTES - 32)), 32, j);

			// Shuffle the bytes
			shuffleBytes(m_pBuf, DIGEST_BYTES);

			// Hash each 64-bit chunk with sha-512
			unsigned char* pChunk = m_pBuf;
			for(size_t i = 0; i < (DIGEST_BYTES / 64); i++)
			{
				sha512_ctx ctx;
				sha512_begin(&ctx);
				sha512_hash(pChunk, 64, &ctx);
				sha512_end(pChunk, &ctx);
				pChunk += 64;
			}
		}

		// Add it to the hash
		uint64_t* pA = (uint64_t*)m_pBuf;
		uint64_t* pB = (uint64_t*)m_pHash;
		for(size_t i = 0; i < (DIGEST_BYTES / sizeof(uint64_t)); i++)
		{
			(*pB) += (*pA);
			pB++;
			pA++;
		}
	}
};




GCrypto::GCrypto(const char* pPassphrase, size_t nPassphraseSize)
: m_blockPos(DIGEST_BYTES)
{
	m_pMangler = new Mangler();
	m_pBuf = new unsigned char[3 * DIGEST_BYTES];
	m_pMangler->wipe(m_pBuf, 2 * DIGEST_BYTES);
	size_t size = 2 * DIGEST_BYTES;
	unsigned char* pBuf = m_pBuf;
	while(size > 0)
	{
		size_t chunk = std::min(size, nPassphraseSize);
		Xor_buffers(pBuf, (const unsigned char*)pPassphrase, chunk);
		pBuf += chunk;
		size -= chunk;
	}
}

GCrypto::~GCrypto()
{
	m_pMangler->purge();
	m_pMangler->wipe(m_pBuf, 3 * DIGEST_BYTES);
	delete[] m_pBuf;
	delete(m_pMangler);
}

void GCrypto::doChunk(char* pData, size_t nDataSize)
{
	while(nDataSize > 0)
	{
		if(m_blockPos >= DIGEST_BYTES)
		{
			// Compute encryption pad
			m_blockPos = 0;
			m_pMangler->reset(NULL);
			m_pMangler->add(m_pBuf, 2 * DIGEST_BYTES);
			memcpy(m_pBuf + 2 * DIGEST_BYTES, m_pMangler->getHash(), DIGEST_BYTES);

			// Advance internal state
			m_pMangler->shuffleBytes(m_pBuf, 2 * DIGEST_BYTES);
			m_pMangler->reset(m_pBuf + 2 * DIGEST_BYTES);
			m_pMangler->add(m_pBuf, 2 * DIGEST_BYTES);
			memcpy(m_pBuf + DIGEST_BYTES, m_pMangler->getHash(), DIGEST_BYTES);
			Taint_buffer(m_pBuf, m_pBuf + 2 * DIGEST_BYTES, DIGEST_BYTES, 3);
		}
		size_t chunkSize = std::min((size_t)DIGEST_BYTES - m_blockPos, nDataSize);
		Xor_buffers((unsigned char*)pData, m_pBuf + 2 * DIGEST_BYTES + m_blockPos, chunkSize);
		m_blockPos += chunkSize;
		pData += chunkSize;
		nDataSize -= chunkSize;
	}
}

#ifndef NO_TEST_CODE
// static
void GCrypto::test()
{
	size_t testLen = DIGEST_BYTES * 5 / 2;
	char* pA = new char[3 * testLen];
	ArrayHolder<char> hA(pA);
	char* pB = pA + testLen;
	char* pC = pB + testLen;
	GRand r(1234);
	for(size_t i = 0; i < testLen; i++)
		pA[i] = (unsigned char)r.next();
	memcpy(pB, pA, testLen);
	{
		GCrypto c("password", 8);
		c.doChunk(pB, testLen);
		if(memcmp(pA, pB, testLen) == 0)
			ThrowError("Didn't do anything");
	}
	memcpy(pC, pB, testLen);
	{
		GCrypto c("passworx", 8);
		c.doChunk(pB, testLen);
		if(memcmp(pA, pB, testLen) == 0)
			ThrowError("This should have failed");
	}
	{
		GCrypto c("password", 8);
		c.doChunk(pC, testLen);
		if(memcmp(pA, pC, testLen) != 0)
			ThrowError("This should have worked");
	}
}
#endif // NO_TEST_CODE

}
