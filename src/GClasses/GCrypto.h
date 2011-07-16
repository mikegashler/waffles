/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __CRYPTO_H__
#define __CRYPTO_H__

#include "GError.h"

namespace GClasses {

class GRand;
class Mangler;

/// This is a symmetric-key block-cypher. It utilizes a 2048-byte
/// internal state which is initialized using the passphrase. It uses
/// repeated applications of sha-512 to advance the internal state, and
/// to generate an 1024-byte pad that it xor's with your data to
/// encrypt or decrypt it. Warning: You use this algorithm at your own
/// risk. Many encryption algorithms eventually turn out to be insecure,
/// and to my knowledge, this algorithm has not yet been extensively
/// scrutinized.
class GCrypto
{
private:
	Mangler* m_pMangler;
	GRand* m_pRand;
	unsigned char* m_pBuf;
	bool m_encrypt;
	size_t m_blockPos;

public:
	GCrypto(const char* pPassphrase, size_t nPassphraseSize);
	~GCrypto();

	void doChunk(char* pData, size_t nDataSize);

#ifndef NO_TEST_CODE
	static void test();
#endif
};

}

#endif // __CRYPTO_H__
