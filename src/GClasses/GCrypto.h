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
	unsigned char* m_pBuf;
	size_t m_blockPos;

public:
	GCrypto(const char* pPassphrase, size_t nPassphraseSize);
	~GCrypto();

	void doChunk(char* pData, size_t nDataSize);

	static void test();
};

}

#endif // __CRYPTO_H__
