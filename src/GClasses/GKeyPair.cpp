/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
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

#include <stdio.h>
#include "GKeyPair.h"
#include "GBigInt.h"
#include "GError.h"
#include "GRand.h"
#include "GDom.h"
#include <time.h>

namespace GClasses {

GKeyPair::GKeyPair()
{
	m_pPrivateKey = NULL;
	m_pPublicKey = NULL;
	m_pN = NULL;
}

GKeyPair::GKeyPair(GDomNode* pNode)
{
	m_pN = new GBigInt(pNode->get("n"));
	m_pPublicKey = new GBigInt(pNode->get("public"));
	GDomNode* pPrivate = pNode->getIfExists("private");
	if(pPrivate)
		m_pPrivateKey = new GBigInt(pPrivate);
	else
		m_pPrivateKey = NULL;
}

GKeyPair::~GKeyPair()
{
	if(m_pPrivateKey)
		m_pPrivateKey->setToZero();
	if(m_pPublicKey)
		m_pPublicKey->setToZero();
	if(m_pN)
		m_pN->setToZero();
	delete(m_pPrivateKey);
	delete(m_pPublicKey);
	delete(m_pN);
}

void GKeyPair::setPublicKey(GBigInt* pPublicKey)
{
	delete(m_pPublicKey);
	m_pPublicKey = pPublicKey;
}

void GKeyPair::setPrivateKey(GBigInt* pPrivateKey)
{
	delete(m_pPrivateKey);
	m_pPrivateKey = pPrivateKey;
}

void GKeyPair::setN(GBigInt* pN)
{
	delete(m_pN);
	m_pN = pN;
}

void GKeyPair::copyPublicKey(GBigInt* pPublicKey)
{
	delete(m_pPublicKey);
	m_pPublicKey = new GBigInt();
	m_pPublicKey->copy(pPublicKey);
}

void GKeyPair::copyPrivateKey(GBigInt* pPrivateKey)
{
	delete(m_pPrivateKey);
	m_pPrivateKey = new GBigInt();
	m_pPrivateKey->copy(pPrivateKey);
}

void GKeyPair::copyN(GBigInt* pN)
{
	delete(m_pN);
	m_pN = new GBigInt();
	m_pN->copy(pN);
}

GBigInt* GKeyPair::publicKey()
{
	return m_pPublicKey;
}

GBigInt* GKeyPair::privateKey()
{
	return m_pPrivateKey;
}

GBigInt* GKeyPair::n()
{
	return m_pN;
}

void GKeyPair::generateKeyPair(unsigned int uintCount, const unsigned int* pRawCryptographicBytes1, const unsigned int* pRawCryptographicBytes2, const unsigned int* pRawCryptographicBytes3)
{
	// Make places to put the data
	GBigInt* pOutPublicKey = new GBigInt();
	GBigInt* pOutPrivateKey = new GBigInt();
	GBigInt* pOutN = new GBigInt();

	// Find two primes
	GBigInt p;
	GBigInt q;
	int i;
	for(i = (int)uintCount - 1; i >= 0; i--)
		p.setUInt(i, pRawCryptographicBytes1[i]);
	for(i = uintCount - 1; i >= 0; i--)
		q.setUInt(i, pRawCryptographicBytes2[i]);
	p.setBit(0, true);
	q.setBit(0, true);
	int nTries = 0;
	while(!p.isPrime())
	{
		p.increment();
		p.increment();
		nTries++;
	}
	nTries = 0;
	while(!q.isPrime())
	{
		q.increment();
		q.increment();
		nTries++;
	}

	// Calculate N (the product of the two primes)
	pOutN->multiply(&p, &q);

	// Calculate prod ((p - 1) * (q - 1))
	p.decrement();
	q.decrement();
	GBigInt prod;
	prod.multiply(&p, &q);

	// Calculate public and private keys
	pOutPublicKey->selectPublicKey(pRawCryptographicBytes3, uintCount, &prod);
	pOutPrivateKey->calculatePrivateKey(pOutPublicKey, &prod);

	// Fill in "this" GKeyPair object
	setPublicKey(pOutPublicKey);
	setPrivateKey(pOutPrivateKey);
	setN(pOutN);
}

GDomNode* GKeyPair::serialize(GDom* pDoc, bool bIncludePrivateKey)
{
	GDomNode* pNode = pDoc->newObj();
	if(!n() || !publicKey())
		throw Ex("No key has been made yet");
	if(bIncludePrivateKey && !privateKey())
		throw Ex("This key-pair doesn't include the private key");
	pNode->add(pDoc, "n", n()->serialize(pDoc));
	pNode->add(pDoc, "public", publicKey()->serialize(pDoc));
	if(bIncludePrivateKey)
		pNode->add(pDoc, "private", privateKey()->serialize(pDoc));
	return pNode;
}

int GKeyPair::maxBlockSize()
{
	return (m_pN->getBitCount() - 1) / 8;
}

unsigned char* GKeyPair::powerMod(const unsigned char* pInput, int nInputSize, bool bPublicKey, int* pnOutputSize)
{
	GBigInt input;
	input.fromByteBuffer(pInput, nInputSize);
	GBigInt results;
	results.powerMod(&input, bPublicKey ? publicKey() : privateKey(), n());
	*pnOutputSize = results.getUIntCount() * sizeof(unsigned int);
	unsigned char* pOutput = (unsigned char*)results.toBufferGiveOwnership();
	while(pOutput[(*pnOutputSize) - 1] == 0)
		(*pnOutputSize)--;
	return pOutput;
}

/*static*/ void GKeyPair::test()
{
	GRand prng(0);
	unsigned int buf[6];
	for(int i = 0; i < 6; i++)
		buf[i] = (unsigned int)prng.next();
	GKeyPair kp;
	kp.generateKeyPair(2, buf, buf + 2, buf + 4);

	// Make up a message
	GBigInt message;
	message.setUInt(0, 0x6a54);

	// Encrypt it
	GBigInt cypher;
	cypher.powerMod(&message, kp.privateKey(), kp.n());

	// Decrypt it
	GBigInt final;
	final.powerMod(&cypher, kp.publicKey(), kp.n());

	// Check the final value
	if(final.compareTo(&message) != 0)
		throw Ex("failed");
}

} // namespace GClasses
