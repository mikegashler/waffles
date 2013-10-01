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

#include "GBits.h"
#include "GRand.h"

using namespace GClasses;

//static
bool GBits::isValidFloat(const char* pString, size_t len)
{
	if(len == 0)
		return false;
	if(*pString == '-' || *pString == '+')
	{
		pString++;
		len--;
		if(len == 0)
			return false;
	}
	int digits = 0;
	int decimals = 0;
	while(len > 0)
	{
		if(*pString == '.')
			decimals++;
		else if(*pString >= '0' && *pString <= '9')
			digits++;
		else
			break;
		pString++;
		len--;
	}
	if(decimals > 1)
		return false;
	if(digits < 1)
		return false;
	if(len > 0 && (*pString == 'e' || *pString == 'E'))
	{
		pString++;
		len--;
		if(len == 0)
			return false;
		if(*pString == '-' || *pString == '+')
		{
			pString++;
			len--;
		}
		if(len == 0)
			return false;
		while(*pString >= '0' && *pString <= '9')
		{
			pString++;
			len--;
		}
	}
	if(len > 0)
		return false;
	return true;
}

#ifndef NO_TEST_CODE
size_t count_trailing_zeros(size_t n)
{
	size_t count = 0;
	for(size_t i = 0; i < 32; i++)
	{
		if(n & 1)
			return count;
		count++;
		n = n >> 1;
	}
	return (size_t)-1;
}

void test_boundingShift()
{
	GRand rand(0);
	for(size_t i = 0; i < 1000; i++)
	{
		size_t bits = (size_t)rand.next(31);
		int n = 1 << bits;
		if(GBits::boundingShift(n) != bits)
			throw Ex("failed");
		n++;
		if(GBits::boundingShift(n) != bits + 1)
			throw Ex("failed");
	}
}

void test_countTrailingZeros()
{
	for(size_t i = 0; i < 10000; i++)
	{
		if(count_trailing_zeros(i) != GBits::countTrailingZeros(i))
			throw Ex("failed");
	}
}

void GBits::test()
{
	test_boundingShift();
	test_countTrailingZeros();
}
#endif // NO_TEST_CODE
