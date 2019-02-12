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

#include "GString.h"
#include "GError.h"
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

using std::string;

namespace GClasses {

size_t safe_strcpy(char* szDest, const char* szSrc, size_t nMaxSize)
{
	if(nMaxSize == 0)
		return 0;
	nMaxSize--;
	size_t n;
	for(n = 0; szSrc[n] != '\0' && n < nMaxSize; n++)
		szDest[n] = szSrc[n];
	szDest[n] = '\0';
	return n;
}


std::string to_fixed_str(size_t val, size_t total_chars, char pad)
{
	std::string s = to_str(val);
	if(s.length() < total_chars)
		return std::string(total_chars - s.length(), pad) + s;
	else if(s.length() > total_chars)
		throw Ex("Not enough characters to represent the value ", s);
	return s;
}



std::string to_fixed_str(double val, size_t total_chars, char pad)
{
	std::ostringstream os;
	os.precision(total_chars);
	os << val;
	std::string s = os.str();
	if(s.length() < total_chars)
		return std::string(total_chars - s.length(), pad) + s;
	else if(s.length() > total_chars)
	{
		size_t dot = s.find_last_of(".");
		if(dot == string::npos)
			throw Ex("Not enough characters to represent the value ", s);
		size_t e = s.find_last_of("eE");
		if(e == string::npos)
		{
			if(dot + 1 > total_chars)
				throw Ex("Not enough characters to represent the value ", s);
			while(s.length() > total_chars)
				s.pop_back();
			return s;
		}
		else
		{
			size_t b = e - (s.length() - total_chars);
			if(b <= dot || b >= e)
				throw Ex("Not enough characters to represent the value ", s);
			s.erase(b, e - b);
			return s;
		}
	}
	else
		return s;
}





GStringChopper::GStringChopper(const char* szString, size_t nMinLength, size_t nMaxLength, bool bDropLeadingWhitespace)
{
	GAssert(nMinLength > 0 && nMaxLength >= nMinLength); // lengths out of range
	m_bDropLeadingWhitespace = bDropLeadingWhitespace;
	if(nMinLength < 1)
		nMinLength = 1;
	if(nMaxLength < nMinLength)
		nMaxLength = nMinLength;
	m_nMinLen = nMinLength;
	m_nMaxLen = nMaxLength;
	m_szString = szString;
	m_nLen = strlen(szString);
	if(m_nLen > nMaxLength)
		m_pBuf = new char[nMaxLength + 1];
	else
		m_pBuf = NULL;
}

GStringChopper::~GStringChopper()
{
	delete[] m_pBuf;
}

void GStringChopper::reset(const char* szString)
{
	m_szString = szString;
	m_nLen = strlen(szString);
	if(!m_pBuf && m_nLen > m_nMaxLen)
		m_pBuf = new char[m_nMaxLen + 1];
}

const char* GStringChopper::next()
{
	if(m_nLen <= 0)
		return NULL;
	if(m_nLen <= m_nMaxLen)
	{
		m_nLen = 0;
		return m_szString;
	}
	size_t i;
	for(i = m_nMaxLen; i >= m_nMinLen && m_szString[i] > ' '; i--)
	{
	}
	if(i < m_nMinLen)
		i = m_nMaxLen;
	memcpy(m_pBuf, m_szString, i);
	m_pBuf[i] = '\0';
	m_szString += i;
	m_nLen -= i;
	if(m_bDropLeadingWhitespace)
	{
		while(m_nLen > 0 && m_szString[0] <= ' ')
		{
			m_szString++;
			m_nLen--;
		}
	}
	return m_pBuf;
}


} // namespace GClasses

