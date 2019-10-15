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

#include "GTokenizer.h"
#include "GError.h"
#include "GHolders.h"
#include "GFile.h"
#include "GString.h"
#include "GBitTable.h"
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <errno.h>
#ifdef WINDOWS
	#include <io.h>
#else
	#include <unistd.h>
#endif

using std::string;
using std::map;
using std::string;

namespace GClasses {


GCharSet::GCharSet(const char* szChars)
: m_bt(256)
{
	char c = '\0';
	while(*szChars != '\0')
	{
		if(*szChars == '-')
		{
			if(c == '\0')
				m_bt.set((unsigned char)*szChars);
			else
			{
				char d = szChars[1];
				if(d <= c)
					throw Ex("invalid character range");
				for(c++; c <= d && c != 0; c++)
					m_bt.set((unsigned char)c);
				szChars++;
			}
		}
		else
			m_bt.set((unsigned char)*szChars);
		c = *szChars;
		szChars++;
	}
}

bool GCharSet::find(char c) const
{
	return m_bt.bit((unsigned char)c);
}

bool GCharSet::equals(const GCharSet& other) const
{
	return m_bt.equals(other.m_bt);
}



GTokenizer::GTokenizer(const char* szFilename)
{
	std::ifstream* pStream = new std::ifstream();
	m_pStream = pStream;
	pStream->exceptions(std::ios::badbit); // don't include std::ios::failbit here because the has_more method reads first, then checks for EOF.
	try
	{
		pStream->open(szFilename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to open the file, ", szFilename, ". ", strerror(errno));
	}
	if(pStream->fail())
		throw Ex("Error while trying to open the file, ", szFilename, ". ", strerror(errno));
	m_qPos = 0;
	m_qCount = 0;
	m_pBufStart = new char[256];
	m_pBufPos = m_pBufStart;
	m_pBufEnd = m_pBufStart + 256;
	m_pos = 0;
	m_lineStart = 0;
	m_line = 1;
}

GTokenizer::GTokenizer(const char* pFile, size_t len)
{
	if(len > 0)
	{
		m_pStream = new std::istringstream(string(pFile, len));
	}
	else
	{
		m_pStream = new std::istringstream(pFile);
	}
	m_qPos = 0;
	m_qCount = 0;
	m_pBufStart = new char[256];
	m_pBufPos = m_pBufStart;
	m_pBufEnd = m_pBufStart + 256;
	m_pos = 0;
	m_lineStart = 0;
	m_line = 1;
}

GTokenizer::~GTokenizer()
{
	delete[] m_pBufStart;
	delete(m_pStream);
}

bool GTokenizer::has_more()
{
	if(m_qCount == 0)
	{
		int c = m_pStream->get();
		if(c == EOF)
			return false;
		m_q[m_qPos] = (char)c;
		m_qCount = 1;
	}
	return true;
}

char GTokenizer::peek()
{
	if(m_qCount == 0)
	{
		int c = m_pStream->get();
		if(c == EOF)
			return '\0';
		m_q[m_qPos] = (char)c;
		m_qCount = 1;
	}
	return m_q[m_qPos];
}

char GTokenizer::peek(size_t n)
{
	if(n >= GTOKENIZER_MAX_LOOKAHEAD)
		throw Ex("out of range");
	while(m_qCount <= n)
	{
		int c = m_pStream->get();
		if(c == EOF)
			return '\0';
		m_q[(m_qPos + m_qCount) % GTOKENIZER_MAX_LOOKAHEAD] = (char)c;
		m_qCount++;
	}
	return m_q[(m_qPos + n) % GTOKENIZER_MAX_LOOKAHEAD];
}

char GTokenizer::get()
{
	if(m_qCount == 0)
	{
		int cc = m_pStream->get();
		if(cc == EOF)
			return '\0';
		m_q[m_qPos] = (char)cc;
		m_qCount = 1;
	}
	char c = m_q[m_qPos];
	if(++m_qPos >= GTOKENIZER_MAX_LOOKAHEAD)
		m_qPos = 0;
	m_qCount--;
	m_pos++;
	if(c == '\n')
	{
		m_line++;
		m_lineStart = m_pos;
	}
	return c;
}

void GTokenizer::growBuf()
{
	size_t len = m_pBufEnd - m_pBufStart;
	char* pNewBuf = new char[len * 2];
	m_pBufEnd = pNewBuf + (len * 2);
	memcpy(pNewBuf, m_pBufStart, len);
	m_pBufPos = pNewBuf + len;
	delete[] m_pBufStart;
	m_pBufStart = pNewBuf;
}

void GTokenizer::bufferChar(char c)
{
	if(m_pBufPos == m_pBufEnd)
		growBuf();
	*m_pBufPos = c;
	m_pBufPos++;
}

char* GTokenizer::nullTerminate()
{
	if(m_pBufPos == m_pBufEnd)
		growBuf();
	*m_pBufPos = '\0';
	return m_pBufStart;
}

char* GTokenizer::appendToToken(const char* string)
{
	while(*string != '\0')
	{
		bufferChar(*string);
		string++;
	}
	return nullTerminate();
}

char* GTokenizer::readUntil(const GCharSet& delimeters, size_t minLen, size_t maxLen)
{
	m_pBufPos = m_pBufStart;
	while(has_more() && maxLen > 0)
	{
		char c = peek();
		if(delimeters.find(c))
			break;
		c = get();
		bufferChar(c);
		maxLen--;
	}
	if((size_t)(m_pBufPos - m_pBufStart) < minLen)
		throw Ex("On line ", to_str(m_line), ", col ", to_str(col()), ", expected a token of at least size ", to_str(minLen), ", but got only ", to_str(m_pBufPos - m_pBufStart));
	return nullTerminate();
}

char* GTokenizer::readUntil_escaped(char escapeChar, const GCharSet& delimeters)
{
	m_pBufPos = m_pBufStart;
	char cCur = '\0';
	while(has_more())
	{
		char c = peek();
		if(delimeters.find(c) && cCur != escapeChar)
			break;
		c = get();
		bufferChar(c);
		cCur = c;
	}
	return nullTerminate();
}

char* GTokenizer::readWhile(const GCharSet& set, size_t minLen)
{
	m_pBufPos = m_pBufStart;
	while(has_more())
	{
		char c = peek();
		if(!set.find(c))
			break;
		c = get();
		bufferChar(c);
	}
	if((size_t)(m_pBufPos - m_pBufStart) < minLen)
		throw Ex("Unexpected token on line ", to_str(m_line), ", col ", to_str(col()));
	return nullTerminate();
}

void GTokenizer::skipWhile(const GCharSet& delimeters)
{
	while(has_more())
	{
		char c = peek();
		if(!delimeters.find(c))
			break;
		c = get();
	}
}

void GTokenizer::skipUntil(const GCharSet& delimeters)
{
	while(has_more())
	{
		char c = peek();
		if(delimeters.find(c))
			break;
		c = get();
	}
}

char* GTokenizer::readUntil_escaped_quoted(const GCharSet& delimiters, char escapeChar)
{
	m_pBufPos = m_pBufStart;
	char c = peek();
	if(c == '"')
	{
		bufferChar('"');
		skip(1);
		while(has_more())
		{
			char c2 = peek();
			if(c2 == '\"' || c2 == '\n')
				break;
			c2 = get();
			bufferChar(c2);
		}
		if(peek() != '"')
			throw Ex("Expected matching double-quotes on line ",
								 to_str(m_line), ", col ", to_str(col()));
		bufferChar('"');
		skip(1);
		while(!delimiters.find(peek()))
			skip(1);
		return nullTerminate();
	}
	else if(c == '\'')
	{
		bufferChar('\'');
		skip(1);
		while(has_more())
		{
			char c2 = peek();
			if(c2 == '\'' || c2 == '\n')
				break;
			c2 = get();
			bufferChar(c2);
		}
		if(peek() != '\'')
			throw Ex("Expected a matching single-quote on line ", to_str(m_line),
								 ", col ", to_str(col()));
		bufferChar('\'');
		skip(1);
		while(!delimiters.find(peek()))
			skip(1);
		return nullTerminate();
	}

	bool inEscapeMode = false;
	while(has_more())
	{
		char c2 = peek();
		if(inEscapeMode)
		{
			if(c2 == '\n')
			{
				throw Ex("Error: '", to_str(escapeChar), "' character used as "
									 "last character on a line to attempt to extend string over "
									 "two lines on line" , to_str(m_line), ", col ",
									 to_str(col()) );
			}
			c2 = get();
			bufferChar(c2);
			inEscapeMode = false;
		}
		else
		{
			if(c2 == '\n' || delimiters.find(c2)){ break; }
			c2 = get();
			if(c2 == escapeChar)	{	inEscapeMode = true;	}
			else { bufferChar(c2);	}
		}
	}

	return nullTerminate();
}

void GTokenizer::skip(size_t n)
{
	while(n > 0 && has_more())
	{
		get();
		n--;
	}
}

size_t GTokenizer::line()
{
	return m_line;
}

void GTokenizer::skipExact(const char* szString)
{
	while(*szString != '\0' && has_more())
	{
		char c = get();
		if(c != *szString)
			throw Ex("Expected \"", szString, "\" on line ", to_str(m_line), ", col ", to_str(col()));
		szString++;
	}
	if(*szString != '\0')
		throw Ex("Expected \", szString, \". Reached end-of-file instead.");
}

size_t GTokenizer::tokenLength()
{
	return m_pBufPos - m_pBufStart;
}

char* GTokenizer::trim(const GCharSet& set)
{
	char* pStart = m_pBufStart;
	while(pStart < m_pBufPos && set.find(*pStart))
		pStart++;
	for(char* pEnd = m_pBufPos - 1; pEnd >= pStart && set.find(*pEnd); pEnd--)
		*pEnd = '\0';
	return pStart;
}

char* GTokenizer::filter(const GCharSet& set)
{
	size_t end = m_pBufPos - m_pBufStart;
	for(size_t i = 0; i < end; i++)
	{
		if(!set.find(m_pBufStart[i]))
		{
			for(size_t j = i; j < end - 1; j++)
				m_pBufStart[j] = m_pBufStart[j + 1];
			end--;
			m_pBufStart[end] = '\0';
			i--;
		}
	}
	return m_pBufStart;
}

size_t GTokenizer::col()
{
	return m_pos - m_lineStart;
}

} // namespace GClasses
