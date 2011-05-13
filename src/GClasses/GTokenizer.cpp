/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GTokenizer.h"
#include "GError.h"
#include "GHolders.h"
#include "GFile.h"
#include "GString.h"
#include <stdio.h>
#include <string.h>
#include <fstream>

using std::string;

namespace GClasses {


GTokenizer::GTokenizer(const char* szFilename)
{
	std::ifstream* pStream = new std::ifstream();
	m_pStream = pStream;
	pStream->exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		pStream->open(szFilename, std::ios::binary);
		pStream->seekg(0, std::ios::end);
		m_len = (size_t)pStream->tellg();
		pStream->seekg(0, std::ios::beg);
	}
	catch(const std::exception&)
	{
		if(GFile::doesFileExist(szFilename))
			ThrowError("Error while trying to open the existing file: ", szFilename);
		else
			ThrowError("File not found: ", szFilename);
	}
	m_pBufStart = new char[256];
	m_pBufPos = m_pBufStart;
	m_pBufEnd = m_pBufStart + 256;
	m_lineStart = m_len;
	m_line = 1;
}

GTokenizer::GTokenizer(const char* pFile, size_t len)
{
	if(len > 0)
	{
		m_pStream = new std::istringstream(string(pFile, len));
		m_len = len;
	}
	else
	{
		string s(pFile);
		m_len = s.length();
		m_pStream = new std::istringstream(s);
	}
	m_pBufStart = new char[256];
	m_pBufPos = m_pBufStart;
	m_pBufEnd = m_pBufStart + 256;
	m_lineStart = m_len;
	m_line = 1;
}

GTokenizer::~GTokenizer()
{
	delete[] m_pBufStart;
	delete(m_pStream);
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

const char* GTokenizer::next(const char* szDelimeters)
{
	m_pBufPos = m_pBufStart;
	while(m_len > 0)
	{
		char c = m_pStream->peek();
		bool isDelimeter = false;
		if(szDelimeters)
		{
			for(const char* pDel = szDelimeters; *pDel != '\0'; pDel++)
			{
				if(c == *pDel)
				{
					isDelimeter = true;
					break;
				}
			}
		}
		else
		{
			if(c <= ' ')
				isDelimeter = true;
		}
		if(isDelimeter)
			break;
		c = m_pStream->get();
		m_len--;
		if(c == '\n')
		{
			m_line++;
			m_lineStart = m_len;
		}
		if(m_pBufPos == m_pBufEnd)
			growBuf();
		*m_pBufPos = c;
		m_pBufPos++;
	}
	if(m_pBufPos == m_pBufEnd)
		growBuf();
	GAssert(m_pBufPos != m_pBufStart); // Empty token! Did you forget to skip past the delimeter after the previous call to next?
	*m_pBufPos = '\0';
	return m_pBufStart;
}

void GTokenizer::skip(const char* szDelimeters)
{
	while(m_len > 0)
	{
		char c = m_pStream->peek();
		bool isDelimeter = false;
		if(szDelimeters)
		{
			for(const char* pDel = szDelimeters; *pDel != '\0'; pDel++)
			{
				if(c == *pDel)
				{
					isDelimeter = true;
					break;
				}
			}
		}
		else
		{
			if(c <= ' ')
				isDelimeter = true;
		}
		if(!isDelimeter)
			break;
		c = m_pStream->get();
		m_len--;
		if(c == '\n')
		{
			m_line++;
			m_lineStart = m_len;
		}
	}
}

void GTokenizer::skipTo(const char* szDelimeters)
{
	while(m_len > 0)
	{
		char c = m_pStream->peek();
		bool isDelimeter = false;
		if(szDelimeters)
		{
			for(const char* pDel = szDelimeters; *pDel != '\0'; pDel++)
			{
				if(c == *pDel)
				{
					isDelimeter = true;
					break;
				}
			}
		}
		else
		{
			if(c <= ' ')
				isDelimeter = true;
		}
		if(isDelimeter)
			break;
		c = m_pStream->get();
		m_len--;
		if(c == '\n')
		{
			m_line++;
			m_lineStart = m_len;
		}
	}
}

const char* GTokenizer::nextTok(const char* szDelimeters)
{
	skip(szDelimeters);
	return next(szDelimeters);
}

const char* GTokenizer::nextArg()
{
	char c = m_pStream->peek();
	if(c == '"')
	{
		advance(1);
		next("\"\n");
		if(peek() != '"')
			ThrowError("Expected matching double-quotes on line ", to_str(m_line), ", col ", to_str(col()));
		advance(1);
		return m_pBufStart;
	}
	else if(c == '\'')
	{
		advance(1);
		next("'\n");
		if(peek() != '\'')
			ThrowError("Expected a matching single-quote on line ", to_str(m_line), ", col ", to_str(col()));
		advance(1);
		return m_pBufStart;
	}
	else
		return next(" \t\n{\r");
}

void GTokenizer::advance(size_t n)
{
	while(n > 0 && m_len > 0)
	{
		char c = m_pStream->get();
		m_len--;
		if(c == '\n')
		{
			m_line++;
			m_lineStart = m_len;
		}
		n--;
	}
}

char GTokenizer::peek()
{
	if(m_len > 0)
		return m_pStream->peek();
	else
		return '\0';
}

size_t GTokenizer::line()
{
	return m_line;
}

size_t GTokenizer::remaining()
{
	return m_len;
}

void GTokenizer::expect(const char* szString)
{
	while(*szString != '\0' && m_len > 0)
	{
		char c = m_pStream->get();
		m_len--;
		if(c == '\n')
		{
			m_line++;
			m_lineStart = m_len;
		}
		if(c != *szString)
			ThrowError("Expected \"", szString, "\" on line ", to_str(m_line), ", col ", to_str(col()));
		szString++;
	}
}

size_t GTokenizer::tokenLength()
{
	return m_pBufPos - m_pBufStart;
}

const char* GTokenizer::trim()
{
	const char* pStart = m_pBufStart;
	while(pStart < m_pBufPos && *pStart <= ' ')
		pStart++;
	for(char* pEnd = m_pBufPos - 1; pEnd >= pStart && *pEnd <= ' '; pEnd--)
		*pEnd = '\0';
	return pStart;
}

size_t GTokenizer::col()
{
	return m_lineStart - m_len + 1;
}

} // namespace GClasses
