/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GXML.h"
#include "../GClasses/GHolders.h"
#include "../GClasses/GBits.h"
#include <stdio.h>
#include <string.h>
#ifdef WINDOWS
#	include <io.h>
#endif
#include <fstream>
#include "../GClasses/GFile.h"

using std::vector;
using std::ofstream;
using std::ostream;

namespace GClasses {

#ifndef UCHAR
#	define UCHAR(c) ((c) & (~32))
#endif // UCHAR


// This is a helper class used by GXMLTag::ParseXMLFile
class GXMLParser
{
protected:
	static const char* s_szCommentError;

	GXMLTag* m_pRootTag;
	GXMLTag* m_pCurrentTag;
	int m_nPos;
	const char* m_pFile;
	int m_nLength;
	int m_nLine;
	int m_nLineStart;
	const char* m_szErrorMessage;
	int m_nErrorOffset;
	int m_nErrorLine;
	int m_nErrorColumn;

public:
	GXMLParser(const char* pFile, int nSize);
	virtual ~GXMLParser();

	GXMLTag* Parse(const char** pszErrorMessage, int* pnErrorOffset, int* pnErrorLine, int* pnErrorColumn);

protected:
	void SetError(const char* szMessage);
	GXMLTag* ParseTag();
	bool ParseCloser(int nNameStart, int nNameLength);
	int ParseName();
	void EatWhitespace();
	GXMLAttribute* ParseAttribute();
	bool UnescapeAttrValue(const char* pValue, int nLength, char* pBuffer);
	void MoveIntoNextTag();
	GXMLTag* ParseTagInternal();
};

/*static*/ const char* GXMLParser::s_szCommentError = "The tag is a comment";

GXMLParser::GXMLParser(const char* pFile, int nSize)
{
	m_pRootTag = NULL;
	m_pCurrentTag = NULL;
	m_nPos = 0;
	m_nLineStart = 0;
	m_pFile = pFile;
	m_nLength = nSize;
	m_nLine = 1;
	m_szErrorMessage = NULL;
	m_nErrorOffset = 0;
	m_nErrorLine = 0;
	m_nErrorColumn = 0;
}

GXMLParser::~GXMLParser()
{

}

inline bool isWhitespace(char c)
{
	return c <= ' ' ? true : false;
}

void GXMLParser::SetError(const char* szMessage)
{
	GAssert(!m_szErrorMessage); // Error message already set
	m_szErrorMessage = szMessage;
	m_nErrorOffset = m_nPos;
	m_nErrorLine = m_nLine;
	m_nErrorColumn = m_nPos - m_nLineStart;
}

void GXMLParser::EatWhitespace()
{
	char c;
	while(true)
	{
		if(m_nPos >= m_nLength)
			break;
		c = m_pFile[m_nPos];
		if(!isWhitespace(c))
			break;
		if(c == '\n')
		{
			m_nLine++;
			m_nLineStart = m_nPos + 1;
		}
		m_nPos++;
	}
}

GXMLTag* GXMLParser::Parse(const char** pszErrorMessage, int* pnErrorOffset, int* pnErrorLine, int* pnErrorColumn)
{
	GXMLTag* pRoot = NULL;
	m_szErrorMessage = s_szCommentError;
	while(m_szErrorMessage == s_szCommentError)
	{
		m_szErrorMessage = NULL;
		pRoot = ParseTag();
	}
	if(pszErrorMessage)
		*pszErrorMessage = m_szErrorMessage;
	if(pnErrorOffset)
		*pnErrorOffset = m_nErrorOffset;
	if(pnErrorLine)
		*pnErrorLine = m_nErrorLine;
	if(pnErrorColumn)
		*pnErrorColumn = m_nErrorColumn;
	return pRoot;
}

GXMLTag* GXMLParser::ParseTag()
{
	MoveIntoNextTag();
	if(m_nPos >= m_nLength)
	{
		SetError("Expected a tag");
		return NULL;
	}
	return ParseTagInternal();
}

void GXMLParser::MoveIntoNextTag()
{
	// Parse the name
	while(m_nPos < m_nLength && m_pFile[m_nPos] != '<')
		m_nPos++;
	if(m_nPos < m_nLength)
		m_nPos++;
}

GXMLTag* GXMLParser::ParseTagInternal()
{
	int nNameStart = m_nPos;
	int nNameLength = ParseName();
	if(nNameLength < 1)
	{
		SetError("Expected a tag name");
		return NULL;
	}
	if(m_pFile[nNameStart] == '?')
	{
		while(true)
		{
			if(m_nPos >= m_nLength)
				break;
			if(m_pFile[m_nPos] == '>')
				break;
			else if(m_pFile[m_nPos] == '\n')
			{
				m_nLine++;
				m_nLineStart = m_nPos + 1;
			}
			m_nPos++;
		}
		if(m_nPos < m_nLength)
			m_nPos++;
		SetError(s_szCommentError);
		return NULL;
	}
	else if(nNameLength >= 3 && strncmp(&m_pFile[nNameStart], "!--", 3) == 0)
	{
		m_nPos -= nNameLength;

		// Skip to the end of the comment
		int nNests = 1;
		while(true)
		{
			if(m_nPos + 3 >= m_nLength)
				break;
			if(strncmp(&m_pFile[m_nPos], "<!--", 4) == 0)
				nNests++;
			if(strncmp(&m_pFile[m_nPos], "-->", 3) == 0)
			{
				nNests--;
				if(nNests <= 0)
					break;
			}
			else if(m_pFile[m_nPos] == '\n')
			{
				m_nLine++;
				m_nLineStart = m_nPos + 1;
			}
			m_nPos++;
		}
		m_nPos += 3;
		SetError(s_szCommentError);
		return NULL;
	}
	GXMLTag* pNewTag = new GXMLTag(&m_pFile[nNameStart], nNameLength);
	Holder<GXMLTag> hNewTag(pNewTag);
	pNewTag->setLineNumber(m_nLine);
	int nStartColumn = m_nPos - m_nLineStart + 1;

	// Parse Attributes
	while(true)
	{
		EatWhitespace();
		if(m_nPos >= m_nLength)
		{
			SetError("Expected an attribute, a '/', or a '>'");
			return NULL;
		}
		if(m_pFile[m_nPos] == '/' || m_pFile[m_nPos] == '>')
			break;
		GXMLAttribute* pNewAttr = ParseAttribute();
		if(!pNewAttr)
			return NULL;
		pNewTag->attributes().push_back(pNewAttr);
	}
	int nEndColumn = m_nPos - m_nLineStart + 1;
	pNewTag->setColumnAndWidth(nStartColumn, nEndColumn - nStartColumn);

	if(m_pFile[m_nPos] == '>')
	{
		// Parse the children
		m_nPos++;
		while(true)
		{
			MoveIntoNextTag();
			if(m_nPos >= m_nLength)
			{
				SetError("Expected a closer tag");
				return NULL;
			}
			if(m_pFile[m_nPos] == '/')
			{
				if(!ParseCloser(nNameStart, nNameLength))
					return NULL;
				break;
			}
			GXMLTag* pChildTag = ParseTagInternal();
			if(!pChildTag)
			{
				if(m_szErrorMessage == s_szCommentError)
				{
					m_szErrorMessage = NULL;
					continue;
				}
				else
					return NULL;
			}
			pNewTag->children().push_back(pChildTag);
		}
	}
	else
	{
		// Parse the end of the tag
		GAssert(m_pFile[m_nPos] == '/'); // internal error
		m_nPos++;
		EatWhitespace();
		if(m_nPos >= m_nLength || m_pFile[m_nPos] != '>')
		{
			SetError("Expected a '>'");
			return NULL;
		}
		m_nPos++;
	}
	return hNewTag.release();
}

int GXMLParser::ParseName()
{
	EatWhitespace();
	char c;
	int nLength = 0;
	while(true)
	{
		if(m_nPos >= m_nLength)
			break;
		c = m_pFile[m_nPos];
		if(c == '/' || c == '>' || c == '=')
			break;
		else if(c == '\n')
		{
			m_nLine++;
			m_nLineStart = m_nPos + 1;
			break;
		}
		else if(isWhitespace(c))
			break;
		m_nPos++;
		nLength++;
	}
	return nLength;
}

bool GXMLParser::ParseCloser(int nNameStart, int nNameLength)
{
	if(m_nPos >= m_nLength || m_pFile[m_nPos] != '/')
	{
		SetError("Expected a '/'");
		return false;
	}
	m_nPos++;
	EatWhitespace();
	int nCloserStart = m_nPos;
	int nCloserLength = ParseName();
	if(nCloserLength != nNameLength || memcmp(&m_pFile[nNameStart], &m_pFile[nCloserStart], nCloserLength) != 0)
	{
		SetError("Closer name doesn't match tag name");
		return false;
	}
	EatWhitespace();
	if(m_nPos >= m_nLength || m_pFile[m_nPos] != '>')
	{
		SetError("Expected a '>'");
		return false;
	}
	m_nPos++;
	return true;
}

GXMLAttribute* GXMLParser::ParseAttribute()
{
	int nNameStart = m_nPos;
	int nNameLength = ParseName();
	if(nNameLength < 1)
	{
		SetError("Expected an attribute name");
		return NULL;
	}
	EatWhitespace();
	if(m_nPos >= m_nLength || m_pFile[m_nPos] != '=')
	{
		SetError("Expected a '='");
		return NULL;
	}
	m_nPos++;
	EatWhitespace();
	if(m_nPos >= m_nLength || m_pFile[m_nPos] != '"')
	{
		SetError("Expected a '\"'");
		return NULL;
	}
	m_nPos++;
	int nValueStart = m_nPos;
	char c;
	int nValueLength = 0;
	while(true)
	{
		if(m_nPos >= m_nLength)
		{
			SetError("Expected a '\"'");
			return NULL;
		}
		c = m_pFile[m_nPos];
		if(c == '"')
			break;
		else if(c == '\n')
		{
			SetError("Expected a '\"'");
			return NULL;
		}
		m_nPos++;
		nValueLength++;
	}
	if(m_nPos >= m_nLength || m_pFile[m_nPos] != '"')
	{
		SetError("Expected a '\"'");
		return NULL;
	}
	m_nPos++;
	
	char szTmp[512];
	char* szBuff = szTmp;
	if(nValueLength >= 512)
		szBuff = new char[nValueLength + 1];
	if(!UnescapeAttrValue(&m_pFile[nValueStart], nValueLength, szBuff))
	{
		SetError("Unrecognized escape sequence");
		return NULL;
	}
	GXMLAttribute* pAttr = new GXMLAttribute(&m_pFile[nNameStart], nNameLength, szBuff);
	if(szBuff != szTmp)
		delete[] szBuff;
	return pAttr;
}

bool GXMLParser::UnescapeAttrValue(const char* pValue, int nLength, char* pBuffer)
{
	while(nLength > 0)
	{
		if(*pValue == '&')
		{
			pValue++;
			nLength--;
			switch(*pValue)
			{
			case '#':
				{
					pValue++;
					nLength--;
					if(UCHAR(*pValue) != 'X')
						return false;
					pValue++;
					nLength--;
					int nNum = 0;
					while(*pValue != ';' && nLength > 0)
					{
						if(UCHAR(*pValue) < '0' || UCHAR(*pValue) > 'F')
							return false;
						if(UCHAR(*pValue) > '9' && UCHAR(*pValue) < 'A')
							return false;
						nNum <<= 4;
						if(UCHAR(*pValue) <= '9')
							nNum += UCHAR(*pValue) - '0';
						else
							nNum += UCHAR(*pValue) - 'A' + 10;
						pValue++;
						nLength--;
					}
					if(nLength < 1)
						return false;
					pValue++;
					nLength--;
					*pBuffer = nNum;
				}
				break;
			case 'a':
			case 'A':
				pValue++;
				nLength--;
				if(UCHAR(*pValue) != 'M')
					return false;
				pValue++;
				nLength--;
				if(UCHAR(*pValue) != 'P')
					return false;
				pValue++;
				nLength--;
				if(*pValue != ';')
					return false;
				*pBuffer = '&';
				pValue++;
				nLength--;
				break;
			case 'g':
			case 'G':
				pValue++;
				nLength--;
				if(UCHAR(*pValue) != 'T')
					return false;
				pValue++;
				nLength--;
				if(*pValue != ';')
					return false;
				*pBuffer = '>';
				pValue++;
				nLength--;
				break;
			case 'l':
			case 'L':
				pValue++;
				nLength--;
				if(UCHAR(*pValue) != 'T')
					return false;
				pValue++;
				nLength--;
				if(*pValue != ';')
					return false;
				*pBuffer = '<';
				pValue++;
				nLength--;
				break;
			case 'q':
			case 'Q':
				pValue++;
				nLength--;
				if(UCHAR(*pValue) != 'U')
					return false;
				pValue++;
				nLength--;
				if(UCHAR(*pValue) != 'O')
					return false;
				pValue++;
				nLength--;
				if(UCHAR(*pValue) != 'T')
					return false;
				pValue++;
				nLength--;
				if(*pValue != ';')
					return false;
				*pBuffer = '"';
				pValue++;
				nLength--;
				break;
			default:
				return false;
			}
		}
		else
		{
			*pBuffer = *pValue;
			pValue++;
			nLength--;
		}
		pBuffer++;
	}
	*pBuffer = '\0';
	return true;
}








GXMLAttribute::GXMLAttribute(const char* szName, const char* szValue)
{
	m_pName = NULL;
	m_pValue = NULL;
	setName(szName);
	setValue(szValue);
}

GXMLAttribute::GXMLAttribute(const char* pName, int nNameLength, const char* pValue)
{
	m_pName = new char[nNameLength + 1];
	memcpy(m_pName, pName, nNameLength);
	m_pName[nNameLength] = '\0';
	m_pValue = NULL;
	setValue(pValue);
}

GXMLAttribute::GXMLAttribute(const char* pName, const char* pValue, int nValueLength)
{
	m_pName = NULL;
	setName(pName);
	m_pValue = new char[nValueLength + 1];
	memcpy(m_pValue, pValue, nValueLength);
	m_pValue[nValueLength] = '\0';
}

GXMLAttribute::~GXMLAttribute()
{
	delete[] m_pName;
	delete[] m_pValue;
}

void GXMLAttribute::setName(const char* szName)
{
	delete(m_pName);
	m_pName = new char[strlen(szName) + 1];
	strcpy(m_pName, szName);
	int n;
	for(n = 0; m_pName[n] != '\0'; n++)
	{
		if(m_pName[n] < ' ')
			m_pName[n] = ' ';
	}
}

void GXMLAttribute::setValue(const char* szValue)
{
	if(szValue == m_pValue)
		return;
	delete[] m_pValue;
	m_pValue = new char[strlen(szValue) + 1];
	strcpy(m_pValue, szValue);
}

int EscapeAttrChar(char c, char* pBuffer)
{
	if(c < ' ')
	{
		char szTmp[3];
		GBits::byteToHexBigEndian(c, szTmp);
		szTmp[2] = '\0';
		if(pBuffer)
		{
			strcpy(pBuffer, "&#x");
			strcpy(pBuffer + 3, szTmp);
			strcpy(pBuffer + 5, ";");
		}
		return 4 + (int)strlen(szTmp);
	}
	else
	{
		switch(c)
		{
		case '&':
			if(pBuffer)
				strcpy(pBuffer, "&amp;");
			return 5;
		case '<':
			if(pBuffer)
				strcpy(pBuffer, "&lt;");
			return 4;
		case '>':
			if(pBuffer)
				strcpy(pBuffer, "&gt;");
			return 4;
		case '"':
			if(pBuffer)
				strcpy(pBuffer, "&quot;");
			return 6;
		default:
			if(pBuffer)
				*pBuffer = c;
			return 1;
		}
	}
}

int GXMLAttribute::toString(char* pBuffer, bool bEscapeQuotes)
{
	int nPos = 0;
	if(pBuffer)
		pBuffer[nPos] = ' ';
	nPos++;
	if(pBuffer)
		strcpy(&pBuffer[nPos], m_pName);
	nPos += (int)strlen(m_pName);
	if(bEscapeQuotes)
	{
		if(pBuffer)
			strcpy(&pBuffer[nPos], "=\\\"");
		nPos += 3;
	}
	else
	{
		if(pBuffer)
			strcpy(&pBuffer[nPos], "=\"");
		nPos += 2;
	}
	char* pChar = m_pValue;
	while(*pChar != '\0')
	{
		nPos += EscapeAttrChar(*pChar, pBuffer ? &pBuffer[nPos] : NULL);
		pChar++;
	}
	if(bEscapeQuotes)
	{
		if(pBuffer)
			strcpy(&pBuffer[nPos], "\\\"");
		nPos += 2;
	}
	else
	{
		if(pBuffer)
			pBuffer[nPos] = '"';
		nPos++;
	}
	return nPos;
}



// **************************************************************

GXMLTag::GXMLTag(const char* szName)
{
	m_pParent = NULL;
	m_pName = NULL;
	setName(szName);
	m_nLineNumber = 0;
	m_nColumnAndWidth = 0;
#ifdef _DEBUG
	m_DEBUG_ONLY_attributes[0] = NULL;
	m_DEBUG_ONLY_attributes[1] = NULL;
	m_DEBUG_ONLY_attributes[2] = NULL;
	m_DEBUG_ONLY_attributes[3] = NULL;
#endif
}

GXMLTag::GXMLTag(const char* pName, int nLength)
{
	m_pParent = NULL;
	m_pName = new char[nLength + 1];
	memcpy(m_pName, pName, nLength);
	m_pName[nLength] = '\0';
	m_nLineNumber = 0;
	m_nColumnAndWidth = 0;
#ifdef _DEBUG
	m_DEBUG_ONLY_attributes[0] = NULL;
	m_DEBUG_ONLY_attributes[1] = NULL;
	m_DEBUG_ONLY_attributes[2] = NULL;
	m_DEBUG_ONLY_attributes[3] = NULL;
#endif
}

GXMLTag::~GXMLTag()
{
	for(vector<GXMLAttribute*>::iterator it = m_attributes.begin(); it != m_attributes.end(); it++)
		delete(*it);
	for(vector<GXMLTag*>::iterator it = m_children.begin(); it != m_children.end(); it++)
		delete(*it);
	delete[] m_pName;
}

const char* GXMLTag::attrValueIfExists(const char* name)
{
	for(vector<GXMLAttribute*>::iterator it = m_attributes.begin(); it != m_attributes.end(); it++)
	{
		if(strcmp(name, (*it)->name()) == 0)
			return (*it)->value();
	}
	return NULL;
}

const char* GXMLTag::attrValue(const char* _name)
{
	const char* pVal = attrValueIfExists(_name);
	if(!pVal)
		ThrowError("The <%s> tag has no attribute named \"%s\"", name(), _name);
	return pVal;
}

GXMLTag* GXMLTag::childTag(const char* name)
{
	for(vector<GXMLTag*>::iterator it = m_children.begin(); it != m_children.end(); it++)
	{
		if(strcmp((*it)->m_pName, name) == 0)
			return *it;
	}
	return NULL;
}

void GXMLTag::setName(const char* szName)
{
	delete(m_pName);
	m_pName = new char[strlen(szName) + 1];
	strcpy(m_pName, szName);
}

char* GXMLTag::toString(const char* szLineStart, const char* szLineEnd, bool bEscapeQuotes)
{
	int nSize = toString(NULL, 0, szLineStart, szLineEnd, bEscapeQuotes);
	char* pBuffer = new char[nSize + 1];
	int nSize2 = toString(pBuffer, 0, szLineStart, szLineEnd, bEscapeQuotes);
	if(nSize != nSize2)
	{
		GAssert(false); // size changed
	}
	pBuffer[nSize] = '\0';
	return pBuffer;
}

char* GXMLTag::toString()
{
	return toString("", "\n", false);
}

int GXMLTag::toString(char* pBuffer, int nTabs, const char* szLineStart, const char* szLineEnd, bool bEscapeQuotes)
{
	int nPos = 0;
	if(pBuffer)
		memset(pBuffer, '\t', nTabs);
	nPos += nTabs;
	int nLineStartLength = (int)strlen(szLineStart);
	int nLineEndLength = (int)strlen(szLineEnd);
	if(pBuffer)
		memcpy(&pBuffer[nPos], szLineStart, nLineStartLength);
	nPos += nLineStartLength;
	if(pBuffer)
		pBuffer[nPos] = '<';
	nPos++;
	if(pBuffer)
		strcpy(&pBuffer[nPos], m_pName);
	nPos += (int)strlen(m_pName);
	for(vector<GXMLAttribute*>::iterator it = m_attributes.begin(); it != m_attributes.end(); it++)
		nPos += (*it)->toString(pBuffer ? &pBuffer[nPos] : NULL, bEscapeQuotes);
	if(m_children.size() > 0)
	{
		if(pBuffer)
			strcpy(&pBuffer[nPos], ">");
		nPos += 1;
		if(pBuffer)
			memcpy(&pBuffer[nPos], szLineEnd, nLineEndLength);
		nPos += nLineEndLength;
		for(vector<GXMLTag*>::iterator it = m_children.begin(); it != m_children.end(); it++)
			nPos += (*it)->toString(pBuffer ? &pBuffer[nPos] : NULL, nTabs + 1, szLineStart, szLineEnd, bEscapeQuotes);
		if(pBuffer)
			memset(&pBuffer[nPos], '\t', nTabs);
		nPos += nTabs;
		if(pBuffer)
			memcpy(&pBuffer[nPos], szLineStart, nLineStartLength);
		nPos += nLineStartLength;
		if(pBuffer)
			strcpy(&pBuffer[nPos], "</");
		nPos += 2;
		if(pBuffer)
			strcpy(&pBuffer[nPos], m_pName);
		nPos += (int)strlen(m_pName);
		if(pBuffer)
			strcpy(&pBuffer[nPos], ">");
		nPos += 1;
		if(pBuffer)
			memcpy(&pBuffer[nPos], szLineEnd, nLineEndLength);
		nPos += nLineEndLength;
	}
	else
	{
		if(pBuffer)
			strcpy(&pBuffer[nPos], " />");
		nPos += 3;
		if(pBuffer)
			memcpy(&pBuffer[nPos], szLineEnd, nLineEndLength);
		nPos += nLineEndLength;
	}
	return nPos;
}

/*static*/ GXMLTag* GXMLTag::fromString(const char* pBuffer, int nSize, const char** pszErrorMessage /*=NULL*/, int* pnErrorOffset /*=NULL*/, int* pnErrorLine /*=NULL*/, int* pnErrorColumn /*=NULL*/)
{
	if(nSize < 1)
	{
		if(pszErrorMessage)
			*pszErrorMessage = "an empty string is not valid XML";
		if(pnErrorOffset)
			*pnErrorOffset = 0;
		if(pnErrorLine)
			*pnErrorLine = 1;
		if(pnErrorColumn)
			*pnErrorColumn = 0;
		return NULL;
	}
	GXMLParser parser(pBuffer, nSize);
	return parser.Parse(pszErrorMessage, pnErrorOffset, pnErrorLine, pnErrorColumn);
}

void GXMLTag::toFile(const char* szFilename)
{
	std::ofstream s;
	s.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		s.open(szFilename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		ThrowError("Error creating file: ", szFilename);
	}
	toFile(s);
}

void GXMLTag::toFile(ostream& os)
{
	char* pBuffer = toString();
	ArrayHolder<char> hBuffer(pBuffer);
	if(!pBuffer)
		ThrowError("Error serializing xml");
	os.write(pBuffer, strlen(pBuffer));
}

void GXMLTag::toCppFile(const char* szFilename, const char* szVarName, const char* szHeader)
{
	std::ofstream s;
	s.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		s.open(szFilename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		ThrowError("Error creating file: ", szFilename);
	}
	char* pBuffer = toString("\"", "\"\n", true);
	if(!pBuffer)
		ThrowError("Error serializing xml");
	ArrayHolder<char> hBuffer(pBuffer);
	s.write(szHeader, strlen(szHeader));
	const char* szDecl = "const char* ";
	s.write(szDecl, strlen(szDecl));
	s.write(szVarName, strlen(szVarName));
	const char* szHead = " = \n";
	s.write(szHead, strlen(szHead));
	s.write(pBuffer, strlen(pBuffer));
	const char* szTail = ";\n\n";
	s.write(szTail, strlen(szTail));
}

/*static*/ GXMLTag* GXMLTag::fromFile(const char* szFilename, const char** pszErrorMessage /*=NULL*/, int* pnErrorOffset /*=NULL*/, int* pnErrorLine /*=NULL*/, int* pnErrorColumn /*=NULL*/)
{
	if(pszErrorMessage)
		*pszErrorMessage = "<unknown error>";
	if(pnErrorOffset)
		*pnErrorOffset = 0;
	if(pnErrorLine)
		*pnErrorLine = 0;
	if(pnErrorColumn)
		*pnErrorColumn = 0;
	size_t nFileSize;
	char* pBuffer = GFile::loadFile(szFilename, &nFileSize);
	ArrayHolder<char> hBuffer(pBuffer);
	GXMLTag* pTag = fromString(pBuffer, (int)nFileSize, pszErrorMessage, pnErrorOffset, pnErrorLine, pnErrorColumn);
	return pTag;
}

} // namespace GClasses

