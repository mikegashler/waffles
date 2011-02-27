/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "GTwt.h"
//#include "GXML.h"
#include "GFile.h"
#include "GHolders.h"
#include <vector>
#include <deque>
#include <sstream>
#include <fstream>

namespace GClasses {

using std::vector;
using std::deque;

struct GTwtObjField
{
	const char* m_pName;
	GTwtNode* m_pValue;
	GTwtObjField* m_pNext;
};


GTwtNode* GTwtNode::fieldIfExists(const char* szName)
{
	if(m_type != type_obj)
		ThrowError("not an obj");
	GTwtObjField* pField;
	for(pField = m_value.m_pFirstField; pField; pField = pField->m_pNext)
	{
		if(strcmp(szName, pField->m_pName) == 0)
			return pField->m_pValue;
	}
	return NULL;
}

void GTwtNode::checkFieldName(const char* szName)
{
	for(size_t i = 0; szName[i] != '\0'; i++)
	{
		if(szName[i] == ':')
			ThrowError("field names may not contain ':'");
	}
}

void GTwtNode::addField(GTwtObjField* pNewField)
{
	pNewField->m_pNext = m_value.m_pFirstField;
	m_value.m_pFirstField = pNewField;
}

void GTwtNode::reverseFieldOrder()
{
	GTwtObjField* pNewHead = NULL;
	while(m_value.m_pFirstField)
	{
		GTwtObjField* pTemp = m_value.m_pFirstField;
		m_value.m_pFirstField = pTemp->m_pNext;
		pTemp->m_pNext = pNewHead;
		pNewHead = pTemp;
	}
	m_value.m_pFirstField = pNewHead;
}

void GTwtNode::addField(GTwtDoc* pDoc, const char* szName, GTwtNode* pNode, bool copyName)
{
	if(m_type != type_obj)
		ThrowError("not an obj");
	checkFieldName(szName);
	GTwtObjField* pField = pDoc->newField();
	if(copyName)
	{
		GHeap* pHeap = pDoc->heap();
		pField->m_pName = pHeap->add(szName);
	}
	else
		pField->m_pName = szName;
	pField->m_pValue = pNode;
	addField(pField);
}

// static
void GTwtNode::printEscapedString(std::ostream& stream, const char* szString, bool forEmbeddedCode)
{
	if(forEmbeddedCode)
	{
		for(size_t i = 0; szString[i] != '\0'; i++)
		{
			if(szString[i] == '\n')
				stream << "\\\\n";
			else if(szString[i] == '\\')
				stream << "\\\\\\\\";
			else if(szString[i] == '\r')
				stream << "\\r";
			else
				stream << szString[i];
		}
	}
	else
	{
		for(size_t i = 0; szString[i] != '\0'; i++)
		{
			if(szString[i] == '\n')
				stream << "\\n";
			else if(szString[i] == '\\')
				stream << "\\\\";
			else
				stream << szString[i];
		}
	}
}

// static
void GTwtNode::unescapeString(char* szString, size_t line)
{
	char* pSrc = szString;
	char* pDest = szString;
	while(*pSrc != '\0')
	{
		if(*pSrc == '\n')
			ThrowError("Newline characters should be escaped as '\\n'");
		else if(*pSrc == '\\')
		{
			pSrc++;
			if(*pSrc == 'n')
				*pDest = '\n';
			else if(*pSrc == '\\')
				*pDest = '\\';
			else
				ThrowError("Unrecognized escape sequence \"\\", to_str(*pSrc), "\" at line ", to_str(line));
		}
		else
			*pDest = *pSrc;
		pDest++;
		pSrc++;
	}
	*pDest = '\0';
}

void GTwtNode::write(std::ostream& stream, const char* szLabel, int level, bool forEmbeddedCode)
{
	if(forEmbeddedCode)
		stream << "\"";
	stream << level << " ";
	if(szLabel)
	{
		printEscapedString(stream, szLabel, forEmbeddedCode);
		stream << ":";
	}
	switch(m_type)
	{
		case type_obj:
			{
				stream << "o";
				if(forEmbeddedCode)
					stream << "\\\\n\"\n";
				else
					stream << "\n";
				GTwtObjField* pField;
				reverseFieldOrder();
				for(pField = m_value.m_pFirstField; pField; pField = pField->m_pNext)
					pField->m_pValue->write(stream, pField->m_pName, level + 1, forEmbeddedCode);
				reverseFieldOrder();
			}
			return;
		case type_list:
			{
				stream << "l " << m_value.m_list.m_itemCount;
				if(forEmbeddedCode)
					stream << "\\\\n\"\n";
				else
					stream << "\n";
				for(size_t i = 0; i < m_value.m_list.m_itemCount; i++)
					m_value.m_list.m_items[i]->write(stream, NULL, level + 1, forEmbeddedCode);
			}
			return;
		case type_bool:
			stream << "b " << (m_value.m_bool ? "t" : "f");
			break;
		case type_int:
			stream << "i " << m_value.m_int;
			break;
		case type_double:
			stream << "d " << m_value.m_double;
			break;
		case type_string:
			stream << "s ";
			printEscapedString(stream, m_value.m_string, forEmbeddedCode);
			break;
		default:
			GAssert(false); // unrecognized type
			stream << "* Error, unrecognized node type *";
			break;
	}
	if(forEmbeddedCode)
		stream << "\\\\n\"\n";
	else
		stream << "\n";
}
/*
GXMLTag* GTwtNode::toXml()
{
	GXMLTag* pTag;
	switch(m_type)
	{
		case type_obj:
			{
				pTag = new GXMLTag("obj");
				GTwtObjField* pField;
				reverseFieldOrder();
				for(pField = m_value.m_pFirstField; pField; pField = pField->m_pNext)
				{
					GXMLTag* pChild = pField->m_pValue->toXml();
					pChild->attributes().push_back(new GXMLAttribute("field", pField->m_pName));
					pTag->children().push_back(pChild);
				}
				reverseFieldOrder();
				return pTag;
			}
		case type_list:
			{
				pTag = new GXMLTag("list");
				for(size_t i = 0; i < m_value.m_list.m_itemCount; i++)
					pTag->children().push_back(m_value.m_list.m_items[i]->toXml());
				return pTag;
			}
		case type_bool:
			{
				pTag = new GXMLTag("bool");
				pTag->attributes().push_back(new GXMLAttribute("value", m_value.m_bool ? "true" : "false"));
				return pTag;
			}
		case type_int:
			{
				std::ostringstream os;
				os << m_value.m_int;
				pTag = new GXMLTag("int");
				string tmp = os.str();
				pTag->attributes().push_back(new GXMLAttribute("value", tmp.c_str()));
				return pTag;
			}
		case type_double:
			{
				std::ostringstream os;
				os.precision(14);
				os << m_value.m_double;
				string tmp = os.str();
				pTag = new GXMLTag("double");
				pTag->attributes().push_back(new GXMLAttribute("value", tmp.c_str()));
				return pTag;
			}
		case type_string:
			pTag = new GXMLTag("string");
			pTag->attributes().push_back(new GXMLAttribute("value", m_value.m_string));
			return pTag;
		default:
			ThrowError("unrecognized type");
	}
	return NULL;
}

GTwtNode* GTwtNode::fromXml(GTwtDoc* pDoc, GXMLTag* pTag)
{
	const char* szName = pTag->name();
	if(strcmp(szName, "obj") == 0)
	{
		GTwtNode* pNode = pDoc->newObj();
		for(vector<GXMLTag*>::iterator it = pTag->children().begin(); it != pTag->children().end(); it++)
		{
			GTwtNode* pChildNode = fromXml(pDoc, (*it));
			if(pChildNode)
				pNode->addField(pDoc, (*it)->attrValue("field"), pChildNode);
		}
		return pNode;
	}
	else if(strcmp(szName, "list") == 0)
	{
		deque<GTwtNode*> q;
		for(vector<GXMLTag*>::iterator it = pTag->children().begin(); it != pTag->children().end(); it++)
		{
			GTwtNode* pChildNode = fromXml(pDoc, *it);
			if(pChildNode)
				q.push_back(pChildNode);
		}
		size_t count = q.size();
		GTwtNode* pNode = pDoc->newList(count);
		for(size_t i = 0; i < count; i++)
		{
			pNode->setItem(i, q.front());
			q.pop_front();
		}
		return pNode;
	}
	else if(strcmp(szName, "bool") == 0)
		return pDoc->newBool(stricmp(pTag->attrValue("value"), "true") == 0 ? true : false);
	else if(strcmp(szName, "int") == 0)
	{
#ifdef WINDOWS
		return pDoc->newInt(_atoi64(pTag->attrValue("value")));
#else
		return pDoc->newInt(atoll(pTag->attrValue("value")));
#endif
	}
	else if(strcmp(szName, "double") == 0)
		return pDoc->newDouble(atof(pTag->attrValue("value")));
	else if(strcmp(szName, "string") == 0)
		return pDoc->newString(pTag->attrValue("value"));
	return NULL;
}
*/
// -------------------------------------------------------------------------------

class Bogus1
{
public:
	int m_type;
	double m_double;
};

GTwtNode* GTwtDoc::newObj()
{
	GTwtNode* pNewObj = (GTwtNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(GTwtObjField*));
	pNewObj->m_type = GTwtNode::type_obj;
	pNewObj->m_value.m_pFirstField = NULL;
	return pNewObj;
}

GTwtNode* GTwtDoc::newList(size_t itemCount)
{
	GTwtNode* pNewList = (GTwtNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + offsetof(GTwtList, m_items) + itemCount * sizeof(GTwtObjField*));
	pNewList->m_type = GTwtNode::type_list;
	pNewList->m_value.m_list.m_itemCount = itemCount;
	for(size_t i = 0; i < itemCount; i++)
		pNewList->m_value.m_list.m_items[i] = NULL;
	return pNewList;
}

GTwtNode* GTwtDoc::newBool(bool b)
{
	GTwtNode* pNewBool = (GTwtNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(bool));
	pNewBool->m_type = GTwtNode::type_bool;
	pNewBool->m_value.m_bool = b;
	return pNewBool;
}

GTwtNode* GTwtDoc::newInt(long long n)
{
	GTwtNode* pNewInt = (GTwtNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(long long));
	pNewInt->m_type = GTwtNode::type_int;
	pNewInt->m_value.m_int = n;
	return pNewInt;
}

GTwtNode* GTwtDoc::newDouble(double d)
{
	GTwtNode* pNewDouble = (GTwtNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(double));
	pNewDouble->m_type = GTwtNode::type_double;
	pNewDouble->m_value.m_double = d;
	return pNewDouble;
}

GTwtNode* GTwtDoc::newString(const char* pString, size_t len)
{
	GTwtNode* pNewString = (GTwtNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + len + 1);
	pNewString->m_type = GTwtNode::type_string;
	memcpy(pNewString->m_value.m_string, pString, len);
	pNewString->m_value.m_string[len] = '\0';
	return pNewString;
}

GTwtNode* GTwtDoc::newString(const char* szString)
{
	return newString(szString, strlen(szString));
}

GTwtObjField* GTwtDoc::newField()
{
	return (GTwtObjField*)m_heap.allocAligned(sizeof(GTwtObjField));
}

void GTwtDoc::advanceToNextLine()
{
	while(m_len > 0 && *m_pDoc != '\n')
	{
		m_pDoc++;
		m_len--;
	}
	if(m_len > 0)
	{
		m_pDoc++;
		m_len--;
	}
	m_line++;
}

GTwtNode* GTwtDoc::parseLine(int level, char** pOutLabel)
{
	// Parse the level
	if(m_len <= 0)
		return NULL;
	if(*m_pDoc < '0' || *m_pDoc > '9')
		ThrowError("Expected a level number at the start of line ", to_str(m_line));
	int lev = atoi(m_pDoc);
	if(lev < level)
		return NULL;
	if(lev != level)
		ThrowError("Invalid level at the start of line ", to_str(m_line), ". Expected a value <= ", to_str(level));
	while(m_len > 0 && *m_pDoc != ' ' && *m_pDoc != '\n')
	{
		m_pDoc++;
		m_len--;
	}
	if(m_len <= 1 || *m_pDoc != ' ')
		ThrowError("Expected more on line ", to_str(m_line));
	m_pDoc++;
	m_len--;

	// Parse the label
	if(pOutLabel)
	{
		size_t i;
		for(i = 0; i < m_len && m_pDoc[i] != ':' && m_pDoc[i] != '\n'; i++)
		{
		}
		if(i >= m_len || m_pDoc[i] != ':')
			ThrowError("Expected a ':' on line ", to_str(m_line));
		*pOutLabel = m_heap.add(m_pDoc, i);
		GTwtNode::unescapeString(*pOutLabel, m_line);
		m_pDoc += (i + 1);
		m_len -= (i + 1);
		if(m_len == 0)
			ThrowError("Expected more on line ", to_str(m_line));
	}

	// Parse the type
	GTwtNode* pNewObj = NULL;
	switch(*m_pDoc)
	{
		case 'o':
			{
				m_pDoc++;
				m_len--;
				pNewObj = newObj();
				advanceToNextLine();
				char* pLabel;
				GTwtNode* pValue;
				while(true)
				{
					pValue = parseLine(level + 1, &pLabel);
					if(!pValue)
						break;
					pNewObj->addField(this, pLabel, pValue, false);
				}
			}
			break;
		case 'l':
			{
				m_pDoc += 2;
				m_len -= 2;
				if(m_len <= 0 || *m_pDoc < '0' || *m_pDoc > '9')
					ThrowError("Expected an item count on line ", to_str(m_line));
#ifdef WINDOWS
				size_t itemCount = (size_t)_strtoui64(m_pDoc, (char**)NULL, 10);
#else
				size_t itemCount = strtoull(m_pDoc, (char**)NULL, 10);
#endif
				pNewObj = newList(itemCount);
				advanceToNextLine();
				GTwtNode* pItem;
				for(size_t i = 0; i < itemCount; i++)
				{
					pItem = parseLine(level + 1, NULL);
					if(!pItem)
						ThrowError("Not enough items in list at line ", to_str(m_line));
					pNewObj->setItem(i, pItem);
				}
				if(parseLine(level + 1, NULL))
					ThrowError("Too many items in list at line ", to_str(m_line));
			}
			break;
		case 'b':
			m_pDoc += 2;
			m_len -= 2;
			if(m_len <= 0)
				ThrowError("Expected a 't' or 'f' on line ", to_str(m_line));
			if(*m_pDoc == 't' || *m_pDoc == 'T')
				pNewObj = newBool(true);
			else if(*m_pDoc == 'f' || *m_pDoc == 'F')
				pNewObj = newBool(false);
			else
				ThrowError("Expected a 't' or 'f' on line ", to_str(m_line));
			advanceToNextLine();
			break;
		case 'i':
			m_pDoc += 2;
			m_len -= 2;
			if(m_len <= 0 || (*m_pDoc != '-' && (*m_pDoc < '0' || *m_pDoc > '9')))
				ThrowError("Expected an integer value on line ", to_str(m_line));
#ifdef WINDOWS
			pNewObj = newInt(_atoi64(m_pDoc));
#else
			pNewObj = newInt(strtoll(m_pDoc, (char**)NULL, 10));
#endif
			advanceToNextLine();
			break;
		case 'd':
			m_pDoc += 2;
			m_len -= 2;
			if(m_len <= 0 || (*m_pDoc != '-' && *m_pDoc != '.' && (*m_pDoc < '0' || *m_pDoc > '9')))
				ThrowError("Expected a double value on line ", to_str(m_line));
			pNewObj = newDouble(atof(m_pDoc));
			advanceToNextLine();
			break;
		case 's':
			{
				m_pDoc += 2;
				m_len -= 2;
				size_t i;
				for(i = 0; i < m_len && m_pDoc[i] != '\n' && m_pDoc[i] != '\r'; i++)
				{
				}
				pNewObj = newString(m_pDoc, i);
				GTwtNode::unescapeString(pNewObj->m_value.m_string, m_line);
				m_pDoc += i;
				m_len -= i;
				advanceToNextLine();
			}
			break;
		default:
			ThrowError("unexpected type '", to_str(*m_pDoc), "' at line ", to_str(m_line), ". Expected 'o', 'l', 'b', 'i', 'd', or 's'");
	}
	return pNewObj;
}

void GTwtDoc::parse(const char* pDoc, size_t len)
{
	m_pDoc = pDoc;
	m_line = 1;
	m_len = len;
	m_pRoot = parseLine(0, NULL);
	m_pDoc = NULL;
	m_line = 0;
	m_len = 0;
}

void GTwtDoc::load(const char* szFilename)
{
	size_t len;
	char* pFile = GFile::loadFile(szFilename, &len);
	ArrayHolder<char> hFile(pFile);
	parse(pFile, len);
	if(!m_pRoot)
		ThrowError("Empty document: ", szFilename);
}

void GTwtDoc::write(std::ostream& stream)
{
	if(!m_pRoot)
		ThrowError("No root node has been set");
	stream.precision(14);
	m_pRoot->write(stream, NULL, 0, false);
}

void GTwtDoc::save(const char* szFilename)
{
	std::ofstream os;
	os.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		os.open(szFilename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		ThrowError("Error creating file: ", szFilename);
	}
	write(os);
}

void GTwtDoc::writeForEmbeddedCode(std::ostream& stream)
{
	if(!m_pRoot)
		ThrowError("No root node has been set");
	stream.precision(14);
	stream << "const char* g_change_me =\n";
	m_pRoot->write(stream, NULL, 0, true);
	stream << ";\n";
}

#ifndef NO_TEST_CODE
// static
void GTwtDoc::test()
{
	const char* szTestFile =
		"0 o\n"
		"1 name:s Bob\\nis\\ncool\n"
		"1 pet:o\n"
		"2 hair:s thick\n"
		"2 age:i 12\n"
		"1 acquantances:l 3\n"
		"2 o\n"
		"3 name:s Bill\n"
		"3 age:i 31\n"
		"2 o\n"
		"3 name:s Sally\n"
		"3 age:i 24\n"
		"2 o\n"
		"3 name:s George\n"
		"3 age:i 18\n"
		"1 names:l 2\n"
		"2 s Bob\n"
		"2 s Doe\n"
		"1 male:b t\n"
		"1 height:d 5.8\n"
		"1 temp:d 98.6\n";
	size_t len = strlen(szTestFile);
	GTwtDoc doc;
	doc.parse(szTestFile, len);
}
#endif // NO_TEST_CODE

} // namespace GClasses

