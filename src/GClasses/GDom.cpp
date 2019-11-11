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

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "GDom.h"
#include "GFile.h"
#include "GHolders.h"
#include <vector>
#include <deque>
#include <sstream>
#include <fstream>
#include <map>
#include <errno.h>
#include "GTokenizer.h"
#include "GString.h"


namespace GClasses {

using std::vector;
using std::deque;
using std::map;
using std::string;


class GDomObjField
{
public:
	const char* m_pName;
	GDomNode* m_pValue;
	GDomObjField* m_pPrev;
};

/// An element in a GDom list
class GDomListItem
{
public:
	/// Pointer to the value contained in this list item
	GDomNode* m_pValue;

	/// Pointer to the previous node in the list
	GDomListItem* m_pPrev;
};

class GDomArrayList
{
public:
	/// Total number of elements in this list
	size_t m_size;
	size_t m_capacity;

	/// The items in the list
	GDomNode* m_items[2]; // 2 is a bogus value
};



GDomListIterator::GDomListIterator(const GDomNode* pNode)
{
	if(pNode->m_type != GDomNode::type_list)
		throw Ex(to_str_brief(*pNode), " is not a list type");
	m_pList = pNode;
	m_index = 0;
}

GDomListIterator::~GDomListIterator()
{
}

GDomNode* GDomListIterator::current()
{
	if(!m_pList->m_value.m_pArrayList)
		return nullptr;
	if(m_index < m_pList->m_value.m_pArrayList->m_size)
		return m_pList->m_value.m_pArrayList->m_items[m_index];
	else
		return nullptr;
}

bool GDomListIterator::currentBool()
{
	return current()->asBool();
}

long long GDomListIterator::currentInt()
{
	return current()->asInt();
}

double GDomListIterator::currentDouble()
{
	return current()->asDouble();
}

const char* GDomListIterator::currentString()
{
	return current()->asString();
}

void GDomListIterator::advance()
{
	m_index++;
}

size_t GDomListIterator::remaining()
{
	if(!m_pList->m_value.m_pArrayList)
		return 0;
	return m_pList->m_value.m_pArrayList->m_size - m_index;
}


GDomNode* GDomNode::getIfExists(const char* szName) const
{
	if(m_type != type_obj)
		throw Ex(to_str_brief(*this), " is not an obj");
	GDomObjField* pField;
	for(pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
	{
		if(strcmp(szName, pField->m_pName) == 0)
			return pField->m_pValue;
	}
	return NULL;
}

size_t GDomNode::reverseFieldOrder() const
{
	GAssert(m_type == type_obj);
	size_t count = 0;
	GDomObjField* pNewHead = NULL;
	while(m_value.m_pLastField)
	{
		GDomObjField* pTemp = m_value.m_pLastField;
		((GDomNode*)this)->m_value.m_pLastField = pTemp->m_pPrev;
		pTemp->m_pPrev = pNewHead;
		pNewHead = pTemp;
		count++;
	}
	((GDomNode*)this)->m_value.m_pLastField = pNewHead;
	return count;
}

size_t GDomNode::size() const
{
	GAssert(m_type == type_list);
	if(!m_value.m_pArrayList)
		return 0;
	return m_value.m_pArrayList->m_size;
}

GDomNode* GDomNode::get(size_t index) const
{
	GAssert(m_type == type_list);
	GAssert(index < m_value.m_pArrayList->m_size);
	return m_value.m_pArrayList->m_items[index];
}

GDomNode* GDomNode::set(GDom* pDoc, const char* szName, GDomNode* pNode)
{
	if(m_type != type_obj)
		throw Ex(to_str_brief(*this), " is not an obj");
	GDomObjField* pField;
	for(pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
	{
		if(strcmp(szName, pField->m_pName) == 0)
		{
			pField->m_pValue = pNode;
			return pNode;
		}
	}
	pField = pDoc->newField();
	pField->m_pPrev = m_value.m_pLastField;
	m_value.m_pLastField = pField;
	GHeap* pHeap = pDoc->heap();
	pField->m_pName = pHeap->add(szName);
	pField->m_pValue = pNode;
	return pNode;
}

GDomNode* GDomNode::set(GDom* pDoc, size_t index, GDomNode* pNode)
{
	if(m_type != type_list)
		throw Ex(to_str_brief(*this), " is not a list");
	if(index == m_value.m_pArrayList->m_size)
		return add(pDoc, pNode);
	else if(index < m_value.m_pArrayList->m_size)
	{
		m_value.m_pArrayList->m_items[index] = pNode;
		return pNode;
	}
	else
		throw Ex("tried to set index ", to_str(index), " in a list of size ", to_str(m_value.m_pArrayList->m_size));
}

GDomNode* GDomNode::add(GDom* pDoc, const char* szName, GDomNode* pNode)
{
	if(m_type != type_obj)
		throw Ex(to_str_brief(*this), " is not an obj");
	GDomObjField* pField = pDoc->newField();
	pField->m_pPrev = m_value.m_pLastField;
	m_value.m_pLastField = pField;
	GHeap* pHeap = pDoc->heap();
	pField->m_pName = pHeap->add(szName);
	pField->m_pValue = pNode;
	return pNode;
}

GDomNode* GDomNode::add(GDom* pDoc, const char* szName, bool b)
{
	return add(pDoc, szName, pDoc->newBool(b));
}

GDomNode* GDomNode::add(GDom* pDoc, const char* szName, long long n)
{
	return add(pDoc, szName, pDoc->newInt(n));
}

GDomNode* GDomNode::add(GDom* pDoc, const char* szName, size_t n)
{
	return add(pDoc, szName, pDoc->newInt((long long)n));
}

GDomNode* GDomNode::add(GDom* pDoc, const char* szName, double d)
{
	return add(pDoc, szName, pDoc->newDouble(d));
}

GDomNode* GDomNode::add(GDom* pDoc, const char* szName, const char* str)
{
	return add(pDoc, szName, pDoc->newString(str));
}

GDomNode* GDomNode::add(GDom* pDoc, GDomNode* pNode)
{
	if(m_type != type_list)
		throw Ex(to_str_brief(*this), " is not a list");
	if(!m_value.m_pArrayList || m_value.m_pArrayList->m_size >= m_value.m_pArrayList->m_capacity)
	{
		// Reallocate the array of node pointers
		size_t newCapacity = std::max((size_t)4, (m_value.m_pArrayList ? m_value.m_pArrayList->m_size * 2 : 0));
		GDomArrayList* pArrayList = (GDomArrayList*)pDoc->m_heap.allocAligned(sizeof(size_t) + sizeof(size_t) + sizeof(GDomNode*) * newCapacity);
		if(m_value.m_pArrayList)
		{
			for(size_t i = 0; i < m_value.m_pArrayList->m_size; i++)
				pArrayList->m_items[i] = m_value.m_pArrayList->m_items[i];
			pArrayList->m_size = m_value.m_pArrayList->m_size;
		}
		else
			pArrayList->m_size = 0;
		pArrayList->m_capacity = newCapacity;
		m_value.m_pArrayList = pArrayList;
	}
	m_value.m_pArrayList->m_items[m_value.m_pArrayList->m_size] = pNode;
	m_value.m_pArrayList->m_size++;
	return pNode;
}

GDomNode* GDomNode::add(GDom* pDoc, bool b)
{
	return add(pDoc, pDoc->newBool(b));
}

GDomNode* GDomNode::add(GDom* pDoc, long long n)
{
	return add(pDoc, pDoc->newInt(n));
}

GDomNode* GDomNode::add(GDom* pDoc, size_t n)
{
	return add(pDoc, pDoc->newInt((long long)n));
}

GDomNode* GDomNode::add(GDom* pDoc, double d)
{
	return add(pDoc, pDoc->newDouble(d));
}

GDomNode* GDomNode::add(GDom* pDoc, const char* str)
{
	return add(pDoc, pDoc->newString(str));
}

void GDomNode::del(const char* szField)
{
	if(m_type != type_obj)
		throw Ex(to_str_brief(*this), " is not an obj");
	if(strcmp(m_value.m_pLastField->m_pName, szField) == 0)
	{
		m_value.m_pLastField = m_value.m_pLastField->m_pPrev;
		return;
	}
	else
	{
		for(GDomObjField* pField = m_value.m_pLastField; pField->m_pPrev; pField = pField->m_pPrev)
		{
			if(strcmp(pField->m_pPrev->m_pName, szField) == 0)
			{
				pField->m_pPrev = pField->m_pPrev->m_pPrev;
				return;
			}
		}
		throw Ex("No field named ", szField);
	}
}

void GDomNode::del(size_t index)
{
	if(m_type != type_list)
		throw Ex(to_str_brief(*this), " is not a list");
	if(index >= m_value.m_pArrayList->m_size)
		throw Ex("Index out of range. Index ", to_str(index), ". Size ", to_str(m_value.m_pArrayList->m_size));
	for(size_t i = index; i + 1 < m_value.m_pArrayList->m_size; i++)
		m_value.m_pArrayList->m_items[i] = m_value.m_pArrayList->m_items[i + 1];
	m_value.m_pArrayList->m_size--;
}

void writeJSONString(std::ostream& stream, const char* szString)
{
	stream << '"';
	while(*szString != '\0')
	{
		if(*szString < ' ')
		{
			switch(*szString)
			{
				case '\b': stream << "\\b"; break;
				case '\f': stream << "\\f"; break;
				case '\n': stream << "\\n"; break;
				case '\r': stream << "\\r"; break;
				case '\t': stream << "\\t"; break;
				default:
					stream << (*szString);
			}
		}
		else if(*szString == '\\')
			stream << "\\\\";
		else if(*szString == '"')
			stream << "\\\"";
		else
			stream << (*szString);
		szString++;
	}
	stream << '"';
}

size_t writeJSONStringCpp(std::ostream& stream, const char* szString)
{
	stream << "\\\"";
	size_t chars = 2;
	while(*szString != '\0')
	{
		if(*szString < ' ')
		{
			switch(*szString)
			{
				case '\b': stream << "\\\\b"; break;
				case '\f': stream << "\\\\f"; break;
				case '\n': stream << "\\\\n"; break;
				case '\r': stream << "\\\\r"; break;
				case '\t': stream << "\\\\t"; break;
				default:
					stream << (*szString);
			}
			chars += 3;
		}
		else if(*szString == '\\')
		{
			stream << "\\\\\\\\";
			chars += 4;
		}
		else if(*szString == '"')
		{
			stream << "\\\\\\\"";
			chars += 4;
		}
		else
		{
			stream << (*szString);
			chars++;
		}
		szString++;
	}
	stream << "\\\"";
	chars += 2;
	return chars;
}

void GDomNode::saveJson(const char* filename) const
{
	GDom doc;
	doc.setRoot(this);
	doc.saveJson(filename);
}

void GDomNode::writeJson(std::ostream& stream) const
{
	std::ios_base::fmtflags oldflags = stream.flags();
	stream << std::fixed;
	switch(m_type)
	{
		case type_obj:
			stream << "{";
			reverseFieldOrder();
			for(GDomObjField* pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
			{
				if(pField != m_value.m_pLastField)
					stream << ",";
				writeJSONString(stream, pField->m_pName);
				stream << ":";
				pField->m_pValue->writeJson(stream);
			}
			reverseFieldOrder();
			stream << "}";
			break;
		case type_list:
			stream << "[";
			if(m_value.m_pArrayList)
			{
				if(m_value.m_pArrayList->m_size > 0)
					m_value.m_pArrayList->m_items[0]->writeJson(stream);
				for(size_t i = 1; i < m_value.m_pArrayList->m_size; i++)
				{
					stream << ",";
					m_value.m_pArrayList->m_items[i]->writeJson(stream);
				}
			}
			stream << "]";
			break;
		case type_bool:
			stream << (m_value.m_bool ? "true" : "false");
			break;
		case type_int:
			stream << m_value.m_int;
			break;
		case type_double:
			stream << m_value.m_double;
			break;
		case type_string:
			writeJSONString(stream, m_value.m_string);
			break;
		case type_null:
			stream << "null";
			break;
		default:
			throw Ex("Unrecognized node type");
	}
	stream.flags(oldflags);
}

void newLineAndIndent(std::ostream& stream, size_t indents)
{
	stream << "\n";
	for(size_t i = 0; i < indents; i++)
		stream << "	";
}

void GDomNode::writeJsonPretty(std::ostream& stream, size_t indents) const
{
	std::ios_base::fmtflags oldflags = stream.flags();
	stream << std::fixed;
	switch(m_type)
	{
		case type_obj:
			stream << "{";
			reverseFieldOrder();
			for(GDomObjField* pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
			{
				newLineAndIndent(stream, indents + 1); writeJSONString(stream, pField->m_pName);
				stream << ":";
				pField->m_pValue->writeJsonPretty(stream, indents + 1);
				if(pField->m_pPrev)
					stream << ",";
			}
			reverseFieldOrder();
			newLineAndIndent(stream, indents); stream << "}";
			break;
		case type_list:
			{
				// Check whether all items in the list are atomic
				bool allAtomic = true;
				if(m_value.m_pArrayList)
				{
					if(m_value.m_pArrayList->m_size >= 1024)
						allAtomic = false;
					for(size_t i = 0; i < m_value.m_pArrayList->m_size && allAtomic; i++)
					{
						GDomNode* pNode = m_value.m_pArrayList->m_items[i];
						if(pNode->type() == GDomNode::type_obj || pNode->type() == GDomNode::type_list)
							allAtomic = false;
					}
				}

				// Print the items
				if(allAtomic)
				{
					// All items are atomic, so let's put them all on one line
					stream << "[";
					if(m_value.m_pArrayList)
					{
						for(size_t i = 0; i < m_value.m_pArrayList->m_size; i++)
						{
							if(i > 0)
							{
								stream << ",";
								if(i % 100 == 0)
									newLineAndIndent(stream, indents);
							}
							GDomNode* pNode = m_value.m_pArrayList->m_items[i];
							pNode->writeJson(stream);
						}
					}
					stream << "]";
				}
				else
				{
					// Some items are non-atomic, so let's spread across multiple lines
					newLineAndIndent(stream, indents);
					stream << "[";
					for(size_t i = 0; i < m_value.m_pArrayList->m_size; i++)
					{
						GDomNode* pNode = m_value.m_pArrayList->m_items[i];
						newLineAndIndent(stream, indents + 1);
						pNode->writeJsonPretty(stream, indents + 1);
						if(i + 1 < m_value.m_pArrayList->m_size)
							stream << ",";
					}
					newLineAndIndent(stream, indents);
					stream << "]";
				}
			}
			break;
		case type_bool:
			stream << (m_value.m_bool ? "true" : "false");
			break;
		case type_int:
			stream << m_value.m_int;
			break;
		case type_double:
			stream << m_value.m_double;
			break;
		case type_string:
			writeJSONString(stream, m_value.m_string);
			break;
		case type_null:
			stream << "null";
			break;
		default:
			throw Ex("Unrecognized node type");
	}
	stream.flags(oldflags);
}

size_t GDomNode::writeJsonCpp(std::ostream& stream, size_t col) const
{
	std::ios_base::fmtflags oldflags = stream.flags();
	stream << std::fixed;
	switch(m_type)
	{
		case type_obj:
			stream << "{";
			col++;
			reverseFieldOrder();
			for(GDomObjField* pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
			{
				if(pField != m_value.m_pLastField)
				{
					stream << ",";
					col++;
				}
				if(col >= 200)
				{
					stream << "\"\n\"";
					col = 0;
				}
				col += writeJSONStringCpp(stream, pField->m_pName);
				stream << ":";
				col++;
				col = pField->m_pValue->writeJsonCpp(stream, col);
			}
			reverseFieldOrder();
			stream << "}";
			col++;
			break;
		case type_list:
			stream << "[";
			col++;
			if(m_value.m_pArrayList)
			{
				for(size_t i = 0; i < m_value.m_pArrayList->m_size; i++)
				{
					if(i + 1 < m_value.m_pArrayList->m_size)
					{
						stream << ",";
						col++;
					}
					if(col >= 200)
					{
						stream << "\"\n\"";
						col = 0;
					}
					col = m_value.m_pArrayList->m_items[i]->writeJsonCpp(stream, col);
				}
			}
			stream << "]";
			col++;
			break;
		case type_bool:
			stream << (m_value.m_bool ? "true" : "false");
			col += 4;
			break;
		case type_int:
			stream << m_value.m_int;
			col += 4; // just a guess
			break;
		case type_double:
			stream << m_value.m_double;
			col += 8; // just a guess
			break;
		case type_string:
			col += writeJSONStringCpp(stream, m_value.m_string);
			break;
		case type_null:
			stream << "null";
			col += 4;
			break;
		default:
			throw Ex("Unrecognized node type");
	}
	if(col >= 200)
	{
		stream << "\"\n\"";
		col = 0;
	}
	stream.flags(oldflags);
	return col;
}

bool isXmlInlineType(int type)
{
	if(type == GDomNode::type_string ||
		type == GDomNode::type_int ||
		type == GDomNode::type_double ||
		type == GDomNode::type_bool ||
		type == GDomNode::type_null)
		return true;
	else
		return false;
}

void GDomNode::writeXmlInlineValue(std::ostream& stream)
{
	switch(m_type)
	{
		case type_string:
			stream << m_value.m_string; // todo: escape the string as necessary
			break;
		case type_int:
			stream << m_value.m_int;
			break;
		case type_double:
			stream << m_value.m_double;
			break;
		case type_bool:
			stream << (m_value.m_bool ? "true" : "false");
			break;
		case type_null:
			stream << "null";
			break;
		default:
			throw Ex("Type cannot be inlined");
	}
}

void GDomNode::writeXml(std::ostream& stream, const char* szLabel) const
{
	switch(m_type)
	{
		case type_obj:
			{
			stream << "<" << szLabel;
			reverseFieldOrder();
			size_t nonInlinedChildren = 0;
			for(GDomObjField* pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
			{
				if(isXmlInlineType(pField->m_pValue->m_type))
				{
					stream << " " << pField->m_pName << "=\"";
					pField->m_pValue->writeXmlInlineValue(stream);
					stream << "\"";
				}
				else
					nonInlinedChildren++;
			}
			if(nonInlinedChildren == 0)
				stream << " />";
			else
			{
				stream << ">";
				for(GDomObjField* pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
				{
					if(!isXmlInlineType(pField->m_pValue->m_type))
						pField->m_pValue->writeXml(stream, pField->m_pName);
				}
				stream << "</" << szLabel << ">";
			}
			reverseFieldOrder();
			}
			return;
		case type_list:
			stream << "<" << szLabel << ">";
			if(m_value.m_pArrayList)
			{
				for(size_t i = 0; i < m_value.m_pArrayList->m_size; i++)
				{
					GDomNode* pNode = m_value.m_pArrayList->m_items[i];
					pNode->writeXml(stream, "i");
				}
			}
			stream << "</" << szLabel << ">";
			return;
		case type_bool:
			stream << "<" << szLabel << ">";
			stream << (m_value.m_bool ? "true" : "false");
			stream << "</" << szLabel << ">";
			break;
		case type_int:
			stream << "<" << szLabel << ">";
			stream << m_value.m_int;
			stream << "</" << szLabel << ">";
			break;
		case type_double:
			stream << "<" << szLabel << ">";
			stream << m_value.m_double;
			stream << "</" << szLabel << ">";
			break;
		case type_string:
			stream << "<" << szLabel << ">";
			stream << m_value.m_string; // todo: escape the string as necessary
			stream << "</" << szLabel << ">";
			break;
		case type_null:
			stream << "<" << szLabel << ">";
			stream << "null";
			stream << "</" << szLabel << ">";
			break;
		default:
			throw Ex("Unrecognized node type");
	}
}

bool GDomNode::isEqual(const GDomNode* pOther) const
{
	switch(m_type)
	{
		case type_obj:
			if(pOther->m_type != type_obj)
				return false;
			{
				GDomObjField* pOth = pOther->m_value.m_pLastField;
				for(GDomObjField* pField = m_value.m_pLastField; pField; pField = pField->m_pPrev)
				{
					if(pOth && strcmp(pField->m_pName, pOth->m_pName) == 0)
					{
						if(!pField->m_pValue->isEqual(pOth->m_pValue))
							return false;
					}
					else
					{
						for(pOth = pOther->m_value.m_pLastField; pOth; pOth = pOth->m_pPrev)
						{
							if(strcmp(pField->m_pName, pOth->m_pName) == 0)
							{
								if(!pField->m_pValue->isEqual(pOth->m_pValue))
									return false;
								break;
							}
						}
					}
					if(pOth)
						pOth = pOth->m_pPrev;
				}
			}
			return true;

		case type_list:
			if(pOther->m_type != type_list)
				return false;
			if(size() != pOther->size())
				return false;
			for(size_t i = 0; i < size(); i++)
			{
				if(!get(i)->isEqual(pOther->get(i)))
					return false;
			}
			return true;

		case type_bool:
			if(pOther->m_type != type_bool)
				return false;
			if(m_value.m_bool == pOther->m_value.m_bool)
				return true;
			else
				return false;

		case type_int:
			if(pOther->m_type != type_int)
				return false;
			if(m_value.m_int == pOther->m_value.m_int)
				return true;
			else
				return false;

		case type_double:
			if(pOther->m_type != type_double)
				return false;
			if(m_value.m_double == pOther->m_value.m_double)
				return true;
			else
				return false;

		case type_string:
			if(pOther->m_type != type_string)
				return false;
			if(strcmp(m_value.m_string, pOther->m_value.m_string) == 0)
				return true;
			else
				return false;

		case type_null:
			if(pOther->m_type != type_bool)
				return false;
			else
				return true;

		default:
			throw Ex("Unrecognized node type");
	}
}


// -------------------------------------------------------------------------------

class GJsonTokenizer : public GTokenizer
{
public:
	GCharSet m_whitespace, m_real, m_quot;

	GJsonTokenizer(const char* szFilename) : GTokenizer(szFilename),
	m_whitespace("\t\n\r "), m_real("-.+0-9eE"), m_quot("\"") {}
	GJsonTokenizer(const char* pFile, size_t len) : GTokenizer(pFile, len),
	m_whitespace("\t\n\r "), m_real("-.+0-9eE"), m_quot("\"") {}
	virtual ~GJsonTokenizer() {}
};

class Bogus1
{
public:
	char m_type;
	double m_double;
};

GDom::GDom()
: m_heap(2000), m_pRoot(NULL), m_line(0), m_len(0), m_pDoc(NULL)
{
}

GDom::~GDom()
{
}

void GDom::clear()
{
	m_pRoot = nullptr;
	m_heap.clear();
}

GDomNode* GDom::newObj()
{
	GDomNode* pNewObj = (GDomNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(GDomObjField*));
	pNewObj->m_type = GDomNode::type_obj;
	pNewObj->m_value.m_pLastField = nullptr;
	return pNewObj;
}

GDomNode* GDom::newList()
{
	GDomNode* pNewList = (GDomNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(GDomListItem*));
	pNewList->m_type = GDomNode::type_list;
	pNewList->m_value.m_pArrayList = nullptr;
	return pNewList;
}

GDomNode* GDom::newNull()
{
	GDomNode* pNewNull = (GDomNode*)m_heap.allocAligned(offsetof(Bogus1, m_double));
	pNewNull->m_type = GDomNode::type_null;
	return pNewNull;
}

GDomNode* GDom::newBool(bool b)
{
	GDomNode* pNewBool = (GDomNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(bool));
	pNewBool->m_type = GDomNode::type_bool;
	pNewBool->m_value.m_bool = b;
	return pNewBool;
}

GDomNode* GDom::newInt(long long n)
{
	GDomNode* pNewInt = (GDomNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(long long));
	pNewInt->m_type = GDomNode::type_int;
	pNewInt->m_value.m_int = n;
	return pNewInt;
}

GDomNode* GDom::newDouble(double d)
{
	if(d >= -1.5e308 && d <= 1.5e308)
	{
		GDomNode* pNewDouble = (GDomNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + sizeof(double));
		pNewDouble->m_type = GDomNode::type_double;
		pNewDouble->m_value.m_double = d;
		return pNewDouble;
	}
	else
	{
		throw Ex("Invalid value: ", to_str(d));
		return NULL;
	}
}

GDomNode* GDom::newString(const char* pString, size_t len)
{
	GDomNode* pNewString = (GDomNode*)m_heap.allocAligned(offsetof(Bogus1, m_double) + len + 1);
	pNewString->m_type = GDomNode::type_string;
	memcpy(pNewString->m_value.m_string, pString, len);
	pNewString->m_value.m_string[len] = '\0';
	return pNewString;
}

GDomNode* GDom::newString(const char* szString)
{
	return newString(szString, strlen(szString));
}

GDomObjField* GDom::newField()
{
	return (GDomObjField*)m_heap.allocAligned(sizeof(GDomObjField));
}

GDomNode* GDom::clone(GDomNode* pNode)
{
	switch(pNode->type())
	{
		case GDomNode::type_obj:
			{
				GDomNode* pClone = newObj();
				for(GDomObjField* pField = pNode->m_value.m_pLastField; pField; pField = pField->m_pPrev)
				{
					GDomObjField* pFieldClone = newField();
					pFieldClone->m_pPrev = pClone->m_value.m_pLastField;
					pClone->m_value.m_pLastField = pFieldClone;
					pFieldClone->m_pName = m_heap.add(pField->m_pName);
					pFieldClone->m_pValue = clone(pField->m_pValue);
				}
				return pClone;
			}
		case GDomNode::type_list:
			{
				GDomNode* pClone = newList();
				for(size_t i = 0; i < pNode->size(); i++)
					pClone->add(this, clone(pNode->get(i)));
				return pClone;
			}
		case GDomNode::type_bool: return newBool(pNode->asBool());
		case GDomNode::type_int: return newInt(pNode->asInt());
		case GDomNode::type_double: return newDouble(pNode->asDouble());
		case GDomNode::type_string: return newString(pNode->asString());
		case GDomNode::type_null: return newNull();;
		default: throw Ex("Unexpected type");
	}
}

char* GDom::loadJsonString(GJsonTokenizer& tok)
{
	tok.skipExact("\"");
	char* szTok = tok.readUntil_escaped('\\', tok.m_quot);
	tok.skip(1);
	size_t eat = 0;
	char* szString = szTok;
	while(szString[eat] != '\0')
	{
		char c = szString[eat];
		if(c == '\\')
		{
			switch(szString[eat + 1])
			{
				case '"': c = '"'; break;
				case '\\': c = '\\'; break;
				case '/': c = '/'; break;
				case 'b': c = '\b'; break;
				case 'f': c = '\f'; break;
				case 'n': c = '\n'; break;
				case 'r': c = '\r'; break;
				case 't': c = '\t'; break;
				case 'u':
					GAssert(false); // Escaped unicode characters are not yet supported
					c = '_';
					eat += 3;
					break;
				default:
					throw Ex("Unrecognized escape sequence");
			}
			eat++;
		}
		*szString = c;
		szString++;
	}
	*szString = '\0';
	return szTok;
}

GDomNode* GDom::loadJsonObject(GJsonTokenizer& tok)
{
	tok.skipExact("{");
	GDomNode* pNewObj = newObj();
	bool readyForField = true;
	GCharSet& whitespace = tok.m_whitespace;
	while(tok.has_more())
	{
		tok.skipWhile(whitespace);
		char c = tok.peek();
		if(c == '}')
		{
			tok.skip(1);
			break;
		}
		else if(c == ',')
		{
			if(readyForField)
				throw Ex("Unexpected ',' in JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
			tok.skip(1);
			readyForField = true;
		}
		else if(c == '\"')
		{
			if(!readyForField)
				throw Ex("Expected a ',' before the next field in JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
			GDomObjField* pNewField = newField();
			pNewField->m_pPrev = pNewObj->m_value.m_pLastField;
			pNewObj->m_value.m_pLastField = pNewField;
			pNewField->m_pName = m_heap.add(loadJsonString(tok));
			tok.skipWhile(whitespace);
			tok.skipExact(":");
			tok.skipWhile(whitespace);
			pNewField->m_pValue = loadJsonValue(tok);
			readyForField = false;
		}
		else if(c == '\0')
			throw Ex("Expected a matching '}' in JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
		else
			throw Ex("Expected a '}' or a '\"' at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
	}
	return pNewObj;
}

GDomNode* GDom::loadJsonArray(GJsonTokenizer& tok)
{
	tok.skipExact("[");
	GDomNode* pNewList = newList();
	bool readyForValue = true;
	while(tok.has_more())
	{
		tok.skipWhile(tok.m_whitespace);
		char c = tok.peek();
		if(c == ']')
		{
			tok.skip(1);
			break;
		}
		else if(c == ',')
		{
			if(readyForValue)
				throw Ex("Unexpected ',' in JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
			tok.skip(1);
			readyForValue = true;
		}
		else if(c == '\0')
			throw Ex("Expected a matching ']' in JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
		else
		{
			if(!readyForValue)
				throw Ex("Expected a ',' or ']' in JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
			GDomNode* pValue = loadJsonValue(tok);
			pNewList->add(this, pValue);
			readyForValue = false;
		}
	}
	return pNewList;
}

GDomNode* GDom::loadJsonNumber(GJsonTokenizer& tok)
{
	char* szString = tok.readWhile(tok.m_real);
	bool hasPeriod = false;
	for(char* szChar = szString; *szChar != '\0'; szChar++)
	{
		if(*szChar == '.')
			hasPeriod = true;
	}
	if(hasPeriod)
		return newDouble(atof(szString));
	else
	{
#ifdef WINDOWS
		return newInt(_atoi64(szString));
#else
		return newInt(strtoll(szString, (char**)NULL, 10));
#endif
	}
}

GDomNode* GDom::loadJsonValue(GJsonTokenizer& tok)
{
	char c = tok.peek();
	if(c == '"')
		return newString(loadJsonString(tok));
	else if(c == '{')
		return loadJsonObject(tok);
	else if(c == '[')
		return loadJsonArray(tok);
	else if(c == 't')
	{
		tok.skipExact("true");
		return newBool(true);
	}
	else if(c == 'f')
	{
		tok.skipExact("false");
		return newBool(false);
	}
	else if(c == 'n')
	{
		tok.skipExact("null");
		return newNull();
	}
	else if((c >= '0' && c <= '9') || c == '-')
		return loadJsonNumber(tok);
	else if(c == '\0')
	{
		throw Ex("Unexpected end of file while parsing JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
		return NULL;
	}
	else
	{
		throw Ex("Unexpected token, \"", to_str(c), "\", while parsing JSON file at line ", to_str(tok.line()), ", col ", to_str(tok.col()));
		return NULL;
	}
}

void GDom::parseJson(const char* pJsonString, size_t len)
{
	GJsonTokenizer tok(pJsonString, len);
	tok.skipWhile(tok.m_whitespace);
	setRoot(loadJsonValue(tok));
}

void GDom::loadJson(const char* szFilename)
{
	GJsonTokenizer tok(szFilename);
	tok.skipWhile(tok.m_whitespace);
	setRoot(loadJsonValue(tok));
}

void GDom::writeJson(std::ostream& stream) const
{
	if(!m_pRoot)
		throw Ex("No root node has been set");
	stream.precision(14);
	m_pRoot->writeJson(stream);
}

void GDom::writeJsonPretty(std::ostream& stream) const
{
	if(!m_pRoot)
		throw Ex("No root node has been set");
	stream.precision(14);
	m_pRoot->writeJsonPretty(stream, 0);
}

void GDom::writeJsonCpp(std::ostream& stream) const
{
	if(!m_pRoot)
		throw Ex("No root node has been set");
	stream.precision(14);
	stream << "const char* g_rename_me = \"";
	m_pRoot->writeJsonCpp(stream, 0);
	stream << "\";\n\n";
}

void GDom::saveJson(const char* szFilename) const
{
	std::ofstream os;
	os.exceptions(std::ios::badbit | std::ios::failbit);
	try
	{
		os.open(szFilename, std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to create the file, ", szFilename, ". ", strerror(errno));
	}
	writeJson(os);
}

void GDom::writeXml(std::ostream& stream) const
{
	if(!m_pRoot)
		throw Ex("No root node has been set");
	stream.precision(14);
	stream << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>";
	m_pRoot->writeXml(stream, "root");
}

std::string to_str(const GDomNode& node)
{
	std::ostringstream os;
	node.writeJsonPretty(os, 0);
	return os.str();
}

std::string to_str_brief(const GDomNode& node)
{
	std::ostringstream os;
	node.writeJsonPretty(os, 0);
	std::string s = os.str();
	if(s.length() < 60)
		return s;
	std::string s2 = s.substr(0, 30);
	s2 += " ... ";
	s2 += s.substr(s.length() - 30, 30);
	return s2;
}


std::string to_str(const GDom& doc)
{
	std::ostringstream os;
	doc.writeJsonPretty(os);
	return os.str();
}

// static
void GDom::test()
{
	const char* szTestFile =
		"{\n"
		"	\"name\":\"Bob\\nis\\\\cool\",\n"
		"	\"pet\":{\n"
		"		\"hair\":\"thick\",\n"
		"		\"age\":12\n"
		"	},\n"
		"	\"acquantances\":[\n"
		"		{\n"
		"			\"name\":\"Bill\",\n"
		"			\"age\":31\n"
		"		},\n"
		"		{\n"
		"			\"name\":\"Sally\",\n"
		"			\"age\":24\n"
		"		},\n"
		"		{\n"
		"			\"name\":\"George\",\n"
		"			\"age\":18\n"
		"		}\n"
		"	],\n"
		"	\"names\":[\n"
		"		\"Bob\",\n"
		"		\"Doe\"\n"
		"	],\n"
		"	\"male\"  :  true , \n"
		"	\"height\": 5.8,\n"
		"	\"temp\":98.6\n"
		"}\n";
	GDom doc;
	doc.parseJson(szTestFile, strlen(szTestFile));
}

















GJsonAsADatabase::GJsonAsADatabase(const char* szBasePath)
: m_basePath(szBasePath)
{
}

GJsonAsADatabase::~GJsonAsADatabase()
{
	flush(true);
}

void GJsonAsADatabase::flush(bool clear_cache)
{
	map<string, GJaadDom*>::iterator it = m_doms.begin();
	while(it != m_doms.end())
	{
		const string& filename = it->first;
		GJaadDom* pDom = it->second;
		if(pDom->modCount() > 0)
		{
			string fullpath = m_basePath;
			fullpath += filename;
			pDom->saveJson(fullpath.c_str());
			pDom->resetModCount();
		}
		++it;
	}
	if(clear_cache)
	{
		map<string, GJaadDom*>::iterator it = m_doms.begin();
		while(it != m_doms.end())
		{
			GJaadDom* pDom = it->second;
			delete(pDom);
			++it;
		}
		m_doms.clear();
	}
}

// static
size_t GJsonAsADatabase::findTok(const char* s, char tok, size_t start)
{
	bool q = false;
	bool qq = false;
	size_t esc = 0;
	size_t brace = 0;
	size_t brack = 0;
	size_t i;
	for(i = start; s[i] != '\0'; i++)
	{
		if(!q && !qq)
		{
			if(s[i] == tok && brace == 0 && brack == 0)
				break;
			else if(s[i] == '[')
				brace++;
			else if(s[i] == ']')
				brace--;
			else if(s[i] == '{')
				brack++;
			else if(s[i] == '}')
				brack--;
			else if(s[i] == '\'')
				q = true;
			else if(s[i] == '"')
				qq = true;
		}
		else
		{
			if(esc > 0)
				esc--;
			else if(s[i] == '\\')
				esc = 2;
			else if(q && s[i] == '\'')
				q = false;
			else if(qq && s[i] == '"')
				qq = false;
		}
	}
	return i;
}


GDomNode* GJsonAsADatabase::findNode(GDom* pDoc, GDomNode* pOb, GDom* pResponseDom, const char* szCmd, char op)
{
	size_t pos = 0;
	while(true)
	{
		// Eat whitespace
		while(szCmd[pos] != '\0' && szCmd[pos] <= ' ')
			pos++;
		char c = szCmd[pos];
		if(c == '\0')
			break;

		if(c == '.')
		{
			size_t i;
			for(i = pos + 1; szCmd[i] != '\0' && szCmd[i] != '.' && szCmd[i] != '['; i++)
			{
			}
			string field(szCmd + pos + 1, i - (pos + 1));
			field = trim(field);
			if(field.length() == 0)
				throw Ex("Expected a field name after '.'");
			if(!pOb && op != '\0')
				pOb = (GDomNode*)pDoc->setRoot(pDoc->newObj());
			if(pOb->type() != GDomNode::type_obj)
				throw Ex("'.' can only be used on object types.");
			GDomNode* pChild = pOb->getIfExists(field.c_str());
			if(!pChild)
			{
				if(op == '\0')
					throw Ex("No field named ", field);
				else if(szCmd[i] == '.') // the next connector will be a '.'
					pChild = pOb->add(pDoc, field.c_str(), pDoc->newObj());
				else if(szCmd[i] == '[') // the next connector will be a '['
					pChild = pOb->add(pDoc, field.c_str(), pDoc->newList());
				else if(op == '+') // There is no next connector, so use the operator
					pChild = pOb->add(pDoc, field.c_str(), pDoc->newList());
				else
					throw Ex("Cannot determine the type for missing field ", field);
			}
			pOb = pChild;
			pos = i;
		}
		else if(c == '[')
		{
			if(pOb && pOb->type() != GDomNode::type_list)
				throw Ex("'[]' can only be used with list types");

			// Find the corresponding ']'
			size_t i = findTok(szCmd, ']', pos + 1);
			size_t indStart = pos + 1;
			while(indStart < i && szCmd[indStart] <= ' ')
				indStart++;
			if(indStart == i)
				throw Ex("Expected an index or equation after '['");
			string sIndex(szCmd + indStart, i - indStart);
			if(!pOb && op != '\0')
				pOb = (GDomNode*)pDoc->setRoot(pDoc->newList());

			// Parse the index
			char c = sIndex[0];
			if(c >= '0' && c <= '9')
			{
				// It's a numerical index
				std::stringstream sstream(sIndex);
				size_t index;
				sstream >> index;
				if(index >= pOb->size())
					throw Ex("index out of range. Index: ", to_str(index), ". Size: ", to_str(pOb->size()));
				pOb = pOb->get(index);
			}
			else
			{
				// It's an equation
				if(c != '.' && c != '[' && c != '=')
					throw Ex("Unexpected index format");

				// Find the "=="
				size_t eq = findTok(sIndex.c_str(), '=');
				if(eq >= sIndex.length())
					throw Ex("Expected an '=' in an equation");
				string pre(sIndex, 0, eq);
				if(sIndex.length() > eq && sIndex[eq + 1] == '=')
					eq++;
				pResponseDom->parseJson(sIndex.c_str() + eq + 1, sIndex.length() - eq - 1);
				GDomNode* pPost = pResponseDom->root();
				size_t j;
				for(j = 0; j < pOb->size(); j++)
				{
					GDomNode* pCand = pOb->get(j);
					GDomNode* pComp = findNode(pDoc, pCand, pResponseDom, pre.c_str(), op);
					if(pComp->isEqual(pPost))
					{
						pOb = pCand;
						break;
					}
				}
				if(j >= pOb->size())
					throw Ex("No matching node found");
			}

			// Move past the ']'
			pos = i;
			if(szCmd[pos] == ']')
				pos++;
		}
		else
			throw Ex("Unexpected token: ", string(szCmd + pos, 1));
	}
	return pOb;
}

GDomNode* GJsonAsADatabase::findLValue(GDom* pDoc, GDomNode* pOb, GDom* pResponseDom, const char* szCmd, string* pOutField, size_t* pOutIndex)
{
	GDomNode* pPrevOb = nullptr;
	size_t pos = 0;
	while(true)
	{
		// Eat whitespace
		while(szCmd[pos] != '\0' && szCmd[pos] <= ' ')
			pos++;
		char c = szCmd[pos];
		if(c == '\0')
			break;

		pPrevOb = pOb;
		if(c == '.')
		{
			size_t i;
			for(i = pos + 1; szCmd[i] != '\0' && szCmd[i] != '.' && szCmd[i] != '['; i++)
			{
			}
			string field(szCmd + pos + 1, i - (pos + 1));
			if(pOutField)
				*pOutField = field;
			if(field.length() == 0)
				throw Ex("Expected a field name after '.'");
			if(pOb->type() != GDomNode::type_obj)
				throw Ex("'.' can only be used on object types.");
			pOb = pOb->getIfExists(field.c_str());
			if(!pOb)
			{
				size_t nextPos = i;
				while(szCmd[nextPos] != '\0' && szCmd[nextPos] <= ' ')
					nextPos++;
				if(szCmd[nextPos] == '.')
					pOb = pOb->add(pDoc, field.c_str(), pDoc->newObj());
				else if(szCmd[nextPos] == '[')
					pOb = pOb->add(pDoc, field.c_str(), pDoc->newList());
				else if(szCmd[nextPos] == '\0')
					return pOb;
				else
					throw Ex("Unexpected token: ", string(szCmd + nextPos, 1));
			}
			pos = i;
		}
		else if(c == '[')
		{
			if(pOb->type() != GDomNode::type_list)
				throw Ex("'[]' can only be used with list types");

			// Find the corresponding ']'
			size_t i = findTok(szCmd, ']', pos + 1);
			size_t indStart = pos + 1;
			while(indStart < i && szCmd[indStart] <= ' ')
				indStart++;
			if(indStart == i)
				throw Ex("Expected an index or equation after '['");
			string sIndex(szCmd + indStart, i - indStart);

			// Parse the index
			char c = sIndex[0];
			if(c >= '0' && c <= '9')
			{
				// It's a numerical index
				std::stringstream sstream(sIndex);
				size_t index;
				sstream >> index;
				if(pOutIndex)
					*pOutIndex = index;
				if(index > pOb->size())
					throw Ex("index out of range. Index: ", to_str(index), ". Size: ", to_str(pOb->size()));
				if(index == pOb->size())
				{
					size_t nextPos = i;
					if(szCmd[nextPos] == ']')
						nextPos++;
					while(szCmd[nextPos] != '\0' && szCmd[nextPos] <= ' ')
						nextPos++;
					if(szCmd[nextPos] == '.')
						pOb = pOb->add(pDoc, pDoc->newObj());
					else if(szCmd[nextPos] == '[')
						pOb = pOb->add(pDoc, pDoc->newList());
					else if(szCmd[nextPos] == '\0')
						return pOb;
					else
						throw Ex("Unexpected token: ", string(szCmd + nextPos, 1));
				}
				pOb = pOb->get(index);
			}
			else
			{
				// It's an equation
				if(c != '.' && c != '[' && c != '=')
					throw Ex("Unexpected index format");

				// Find the "=="
				size_t eq = findTok(sIndex.c_str(), '=');
				if(eq >= sIndex.length())
					throw Ex("Expected an '=' in an equation");
				string pre(sIndex, 0, eq);
				if(sIndex.length() > eq && sIndex[eq + 1] == '=')
					eq++;
				pResponseDom->parseJson(sIndex.c_str() + eq + 1, sIndex.length() - eq - 1);
				GDomNode* pPost = pResponseDom->root();
				size_t j;
				for(j = 0; j < pOb->size(); j++)
				{
					GDomNode* pCand = pOb->get(j);
					GDomNode* pComp = findNode(pDoc, pCand, pResponseDom, pre.c_str(), '=');
					if(pComp->isEqual(pPost))
					{
						if(pOutIndex)
							*pOutIndex = j;
						pOb = pCand;
						break;
					}
				}
				if(j >= pOb->size())
					throw Ex("No matching node found");
			}

			// Move past the ']'
			pos = i;
			if(szCmd[pos] == ']')
				pos++;
		}
		else
			throw Ex("Unexpected token: ", string(szCmd + pos, 1));
	}
	return pPrevOb;
}

GJaadDom* GJsonAsADatabase::getDom(const char* szFile)
{
	map<string, GJaadDom*>::iterator it = m_doms.find(szFile);
	if(it == m_doms.end())
	{
		// Load the DOM from file
		GJaadDom* pDoc = new GJaadDom();
		Holder<GJaadDom> hDoc(pDoc);
		string filename = m_basePath;
		filename += szFile;
		if(GFile::doesFileExist(filename.c_str()))
			pDoc->loadJson(filename.c_str());
		else
		{
			PathData pd;
			GFile::parsePath(filename.c_str(), &pd);
			std::string dir = filename.substr(0, pd.fileStart);
			if(!GFile::doesDirExist(dir.c_str()))
				throw Ex("No such directory: ", dir.substr(m_basePath.length()));
		}
		hDoc.release();
		m_doms.insert(std::pair<string, GJaadDom*>(szFile, pDoc));
		return pDoc;
	}
	else
		return it->second;
}

void GJsonAsADatabase::add(GDomNode* pRequest, GDom* pDoc, GDomNode* pOb)
{
	if(pOb->type() == GDomNode::type_obj)
	{
		const char* szName = pRequest->getString("name");
		GDomNode* pValue = pRequest->get("val");
		pOb->add(pDoc, szName, pDoc->clone(pValue));
	}
	else if(pOb->type() == GDomNode::type_list)
	{
		GDomNode* pValue = pRequest->get("val");
		pOb->add(pDoc, pDoc->clone(pValue));
	}
	else
		throw Ex("An object or list type is needed for add");
}

void GJsonAsADatabase::del(GDomNode* pRequest, GDom* pDoc, GDomNode* pOb)
{
	if(pOb->type() == GDomNode::type_obj)
	{
		const char* szName = pRequest->getString("name");
		pOb->del(szName);
	}
	else if(pOb->type() == GDomNode::type_list)
	{
		size_t index = pRequest->getInt("index");
		pOb->del(index);
	}
	else
		throw Ex("An object or list type is needed for del");
}

const GDomNode* GJsonAsADatabase::apply(const char* szFilename, const char* szCmd, GDom* pResponseDom)
{
	// Find or load the DOM
	GJaadDom* pDoc = getDom(szFilename);

	// Do the command
	size_t op = findTok(szCmd, '='); // possible bug: the '==' operator should not be detected here
	if(szCmd[op] == '\0')
	{
		// No '=', so the user must be requesting an object be returned
		GDomNode* pNode = findNode(pDoc, pDoc->root(), pResponseDom, szCmd, '\0');
		return pNode;
	}
	pDoc->incModCount();
	if(op > 0 && szCmd[op - 1] == '+')
	{
		// It's a "+=" operation
		string sLeft(szCmd, op - 1);
		GDomNode* pNode = findNode(pDoc, pDoc->root(), pResponseDom, sLeft.c_str(), '+');
		if(pNode)
		{
			if(pNode->type() != GDomNode::type_list)
				throw Ex("The left part before '+=' is not a list type");
			GDomNode* pOldRoot = pDoc->root();
			pDoc->parseJson(szCmd + op + 1, strlen(szCmd + op + 1));
			pNode->add(pDoc, pDoc->root());
			pDoc->setRoot(pOldRoot);
		}
		else
		{
			// Empty document. Make a root list node.
			pNode = pDoc->newList();
			pDoc->parseJson(szCmd + op + 1, strlen(szCmd + op + 1));
			pNode->add(pDoc, pDoc->root());
			pDoc->setRoot(pNode);
		}
		return nullptr;
	}
	else if(op > 0 && szCmd[op - 1] == '-')
	{
		// It's a "-=" operation
		string sLeft(szCmd, op - 1);
		GDomNode* pNode = findNode(pDoc, pDoc->root(), pResponseDom, sLeft.c_str(), '-');
		if(pNode)
		{
			string sField;
			size_t index;
			GDomNode* pVal = findLValue(pDoc, pNode, pResponseDom, szCmd + op + 1, &sField, &index);
			if(pVal->type() == GDomNode::type_list)
			{
				pVal->del(index);
				return nullptr;
			}
			else if(pVal->type() == GDomNode::type_obj)
			{
				pVal->del(sField.c_str());
				return nullptr;
			}
			else
				throw Ex("Can only remove from a list or object type");
		}
		else
			throw Ex("No document to remove from");
	}
	else
	{
		// It must be an "=" operation
		string sLeft(szCmd, op);
		string sField;
		size_t index;
		GDomNode* pNode = findLValue(pDoc, pDoc->root(), pResponseDom, sLeft.c_str(), &sField, &index);
		if(!pNode)
			pDoc->parseJson(szCmd + op + 1, strlen(szCmd + op + 1));
		else if(pNode->type() == GDomNode::type_obj)
		{
			GDomNode* pOldRoot = pDoc->root();
			pDoc->parseJson(szCmd + op + 1, strlen(szCmd + op + 1));
			GDomNode* pVal = pDoc->root();
			pDoc->setRoot(pOldRoot);
			pNode->set(pDoc, sField.c_str(), pVal);
		}
		else if(pNode->type() == GDomNode::type_list)
		{
			GDomNode* pOldRoot = pDoc->root();
			pDoc->parseJson(szCmd + op + 1, strlen(szCmd + op + 1));
			GDomNode* pVal = pDoc->root();
			pDoc->setRoot(pOldRoot);
			pNode->set(pDoc, index, pVal);
		}
		else
			throw Ex("Not an LValue");
		return nullptr;
	}
}




} // namespace GClasses
