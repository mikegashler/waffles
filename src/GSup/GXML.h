/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GXML_H__
#define __GXML_H__

#include <stddef.h>
#include "../GClasses/GError.h"
#include <vector>
#include <ostream>

namespace GClasses {

class GXMLTag;

/// Represents an attribute in an XML DOM
class GXMLAttribute
{
friend class GXMLTag;
protected:
	char* m_pName;
	char* m_pValue;

public:
	GXMLAttribute(const char* szName, const char* szValue);
	GXMLAttribute(const char* pName, int nNameLength, const char* pValue);
	GXMLAttribute(const char* pName, const char* pValue, int nValueLength);

	virtual ~GXMLAttribute();

	const char* name() { return m_pName; }
	const char* value() { return m_pValue; }

	void setName(const char* szName);
	void setValue(const char* szValue);
	
protected:
	int toString(char* pBuffer, bool bEscapeQuotes);
};

/// Represents a tag in an XML DOM
class GXMLTag
{
protected:
	char* m_pName;
#ifdef _DEBUG
	GXMLAttribute* m_DEBUG_ONLY_attributes[4];
#endif
	std::vector<GXMLTag*> m_children;
	std::vector<GXMLAttribute*> m_attributes;
	GXMLTag* m_pParent;
	int m_nLineNumber;
	unsigned int m_nColumnAndWidth;

public:
	GXMLTag(const char* szName);
	GXMLTag(const char* pName, int nLength);
	virtual ~GXMLTag();

	const char* name() { return m_pName; }
	GXMLTag* parentTag() { return m_pParent; }
	std::vector<GXMLTag*>& children() { return m_children; }
	std::vector<GXMLAttribute*>& attributes() { return m_attributes; }
	int lineNumber() { return m_nLineNumber; }
	GXMLTag* childTag(const char* name);

	/// Returns NULL if it doesn't exist
	const char* attrValueIfExists(const char* name);

	/// Throws if it doesn't exist
	const char* attrValue(const char* name);

	void setColumnAndWidth(int nColumn, int nWidth)
	{
		m_nColumnAndWidth = ((nColumn & 0xffff)) << 16 | (nWidth & 0xffff);
	}

	void offsetAndWidth(int* pnColumn, int* pnWidth)
	{
		*pnWidth = m_nColumnAndWidth & 0xffff;
		*pnColumn = ((m_nColumnAndWidth >> 16) & 0xffff);
	}

	/// Sets the name of the tag
	void setName(const char* szName);

	/// Set the line number for this XML tag
	void setLineNumber(int nLineNumber) { m_nLineNumber = nLineNumber; }

	/// Convert the entire XML tree to a string. You are responsible to delete the string this returns
	char* toString();

	/// You are responsible to delete the tag this returns
	static GXMLTag* fromString(const char* pBuffer, int nSize, const char** pszErrorMessage = NULL, int* pnErrorOffset = NULL, int* pnErrorLine = NULL, int* pnErrorColumn = NULL);

	/// Save the XML tree to the specified file
	void toFile(const char* szFilename);

	/// Save the XML tree to the specified stream
	void toFile(std::ostream& os);

	/// Save the XML tree to a text file with C++ syntax so you can embed the XML in a .CPP file
	void toCppFile(const char* szFilename, const char* szVarName, const char* szHeader);

	/// Parse an XML file and return the root tag. You are responsible to delete the tag this returns
	static GXMLTag* fromFile(const char* szFilename, const char** pszErrorMessage = NULL, int* pnErrorOffset = NULL, int* pnErrorLine = NULL, int* pnErrorColumn = NULL);

protected:
	int toString(char* pBuffer, int nTabs, const char* szLineStart, const char* szLineEnd, bool bEscapeQuotes);
	char* toString(const char* szLineStart, const char* szLineEnd, bool bEscapeQuotes);
};

} // namespace GClasses

#endif // __GXML_H__
