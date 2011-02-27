/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GTWT_H__
#define __GTWT_H__

#include "GError.h"
#include "GHeap.h"
#include <iostream>

namespace GClasses {

class GTwtNode;
class GTwtDoc;
struct GTwtObjField;
//class GXMLTag;

#ifdef WINDOWS
//	ensure compact packing
#	pragma pack(1)
#endif
struct GTwtList
{
	size_t m_itemCount;
	GTwtNode* m_items[2]; // 2 is a bogus value
};


/// Represents a single node in the DOM of a GTwtDoc
class GTwtNode
{
friend class GTwtDoc;
public:
	enum nodetype
	{
		type_obj = 0,
		type_list,
		type_bool,
		type_int,
		type_double,
		type_string,
	};

private:
	int m_type;
	union
	{
		GTwtObjField* m_pFirstField;
		GTwtList m_list;
		bool m_bool;
		long long m_int;
		double m_double;
		char m_string[8]; // 8 is a bogus value
	} m_value;

	GTwtNode() {}
	~GTwtNode() {}

public:
	/// Returns the type of this node
	nodetype type()
	{
		return (nodetype)m_type;
	}

	/// Returns the boolean value stored by this node. Throws if this is not a bool type
	bool asBool()
	{
		if(m_type != type_bool)
			ThrowError("not an bool");
		return m_value.m_bool;
	}

	/// Returns the 64-bit integer value stored by this node. Throws if this is not an integer type
	long long asInt()
	{
		if(m_type != type_int)
			ThrowError("not an int");
		return m_value.m_int;
	}

	/// Returns the double value stored by this node. Throws if this is not a double type
	double asDouble()
	{
		if(m_type != type_double)
			ThrowError("not a double");
		return m_value.m_double;
	}

	/// Returns the string value stored by this node. Throws if this is not a string type
	const char* asString()
	{
		if(m_type != type_string)
			ThrowError("not a string");
		return m_value.m_string;
	}

	/// Returns the number of items in this list. Throws if this is not a list type
	size_t itemCount()
	{
		if(m_type != type_list)
			ThrowError("not a list");
		return m_value.m_list.m_itemCount;
	}

	/// Returns the "index" item in this list. Throws if this is not a list type
	GTwtNode* item(size_t index)
	{
		if(m_type != type_list)
			ThrowError("not a list");
		if(index >= m_value.m_list.m_itemCount)
			ThrowError("out of range");
		return m_value.m_list.m_items[index];
	}

	/// Sets the "index" item in this list. Throws if this is not a list type.
	/// Returns the pNode. Yes, it returns the same node that you pass in.
	GTwtNode* setItem(size_t index, GTwtNode* pNode)
	{
		if(m_type != type_list)
			ThrowError("not a list");
		if(index >= m_value.m_list.m_itemCount)
			ThrowError("out of range");
		m_value.m_list.m_items[index] = pNode;
		return pNode;
	}

	/// Returns the node with the specified field name. Throws if this is not an object type. Returns
	/// NULL if this is an object type, but there is no field with the specified name
	GTwtNode* fieldIfExists(const char* szName);

	/// Returns the node with the specified field name. Throws if this is not an object type. Throws
	/// if there is no field with the name specified by szName
	GTwtNode* field(const char* szName)
	{
		GTwtNode* pNode = fieldIfExists(szName);
		if(!pNode)
			ThrowError("There is no field named ", szName);
		return pNode;
	}

	/// Adds a field with the specified name to this object. Throws if this is not an object type
	/// Returns the pNode. Yes, it returns the same node that you pass in. Why?
	GTwtNode* addField(GTwtDoc* pDoc, const char* szName, GTwtNode* pNode)
	{
		addField(pDoc, szName, pNode, true);
		return pNode;
	}
/*
	/// Converts this branch to XML
	GXMLTag* toXml();

	/// Converts from XML. (This should be able to round-trip with ToXml.) It will ignore
	/// anything that it doesn't recognize.
	GTwtNode* fromXml(GTwtDoc* pDoc, GXMLTag* pTag);
*/
protected:
	void reverseFieldOrder();
	void addField(GTwtDoc* pDoc, const char* szName, GTwtNode* pNode, bool copyName);
	void addField(GTwtObjField* pNewField);
	void checkFieldName(const char* szName);
	void write(std::ostream& stream, const char* szLabel, int level, bool forEmbeddedCode);
	static void printEscapedString(std::ostream& stream, const char* szString, bool forEmbeddedCode);
	static void unescapeString(char* szString, size_t line);
};
#ifdef WINDOWS
//	reset packing to the default
#	pragma pack()
#endif



/// Twt is a text-based data format, somewhat like XML. The major differences are:
/// XML is bloated, slow, designed for human readability, feature-rich, and hard to fully implement.
/// Twt is compact, fast, designed for simplicity of machine parsing, simple, and easy to fully implement.
/// (Plus, GTwtNode has methods that will convert to/from XML, so you're never stuck if you use Twt as your file format.)
class GTwtDoc
{
friend class GTwtNode;
protected:
	GHeap m_heap;
	GTwtNode* m_pRoot;
	int m_line;
	size_t m_len;
	const char* m_pDoc;

public:
	GTwtDoc()
	 : m_heap(2000), m_pRoot(NULL), m_line(0), m_len(0), m_pDoc(NULL)
	{
	}

	~GTwtDoc()
	{
	}

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif

	/// Parse a text file into a twt DOM
	void parse(const char* pDoc, size_t len);

	/// Loads and parses from a file
	void load(const char* szFilename);

	/// Saves to a file
	void save(const char* szFilename);

	/// Writes this doc to the specified stream. Use fopen to open a stream to a file. Use
	/// open_memstream to open a stream to memory.
	void write(std::ostream& stream);

	/// Writes this doc to the specified stream in a manner suitable for embedding in C++ code
	void writeForEmbeddedCode(std::ostream& stream);

	/// Gets a pointer to the heap used by this doc
	GHeap* heap() { return &m_heap; }

	/// Sets the root document node. (Returns the same node that you pass in.)
	GTwtNode* setRoot(GTwtNode* pNode) { m_pRoot = pNode; return pNode; }

	/// Gets the root document node
	GTwtNode* root() { return m_pRoot; }

	/// Makes a new object node
	GTwtNode* newObj();

	/// Makes a new list node
	GTwtNode* newList(size_t itemCount);

	/// Makes a new boolean node
	GTwtNode* newBool(bool b);

	/// Makes a new integer node
	GTwtNode* newInt(long long n);

	/// Makes a new double node
	GTwtNode* newDouble(double d);

	/// Makes a new string node from a null-terminated string
	GTwtNode* newString(const char* szString);

	/// Makes a new string node from the specified string segment
	GTwtNode* newString(const char* pString, size_t len);

protected:
	GTwtObjField* newField();
	void advanceToNextLine();
	GTwtNode* parseLine(int level, char** pOutLabel);
};

} // namespace GClasses

#endif // __GTWT_H__
