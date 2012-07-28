/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GDOM_H__
#define __GDOM_H__

#include "GError.h"
#include "GHeap.h"
#include <iostream>

namespace GClasses {

class GDomNode;
class GDom;
class GDomObjField;
class GDomListItem;
class GJsonTokenizer;


#ifdef WINDOWS
//	ensure compact packing
#	pragma pack(1)
#endif

/// This class iterates over the items in a list node
class GDomListIterator
{
protected:
	GDomNode* m_pList;
	GDomListItem* m_pCurrent;
	size_t m_remaining;

public:
	GDomListIterator(GDomNode* pNode);
	~GDomListIterator();

	/// Returns the current item in the list
	GDomNode* current();

	/// Advances to the next item in the list
	void advance();

	/// Returns the number of items remaining to be visited.  When
	/// the current item in the list is the first item, the number
	/// remaining is the number of items in the list.
	size_t remaining();
};


/// Represents a single node in a DOM
class GDomNode
{
friend class GDom;
friend class GDomListIterator;
public:
	enum nodetype
	{
		type_obj = 0,
		type_list,
		type_bool,
		type_int,
		type_double,
		type_string,
		type_null,
	};

private:
	int m_type;
	union
	{
		GDomObjField* m_pLastField;
		GDomListItem* m_pLastItem;
		bool m_bool;
		long long m_int;
		double m_double;
		char m_string[8]; // 8 is a bogus value
	} m_value;

	GDomNode() {}
	~GDomNode() {}

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
			throw Ex("not an bool");
		return m_value.m_bool;
	}

	/// Returns the 64-bit integer value stored by this node. Throws if this is not an integer type
	long long asInt()
	{
		if(m_type != type_int)
			throw Ex("not an int");
		return m_value.m_int;
	}

	/// Returns the double value stored by this node. Throws if this is not a double type
	double asDouble()
	{
		if(m_type == type_double)
			return m_value.m_double;
		else if(m_type == type_int)
			return (double)m_value.m_int;
		else
			throw Ex("not a double");
		return 0.0;
	}

	/// Returns the string value stored by this node. Throws if this is not a string type
	const char* asString()
	{
		if(m_type != type_string)
			throw Ex("not a string");
		return m_value.m_string;
	}

	/// Returns the node with the specified field name. Throws if this is not an object type. Returns
	/// NULL if this is an object type, but there is no field with the specified name
	GDomNode* fieldIfExists(const char* szName);

	/// Returns the node with the specified field name. Throws if this is not an object type. Throws
	/// if there is no field with the name specified by szName
	GDomNode* field(const char* szName)
	{
		GDomNode* pNode = fieldIfExists(szName);
		if(!pNode)
			throw Ex("There is no field named ", szName);
		return pNode;
	}

	/// Adds a field with the specified name to this object. Throws if this is not an object type
	/// Returns pNode. (Yes, it returns the same node that you pass in. This is useful for
	/// writing compact marshalling code.)
	GDomNode* addField(GDom* pDoc, const char* szName, GDomNode* pNode);

	/// Adds an item to a list node. Returns a pointer to the item passed in (pNode).
	GDomNode* addItem(GDom* pDoc, GDomNode* pNode);

	/// Writes this node in JSON format.
	void writeJson(std::ostream& stream) const;

	/// Writes this node in JSON format indented in a manner suitable for human readability.
	void writeJsonPretty(std::ostream& stream, size_t indents) const;

	/// Writes this node in JSON format, escaped in a manner suitable for hard-coding in a C/C++ string.
	size_t writeJsonCpp(std::ostream& stream, size_t col) const;

	/// Writes this node as XML
	void writeXml(std::ostream& stream, const char* szLabel) const;

protected:
	/// Reverses the order of the fiels in the object and returns
	/// the number of fields.  Assumes this GDomNode is
	/// a object node.  Behavior is undefined if it is not an object
	/// node.
	///
	/// \note This method is hackishly marked const because it is always used twice, such that it has no net effect.
	///
	/// \return The number of fields
	size_t reverseFieldOrder() const;

	/// Reverses the order of the items in the list and returns
	/// the number of items in the list.  Assumes this GDomNode is
	/// a list node.  Behavior is undefined if it is not a list
	/// node.
	///
	/// \note This method is hackishly marked const because it is always used twice, such that it has no net effect.
	///
	/// \return The number of items in the list
	size_t reverseItemOrder() const;

	void writeXmlInlineValue(std::ostream& stream);
};
#ifdef WINDOWS
//	reset packing to the default
#	pragma pack()
#endif



/// A Document Object Model. This represents a document as a hierarchy of objects.
/// The DOM can be loaded-from or saved-to a file in JSON (JavaScript Object Notation)
/// format. (See http://json.org.) In the future, support for XML and/or other formats
/// may be added.
class GDom
{
friend class GDomNode;
protected:
	GHeap m_heap;
	GDomNode* m_pRoot;
	int m_line;
	size_t m_len;
	const char* m_pDoc;

public:
	GDom();
	~GDom();

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // MIN_PREDICT

	/// Clears the DOM.
	void clear();

	/// Load from the specified file in JSON format. (See http://json.org.)
	void loadJson(const char* szFilename);

	/// Saves to a file in JSON format. (See http://json.org.)
	void saveJson(const char* szFilename) const;

	/// Parses JSON format from a tokenizer (which wraps a stream).
	void parseJson(const char* pFile, size_t len);

	/// Writes this doc to the specified stream in JSON format. (See http://json.org.)
	/// (If you want to write to a memory buffer, you can use open_memstream.)
	void writeJson(std::ostream& stream) const;

	/// Writes this doc to the specified stream in JSON format with indentation to make it human-readable.
	/// (If you want to write to a memory buffer, you can use open_memstream.)
	void writeJsonPretty(std::ostream& stream) const;

	/// Writes this doc to the specified stream as an inlined C++ string in JSON format.
	/// (This method would be useful for hard-coding a serialized object in a C++ program.)
	void writeJsonCpp(std::ostream& stream) const;

	/// Write as XML to the specified stream.
	void writeXml(std::ostream& stream) const;

	/// Gets the root document node
	GDomNode* root() const { return m_pRoot; }

	/// Sets the root document node. (Returns the same node that you pass in.)
	GDomNode* setRoot(GDomNode* pNode) { m_pRoot = pNode; return pNode; }

	/// Makes a new object node
	GDomNode* newObj();

	/// Makes a new list node
	GDomNode* newList();

	/// Makes a new node to represent a null value
	GDomNode* newNull();

	/// Makes a new boolean node
	GDomNode* newBool(bool b);

	/// Makes a new integer node
	GDomNode* newInt(long long n);

	/// Makes a new double node
	GDomNode* newDouble(double d);

	/// Makes a new string node from a null-terminated string
	GDomNode* newString(const char* szString);

	/// Makes a new string node from the specified string segment
	GDomNode* newString(const char* pString, size_t len);

	/// Returns a pointer to the heap used by this doc
	GHeap* heap() { return &m_heap; }

protected:
	GDomObjField* newField();
	GDomListItem* newItem();
	GDomNode* loadJsonObject(GJsonTokenizer& tok);
	GDomNode* loadJsonArray(GJsonTokenizer& tok);
	GDomNode* loadJsonNumber(GJsonTokenizer& tok);
	GDomNode* loadJsonValue(GJsonTokenizer& tok);
	char* loadJsonString(GJsonTokenizer& tok);
};

///\brief Converts a GDomNode to a string
std::string to_str(const GDomNode& node);

///\brief Converts a GDom to a string
std::string to_str(const GDom& doc);

} // namespace GClasses




#endif // __GDOM_H__
