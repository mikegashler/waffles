/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberateFiterly misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#ifndef __GDOM_H__
#define __GDOM_H__

#include "GError.h"
#include "GHeap.h"
#include "GString.h"
#include <iostream>

namespace GClasses {

class GDomNode;
class GDom;
class GDomObjField;
class GDomArrayList;
class GJsonTokenizer;


#ifdef WINDOWS
//	ensure compact packing
#	pragma pack(1)
#endif

///\brief Converts a GDomNode to a string
std::string to_str(const GDomNode& node);

///\brief Converts a GDomNode to a string, but if the resulting string is more than about 60 characters then it will be shortened to show only the first and last segments separated by ellipses.
std::string to_str_brief(const GDomNode& node);

///\brief Converts a GDom to a string
std::string to_str(const GDom& doc);

/// This class iterates over the items in a list node
class GDomListIterator
{
friend class GDomNode;
protected:
	const GDomNode* m_pList;
	size_t m_index;

public:
	GDomListIterator(const GDomNode* pNode);
	~GDomListIterator();

	/// Returns the current item in the list
	GDomNode* current();

	/// Returns the current bool in the list
	bool currentBool();

	/// Returns the current int in the list
	long long currentInt();

	/// Returns the current double in the list
	double currentDouble();

	/// Returns the current string in the list
	const char* currentString();

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
	char m_type;
	union
	{
		GDomObjField* m_pLastField;
		GDomArrayList* m_pArrayList;
		bool m_bool;
		long long m_int;
		double m_double;
		char m_string[8]; // 8 is a bogus value
	} m_value;

	GDomNode() {}
	~GDomNode() {}

public:
	/// Returns the type of this node
	nodetype type() const
	{
		return (nodetype)m_type;
	}

	/// Returns the boolean value stored by this node. Throws if this is not a bool type
	bool asBool() const
	{
		if(m_type != type_bool)
			throw Ex("\"", to_str(this), "\" is not a bool");
		return m_value.m_bool;
	}

	/// Returns the 64-bit integer value stored by this node. Throws if this is not an integer type
	long long asInt() const
	{
		if(m_type != type_int)
			throw Ex("\"", to_str(this), "\" is not an int");
		return m_value.m_int;
	}

	/// Returns the double value stored by this node. Throws if this is not a double type
	double asDouble() const
	{
		if(m_type == type_double)
			return m_value.m_double;
		else if(m_type == type_int)
			return (double)m_value.m_int;
		else
			throw Ex("\"", to_str(this), "\" is not a double");
		return 0.0;
	}

	/// Returns the string value stored by this node. Throws if this is not a string type
	const char* asString() const
	{
		if(m_type != type_string)
			throw Ex("\"", to_str(this), "\" is not a string");
		return m_value.m_string;
	}

	/// Returns the node with the specified field name. Throws if this is not an object type. Returns
	/// NULL if this is an object type, but there is no field with the specified name
	GDomNode* getIfExists(const char* szName) const;

	/// Returns the node with the specified field name. Throws if this is not an object type. Throws
	/// if there is no field with the name specified by szName
	GDomNode* get(const char* szName) const
	{
		GDomNode* pNode = getIfExists(szName);
		if(!pNode)
			throw Ex("There is no field named ", szName);
		return pNode;
	}

	bool getBool(const char* szName) const
	{
		return get(szName)->asBool();
	}

	long long getInt(const char* szName) const
	{
		return get(szName)->asInt();
	}

	double getDouble(const char* szName) const
	{
		return get(szName)->asDouble();
	}

	const char* getString(const char* szName) const
	{
		return get(szName)->asString();
	}

	size_t size() const;

	GDomNode* get(size_t index) const;

	bool getBool(size_t index) const
	{
		return get(index)->asBool();
	}

	long long getInt(size_t index) const
	{
		return get(index)->asInt();
	}

	double getDouble(size_t index) const
	{
		return get(index)->asDouble();
	}

	const char* getString(size_t index) const
	{
		return get(index)->asString();
	}

	/// Replaces an existing field with the specified name.
	/// If no existing field is found, adds a new one.
	GDomNode* set(GDom* pDoc, const char* szName, GDomNode* pNode);

	/// Sets the specified element in a list
	GDomNode* set(GDom* pDoc, size_t index, GDomNode* pNode);

	/// Adds a field with the specified name to this object. Throws if this is not an object type
	/// Returns pNode. (Yes, it returns the same node that you pass in. This is useful for
	/// writing compact marshalling code.)
	/// Assumes the user already knows there is no existing field with that name.
	GDomNode* add(GDom* pDoc, const char* szName, GDomNode* pNode);

	/// Adds a boolean field
	GDomNode* add(GDom* pDoc, const char* szName, bool b);

	/// Adds an int field
	GDomNode* add(GDom* pDoc, const char* szName, long long n);

	/// Adds an int field
	GDomNode* add(GDom* pDoc, const char* szName, size_t n);

	/// Adds a double field
	GDomNode* add(GDom* pDoc, const char* szName, double d);

	/// Adds a string field
	GDomNode* add(GDom* pDoc, const char* szName, const char* str);

	/// Adds an item to a list node. Returns a pointer to the item passed in (pNode).
	GDomNode* add(GDom* pDoc, GDomNode* pNode);

	/// Adds a boolean list item
	GDomNode* add(GDom* pDoc, bool b);

	/// Adds an int list item
	GDomNode* add(GDom* pDoc, long long n);

	/// Adds an int list item
	GDomNode* add(GDom* pDoc, size_t n);

	/// Adds a double list item
	GDomNode* add(GDom* pDoc, double d);

	/// Adds a string list item
	GDomNode* add(GDom* pDoc, const char* str);

	/// Removes the specified field from an object
	void del(const char* szField);

	/// Removes the specified element from a list. An O(n) operation, since it shifts the remaining data.
	void del(size_t index);

	/// Writes this node to a JSON file
	void saveJson(const char* filename) const;

	/// Writes this node in JSON format.
	void writeJson(std::ostream& stream) const;

	/// Writes this node in JSON format indented in a manner suitable for human readability.
	void writeJsonPretty(std::ostream& stream, size_t indents) const;

	/// Writes this node in JSON format, escaped in a manner suitable for hard-coding in a C/C++ string.
	size_t writeJsonCpp(std::ostream& stream, size_t col) const;

	/// Writes this node as XML
	void writeXml(std::ostream& stream, const char* szLabel) const;

	/// Returns true iff pOther is equivalent to this node
	bool isEqual(const GDomNode* pOther) const;

protected:
	/// Reverses the order of the fields in the object and returns
	/// the number of fields.  Assumes this GDomNode is
	/// a object node.  Behavior is undefined if it is not an object
	/// node.
	///
	/// \note This method is hackishly marked const because it is always used twice, such that it has no net effect.
	///
	/// \return The number of fields
	size_t reverseFieldOrder() const;

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
	virtual ~GDom();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Clears the DOM.
	void clear();

	/// Load from the specified file in JSON format. (See http://json.org.)
	void loadJson(const char* szFilename);

	/// Saves to a file in JSON format. (See http://json.org.)
	void saveJson(const char* szFilename) const;

	/// Parses a JSON string. The resulting DOM can be retrieved by calling root().
	void parseJson(const char* pJsonString, size_t len);

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
	const GDomNode* root() const { return m_pRoot; }
	GDomNode* root() { return m_pRoot; }

	/// Sets the root document node. (Returns the same node that you pass in.)
	const GDomNode* setRoot(const GDomNode* pNode) { m_pRoot = (GDomNode*)pNode; return pNode; }

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

	/// Makes a new string node from a null-terminated string. (If you want
	/// to parse a JSON string, call parseJson instead. This method just wraps
	/// the string in a node.)
	GDomNode* newString(const char* szString);

	/// Makes a new string node from the specified string segment
	GDomNode* newString(const char* pString, size_t len);

	/// Makes a deep copy of pNode in this DOM
	GDomNode* clone(GDomNode* pNode);

	/// Returns a pointer to the heap used by this doc
	GHeap* heap() { return &m_heap; }

protected:
	GDomObjField* newField();
	GDomNode* loadJsonObject(GJsonTokenizer& tok);
	GDomNode* loadJsonArray(GJsonTokenizer& tok);
	GDomNode* loadJsonNumber(GJsonTokenizer& tok);
	GDomNode* loadJsonValue(GJsonTokenizer& tok);
	char* loadJsonString(GJsonTokenizer& tok);
};


/// A Dom with a mod counter, for use with Jaad.
class GJaadDom : public GDom
{
protected:
	size_t m_modCount;

public:
	GJaadDom() : GDom(), m_modCount(0) {}
	virtual ~GJaadDom() {}

	/// Gets the mod count
	size_t modCount() { return m_modCount; }

	/// Resets the mod count
	void resetModCount() { m_modCount = 0; }

	/// Increments the mod count
	void incModCount() { ++m_modCount; }
};


/// Represents a collection of JSON files that can be edited remotely, like a database.
/// (The philosophy is to insert implied structures as needed, and fail only when
/// a command cannot be made to work.)
///
/// Example packet:
/// {
///   "file":"/somepath/somefile.json",
///   "auth":"sometoken",
///   "cmd":".somefield[3].anotherfield"
/// }
///
/// Some example commands:
/// ".somefield[3]"                      Get an array element
/// ".somefield[3].anotherfield"         Get a field value
/// ".somefield += {'x':123, 'y':456}"   Add an object to a list
/// ".somefield[3].newfield = 'newval'"  Set a field value
/// ".somefield -= [3]"                  Remove a list element
/// "-= .somefield[3]"                   Remove a list element (same as previous one)
/// ".somefield[3] -= .newfield"         Remove a field
/// "-= .somefield[3].newfield"          Remove a field (same as previous one)
/// ".somefield[.field == 'val']"        Get the object with a field of 'val'
/// ".somefield[.field == 'val'].f2 = 3" Set f2 of the object with a field of 'val' to 3
/// "[3]"                                Get an array element from a root list
/// "+= {'x':123, 'y':456}"              Add an object to a root list
/// ".somefield = {'x':123, 'y':456}"    Set a field in the root object
///
class GJsonAsADatabase
{
protected:
	std::string m_basePath;
	std::map<std::string, GJaadDom*> m_doms;

public:
	GJsonAsADatabase(const char* szBasePath);
	virtual ~GJsonAsADatabase();

	/// Throws an exception if anything goes wrong.
	/// Returns nullptr if an operation was successfully completed.
	/// Returns the requested data if data was requested.
	const GDomNode* apply(const char* szFilename, const char* szCmd, GDom* pResponseDom);

	/// Writes any changed files to disk. If clear_cache is true, it also clears its cache.
	void flush(bool clear_cache);

protected:
	/// Finds the specified token, ignoring braced or bracked regions, escaped characters, and quoted or double-quoted strings.
	/// Starts at position start. Returns the index of the token if it is found.
	/// Returns s.size() if the token is not found.
	static size_t findTok(const char* s, char tok, size_t start = 0);

	GJaadDom* getDom(const char* szFile);
	GDomNode* findNode(GDom* pDoc, GDomNode* pOb, GDom* pResponseDom, const char* szCmd, char op);
	GDomNode* findLValue(GDom* pDoc, GDomNode* pOb, GDom* pResponseDom, const char* szCmd, std::string* pOutField, size_t* pOutIndex);
	void add(GDomNode* pRequest, GDom* pDoc, GDomNode* pOb);
	void del(GDomNode* pRequest, GDom* pDoc, GDomNode* pOb);
};




} // namespace GClasses




#endif // __GDOM_H__
