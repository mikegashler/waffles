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

protected:
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

public:
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

	/// Adds a field with the specified name to this object. Throws if this is not an object type
	/// Returns pNode. (Yes, it returns the same node that you pass in. This is useful for
	/// writing compact marshalling code.)
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


/// Represents a collection of JSON files that can be edited remotely like a database.
///
/// Example actions:
///
/// ### Get a value
/// {
///   "file":"/somepath/somefile.json",
///   "auth":"sometoken",
///   "act":"get",
///   "ob":".somefield[3].anotherfield"
/// }
///
/// ### Add a value to a list
/// {
///   "file":"/somepath/somefile.json",
///   "auth":"sometoken",
///   "act":"add",
///   "ob":".somefield",
///   "val":{"list":[{"x":1,"y":2},{"x":3,"y":4}]}
/// }
///
/// ### Add a value to an object
/// {
///   "file":"/somepath/somefile.json",
///   "auth":"sometoken",
///   "act":"add",
///   "ob":".somefield[3]",
///   "name":"newfieldname",
///   "val":{"list":[{"x":1,"y":2},{"x":3,"y":4}]}
/// }
///
/// ### Delete an element from a list
/// {
///   "file":"/somepath/somefile.json",
///   "auth":"sometoken",
///   "act":"del",
///   "ob":".somefield",
///   "index":2
/// }
///
/// ### Delete a field from an object
/// {
///   "file":"/somepath/somefile.json",
///   "auth":"sometoken",
///   "act":"del",
///   "ob":".somefield[1]"
///   "name":"fieldtodelete",
/// }
///
class GJsonAsADatabase
{
protected:
	std::string m_basePath;
	std::map<std::string, GDom*> m_doms;

public:
	GJsonAsADatabase(const char* szBasePath);
	~GJsonAsADatabase();

	/// Evaluates the szAuth parameter to determine whether the user has
	/// permission to modify the specified file, szFilename.
	/// The default implementation disallows everything.
	/// You must override this method to allow valid actions to be performed.
	virtual bool checkPermission(const char* szFilename, const char* szAuth) { return false; }

	const GDomNode* apply(GDomNode* pRequest, GDom* pResponseDom);

protected:
	GDom* getDom(const char* szFile);
	GDomNode* findNode(GDomNode* pOb, GDom* pResponseDom, const char* szOb);
	void add(GDomNode* pRequest, GDom* pDoc, GDomNode* pOb);
	void del(GDomNode* pRequest, GDom* pDoc, GDomNode* pOb);
};




} // namespace GClasses




#endif // __GDOM_H__
