/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
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

#ifndef __GHTML_H__
#define __GHTML_H__

#include "GError.h"
#include "GTokenizer.h"
#include <vector>
#include <string>
#include <map>
#include <ostream>

namespace GClasses {

class GConstStringHashTable;
struct GHtmlTagHandlerStruct;

class GHtmlElement
{
public:
	GHtmlElement* parent;
	bool text;
	bool singleton;
	std::string name; // or the text if this is a text element
	std::vector<std::string> attrNames;
	std::vector<std::string> attrValues;
	std::vector<GHtmlElement*> children;

	/// Makes an element and inserts it at the specified position of its parent's children.
	/// If pos is -1 (the default), then it is added at the end of the children.
	GHtmlElement(GHtmlElement* par, const char* szTagName, int pos = -1)
	: parent(par), text(false), singleton(false), name(szTagName)
	{
		if(par)
		{
			if(pos < 0)
				par->children.push_back(this);
			else
				par->children.insert(par->children.begin() + pos, this);
		}
	}

	~GHtmlElement()
	{
		for(size_t i = 0; i < children.size(); i++)
			delete(children[i]);
	}

	void populateIdMap(std::map<std::string, GHtmlElement*>& id_to_el);

	void write(std::ostream& stream) const;

	void writePretty(std::ostream& stream, size_t depth = 0) const;

	void writeTextOnly(std::ostream& stream) const;

	/// Returns a pointer to the first immediate child element with the specified tag name
	GHtmlElement* childTag(const char* szTagName);

	/// Adds an attribute name/value pair to this element
	void addAttr(const char* szName, const char* szValue);

	/// Drops the first attribute with the specified name
	void dropAttr(const char* szName);

	/// Swaps the positions of two html element-branches in their DOMs. (Assumes one does not descend from the other--that would create nasty cycles.)
	void swap(GHtmlElement* that);
};



class GHtmlDoc
{
protected:
	GHtmlElement* m_pDocument;
	std::map<std::string, GHtmlElement*>* m_id_to_el;

public:
	GHtmlDoc(const char* szFilename);

	GHtmlDoc(const char* pFile, size_t len);

	~GHtmlDoc();

	GHtmlElement* document() { return m_pDocument; }

	GHtmlElement* getElementById(const char* id);

	//GHtmlElement* getBody();

	static void test();
};



} // namespace GClasses

#endif // __GHTML_H__
