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

#include "GHtml.h"
#include "GHashTable.h"
#include "GTokenizer.h"
#include <set>
#include <string.h>

using std::string;

namespace GClasses {

struct InsensitiveCompare
{
    bool operator() (const std::string& a, const std::string& b) const
    {
        return _stricmp(a.c_str(), b.c_str()) < 0;
    }
};



// A helper class used by the GHtmlDoc constructor to parse HTML files into a DOM
class GHtmlParser : public GTokenizer
{
protected:
	GCharSet m_tagStart;
	GCharSet m_tagEnd;
	GCharSet m_endTagName;
	GCharSet m_endAttrName;
	GCharSet m_hyphen;
	GCharSet m_whitespace;
	std::set<std::string, InsensitiveCompare> singletons;
	//std::set<std::string, InsensitiveCompare> self_closing;
	std::set<std::string, InsensitiveCompare> special;

public:
	void addSingletons()
	{
		singletons.insert("!doctype");
		singletons.insert("area");
		singletons.insert("base");
		singletons.insert("br");
		singletons.insert("col");
		singletons.insert("command");
		singletons.insert("embed");
		singletons.insert("hr");
		singletons.insert("img");
		singletons.insert("input");
		singletons.insert("keygen");
		singletons.insert("link");
		singletons.insert("meta");
		singletons.insert("param");
		singletons.insert("source");
		singletons.insert("track");
		singletons.insert("wbr");
		/*self_closing.insert("html");
		self_closing.insert("head");
		self_closing.insert("body");
		self_closing.insert("p");
		self_closing.insert("dt");
		self_closing.insert("dd");
		self_closing.insert("li");
		self_closing.insert("option");
		self_closing.insert("thead");
		self_closing.insert("th");
		self_closing.insert("tbody");
		self_closing.insert("tr");
		self_closing.insert("td");
		self_closing.insert("tfoot");
		self_closing.insert("colgroup");*/
		special.insert("script");
		special.insert("style");

	}

	GHtmlParser(const char* szFilename) : GTokenizer(szFilename),
		m_tagStart("<"),
		m_tagEnd(">"),
		m_endTagName("> \t\r\n/"),
		m_endAttrName(">= \t\r\n/"),
		m_hyphen("-"),
		m_whitespace(" \t\r\n")
	{
		addSingletons();
	}

	GHtmlParser(const char* pFile, size_t len) : GTokenizer(pFile, len),
		m_tagStart("<"),
		m_tagEnd(">"),
		m_endTagName("> \t\r\n"),
		m_endAttrName(">= \t\r\n"),
		m_hyphen("-"),
		m_whitespace(" \t\r\n")
	{
		addSingletons();
	}

	virtual ~GHtmlParser()
	{
	}

	const char* parseCloser()
	{
		skip(2); // Move past the "</"
		char* szTagName = readUntil(m_tagEnd);
		skipWhile(m_tagEnd);
		return szTagName;
	}

	GHtmlElement* parseTag(GHtmlElement* par)
	{
		// Parse the tag name
		skip(1); // Move past the '<'
		char* szTagName = readUntil(m_endTagName);

		// Handle comments
		if(szTagName[0] == '!' && szTagName[1] == '-' && szTagName[2] == '-')
		{
			string s = szTagName;
			while(true)
			{
				size_t len = s.length();
				if(len >= 5 && s[len - 1] == '-' && s[len - 2] == '-')
					break;
				char* szSpaces = readWhile(m_endTagName);
				s += szSpaces;
				if(!has_more())
					break;
				char* szMore = readUntil(m_endTagName);
				s += szMore;
			}
			skip(1);
			GHtmlElement* pEl = new GHtmlElement(par, s.c_str());
			pEl->singleton = true;
			return pEl;
		}

		GHtmlElement* pEl = new GHtmlElement(par, szTagName);

		// Digest the attributes
		while(true)
		{
			// Eat whitespace
			skipWhile(m_whitespace);

			// Detect the end of the tag
			if(!has_more())
				break;
			char c = peek();
			if(c == '/' && peek(1) == '>')
			{
				skip(2);
				pEl->singleton = true;
				break;
			}
			if(c == '>')
			{
				skip(1);
				break;
			}

			// Parse the name and value
			char* szAttrName = readUntil(m_endAttrName);
			string sName = szAttrName;
			skipWhile(m_whitespace);
			string sValue = "";
			c = peek();
			if(c == '=')
			{
				skip(1);
				skipWhile(m_whitespace);
				if(peek() != '>')
				{
					char* szAttrValue = readUntil_escaped_quoted(m_endTagName, '\0');
					sValue = szAttrValue;
				}
			}

			// Add the name/value pair
			pEl->attrNames.push_back(sName);
			pEl->attrValues.push_back(sValue);
		}

		if(!pEl->singleton && singletons.find(pEl->name) != singletons.end())
			pEl->singleton = true;
		return pEl;
	}

	GHtmlElement* parse()
	{
		GHtmlElement* pDocument = new GHtmlElement(nullptr, "document");
		pDocument->text = true; // text and singleton is a unique combination of flags for the document tag
		pDocument->singleton = true; // text and singleton is a unique combination of flags for the document tag
		GHtmlElement* pCur = pDocument;
		bool parsing_special = false; // indicates a tag that surrounds a special language, such as <style> or <script>
		while(has_more())
		{
			char c = peek();
			if(c == '<')
			{
				// Parse a tag
				if(peek(1) == '/')
				{
					// Parse a closer tag
					const char* szCloseName = parseCloser();
					while(true)
					{
						GHtmlElement* pChild = pCur;
						pCur = pCur->parent;
						if(_stricmp(szCloseName, pChild->name.c_str()) == 0)
							break;
						if(!pCur)
							throw Ex("No matching tag to close: ", szCloseName);
					}
					parsing_special = false;
				}
				else if(parsing_special)
				{
					// Parse a text element
					const char* szText = readWhile(m_tagStart);
					GHtmlElement* pEl = new GHtmlElement(pCur, szText);
					pEl->text = true;
				}
				else
				{
					// Parse an opener tag
					GHtmlElement* pEl = parseTag(pCur);
					if(!pEl->singleton)
						pCur = pEl;
					if(special.find(pEl->name) != special.end())
						parsing_special = true;
				}
			}
			else
			{
				// Parse a text element
				const char* szText = readUntil(m_tagStart, 1, 2048);
				GHtmlElement* pEl = new GHtmlElement(pCur, szText);
				pEl->text = true;
			}
		}
		return pDocument;
	}
};









string attrValueTrim(string& s)
{
	size_t beg = 0;
	while(beg < s.length() && (s[beg] <= ' ' || s[beg] == '"' || s[beg] == '\''))
		beg++;
	size_t end = s.length();
	while(end > beg && (s[end - 1] <= ' ' || s[end - 1] == '"' || s[end - 1] == '\''))
		end--;
	if(beg == 0 && end == s.length())
		return s;
	return s.substr(beg, end - beg);
}

void GHtmlElement::populateIdMap(std::map<std::string, GHtmlElement*>& id_to_el)
{
	for(size_t i = 0; i < attrNames.size(); i++)
	{
		if(_stricmp(attrNames[i].c_str(), "id") == 0 && attrValues.size() > i)
		{
			id_to_el.insert(std::pair<std::string, GHtmlElement*>(attrValueTrim(attrValues[i]), this));
		}
	}
	for(size_t i = 0; i < children.size(); i++)
		children[i]->populateIdMap(id_to_el);
}

void GHtmlElement::write(std::ostream& stream) const
{
	if(text)
	{
		if(singleton) // unique combination of flags for the document tag
		{
			for(size_t i = 0; i < children.size(); i++)
				children[i]->write(stream);
		}
		else
			stream << name;
	}
	else
	{
		stream << "<" << name;
		for(size_t i = 0; i < attrNames.size() && i < attrValues.size(); i++)
		{
			// Print attribute name=value
			stream << " " << attrNames[i];
			if(attrValues[i].length() > 0)
				stream << "=" << attrValues[i];
		}
		if(singleton)
			stream << ">"; // "/>";
		else
		{
			stream << ">";
			for(size_t i = 0; i < children.size(); i++)
				children[i]->write(stream);
			stream << "</" << name << ">";
		}
	}
}

void GHtmlElement::writePretty(std::ostream& stream, size_t depth) const
{
	if(text)
	{
		if(singleton) // unique combination of flags for the document tag
		{
			for(size_t i = 0; i < children.size(); i++)
				children[i]->writePretty(stream, depth);
		}
		else
		{
			size_t first = name.find_first_not_of(" \t\r\n");
			if(first == string::npos)
				stream << name << "\n";
			else
			{
				size_t last = name.find_last_not_of(" \t\r\n");
				stream << name.substr(first, (last - first + 1)) << "\n";
			}
		}
	}
	else
	{
		for(size_t i = 0; i < depth; i++)
			stream << "\t";
		stream << "<" << name;
		for(size_t i = 0; i < attrNames.size() && i < attrValues.size(); i++)
		{
			// Print attribute name=value
			stream << " " << attrNames[i];
			if(attrValues[i].length() > 0)
				stream << "=" << attrValues[i];
		}
		if(singleton)
			stream << "/>\n";
		else
		{
			stream << ">\n";
			for(size_t i = 0; i < children.size(); i++)
				children[i]->writePretty(stream, depth + 1);
			for(size_t i = 0; i < depth; i++)
				stream << "\t";
			stream << "</" << name << ">\n";
		}
	}
}

void GHtmlElement::writeTextOnly(std::ostream& stream) const
{
	if(text)
	{
		if(singleton) // unique combination of flags for the document tag
		{
			for(size_t i = 0; i < children.size(); i++)
				children[i]->writeTextOnly(stream);
		}
		else
			stream << name << "\n";
	}
	else
	{
		if(!singleton)
		{
			const char* szName = name.c_str();
			if(_stricmp(szName, "style") != 0 &&
			_stricmp(szName, "script") != 0)
			{
				for(size_t i = 0; i < children.size(); i++)
					children[i]->writeTextOnly(stream);
			}
		}
	}
}

GHtmlElement* GHtmlElement::childTag(const char* szTagName)
{
	for(size_t i = 0; i < children.size(); i++)
	{
		GHtmlElement* pEl = children[i];
		if(_stricmp(pEl->name.c_str(), szTagName) == 0)
			return pEl;
	}
	return nullptr;
}

void GHtmlElement::addAttr(const char* szName, const char* szValue)
{
	attrNames.push_back(szName);
	attrValues.push_back(szValue);
}

void GHtmlElement::dropAttr(const char* szName)
{
	for(size_t i = 0; i < attrNames.size(); i++)
	{
		if(_stricmp(attrNames[i].c_str(), szName) == 0)
		{
			attrNames.erase(attrNames.begin() + i);
			attrValues.erase(attrValues.begin() + i);
		}
	}
}

void GHtmlElement::swap(GHtmlElement* that)
{
	// Find the indexes of "this"
	size_t indexThis;
	for(indexThis = 0; indexThis < this->parent->children.size(); indexThis++)
	{
		if(this->parent->children[indexThis] == this)
			break;
	}
	if(indexThis >= this->parent->children.size())
		throw Ex("Failed to find this element among its parent's children");

	// Find the indexes of "that"
	size_t indexThat;
	for(indexThat = 0; indexThat < that->parent->children.size(); indexThat++)
	{
		if(that->parent->children[indexThat] == that)
			break;
	}
	if(indexThis >= this->parent->children.size())
		throw Ex("Failed to find that element among its parent's children");

	std::swap(this->parent->children[indexThis], that->parent->children[indexThat]);
	std::swap(this->parent, that->parent);
}







GHtmlDoc::GHtmlDoc(const char* szFilename)
: m_id_to_el(nullptr)
{
	GHtmlParser parser(szFilename);
	m_pDocument = parser.parse();
}

GHtmlDoc::GHtmlDoc(const char* pDoc, size_t size)
: m_id_to_el(nullptr)
{
	GHtmlParser parser(pDoc, size);
	m_pDocument = parser.parse();
}

GHtmlDoc::~GHtmlDoc()
{
	delete(m_id_to_el);
	delete(m_pDocument);
}

GHtmlElement* GHtmlDoc::getElementById(const char* id)
{
	if(!m_id_to_el)
	{
		m_id_to_el = new std::map<std::string, GHtmlElement*>();
		m_pDocument->populateIdMap(*m_id_to_el);
	}
	std::map<std::string, GHtmlElement*>::iterator it = m_id_to_el->find(id);
	if(it == m_id_to_el->end())
		return nullptr;
	return it->second;
}

/*
GHtmlElement* GHtmlDoc::getBody()
{
	GHtmlElement* pHtml = m_pDocument->childTag("html");
	if(!pHtml)
		return nullptr;
	return pHtml->childTag("body");
}
*/

void GHtmlDoc::test()
{
	const char* raw =
	"<!doctype html>\n"
	"<html>\n"
	"<head>\n"
	"  <meta http-equiv=\"Content-Type\" content=\"text/html;charset=UTF-8\">\n"
	"  <title>Some Title</title>\n"
	"  <style>\n"
	"  body {\n"
	"    background-color: #e0d8d0;\n"
	"    background-image:url('back.png');\n"
	"  }\n"
	"  </style>\n"
	"</head>\n"
	"<body>\n"
	"<a href=\"http://somesite.com/index.html\"><img src=\"pic.png\" id=\"mypic\"></a>\n"
	"<div id=\"mydiv\">Div contents</div>\n"
	"\n"
	"<br><br>\n"
	"<table bgcolor=#f0e8e0 cellpadding=40 width=800 align=center><tr><td>\n"
	"<br>\n"
	"<!-- Here comes the title -->\n"
	"<h1>My <b>elegant</b> title</h1>\n"
	"<p>Bla bla blah!\n"
	"</p>\n"
	"	   <br><br>\n"
	"<iframe width=\"720\" height=\"405\" src=\"https://www.youtube.com/embed/KzGjEkp772s?start=25\" frameborder=\"0\" allow=\"accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture\" allowfullscreen></iframe>\n"
	"</td></tr></table></body></html>\n";
	GHtmlDoc doc(raw, strlen(raw));
	std::ostringstream os;
	doc.document()->write(os);
	string s = os.str();
	const char* rt = s.c_str();

	// Make sure the before and after strings match
	size_t i = 0;
	while(raw[i] != '\0' || rt[i] != '\0')
	{
		if(raw[i] == rt[i])
			++i;
		else
		{
            std::cout << "\n";
            size_t pre = std::min(i, (size_t)20);
            for(size_t j = 0; j < pre; j++)
                std::cout << " ";
            std::cout << "v\n";
            for(size_t j = 0; j < 40; j++)
            {
                if(raw[i - pre + j] >= ' ')
                    std::cout << raw[i - pre + j];
                else
                    std::cout << '.';
            }
            std::cout << "\n";
            for(size_t j = 0; j < 40; j++)
            {
                if(rt[i - pre + j] >= ' ')
                    std::cout << rt[i - pre + j];
                else
                    std::cout << '.';
            }
            std::cout << "\n";
            for(size_t j = 0; j < pre; j++)
                std::cout << " ";
            std::cout << "^\n";
            throw Ex("strings differ");
        }
	}
}

}
