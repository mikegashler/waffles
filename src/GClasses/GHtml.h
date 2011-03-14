#ifndef __GHTML_H__
#define __GHTML_H__

#include "GError.h"

namespace GClasses {

class GConstStringHashTable;
struct GHtmlTagHandlerStruct;

/// This class is for parsing HTML files. It's designed to be very simple.
/// This class might be useful, for example, for building a web-crawler
/// or for extracting readable text from a web page.
class GHtml
{
protected:
	const char* m_pDoc;
	size_t m_nSize;
	size_t m_nPos;

public:
	GHtml(const char* pDoc, size_t nSize);
	virtual ~GHtml();

	/// You should call this method in a loop until it returns false. It parses a
	/// little bit more of the document each time you call it. It returns false
	/// if there was nothing more to parse. The various virtual methods are called
	/// whenever it finds something interesting.
	bool parseSomeMore();

	/// This method will be called whenever the parser finds a section of display text
	virtual void onTextChunk(const char* pChunk, size_t chunkSize) {}

	/// This method is called whenever a new tag is found
	virtual void onTag(const char* pTagName, size_t tagNameLen) {}

	/// This method is called for each parameter in the tag
	virtual void onTagParam(const char* pTagName, size_t tagNameLen, const char* pParamName, size_t paramNameLen, const char* pValue, size_t valueLen) {}

	/// This method is called when an HTML comment (<!--comment-->) is found
	virtual void onComment(const char* pComment, size_t len) {}

protected:
	void parseTag();
};

} // namespace GClasses

#endif // __GHTML_H__
