#ifndef __GHTML_H__
#define __GHTML_H__

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
	int m_nSize;
	int m_nPos;

public:
	GHtml(const char* pDoc, int nSize);
	virtual ~GHtml();

	/// You should call this method in a loop until it returns false. It parses a
	/// little bit more of the document each time you call it. It returns false
	/// if there was nothing more to parse. The various virtual methods are called
	/// whenever it finds something interesting.
	bool parseSomeMore();

	/// This method will be called whenever the parser finds a section of display text
	virtual void onTextChunk(const char* pChunk, int chunkSize) {}

	/// This method is called whenever a new tag is found
	virtual void onTag(const char* pTagName, int tagNameLen) {}

	/// This method is called for each parameter in the tag
	virtual void onTagParam(const char* pTagName, int tagNameLen, const char* pParamName, int paramNameLen, const char* pValue, int valueLen) {}

	/// This method is called when an HTML comment (<!--comment-->) is found
	virtual void onComment(const char* pComment, int len) {}

protected:
	void parseTag();
};

} // namespace GClasses

#endif // __GHTML_H__
