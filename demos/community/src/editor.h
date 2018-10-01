/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/


#ifndef EDITOR
#define EDITOR

#include <GClasses/GDynamicPage.h>
#include <iostream>

class Server;

using namespace GClasses;

// Provides pages for editing web pages with version control
class Editor
{
public:

	/// Produces a list of all the files in the current directory
	static void ajaxFilelist(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut);

	/// Saves edited raw text back to a file
	static void ajaxSaveText(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut);

	/// Writes changes from the graphical editor to the file
	static void ajaxSaveGui(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut);

	/// Browse the HTML documents in the user's current folder
	static void pageBrowse(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Edit the raw source of an HTML document
	static void pageEditText(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Edit an HTML page with a graphical editor
	static void pageEditGui(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Diff-merge two files
	static void pageDiff(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

protected:
	/// Writes text into an HTML <pre> tag, escaping '&', '<', and '>' characters.
	static void writeHtmlPre(std::ostream& stream, char* file, size_t len);

	/// Saves a file into its historical record
	static void archiveFile(const char* szFilename);
};

#endif // EDITOR
