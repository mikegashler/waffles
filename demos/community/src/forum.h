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


#ifndef FORUM
#define FORUM

#include <GClasses/GDynamicPage.h>
#include <iostream>

class Server;
class Account;

using namespace GClasses;

// Provides tools for making web forums or comment sections
class Forum
{
public:
	/// Adds a forum to another web page
	static void pageForumWrapper(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// A page for an admin to monitor recent comments
	static void pageFeed(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Returns all of the comments (so you can inject them into a web page by setting .innerHTML)
	static void ajaxGetForumHtml(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut);

	/// Adds a comment to a forum
	static void ajaxAddComment(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut);

protected:
	static void format_comment_recursive(Server* pServer, GDomNode* pEntry, std::ostream& os, std::string& id, bool allow_direct_reply, size_t depth);

};

#endif // FORUM
