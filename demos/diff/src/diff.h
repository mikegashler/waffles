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
class Account;

using namespace GClasses;

// Provides pages for editing web pages with version control
class Editor
{
public:
	/// Diff-merge two files
	static void pageDiff(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

protected:

	/// Writes text into an HTML <pre> tag, escaping '&', '<', and '>' characters.
	static void writeHtmlPre(std::ostream& stream, char* file, size_t len);
};

#endif // EDITOR
