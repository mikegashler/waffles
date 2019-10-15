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

#include <GClasses/GRand.h>
#include <GClasses/GApp.h>
#include <GClasses/GTime.h>
#include <map>
#include "server.h"
#include "diff.h"

using std::cout;
using std::cerr;
using std::ostream;
using std::map;
using std::string;
using std::make_pair;


Server::Server(int port, GRand* pRand)
: GDynamicPageServer(port, pRand)
{
}

// virtual
Server::~Server()
{
}

// virtual
GDynamicPageSessionExtension* Server::deserializeSessionExtension(const GDomNode* pNode)
{
	return new Terminal(pNode, this);
}

// virtual
GDynamicPageConnection* Server::makeConnection(SOCKET sock)
{
	return new Connection(sock, this);
}


void makeHeader(ostream& response)
{
	//response << "<!DOCTYPE html>";
	response << "<html><head>\n";
	response << "	<meta charset=\"utf-8\">\n";
	response << "	<title>Community Modeler</title>\n";
	response << "	<style>\n";
	response << "body,center { font-family: verdana, tahoma, geneva, sans-serif; font-size: 18px; }\n";
	response << "a { text-decoration: none; }\n";
	response << "a:hover { text-decoration: underline; }\n";
	response << "pre { white-space: pre-wrap; }\n";
	response << ".mono { font-family: \"Lucida Console\", Monaco, monospace }\n";
	response << ".resolved { display: inline; text-decoration: none; pointer-events: none; }\n";
	response << ".c0 { background-color: #ffd0e0; }\n"; // pink
	response << ".c1 { background-color: #c0c0ff; }\n"; // purple
	response << ".c2 { background-color: #ffc020; }\n"; // orange
	response << ".c3 { background-color: #80f080; }\n"; // green
	response << ".dc { text-decoration: none; display: inline; }\n";
	response << "	</style>\n";
	response << "</head><body>\n";
	response << "<table align=center width=1200 cellpadding=0 cellspacing=0><tr><td>\n";
	response << "<table cellpadding=0 cellspacing=0>\n";

	// The header row
	response << "<tr><td colspan=2 id=\"header\">";
	response << "</td></tr>\n";

	// The main row
	response << "<tr>\n";

	// The left sidebar
	response << "<td id=\"sidebar\">";
	response << "</td><td id=\"mainbody\">\n\n\n\n";
}

void makeFooter(ostream& response)
{
	// End of main row
	response << "\n\n\n\n</td></tr>\n";

	// Footer row
	response << "<tr><td colspan=2 id=\"footer\">\n";
	response << "</td></tr></table>\n";
	response << "</td></tr></table></body></html>\n";
}


// virtual
void Connection::handleRequest(GDynamicPageSession* pSession, ostream& response)
{
	try
	{
		Server* pServer = (Server*)m_pServer;
		cout << "Requested: " << pSession->url() << "\n";
		makeHeader(response);
		Editor::pageDiff(pServer, pSession, response);
		makeFooter(response);
	}
	catch(std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << " when processing the url: " << m_szUrl << "\n";
		response << "Sorry, an error occurred. Please yell at the operator of this site.\n";
	}
}
