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

#ifndef SERVER
#define SERVER

#include <map>
#include <string>
#include <GClasses/GFile.h>
#include <GClasses/GDynamicPage.h>

class Account;

using namespace GClasses;


class Server : public GDynamicPageServer
{
protected:
	GFileCache m_fileCache;

public:
	std::string m_basePath;

	Server(int port, GRand* pRand);
	virtual ~Server();
	virtual GDynamicPageSessionExtension* deserializeSessionExtension(const GDomNode* pNode);
	virtual GDynamicPageConnection* makeConnection(SOCKET sock);
};






class Terminal : public GDynamicPageSessionExtension
{
public:
	Terminal() : GDynamicPageSessionExtension()
	{
	}

	Terminal(const GDomNode* pNode, Server* pServer)
	{
	}

	virtual ~Terminal()
	{
	}

	virtual GDomNode* serialize(GDom* pDoc)
	{
		return nullptr;
	}

	/// Called when the sessions are destroyed, or a new GDynamicPageSessionExtension is
	/// explicitly associated with this cookie.
	virtual void onDisown()
	{
	}
};


class Connection : public GDynamicPageConnection
{
public:
	Connection(SOCKET sock, GDynamicPageServer* pServer) : GDynamicPageConnection(sock, pServer)
	{
	}
	
	virtual ~Connection()
	{
	}

	virtual void handleRequest(GDynamicPageSession* pSession, std::ostream& response);
};







Terminal* getTerminal(GDynamicPageSession* pSession);

Account* getAccount(GDynamicPageSession* pSession);



#endif // SERVER
