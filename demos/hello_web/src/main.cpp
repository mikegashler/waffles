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

#include <stdio.h>
#include <stdlib.h>
#ifdef WINDOWS
#	include <windows.h>
#	include <process.h>
#	include <direct.h>
#else
#	include <unistd.h>
#endif
#include <GClasses/GDynamicPage.h>
#include <GClasses/GApp.h>
#include <GClasses/GDom.h>
#include <GClasses/GString.h>
#include <GClasses/GHolders.h>
#include <GClasses/GFile.h>
#include <GClasses/GTime.h>
#include <GClasses/GRand.h>
#include <GClasses/GHashTable.h>
#include <GClasses/sha1.h>
#include <wchar.h>
#include <string>
#include <exception>
#include <iostream>
#include <sstream>

using namespace GClasses;
using std::cout;
using std::cerr;
using std::string;
using std::ostream;


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

class Server : public GDynamicPageServer
{
public:
	std::string m_basePath;

	Server(int port, GRand* pRand);
	virtual ~Server() {}
	virtual void onEverySixHours() {}
	virtual void onStateChange() {}
	virtual void onShutDown() {}

	virtual GDynamicPageConnection* makeConnection(SOCKET sock)
	{
		return new Connection(sock, this);
	}
};


// virtual
void Connection::handleRequest(GDynamicPageSession* pSession, std::ostream& response)
{
	if(strcmp(m_szUrl, "/favicon.ico") == 0)
		return;
	else if(strncmp(m_szUrl, "/hello", 6) == 0)
	{
		response << "<html><head>\n";
		response << "	<title>My Hello Site</title>\n";
		response << "	<link rel=\"stylesheet\" type=\"text/css\" href=\"/style.css\" />\n";
		response << "</head><body>\n";
		response << "	Hello Web! <img src=\"smiley.png\"><br>\n";
		response << "</body></html>\n";
	}
	else if(strcmp(m_szUrl, "/smiley.png") == 0)
		sendFileSafe(((Server*)m_pServer)->m_basePath.c_str(), m_szUrl + 1, response);
	else
		response << "<h1>404 - Not found!</h1>";
}


Server::Server(int port, GRand* pRand) : GDynamicPageServer(port, pRand)
{
	char buf[300];
	GTime::asciiTime(buf, 256, false);
	cout << "Server starting at: " << buf << "\n";
	GApp::appPath(buf, 256, true);
	strcat(buf, "web/");
	GFile::condensePath(buf);
	m_basePath = buf;
}

void getLocalStorageFolder(char* buf)
{
	if(!GFile::localStorageDirectory(buf))
		throw Ex("Failed to find local storage folder");
	strcat(buf, "/.hello/");
	GFile::makeDir(buf);
	if(!GFile::doesDirExist(buf))
		throw Ex("Failed to create folder in storage area");
}

void LaunchBrowser(const char* szAddress)
{
	int addrLen = strlen(szAddress);
	GTEMPBUF(char, szUrl, addrLen + 20);
	strcpy(szUrl, szAddress);
	strcpy(szUrl + addrLen, "/hello");
	if(!GApp::openUrlInBrowser(szUrl))
	{
		cout << "Failed to open the URL: " << szUrl << "\nPlease open this URL manually.\n";
		cout.flush();
	}
}

void doit(void* pArg)
{
	int port = 8989;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	GRand prng(seed);
	Server server(port, &prng);
	LaunchBrowser(server.myAddress());
	// Pump incoming HTTP requests (this is the main loop)
	cout << "Server is up\n";
	cout.flush();
	server.go();
	cout << "Goodbye.\n";
}

void doItAsDaemon()
{
	char path[300];
	getLocalStorageFolder(path);
	string s1 = path;
	s1 += "stdout.log";
	string s2 = path;
	s2 += "stderr.log";
	GApp::launchDaemon(doit, path, s1.c_str(), s2.c_str());
	cout << "Daemon running.\n	stdout >> " << s1.c_str() << "\n	stderr >> " << s2.c_str() << "\n";
}

int main(int nArgs, char* pArgs[])
{
	int nRet = 1;
	try
	{
		doit(NULL);
		//doItAsDaemon();
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
