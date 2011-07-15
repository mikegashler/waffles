// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

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



class Server : public GDynamicPageServer
{
protected:
	std::string m_basePath;

public:
	Server(int port, GRand* pRand);
	virtual ~Server() {}
	virtual void handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, std::ostream& response);
	virtual void onEverySixHours() {}
	virtual void onStateChange() {}
	virtual void onShutDown() {}
};


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

// virtual
void Server::handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, std::ostream& response)
{
	if(strcmp(szUrl, "/") == 0)
		szUrl = "/hello";
	if(strcmp(szUrl, "/favicon.ico") == 0)
		return;
	if(strncmp(szUrl, "/hello", 6) == 0)
	{
		response << "<html><head>\n";
		response << "	<title>My Hello Site</title>\n";
		response << "	<link rel=\"stylesheet\" type=\"text/css\" href=\"/style.css\" />\n";
		response << "</head><body>\n";
		response << "	Hello Web! <img src=\"smiley.png\"><br>\n";
		response << "</body></html>\n";
	}
	else
		sendFileSafe(m_basePath.c_str(), szUrl + 1, response);
}

void getLocalStorageFolder(char* buf)
{
	if(!GFile::localStorageDirectory(buf))
		ThrowError("Failed to find local storage folder");
	strcat(buf, "/.hello/");
	GFile::makeDir(buf);
	if(!GFile::doesDirExist(buf))
		ThrowError("Failed to create folder in storage area");
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
