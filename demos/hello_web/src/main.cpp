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
#include <GClasses/GTwt.h>
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


void OpenUrl(const char* szUrl)
{
#ifdef WINDOWS
	// Windows
	ShellExecute(NULL, NULL, szUrl, NULL, NULL, SW_SHOW);
#else
#ifdef DARWIN
	// Mac
	GTEMPBUF(char, pBuf, 32 + strlen(szUrl));
	strcpy(pBuf, "open ");
	strcat(pBuf, szUrl);
	strcat(pBuf, " &");
	system(pBuf);
#else // DARWIN
	GTEMPBUF(char, pBuf, 32 + strlen(szUrl));

	// Gnome
	strcpy(pBuf, "gnome-open ");
	strcat(pBuf, szUrl);
	if(system(pBuf) != 0)
	{
		// KDE
		//strcpy(pBuf, "kfmclient exec ");
		strcpy(pBuf, "konqueror ");
		strcat(pBuf, szUrl);
		strcat(pBuf, " &");
		if(system(pBuf) != 0)
			cout << "Failed to open " << szUrl << ". Please open it manually.\n";
	}
#endif // !DARWIN
#endif // !WINDOWS
}

void LaunchBrowser(const char* szAddress)
{
	int addrLen = strlen(szAddress);
	GTEMPBUF(char, szUrl, addrLen + 20);
	strcpy(szUrl, szAddress);
	strcpy(szUrl + addrLen, "/hello");
	OpenUrl(szUrl);
}

void redirectStandardStreams(const char* pPath)
{
	string s1(pPath);
	s1 += "stdout.log";
	if(!freopen(s1.c_str(), "a", stdout))
	{
		cout << "Error redirecting stdout\n";
		cerr << "Error redirecting stdout\n";
		ThrowError("Error redirecting stdout");
	}
	string s2(pPath);
	s2 += "stderr.log";
	if(!freopen(s2.c_str(), "a", stderr))
	{
		cout << "Error redirecting stderr\n";
		cerr << "Error redirecting stderr\n";
		ThrowError("Error redirecting stderr");
	}
}

// ********* Uncomment the following line to run as a daemon **********
//#define RUN_AS_DAEMON

void doit(void* pArg)
{
	int port = 8989;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	GRand prng(seed);
#ifdef RUN_AS_DAEMON
	redirectStandardStreams((const char*)pArg);
	Server server(port, &prng);
#else
	Server server(port, &prng);
	LaunchBrowser(server.myAddress());
#endif
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
	int pid = GApp::launchDaemon(doit, path);
	cout << "Daemon running.\n	pid=" << pid << "\n	stdout >> " << s1.c_str() << "\n	stderr >> " << s2.c_str() << "\n";
}

int main(int nArgs, char* pArgs[])
{
	int nRet = 1;
	try
	{
#ifdef RUN_AS_DAEMON
		doItAsDaemon();
#else
		doit(NULL);
#endif
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
