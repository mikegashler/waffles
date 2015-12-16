/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include "GDynamicPage.h"
#include "GApp.h"
#include "GBits.h"
#include "GHashTable.h"
#include "GDom.h"
#include "GImage.h"
#include "GHolders.h"
#include "GFile.h"
#include "GThread.h"
#include "GString.h"
#include "GSocket.h"
#include "GTime.h"
#include "GRand.h"
#include "sha1.h"
#ifndef WINDOWS
#	include <unistd.h>
#	include <arpa/inet.h>
#endif
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <memory>

using namespace GClasses;
using std::vector;
using std::string;
using std::cout;
using std::make_pair;
using std::map;
using std::ostream;
using std::ostringstream;


GDynamicPageSession::GDynamicPageSession(GDynamicPageServer* pServer, unsigned long long ident)
{
	m_id = ident;
	m_pServer = pServer;
	onAccess();
	m_pExtension = NULL;
}

// virtual
GDynamicPageSession::~GDynamicPageSession()
{
	if(m_pExtension)
		m_pExtension->onDisown();
}

void GDynamicPageSession::onAccess()
{
	time_t t;
	m_tLastAccessed = time(&t);
}

void GDynamicPageSession::setExtension(GDynamicPageSessionExtension* pExtension)
{
	if(m_pExtension)
		m_pExtension->onDisown();
	m_pExtension = pExtension;
}

// ------------------------------------------------------

GDynamicPageConnection::GDynamicPageConnection(SOCKET sock, GDynamicPageServer* pServer)
: GHttpConnection(sock), m_pServer(pServer)
{
}

// virtual
GDynamicPageConnection::~GDynamicPageConnection()
{
}

// virtual
bool GDynamicPageConnection::hasBeenModifiedSince(const char* szUrl, const char* szDate)
{
	return true;
}

GDynamicPageSession* GDynamicPageConnection::establishSession()
{
	// Find existing session
	unsigned long long nSessionID;
	GDynamicPageSession* pSession = NULL;
	if(m_szCookieIncoming[0] != '\0')
	{
		const char* crumb = strstr(m_szCookieIncoming, "GDPSI=");
		if(crumb)
		{
			crumb += 6;
#ifdef WINDOWS
			nSessionID = _strtoui64(crumb, NULL, 10);
#else
			nSessionID = strtoull(crumb, NULL, 10);
#endif
			pSession = m_pServer->findSession(nSessionID);
			if(!pSession)
			{
				cout << "Unrecognized session id cookie crumb from " << inet_ntoa(ipAddr()) << ": " << crumb << "\n";
				cout.flush();
			}
		}
		else
		{
			cout << "Cookie with no GDPSI cookie crumb from " << inet_ntoa(ipAddr()) << ": " << m_szCookieIncoming << "\n";
			cout.flush();
		}
	}

	// Make a new session
	if(!pSession)
	{
		// Make a new cookie
		nSessionID = (unsigned long long)m_pServer->prng()->next() ^ (unsigned long long)(GTime::seconds() * 10000);
		std::ostringstream os;
		os << "GDPSI=";
		os << nSessionID;
		string tmp = os.str();
		setCookie(tmp.c_str(), true);
		pSession = m_pServer->makeNewSession(nSessionID);
	}

	return pSession;
}

// virtual
void GDynamicPageConnection::doGet(ostream& response)
{
	// Set up the session
	GDynamicPageSession* pSession = establishSession();
	pSession->setCurrentUrl(m_szUrl, m_pContent, m_nContentLength);

	// Handle the request
	setContentType("text/html");
	handleRequest(pSession, response);
}

// virtual
void GDynamicPageConnection::doPost(ostream& response)
{
	doGet(response);
}

// virtual
void GDynamicPageConnection::setHeaders(const char* szUrl, const char* szParams)
{
	// todo: write me
}

void GDynamicPageConnection::sendFile(const char* szMimeType, const char* szFilename, ostream& response)
{
	// Load the file
	size_t nSize;
	char* pFile = GFile::loadFile(szFilename, &nSize);
	std::unique_ptr<char[]> hFile(pFile);

	// Set the headers
	setContentType(szMimeType);
	setModifiedTime(GFile::modifiedTime(szFilename));

	// Send the file
	response.write(pFile, nSize);
}

void GDynamicPageConnection::sendFileSafe(const char* szJailPath, const char* szLocalPath, ostream& response)
{
	// Make sure the file is within the jail
	size_t jailLen = strlen(szJailPath);
	size_t localLen = strlen(szLocalPath);
	GTEMPBUF(char, buf, jailLen + localLen + 1);
	strcpy(buf, szJailPath);
	strcpy(buf + jailLen, szLocalPath);
	GFile::condensePath(buf);
	if(strncmp(buf, szJailPath, jailLen) != 0)
		return;

	// Send the file
	if(GFile::doesFileExist(buf))
	{
		// Send the file
		try
		{
			sendFile(extensionToMimeType(buf), buf, response);
		}
		catch(const char* szError)
		{
			cout << "Error sending file: " << buf << "\n" << szError;
			return;
		}
	}
	else
	{
		cout << "Not found: " << szJailPath << szLocalPath << "\n";
		response << "404 - not found.<br><br>\n";
	}
}

const char* g_pExtensionToMimeTypeHackTable[] =
{
	".png", "image/png",
	".js", "text/javascript",
	".jpg", "image/jpeg",
	".gif", "image/gif",
};

// static
const char* GDynamicPageConnection::extensionToMimeType(const char* szFilename)
{
	PathData pd;
	GFile::parsePath(szFilename, &pd);
	const char* szExt = szFilename + pd.extStart;
	const char* szMimeType = "text/html";
	for(size_t i = 0; i < sizeof(g_pExtensionToMimeTypeHackTable) / sizeof(const char*); i += 2)
	{
		if(_stricmp(szExt, g_pExtensionToMimeTypeHackTable[i]) == 0)
		{
			szMimeType = g_pExtensionToMimeTypeHackTable[i + 1];
			break;
		}
	}
	return szMimeType;
}


// ------------------------------------------------------

GDynamicPageServer::GDynamicPageServer(int port, GRand* pRand)
: GHttpServer(port), m_pRand(pRand)
{
	m_bKeepGoing = true;

	// Init captcha salt
	int i;
	for(i = 0; i < 14; i++)
	{
		do
		{
			m_daemonSalt[i] = ' ' + (char)m_pRand->next(95);
		} while(m_daemonSalt[i] == '"' ||
			m_daemonSalt[i] == '&' ||
			m_daemonSalt[i] == '<' ||
			m_daemonSalt[i] == '>');
	}
	m_daemonSalt[i] = '\0';
	computePasswordSalt();

	// Determine my address
	char szBuf[300];
	strcpy(szBuf, "http://");
	try
	{
		GTCPServer::hostName(szBuf + 7, 256);
	}
	catch(...)
	{
		strcpy(szBuf + 7, "localhost");
	}
	strcat(szBuf + 7, ":");
	std::ostringstream os;
	os << port;
	string tmp = os.str();
	strcat(szBuf + 8, tmp.c_str());
	m_szMyAddress = new char[strlen(szBuf) + 1];
	strcpy(m_szMyAddress, szBuf);
}

GDynamicPageServer::~GDynamicPageServer()
{
	flushSessions();
	delete[] m_szMyAddress;
}

void GDynamicPageServer::flushSessions()
{
	for(map<unsigned long long, GDynamicPageSession*>::iterator it = m_sessions.begin(); it != m_sessions.end(); it++)
		delete(it->second);
	m_sessions.clear();
}

GDynamicPageSession* GDynamicPageServer::makeNewSession(unsigned long long id)
{
	GDynamicPageSession* pSession = new GDynamicPageSession(this, id);
	m_sessions.insert(make_pair(id, pSession));
	return pSession;
}

GDynamicPageSession* GDynamicPageServer::findSession(unsigned long long id)
{
	map<unsigned long long, GDynamicPageSession*>::iterator it = m_sessions.find(id);
	if(it == m_sessions.end())
		return NULL;
	else
		return it->second;
}

void GDynamicPageServer::printSessionIds(std::ostream& stream)
{
	map<unsigned long long, GDynamicPageSession*>::iterator it;
	for(it = m_sessions.begin(); it != m_sessions.end(); it++)
	{
		stream << "	" << it->first << "\n";
	}
	stream.flush();
}

void GDynamicPageServer::go()
{
	double dLastMaintenance = GTime::seconds();
	GSignalHandler sh;
	while(m_bKeepGoing && sh.check() == 0)
	{
		if(!process())
		{
			if(GTime::seconds() - dLastMaintenance > 14400)	// 4 hours
			{
				doMaintenance();
				dLastMaintenance = GTime::seconds();
			}
			else
				GThread::sleep(100);
		}
	}
	onShutDown();
}

void GDynamicPageServer::shutDown()
{
	m_bKeepGoing = false;
}

void GDynamicPageServer::doMaintenance()
{
	onEverySixHours();
}

const char* GDynamicPageServer::myAddress()
{
	return m_szMyAddress;
}

const char* GDynamicPageServer::daemonSalt()
{
	return m_daemonSalt;
}

void GDynamicPageServer::setDaemonSalt(const char* szSalt)
{
	if(strlen(szSalt) != 14)
		throw Ex("Salt has unexpected length");
	strcpy(m_daemonSalt, szSalt);
	computePasswordSalt();
}

const char* GDynamicPageServer::passwordSalt()
{
	return m_passwordSalt;
}

void GDynamicPageServer::computePasswordSalt()
{
	unsigned char digest[20];
	SHA_CTX ctx;
	memset(&ctx, '\0', sizeof(SHA_CTX));
	SHA1_Init(&ctx);
	SHA1_Update(&ctx, (unsigned char*)m_daemonSalt, (unsigned int)strlen(m_daemonSalt));
	SHA1_Update(&ctx, (unsigned char*)"ajbiekistwgcdpcm", 16);
	SHA1_Final(digest, &ctx);
	for(int i = 0; i < 14; i++)
	{
		char c = digest[i] % 52;
		if(c >= 26)
			c += 6;
		c += 'A';
		m_passwordSalt[i] = c;
	}
	m_passwordSalt[14] = '\0';
}

void GDynamicPageServer::redirect(std::ostream& response, const char* szUrl)
{
	ostringstream& r = reinterpret_cast<ostringstream&>(response);
	r.str("");
	r.clear();
	r << "<html><head><META HTTP-EQUIV=\"Refresh\" CONTENT=\"0; URL=";
	r << szUrl;
	r << "\"></head><body>If your browser doesn't automatically redirect, please click <a href=\"";
	r << szUrl;
	r << "\">here</a>\n";
}

