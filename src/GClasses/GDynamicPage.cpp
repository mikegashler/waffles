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
#include "sha2.h"
#include "GRand.h"
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


GDynamicPageSession::GDynamicPageSession(unsigned long long ident)
: m_pConnection(nullptr)
{
	m_id = ident;
	onAccess();
	m_pExtension = nullptr;
}

GDynamicPageSession::GDynamicPageSession(GDynamicPageServer* pServer, GDomNode* pNode)
: m_pConnection(nullptr)
{
	m_id = (unsigned long long)pNode->getInt("id");
	m_tLastAccessed = (time_t)pNode->getInt("acc");
	GDomNode* pExt = pNode->getIfExists("ext");
	if(pExt)
		m_pExtension = pServer->deserializeSessionExtension(pExt);
	else
		m_pExtension = nullptr;
}

// virtual
GDynamicPageSession::~GDynamicPageSession()
{
	if(m_pExtension)
		m_pExtension->onDisown();
}

GDynamicPageServer* GDynamicPageSession::server()
{
	return m_pConnection->m_pServer;
}

void GDynamicPageSession::setConnection(GDynamicPageConnection* pConn)
{
	m_pConnection = pConn;
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

GDomNode* GDynamicPageSession::serialize(GDom* pDoc)
{
	GDomNode* pObj = pDoc->newObj();
	pObj->add(pDoc, "id", (long long)m_id);
	pObj->add(pDoc, "acc", (long long)m_tLastAccessed);
	if(m_pExtension)
		pObj->add(pDoc, "ext", m_pExtension->serialize(pDoc));
	return pObj;
}

const char* GDynamicPageSession::url()
{
	return m_pConnection->m_szUrl;
}

const char* GDynamicPageSession::params()
{
	return m_pConnection->m_pContent;
}

size_t GDynamicPageSession::paramsLen()
{
	return m_pConnection->m_nContentLength;
}

void GDynamicPageSession::addAjaxCookie(GDom& docOut, GDomNode* pOutNode)
{
  std::ostringstream os;
  os << "GDPSI=";
	os << m_id;
	pOutNode->add(&docOut, "cookie", os.str().c_str());
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

string remove_cr_lf(const char* str)
{
	string s = str;
	for(size_t i = 0; i < s.length(); i++)
	{
		if(s[i] == '\r' || s[i] == '\n')
			s[i] = '_';
	}
	return s;
}

const char* _strnstr(const char* big, const char* small, size_t len)
{
	for(size_t i = 0; i < len; i++)
	{
		const char* a = big;
		const char* b = small;
		while(*b != '\0' && *a == *b)
		{
			++a;
			++b;
		}
		if(*b == '\0')
			return big;
		++big;
	}
	return nullptr;
}

GDynamicPageSession* GDynamicPageConnection::establishSession()
{
	// Check for a cookie in the HTTP header
	GDynamicPageSession* pSession = NULL;
	if(m_szCookieIncoming[0] != '\0')
	{
		const char* crumb = strstr(m_szCookieIncoming, "GDPSI=");
		if(crumb)
		{
			crumb += 6;
#ifdef WINDOWS
			unsigned long long nSessionID = _strtoui64(crumb, NULL, 10);
#else
			unsigned long long nSessionID = strtoull(crumb, NULL, 10);
#endif
			pSession = m_pServer->findSession(nSessionID);
			if(!pSession)
			{
				cout << "Unrecognized session id cookie crumb from " << inet_ntoa(ipAddr()) << ": " << remove_cr_lf(crumb) << "\n";
				cout.flush();
			}
		}
		else
		{
			cout << "Cookie with no GDPSI= cookie crumb from " << inet_ntoa(ipAddr()) << ": " << m_szCookieIncoming << "\n";
			cout.flush();
		}
	}

	// Check for a cookie in the content
	if(!pSession)
	{
		const char* crumb = _strnstr(m_pContent, "GDPSI=", m_nContentLength);
		if(crumb)
		{
			crumb += 6;
#ifdef WINDOWS
			unsigned long long nSessionID = _strtoui64(crumb, NULL, 10);
#else
			unsigned long long nSessionID = strtoull(crumb, NULL, 10);
#endif
			pSession = m_pServer->findSession(nSessionID);
			if(!pSession)
			{
				cout << "Unrecognized session id cookie crumb from " << inet_ntoa(ipAddr()) << "\n";
				cout.flush();
			}
		}
	}

	// Make a new session
	if(!pSession)
	{
		// Make a new cookie
		unsigned long long nSessionID = (unsigned long long)m_pServer->prng()->next() ^ (unsigned long long)(GTime::seconds() * 10000);
		std::ostringstream os;
		os << "GDPSI=";
		os << nSessionID;
		string tmp = os.str();
		setCookie(tmp.c_str(), true);
		pSession = m_pServer->makeNewSession(nSessionID);
	}
	pSession->setConnection(this); // just sets a pointer back to this object

	return pSession;
}

// virtual
void GDynamicPageConnection::doGet(ostream& response)
{
	// Set up the session
	GDynamicPageSession* pSession = establishSession();

	// Handle the request
	setContentType("text/html");
	handleRequest(pSession, response);
}

// virtual
void GDynamicPageConnection::doPost(ostream& response)
{
	GDynamicPageSession* pSession = establishSession();
	handleRequest(pSession, response);
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
	//cout << "Requested file: " << szLocalPath;
	//cout.flush();

	// Make sure the file is within the jail
	size_t jailLen = strlen(szJailPath);
	size_t localLen = strlen(szLocalPath);
	GTEMPBUF(char, buf, jailLen + localLen + 1);
	strcpy(buf, szJailPath);
	strcpy(buf + jailLen, szLocalPath);
	GFile::condensePath(buf);
	if(strncmp(buf, szJailPath, jailLen) != 0)
	{
		//cout << "Denied because " << buf << " it is not within " << szJailPath << "\n";
		return;
	}

	// Send the file
	if(GFile::doesFileExist(buf))
	{
		// Send the file
		try
		{
			sendFile(extensionToMimeType(buf), buf, response);
			//cout << ", Sent: " << buf << "\n";
			//cout.flush();
		}
		catch(const char* szError)
		{
			//cout << ", Error sending file: " << buf << ", " << szError << "\n";
			//cout.flush();
			return;
		}
	}
	else
	{
		//cout << ", Not found: " << buf << "\n";
		//cout.flush();
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
	GDynamicPageSession* pSession = new GDynamicPageSession(id);
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
	unsigned char digest[SHA512_DIGEST_LENGTH];
	sha512_ctx ctx;
	memset(&ctx, '\0', sizeof(sha512_ctx));
	sha512_begin(&ctx);
	sha512_hash((unsigned char*)m_daemonSalt, (unsigned int)strlen(m_daemonSalt), &ctx);
	sha512_hash((unsigned char*)"ajbiekistwgcdpcm", 16, &ctx);
	sha512_end(digest, &ctx);
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

GDomNode* GDynamicPageServer::serialize(GDom* pDoc)
{
	GDomNode* pObj = pDoc->newObj();
	pObj->add(pDoc, "daemonSalt", m_daemonSalt);
	pObj->add(pDoc, "pwSalt", m_passwordSalt);
	GDomNode* pSessionList = pDoc->newList();
	pObj->add(pDoc, "sessions", pSessionList);
	for(std::map<unsigned long long, GDynamicPageSession*>::iterator it = m_sessions.begin(); it != m_sessions.end(); it++)
	{
		GDynamicPageSession* pSession = it->second;
		GDomNode* pSessionNode = pSession->serialize(pDoc);
		pSessionList->add(pDoc, pSessionNode);
	}
	return pObj;
}

void GDynamicPageServer::deserialize(const GDomNode* pNode)
{
	const char* szDaemonSalt = pNode->getString("daemonSalt");
	if(strlen(szDaemonSalt) > 14)
		throw Ex("bad daemon salt size");
	strcpy(m_daemonSalt, szDaemonSalt);
	const char* szPasswordSalt = pNode->getString("pwSalt");
	if(strlen(szPasswordSalt) > 14)
		throw Ex("bad pw salt size");
	strcpy(m_passwordSalt, szPasswordSalt);
	flushSessions();
	GDomNode* pSessionList = pNode->get("sessions");
	GDomListIterator it(pSessionList);
	while(it.remaining() > 0)
	{
		GDomNode* pSessionNode = it.current();
		unsigned long long id = (unsigned long long)pSessionNode->getInt("id");
		GDynamicPageSession* pSession = new GDynamicPageSession(this, pSessionNode);
		m_sessions.insert(std::pair<unsigned long long, GDynamicPageSession*>(id, pSession));
		it.advance();
	}
}
