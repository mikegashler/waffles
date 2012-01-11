/*
	Copyright (C) 2009, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GDynamicPage.h"
#include "GApp.h"
#include "GBits.h"
#include "GHashTable.h"
#include "GDom.h"
#include "GHeap.h"
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
#endif
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>

using namespace GClasses;
using std::vector;
using std::string;
using std::cout;
using std::make_pair;
using std::map;
using std::ostream;
using std::ostringstream;


GDynamicPageSession::GDynamicPageSession(GDynamicPageServer* pServer, unsigned long long id)
{
	m_id = id;
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
		ThrowError("Salt has unexpected length");
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

// virtual
bool GDynamicPageServer::hasBeenModifiedSince(const char* szUrl, const char* szDate)
{
	return true;
}

GDynamicPageSession* GDynamicPageServer::establishSession(const char* szCookie)
{
	// Find existing session
	unsigned long long nSessionID;
	GDynamicPageSession* pSession = NULL;
	if(szCookie)
	{
#ifdef WINDOWS
		nSessionID = _strtoui64(szCookie, NULL, 10);
#else
		nSessionID = strtoull(szCookie, NULL, 10);
#endif
		map<unsigned long long, GDynamicPageSession*>::iterator it = m_sessions.find(nSessionID);
		if(it != m_sessions.end())
			pSession = it->second;
	}

	// Make a new session
	if(!pSession)
	{
		if(szCookie && *szCookie >= '0' && *szCookie <= '9')
		{
			// It's an old session of which we no longer have record
#ifdef WIN32
			nSessionID = _strtoui64(szCookie, NULL, 10);
#else
			nSessionID = strtoull(szCookie, NULL, 10);
#endif
		}
		else
		{
			// Make a new cookie
			nSessionID = (unsigned long long)m_pRand->next();
			std::ostringstream os;
			os << nSessionID;
			string tmp = os.str();
			setCookie(tmp.c_str(), true);
		}
		pSession = new GDynamicPageSession(this, nSessionID);
		m_sessions.insert(make_pair(nSessionID, pSession));
	}

	return pSession;
}

// virtual
void GDynamicPageServer::doGet(const char* szUrl, const char* szParams, size_t nParamsLen, const char* szCookie, ostream& response)
{
	// Set up the session
	GDynamicPageSession* pSession = establishSession(szCookie);
	pSession->setCurrentUrl(szUrl, szParams, nParamsLen);

	// Handle the request
	setContentType("text/html");
	handleRequest(szUrl, szParams, (int)nParamsLen, pSession, response);
}

// virtual
void GDynamicPageServer::doPost(const char* szUrl, unsigned char* pData, size_t nDataSize, const char* szCookie, ostream& pResponse)
{
	doGet(szUrl, (const char*)pData, nDataSize, szCookie, pResponse);
	delete[] pData;
}

void GDynamicPageServer::sendFile(const char* szMimeType, const char* szFilename, ostream& response)
{
	// Load the file
	size_t nSize;
	char* pFile = GFile::loadFile(szFilename, &nSize);
	ArrayHolder<char> hFile(pFile);

	// Set the headers
	setContentType(szMimeType);
	setModifiedTime(GFile::modifiedTime(szFilename));

	// Send the file
	response.write(pFile, nSize);
}

// virtual
void GDynamicPageServer::setHeaders(const char* szUrl, const char* szParams)
{
	// todo: write me
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

const char* g_pExtensionToMimeTypeHackTable[] =
{
	".png", "image/png",
	".js", "text/javascript",
	".jpg", "image/jpeg",
	".gif", "image/gif",
};

void GDynamicPageServer::sendFileSafe(const char* szJailPath, const char* szLocalPath, ostream& response)
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
		// Find the extension
		PathData pd;
		GFile::parsePath(buf, &pd);
		const char* szExt = buf + pd.extStart;

		// Determine the mime type
		const char* szMimeType = "text/html";
		for(size_t i = 0; i < sizeof(g_pExtensionToMimeTypeHackTable) / sizeof(const char*); i += 2)
		{
			if(_stricmp(szExt, g_pExtensionToMimeTypeHackTable[i]) == 0)
			{
				szMimeType = g_pExtensionToMimeTypeHackTable[i + 1];
				break;
			}
		}

		// Send the file
		try
		{
			sendFile(szMimeType, buf, response);
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
