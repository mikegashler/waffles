/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "../GClasses/GFile.h"
#include "GHttp.h"
#include "GSocket.h"
#include "../GClasses/GString.h"
#include "../GClasses/GError.h"
#include "../GClasses/GHolders.h"
#include "../GClasses/GTime.h"
#include "../GClasses/GThread.h" // Sleep
#include "../GClasses/GHashTable.h"
#include "../GClasses/GHeap.h"
#include <time.h>
#include <string>
#include <sstream>

using namespace GClasses;
using std::vector;
using std::string;
using std::ostream;
using std::ostringstream;

namespace GClasses{

class GHttpClientSocket : public GSocketClient
{
protected:
	GHttpClient* m_pParent;

public:
	GHttpClientSocket(GHttpClient* pParent, int nMaxPacketSize) : GSocketClient(false, nMaxPacketSize)
	{
		m_pParent = pParent;
	}

	virtual ~GHttpClientSocket()
	{
	}

	static GHttpClientSocket* ConnectToTCPSocket(GHttpClient* pParent, const char* szHost, int nPort)
	{
		GHttpClientSocket* pSocket = new GHttpClientSocket(pParent, 0);
		if(!pSocket)
			return NULL;
		if(!pSocket->Connect(szHost, nPort))
		{
			delete(pSocket);
			return NULL;
		}
		return pSocket;
	}

protected:
	virtual void onLoseConnection(int nSocketNumber)
	{
		m_pParent->onLoseConnection();
	}
};
}

// -------------------------------------------------------------------------------

GHttpClient::GHttpClient()
{
	m_pSocket = NULL;
	m_status = Error;
	m_pData = NULL;
	m_bPastHeader = false;
	m_aborted = false;
	strcpy(m_szServer, "\0");
	m_szRedirect = NULL;
	m_dLastReceiveTime = 0;
	strcpy(m_szClientName, "GHttpClient/1.0");
}

GHttpClient::~GHttpClient()
{
	delete(m_pSocket);
	delete(m_pData);
	delete(m_szRedirect);
}

void GHttpClient::abort()
{
	m_pSocket->Disconnect();
	m_status = Aborted;
}

void GHttpClient::setClientName(const char* szClientName)
{
	strncpy(m_szClientName, szClientName, 32);
	m_szClientName[31] = '\0';
}

GHttpClient::Status GHttpClient::status(float* pfProgress)
{
	while(true)
	{
		if(m_aborted || !m_pSocket)
			return Aborted;
		if(m_pSocket->GetMessageCount() <= 0)
			break;
		m_dLastReceiveTime = GTime::seconds();
		int nSize;
		const unsigned char* szChunk = m_pSocket->GetNextMessage(&nSize);
		onReceiveData(szChunk, nSize);
		if(m_bPastHeader)
			processBody(szChunk, nSize);
		else
			processHeader(szChunk, nSize);
		if(pfProgress)
		{
			if(m_nContentSize)
				*pfProgress = (float)m_nDataPos / m_nContentSize;
			else
				*pfProgress = 0;
		}
	}
	return m_status;
}

bool GHttpClient::get(const char* szUrl, bool actuallyGetData) // actuallyGetData default = true todo rename sendRequestToserver
{
	// todo: make it more lenient with the timeout values for downloading the heads...

	GTEMPBUF(char, szNewUrl, (int)strlen(szUrl) + 1);
	strcpy(szNewUrl, szUrl);
	GFile::condensePath(szNewUrl);

	// Get the port
	int nHostIndex, nPortIndex, nPathIndex;
	GHttpClientSocket::parseURL(szNewUrl, &nHostIndex, &nPortIndex, &nPathIndex, NULL);
	int nPort;
	if(nPathIndex > nPortIndex)
		nPort = atoi(&szNewUrl[nPortIndex + 1]); // the "+1" is for the ':'
	else
		nPort = 80;

	// Copy the host name
	int nTempBufSize = nPortIndex - nHostIndex + 1;
	GTEMPBUF(char, szHost, nTempBufSize);
	memcpy(szHost, &szNewUrl[nHostIndex], nPortIndex - nHostIndex);
	szHost[nPortIndex - nHostIndex] = '\0';

	// Connect
	m_aborted = true;
	if(!m_pSocket || GTime::seconds() - m_dLastReceiveTime > 10 || !m_pSocket->isConnected() || strcmp(szHost, m_szServer) != 0)
	{
		delete(m_pSocket);
		m_pSocket = NULL;
		try
		{
			m_pSocket = GHttpClientSocket::ConnectToTCPSocket(this, szHost, nPort);
		}
		catch(...)
		{
			return false;
		}
		if(!m_pSocket)
			return false;
		strncpy(m_szServer, szHost, 255);
		m_szServer[255] = '\0';
	}

	// Send the request
	const char* szPath = &szNewUrl[nPathIndex];
	if(szPath[0] == 0)
		szPath = "/index.html";
	string s;
	if(actuallyGetData)
	{
		//do GET
		m_bAmCurrentlyDoingJustHeaders = false;
		s = "GET ";
	}
	else
	{
		//Do HEAD
		m_bAmCurrentlyDoingJustHeaders = true;
		s = "HEAD ";
	}
	while(*szPath != '\0')
	{
		if(*szPath == ' ')
			s += "%20";
		else
			s += *szPath;
		szPath++;
	}
	s += " HTTP/1.1\r\n";
	s += "Host: ";
	s += szHost;
	s += ":";
	s += nPort;
// todo: undo the next line
//	s += "\r\nUser-Agent: Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.7.12) Gecko/20051010 Firefox/1.0.7 (Ubuntu package 1.0.7)\r\nAccept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5\r\nAccept-Language: en-us,en;q=0.5\r\nAccept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\nKeep-Alive: 300\r\nConnection: keep-alive\r\n\r\n";
	s += "\r\nUser-Agent: ";
	s += m_szClientName;
	s += "\r\nKeep-Alive: 60\r\nConnection: keep-alive\r\n\r\n";
	if(!m_pSocket->Send((unsigned char*)s.c_str(), (int)s.length()))
		return false;

	// Update [reset] status
	m_nContentSize = 0;
	m_nDataPos = 0;
	m_bChunked = false;
	m_bPastHeader = false;
	m_nHeaderPos = 0;
	delete(m_pData);
	m_pData = NULL;
	m_status = Downloading;
	m_aborted = false;
	return true;
}

void GHttpClient::processHeader(const unsigned char* szData, int nSize)
{
	while(nSize > 0)
	{
		if(m_nHeaderPos < 256)
			m_szHeaderBuf[m_nHeaderPos++] = *szData;
		if(*szData == '\n')
		{
			m_szHeaderBuf[m_nHeaderPos] = '\0';
			if(m_nHeaderPos <= 2)
			{
				szData++;
				nSize--;
				m_bPastHeader = true;
				if(m_bAmCurrentlyDoingJustHeaders)
				{
					m_status = Done;
					GAssert(!(nSize > 0 && (m_bChunked || m_nContentSize > 0))); // uh we shouldn't have a body we were just waiting for the headers!
				}

				if(m_szRedirect)
				{
					if(!get(m_szRedirect))
						m_status = Error;
					delete(m_szRedirect);
					m_szRedirect = NULL;
				}
				else if(nSize > 0 && (m_bChunked || m_nContentSize > 0))
				{
					processBody(szData, nSize);
				}
				return;
			}
			if(_strnicmp(m_szHeaderBuf, "HTTP/", 5) == 0)
			{
				char* szTmp = m_szHeaderBuf + 5;
				while(*szTmp != ' ' && *szTmp != '\n')
					szTmp++;
				if(*szTmp == ' ')
					szTmp++;
				if(*szTmp == '4')
					m_status = NotFound;
				else if(*szTmp != '2')
					m_status = Error;
				else
					m_nContentSize = 0;
			}
			else if(_strnicmp(m_szHeaderBuf, "Content-Length:", 15) == 0)
			{
				char* szTmp = m_szHeaderBuf + 15;
				while(*szTmp != '\n' && *szTmp <= ' ')
					szTmp++;
				m_nContentSize = atoi(szTmp);
				if(m_nContentSize > 0)
				{
					m_pData = new unsigned char[m_nContentSize + 1];
					m_nDataPos = 0;
				}
			}
			else if(_strnicmp(m_szHeaderBuf, "Transfer-Encoding: chunked", 26) == 0)
			{
				m_bChunked = true;
			}
			else if(_strnicmp(m_szHeaderBuf, "Location:", 9) == 0)
			{
				const char* szLoc = m_szHeaderBuf + 9;
				while(*szLoc > '\0' && *szLoc <= ' ')
					szLoc++;
				int nLen = (int)strlen(szLoc);
				delete(m_szRedirect);
				m_szRedirect = new char[nLen + 1];
				strcpy(m_szRedirect, szLoc);
			}
/*			else if(strnicmp(m_szHeaderBuf, "Last-Modified:", 14) == 0)
			{
				strcpy(caLastModifiedString, m_szHeaderBuf);
			}*/

			m_nHeaderPos = 0;
		}
		szData++;
		nSize--;
	}
}

void GHttpClient::processBody(const unsigned char* szData, int nSize)
{
	if(m_bChunked)
		processChunkBody(szData, nSize);
	else if(m_nContentSize > 0)
	{
		if(m_nDataPos + nSize > m_nContentSize)
			nSize = m_nContentSize - m_nDataPos;
		memcpy(m_pData + m_nDataPos, szData, nSize);
		m_nDataPos += nSize;
		if(m_nDataPos >= m_nContentSize)
		{
			if(m_status == Downloading)
			{
				m_status = Done; // too bad this involves polling...teeny teeny lag :)
			}
			m_pData[m_nContentSize] = '\0';
			m_bPastHeader = false; // reset this baaad boy.  Hmm.
		}
	}
	else
		m_chunkBuf.write((const char*)szData, nSize);
}

void GHttpClient::onLoseConnection()
{
	if(m_bChunked)
	{
		if(m_status == Downloading)
			m_status = Error;
	}
	else if(m_nContentSize > 0)
	{
		if(m_status == Downloading)
			m_status = Error;
	}
	else
	{
		if(m_status == Downloading)
			m_status = Done;
		string s = m_chunkBuf.str();
		m_chunkBuf.str("");
		m_chunkBuf.clear();
		delete(m_pData);
		m_pData = new unsigned char[s.length() + 1];
		memcpy(m_pData, s.c_str(), s.length());
		m_pData[s.length()] = '\0';
		m_nContentSize = s.length();
		m_bPastHeader = false;
	}
	m_aborted = true;
}

void GHttpClient::processChunkBody(const unsigned char* szData, int nSize)
{
	while(nSize > 0)
	{
		if(m_nContentSize == 0)
		{
			// Read the chunk size
			int n;
			for(n = 0; (szData[n] < '0' || szData[n] > 'f') && n < nSize; n++)
			{
			}
			int nHexStart = n;
			for( ; szData[n] >= '0' && szData[n] <= 'f' && n < nSize; n++)
			{
			}
			if(n >= nSize)
				break;

			// Convert it from hex to an integer
			int nPow = 1;
			int nDig;
			int i;
			for(i = n - 1; i >= nHexStart; i--)
			{
				if(szData[i] >= '0' && szData[i] <= '9')
					nDig = szData[i] - '0';
				else if(szData[i] >= 'a' && szData[i] <= 'f')
					nDig = szData[i] - 'a' + 10;
				else if(szData[i] >= 'A' && szData[i] <= 'F')
					nDig = szData[i] - 'A' + 10;
				else
				{
					nDig = 0;
					GAssert(false); // expected a hex digit
				}
				m_nContentSize += (nDig * nPow);
				nPow *= 16;
			}
			for( ; szData[n] != '\n' && n < nSize; n++)
			{
			}
			if(n < nSize && szData[n] == '\n')
				n++;
			szData += n;
			nSize -= n;
		}
		if(m_nContentSize == 0)
		{
			string s = m_chunkBuf.str();
			m_chunkBuf.str("");
			m_chunkBuf.clear();
			delete(m_pData);
			m_pData = new unsigned char[s.length() + 1];
			memcpy(m_pData, s.c_str(), s.length());
			m_pData[s.length()] = '\0';
			m_nContentSize = s.length();
			m_bChunked = false;
			m_bPastHeader = false;
			if(m_status == Downloading)
				m_status = Done;
			break;
		}
		else
		{
			int nChunkSize = std::min(m_nContentSize, nSize);
			m_chunkBuf.write((const char*)szData, nChunkSize);
			szData += nChunkSize;
			nSize -= nChunkSize;
			m_nContentSize -= nChunkSize;
		}
	}
}

// todo: this is a hack--fix it properly
void GHttpClient::gimmeWhatYouGot()
{
	if(m_bChunked)
	{
		string s = m_chunkBuf.str();
		m_chunkBuf.str("");
		m_chunkBuf.clear();
		if(s.length() > 64)
		{
			delete(m_pData);
			m_pData = new unsigned char[s.length() + 1];
			memcpy(m_pData, s.c_str(), s.length());
			m_pData[s.length()] = '\0';
			m_nContentSize = s.length();
			m_bChunked = false;
			m_bPastHeader = false;
			if(m_status == Downloading)
				m_status = Done;
		}
	}
	else if(m_nContentSize > 0)
	{
		if(m_nDataPos > 64)
		{
			if(m_status == Downloading)
				m_status = Done;
			m_pData[m_nDataPos] = '\0';
			m_bPastHeader = false;
			m_nContentSize = m_nDataPos;
		}
	}
	else
	{
		string s = m_chunkBuf.str();
		m_chunkBuf.str("");
		m_chunkBuf.clear();
		if(s.length() > 64)
		{
			delete(m_pData);
			m_pData = new unsigned char[s.length() + 1];
			memcpy(m_pData, s.c_str(), s.length());
			m_pData[s.length()] = '\0';
			m_nContentSize = s.length();
			if(m_status == Downloading)
				m_status = Done;
			m_bPastHeader = false;
		}
	}
}

unsigned char* GHttpClient::getData(int* pnSize)
{
	if(m_status != Done)
	{
		gimmeWhatYouGot();
		if(m_status == Done)
		{
			//GAssert(false); // todo: why are we giving it out early?
		}

	}

	if(m_status != Done)
	{
		// it's really not done, so don't return anything [rdp]
		*pnSize = 0;
		return NULL;
	}
	*pnSize = m_nContentSize;
	return m_pData;
}

unsigned char* GHttpClient::releaseData(int* pnSize)
{
	unsigned char* pData = getData(pnSize);
	if(!pData)
		return NULL;
	m_pData = NULL;
	return pData;
}


// -----------------------------------------------------------------------

namespace GClasses {
class GHttpServerBuffer
{
public:
	enum RequestType
	{
		None,
		Get,
		Head,
		Post,
	};

	int m_nPos;
	char m_szLine[MAX_SERVER_LINE_SIZE];
	char m_szUrl[MAX_SERVER_LINE_SIZE];
	char m_szParams[MAX_SERVER_LINE_SIZE];
	char m_szDate[MAX_SERVER_LINE_SIZE];
	char m_szCookie[MAX_COOKIE_SIZE];
	unsigned char* m_pPostBuffer;
	RequestType m_eRequestType;
	int m_nContentLength;

	GHttpServerBuffer()
	{
		m_eRequestType = None;
		m_nPos = 0;
		m_nContentLength = 0;
		m_pPostBuffer = NULL;
	}

	~GHttpServerBuffer()
	{
		delete[] m_pPostBuffer;
	}

	void reset()
	{
		m_szUrl[0] = '\0';
		m_szParams[0] = '\0';
		m_szDate[0] = '\0';
		m_szCookie[0] = '\0';
		m_nContentLength = 0;
		delete[] m_pPostBuffer;
		m_pPostBuffer = NULL;
	}
};
}

GHttpServer::GHttpServer(int nPort)
{
	m_pSocket = new GSocketServer(false, 0, nPort, 1000);
	if(!m_pSocket)
		throw("failed to open port");
	setContentType("text/html");
	setCookie("", false);
	m_modifiedTime = 0;
}

GHttpServer::~GHttpServer()
{
	for(vector<GHttpServerBuffer*>::iterator it = m_buffers.begin(); it != m_buffers.end(); it++)
		delete(*it);
	delete(m_pSocket);
}

bool GHttpServer::process()
{
	int nMessageSize;
	int nConnection;
	unsigned char* pMessage;
	unsigned char* pIn;
	char c;
	GHttpServerBuffer* pBuffer;
	bool bDidSomething = false;
	while(m_pSocket->GetMessageCount() > 0)
	{
		bDidSomething = true;
		pMessage = m_pSocket->GetNextMessage(&nMessageSize, &nConnection);
		pIn = pMessage;
		while((int)m_buffers.size() <= nConnection)
			m_buffers.push_back(new GHttpServerBuffer());
		pBuffer = m_buffers[nConnection];
		while(nMessageSize > 0)
		{
			if(pBuffer->m_pPostBuffer)
			{
				processPostData(nConnection, pBuffer, pIn, nMessageSize);
				break;
			}
			else
			{
				// Obtain a single header line
				while(nMessageSize > 0)
				{
					c = *pIn;
					pBuffer->m_szLine[pBuffer->m_nPos++] = c;
					pIn++;
					nMessageSize--;
					if(c == '\n' || pBuffer->m_nPos >= MAX_SERVER_LINE_SIZE - 1)
					{
						pBuffer->m_szLine[pBuffer->m_nPos] = '\0';
						processHeaderLine(nConnection, pBuffer, pBuffer->m_szLine);
						pBuffer->m_nPos = 0;
						break;
					}
				}
			}
		}
		delete[] pMessage;
	}
	return bDidSomething;
}

void GHttpServer::beginRequest(GHttpServerBuffer* pClient, int eType, const char* szIn)
{
	pClient->reset();
	pClient->m_eRequestType = (GHttpServerBuffer::RequestType)eType;
	char* szOut = pClient->m_szUrl;
	while(*szIn > ' ' && *szIn != '?')
	{
		*szOut = *szIn;
		szIn++;
		szOut++;
	}
	*szOut = '\0';
	if(eType == GHttpServerBuffer::Get && *szIn == '?')
	{
		szIn++;
		szOut = pClient->m_szParams;
		while(*szIn > ' ')
		{
			*szOut = *szIn;
			szIn++;
			szOut++;
		}
		*szOut = '\0';
	}
	else
		pClient->m_szParams[0] = '\0';
}

void GHttpServer::onReceiveFullPostRequest(GHttpServerBuffer* pClient, int nConnection)
{
	char* szCookie = NULL;
	if(pClient->m_szCookie[0] != '\0')
		szCookie = pClient->m_szCookie;
	m_modifiedTime = 0;
	doPost(pClient->m_szUrl, pClient->m_pPostBuffer, pClient->m_nContentLength, szCookie, m_stream);
	pClient->m_pPostBuffer = NULL;
	pClient->m_nPos = 0;
	sendResponse(pClient, nConnection);
}

void GHttpServer::processPostData(int nConnection, GHttpServerBuffer* pClient, const unsigned char* pData, int nDataSize)
{
	if(nDataSize > pClient->m_nContentLength - pClient->m_nPos)
		nDataSize = pClient->m_nContentLength - pClient->m_nPos;
	memcpy(pClient->m_pPostBuffer + pClient->m_nPos, pData, nDataSize);
	pClient->m_nPos += nDataSize;
	if(pClient->m_nPos >= pClient->m_nContentLength)
	{
		pClient->m_pPostBuffer[pClient->m_nContentLength] = '\0';
		onReceiveFullPostRequest(pClient, nConnection);
	}
}

void GHttpServer::processHeaderLine(int nConnection, GHttpServerBuffer* pClient, const char* szLine)
{
	onProcessLine(nConnection, szLine);

	// Skip whitespace
	while(*szLine > '\0' && *szLine <= ' ')
		szLine++;

	if(*szLine == '\0')
	{
		if(pClient->m_eRequestType == GHttpServerBuffer::Get)
		{
			bool bModified = true;
			if(pClient->m_szDate[0] != '\0')
				bModified = hasBeenModifiedSince(pClient->m_szUrl, pClient->m_szDate);
			if(bModified)
			{
				char* szCookie = NULL;
				if(pClient->m_szCookie[0] != '\0')
					szCookie = pClient->m_szCookie;
				m_modifiedTime = 0;
				doGet(pClient->m_szUrl, pClient->m_szParams, (int)strlen(pClient->m_szParams), szCookie, m_stream);
				sendResponse(pClient, nConnection);
			}
			else
				sendNotModifiedResponse(pClient, nConnection);
		}
		else if(pClient->m_eRequestType == GHttpServerBuffer::Head)
		{
			m_modifiedTime = 0;
			setHeaders(pClient->m_szUrl, pClient->m_szParams);
			sendResponse(pClient, nConnection);
		}
		else if(pClient->m_eRequestType == GHttpServerBuffer::Post)
		{
			if(pClient->m_nContentLength >= 0)
			{
				if(pClient->m_nContentLength > 0)
					pClient->m_pPostBuffer = new unsigned char[pClient->m_nContentLength + 1];
				else
				{
					pClient->m_pPostBuffer = NULL;
					onReceiveFullPostRequest(pClient, nConnection);
				}
				pClient->m_nPos = 0;
			}
			else
			{
				GAssert(false); // bad post data size
				m_pSocket->Disconnect(nConnection);
			}
		}
	}
	else if(_strnicmp(szLine, "GET ", 4) == 0)
		beginRequest(pClient, GHttpServerBuffer::Get, szLine + 4);
	else if(_strnicmp(szLine, "HEAD ", 5) == 0)
		beginRequest(pClient, GHttpServerBuffer::Head, szLine + 5);
	else if(_strnicmp(szLine, "POST ", 5) == 0)
		beginRequest(pClient, GHttpServerBuffer::Post, szLine + 5);
	else if(_strnicmp(szLine, "Content-Length: ", 16) == 0)
		pClient->m_nContentLength = atoi(szLine + 16);
	else if(_strnicmp(szLine, "Cookie: ", 8) == 0)
	{
		int i = 17; // strlen("Cookie: attribute")
		while(szLine[i] != '=' && szLine[i] != '\0')
			i++;
		if(szLine[i] == '=')
			i++;
		strcpy(pClient->m_szCookie, szLine + i);
	}
	else if(_strnicmp(szLine, "If-Modified-Since: ", 19) == 0)
	{
		const char* szIn = szLine + 19;
		char* szOut = pClient->m_szDate;
		while(*szIn >= ' ')
		{
			*szOut = *szIn;
			szIn++;
			szOut++;
		}
		*szOut = '\0';
	}
}

void GHttpServer::setContentType(const char* szContentType)
{
	safe_strcpy(m_szContentType, szContentType, 64);
}

void GHttpServer::setCookie(const char* szPayload, bool bPersist)
{
	m_bPersistCookie = bPersist;
	safe_strcpy(m_szCookie, szPayload, MAX_COOKIE_SIZE);
}

void AscTimeToGMT(const char* szIn, char* szOut)
{
	// Copy the day
	while(*szIn > ' ')
		*(szOut++) = *(szIn++);
	*(szOut++) = ',';
	*(szOut++) = ' ';
	while(*szIn == ' ')
		szIn++;

	// Copy the day of the month
	const char* pMonth = szIn;
	while(*szIn > ' ')
		szIn++;
	while(*szIn == ' ')
		szIn++;
	while(*szIn > ' ')
		*(szOut++) = *(szIn++);
	*(szOut++) = ' ';
	while(*szIn == ' ')
		szIn++;
	const char* pTime = szIn;

	// Copy the month
	while(*pMonth > ' ')
		*(szOut++) = *(pMonth++);
	*(szOut++) = ' ';

	// Copy the year
	while(*szIn > ' ')
		szIn++;
	while(*szIn == ' ')
		szIn++;
	while(*szIn > ' ')
		*(szOut++) = *(szIn++);
	*(szOut++) = ' ';

	// Copy the time
	while(*pTime > ' ')
		*(szOut++) = *(pTime++);
	*(szOut++) = ' ';

	// Add "GMT"
	strcpy(szOut, "GMT");
}

void GHttpServer::sendResponse(GHttpServerBuffer* pClient, int nConnection)
{
	// Convert the payload to a string
	string sPayload = m_stream.str();
	m_stream.str("");
	m_stream.clear();

	// Make the header
	const char pTmp1[] = "HTTP/1.1 200 OK\r\nContent-Type: ";
	m_pSocket->Send(pTmp1, sizeof(pTmp1) - 1, nConnection);
	m_pSocket->Send(m_szContentType, strlen(m_szContentType), nConnection);

	if(pClient->m_eRequestType != GHttpServerBuffer::Head)
	{
		std::ostringstream os;
		os << "\r\nContent-Length: ";
		os << sPayload.length();
		os << "\r\n";
		string s = os.str();
		m_pSocket->Send(s.c_str(), s.length(), nConnection);
	}

	// Set the date header
	{
		std::ostringstream os;
		os << "Date: ";
		time_t t = time((time_t*)0);
#ifdef WINDOWS
		struct tm thetime;
		gmtime_s(&thetime, &t);
		struct tm* pTime = &thetime;
#else
		struct tm* pTime = gmtime(&t);
#endif
		const char* szAscTime = asctime(pTime);
		char szGMT[40];
		AscTimeToGMT(szAscTime, szGMT);
		os << szGMT;
		os << "\r\n";
		string s = os.str();
		m_pSocket->Send(s.c_str(), s.length(), nConnection);
	}

	// Set the last-modified header
	if(m_modifiedTime != 0)
	{
		std::ostringstream os;
		os << "Last-Modified: ";
		struct tm* pTime = gmtime(&m_modifiedTime);
		const char* szAscTime = asctime(pTime);
		char szGMT[40];
		AscTimeToGMT(szAscTime, szGMT);
		os << szGMT;
		os << "\r\n";
		string s = os.str();
		m_pSocket->Send(s.c_str(), s.length(), nConnection);
	}

	// Set cookie
	if(m_szCookie[0] != '\0')
	{
		std::ostringstream os;
		os << "Set-Cookie: attribute=";
		os << m_szCookie;
		os << "; path=/";
		if(m_bPersistCookie)
			os << "; expires=Sat, 01-Jan-2060 00:00:00 GMT";
		os << "\r\n";
		string s = os.str();
		m_pSocket->Send(s.c_str(), s.length(), nConnection);
	}

	// End of header
	m_pSocket->Send("\r\n", 2, nConnection);

	// Send the payload
	if(pClient->m_eRequestType != GHttpServerBuffer::Head)
		m_pSocket->Send(sPayload.c_str(), sPayload.length(), nConnection);
}

void GHttpServer::sendNotModifiedResponse(GHttpServerBuffer* pClient, int nConnection)
{
	ostringstream os;
	os << "HTTP/1.1 304 Not Modified\r\nDate: ";
	os << pClient->m_szDate;
	os << "\r\n\r\n";
	string s = os.str();
	m_pSocket->Send(s.c_str(), s.length(), nConnection);
}

/*static*/ void GHttpServer::unescapeUrl(char* szOut, const char* szIn, int nInLen)
{
	int c1, c2, n1, n2;
	while(nInLen > 0 && *szIn != '\0')
	{
		if(*szIn == '%')
		{
			if(--nInLen == 0)
				break;
			szIn++;
			n1 = *szIn;
			if(--nInLen == 0)
				break;
			szIn++;
			n2 = *szIn;
			if(n1 >= '0' && n1 <= '9')
				c1 = n1 - '0';
			else if(n1 >= 'a' && n1 <= 'z')
				c1 = n1 - 'a' + 10;
			else if(n1 >= 'A' && n1 <= 'Z')
				c1 = n1 - 'A' + 10;
			else
				c1 = 2;
			if(n2 >= '0' && n2 <= '9')
				c2 = n2 - '0';
			else if(n2 >= 'a' && n2 <= 'z')
				c2 = n2 - 'a' + 10;
			else if(n2 >= 'A' && n2 <= 'Z')
				c2 = n2 - 'A' + 10;
			else
				c2 = 0;
			*szOut = 16 * c1 + c2;
		}
		else if(*szIn == '+')
			*szOut = ' ';
		else
			*szOut = *szIn;
		nInLen--;
		szIn++;
		szOut++;
	}
	*szOut = '\0';
}

// static
bool GHttpServer::parseFileParam(const char* pParams, int nParamsLen, const char** ppFilename, int* pFilenameLen, const unsigned char** ppFile, int* pFileLen)
{
	// Measure the length of the unique divider string
	int dividerLen;
	for(dividerLen = 0; dividerLen < nParamsLen && pParams[dividerLen] != '\r'; dividerLen++)
	{
	}
	if(dividerLen >= nParamsLen)
		return false;
	GAssert(dividerLen > 5 && dividerLen < 150); // divider length doesn't seem right

	// Move to the next line of data
	int pos = dividerLen + 1;
	while(pos < nParamsLen && pParams[pos] <= ' ')
		pos++;
	if(pos >= nParamsLen)
		return false;
	GAssert(strncmp(&pParams[pos], "Content-Disposition: form-data", 30) == 0); // unexpected line

	// Find the filename
	while(pos < nParamsLen && strncmp(&pParams[pos], "filename=\"", 10) != 0)
		pos++;
	if(pos >= nParamsLen)
		return false;
	pos += 10;
	*ppFilename = &pParams[pos];
	int nFilenameLen = 0;
	while(pos < nParamsLen && pParams[pos] != '"')
	{
		pos++;
		nFilenameLen++;
	}
	if(pos >= nParamsLen)
		return false;
	*pFilenameLen = nFilenameLen;

	// Find the start of the file
	while(pos < nParamsLen && (
			pParams[pos] != '\r' ||
			pParams[pos + 1] != '\n' ||
			pParams[pos + 2] != '\r' ||
			pParams[pos + 3] != '\n'))
		pos++;
	if(pos >= nParamsLen)
		return false;
	pos += 4;
	*ppFile = (const unsigned char*)&pParams[pos];
	int nDataLen = 0;
	while(pos < nParamsLen && strncmp(&pParams[pos], pParams, dividerLen) != 0)
	{
		pos++;
		nDataLen++;
	}
	if(pos >= nParamsLen)
		return false;
	*pFileLen = nDataLen - 2;
	return true;
}








GHttpParamParser::GHttpParamParser(const char* szParams, bool scrub)
{
	m_pHeap = NULL;
	if(szParams)
	{
		m_pHeap = new GHeap(1024);
		int nNameStart = 0;
		int nNameLen, nValueStart, nValueLen;
		while(true)
		{
			for(nNameLen = 0; szParams[nNameStart + nNameLen] != '=' && szParams[nNameStart + nNameLen] != '\0'; nNameLen++)
			{
			}
			if(szParams[nNameStart + nNameLen] == '\0')
				break;
			nValueStart = nNameStart + nNameLen + 1;
			for(nValueLen = 0; szParams[nValueStart + nValueLen] != '&' && szParams[nValueStart + nValueLen] != '\0'; nValueLen++)
			{
			}
			char* szName = m_pHeap->allocate(nNameLen + 1);
			GHttpServer::unescapeUrl(szName, &szParams[nNameStart], nNameLen);
			char* szValue = m_pHeap->allocate(nValueLen + 1);
			GHttpServer::unescapeUrl(szValue, &szParams[nValueStart], nValueLen);
			if(scrub)
				scrubValue(szValue);
			m_map.insert(std::make_pair(szName, szValue));
			if(szParams[nValueStart + nValueLen] == '\0')
				break;
			nNameStart = nValueStart + nValueLen + 1;
		}
	}
}

GHttpParamParser::~GHttpParamParser()
{
	delete(m_pHeap);
}

// static
void GHttpParamParser::scrubValue(char* value)
{
	while(*value != '\0')
	{
		if(*value >= 'a' && *value <= 'z') {}
		else if(*value >= '(' && *value <= ':') {}
		else if(*value >= ' ' && *value <= '!') {}
		else if(*value >= '<' && *value <= 'Z') {}
		else if(*value >= '#' && *value <= '%') {}
		else
			*value = '_';
		value++;
	}
}

const char* GHttpParamParser::find(const char* szName)
{
	std::map<const char*, const char*, strComp>::iterator it = m_map.find(szName);
	if(it == m_map.end())
		return NULL;
	return it->second;
}










GHttpMultipartParser::GHttpMultipartParser(const char* pRawData, int len)
{
	m_pRawData = pRawData;
	m_len = len;
	for(m_sentinelLen = 0; m_sentinelLen < len && pRawData[m_sentinelLen] != '\r'; m_sentinelLen++)
	{
	}
	m_pos = m_sentinelLen;
	for(m_repeatLen = 1; pRawData[m_repeatLen] == pRawData[0]; m_repeatLen++)
	{
	}
}

GHttpMultipartParser::~GHttpMultipartParser()
{
}

bool GHttpMultipartParser::next(int* pNameStart, int* pNameLen, int* pValueStart, int* pValueLen, int* pFilenameStart, int* pFilenameLen)
{
	// Find the terminating sentinel
	int start = m_pos;
	for( ; m_pos < m_len && strncmp(m_pRawData, m_pRawData + m_pos, m_sentinelLen) != 0; m_pos++)
	{
		if(m_pRawData[m_pos + m_repeatLen] != m_pRawData[0])
			m_pos += m_repeatLen;
	}
	if(m_pos >= m_len)
		return false;
	int valueEnd = m_pos;
	if(valueEnd > start && m_pRawData[valueEnd - 1] == '\n')
		valueEnd--;
	if(valueEnd > start && m_pRawData[valueEnd - 1] == '\r')
		valueEnd--;
	m_pos += m_sentinelLen;

	// Find the start of all the parts
	int nameStart = -1;
	int nameEnd = -1;
	int filenameStart = -1;
	int filenameEnd = -1;
	int valueStart;
	for(valueStart = start; valueStart < m_len && strncmp(m_pRawData + valueStart, "\r\n\r\n", 4) != 0; valueStart++)
	{
		if(strncmp(m_pRawData + valueStart, "name=\"", 6) == 0)
			nameStart = valueStart + 6;
		if(strncmp(m_pRawData + valueStart, "filename=\"", 10) == 0)
			filenameStart = valueStart + 10;
	}
	if(valueStart >= m_len)
		return false;
	valueStart += 4;

	// Find ends
	if(nameStart >= 0)
	{
		for(nameEnd = nameStart; nameEnd < valueStart && m_pRawData[nameEnd] != '"'; nameEnd++)
		{
		}
	}
	if(filenameStart >= 0)
	{
		for(filenameEnd = filenameStart; filenameEnd < valueStart && m_pRawData[filenameEnd] != '"'; filenameEnd++)
		{
		}
	}

	// Return results
	*pNameStart = nameStart;
	*pNameLen = nameEnd - nameStart;
	*pValueStart = valueStart;
	*pValueLen = valueEnd - valueStart;
	*pFilenameStart = filenameStart;
	*pFilenameLen = filenameEnd - filenameStart;
	return true;
}
