/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GSocket.h"
#include <time.h>
#include "../GClasses/GSpinLock.h"
#include "../GClasses/GHolders.h"
#include "../GClasses/GError.h"
#include "../GClasses/GThread.h"
#include "../GClasses/GString.h"
#include "../GClasses/GBits.h"
#include <wchar.h>
#include <sstream>
#include <iostream>
#ifdef WINDOWS
#	include "GWindows.h"
#	include <Ws2tcpip.h>
#else
#	include <unistd.h>
#	include <time.h>
#	include <string.h>
#	include <sys/socket.h>
#	include <arpa/inet.h>
#	include <netdb.h>
#	include <stdlib.h>
#	include <sys/ioctl.h>
#	include <errno.h>
#	define SOCKET_ERROR -1
#endif

namespace GClasses {

using std::cerr;
using std::vector;
using std::string;

const char* gsocket_GetLastError()
{
	const char* szMsg = NULL;
#ifdef WINDOWS
	int n = WSAGetLastError();
	switch(n)
	{
		case WSAECONNRESET: 		szMsg = "An incoming connection was indicated, but was subsequently terminated by the remote peer prior to accepting the call."; break;
		case WSAEFAULT: 			szMsg = "The addrlen parameter is too small or addr is not a valid part of the user address space."; break;
		case WSAEINTR: 				szMsg = "A blocking Windows Sockets 1.1 call was canceled through WSACancelBlockingCall."; break;
		case WSAEINVAL: 			szMsg = "The listen function was not invoked prior to accept."; break;
		case WSAEINPROGRESS: 		szMsg = "A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function."; break;
		case WSAEMFILE: 			szMsg = "The queue is nonempty upon entry to accept and there are no descriptors available."; break;
		case WSAENETDOWN: 			szMsg = "The network subsystem has failed."; break;
		case WSAENOBUFS: 			szMsg = "No buffer space is available."; break;
		case WSAENOTSOCK: 			szMsg = "The descriptor is not a socket."; break;
		case WSAEOPNOTSUPP: 		szMsg = "The referenced socket is not a type that supports connection-oriented service."; break;
		case WSAEWOULDBLOCK: 		szMsg = "The socket is marked as nonblocking and no connections are present to be accepted."; break;
		case WSANOTINITIALISED:		szMsg = "A successful WSAStartup must occur before using this function.";   break;
		case WSAEALREADY:			szMsg = "A nonblocking connect call is in progress on the specified socket.";   break;
		case WSAEADDRNOTAVAIL:		szMsg = "The remote address is not a valid address (such as ADDR_ANY).";   break;
		case WSAEAFNOSUPPORT:		szMsg = "Addresses in the specified family cannot be used with this socket.";   break;
		case WSAECONNREFUSED:		szMsg = "The attempt to connect was forcefully rejected.";   break;
		case WSAEISCONN:			szMsg = "The socket is already connected (connection-oriented sockets only).";   break;
		case WSAENETUNREACH:		szMsg = "The network cannot be reached from this host at this time.";   break;
		case WSAETIMEDOUT:			szMsg = "Attempt to connect timed out without establishing a connection.";   break;
		case WSASYSNOTREADY:		szMsg = "network subsystem not ready for communication.";   break;
		case WSAVERNOTSUPPORTED:	szMsg = "The version of Windows Sockets support requested is not provided by this implementation.";   break;
		case WSAEPROCLIM:			szMsg = "Limit on the number of tasks supported has been reached.";   break;
		case WSAEHOSTUNREACH:		szMsg = "Host unreacheable"; break;
		case WSAENOTCONN:			szMsg = "Not Connected"; break;
		case WSAECONNABORTED:		szMsg = "Connection Aborted"; break;
		case 0x2740:				szMsg = "Port already in use"; break;
		case WSAHOST_NOT_FOUND: 	szMsg = "Authoritative answer host not found."; break;
		case WSATRY_AGAIN: 			szMsg = "Nonauthoritative host not found, or server failure."; break;
		case WSANO_RECOVERY: 		szMsg = "A nonrecoverable error occurred."; break;
		case WSANO_DATA: 			szMsg = "Valid name, no data record of requested type."; break;
		default:					szMsg = "An unrecognized socket error occurred"; break;
	}
#else
	switch(errno)
	{
		case EBADF:			szMsg = "not a valid socket descriptor."; break;
		case EINVAL:		szMsg = "The socket is already bound to an address or addrlen is wrong."; break;
		case EACCES:		szMsg = "Access permission is denied."; break;
		case ENOTSOCK:		szMsg = "Argument is a descriptor for a file, not a socket."; break;
		case EROFS:			szMsg = "The  socket inode would reside on a read-only file system."; break;
		case EFAULT:		szMsg = "the addr parameter points outside the user's accessible address space."; break;
		case ENAMETOOLONG:	szMsg = "A component of a pathname exceeded {NAME_MAX} characters, or an entire path name exceeded {PATH_MAX} characters."; break;
		case ENOENT:		szMsg = "The file or named socket does not exist."; break;
		case ENOMEM:		szMsg = "Insufficient kernel memory was available."; break;
		case ENOTDIR:		szMsg = "A component of the path prefix is not a directory."; break;
		case ELOOP:			szMsg = "Too many symbolic links were encountered in resolving my_addr."; break;
		case EOPNOTSUPP:	szMsg = "The referenced socket is not of type SOCK_STREAM."; break;
		case EWOULDBLOCK:	szMsg = "The socket is marked non-blocking and no connections are present to be accepted."; break;
		case EMFILE:		szMsg = "The per-process descriptor table is full."; break;
		case ENFILE:		szMsg = "The system file table is full."; break;
		case EADDRNOTAVAIL:	szMsg = "The specified address is not available on this machine."; break;
		case EAFNOSUPPORT:	szMsg = "Addresses in the specified address family cannot be used with this socket."; break;
		case EISCONN:		szMsg = "The socket is already connected."; break;
		case ETIMEDOUT:		szMsg = "Attempt to connect timed out without establishing a connection."; break;
		case ECONNREFUSED:	szMsg = "The attempt to connect was forcefully rejected."; break;
		case ENETUNREACH:	szMsg = "The network isn't reachable from this host."; break;
		case EADDRINUSE:	szMsg = "The address is already in use."; break;
		case EINPROGRESS:	szMsg = "The socket is non-blocking and the connection cannot be completed immediately.  It is possible to select(2) for completion by selecting the socket for writing."; break;
		case EALREADY:		szMsg = "The socket is non-blocking and a previous connection attempt has not yet been completed."; break;
		case HOST_NOT_FOUND:	szMsg = "The specified host is unknown."; break;
		case NO_ADDRESS:	szMsg = "The requested name is valid but does not have an IP address."; break;
		case NO_RECOVERY:	szMsg = "A non-recoverable name server error occurred."; break;
		//case TRY_AGAIN:		szMsg = "A temporary error occurred on an authoritative name server.  Try again later."; break;
		default:		szMsg = "An unrecognized socket error occurred"; break;
	}
#endif
	return szMsg;
}

void gsocket_LogError()
{
	const char* szErrorMessage = gsocket_GetLastError();
	ThrowError(szErrorMessage);
}

inline void SetSocketToBlockingMode(SOCKET s)
{
	unsigned long ulMode = 0;
#ifdef WINDOWS
	if(ioctlsocket(s, FIONBIO, &ulMode) != 0)
#else
	if(ioctl(s, FIONBIO, &ulMode) != 0)
#endif
	{
		gsocket_LogError();
	}
}

inline void SetSocketToNonBlockingMode(SOCKET s)
{
	unsigned long ulMode = 1;
#ifdef WINDOWS
	if(ioctlsocket(s, FIONBIO, &ulMode) != 0)
#else
	if(ioctl(s, FIONBIO, &ulMode) != 0)
#endif
		gsocket_LogError();
}

inline void CloseSocket(SOCKET s)
{
#ifdef WINDOWS
	closesocket(s);
#else
	close(s);
#endif
}


#ifdef WINDOWS
bool gsocket_InitWinSock()
{
	// Initializing Winsock
	WORD wVersionRequested;
	WSADATA wsaData;
	int err;
	wVersionRequested = MAKEWORD(1, 1);
	err = WSAStartup( wVersionRequested, &wsaData );
	if ( err != 0 )
		ThrowError("Failed to find a usable WinSock DLL");

	// Confirm that the WinSock DLL supports 2.2.
	// Note that if the DLL supports versions greater
	// than 2.2 in addition to 2.2, it will still return
	// 2.2 in wVersion since that is the version we
	// requested.
	if ( LOBYTE( wsaData.wVersion ) != 1 ||
			HIBYTE( wsaData.wVersion ) != 1 )
	{
		int n1 = LOBYTE( wsaData.wVersion );
		int n2 = HIBYTE( wsaData.wVersion );
		WSACleanup();
		ThrowError("Found a Winsock DLL, but it only supports an older version.  It needs to support version 2.2");
	}
	return true;
}
#endif // WINDOWS

// ------------------------------------------------------------------------------

GSocketClientBase::GSocketClientBase(bool bUDP)
{
	m_hListenThread = BAD_HANDLE;
	m_s = INVALID_SOCKET;
	m_bKeepListening = true;
	m_bUDP = bUDP;
#ifdef WINDOWS
	if(!gsocket_InitWinSock())
		throw "Error initializing WinSock";
#endif
}

GSocketClientBase::~GSocketClientBase()
{
	Disconnect();
}

void GSocketClientBase::joinListenThread()
{
	m_bKeepListening = false;
	time_t tStart;
	time_t tNow;
	time(&tStart);
	while(m_hListenThread != BAD_HANDLE)
	{
		GThread::sleep(0);
		time(&tNow);
		if(tNow - tStart > 4)
		{
			GAssert(false); // Error, took too long for the listen thread to exit
			break;
		}
	}
}

void GSocketClientBase::joinListenThread(int nConnectionNumber)
{
	ThrowError("not implemented yet");
}

void GSocketClientBase::listen()
{
	char szReceiveBuff[520];
	SOCKET s = m_s;

	// Mark the socket as blocking so we can call "recv" which is a blocking operation
	SetSocketToBlockingMode(s);

	// Start receiving messages
	unsigned long dwBytesReadyToRead = 0;
	int nBytesRead = 0;
	while(m_bKeepListening)
	{
		// See how much data is ready to be received
#ifdef WINDOWS
		GWindows::yieldToWindows(); // This is necessary because incoming packets go through the Windows message pump
		if(ioctlsocket(s, FIONREAD, &dwBytesReadyToRead) != 0)
			gsocket_LogError();
#else
		if(ioctl(s, FIONREAD, &dwBytesReadyToRead) != 0)
			gsocket_LogError();
#endif
		if(dwBytesReadyToRead > 0)
		{
			nBytesRead = recv(s, szReceiveBuff, 512, 0); // read from the queue (This blocks until there is some to read or connection is closed, but we already know there is something to read)
			if(nBytesRead > 0) // recv reads in as much data as is currently available up to the size of the buffer
				Receive((unsigned char*)szReceiveBuff, nBytesRead);
			else
				break; // The socket has been closed
		}
		else
		{
			// There's nothing to receive, so let's sleep for a while
			GThread::sleep(50);
		}
	}

#ifdef WINDOWS
	if(nBytesRead == SOCKET_ERROR)
	{
		int n = WSAGetLastError();
		switch(n)
		{
			case WSAECONNABORTED:	break;
			case WSAECONNRESET:		break;
			default:				gsocket_LogError();		break;
		}
	}
#endif

	onCloseConnection();
	shutdown(m_s, 2);
	CloseSocket(m_s);
	m_s = INVALID_SOCKET;
	m_hListenThread = BAD_HANDLE;
}

unsigned int ListenThread(void* pData)
{
	((GSocketClientBase*)pData)->listen();
	return 0;
}

/*virtual*/ void GSocketClientBase::onCloseConnection()
{
}

bool GSocketClientBase::isIPAddress(const char* szHost)
{
	int n;
	for(n = 0; szHost[n] != '.' && szHost[n] != '\0'; n++)
	{
		if(szHost[n] < '0' || szHost[n] > '9')
			return false;
	}
	if(szHost[n] == '.')
	{
		for(n++; szHost[n] != '.' && szHost[n] != '\0'; n++)
		{
			if(szHost[n] < '0' || szHost[n] > '9')
				return false;
		}
	}
	return true;
}


// This is for parsing a URL
//static
void GSocketClientBase::parseURL(const char* szUrl, int* pnHostIndex, int* pnPortIndex, int* pnPathIndex, int* pnParamsIndex)
{
	// Find the host
	int nHost = 0;
	int i;
	for(i = 0; szUrl[i] != ':' && szUrl[i] != '?' && szUrl[i] != '\0'; i++)
	{
	}
	if(strncmp(&szUrl[i], "://", 3) == 0)
		nHost = i + 3;

	// Find the port
	int nPort = -1;
	for(i = nHost; szUrl[i] != ':' && szUrl[i] != '?' && szUrl[i] != '\0'; i++)
	{
	}
	if(szUrl[i] == ':')
		nPort = i;

	// Find the path
	int nPath;
	for(nPath = std::max(nHost, nPort); szUrl[nPath] != '/' && szUrl[nPath] != '?' && szUrl[nPath] != '\0'; nPath++)
	{
	}
	if(nPort < 0)
		nPort = nPath;

	// Find the params
	if(pnParamsIndex)
	{
		int nParams;
		for(nParams = nPath; szUrl[nParams] != '?' && szUrl[nParams] != '\0'; nParams++)
		{
		}
		*pnParamsIndex = nParams;
	}

	// Set the return values
	if(pnHostIndex)
		*pnHostIndex = nHost;
	if(pnPortIndex)
		*pnPortIndex = nPort;
	if(pnPathIndex)
		*pnPathIndex = nPath;
}

//static
int GSocketClientBase::parseUrlParams(const char* szParams, int nMaxParams, char** pNames, int* pNameLengths, char** pValues, int* pValueLengths)
{
	if(*szParams == '?')
		szParams++;
	int nParams = 0;
	while(true)
	{
		if(*szParams == '\0' || nParams >= nMaxParams)
			return nParams;
		pNames[nParams] = (char*)szParams;
		pNameLengths[nParams] = 0;
		while(*szParams != '\0' && *szParams != '=' && *szParams != '&')
		{
			szParams++;
			pNameLengths[nParams]++;
		}
		if(*szParams == '=')
			szParams++;
		pValues[nParams] = (char*)szParams;
		pValueLengths[nParams] = 0;
		while(*szParams != '\0' && *szParams != '&')
		{
			szParams++;
			pValueLengths[nParams]++;
		}
		if(*szParams == '&')
			szParams++;
		nParams++;
	}
}
/*
in_addr GSocketClientBase::StringToAddr(const char* szURL)
{
	// Extract the host and port from the URL
	GTEMPBUF(szHost, strlen(szURL));
	ParseURL(szURL, NULL, szHost, NULL, NULL, NULL);

	// Determine if it is a friendly-URL or an IP address
	if(IsThisAnIPAddress(szHost))
	{
		in_addr iaTmp;
#ifdef WINDOWS
		iaTmp.S_un.S_addr = inet_addr(szHost);
#else
		iaTmp.s_addr = inet_addr(szHost);
#endif
		return iaTmp;
	}
	else
	{
		struct hostent* psh = gethostbyname(szHost);
		if(!psh)
		{
			gsocket_LogError();
			in_addr iaTmp;
#ifdef WINDOWS
			iaTmp.S_un.S_addr = NULL;
#else
            iaTmp.s_addr = 0;
#endif
			return iaTmp;
		}
		return *(in_addr*)psh->h_addr_list[0];
	}
}
*/

bool GSocketClientBase::Connect(const char* szHost, unsigned short nPort, int nTimeout)
{
	if(isConnected())
		Disconnect();
	struct addrinfo hints, *res, *res0;
	int error;
	res0 = NULL;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	std::ostringstream os;
	os << nPort;
	string tmp = os.str();
	error = getaddrinfo(szHost, tmp.c_str(), &hints, &res0);
	if(error)
		ThrowError(gai_strerror(error));
	m_s = INVALID_SOCKET;
	for(res = res0; res; res = res->ai_next)
	{
//cout << "Attempting to connect to " << ((unsigned char*)&res->ai_addr->sa_data)[2] << "." << ((unsigned char*)&res->ai_addr->sa_data)[3] << "." << ((unsigned char*)&res->ai_addr->sa_data)[4] << "." << ((unsigned char*)&res->ai_addr->sa_data)[5] << " on port " << nPort;

		m_s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
		if(m_s < 0)
			continue;


/*
		if(connect(m_s, res->ai_addr, (int)res->ai_addrlen) < 0)
		{
			CloseSocket(m_s);
			m_s = INVALID_SOCKET;
			continue;
		}
		break;  // we got a connection
*/



		struct timeval timeout;
		fd_set socketSet;
		SetSocketToNonBlockingMode(m_s);

		// Trying to connect with timeout
		if(connect(m_s, res->ai_addr, (int)res->ai_addrlen) < 0)
		{
#ifdef WINDOWS

			int n = WSAGetLastError();
			if(n == WSAEWOULDBLOCK || n == WSAEINPROGRESS)
#else
			if(errno == EINPROGRESS)
#endif
			{
				timeout.tv_sec = nTimeout;
				timeout.tv_usec = 0;
				FD_ZERO(&socketSet);
				FD_SET(m_s, &socketSet);
#ifdef WINDOWS
				int res = select((int)m_s + 1, NULL, &socketSet, NULL, &timeout);
				if(res < 0 && WSAGetLastError() != WSAEINTR)
#else
				int res = select(m_s + 1, NULL, &socketSet, NULL, &timeout);
				if(res < 0 && errno != EINTR)
#endif
				{
					// Failed to connect
					gsocket_LogError();
				}
				else if(res > 0)
				{
					// Socket selected for write
					socklen_t lon = sizeof(int);
					int valopt;
					if(getsockopt(m_s, SOL_SOCKET, SO_ERROR, (char*)(&valopt), &lon) < 0)
					{
						// error calling getsockopt
						gsocket_LogError();
					}
					if(valopt)
					{
						//gsocket_LogError();
						CloseSocket(m_s);
						m_s = INVALID_SOCKET;
						continue;
					}

					// Got a connection!
					SetSocketToBlockingMode(m_s);
					break;
				}
				else
				{
					// Timeout exceeded
					CloseSocket(m_s);
					m_s = INVALID_SOCKET;
					continue;
				}
			}
			else
			{
				// Failed to connect to this address
				CloseSocket(m_s);
				m_s = INVALID_SOCKET;
				continue;
			}
		}











	}
	freeaddrinfo(res0);
	if(m_s == INVALID_SOCKET)
	{
		// todo: handle the error
		return false;
	}

	// Spawn the listener thread
	m_hListenThread = GThread::spawnThread(ListenThread, this);
	if(m_hListenThread == BAD_HANDLE)
		ThrowError("Failed to spawn listening thread\n");
	GThread::sleep(0);

	return true;
}

void GSocketClientBase::Disconnect()
{
	joinListenThread();

	// Disconnect the connection
	if(m_s != INVALID_SOCKET)
	{
		shutdown(m_s, 2);
		CloseSocket(m_s);
		m_s = INVALID_SOCKET;
	}
}

bool GSocketClientBase::Send(const unsigned char *pBuff, size_t len)
{
	int nBytesSent;
	do
	{
		nBytesSent = send(m_s, (const char*)pBuff, (int)len, 0);
		if(nBytesSent < 0)
		{
#ifdef WINDOWS
			int n = WSAGetLastError();
			switch(n)
			{
				case WSAECONNABORTED:	break;
				case WSAECONNRESET:	break;
				default:	gsocket_LogError();		break;
			}
#else // WINDOWS
			if(errno == EAGAIN)
			{
				GThread::sleep(10); // todo: we should call "select" to determine when the socket is ready again
				continue;
			}
#endif // !WINDOWS
			return false;
		}
		else
		{
			len -= nBytesSent;
			pBuff += nBytesSent;
		}
	} while(len > 0);
	return true;
}

bool GSocketClientBase::isConnected()
{
	if(m_hListenThread == BAD_HANDLE)
		return false;
	else
		return true;
}

SOCKET GSocketClientBase::socketHandle()
{
	return m_s;
}

in_addr GSocketClientBase::myIPAddr()
{
	struct sockaddr sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(getsockname(m_s, &sAddr, &l))
	{
		gsocket_LogError();
	}
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	return pInfo->sin_addr;
}

char* GSocketClientBase::myIPAddr(char* szBuff, size_t nBuffSize)
{
	safe_strcpy(szBuff, inet_ntoa(myIPAddr()), nBuffSize);
	return szBuff;
}

u_short GSocketClientBase::myPort()
{
	SOCKADDR sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(getsockname(m_s, &sAddr, &l))
	{
		gsocket_LogError();
	}
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	return htons(pInfo->sin_port);
}

char* GSocketClientBase::myName(char* szBuff, size_t nBuffSize)
{
	SOCKADDR sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(getsockname(m_s, &sAddr, &l))
	{
		gsocket_LogError();
	}
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	HOSTENT* namestruct = gethostbyaddr((const char*)&pInfo->sin_addr, 4, pInfo->sin_family);
	if(!namestruct)
	{
		ThrowError("Error calling gethostbyaddr");
	}
	safe_strcpy(szBuff, namestruct->h_name, nBuffSize);
	return(szBuff);
}

in_addr GSocketClientBase::otherIPAddr()
{
	struct sockaddr sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(getpeername(m_s, &sAddr, &l))
		gsocket_LogError();
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	return pInfo->sin_addr;
}

char* GSocketClientBase::otherIPAddr(char* szBuff, size_t nBuffSize)
{
	safe_strcpy(szBuff, inet_ntoa(otherIPAddr()), nBuffSize);
	return szBuff;
}

u_short GSocketClientBase::otherPort()
{
	SOCKADDR sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(getpeername(m_s, &sAddr, &l))
		gsocket_LogError();
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	return htons(pInfo->sin_port);
}

char* GSocketClientBase::otherName(char* szBuff, size_t nBuffSize)
{
	SOCKADDR sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(getpeername(m_s, &sAddr, &l))
		gsocket_LogError();
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	HOSTENT* namestruct = gethostbyaddr((const char*)&pInfo->sin_addr, 4, pInfo->sin_family);
	if(!namestruct)
	{
		ThrowError("Error calling gethostbyaddr");
	}
	safe_strcpy(szBuff, namestruct->h_name, nBuffSize);
	return(szBuff);
}

// ------------------------------------------------------------------------------

GSocketServerBase::GSocketServerBase(bool bUDP, int nPort, int nMaxConnections)
{
	m_hWorkerThread = BAD_HANDLE;
	m_socketConnectionListener = INVALID_SOCKET;
	m_bKeepWorking = true;
	m_bUDP = false;
	m_szReceiveBuffer = new char[2048];
	m_pConnectionsLock = new GSpinLock();
	Init(bUDP, nPort, nMaxConnections);
}

GSocketServerBase::~GSocketServerBase()
{
	DontAcceptAnyMoreConnections();
	JoinWorkerThread();
	{
		GSpinLockHolder hLock(m_pConnectionsLock, "~GSocketServerBase");
		size_t nCount = m_connections.size();
		size_t n;
		SOCKET sock;
		for(n = 0; n < nCount; n++)
		{
			sock = m_connections[n];
			if(sock != INVALID_SOCKET)
			{
				shutdown(sock, 2);
				CloseSocket(sock);
				if(m_connections.size() > n)
					m_connections[n] = INVALID_SOCKET;
			}
		}
		m_connections.clear();
	}
	delete[] m_szReceiveBuffer;
	delete(m_pConnectionsLock);
}

void GSocketServerBase::JoinWorkerThread()
{
	m_bKeepWorking = false;
	time_t tStart;
	time_t tNow;
	time(&tStart);
	while(m_hWorkerThread != BAD_HANDLE)
	{
		GThread::sleep(0);
		time(&tNow);
		if(tNow - tStart > 4)
		{
			GAssert(false); // Error, took too long for the worker thread to exit
			break;
		}
	}
}

int GSocketServerBase::GetFirstAvailableConnectionNumber()
{
	// Find the first empty Handle slot for the listening thread
	GAssert(m_pConnectionsLock->isLocked()); // The connections lock should be held when this is called
	int nSize = (int)m_connections.size();
	int nSocketNumber = -1;
	int n;
	for(n = 0; n < nSize; n++)
	{
		if(m_connections[n] == INVALID_SOCKET)
		{
			nSocketNumber = n;
			break;
		}
	}

	// Add a new slot if we couldn't find one
	if(nSocketNumber < 0 && (int)m_connections.size() < m_nMaxConnections)
	{
		nSocketNumber = nSize;
		m_connections.push_back(INVALID_SOCKET);
	}

	return nSocketNumber;
}

SOCKET GSocketServerBase::RefreshSocketSet()
{
	// Clear the set
	FD_ZERO(&m_socketSet);

	// Add the connection listener socket so that select() will return if a new connection comes in
	SOCKET highSocket = 0;
	if(m_socketConnectionListener != INVALID_SOCKET)
	{
		FD_SET(m_socketConnectionListener, &m_socketSet);
		if(m_socketConnectionListener > highSocket)
			highSocket = m_socketConnectionListener;
	}

	// Add all the current connections to the set
	{
		GSpinLockHolder hLock(m_pConnectionsLock, "RefreshSocketSet");
		int nCount = (int)m_connections.size();
		SOCKET s;
		int n;
		for(n = 0; n < nCount; n++)
		{
			s = m_connections[n];
			if(s != INVALID_SOCKET)
			{
				FD_SET(s, &m_socketSet);
				if(s > highSocket)
					highSocket = s;
			}
		}
	}
	GAssert(!m_pConnectionsLock->isLocked()); // Didn't release lock

	return highSocket;
}

int GSocketServerBase::HandleNewConnection()
{
	// Accept the connection
	SOCKET s;
	SOCKADDR_IN sHostAddrIn;
	socklen_t nStructSize = sizeof(struct sockaddr);
	s = accept(m_socketConnectionListener, (struct sockaddr*)&sHostAddrIn, &nStructSize);

	// Set the connection to non-blocking mode
	SetSocketToNonBlockingMode(s);

	// Find a place for the new socket
	int nConnection = GetFirstAvailableConnectionNumber();
	if(nConnection < 0)
	{
		GAssert(false); // no room for this connection

		// We accepted the connection even though we didn't have room for it so
		// we can close it so it won't keep bugging us about accepting it.
		CloseSocket(s);
		return -1;
	}
	m_connections[nConnection] = s;
	return nConnection;
}

unsigned int ServerWorkerThread(void* pData)
{
	((GSocketServerBase*)pData)->ServerWorker();
	return 0;
}

void GSocketServerBase::ServerWorker()
{
#ifdef WINDOWS
	GWindows::yieldToWindows();
#endif
	int n, nCount, nBytes;
	struct timeval timeout;
	int nReadySocketCount; // the number of sockets ready for reading
	SOCKET s, highSocket;
	while(m_bKeepWorking)
	{
		// We need to refresh the socket set each time we loop because select() changes the set
		highSocket = RefreshSocketSet();

		// Check which sockets are ready for reading
		timeout.tv_sec = 1;
		timeout.tv_usec = 0;
#ifdef WINDOWS
		nReadySocketCount = select((int)highSocket + 1, &m_socketSet, NULL, NULL, &timeout);
#else
		nReadySocketCount = select(highSocket + 1, &m_socketSet, NULL, NULL, &timeout);
#endif
		// Handle errors
		if(nReadySocketCount < 0)
		{
			const char* szError = gsocket_GetLastError();
			cerr << "*** Socket error: " << szError << "\n";
			break;
		}

		// Read from the ready sockets
		if(nReadySocketCount > 0)
		{
			// Check the connection listener socket for incoming connections
			if(m_socketConnectionListener != INVALID_SOCKET)
			{
				int nNewConnection = -1;
				{
					GSpinLockHolder hLock(m_pConnectionsLock, "ServerWorker 1");
					if(m_socketConnectionListener != INVALID_SOCKET)
					{
						if(FD_ISSET(m_socketConnectionListener, &m_socketSet))
							nNewConnection = HandleNewConnection();
					}
				}

				// WARNING: the accept function will return as soon as it gets
				//         an ACK packet back from the client, but the connection
				//         isn't actually established until more data is
				//         received.  Therefore, if you try to send data immediately
				//         (which someone might want to do in OnAcceptConnetion, the
				//         data might be lost since the connection might not be
				//         fully open.
				if(nNewConnection >= 0)
					onAcceptConnection(nNewConnection);

			}
			GAssert(!m_pConnectionsLock->isLocked()); // Didn't release lock

			// Check each connection socket for incoming data
			nCount = (int)m_connections.size();
			for(n = 0; n < nCount; n++)
			{
				s = m_connections[n];
				if(s != INVALID_SOCKET && FD_ISSET(s, &m_socketSet))
				{
					// The recv() function blocks until there is something to read or the
					// connection is closed, (but in this case we already know there is
					// something to read, so we won't be blocked for very long.)
					nBytes = recv(s, m_szReceiveBuffer, 2048, 0);
					if(nBytes > 0)
					{
						Receive((unsigned char*)m_szReceiveBuffer, nBytes, n);
					}
					else
					{
						// The socket was closed or an error occurred. Either way, close the socket
						onCloseConnection(n);
						CloseSocket(s);
						m_connections[n] = INVALID_SOCKET;
						GSpinLockHolder hLock(m_pConnectionsLock, "ServerWorker 2");
						ReduceConnectionList();
					}
					GAssert(!m_pConnectionsLock->isLocked()); // Didn't release lock
				}
			}
		}
		else
			GThread::sleep(100);
	}
	m_hWorkerThread = BAD_HANDLE;
}

void GSocketServerBase::Init(bool bUDP, int nPort, int nMaxConnections)
{
	m_nMaxConnections = nMaxConnections;
#ifdef WINDOWS
	if(!gsocket_InitWinSock())
		throw "failed to init WinSock";
#endif

	GAssert(nPort > 0); // invalid port number
	if(m_bUDP)
	{
		ThrowError("UDP not implemented yet");
	}
	m_bUDP = bUDP;

	// Make the Socket
	m_socketConnectionListener = socket(AF_INET, m_bUDP ? SOCK_DGRAM : SOCK_STREAM, 0);
	if(m_socketConnectionListener == INVALID_SOCKET)
	{
		gsocket_LogError();
		throw "faled to make a socket";
	}

	// Put the socket into non-blocking mode (so the call to "accept" will return immediately
	// if there are no connections in the queue ready to be accepted)
	SetSocketToNonBlockingMode(m_socketConnectionListener);

	// Tell the socket that it's okay to reuse an old crashed socket that hasn't timed out yet
	int flag = 1;
	setsockopt(m_socketConnectionListener, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(flag));

	// Prepare the socket for accepting
	memset(&m_sHostAddrIn, '\0', sizeof(SOCKADDR_IN));
	m_sHostAddrIn.sin_family = AF_INET;
	m_sHostAddrIn.sin_port = htons((u_short)nPort);
	m_sHostAddrIn.sin_addr.s_addr = htonl(INADDR_ANY);
	if(bind(m_socketConnectionListener, (struct sockaddr*)&m_sHostAddrIn, sizeof(SOCKADDR)))
	{
		gsocket_LogError();
		throw "failed to bind a socket";
	}

	// Start listening for connections
	if(listen(m_socketConnectionListener, nMaxConnections))
	{
		gsocket_LogError();
		throw "Failed to listen on a socket";
	}

	// Spawn the worker thread
	m_hWorkerThread = GThread::spawnThread(ServerWorkerThread, this);
	if(m_hWorkerThread == BAD_HANDLE)
		throw "Failed to spawn worker thread";

	// Give the worker thread a chance to awake
	GThread::sleep(0);
}

void GSocketServerBase::ReduceConnectionList()
{
	GAssert(m_pConnectionsLock->isLocked()); // Should be locked when this is called
	while(true)
	{
		int n = (int)m_connections.size();
		if(n <= 0)
			break;
		if(m_connections[n - 1] != INVALID_SOCKET)
			break;
		m_connections.pop_back();
	}
}

void GSocketServerBase::Disconnect(int nConnectionNumber)
{
	GSpinLockHolder hLock(m_pConnectionsLock, "Disconnect");
	if(nConnectionNumber < 0 || nConnectionNumber >= (int)m_connections.size())
		return;
	SOCKET s = m_connections[nConnectionNumber];
	if(s != INVALID_SOCKET)
	{
		onCloseConnection(nConnectionNumber);
		shutdown(s, 2);
		m_connections[nConnectionNumber] = INVALID_SOCKET;
		CloseSocket(s);
		ReduceConnectionList();
	}
}

bool GSocketServerBase::Send(const unsigned char *pBuff, size_t len, int nConnectionNumber)
{
	SOCKET s;
	{
		GSpinLockHolder hLock(m_pConnectionsLock, "GSocketServerBase::Send");
		if(nConnectionNumber < 0 || nConnectionNumber >= (int)m_connections.size())
			return false;
		s = m_connections[nConnectionNumber];
	}
	GAssert(!m_pConnectionsLock->isLocked()); // didn't release lock
	if(s == SOCKET_ERROR)
		return false;
	int nBytesSent;
	do
	{
		nBytesSent = send(s, (const char*)pBuff, (int)len, 0);
		if(nBytesSent < 0)
		{
#ifdef WINDOWS
			int n = WSAGetLastError();
			switch(n)
			{
				case WSAECONNABORTED:	break;
				case WSAECONNRESET:	break;
				default:	gsocket_LogError();		break;
			}
#else // WINDOWS
			if(errno == EAGAIN)
			{
				GThread::sleep(10); // todo: we should call "select" to determine when the socket is ready again
				continue;
			}
#endif // !WINDOWS
			return false;
		}
		else
		{
			len -= nBytesSent;
			pBuff += nBytesSent;
		}
	} while(len > 0);
	return true;
}

void GSocketServerBase::onCloseConnection(int nConnection)
{

}

void GSocketServerBase::onAcceptConnection(int nConnection)
{

}

bool GSocketServerBase::IsConnected(int nConnectionNumber)
{
	ThrowError("Not implemented yet");
	/*
	if(nConnectionNumber == 0)
	{
		if(m_hWorkerThread == BAD_HANDLE)
			return false;
		else
			return true;
	}
	else
	{
		if(m_pHostListenThreads->GetSize() < nConnectionNumber)
			return false;
		if(m_pHostListenThreads->GetHandle(nConnectionNumber - 1) == BAD_HANDLE)
			return false;
		else
			return true;
	}
	*/
	return false;
}

void GSocketServerBase::DontAcceptAnyMoreConnections()
{
	GSpinLockHolder hLock(m_pConnectionsLock, "DontAcceptAnyMoreConnections");
	if(m_socketConnectionListener != INVALID_SOCKET)
	{
		shutdown(m_socketConnectionListener, 2);
		CloseSocket(m_socketConnectionListener);
		m_socketConnectionListener = INVALID_SOCKET;
	}
}

SOCKET GSocketServerBase::GetSocketHandle(int nConnectionNumber)
{
	if(nConnectionNumber < 0)
		return m_socketConnectionListener;
	else
		return m_connections[nConnectionNumber];
}

in_addr GSocketServerBase::GetIPAddr(int nConnectionNumber)
{
	if(nConnectionNumber < 0)
	{
		char szHostName[256];
		if(gethostname(szHostName, sizeof(szHostName)) == SOCKET_ERROR)
			gsocket_LogError();
		struct hostent* pHostEnt = gethostbyname(szHostName);
		if(pHostEnt == 0)
			ThrowError("Couldn't resolve an IP address for ", szHostName);
		struct in_addr addr, cand;
		int nGoodness = -1;

		// Find the most-accessible address
		for (int i = 0; pHostEnt->h_addr_list[i] != 0; ++i)
		{
			memcpy(&cand, pHostEnt->h_addr_list[i], sizeof(struct in_addr));
			const char* szAddr = inet_ntoa(cand);
			int nScore;
			if(strcmp(szAddr, "127.0.0.1") == 0) // 127.0.0.1 is the loopback address for localhost
				nScore = 0;
			else if(strncmp(szAddr, "10.", 3) == 0) // 10.0.0.0 - 10.255.255.255 are reserved for private networks
				nScore = 1;
			else if(strncmp(szAddr, "192.168.", 8) == 0) // 192.168.0.0 - 192.168.255.255 are reserved for private networks
				nScore = 2;
			else if(strncmp(szAddr, "169.254.", 8) == 0) // 169.254.0.0 - 169.254.255.255 are reserved for automatic private addressing
				nScore = 3;
			else if(strncmp(szAddr, "172.", 4) == 0) // 172.16.0.0 - 172.31.255.255 are reserved for private networks
				nScore = 4;
			else
				nScore = 5;
			if(nScore > nGoodness)
			{
				memcpy(&addr, pHostEnt->h_addr_list[i], sizeof(struct in_addr));
				nGoodness = nScore;
			}
		}
		return addr;
	}
	else
	{
		struct sockaddr sAddr;
		socklen_t l;
		l = sizeof(SOCKADDR);
		if(getpeername(m_connections[nConnectionNumber], &sAddr, &l))
			gsocket_LogError();
		if(sAddr.sa_family != AF_INET)
			ThrowError("Error, family is not AF_INET");
		SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
		return pInfo->sin_addr;
	}
}

char* GSocketServerBase::GetIPAddr(char* szBuff, size_t nBuffSize, int nConnectionNumber)
{
	safe_strcpy(szBuff, inet_ntoa(GetIPAddr(nConnectionNumber)), nBuffSize);
	return szBuff;
}

u_short GSocketServerBase::GetPort(int nConnectionNumber)
{
	SOCKADDR sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(nConnectionNumber < 0)
	{
		if(getsockname(m_socketConnectionListener, &sAddr, &l))
			gsocket_LogError();
	}
	else
	{
		if(getpeername(m_connections[nConnectionNumber], &sAddr, &l))
			gsocket_LogError();
	}
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	return htons(pInfo->sin_port);
}

char* GSocketServerBase::GetName(char* szBuff, size_t nBuffSize, int nConnectionNumber)
{
	SOCKADDR sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(nConnectionNumber < 0)
	{
		if(getsockname(m_socketConnectionListener, &sAddr, &l))
			gsocket_LogError();
	}
	else
	{
		if(getpeername(m_connections[nConnectionNumber], &sAddr, &l))
			gsocket_LogError();
	}
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, family is not AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	HOSTENT* namestruct = gethostbyaddr((const char*)&pInfo->sin_addr, 4, pInfo->sin_family);
	if(!namestruct)
		ThrowError("Error calling gethostbyaddr");
	safe_strcpy(szBuff, namestruct->h_name, nBuffSize);
	return(szBuff);
}

// --------------------------------------------------------------------------

const char GSocketTag[] = "GSKT";

class GSocketServerBuffer
{
public:
	unsigned char* m_pBuffer;
	size_t m_nBufferPos;

	GSocketServerBuffer(size_t nMaxPacketSize)
	{
		m_pBuffer = new unsigned char[nMaxPacketSize + sizeof(struct GEZSocketPacketHeader)];
		m_nBufferPos = 0;
	}

	~GSocketServerBuffer()
	{
		delete[] m_pBuffer;
	}
};

GSocketServer::GSocketServer(bool bUDP, size_t nMaxPacketSize, int nPort, int nMaxConnections) : GSocketServerBase(bUDP, nPort, nMaxConnections)
{
	GAssert(sizeof(struct GEZSocketPacketHeader) == 8); // packing issue
	m_nMaxPacketSize = nMaxPacketSize;
	if(nMaxPacketSize > 0)
		m_pBuffers = new vector<GSocketServerBuffer*>();
	else
		m_pBuffers = NULL;
	m_pMessageQueueLock = new GSpinLock();
}

GSocketServer::~GSocketServer()
{
	// Join the worker thread now it doesn't try
	// to queue up a message after we delete the
	// message queue
	JoinWorkerThread();

	if(m_pBuffers)
	{
		int nCount = (int)m_pBuffers->size();
		int n;
		for(n = 0; n < nCount; n++)
			delete((*m_pBuffers)[n]);
		delete(m_pBuffers);
	}
	while(m_messageQueue.size() > 0)
	{
		delete(m_messageQueue.front());
		m_messageQueue.pop_front();
	}
	delete(m_pMessageQueueLock);
}

void GSocketServer::QueueMessage(unsigned char* pBuf, size_t nLen, int nConnectionNumber)
{
	GSpinLockHolder hLock(m_pMessageQueueLock, "GSocketServer::QueueMessage");
	GSocketMessage* pNewMessage = new GSocketMessage(pBuf, nLen, nConnectionNumber);
	m_messageQueue.push_back(pNewMessage);
}

size_t GSocketServer::GetMessageCount()
{
	return m_messageQueue.size();
}

unsigned char* GSocketServer::GetNextMessage(size_t* pnSize, int* pnOutConnectionNumber)
{
	GSocketMessage* pMessage;
	{
		GSpinLockHolder hLock(m_pMessageQueueLock, "GSocketClient::GetNextMessage");
		if(m_messageQueue.size() == 0)
		{
			*pnOutConnectionNumber = -1;
			return NULL;
		}
		pMessage = m_messageQueue.front();
		m_messageQueue.pop_front();
	}
	*pnSize = pMessage->GetMessageSize();
	*pnOutConnectionNumber = pMessage->GetConnection();
	unsigned char* pBuf = pMessage->TakeBuffer();
	delete(pMessage);
	return pBuf;
}

bool GSocketServer::Receive(unsigned char *pBuf, size_t nLen, int nConnectionNumber)
{
	if(m_nMaxPacketSize == 0)
	{
		QueueMessage(pBuf, nLen, nConnectionNumber);
	}
	else
	{
		while(nConnectionNumber >= (int)m_pBuffers->size())
			m_pBuffers->push_back(new GSocketServerBuffer(m_nMaxPacketSize));
		GSocketServerBuffer* pBuffer = (*m_pBuffers)[nConnectionNumber];
		while(nLen > 0)
		{
			if(pBuffer->m_nBufferPos == 0 &&
				nLen >= sizeof(struct GEZSocketPacketHeader) &&
				nLen >= sizeof(struct GEZSocketPacketHeader) + GBits::littleEndianToN32(((struct GEZSocketPacketHeader*)pBuf)->nPayloadSize))
			{
				// We've got a whole packet, so just queue it up
				GAssert(*(unsigned int*)pBuf == *(unsigned int*)GSocketTag); // Bad Packet
				int nSize = GBits::littleEndianToN32(((struct GEZSocketPacketHeader*)pBuf)->nPayloadSize);
				pBuf += sizeof(struct GEZSocketPacketHeader);
				nLen -= sizeof(struct GEZSocketPacketHeader);
				QueueMessage(pBuf, nSize, nConnectionNumber);
				pBuf += nSize;
				nLen -= nSize;
			}
			else
			{
				// We've only got a partial packet, so we need to buffer it
				while(pBuffer->m_nBufferPos < (int)sizeof(struct GEZSocketPacketHeader) && nLen > 0)
				{
					pBuffer->m_pBuffer[pBuffer->m_nBufferPos] = *pBuf;
					if(pBuffer->m_nBufferPos < 4 && *pBuf != GSocketTag[pBuffer->m_nBufferPos])
					{
						GAssert(false); // bad packet
						pBuffer->m_nBufferPos = 0;
					}
					else
						pBuffer->m_nBufferPos++;
					pBuf++;
					nLen--;
				}
				if(pBuffer->m_nBufferPos < (int)sizeof(struct GEZSocketPacketHeader))
					return true;
				struct GEZSocketPacketHeader* pHeader = (struct GEZSocketPacketHeader*)pBuffer->m_pBuffer;
				int nSize = GBits::littleEndianToN32(pHeader->nPayloadSize);
				if((size_t)nSize > m_nMaxPacketSize)
				{
					GAssert(false); // Received a packet that was too big
					pHeader->nPayloadSize = (unsigned int)m_nMaxPacketSize;
				}
				while(pBuffer->m_nBufferPos < sizeof(struct GEZSocketPacketHeader) + nSize && nLen > 0)
				{
					pBuffer->m_pBuffer[pBuffer->m_nBufferPos] = *pBuf;
					pBuffer->m_nBufferPos++;
					pBuf++;
					nLen--;
				}
				if(pBuffer->m_nBufferPos < sizeof(struct GEZSocketPacketHeader) + nSize)
					return true;
				QueueMessage(pBuffer->m_pBuffer + sizeof(struct GEZSocketPacketHeader), nSize, nConnectionNumber);
				pBuffer->m_nBufferPos = 0;
			}
		}
	}
	return true;
}

bool GSocketServer::Send(const void* pBuf, size_t nLen, int nConnectionNumber)
{
	if(m_nMaxPacketSize > 0)
	{
		GAssert(nLen <= m_nMaxPacketSize); // packet too big
		struct GEZSocketPacketHeader header;
		header.tag[0] = GSocketTag[0];
		header.tag[1] = GSocketTag[1];
		header.tag[2] = GSocketTag[2];
		header.tag[3] = GSocketTag[3];
		header.nPayloadSize = GBits::n32ToLittleEndian((unsigned int)nLen);
		if(!GSocketServerBase::Send((const unsigned char*)&header, sizeof(struct GEZSocketPacketHeader), nConnectionNumber))
			return false;
	}
	bool bRet = GSocketServerBase::Send((const unsigned char*)pBuf, nLen, nConnectionNumber);
	return bRet;
}

bool GSocketServer::Send2(const void* pBuf1, size_t nLen1, const void* pBuf2, size_t nLen2, int nConnectionNumber)
{
	if(m_nMaxPacketSize > 0)
	{
		GAssert(nLen1 + nLen2 <= m_nMaxPacketSize); // packet too big
		struct GEZSocketPacketHeader header;
		header.tag[0] = GSocketTag[0];
		header.tag[1] = GSocketTag[1];
		header.tag[2] = GSocketTag[2];
		header.tag[3] = GSocketTag[3];
		header.nPayloadSize = GBits::n32ToLittleEndian((unsigned int)(nLen1 + nLen2));
		if(!GSocketServerBase::Send((const unsigned char*)&header, sizeof(struct GEZSocketPacketHeader), nConnectionNumber))
			return false;
	}
	bool bRet = GSocketServerBase::Send((const unsigned char*)pBuf1, nLen1, nConnectionNumber);
	if(bRet)
		bRet = GSocketServerBase::Send((const unsigned char*)pBuf2, nLen2, nConnectionNumber);
	return bRet;
}

// --------------------------------------------------------------------------

GSocketClient::GSocketClient(bool bUDP, size_t nMaxPacketSize) : GSocketClientBase(bUDP)
{
	m_nMaxPacketSize = nMaxPacketSize;
	if(nMaxPacketSize > 0)
		m_pBuffer = new unsigned char[nMaxPacketSize + sizeof(struct GEZSocketPacketHeader)];
	else
		m_pBuffer = NULL;
	m_nBufferPos = 0;
	m_pMessageQueueLock = new GSpinLock();
}

GSocketClient::~GSocketClient()
{
	// Join the other threads now so they don't try
	// to queue up a message after we delete the
	// message queue
	joinListenThread();

	delete[] m_pBuffer;
	while(m_messageQueue.size() > 0)
	{
		delete(m_messageQueue.front());
		m_messageQueue.pop_front();
	}
	delete(m_pMessageQueueLock);
}

void GSocketClient::QueueMessage(unsigned char* pBuf, size_t nLen)
{
	GSpinLockHolder hLock(m_pMessageQueueLock, "GSocketClient::QueueMessage");
	m_messageQueue.push_back(new GSocketMessage(pBuf, nLen, 0));
}

size_t GSocketClient::GetMessageCount()
{
	return m_messageQueue.size();
}

unsigned char* GSocketClient::GetNextMessage(size_t* pnSize)
{
	if(m_messageQueue.size() == 0)
	{
		*pnSize = 0;
		return NULL;
	}
	GSocketMessage* pMessage;
	{
		GSpinLockHolder hLock(m_pMessageQueueLock, "GSocketClient::GetNextMessage");
		pMessage = m_messageQueue.front();
		m_messageQueue.pop_front();
	}
	*pnSize = pMessage->GetMessageSize();
	unsigned char* pBuf = pMessage->TakeBuffer();
	delete(pMessage);
	return pBuf;
}

bool GSocketClient::Receive(unsigned char *pBuf, size_t nLen)
{
	if(m_nMaxPacketSize == 0)
		QueueMessage(pBuf, nLen);
	else
	{
		while(nLen > 0)
		{
			if(m_nBufferPos == 0 &&
				nLen >= (int)sizeof(struct GEZSocketPacketHeader) &&
				nLen >= (int)sizeof(struct GEZSocketPacketHeader) + GBits::littleEndianToN32(((struct GEZSocketPacketHeader*)pBuf)->nPayloadSize))
			{
				// We've got a whole packet, so just queue it up
				GAssert(*(unsigned int*)pBuf == *(unsigned int*)GSocketTag); // Bad Packet
				int nSize = GBits::littleEndianToN32(((struct GEZSocketPacketHeader*)pBuf)->nPayloadSize);
				pBuf += sizeof(struct GEZSocketPacketHeader);
				nLen -= sizeof(struct GEZSocketPacketHeader);
				QueueMessage(pBuf, nSize);
				pBuf += nSize;
				nLen -= nSize;
			}
			else
			{
				// We've only got a partial packet, so we need to buffer it
				while(m_nBufferPos < (int)sizeof(struct GEZSocketPacketHeader) && nLen > 0)
				{
					m_pBuffer[m_nBufferPos] = *pBuf;
					if(m_nBufferPos < 4 && *pBuf != GSocketTag[m_nBufferPos])
						m_nBufferPos = -1;
					m_nBufferPos++;
					pBuf++;
					nLen--;
				}
				if(m_nBufferPos < (int)sizeof(struct GEZSocketPacketHeader))
					return true;
				struct GEZSocketPacketHeader* pHeader = (struct GEZSocketPacketHeader*)m_pBuffer;
				int nSize = GBits::littleEndianToN32(pHeader->nPayloadSize);
				if((size_t)nSize > m_nMaxPacketSize)
				{
					GAssert(false); // Received a packet that was too big
					pHeader->nPayloadSize = (unsigned int)m_nMaxPacketSize;
				}
				while(m_nBufferPos < sizeof(struct GEZSocketPacketHeader) + nSize && nLen > 0)
				{
					m_pBuffer[m_nBufferPos] = *pBuf;
					m_nBufferPos++;
					pBuf++;
					nLen--;
				}
				if(m_nBufferPos < sizeof(struct GEZSocketPacketHeader) + nSize)
					return true;
				QueueMessage(m_pBuffer + sizeof(struct GEZSocketPacketHeader), nSize);
				m_nBufferPos = 0;
			}
		}
	}
	return true;
}

bool GSocketClient::Send(const void* pBuf, size_t nLen)
{
	if(m_nMaxPacketSize > 0)
	{
		GAssert(nLen <= m_nMaxPacketSize); // packet too big
		struct GEZSocketPacketHeader header;
		header.tag[0] = GSocketTag[0];
		header.tag[1] = GSocketTag[1];
		header.tag[2] = GSocketTag[2];
		header.tag[3] = GSocketTag[3];
		header.nPayloadSize = GBits::n32ToLittleEndian((unsigned int)nLen);
		if(!GSocketClientBase::Send((const unsigned char*)&header, sizeof(struct GEZSocketPacketHeader)))
			return false;
	}
	return GSocketClientBase::Send((const unsigned char*)pBuf, nLen);
}

bool GSocketClient::Send2(const void* pBuf1, size_t nLen1, const void* pBuf2, size_t nLen2)
{
	if(m_nMaxPacketSize > 0)
	{
		GAssert(nLen1 + nLen2 <= m_nMaxPacketSize); // packet too big
		struct GEZSocketPacketHeader header;
		header.tag[0] = GSocketTag[0];
		header.tag[1] = GSocketTag[1];
		header.tag[2] = GSocketTag[2];
		header.tag[3] = GSocketTag[3];
		header.nPayloadSize = GBits::n32ToLittleEndian((unsigned int)(nLen1 + nLen2));
		if(!GSocketClientBase::Send((const unsigned char*)&header, sizeof(struct GEZSocketPacketHeader)))
			return false;
	}
	bool bRet = GSocketClientBase::Send((const unsigned char*)pBuf1, nLen1);
	if(bRet)
		bRet = GSocketClientBase::Send((const unsigned char*)pBuf2, nLen2);
	return bRet;
}

#ifndef NO_TEST_CODE
#define TEST_SOCKET_PORT 4464

void TestGSocketSerial(bool bGash, const char* szAddr, int port)
{
	// Establish a socket connection
	GSocketServer* pServer = new GSocketServer(false, bGash ? 5000 : 0, port, 1000);
	Holder<GSocketServer> hServer(pServer);
	if(!pServer)
		throw "failed to init the server";
	GSocketClient* pClient = new GSocketClient(false, 5000);
	Holder<GSocketClient> hClient(pClient);
	if(!pClient->Connect(szAddr, port, 10))
		throw "failed to make the client";

	// Send a bunch of data
	int i;
	char szBuf[5000];
	for(i = 0; i < 5000; i++)
		szBuf[i] = (char)i;
	pClient->Send(szBuf, 5000);
#ifdef WINDOWS
	GWindows::yieldToWindows();
#endif
	for(i = 10; i < 60; i++)
	{
		if(!pClient->Send(szBuf + i, i))
			ThrowError("failed");
#ifdef WINDOWS
		GWindows::yieldToWindows();
#endif
	}
	pClient->Send(szBuf, 5000);
#ifdef WINDOWS
	GWindows::yieldToWindows();
#endif

	// Wait for the data to arrive
	int nTimeout;
	for(nTimeout = 500; nTimeout > 0; nTimeout--)
	{
		if(pServer->GetMessageCount() == 52)
			break;
		GThread::sleep(50);
	}

	if(pServer->GetMessageCount() != 52)
		ThrowError("failed");

	// Check the data and send some of it back to the client
	size_t nSize;
	int nConnection;
	{
		ArrayHolder<unsigned char> hData(pServer->GetNextMessage(&nSize, &nConnection));
		unsigned char* pData = hData.get();
		if(!pData || pData == (unsigned char*)szBuf)
			ThrowError("failed");
		if(nSize != 5000)
			ThrowError("failed");
		if(pData[122] != (char)122)
			ThrowError("failed");
	}
	for(i = 10; i < 60; i++)
	{
		ArrayHolder<unsigned char> hData(pServer->GetNextMessage(&nSize, &nConnection));
		unsigned char* pData = hData.get();
		if(nSize != i || !pData)
			ThrowError("failed");
		if(pData[0] != (char)i)
			ThrowError("failed");
		if(pData[i - 1] != (char)(i + i - 1))
			ThrowError("failed");
		if(!pServer->Send(pData, 10, nConnection))
			ThrowError("failed");
#ifdef WINDOWS
		GWindows::yieldToWindows();
#endif
	}
	{
		ArrayHolder<unsigned char> hData(pServer->GetNextMessage(&nSize, &nConnection));
		unsigned char* pData = hData.get();
		if(!pData || pData == (unsigned char*)szBuf)
			ThrowError("failed");
		if(nSize != 5000)
			ThrowError("failed");
		if(pData[122] != (char)122)
			ThrowError("failed");
	}

	// Wait for the data to arrive
	for(nTimeout = 500; nTimeout > 0; nTimeout--)
	{
		if(pClient->GetMessageCount() == 50)
			break;
		GThread::sleep(50);
	}
	if(pClient->GetMessageCount() != 50)
		ThrowError("failed");

	// Check the data
	for(i = 10; i < 60; i++)
	{
		ArrayHolder<unsigned char> hData(pClient->GetNextMessage(&nSize));
		unsigned char* pData = hData.get();
		if(nSize != 10 || !pData)
			ThrowError("failed");
		if(pData[0] != (char)i)
			ThrowError("failed");
		if(pData[9] != (char)(i + 9))
			ThrowError("failed");
	}
}

struct TestGSocketParallelData
{
};

unsigned int TestGSocketParallelClientThread(void* pParam)
{
	//TestGSocketParallelData* pData = (TestGSocketParallelData*)pParam;
	GSocketClient* pClient = new GSocketClient(false, 5000);
	Holder<GSocketClient> hClient(pClient);
	if(!pClient->Connect("localhost", TEST_SOCKET_PORT, 10))
		ThrowError("failed to connect");
	char szBuf[1025];
	memset(szBuf, 'f', 1025);
	int n;
	for(n = 0; n < 100; n++)
		pClient->Send(szBuf, 1025);
	return 0;
}

void TestGSocketParallel()
{
	// Establish a socket connection
	GSocketServer* pServer = new GSocketServer(false, 5000, TEST_SOCKET_PORT, 1000);
	Holder<GSocketServer> hServer(pServer);
	TestGSocketParallelData sData;
	if(!pServer)
		ThrowError("failed");
	GThread::spawnThread(TestGSocketParallelClientThread, &sData);
	GThread::spawnThread(TestGSocketParallelClientThread, &sData);
	GThread::spawnThread(TestGSocketParallelClientThread, &sData);
	GThread::spawnThread(TestGSocketParallelClientThread, &sData);
	GThread::spawnThread(TestGSocketParallelClientThread, &sData);
	int i;
	size_t nSize;
	int nConnection;
	for(i = 0; i < 500; i++)
	{
		ArrayHolder<unsigned char> hData(pServer->GetNextMessage(&nSize, &nConnection));
		if(!hData.get())
		{
			i--;
			GThread::sleep(0);
			continue;
		}
		if(nSize != 1025)
			ThrowError("failed");
	}
}

void GSocketClient::test()
{
	TestGSocketSerial(true, "localhost", TEST_SOCKET_PORT);
	TestGSocketParallel();
}
#endif // !NO_TEST_CODE

} // namespace GClasses

