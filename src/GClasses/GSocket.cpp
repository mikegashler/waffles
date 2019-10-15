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

#include "GSocket.h"
#include <time.h>
#include "GHolders.h"
#include "GError.h"
#include "GThread.h"
#include "GRand.h"
#include "GApp.h"
#include "GTime.h"
#include <wchar.h>
#include <sstream>
#include <iostream>
#include <queue>
#ifdef WINDOWS
#	include "GWindows.h"
#	include <Ws2tcpip.h>
#else
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

using std::cerr;
using std::vector;
using std::string;
using std::set;

#ifndef WINDOWS
typedef struct sockaddr SOCKADDR;
typedef struct sockaddr_in SOCKADDR_IN;
#endif

namespace GClasses {


#ifdef WINDOWS
string winstrerror(int err)
{
	char buf[1024];
	FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
		FORMAT_MESSAGE_MAX_WIDTH_MASK, NULL, err,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPSTR)buf, 1024, NULL);
    return string(buf);
}
#endif

bool GSocket_isReady(SOCKET s)
{
	fd_set sockSet;
	FD_ZERO(&sockSet);
	FD_SET(s, &sockSet);

	// Check which sockets are ready for reading
	struct timeval timeout;
	timeout.tv_sec = 0;
	timeout.tv_usec = 0;
	int ret = select((int)s + 1, &sockSet, NULL, NULL, &timeout);
	if(ret < 0)
	{
#ifdef WINDOWS
		throw Ex("select failed: ", winstrerror(WSAGetLastError()));
#else
		throw Ex("select failed: ", strerror(errno));
#endif
	}
	return ret == 0 ? false : true;
}

void GSocket_setSocketMode(SOCKET s, bool blocking)
{
	unsigned long ulMode = blocking ? 0 : 1;
#ifdef WINDOWS
	if(ioctlsocket(s, FIONBIO, &ulMode) != 0)
#else
	if(ioctl(s, FIONBIO, &ulMode) != 0)
#endif
		throw Ex("Error changing the mode of a socket");
}

size_t GSocket_bytesReady(SOCKET s)
{
	if(s == INVALID_SOCKET)
		return 0;
	unsigned long bytesReadyToRead = 0;
#ifdef WINDOWS
	GWindows::yield(); // This is necessary because incoming packets go through the Windows message pump
	if(ioctlsocket(s, FIONREAD, &bytesReadyToRead) != 0)
		throw Ex("ioctlsocket failed: ", winstrerror(WSAGetLastError()));
#else
	if(ioctl(s, FIONREAD, &bytesReadyToRead) != 0)
		throw Ex("ioctl failed: ", strerror(errno));
#endif
	return bytesReadyToRead;
}

void GSocket_closeSocket(SOCKET s)
{
	shutdown(s, 2/*SHUT_RDWR*/);
#ifdef WINDOWS
	closesocket(s);
#else
	close(s);
#endif
}

in_addr GSocket_ipAddr(SOCKET s)
{
	struct sockaddr sAddr;
	socklen_t l;
	l = sizeof(SOCKADDR);
	if(getpeername(s, &sAddr, &l))
	{
#ifdef WINDOWS
		throw Ex("getpeername failed: ", winstrerror(WSAGetLastError()));
#else
		throw Ex("getpeername failed: ", strerror(errno));
#endif
	}
	if(sAddr.sa_family != AF_INET)
		throw Ex("Error, expected family to be AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	return pInfo->sin_addr;
}

size_t GSocket_send(SOCKET s, const char* buf, size_t len)
{
	if(s == INVALID_SOCKET)
		throw Ex("Tried to send over a socket that was not connected");
	ssize_t bytesSent = ::send(s, buf, (int)len, 0);
	if(bytesSent < 0)
	{
#ifdef WINDOWS
		int err = WSAGetLastError();
		if(err == WSAEWOULDBLOCK)
			return 0;
		else
			throw Ex("Error sending in GTCPClient::send: ", winstrerror(err));
#else
		if(errno == EWOULDBLOCK)
			return 0;
		else
			throw Ex("Error sending in GTCPClient::send: ", strerror(errno));
#endif
		return 0;
	}
	else
		return (size_t)bytesSent;
}

void GSocket_init()
{
#ifdef WINDOWS
	// Initialize Winsock
	WORD wVersionRequested;
	WSADATA wsaData;
	wVersionRequested = MAKEWORD(1, 1);
	int err = WSAStartup(wVersionRequested, &wsaData);
	if(err != 0)
		throw Ex("Failed to find a usable WinSock DLL");

	// Confirm that the WinSock DLL supports at least 2.2.
	if ( LOBYTE( wsaData.wVersion ) != 1 ||
			HIBYTE( wsaData.wVersion ) != 1 )
	{
		int n1 = LOBYTE( wsaData.wVersion );
		int n2 = HIBYTE( wsaData.wVersion );
		WSACleanup();
		throw Ex("Found a Winsock DLL, but it only supports an older version. It needs to support version 2.2");
	}
#endif
}

SOCKET GSocket_connect(const char* addr, unsigned short port, int timeoutSecs)
{
	struct addrinfo hints, *res, *res0;
	int error;
	res0 = NULL;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	std::ostringstream os;
	os << port;
	string tmp = os.str();
	error = getaddrinfo(addr, tmp.c_str(), &hints, &res0);
	if(error)
		throw Ex(gai_strerror(error));
	SOCKET sock = INVALID_SOCKET;
	for(res = res0; res; res = res->ai_next)
	{
		sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
		if(sock < 0)
			continue;
		struct timeval timeout;
		fd_set socketSet;
		GSocket_setSocketMode(sock, false);

		// Trying to connect with timeout
		if(::connect(sock, res->ai_addr, (int)res->ai_addrlen) < 0)
		{
#ifdef WINDOWS
			int n = WSAGetLastError();
			if(n == WSAEWOULDBLOCK || n == WSAEINPROGRESS)
#else
			if(errno == EINPROGRESS)
#endif
			{
				timeout.tv_sec = timeoutSecs;
				timeout.tv_usec = 0;
				FD_ZERO(&socketSet);
				FD_SET(sock, &socketSet);
#ifdef WINDOWS
				GWindows::yield();
				int result = select((int)sock + 1, NULL, &socketSet, NULL, &timeout);
				if(result < 0 && WSAGetLastError() != WSAEINTR)
#else
				int result = select(sock + 1, NULL, &socketSet, NULL, &timeout);
				if(result < 0 && errno != EINTR)
#endif
					throw Ex("Failed to connect to ", addr, " on port ", to_str(port));
				else if(result > 0)
				{
					// Socket selected for write
					socklen_t lon = sizeof(int);
					int valopt;
					if(getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)(&valopt), &lon) < 0)
						throw Ex("getsockopt failed");
					if(valopt)
					{
						GSocket_closeSocket(sock);
						sock = INVALID_SOCKET;
						continue;
					}

					// Got a connection!
					break;
				}
				else
				{
					// Timeout exceeded
					GSocket_closeSocket(sock);
					sock = INVALID_SOCKET;
					continue;
				}
			}
			else
			{
				// Failed to connect to this address
				GSocket_closeSocket(sock);
				sock = INVALID_SOCKET;
				continue;
			}
		}
	}
	freeaddrinfo(res0);
	if(sock == INVALID_SOCKET)
		throw Ex("Failed to connect to ", addr, " on port ", to_str(port));
	return sock;
}




GTCPClient::GTCPClient()
: m_sock(INVALID_SOCKET)
{
	GSocket_init();
}

GTCPClient::~GTCPClient()
{
	disconnect();
}

void GTCPClient::disconnect()
{
	if(m_sock == INVALID_SOCKET)
		return;
	onDisconnect();
	GSocket_closeSocket(m_sock);
	m_sock = INVALID_SOCKET;
}

bool GTCPClient::isConnected()
{
	return m_sock == INVALID_SOCKET ? false : true;
}

void GTCPClient::connect(const char* addr, unsigned short port, int timeoutSecs)
{
	disconnect();
	m_sock = GSocket_connect(addr, port, timeoutSecs);
}

void GTCPClient::send(const char* buf, size_t len)
{
	try
	{
		while(len > 0)
		{
			size_t bytesSent = GSocket_send(m_sock, buf, len);
			if(bytesSent > 0)
			{
				buf += bytesSent;
				len -= bytesSent;
			}
			else
				GThread::sleep(0);
		}
	}
	catch(const std::exception& e)
	{
		disconnect();
		throw e;
	}
}

size_t GTCPClient::receive(char* buf, size_t len)
{
	size_t bytesReady = GSocket_bytesReady(m_sock);
	if(bytesReady > 0)
	{
		ssize_t bytesReceived = recv(m_sock, buf, (int)len, 0);
		if(bytesReceived > 0)
			return size_t(bytesReceived);
		else if(bytesReceived == 0)
		{
			disconnect();
			return 0;
		}
		else
		{
#ifdef WINDOWS
			throw Ex("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
			throw Ex("Error calling recv: ", strerror(errno));
#endif
		}
	}
	return 0;
}










in_addr GTCPConnection::ipAddr()
{
	return GSocket_ipAddr(m_sock);
}

const char* GTCPConnection::getIPAddress()
{
	return inet_ntoa(ipAddr());
}






GTCPServer::GTCPServer(unsigned short port)
{
	GSocket_init();
	m_sock = socket(AF_INET, SOCK_STREAM, 0); // use SOCK_DGRAM for UDP

	// Tell the socket that it's okay to reuse an old crashed socket that hasn't timed out yet
	int flag = 1;
	setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(flag));

	// Bind the socket to the port
	SOCKADDR_IN sHostAddrIn;
	memset(&sHostAddrIn, '\0', sizeof(SOCKADDR_IN));
	sHostAddrIn.sin_family = AF_INET;
	sHostAddrIn.sin_port = htons(port);
	sHostAddrIn.sin_addr.s_addr = htonl(INADDR_ANY);
	if(bind(m_sock, (struct sockaddr*)&sHostAddrIn, sizeof(SOCKADDR)) != 0)
	{
#ifdef WINDOWS
		throw Ex("Failed to bind to port ", to_str(port), ": ", winstrerror(WSAGetLastError()));
#else
		throw Ex("Failed to bind to port ", to_str(port), ": ", strerror(errno));
#endif
	}

	// Start listening for connections
	if(listen(m_sock, SOMAXCONN) != 0)
#ifdef WINDOWS
		throw Ex("Failed to listen on the socket: ", winstrerror(WSAGetLastError()));
#else
		throw Ex("Failed to listen on the socket: ", strerror(errno));
#endif
}

GTCPServer::~GTCPServer()
{
	while(m_socks.size() > 0)
		disconnect(*m_socks.begin());
	GSocket_closeSocket(m_sock);
}

void GTCPServer::disconnect(GTCPConnection* pConn)
{
	onDisconnect(pConn);
	GSocket_closeSocket(pConn->socket());
	m_socks.erase(pConn);
	delete(pConn);
}

void GTCPServer::checkForNewConnections()
{
	if(!GSocket_isReady(m_sock))
		return;

	// Accept the connection
	SOCKADDR_IN sHostAddrIn;
	socklen_t nStructSize = sizeof(struct sockaddr);
	SOCKET s = accept(m_sock, (struct sockaddr*)&sHostAddrIn, &nStructSize);
	if(s < 0)
	{
#ifdef WINDOWS
		if(WSAGetLastError() == WSAEWOULDBLOCK) // no connections are ready to be accepted
			return;
#else
		if(errno == EAGAIN) // no connections are ready to be accepted
			return;
#endif
		string s2 = "Received bad data while trying to accept a connection: ";
#ifdef WINDOWS
		s2 += winstrerror(WSAGetLastError());
#else
		s2 += strerror(errno);
#endif
		onReceiveBadData(s2.c_str());
		return;
	}
	GSocket_setSocketMode(s, false);
	m_socks.insert(makeConnection(s));
}

size_t GTCPServer::receive(char* buf, size_t len, GTCPConnection** pOutConn)
{
	checkForNewConnections();
	for(set<GTCPConnection*>::iterator it = m_socks.begin(); it != m_socks.end(); it++)
	{
		GTCPConnection* pConn = *it;
		if(GSocket_bytesReady(pConn->socket()) > 0)
		{
			ssize_t bytesReceived = recv(pConn->socket(), buf, (int)len, 0);
			if(bytesReceived > 0)
			{
				*pOutConn = pConn;
				return size_t(bytesReceived);
			}
			else if(bytesReceived == 0)
			{
				// The client has disconnected gracefully
				disconnect(pConn);

				// Recurse since the previous operation will invalidate the iterator
				return receive(buf, len, pOutConn);
			}
			else
			{
#ifdef WINDOWS
				throw Ex("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
				throw Ex("Error calling recv: ", strerror(errno));
#endif
			}
		}
	}
	return 0;
}

void GTCPServer::send(const char* buf, size_t len, GTCPConnection* pConn)
{
	try
	{
		while(len > 0)
		{
			size_t bytesSent = GSocket_send(pConn->socket(), buf, len);
			if(bytesSent > 0)
			{
				buf += bytesSent;
				len -= bytesSent;
			}
			else
				GThread::sleep(0);
		}
	}
	catch(const std::exception& e)
	{
		disconnect(pConn);
		throw e;
	}
}

// static
void GTCPServer::hostName(char* buf, size_t len)
{
	if(gethostname(buf, (int)len) == SOCKET_ERROR)
		throw Ex("failed to get host noame");
}

/*
// static
in_addr GTCPServer::hostNameToIPAddress(char* szHostName)
{
	struct hostent* pHostEnt = gethostbyname(szHostName);
	if(pHostEnt == 0)
		throw Ex("Couldn't resolve an IP address for ", szHostName);
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
*/



#define MAGIC_VALUE 0x0b57ac1e

int GPackageConnection::receive(unsigned int maxBufSize, unsigned int maxPackageSize)
{
	if(GSocket_bytesReady(m_sock) == 0)
		return 0; // Nothing bad happened
	if(m_headerBytes < 2 * sizeof(unsigned int)) // if the header is still incomplete...
	{
		// Receive the header
		ssize_t bytesReceived = recv(m_sock, ((char*)m_header) + m_headerBytes, 2 * sizeof(unsigned int) - m_headerBytes, 0);
		if(bytesReceived > 0) // if we successfully received something...
		{
			m_headerBytes += (unsigned int)bytesReceived;
			if(m_headerBytes >= 2 * sizeof(unsigned int))
			{
				if(m_header[0] == MAGIC_VALUE)
				{
					if(m_header[1] > maxPackageSize)
						return 2; // The package is too big
					if(m_bufSize < m_header[1])
					{
						m_bufSize = std::max(m_header[1], std::min(maxBufSize, 2 * m_bufSize));
						delete[] m_pBuf;
						m_pBuf = new char[m_bufSize];
					}
					else if(m_bufSize > maxBufSize && m_header[1] < maxBufSize)
					{
						m_bufSize = m_header[1];
						delete[] m_pBuf;
						m_pBuf = new char[m_bufSize];
					}
					return receive(maxBufSize, maxPackageSize); // do it again to get the body of the package
				}
				else
				{
					m_headerBytes = 0;
					m_payloadBytes = 0;
					return 3; // The header is incorrect
				}
			}
			else
				return 0; // Nothing bad happened
		}
		else if(bytesReceived == 0)
			return 1; // The other end disconnected
		else
		{
#ifdef WINDOWS
			throw Ex("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
			throw Ex("Error calling recv: ", strerror(errno));
#endif
			return 1; // The other end disconnected
		}
	}
	else
	{
		// Receive the payload
		ssize_t bytesReceived = recv(m_sock, m_pBuf + m_payloadBytes, m_header[1] - m_payloadBytes, 0);
		if(bytesReceived > 0) // if we successfully received something...
		{
			m_payloadBytes += (unsigned int)bytesReceived;
			if(m_payloadBytes >= m_header[1])
			{
				m_q.push(GPackageConnectionBuf(m_pBuf, m_bufSize, m_payloadBytes));
				m_pBuf = NULL;
				m_bufSize = 0;
				m_payloadBytes = 0;
				m_headerBytes = 0;
			}
			return 0; // nothing bad happened
		}
		else if(bytesReceived == 0)
			return 1; // The other end disconnected
		else
		{
#ifdef WINDOWS
			throw Ex("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
			throw Ex("Error calling recv: ", strerror(errno));
#endif
			return 1; // The other end disconnected
		}
	}
}

char* GPackageConnection::next(size_t* pOutSize)
{
	if(m_q.size() == 0)
		return NULL;
	GPackageConnectionBuf& package = m_q.front();
	if(m_bufSize == 0)
	{
		m_pBuf = package.m_pBuf;
		m_bufSize = package.m_bufSize;
		*pOutSize = package.m_dataSize;
		m_q.pop();
		return m_pBuf;
	}
	else
	{
		delete[] m_pCondemned;
		m_pCondemned = package.m_pBuf;
		*pOutSize = package.m_dataSize;
		m_q.pop();
		return m_pCondemned;
	}
}




















GPackageClient::GPackageClient()
: m_conn(INVALID_SOCKET), m_maxBufSize(8192), m_maxPackageSize(0x1000000)
{
}

GPackageClient::~GPackageClient()
{
}

void GPackageClient::disconnect()
{
	if(m_conn.socket() == INVALID_SOCKET)
		return;
	onDisconnect();
	GSocket_closeSocket(m_conn.socket());
	m_conn.setSocket(INVALID_SOCKET);
}

void GPackageClient::connect(const char* addr, unsigned short port, int timeoutSecs)
{
	disconnect();
	m_conn.setSocket(GSocket_connect(addr, port, timeoutSecs));
}

// virtual
void GPackageClient::pump()
{
	int status = m_conn.receive(m_maxBufSize, m_maxPackageSize);
	if(status)
	{
		if(status == 1)
		{
			disconnect();
			return;
		}
		else if(status == 2)
			onReceiveBadData("Package too big");
		else if(status == 3)
			onReceiveBadData("Breach of protocol");
	}
	GThread::sleep(0);
}

void GPackageClient::send(const char* buf, size_t len)
{
	unsigned int header[2];
	header[0] = MAGIC_VALUE;
	header[1] = (unsigned int)len;
	char* pH = (char*)header;
	size_t hl = 2 * sizeof(unsigned int);
	try
	{
		while(hl > 0)
		{
			size_t bytesSent = GSocket_send(m_conn.socket(), pH, hl);
			if(bytesSent > 0)
			{
				pH += bytesSent;
				hl -= bytesSent;
			}
			else
				pump();
		}
		while(len > 0)
		{
			size_t bytesSent = GSocket_send(m_conn.socket(), buf, len);
			if(bytesSent > 0)
			{
				buf += bytesSent;
				len -= bytesSent;
			}
			else
				pump();
		}
	}
	catch(const std::exception& e)
	{
		disconnect();
		throw e;
	}
}

char* GPackageClient::receive(size_t* pLen)
{
	int status = m_conn.receive(m_maxBufSize, m_maxPackageSize);
	if(status)
	{
		if(status == 1)
			disconnect();
		else if(status == 2)
			onReceiveBadData("Package too big");
		else if(status == 3)
			onReceiveBadData("Breach of protocol");
		return NULL;
	}
	else
		return m_conn.next(pLen);
}








GPackageServer::GPackageServer(unsigned short port)
: m_maxBufSize(8192), m_maxPackageSize(0x1000000)
{
	GSocket_init();
	m_sock = socket(AF_INET, SOCK_STREAM, 0); // use SOCK_DGRAM for UDP

	// Tell the socket that it's okay to reuse an old crashed socket that hasn't timed out yet
	int flag = 1;
	setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(flag));

	// Bind the socket to the port
	SOCKADDR_IN sHostAddrIn;
	memset(&sHostAddrIn, '\0', sizeof(SOCKADDR_IN));
	sHostAddrIn.sin_family = AF_INET;
	sHostAddrIn.sin_port = htons(port);
	sHostAddrIn.sin_addr.s_addr = htonl(INADDR_ANY);
	if(bind(m_sock, (struct sockaddr*)&sHostAddrIn, sizeof(SOCKADDR)) != 0)
	{
#ifdef WINDOWS
		throw Ex("Failed to bind to port ", to_str(port), ": ", winstrerror(WSAGetLastError()));
#else
		throw Ex("Failed to bind to port ", to_str(port), ": ", strerror(errno));
#endif
	}

	// Start listening for connections
	if(listen(m_sock, SOMAXCONN) != 0)
#ifdef WINDOWS
		throw Ex("Failed to listen on the socket: ", winstrerror(WSAGetLastError()));
#else
		throw Ex("Failed to listen on the socket: ", strerror(errno));
#endif
}

GPackageServer::~GPackageServer()
{
	while(m_socks.size() > 0)
		disconnect(*m_socks.begin());
	GSocket_closeSocket(m_sock);
}

void GPackageServer::disconnect(GPackageConnection* pConn)
{
	onDisconnect(pConn);
	GSocket_closeSocket(pConn->socket());
	m_socks.erase(pConn);
	delete(pConn);
}

// virtual
void GPackageServer::pump(GPackageConnection* pConn)
{
	int status = pConn->receive(m_maxBufSize, m_maxPackageSize);
	if(status)
	{
		if(status == 2)
			onReceiveBadData("Package too big");
		else if(status == 3)
			onReceiveBadData("Breach of protocol");
		disconnect(pConn);
	}
	GThread::sleep(0);
}

void GPackageServer::checkForNewConnections()
{
	if(!GSocket_isReady(m_sock))
		return;

	// Accept the connection
	SOCKADDR_IN sHostAddrIn;
	socklen_t nStructSize = sizeof(struct sockaddr);
	SOCKET s = accept(m_sock, (struct sockaddr*)&sHostAddrIn, &nStructSize);
	if(s < 0)
	{
#ifdef WIN32
		if(WSAGetLastError() == WSAEWOULDBLOCK) // no connections are ready to be accepted
			return;
#else
		if(errno == EAGAIN) // no connections are ready to be accepted
			return;
#endif
		string s2 = "Received bad data while trying to accept a connection: ";
#ifdef WINDOWS
		s2 += winstrerror(WSAGetLastError());
#else
		s2 += strerror(errno);
#endif
		onReceiveBadData(s2.c_str());
		return;
	}
	GSocket_setSocketMode(s, false);
	m_socks.insert(makeConnection(s));
}

void GPackageServer::send(const char* buf, size_t len, GPackageConnection* pConn)
{
	unsigned int header[2];
	header[0] = MAGIC_VALUE;
	header[1] = (unsigned int)len;
	char* pH = (char*)header;
	size_t hl = 2 * sizeof(unsigned int);
	try
	{
		while(hl > 0)
		{
			size_t bytesSent = GSocket_send(pConn->socket(), pH, hl);
			if(bytesSent > 0)
			{
				pH += bytesSent;
				hl -= bytesSent;
			}
			else
				pump(pConn);
		}
		while(len > 0)
		{
			size_t bytesSent = GSocket_send(pConn->socket(), buf, len);
			if(bytesSent > 0)
			{
				buf += bytesSent;
				len -= bytesSent;
			}
			else
				pump(pConn);
		}
	}
	catch(const std::exception& e)
	{
		onReceiveBadData(e.what());
	}
}

char* GPackageServer::receive(size_t* pOutLen, GPackageConnection** pOutConn)
{
	checkForNewConnections();
	for(set<GPackageConnection*>::iterator it = m_socks.begin(); it != m_socks.end(); it++)
	{
		GPackageConnection* pConn = *it;
		int status = pConn->receive(m_maxBufSize, m_maxPackageSize);
		if(status)
		{
			if(status == 2)
				onReceiveBadData("Package too big");
			else if(status == 3)
				onReceiveBadData("Breach of protocol");
			disconnect(pConn);
		}
		else
		{
			char* pPackage = pConn->next(pOutLen);
			if(pPackage)
			{
				*pOutConn = pConn;
				return pPackage;
			}
		}
	}
	return NULL;
}


#define TEST_PORT 7251
#define CLIENT_COUNT 5
#define TEST_LEN 1000
#define MAX_PACKET_LEN 257

void GPackageServer_serial_test()
{
	GPackageServer server(TEST_PORT);
	GPackageClient clients[CLIENT_COUNT];
	for(size_t i = 0; i < CLIENT_COUNT; i++)
		clients[i].connect("localhost", TEST_PORT, 5);
	std::queue<size_t> q[CLIENT_COUNT];
	GRand randMaster(0);
	GRand randData(1234);
	char buf[MAX_PACKET_LEN];
	size_t sends = 0;
	size_t receives = 0;
	size_t bounces = 0;
	size_t iters = 0;
	while(receives < TEST_LEN)
	{
		size_t turn = size_t(randMaster.next(3 * CLIENT_COUNT));
		if(turn < CLIENT_COUNT)
		{
			if(sends < TEST_LEN && sends < receives + 50)
			{
				// The client sends a random packet of random size
				size_t len = (size_t)randMaster.next(MAX_PACKET_LEN - 5) + 5;
				size_t seed = (size_t)randMaster.next(1000000);
				randData.setSeed(seed);
				q[turn].push(len);
				q[turn].push(seed);
				char* pB = buf;
				for(size_t i = 0; i < len; i++)
					*(pB++) = (char)randData.next();
//cout << "Client " << to_str(turn) << " sent seed=" << to_str(seed) << ", len=" << len << ", data=" << to_str((int)buf[0]) << to_str((int)buf[1]) << to_str((int)buf[2]) << "\n"; cout.flush();
				clients[turn].send(buf, len);
				sends++;
			}
			else
				GThread::sleep(0);
		}
		else if(turn < CLIENT_COUNT + CLIENT_COUNT)
		{
			// The client receives the next packet and checks it
			turn -= CLIENT_COUNT;
			size_t len;
			char* pPackage = clients[turn].receive(&len);
			if(pPackage)
			{
				if(q[turn].size() == 0)
					throw Ex("unexpected package");
				size_t ll = q[turn].front();
				q[turn].pop();
				if(len != ll)
					throw Ex("incorrect package size");
				size_t seed = q[turn].front();
				q[turn].pop();
				randData.setSeed(seed);
				char* pB = pPackage;
//cout << "Client " << to_str(turn) << " received seed=" << to_str(seed) << ", len=" << len << ", data=" << to_str((int)pB[0]) << to_str((int)pB[1]) << to_str((int)pB[2]) << "\n"; cout.flush();
				for(size_t i = 0; i < len; i++)
				{
					if(*(pB++) != (char)randData.next())
						throw Ex("package corruption");
				}
				receives++;
			}
			else
				GThread::sleep(0);
		}
		else
		{
			size_t len;
			GPackageConnection* pConn;
			char* pPackage = server.receive(&len, &pConn);
			if(pPackage)
			{
//cout << "Server bounced " << to_str((int)pPackage[0]) << to_str((int)pPackage[1]) << to_str((int)pPackage[2]) << "\n"; cout.flush();
				server.send(pPackage, len, pConn);
				bounces++;
			}
			else
				GThread::sleep(0);
		}
		iters++;
	}
	if(sends != TEST_LEN || receives != TEST_LEN)
		throw Ex("something is amiss");
}

void GPackageServer::test()
{
	GPackageServer_serial_test();
	//GPackageServer_threaded_test();
}





void GDomClient::send(GDomNode* pNode)
{
	std::ostringstream os;
	m_doc.setRoot(pNode);
	m_doc.writeJson(os);
	string s = os.str();
	GPackageClient::send(s.c_str(), s.length());
}

const GDomNode* GDomClient::receive()
{
	m_doc.clear();
	size_t len;
	char* pPackage = GPackageClient::receive(&len);
	if(pPackage)
	{
		m_doc.parseJson(pPackage, len);
		return m_doc.root();
	}
	else
		return NULL;
}





void GDomServer::send(GDomNode* pNode, GPackageConnection* pConn)
{
	std::ostringstream os;
	m_doc.setRoot(pNode);
	m_doc.writeJson(os);
	string s = os.str();
	GPackageServer::send(s.c_str(), s.length(), pConn);
}

const GDomNode* GDomServer::receive(GPackageConnection** pOutConn)
{
	m_doc.clear();
	size_t len;
	char* pPackage = GPackageServer::receive(&len, pOutConn);
	if(pPackage)
	{
		m_doc.parseJson(pPackage, len);
		return m_doc.root();
	}
	else
		return NULL;
}




} // namespace GClasses
