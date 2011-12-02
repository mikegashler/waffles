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
		ThrowError("select failed: ", winstrerror(WSAGetLastError()));
#else
		ThrowError("select failed: ", strerror(errno));
#endif
	}
	return ret == 0 ? false : true;
}

size_t GSocket_bytesReady(SOCKET s)
{
	unsigned long bytesReadyToRead = 0;
#ifdef WINDOWS
	GWindows::yield(); // This is necessary because incoming packets go through the Windows message pump
	if(ioctlsocket(s, FIONREAD, &bytesReadyToRead) != 0)
		ThrowError("ioctlsocket failed: ", winstrerror(WSAGetLastError()));
#else
	if(ioctl(s, FIONREAD, &bytesReadyToRead) != 0)
		ThrowError("ioctl failed: ", strerror(errno));
#endif
	return bytesReadyToRead;
}

void GSocket_setSocketMode(SOCKET s, bool blocking)
{
	unsigned long ulMode = blocking ? 0 : 1;
#ifdef WINDOWS
	if(ioctlsocket(s, FIONBIO, &ulMode) != 0)
#else
	if(ioctl(s, FIONBIO, &ulMode) != 0)
#endif
		ThrowError("Error changing the mode of a socket");
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
		ThrowError("getpeername failed: ", winstrerror(WSAGetLastError()));
#else
		ThrowError("getpeername failed: ", strerror(errno));
#endif
	}
	if(sAddr.sa_family != AF_INET)
		ThrowError("Error, expected family to be AF_INET");
	SOCKADDR_IN* pInfo = (SOCKADDR_IN*)&sAddr;
	return pInfo->sin_addr;
}

void GSocket_send(SOCKET s, const char* buf, size_t len)
{
	while(true)
	{
		ssize_t bytesSent = ::send(s, buf, len, 0);
		if(bytesSent > 0)
		{
			buf += bytesSent;
			len -= bytesSent;
			if(len == 0)
				return;
		}
		else
		{
#ifdef WINDOWS
			int err = WSAGetLastError();
			if(bytesSent == 0 || err == WSAEWOULDBLOCK)
				GThread::sleep(0);
			else
				ThrowError("Error sending in GTCPClient::send: ", winstrerror(err));
#else
			if(bytesSent == 0 || errno == EWOULDBLOCK)
				GThread::sleep(0);
			else
				ThrowError("Error sending in GTCPClient::send: ", strerror(errno));
#endif
		}
	}
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
		ThrowError("Failed to find a usable WinSock DLL");

	// Confirm that the WinSock DLL supports at least 2.2.
	if ( LOBYTE( wsaData.wVersion ) != 1 ||
			HIBYTE( wsaData.wVersion ) != 1 )
	{
		int n1 = LOBYTE( wsaData.wVersion );
		int n2 = HIBYTE( wsaData.wVersion );
		WSACleanup();
		ThrowError("Found a Winsock DLL, but it only supports an older version. It needs to support version 2.2");
	}
#endif
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
		ThrowError(gai_strerror(error));
	m_sock = INVALID_SOCKET;
	for(res = res0; res; res = res->ai_next)
	{
		m_sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
		if(m_sock < 0)
			continue;
		struct timeval timeout;
		fd_set socketSet;
		GSocket_setSocketMode(m_sock, false);

		// Trying to connect with timeout
		if(::connect(m_sock, res->ai_addr, (int)res->ai_addrlen) < 0)
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
				FD_SET(m_sock, &socketSet);
#ifdef WINDOWS
				int res = select((int)m_sock + 1, NULL, &socketSet, NULL, &timeout);
				if(res < 0 && WSAGetLastError() != WSAEINTR)
#else
				int res = select(m_sock + 1, NULL, &socketSet, NULL, &timeout);
				if(res < 0 && errno != EINTR)
#endif
					ThrowError("Failed to connect to ", addr, " on port ", to_str(port));
				else if(res > 0)
				{
					// Socket selected for write
					socklen_t lon = sizeof(int);
					int valopt;
					if(getsockopt(m_sock, SOL_SOCKET, SO_ERROR, (char*)(&valopt), &lon) < 0)
						ThrowError("getsockopt failed");
					if(valopt)
					{
						GSocket_closeSocket(m_sock);
						m_sock = INVALID_SOCKET;
						continue;
					}

					// Got a connection!
					break;
				}
				else
				{
					// Timeout exceeded
					GSocket_closeSocket(m_sock);
					m_sock = INVALID_SOCKET;
					continue;
				}
			}
			else
			{
				// Failed to connect to this address
				GSocket_closeSocket(m_sock);
				m_sock = INVALID_SOCKET;
				continue;
			}
		}
	}
	freeaddrinfo(res0);
	if(m_sock == INVALID_SOCKET)
		ThrowError("Failed to connect to ", addr, " on port ", to_str(port));
}

void GTCPClient::send(const char* buf, size_t len)
{
	try
	{
		GSocket_send(m_sock, buf, len);
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
		ssize_t bytesReceived = recv(m_sock, buf, len, 0);
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
			ThrowError("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
			ThrowError("Error calling recv: ", strerror(errno));
#endif
		}
	}
	return 0;
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
		ThrowError("Failed to bind to port ", to_str(port), ": ", winstrerror(WSAGetLastError()));
#else
		ThrowError("Failed to bind to port ", to_str(port), ": ", strerror(errno));
#endif
	}

	// Start listening for connections
	if(listen(m_sock, SOMAXCONN) != 0)
#ifdef WINDOWS
		ThrowError("Failed to listen on the socket: ", winstrerror(WSAGetLastError()));
#else
		ThrowError("Failed to listen on the socket: ", strerror(errno));
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
#ifdef WIN32
		if(WSAGetLastError() == WSAEWOULDBLOCK) // no connections are ready to be accepted
			return;
#else
		if(errno == EAGAIN) // no connections are ready to be accepted
			return;
#endif
		string s = "Received bad data while trying to accept a connection: ";
#ifdef WINDOWS
		s += winstrerror(WSAGetLastError());
#else
		s += strerror(errno);
#endif
		onReceiveBadData(s.c_str());
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
			ssize_t bytesReceived = recv(pConn->socket(), buf, len, 0);
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
				ThrowError("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
				ThrowError("Error calling recv: ", strerror(errno));
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
		GSocket_send(pConn->socket(), buf, len);
	}
	catch(const std::exception& e)
	{
		disconnect(pConn);
		throw e;
	}
}

in_addr GTCPServer::ipAddr(GTCPConnection* pConn)
{
	return GSocket_ipAddr(pConn->socket());
}

// static
void GTCPServer::hostName(char* buf, size_t len)
{
	if(gethostname(buf, len) == SOCKET_ERROR)
		ThrowError("failed to get host noame");
}

/*
// static
in_addr GTCPServer::hostNameToIPAddress(char* szHostName)
{
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
*/



#define MAGIC_VALUE 0x0b57ac1e

GPackageClient::GPackageClient()
: GTCPClient(), m_headerBytes(0), m_payloadBytes(0), m_bufSize(0), m_maxBufSize(8192), m_maxPackageSize(0x1000000), m_pBuf(NULL)
{
}

GPackageClient::~GPackageClient()
{
	delete[] m_pBuf;
}

void GPackageClient::send(const char* buf, size_t len)
{
	unsigned int header[2];
	header[0] = MAGIC_VALUE;
	header[1] = (unsigned int)len;
	GTCPClient::send((char*)header, 2 * sizeof(unsigned int));
	GTCPClient::send(buf, len);
}

char* GPackageClient::receive(size_t* pLen)
{
	if(m_headerBytes < 2 * sizeof(unsigned int))
	{
		m_headerBytes += GTCPClient::receive(((char*)m_header) + m_headerBytes, 2 * sizeof(unsigned int) - m_headerBytes);
		if(m_headerBytes >= 2 * sizeof(unsigned int))
		{
			if(m_header[0] == MAGIC_VALUE)
			{
				if(m_header[1] > m_maxPackageSize)
					onReceiveBadData("Package too big");
				if(m_bufSize < m_header[1])
				{
					m_bufSize = std::max(m_header[1], std::min(m_maxBufSize, 2 * m_bufSize));
					delete[] m_pBuf;
					m_pBuf = new char[m_bufSize];
				}
				else if(m_bufSize > m_maxBufSize && m_header[1] < m_maxBufSize)
				{
					m_bufSize = m_header[1];
					delete[] m_pBuf;
					m_pBuf = new char[m_bufSize];
				}
				return receive(pLen);
			}
			else
			{
				onReceiveBadData("Breach of protocol");
				m_headerBytes = 0;
				m_payloadBytes = 0;
				return NULL;
			}
		}
		else
			return NULL;
	}
	else
	{
		m_payloadBytes += GTCPClient::receive(m_pBuf + m_payloadBytes, m_header[1] - m_payloadBytes);
		if(m_payloadBytes >= m_header[1])
		{
			m_headerBytes = 0;
			*pLen = m_payloadBytes;
			m_payloadBytes = 0;
			return m_pBuf;
		}
		else
			return NULL;
	}
}





/// This is a helper class used by GPackageServer
class GPackageConnection : public GTCPConnection
{
public:
	unsigned int m_headerBytes;
	unsigned int m_payloadBytes;
	unsigned int m_bufSize;
	unsigned int m_header[2];
	char* m_pBuf;

	GPackageConnection(SOCKET sock)
	: GTCPConnection(sock), m_headerBytes(0), m_payloadBytes(0), m_bufSize(0), m_pBuf(NULL)
	{
	}

	virtual ~GPackageConnection()
	{
		delete[] m_pBuf;
	}
};

GPackageServer::GPackageServer(unsigned short port)
: GTCPServer(port), m_maxBufSize(8192), m_maxPackageSize(0x1000000)
{
}

GPackageServer::~GPackageServer()
{
}

// virtual
GTCPConnection* GPackageServer::makeConnection(SOCKET s)
{
	return new GPackageConnection(s);
}

void GPackageServer::send(const char* buf, size_t len, GTCPConnection* pConn)
{
	unsigned int header[2];
	header[0] = MAGIC_VALUE;
	header[1] = (unsigned int)len;
	GTCPServer::send((char*)header, 2 * sizeof(unsigned int), pConn);
	GTCPServer::send(buf, len, pConn);
}

char* GPackageServer::receive(size_t* pOutLen, GTCPConnection** pOutConn)
{
	checkForNewConnections();
	for(set<GTCPConnection*>::iterator it = m_socks.begin(); it != m_socks.end(); )
	{
		GPackageConnection* pConn = (GPackageConnection*)*it;
		bool doagain = false;
		if(GSocket_bytesReady(pConn->socket()) > 0) // if there is something ready to receive...
		{
			if(pConn->m_headerBytes < 2 * sizeof(unsigned int)) // if the header is still incomplete...
			{
				// Receive the header
				ssize_t bytesReceived = recv(pConn->socket(), ((char*)pConn->m_header) + pConn->m_headerBytes, 2 * sizeof(unsigned int) - pConn->m_headerBytes, 0);
				if(bytesReceived > 0) // if we successfully received something...
				{
					pConn->m_headerBytes += bytesReceived;
					if(pConn->m_headerBytes >= 2 * sizeof(unsigned int))
					{
						if(pConn->m_header[0] == MAGIC_VALUE)
						{
							if(pConn->m_header[1] > m_maxPackageSize)
								onReceiveBadData("Package too big");
							if(pConn->m_bufSize < pConn->m_header[1])
							{
								pConn->m_bufSize = std::max(pConn->m_header[1], std::min(m_maxBufSize, 2 * pConn->m_bufSize));
								delete[] pConn->m_pBuf;
								pConn->m_pBuf = new char[pConn->m_bufSize];
							}
							else if(pConn->m_bufSize > m_maxBufSize && pConn->m_header[1] < m_maxBufSize)
							{
								pConn->m_bufSize = pConn->m_header[1];
								delete[] pConn->m_pBuf;
								pConn->m_pBuf = new char[pConn->m_bufSize];
							}
							doagain = true; // let's visit this connection again
						}
						else
						{
							onReceiveBadData("breach of protocol");
							pConn->m_headerBytes = 0;
							pConn->m_payloadBytes = 0;
						}
					}
				}
				else if(bytesReceived == 0) // if the client disconnected...
				{
					disconnect(pConn);
					return receive(pOutLen, pOutConn); // Recurse since the previous operation will invalidate the iterator
				}
				else
				{
#ifdef WINDOWS
					ThrowError("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
					ThrowError("Error calling recv: ", strerror(errno));
#endif
				}
			}
			else
			{
				// Receive the payload
				ssize_t bytesReceived = recv(pConn->socket(), pConn->m_pBuf + pConn->m_payloadBytes, pConn->m_header[1] - pConn->m_payloadBytes, 0);
				if(bytesReceived > 0) // if we successfully received something...
				{
					pConn->m_payloadBytes += bytesReceived;
					if(pConn->m_payloadBytes >= pConn->m_header[1])
					{
						*pOutLen = pConn->m_payloadBytes;
						*pOutConn = pConn;
						pConn->m_headerBytes = 0;
						pConn->m_payloadBytes = 0;
						return pConn->m_pBuf;
					}
				}
				else if(bytesReceived == 0) // if the client disconnected...
				{
					disconnect(pConn);
					return receive(pOutLen, pOutConn); // Recurse since the previous operation will invalidate the iterator
				}
				else
				{
#ifdef WINDOWS
					ThrowError("Error calling recv: ", winstrerror(WSAGetLastError()));
#else
					ThrowError("Error calling recv: ", strerror(errno));
#endif
				}
			}
		}
		if(!doagain) // if we just completed a header, there is a good chance the payload may be ready too
			it++;
	}
	return NULL;
}

#ifndef NO_TEST_CODE
#define TEST_PORT 7251
#define CLIENT_COUNT 5
#define TEST_LEN 5000
#define MAX_PACKET_LEN 2345

class GPackageServer_test_struct
{
public:
	volatile bool serverIsUp;
	volatile bool keepRunning;
	volatile bool running;

	GPackageServer_test_struct()
	: serverIsUp(false), keepRunning(true), running(true)
	{
	}

	~GPackageServer_test_struct()
	{
		keepRunning = false;
		size_t timeout = 200;
		while(running && --timeout != 0)
			GThread::sleep(50);
		if(running)
			ThrowError("Failed to join server thread");
	}
};



unsigned int GPackageServer_test_server(void* pThis)
{
	GPackageServer_test_struct* pStruct = (GPackageServer_test_struct*)pThis;
	if(pStruct->serverIsUp)
	{
		pStruct->running = false;
		return 1;
	}
	GPackageServer server(TEST_PORT);
	pStruct->serverIsUp = true;
	size_t bounces = 0;
	double lastReceiveTime = GTime::seconds();
	while(bounces < TEST_LEN && pStruct->keepRunning)
	{
		size_t len;
		GTCPConnection* pConn;
		char* pPackage = server.receive(&len, &pConn);
		if(pPackage)
		{
			server.send(pPackage, len, pConn);
			bounces++;
			lastReceiveTime = GTime::seconds();
		}
		else
		{
			if(GTime::seconds() - lastReceiveTime > 10)
			{
				cerr << "\nServer aborting after " << bounces << "/" << TEST_LEN << " successful bounces due to inactivity.\n";
				break;
			}
			GThread::sleep(0);
		}
	}
	pStruct->running = false;
	return 0;
}

void GPackageServer::test()
{
	// Spawn a bounce server in another thread
	GPackageServer_test_struct str;
	GThread::spawnThread(GPackageServer_test_server, &str);
	size_t timeout = 200;
	while(!str.serverIsUp)
	{
		GThread::sleep(50);
		if(--timeout == 0)
			ThrowError("Failed to spawn server thread");
	}

	// Connect to it with several clients
	GPackageClient clients[CLIENT_COUNT];
	for(size_t i = 0; i < CLIENT_COUNT; i++)
		clients[i].connect("localhost", TEST_PORT, 5);
	std::queue<size_t> q[CLIENT_COUNT];
	GRand rand(0);
	char buf[MAX_PACKET_LEN];
	size_t sends = 0;
	size_t receives = 0;
	size_t iters = 0;
	while(receives < TEST_LEN)
	{
		size_t turn = size_t(rand.next(2 * CLIENT_COUNT));
		if(turn < CLIENT_COUNT)
		{
			if(sends < TEST_LEN && sends < receives + 30)
			{
				if(!str.running)
					ThrowError("The server aborted prematurely");

				// The client sends a random packet of random size
				size_t len = (size_t)rand.next(MAX_PACKET_LEN - 5) + 5;
				size_t seed = (size_t)rand.next(1000000);
				rand.setSeed(seed);
				q[turn].push(len);
				q[turn].push(seed);
				char* pB = buf;
				for(size_t i = 0; i < len; i++)
					*(pB++) = (char)rand.next();
				clients[turn].send(buf, len);
				sends++;
			}
		}
		else
		{
			// The client receives the next packet and checks it
			turn -= CLIENT_COUNT;
			size_t len;
			char* pPackage = clients[turn].receive(&len);
			if(pPackage)
			{
				if(q[turn].size() == 0)
					ThrowError("unexpected package");
				if(len != q[turn].front())
					ThrowError("incorrect package size");
				q[turn].pop();
				rand.setSeed(q[turn].front());
				q[turn].pop();
				char* pB = pPackage;
				for(size_t i = 0; i < len; i++)
				{
					if(*(pB++) != (char)rand.next())
						ThrowError("package corruption");
				}
				receives++;
			}
			else
				GThread::sleep(0);
		}
		iters++;
	}
	if(sends != TEST_LEN || receives != TEST_LEN)
		ThrowError("something is amiss");
}
#endif // !NO_TEST_CODE

} // namespace GClasses

