/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GSOCKET_H__
#define __GSOCKET_H__

#include "GError.h"
#include "GThread.h"

#ifndef WINDOWS
#	define SOCKET int
#	define INVALID_SOCKET -1
#endif

#ifdef WINDOWS
#	ifndef _MT
#		error ERROR: _MT not defined.  GSocket requires using the multithreaded libraries
#		define _MT
#	endif // !_MT
#endif


#ifdef WINDOWS
#	ifndef IPPROTO_IP
#		include <winsock2.h>
#	endif // IPPROTO_IP
#else
#	include <sys/types.h>
#	include <sys/socket.h>
#	include <netinet/in.h>
#endif
#include <vector>
#include <deque>
#include <set>
#include <string.h>

#ifndef WINDOWS
typedef struct sockaddr SOCKADDR;
typedef struct sockaddr_in SOCKADDR_IN;
typedef struct hostent HOSTENT;
#endif

namespace GClasses {

class GSpinLock;
class GSocketServerBuffer;
/*


class GSocketClientBase
{
protected:
	volatile bool m_bKeepListening;
	SOCKET m_s;
	THREAD_HANDLE m_hListenThread;
	bool m_bUDP;

public:
	GSocketClientBase(bool bUDP);
	virtual ~GSocketClientBase();

	static bool isIPAddress(const char* szHost);

	/// This returns the SOCKET
	SOCKET socketHandle();

	/// nTimeout is specified in seconds
	bool Connect(const char* szHost, unsigned short nPort, int nTimeout = 10);
	void Disconnect();

	bool Send(const unsigned char *pBuf, size_t nLen);

	/// This method is abstract because you need to implement something here
	virtual bool Receive(unsigned char *pBuf, size_t nLen) = 0; // Override me


	u_short myPort();
	struct in_addr myIPAddr();
	bool isConnected();
	char* myIPAddr(char* szBuff, size_t nBuffSize);
	char* myName(char* szBuff, size_t nBuffSize);
	u_short otherPort();
	struct in_addr otherIPAddr();
	char* otherIPAddr(char* szBuff, size_t nBuffSize);
	char* otherName(char* szBuff, size_t nBuffSize);

	void listen(); // Don't call this method directly

	/// This parses a URL into its parts
	static void parseURL(const char* szUrl, int* pnHostIndex, int* pnPortIndex, int* pnPathIndex, int* pnParamsIndex);

	/// Parses the parameter portion of a URL
	static int parseUrlParams(const char* szParams, int nMaxParams, char** pNames, int* pNameLengths, char** pValues, int* pValueLengths);

protected:
	bool goIntoHostMode(unsigned short nListenPort, int nMaxConnections);
	int firstAvailableSocketNumber();
	void joinListenThread();
	void joinListenThread(int nConnectionNumber);

	/// This method is empty. It's just here so you can override it.
	/// This is called when the connection is gracefully closed. (There is no built-in support
	/// for detecting ungraceful disconnects. This is a feature, not a bug, because it makes
	/// it robust to sporadic hardware. I recommend implementing a system where the
	/// server requires the client to send periodic heartbeat packets and you call Disconnect()
	/// if the responses don't come regularly.)
	virtual void onCloseConnection();
};








class GSocketServerBase
{
protected:
	SOCKET m_socketConnectionListener;
	std::vector<SOCKET> m_connections;
	GSpinLock* m_pConnectionsLock;
	fd_set m_socketSet; // structure used by select()
	THREAD_HANDLE m_hWorkerThread;
	volatile bool m_bKeepWorking;
	SOCKADDR_IN m_sHostAddrIn;
	bool m_bUDP;
	int m_nMaxConnections;
	char* m_szReceiveBuffer;

public:
	GSocketServerBase(bool bUDP, int nPort, int nMaxConnections);
	virtual ~GSocketServerBase();

	/// This returns the SOCKET
	SOCKET GetSocketHandle(int nConnectionNumber);

	void Disconnect(int nConnectionNumber);

	bool Send(const unsigned char *pBuf, size_t nLen, int nConnectionNumber);

	/// This method is abstract because you need to implement something here
	virtual bool Receive(unsigned char *pBuf, size_t nLen, int nConnectionNumber) = 0; // Override me

	bool IsConnected(int nConnectionNumber);

	/// Stops listening for new connections
	void DontAcceptAnyMoreConnections();

	/// If nConnectionNumber is less than 0, returns the address of the server
	char* GetIPAddr(char* szBuff, size_t nBuffSize, int nConnectionNumber);

	/// If nConnectionNumber is less than 0, returns the address of the server
	struct in_addr GetIPAddr(int nConnectionNumber);

	/// If nConnectionNumber is less than 0, returns the port of the server
	u_short GetPort(int nConnectionNumber);

	/// If nConnectionNumber is less than 0, returns the name of the server
	char* GetName(char* szBuff, size_t nBuffSize, int nConnectionNumber);

	void ServerWorker(); // Don't call this method directly

protected:
	void Init(bool bUDP, int nPort, int nMaxConnections);
	int GetFirstAvailableConnectionNumber();
	void JoinWorkerThread();

	/// This method is empty. It's just here so you can override it.
	/// This is called when the connection is gracefully closed. (There is no built-in support
	/// for detecting ungraceful disconnects. This is a feature, not a bug, because it makes
	/// it robust to sporadic hardware. To detect ungraceful disconnects, I recommend requiring
	/// the client to send periodic heartbeat packets and calling Disconnect() if they stop coming.)
	virtual void onCloseConnection(int nConnection);

	/// This method is empty. It's just here so you can override it.
	/// WARNING: the connection isn't fully open at the time this method is called,
	///          so don't send anything back to the client inside this callback
	virtual void onAcceptConnection(int nConnection);

	SOCKET RefreshSocketSet();
	int HandleNewConnection();
	void ReduceConnectionList();
};

// --------------------------------------------------------------------------

struct GEZSocketPacketHeader
{
	char tag[4];
	unsigned int nPayloadSize;
};


class GSocketMessage
{
protected:
	unsigned char* m_pMessage;
	size_t m_nMessageSize;
	int m_nConnection;

public:
	GSocketMessage(unsigned char* pMessage, size_t nMessageSize, int nConnection)
	{
		m_pMessage = new unsigned char[nMessageSize];
		memcpy(m_pMessage, pMessage, nMessageSize);
		m_nMessageSize = nMessageSize;
		m_nConnection = nConnection;
	}

	virtual ~GSocketMessage()
	{
		delete(m_pMessage);
	}

	const unsigned char* GetTheMessage() { return m_pMessage; }
	size_t GetMessageSize() { return m_nMessageSize; }
	int GetConnection() { return m_nConnection; }

	/// you must delete the buffer this returns
	unsigned char* TakeBuffer()
	{
		unsigned char* pMessage = m_pMessage;
		m_pMessage = NULL;
		return pMessage;
	}
};


// --------------------------------------------------------------------------


/// This class is designed to make network communication easy
class GSocketServer : public GSocketServerBase
{
protected:
	std::vector<GSocketServerBuffer*>* m_pBuffers;
	size_t m_nMaxPacketSize;
	std::deque<GSocketMessage*> m_messageQueue;
	GSpinLock* m_pMessageQueueLock;

	virtual bool Receive(unsigned char *pBuf, size_t len, int nConnectionNumber);
	void QueueMessage(unsigned char* pBuf, size_t nLen, int nConnectionNumber);

public:
	/// if nMaxPacketSize = 0, the socket will speak raw UDP or TCP.
	/// if nMaxPacketSize > 0, it will speak GSKT over TCP. (GSKT guarantees
	///          same-size delivery of packets, but has a maximum packet size.)
	GSocketServer(bool bUDP, size_t nMaxPacketSize, int nPort, int nMaxConnections);

	virtual ~GSocketServer();

	/// Send some data
	bool Send(const void* pBuf, size_t nLen, int nConnectionNumber);

	/// Concat two blobs and send as one packet
	bool Send2(const void* pBuf1, size_t nLen1, const void* pBuf2, size_t nLen2, int nConnectionNumber);

	/// Returns the number of messages waiting to be received
	size_t GetMessageCount();

	/// Receive the next message. (You are responsible to delete the buffer this returns)
	unsigned char* GetNextMessage(size_t* pnSize, int* pnOutConnectionNumber);
};

// --------------------------------------------------------------------------

/// This class is designed to make network communication easy
class GSocketClient : public GSocketClientBase
{
protected:
	unsigned char* m_pBuffer;
	size_t m_nBufferPos;
	size_t m_nMaxPacketSize;
	std::deque<GSocketMessage*> m_messageQueue;
	GSpinLock* m_pMessageQueueLock;

public:
	/// if nMaxPacketSize = 0, the socket will speak raw UDP or TCP.
	/// if nMaxPacketSize > 0, it will speak GSKT over TCP. (GSKT guarantees
	///          same-size delivery of packets, but has a maximum packet size.)
	GSocketClient(bool bUDP, size_t nMaxPacketSize);

	virtual ~GSocketClient();

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !NO_TEST_CODE

	/// Send some data
	bool Send(const void* pBuf, size_t nLen);
	
	/// Concat two blobs and send as one packet
	bool Send2(const void* pBuf1, size_t nLen1, const void* pBuf2, size_t nLen2);

	/// Returns the number of messages waiting to be received
	size_t GetMessageCount();

	/// Receive the next message. (You are responsible to delete the buffer this returns)
	unsigned char* GetNextMessage(size_t* pnSize);

protected:
	virtual bool Receive(unsigned char *pBuf, size_t len);
	void QueueMessage(unsigned char* pBuf, size_t nLen);
};
*/

/// This class is an abstraction of a TCP client socket connection
class GTCPClient
{
protected:
	SOCKET m_sock;

public:
	GTCPClient();
	virtual ~GTCPClient();

	/// Send some data. Throws an exception if the send fails for any reason (including
	/// common reasons, such as if the server has closed the connection), so it is generally
	/// a good idea to send within a try/catch block.
	void send(const char* buf, size_t len);

	/// This method receives any data that is ready to be received. It returns the
	/// number of bytes received. It immediately returns 0 if nothing is ready to
	/// be recevied. It also returns 0 if the server has closed the connection, so
	/// you should periodically call isConnected() to make sure the connection is
	/// still open.
	size_t receive(char* buf, size_t len);

	/// Connect to a server. Throws an exception if it fails to connect within the
	/// specified timout period.
	void connect(const char* addr, unsigned short port, int timeoutSecs = 20);

	/// Returns true iff the socket is connected
	bool isConnected();

	/// Disconnect from the server
	void disconnect();

protected:
	/// This is called when the connection is first known to have disconnected.
	virtual void onDisconnect() {}
};


/// This class is used by GTCPServer to represent a connection with one of
/// the clients. (If you want to associate some additional objects with each
/// connection, you can inherrit from this class, and overload GTCPServer::makeConnection
/// to return your own custom object.)
class GTCPConnection
{
protected:
	SOCKET m_sock;

public:
	GTCPConnection(SOCKET sock) : m_sock(sock) {}
	virtual ~GTCPConnection() {}

	/// Returns the socket associated with this connection
	SOCKET socket() { return m_sock; }
};


/// This class is an abstraction of a TCP server, which maintains a set of socket connections
class GTCPServer
{
protected:
	SOCKET m_sock; // used to listen for incoming connections
	std::set<GTCPConnection*> m_socks; // used to communicate with each connected client

public:
	GTCPServer(unsigned short port);
	virtual ~GTCPServer();

	/// Send some data to the specified client. Throws an exception if the send fails for any reason (including
	/// common reasons, such as if the client has closed the connection), so it is generally a good idea to send
	/// within a try/catch block.
	void send(const char* buf, size_t len, GTCPConnection* pConn);

	/// This method receives any data that is ready to be received. It returns the
	/// number of bytes received. It immediately returns 0 if nothing is ready to
	/// be recevied. The value at pOutConn will be set to indicate which client
	/// it received the data from.
	size_t receive(char* buf, size_t len, GTCPConnection** pOutConn);

	/// Disconnect from the specified client.
	void disconnect(GTCPConnection* pConn);

	/// Returns the client's IP address for the specified connection.
	in_addr ipAddr(GTCPConnection* pConn);

	/// Obtains the name of this host.
	static void hostName(char* buf, size_t len);

protected:
	/// This is called just before a new connection is accepted. It
	/// returns a pointer to a new GTCPConnection object to
	/// associate with this connection. (The connection, however, isn't yet fully
	/// established, so it might cause an error if you send something to the
	/// client in an overload of this method.)
	virtual GTCPConnection* makeConnection(SOCKET s) { return new GTCPConnection(s); }

	/// This is called when a connection is first known to have disconnected.
	virtual void onDisconnect(GTCPConnection* pConn) {}

	/// This is called when a client sends some bad data.
	virtual void onReceiveBadData(const char* message) {}

	/// Accept any new incoming connections
	void checkForNewConnections();
};


/// This class abstracts a client that speaks a home-made protocol that
/// guarantees packages will arrive in the same order and size as when
/// they were sent. This protocol is a simple layer on top of TCP.
class GPackageClient : public GTCPClient
{
protected:
	unsigned int m_header[2];
	unsigned int m_headerBytes;
	unsigned int m_payloadBytes;
	unsigned int m_bufSize;
	unsigned int m_maxBufSize;
	unsigned int m_maxPackageSize;
	char* m_pBuf;

public:
    GPackageClient();
	virtual ~GPackageClient();

	/// Send a package, which guarantees to arrive in the
	/// same order and size as it was sent.
	void send(const char* buf, size_t len);

	/// Receive the next available package. (This returns a
	/// pointer to an internal buffer, the contents of which only remain
	/// valid until the next time you call receive.)
	char* receive(size_t* pLen);

	/// Sets some internal values that guide how it reallocates the internal
	/// buffer. 'a' is the maximum buffer size to keep around. 'b' is the
	/// maximum size for the buffer ever. If a package bigger than 'b' is sent,
	/// an exception will be thrown. If a package bigger than 'a' is sent,
	/// then the buffer will be grown to that size, but it will be made small
	/// again the next time a package is received.
	void setMaxBufferSizes(size_t a, size_t b) { m_maxBufSize = (unsigned int)a; m_maxPackageSize = (unsigned int)b; }

protected:
	/// This method is called if the peer sends data that does
	/// not follow the expected protocol.
	virtual void onReceiveBadData(const char* message) {}
};


/// This class abstracts a server that speaks a home-made protocol that
/// guarantees packages will arrive in the same order and size as when
/// they were sent. This protocol is a simple layer on top of TCP.
class GPackageServer : public GTCPServer
{
protected:
	unsigned int m_maxBufSize;
	unsigned int m_maxPackageSize;

public:
	GPackageServer(unsigned short port);
	virtual ~GPackageServer();

	/// Send a package, which guarantees to arrive in the
	/// same order and size as it was sent.
	void send(const char* buf, size_t len, GTCPConnection* pConn);

	/// Receives the next available package. (The order and size is
	/// guaranteed to arrive the same as when it was sent.)
	/// NULL is returned if no package is ready in its entirety.
	/// The returned value is a pointer to an internal buffer, the
	/// contents of which is only valid until the next time receive
	/// is called.
	char* receive(size_t* pOutLen, GTCPConnection** pOutConn);

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !NO_TEST_CODE

	/// Sets some internal values that guide how it reallocates the internal
	/// buffer. 'a' is the maximum buffer size to keep around. 'b' is the
	/// maximum size for the buffer ever. If a package bigger than 'b' is sent,
	/// an exception will be thrown. If a package bigger than 'a' is sent,
	/// then the buffer will be grown to that size, but it will be made small
	/// again the next time a package is received.
	void setMaxBufferSizes(size_t a, size_t b) { m_maxBufSize = (unsigned int)a; m_maxPackageSize = (unsigned int)b; }

protected:
	/// See the comment for GTCPConnection::makeConnection.
	virtual GTCPConnection* makeConnection(SOCKET s);
};



} // namespace GClasses

#endif // __GSOCKET_H__
