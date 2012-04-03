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
#include "GDom.h"

#ifndef WINDOWS
#	define SOCKET int
#	define INVALID_SOCKET -1
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
#include <set>
#include <queue>
#include <string.h>

namespace GClasses {

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

	/// Sets the socket for this connection
	void setSocket(SOCKET sock) { m_sock = sock; }

	/// Returns the client's IP address for this connection.
	/// (You can use inet_ntoa to convert the value this returns to a string.)
	in_addr ipAddr();
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

	/// Obtains the name of this host.
	static void hostName(char* buf, size_t len);

	/// Returns a reference to the current set of connections
	std::set<GTCPConnection*>& connections() { return m_socks; }

	/// Accept any new incoming connections
	void checkForNewConnections();

protected:
	/// This is called just before a new connection is accepted. It
	/// returns a pointer to a new GTCPConnection object to
	/// associate with this connection. (The connection, however, isn't yet fully
	/// established, so it might cause an error if you send something to the
	/// client in an overload of this method.) If you want to associate
	/// some data with each connection, you can overload this method to
	/// create a custom object.
	virtual GTCPConnection* makeConnection(SOCKET s) { return new GTCPConnection(s); }

	/// This is called when a connection is first known to have disconnected.
	virtual void onDisconnect(GTCPConnection* pConn) {}

	/// This is called when a client sends some bad data.
	virtual void onReceiveBadData(const char* message) {}
};


/// This is a helper class used by GPackageConnection.
class GPackageConnectionBuf
{
public:
	char* m_pBuf;
	unsigned int m_bufSize;
	unsigned int m_dataSize;

	GPackageConnectionBuf(char* pBuf, unsigned int bufSize, unsigned int dataSize)
	: m_pBuf(pBuf), m_bufSize(bufSize), m_dataSize(dataSize)
	{
	}

	GPackageConnectionBuf(const GPackageConnectionBuf& that)
	: m_pBuf(that.m_pBuf), m_bufSize(that.m_bufSize), m_dataSize(that.m_dataSize)
	{
	}
};


/// This is a helper class used by GPackageServer. If you implement a custom
/// connection object for a sub-class of GPackageServer, then it should inherrit
/// from this class.
class GPackageConnection : public GTCPConnection
{
public:
	std::queue<GPackageConnectionBuf> m_q;
	char* m_pCondemned;
	char* m_pBuf;
	unsigned int m_header[2];
	unsigned int m_headerBytes;
	unsigned int m_payloadBytes;
	unsigned int m_bufSize;

	GPackageConnection(SOCKET sock)
	: GTCPConnection(sock), m_pCondemned(NULL), m_pBuf(NULL), m_headerBytes(0), m_payloadBytes(0), m_bufSize(0)
	{
	}

	virtual ~GPackageConnection()
	{
		delete[] m_pCondemned;
		delete[] m_pBuf;
	}

	/// Receives any available incoming data and adds it to the queue when a complete package is received.
	/// Returns 0 if no errors occur (whether or not a full package was received).
	/// Returns 1 if the other end disconnected. Returns 2 if the other end tried to send a package
	/// that was too big. Returns 3 if the other end breached protocol by sending a bad header.
	int receive(unsigned int maxBufSize, unsigned int maxPackageSize);

	/// Returns the next ready package. Returns NULL if no complete package is ready.
	char* next(size_t* pOutSize);
};


/// This class abstracts a client that speaks a home-made protocol that
/// guarantees packages will arrive in the same order and size as when
/// they were sent. This protocol is a simple layer on top of TCP.
class GPackageClient
{
protected:
	GPackageConnection m_conn;
	unsigned int m_maxBufSize;
	unsigned int m_maxPackageSize;

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

	/// Connect to a server. Throws an exception if it fails to connect within the
	/// specified timout period.
	void connect(const char* addr, unsigned short port, int timeoutSecs = 20);

	/// Disconnect from the server
	void disconnect();

	/// Receives any pending messages into an internal buffer (to unblock the
	/// server, in case its send buffer is full.)
	virtual void pump();

protected:
	/// This method is called if the peer sends data that does
	/// not follow the expected protocol.
	virtual void onReceiveBadData(const char* message) {}

	/// This is called when the connection is first known to have disconnected.
	virtual void onDisconnect() {}

};


/// This class abstracts a server that speaks a home-made protocol that
/// guarantees packages will arrive in the same order and size as when
/// they were sent. This protocol is a simple layer on top of TCP.
class GPackageServer
{
protected:
	SOCKET m_sock; // used to listen for incoming connections
	std::set<GPackageConnection*> m_socks; // used to communicate with each connected client
	unsigned int m_maxBufSize;
	unsigned int m_maxPackageSize;

public:
	GPackageServer(unsigned short port);
	virtual ~GPackageServer();

	/// Send a package, which guarantees to arrive in the
	/// same order and size as it was sent.
	void send(const char* buf, size_t len, GPackageConnection* pConn);

	/// Receives the next available package. (The order and size is
	/// guaranteed to arrive the same as when it was sent.)
	/// NULL is returned if no package is ready in its entirety.
	/// The returned value is a pointer to an internal buffer, the
	/// contents of which is only valid until the next time receive
	/// is called.
	char* receive(size_t* pOutLen, GPackageConnection** pOutConn);

	/// Disconnect from the specified client.
	void disconnect(GPackageConnection* pConn);

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

	/// Receives any pending messages into an internal buffer (to unblock the
	/// client, in case its send buffer is full.)
	virtual void pump(GPackageConnection* pConn);

	/// Returns a reference to the current set of connections
	std::set<GPackageConnection*>& connections() { return m_socks; }

	/// Accept any new incoming connections
	void checkForNewConnections();

protected:
	/// This is called just before a new connection is accepted. It
	/// returns a pointer to a new GPackageConnection object to
	/// associate with this connection. (The connection, however, isn't yet fully
	/// established, so it might cause an error if you send something to the
	/// client in an overload of this method.) If you want to associate
	/// some data with each connection, you can overload this method to
	/// create a custom object.
	virtual GPackageConnection* makeConnection(SOCKET s) { return new GPackageConnection(s); }

	/// This is called when a connection is first known to have disconnected.
	virtual void onDisconnect(GTCPConnection* pConn) {}

	/// This is called when a client sends some bad data.
	virtual void onReceiveBadData(const char* message) {}
};



/// This is a socket client that sends and receives DOM nodes
class GDomClient : public GPackageClient
{
protected:
	GDom m_doc;

public:
	GDomClient() : GPackageClient() {}
	virtual ~GDomClient() {}

	/// Send the specified DOM node.
	void send(GDomNode* pNode);

	/// Receive the next available DOM node, or NULL if none are ready.
	GDomNode* receive();
};


/// This is a socket server that sends and receives DOM nodes
class GDomServer : public GPackageServer
{
protected:
	GDom m_doc;

public:
	GDomServer(unsigned int port) : GPackageServer(port) {}
	virtual ~GDomServer() {}

	/// Send the specified DOM node.
	void send(GDomNode* pNode, GPackageConnection* pConn);

	/// Receive the next available DOM node, or NULL if none are ready.
	GDomNode* receive(GPackageConnection** pOutConn);

};

} // namespace GClasses

#endif // __GSOCKET_H__
