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

#include "../GClasses/GError.h"
#include "../GClasses/GThread.h"

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
#include <string.h>

#ifndef WINDOWS
typedef struct sockaddr SOCKADDR;
typedef struct sockaddr_in SOCKADDR_IN;
typedef struct hostent HOSTENT;
#endif

namespace GClasses {

class GSpinLock;
class GSocketServerBuffer;



class GSocketClientBase
{
protected:
	bool m_bKeepListening;
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

	bool Send(const unsigned char *pBuf, int nLen);

	/// This method is abstract because you need to implement something here
	virtual bool Receive(unsigned char *pBuf, int nLen) = 0; // Override me


	u_short myPort();
	struct in_addr myIPAddr();
	bool isConnected();
	char* myIPAddr(char* szBuff, int nBuffSize);
	char* myName(char* szBuff, int nBuffSize);
	u_short otherPort();
	struct in_addr otherIPAddr();
	char* otherIPAddr(char* szBuff, int nBuffSize);
	char* otherName(char* szBuff, int nBuffSize);

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
	bool m_bKeepWorking;
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

	bool Send(const unsigned char *pBuf, int nLen, int nConnectionNumber);

	/// This method is abstract because you need to implement something here
	virtual bool Receive(unsigned char *pBuf, int nLen, int nConnectionNumber) = 0; // Override me

	bool IsConnected(int nConnectionNumber);

	/// Stops listening for new connections
	void DontAcceptAnyMoreConnections();

	/// If nConnectionNumber is less than 0, returns the address of the server
	char* GetIPAddr(char* szBuff, int nBuffSize, int nConnectionNumber);

	/// If nConnectionNumber is less than 0, returns the address of the server
	struct in_addr GetIPAddr(int nConnectionNumber);

	/// If nConnectionNumber is less than 0, returns the port of the server
	u_short GetPort(int nConnectionNumber);

	/// If nConnectionNumber is less than 0, returns the name of the server
	char* GetName(char* szBuff, int nBuffSize, int nConnectionNumber);

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
	int nPayloadSize;
};


class GSocketMessage
{
protected:
	unsigned char* m_pMessage;
	int m_nMessageSize;
	int m_nConnection;

public:
	GSocketMessage(unsigned char* pMessage, int nMessageSize, int nConnection)
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
	int GetMessageSize() { return m_nMessageSize; }
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
	int m_nMaxPacketSize;
	std::deque<GSocketMessage*> m_messageQueue;
	GSpinLock* m_pMessageQueueLock;

	virtual bool Receive(unsigned char *pBuf, int len, int nConnectionNumber);
	void QueueMessage(unsigned char* pBuf, int nLen, int nConnectionNumber);

public:
	/// if nMaxPacketSize = 0, the socket will speak raw UDP or TCP.
	/// if nMaxPacketSize > 0, it will speak GSKT over TCP. (GSKT guarantees
	///          same-size delivery of packets, but has a maximum packet size.)
	GSocketServer(bool bUDP, int nMaxPacketSize, int nPort, int nMaxConnections);

	virtual ~GSocketServer();

	/// Send some data
	bool Send(const void* pBuf, int nLen, int nConnectionNumber);

	/// Concat two blobs and send as one packet
	bool Send2(const void* pBuf1, int nLen1, const void* pBuf2, int nLen2, int nConnectionNumber);

	/// Returns the number of messages waiting to be received
	int GetMessageCount();

	/// Receive the next message. (You are responsible to delete the buffer this returns)
	unsigned char* GetNextMessage(int* pnSize, int* pnOutConnectionNumber);
};

// --------------------------------------------------------------------------

/// This class is designed to make network communication easy
class GSocketClient : public GSocketClientBase
{
protected:
	unsigned char* m_pBuffer;
	int m_nBufferPos;
	int m_nMaxPacketSize;
	std::deque<GSocketMessage*> m_messageQueue;
	GSpinLock* m_pMessageQueueLock;

public:
	/// if nMaxPacketSize = 0, the socket will speak raw UDP or TCP.
	/// if nMaxPacketSize > 0, it will speak GSKT over TCP. (GSKT guarantees
	///          same-size delivery of packets, but has a maximum packet size.)
	GSocketClient(bool bUDP, int nMaxPacketSize);

	virtual ~GSocketClient();

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !NO_TEST_CODE

	/// Send some data
	bool Send(const void* pBuf, int nLen);
	
	/// Concat two blobs and send as one packet
	bool Send2(const void* pBuf1, int nLen1, const void* pBuf2, int nLen2);

	/// Returns the number of messages waiting to be received
	int GetMessageCount();

	/// Receive the next message. (You are responsible to delete the buffer this returns)
	unsigned char* GetNextMessage(int* pnSize);

protected:
	virtual bool Receive(unsigned char *pBuf, int len);
	void QueueMessage(unsigned char* pBuf, int nLen);
};


} // namespace GClasses

#endif // __GSOCKET_H__
