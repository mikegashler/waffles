/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GHTTP_H__
#define __GHTTP_H__

#include <time.h>
#include <vector>
#include <sstream>
#include <map>
#include <string.h>

namespace GClasses {

class GHttpClientSocket;
class GWebSocketClientSocket;
class GHttpServerSocket;
class GTCPServer;
class GHttpConnection;
class GHttpServerBuffer;
class GHeap;
class GConstStringHashTable;


/// This class allows you to get files using the HTTP protocol
class GHttpClient
{
public:
	enum Status
	{
		Downloading,
		Error,
		NotFound,
		Done,
		Aborted,
	};

protected:
	char* m_pReceiveBuf;
	char m_szHeaderBuf[258];
	char m_szServer[256];
	char m_szClientName[32];
	size_t m_nHeaderPos;
	size_t m_nContentSize;
	bool m_bChunked;
	bool m_aborted;
	unsigned char* m_pData;
	size_t m_nDataPos;
	GHttpClientSocket* m_pSocket;
	Status m_status;
	std::ostringstream m_chunkBuf;
	bool m_bPastHeader;
	char* m_szRedirect;
	double m_dLastReceiveTime;
	bool m_bAmCurrentlyDoingJustHeaders;

public:
	GHttpClient();
	virtual ~GHttpClient();

	virtual void onReceiveData(const unsigned char* pData, size_t nLen) {}

	/// Send a request to get a file.  Returns immediately (before the file
	/// is downloaded).
	bool get(const char* szUrl, bool actuallyDownloadTheData = true);

	/// See what the status of the download is.  If everything is going okay,
	/// it will return "Downloading" while downloading and "Done" when the file
	/// is available.  pfProgress is an optional parameter.  If it is non-NULL,
	/// it will return a number between 0 and 1 that indicates the ratio of
	/// content (not including header data) already downloaded.
	Status status(float* pfProgress);

	/// Don't call this until the status is "Done".  It returns a pointer to the
	/// file that was downloaded.  The buffer will be deleted when this object is
	/// deleted, so if you want to retain the buffer, call releaseData instead.
	unsigned char* getData(size_t* pnSize);

	/// Just like getData except it forgets about the buffer so you'll have to
	/// delete it yourself.
	unsigned char* releaseData(size_t* pnSize);

	/// This is called when the connection is lost
	void onLoseConnection();

	void setClientName(const char* szClientName);
	
	void abort();	/// called by the consumer, when an abort is desired. 

protected:
	void processHeader(const unsigned char* szData, size_t nSize);
	void processBody(const unsigned char* szData, size_t nSize);
	void processChunkBody(const unsigned char* szData, size_t nSize);
	void gimmeWhatYouGot();

};

class GWebSocketClient
{
protected:
	GWebSocketClientSocket* m_pSocket;

public:
	GWebSocketClient();
	~GWebSocketClient();

	bool get(const char* szUrl);

	void onLoseConnection();
};



#define MAX_SERVER_LINE_SIZE 300
#define MAX_COOKIE_SIZE 300

/// This class allows you to implement a simple HTTP daemon
class GHttpServer
{
protected:
	char* m_pReceiveBuf;
	GHttpServerSocket* m_pSocket;
	std::vector<GHttpServerBuffer*> m_buffers;
	std::ostringstream m_stream;
	char m_szContentType[64];
	char m_szCookie[MAX_COOKIE_SIZE];
	bool m_bPersistCookie;
	time_t m_modifiedTime;

public:
	GHttpServer(int nPort);
	virtual ~GHttpServer();

	/// You should call this method constantly inside the main loop.
	/// It returns true if it did anything, and false if it didn't, so
	/// if it returns false you may want to sleep for a little while.
	bool process();

	/// Unescapes a URL. (i.e. replace "%20" with " ", etc.). szOut should point
	/// to a buffer at least as big as szIn (including the null terminator). This
	/// will stop when it hits the null-terminator in szIn or when nInLen characters
	/// have been parsed. So if szIn is null-terminated, you can safely pass in a
	/// huge arbitrary value for nInLen.
	static void unescapeUrl(char* szOut, const char* szIn, size_t nInLen);

	/// This is a rather hacky method that parses the parameters of a specific upload-file form
	static bool parseFileParam(const char* pParams, size_t nParamsLen, const char** ppFilename, size_t* pFilenameLen, const unsigned char** ppFile, size_t* pFileLen);

	/// Specifies the content-type of the response
	void setContentType(const char* szContentType);

	/// Specifies the set-cookie header to be sent with the response
	void setCookie(const char* szPayload, bool bPersist);

	/// Sets the date (modified time) to be sent with the file so the client can cache it
	void setModifiedTime(time_t t) { m_modifiedTime = t; }

	/// Returns a reference to the socket on which this server listens
	GTCPServer* socket() { return (GTCPServer*)m_pSocket; }

protected:
	virtual void onProcessLine(GHttpConnection* pConn, const char* szLine) {}
	void processPostData(GHttpConnection* pConn, const unsigned char* pData, size_t nDataSize);
	void processHeaderLine(GHttpConnection* pConn, const char* szLine);
	void beginRequest(GHttpConnection* pConn, int eType, const char* szIn);
	void sendResponse(GHttpConnection* pConn);
	void sendNotModifiedResponse(GHttpConnection* pConn);
	void onReceiveFullPostRequest(GHttpConnection* pConn);

	/// This method should set the content type and the date headers, and any other
	/// headers deemed necessary
	virtual void setHeaders(const char* szUrl, const char* szParams) = 0;

	/// The primary purpose of this method is to push a response into pResponse.
	/// Typically this method will call SetHeaders.
	virtual void doGet(const char* szUrl, const char* szParams, size_t nParamsLen, const char* szCookie, std::ostream& response) = 0;

	/// This method takes ownership of pData. Don't forget to delete it. When the POST is
	/// caused by an HTML form, it's common for this method to just call DoGet (passing
	/// pData for szParams) and then delete pData. (For convenience, a '\0' is already appended
	/// at the end of pData.)
	virtual void doPost(const char* szUrl, unsigned char* pData, size_t nDataSize, const char* szCookie, std::ostream& response) = 0;

	/// This is called when the client does a conditional GET. It should return true
	/// if you wish to re-send the file, and DoGet will be called.
	virtual bool hasBeenModifiedSince(const char* szUrl, const char* szDate) = 0;
};


struct strComp
{
	bool operator()(const char* a, const char* b) const { return strcmp(a, b) < 0; }
};


/// A class for parsing the name/value pairs that follow the "?" in a URL.
class GHttpParamParser
{
protected:
	GHeap* m_pHeap;
	std::map<const char*, const char*, strComp> m_map;

public:
	/// szParams should be everything in the URL after the "?".
	/// If scrub is true, then the values will be scrubbed. That is,
	/// all characters not in {a-z;A-Z;0-9;!;@;#;$;-;+;*;/;(;);.;,;:}
	/// will be replaced with an '_' character.
	GHttpParamParser(const char* szParams, bool scrub = true);
	~GHttpParamParser();

	/// Returns the value associated with the specified name. Returns NULL if the name is not found.
	const char* find(const char* szName);

	/// Returns a map of the name/value pairs
	std::map<const char*, const char*, strComp>& map() { return m_map; }

protected:
	static void scrubValue(char* value);
};


class GHttpMultipartParser
{
protected:
	const char* m_pRawData;
	size_t m_sentinelLen;
	size_t m_repeatLen;
	size_t m_pos;
	size_t m_len;

public:
	GHttpMultipartParser(const char* pRawData, size_t len);
	~GHttpMultipartParser();

	bool next(size_t* pNameStart, size_t* pNameLen, size_t* pValueStart, size_t* pValueLen, size_t* pFilenameStart, size_t* pFilenameLen);
};

} // namespace GClasses

#endif // __GHTTP_H__
