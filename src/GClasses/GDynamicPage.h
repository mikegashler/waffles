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

#ifndef __GDYNAMICPAGE_H__
#define __GDYNAMICPAGE_H__

#include "GHttp.h"
#include <stddef.h>
#include <vector>
#include <string>
#include <map>

namespace GClasses {

class GConstStringHashTable;
class GDynamicPageServer;
class GDynamicPageSession;
class GDynamicPageConnection;
class GRand;
class GDom;



class GDynamicPageSessionExtension
{
public:
	GDynamicPageSessionExtension() {}
	virtual ~GDynamicPageSessionExtension() {}

	/// This method is called when a parent session is deleted,
	/// or when another extension takes the place of this one.
	/// If you consider this extension to be owned by that session,
	/// then you should probably call "delete(this)" in
	/// this method. Otherwise, this method just exists to let
	/// you know that the session will no longer be referring to
	/// this extension.
	virtual void onDisown() = 0;

	/// Marshal this object into a Dom node.
	virtual GDomNode* serialize(GDom* pDoc) = 0;
};


class GDynamicPageSession
{
protected:
	unsigned long long m_id;
	time_t m_tLastAccessed;
	GDynamicPageSessionExtension* m_pExtension;
	GDynamicPageConnection* m_pConnection;

public:
	GDynamicPageSession(unsigned long long id);
	GDynamicPageSession(GDynamicPageServer* pServer, GDomNode* pNode);
	virtual ~GDynamicPageSession();

	/// Returns the server object associated with this session
	GDynamicPageServer* server();

	/// Returns the id associated with this session
	unsigned long long id() { return m_id; }

	/// Use this to store your own custom object with the session.
	/// Your object should inherit from GDynamicPageSessionExtension.
	/// (The extension will not be deleted when the session is deleted,
	/// but the GDynamicPageSessionExtension::onDisown method will be
	/// called, which you can use to delete the extension.)
	void setExtension(GDynamicPageSessionExtension* pExtension);

	/// Stamp the session as having been accessed at the current time
	void onAccess();

	/// Associate the current connection with this session
	void setConnection(GDynamicPageConnection* pConn);

	GDynamicPageConnection* connection() { return m_pConnection; }

	/// Retrieve the extension object that was associated with this
	/// session by a call to setExtension.
	GDynamicPageSessionExtension* extension() { return m_pExtension; }

	/// Marshal this object into a Dom node
	GDomNode* serialize(GDom* pDom);

	// Get the current URL from the connection. (Convenience method.)
	const char* url();

	// Get the current params from the connection. (Convenience method.)
	const char* params();

	// Get the length of the params
	size_t paramsLen();

	// Add a 'cookie' field to pOutNode with this session's cookie.
	// (This is needed because some browsers block cookies in AJAX requests.)
	void addAjaxCookie(GDom& docOut, GDomNode* pOutNode);

};




class GDynamicPageConnection : public GHttpConnection
{
public:
	GDynamicPageServer* m_pServer;


	GDynamicPageConnection(SOCKET sock, GDynamicPageServer* pServer);
	virtual ~GDynamicPageConnection();

	virtual void doGet(std::ostream& response);
	virtual void doPost(std::ostream& response);

	/// Concatenates szJailPath+szLocalPath, and makes sure that the result is within szJailPath, then
	/// it automatically determines the mime type from the extension, and sends the file.
	void sendFileSafe(const char* szJailPath, const char* localPath, std::ostream& response);

protected:
	/// This method is called by doGet or doPost when a client requests something from the server
	virtual void handleRequest(GDynamicPageSession* pSession, std::ostream& response) = 0;

	virtual bool hasBeenModifiedSince(const char* szUrl, const char* szDate);
	virtual void setHeaders(const char* szUrl, const char* szParams);
	GDynamicPageSession* establishSession();

	void sendFile(const char* szMimeType, const char* szFilename, std::ostream& response);

	/// Determines an appropriate mime type for the given filename based on its extension.
	/// (Currently only recognizes a very small number of extensions.)
	static const char* extensionToMimeType(const char* szFilename);
};



class GDynamicPageServer : public GHttpServer
{
protected:
	GRand* m_pRand;
	std::map<unsigned long long, GDynamicPageSession*> m_sessions;
	char* m_szMyAddress;
	char m_daemonSalt[16];
	char m_passwordSalt[16];

public:
	GDynamicPageServer(int port, GRand* pRand);
	virtual ~GDynamicPageServer();

	virtual GDynamicPageSessionExtension* deserializeSessionExtension(const GDomNode* pNode)
	{
		throw Ex("deserializeSessionExtension has not be overridden");
	}

	void flushSessions();

	/// Returns the session with the specified id. If no session is found with that id, returns NULL.
	GDynamicPageSession* findSession(unsigned long long id);

	/// Makes a new session with the specified id.
	GDynamicPageSession* makeNewSession(unsigned long long id);

	/// Prints all known session ids to the stream. (This is used for debugging purposes.)
	void printSessionIds(std::ostream& stream);

	/// Returns the account if the password is correct. Returns NULL if not.
	const char* myAddress();

	void setDaemonSalt(const char* szSalt);
	const char* daemonSalt();
	const char* passwordSalt();

	GRand* prng() { return m_pRand; }
	void redirect(std::ostream& response, const char* szUrl);

	GDomNode* serialize(GDom* pDoc);
	void deserialize(const GDomNode* pNode);

protected:
	void doMaintenance();
	void computePasswordSalt();
};

} // namespace GClasses

#endif // __GDYNAMICPAGE_H__
