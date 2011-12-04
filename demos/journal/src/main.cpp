// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#ifdef WINDOWS
#	include <windows.h>
#	include <process.h>
#	include <direct.h>
#else
#	include <unistd.h>
#endif
#include <GClasses/GDynamicPage.h>
#include <GClasses/GImage.h>
#include <GClasses/GDirList.h>
#include <GClasses/GHolders.h>
#include <GClasses/GApp.h>
#include <GClasses/GDom.h>
#include <GClasses/GString.h>
#include <GClasses/GHeap.h>
#include <GClasses/GHttp.h>
#include <GClasses/GFile.h>
#include <GClasses/GTime.h>
#include <GClasses/GRand.h>
#include <GClasses/GHashTable.h>
#include <GClasses/sha1.h>
#include <wchar.h>
#include <string>
#include <vector>
#include <exception>
#include <iostream>
#include <sstream>

using namespace GClasses;
using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::ostream;
using std::map;


class View;
class Account;


class Server : public GDynamicPageServer
{
protected:
	std::string m_basePath;
	View* m_pViewAdmin;
	View* m_pViewUpload;
	View* m_pViewLogin;
	View* m_pViewNewAccount;

	// Typically the accounts would be stored in a database, but since this is
	// just a demo, we'll keep them all in memory for simplicity.
	std::map<std::string,Account*> m_accounts;

public:
	Server(int port, GRand* pRand);
	virtual ~Server();
	void loadState();
	void saveState();
	virtual void handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, std::ostream& response);
	void getStatePath(char* buf);
	virtual void onEverySixHours();
	virtual void onStateChange();
	virtual void onShutDown();
	Account* loadAccount(const char* szUsername, const char* szPasswordHash);
	Account* newAccount(const char* szUsername, const char* szPasswordHash);
	GDomNode* serializeState(GDom* pDoc);
	void deserializeState(GDomNode* pNode);
};

class Account : public GDynamicPageSessionExtension
{
protected:
	string m_afterLoginUrl;
	string m_afterLoginParams;
	string m_username;
	string m_passwordHash;

public:
	Account(const char* szUsername, const char* szPasswordHash)
	: m_username(szUsername), m_passwordHash(szPasswordHash)
	{
	}

	virtual ~Account()
	{
	}

	virtual void onDisown()
	{
	}

	void setAfterLoginUrlAndParams(const char* szUrl, const char* szParams)
	{
		m_afterLoginUrl = szUrl;
		m_afterLoginParams = szParams;
	}

	void clearAfterLoginStuff()
	{
		m_afterLoginUrl.clear();
		m_afterLoginParams.clear();
	}

	const char* afterLoginUrl()
	{
		return m_afterLoginUrl.c_str();
	}

	const char* afterLoginParams()
	{
		return m_afterLoginParams.c_str();
	}

	static Account* fromDom(GDomNode* pNode)
	{
		Account* pAccount = new Account(pNode->field("username")->asString(), pNode->field("password")->asString());
		return pAccount;
	}

	GDomNode* toDom(GDom* pDoc)
	{
		GDomNode* pAccount = pDoc->newObj();
		pAccount->addField(pDoc, "username", pDoc->newString(m_username.c_str()));
		pAccount->addField(pDoc, "password", pDoc->newString(m_passwordHash.c_str()));
		return pAccount;
	}

	const char* username() { return m_username.c_str(); }
	const char* passwordHash() { return m_passwordHash.c_str(); }

	bool doesHavePassword()
	{
		return m_passwordHash.length() > 0;
	}
};

Account* getAccount(GDynamicPageSession* pSession)
{
	Account* pAccount = (Account*)pSession->extension();
	if(!pAccount)
	{
		Server* pServer = (Server*)pSession->server();
		std::ostringstream oss;
		oss << "_";
		oss << pSession->id();
		string tmp = oss.str();
		const char* szGenericUsername = tmp.c_str();
		pAccount = pServer->loadAccount(szGenericUsername, NULL);
		if(!pAccount)
		{
			pAccount = pServer->newAccount(szGenericUsername, NULL);
			if(!pAccount)
				ThrowError("Failed to create account");
		}
		pSession->setExtension(pAccount);
	}
	return pAccount;
}

void makeHeader(GDynamicPageSession* pSession, ostream& response)
{
	Account* pAccount = getAccount(pSession);
	response << "<html><head>\n";
	response << "	<title>The Empirically Reviewed Journal of Machine Learning</title>\n";
	response << "	<link rel=\"stylesheet\" type=\"text/css\" href=\"/style/style.css\" />\n";
	response << "</head><body><div id=\"wholepage\">\n";
	response << "\n\n\n\n\n<!-- Header Area --><div id=\"header\">\n";
	response << "	The Empirically Reviewed Journal of Machine Learning\n";
	response << "</div>\n\n\n\n\n<!-- Left Bar Area --><div id=\"sidebar\">\n";
	response << "	<center><img src=\"style/logo.png\"><br>\n";
	if(pAccount->doesHavePassword())
	{
		response << "	Logged in as ";
		response << pAccount->username();
		response << "<br>\n";
		response << "	<a href=\"/login?action=logout\">Log out</a><br>\n";
	}
	response << "	</center><br>\n";
	response << "	<a href=\"/main.hbody\">Main Page</a><br>\n";
	response << "	<a href=\"/competitions.hbody\">Competition Specs</a><br>\n";
	response << "	<a href=\"/admin\">Admin Page</a><br>\n";
	response << "	<a href=\"/upload\">Upload a file</a><br>\n";
	response << "	<a href=\"http://code.google.com/p/gpeerreview/\">GPeerReview</a><br>\n";
	response << "	<a href=\"http://waffles.sourceforge.net/\">Waffles</a><br>\n";
	response << "	<br><br><br><br><br><br>\n";
	response << "</div>\n\n\n\n\n<!-- Main Body Area --><div id=\"mainbody\">\n";
}

void makeFooter(GDynamicPageSession* pSession, ostream& response)
{
	response << "</div>\n\n\n\n\n<!-- Footer Area --><div id=\"footer\">\n";
	response << "	The contents of this page are distributed under the <a href=\"http://creativecommons.org/publicdomain/zero/1.0/\">CC0 1.0 license</a>. <img src=\"http://i.creativecommons.org/l/zero/1.0/80x15.png\" border=\"0\" alt=\"CC0\" />\n";
	response << "</div>\n\n\n\n\n";
	response << "</div></body></html>\n";
}

class View
{
protected:
	Server* m_pServer;

public:
	View(Server* pServer) : m_pServer(pServer) {}
	virtual ~View() {}

	virtual void makePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);
		makeBody(pSession, response);
		makeFooter(pSession, response);
	}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response) = 0;
	virtual bool isLogInRequired() = 0;
};

// ------------------------------------------------------

class ViewAdmin : public View
{
public:
	ViewAdmin(Server* pServer) : View(pServer) {}
	virtual ~ViewAdmin() {}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		if(_stricmp(pAccount->username(), "root") != 0)
		{
			response << "Sorry, only the user with username <i>root</i> can access the admin page (and if you are not the administrator, then this username has probably already been taken).<br><br>\n";
			return;
		}
		if(pSession->paramsLen() > 0)
		{
			GHttpParamParser params(pSession->params());
			const char* szAction = params.find("action");
			if(szAction)
			{
				// Do the action
				if(_stricmp(szAction, "shutdown") == 0)
				{
					cout << "root has told the server to shut down.\n";
					cout.flush();
					cerr.flush();
					m_pServer->shutDown();
				}
				else
					response << "[Unknown action! No action taken]<br>\n";
			}
		}

		response << "<form name=\"shutdownform\" action=\"/admin\" method=\"get\">\n";
		response << "	<input type=\"hidden\" name=\"action\" value=\"shutdown\" />\n";
		response << "	<input type=\"submit\" value=\"Gracefully Shut Down Server\">\n";
		response << "</form><br>\n\n";
	}

	virtual bool isLogInRequired() { return true; }
};

// ------------------------------------------------------

class ViewUpload : public View
{
protected:
	string m_basePath;

public:
	ViewUpload(Server* pServer, const char* basePath) : View(pServer)
	{
		m_basePath = basePath;
	}

	virtual ~ViewUpload()
	{
	}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		if(pSession->paramsLen() <= 0)
		{
			// Show the upload form
			response << "You can submit a file here. It must be less than 25 MB.<br>\n";
			response << "<form method='POST' enctype='multipart/form-data' action='/upload'>\n";
			response << "	Competition: <select name=competition>\n";
			response << "		<option value=\"1\">1) Classification predictive accuracy</option>\n";
			response << "		<option value=\"2\">2) Area under ROC curve with binary classification sets</option>\n";
			response << "		<option value=\"3\">3) Classification of hand-written numbers</option>\n";
			response << "		<option value=\"4\">4) Precision-recall with multi-class data</option>\n";
			response << "		<option value=\"5\">5) Efficient classification with large datasets</option>\n";
			response << "		<option value=\"6\">6) Precision-recall with multi-class data</option>\n";
			response << "		<option value=\"7\">7) Incremental/stream learning tests</option>\n";
			response << "		<option value=\"8\">8) Active learning tests</option>\n";
			response << "		<option value=\"9\">9) Robustness to irrelevant features tests</option>\n";
			response << "	</select><br>\n";
			response << "	File to upload: <input type=file name=upfile><br>\n";
			response << "	<input type=submit value=Submit> (Please press submit only once!)\n";
			response << "</form><br><br>\n";
		}
		else
		{
			// Make a folder for this upload
			Account* pAccount = getAccount(pSession);
			string s = m_basePath;
			s += "uploads/";
			s += pAccount->username();
			s += "/";
			string timestamp;
			GTime::appendTimeStampValue(&timestamp, "-", "_", "-", true/*use GMT*/);
			s += timestamp;
			s += "/";
			GFile::makeDir(s.c_str());

			// Extract and save the uploaded data
			string meta = "username=";
			meta += pAccount->username();
			meta += "\ntimestamp=";
			meta += timestamp;
			meta += "\n";
			size_t nameStart, nameLen, valueStart, valueLen, filenameStart, filenameLen;
			GHttpMultipartParser parser(pSession->params(), pSession->paramsLen());
			while(parser.next(&nameStart, &nameLen, &valueStart, &valueLen, &filenameStart, &filenameLen))
			{
				if(nameStart >= 0)
					meta.append(pSession->params() + nameStart, nameLen);
				else
					meta += "?";
				meta += "=";
				if(filenameStart >= 0)
				{
					meta.append(pSession->params() + filenameStart, filenameLen);
					string s2 = s;
					s2.append(pSession->params() + filenameStart, filenameLen);
					GFile::saveFile(pSession->params() + valueStart, valueLen, s2.c_str());
				}
				else
					meta.append(pSession->params() + valueStart, valueLen);
				meta += "\n";
			}
			s += "meta.txt";
			GFile::saveFile(meta.c_str(), meta.length(), s.c_str());

			// Send an email to notify the administrator
//			GSmtp::sendEmail("username@domain.com", "automated@nowhere.com", "The following file has been uploaded", meta.c_str(), "smtp.domain.com");

			// Inform the user that the submission was received
			response << "The following submission was received:<br><pre>\n";
			response << meta.c_str();
			response << "</pre><br><br>\n\n";
			response << "Have a nice day!\n";
		}
	}

	virtual bool isLogInRequired() { return true; }
};

// ------------------------------------------------------

class ViewLogin : public View
{
public:
	ViewLogin(Server* pServer) : View(pServer) {}
	virtual ~ViewLogin() {}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		if(pSession->paramsLen() >= 0)
		{
			// See if the user wants to log out
			GHttpParamParser params(pSession->params());
			const char* szAction = params.find("action");
			if(szAction)
			{
				if(_stricmp(szAction, "logout") == 0)
				{
					m_pServer->redirect(response, "/main.hbody");
					pSession->setExtension(NULL); // disconnect the account from this session
					return;
				}
				else
					response << "Unrecognized action: " << szAction << "<br><br>\n\n";
			}

			// Check the password
			const char* szUsername = params.find("username");
			const char* szPasswordHash = params.find("password");
			if(szUsername)
			{
				Account* pNewAccount = m_pServer->loadAccount(szUsername, szPasswordHash);
				if(pNewAccount)
				{
					string s;
					if(pAccount)
					{
						if(strlen(pAccount->afterLoginUrl()) > 0)
							s = pAccount->afterLoginUrl();
						if(strlen(pAccount->afterLoginParams()) > 0)
						{
							s += "?";
							s += pAccount->afterLoginParams();
						}
						if(s.length() < 1)
						{
							s = "/main.hbody?";
							s += to_str((size_t)m_pServer->prng()->next()); // ensure no caching
						}
					}
					else
					{
						s = "/main.hbody";
						s += to_str((size_t)m_pServer->prng()->next()); // ensure no caching
					}

					// Log in with the new account
					pSession->setExtension(pNewAccount);
					m_pServer->redirect(response, s.c_str());
				}
				else
					response << "<big><big>Incorrect Password! Please try again</big></big><br><br>\n";
			}
		}

		response << "<br><br>\n";
		response << "<SCRIPT language=\"JavaScript\" src=\"/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
		if(pAccount)
		{
			response << "Your current username is: " << pAccount->username() << ".<br>\n";
			if(pAccount->doesHavePassword())
				response << "Click here to <a href=\"/login?action=logout\">log out</a>.<br>\n";
			response << "Log in as a different user:<br>\n";
		}
		else
			response << "Please enter credentials to log in:<br>\n";
		response << "<form name=\"loginform\" action=\"/login\" method=\"get\" onsubmit=\"return HashPassword('";
		response << m_pServer->passwordSalt();
		response << "')\">\n";
		response << "	Username:<input type=\"text\" name=\"username\" ><br>\n";
		response << "	Password:<input type=\"password\" name=\"password\" ><br>\n";
		response << "	<input type=\"submit\" value=\"Log In\">\n";
		response << "</form><br><br>\n\n";

		response << "or click here to <a href=\"/newaccount\">create a new account</a><br><br>\n";
	}

	virtual bool isLogInRequired() { return false; }
};

// ------------------------------------------------------

class ViewNewAccount : public View
{
public:
	ViewNewAccount(Server* pServer);
	virtual ~ViewNewAccount();

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response);
	virtual bool isLogInRequired() { return false; }

protected:
	void GetCaptchaText(char* szOut, const char* szID);
	void MakeCaptcha(const char* szID, ostream& response);
	void SendCaptcha(const char* szText, ostream& response);
	bool CheckCaptchaText(const char* a, const char* b);
};

ViewNewAccount::ViewNewAccount(Server* pServer)
: View(pServer)
{
}

/*virtual*/ ViewNewAccount::~ViewNewAccount()
{
}

bool ViewNewAccount::CheckCaptchaText(const char* pA, const char* pB)
{
	while(*pA != '\0')
	{
		char a = (*pA | 32);
		char b = (*pB | 32);

		// Close enough
		if(a == 'O') a = '0';	if(b == 'O') b = '0';
		if(a == 'S') a = '5';	if(b == 'S') b = '5';
		if(a == 'G') a = '6';	if(b == 'G') b = '6';

		if(a != b)
			return false;
		pA++;
		pB++;
	}
	if(*pB != '\0')
		return false;
	return true;
}

void ViewNewAccount::GetCaptchaText(char* szOut, const char* szID)
{
	unsigned char digest[20];
	SHA_CTX ctx;
	memset(&ctx, '\0', sizeof(SHA_CTX));
	SHA1_Init(&ctx);
	const char* daemonSalt = m_pServer->daemonSalt();
	SHA1_Update(&ctx, (unsigned char*)szID, strlen(szID));
	SHA1_Update(&ctx, (unsigned char*)daemonSalt, strlen(daemonSalt));
	SHA1_Final(digest, &ctx);
	int i;
	for(i = 0; i < 6; i++)
	{
		szOut[i] = digest[i] % 36;
		if(szOut[i] >= 10)
			szOut[i] += 'A' - 10;
		else
			szOut[i] += '0';
	}
	szOut[i] = '\0';
}

#ifndef WINDOWS
void DeleteFile(const char* szFilename)
{
	char szBuf[64];
	strcpy(szBuf, "rm ");
	strcat(szBuf, szFilename);
	strcat(szBuf, " &");
	if(system(szBuf) == -1)
		ThrowError("Failed to delete file");
}
#endif

void ViewNewAccount::MakeCaptcha(const char* szID, ostream& response)
{
	char szText[32];
	GetCaptchaText(szText, szID);
	std::ostringstream& r = reinterpret_cast<std::ostringstream&>(response);
	r.str("");
	r.clear();

	// Make the filename
	char szTemp[512];
	GFile::tempFilename(szTemp);

	// Make the captcha
	GImage image;
	image.captcha(szText, m_pServer->prng());
	image.savePng(szTemp);
	m_pServer->sendFile("image/png", szTemp, response);
	DeleteFile(szTemp);
}

/*virtual*/ void ViewNewAccount::makeBody(GDynamicPageSession* pSession, ostream& response)
{
	if(_strnicmp(pSession->url(), "/captcha", 8) == 0)
	{
		char szID[9];
		memcpy(szID, pSession->url() + 8, 8);
		szID[8] = '\0';
		MakeCaptcha(szID, response);
		return;
	}

	const char* szUsername = "";
	const char* szEmail = "";
	const char* szPassword = "";
	const char* szPWAgain = "";
	const char* szCaptchaId = "";
	const char* szCaptcha = "";
	GHttpParamParser params(pSession->params());
	if(pSession->paramsLen() > 0)
	{
		// Get the action
		const char* szError = NULL;
		const char* szAction = params.find("action");
		if(!szAction)
			szError = "Expected an action param";
		if(!szError && _stricmp(szAction, "newaccount") != 0)
			szError = "Unrecognized action";

		szUsername = params.find("username");
		szEmail = params.find("email");
		szPassword = params.find("password");
		szPWAgain = params.find("pwagain");
		szCaptchaId = params.find("captchaid");

		// Check the parameters
		if(!szUsername || strlen(szUsername) < 1)
			szError = "The username is not valid";
		if(!szEmail || strlen(szEmail) < 1)
			szError = "The email address is not valid";
		if(!szPassword || strlen(szPassword) < 1)
			szError = "The password is not valid";
		if(!szPWAgain || strcmp(szPassword, szPWAgain) != 0)
			szError = "The passwords don't match";
		if(!szCaptchaId)
			szError = "Expected a hidden captcha id";
		char szExpectedCaptchaText[32];
		GetCaptchaText(szExpectedCaptchaText, szCaptchaId);
		szCaptcha = params.find("captcha");
		if(!szCaptcha || strlen(szCaptcha) < 1)
			szError = "You must enter the captcha text as shown in the image";
		if(!CheckCaptchaText(szCaptcha, szExpectedCaptchaText))
			szError = "The captcha text is incorrect";
		if(!szError)
		{
			// Create the account
			Account* pAccount = m_pServer->newAccount(szUsername, szPassword);
			if(!pAccount)
				szError = "That username is already taken.";
			else
			{
				m_pServer->saveState();
				response << "<big>An account has been successfully created.</big><br><br> Click here to <a href=\"/login\">log in</a><br>\n";
				return;
			}
		}
		if(szError)
		{
			response << "<center>";
			response << szError;
			response << "</center><br><br>\n\n";
			szPassword = "";
			szPWAgain = "";
		}
	}

	// Make a captcha ID
	char szCaptchaID[32];
	int i;
	for(i = 0; i < 8; i++)
		szCaptchaID[i] = (char)m_pServer->prng()->next(26) + 'a';
	szCaptchaID[i] = '\0';

	response << "<br><center><table width=\"400\" border=\"0\" cellpadding=\"0\" cellspacing=\"0\"><tr><td>\n";
	response << "<SCRIPT language=\"JavaScript\" src=\"/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
	response << "	<big><big><b>Create a new account</b></big></big><br><br>\n";
	response << "	<form name=\"newaccountform\" action=\"/newaccount\" method=\"post\" onsubmit=\"return HashNewAccount('";
	response << m_pServer->passwordSalt();
	response << "')\">\n";
	response << "		<input type=\"hidden\" name=\"action\" value=\"newaccount\" />\n";
	response << "		<input type=\"hidden\" name=\"captchaid\" value=\"";
	response << szCaptchaID;
	response << "\" />\n";
	response << "		Username: <input type=\"text\" size=\"15\" name=\"username\" value=\"";
	response << szUsername;
	response << "\"><br><br>\n";
	response << "		Email: <input type=\"text\" size=\"15\" name=\"email\" value=\"";
	response << szEmail;
	response << "\"><br><br>\n";
	response << "		Password: <input type=\"password\" name=\"password\" size=\"15\" value=\"";
	response << szPassword;
	response << "\"><br>\n";
	response << "		PW Again: <input type=\"password\" name=\"pwagain\" size=\"15\" value=\"";
	response << szPWAgain;
	response << "\"><br><br>\n";
	response << "		<img src=\"captcha";
	response << szCaptchaID;
	response << ".png\"><br>\n";
	response << "		If you can't read the captcha, click here to <a href=\"/newaccount\">get another one</a><br>\n";
	response << "		Captcha: <input type=\"text\" size=\"15\" name=\"captcha\"><br>\n";
	response << "		<br><input type=\"submit\" value=\"Submit\">\n";
	response << "	</form><br>\n\n";
	response << "</tr></td></table></center>\n";
}

// ------------------------------------------------------

Server::Server(int port, GRand* pRand) : GDynamicPageServer(port, pRand)
{
	char buf[300];
	GTime::asciiTime(buf, 256, false);
	cout << "Server starting at: " << buf << "\n";
	GApp::appPath(buf, 256, true);
	strcat(buf, "web/");
	GFile::condensePath(buf);
	m_basePath = buf;
	m_pViewAdmin = new ViewAdmin(this);
	m_pViewUpload = new ViewUpload(this, buf);
	m_pViewLogin = new ViewLogin(this);
	m_pViewNewAccount = new ViewNewAccount(this);
	loadState();
}

// virtual
Server::~Server()
{
	saveState();
	delete(m_pViewAdmin);
	delete(m_pViewUpload);
	delete(m_pViewLogin);
	delete(m_pViewNewAccount);

	// Delete all the accounts
	flushSessions(); // ensure that there are no sessions referencing the accounts
	for(map<string,Account*>::iterator it = m_accounts.begin(); it != m_accounts.end(); it++)
		delete(it->second);
}

void Server::loadState()
{
	char statePath[300];
	getStatePath(statePath);
	if(GFile::doesFileExist(statePath))
	{
		GDom doc;
		doc.loadJson(statePath);
		deserializeState(doc.root());
		cout << "State loaded from: " << statePath << "\n";
	}
	else
		cout << "No state file (" << statePath << ") found. Creating new state.\n";
}

void Server::saveState()
{
	GDom doc;
	doc.setRoot(serializeState(&doc));
	char szStoragePath[300];
	getStatePath(szStoragePath);
	doc.saveJson(szStoragePath);
	char szTime[256];
	GTime::asciiTime(szTime, 256, false);
	cout << "Server state saved at: " << szTime << "\n";
}

// virtual
void Server::handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, ostream& response)
{
	View* pView = NULL;
	if(strcmp(szUrl, "/") == 0)
		szUrl = "/main.hbody";
	else if(strcmp(szUrl, "/favicon.ico") == 0)
		return;
	else if(strncmp(szUrl, "/login", 6) == 0)
		pView = m_pViewLogin;
	else if(strncmp(szUrl, "/admin", 6) == 0)
		pView = m_pViewAdmin;
	else if(strncmp(szUrl, "/upload", 7) == 0)
		pView = m_pViewUpload;
	else if(strncmp(szUrl, "/newaccount", 11) == 0)
		pView = m_pViewNewAccount;
	else if(strncmp(szUrl, "/captcha", 7) == 0)
		pView = m_pViewNewAccount;
	if(pView)
	{
		Account* pAccount = getAccount(pSession);
		if(pView->isLogInRequired() && !pAccount->doesHavePassword())
		{
			pAccount->setAfterLoginUrlAndParams(szUrl, szParams);
			pView = m_pViewLogin;
		}
		pView->makePage(pSession, response);
	}
	else
	{
		size_t len = strlen(szUrl);
		if(len > 6 && strcmp(szUrl + len - 6, ".hbody") == 0)
		{
			makeHeader(pSession, response);
			sendFileSafe(m_basePath.c_str(), szUrl + 1, response);
			makeFooter(pSession, response);
		}
		else
			sendFileSafe(m_basePath.c_str(), szUrl + 1, response);
	}
}


void getLocalStorageFolder(char* buf)
{
	if(!GFile::localStorageDirectory(buf))
		ThrowError("Failed to find local storage folder");
	strcat(buf, "/.erjml/");
	GFile::makeDir(buf);
	if(!GFile::doesDirExist(buf))
		ThrowError("Failed to create folder in storage area");
}

void Server::getStatePath(char* buf)
{
	getLocalStorageFolder(buf);
	strcat(buf, "state.json");
}

// virtual
void Server::onEverySixHours()
{
	saveState();
	fflush(stdout);
}

// virtual
void Server::onStateChange()
{
}

// virtual
void Server::onShutDown()
{
}


Account* Server::loadAccount(const char* szUsername, const char* szPasswordHash)
{
	if(!szPasswordHash)
		szPasswordHash = "";

	// Find the account
	map<string,Account*>::iterator it = m_accounts.find(szUsername);
	if(it == m_accounts.end())
		return NULL;
	Account* pAccount = it->second;

	// Check the password hash
	if(_stricmp(pAccount->passwordHash(), szPasswordHash) != 0)
		return NULL;
	return pAccount;
}

Account* Server::newAccount(const char* szUsername, const char* szPasswordHash)
{
	if(!szPasswordHash)
		szPasswordHash = "";

	// See if that username already exists
	map<string,Account*>::iterator it = m_accounts.find(szUsername);
	if(it != m_accounts.end())
		return NULL;

	// Make the account
	Account* pAccount = new Account(szUsername, szPasswordHash);
	m_accounts.insert(make_pair(string(szUsername), pAccount));
	return pAccount;
}

GDomNode* Server::serializeState(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();

	// Captcha salt
	pNode->addField(pDoc, "daemonSalt", pDoc->newString(daemonSalt()));

	// Save the accounts
	GDomNode* pAccounts = pNode->addField(pDoc, "accounts", pDoc->newList());
	size_t i = 0;
	for(map<string,Account*>::iterator it = m_accounts.begin(); it != m_accounts.end(); it++)
	{
		Account* pAccount = it->second;
		pAccounts->addItem(pDoc, pAccount->toDom(pDoc));
		i++;
	}

	return pNode;
}

void Server::deserializeState(GDomNode* pNode)
{
	// Captcha salt
	const char* daemonSalt = pNode->fieldIfExists("daemonSalt")->asString();
	if(daemonSalt)
		setDaemonSalt(daemonSalt);

	// Load the accounts
	GDomNode* pAccounts = pNode->field("accounts");
	for(GDomListIterator it(pAccounts); it.current(); it.advance())
	{
		Account* pAccount = Account::fromDom(it.current());
		m_accounts.insert(make_pair(string(pAccount->username()), pAccount));
	}
}







void OpenUrl(const char* szUrl)
{
#ifdef WINDOWS
	// Windows
	ShellExecute(NULL, NULL, szUrl, NULL, NULL, SW_SHOW);
#else
#ifdef DARWIN
	// Mac
	GTEMPBUF(char, pBuf, 32 + strlen(szUrl));
	strcpy(pBuf, "open ");
	strcat(pBuf, szUrl);
	strcat(pBuf, " &");
	system(pBuf);
#else // DARWIN
	GTEMPBUF(char, pBuf, 32 + strlen(szUrl));

	// Gnome
	strcpy(pBuf, "gnome-open ");
	strcat(pBuf, szUrl);
	if(system(pBuf) != 0)
	{
		// KDE
		//strcpy(pBuf, "kfmclient exec ");
		strcpy(pBuf, "konqueror ");
		strcat(pBuf, szUrl);
		strcat(pBuf, " &");
		if(system(pBuf) != 0)
			cout << "Failed to open " << szUrl << ". Please open it manually.\n";
	}
#endif // !DARWIN
#endif // !WINDOWS
}

void LaunchBrowser(const char* szAddress)
{
	int addrLen = strlen(szAddress);
	GTEMPBUF(char, szUrl, addrLen + 20);
	strcpy(szUrl, szAddress);
	strcpy(szUrl + addrLen, "/main.hbody");
	if(!GApp::openUrlInBrowser(szUrl))
	{
		cout << "Failed to open the URL: " << szUrl << "\nPlease open this URL manually.\n";
		cout.flush();
	}
}

void doit(void* pArg)
{
	int port = 8989;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	GRand prng(seed);
	Server server(port, &prng);
	LaunchBrowser(server.myAddress());

	// Pump incoming HTTP requests (this is the main loop)
	server.go();
	cout << "Goodbye.\n";
}

void doItAsDaemon()
{
	char path[300];
	getLocalStorageFolder(path);
	string s1 = path;
	s1 += "stdout.log";
	string s2 = path;
	s2 += "stderr.log";
	GApp::launchDaemon(doit, path, s1.c_str(), s2.c_str());
	cout << "Daemon running.\n	stdout >> " << s1.c_str() << "\n	stderr >> " << s2.c_str() << "\n";
}

int main(int nArgs, char* pArgs[])
{
	int nRet = 1;
	try
	{
		doit(NULL);
		//doItAsDaemon();
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
