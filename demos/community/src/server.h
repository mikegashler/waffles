/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#ifndef SERVER
#define SERVER

#include <map>
#include <string>
#include <GClasses/GFile.h>
#include <GClasses/GDynamicPage.h>
#include "recommender.h"

class Account;

using namespace GClasses;

class MyJaad : public GJsonAsADatabase
{
public:
	MyJaad(const char* szBasePath) : GJsonAsADatabase(szBasePath)
	{}

	virtual ~MyJaad()
	{}

	virtual bool checkPermission(const char* szFilename)
	{
		// Make sure it's not too long
		size_t len = strlen(szFilename);
		if(len > 128)
			return false;

		// Make sure there are no funny characters
		for(size_t i = 0; i < len; i++)
		{
			if(szFilename[i] == '.' || szFilename[i] == '/' ||
				szFilename[i] == '_' || szFilename[i] == '-' ||
				(szFilename[i] >= 'a' && szFilename[i] <= 'z') ||
				(szFilename[i] >= 'A' && szFilename[i] <= 'Z') ||
				(szFilename[i] >= '0' && szFilename[i] <= '9'))
				{}
			else
				return false;
		}

		// No breaking out of the base path
		if(strstr(szFilename, ".."))
			return false;

		// Make sure it ends with "comments.json"
		std::string sFile = "comments.json";
		if(len < sFile.length())
			return false;
		std::string s = szFilename;
		if(s.compare (s.length() - sFile.length(), sFile.length(), sFile) != 0)
			return false;

		return true;
	}
};


class Server : public GDynamicPageServer
{
protected:
	std::map<std::string,size_t> m_username_to_index; // Mapping from username to account index
	std::vector<Account*> m_accounts;
	NeuralRecommender m_recommender; // The recommender system
	GFileCache m_fileCache;
	MyJaad* m_pJaad;

public:
	bool m_keepGoing;
	std::string m_basePath;
	std::string m_toolsPath;

	Server(int port, GRand* pRand);
	virtual ~Server();
	NeuralRecommender& recommender() { return m_recommender; }
	GJsonAsADatabase& jaad() { return *m_pJaad; }
	void saveState();
	void doSomeRecommenderTraining();
	virtual void onStateChange();
	virtual void onShutDown();
	std::vector<Account*>& accounts() { return m_accounts; }
	bool isUsernameTaken(const char* szUsername);
	Account* findAccount(const char* szUsername);
	Account* get_account(size_t index) { return m_accounts[index]; }
	size_t user_id(const char* szUsername);
	bool isValidUsername(const char* szUsername);
	Account* newAccount(const char* szUsername, const char* szPasswordHash);
	//void deleteAccount(Account* pAccount);
	bool onRenameAccount(const char* szOldName, const char* szNewName);
	GDomNode* serializeState(GDom* pDoc);
	void deserializeState(const GDomNode* pNode);
	virtual GDynamicPageSessionExtension* deserializeSessionExtension(const GDomNode* pNode);
	virtual GDynamicPageConnection* makeConnection(SOCKET sock);
	std::string& cache(const char* szFilename);
	static void getLocalStorageFolder(char* buf);

	/// Make the header part of standardized pages
	static void makeHeader(GDynamicPageSession* pSession, std::ostream& response, const char* szParamsMessage);

	/// Make the footer part of standardized pages
	static void makeFooter(GDynamicPageSession* pSession, std::ostream& response);

	/// Prints a timestamp and message to stdout, and flushes stdout
	void log(const char* message);

	/// This is called at periodic intervals, a few times per day
	void do_maintenance();
};






class Account
{
protected:
	std::string m_username;
	std::string m_passwordHash;
	std::string m_path; // Within .../usercontent/username/pages/. Does not start with a slash.
	User* m_pUser; // A redundant pointer to the User object that goes with this account
	bool m_admin;
	bool m_assistant;
	bool m_banned;

public:
	Account(const char* szUsername, const char* szPasswordHash)
	: m_username(szUsername), m_passwordHash(szPasswordHash), m_pUser(nullptr), m_admin(false), m_assistant(false), m_banned(false)
	{
	}

	virtual ~Account()
	{
	}

	bool isAdmin()
	{
		return m_admin;
	}

	bool isAssistant()
	{
		return m_assistant;
	}

	bool isBanned()
	{
		return m_banned;
	}

	void makeAdmin(bool admin)
	{
		m_admin = admin;
	}

	void makeAssistant(bool assistant)
	{
		m_assistant = assistant;
	}

	void makeBanned(bool banned)
	{
		m_banned = banned;
	}

	static Account* fromDom(GDomNode* pNode, GRand& rand)
	{
		const char* un = pNode->getString("username");
		const char* pw = pNode->getString("password");
		Account* pAccount = new Account(un, pw);
		if(pNode->getIfExists("admin"))
			pAccount->makeAdmin(true);
		if(pNode->getIfExists("assistant"))
			pAccount->makeAssistant(true);
		if(pNode->getIfExists("banned"))
			pAccount->makeBanned(true);
		return pAccount;
	}

	GDomNode* toDom(GDom* pDoc)
	{
		GDomNode* pAccount = pDoc->newObj();
		pAccount->add(pDoc, "username", m_username.c_str());
		pAccount->add(pDoc, "password", m_passwordHash.c_str());
		if(m_admin)
			pAccount->add(pDoc, "admin", "true");
		if(m_assistant)
			pAccount->add(pDoc, "assistant", "true");
		if(m_banned)
			pAccount->add(pDoc, "banned", "true");
		return pAccount;
	}

	std::string& path() { return m_path; }
	const char* username() { return m_username.c_str(); }
	const char* passwordHash() { return m_passwordHash.c_str(); }

	User* getUser(NeuralRecommender& recommender)
	{
		if(!m_pUser)
			m_pUser = recommender.findOrMakeUser(m_username.c_str());
		return m_pUser;
	}

	User* user() { return m_pUser; }
	void setUser(User* pUser) { m_pUser = pUser; }

	bool changeUsername(Server* pServer, const char* szNewName)
	{
		if(!pServer->onRenameAccount(m_username.c_str(), szNewName))
			return false;
		m_username = szNewName;
		return true;
	}

	void changePasswordHash(Server* pServer, const char* szNewPasswordHash)
	{
		m_passwordHash = szNewPasswordHash;
	}

	bool doesHavePassword()
	{
		return m_passwordHash.length() > 0;
	}
};




class Terminal : public GDynamicPageSessionExtension
{
protected:
	size_t m_currentAccount; // The currently logged-in account. (nullptr if not logged in.)
	std::vector<Account*> m_accounts; // All accounts associated with this machine
	std::vector<bool> m_requirePassword; // Whether or not the password is required on this machine

public:
	Terminal() : GDynamicPageSessionExtension(), m_currentAccount((size_t)-1)
	{
	}

	Terminal(const GDomNode* pNode, Server* pServer)
	{
		m_currentAccount = pNode->getInt("loggedin");
		GDomListIterator it(pNode->get("accounts"));
		while(it.remaining() > 0)
		{
			const char* username = it.currentString();
			Account* pAccount = pServer->findAccount(username);
			if(!pAccount)
				throw Ex("Account not found for ", username);
			m_accounts.push_back(pAccount);
			it.advance();
		}
		GDomListIterator it2(pNode->get("reqpw"));
		while(it2.remaining() > 0)
		{
			bool b = it2.currentBool();
			m_requirePassword.push_back(b);
			it2.advance();
		}
		if(m_accounts.size() != m_requirePassword.size())
			throw Ex("Mismatching sizes");
	}

	virtual ~Terminal()
	{
		for(size_t i = 0; i < m_accounts.size(); i++)
			delete(m_accounts[i]);
	}

	/// Called when the sessions are destroyed, or a new GDynamicPageSessionExtension is
	/// explicitly associated with this cookie.
	virtual void onDisown()
	{
	}

	virtual GDomNode* serialize(GDom* pDoc)
	{
		GDomNode* pObj = pDoc->newObj();
		pObj->add(pDoc, "loggedin", m_currentAccount);
		GDomNode* pAccList = pDoc->newList();
		pObj->add(pDoc, "accounts", pAccList);
		for(size_t i = 0; i < m_accounts.size(); i++)
			pAccList->add(pDoc, m_accounts[i]->username());
		GDomNode* pReqList = pDoc->newList();
		pObj->add(pDoc, "reqpw", pReqList);
		for(size_t i = 0; i < m_requirePassword.size(); i++)
			pReqList->add(pDoc, m_requirePassword[i]);
		return pObj;
	}

	/// Returns the current account, or nullptr if not logged in.
	Account* currentAccount()
	{
		return (m_currentAccount == (size_t)-1 ? nullptr : m_accounts[m_currentAccount]);
	}

	size_t accountCount() { return m_accounts.size(); }
	Account* account(size_t index) { return m_accounts[index]; }
	bool requirePassword() { return m_requirePassword[m_currentAccount]; }
	bool requirePassword(size_t index) { return m_requirePassword[index]; }

	void logOut()
	{
		m_currentAccount = (size_t)-1;
	}

	bool forgetAccount(const char* szUsername)
	{
		for(size_t i = 0; i < m_accounts.size(); i++)
		{
			if(strcmp(m_accounts[i]->username(), szUsername) == 0)
			{
				m_accounts.erase(m_accounts.begin() + i);
				m_requirePassword.erase(m_requirePassword.begin() + i);
				return true;
			}
		}
		return false;
	}

	static std::string generateUsername(GRand& rand);

	Account* makeNewAccount(Server* pServer)
	{
		Account* pNewAccount = nullptr;
		for(size_t patience = 10; patience > 0; patience--)
		{
			std::string userName = generateUsername(*pServer->prng());
			if(!pServer->findAccount(userName.c_str()))
			{
				pNewAccount = pServer->newAccount(userName.c_str(), nullptr);
				if(!pNewAccount)
					throw Ex("Failed to create account");
				break;
			}
		}
		if(!pNewAccount)
			throw Ex("Failed to generate a unique username");
		m_currentAccount = m_accounts.size();
		m_accounts.push_back(pNewAccount);
		m_requirePassword.push_back(false);
		if(m_currentAccount == 0)
			pNewAccount->makeAdmin(true);
		return pNewAccount;
	}

	void setRequirePassword(bool require)
	{
		if(m_currentAccount == (size_t)-1)
			throw Ex("not logged in");
		m_requirePassword[m_currentAccount] = require;
	}

	bool logIn(Account* pAccount, const char* szPasswordHash)
	{
		bool found = false;
		bool requirePassword = true;
		size_t accountIndex = (size_t)-1;
		for(size_t i = 0; i < m_accounts.size() && !found; i++)
		{
			if(m_accounts[i] == pAccount)
			{
				found = true;
				accountIndex = i;
				requirePassword = m_requirePassword[i];
			}
		}
		if(requirePassword)
		{
			if(!pAccount->passwordHash())
				return false;
			if(strlen(pAccount->passwordHash()) < 1)
				return false;
			if(!szPasswordHash)
				return false;
			if(strcmp(pAccount->passwordHash(), szPasswordHash) != 0)
				return false;
		}
		if(!found)
		{
			accountIndex = m_accounts.size();
			m_accounts.push_back(pAccount);
			m_requirePassword.push_back(requirePassword);
		}
		m_currentAccount = accountIndex;
		return true;
	}
};


class Connection : public GDynamicPageConnection
{
public:
	Connection(SOCKET sock, GDynamicPageServer* pServer) : GDynamicPageConnection(sock, pServer)
	{
	}

	virtual ~Connection()
	{
	}

	virtual void handleRequest(GDynamicPageSession* pSession, std::ostream& response);
	void handleAjax(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);
	void handleTools(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);
	bool is_admin(Terminal* pTerminal, const char** ppParamsMessage);
	const char* processParams(GDynamicPageSession* pSession);
};







Terminal* getTerminal(GDynamicPageSession* pSession);

Account* getAccount(GDynamicPageSession* pSession);



#endif // SERVER
