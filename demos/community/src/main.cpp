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
#include <GClasses/GMatrix.h>
#include <GClasses/GDirList.h>
#include <GClasses/GApp.h>
#include <GClasses/GDom.h>
#include <GClasses/GString.h>
#include <GClasses/GHeap.h>
#include <GClasses/GHttp.h>
#include <GClasses/GFile.h>
#include <GClasses/GTime.h>
#include <GClasses/GPlot.h>
#include <GClasses/GThread.h>
#include <GClasses/GRand.h>
#include <GClasses/GHashTable.h>
#include <GClasses/sha1.h>
#include <GClasses/GVec.h>
#include <GClasses/GHolders.h>
#include <GClasses/GBitTable.h>
#include <wchar.h>
#include <math.h>
#include <string>
#include <vector>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <cmath>
#include "recommender.h"

using namespace GClasses;
using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::ostream;
using std::map;
using std::set;
using std::pair;
using std::make_pair;
using std::multimap;

class View;
class Account;
class ViewStats;


const char* g_auto_name_1[] =
{
	"amazing",
	"awesome",
	"blue",
	"brave",
	"calm",
	"cheesy",
	"confused",
	"cool",
	"crazy",
	"delicate",
	"diligent",
	"dippy",
	"exciting",
	"fearless",
	"flaming",
	"fluffy",
	"friendly",
	"funny",
	"gentle",
	"glowing",
	"golden",
	"greasy",
	"green",
	"gritty",
	"happy",
	"jumpy",
	"killer",
	"laughing",
	"liquid",
	"lovely",
	"lucky",
	"malted",
	"meaty",
	"mellow",
	"melted",
	"moldy",
	"peaceful",
	"pickled",
	"pious",
	"purple",
	"quiet",
	"red",
	"rubber",
	"sappy",
	"silent",
	"silky",
	"silver",
	"sneaky",
	"stellar",
	"subtle",
	"super",
	"trippy",
	"uber",
	"valiant",
	"vicious",
	"wild",
	"yellow",
	"zippy"
};
#define AUTO_NAME_1_COUNT (sizeof(g_auto_name_1) / sizeof(const char*))

const char* g_auto_name_2[] =
{
	"alligator",
	"ant",
	"armadillo",
	"bat",
	"bear",
	"bee",
	"beaver",
	"camel",
	"cat",
	"cheetah",
	"chicken",
	"cricket",
	"deer",
	"dinosaur",
	"dog",
	"dolphin",
	"duck",
	"eagle",
	"elephant",
	"fish",
	"frog",
	"giraffe",
	"hamster",
	"hawk",
	"hornet",
	"horse",
	"iguana",
	"jaguar",
	"kangaroo",
	"lion",
	"lemur",
	"leopard",
	"llama",
	"monkey",
	"mouse",
	"newt",
	"ninja",
	"ox",
	"panda",
	"panther",
	"parrot",
	"porcupine",
	"possum",
	"raptor",
	"rat",
	"salmon",
	"shark",
	"snake",
	"spider",
	"squid",
	"tiger",
	"toad",
	"toucan",
	"turtle",
	"unicorn",
	"walrus",
	"warrior",
	"wasp",
	"wizard",
	"yak",
	"zebra"
};
#define AUTO_NAME_2_COUNT (sizeof(g_auto_name_2) / sizeof(const char*))

const char* g_auto_name_3[] =
{
	"arms",
	"beak",
	"beard",
	"belly",
	"belt",
	"brain",
	"bray",
	"breath",
	"brow",
	"burrito",
	"button",
	"cheeks",
	"chin",
	"claw",
	"crown",
	"dancer",
	"dream",
	"eater",
	"elbow",
	"eye",
	"feather",
	"finger",
	"fist",
	"foot",
	"forehead",
	"fur",
	"grin",
	"hair",
	"hands",
	"head",
	"horn",
	"jaw",
	"knee",
	"knuckle",
	"legs",
	"mouth",
	"neck",
	"nose",
	"pants",
	"party",
	"paw",
	"pelt",
	"pizza",
	"roar",
	"scalp",
	"shoe",
	"shoulder",
	"skin",
	"smile",
	"taco",
	"tail",
	"tamer",
	"toe",
	"tongue",
	"tooth",
	"wart",
	"wing",
	"zit"
};
#define AUTO_NAME_3_COUNT (sizeof(g_auto_name_3) / sizeof(const char*))



string to_string(double d, int decimal_places)
{
	double p = pow(10, decimal_places);
	d *= p;
	d += 0.5;
	d = floor(d);
	d /= p;
	return to_str(d);
}


class Server : public GDynamicPageServer
{
protected:
	std::map<std::string,Account*> m_accounts; // Mapping from username to account
	Recommender m_recommender; // The recommender system

public:
	std::string m_basePath;

	Server(int port, GRand* pRand);
	virtual ~Server();
	Recommender& recommender() { return m_recommender; }
	void saveState();
	virtual void onEverySixHours();
	virtual void onStateChange();
	virtual void onShutDown();
	bool isUsernameTaken(const char* szUsername);
	Account* findAccount(const char* szUsername);
	Account* newAccount(const char* szUsername, const char* szPasswordHash);
	void deleteAccount(Account* pAccount);
	void onRenameAccount(const char* szOldName, Account* pAccount);
	GDomNode* serializeState(GDom* pDoc);
	void deserializeState(const GDomNode* pNode);
	virtual GDynamicPageSessionExtension* deserializeSessionExtension(const GDomNode* pNode);
	virtual GDynamicPageConnection* makeConnection(SOCKET sock);
};






class Account
{
protected:
	string m_username;
	string m_passwordHash;
	size_t m_currentTopic;
	User* m_pUser; // A redundant pointer to the User object that goes with this account
	bool m_admin;

public:
	Account(const char* szUsername, const char* szPasswordHash)
	: m_username(szUsername), m_passwordHash(szPasswordHash), m_currentTopic(-1), m_pUser(nullptr), m_admin(false)
	{
	}

	virtual ~Account()
	{
	}

	bool isAdmin()
	{
		return m_admin;
	}

	void makeAdmin(bool admin)
	{
		m_admin = admin;
	}

	static Account* fromDom(GDomNode* pNode, GRand& rand)
	{
		const char* un = pNode->getString("username");
		const char* pw = pNode->getString("password");
		Account* pAccount = new Account(un, pw);
		if(pNode->getIfExists("admin"))
			pAccount->makeAdmin(true);
		return pAccount;
	}

	GDomNode* toDom(GDom* pDoc)
	{
		GDomNode* pAccount = pDoc->newObj();
		pAccount->add(pDoc, "username", m_username.c_str());
		pAccount->add(pDoc, "password", m_passwordHash.c_str());
		if(m_admin)
			pAccount->add(pDoc, "admin", "true");
		return pAccount;
	}

	const char* username() { return m_username.c_str(); }
	const char* passwordHash() { return m_passwordHash.c_str(); }
	size_t currentTopic() { return m_currentTopic; }
	void setCurrentTopic(size_t topic) { m_currentTopic = topic; }

	User* getUser(Recommender& recommender)
	{
		if(!m_pUser)
			m_pUser = recommender.findOrMakeUser(m_username.c_str());
		return m_pUser;
	}
	
	User* user() { return m_pUser; }
	void setUser(User* pUser) { m_pUser = pUser; }

	bool changeUsername(Server* pServer, const char* szNewName)
	{
		if(pServer->findAccount(szNewName))
			return false;
		string oldName = m_username;
		m_username = szNewName;
		pServer->onRenameAccount(oldName.c_str(), this);
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

	static string generateUsername(GRand& rand)
	{
		string s = g_auto_name_1[rand.next(AUTO_NAME_1_COUNT)];
		s += " ";
		s += g_auto_name_2[rand.next(AUTO_NAME_2_COUNT)];
		s += " ";
		s += g_auto_name_3[rand.next(AUTO_NAME_3_COUNT)];
		return s;
	}

	Account* makeNewAccount(Server* pServer)
	{
		Account* pNewAccount = nullptr;
		for(size_t patience = 10; patience > 0; patience--)
		{
			string userName = generateUsername(*pServer->prng());
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



Terminal* getTerminal(GDynamicPageSession* pSession)
{
	Terminal* pTerminal = (Terminal*)pSession->extension();
	if(!pTerminal)
	{
		pTerminal = new Terminal();
		pSession->setExtension(pTerminal);
	}
	return pTerminal;
}

Account* getAccount(GDynamicPageSession* pSession)
{
	// Get the terminal
	Terminal* pTerminal = getTerminal(pSession);
	Account* pAccount = pTerminal->currentAccount();
	GAssert(!pAccount || pAccount->username()[0] != '\0'); // every account should have a unique username
	return pAccount;
}


class ItemStats
{
protected:
	Item& m_item;
	size_t m_id;
	unsigned int m_agree, m_uncertain, m_disagree;
	unsigned int m_agg, m_dis;
	double m_deviation;

public:
	ItemStats(size_t topicId, Item& itm, size_t itemId, User** pUsers, size_t accCount)
	: m_item(itm), m_id(itemId), m_agree(0), m_uncertain(0), m_disagree(0), m_agg(0), m_dis(0)
	{
		// Compute the mean
		User** Us = pUsers;
		float rating;
		double mean = 0.0;
		size_t count = 0;
		for(size_t i = 0; i < accCount; i++)
		{
			if((*Us)->getRating(topicId, itemId, &rating))
			{
				mean += rating;
				count++;
				if(rating < -0.333333)
					m_disagree++;
				else if(rating > 0.333333)
					m_agree++;
				else
					m_uncertain++;
				if(rating < 0.5)
					m_dis++;
				else
					m_agg++;
			}
			Us++;
		}
		mean /= count;

		// Compute the deviation
		Us = pUsers;
		double var = 0.0;
		for(size_t i = 0; i < accCount; i++)
		{
			if((*Us)->getRating(topicId, itemId, &rating))
			{
				double d = mean - rating;
				var += (d * d);
			}
			Us++;
		}
		m_deviation = sqrt(var / count);
	}

	Item& item() { return m_item; }
	size_t id() { return m_id; }
	unsigned int disagree() { return m_disagree; }
	unsigned int uncertain() { return m_uncertain; }
	unsigned int agree() { return m_agree; }
	unsigned int split() { return std::min(m_agg, m_dis); }

	double controversy() const
	{
		return m_deviation * (m_agg + m_dis);
	}

	static bool comparer(const ItemStats* pA, const ItemStats* pB)
	{
		return pA->controversy() > pB->controversy();
	}
};

class UpdateComparer
{
public:
	UpdateComparer()
	{
	}

	bool operator() (const pair<size_t,float>& a, const pair<size_t,float>& b) const
	{
		return a.second > b.second;
	}
};


bool str_endswith(const char* szFull, const char* szTail)
{
	size_t lenFull = strlen(szFull);
	size_t lenTail = strlen(szTail);
	if(lenTail > lenFull)
		return false;
	if(strcmp(szFull + lenFull - lenTail, szTail) == 0)
		return true;
	return false;
}





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
	void handleAjax(GDynamicPageSession* pSession, ostream& response);
	const char* processParams(GDynamicPageSession* pSession);

	void makeHeader(GDynamicPageSession* pSession, ostream& response, const char* szParamsMessage)
	{
		//response << "<!DOCTYPE html>";
		response << "<html><head>\n";
		Account* pAccount = getAccount(pSession);
		response << "	<title>Community Modeler</title>\n";
		response << "	<link rel=\"stylesheet\" type=\"text/css\" href=\"/style/style.css\" />\n";
		response << "</head><body>\n";
		response << "<table align=center width=1200 cellpadding=0 cellspacing=0><tr><td>\n";
		response << "<table cellpadding=0 cellspacing=0>\n";

		// The header row
		response << "<tr><td colspan=2 id=\"header\">";
		if(pAccount)
		{
			response << "Welcome, ";
			const char* szUsername = pAccount->username();
			response << szUsername;
			response << ".";
		}
		else
		{
			response << "Please log in.";
		}
		if(szParamsMessage)
		{
			response << "<br>" << szParamsMessage;
		}
		response << "</td></tr>\n";

		// The main row
		response << "<tr>\n";

		// The left sidebar
		response << "<td id=\"sidebar\">";
		if(pAccount)
		{
			response << "	<a href=\"/survey\">Survey</a><br><br>\n";
			response << "	<a href=\"/account?action=logout\">Log out</a><br><br>\n";
			response << "	<a href=\"/account\">Account</a><br><br>\n";
			if(pAccount->isAdmin())
				response << "	<a href=\"/admin\">Admin</a><br><br>\n";
		}
		else
		{
			response << "	<a href=\"/account?action=newaccount\">New account</a><br><br>\n";
		}
	//	response << "	<a href=\"/main.hbody\">Overview</a><br>\n";
		response << "</td><td id=\"mainbody\">\n\n\n\n";
	}

	void makeFooter(GDynamicPageSession* pSession, ostream& response)
	{
		// End of main row
		response << "\n\n\n\n</td></tr>\n";

		// Footer row
		response << "<tr><td colspan=2 id=\"footer\">\n";
		response << "</td></tr></table>\n";
		response << "</td></tr></table></body></html>\n";
	}

	void pageRedirect(GDynamicPageSession* pSession, ostream& response, const char* url)
	{
		// Attempt an HTTP redirect
		cout << "Redirecting from " << pSession->url() << " to " << url << "\n";
		m_pServer->redirect(response, url);

		// Do an HTML redirect as a backup
		response << "<html><head>";
		response << "<meta http-equiv=\"refresh\" content=\"0; url=" << url << "\">";
		response << "</head>\n";

		// Give the user a link as a last resort
		response << "<body>";
		response << "<a href=\"" << url << "\">Please click here to continue</a>\n";
		response << "</body></html>\n";
	}

	void makeSliderScript(ostream& response)
	{
		response << "<script language=\"JavaScript\" src=\"style/slider.js\"></script>\n";
		response << "<script language=\"JavaScript\">\n";
		response << "	var A_TPL = { 'b_vertical' : false, 'b_watch': true, 'n_controlWidth': 321, 'n_controlHeight': 22, 'n_sliderWidth': 19, 'n_sliderHeight': 20, 'n_pathLeft' : 1, 'n_pathTop' : 1, 'n_pathLength' : 300, 's_imgControl': 'style/slider_bg.png', 's_imgSlider': 'style/slider_tab.png', 'n_zIndex': 1 }\n";
		response << "</script>\n";
	}

	void makeUrlSlider(Account* pAccount, size_t itemId, ostream& response)
	{
		// Compute the rating (or predicted rating if this item has not been rated)
		size_t currentTopic = pAccount->currentTopic();
		Topic* pCurrentTopic = ((Server*)m_pServer)->recommender().topics()[currentTopic];
		Item& item = pCurrentTopic->item(itemId);
		float score;
		User* pUser = pAccount->getUser(((Server*)m_pServer)->recommender());
		if(!pUser->getRating(currentTopic, itemId, &score))
			score = pUser->predictRating(item);
/*		score *= 500.0;
		score += 500.0;
		score = 0.1 * floor(score);*/

		// Display the slider
		response << "<table cellpadding=0 cellspacing=0><tr><td width=300>\n	";
		response << item.left() << "\n";
		response << "</td><td>\n";
		response << "	<input type=checkbox name=\"check_slider" << itemId << "\" id=\"check_slider" << itemId << "\">\n";
		response << "	<input name=\"slider" << itemId << "\" id=\"slider" << itemId << "\" type=\"Text\" size=\"3\">\n";
		response << "</td><td>\n";
		response << "<script language=\"JavaScript\">\n";
		response << "	var A_INIT1 = { 's_checkname': 'check_slider" << itemId << "', 's_name': 'slider" << itemId << "', 'n_minValue' : -1, 'n_maxValue' : 1, 'n_value' : " << score << ", 'n_step' : 0.01 }\n";
		response << "	new slider(A_INIT1, A_TPL);\n";
		response << "</script>\n";
		response << "</td><td width=300>\n";
		response << item.right() << "\n";
		response << "</td></tr></table>\n";
	}


	virtual void pageSurvey(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		User* pUser = pAccount->getUser(((Server*)m_pServer)->recommender());
		size_t currentTopic = pAccount->currentTopic();
		if(pSession->paramsLen() > 0)
		{
			// Get the topic
			GHttpParamParser params(pSession->params());
			const char* szTopic = params.find("topic");
			if(szTopic)
			{
				const vector<Topic*>& topics = ((Server*)m_pServer)->recommender().topics();
#ifdef WINDOWS
				size_t i = (size_t)_strtoui64(szTopic, NULL, 10);
#else
				size_t i = (size_t)strtoull(szTopic, NULL, 10);
#endif
				if(i < topics.size())
					pAccount->setCurrentTopic(i);
				else
					pAccount->setCurrentTopic((size_t)-1);
				currentTopic = pAccount->currentTopic();
			}

			// Check for topic proposals
			const char* szProposal = NULL;
			if(pAccount->doesHavePassword())
				szProposal = params.find("proposal");
			if(szProposal)
				((Server*)m_pServer)->recommender().proposeTopic(pAccount->username(), szProposal);

			// Do the action
			if(currentTopic < ((Server*)m_pServer)->recommender().topics().size())
			{
				const char* szAction = params.find("action");
				if(!szAction)
				{
				}
				else if(_stricmp(szAction, "add") == 0)
				{
					const char* szLeft = params.find("left");
					const char* szRight = params.find("right");
					if(!szLeft || !szRight)
						response << "[invalid params]<br>\n";
					else
					{
						((Server*)m_pServer)->recommender().addItem(currentTopic, szLeft, szRight, pAccount->username());
						cout << pAccount->username() << "added: " << szLeft << " <-----> " << szRight << "\n";
						((Server*)m_pServer)->saveState();
					}
				}
				else if(_stricmp(szAction, "rate") == 0)
				{
					// Make an set of all the checked ids
					set<size_t> checks;
					map<const char*, const char*, strComp>& paramMap = params.map();
					for(map<const char*, const char*, strComp>::iterator it = paramMap.begin(); it != paramMap.end(); it++)
					{
						const char* szName = it->first;
						if(_strnicmp(szName, "check_slider", 12) == 0)
						{
#ifdef WINDOWS
							size_t itemId = (size_t)_strtoui64(szName + 12, NULL, 10);
#else
							size_t itemId = (size_t)strtoull(szName + 12, NULL, 10);
#endif
							checks.insert(itemId);
						}
					}

					// find the corresponding scores for each topic id, and add the rating
					for(map<const char*, const char*, strComp>::iterator it = paramMap.begin(); it != paramMap.end(); it++)
					{
						const char* szName = it->first;
						if(_strnicmp(szName, "slider", 6) == 0)
						{
#ifdef WINDOWS
							size_t itemId = (size_t)_strtoui64(szName + 6, NULL, 10);
#else
							size_t itemId = (size_t)strtoull(szName + 6, NULL, 10);
#endif
							set<size_t>::iterator tmp = checks.find(itemId);
							if(tmp != checks.end())
							{
								float score = (float)atof(it->second);
								if(score >= -1.0f && score <= 1.0f)
								{
									pUser->updateRating(currentTopic, itemId, score);
									response << "[Rating recorded. Thank you.]<br>\n";
								}
								else
									response << "[the rating of " << score << " is out of range.]<br>\n";
							}
						}
					}

					// Do some training
					((Server*)m_pServer)->recommender().refinePersonality(pUser, pAccount->currentTopic(), ON_RATE_TRAINING_ITERS); // trains just personalities
					((Server*)m_pServer)->recommender().refineModel(currentTopic, ON_RATE_TRAINING_ITERS); // trains both personalities and weights
					((Server*)m_pServer)->saveState();
				}
			}
		}

		if(currentTopic < ((Server*)m_pServer)->recommender().topics().size()) // if a topic has been selected...
		{
			Topic* pCurrentTopic = ((Server*)m_pServer)->recommender().topics()[currentTopic];
			
			// Display the topic
			makeSliderScript(response);
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";
			response << "<form name=\"formname\" action=\"/survey\" method=\"post\">\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";

			// Random picks
			size_t* pIndexes = new size_t[pCurrentTopic->size()];
			Holder<size_t> hIndexes(pIndexes);
			GIndexVec::makeIndexVec(pIndexes, pCurrentTopic->size());
			GIndexVec::shuffle(pIndexes, pCurrentTopic->size(), m_pServer->prng());
			size_t sliderCount = 0;
			for(size_t i = 0; i < pCurrentTopic->size(); i++)
			{
				if(sliderCount >= 8)
					break;
				size_t itemId = pIndexes[i];
				float rating;
				if(pUser->getRating(currentTopic, itemId, &rating))
					continue;
				if(sliderCount == 0)
				{
					response << "<h3>A few statements for your evaluation:</h3>\n";
					response << "<p>It is okay to skip statements you find ambiguous, invasive, or uninteresting. For your convenience, the sliders have been set to reflect predictions of your opinions. As you express more opinions, these predictions should improve.</p>\n";
				}
				makeUrlSlider(pAccount, itemId, response);
				response << "<br><br>\n";
				sliderCount++;
			}

			// The update ratings button
			if(sliderCount > 0)
			{
				response << "<br><table><tr><td width=330></td><td>";
				response << "<input type=\"submit\" value=\"Update opinions\">";
				response << "</td></tr><tr><td></td><td>";
				response << "(Only checked items will be updated.)";
				response << "</td></tr></table>\n";
			}
			else
			{
				if(pCurrentTopic->size() == 0)
				{
					response << "There are not yet any survey questions in this topic.<br><br>\n";
				}
				else
				{
					response << "Thank you. You have expressed your opinion about all ";
					response << to_str(pCurrentTopic->size());
					response << " survey statements in this topic.<br><br>\n";
				}
			}

			response << "</form><br><br>\n\n";

			// The choices links at the bottom of the page
			response << "<a href=\"/submit\">Submit a new statement</a>";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/survey?topic=-1\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/update\">My opinions</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/stats\">Vizualize</a>\n";

/*
			response << "Stats:<br>\n";
			response << "Total Number of users: " << ((Server*)m_pServer)->accounts().size() << "<br>\n";
			response << "Number of items in this topic: " << pCurrentTopic->size() << "<br>\n";
			std::map<size_t, float>* pMap = currentTopic < pAccount->ratings().size() ? pAccount->ratings()[currentTopic] : NULL;
			response << "Number of items you have rated in this topic: " << (pMap ? pMap->size() : (size_t)0) << "<br>\n<br>\n";
*/
		}
		else
		{
			const vector<Topic*>& topics = ((Server*)m_pServer)->recommender().topics();
			response << "<h3>Choose a topic:</h3>\n";
			if(topics.size() > 0)
			{
				response << "<ul>\n";
				size_t i = 0;
				for(vector<Topic*>::const_iterator it = topics.begin(); it != topics.end(); it++)
				{
					response << "	<li><a href=\"/survey?topic=" << i << "\">" << (*it)->descr() << "</a></li>\n";
					i++;
				}
				response << "</ul><br><br><br>\n";
			}
			else
			{
				response << "There are currently no topics. Please ";
				if(!pAccount->isAdmin())
					response << "ask the administrator to ";
				response << "go to the <a href=\"/admin\">admin</a> page and add at least one topic.<br><br><br>";
			}
			response << "<br><br>\n";
/*
			// Make the form to propose new topics
			if(!pAccount->isAdmin())
			{
				response << "<form name=\"propose\" action=\"/survey\" method=\"get\">\n";
				response << "	<h3>Propose a new topic:</h3>\n";
				response << "	<input type=\"text\" name=\"proposal\" size=\"55\"><input type=\"submit\" value=\"Submit\"><br>\n";
				response << "	(Your proposed topic will be added to a log file. Hopefully, someone actually reads the log file.)\n";
				response << "</form><br>\n\n";
			}
*/
		}
	}

	virtual void pageSubmit(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->recommender().topics().size())
		{
			m_pServer->redirect(response, "/survey");
		}
		else
		{
			// Display the topic
			Topic* pCurrentTopic = ((Server*)m_pServer)->recommender().topics()[currentTopic];
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";

			// Make the form to submit a new item
			response << "<h3>Submit a new survey question to this topic</h3>\n";
			response << "<form name=\"formname\" action=\"/survey\" method=\"post\">\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"add\" />\n";
			response << "Left Statement: <input type=\"text\" name=\"left\" size=\"40\"><br>\n";
			response << "Opposing right Statement:<input type=\"text\" name=\"right\" size=\"40\"><br>\n";
			response << "	<input type=\"submit\" value=\"Submit\">";
			response << "</form><br><br>\n\n";

			// The choices links at the bottom of the page
			response << "<br>\n";
			response << "<a href=\"/survey?topic=-1\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/update\">My opinions</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/survey\">" << "Survey</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/stats\">Vizualize</a>\n";
		}
	}

	double computeVariance(double* pCentroid, Topic& topic, size_t topicId, User** pUsers, size_t accCount)
	{
		// Compute the centroid
		GVec::setAll(pCentroid, 0.0, topic.size());
		User** pUs = pUsers;
		for(size_t j = 0; j < accCount; j++)
		{
			double* pC = pCentroid;
			for(size_t i = 0; i < topic.size(); i++)
			{
				float rating;
				if(!(*pUs)->getRating(topicId, i, &rating))
					rating = (*pUs)->predictRating(topic.item(i));
				(*pC) += rating;
				pC++;
			}
			pUs++;
		}
		double t = 1.0 / accCount;
		for(size_t i = 0; i < topic.size(); i++)
			pCentroid[i] *= t;

		// Measure the sum-squared error with the centroid
		double sse = 0.0;
		pUs = pUsers;
		for(size_t j = 0; j < accCount; j++)
		{
			double* pC = pCentroid;
			for(size_t i = 0; i < topic.size(); i++)
			{
				float rating;
				if(!(*pUs)->getRating(topicId, i, &rating))
					rating = (*pUs)->predictRating(topic.item(i));
				double d = *pC - rating;
				sse += (d * d);
				pC++;
			}
			pUs++;
		}
		return sse;
	}

	size_t divideAccounts(Topic& topic, size_t topicId, User** pUsers, size_t accCount, size_t itm)
	{
		size_t head = 0;
		size_t tail = accCount;
		while(tail > head)
		{
			float rating;
			if(!pUsers[head]->getRating(topicId, itm, &rating))
				rating = pUsers[head]->predictRating(topic.item(itm));
			if(rating > 0.0)
			{
				tail--;
				std::swap(pUsers[head], pUsers[tail]);
			}
			else
				head++;
		}
		GAssert(head == tail);
		return head;
	}

	void makeTree(Topic& topic, size_t topicId, GBitTable& bt, User** pUsers, size_t accCount, ostream& response, vector<char>& prefix, int type)
	{
		// Try splitting on each of the remaining statements
		size_t best = (size_t)-1;
		double mostCont = 0.0;
		double* pCentroid = new double[topic.size()];
		ArrayHolder<double> hCentroid(pCentroid);
		size_t tieCount = 0;
		for(size_t i = 0; i < topic.size(); i++)
		{
			if(bt.bit(i))
				continue;
			ItemStats is(topicId, topic.item(i), i, pUsers, accCount);
			double c = is.controversy();
			if(is.split() > 0)
			{
				if(c > mostCont)
				{
					mostCont = c;
					best = i;
					tieCount = 0;
				}
				else if(c == mostCont)
				{
					tieCount++;
					if(m_pServer->prng()->next(tieCount + 1) == 0)
						best = i;
				}
			}
		}

		if(best != (size_t)-1)
		{
			// Divide on the best statement
			size_t firstHalfSize = divideAccounts(topic, topicId, pUsers, accCount, best);
			bt.set(best);

			// Recurse
			prefix.push_back(' ');
			if(type >= 0) prefix.push_back(' '); else prefix.push_back('|');
			prefix.push_back(' ');
			prefix.push_back(' ');
			makeTree(topic, topicId, bt, pUsers, firstHalfSize, response, prefix, 1);

			for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
				response << *it;
			response << "/ (" << topic.item(best).left() << ")\n";
			prefix.pop_back(); prefix.pop_back(); prefix.pop_back(); prefix.pop_back();
			for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
				response << *it;
			if(type == 0)
				response << "---(\n";
			else
				response << " +-(\n";
			prefix.push_back(' ');
			if(type <= 0) prefix.push_back(' '); else prefix.push_back('|');
			prefix.push_back(' ');
			prefix.push_back(' ');
			for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
				response << *it;
			response << "\\ (" << topic.item(best).right() << ")\n";

			makeTree(topic, topicId, bt, pUsers + firstHalfSize, accCount - firstHalfSize, response, prefix, -1);
			prefix.pop_back(); prefix.pop_back(); prefix.pop_back(); prefix.pop_back();

			bt.unset(best);
		}
		else
		{
			for(size_t j = 0; j < accCount; j++)
			{
				for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
					response << *it;
				response << " +-&gt;<a href=\"/stats?user=" << pUsers[j]->username() << "\">";
				response << pUsers[j]->username() << "</a>\n";
			}
		}
	}

	virtual void makeItemBody(GDynamicPageSession* pSession, ostream& response, size_t topicId, size_t itemId, Item& item, User** pUsers, size_t accCount)
	{
		std::multimap<double,User*> mm;
		while(accCount > 0)
		{
			float rating;
			if((*pUsers)->getRating(topicId, itemId, &rating))
				mm.insert(std::pair<double,User*>(rating,*pUsers));
			accCount--;
			pUsers++;
		}

		// First show all the left-leaning answers from the sorted map
		response << "<h3>" << item.left() << "</h3>\n";
		response << "<table>\n";
		size_t hh = 0;
		for(std::multimap<double,User*>::iterator it = mm.begin(); it != mm.end(); it++)
		{
			if(hh == 0 && it->first > -0.3333) // When they cross into being uncertain
			{
				response << "</table>\n<h3>Uncertain</h3>\n<table>\n";
				hh++;
			}
			if(hh == 1 && it->first > 0.3333) // when they cross into being right-learning
			{
				response << "</table>\n<h3>" << item.right() << "</h3>\n<table>\n";
				hh++;
			}
			response << "<tr><td>" << it->second->username() << "</td><td>" << to_string(it->first, 2) << "</td></tr>\n";
		}

		if(hh == 0) // in the obscure case where we haven't found any uncertain people yet...
		{
			response << "</table>\n<h3>Uncertain</h3>\n<table>\n";
			hh++;
		}
		if(hh == 1) // in the obscure case where we haven't found any right-leaning people yet...
		{
			response << "</table>\n<h3>" << item.right() << "</h3>\n<table>\n";
			hh++;
		}
		response << "</table>\n";
	}

	virtual void makeUserBody(GDynamicPageSession* pSession, ostream& response, User* pA, User* pB, size_t topicId, Topic& topic)
	{
		std::multimap<float,size_t> m;
		float rA = 0.0f;
		float rB = 0.0f;
		for(size_t i = 0; i < topic.size(); i++)
		{
			if(pA->getRating(topicId, i, &rA))
			{
				if(pB->getRating(topicId, i, &rB))
					m.insert(std::pair<float,size_t>(-std::abs(rB - rA), i));
			}
		}
		if(m.size() == 0)
		{
			response << "You have no ratings in common.<br><br>\n";
			return;
		}
		response << "<table><tr><td><u>" << pA->username() << "</u></td><td><u>" << pB->username() << "</u></td><td><u>product</u></td><td><u>Left</u></td><td><u>Right</u></td></tr>\n";
		for(std::multimap<float,size_t>::iterator it = m.begin(); it != m.end(); it++)
		{
			pA->getRating(topicId, it->second, &rA);
			pB->getRating(topicId, it->second, &rB);
			response << "<tr><td>" << to_str(0.1 * floor(10 * rA)) << "</td><td>" << to_str(0.1 * floor(10 * rB)) << "</td><td>" << to_str(0.1 * floor(10 * rA * rB)) << "</td><td>" << topic.item(it->second).left() << "</td><td>" << topic.item(it->second).right() << "</td></tr>\n";
		}
		response << "</table>\n";
	}

	void pageStats(GDynamicPageSession* pSession, ostream& response)
	{
		// Get the topic
		Account* pAccount = getAccount(pSession);
		User* pUser = pAccount->getUser(((Server*)m_pServer)->recommender());
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->recommender().topics().size())
		{
			response << "Unrecognized topic.";
			return;
		}
		Topic& topic = *((Server*)m_pServer)->recommender().topics()[currentTopic];

		// Copy the account pointers into an array
		const std::vector<User*>& users = ((Server*)m_pServer)->recommender().users();
		User** pAccs = new User*[users.size()];
		ArrayHolder<User*> hAccs(pAccs);
		User** pAc = pAccs;
		size_t accountCount = 0;
		for(std::vector<User*>::const_iterator it = users.begin(); it != users.end(); it++)
		{
			*(pAc++) = *it;
			accountCount++;
		}

		// Check the params
		GHttpParamParser params(pSession->params());
		const char* szItemId = params.find("item");
		if(szItemId)
		{
			size_t itemId = atoi(szItemId);
			makeItemBody(pSession, response, currentTopic, itemId, topic.item(itemId), pAccs, accountCount);
			return;
		}
		const char* szOtherUser = params.find("user");
		if(szOtherUser)
		{
			User* pOther = ((Server*)m_pServer)->recommender().findUser(szOtherUser);
			if(!pOther)
				response << "[No such user]<br><br>\n";
			else
				makeUserBody(pSession, response, pUser, pOther, currentTopic, topic);
			return;
		}

		GBitTable bt(topic.size());
		vector<char> prefix;
		response << "This ascii-art tree was constructed by dividing on the most controversial statements within each branch.\n";
		response << "This tree is arranged such that the ascending branches lead to the usernames of people who support the left statement, and the descending branches lead to the usernames of people who support the right statement.\n";
		response << "(In cases lacking response, predictions were used to make any judgement calls necessary to construct this tree, so some placements may be estimated.)\n";
		response << "<br><br>\n";
		response << "<pre>\n";
		makeTree(topic, currentTopic, bt, pAccs, accountCount, response, prefix, 0);
		response << "</pre>\n";
		response << "<br><br>\n";

		// Make a table of items sorted by controversy
		std::vector<ItemStats*> items;
		for(size_t i = 0; i < topic.size(); i++)
			items.push_back(new ItemStats(currentTopic, topic.item(i), i, pAccs, accountCount));
		sort(items.begin(), items.end(), ItemStats::comparer);
		response << "<table><tr><td><b><i><u>Statement</u></i></b></td><td><b><i><u>Lean Left</u></i></b></td><td><b><i><u>Uncertain</u></i></b></td><td><b><i><u>Lean Right</u></i></b></td><td><b><i><u>Controversy</u></i></b></td></tr>\n";
		for(vector<ItemStats*>::iterator it = items.begin(); it != items.end(); it++)
		{
			response << "<tr><td>";
			response << "<a href=\"/stats?item=" << to_str((*it)->id()) << "\">" << (*it)->item().left() << " / " << (*it)->item().right() << "</a>";
			response << "</td><td>";
			response << to_str((*it)->disagree());
			response << "</td><td>";
			response << to_str((*it)->uncertain());
			response << "</td><td>";
			response << to_str((*it)->agree());
			response << "</td><td>";
			response << to_str((*it)->controversy());
			response << "</td></tr>\n";
			delete(*it);
		}
		response << "</table><br><br>\n";

		response << "<h3>A vizualization of the users in this community:</h3>\n";
		response << "<img src=\"users.svg\"><br><br>\n";
		response << "<h3>A vizualization of the items in this community:</h3>\n";
		response << "<img src=\"items.svg\"><br><br>\n";

		// The choices links at the bottom of the page
		response << "<a href=\"/submit\">Submit a new question</a>";
		response << "&nbsp;&nbsp;&nbsp;&nbsp;";
		response << "<a href=\"/survey?topic=-1\">Change topic</a>\n";
		response << "&nbsp;&nbsp;&nbsp;&nbsp;";
		response << "<a href=\"/update\">My opinions</a>\n";
		response << "&nbsp;&nbsp;&nbsp;&nbsp;";
		response << "<a href=\"/survey\">" << "Survey</a>\n";
	}

	void plotUsers(GDynamicPageSession* pSession, ostream& response)
	{
		setContentType("image/svg+xml");
		GSVG svg(800, 800);

		const std::vector<User*>& users = ((Server*)m_pServer)->recommender().users();
		double xmin = 0;
		double ymin = 0;
		double xmax = 0;
		double ymax = 0;
		for(std::vector<User*>::const_iterator it = users.begin(); it != users.end(); it++)
		{
			User* pUser = *it;
			vector<double>& profile = pUser->personality();
			xmin = std::min(xmin, profile[1]);
			xmax = std::max(xmax, profile[1]);
			ymin = std::min(ymin, profile[2]);
			ymax = std::max(ymax, profile[2]);
		}
		double wid = xmax - xmin;
		double hgt = ymax - ymin;
		xmin -= 0.1 * wid;
		xmax += 0.1 * wid;
		ymin -= 0.1 * hgt;
		ymax += 0.1 * hgt;
		if(xmax - xmin < 1e-4)
			xmax += 1e-4;
		if(ymax - ymin < 1e-4)
			ymax += 1e-4;
		svg.newChart(xmin, ymin, xmax, ymax, 0, 0, 0);
		for(std::vector<User*>::const_iterator it = users.begin(); it != users.end(); it++)
		{
			User* pUser = *it;
			vector<double>& profile = pUser->personality();
			svg.dot(profile[1], profile[2], 0.75, 0x008080);
			svg.text(profile[1], profile[2], (*it)->username(), 0.75);
		}
		svg.print(response);
	}

	void plotItems(GDynamicPageSession* pSession, ostream& response)
	{
		// Get the topic
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->recommender().topics().size())
		{
			response << "Unrecognized topic.";
			return;
		}
		Topic& topic = *((Server*)m_pServer)->recommender().topics()[currentTopic];

		setContentType("image/svg+xml");
		GSVG svg(800, 800);

		double xmin = 0;
		double ymin = 0;
		double xmax = 0;
		double ymax = 0;
		for(size_t i = 0; i < topic.size(); i++)
		{
			Item& item = topic.item(i);
			vector<double>& weights = item.weights();
			xmin = std::min(xmin, weights[1]);
			xmax = std::max(xmax, weights[1]);
			ymin = std::min(ymin, weights[2]);
			ymax = std::max(ymax, weights[2]);
		}
		double wid = xmax - xmin;
		double hgt = ymax - ymin;
		xmin -= 0.1 * wid;
		xmax += 0.1 * wid;
		ymin -= 0.1 * hgt;
		ymax += 0.1 * hgt;
		if(xmax - xmin < 1e-4)
			xmax += 1e-4;
		if(ymax - ymin < 1e-4)
			ymax += 1e-4;
		svg.newChart(xmin, ymin, xmax, ymax, 0, 0, 0);
		for(size_t i = 0; i < topic.size(); i++)
		{
			Item& item = topic.item(i);
			const char* szTitle = item.left();
			vector<double>& weights = item.weights();
			svg.dot(weights[1], weights[2], 0.75, 0x008080);
			svg.text(weights[1], weights[2], szTitle, 0.75);
		}
		svg.print(response);
	}

	virtual void pageUpdate(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		User* pUser = pAccount->getUser(((Server*)m_pServer)->recommender());
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->recommender().topics().size())
		{
			m_pServer->redirect(response, "/survey");
		}
		else
		{
			makeSliderScript(response);

			// Display the topic
			Topic* pCurrentTopic = ((Server*)m_pServer)->recommender().topics()[currentTopic];
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";

			// Display the items you have rated
			if(pUser->ratings().size() > currentTopic)
			{
				vector<pair<size_t, float> >& v = pUser->ratings()[currentTopic]->m_vec;
				if(v.size() > 0)
				{
					response << "<h3>Your opinions</h3>\n";
					response << "<form name=\"formname\" action=\"/survey\" method=\"post\">\n";
					response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";
					UpdateComparer comparer;
					std::sort(v.begin(), v.end(), comparer);
					for(vector<pair<size_t, float> >::iterator it = v.begin(); it != v.end(); it++)
						makeUrlSlider(pAccount, it->first, response);
					response << "<br><table><tr><td width=330></td><td>";
					response << "<input type=\"submit\" value=\"Update ratings\">";
					response << "</td></tr><tr><td></td><td>";
					response << "(Only checked items will be updated.)";
					response << "</td></tr></table>\n";
					response << "</form><br><br>\n\n";
				}
				else
					response << "You have not yet rated anything in this topic<br><br>\n";
			}
			else
				response << "You have not yet rated anything in this topic<br><br>\n";

			// The choices links at the bottom of the page
			response << "<a href=\"/submit\">Submit a new item</a>";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/survey?topic=-1\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/survey\">" << "Survey</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/stats\">Vizualize</a>\n";
		}
	}

	virtual void pageAdmin(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		if(pSession->paramsLen() > 0)
		{
			GHttpParamParser params(pSession->params());
			const char* szDel = params.find("del");
			if(szDel)
			{
				size_t currentTopic = pAccount->currentTopic();
				if(currentTopic >= ((Server*)m_pServer)->recommender().topics().size())
					response << "[invalid topic id]<br><br>\n";
				else
				{
					size_t index = atoi(szDel);
					const std::vector<User*>& users = ((Server*)m_pServer)->recommender().users();
					Topic& topic = *((Server*)m_pServer)->recommender().topics()[currentTopic];
					if(index >= topic.size())
						response << "[invalid item index]<br><br>\n";
					else
					{
						cout << "Deleted item " << topic.item(index).left() << " / " << topic.item(index).right() << "\n";
						for(std::vector<User*>::const_iterator it = users.begin(); it != users.end(); it++)
						{
							(*it)->withdrawRating(currentTopic, index);
							(*it)->swapItems(currentTopic, index, topic.size() - 1);
						}
						topic.deleteItemAndSwapInLast(index);
						response << "[Item successfully deleted]<br><br>\n";
					}
				}
			}
		}

		// Form to delete a statement
		response << "<h2>Delete Statements</h2>\n\n";
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->recommender().topics().size())
			response << "<p>No topic has been selected. If you want to delete one or more statements, please click on \"Survey\", choose a topic, then return to here.</p>\n";
		else
		{
			response << "<p>If a statement can be corrected, it is courteous to submit a corrected version after you delete it. Valid reasons to delete a statement include: not controversial enough, too long-winded, confusing, difficult to negate, ambiguous, off-topic, etc.</p>";
			Topic& topic = *((Server*)m_pServer)->recommender().topics()[currentTopic];
			response << "<table>\n";
			for(size_t i = 0; i < topic.size(); i++)
			{
				Item& itm = topic.item(i);
				response << "<tr><td>";
				response << "<form name=\"delitem\" action=\"/admin\" method=\"get\"><input type=\"hidden\" name=\"del\" value=\"" << to_str(i) << "\"><input type=\"submit\" value=\"Delete\"></form>";
				response << "</td><td>" << itm.left() << " / " << itm.right() << "</td></tr>\n";
			}
			response << "</table>\n";
			response << "<br>\n";
			response << "</form><br><br>\n\n";
		}

		// Admin controls
		if(pAccount->isAdmin())
		{
			response << "<h2>Admin controls</h2>\n\n";

			// Form to add a new topic
			response << "<form name=\"newtopicform\" action=\"/admin\" method=\"get\">\n";
			response << "	Add a new topic:<br>\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"newtopic\" />\n";
			response << "	<input type=\"text\" name=\"descr\" size=\"55\"><input type=\"submit\" value=\"Add\"><br>\n";
			response << "</form><br><br>\n\n";

			// Form to shut down the server
			response << "<form name=\"trainform\" action=\"/admin\" method=\"get\">\n";
			response << "	Refine the recommender system:<br>\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"train\" />\n";
			response << "	<input type=\"submit\" value=\"Train\">\n";
			response << "</form><br><br>\n\n";

			// Form to shut down the server
			response << "<form name=\"shutdownform\" action=\"/admin\" method=\"get\">\n";
			response << "	Shut down the daemon:<br>\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"shutdown\" />\n";
			response << "	<input type=\"submit\" value=\"Shut down now\">\n";
			response << "</form><br><br>\n\n";
		}
	}

	virtual void pageAccount(GDynamicPageSession* pSession, ostream& response)
	{
		Terminal* pTerminal = getTerminal(pSession);
		Account* pAccount = pTerminal->currentAccount();
		if(pAccount)
		{
			response << "<h3>Manage your account</h3>\n";

			// Change username
			response << "<form><input type=\"hidden\" name=\"action\" value=\"changename\">";
			response << "Change your username to: <input type=\"text\" name=\"newname\" value=\"" << pAccount->username() << "\">";
			response << "<input type=\"submit\" value=\"Change username\">";
			response << "</form>";
			response << "<br><br>";

			// Add a password
			bool havePassword = true;
			if(pAccount->passwordHash() == nullptr || strlen(pAccount->passwordHash()) < 1)
				havePassword = false;
			if(!havePassword)
			{
				response << "You do not yet have a password! Please set a password so you can access this account from other devices.<br>";
			}
			response << "<form><input type=\"hidden\" name=\"action\" value=\"changepassword\">";
			response << "<table><tr><td align=right>" << (havePassword ? "Change" : "Set") << " your password to:</td><td><input type=\"password\" name=\"newpw\"></td><td></td></tr>";
			response << "<tr><td align=right>Again:</td><td><input type=\"password\" name=\"pwagain\"></td><td><input type=\"submit\" value=\"Change password\"></td></tr></table>";
			response << "</form>";
			response << "<br><br>";

			// Check box to require password on this account
			if(havePassword)
			{
				response << "<form><input type=\"hidden\" name=\"action\" value=\"requirepw\">";
				response << "<input type=\"checkbox\" name=\"cb\"" << (pTerminal->requirePassword() ? " checked" : "") << " onChange=\"this.form.submit()\">Require a password when I visit using this device.";
				response << "</form>";
				response << "<br><br>";
			}
/*
			// Button to nuke my account
			response << "Warning: Don't push this button unless you really mean it! ";
			response << "<form name=\"nukeself\" action=\"/account\" method=\"get\">\n";
			response << "<input type=\"hidden\" name=\"action\" value=\"nukeself\">\n";
			response << "<input type=\"submit\" value=\"Nuke My Account\">\n";
			response << "</form>\n";
*/
		}
		else
		{
			// Not logged in
			response << "Click on an account to log in:<br>(You can change the username or password after you log in.)<br><br>\n";
			response << "<table>\n";
			for(size_t i = 0; i < pTerminal->accountCount(); i++)
			{
				bool reqpw = pTerminal->requirePassword(i);
				response << "<tr><td>";
				Account* pAcc = pTerminal->account(i);
				if(!reqpw)
					response << "<a href=\"/account?action=login&name=" << pAcc->username() << "\">";
				response << "<div style=\"";
				response << "background-color:#203050;width:220px;height:60px;";
				response << "border:1px solid #000000;border-radius:15px;";
				response << "text-align:center;color:white";
				response << "\">";
				if(!pTerminal->requirePassword(i))
					response << "<br>";
				response << pAcc->username();
				if(pAcc->isAdmin())
					response << " (Admin)";
				if(reqpw)
				{
					response << "<form>";
					response << "<input type=\"hidden\" name=\"action\" value=\"login\">";
					response << "<input type=\"hidden\" name=\"name\" value=\"" << pAcc->username() << "\">";
					response << "<input type=\"password\" name=\"password\">";
					response << "</form>";
				}
				response << "</div>";
				if(!reqpw)
					response << "</a>";
				response << "</td><td>";
				response << "<a href=\"/account?action=forget&name=" << pAcc->username() << "\">Forget account</a>";
				response << "</td></tr>\n";
			}
			response << "</table>\n";
		}
/*
		Account* pAccount = getAccount(pSession);
		if(pSession->paramsLen() > 0)
		{
			// Check the password
			const char* szUsername = params.find("username");
			const char* szPasswordHash = params.find("password");
			if(szUsername)
			{
				Account* pLoadedAccount = ((Server*)m_pServer)->loadAccount(szUsername, szPasswordHash);
				if(pLoadedAccount)
					getTerminal(pSession)->logIn(pLoadedAccount);
				else
					response << "<big><big>Incorrect Password! Please try again</big></big><br><br>\n";
			}
		}

		response << "<br><br>\n";
		response << "<SCRIPT language=\"JavaScript\" src=\"/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
		if(pAccount)
		{
			response << "Your current username is: ";
			const char* szUsername = pAccount->username();
			if(*szUsername == '_')
				response << "anonymous";
			else
				response << szUsername;
			response << ".<br>\n";
			response << "Switch user:<br>\n";
		}
		else
			response << "Please log in:<br><br>\n";
		response << "<form name=\"loginform\" action=\"/account\" method=\"get\" onsubmit=\"return HashPassword('";
		response << ((Server*)m_pServer)->passwordSalt();
		response << "')\">\n";
		response << "	Username:<input type=\"text\" name=\"username\" ><br>\n";
		response << "	Password:<input type=\"password\" name=\"password\" ><br>\n";
		response << "	<input type=\"submit\" value=\"Log In\">\n";
		response << "</form><br>\n\n";

		response << "or <a href=\"/newaccount\">create a new account</a><br><br><br>\n";
*/
	}

	void pageError(ostream& response)
	{
		response << "Sorry, an error occurred. Please yell at the operator of this site.\n";
	}

	void pageNewAccount(GDynamicPageSession* pSession, ostream& response)
	{
		const char* szUsername = "";
		const char* szPassword = "";
		const char* szPWAgain = "";
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
			szPassword = params.find("password");
			szPWAgain = params.find("pwagain");

			// Check the parameters
			if(!szUsername || strlen(szUsername) < 1)
				szError = "The username is not valid";
			if(!szPassword || strlen(szPassword) < 1)
				szError = "The password is not valid";
			if(!szPWAgain || strcmp(szPassword, szPWAgain) != 0)
				szError = "The passwords don't match";
			if(!szError)
			{
				// Create the account
				Account* pAccount = ((Server*)m_pServer)->newAccount(szUsername, szPassword);
				if(!pAccount)
					szError = "That username is already taken.";
				else
				{
					((Server*)m_pServer)->saveState();
					response << "<big>An account has been successfully created.</big><br><br> Click here to <a href=\"/account\">log in</a><br>\n";
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

		response << "<br><center><table width=\"400\" border=\"0\" cellpadding=\"0\" cellspacing=\"0\"><tr><td>\n";
		response << "<SCRIPT language=\"JavaScript\" src=\"/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
		response << "	<big><big><b>Create a new account</b></big></big><br><br>\n";
		response << "	<form name=\"newaccountform\" action=\"/newaccount\" method=\"post\" onsubmit=\"return HashNewAccount('";
		response << ((Server*)m_pServer)->passwordSalt();
		response << "')\">\n";
		response << "		<input type=\"hidden\" name=\"action\" value=\"newaccount\" />\n";
		response << "		Username: <input type=\"text\" size=\"15\" name=\"username\" value=\"";
		response << szUsername;
		response << "\"><br><br>\n";
		response << "		Password: <input type=\"password\" name=\"password\" size=\"15\" value=\"";
		response << szPassword;
		response << "\"><br>\n";
		response << "		PW Again: <input type=\"password\" name=\"pwagain\" size=\"15\" value=\"";
		response << szPWAgain;
		response << "\"><br><br>\n";
		response << "		<input type=\"submit\" value=\"Submit\">\n";
		response << "	</form><br>\n\n";
		response << "</tr></td></table></center>\n";
	}

	void pageTest(GDynamicPageSession* pSession, ostream& response)
	{
		response << "<script type=\"text/javascript\" src=\"ajax.js\"></script>\n";
	}
};

const char* Connection::processParams(GDynamicPageSession* pSession)
{
	const char* szParamsMessage = nullptr;
	GHttpParamParser params(pSession->params());
	const char* szAction = params.find("action");
	if(szAction)
	{
		Terminal* pTerminal = getTerminal(pSession);
		if(strcmp(szAction, "logout") == 0)
			pTerminal->logOut();
		else if(strcmp(szAction, "login") == 0)
		{
			const char* szUsername = params.find("name");
			if(szUsername)
			{
				const char* szPasswordHash = params.find("password");
				Account* pAccount = ((Server*)m_pServer)->findAccount(szUsername);
				if(pAccount)
				{
					if(!pTerminal->logIn(pAccount, szPasswordHash))
					{
						if(pTerminal->currentAccount())
							throw Ex("say wha?");
						szParamsMessage = "Incorrect password";
					}
				}
				else
					szParamsMessage = "Invalid username";
			}
			else
				szParamsMessage = "Expected a username";
		}
		else if(strcmp(szAction, "newaccount") == 0)
		{
			pTerminal->makeNewAccount((Server*)m_pServer);
		}
		else if(strcmp(szAction, "changename") == 0)
		{
			Account* pAccount = pTerminal->currentAccount();
			if(pAccount)
			{
				const char* szNewName = params.find("newname");
				if(szNewName && strlen(szNewName) > 0)
				{
					if(!pAccount->changeUsername((Server*)m_pServer, szNewName))
						szParamsMessage = "Sorry, that name is already taken";
				}
				else
					szParamsMessage = "Expected a new name";
			}
			else
				szParamsMessage = "You must log in to change the name";
		}
		else if(strcmp(szAction, "changepassword") == 0)
		{
			Account* pAccount = pTerminal->currentAccount();
			if(pAccount)
			{
				const char* szNewPw = params.find("newpw");
				const char* szPwAgain = params.find("pwagain");
				if(szNewPw && szPwAgain && strlen(szNewPw) > 0 && strlen(szPwAgain))
				{
					if(strcmp(szNewPw, szPwAgain) == 0)
						pAccount->changePasswordHash((Server*)m_pServer, szNewPw);
					else
						szParamsMessage = "The passwords do not match";
				}
				else
					szParamsMessage = "Expected the password to be entered two times";
			}
			else
				szParamsMessage = "You must log in to change the name";
		}
		else if(strcmp(szAction, "forget") == 0)
		{
			const char* szName = params.find("name");
			if(szName && strlen(szName) > 0)
			{
				if(!pTerminal->forgetAccount(szName))
					szParamsMessage = "No such account to forget";
			}
			else
				szParamsMessage = "Expected a username to forget";
		}
		else if(strcmp(szAction, "requirepw") == 0)
		{
			const char* szCheckbox = params.find("cb");
			if(szCheckbox)
				pTerminal->setRequirePassword(true);
			else
				pTerminal->setRequirePassword(false);
		}
		else if(strcmp(szAction, "shutdown") == 0)
		{
			Account* pAccount = pTerminal->currentAccount();
			if(pAccount)
			{
				if(pAccount->isAdmin())
				{
					cout << "An admin has directed the server to shut down\n";
					cout.flush();
					m_pServer->shutDown();
				}
				else
					szParamsMessage = "Sorry, only an administrator may perform that action";
			}
			else
				szParamsMessage = "You must log in to do this action";
		}
		else if(strcmp(szAction, "newtopic") == 0)
		{
			Account* pAccount = pTerminal->currentAccount();
			if(pAccount)
			{
				if(pAccount->isAdmin())
				{
					const char* szDescr = params.find("descr");
					if(szDescr && strlen(szDescr) > 0)
					{
						((Server*)m_pServer)->recommender().newTopic(szDescr);
					}
					else
						szParamsMessage = "You must enter a topic description";
				}
				else
					szParamsMessage = "Sorry, only an administrator may perform that action";
			}
			else
				szParamsMessage = "You must log in to do this action";
		}
		else if(strcmp(szAction, "nukeself") == 0)
		{
			Account* pAccount = pTerminal->currentAccount();
			const char* szUsername = pAccount->username();
			((Server*)m_pServer)->deleteAccount(pAccount);
			pTerminal->forgetAccount(szUsername);
			pTerminal->logOut();
		}
		else if(strcmp(szAction, "train") == 0)
		{
			Account* pAccount = pTerminal->currentAccount();
			((Server*)m_pServer)->recommender().refineModel(pAccount->currentTopic(), ON_TRAIN_TRAINING_ITERS);
		}
	}
	return szParamsMessage;
}

void Connection::handleAjax(GDynamicPageSession* pSession, ostream& response)
{
	GDom docIn;
	double d = 0;
	try
	{
		docIn.parseJson(pSession->params(), pSession->paramsLen());
		const GDomNode* pNode = docIn.root();
		d = pNode->getDouble("burrito");
	}
	catch(std::exception& e)
	{
		cout << "Invalid incoming AJAX blob. " << e.what();
		return;
	}

	GDom doc;
	GDomNode* pNode = doc.newObj();
	pNode->add(&doc, "someval", 3.14159);
	pNode->add(&doc, "yousaid", d);
	
	setContentType("application/json");
	pNode->writeJson(response);
}

bool isurl(const char* url, const char* dynpage)
{
	while(true)
	{
		if(*dynpage == '\0')
			break;
		if(*url != *dynpage)
			return false;
		url++;
		dynpage++;
	}
	if(*url == '\0' || *url == '?')
		return true;
	return false;
}

// virtual
void Connection::handleRequest(GDynamicPageSession* pSession, ostream& response)
{
	if(isurl(m_szUrl, "/ajax"))
	{
		handleAjax(pSession, response);
		return;
	}

	const char* szParamsMessage = processParams(pSession);

	// Fine a method to make the requested page
	bool headers = true;
	void (Connection::*makePage)(GDynamicPageSession* pSession, ostream& response) = nullptr;
	if(isurl(m_szUrl, "/")) makePage = &Connection::pageSurvey;
	else if(isurl(m_szUrl, "/account")) makePage = &Connection::pageAccount;
	else if(isurl(m_szUrl, "/survey")) makePage = &Connection::pageSurvey;
	else if(isurl(m_szUrl, "/submit")) makePage = &Connection::pageSubmit;
	else if(isurl(m_szUrl, "/stats")) makePage = &Connection::pageStats;
	else if(isurl(m_szUrl, "/test")) makePage = &Connection::pageTest;
	else if(isurl(m_szUrl, "/update")) makePage = &Connection::pageUpdate;
	else if(isurl(m_szUrl, "/admin")) makePage = &Connection::pageAdmin;
	else if(isurl(m_szUrl, "/newaccount")) makePage = &Connection::pageNewAccount;
	else if(isurl(m_szUrl, "/users.svg")) { makePage = &Connection::plotUsers; headers = false; }
	else if(isurl(m_szUrl, "/items.svg")) { makePage = &Connection::plotItems; headers = false; }

	try
	{
		if(makePage)
		{
			// Make sure there is at least one account
			Terminal* pTerminal = getTerminal(pSession);
			if(pTerminal->accountCount() == 0)
				pTerminal->makeNewAccount((Server*)m_pServer);
			Account* pAccount = pTerminal->currentAccount();
			if(pAccount)
			{
				if(headers)
					makeHeader(pSession, response, szParamsMessage);
				(*this.*makePage)(pSession, response);
				if(headers)
					makeFooter(pSession, response);
			}
			else
			{
				if(headers)
					makeHeader(pSession, response, szParamsMessage);
				pageAccount(pSession, response);
				if(headers)
					makeFooter(pSession, response);
			}
		}
		else
		{
			size_t len = strlen(m_szUrl);
			if(len > 6 && strcmp(m_szUrl + len - 6, ".hbody") == 0)
			{
				if(headers)
					makeHeader(pSession, response, szParamsMessage);
				sendFileSafe(((Server*)m_pServer)->m_basePath.c_str(), m_szUrl + 1, response);
				if(headers)
					makeFooter(pSession, response);
			}
			else
				sendFileSafe(((Server*)m_pServer)->m_basePath.c_str(), m_szUrl + 1, response);
		}
	}
	catch(std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
		pageError(response);
	}
}

// ------------------------------------------------------

Server::Server(int port, GRand* pRand)
: GDynamicPageServer(port, pRand),
m_recommender(*pRand)
{
	char buf[300];
	GApp::appPath(buf, 256, true);
	strcat(buf, "web/");
	GFile::condensePath(buf);
	m_basePath = buf;
	cout << "Base path: " << m_basePath << "\n";
}

// virtual
Server::~Server()
{
	saveState();

	// Delete all the accounts
	flushSessions(); // ensure that there are no sessions referencing the accounts
	for(map<std::string,Account*>::iterator it = m_accounts.begin(); it != m_accounts.end(); it++)
		delete(it->second);
}

void getLocalStorageFolder(char* buf)
{
	if(!GFile::localStorageDirectory(buf))
		throw Ex("Failed to find local storage folder");
	strcat(buf, "/.community/");
	GFile::makeDir(buf);
	if(!GFile::doesDirExist(buf))
		throw Ex("Failed to create folder in storage area");
}

void Server::saveState()
{
	GDom doc;
	doc.setRoot(serializeState(&doc));
	char szStoragePath[300];
	getLocalStorageFolder(szStoragePath);
	strcat(szStoragePath, "state.json");
	doc.saveJson(szStoragePath);
	char szTime[256];
	GTime::asciiTime(szTime, 256, false);
	cout << "Server state saved at: " << szTime << "\n";
}

// virtual
void Server::onEverySixHours()
{
	for(size_t i = 0; i < 3; i++)
	{
		size_t topicId = m_pRand->next(recommender().topics().size());
		recommender().refineModel(topicId, ON_TRAIN_TRAINING_ITERS);
	}
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

Account* Server::findAccount(const char* szUsername)
{
	map<string,Account*>::iterator it = m_accounts.find(szUsername);
	if(it == m_accounts.end())
		return nullptr;
	Account* pAccount = it->second;
	return pAccount;
}

Account* Server::newAccount(const char* szUsername, const char* szPasswordHash)
{
	if(!szPasswordHash)
		szPasswordHash = "";

	// See if that username already exists
	if(findAccount(szUsername))
		return nullptr;

	// Make the account
	Account* pAccount = new Account(szUsername, szPasswordHash);
	m_accounts.insert(make_pair(string(szUsername), pAccount));
	cout << "Made new account for " << szUsername << "\n";
	cout.flush();
	return pAccount;
}

void Server::onRenameAccount(const char* szOldName, Account* pAccount)
{
	m_accounts.erase(szOldName);
	m_accounts.insert(make_pair(pAccount->username(), pAccount));
}

void Server::deleteAccount(Account* pAccount)
{
	string s;
	for(std::map<std::string,Account*>::iterator it = m_accounts.begin(); it != m_accounts.end(); it++)
	{
		if(it->second == pAccount)
		{
			s = it->first;
			break;
		}
	}
	m_accounts.erase(s);
	cout << "Account " << pAccount->username() << " deleted.\n";
	saveState();
}

GDomNode* Server::serializeState(GDom* pDoc)
{
	GDomNode* pNode = GDynamicPageServer::serialize(pDoc);
	GDomNode* pAccounts = pDoc->newList();
	pNode->add(pDoc, "accounts", pAccounts);
	for(std::map<std::string,Account*>::iterator it = m_accounts.begin(); it != m_accounts.end(); it++)
	{
		Account* pAcc = it->second;
		pAccounts->add(pDoc, pAcc->toDom(pDoc));
	}
	pNode->add(pDoc, "recommender", m_recommender.serialize(pDoc));
	return pNode;
}

void Server::deserializeState(const GDomNode* pNode)
{
	// Load the accounts
	GAssert(m_accounts.size() == 0);
	GDomNode* pAccounts = pNode->get("accounts");
	for(GDomListIterator it(pAccounts); it.current(); it.advance())
	{
		Account* pAccount = Account::fromDom(it.current(), *m_pRand);
		m_accounts.insert(make_pair(string(pAccount->username()), pAccount));
	}

	// Load the base stuff
	GDynamicPageServer::deserialize(pNode);

	// Recommender system
	m_recommender.deserialize(pNode->get("recommender"));
}

GDynamicPageSessionExtension* Server::deserializeSessionExtension(const GDomNode* pNode)
{
	return new Terminal(pNode, this);
}

// virtual
GDynamicPageConnection* Server::makeConnection(SOCKET sock)
{
	return new Connection(sock, this);
}






void LaunchBrowser(const char* szAddress, GRand* pRand)
{
	string s = szAddress;
	s += "/survey";
	if(!GApp::openUrlInBrowser(s.c_str()))
	{
		cout << "Failed to open the URL: " << s.c_str() << "\nPlease open this URL manually.\n";
		cout.flush();
	}
}

void redirectStandardStreams(const char* pPath)
{
	string s1(pPath);
	s1 += "stdout.log";
	if(!freopen(s1.c_str(), "a", stdout))
	{
		cout << "Error redirecting stdout\n";
		cerr << "Error redirecting stdout\n";
		throw Ex("Error redirecting stdout");
	}
	string s2(pPath);
	s2 += "stderr.log";
	if(!freopen(s2.c_str(), "a", stderr))
	{
		cout << "Error redirecting stderr\n";
		cerr << "Error redirecting stderr\n";
		throw Ex("Error redirecting stderr");
	}
}

void doit(void* pArg)
{
	{
#ifdef _DEBUG
		int port = 8987;
#else
		int port = 8988;
#endif
		size_t seed = getpid() * (size_t)time(NULL);
		GRand prng(seed);
		char statePath[300];
		getLocalStorageFolder(statePath);
		strcat(statePath, "state.json");
		Server* pServer = new Server(port, &prng);
		Holder<Server> hServer(pServer);
		if(GFile::doesFileExist(statePath))
		{
			GDom doc;
			cout << "Loading state...\n";
			cout.flush();
			doc.loadJson(statePath);
			pServer->deserializeState(doc.root());
			cout << "Server state loaded from " << statePath << "\n";
			cout.flush();

/*
			// Do some training to make sure the model is in good shape
			cout << "doing some training...\n";
			for(size_t i = 0; i < recommender().topics().size(); i++)
				recommender().refineModel(i, ON_STARTUP_TRAINING_ITERS);
*/
		}
		else
			cout << "No saved state. Made a new server\n";

		char buf[300];
		GTime::asciiTime(buf, 256, false);
		cout << "Server started at: " << buf << "\n";
		cout.flush();

		LaunchBrowser(pServer->myAddress(), &prng);
		pServer->go();
	}
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
	if(chdir(path) != 0)
		throw Ex("Failed to change dir to ", path);
	cout << "Launching daemon...\n";
	GApp::launchDaemon(doit, path, s1.c_str(), s2.c_str());
	if(!getcwd(path, 300))
	{
	}
	cout << "Daemon running in " << path << ".\n	stdout >> " << s1.c_str() << "\n	stderr >> " << s2.c_str() << "\n";
}

int main(int nArgs, char* pArgs[])
{
cout << "an1=" << to_str(AUTO_NAME_1_COUNT) << "," << to_str(AUTO_NAME_2_COUNT) << "," << to_str(AUTO_NAME_3_COUNT) << "\n";
	int nRet = 1;
	try
	{
		if(nArgs > 1 && strcmp(pArgs[1], "daemon") == 0)
			doItAsDaemon();
		else
			doit(NULL);
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
