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

#include <GClasses/GRand.h>
#include <GClasses/GApp.h>
#include <GClasses/GTime.h>
#include <map>
#include "server.h"
#include "editor.h"
#include "login.h"

using std::cout;
using std::cerr;
using std::ostream;
using std::map;
using std::string;
using std::make_pair;


Server::Server(int port, GRand* pRand)
: GDynamicPageServer(port, pRand),
m_recommender(*pRand)
{
	char buf[300];
	GApp::appPath(buf, 256, true);
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

void Server::makeHeader(GDynamicPageSession* pSession, ostream& response, const char* szParamsMessage)
{
	//response << "<!DOCTYPE html>";
	response << "<html><head>\n";
	response << "	<meta charset=\"utf-8\">\n";
	Account* pAccount = getAccount(pSession);
	response << "	<title>Community Modeler</title>\n";
	response << "	<link rel=\"stylesheet\" type=\"text/css\" href=\"style/style.css\" />\n";
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
		response << "	<a href=\"/browse\">My pages</a><br><br>\n";
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

void Server::makeFooter(GDynamicPageSession* pSession, ostream& response)
{
	// End of main row
	response << "\n\n\n\n</td></tr>\n";

	// Footer row
	response << "<tr><td colspan=2 id=\"footer\">\n";
	response << "</td></tr></table>\n";
	response << "</td></tr></table></body></html>\n";
}





void Server::getLocalStorageFolder(char* buf)
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

string& Server::cache(const char* szFilename)
{
	string s = m_basePath;
	s += "web/";
	s += szFilename;
	return m_fileCache.get(s.c_str());
}







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


std::string Terminal::generateUsername(GRand& rand)
{
	std::string s = g_auto_name_1[rand.next(AUTO_NAME_1_COUNT)];
	s += " ";
	s += g_auto_name_2[rand.next(AUTO_NAME_2_COUNT)];
	s += " ";
	s += g_auto_name_3[rand.next(AUTO_NAME_3_COUNT)];
	return s;
}



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

















void Connection::handleAjax(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	GDom docIn;
	GDom docOut;
	GDomNode* pOutNode = nullptr;
	setContentType("application/json");
	try
	{
		docIn.parseJson(pSession->params(), pSession->paramsLen());
		const GDomNode* pInNode = docIn.root();
		const char* action = pInNode->getString("action");
		pOutNode = docOut.newObj();
		if(strcmp(action, "save_gui") == 0) Editor::ajaxSaveGui(pServer, pSession, pInNode, docOut, pOutNode);
		else if(strcmp(action, "save_text") == 0) Editor::ajaxSaveText(pServer, pSession, pInNode, docOut, pOutNode);
		else if(strcmp(action, "filelist") == 0) Editor::ajaxFilelist(pServer, pSession, pInNode, docOut, pOutNode);
		else
		{
			string s = "Error: unrecognized action, ";
			s += action;
			pOutNode->add(&docOut, "msg", s.c_str());
		}
	}
	catch(std::exception& e)
	{
		cout << "\nProblem during AJAX request. " << e.what() << "\n";
		cout.flush();
		return;
	}
	pOutNode->writeJson(response);
}

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

/*
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
*/

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

/*
	void pageRedirect(Server* pServer, GDynamicPageSession* pSession, ostream& response, const char* url)
	{
		// Attempt an HTTP redirect
		cout << "Redirecting from " << pSession->url() << " to " << url << "\n";
		pServer->redirect(response, url);

		// Do an HTML redirect as a backup
		response << "<html><head>";
		response << "<meta http-equiv=\"refresh\" content=\"0; url=" << url << "\">";
		response << "</head>\n";

		// Give the user a link as a last resort
		response << "<body>";
		response << "<a href=\"" << url << "\">Please click here to continue</a>\n";
		response << "</body></html>\n";
	}
*/

// virtual
void Connection::handleRequest(GDynamicPageSession* pSession, ostream& response)
{
	pSession->setConnection(this);
	Server* pServer = (Server*)m_pServer;

	//cout << "Handling: " << m_szUrl << "?" << pSession->params() << "\n";
	if(isurl(m_szUrl, "/ajax"))
	{
		handleAjax(pServer, pSession, response);
		return;
	}

	const char* szParamsMessage = processParams(pSession);

	// Fine a method to make the requested page
	bool headers = true;
	void (*makePage)(Server* pServer, GDynamicPageSession* pSession, ostream& response) = nullptr;
	if(isurl(m_szUrl, "/")) makePage = &Survey::pageSurvey;
	else if(isurl(m_szUrl, "/account")) makePage = &Login::pageAccount;
	else if(isurl(m_szUrl, "/browse")) makePage = &Editor::pageBrowse;
	else if(isurl(m_szUrl, "/diff")) makePage = &Editor::pageDiff;
	else if(isurl(m_szUrl, "/edit")) makePage = &Editor::pageEditGui;
	else if(isurl(m_szUrl, "/edittext")) makePage = &Editor::pageEditText;
	else if(isurl(m_szUrl, "/survey")) makePage = &Survey::pageSurvey;
	else if(isurl(m_szUrl, "/submit")) makePage = &Survey::pageNewSurveyItem;
	else if(isurl(m_szUrl, "/stats")) makePage = &Survey::pageStats;
	else if(isurl(m_szUrl, "/update")) makePage = &Survey::pageUpdateResponses;
	else if(isurl(m_szUrl, "/admin")) makePage = &Login::pageAdmin;
	else if(isurl(m_szUrl, "/newaccount")) makePage = &Login::pageNewAccount;
	else if(isurl(m_szUrl, "/users.svg")) { makePage = &Survey::plotUsers; headers = false; }
	else if(isurl(m_szUrl, "/items.svg")) { makePage = &Survey::plotItems; headers = false; }

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
					Server::makeHeader(pSession, response, szParamsMessage);
				(*makePage)(pServer, pSession, response);
				if(headers)
					Server::makeFooter(pSession, response);
			}
			else
			{
				if(headers)
					Server::makeHeader(pSession, response, szParamsMessage);
				Login::pageAccount(pServer, pSession, response);
				if(headers)
					Server::makeFooter(pSession, response);
			}
		}
		else
		{
			string webpath = ((Server*)m_pServer)->m_basePath;
			webpath += "web/";
			size_t len = strlen(m_szUrl);
			if(len > 6 && strcmp(m_szUrl + len - 6, ".hbody") == 0)
			{
				if(headers)
					Server::makeHeader(pSession, response, szParamsMessage);
				sendFileSafe(webpath.c_str(), m_szUrl + 1, response);
				if(headers)
					Server::makeFooter(pSession, response);
			}
			else
				sendFileSafe(webpath.c_str(), m_szUrl + 1, response);
		}
	}
	catch(std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
		response << "Sorry, an error occurred. Please yell at the operator of this site.\n";
	}
}
