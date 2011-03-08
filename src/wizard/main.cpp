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
#include "../GSup/GDynamicPage.h"
#include "../GClasses/GApp.h"
#include "../GClasses/GTwt.h"
#include "../GClasses/GString.h"
#include "../GClasses/GHolders.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GTime.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GHashTable.h"
#include "../GSup/sha1.h"
#include <wchar.h>
#include <string>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>
#include "usage.h"
#include <sstream>

using namespace GClasses;
using std::cout;
using std::cerr;
using std::string;
using std::ostream;
using std::vector;
using std::ostringstream;

class Page;

class MySession : public GDynamicPageSessionExtension
{
protected:
	GDynamicPageSession* m_pDPSession;
	size_t m_page;
	Page* m_pRootPage;
	Page* m_pCurrentPage;
	Page* m_pGoodbyePage;
	UsageNode* m_pRootNode;
	UsageNode* m_pDefaultNode;

public:
	MySession(GDynamicPageSession* pDPSession);

	virtual ~MySession()
	{
		delete(m_pRootNode);
		delete(m_pDefaultNode);
	}

	virtual void onDisown()
	{
		m_pDPSession = NULL;
		delete(this);
	}

	size_t page() { return m_page; }
	Page* currentPage() { return m_pCurrentPage; }
	UsageNode* defaultNode() { return m_pDefaultNode; }

	void doNext(const char* szParams);
};

MySession* getSession(GDynamicPageSession* pDPSession)
{
	MySession* pSess = (MySession*)pDPSession->extension();
	if(!pSess)
	{
		pSess = new MySession(pDPSession);
		pDPSession->setExtension(pSess);
	}
	return pSess;
}

class Server : public GDynamicPageServer
{
protected:
	std::string m_basePath;
	vector<UsageNode*> m_globals;

public:
	Server(int port, GRand* pRand);
	virtual ~Server();
	virtual void handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, std::ostream& response);
	UsageNode* globalUsageNode(const char* name, UsageNode* pDefaultNode);
	virtual void onEverySixHours() {}
	virtual void onStateChange() {}
	virtual void onShutDown() {}
};


class Page
{
protected:
	Server* m_pController;
	MySession* m_pSession;
	Page* m_pParent;
	UsageNode* m_pNode;
	vector<Page*> m_children;
	int m_selection;
	vector<string> m_structData;
	vector<size_t> m_options;
	bool m_doAutoNext;

	enum part_type
	{
		pt_string,
		pt_struct,
		pt_loner,
	};

	enum mode
	{
		mode_choose_one,
		mode_options,
		mode_goodbye,
		mode_struct,
		mode_first_mave,
	};

	mode m_mode;

public:
	Page(Server* pController, MySession* pSession, Page* pParent, UsageNode* pNode) : m_pController(pController), m_pSession(pSession), m_pParent(pParent), m_pNode(pNode)
	{
		m_selection = 0;
		m_doAutoNext = false;
		if(!pNode)
		{
			m_mode = mode_goodbye;
			return;
		}
		if(pNode->parts().size() == 2 && pNode->choices().size() == 1 && strcmp(pNode->parts()[1].c_str(), pNode->choices()[0]->tok()) == 0)
			pNode = pNode->choices()[0];
		if(pNode->choices().size() > 0 && ((pNode->parts().size() == 1 && pNode->tok()[0] == '[') || (pNode->parts().size() == 2 && pNode->tok()[0] != '[' && pNode->tok()[0] != '<' && pNode->parts()[1].c_str()[0] == '[')))
			m_mode = mode_choose_one;
		else if(pNode->tok()[0] == '<')
			m_mode = mode_options;
		else if(pNode->parts().size() == 1 && pNode->tok()[0] != '[')
		{
			m_mode = mode_options;
			m_doAutoNext = true;
		}
		else if(countStructureParts(pNode) > 0)
			m_mode = mode_struct;
		else
		{
			m_mode = mode_struct;
			m_doAutoNext = true;
		}
	}

	~Page()
	{
		clearChildren();
	}

	void setCommand(string& s) { m_structData.push_back(s); }

	bool autoDoNext() { return m_doAutoNext; }

	void makeBody(std::ostream& response)
	{
		response << "<h2>Waffles Command-building Wizard</h2>\n";
		response << "<table border=\"1\" width=\"1000\"><tr><td>\n";
		response << "<form name=\"input\" action=\"wizard\" method=\"post\">\n";

		if(m_mode == mode_goodbye)
		{
			response << "Done! The command to perform the specified task is:<br><br>\n\n";
			response << "<pre>\n";
			for(size_t i = 0; i < m_structData.size(); i++)
				response << "	" << m_structData[i] << "\n";
			response << "</pre><br>\n\n";
			response << "To execute this command, just paste it into a console window. To use it in a script, just paste it into the script. (You may close this window now.)";
			m_pController->shutDown();
			return;
		}
		else if(m_mode == mode_choose_one)
		{
			response << "Please choose a value for <i>" << m_pNode->tok() << "</i><br>\n";
			response << m_pNode->descr() << "<br><br>\n\n";
			response << "<table width=60%><tr><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Start Over\" /></td><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Next\" /></td></tr></table><br>\n";
			response << "<table>\n";
			for(size_t i = 0; i < m_pNode->choices().size(); i++)
			{
				UsageNode* pChoice = m_pNode->choices()[i];
				response << "	<tr><td valign=top><input type=\"radio\" name=\"choice\" value=\"" << i << "\"";
				if(i == 0)
					response << " checked";
				response << "></td>";
				response << "<td valign=top>" << pChoice->tok() << "</td>";
				response << "<td>" << pChoice->descr() << "<br><br></td></tr>\n";
			}
			response << "</table>\n";
		}
		else if(m_mode == mode_options)
		{
			response << "Check the options you want for <i>" << m_pNode->tok() << "</i><br><br>\n\n";
			response << "<table width=60%><tr><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Start Over\" /></td><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Next\" /></td></tr></table><br>\n";
			response << "<table>\n";
			for(size_t i = 0; i < m_pNode->choices().size(); i++)
			{
				UsageNode* pChoice = m_pNode->choices()[i];
				string s;
				pChoice->sig(&s);
				response << "	<tr><td valign=top><input type=checkbox name=\"" << i << "\"></td>";
				response << "<td valign=top>" << s << "</td>";
				response << "<td>" << pChoice->descr() << "<br><br></td><tr>\n";
			}
			response << "</table>\n";
		}
		else if(m_mode == mode_struct)
		{
			response << "Please provide values for <i>" << m_pNode->tok() << "</i><br>\n";
			response << m_pNode->descr() << "<br><br>\n\n";
			response << "<table width=60%><tr><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Start Over\" /></td><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Next\" /></td></tr></table><br>\n";
			response << "<table>\n";
			size_t index = 0;
			for(size_t i = 0; i < m_pNode->parts().size(); i++)
			{
				if(partType(m_pNode, i) == pt_struct)
				{
					string arg = m_pNode->parts()[i];
					UsageNode* pChoice = m_pNode->choice(arg.c_str());
					if(!pChoice)
						pChoice = globalUsageNode(arg.c_str());
					response << "	<tr><td valign=top>" << pChoice->tok() << "</td>";
					response << "<td valign=top><input type=\"text\" name=\"" << index << "\" ";
					string s = pChoice->default_value();
					if(s.length() == 0)
						s = m_pNode->default_value();
					if(s.length() > 0)
						response << "value=\"" << s << "\"";
					response << "/></td>";
					response << "<td>" << pChoice->descr() << "<br><br></td></tr><br>\n";
					index++;
				}
			}
			response << "</table>\n";
		}

		response << "<br><br>\n\n";
		response << "<table width=60%><tr><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Start Over\" /></td><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Next\" /></td></tr></table>\n";
		response << "<input type=\"hidden\" name=\"pageid\" value=\"" << m_pSession->page() << "\" />\n";
		response << "</form>\n";
		response << "</td></tr></table>\n";
	}

	void setChoices(GHttpParamParser& pp)
	{
		if(m_mode == mode_choose_one)
		{
			const char* pChoice = pp.find("choice");
			if(!pChoice)
				ThrowError("Expected a \"choice\" parameter value");
			m_selection = atoi(pChoice);
		}
		else if(m_mode == mode_options)
		{
			m_options.clear();
			for(size_t i = 0; i < m_pNode->choices().size(); i++)
			{
				string s = to_str(i);
				const char* pValue = pp.find(s.c_str());
				if(pValue && _stricmp(pValue, "off") != 0)
					m_options.push_back(i);
			}
		}
		else if(m_mode == mode_struct)
		{
			size_t index = 0;
			for(size_t i = 0; i < m_pNode->parts().size(); i++)
			{
				if(partType(m_pNode, i) == pt_struct)
				{
					// Find the value
					string s = to_str(index);
					const char* pValue = pp.find(s.c_str());
					if(!pValue || strlen(pValue) == 0)
						pValue = "left_blank";

					// Make sure it is properly quoted
					bool quot = false;
					bool needQuotes = false;
					for(size_t j = 0; pValue[j] != '\0'; j++)
					{
						if(pValue[j] == '"' || pValue[j] == '\'')
							quot = !quot;
						else if(!quot && pValue[j] == ' ')
						{
							needQuotes = true;
							break;
						}
					}
					string s2;
					if(needQuotes)
					{
						s2 = "\"";
						s2 += pValue;
						s2 += "\"";
						pValue = s2.c_str();
					}

					// Store the parameter
					m_structData.push_back(pValue);
					index++;
				}
			}
		}
	}

	part_type partType(UsageNode* pNode, int part)
	{
		const char* name = pNode->parts()[part].c_str();
		if(name[0] == '<')
			return pt_loner;
		if(name[0] != '[')
			return pt_string;
		UsageNode* pChoice = pNode->choice(name);
		if(!pChoice)
			pChoice = globalUsageNode(name);
		if(pChoice->choices().size() > 0)
			return pt_loner;
		else
			return pt_struct;
	}

	UsageNode* globalUsageNode(const char* name)
	{
		return m_pController->globalUsageNode(name, m_pSession->defaultNode());
	}

	int countStructureParts(UsageNode* pNode)
	{
		int count = 0;
		for(size_t i = 0; i < pNode->parts().size(); i++)
		{
			if(partType(pNode, i) == pt_struct)
				count++;
		}
		return count;
	}

	Page* parent()
	{
		return m_pParent;
	}

	vector<Page*>& children()
	{
		return m_children;
	}

	void clearChildren()
	{
		for(vector<Page*>::iterator it = m_children.begin(); it != m_children.end(); it++)
			delete(*it);
		m_children.clear();
	}

	bool createChildPages()
	{
		clearChildren();
		if(m_mode == mode_choose_one)
		{
			UsageNode* pChoice = m_pNode->choices()[m_selection];
			m_children.push_back(new Page(m_pController, m_pSession, this, pChoice));
		}
		else if(m_mode == mode_struct)
		{
			for(size_t i = 0; i < m_pNode->parts().size(); i++)
			{
				if(partType(m_pNode, i) == pt_loner)
				{
					string arg = m_pNode->parts()[i];
					UsageNode* pChoice = m_pNode->choice(arg.c_str());
					if(!pChoice)
						pChoice = globalUsageNode(arg.c_str());
					m_children.push_back(new Page(m_pController, m_pSession, this, pChoice));
				}
			}
		}
		else if(m_mode == mode_options)
		{
			for(size_t i = 0; i < m_options.size(); i++)
			{
				size_t option = m_options[i];
				UsageNode* pChoice = m_pNode->choices()[option];
				m_children.push_back(new Page(m_pController, m_pSession, this, pChoice));
			}
		}
		else
			ThrowError("Unrecognized mode");
		return true;
	}

	void makeCommand(ostream& stream)
	{
		if(m_mode == mode_choose_one)
		{
			GAssert(m_children.size() == 1); // expected one child page
			if(partType(m_pNode, 0) == pt_string)
				stream << m_pNode->tok() << " ";
			m_children[0]->makeCommand(stream);
		}
		else if(m_mode == mode_struct)
		{
			int stringPos = 0;
			int structPos = 0;
			int lonerPos = 0;
			for(size_t i = 0; i < m_pNode->parts().size(); i++)
			{
				part_type pt = partType(m_pNode, i);
				if(pt == pt_string)
				{
					stream << m_pNode->parts()[i] << " ";
					stringPos++;
				}
				else if(pt == pt_struct)
					stream << m_structData[structPos++] << " ";
				else
				{
					GAssert(pt == pt_loner); // unexpected value
					m_children[lonerPos++]->makeCommand(stream);
				}
			}
			GAssert((size_t)(stringPos + structPos + lonerPos) == m_pNode->parts().size());
		}
		else if(m_mode == mode_options)
		{
			if(partType(m_pNode, 0) == pt_string)
				stream << m_pNode->tok() << " ";
			for(size_t i = 0; i < m_children.size(); i++)
				m_children[i]->makeCommand(stream);
		}
		else
			ThrowError("Unrecognized mode");
	}
};










MySession::MySession(GDynamicPageSession* pDPSession)
: GDynamicPageSessionExtension(), m_pDPSession(pDPSession)
{
	m_page = 0;
	m_pRootNode = makeMasterUsageTree();
	m_pRootPage = NULL;
	m_pGoodbyePage = NULL;
	m_pCurrentPage = NULL;
	m_pDefaultNode = new UsageNode("", "");
}

void MySession::doNext(const char* szParams)
{
	// Check the page id. Start over if it is not what is expected.
	GHttpParamParser pp(szParams);
	const char* szPageId = pp.find("pageid");
	size_t pageId = (size_t)-1;
	if(szPageId)
		pageId = atoi(szPageId);
	const char* szBtn = pp.find("btn");
	if(szBtn && _stricmp(szBtn, "Start Over") == 0)
		pageId = (size_t)-1;
	if(pageId != m_page || !m_pCurrentPage || !m_pRootPage)
	{
		// Start over at the beginning
		m_page = 1;
		delete(m_pRootPage);
		m_pRootPage = new Page((Server*)m_pDPSession->server(), this, NULL, m_pRootNode);
		m_pCurrentPage = m_pRootPage;
		return;
	}
	else
	{
		m_pCurrentPage->setChoices(pp);
		m_page++;
	}

	// If there are children, pick the first child
	if(!m_pCurrentPage->createChildPages())
		return;
	Page* pNext = NULL;
	if(m_pCurrentPage->children().size() > 0)
		pNext = m_pCurrentPage->children()[0];

	if(!pNext)
	{
		Page* pPar = m_pCurrentPage->parent();
		if(pPar)
		{
			// Find the current page
			size_t i;
			for(i = 0; i < pPar->children().size(); i++)
			{
				if(pPar->children()[i] == m_pCurrentPage)
					break;
			}
			if(i >= pPar->children().size())
				ThrowError("internal error"); // failed to find current page

			// Pick the next sibling if there is one
			if(i + 1 < pPar->children().size())
				pNext = pPar->children()[i + 1];

			if(!pNext)
			{
				// Pick the next-sibling of the nearest ancestor with a next-sibling
				while(true)
				{
					Page* pParPar = pPar->parent();
					if(!pParPar)
						break;
					
					// Find the parent page
					size_t i;
					for(i = 0; i < pParPar->children().size(); i++)
					{
						if(pParPar->children()[i] == pPar)
							break;
					}
					if(i >= pParPar->children().size())
						ThrowError("internal error"); // failed to find pPar page

					// Pick the next sibling of pPar
					if(i + 1 < pParPar->children().size())
					{
						pNext = pParPar->children()[i + 1];
						break;
					}
					pPar = pParPar;
				}
			}
		}
	}
	if(pNext)
		m_pCurrentPage = pNext;
	else
	{
		// Print the command and terminate
		ostringstream stream;
		m_pRootPage->makeCommand(stream);
		string sCommand = stream.str();
		cout << "The command is:\n\n" << sCommand << "\n\n";
		cout.flush();
		delete(m_pGoodbyePage);
		m_pGoodbyePage = new Page((Server*)m_pDPSession->server(), this, NULL, NULL);
		m_pGoodbyePage->setCommand(sCommand);
		m_pCurrentPage = m_pGoodbyePage;
	}
}

















Server::Server(int port, GRand* pRand) : GDynamicPageServer(port, pRand)
{
	char buf[300];
	GApp::appPath(buf, 256, true);
	strcat(buf, "web/");
	GFile::condensePath(buf);
	m_basePath = buf;

	m_globals.push_back(makeAlgorithmUsageTree());
	m_globals.push_back(makeNeighborUsageTree());
}

// virtual
Server::~Server()
{
	for(vector<UsageNode*>::iterator it = m_globals.begin(); it != m_globals.end(); it++)
		delete(*it);
}


UsageNode* Server::globalUsageNode(const char* name, UsageNode* pDefaultNode)
{
	for(vector<UsageNode*>::iterator it = m_globals.begin(); it != m_globals.end(); it++)
	{
		if(strcmp((*it)->tok(), name) == 0)
			return *it;
	}
	pDefaultNode->setTok(name);
	return pDefaultNode;
}

// virtual
void Server::handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pDPSession, std::ostream& response)
{
	if(strcmp(szUrl, "/") == 0)
		szUrl = "/hello";
	if(strcmp(szUrl, "/favicon.ico") == 0)
		return;
	if(strncmp(szUrl, "/wizard", 6) == 0)
	{
		MySession* pSession = getSession(pDPSession);
		pSession->doNext(szParams);
		response << "<html><head>\n";
		response << "</head><body>\n";
		while(pSession->currentPage()->autoDoNext())
		{
			Page* pPrev = pSession->currentPage();
			string sParams = "pageid=";
			sParams += to_str(pSession->page());
			pSession->doNext(sParams.c_str());
			if(pSession->currentPage() == pPrev)
				ThrowError("Internal error");
		}
		pSession->currentPage()->makeBody(response);
		response << "</body></html>\n";
	}
	else
		sendFileSafe(m_basePath.c_str(), szUrl + 1, response);
}



void getLocalStorageFolder(char* buf)
{
	if(!GFile::localStorageDirectory(buf))
		ThrowError("Failed to find local storage folder");
	strcat(buf, "/.hello/");
	GFile::makeDir(buf);
	if(!GFile::doesDirExist(buf))
		ThrowError("Failed to create folder in storage area");
}


void OpenUrl(const char* szUrl)
{
#ifdef WIN32
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
#endif // !WIN32
}

void LaunchBrowser(const char* szAddress)
{
	int addrLen = strlen(szAddress);
	GTEMPBUF(char, szUrl, addrLen + 20);
	strcpy(szUrl, szAddress);
	strcpy(szUrl + addrLen, "/wizard");
	cout << "Opening browser to: " << szUrl << "\n";
	if(!GApp::openUrlInBrowser(szUrl))
	{
		cout << "Failed to open the URL: " << szUrl << "\nPlease open this URL manually.\n";
		cout.flush();
	}
}

void doit(void* pArg)
{
	int port = 8421;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	GRand prng(seed);
	Server server(port, &prng);
	LaunchBrowser(server.myAddress());

	// Pump incoming HTTP requests (this is the main loop)
	server.go();
	cout << "Goodbye.\n";
}

int main(int nArgs, char* pArgs[])
{
	int nRet = 1;
	try
	{
		doit(NULL);
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
