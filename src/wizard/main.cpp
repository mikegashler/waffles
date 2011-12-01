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
#include "../GClasses/GDynamicPage.h"
#include "../GClasses/GApp.h"
#include "../GClasses/GDom.h"
#include "../GClasses/GString.h"
#include "../GClasses/GHolders.h"
#include "../GClasses/GFile.h"
#include "../GClasses/GTime.h"
#include "../GClasses/GThread.h"
#include "../GClasses/GRand.h"
#include "../GClasses/GHashTable.h"
#include "../GClasses/sha1.h"
#include "../GClasses/GDirList.h"
#include <wchar.h>
#include <string>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>
#include "usage.h"
#include <sstream>
#include <stack>

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
	std::stack<Page*> m_pageStack;

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

	void onShowPage()
	{
		m_pageStack.push(m_pCurrentPage);
	}

	std::stack<Page*>& pageStack() { return m_pageStack; }
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
	int m_port;

public:
	Server(int port, GRand* pRand);
	virtual ~Server();
	virtual void handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, std::ostream& response);
	UsageNode* globalUsageNode(const char* name, UsageNode* pDefaultNode);
	virtual void onEverySixHours() {}
	virtual void onStateChange() {}
	virtual void onShutDown() {}
	void pump();

protected:
	void addScript(std::ostream& response);
};


class Page
{
protected:
	Server* m_pController;
	MySession* m_pSession;
	Page* m_pParent;
	UsageNode* m_pOrigNode;
	UsageNode* m_pEffectiveNode;
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
	Page(Server* pController, MySession* pSession, Page* pParent, UsageNode* pNode) : m_pController(pController), m_pSession(pSession), m_pParent(pParent), m_pOrigNode(pNode)
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
		m_pEffectiveNode = pNode;
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

	static bool looksLikeFilename(const char* szVal)
	{
		size_t periods = 0;
		size_t alphas = 0;
		while(*szVal != '\0')
		{
			if(*szVal == '.')
				periods++;
			else if(*szVal >= 'a' && *szVal <= 'z' && *szVal != 'e')
				alphas++;
			else if(*szVal >= 'A' && *szVal <= 'Z' && *szVal != 'e')
				alphas++;
			szVal++;
		}
		return periods == 1 && alphas >= 1;
	}

	void makeStandardButtons(std::ostream& response, MySession* pSession, bool haveNext)
	{
		response << "<table width=100%><tr>";
		if(pSession->pageStack().size() > 1)
			response << "<td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Go back\" /></td>";
		else
			response << "<td align=\"center\">&nbsp;&nbsp;&nbsp;</td>";
		if(haveNext)
			response << "<td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Next\" /></td>";
		else
			response << "<td align=\"center\">&nbsp;&nbsp;&nbsp;</td>";
		if(pSession->pageStack().size() > 1)
			response << "<td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Start Over\" /></td>";
		else
			response << "<td align=\"center\">&nbsp;&nbsp;&nbsp;</td>";
		response << "<td align=\"center\"><input type=\"button\" name=\"btn\" value=\"Quit\" onclick=\"location.href='/shutdown'\" /></td>";
		response << "</tr></table><br>\n";
	}

	void makeBody(std::ostream& response, MySession* pSession)
	{
		response << "<h2>Waffles Command-building Wizard</h2>\n";
		response << "<table border=\"1\" width=\"1000\"><tr><td>\n";

		if(m_mode == mode_goodbye)
		{
			response << "Done! The command to perform the specified task is:<br><br>\n\n";
			response << "<pre>\n";
			for(size_t i = 0; i < m_structData.size(); i++)
				response << "	" << m_structData[i] << "\n";
			response << "</pre><br>\n\n";
			response << "To execute this command, just paste it into a console window. To use it in a script, just paste it into the script.";

			response << "<form name=\"input\" action=\"wizard\" method=\"post\">\n";
			response << "<table width=60%><tr><td align=\"center\"><input type=\"submit\" name=\"btn\" value=\"Start Over\" /></td><td align=\"center\"><input type=\"button\" name=\"btn\" value=\"Finished\" onclick=\"location.href='/shutdown'\" /></td></tr></table>\n";
			response << "</form>\n";
		}
		else if(m_mode == mode_choose_one)
		{
			response << "<form name=\"input\" action=\"wizard\" method=\"post\">\n";
			response << "Please choose a value for <i>" << m_pEffectiveNode->tok() << "</i><br>\n";
			response << m_pEffectiveNode->descr() << "<br><br>\n\n";
			makeStandardButtons(response, pSession, false);
			response << "<table>\n";
			for(size_t i = 0; i < m_pEffectiveNode->choices().size(); i++)
			{
				UsageNode* pChoice = m_pEffectiveNode->choices()[i];
				response << "	<tr><td valign=top><input type=\"submit\" name=\"btn\" value=\"" << pChoice->tok() << "\"></td>";
				response << "<td>" << pChoice->descr() << "<br><br></td></tr>\n";
			}
			response << "</table>\n";
			response << "<br><br>\n\n";
			makeStandardButtons(response, pSession, false);
			response << "<input type=\"hidden\" name=\"pageid\" value=\"" << m_pSession->page() << "\" />\n";
			response << "</form>\n";
		}
		else if(m_mode == mode_options)
		{
			response << "<form name=\"input\" action=\"wizard\" method=\"post\">\n";
			response << "Check the options you want:<br><br>\n\n";
			makeStandardButtons(response, pSession, true);
			response << "<table>\n";
			for(size_t i = 0; i < m_pEffectiveNode->choices().size(); i++)
			{
				UsageNode* pChoice = m_pEffectiveNode->choices()[i];
				string s;
				pChoice->sig(&s);
				response << "	<tr><td valign=top><input type=checkbox name=\"" << i << "\"></td>";
				response << "<td valign=top>" << s << "</td>";
				response << "<td>" << pChoice->descr() << "<br><br></td><tr>\n";
			}
			response << "</table>\n";
			response << "<br><br>\n\n";
			makeStandardButtons(response, pSession, true);
			response << "<input type=\"hidden\" name=\"pageid\" value=\"" << m_pSession->page() << "\" />\n";
			response << "</form>\n";
		}
		else if(m_mode == mode_struct)
		{
			response << "<form name=\"input\" action=\"wizard\" method=\"post\">\n";
			response << "Please provide values for <i>" << m_pEffectiveNode->tok() << "</i><br>\n";
			response << m_pEffectiveNode->descr() << "<br><br>\n\n";
			makeStandardButtons(response, pSession, true);
			response << "<table>\n";
			size_t index = 0;
			for(size_t i = 0; i < m_pEffectiveNode->parts().size(); i++)
			{
				if(partType(m_pEffectiveNode, i) == pt_struct)
				{
					string arg = m_pEffectiveNode->parts()[i];
					UsageNode* pChoice = m_pEffectiveNode->choice(arg.c_str());
					if(!pChoice)
						pChoice = globalUsageNode(arg.c_str());
					response << "	<tr><td valign=top>" << pChoice->tok() << "</td>";
					response << "<td valign=top><input type=\"text\" name=\"" << index << "\" id=\"" << index << "\" ";
					string s = pChoice->default_value();
					if(s.length() == 0)
						s = m_pEffectiveNode->default_value();
					if(s.length() > 0)
						response << "value=\"" << s << "\" ";
					response << "/>";
					if(s.length() > 0 && looksLikeFilename(s.c_str()))
					{
						response << "<br><input type=\"button\" id=\"b" << index << "\" value=\"Browse\" onClick=\"onClickBrowse(this.id)\">";
					}
					response << "</td>";
					response << "<td>" << pChoice->descr() << "<br><br></td></tr><br>\n";
					index++;
				}
			}
			response << "</table>\n";
			response << "<br><br>\n\n";
			makeStandardButtons(response, pSession, true);
			response << "<input type=\"hidden\" name=\"pageid\" value=\"" << m_pSession->page() << "\" />\n";
			response << "</form>\n";
		}

		response << "</td></tr></table>\n";
	}

	void setChoices(GHttpParamParser& pp)
	{
		if(m_mode == mode_choose_one)
		{
			const char* pBtn = pp.find("btn");
			if(!pBtn)
				ThrowError("Expected a \"btn\" parameter value");
			m_selection = -1;
			for(size_t i = 0; i < m_pEffectiveNode->choices().size(); i++)
			{
				if(strcmp(m_pEffectiveNode->choices()[i]->tok(), pBtn) == 0)
				{
					m_selection = i;
					break;
				}
			}
			if(m_selection < 0)
				ThrowError("Unrecognized choice: ", pBtn);
		}
		else if(m_mode == mode_options)
		{
			m_options.clear();
			for(size_t i = 0; i < m_pEffectiveNode->choices().size(); i++)
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
			for(size_t i = 0; i < m_pEffectiveNode->parts().size(); i++)
			{
				if(partType(m_pEffectiveNode, i) == pt_struct)
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

	part_type partType(UsageNode* pNode, size_t part)
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
			UsageNode* pChoice = m_pEffectiveNode->choices()[m_selection];
			m_children.push_back(new Page(m_pController, m_pSession, this, pChoice));
		}
		else if(m_mode == mode_struct)
		{
			for(size_t i = 0; i < m_pEffectiveNode->parts().size(); i++)
			{
				if(partType(m_pEffectiveNode, i) == pt_loner)
				{
					string arg = m_pEffectiveNode->parts()[i];
					UsageNode* pChoice = m_pEffectiveNode->choice(arg.c_str());
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
				UsageNode* pChoice = m_pEffectiveNode->choices()[option];
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
			if(partType(m_pOrigNode, 0) == pt_string)
				stream << m_pOrigNode->tok() << " ";
			m_children[0]->makeCommand(stream);
		}
		else if(m_mode == mode_struct)
		{
			int stringPos = 0;
			int structPos = 0;
			int lonerPos = 0;
			for(size_t i = 0; i < m_pOrigNode->parts().size(); i++)
			{
				part_type pt = partType(m_pOrigNode, i);
				if(pt == pt_string)
				{
					stream << m_pOrigNode->parts()[i] << " ";
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
			GAssert((size_t)(stringPos + structPos + lonerPos) == m_pOrigNode->parts().size());
		}
		else if(m_mode == mode_options)
		{
			if(partType(m_pOrigNode, 0) == pt_string)
				stream << m_pOrigNode->tok() << " ";
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
	Page* pNext = NULL;

	// Find the page ID
	GHttpParamParser pp(szParams);
	const char* szPageId = pp.find("pageid");
	size_t pageId = (size_t)-1;
	if(szPageId)
		pageId = atoi(szPageId);

	// Check if the back button was pressed
	const char* szBtn = pp.find("btn");
	bool backingup = false;
	if(szBtn && _stricmp(szBtn, "Go back") == 0)
	{
		m_pageStack.pop();
		pNext = m_pageStack.top();
		m_pageStack.pop();
		backingup = true;
	}

	// Check if the start-over button was pressed
	if(szBtn && _stricmp(szBtn, "Start Over") == 0)
		pageId = (size_t)-1;

	// Check the page id. Start over if it is not what is expected.
	if(pageId != m_page || !m_pCurrentPage || !m_pRootPage)
	{
		// Start over at the beginning
		while(m_pageStack.size() > 0)
			m_pageStack.pop();
		m_page = 1;
		delete(m_pRootPage);
		m_pRootPage = new Page((Server*)m_pDPSession->server(), this, NULL, m_pRootNode);
		m_pCurrentPage = m_pRootPage;
		return;
	}
	else
	{
		if(!backingup)
			m_pCurrentPage->setChoices(pp);
		m_page++;
	}

	// If there are children, pick the first child
	if(!backingup)
	{
		if(!m_pCurrentPage->createChildPages())
			return;
	}
	if(!pNext && m_pCurrentPage->children().size() > 0)
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

















Server::Server(int port, GRand* pRand) : GDynamicPageServer(port, pRand), m_port(port)
{
	char buf[300];
	GApp::appPath(buf, 256, true);
	strcat(buf, "web/");
	GFile::condensePath(buf);
	m_basePath = buf;

	m_globals.push_back(makeAlgorithmUsageTree());
	m_globals.push_back(makeNeighborUsageTree());
	m_globals.push_back(makeCollaborativeFilterUsageTree());
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

void Server::addScript(std::ostream& response)
{
	response << "<script type=\"text/javascript\">\n";
	response << "var g_httpClient = null\n";
	response << "var g_fileBoxes = []\n";
	response << "var g_path = \"\"\n";

	response << "function httpPost(url, payload) {\n";
	response << "	g_httpClient = (window.XMLHttpRequest) ? new XMLHttpRequest() : new ActiveXObject('Microsoft.XMLHTTP')\n";
	response << "	g_httpClient.onreadystatechange = handleHttpResponse\n";
	response << "	g_httpClient.open('post', url, true)\n";
	response << "	g_httpClient.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded')\n";
	response << "	g_httpClient.send(payload)\n";
	response << "}\n";

	response << "function handleHttpResponse() {\n";
	response << "	if(g_httpClient.readyState != 4)\n";
	response << "		return\n";
	response << "	receiveFolderList(g_httpClient.responseText)\n";
	response << "}\n";

	response << "function receiveFolderList(t) {\n";
	response << "	for(var i = 0; i < g_fileBoxes.length; i++) {\n";
	response << "		var fb = g_fileBoxes[i]\n";
	response << "		var files = JSON.parse(t)\n";
	response << "		fb.options.length=0\n";
	response << "		for(var j = 0; j < files.length; j++)\n";
	response << "			fb.options[j] = new Option(files[j], \"0\")\n";
	response << "	}\n";
	response << "}\n";

	response << "function trimPath(path) {\n";
	response << "	var begin = 0\n";
	response << "	for(var i = 0; i < path.length; i++) {\n";
	response << "		if(path[i] == '/') {\n";
	response << "			if(path.length > i + 3 && path[i + 1] == '.' && path[i + 2] == '.' && path[i + 3] == '/' && (i - begin != 2 || path[begin] != '.' || path[begin + 1] != '.'))\n";
	response << "				return trimPath(path.substr(0, begin) + path.substr(i + 4))\n";
	response << "			begin = i + 1;\n";
	response << "		}\n";
	response << "	}\n";
	response << "	return path\n";
	response << "}\n";

	response << "function onClickFileBox(lb) {\n";
	response << "	var index = lb.selectedIndex\n";
	response << "	var val = lb.options[index].text\n";
	response << "	if(val[0] == '[') {\n";
	response << "		var dir = val.substr(1, val.length - 2)\n";
	response << "		httpPost('listfiles', 'cd=' + dir);\n";
	response << "		g_path = trimPath(g_path + dir + '/')\n";
	response << "	}\n";
	response << "	else {\n";
	response << "		var tb = lb.targetTextBox\n";
	response << "		tb.value = g_path + val\n";
	response << "		var par = lb.parentNode\n";
	response << "		var btn = document.createElement('input')\n";
	response << "		btn.setAttribute(\"type\", \"button\")\n";
	response << "		btn.setAttribute(\"id\", \"b\" + lb.targetTextBox.id)\n";
	response << "		btn.setAttribute(\"value\", \"Browse\")\n";
	response << "		btn.setAttribute(\"onClick\", \"onClickBrowse(this.id)\")\n";
	response << "		par.removeChild(lb)\n";
	response << "		par.appendChild(btn)\n";
	response << "	}\n";
	response << "}\n";

	response << "function onClickBrowse(buttonid) {\n";
	response << "	var btn = document.getElementById(buttonid)\n";
	response << "	var par = btn.parentNode\n";
	response << "	var lb = document.createElement('select')\n";
	response << "	lb.setAttribute(\"onChange\", \"onClickFileBox(this)\")\n";
	response << "	lb.size = 10\n";
	response << "	lb.targetTextBox = document.getElementById(buttonid.substr(1))\n";
	response << "	par.removeChild(btn)\n";
	response << "	par.appendChild(lb)\n";
	response << "	g_fileBoxes.push(lb)\n";
	response << "	httpPost('listfiles')\n";
	response << "}\n";
	response << "</script>\n";
}

// virtual
void Server::handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pDPSession, std::ostream& response)
{
	if(strcmp(szUrl, "/") == 0)
		szUrl = "/wizard";
	if(strcmp(szUrl, "/favicon.ico") == 0)
		return;
	if(strncmp(szUrl, "/wizard", 6) == 0)
	{
		MySession* pSession = getSession(pDPSession);
		pSession->doNext(szParams);
		response << "<html><head>\n";
		addScript(response);
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
		pSession->onShowPage();
		pSession->currentPage()->makeBody(response, pSession);
		response << "</body></html>\n";
	}
	else if(strncmp(szUrl, "/listfiles", 10) == 0)
	{
		char buf[300];
		if(!getcwd(buf, 300))
			ThrowError("getcwd failed");
		if(nParamsLen >= 3 && strncmp(szParams, "cd=", 3) == 0)
		{
			if(chdir(szParams + 3) != 0)
				cerr << "Failed to change dir from " << buf << " to " << szUrl << "\n";
		}
		response << "[";
		bool first = true;
		if(strlen(buf) >= 4)
		{
			response << "\"[..]\"";
			first = false;
		}
		{
			vector<string> folders;
			GFile::folderList(folders);
			for(vector<string>::iterator it = folders.begin(); it != folders.end(); it++)
			{
				const char* szDirName = it->c_str();
				if(first)
					first = false;
				else
					response << ",";
				response << "\"[" << szDirName << "]\"";
			}
		}
		{
			vector<string> files;
			GFile::fileList(files);
			for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
			{
				const char* szFilename = it->c_str();
				if(first)
					first = false;
				else
					response << ",";
				response << "\"" << szFilename << "\"";
			}
		}
		response << "]";
	}
	else if(strncmp(szUrl, "/shutdown", 9) == 0)
	{
		response << "<html><head></head><body onLoad=\"var closure=function() { window.top.opener = null; window.open('','_parent',''); window.close()}; setTimeout(closure,500)\"><h3>Goodbye!</h3></body></html>\n";
		shutDown();
	}
	else
		sendFileSafe(m_basePath.c_str(), szUrl + 1, response);
}

void Server::pump()
{
	double dLastActivity = GTime::seconds();
	GSignalHandler sh;
	while(m_bKeepGoing && sh.check() == 0)
	{
		if(process())
			dLastActivity = GTime::seconds();
		else
		{
			if(GTime::seconds() - dLastActivity > 900)	// 15 minutes
			{
				cout << "Shutting down due to inactivity for 15 minutes.\n";
				m_bKeepGoing = false;
			}
			else
				GThread::sleep(100);
		}
	}
	onShutDown();
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
	size_t addrLen = strlen(szAddress);
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

// Launch the web-based wizard tool
void do_wizard()
{
	int port = 8421;
	unsigned int seed = getpid() * (unsigned int)time(NULL);
	GRand prng(seed);
	bool serverStarted = false;
	try
	{
		Server server(port, &prng);
		serverStarted = true;
		string s = "http://localhost:";
		s += to_str(port);
		LaunchBrowser(s.c_str());

		// Pump incoming HTTP requests (this is the main loop)
		server.pump();
	}
	catch(std::exception& e)
	{
		if(serverStarted)
			throw e;
		else
		{
			cout << "Failed to launch the server. Typically, this means there is already another instance running on port " << port << ", so I am just going to open the browser and then exit.\n";
			string s = "http://localhost:";
			s += to_str(port);
			LaunchBrowser(s.c_str());
		}
	}
	cout << "Goodbye.\n";
}

// Generate a script that can be installed in /etc/bash_completion.d to tell
// BASH how to use waffles_wizard to complete commands.
void make_bash_completion_file()
{
	UsageNode* pRoot = makeMasterUsageTree();
	Holder<UsageNode> hRoot(pRoot);
	vector<UsageNode*>& choices = pRoot->choices();
	for(vector<UsageNode*>::iterator it = choices.begin(); it != choices.end(); it++)
	{
		UsageNode* pNode = *it;
		cout << "_" << pNode->tok() << "()\n{\n";
		cout << "	COMPREPLY=( $(waffles_wizard complete ${COMP_CWORD} ${COMP_WORDS[@]}) )\n";
		cout << "}\ncomplete -F _" << pNode->tok() << " " << pNode->tok() << "\n\n";
	}
}

// Returns true iff the first strlen(part) characters in "full" match "part"
bool doesMatch(const char* full, const char* part)
{
	while(*part != '\0')
	{
		if(*(full++) != *(part++))
			return false;
	}
	return true;
}

// This is a helper-class used by complete_command to complete commands
class CommandCompleter
{
protected:
	bool m_done;
	UsageNode* m_pAlgs;
	UsageNode* m_pCF;
	UsageNode* m_pNF;

public:
	CommandCompleter()
	: m_done(false), m_pAlgs(NULL), m_pCF(NULL), m_pNF(NULL)
	{
	}

	~CommandCompleter()
	{
		delete(m_pAlgs);
		delete(m_pCF);
		delete(m_pNF);
	}

	UsageNode* trySpecial(const char* tok)
	{
		if(strcmp(tok, "[algorithm]") == 0)
		{
			if(!m_pAlgs)
				m_pAlgs = makeAlgorithmUsageTree();
			return m_pAlgs;
		}
		else if(strcmp(tok, "[collab-filter]") == 0)
		{
			if(!m_pCF)
				m_pCF = makeCollaborativeFilterUsageTree();
			return m_pCF;
		}
		else if(strcmp(tok, "[neighbor-finder]") == 0)
		{
			if(!m_pNF)
				m_pNF = makeNeighborUsageTree();
			return m_pNF;
		}
		return NULL;
	}

	static void completeFilename(const char* tok)
	{
		string s;
		if(*tok == '~')
		{
			s = getenv("HOME");
			s += (tok + 1);
			tok = s.c_str();
		}
		int lastSlash = -1;
		for(int i = 0; tok[i] != '\0'; i++)
		{
			if(tok[i] == '/')
				lastSlash = i;
		}
		char origDir[300];
		char folder[300];
		folder[0] = '\0';
		if(lastSlash >= 0)
		{
			if(!getcwd(origDir, 300))
				ThrowError("getcwd failed");
			size_t len = std::min(299, lastSlash + 1);
			memcpy(folder, tok, len);
			folder[len] = '\0';
			if(chdir(folder) != 0)
				return;
			tok += lastSlash + 1;
		}
		if(strcmp(tok, "..") == 0)
		{
			cout << folder << "../.\n";
			cout << folder << "../_\n";
		}

		// Complete with matching dir names
		{
			vector<string> folders;
			GFile::folderList(folders);
			for(vector<string>::iterator it = folders.begin(); it != folders.end(); it++)
			{
				const char* fn = it->c_str();
				if(doesMatch(fn, tok))
				{
					// Do two completions (".", and "_") to trick BASH into not inserting a space. If
					// the user presses TAB twice to see the full list of completions, this might be
					// confusing--well, do you know of a better solution?
					cout << folder << fn << "/.\n";
					cout << folder << fn << "/_\n";
				}
			}
		}

		// Complete with matching filenames
		{
			vector<string> files;
			GFile::fileList(files);
			for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
			{
				const char* fn = it->c_str();
				if(doesMatch(fn, tok))
					cout << folder << fn << "\n";
			}
		}

		// Restore the original folder
		if(lastSlash >= 0)
		{
			if(chdir(origDir) != 0)
			{
			}
		}
	}

	bool doCompletion(GArgReader& args, UsageNode* pNode, size_t nodePos)
	{
		vector<string>& parts = pNode->parts();
		size_t flex = 0; // the number of flexible parts
		for(size_t i = 0; i < parts.size(); i++)
		{
			if(parts[i][0] == '[' || parts[i][0] == '<')
				flex++;
		}
		while(nodePos < parts.size())
		{
			const char* part = parts[nodePos].c_str();
			const char* tok = args.peek();
			if(args.size() <= 1) // If this is the arg to complete
			{
				if(part[0] != '[' && part[0] != '<' && doesMatch(part, tok))
				{
					cout << part << "\n";
					m_done = true;
					return true;
				}
				else
				{
					UsageNode* pExpanded = pNode->choice(part);
					if(!pExpanded && nodePos > 0)
						pExpanded = trySpecial(part);
					if(pExpanded)
						doCompletion(args, pExpanded, 0);
					else
					{
						vector<UsageNode*>& choices = pNode->choices();
						if(flex == 1 && choices.size() > 0)
						{
							// Complete with matching choices
							for(vector<UsageNode*>::iterator it = choices.begin(); it != choices.end(); it++)
							{
								if(doesMatch((*it)->tok(), tok))
									cout << (*it)->tok() << "\n";
							}
						}
						else if(pNode->default_value().length() > 0)
						{
							if(Page::looksLikeFilename(pNode->default_value().c_str()))
								completeFilename(tok);
							else
								cout << pNode->default_value() << "\n";
						}
						else
						{
							// ?
						}
						if(part[0] != '<')
							m_done = true;
						return true;
					}
				}
			}
			else
			{
				if(strcmp(part, tok) == 0)
					args.pop_string();
				else
				{
					UsageNode* pExpanded = pNode->choice(part); // find a choice matching the template
					if(!pExpanded)
						pExpanded = pNode->choice(tok); // find a choice matching the token
					if(!pExpanded && nodePos > 0)
						pExpanded = trySpecial(part); // try to match a special tag (like [algorithm])
					vector<UsageNode*>& choices = pNode->choices();
					if(!pExpanded && choices.size() == 1 && choices[0]->tok()[0] == '[' && tok[0] >= '0' && tok[0] <= '9') // This check is a special case to accept the [instance_count] portion of the <contents> tag in some ensemble algorithms.
						pExpanded = choices[0];
					if(pExpanded)
					{
						if(doCompletion(args, pExpanded, 0) && part[0] == '<')
							nodePos--; // Allow for another option
					}
					else
					{
						if(choices.size() == 0 || flex != 1)
							args.pop_string(); // it's a free-form value, so accept anything
						else
						{
							if(part[0] != '<')
								ThrowError("Unexpected token, ", tok, ", in arg ", to_str(args.get_pos() - 3));
							GAssert(nodePos + 1 >= parts.size());
							return false; // Failed to match any option, so move on to the next template item. (All the other cases return true to indicate that it is possible to match the same template item again.)
						}
					}
				}
			}
			if(m_done)
				return true;
			nodePos++;
		}
		return true;
	}
};

// Complete the command specified starting with arg 3. All completion candidates are printed to stdout,
// separated by a newline.
//
// pArgs[0] = the name of this app, "waffles_wizard"
// pArgs[1] = the command, "complete"
// pArgs[2] = the index of the arg that should be completed (as if pArg[3] were at index 0)
// pArgs[3] to pArgs[*] = the command to complete
void complete_command(int nArgs, char* pArgs[])
{
	try
	{
		if(nArgs < 3)
			ThrowError("Expected more args");
		size_t cur = atoi(pArgs[2]);
		ArrayHolder<char*> hArgs;
		if((int)cur >= nArgs - 3)
		{
			// Append an empty-string to the args
			if((int)cur > nArgs - 3)
				ThrowError("Out of range");
			char** pArgsNew = new char*[nArgs + 1];
			hArgs.reset(pArgsNew);
			for(size_t i = 0; i < (size_t)nArgs; i++)
				pArgsNew[i] = (char*)pArgs[i];
			pArgsNew[nArgs] = (char*)"";
			pArgs = pArgsNew;
			nArgs++;
		}
		GArgReader args(nArgs, pArgs);
		args.pop_string(); // waffles_wizard
		args.pop_string(); // complete
		args.pop_string(); // the current arg number
		if(cur < 1)
			ThrowError("expected cur to be >= 1");
		const char* szApp = args.pop_string();
		UsageNode* pNode = NULL;
		if(doesMatch(szApp, "waffles_learn"))
			pNode = makeLearnUsageTree();
		else if(doesMatch(szApp, "waffles_transform"))
			pNode = makeTransformUsageTree();
		else if(doesMatch(szApp, "waffles_recommend"))
			pNode = makeRecommendUsageTree();
		else if(doesMatch(szApp, "waffles_plot"))
			pNode = makePlotUsageTree();
		else if(doesMatch(szApp, "waffles_dimred"))
			pNode = makeDimRedUsageTree();
		else if(doesMatch(szApp, "waffles_cluster"))
			pNode = makeClusterUsageTree();
		else if(doesMatch(szApp, "waffles_generate"))
			pNode = makeGenerateUsageTree();
		else if(doesMatch(szApp, "waffles_audio"))
			pNode = makeAudioUsageTree();
		else if(doesMatch(szApp, "waffles_sparse"))
			pNode = makeSparseUsageTree();
		else
			ThrowError("unrecognized app");
		Holder<UsageNode> hNode(pNode);
		CommandCompleter cc;
		cc.doCompletion(args, pNode, 1);
	}
	catch(std::exception& e)
	{
		string s = "Error: ";
		s += e.what();
		for(int i = s.length() - 1; i >= 0; i--)
		{
			if(s[i] == ' ')
				s[i] = '_';
		}
		cout << s << "\n";
	}
}

int main(int nArgs, char* pArgs[])
{
	int ret = 1;
	try
	{
		if(nArgs < 2)
			do_wizard(); // do a web-based wizard utility
		else if(strcmp(pArgs[1], "make_bash_completion_file") == 0)
			make_bash_completion_file(); // generate a script that is used to set up bash command-completion
		else if(strcmp(pArgs[1], "complete") == 0)
			complete_command(nArgs, pArgs); // complete the specified command
		else
			ThrowError("Unrecognized command: ", pArgs[1]);
		ret = 0;
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return ret;
}
