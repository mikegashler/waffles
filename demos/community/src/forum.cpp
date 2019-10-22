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


#include <GClasses/GHolders.h>
#include <GClasses/GHtml.h>
#include <GClasses/GText.h>
#include <GClasses/GTime.h>
#include <GClasses/GFile.h>
#include <vector>
#include <sstream>
#include "forum.h"
#include "server.h"

using std::ostream;
using std::string;
using std::vector;
using std::cout;

void Forum::format_comment_recursive(GDomNode* pEntry, std::ostream& os, std::string& id, bool allow_direct_reply, size_t depth)
{
	// Extract relevant data
	const char* szUsername = pEntry->getString("user");
	const char* szDate = pEntry->getString("date");
	const char* szComment = pEntry->getString("comment");
	GDomNode* pReplies = pEntry->getIfExists("replies");

	// Add the comment enclosed in a "bubble" div
	os << "<div class=\"bubble\"><table cellpadding=10px><tr>\n";
	os << "<td valign=top align=right>";
	os << szUsername << "<br>";
	os << szDate;
	os << "</td><td valign=top>";
	os << szComment;
	os << "<br><a href=\"#javascript:void(0)\" onclick=\"tog_viz('" << id << "')\">reply</a>";
	os << "</td></tr>\n";
	os << "</table></div><br>\n";

	if(depth > 0)
	{
		// Recursively add replies
		if(pReplies)
		{
			os << "<div class=\"indent\">";
			for(size_t i = 0; i < pReplies->size(); i++)
			{
				string child_id = id;
				child_id += "_";
				child_id += to_str(i);
				bool child_allow_direct_replies = true;
				if(i == pReplies->size() - 1)
					child_allow_direct_replies = false;
				format_comment_recursive(pReplies->get(i), os, child_allow_direct_replies ? child_id : id, child_allow_direct_replies, depth - 1);
			}
			os << "</div>\n";
		}

		if(allow_direct_reply)
		{
			// Add a hidden div with a reply field and post button
			os << "<div class=\"hidden indent\" id=\"" << id << "\"><textarea id=\"" << id << "t\" rows=\"2\" cols=\"50\"></textarea><br>";
			os << "<button type=\"button\" onclick=\"post_comment('" << id << "t')\">Post</button><br><br></div>\n";
		}
	}
}


void Forum::ajaxGetForumHtml(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut)
{
	// Make a request node
	GDomNode* pRequest = doc.newObj();
	pRequest->add(&doc, "file", pIn->getString("file"));
	pRequest->add(&doc, "cmd", "");

	// Request the whole file
	GJsonAsADatabase& jaad = pServer->jaad();
	const GDomNode* pResponse = jaad.apply(pRequest, &doc);
	if(pResponse)
	{
		// Convert to HTML
		std::ostringstream os;
		if(pResponse->type() == GDomNode::type_list)
		{
			// Convert hierarchical list of comments into HTML
			os << "<h2>Visitor Comments:</h2>\n";
			for(size_t i = 0; i < pResponse->size(); i++)
			{
				GDomNode* pEntry = pResponse->get(i);
				string id = "r";
				id += to_str(i);
				format_comment_recursive(pEntry, os, id, true, 12);
			}
			os << "<textarea id=\"rt\" rows=\"2\" cols=\"50\"></textarea><br>\n";
			os << "<input type=\"button\" onclick=\"post_comment('rt');\" value=\"Post\">\n";
		}
		else
		{
			// Convert error message into HTML
			os << "<br><font color=\"red\">[Comments currently unavailable because: ";
			os << pResponse->getString("jaad_error");
			os << "]</font><br>\n";
		}
		pOut->add(&doc, "html", os.str().c_str());
	}
}

void Forum::ajaxAddComment(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut)
{
	// Get the data
	Account* pAccount = getAccount(pSession);
	if(!pAccount)
		throw Ex("You must be logged in to comment.");
	const char* szUsername = pAccount->username();
	const char* szFilename = pIn->getString("file");
	const char* szId = pIn->getString("id");
	const char* szIpAddress = pSession->connection()->getIPAddress();
	const char* szComment = pIn->getString("comment");
	string sDate;
	GTime::appendTimeStampValue(&sDate, "-", " ", ":", true);

	// Parse the ID (to determine where to insert the comment)
	if(*szId != 'r')
		throw Ex("Invalid ID");
	if(*szId == '_')
		throw Ex("Invalid ID");
	szId++;
	string cmd = "";
	size_t depth = 0;
	while(true)
	{
		if(*szId == 't')
			break;
		else if(*szId == '_')
		{
			++szId;
			if(*szId == '_' || *szId == 't')
				throw Ex("Invalid ID");
		}
		else if(*szId >= '0' && *szId <= '9')
		{
			if(++depth > 20)
				throw Ex("Invalid ID");
			cmd += '[';
			while(*szId >= '0' && *szId <= '9')
			{
				cmd += *szId;
				++szId;
			}
			cmd += "].replies";
		}
		else
			throw Ex("Invalid ID");
	}

	// Construct the JAAD command
	cmd += " += {\"ip\":\"";
	cmd += szIpAddress;
	cmd += "\",\"user\":\"";
	cmd += szUsername;
	cmd += "\",\"date\":\"";
	cmd += sDate;
	cmd += "\",\"comment\":\"";
	cmd += szComment;
	cmd += "\"}";

	// Make a request node
	GDomNode* pRequest = doc.newObj();
	pRequest->add(&doc, "file", szFilename);
	pRequest->add(&doc, "cmd", cmd.c_str());

	// Send the request
	GJsonAsADatabase& jaad = pServer->jaad();
	const GDomNode* pResponse = jaad.apply(pRequest, &doc);
	pOut->add(&doc, "response", pResponse);

	// Log this comment
	string cmd2 = "+={\"ip\":\"";
	cmd2 += szIpAddress;
	cmd2 += "\",\"user\":\"";
	cmd2 += szUsername;
	cmd2 += "\",\"date\":\"";
	cmd2 += sDate;
	cmd2 += "\",\"file\":\"";
	cmd2 += szFilename;
	cmd2 += "\",\"comment\":\"";
	cmd2 += szComment;
	cmd2 += "\"}";
	GDomNode* pReq2 = doc.newObj();
	pReq2->add(&doc, "file", "comments.json");
	pReq2->add(&doc, "cmd", cmd.c_str());
	jaad.apply(pReq2, &doc);
}

void Forum::pageFeed(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Check access privileges
	Account* pAccount = getAccount(pSession);
	if(!pAccount->isAdmin())
	{
		response << "Sorry, you must be an admin to access this page.";
		return;
	}

	// Load the log file
	string filename = pServer->m_basePath;
	filename += "comments_log.json";
	GDom dom;
	dom.loadJson(filename.c_str());
	GDomNode* pNode = dom.root();

	// Generate a page
	response << "<h2>Recent comments</h2>\n";
	response << "<table><tr><td>Ban user</td><td>Date</td><td>Username</td><td>IP</td><td>Comment</td></tr>\n";
	for(size_t i = 0; i < pNode->size(); i++)
	{
		GDomNode* pComment = pNode->get(i);
		response << "<tr><td><input type=\"checkbox\"></td>";
		response << "<td>" << pComment->getString("date") << "</td>";
		response << "<td>" << pComment->getString("user") << "</td>";
		response << "<td>" << pComment->getString("ip") << "</td>";
		response << "<td>" << pComment->getString("comment") << "</td>";
		response << "\n";
	}
	response << "</table>\n";
}

void Forum::pageForumWrapper(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Parse the url
	string s = pSession->url();
	if(s.length() < 3 || s.substr(0, 3).compare("/c/") != 0)
		throw Ex("Unexpected url");
	s = s.substr(3);
	if(s[s.length() - 1] == '/')
		s += "index.html";
	PathData pd;
	GFile::parsePath(s.c_str(), &pd);
	if(pd.extStart == pd.len)
	{
		s += "/index.html";
		GFile::parsePath(s.c_str(), &pd);
	}

	// If it's not an HTML file, just send the file
	if(s.substr(pd.extStart).compare(".html") != 0)
	{
		pSession->connection()->sendFileSafe(pServer->m_basePath.c_str(), s.c_str(), response);
		return;
	}

	// Parse the HTML
	string fullPath = pServer->m_basePath;
	fullPath += s;
	GHtmlDoc doc(fullPath.c_str());
	GHtmlElement* pElHtml = doc.document()->childTag("html");
	if(!pElHtml)
		return throw Ex("Expected an html tag");
	GHtmlElement* pElHead = pElHtml->childTag("head");
	if(!pElHead)
		pElHead = new GHtmlElement(pElHtml, "head", 0);
	GHtmlElement* pElStyle = pElHead->childTag("style");
	if(!pElStyle)
		pElStyle = new GHtmlElement(pElHead, "style");
	GHtmlElement* pElBody = pElHtml->childTag("body");
	if(!pElBody)
		return throw Ex("Expected a body tag");

	// Inject the comments stuff
	string& sStyle = pServer->cache("chat_style.css");
	GHtmlElement* pAddedStyle = new GHtmlElement(pElStyle, sStyle.c_str());
	pAddedStyle->text = true;
	string sScript = "let comments_file = \"";
	sScript += s.substr(0, pd.extStart);
	sScript += "_comments.json\";\n";
	sScript += pServer->cache("chat_script.js");
	GHtmlElement* pAddedScript = new GHtmlElement(pElBody, "script", 0);
	pAddedScript->addAttr("type", "\"text/javascript\"");
	GHtmlElement* pAddedScriptContent = new GHtmlElement(pAddedScript, sScript.c_str());
	pAddedScriptContent->text = true;
	GHtmlElement* pAddedComments = new GHtmlElement(pElBody, "div");
	pAddedComments->addAttr("id", "\"comments\"");

	doc.document()->write(response);
}
