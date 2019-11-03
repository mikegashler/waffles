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

void Forum::format_comment_recursive(Server* pServer, GDomNode* pEntry, std::ostream& os, std::string& id, bool allow_direct_reply, size_t depth)
{
	// Extract relevant data
	const char* szUsername = pServer->get_account(pEntry->getInt("user"))->username();
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
				format_comment_recursive(pServer, pReplies->get(i), os, child_allow_direct_replies ? child_id : id, child_allow_direct_replies, depth - 1);
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
	// Change the username if requested
	Account* pAccount = getAccount(pSession);
	GDomNode* pChangeName = pIn->getIfExists("changename");
	if(pChangeName)
	{
		const char* szNewName = pChangeName->asString();
		string sNewName = szNewName;
		for(size_t i = 0; i < sNewName.length(); i++)
		{
			if(sNewName[i] == ' ')
				sNewName[i] = '_';
		}
		if(pAccount)
		{
			if(!pAccount->changeUsername(pServer, sNewName.c_str()))
				pOut->add(&doc, "error", "Sorry, that username is not available");
		}
		else
			pOut->add(&doc, "error", "Not currently logged in");
	}

	// Request the whole comments file
	GJsonAsADatabase& jaad = pServer->jaad();
	const GDomNode* pResponse = jaad.apply(pIn->getString("file"), "", &doc);
	std::ostringstream os;

	// Convert to HTML
	if(pResponse)
	{
		if(pResponse->type() == GDomNode::type_list)
		{
			// Convert hierarchical list of comments into HTML
			os << "<br><br><h2>Visitor Comments:</h2>\n";

			// Add the username
			if(pAccount)
			{
				os << "Posting as <a href=\"#javascript:void(0)\" onclick=\"tog_viz('change_username')\">" << pAccount->username() << "</a><br>\n";
				os << "<div class=\"hidden\" id=\"change_username\">Change username to <input type=\"text\" id=\"username\" value=\"" << pAccount->username() << "\"><input type=\"button\" onclick=\"change_username()\" value=\"Change\"></div>\n";
			}
			else
				os << "Not logged in<br>\n";

			// Add the comments
			for(size_t i = 0; i < pResponse->size(); i++)
			{
				GDomNode* pEntry = pResponse->get(i);
				string id = "r";
				id += to_str(i);
				format_comment_recursive(pServer, pEntry, os, id, true, 12);
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
	}
	else
	{
		os << "<br><br><h2>Visitor Comments:</h2>\n";

		// Add the username
		if(pAccount)
		{
			os << "Posting as <a href=\"#javascript:void(0)\" onclick=\"tog_viz('change_username')\">" << pAccount->username() << "</a><br>\n";
			os << "<div class=\"hidden\" id=\"change_username\">Change username to <input type=\"text\" id=\"username\" value=\"" << pAccount->username() << "\"><input type=\"button\" onclick=\"change_username()\" value=\"Change\"></div>\n";
		}
		else
			os << "Not logged in<br>\n";
		os << "<textarea id=\"rt\" rows=\"2\" cols=\"50\"></textarea><br>\n";
		os << "<input type=\"button\" onclick=\"post_comment('rt');\" value=\"Post\">\n";
	}
	pOut->add(&doc, "html", os.str().c_str());
}

string HTML_scrub_string(const char* szString)
{
	std::ostringstream stream;
	while(*szString != '\0')
	{
		if(*szString == '&')
			stream << "&amp;";
		else if(*szString == '<')
			stream << "&lt;";
		else if(*szString == '>')
			stream << "&gt;";
		else
			stream << *szString;
		++szString;
	}
	return stream.str();
}

string JSON_encode_string(const char* szString)
{
	std::ostringstream stream;
	stream << '"';
	while(*szString != '\0')
	{
		if(*szString < ' ')
		{
			switch(*szString)
			{
				case '\b': stream << "\\b"; break;
				case '\f': stream << "\\f"; break;
				case '\n': stream << "\\n"; break;
				case '\r': stream << "\\r"; break;
				case '\t': stream << "\\t"; break;
				default:
					stream << (*szString);
			}
		}
		else if(*szString == '\\')
			stream << "\\\\";
		else if(*szString == '"')
			stream << "\\\"";
		else
			stream << (*szString);
		++szString;
	}
	stream << '"';
	return stream.str();
}

size_t portions(const char* szString, double* whitespace, double* letters, double* caps)
{
	size_t _letters = 0;
	size_t _caps = 0;
	size_t _chars = 0;
	size_t _space = 0;
	while(*szString != '\0')
	{
		if(*szString >= 'a' && *szString <= 'z')
			++_letters;
		else if(*szString >= 'A' && *szString <= 'Z')
		{
			++_letters;
			++_caps;
		}
		else if(*szString <= ' ')
			++_space;
		++_chars;
		++szString;
	}
	*whitespace = (double)_space / _chars;
	*letters = (double)_letters / _chars;
	*caps = (double)_caps / _letters;
	return _chars;
}

void Forum::ajaxAddComment(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut)
{
	// Get the data
	Account* pAccount = getAccount(pSession);
	if(!pAccount)
		throw Ex("You must be logged in to comment.");
	size_t user_id = pServer->user_id(pAccount->username());
	const char* szFilename = pIn->getString("file");
	const char* szId = pIn->getString("id");
	const char* szIpAddress = pSession->connection()->getIPAddress();
	const char* szComment = pIn->getString("comment");

	// Evaluate the comment
	if(strstr(szComment, "://"))
		throw Ex("Comment rejected. Hyperlinks are not allowed.");
	if(strstr(szComment, "href="))
		throw Ex("Comment rejected. Hyperlinks are not allowed.");
	double _ws, _letters, _caps;
	size_t len = portions(szComment, &_ws, &_letters, &_caps);
	if(len > 3 && _ws > 0.5)
		throw Ex("Comment rejected. Too much whitespace.");
	if(len > 25 && _ws < 0.02)
		throw Ex("Comment rejected. Use more spaces.");
	if(_letters < 0.6)
		throw Ex("Comment rejected. Comments should be mostly words, not symbols");
	if(_caps > 0.3)
		throw Ex("Comment rejected. Too many capitalized letters.");

	// Parse the ID (to determine where to insert the comment)
	if(*szId != 'r')
		throw Ex("Invalid ID");
	if(*szId == '_')
		throw Ex("Invalid ID");
	szId++;
	std::ostringstream cmd;
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
			cmd << '[';
			while(*szId >= '0' && *szId <= '9')
			{
				cmd << *szId;
				++szId;
			}
			cmd << "].replies";
		}
		else
			throw Ex("Invalid ID");
	}

	// Construct the JAAD command
	string sDate;
	GTime::appendTimeStampValue(&sDate, "-", " ", ":", true);
	string encodedIP = JSON_encode_string(szIpAddress);
	string encodedDate = JSON_encode_string(sDate.c_str());
	string encodedComment = JSON_encode_string(HTML_scrub_string(szComment).c_str());
	cmd << " += {\"ip\":" << encodedIP;
	cmd << ",\"user\":" << to_str(user_id);
	cmd << ",\"date\":" << encodedDate;
	cmd << ",\"comment\":" << encodedComment;
	cmd << "}";

	// Send the request
	GJsonAsADatabase& jaad = pServer->jaad();
	const GDomNode* pResponse = jaad.apply(szFilename, cmd.str().c_str(), &doc);
	pOut->add(&doc, "response", pResponse);

	// Log this comment
	std::ostringstream cmd2;
	cmd2 << "+={\"ip\":" << encodedIP;
	cmd2 << ",\"user\":" << to_str(user_id);
	cmd2 << ",\"date\":" << encodedDate;
	cmd2 << ",\"file\":" << JSON_encode_string(szFilename);
	cmd2 << ",\"comment\":" << encodedComment;
	cmd2 << "}";
	jaad.apply("comments.json", cmd2.str().c_str(), &doc);
}

void Forum::pageFeed(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Process params
	if(pSession->paramsLen() > 0)
	{
		GHttpParamParser params(pSession->params());
		const char* szAction = params.find("action");
		if(strcmp(szAction, "ban") == 0)
		{
			// Ban checked users
			pServer->log("Banning users");
			auto map = params.map();
			for(std::map<const char*, const char*, GClasses::strComp>::iterator it = map.begin(); it != map.end(); it++)
			{
				if(strncmp(it->first, "ban_", 4) == 0)
				{
					size_t account_index = atoi(it->first + 4);
					Account* pUserAccount = pServer->get_account(account_index);
					if(strcmp(it->second, "true") == 0)
					{
						string s = "Banning user: ";
						s += pUserAccount->username();
						pServer->log(s.c_str());
						pUserAccount->makeBanned(true);
					}
				}
			}

			// Scrub the comments
			GJsonAsADatabase& jaad = pServer->jaad();
			jaad.flush(true);
			scrub_all_comment_files(pServer);
			response << "[Comments files scrubbed]<br>\n";
		}
		else if(strcmp(szAction, "flush") == 0)
		{
			pServer->jaad().flush(true);
			response << "[Comments files flushed]<br>\n";
		}
	}

	// Check access privileges
	Account* pAccount = getAccount(pSession);
	if(!pAccount->isAdmin())
	{
		response << "Sorry, you must be an admin to access this page.";
		return;
	}

	// Load the log file
	GJsonAsADatabase& jaad = pServer->jaad();
	GDom doc;
	const GDomNode* pNode = jaad.apply("comments.json", "", &doc);

	// Generate a page
	if(pNode)
	{
		response << "<h2>Recent comments</h2>\n";
		response << "<form method=\"post\">";
		response << "<input type=\"hidden\" name=\"action\" value=\"ban\" />\n";
		response << "<table><tr><td>Ban user</td><td>Date</td><td>Username</td><td>IP</td><td>Comment</td></tr>\n";
		for(size_t i = 0; i < pNode->size(); i++)
		{
			GDomNode* pComment = pNode->get(i);
			response << "<tr><td><input type=\"checkbox\" name=\"ban_" << to_str(pComment->getInt("user")) << "\"></td>";
			response << "<td>" << pComment->getString("date") << "</td>";
			response << "<td>" << pServer->get_account(pComment->getInt("user"))->username() << "</td>";
			response << "<td>" << pComment->getString("ip") << "</td>";
			response << "<td>" << pComment->getString("comment") << "</td>";
			response << "\n";
		}
		response << "</table>\n";
		response << "<input type=\"submit\" value=\"Ban checked users and remove all of their comments\"></form>";
		response << "</form>";
	}
	else
		response << "No recent comments\n";

	// Form to flush the comments files
	response << "<br><br><br><br>\n";
	response << "<form name=\"flushform\" method=\"get\">\n";
	response << "	Flush the comments files:<br>\n";
	response << "	<input type=\"hidden\" name=\"action\" value=\"flush\" />\n";
	response << "	<input type=\"submit\" value=\"Flush\">\n";
	response << "</form><br><br>\n\n";
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
	/*if(pd.extStart == pd.len)
	{
		s += "/index.html";
		GFile::parsePath(s.c_str(), &pd);
	}*/

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
	string sScript = "\nlet comments_file = \"";
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

bool Forum::purge_comments_from_banned_users(Server* pServer, GDomNode* pList)
{
	bool made_changes = false;
	if(pList->type() != GDomNode::type_list)
		return made_changes;
	for(size_t i = pList->size() - 1; i < pList->size(); --i)
	{
		GDomNode* pComment = pList->get(i);
		GDomNode* pUserId = pComment->getIfExists("user");
		if(pUserId)
		{
			size_t userId = pUserId->asInt();
			if(userId >= 0 && userId < pServer->account_count())
			{
				Account* pAccount = pServer->get_account(userId);
				if(pAccount->isBanned())
				{
					pList->del(i);
					made_changes = true;
				}
				else
				{
					GDomNode* pReplies = pComment->getIfExists("replies");
					if(pReplies)
					{
						if(purge_comments_from_banned_users(pServer, pReplies))
							made_changes = true;
					}
				}
			}
		}
	}
	return made_changes;
}

void Forum::scrub_all_comment_files(Server* pServer)
{
	std::vector<std::string> file_list;
	GFile::fileListRecursive(file_list, pServer->m_basePath.c_str());
	for(size_t i = 0; i < file_list.size(); i++)
	{
		if(ends_with(file_list[i], "comments.json"))
		{
			GDom dom;
			dom.loadJson(file_list[i].c_str());
			GDomNode* pRootList = dom.root();
			if(purge_comments_from_banned_users(pServer, pRootList))
				dom.saveJson(file_list[i].c_str());
		}
	}
}
