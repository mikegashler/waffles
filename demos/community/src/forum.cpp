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
cout << "id=" << id << "\n";
	// Extract relevant data
	const char* szUser = pEntry->getString("user");
	const char* szDate = "2019-01-01 22:21"; // TODO: Make this dynamic
	const char* szComment = pEntry->getString("comment");
	GDomNode* pReplies = pEntry->getIfExists("replies");

	// Add the comment enclosed in a "bord" div
	os << "<div class=\"bord\"><table cellpadding=10px><tr>\n";
	os << "<td valign=top align=right>";
	os << szUser << "<br>";
	os << szDate;
	os << "</td><td valign=top>";
	os << szComment;
	os << "<br><a href=\"#\" onclick=\"tog_viz('" << id << "')\">reply</a>";
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
	pRequest->add(&doc, "auth", "7b4932af10354c01");
	pRequest->add(&doc, "cmd", "");

	// Request the whole file
	GJsonAsADatabase& jaad = pServer->jaad();
	const GDomNode* pResponse = jaad.apply(pRequest, &doc);
	if(pResponse)
	{
		// Convert to HTML
		if(pResponse->type() != GDomNode::type_list)
			throw Ex("Expected a list type");
		std::ostringstream os;
		for(size_t i = 0; i < pResponse->size(); i++)
		{
			GDomNode* pEntry = pResponse->get(i);
			string id = "r";
			id += to_str(i);
			format_comment_recursive(pEntry, os, id, true, 12);
		}
		pOut->add(&doc, "html", os.str().c_str());
	}
}

void Forum::ajaxAddComment(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut)
{
	// Get the data
	const char* szFilename = pIn->getString("file");
	const char* szUser = pIn->getString("user");
	const char* szId = pIn->getString("id");
	const char* szIpAddress = pSession->connection()->getIPAddress();
	const char* szComment = pIn->getString("comment");

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
	cmd += szUser;
	cmd += "\",\"comment\":\"";
	cmd += szComment;
	cmd += "\"}";

	// Make a request node
	GDomNode* pRequest = doc.newObj();
	pRequest->add(&doc, "file", szFilename);
	pRequest->add(&doc, "auth", "7b4932af10354c01");
	pRequest->add(&doc, "cmd", cmd.c_str());

	// Send the request
	GJsonAsADatabase& jaad = pServer->jaad();
	const GDomNode* pResponse = jaad.apply(pRequest, &doc);
	pOut->add(&doc, "response", pResponse);
}
