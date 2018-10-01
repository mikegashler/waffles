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
#include "editor.h"
#include "server.h"

using std::ostream;
using std::string;
using std::vector;
using std::cout;


string goUpOneDirectory(string& dir)
{
	size_t lastSlash = dir.find_last_of("/");
	if(lastSlash == string::npos)
		return dir;
	else
	{
		return dir.substr(0, lastSlash);
	}
}

void ensureFolderExists(string& folderName)
{
	bool ok;
#ifdef WINDOWS
	ok = (mkdir(folderName.c_str()) == 0);
#else
	ok = (mkdir(folderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0); // read/write/search permissions for owner and group, and with read/search permissions for others
#endif
	if(!ok)
	{
		//throw Ex("Failed to make directory: ", folderName);
	}
}

void Editor::ajaxFilelist(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut)
{
	Account* pAccount = getAccount(pSession);
	string foldername = pServer->m_basePath.c_str();
	foldername += "usercontent/";
	foldername += pAccount->username();
	ensureFolderExists(foldername);
	foldername += "/pages";
	ensureFolderExists(foldername);
	foldername += "/";
	const char* szCD = pIn->getString("folder");
	if(strstr(szCD, ".") != nullptr)
	{
		// found a dot
		if(szCD[0] == '.' && szCD[1] == '.')
		{
			// It's ".."
			pAccount->path() = goUpOneDirectory(pAccount->path());
		}
		else
		{
			// It's probably just a single dot, so don't change the path
		}
	}
	else
	{
		// Change into the specified folder
		pAccount->path() += "/";
		pAccount->path() += szCD;
	}
	foldername += pAccount->path();

	// Add the path
	pOut->add(&doc, "path", pAccount->path().c_str());

	// Add the folder list
	GDomNode* pFolders = doc.newList();
	pOut->add(&doc, "folders", pFolders);
	vector<string> folderList;
	GFile::folderList(folderList, foldername.c_str());
	for(size_t i = 0; i < folderList.size(); i++)
		pFolders->add(&doc, folderList[i].c_str());

	// Add the file list
	GDomNode* pFiles = doc.newList();
	pOut->add(&doc, "files", pFiles);
	vector<string> fileList;
	GFile::fileList(fileList, foldername.c_str());
	for(size_t i = 0; i < fileList.size(); i++)
		pFiles->add(&doc, fileList[i].c_str());
}

void Editor::ajaxSaveText(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut)
{
	// Find the filename
	Account* pAccount = getAccount(pSession);
	string filename = pServer->m_basePath.c_str();
	filename += "usercontent/";
	filename += pAccount->username();
	filename += "/pages";
	filename += "/";
	filename += pIn->getString("filename");

	// Get the new page
	const char* szNewPage = pIn->getString("page");
	size_t pageLen = strlen(szNewPage);

	// Write the file
	std::ofstream s;
	s.exceptions(std::ios::badbit | std::ios::failbit);
	try
	{
		s.open(filename.c_str(), std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to create the file, ", filename, ". ", strerror(errno));
	}
	s.write(szNewPage, pageLen);

	cout << "User " << pAccount->username() << " saved raw text to: " << filename << "\n";
	cout.flush();
}

void Editor::archiveFile(const char* szFilename)
{
	// Divide into folder and file
	string filename = szFilename;
	string folderOnly;
	string fileOnly;
	size_t pos = filename.find_last_of("/");
	if(pos == string::npos)
	{
		cout << "Error, expected a fully-qualified filename: " << szFilename << "\n";
		folderOnly = "";
		fileOnly = szFilename;
	}
	else
	{
		folderOnly = filename.substr(0, pos + 1);
		fileOnly = filename.substr(pos + 1);
	}

	// Make the archive filename
	string filename_without_extension;
	string extension;
	size_t lastDotPos = fileOnly.find_last_of(".");
	if(lastDotPos == string::npos)
	{
		filename_without_extension = fileOnly;
		extension = "";
	}
	else
	{
		filename_without_extension = fileOnly.substr(0, lastDotPos);
		extension = fileOnly.substr(lastDotPos);
	}
	string newFilename = folderOnly;
	newFilename += "history/";
	newFilename += filename_without_extension;
	newFilename += "_";
	GTime::appendTimeStampValue(&newFilename, "-", "_", "-");
	newFilename += extension;
	cout << "Archiving " << szFilename << " to " << newFilename << "\n";
	if(!GFile::copyFile(filename.c_str(), newFilename.c_str()))
		cout << "Failed to copy file " << filename << " to " << newFilename << "\n";
}

void Editor::ajaxSaveGui(Server* pServer, GDynamicPageSession* pSession, const GDomNode* pIn, GDom& doc, GDomNode* pOut)
{
	// Load and parse the original file
	Account* pAccount = getAccount(pSession);
	string filename = pServer->m_basePath.c_str();
	filename += "usercontent/";
	filename += pAccount->username();
	filename += "/pages";
	filename += "/";
	filename += pIn->getString("filename");
	archiveFile(filename.c_str());
	GHtmlDoc domOld(filename.c_str());

	// Find the old content
	GHtmlElement* pOldContent = domOld.getElementById("content");
	if(!pOldContent)
	{
		pOldContent = domOld.getBody();
		if(!pOldContent)
			throw Ex("Could not find content or body in the old dom");
	}

	// Parse the new content and strip the contenteditable attribute
	const char* szNewContent = pIn->getString("content");
	GHtmlDoc domNew(szNewContent, strlen(szNewContent));
	GHtmlElement* pNewContent = domNew.getElementById("content");
	if(!pNewContent)
		throw Ex("Expected a tag with id=\"content\"");
	pNewContent->dropAttr("contenteditable");

	// Swap in the new content
	pOldContent->swap(pNewContent);

	// Write the file
	std::ofstream s;
	s.exceptions(std::ios::badbit | std::ios::failbit);
	try
	{
		s.open(filename.c_str(), std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to create the file, ", filename, ". ", strerror(errno));
	}
	domOld.document()->write(s);

	cout << "User " << pAccount->username() << " saved a page to: " << filename << "\n";
	cout.flush();
}

void Editor::pageBrowse(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	//Account* pAccount = getAccount(pSession);
	response << "<script type=\"text/javascript\" src=\"browse.js\"></script>\n";
	//response << "<input type=\"hidden\" id=\"username\" value=\"" << pAccount->username() << "\">\n";
	response << "Path: <input type=\"text\" id=\"path\" size=\"60\" disabled=\"disabled\">\n";
	response << "<table>\n";
	response << "	<tr><td width=200px>Folders:</td><td width=400px>Files:</td></tr>\n";
	response << "	<tr>\n";
	response << "		<td>\n";
	response << "			<select id=\"folders\" size=\"12\" onclick=\"closeDetails(0)\" ondblclick=\"onfolderchange()\" style=\"width:100%\">\n";
	response << "			</select>\n";
	response << "		</td>\n";
	response << "		<td>\n";
	response << "			<select id=\"files\" size=\"12\" onclick=\"closeDetails(0)\" style=\"width:100%\">\n";
	response << "			</select>\n";
	response << "		</td>\n";
	response << "	</tr>\n";
	response << "	<tr>\n";
	response << "		<td valign=top>\n";
	response << "			<details id=\"d1\" onclick=\"onclickd1()\"><summary>Delete selected folder</summary>\n";
	response << "				Are you sure? Everything inside it will be deleted too. <input type=\"submit\" value=\"Yes, delete it\">\n";
	response << "			</details><br>\n";
	response << "			<details id=\"d2\" onclick=\"onclickd2()\"><summary>New folder</summary>\n";
	response << "				Name: <input type=\"text\" id=\"newfoldername\"><input type=\"button\" onclick=\"newfolder()\" value=\"Create\">\n";
	response << "			</details>\n";
	response << "		</td>\n";
	response << "		<td valign=top>\n";
	response << "			<a onclick=\"editgui()\" href=\"#\">wysiwyg editor</a><br><br>\n";
	response << "			<a onclick=\"edittext()\" href=\"#\">text editor</a><br><br>\n";
	response << "			<details id=\"d3\" onclick=\"onclickd3()\"><summary>Delete selected file</summary>\n";
	response << "				Are you sure? <input type=\"submit\" value=\"Yes, delete it\">\n";
	response << "			</details><br>\n";
	response << "			<details id=\"d4\" onclick=\"onclickd4()\"><summary>New page</summary>\n";
	response << "				<form action=\"/edit\">Name: <input type=\"text\" name=\"pagename\"><input type=\"submit\" value=\"Create\"></form>\n";
	response << "			</details>\n";
	response << "		</td>\n";
	response << "	</tr>\n";
	response << "</table>\n";
	response << "<br><br>\n";
}

void Editor::pageEditText(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Determine the file name from a URL parameter
	GHttpParamParser params(pSession->params());
	const char* pagename = params.find("pagename");
	if(!pagename || strlen(pagename) < 1)
	{
		response << "Expected a page name.";
		return;
	}

	// Load the file
	Account* pAccount = getAccount(pSession);
	string filename = pServer->m_basePath.c_str();
	filename += "usercontent/";
	filename += pAccount->username();
	filename += "/pages/";
	filename += pagename;
	size_t fileLen;
	char* pFile = GFile::loadFile(filename.c_str(), &fileLen);
	ArrayHolder<char> hFile(pFile);

	// Generate the page content
	response << "<input type=\"hidden\" id=\"filename\" value=\"" << pagename << "\">";
	response << pServer->cache("editorText.html");
	response << "<tr><td><textarea id=\"page\" cols=\"100\" rows=\"30\">";
	response.write(pFile, fileLen);
	response << "</textarea></td></tr></table>\n";
}

void Editor::pageEditGui(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Determine the file name from a URL parameter
	GHttpParamParser params(pSession->params());
	const char* pagename = params.find("pagename");
	if(!pagename || strlen(pagename) < 1)
	{
		response << "Expected a page name.";
		return;
	}

	// Load the file
	Account* pAccount = getAccount(pSession);
	string filename = pServer->m_basePath.c_str();
	filename += "usercontent/";
	filename += pAccount->username();
	filename += "/pages/";
	filename += pagename;
	GHtmlDoc dom(filename.c_str());

	// Find the content div and make it content-editable
	GHtmlElement* pContent = dom.getElementById("content");
	if(!pContent)
	{
		pContent = dom.getBody();
		if(!pContent)
			throw Ex("Could not find content or body");
		pContent->name = "div";
		pContent->addAttr("id", "\"content\"");
	}
	pContent->addAttr("contenteditable", "true");

	// Generate the page content
	response << "<input type=\"hidden\" id=\"filename\" value=\"" << pagename << "\">";
	response << pServer->cache("editorGui.html");
	response << "<tr><td>";
	pContent->write(response);
	response << "</td></tr></table>\n";
}

void Editor::writeHtmlPre(ostream& stream, char* file, size_t len)
{
	for(size_t i = 0; i < len; i++)
	{
		if(*file == '&')
			stream << "&amp;";
		else if(*file == '<')
			stream << "&lt;";
		else if(*file == '>')
			stream << "&gt;";
		else
			stream << *file;
		file++;
	}
}

void Editor::pageDiff(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Determine the file name from a URL parameter
	GHttpParamParser params(pSession->params());
	const char* leftname = params.find("left");
	if(!leftname || strlen(leftname) < 1)
	{
		response << "Expected a left page.";
		return;
	}
	const char* rightname = params.find("right");
	if(!rightname || strlen(rightname) < 1)
	{
		response << "Expected a right page.";
		return;
	}

	// Load the files
	Account* pAccount = getAccount(pSession);
	string filename = pServer->m_basePath.c_str();
	filename += "usercontent/";
	filename += pAccount->username();
	filename += "/pages/";
	string leftfilename = filename + leftname;
	string rightfilename = filename + rightname;
	size_t leftSize, rightSize;
	char* leftFile = GFile::loadFile(leftfilename.c_str(), &leftSize);
	ArrayHolder<char> hLeftFile(leftFile);
	char* rightFile = GFile::loadFile(rightfilename.c_str(), &rightSize);
	ArrayHolder<char> hRightfile(rightFile);

	// Diff the files
	GDiffer diff;
	diff.compare(leftFile, leftSize, 0, rightFile, rightSize, 0);
	diff.simplify(12);
	std::vector<GDiffChunk*>& chunks = diff.chunks();

	// Build the diff content
	std::ostringstream ssLeft;
	std::ostringstream ssRight;
	size_t spotCount = 0;
	for(size_t i = 0; i < chunks.size(); i++)
	{
		GDiffChunk* pChunk = chunks[i];
		if(pChunk->left != INVALID_INDEX && pChunk->right != INVALID_INDEX)
		{
			// Match
			writeHtmlPre(ssLeft, leftFile + pChunk->left, pChunk->len);
			writeHtmlPre(ssRight, rightFile + pChunk->right, pChunk->len);
		}
		else
		{
			// Left side
			ssLeft << "<a id=\"a" << to_str(spotCount) << "\" ";
			ssLeft << "class=\"dc c" << to_str(spotCount % 4) << "\" ";
			ssLeft << "onclick=\"take('a" << to_str(spotCount) << "', 'b" << to_str(spotCount) << "')\" ";
			ssLeft << "href=\"#\">";

			// Right side
			ssRight << "<a id=\"b" << to_str(spotCount) << "\" ";
			ssRight << "class=\"dc c" << to_str(spotCount % 4) << "\" ";
			ssRight << "onclick=\"take('b" << to_str(spotCount) << "', 'a" << to_str(spotCount) << "')\" ";
			ssRight << "href=\"#\">";

			GDiffChunk* pChunk2 = (i + 1 < chunks.size() ? chunks[i + 1] : nullptr);
			if(!pChunk2 || (pChunk2->left != INVALID_INDEX && pChunk2->right != INVALID_INDEX))
			{
				// Make a loner segment
				if(pChunk->left != INVALID_INDEX)
					writeHtmlPre(ssLeft, leftFile + pChunk->left, pChunk->len);
				else
				{
					for(size_t j = 0; j < pChunk->len; j++)
						ssLeft << "░";
				}
				if(pChunk->right != INVALID_INDEX)
					writeHtmlPre(ssRight, rightFile + pChunk->right, pChunk->len);
				else
				{
					for(size_t j = 0; j < pChunk->len; j++)
						ssRight << "░";
				}
			}
			else
			{
				// Make a differing segment
				if(pChunk->right != INVALID_INDEX)
					std::swap(pChunk, pChunk2);
				GAssert(pChunk->left != INVALID_INDEX);
				GAssert(pChunk->right == INVALID_INDEX);
				GAssert(pChunk2->left == INVALID_INDEX);
				GAssert(pChunk2->right != INVALID_INDEX);
				writeHtmlPre(ssLeft, leftFile + pChunk->left, pChunk->len);
				if(pChunk->len < pChunk2->len)
				{
					for(size_t j = pChunk->len; j < pChunk2->len; j++)
						ssLeft << "░";
				}
				writeHtmlPre(ssRight, rightFile + pChunk2->right, pChunk2->len);
				if(pChunk2->len < pChunk->len)
				{
					for(size_t j = pChunk2->len; j < pChunk->len; j++)
						ssRight << "░";
				}
				i++;
			}
			ssLeft << "</a>";
			ssRight << "</a>";
			spotCount++;
		}
	}

	// Generate the diff page
	response << pServer->cache("diff.html");
	response << "<table border=1 width=100% cellpadding=10><tr><td width=50%><pre>\n";
	response << ssLeft.str();
	response << "\n</pre></td><td width=50%><pre>\n";
	response << ssRight.str();
	response << "\n</pre></td></tr></table>\n";
}


