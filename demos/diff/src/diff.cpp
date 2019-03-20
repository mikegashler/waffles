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
#include "diff.h"
#include "server.h"

using std::ostream;
using std::string;
using std::vector;
using std::cout;


string scrub_filename(const string filename)
{
	string copy = filename;
	for(size_t i = 0; i < copy.length(); i++)
	{
		char c = copy[i];
		if(c == '.' && i + 1 < copy.length() && copy[i + 1] == '.')
			copy[i] = '_';
		if(c == ' ' || (c >= '-' && c <= '9') || (c >= 'A' && c <= 'Z') || c == '_' || (c >= 'a' && c <= 'z'))
		{
		}
		else
			copy[i] = '_';
	}
	return copy;
}

class FilenameParser
{
public:
	string folder;
	string name;
	string extension;

	// Scrubs the filename.
	// Then parses it into three parts: folder, name, extension.
	// Ensures that the folder begins and ends with a "/".
	FilenameParser(const string& filename)
	{
		// Scrub the string
		string copy = scrub_filename(filename);

		// Split off the folder
		size_t lastSlashPos = copy.find_last_of("/");
		size_t nameStart;
		if(lastSlashPos == string::npos)
		{
			nameStart = 0;
			folder = "/";
		}
		else
		{
			nameStart = lastSlashPos + 1;
			folder = copy.substr(0, nameStart);
			if(folder[0] != '/')
				folder.insert(0, "/");
		}
		string fn = copy.substr(nameStart);

		// Split off the extension
		size_t lastDotPos = fn.find_last_of(".");
		if(lastDotPos == string::npos)
		{
			name = fn;
			extension = "";
		}
		else
		{
			name = fn.substr(0, lastDotPos);
			extension = fn.substr(lastDotPos);
		}
	}
};



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
	size_t leftSize, rightSize;
	char* leftFile = GFile::loadFile(leftname, &leftSize);
	ArrayHolder<char> hLeftFile(leftFile);
	char* rightFile = GFile::loadFile(rightname, &rightSize);
	ArrayHolder<char> hRightfile(rightFile);

	// Diff the files
	GDiffer diff;
	diff.compare(leftFile, leftSize, 0, rightFile, rightSize, 0);
	if(leftSize + rightSize < 1024)
		diff.simplify(6);
	else
		diff.simplify(9);
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
	response << "<script>\n";
	response << "\n";
	response << "function cut_trailing_underscores(s)\n";
	response << "{\n";
	response << "	let len = s.length;\n";
	response << "	while(len >= 0 && s.charAt(len - 1) == '░')\n";
	response << "		len--;\n";
	response << "	return s.substr(0, len);\n";
	response << "}\n";
	response << "\n";
	response << "function take(keep, replace)\n";
	response << "{\n";
	response << "	let obKeep = document.getElementById(keep);\n";
	response << "	let obReplace = document.getElementById(replace);\n";
	response << "	let newText = cut_trailing_underscores(obKeep.innerHTML);\n";
	response << "	obKeep.innerHTML = newText;\n";
	response << "	obReplace.innerHTML = newText;\n";
	response << "	obReplace.classList.remove(\"dc\");\n";
	response << "	obReplace.classList.remove(\"c0\");\n";
	response << "	obReplace.classList.remove(\"c1\");\n";
	response << "	obReplace.classList.remove(\"c2\");\n";
	response << "	obReplace.classList.remove(\"c3\");\n";
	response << "	obReplace.classList.add(\"resolved\");\n";
	response << "	obKeep.classList.remove(\"dc\");\n";
	response << "	obKeep.classList.remove(\"c0\");\n";
	response << "	obKeep.classList.remove(\"c1\");\n";
	response << "	obKeep.classList.remove(\"c2\");\n";
	response << "	obKeep.classList.remove(\"c3\");\n";
	response << "	obKeep.classList.add(\"resolved\");\n";
	response << "}\n";
	response << "\n";
	response << "</script>\n";
	response << "Click on the segments you want to keep...<br><br>\n";
	response << "<table border=1 width=100% cellpadding=10><tr><td width=50%><pre>\n";
	response << ssLeft.str();
	response << "\n</pre></td><td width=50%><pre>\n";
	response << ssRight.str();
	response << "\n</pre></td></tr></table>\n";
}
